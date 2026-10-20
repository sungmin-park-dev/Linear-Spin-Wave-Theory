"""Stationary points of the classical energy on a fixed magnetic cell (D63).

LSWT, and MAGSWT (D62) for classically unstable states, need a reference
state with vanishing torques: otherwise linear boson terms remain and the
harmonic expansion is not defined. With bond-dependent couplings such as the
``Jzpm3`` term of the BCAO model, a symmetric pattern (e.g. up-up-down) is in
general not stationary; the stationary point with the same magnetic cell is a
canted version of it, and it may be a saddle point rather than a minimum.
:func:`~spintoolkit.methods.classical.refine_classical` minimizes the energy
and so slides off a saddle; :func:`find_stationary_state` instead solves
``dE/dx = 0`` by Newton's method in the tangent coordinates of
:mod:`~spintoolkit.methods.classical`, which converges to the nearby
stationary point whatever its curvature.

Each step solves ``H x = -g`` with the analytic Hessian ``H`` and gradient
``g`` of :func:`~spintoolkit.methods.classical.tangent_expansion`, leaving
alone directions with ``|eig H|`` below ``sqrt(eps) max |eig H|`` (rotations
of an exact symmetry, along which ``g`` vanishes). The step is halved until
the largest torque decreases; if no halving helps, a step along ``-H g``
(steepest descent of ``|g|^2``) is tried. Every step starts from a new chart
at the current state. Gradient and Hessian are covariant under the symmetries
of the model, so a seed with a symmetry keeps it.

No stationary point near the seed is a possible answer: the result then says
``converged=False`` and carries the last state only for inspection; it is not
a reference state for LSWT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from spintoolkit.definitions.defaults import (
    STATIONARY_HALVINGS, STATIONARY_MAX_STEPS, STATIONARY_TORQUE_TOLERANCE)
from spintoolkit.methods.classical import _with_directions, tangent_expansion
from spintoolkit.states.spin_state import SpinState, validate_spin_state
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import SpinModel


@dataclass(frozen=True)
class StationaryResult:
    """Outcome of :func:`find_stationary_state`.

    Attributes
    ----------
    state : SpinState
        Stationary state when ``converged``; otherwise the lowest-torque state
        reached (not a valid LSWT reference).
    converged : bool
        Largest torque at or below ``torque_tolerance``.
    max_torque : float
        Largest ``|S_i x h_i|`` (E0).
    energy : float
        Classical energy per site (E0).
    saddle_index : int
        Number of negative eigenvalues of the tangent Hessian on the cell
        (momenta commensurate with it only): 0 for a minimum on the cell, more
        for a saddle. Not a stability verdict over continuous momenta; use
        :func:`~spintoolkit.methods.stability.local_stability`. -1 if not
        converged.
    flat_directions : int
        Hessian eigenvalues within ``sqrt(eps) max |eig|`` of zero (symmetry
        rotations and accidental degeneracies). -1 if not converged.
    max_rotation_from_seed : float
        Largest angle (rad) between a spin and its seed direction.
    steps : int
    history : list of float
        Largest torque after each step.
    """

    state: SpinState
    converged: bool
    max_torque: float
    energy: float
    saddle_index: int
    flat_directions: int
    max_rotation_from_seed: float
    steps: int
    history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"converged": self.converged, "max_torque": self.max_torque,
                "energy": self.energy, "saddle_index": self.saddle_index,
                "flat_directions": self.flat_directions,
                "max_rotation_from_seed": self.max_rotation_from_seed, "steps": self.steps,
                "history": list(self.history)}


def _flat(w: np.ndarray) -> np.ndarray:
    return np.abs(w) <= np.sqrt(np.finfo(float).eps) * max(float(np.max(np.abs(w))), 1e-300)


def find_stationary_state(model: SpinModel, seed: SpinState,
                          conditions: Optional[ExternalConditions] = None, *,
                          torque_tolerance: float = STATIONARY_TORQUE_TOLERANCE,
                          max_steps: int = STATIONARY_MAX_STEPS) -> StationaryResult:
    """Newton search for a stationary point (minimum or saddle) near ``seed``.

    Parameters
    ----------
    model : SpinModel
    seed : SpinState
        Starting pattern; its magnetic supercell is kept.
    conditions : ExternalConditions, optional
    torque_tolerance : float
        Largest torque (E0) accepted as stationary.
    max_steps : int
        Largest number of Newton steps.

    Returns
    -------
    StationaryResult
    """
    conditions = conditions or ExternalConditions()
    validate_spin_state(seed, model)
    seed_directions = np.array([seed.directions[key] for key in seed.directions])
    state, ex = seed, tangent_expansion(model, seed, conditions)
    n = len(ex.keys)
    history: List[float] = []

    def moved(ex, step):
        vectors = ex.directions + np.einsum("ia,iax->ix", step.reshape(n, 2), ex.frames)
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        trial = _with_directions(seed, ex.keys, vectors, seed.provenance)
        return trial, tangent_expansion(model, trial, conditions)

    steps = 0
    while ex.max_torque > torque_tolerance and steps < max_steps:
        steps += 1
        w, v = np.linalg.eigh(ex.hessian)
        keep = ~_flat(w)
        newton = -v[:, keep] @ ((v[:, keep].T @ ex.gradient) / w[keep])
        descent = -ex.hessian @ ex.gradient
        accepted = None
        for direction in (newton, descent):
            if not np.all(np.isfinite(direction)) or not np.any(direction):
                continue
            if direction is descent:              # scale to the Newton step length
                direction = direction * (np.linalg.norm(newton) / np.linalg.norm(direction)
                                         if np.linalg.norm(newton) > 0 else 1.0)
            alpha = 1.0
            for _ in range(STATIONARY_HALVINGS):
                trial = moved(ex, alpha * direction)
                if trial[1].max_torque < ex.max_torque:
                    accepted = trial
                    break
                alpha /= 2
            if accepted is not None:
                break
        if accepted is None:
            break
        state, ex = accepted
        history.append(ex.max_torque)

    converged = ex.max_torque <= torque_tolerance
    if converged:
        w = np.linalg.eigvalsh(ex.hessian)
        flat = _flat(w)
        saddle_index, flat_count = int(np.sum((w < 0) & ~flat)), int(np.sum(flat))
    else:
        saddle_index = flat_count = -1
    final = np.array([state.directions[key] for key in seed.directions])
    rotation = float(np.max(np.arccos(np.clip(np.sum(final * seed_directions, axis=1), -1, 1)),
                            initial=0.0))
    provenance = dict(seed.provenance)
    provenance["stationary_search"] = {"converged": converged, "max_torque": ex.max_torque,
                                       "steps": steps, "saddle_index": saddle_index}
    state = SpinState(state.model_ref, state.supercell, dict(state.directions), provenance)
    return StationaryResult(state, converged, float(ex.max_torque), float(ex.energy),
                            saddle_index, flat_count, rotation, steps, history)
