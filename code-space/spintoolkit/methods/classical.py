"""Classical energy, local fields and torques of a spin state.

Reads only the model's terms, so it works for any :class:`SpinModel` without
model-specific code. Spins are classical vectors ``S n`` of length ``S``:

    E = sum_bilinear S_i^T J S_j - sum_zeeman b^T g S_i

evaluated over one magnetic supercell and reported per site, dimensionless in
the energy unit E0 of the coefficients (``b`` is ``ExternalConditions.field``). The local field
``h_i = -dE/dS_i`` collects every term that contains spin ``i``; the torque
``S_i x h_i`` vanishes at a classical stationary point, the condition for the
linear boson terms of LSWT to vanish.

Second-order expansions use local tangent coordinates: spin ``i`` moves to
``n_i sqrt(1 - |x_i|^2) + x_i^1 e_i^1 + x_i^2 e_i^2`` with an orthonormal frame
``(e_i^1, e_i^2)`` perpendicular to ``n_i``. Unlike polar angles these
coordinates have no spurious zero mode at the poles. In them

    dE/dx_ia        = -S_i e_ia . h_i
    d2E/dx_ia dx_jb = S_i S_j e_ia . K_ij e_jb + delta_ij delta_ab S_i n_i . h_i

where ``K_ij = d2E/dS_i dS_j`` collects the exchange matrices. Both are
reported per site, like the energy.
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple
import warnings

import numpy as np
from scipy.optimize import minimize

from spintoolkit.definitions.defaults import (
    CLASSICAL_REFINE_GTOL, CLASSICAL_REFINE_NEWTON_STEPS)

from spintoolkit.states.spin_state import SpinState, validate_spin_state
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import BILINEAR, ZEEMAN, SpinModel

#: Term kinds this consumer understands.
SUPPORTED_KINDS = (BILINEAR, ZEEMAN)

Key = Tuple[str, Tuple[int, int]]


def _prepare(model: SpinModel, state: SpinState,
             conditions: Optional[ExternalConditions]):
    unsupported = sorted({t.kind for t in model.terms} - set(SUPPORTED_KINDS))
    if unsupported:
        raise NotImplementedError(f"classical methods do not support term kinds {unsupported}")
    validate_spin_state(state, model)
    conditions = conditions or ExternalConditions()
    field = conditions.field
    if np.any(field != 0):
        coupled = {t.participants[0][0] for t in model.terms_of_kind(ZEEMAN)}
        uncoupled = sorted(set(model.site_ids) - coupled)
        if uncoupled:
            warnings.warn(f"sites {uncoupled} have no zeeman term and do not couple "
                          "to the applied field", UserWarning, stacklevel=3)
    spins = {(site, cell): model.site(site).spin * state.direction(site, cell)
             for site in model.site_ids for cell in state.cells}
    return field, spins


def _shift(cell, offset):
    return (cell[0] + offset[0], cell[1] + offset[1])


def classical_energy(model: SpinModel, state: SpinState,
                     conditions: Optional[ExternalConditions] = None,
                     geometry: Optional[CalculationGeometry] = None) -> float:
    """Classical energy per site in units of E0.

    Parameters
    ----------
    model : SpinModel
    state : SpinState
        Must belong to ``model``.
    conditions : ExternalConditions, optional
        Dimensionless field (default zero). The temperature is not used.
    geometry : CalculationGeometry, optional
        A finite torus evaluates the energy of the torus Hamiltonian with the
        expansion rules shared by all methods (D23): the state must tile the
        torus, and a torus on which a bond folds onto a single site is
        rejected. For a commensurate state the value per site equals the
        thermodynamic-limit value. None or the thermodynamic limit evaluates
        one magnetic supercell.
    """
    if geometry is not None and geometry.kind == "finite_torus":
        return _torus_energy(model, state, conditions, geometry)
    field, spins = _prepare(model, state, conditions)
    energy = 0.0
    for term in model.terms_of_kind(BILINEAR):
        (a, n1), (b, n2) = term.participants
        for cell in state.cells:
            s_i = spins[(a, state.reduce_cell(_shift(cell, n1)))]
            s_j = spins[(b, state.reduce_cell(_shift(cell, n2)))]
            energy += s_i @ term.coefficient @ s_j
    for term in model.terms_of_kind(ZEEMAN):
        site = term.participants[0][0]
        for cell in state.cells:
            energy -= field @ term.coefficient @ spins[(site, cell)]
    return float(energy / (model.num_sites * state.num_cells))


def _torus_energy(model, state, conditions, geometry) -> float:
    from spintoolkit.system.cluster import expand_on_torus

    _prepare(model, state, conditions)
    validate_spin_state(state, model, geometry)
    cluster = expand_on_torus(model, geometry)
    spins = np.array([cluster.spins[i] * state.direction(site, state.reduce_cell(cell))
                      for i, (site, cell) in enumerate(cluster.keys)])
    energy = np.einsum("ma,mab,mb->", spins[cluster.source], cluster.exchange,
                       spins[cluster.target])
    energy -= np.sum(cluster.fields(conditions) * spins)
    return float(energy / cluster.num_sites)


def local_fields(model: SpinModel, state: SpinState,
                 conditions: Optional[ExternalConditions] = None) -> Dict[Key, np.ndarray]:
    """Local field ``h_i = -dE/dS_i`` on every spin of the supercell.

    Returns
    -------
    dict
        ``(site_id, cell) -> (3,)`` array in units of E0.
    """
    field, spins = _prepare(model, state, conditions)
    fields = {key: np.zeros(3) for key in spins}
    for term in model.terms_of_kind(BILINEAR):
        (a, n1), (b, n2) = term.participants
        J = term.coefficient
        for cell in state.cells:
            i = (a, state.reduce_cell(_shift(cell, n1)))
            j = (b, state.reduce_cell(_shift(cell, n2)))
            fields[i] -= J @ spins[j]
            fields[j] -= J.T @ spins[i]
    for term in model.terms_of_kind(ZEEMAN):
        site = term.participants[0][0]
        for cell in state.cells:
            fields[(site, cell)] += term.coefficient.T @ field
    return fields


def torques(model: SpinModel, state: SpinState,
            conditions: Optional[ExternalConditions] = None) -> Dict[Key, np.ndarray]:
    """Torque ``S_i x h_i`` on every spin; all vanish at a stationary state.

    Returns
    -------
    dict
        ``(site_id, cell) -> (3,)`` array in units of E0.
    """
    fields = local_fields(model, state, conditions)
    return {key: np.cross(model.site(key[0]).spin * state.direction(*key), value)
            for key, value in fields.items()}


class _Compiled(NamedTuple):
    """Array form of a model and state: one row per spin of the supercell."""

    keys: Tuple[Key, ...]
    lengths: np.ndarray        # (n,) spin lengths S_i
    directions: np.ndarray     # (n, 3) unit vectors n_i
    source: np.ndarray         # (m,) spin index of each bond source
    target: np.ndarray         # (m,) spin index of each bond target
    exchange: np.ndarray       # (m, 3, 3) exchange matrix of each bond
    zeeman: np.ndarray         # (n, 3) g_i^T b of each spin


def _compile(model: SpinModel, state: SpinState,
             conditions: Optional[ExternalConditions]) -> _Compiled:
    field, spins = _prepare(model, state, conditions)
    keys = tuple(spins)
    index = {key: i for i, key in enumerate(keys)}
    source, target, exchange = [], [], []
    for term in model.terms_of_kind(BILINEAR):
        (a, n1), (b, n2) = term.participants
        for cell in state.cells:
            source.append(index[(a, state.reduce_cell(_shift(cell, n1)))])
            target.append(index[(b, state.reduce_cell(_shift(cell, n2)))])
            exchange.append(term.coefficient)
    zeeman = np.zeros((len(keys), 3))
    for term in model.terms_of_kind(ZEEMAN):
        site = term.participants[0][0]
        for cell in state.cells:
            zeeman[index[(site, cell)]] += term.coefficient.T @ field
    return _Compiled(keys, np.array([model.site(k[0]).spin for k in keys]),
                     np.array([state.direction(*k) for k in keys]),
                     np.array(source, dtype=int), np.array(target, dtype=int),
                     np.array(exchange, dtype=float).reshape(-1, 3, 3), zeeman)


def _energy_and_fields(compiled: _Compiled, directions: np.ndarray):
    """Total energy and local fields ``h_i = -dE/dS_i`` (not divided by the site count)."""
    spins = compiled.lengths[:, None] * directions
    s_i, s_j = spins[compiled.source], spins[compiled.target]
    j_sj = np.einsum("mab,mb->ma", compiled.exchange, s_j)
    jt_si = np.einsum("mba,mb->ma", compiled.exchange, s_i)
    energy = np.sum(s_i * j_sj) - np.sum(compiled.zeeman * spins)
    fields = compiled.zeeman.copy()
    np.subtract.at(fields, compiled.source, j_sj)
    np.subtract.at(fields, compiled.target, jt_si)
    return energy, fields


def tangent_frames(directions: np.ndarray) -> np.ndarray:
    """Orthonormal frames ``(e^1, e^2)`` perpendicular to each unit vector.

    Parameters
    ----------
    directions : (n, 3) array
        Unit vectors.

    Returns
    -------
    (n, 2, 3) array
        ``frames[i, a]`` is ``e_i^a``; ``(e^1, e^2, n)`` is right-handed.
    """
    frames = np.empty((len(directions), 2, 3))
    for i, n in enumerate(np.asarray(directions, dtype=float)):
        helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n, helper)
        e1 /= np.linalg.norm(e1)
        frames[i] = e1, np.cross(n, e1)
    return frames


class TangentExpansion(NamedTuple):
    """Gradient and Hessian of the classical energy per site in tangent coordinates.

    Coordinate ``2 i + a`` is the displacement of spin ``keys[i]`` along
    ``frames[i, a]``, in radians.
    """

    keys: Tuple[Key, ...]
    directions: np.ndarray     # (n, 3)
    frames: np.ndarray         # (n, 2, 3)
    energy: float              # per site
    gradient: np.ndarray       # (2n,)
    hessian: np.ndarray        # (2n, 2n)
    max_torque: float          # max_i |S_i x h_i|, in E0


def tangent_expansion(model: SpinModel, state: SpinState,
                      conditions: Optional[ExternalConditions] = None) -> TangentExpansion:
    """Analytic gradient and Hessian of the classical energy about ``state``.

    The Hessian is exact for the tangent coordinates of the module docstring
    at any state; at a stationary state it is the classical spin-wave
    stiffness of the supercell (wave vectors commensurate with it only).

    Parameters
    ----------
    model : SpinModel
    state : SpinState
    conditions : ExternalConditions, optional

    Returns
    -------
    TangentExpansion
    """
    compiled = _compile(model, state, conditions)
    n = len(compiled.keys)
    directions = compiled.directions
    frames = tangent_frames(directions)
    energy, fields = _energy_and_fields(compiled, directions)
    coupling = np.zeros((n, 3, n, 3))
    for i, j, J in zip(compiled.source, compiled.target, compiled.exchange):
        coupling[i, :, j, :] += J
        coupling[j, :, i, :] += J.T
    scaled = compiled.lengths[:, None, None] * frames                     # S_i e_ia
    hessian = np.einsum("iax,ixjy,jby->iajb", scaled, coupling, scaled).reshape(2 * n, 2 * n)
    diagonal = compiled.lengths * np.einsum("ix,ix->i", directions, fields)
    hessian += np.diag(np.repeat(diagonal, 2))
    gradient = -np.einsum("iax,ix->ia", scaled, fields).ravel()
    torque = np.cross(compiled.lengths[:, None] * directions, fields)
    return TangentExpansion(compiled.keys, directions, frames, float(energy / n),
                            gradient / n, hessian / n,
                            float(np.max(np.linalg.norm(torque, axis=1))))


def _with_directions(state: SpinState, keys, directions, provenance) -> SpinState:
    return SpinState(state.model_ref, state.supercell,
                     {key: vector for key, vector in zip(keys, directions)}, provenance)


def refine_classical(model: SpinModel, state: SpinState,
                     conditions: Optional[ExternalConditions] = None,
                     gtol: float = CLASSICAL_REFINE_GTOL,
                     newton_steps: int = CLASSICAL_REFINE_NEWTON_STEPS) -> SpinState:
    """Polish a classical minimum with the analytic gradient and Hessian.

    L-BFGS-B in tangent coordinates (spins renormalized after each step),
    followed by Newton steps restricted to the non-flat Hessian directions so
    that exactly flat directions of a degenerate manifold are left alone.
    Intended for states that are already near a minimum, such as the result of
    a global search.

    Parameters
    ----------
    model : SpinModel
    state : SpinState
    conditions : ExternalConditions, optional
    gtol : float
        Gradient tolerance of L-BFGS-B (per site, E0 per radian).
    newton_steps : int
        Maximum number of Newton steps after L-BFGS-B.

    Returns
    -------
    SpinState
        Same supercell; provenance gains a ``refinement`` record with the
        energy change and the maximum torque before and after.
    """
    compiled = _compile(model, state, conditions)
    n = len(compiled.keys)
    start = compiled.directions
    frames = tangent_frames(start)

    def moved(x):
        vectors = start + np.einsum("ia,iax->ix", x.reshape(n, 2), frames)
        radii = np.linalg.norm(vectors, axis=1)
        return vectors / radii[:, None], radii

    def objective(x):
        directions, radii = moved(x)
        energy, fields = _energy_and_fields(compiled, directions)
        force = -compiled.lengths[:, None] * fields                          # dE/dn_i
        force -= np.einsum("ix,ix->i", force, directions)[:, None] * directions
        gradient = np.einsum("ix,iax->ia", force, frames) / radii[:, None]
        return energy / n, gradient.ravel() / n

    before = tangent_expansion(model, state, conditions)
    result = minimize(objective, np.zeros(2 * n), jac=True, method="L-BFGS-B",
                      options={"gtol": gtol, "ftol": 0.0, "maxiter": 10000})
    refined = _with_directions(state, compiled.keys, moved(result.x)[0], state.provenance)
    current = tangent_expansion(model, refined, conditions)
    for _ in range(newton_steps):
        w, v = np.linalg.eigh(current.hessian)
        keep = np.abs(w) > np.sqrt(np.finfo(float).eps) * np.max(np.abs(w))
        step = -v[:, keep] @ ((v[:, keep].T @ current.gradient) / w[keep])
        vectors = current.directions + np.einsum("ia,iax->ix", step.reshape(n, 2), current.frames)
        trial = _with_directions(state, current.keys,
                                 vectors / np.linalg.norm(vectors, axis=1)[:, None],
                                 state.provenance)
        expansion = tangent_expansion(model, trial, conditions)
        if expansion.max_torque >= current.max_torque:
            break
        refined, current = trial, expansion
    record = {"method": "L-BFGS-B + Newton (tangent coordinates)",
              "energy_change": current.energy - before.energy,
              "max_torque_before": before.max_torque, "max_torque_after": current.max_torque}
    return _with_directions(state, current.keys, current.directions,
                            {**dict(state.provenance), "refinement": record})
