"""Local stability of a classical state over continuous momenta (D51).

A stationary classical state is a local minimum of the classical energy when
its second variation is non-negative for every deformation. On the magnetic
supercell a deformation with wave vector ``k`` enters through the LSWT matrix
``H(k)`` of :class:`~spintoolkit.methods.lswt.quadratic.QuadraticBoseHamiltonian`:
with the boson operators replaced by classical amplitudes, the energy to second
order is the quadratic form of ``H(k)``. The state is locally stable when
``H(k)`` is positive semidefinite at every ``k`` of the magnetic zone, not only
on a mesh. (For onsite terms the energy is the spin-coherent-state value of
D37, the same as :func:`~spintoolkit.methods.classical.classical_energy`.)

A finite mesh misses an instability between its points, and a long-wavelength
instability next to a zero mode is the easiest to miss: where a zero mode sits
at ``k0``, the lowest eigenvalue nearby is ``q . rho . q`` with ``q = k - k0``,
so a negative stiffness ``rho`` gives values far below any fixed tolerance.
The test here therefore has two parts.

1. Away from zero modes: ``lambda_min(H(k)) / scale`` (``scale`` the largest
   ``|eig H(k)|`` on the mesh) is minimized over continuous ``k`` from the
   lowest points of an unshifted and a half-step shifted mesh (Nelder-Mead),
   outside balls of radius ``ball_radius`` (in units of the shortest magnetic
   reciprocal vector) about the zero modes. A value below
   ``-zero_tolerance`` is an instability at that ``k``; a value within
   ``zero_tolerance`` of zero is a further zero mode, which gets its own ball
   (with its partner at ``-k0``) before the search is repeated.
2. At each zero mode ``k0``, with ``V0`` spanning the null space of ``H(k0)``
   and ``W``, ``D`` the other eigenvectors and eigenvalues, degenerate
   perturbation theory gives the eigenvalues near zero exactly to second
   order in ``q``:

   - linear term ``L(u) = V0^dag (u . dH) V0``; any nonzero ``L`` makes some
     eigenvalue negative on one side, so it is an instability;
   - stiffness ``M(u) = V0^dag (1/2 u u : d2H) V0
     - V0^dag (u . dH) W D^-1 W^dag (u . dH) V0`` for unit directions ``u``;
     its lowest eigenvalue over ``u`` is ``rho_min`` (E0 times length
     squared). ``rho_min < 0`` is an instability; ``rho_min`` zero within the
     tolerance is a flat direction that harmonic order does not decide.

   The ball is also sampled (four radii, 24 directions) as a check beyond
   second order.

The derivatives of ``H`` are analytic, so ``rho`` has no finite-difference
error. The verdict is

- ``"unstable"``: a negative eigenvalue outside the balls, a nonzero linear
  term, a negative stiffness, or a negative sampled value inside a ball;
- ``"marginal"``: nothing negative found, but some zero mode is not fixed at
  harmonic order: a flat stiffness direction, a zero mode at ``k0 != 0``, or
  more flat directions at ``k = 0`` than the broken continuous symmetries
  explain (an accidental degeneracy, e.g. the NBCP three-sublattice orbit with
  J_PD or J_Gamma). Whether the energy along such a direction rises, falls or
  stays exactly flat is a question of higher order;
- ``"stable"``: every eigenvalue positive except Goldstone modes of broken
  symmetries at ``k = 0`` with positive definite stiffness;
- ``"not stationary"``: the largest torque exceeds the stationarity
  tolerance. The second variation about a non-stationary state does not
  decide local stability, so nothing else is computed.

Limits: the search is a local minimization from mesh seeds and can miss a
very narrow negative pocket between seeds; inside a ball only the quadratic
term and the sampled radii are tested, so a negative quartic term between the
sampled radii of a flat direction is not resolved (such a point is
``"marginal"`` anyway). Incommensurate structures (D34) are not handled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from spintoolkit.definitions.defaults import (
    LSWT_STATIONARITY_TOLERANCE, STABILITY_BALL_RADIUS, STABILITY_MESH, STABILITY_SEEDS,
    STABILITY_STIFFNESS_TOLERANCE, STABILITY_ZERO_TOLERANCE)
from spintoolkit.methods.classical import torques
from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.states.spin_state import SpinState, validate_spin_state
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import SpinModel

STABLE = "stable"
UNSTABLE = "unstable"
MARGINAL = "marginal"
NOT_STATIONARY = "not stationary"

_BALL_FRACTIONS = (0.125, 0.25, 0.5, 1.0)
_BALL_DIRECTIONS = 24
_STIFFNESS_ANGLES = 720
_MAX_ROUNDS = 6


@dataclass(frozen=True)
class ZeroModeExpansion:
    """Second-order expansion about one zero mode ``k0``.

    Attributes
    ----------
    k, fractional : (2,) arrays
        Cartesian momentum and coordinates in the magnetic reciprocal basis.
    nullity : int
        Number of eigenvalues of ``H(k0)`` within the zero tolerance.
    origin : str
        ``"goldstone"`` (k0 = 0, flat directions explained by broken
        continuous symmetries), ``"accidental"`` (k0 = 0, more flat
        directions than broken symmetries) or ``"accidental (k != 0)"``.
    linear : float
        Largest ``|eig L(u)|`` over directions, relative to ``scale * length``.
    stiffness_min : float
        ``rho_min`` (E0 length^2): lowest eigenvalue of ``M(u)`` over unit ``u``.
    stiffness_direction : float
        Angle (radians) of the direction ``u`` of ``rho_min``.
    stiffness_relative : float
        ``rho_min / (scale * length^2)``.
    ball_minimum : float
        Lowest sampled ``lambda_min / scale`` inside the ball.
    status : {"positive", "flat", "negative", "linear", "negative in ball"}
    """

    k: np.ndarray
    fractional: np.ndarray
    nullity: int
    origin: str
    linear: float
    stiffness_min: float
    stiffness_direction: float
    stiffness_relative: float
    ball_minimum: float
    status: str


@dataclass(frozen=True)
class StabilityReport:
    """Result of :func:`local_stability`.

    Attributes
    ----------
    verdict : {"stable", "unstable", "marginal", "not stationary"}
    reasons : tuple of str
        Why the verdict is not ``"stable"`` (empty if it is).
    lowest : float
        Lowest ``lambda_min(H(k)) / scale`` found outside the zero-mode balls
        (NaN if not computed).
    lowest_k, lowest_fractional : (2,) arrays
        Where ``lowest`` was found.
    scale : float
        Largest ``|eig H(k)|`` on the mesh (E0).
    length : float
        Root mean square bond length used to make ``rho`` relative.
    zero_modes : tuple of ZeroModeExpansion
    max_torque : float
    settings : dict
    """

    verdict: str
    reasons: Tuple[str, ...]
    lowest: float
    lowest_k: np.ndarray
    lowest_fractional: np.ndarray
    scale: float
    length: float
    zero_modes: Tuple[ZeroModeExpansion, ...]
    max_torque: float
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_stable(self) -> bool:
        """True only for the verdict ``"stable"``."""
        return self.verdict == STABLE

    def summary(self) -> str:
        text = self.verdict
        if np.isfinite(self.lowest):
            text += (f"; lowest lambda_min/scale {self.lowest:.3g} at q="
                     f"{np.round(self.lowest_fractional, 6).tolist()} outside zero modes")
        for z in self.zero_modes:
            text += (f"; zero mode at q={np.round(z.fractional, 6).tolist()} ({z.origin}, "
                     f"nullity {z.nullity}, rho_min {z.stiffness_min:.4g}, {z.status})")
        if self.reasons:
            text += "; " + "; ".join(self.reasons)
        return text

    def to_dict(self) -> Dict[str, Any]:
        from spintoolkit.methods.result import to_jsonable
        return to_jsonable({"verdict": self.verdict, "reasons": list(self.reasons),
                            "lowest": self.lowest, "lowest_k": self.lowest_k,
                            "lowest_fractional": self.lowest_fractional, "scale": self.scale,
                            "length": self.length, "zero_modes": list(self.zero_modes),
                            "max_torque": self.max_torque, "settings": self.settings,
                            "summary": self.summary()})


def local_stability(model: SpinModel, state: SpinState,
                    conditions: Optional[ExternalConditions] = None, *,
                    mesh: int = STABILITY_MESH, seeds: int = STABILITY_SEEDS,
                    ball_radius: float = STABILITY_BALL_RADIUS,
                    zero_tolerance: float = STABILITY_ZERO_TOLERANCE,
                    stiffness_tolerance: float = STABILITY_STIFFNESS_TOLERANCE,
                    stationarity_tolerance: float = LSWT_STATIONARITY_TOLERANCE
                    ) -> StabilityReport:
    """Test whether a classical state is a local minimum over continuous momenta.

    Parameters
    ----------
    model : SpinModel
    state : SpinState
        Commensurate classical state on its magnetic supercell.
    conditions : ExternalConditions, optional
    mesh : int
        Seed mesh ``mesh x mesh`` of the magnetic reciprocal cell, used
        unshifted and shifted by half a step.
    seeds : int
        Number of lowest mesh points from which ``lambda_min`` is minimized.
    ball_radius : float
        Radius of the ball about each zero mode, in units of the shortest
        magnetic reciprocal vector; inside it the expansion decides.
    zero_tolerance : float
        ``|lambda| / scale`` at or below this is zero (both for the null space
        of ``H(k0)`` and for the search).
    stiffness_tolerance : float
        ``|rho| / (scale * length^2)`` and the relative linear term at or
        below this count as zero.
    stationarity_tolerance : float
        Largest torque (E0) accepted.

    Returns
    -------
    StabilityReport
    """
    if not isinstance(state, SpinState):
        raise TypeError("local_stability needs a commensurate SpinState; incommensurate "
                        "structures (D34) are not handled")
    conditions = conditions or ExternalConditions()
    validate_spin_state(state, model)
    settings = {"mesh": int(mesh), "seeds": int(seeds), "ball_radius": float(ball_radius),
                "zero_tolerance": float(zero_tolerance),
                "stiffness_tolerance": float(stiffness_tolerance),
                "stationarity_tolerance": float(stationarity_tolerance)}
    torque = max(float(np.linalg.norm(t)) for t in torques(model, state, conditions).values())
    nan2 = np.full(2, np.nan)
    if torque > stationarity_tolerance:
        return StabilityReport(NOT_STATIONARY, (f"max torque {torque:.3g} E0 exceeds "
                                                f"{stationarity_tolerance:.3g}",),
                               float("nan"), nan2, nan2, float("nan"), float("nan"), (),
                               torque, settings)

    hamiltonian = QuadraticBoseHamiltonian(model, state, conditions)
    magnetic = state.magnetic_lattice(model)
    reciprocal = 2 * np.pi * np.linalg.inv(magnetic).T          # rows b_1, b_2; k = p @ reciprocal
    radius = ball_radius * float(np.min(np.linalg.norm(reciprocal, axis=1)))
    deltas = hamiltonian._deltas
    length = (float(np.sqrt(np.mean(np.sum(deltas ** 2, axis=1)))) if len(deltas)
              else float(np.min(np.linalg.norm(model.lattice, axis=1))))

    def to_k(p):
        return np.asarray(p, dtype=float) @ reciprocal

    def to_p(k):
        p = np.mod(np.asarray(k, dtype=float) @ magnetic.T / (2 * np.pi), 1.0)
        return np.where(np.isclose(p, 1.0, rtol=0, atol=1e-12), 0.0, p)

    grid = np.array([(i / mesh, j / mesh) for i in range(mesh) for j in range(mesh)])
    grid = np.vstack([grid, grid + 0.5 / mesh])
    special = np.array([(a, b) for a in (0, 0.5, 1 / 3, 2 / 3) for b in (0, 0.5, 1 / 3, 2 / 3)])
    grid = np.vstack([special, grid])
    mesh_values = _lowest_eigenvalues(hamiltonian, to_k(grid))
    scale = float(np.max(np.abs(np.linalg.eigvalsh(hamiltonian.at(to_k(grid[::7]))))))
    scale = max(scale, float(np.max(np.abs(mesh_values))), 1e-300)
    relative = mesh_values / scale

    def lowest(p) -> float:
        return float(np.linalg.eigvalsh(hamiltonian.at(to_k(p))[0])[0] / scale)

    def distance(p, q) -> float:
        d = np.mod(np.asarray(p) - np.asarray(q) + 0.5, 1.0) - 0.5
        return float(np.linalg.norm(to_k(d)))

    centres: List[np.ndarray] = []

    def add_centre(p):
        for q in (to_p(to_k(p)), to_p(-to_k(p))):
            if all(distance(q, c) > 1e-9 for c in centres):
                centres.append(q)

    if abs(relative[0]) <= zero_tolerance:                  # grid[0] is the zone centre
        add_centre(np.zeros(2))

    def outside(p) -> bool:
        return all(distance(p, c) > radius for c in centres)

    best_value, best_p = np.inf, np.full(2, np.nan)
    step = 1.0 / mesh
    exhausted = False
    for _ in range(_MAX_ROUNDS):
        order = [i for i in np.argsort(relative) if outside(grid[i])]
        starts: List[np.ndarray] = []
        for i in order:
            if all(distance(grid[i], s) > 1.5 * step for s in starts):
                starts.append(grid[i])
            if len(starts) >= seeds:
                break
        best_value, best_p = np.inf, np.full(2, np.nan)
        for p0 in starts:
            value, p = lowest(p0), p0
            simplex = np.array([p0, p0 + [step, 0], p0 + [0, step]])
            opt = minimize(lambda x: lowest(x) if outside(x) else lowest(p0),
                           p0, method="Nelder-Mead",
                           options={"initial_simplex": simplex, "xatol": 1e-10,
                                    "fatol": 1e-18, "maxiter": 400})
            if opt.fun < value and outside(opt.x):
                value, p = float(opt.fun), to_p(to_k(opt.x))
            if value < best_value:
                best_value, best_p = value, to_p(to_k(p))
        if best_value < -zero_tolerance or best_value > zero_tolerance:
            break
        add_centre(best_p)                                  # a further zero mode
    else:
        exhausted = True
    reasons: List[str] = []
    if best_value < -zero_tolerance:
        reasons.append(f"negative eigenvalue {best_value:.3g} (relative) at q="
                       f"{np.round(best_p, 6).tolist()}")

    origin_counts = None
    expansions = []
    for centre in centres:
        at_origin = distance(centre, np.zeros(2)) < 1e-9
        if at_origin and origin_counts is None:
            from spintoolkit.observables.zero_modes import _symmetry_origin
            origin_counts = _symmetry_origin(model, state, conditions)
        expansions.append(_expand(hamiltonian, to_k(centre), centre, scale, length, radius,
                                  zero_tolerance, stiffness_tolerance, at_origin,
                                  origin_counts))
    for z in expansions:
        where = f"q={np.round(z.fractional, 6).tolist()}"
        if z.status == "linear":
            reasons.append(f"linear term at the zero mode {where}")
        elif z.status == "negative":
            reasons.append(f"negative stiffness rho_min {z.stiffness_min:.3g} at {where}, "
                           f"direction {z.stiffness_direction:.4f} rad")
        elif z.status == "negative in ball":
            reasons.append(f"negative eigenvalue {z.ball_minimum:.3g} (relative) near {where}")
    unstable = bool(reasons)
    if not unstable:
        if exhausted:
            reasons.append(f"zero modes are not isolated: still finding new ones after "
                           f"{_MAX_ROUNDS} rounds (a line or surface of zero modes?)")
        for z in expansions:
            where = f"q={np.round(z.fractional, 6).tolist()}"
            if z.status == "flat":
                reasons.append(f"flat stiffness direction at {where} (higher order decides)")
            if z.origin != "goldstone":
                reasons.append(f"{z.origin} zero mode at {where} (higher order decides)")
    verdict = UNSTABLE if unstable else MARGINAL if reasons else STABLE
    return StabilityReport(verdict, tuple(reasons), float(best_value), to_k(best_p),
                           np.asarray(best_p, dtype=float), scale, length, tuple(expansions),
                           torque, settings)


def _lowest_eigenvalues(hamiltonian: QuadraticBoseHamiltonian, k: np.ndarray,
                        chunk: int = 256) -> np.ndarray:
    return np.concatenate([np.linalg.eigvalsh(hamiltonian.at(k[i:i + chunk]))[:, 0]
                           for i in range(0, len(k), chunk)])


def _expand(hamiltonian, k0, p0, scale, length, radius, zero_tolerance, stiffness_tolerance,
            at_origin, origin_counts) -> ZeroModeExpansion:
    """Linear term, stiffness and ball samples about the zero mode ``k0``."""
    w, v = np.linalg.eigh(hamiltonian.at(k0)[0])
    null = np.abs(w) <= zero_tolerance * scale
    V0, W, D = v[:, null], v[:, ~null], w[~null]
    nullity = int(np.sum(null))
    dx, dy = (d[0] for d in hamiltonian.derivatives_at(k0))
    dxx, dxy, dyy = (d[0] for d in hamiltonian.second_derivatives_at(k0))
    Wd = W / D                                              # W D^-1

    def parts(theta):
        u = np.array([np.cos(theta), np.sin(theta)])
        first = u[0] * dx + u[1] * dy
        second = u[0] ** 2 * dxx + 2 * u[0] * u[1] * dxy + u[1] ** 2 * dyy
        L = V0.conj().T @ first @ V0
        coupling = W.conj().T @ first @ V0
        M = 0.5 * V0.conj().T @ second @ V0 - coupling.conj().T @ (Wd.conj().T @ first @ V0)
        return 0.5 * (L + L.conj().T), 0.5 * (M + M.conj().T)

    def rho(theta) -> float:
        return float(np.linalg.eigvalsh(parts(theta)[1])[0])

    angles = np.linspace(0.0, np.pi, _STIFFNESS_ANGLES, endpoint=False)  # M(u) = M(-u)
    linear = max(float(np.max(np.abs(np.linalg.eigvalsh(parts(t)[0])))) for t in angles[::30])
    values = np.array([rho(t) for t in angles])
    i = int(np.argmin(values))
    h = np.pi / _STIFFNESS_ANGLES
    opt = minimize_scalar(rho, bounds=(angles[i] - h, angles[i] + h), method="bounded",
                          options={"xatol": 1e-10})
    theta_min, rho_min = ((float(opt.x), float(opt.fun)) if opt.fun < values[i]
                          else (float(angles[i]), float(values[i])))
    theta_min = float(np.mod(theta_min, np.pi))
    rho_relative = rho_min / (scale * length ** 2)
    linear_relative = linear / (scale * length)

    directions = np.arange(_BALL_DIRECTIONS) * 2 * np.pi / _BALL_DIRECTIONS
    probes = np.array([k0 + f * radius * np.array([np.cos(t), np.sin(t)])
                       for f in _BALL_FRACTIONS for t in directions])
    ball = float(np.min(_lowest_eigenvalues(hamiltonian, probes)) / scale)

    if np.any(w < -zero_tolerance * scale):
        status = "negative"
    elif linear_relative > stiffness_tolerance:
        status = "linear"
    elif rho_relative < -stiffness_tolerance:
        status = "negative"
    elif ball < -zero_tolerance:
        status = "negative in ball"
    elif rho_relative <= stiffness_tolerance:
        status = "flat"
    else:
        status = "positive"
    if not at_origin:
        origin = "accidental (k != 0)"
    else:
        broken = int(origin_counts.get("broken_symmetries", 0)) if origin_counts else 0
        origin = "goldstone" if nullity <= broken else "accidental"
    return ZeroModeExpansion(np.asarray(k0, dtype=float), np.asarray(p0, dtype=float), nullity,
                             origin, float(linear_relative), rho_min, theta_min,
                             float(rho_relative), ball, status)
