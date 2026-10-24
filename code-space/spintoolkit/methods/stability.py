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
  harmonic order (a flat stiffness direction, a zero mode at ``k0 != 0``, or
  more flat directions at ``k = 0`` than the broken continuous symmetries
  explain: an accidental degeneracy, e.g. the NBCP three-sublattice orbit with
  J_PD or J_Gamma), and the beyond-harmonic test below finds it flat or
  cannot resolve it;
- ``"stable"``: every eigenvalue positive except Goldstone modes of broken
  symmetries at ``k = 0`` with positive definite stiffness, and every zero
  mode harmonic order leaves open rises beyond harmonic order;
- ``"not stationary"``: the largest torque exceeds the stationarity
  tolerance. The second variation about a non-stationary state does not
  decide local stability, so nothing else is computed.

3. Beyond harmonic order (D54, ``higher_order=True``), for the zero modes
   step 2 leaves open:

   - a flat stiffness direction: ``H(k)`` is exact in ``k``, so
     ``lambda_min(H(k0 + q))``, minimized over the direction of ``q``, is
     taken at radii ``ball_radius, ball_radius / 2, ...``;
   - accidental zero modes: the state is put on the supercell that folds
     them to ``k = 0`` (denominators up to ``STABILITY_FOLD_DENOMINATOR``,
     at most ``STABILITY_FOLD_SPINS`` spins); there the Hessian null space
     minus the generators of exact global rotations is the accidental
     subspace, and the classical energy per spin along sampled directions of
     it (both signs), with all stiff directions relaxed, is taken at
     root-mean-square angles ``STABILITY_AMPLITUDE / 2^j`` per spin.

   The smallest amplitude whose change exceeds the floor
   (``zero_tolerance`` and ``STABILITY_AMPLITUDE_FLOOR`` relative to
   ``scale``) gives the sign of the leading term and, against the next
   larger amplitude, its power. A fall in any direction (an odd cubic term
   falls on one side) is an instability; a rise in every direction makes the
   zero mode harmless; a change below the floor up to the largest amplitude in
   some direction is a classically degenerate manifold, left ``"marginal"``
   since fluctuations decide (order by disorder); otherwise unresolved.

Limits: the search is a local minimization from mesh seeds and can miss a
very narrow negative pocket between seeds; inside a ball only the quadratic
term and the sampled radii are tested, so a negative quartic term between the
sampled radii of a flat direction is not resolved by step 2 (step 3 tests
it). A single-Q spiral (D55) is tested in its rotating frame, where the
model is translation invariant on the primitive cell for any pitch; models
not U(1) symmetric about the spiral axis are refused as in D34. The amplitude test samples directions of the accidental subspace (8
for two dimensions, basis vectors and pairs up to six) and relaxes only the
stiff ones, so a fall confined between sampled directions can be missed;
"flat" means flat within the floor, not proved exact. Zero modes on lines
are not tested beyond harmonic order, nor are accidental zero modes at an
incommensurate ``k0`` (no finite cell folds them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from spintoolkit.definitions.defaults import (
    LSWT_STATIONARITY_TOLERANCE, STABILITY_AMPLITUDE, STABILITY_AMPLITUDE_FLOOR,
    STABILITY_BALL_RADIUS, STABILITY_FOLD_DENOMINATOR, STABILITY_FOLD_SPINS, STABILITY_MESH,
    STABILITY_SEEDS, STABILITY_STIFFNESS_TOLERANCE, STABILITY_ZERO_TOLERANCE)
from spintoolkit.methods.classical import (_compile, _energy_and_fields, rotation_generators,
                                           stationarity, tangent_expansion, tangent_frames)
from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.states.incommensurate import IncommensurateStructure, rotation_matrix
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
_HALVINGS = 5                 # amplitudes or radii r, r/2, ..., r/2^4
_MOMENTUM_ANGLES = 360
_MAX_SAMPLED_DIMENSION = 6    # accidental subspaces up to this dimension are sampled

RISES, FALLS, FLAT, UNRESOLVED = "rises", "falls", "flat", "unresolved"


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
class HigherOrderCheck:
    """Beyond-harmonic test of zero modes that harmonic order leaves open (D54).

    Attributes
    ----------
    kind : {"amplitude", "momentum"}
        ``"amplitude"``: classical energy along the accidental zero modes, with
        every stiff direction relaxed, on the cell that folds them to ``k = 0``.
        ``"momentum"``: ``lambda_min(H(k0 + q))`` along the flat stiffness
        direction (``H(k)`` is exact in ``k``, so higher orders in ``q`` are
        harmonic in the amplitude).
    fractional : (n, 2) array
        Zero modes covered, in the magnetic reciprocal basis.
    spins : int
        Spins in the cell (amplitude) or of the magnetic cell (momentum).
    directions : int
        Sampled directions in the accidental subspace (amplitude) or angles
        of ``q`` (momentum).
    outcome : {"rises", "falls", "flat", "unresolved"}
        Sign of the leading change at the smallest resolved amplitude:
        ``"flat"`` if no change exceeds the floor up to the largest amplitude
        in some direction and none falls; ``"unresolved"`` if the leading
        power could not be read off from two consecutive halvings.
    order : float
        Leading power ``p`` of the change, ``log2(f(2a) / f(a))`` (NaN if not
        resolved).
    lowest : float
        Lowest leading change, relative to the stability scale.
    amplitude : float
        Largest root-mean-square angle per spin (rad) or radius of ``q``
        (inverse length) sampled.
    """

    kind: str
    fractional: np.ndarray
    spins: int
    directions: int
    outcome: str
    order: float
    lowest: float
    amplitude: float


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
    higher_order : tuple of HigherOrderCheck
        Beyond-harmonic tests of the zero modes harmonic order leaves open
        (empty if there are none or they were not requested).
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
    higher_order: Tuple[HigherOrderCheck, ...] = ()

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
        for c in self.higher_order:
            text += (f"; beyond harmonic ({c.kind}) at q={np.round(c.fractional, 6).tolist()}: "
                     f"{c.outcome}" + (f", order {c.order:.2f}" if np.isfinite(c.order) else ""))
        gradient = self.settings.get("pitch_gradient")
        tolerance = self.settings.get("stationarity_tolerance", 0.0)
        if gradient is not None and np.linalg.norm(gradient) > tolerance:
            text += (f"; spiral pitch not at an energy extremum (|dE/dQ| = "
                     f"{np.linalg.norm(gradient):.3g} E0): a twisted spiral")
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
                            "higher_order": list(self.higher_order),
                            "summary": self.summary()})


def local_stability(model: SpinModel, state,
                    conditions: Optional[ExternalConditions] = None, *,
                    mesh: int = STABILITY_MESH, seeds: int = STABILITY_SEEDS,
                    ball_radius: float = STABILITY_BALL_RADIUS,
                    zero_tolerance: float = STABILITY_ZERO_TOLERANCE,
                    stiffness_tolerance: float = STABILITY_STIFFNESS_TOLERANCE,
                    stationarity_tolerance: float = LSWT_STATIONARITY_TOLERANCE,
                    higher_order: bool = True) -> StabilityReport:
    """Test whether a classical state is a local minimum over continuous momenta.

    Parameters
    ----------
    model : SpinModel
    state : SpinState or IncommensurateStructure
        Commensurate classical state on its magnetic supercell, or a single-Q
        spiral (D55). A spiral is tested in its rotating frame
        (:func:`~spintoolkit.methods.lswt.spiral.rotating_frame_model`), whose
        ``H(k)`` is the exact second variation for any pitch; momenta in the
        report are then rotating-frame momenta in the primitive reciprocal
        basis, and lab rotations about axes perpendicular to the spiral axis
        are Goldstone modes at ``k = +-Q``. ``settings["pitch_gradient"]``
        holds ``dE/dQ``: without those perpendicular symmetries a spiral of
        the wrong pitch is a twisted state that can be locally stable, so
        the verdict does not say the pitch minimizes the energy.
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
        Largest relative torque ``max_i |S_i x h_i| / max_i S_i |h_i|`` accepted (D68).
    higher_order : bool
        Test zero modes that harmonic order leaves open beyond harmonic order
        (D54) and let the result decide between ``"marginal"``, ``"stable"``
        and ``"unstable"``. With ``False`` the verdict is the harmonic one of
        D51.

    Returns
    -------
    StabilityReport
    """
    conditions = conditions or ExternalConditions()
    spiral: Optional[IncommensurateStructure] = None
    goldstone_p: List[np.ndarray] = []
    perpendicular = np.zeros((3, 0))
    if isinstance(state, IncommensurateStructure):
        from spintoolkit.methods.lswt.spiral import rotating_frame_model
        from spintoolkit.observables.zero_modes import _symmetry_axes

        spiral = state
        n = spiral.rotation_axis
        axes = _symmetry_axes(model, conditions)
        if axes.size:
            u, sv, _ = np.linalg.svd((np.eye(3) - np.outer(n, n)) @ axes, full_matrices=False)
            perpendicular = u[:, sv > 1e-8]
        from spintoolkit.methods.lswt.spiral import spiral_energy_gradient
        pitch_gradient = spiral_energy_gradient(model, spiral, conditions)
        model, state = rotating_frame_model(model, spiral, conditions)
        if perpendicular.shape[1]:                          # lab rotations about axes _|_ n
            q = np.mod(spiral.wave_vector, 1.0)
            goldstone_p = [q, np.mod(-q, 1.0)]
    elif not isinstance(state, SpinState):
        raise TypeError("local_stability needs a SpinState or a single-Q "
                        "IncommensurateStructure")
    else:
        validate_spin_state(state, model)
    settings = {"mesh": int(mesh), "seeds": int(seeds), "ball_radius": float(ball_radius),
                "zero_tolerance": float(zero_tolerance),
                "stiffness_tolerance": float(stiffness_tolerance),
                "stationarity_tolerance": float(stationarity_tolerance),
                "higher_order": bool(higher_order)}
    if spiral is not None:
        # dE/dQ of the lab model: a twisted spiral can be locally stable at a pitch
        # that is not an energy extremum (only without the perpendicular symmetries).
        settings["pitch_gradient"] = np.asarray(pitch_gradient, dtype=float).tolist()
    torque, relative_torque = stationarity(model, state, conditions)
    nan2 = np.full(2, np.nan)
    if relative_torque > stationarity_tolerance:
        return StabilityReport(NOT_STATIONARY, (f"max torque {torque:.3g} E0 (relative "
                                                f"{relative_torque:.3g}) exceeds "
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
    for g in goldstone_p:                                   # spiral: lab Goldstones at +-Q
        if abs(lowest(g)) <= zero_tolerance:
            add_centre(g)

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
        at_q = any(distance(centre, g) < 1e-9 for g in goldstone_p)
        expansions.append(_expand(hamiltonian, to_k(centre), centre, scale, length, radius,
                                  zero_tolerance, stiffness_tolerance, at_origin,
                                  origin_counts, at_q))
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
    checks: List[HigherOrderCheck] = []
    if not unstable:
        if exhausted:
            reasons.append(f"zero modes are not isolated: still finding new ones after "
                           f"{_MAX_ROUNDS} rounds (a line or surface of zero modes?)")
        open_reasons: List[str] = []
        flat = [z for z in expansions if z.status == "flat"]
        accidental = [z for z in expansions if z.origin != "goldstone"]
        for z in flat:
            open_reasons.append(f"flat stiffness direction at q="
                                f"{np.round(z.fractional, 6).tolist()} (higher order decides)")
        for z in accidental:
            open_reasons.append(f"{z.origin} zero mode at q="
                                f"{np.round(z.fractional, 6).tolist()} (higher order decides)")
        if higher_order and open_reasons and not exhausted:
            checks = [_momentum_check(hamiltonian, z, scale, radius, zero_tolerance,
                                      len(hamiltonian.site_keys)) for z in flat]
            if accidental:
                checks.append(_amplitude_check(model, state, conditions,
                                               np.array([z.fractional for z in accidental]),
                                               scale, spiral, perpendicular))
            open_reasons = _higher_order_reasons(checks)
            unstable = any(c.outcome == FALLS for c in checks)
        reasons.extend(open_reasons)
    verdict = UNSTABLE if unstable else MARGINAL if reasons else STABLE
    return StabilityReport(verdict, tuple(reasons), float(best_value), to_k(best_p),
                           np.asarray(best_p, dtype=float), scale, length, tuple(expansions),
                           torque, settings, tuple(checks))


def _higher_order_reasons(checks: List[HigherOrderCheck]) -> List[str]:
    """Reasons for a verdict other than ``"stable"`` from the beyond-harmonic tests."""
    out = []
    for c in checks:
        where = f"q={np.round(c.fractional, 6).tolist()}"
        if c.kind == "amplitude":
            what = f"classical energy along the accidental zero modes at {where}"
            size = f"{c.amplitude:.3g} rad"
        else:
            what = f"lambda_min(H(k0 + q)) along the flat stiffness direction at {where}"
            size = f"|q| = {c.amplitude:.3g}"
        if c.outcome == FALLS:
            out.append(f"{what} falls at order {c.order:.2f} ({c.lowest:.3g} relative)")
        elif c.outcome == FLAT:
            out.append(f"{what} stays flat within the floor up to {size} in some direction "
                       f"(a classically degenerate manifold: fluctuations decide, e.g. "
                       f"order by disorder)")
        elif c.outcome == UNRESOLVED:
            out.append(f"{what}: leading order not resolved up to {size} (higher order decides)")
    return out


def _leading(values: np.ndarray, floor: float) -> Tuple[str, float, float]:
    """Outcome, power and value of the leading change from values at halving amplitudes.

    ``values[j]`` belongs to amplitude ``a / 2^j``. The smallest amplitude
    whose value exceeds ``floor`` decides the sign; its power is read off
    against the next larger amplitude, which must exceed the floor with the
    same sign. Harmonic order vanishes here, so a power below 2.5 means the
    values are not yet asymptotic and the outcome is unresolved.
    """
    resolved = np.flatnonzero(np.abs(values) > floor)
    if resolved.size == 0:
        return FLAT, float("nan"), 0.0
    j = int(resolved[-1])
    if j == 0 or abs(values[j - 1]) <= floor or np.sign(values[j - 1]) != np.sign(values[j]):
        return UNRESOLVED, float("nan"), float(values[j])
    order = float(np.log2(values[j - 1] / values[j]))
    if order < 2.5:
        return UNRESOLVED, order, float(values[j])
    return (RISES if values[j] > 0 else FALLS), order, float(values[j])


def _combine(outcomes) -> Tuple[str, float, float]:
    """Outcome over sampled directions: any fall, else any unresolved, else any flat."""
    outcomes = list(outcomes)
    for kind in (FALLS, UNRESOLVED, FLAT, RISES):
        chosen = [o for o in outcomes if o[0] == kind]
        if chosen:
            outcome, order, value = min(chosen, key=lambda o: o[2])
            return outcome, order, value
    return FLAT, float("nan"), 0.0


def _momentum_check(hamiltonian, z: ZeroModeExpansion, scale, radius, zero_tolerance,
                    spins) -> HigherOrderCheck:
    """``lambda_min(H(k0 + q)) / scale`` minimized over the direction of ``q`` at halving |q|."""
    angles = np.arange(_MOMENTUM_ANGLES) * 2 * np.pi / _MOMENTUM_ANGLES
    units = np.column_stack([np.cos(angles), np.sin(angles)])

    def lowest(r, theta) -> float:
        k = z.k + r * np.array([np.cos(theta), np.sin(theta)])
        return float(np.linalg.eigvalsh(hamiltonian.at(k)[0])[0] / scale)

    values = []
    for j in range(_HALVINGS):
        r = radius / 2 ** j
        sampled = _lowest_eigenvalues(hamiltonian, z.k + r * units) / scale
        i = int(np.argmin(sampled))
        h = 2 * np.pi / _MOMENTUM_ANGLES
        opt = minimize_scalar(lambda t: lowest(r, t), bounds=(angles[i] - h, angles[i] + h),
                              method="bounded", options={"xatol": 1e-12})
        values.append(min(float(sampled[i]), float(opt.fun)))
    outcome, order, value = _leading(np.array(values), zero_tolerance)
    return HigherOrderCheck("momentum", np.atleast_2d(z.fractional), int(spins),
                            _MOMENTUM_ANGLES, outcome, order, value, float(radius))


def _folded(model: SpinModel, state: SpinState, fractional: np.ndarray) -> Optional[SpinState]:
    """The state on the supercell that folds every ``q`` of ``fractional`` to ``k = 0``.

    A zero mode at ``q = p1 b1 + p2 b2`` (magnetic reciprocal basis) is at
    ``k = 0`` of the cell with rows ``n1 A1, n2 A2`` when ``n_i p_i`` are
    integers. ``None`` if some ``p_i`` has no denominator up to
    ``STABILITY_FOLD_DENOMINATOR`` or the cell would exceed
    ``STABILITY_FOLD_SPINS`` spins.
    """
    n = [1, 1]
    for p in np.atleast_2d(fractional):
        for i in (0, 1):
            f = Fraction(float(p[i])).limit_denominator(STABILITY_FOLD_DENOMINATOR)
            if abs(float(f) - p[i]) > 1e-8:
                return None
            n[i] = int(np.lcm(n[i], f.denominator))
    if n == [1, 1]:
        return state
    if n[0] * n[1] * state.num_cells * len(model.site_ids) > STABILITY_FOLD_SPINS:
        return None
    supercell = np.diag(n) @ state.supercell
    return SpinState.from_function(model, supercell, lambda site, cell: state.direction(site, cell),
                                   dict(state.provenance))


def _amplitude_check(model: SpinModel, state: SpinState, conditions: ExternalConditions,
                     fractional: np.ndarray, scale: float,
                     spiral: Optional[IncommensurateStructure] = None,
                     perpendicular: np.ndarray = np.zeros((3, 0))) -> HigherOrderCheck:
    """Classical energy per spin along accidental zero modes, stiff directions relaxed.

    On the folded cell the Hessian null space at ``k = 0`` minus the
    generators of exact global rotations is the accidental subspace. Along
    each sampled unit direction ``u`` of it and its opposite, the energy
    change per spin ``f(a) = min_y E(a sqrt(n) u + y) - E0`` over the stiff
    directions ``y`` (L-BFGS in the fixed tangent chart) is taken at halving
    root-mean-square angles ``a`` per spin;
    its leading term decides (:func:`_leading`). Relaxing the stiff
    directions makes the quartic coefficient the effective one, with the
    cubic couplings to stiff modes included.

    For a spiral, ``model`` and ``state`` are the rotating frame; a lab
    rotation about an axis ``m`` perpendicular to ``n`` acts on the spin of
    cell ``R`` about ``R_n(-phi_R) m`` and is removed with the rest.
    """
    from spintoolkit.observables.zero_modes import _symmetry_axes

    floor = STABILITY_AMPLITUDE_FLOOR
    folded = _folded(model, state, fractional)
    if folded is None:
        return HigherOrderCheck("amplitude", fractional, 0, 0, UNRESOLVED, float("nan"),
                                float("nan"), 0.0)
    expansion = tangent_expansion(model, folded, conditions)
    compiled = _compile(model, folded, conditions)
    n = len(compiled.keys)
    w, v = np.linalg.eigh(expansion.hessian * n)            # total energy, E0 per rad^2
    null = np.abs(w) <= 1e-8 * max(float(np.max(np.abs(w))), scale)
    N, stiff = v[:, null], v[:, ~null]
    axes = _symmetry_axes(model, conditions)
    G = rotation_generators(expansion) @ axes if axes.size else np.zeros((2 * n, 0))
    if spiral is not None and perpendicular.shape[1]:
        columns = []
        for m in perpendicular.T:
            column = []
            for (site, cell), d, frame in zip(compiled.keys, expansion.directions,
                                              expansion.frames):
                axis = rotation_matrix(spiral.rotation_axis, -spiral.phase(cell)) @ m
                column.extend(np.cross(axis, d) @ e for e in frame)
            columns.append(column)
        G = np.column_stack([G, np.array(columns).T])
    if G.shape[1] and N.shape[1]:
        u, sv, _ = np.linalg.svd(G, full_matrices=False)
        G = u[:, sv > 1e-8 * max(1.0, float(sv.max(initial=0.0)))]
        rest = N - G @ (G.T @ N)
        u, sv, _ = np.linalg.svd(rest, full_matrices=False)
        A = u[:, sv > 0.5]
    else:
        A = N
    d = A.shape[1]
    if d == 0 or d > _MAX_SAMPLED_DIMENSION:
        return HigherOrderCheck("amplitude", fractional, n, 0,
                                FLAT if d == 0 else UNRESOLVED, float("nan"),
                                float("nan"), STABILITY_AMPLITUDE)
    if d == 1:
        units = [A[:, 0]]
    elif d == 2:
        units = [np.cos(t) * A[:, 0] + np.sin(t) * A[:, 1]
                 for t in np.arange(8) * np.pi / 8]
    else:
        units = [A[:, i] for i in range(d)]
        units += [(A[:, i] + s * A[:, j]) / np.sqrt(2)
                  for i in range(d) for j in range(i + 1, d) for s in (1, -1)]
    start, frames = compiled.directions, tangent_frames(compiled.directions)

    def energy(x):
        vectors = start + np.einsum("ia,iax->ix", x.reshape(n, 2), frames)
        radii = np.linalg.norm(vectors, axis=1)
        directions = vectors / radii[:, None]
        e, fields = _energy_and_fields(compiled, directions)
        force = -compiled.lengths[:, None] * fields
        force -= np.einsum("ix,ix->i", force, directions)[:, None] * directions
        return e / n, np.einsum("ix,iax->ia", force, frames).ravel() / radii.repeat(2) / n

    e0 = energy(np.zeros(2 * n))[0]

    def change(a, unit):
        """Energy change per spin at ``a * unit``, relaxed in the stiff directions."""
        def relaxed(y):
            e, g = energy(a * unit + stiff @ y)
            return e - e0, stiff.T @ g

        if stiff.shape[1]:
            opt = minimize(relaxed, np.zeros(stiff.shape[1]), jac=True, method="L-BFGS-B",
                           options={"gtol": 1e-15, "ftol": 0.0, "maxiter": 2000})
            return float(opt.fun)
        return relaxed(np.zeros(0))[0]

    def series(unit, sign):
        # rms angle per spin a, a/2, ...
        return _leading(np.array([change(sign * STABILITY_AMPLITUDE * np.sqrt(n) / 2 ** j, unit)
                                  for j in range(_HALVINGS)]) / scale, floor)

    outcomes = [series(unit, sign) for unit in units for sign in (1.0, -1.0)]
    samples = 2 * len(units)
    if d > 1 and all(o[0] == RISES for o in outcomes):
        # Sampled directions can all rise while the leading form is negative between
        # them (a saddle): minimize over the direction in the null space at the smallest
        # amplitude, then classify that direction as the samples (D75).
        a = STABILITY_AMPLITUDE * np.sqrt(n) / 2 ** (_HALVINGS - 1)
        along = lambda y: change(a, A @ y / np.linalg.norm(y))  # noqa: E731
        y0 = A.T @ units[int(np.argmin([o[2] for o in outcomes[::2]]))]
        opt = minimize(along, y0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-16, "maxiter": 200 * d})
        unit = A @ opt.x / np.linalg.norm(opt.x)
        outcomes += [series(unit, 1.0), series(unit, -1.0)]
        samples += 2
    outcome, order, value = _combine(outcomes)
    return HigherOrderCheck("amplitude", fractional, n, samples, outcome, order, value,
                            STABILITY_AMPLITUDE)


def _lowest_eigenvalues(hamiltonian: QuadraticBoseHamiltonian, k: np.ndarray,
                        chunk: int = 256) -> np.ndarray:
    return np.concatenate([np.linalg.eigvalsh(hamiltonian.at(k[i:i + chunk]))[:, 0]
                           for i in range(0, len(k), chunk)])


def _expand(hamiltonian, k0, p0, scale, length, radius, zero_tolerance, stiffness_tolerance,
            at_origin, origin_counts, at_spiral_q=False) -> ZeroModeExpansion:
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
    if at_spiral_q and not at_origin:
        # Rotating frame of a spiral: lab rotations about the two axes perpendicular to n
        # give one zero mode at +Q and its partner at -Q.
        origin = "goldstone" if nullity <= 1 else "accidental (k != 0)"
    elif not at_origin:
        origin = "accidental (k != 0)"
    else:
        broken = int(origin_counts.get("broken_symmetries", 0)) if origin_counts else 0
        origin = "goldstone" if nullity <= broken else "accidental"
    return ZeroModeExpansion(np.asarray(k0, dtype=float), np.asarray(p0, dtype=float), nullity,
                             origin, float(linear_relative), rho_min, theta_min,
                             float(rho_relative), ball, status)
