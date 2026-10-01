"""Zero-point state selection on the classical ground-state manifold (D17, D19, D28).

The zero-point energy E_qm is compared only between classical ground states.
Off the manifold the torques do not vanish, the linear boson terms of LSWT
remain, and minimizing the O(S^2) classical energy E_cl together with the
O(S^0) E_qm moves the state by O(1/S), beyond LSWT order (D18). Exact
symmetries of the Hamiltonian leave E_qm constant, so a selection happens only
along accidental directions, where E_cl alone is flat. The orbit axis is
therefore read from E_cl, not from the symmetry of the Hamiltonian.

Steps of :func:`select_on_manifold`:

1. Refine the classical state with the analytic gradient and Hessian
   (:func:`~spintoolkit.methods.classical.refine_classical`).
2. Hessian of E_cl in tangent coordinates; project the global rotation
   generators ``n x n_i`` on it and find the flat rotation axis ``n``.
3. Relaxed soft path (D28): at equally spaced rotation angles, ``R_n(phi)``
   of the refined state is relaxed classically in every tangent direction
   except the orbit tangent. Only the torque along the orbit remains; a
   torque transverse to the spins does not change H2, so dropping the linear
   term is the constrained one-loop calculation.
4. One-loop effective potential ``Gamma(phi) = E_cl + E_zp`` on that path: a
   least-squares Fourier fit up to a maximum harmonic and a bounded 1D
   minimization. The minimum of Gamma is the selected state; the pure E_cl
   and E_qm minima are kept as candidates with the shifts from them. For a
   flat classical orbit Gamma = const + E_zp, which is D17. ``dGamma/dphi = 0``
   balances the classical and zero-point torques along the soft direction.
5. Validity: the soft coordinate must be softer than every other mode
   (classical softness below one, otherwise ``no_degeneracy``), the other
   modes must be stable at the refined state and the path must exist
   (otherwise ``not_soft``), and the curvature of Gamma over the hard
   stiffness (adiabatic ratio) above ``adiabatic_warning`` gives a warning.

Two criteria modes (D19):

``physics`` (default)
    Flatness is judged against the numerical accuracy of the refined state:
    the remaining Newton correction ``delta`` sets the curvature floor
    ``accuracy_factor * |H| * (delta + roundoff_factor * eps)``. A resolved
    classical curvature is handled by the effective potential (D28), with no
    threshold between classical pinning and quantum selection. "No selection"
    requires the rotation
    to be an exact symmetry of every term (``[K_n, J] = 0``,
    ``g^T b || n``); a non-symmetric orbit whose harmonic amplitude stays at
    the fit residual is "unresolved".
``fixed``
    Constant thresholds: a Hessian eigenvalue is null when
    ``w_min / w_next < gap_ratio``, a generator singular value is zero below
    ``rank_tolerance`` relative to the largest, and "no selection" means the
    harmonic amplitude is below ``max(residual_factor * residual,
    roundoff_factor * eps * |Gamma|)``; the orbit must be a null Hessian mode.

The defaults live in :mod:`spintoolkit.definitions.defaults`; every threshold
used is recorded in :attr:`SelectionResult.diagnostics`.

Only a single global rotation axis is supported. Null directions outside the
span of the global rotations (site-dependent axes, the accidental manifold of
the triangular antiferromagnet in a field) are reported as
``unsupported_manifold``.

Energies are dimensionless in the energy unit E0 of the model, per site.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
import warnings

import numpy as np
from scipy.optimize import minimize_scalar

from spintoolkit.definitions.defaults import (
    SELECTION_ACCURACY_FACTOR, SELECTION_ADIABATIC_WARNING, SELECTION_GAP_RATIO,
    SELECTION_MAX_HARMONIC, SELECTION_MODE, SELECTION_ORBIT_POINTS,
    SELECTION_PATH_TOLERANCE, SELECTION_RANK_TOLERANCE, SELECTION_RESIDUAL_FACTOR,
    SELECTION_ROUNDOFF_FACTOR)
from spintoolkit.methods.classical import (
    classical_energy, refine_classical, tangent_expansion)
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import BILINEAR, ZEEMAN, SpinModel

#: Verdicts of :func:`select_on_manifold`.
NO_DEGENERACY = "no_degeneracy"
SELECTED = "selected"
NO_SELECTION = "no_selection"
UNRESOLVED = "unresolved"
NOT_SOFT = "not_soft"
UNSUPPORTED_MANIFOLD = "unsupported_manifold"
AXIS_REQUIRED = "axis_required"

MODES = ("physics", "fixed")

EPS = np.finfo(float).eps

QuantumEnergy = Callable[[SpinModel, SpinState, ExternalConditions], float]


def _zero_point(quantum_energy, model, state, conditions, axis) -> float:
    """Call a provider, passing the orbit axis to those that accept it (``accepts_axis``)."""
    if getattr(quantum_energy, "accepts_axis", False):
        return float(quantum_energy(model, state, conditions, axis=axis))
    return float(quantum_energy(model, state, conditions))


@dataclass(frozen=True)
class SelectionCriteria:
    """Thresholds of the zero-point selection (D19).

    Parameters
    ----------
    mode : {"physics", "fixed"}
    gap_ratio : float
        Fixed mode: ``w_min / w_next`` below this marks a null Hessian mode.
    rank_tolerance : float
        Fixed mode: generator singular values below this times the largest
        count as zero.
    residual_factor : float
        A harmonic amplitude must exceed this times the fit residual.
    roundoff_factor : float
        Multiples of ``eps * |E|`` treated as round-off.
    accuracy_factor : float
        Physics mode: multiples of the estimated state accuracy.
    adiabatic_warning : float
        Warn when the curvature of Gamma over the hard stiffness exceeds this
        (the verdict does not change).
    path_tolerance : float
        Relative gradient tolerance of the soft-path relaxation.
    orbit_points : int
        Equally spaced orbit samples.
    max_harmonic : int
        Highest Fourier harmonic of the orbit fit; ``orbit_points`` must
        exceed ``2 * max_harmonic + 1`` so that the fit residual is defined.
    refine : bool
        Refine the classical state first (recommended).
    """

    mode: str = SELECTION_MODE
    gap_ratio: float = SELECTION_GAP_RATIO
    rank_tolerance: float = SELECTION_RANK_TOLERANCE
    residual_factor: float = SELECTION_RESIDUAL_FACTOR
    roundoff_factor: float = SELECTION_ROUNDOFF_FACTOR
    accuracy_factor: float = SELECTION_ACCURACY_FACTOR
    adiabatic_warning: float = SELECTION_ADIABATIC_WARNING
    path_tolerance: float = SELECTION_PATH_TOLERANCE
    orbit_points: int = SELECTION_ORBIT_POINTS
    max_harmonic: int = SELECTION_MAX_HARMONIC
    refine: bool = True

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.orbit_points <= 2 * self.max_harmonic + 1:
            raise ValueError("orbit_points must exceed 2 * max_harmonic + 1")
        if self.adiabatic_warning <= 0 or self.path_tolerance <= 0:
            raise ValueError("adiabatic_warning and path_tolerance must be positive")


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of :func:`select_on_manifold`.

    Attributes
    ----------
    verdict : str
        One of the module verdict constants.
    state : SpinState
        The selected state for ``selected``; otherwise the refined input state.
    axis : ndarray or None
        Unit rotation axis of the orbit.
    phi : float or None
        Rotation angle of ``state`` from the refined input state.
    candidates : mapping of str to SpinState
        Minima kept for inspection, e.g. ``classical`` and ``quantum`` for a
        competition.
    diagnostics : dict
        Numbers behind the verdict: Hessian spectrum, generator singular
        values, curvatures, harmonic amplitudes and fit residual, orbit
        energies, thresholds used and the E_qm provider description.
    criteria : SelectionCriteria
    message : str
        One-line explanation of the verdict.
    """

    verdict: str
    state: SpinState
    axis: Optional[np.ndarray]
    phi: Optional[float]
    candidates: Mapping[str, SpinState]
    diagnostics: Dict[str, Any]
    criteria: SelectionCriteria
    message: str


# ---------------------------------------------------------------------------
# Rotations and harmonics
# ---------------------------------------------------------------------------

def _cross_matrix(axis: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                     [-axis[1], axis[0], 0.0]])


def rotation_matrix(axis, angle: float) -> np.ndarray:
    """Rotation by ``angle`` about the unit vector ``axis`` (Rodrigues)."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = _cross_matrix(axis)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def rotate_state(state: SpinState, axis, angle: float,
                 provenance: Optional[Mapping[str, Any]] = None) -> SpinState:
    """Rotate every spin of ``state`` by ``angle`` about ``axis``."""
    R = rotation_matrix(axis, angle)
    return SpinState(state.model_ref, state.supercell,
                     {key: R @ vector for key, vector in state.directions.items()},
                     dict(state.provenance) if provenance is None else provenance)


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return axis * np.sign(axis[np.argmax(np.abs(axis))])


def fit_harmonics(phis: np.ndarray, values: np.ndarray, max_harmonic: int) -> Dict[str, Any]:
    """Least-squares fit ``c_0 + sum_m (a_m cos m phi + b_m sin m phi)``.

    Returns
    -------
    dict
        ``constant``, ``cos`` and ``sin`` (index ``m - 1``), ``amplitudes``
        ``sqrt(a_m^2 + b_m^2)`` and ``residual`` (max absolute residual).
    """
    m = np.arange(1, max_harmonic + 1)
    basis = np.column_stack([np.ones_like(phis), np.cos(np.outer(phis, m)),
                             np.sin(np.outer(phis, m))])
    offset = float(np.mean(values))
    coefficients, *_ = np.linalg.lstsq(basis, values - offset, rcond=None)
    residual = float(np.max(np.abs(basis @ coefficients - (values - offset))))
    cos, sin = coefficients[1:max_harmonic + 1], coefficients[max_harmonic + 1:]
    return {"constant": offset + float(coefficients[0]), "cos": cos.tolist(),
            "sin": sin.tolist(), "amplitudes": np.hypot(cos, sin).tolist(),
            "residual": residual}


def _series(fit: Mapping[str, Any], phi, derivative: int = 0, constant: bool = True):
    m = np.arange(1, len(fit["cos"]) + 1)
    angle = np.multiply.outer(phi, m)
    a, b = np.asarray(fit["cos"]), np.asarray(fit["sin"])
    if derivative == 0:
        # The oscillating part alone keeps full relative precision when the
        # variation (e.g. 1e-12) is far below the constant (e.g. 5e-2).
        return (fit["constant"] if constant else 0.0) + np.cos(angle) @ a + np.sin(angle) @ b
    if derivative == 2:
        return -(np.cos(angle) @ (m ** 2 * a) + np.sin(angle) @ (m ** 2 * b))
    raise ValueError("derivative must be 0 or 2")


def _fit_minimum(fit: Mapping[str, Any], orbit_points: int) -> float:
    """Global minimum of a fitted series: dense grid, then bounded refinement."""
    grid = np.linspace(0.0, 2 * np.pi, 64 * orbit_points, endpoint=False)
    start = grid[np.argmin(_series(fit, grid, constant=False))]
    step = grid[1] - grid[0]
    result = minimize_scalar(lambda x: float(_series(fit, x, constant=False)),
                             bounds=(start - step, start + step),
                             method="bounded", options={"xatol": 1e-12})
    return float(np.mod(result.x, 2 * np.pi))


# ---------------------------------------------------------------------------
# Symmetry and generators
# ---------------------------------------------------------------------------

def rotation_symmetry(model: SpinModel, conditions: ExternalConditions, axis,
                      tolerance: float) -> Tuple[bool, Dict[str, float]]:
    """Whether every rotation about ``axis`` leaves all terms invariant.

    Bilinear and onsite terms need ``[K_n, J] = 0`` with ``K_n`` the generator
    (``n x``); zeeman terms need ``g^T b`` parallel to ``n``. Norms are
    relative to the largest coefficient, compared with ``tolerance``.

    Returns
    -------
    (bool, dict)
        The verdict and the largest relative violation of each kind.
    """
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = _cross_matrix(axis)
    # Onsite matrices transform like exchange matrices under a global rotation.
    exchange = [t.coefficient for t in model.terms_of_kind(BILINEAR)]
    exchange += [t.coefficient for t in model.terms_of_kind("onsite")]
    scale = max([np.linalg.norm(J) for J in exchange] + [0.0])
    bilinear = max([np.linalg.norm(K @ J - J @ K) for J in exchange] + [0.0])
    fields = [t.coefficient.T @ conditions.field for t in model.terms_of_kind(ZEEMAN)]
    field_scale = max([np.linalg.norm(h) for h in fields] + [0.0])
    zeeman = max([np.linalg.norm(np.cross(axis, h)) for h in fields] + [0.0])
    violations = {"bilinear": bilinear / scale if scale else 0.0,
                  "zeeman": zeeman / field_scale if field_scale else 0.0}
    return max(violations.values()) <= tolerance, violations


def _generators(expansion) -> np.ndarray:
    """Tangent components of ``e_k x n_i`` for the three Cartesian axes: (2n, 3)."""
    return np.array([[np.cross(axis, n) @ e for axis in np.eye(3)]
                     for n, frame in zip(expansion.directions, expansion.frames) for e in frame])


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _null_count_by_gap(values: np.ndarray, gap_ratio: float) -> int:
    # Values at round-off level carry no ratio information (0 / 1e-17 is not a gap).
    values = np.maximum(values, EPS * (values.max() if len(values) else 0.0))
    for k in range(len(values) - 1):
        if values[k + 1] > 0 and values[k] / values[k + 1] < gap_ratio:
            return k + 1
    return 0


def _describe(quantum_energy) -> Dict[str, Any]:
    describe = getattr(quantum_energy, "describe", None)
    return describe() if callable(describe) else {"provider": repr(quantum_energy)}


def select_on_manifold(model: SpinModel, state: SpinState,
                       conditions: Optional[ExternalConditions],
                       quantum_energy: QuantumEnergy, axis=None,
                       criteria: SelectionCriteria = SelectionCriteria()) -> SelectionResult:
    """Select among degenerate classical states by the zero-point energy.

    Parameters
    ----------
    model : SpinModel
    state : SpinState
        A classical minimum, e.g. from a global search.
    conditions : ExternalConditions or None
    quantum_energy : callable
        ``quantum_energy(model, state, conditions) -> float``, the zero-point
        energy per site in E0; see :class:`LSWTZeroPointEnergy`. A provider
        with ``accepts_axis = True`` is called with the keyword ``axis`` (the
        orbit axis, whose uniform rotation is the constrained coordinate). An
        optional ``describe()`` method is recorded in the diagnostics.
    axis : array_like, optional
        Rotation axis of the orbit. Required when several rotation directions
        are flat (e.g. an SO(3) degenerate state); otherwise found from the
        Hessian.
    criteria : SelectionCriteria

    Returns
    -------
    SelectionResult
    """
    conditions = conditions or ExternalConditions()
    physics = criteria.mode == "physics"
    if criteria.refine:
        state = refine_classical(model, state, conditions)
    expansion = tangent_expansion(model, state, conditions)
    H, g = expansion.hessian, expansion.gradient
    w, V = np.linalg.eigh(H)
    scale = float(np.max(np.abs(w))) if len(w) else 0.0
    order = np.argsort(np.abs(w))
    sorted_w = np.abs(w[order])

    # Accuracy of the state: remaining Newton correction off the flat directions.
    keep = np.abs(w) > np.sqrt(EPS) * scale
    delta = float(np.max(np.abs(V[:, keep] @ ((V[:, keep].T @ g) / w[keep])), initial=0.0))
    accuracy = delta + criteria.roundoff_factor * EPS
    curvature_floor = criteria.accuracy_factor * scale * accuracy
    rank_tolerance = (criteria.accuracy_factor * accuracy if physics
                      else criteria.rank_tolerance)
    if physics:
        null_count = int(np.sum(sorted_w <= curvature_floor))
    else:
        null_count = _null_count_by_gap(sorted_w, criteria.gap_ratio)
    flat_limit = (curvature_floor if physics
                  else criteria.gap_ratio * (sorted_w[null_count] if null_count < len(sorted_w)
                                             else scale))

    G = _generators(expansion)
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    rank = int(np.sum(s > rank_tolerance * s.max())) if s.max() > 0 else 0
    block_w, block_y = np.linalg.eigh(U[:, :rank].T @ H @ U[:, :rank])
    block_axes = [_canonical_axis(Vt[:rank].T @ (block_y[:, k] / s[:rank]))
                  for k in range(rank)]
    flat_rotations = [k for k in range(rank) if abs(block_w[k]) <= flat_limit]

    diagnostics: Dict[str, Any] = {
        "mode": criteria.mode, "thresholds": asdict(criteria),
        "classical_energy": expansion.energy, "max_torque": expansion.max_torque,
        "refinement": dict(state.provenance).get("refinement"),
        "state_accuracy": delta, "curvature_floor": curvature_floor,
        "rank_tolerance_used": rank_tolerance,
        "hessian_abs_eigenvalues": sorted_w.tolist(), "null_count": null_count,
        "gap_ratio": float(sorted_w[0] / sorted_w[1]) if len(sorted_w) > 1 and sorted_w[1] > 0 else None,
        "generator_relative_singular_values": (s / s.max()).tolist() if s.max() > 0 else [],
        "generator_rank": rank,
        "rotation_curvatures": [float(x) for x in block_w],
        "flat_rotation_count": len(flat_rotations),
    }

    def result(verdict, message, selected=None, axis_=None, phi=None, candidates=None):
        return SelectionResult(verdict, selected or state, axis_, phi, candidates or {},
                               diagnostics, criteria, message)

    if null_count > len(flat_rotations):
        null_vectors = V[:, order[:null_count]]
        Q = U[:, :rank]
        diagnostics["null_norm_outside_rotation_span"] = [
            float(np.linalg.norm(v - Q @ (Q.T @ v))) for v in null_vectors.T]
        return result(UNSUPPORTED_MANIFOLD,
                      f"{null_count} flat Hessian directions but {len(flat_rotations)} flat "
                      "global rotations; site-dependent or non-rotational degeneracy is not "
                      "supported")

    if axis is None:
        if len(flat_rotations) > 1:
            return result(AXIS_REQUIRED, f"{len(flat_rotations)} flat rotation directions; "
                          "give the orbit axis explicitly")
        if flat_rotations:
            axis = block_axes[flat_rotations[0]]
        elif physics and rank:
            axis = block_axes[int(np.argmin(np.abs(block_w)))]
        else:
            return result(NO_DEGENERACY, "no flat Hessian direction (fixed mode gap ratio)"
                          if not physics else "no rotation moves the state")
    axis = _canonical_axis(np.asarray(axis, dtype=float))
    diagnostics["axis"] = axis.tolist()

    tangent = G @ axis
    if np.linalg.norm(tangent) <= rank_tolerance * s.max():
        return result(NO_SELECTION, "the state is invariant under rotations about the axis",
                      axis_=axis)
    c_cl = float(tangent @ H @ tangent)
    tangent_norm = float(np.linalg.norm(tangent))
    hard = _hard_stiffness(H, tangent)
    softness = (c_cl / tangent_norm ** 2) / hard if hard > 0 else np.inf
    diagnostics.update({"C_cl": c_cl, "hard_stiffness": hard, "classical_softness": softness})
    hessian_flat = abs(c_cl) <= flat_limit
    diagnostics["hessian_flat_along_orbit"] = bool(hessian_flat)
    if not hessian_flat and not physics:
        return result(NO_DEGENERACY, "the orbit direction is not a null Hessian mode", axis_=axis)
    if hard < -flat_limit:
        return result(NOT_SOFT, "a hard mode is unstable at the refined state (a classical "
                      "saddle, not a minimum); start from a classical minimum", axis_=axis)
    if not hessian_flat and softness >= 1 - 1e-6:
        return result(NO_DEGENERACY, "no soft coordinate: the classical curvature along the "
                      "orbit is not below the stiffness of the other modes", axis_=axis)

    symmetric, violations = rotation_symmetry(
        model, conditions, axis, criteria.accuracy_factor * accuracy)
    diagnostics["symmetry_violation"] = violations
    diagnostics["exact_symmetry"] = bool(symmetric)
    step = 2 * np.pi / criteria.orbit_points
    if physics and symmetric and hessian_flat:
        phis = np.arange(criteria.orbit_points) * step
        e_cl = np.array([classical_energy(model, rotate_state(state, axis, p), conditions)
                         for p in phis])
        diagnostics["orbit"] = {"phi": phis.tolist(), "E_cl": e_cl.tolist(),
                                "E_cl_span": float(np.ptp(e_cl))}
        return result(NO_SELECTION, "the orbit is an exact symmetry of every term; E_qm is "
                      "constant", axis_=axis)

    # Relaxed soft path and the one-loop effective potential Gamma = E_cl + E_zp (D28).
    path = [soft_path_point(model, state, conditions, axis, p, criteria.path_tolerance * scale)
            for p in np.arange(criteria.orbit_points) * step]
    path_info = {"max_transverse_gradient": max(x.transverse_gradient for x in path),
                 "min_hard_stiffness": min(x.hard_stiffness for x in path),
                 "tolerance": criteria.path_tolerance * scale}
    diagnostics["path"] = path_info
    if not all(x.converged for x in path):
        return result(NOT_SOFT, "the relaxed soft path does not exist (relaxation off the orbit "
                      "does not converge or a hard mode is unstable)", axis_=axis)
    phis = np.array([x.phi for x in path])
    e_cl = np.array([classical_energy(model, x.state, conditions) for x in path])
    e_qm = np.array([_zero_point(quantum_energy, model, x.state, conditions, axis) for x in path])
    diagnostics["quantum_energy_provider"] = _describe(quantum_energy)
    gamma = e_cl + e_qm
    rigid_cl = np.array([classical_energy(model, rotate_state(state, axis, p), conditions)
                         for p in np.arange(criteria.orbit_points) * step])
    path_info["relaxation_energy_max"] = float(np.max(rigid_cl - e_cl))
    fit_qm = fit_harmonics(phis, e_qm, criteria.max_harmonic)
    fit_cl = fit_harmonics(phis, e_cl, criteria.max_harmonic)
    fit_gamma = fit_harmonics(phis, gamma, criteria.max_harmonic)
    amplitude_qm = float(max(fit_qm["amplitudes"]))
    amplitude = float(max(fit_gamma["amplitudes"]))
    threshold = max(criteria.residual_factor * fit_gamma["residual"],
                    criteria.roundoff_factor * EPS * float(np.max(np.abs(gamma))))
    diagnostics["orbit"] = {
        "phi": phis.tolist(), "E_cl": e_cl.tolist(), "E_qm": e_qm.tolist(),
        "Gamma": gamma.tolist(), "E_cl_span": float(np.ptp(e_cl)),
        "E_qm_span": float(np.ptp(e_qm)), "Gamma_span": float(np.ptp(gamma)),
        "E_cl_fit": fit_cl, "E_qm_fit": fit_qm, "Gamma_fit": fit_gamma,
        "E_qm_amplitude": amplitude_qm, "Gamma_amplitude": amplitude,
        "E_qm_dominant_harmonic": int(np.argmax(fit_qm["amplitudes"]) + 1),
        "E_qm_resolution_threshold": max(
            criteria.residual_factor * fit_qm["residual"],
            criteria.roundoff_factor * EPS * float(np.max(np.abs(e_qm)))),
        "Gamma_resolution_threshold": threshold}

    if amplitude <= threshold:
        if physics:
            return result(UNRESOLVED, "Gamma = E_cl + E_qm varies along the soft path by less "
                          "than the fit resolution; increase the mesh or check the provider",
                          axis_=axis)
        return result(NO_SELECTION, "harmonic amplitude of Gamma below the threshold", axis_=axis)

    phi_gamma = _fit_minimum(fit_gamma, criteria.orbit_points)
    c_gamma = float(_series(fit_gamma, phi_gamma, 2))
    fine = np.linspace(0.0, 2 * np.pi, 64 * criteria.orbit_points, endpoint=False)
    fitted = _series(fit_gamma, fine, constant=False)
    is_min = (fitted <= np.roll(fitted, 1)) & (fitted <= np.roll(fitted, -1))
    equivalent = fine[is_min & (fitted - fitted.min() <= threshold)]
    tol = criteria.path_tolerance * scale
    chosen = soft_path_point(model, state, conditions, axis, phi_gamma, tol, target=True)
    selected = SpinState(chosen.state.model_ref, chosen.state.supercell, chosen.state.directions,
                         {**dict(state.provenance),
                          "selection": {"method": "D28 effective potential", "axis": axis.tolist(),
                                        "phi": phi_gamma, "mode": criteria.mode}})
    candidates = {"selected": selected}
    shifts = {}
    for name, fit in (("quantum", fit_qm), ("classical", fit_cl)):
        floor = max(criteria.residual_factor * fit["residual"],
                    criteria.roundoff_factor * EPS * float(np.max(np.abs(gamma))))
        if max(fit["amplitudes"]) <= floor:
            continue
        # nearest of the equivalent minima (e.g. sixfold E_qm) to the minimum of Gamma
        phi_x = _nearest_equivalent_minimum(fit, phi_gamma, floor, criteria.orbit_points)
        candidates[name] = soft_path_point(model, state, conditions, axis, phi_x, tol,
                                           target=True).state
        harmonic = int(np.argmax(fit["amplitudes"]) + 1)
        shift = _wrap(phi_gamma - phi_x)
        shifts[name] = {"phi": phi_x, "shift": shift, "dominant_harmonic": harmonic,
                        "fraction_of_period": shift / (2 * np.pi / harmonic)}
    adiabatic = ((c_gamma / chosen.tangent_norm ** 2) / chosen.hard_stiffness
                 if chosen.hard_stiffness > 0 else np.inf)
    diagnostics.update({
        "phi_gamma": phi_gamma, "C_gamma": c_gamma, "adiabatic_ratio": adiabatic,
        "minima": shifts, "equivalent_minima": equivalent.tolist(),
        "phi_qm": shifts.get("quantum", {}).get("phi"),
        "C_qm": (float(_series(fit_qm, shifts["quantum"]["phi"], 2)) if "quantum" in shifts else None),
        "E_qm_selected": _zero_point(quantum_energy, model, selected, conditions, axis),
        "E_cl_selected": classical_energy(model, selected, conditions)})
    if adiabatic > criteria.adiabatic_warning:
        warnings.warn(f"adiabatic ratio {adiabatic:.3g} > {criteria.adiabatic_warning}: the soft "
                      "coordinate is weakly separated from the other modes", UserWarning, stacklevel=2)
    message = f"Gamma = E_cl + E_qm is minimal at phi = {phi_gamma:.6f} about the axis"
    if "quantum" in shifts:
        message += f"; {shifts['quantum']['shift']:+.3g} rad from the E_qm minimum"
    if "classical" in shifts:
        message += f", {shifts['classical']['shift']:+.3g} rad from the E_cl minimum"
    return result(SELECTED, message, selected=selected, axis_=axis, phi=phi_gamma,
                  candidates=candidates)


# ---------------------------------------------------------------------------
# Relaxed soft path (D28)
# ---------------------------------------------------------------------------

def _wrap(angle: float) -> float:
    return float(np.mod(angle + np.pi, 2 * np.pi) - np.pi)


def _nearest_equivalent_minimum(fit, phi: float, tolerance: float, orbit_points: int) -> float:
    """The minimum of a fitted series, among those within ``tolerance`` of the lowest, nearest to phi."""
    grid = np.linspace(0.0, 2 * np.pi, 64 * orbit_points, endpoint=False)
    values = _series(fit, grid, constant=False)
    is_min = (values <= np.roll(values, 1)) & (values <= np.roll(values, -1))
    candidates = grid[is_min & (values - values.min() <= tolerance)]
    start = candidates[np.argmin(np.abs([_wrap(phi - c) for c in candidates]))]
    step = grid[1] - grid[0]
    result = minimize_scalar(lambda x: float(_series(fit, x, constant=False)),
                             bounds=(start - step, start + step),
                             method="bounded", options={"xatol": 1e-12})
    return float(np.mod(result.x, 2 * np.pi))


def _hard_stiffness(H: np.ndarray, tangent: np.ndarray) -> float:
    """Lowest Hessian eigenvalue on the complement of the orbit tangent."""
    unit = tangent / np.linalg.norm(tangent)
    _, _, vt = np.linalg.svd(unit[None, :])
    P = vt[1:].T
    return float(np.min(np.linalg.eigvalsh(P.T @ H @ P)))


def rotation_angle(reference: SpinState, state: SpinState, axis) -> float:
    """Least-squares rotation angle about ``axis`` that maps ``reference`` onto ``state``."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    cross = dot = 0.0
    for key, r in reference.directions.items():
        v = state.directions[key]
        r_perp, v_perp = r - (r @ axis) * axis, v - (v @ axis) * axis
        cross += np.cross(r_perp, v_perp) @ axis
        dot += r_perp @ v_perp
    return float(np.arctan2(cross, dot))


@dataclass(frozen=True)
class SoftPathPoint:
    """A state on the relaxed soft path.

    Attributes
    ----------
    state : SpinState
    phi : float
        Rotation angle of ``state`` from the reference about the axis.
    transverse_gradient : float
        Largest energy gradient off the orbit tangent (zero on the path).
    orbit_torque : float
        Gradient along the unit orbit tangent (balanced by the zero-point
        torque at the minimum of Gamma).
    hard_stiffness : float
        Lowest Hessian eigenvalue off the orbit tangent (zero when other
        exactly flat directions exist, e.g. an SO(3)-degenerate state).
    tangent_norm : float
        Norm of the orbit tangent per radian of rotation.
    converged : bool
        The transverse gradient vanishes and no hard mode is unstable.
    """

    state: SpinState
    phi: float
    transverse_gradient: float
    orbit_torque: float
    hard_stiffness: float
    tangent_norm: float
    converged: bool


def _relax_transverse(model, state, conditions, axis, tolerance, steps=40):
    for _ in range(steps):
        ex = tangent_expansion(model, state, conditions)
        tangent = _generators(ex) @ axis
        unit = tangent / np.linalg.norm(tangent)
        _, _, vt = np.linalg.svd(unit[None, :])
        P = vt[1:].T
        gP = P.T @ ex.gradient
        if np.max(np.abs(gP)) <= tolerance:
            break
        w, v = np.linalg.eigh(P.T @ ex.hessian @ P)
        floor = np.sqrt(np.finfo(float).eps) * np.max(np.abs(w))
        if np.min(w) < -floor:
            break                                   # a hard mode is unstable
        keep = w > floor                            # other exactly flat directions stay put
        step = -v[:, keep] @ ((v[:, keep].T @ gP) / w[keep])
        directions = ex.directions + np.einsum("ia,iax->ix", (P @ step).reshape(-1, 2), ex.frames)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        state = SpinState(state.model_ref, state.supercell, dict(zip(ex.keys, directions)),
                          state.provenance)
    ex = tangent_expansion(model, state, conditions)
    tangent = _generators(ex) @ axis
    unit = tangent / np.linalg.norm(tangent)
    _, _, vt = np.linalg.svd(unit[None, :])
    P = vt[1:].T
    transverse = float(np.max(np.abs(P.T @ ex.gradient)))
    w = np.linalg.eigvalsh(P.T @ ex.hessian @ P)
    stable = bool(np.min(w) >= -np.sqrt(np.finfo(float).eps) * np.max(np.abs(w)))
    return (state, transverse, float(unit @ ex.gradient), float(np.min(w)),
            float(np.linalg.norm(tangent)), stable)


def soft_path_point(model: SpinModel, reference: SpinState,
                    conditions: Optional[ExternalConditions], axis, phi: float,
                    tolerance: float, target: bool = False) -> SoftPathPoint:
    """Classical state on the relaxed soft path at rotation angle ``phi``.

    Starts from ``R_n(phi) reference`` and minimizes E_cl over every tangent
    direction except the orbit tangent (Newton steps with the analytic
    Hessian). With ``target`` the rotation is corrected so that the relaxed
    state sits at ``phi`` itself (relaxation shifts the angle at second order).
    """
    conditions = conditions or ExternalConditions()
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    angle = phi
    for _ in range(3 if target else 1):
        start = rotate_state(reference, axis, angle)
        state, transverse, torque, hard, norm, stable = _relax_transverse(
            model, start, conditions, axis, tolerance)
        measured = rotation_angle(reference, state, axis)
        if not target:
            break
        angle += _wrap(phi - measured)
    return SoftPathPoint(state, float(np.mod(measured, 2 * np.pi)), transverse, torque, hard, norm,
                         transverse <= tolerance and stable)


@dataclass(frozen=True)
class OrbitLandscape:
    """Energies along the orbit ``R_n(phi)`` of a classical state.

    Attributes
    ----------
    axis : (3,) array
    phi : (m,) array
        Rotation angles from ``state``.
    classical : (m,) array
        ``E_cl(phi)`` per site.
    quantum : (m,) array or None
        ``E_qm(phi)`` per site when a provider was given.
    state : SpinState
        The (refined) reference state at ``phi = 0``.
    diagnostics : dict
        Axis determination (flat Hessian direction, ``C_cl``, exact symmetry)
        and the provider description.
    """

    axis: np.ndarray
    phi: np.ndarray
    classical: np.ndarray
    quantum: Optional[np.ndarray]
    state: SpinState
    diagnostics: Dict[str, Any]


def _no_quantum_energy(model, state, conditions) -> float:
    return 0.0


def orbit_energy_landscape(model: SpinModel, state: SpinState,
                           conditions: Optional[ExternalConditions] = None,
                           quantum_energy: Optional[QuantumEnergy] = None, axis=None,
                           phis=None, criteria: SelectionCriteria = SelectionCriteria(),
                           relax: bool = True) -> OrbitLandscape:
    """E_cl and E_qm along the relaxed soft path (or the rigid orbit) on any angle grid.

    The axis is found as in :func:`select_on_manifold` (classical Hessian and
    global rotation generators, after refinement) unless given. This replaces
    the former MAGSWT grid search (D27): the orbit, not a fixed z rotation, is
    sampled, and the classical energy shows whether the orbit is flat.

    Parameters
    ----------
    model, state, conditions
    quantum_energy : callable, optional
        As in :func:`select_on_manifold`; omitted, only E_cl is evaluated.
    axis : array_like, optional
    phis : array_like, optional
        Angles (default: 360 points in ``[0, 2 pi)``).
    criteria : SelectionCriteria
        Used for refinement and the axis determination.
    relax : bool
        Evaluate on the relaxed soft path (D28; ``phi`` is then the measured
        rotation angle of each relaxed state). False gives the rigid orbit
        ``R_n(phi)``, which overestimates classical pinning when the orbit is
        not flat.
    """
    conditions = conditions or ExternalConditions()
    phis = (np.linspace(0.0, 2 * np.pi, 360, endpoint=False) if phis is None
            else np.asarray(phis, dtype=float).ravel())
    probe = select_on_manifold(model, state, conditions, _no_quantum_energy, axis=axis,
                               criteria=criteria)
    if probe.axis is None:
        raise ValueError(f"no orbit axis: {probe.message}")
    base = refine_classical(model, state, conditions) if criteria.refine else state
    diagnostics = {key: probe.diagnostics.get(key) for key in
                   ("C_cl", "hessian_flat_along_orbit", "exact_symmetry", "flat_rotation_count",
                    "generator_rank", "max_torque", "refinement", "classical_softness")}
    scale = float(np.max(np.abs(np.linalg.eigvalsh(
        tangent_expansion(model, base, conditions).hessian))))
    if relax:
        points = [soft_path_point(model, base, conditions, probe.axis, p,
                                  criteria.path_tolerance * scale, target=True) for p in phis]
        states = [x.state for x in points]
        diagnostics["path_converged"] = all(x.converged for x in points)
        diagnostics["max_transverse_gradient"] = max(x.transverse_gradient for x in points)
        phis = np.array([x.phi for x in points])
    else:
        states = [rotate_state(base, probe.axis, p) for p in phis]
    classical = np.array([classical_energy(model, x, conditions) for x in states])
    quantum = None
    if quantum_energy is not None:
        quantum = np.array([_zero_point(quantum_energy, model, x, conditions, probe.axis)
                            for x in states])
        diagnostics["quantum_energy_provider"] = _describe(quantum_energy)
    diagnostics["E_cl_span"] = float(np.ptp(classical))
    diagnostics["relaxed"] = bool(relax)
    return OrbitLandscape(probe.axis, phis, classical, quantum, base, diagnostics)


# ---------------------------------------------------------------------------
# Zero-point energy provider
# ---------------------------------------------------------------------------

class LSWTZeroPointEnergy:
    """Zero-point energy per site for :func:`select_on_manifold` (D28).

    ``method="constrained"`` (default): the one-loop zero-point energy about
    a constrained classical configuration on the Brillouin-zone mesh of the
    magnetic cell (``BrillouinZone.get_full(N)``, which keeps the point-group
    symmetry). The constrained coordinate is the uniform rotation about the
    orbit axis, one boson mode at the zone centre: with ``axis`` given, only
    that mode is removed there and every other mode is kept. Its curvature is
    the classical one along the orbit (already in E_cl, and negative on part
    of a tilted path), and the linear term dropped on the soft path couples
    only to it. With the rotation angle read by least squares, the removed
    phase-space pair is ``span{t, J t}`` of the orbit tangent ``t``, i.e. the
    boson line ``u_i = sqrt(S_i) (t_theta,i + i t_phi,i)`` in the local frame
    ``(e_theta, e_phi, s_i)``; the zone-centre Hamiltonian is restricted to
    its orthogonal complement (a unitary, hence paraunitary, reduction).
    Without ``axis`` the whole zone centre is dropped (recorded by
    :meth:`describe`). No regularization is added; where ``H(k)`` is not
    positive definite (long-wavelength modes of a constrained state away
    from the classical minimum) the real part of the harmonic frequencies is
    used and the number of such mode pairs is logged. A uniform MAGSWT shift
    must not be used here: it is of the same order as the zero-point energy
    differences and makes E_zp(phi) non-smooth.

    ``method="legacy"``: the existing ``EnergyFunction`` (zone centre
    included, regularization ``reg_type``, MAGSWT by default).

    Parameters
    ----------
    bz_type : str
        Brillouin-zone type of the magnetic cell (``SpinSystem.to_legacy_dict``).
    N : int
        Mesh density.
    method : {"constrained", "legacy"}
    reg_type : int or str
        Regularization of the legacy method.
    """

    accepts_axis = True

    def __init__(self, bz_type: str, N: int, method: str = "constrained",
                 reg_type: Any = "MAGSWT"):
        if method not in ("constrained", "legacy"):
            raise ValueError("method must be 'constrained' or 'legacy'")
        self.bz_type, self.N, self.method, self.reg_type = bz_type, N, method, reg_type
        self._cache: Dict[Any, Any] = {}
        self.regularization: list = []
        self.unstable_pairs: list = []
        self.zone_centre: list = []

    def __call__(self, model: SpinModel, state: SpinState,
                 conditions: Optional[ExternalConditions] = None, axis=None) -> float:
        """Zero-point energy per site; ``axis`` (constrained method) is the orbit axis."""
        from spintoolkit.system.conversion import to_spin_system

        conditions = conditions or ExternalConditions()
        system = to_spin_system(model, state, conditions)
        key = (model.fingerprint(), state.supercell.tobytes(), conditions.field.tobytes())
        if self.method == "legacy":
            from spintoolkit.methods.lswt.energy import EnergyFunction
            if key not in self._cache:
                self._cache[key] = EnergyFunction(system.to_legacy_dict(self.bz_type), N=self.N)
            energy_function = self._cache[key]
            value = float(energy_function.quantum_energy_density_func(
                system.get_angles_flat(), reg_type=self.reg_type))
            self.regularization.append(energy_function.mu_magswt)
            return value
        from spintoolkit.methods.lswt.hamiltonian import LSWTHamiltonian
        from spintoolkit.system.brillouin_zone import BrillouinZone

        data = system.to_legacy_dict(self.bz_type)
        if key not in self._cache:
            _, k, _ = BrillouinZone(data["Lattice/BZ setting"], bz_type=self.bz_type).get_full(self.N)
            self._cache[key] = np.asarray(k, dtype=float)
        k = self._cache[key]
        centre = np.linalg.norm(k, axis=1) <= 1e-12
        if axis is None:
            k = k[~centre]
            centre = centre[~centre]
        angles = system.get_angles_flat()
        H, _ = LSWTHamiltonian(data["Spin info"], data["Couplings"]).Quadratic_Bose_Hamiltonian(
            k, angles=angles)
        H = np.asarray(H)
        ns = H.shape[1] // 2
        reduce = None
        if axis is not None and centre.any():
            u = _soft_boson_line(data["Spin info"], angles, axis)
            q, _ = np.linalg.qr(np.column_stack([u, np.eye(ns)]))
            Q = q[:, 1:ns]                                      # orthonormal complement of u
            zero = np.zeros_like(Q)
            reduce = np.block([[Q, zero], [zero, Q.conj()]])
        total, unstable = 0.0, 0
        for Hk, at_centre in zip(H, centre):
            if at_centre:
                Hk = reduce.conj().T @ Hk @ reduce
            energy, bad = _harmonic_zero_point(Hk)
            total += energy
            unstable += bad
        self.unstable_pairs.append(unstable)
        self.zone_centre.append("excluded (no axis)" if axis is None else "soft pair projected")
        return float(total / len(H) / ns)

    def describe(self) -> Dict[str, Any]:
        out = {"provider": "LSWT zero-point energy via to_spin_system", "method": self.method,
               "bz_type": self.bz_type, "N": self.N}
        if self.method == "legacy":
            values = [x for x in self.regularization if x is not None]
            out.update({"reg_type": self.reg_type, "calls": len(self.regularization),
                        "regularization_min": min(values) if values else None,
                        "regularization_max": max(values) if values else None})
        else:
            out.update({"zone_centre": sorted(set(self.zone_centre)),
                        "regularization": "none; real part where H(k) is not positive definite",
                        "calls": len(self.unstable_pairs),
                        "max_unstable_pairs": max(self.unstable_pairs, default=0)})
        return out


def _soft_boson_line(spin_info, angles, axis) -> np.ndarray:
    """Normalized boson amplitudes of the uniform rotation about ``axis`` (zone centre).

    ``u_i = sqrt(S_i) ((n x s_i).e_theta + i (n x s_i).e_phi)`` in the local
    frame of :class:`~spintoolkit.methods.lswt.hamiltonian.LSWTHamiltonian`.
    """
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    u = []
    for info, (theta, phi) in zip(spin_info.values(), np.asarray(angles).reshape(-1, 2)):
        s = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
        e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0])
        d = np.cross(axis, s)
        u.append(np.sqrt(info["Spin"]) * (d @ e_theta + 1j * (d @ e_phi)))
    u = np.array(u)
    return u / np.linalg.norm(u)


def _harmonic_zero_point(Hk: np.ndarray) -> Tuple[float, int]:
    """``sum(omega) / 2 - Tr H / 4`` of one BdG block and the number of unstable mode pairs."""
    from spintoolkit.methods.lswt.diagonalization import Diagonalizer

    n = Hk.shape[0] // 2
    eta = np.diag(np.r_[np.ones(n), -np.ones(n)])
    try:
        energies = Diagonalizer.Colpa(np.linalg.cholesky(Hk), eta, paraunitary=False)[:n]
        unstable = 0
    except np.linalg.LinAlgError:
        w = np.linalg.eigvals(eta @ Hk)
        stable = np.abs(w.imag) <= 1e-10 * np.max(np.abs(w))
        unstable = int(np.sum(~stable)) // 2
        energies = w.real[stable & (w.real > 0)]
    return float(np.sum(energies) / 2 - np.real(np.trace(Hk)) / 4), unstable


def lswt_zero_point_energy(bz_type: str, N: int, method: str = "constrained",
                           reg_type: Any = "MAGSWT") -> LSWTZeroPointEnergy:
    """Zero-point energy provider for :func:`select_on_manifold` (see LSWTZeroPointEnergy)."""
    return LSWTZeroPointEnergy(bz_type, N, method, reg_type)
