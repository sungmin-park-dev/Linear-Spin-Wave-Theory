"""Zero-point state selection on the classical ground-state manifold (D17, D19).

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
3. Orbit ``R_n(phi)`` of the refined state: E_cl and E_qm at equally spaced
   ``phi``, a least-squares Fourier fit up to a maximum harmonic, and a
   bounded 1D minimization of the fitted E_qm. The fit locates the minimum
   more precisely than the samples when the E_qm variation is tiny.
4. Check that E_cl is constant along the whole orbit (the Hessian guarantees
   flatness only to second order) and record the energies and curvatures used.

Two criteria modes (D19):

``physics`` (default)
    Flatness is judged against the numerical accuracy of the refined state:
    the remaining Newton correction ``delta`` sets the curvature floor
    ``accuracy_factor * |H| * (delta + roundoff_factor * eps)``. A resolved
    classical curvature ``C_cl`` along the orbit is compared with the
    quantum curvature ``C_qm`` at the selected angle (classical pinning,
    competition or quantum selection). "No selection" requires the rotation
    to be an exact symmetry of every term (``[K_n, J] = 0``,
    ``g^T b || n``); a non-symmetric orbit whose harmonic amplitude stays at
    the fit residual is "unresolved".
``fixed``
    Constant thresholds: a Hessian eigenvalue is null when
    ``w_min / w_next < gap_ratio``, a generator singular value is zero below
    ``rank_tolerance`` relative to the largest, and "no selection" means the
    harmonic amplitude is below ``max(residual_factor * residual,
    roundoff_factor * eps * |E_qm|)``.

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

import numpy as np
from scipy.optimize import minimize_scalar

from spintoolkit.definitions.defaults import (
    SELECTION_ACCURACY_FACTOR, SELECTION_COMPETITION_BAND, SELECTION_GAP_RATIO,
    SELECTION_MAX_HARMONIC, SELECTION_MODE, SELECTION_ORBIT_POINTS,
    SELECTION_RANK_TOLERANCE, SELECTION_RESIDUAL_FACTOR, SELECTION_ROUNDOFF_FACTOR)
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
COMPETITION = "competition"
NOT_FLAT = "not_flat"
UNSUPPORTED_MANIFOLD = "unsupported_manifold"
AXIS_REQUIRED = "axis_required"

MODES = ("physics", "fixed")

EPS = np.finfo(float).eps

QuantumEnergy = Callable[[SpinModel, SpinState, ExternalConditions], float]


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
    competition_band : (float, float)
        Physics mode: ``C_cl / C_qm`` inside this range is a competition;
        below it the quantum selection wins, above it classical pinning.
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
    competition_band: Tuple[float, float] = SELECTION_COMPETITION_BAND
    orbit_points: int = SELECTION_ORBIT_POINTS
    max_harmonic: int = SELECTION_MAX_HARMONIC
    refine: bool = True

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.orbit_points <= 2 * self.max_harmonic + 1:
            raise ValueError("orbit_points must exceed 2 * max_harmonic + 1")
        low, high = self.competition_band
        if not 0 < low <= high:
            raise ValueError("competition_band must satisfy 0 < low <= high")


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


def _series(fit: Mapping[str, Any], phi, derivative: int = 0):
    m = np.arange(1, len(fit["cos"]) + 1)
    angle = np.multiply.outer(phi, m)
    a, b = np.asarray(fit["cos"]), np.asarray(fit["sin"])
    if derivative == 0:
        return fit["constant"] + np.cos(angle) @ a + np.sin(angle) @ b
    if derivative == 2:
        return -(np.cos(angle) @ (m ** 2 * a) + np.sin(angle) @ (m ** 2 * b))
    raise ValueError("derivative must be 0 or 2")


def _fit_minimum(fit: Mapping[str, Any], orbit_points: int) -> float:
    """Global minimum of a fitted series: dense grid, then bounded refinement."""
    grid = np.linspace(0.0, 2 * np.pi, 64 * orbit_points, endpoint=False)
    start = grid[np.argmin(_series(fit, grid))]
    step = grid[1] - grid[0]
    result = minimize_scalar(lambda x: float(_series(fit, x)), bounds=(start - step, start + step),
                             method="bounded", options={"xatol": 1e-12})
    return float(np.mod(result.x, 2 * np.pi))


# ---------------------------------------------------------------------------
# Symmetry and generators
# ---------------------------------------------------------------------------

def rotation_symmetry(model: SpinModel, conditions: ExternalConditions, axis,
                      tolerance: float) -> Tuple[bool, Dict[str, float]]:
    """Whether every rotation about ``axis`` leaves all terms invariant.

    Bilinear terms need ``[K_n, J] = 0`` with ``K_n`` the generator
    (``n x``); zeeman terms need ``g^T b`` parallel to ``n``. Norms are
    relative to the largest coefficient, compared with ``tolerance``.

    Returns
    -------
    (bool, dict)
        The verdict and the largest relative violation of each kind.
    """
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = _cross_matrix(axis)
    exchange = [t.coefficient for t in model.terms_of_kind(BILINEAR)]
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
        energy per site in E0; see :class:`LSWTZeroPointEnergy`. An optional
        ``describe()`` method is recorded in the diagnostics.
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
    low, high = criteria.competition_band
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
    diagnostics["C_cl"] = c_cl
    hessian_flat = abs(c_cl) <= flat_limit
    diagnostics["hessian_flat_along_orbit"] = bool(hessian_flat)
    if not hessian_flat and not physics:
        return result(NO_DEGENERACY, "the orbit direction is not a null Hessian mode", axis_=axis)

    symmetric, violations = rotation_symmetry(
        model, conditions, axis, criteria.accuracy_factor * accuracy)
    diagnostics["symmetry_violation"] = violations
    diagnostics["exact_symmetry"] = bool(symmetric)

    # Classical pinning screen: compare energy changes over one orbit step.
    step = 2 * np.pi / criteria.orbit_points
    quantum_info = {}
    if physics and not hessian_flat:
        e0 = quantum_energy(model, state, conditions)
        e_pm = [quantum_energy(model, rotate_state(state, axis, a), conditions) for a in (step, -step)]
        quantum_change = max(abs(e - e0) for e in e_pm)
        classical_change = 0.5 * c_cl * step ** 2
        diagnostics["pinning_screen"] = {"orbit_step": step, "classical_change": classical_change,
                                         "quantum_change": quantum_change}
        if classical_change > high * quantum_change:
            diagnostics["quantum_energy_provider"] = _describe(quantum_energy)
            return result(NO_DEGENERACY, "classical curvature along the softest rotation "
                          "dominates the zero-point variation (classical pinning)", axis_=axis)

    if physics and symmetric and hessian_flat:
        phis = np.arange(criteria.orbit_points) * step
        e_cl = np.array([classical_energy(model, rotate_state(state, axis, p), conditions)
                         for p in phis])
        diagnostics["orbit"] = {"phi": phis.tolist(), "E_cl": e_cl.tolist(),
                                "E_cl_span": float(np.ptp(e_cl))}
        return result(NO_SELECTION, "the orbit is an exact symmetry of every term; E_qm is "
                      "constant", axis_=axis)

    # Orbit.
    phis = np.arange(criteria.orbit_points) * step
    orbit_states = [rotate_state(state, axis, p) for p in phis]
    e_cl = np.array([classical_energy(model, x, conditions) for x in orbit_states])
    e_qm = np.array([quantum_energy(model, x, conditions) for x in orbit_states])
    diagnostics["quantum_energy_provider"] = _describe(quantum_energy)
    fit_qm = fit_harmonics(phis, e_qm, criteria.max_harmonic)
    fit_cl = fit_harmonics(phis, e_cl, criteria.max_harmonic)
    amplitude = float(max(fit_qm["amplitudes"]))
    threshold = max(criteria.residual_factor * fit_qm["residual"],
                    criteria.roundoff_factor * EPS * float(np.max(np.abs(e_qm))))
    cl_floor = criteria.roundoff_factor * EPS * float(np.max(np.abs(e_cl)))
    diagnostics["orbit"] = {
        "phi": phis.tolist(), "E_cl": e_cl.tolist(), "E_qm": e_qm.tolist(),
        "E_cl_span": float(np.ptp(e_cl)), "E_qm_span": float(np.ptp(e_qm)),
        "E_cl_roundoff_floor": cl_floor,
        "E_qm_fit": fit_qm, "E_cl_fit": fit_cl,
        "E_qm_amplitude": amplitude,
        "E_qm_dominant_harmonic": int(np.argmax(fit_qm["amplitudes"]) + 1),
        "E_qm_resolution_threshold": threshold}

    if amplitude <= threshold:
        if physics:
            return result(UNRESOLVED, "E_qm varies along a non-symmetric orbit by less than "
                          "the fit resolution; increase the mesh or check the provider",
                          axis_=axis)
        return result(NO_SELECTION, "harmonic amplitude of E_qm below the threshold", axis_=axis)

    phi_qm = _fit_minimum(fit_qm, criteria.orbit_points)
    c_qm = float(_series(fit_qm, phi_qm, 2))
    tolerance = max(criteria.residual_factor * fit_qm["residual"],
                    criteria.roundoff_factor * EPS * float(np.max(np.abs(e_qm))))
    fine = np.linspace(0.0, 2 * np.pi, 64 * criteria.orbit_points, endpoint=False)
    fitted = _series(fit_qm, fine)
    is_min = (fitted <= np.roll(fitted, 1)) & (fitted <= np.roll(fitted, -1))
    equivalent = fine[is_min & (fitted - fitted.min() <= tolerance)]
    selected = rotate_state(state, axis, phi_qm,
                            {**dict(state.provenance),
                             "selection": {"method": "D17 orbit", "axis": axis.tolist(),
                                           "phi": phi_qm, "mode": criteria.mode}})
    diagnostics.update({
        "phi_qm": phi_qm, "C_qm": c_qm,
        "equivalent_minima": equivalent.tolist(),
        "E_qm_selected": float(quantum_energy(model, selected, conditions)),
        "E_cl_selected": classical_energy(model, selected, conditions)})
    candidates = {"quantum": selected}

    # Classical variation along the orbit.
    if physics:
        span_cl = float(np.ptp(e_cl))
        if hessian_flat:
            ratio = 0.0 if span_cl <= cl_floor else span_cl / float(np.ptp(e_qm))
            kind = "orbit span ratio E_cl / E_qm"
        else:
            ratio = c_cl / c_qm if c_qm > 0 else np.inf
            kind = "curvature ratio C_cl / C_qm"
        diagnostics["classical_to_quantum"] = {"kind": kind, "ratio": ratio}
        if ratio >= low:
            phi_cl = _fit_minimum(fit_cl, criteria.orbit_points)
            candidates["classical"] = rotate_state(state, axis, phi_cl)
            diagnostics["phi_cl"] = phi_cl
            if ratio <= high:
                return result(COMPETITION, f"{kind} = {ratio:.3g} inside the competition band; "
                              "both minima kept", axis_=axis, candidates=candidates)
            if hessian_flat:
                return result(NOT_FLAT, "E_cl varies along the orbit beyond second order "
                              "and dominates E_qm", axis_=axis, candidates=candidates)
            return result(NO_DEGENERACY, "classical pinning dominates the zero-point "
                          "selection", axis_=axis, candidates=candidates)
    elif float(np.ptp(e_cl)) > cl_floor:
        return result(NOT_FLAT, "E_cl is not constant along the orbit", axis_=axis,
                      candidates=candidates)

    return result(SELECTED, f"E_qm selects phi = {phi_qm:.6f} about the axis "
                  f"(harmonic {diagnostics['orbit']['E_qm_dominant_harmonic']} dominant)",
                  selected=selected, axis_=axis, phi=phi_qm, candidates=candidates)


# ---------------------------------------------------------------------------
# Zero-point energy provider
# ---------------------------------------------------------------------------

class LSWTZeroPointEnergy:
    """Zero-point energy per site from the existing LSWT ``EnergyFunction``.

    The model and state are converted with
    :func:`~spintoolkit.system.conversion.to_spin_system`; one
    ``EnergyFunction`` is built per model, supercell and field and reused for
    every orientation. The regularization value returned by the existing code
    is logged: with MAGSWT a uniform shift of at least 1e-9 is always added.

    Parameters
    ----------
    bz_type : str
        Brillouin-zone type of the magnetic cell (``SpinSystem.to_legacy_dict``).
    N : int
        Mesh density.
    reg_type : int or str
        Regularization passed to ``quantum_energy_density_func``.
    """

    def __init__(self, bz_type: str, N: int, reg_type: Any = "MAGSWT"):
        self.bz_type, self.N, self.reg_type = bz_type, N, reg_type
        self._cache: Dict[Any, Any] = {}
        self.regularization: list = []

    def __call__(self, model: SpinModel, state: SpinState,
                 conditions: Optional[ExternalConditions] = None) -> float:
        from spintoolkit.methods.lswt.energy import EnergyFunction
        from spintoolkit.system.conversion import to_spin_system

        conditions = conditions or ExternalConditions()
        system = to_spin_system(model, state, conditions)
        key = (model.fingerprint(), state.supercell.tobytes(), conditions.field.tobytes())
        if key not in self._cache:
            self._cache[key] = EnergyFunction(system.to_legacy_dict(self.bz_type), N=self.N)
        energy_function = self._cache[key]
        value = float(energy_function.quantum_energy_density_func(
            system.get_angles_flat(), reg_type=self.reg_type))
        self.regularization.append(energy_function.mu_magswt)
        return value

    def describe(self) -> Dict[str, Any]:
        values = [x for x in self.regularization if x is not None]
        return {"provider": "LSWT EnergyFunction via to_spin_system", "bz_type": self.bz_type,
                "N": self.N, "reg_type": self.reg_type, "calls": len(self.regularization),
                "regularization_min": min(values) if values else None,
                "regularization_max": max(values) if values else None}


def lswt_zero_point_energy(bz_type: str, N: int, reg_type: Any = "MAGSWT") -> LSWTZeroPointEnergy:
    """Zero-point energy provider for :func:`select_on_manifold`."""
    return LSWTZeroPointEnergy(bz_type, N, reg_type)
