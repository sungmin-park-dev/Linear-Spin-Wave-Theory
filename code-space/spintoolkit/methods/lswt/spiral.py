"""Rotating-frame LSWT of single-Q spirals (D34).

Reference: S. Toth and B. Lake, J. Phys.: Condens. Matter 27, 166002 (2015).

The spiral of :class:`~spintoolkit.states.incommensurate.IncommensurateStructure`
puts spin ``a`` of cell ``R`` along ``R_n(phi_R) d_a`` with ``phi_R = 2 pi q . R``.
Write every spin in the frame that rotates with it, ``S_i = R_n(phi_i) S'_i``.
A bilinear term between cells ``R + n1`` and ``R + n2`` becomes

    S_i^T J S_j = S'_i^T R_n(phi_i)^T J R_n(phi_j) S'_j.

If ``J`` commutes with every rotation about ``n`` (the model is U(1) symmetric
about ``n``: ``J = alpha (1 - n n^T) + beta n n^T + gamma [n]_x``, i.e. XXZ
exchange with the axis ``n`` plus a DM vector along ``n``), then
``R_n(phi_i)^T J R_n(phi_j) = J R_n(phi_j - phi_i)`` and the rotating-frame
exchange

    J'(n2 - n1) = J R_n(2 pi q . (n2 - n1))

depends only on the bond, not on ``R``. A Zeeman field ``h_a = g_a^T b`` along
``n`` is unchanged. The rotating-frame model is therefore an ordinary
translation-invariant model on the primitive cell, its classical state is
``d_a`` on the primitive cell, and the existing LSWT (``solve_lswt``) gives the
exact LSWT of the spiral for any ``q``, rational or not. Its momentum ``k`` is
the rotating-frame (magnon) momentum and its bands are the spiral magnon
dispersion ``omega(k)``.

Without the U(1) symmetry the rotating-frame exchange depends on ``R``, the
single-Q spiral is in general not a classical stationary state (higher
harmonics are generated) and magnons at ``k`` mix with ``k +- 2Q``. A
truncated single-Q LSWT is then not a controlled expansion, so this module
rejects such models instead of returning an approximation; a commensurate
spiral can still be treated on its supercell with ``solve_lswt``.

The rotating-frame result is not a lab-frame result: lab-frame spin
correlations mix ``k`` with ``k +- Q``
(:func:`~spintoolkit.observables.structure_factor.spiral_structure_factor`).
Observables that assume lab-frame spins refuse it (see
:func:`~spintoolkit.methods.lswt.run.require_lab_frame`). Energies, magnon
bands, boson numbers and thermodynamics are frame independent and are read
from it directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
from scipy.optimize import minimize

from spintoolkit.definitions.defaults import (
    LSWT_STATIONARITY_TOLERANCE, SPIRAL_SYMMETRY_TOLERANCE)
from spintoolkit.methods.lswt.run import LSWTResult, LSWTSettings, solve_lswt
from spintoolkit.methods.result import ResultHeader, to_jsonable
from spintoolkit.states.incommensurate import (
    IncommensurateStructure, rotation_matrix, cross_matrix, validate_spiral)
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import BILINEAR, ZEEMAN, SpinModel, Term

ROTATING_FRAME = "rotating"


class SpiralSymmetryError(ValueError):
    """The model is not U(1) symmetric about the spiral axis; lists every offending term."""

    def __init__(self, violations):
        self.violations = list(violations)
        lines = "\n".join(f"  - {v}" for v in self.violations)
        super().__init__(
            f"rotating-frame LSWT needs a Hamiltonian that is symmetric under rotations about "
            f"the spiral axis; {len(self.violations)} term(s) break it:\n{lines}\n"
            "Without this symmetry the single-Q spiral is not an exact classical state and "
            "magnons at k mix with k +- 2Q. For a commensurate wave vector use "
            "IncommensurateStructure.to_spin_state and solve_lswt on the supercell.")


def symmetry_violations(model: SpinModel, axis, conditions: Optional[ExternalConditions] = None,
                        tolerance: float = SPIRAL_SYMMETRY_TOLERANCE) -> List[str]:
    """Terms that break the rotational symmetry about ``axis``.

    A bilinear ``J`` must commute with ``R_n(1 rad)`` (a generic angle: its
    eigenvalues ``1, e^{+-i}`` are distinct, so commuting with it is commuting
    with every rotation about ``n``). The Zeeman vector ``h_a = g_a^T b`` must
    be parallel to ``n``. Both relative to ``max(1, |.|)``.
    """
    n = np.asarray(axis, dtype=float)
    R = rotation_matrix(n, 1.0)
    out = []
    for index, term in enumerate(model.terms_of_kind(BILINEAR)):
        J = term.coefficient
        defect = np.max(np.abs(R @ J - J @ R))
        if defect > tolerance * max(1.0, np.max(np.abs(J))):
            name = term.label or f"bilinear {index}"
            out.append(f"{name} {term.participants}: |[J, R_n]| = {defect:.3g} (exchange "
                       "anisotropy or DM component not along the axis)")
    b = (conditions or ExternalConditions()).field
    for term in model.terms_of_kind(ZEEMAN):
        h = term.coefficient.T @ b
        transverse = np.linalg.norm(h - (h @ n) * n)
        if transverse > tolerance * max(1.0, np.linalg.norm(h)):
            out.append(f"zeeman {term.participants[0][0]!r}: field component {transverse:.3g} "
                       "perpendicular to the axis")
    return out


def rotating_frame_model(model: SpinModel, structure: IncommensurateStructure,
                         conditions: Optional[ExternalConditions] = None,
                         check_symmetry: bool = True) -> Tuple[SpinModel, SpinState]:
    """The translation-invariant rotating-frame model and its primitive-cell state.

    Parameters
    ----------
    model : SpinModel
    structure : IncommensurateStructure
    conditions : ExternalConditions, optional
        Needed for the symmetry check of the field.
    check_symmetry : bool
        Raise :class:`SpiralSymmetryError` if the model is not U(1) symmetric
        about the axis (default). Only the energy functions switch it off,
        where ``E(q)`` of the given single-Q state is still well defined.

    Returns
    -------
    model : SpinModel
        Same sites and Zeeman terms; bilinear ``J'(n2 - n1) = J R_n(2 pi q . (n2 - n1))``.
    state : SpinState
        ``d_a`` on the primitive cell.
    """
    validate_spiral(structure, model)
    if check_symmetry:
        violations = symmetry_violations(model, structure.rotation_axis, conditions)
        if violations:
            raise SpiralSymmetryError(violations)
    terms = []
    for term in model.terms:
        if term.kind == BILINEAR:
            (a, n1), (b, n2) = term.participants
            angle = structure.phase((n2[0] - n1[0], n2[1] - n1[1]))
            J = term.coefficient @ rotation_matrix(structure.rotation_axis, angle)
            terms.append(Term(BILINEAR, term.participants, J, term.label))
        else:
            terms.append(term)
    metadata = dict(model.metadata)
    metadata["model_id"] = f"{model.metadata['model_id']}@rotating_frame"
    metadata["parameters"] = {**dict(model.metadata.get("parameters", {})),
                              "spiral_wave_vector": structure.wave_vector.tolist(),
                              "spiral_axis": structure.rotation_axis.tolist()}
    rotated = SpinModel(model.lattice, model.sites, terms, metadata)
    state = SpinState(rotated.fingerprint(), np.eye(2, dtype=int),
                      {(s, (0, 0)): structure.directions[s] for s in model.site_ids},
                      {"origin": "rotating_frame", **dict(structure.provenance)})
    return rotated, state



def _bond_arrays(model: SpinModel):
    index = {s: i for i, s in enumerate(model.site_ids)}
    spins = np.array([s.spin for s in model.sites])
    bonds = model.terms_of_kind(BILINEAR)
    a = np.array([index[t.participants[0][0]] for t in bonds], dtype=int)
    b = np.array([index[t.participants[1][0]] for t in bonds], dtype=int)
    dn = np.array([np.subtract(t.participants[1][1], t.participants[0][1]) for t in bonds],
                  dtype=float).reshape(-1, 2)
    J = np.array([t.coefficient for t in bonds]).reshape(-1, 3, 3)
    return index, spins, a, b, dn, J


def _fields(model: SpinModel, conditions: Optional[ExternalConditions]) -> np.ndarray:
    b = (conditions or ExternalConditions()).field
    h = np.zeros((model.num_sites, 3))
    for i, s in enumerate(model.site_ids):
        for term in model.terms_of_kind(ZEEMAN):
            if term.participants[0][0] == s:
                h[i] = term.coefficient.T @ b
    return h


def _energy_and_gradient(arrays, h, axis, q, d, directions_gradient=False):
    """Energy per site, ``dE/dq`` and optionally ``dE/dd_a`` (Ns, 3), all analytic.

    ``E = (1/Ns) [sum_bonds S_a S_b d_a^T J R_n(2 pi q . dn) d_b - sum_a S_a h_a . d_a]``
    and ``dR_n(phi)/dphi = R_n(phi) [n]_x``.
    """
    _, spins, a, b, dn, J = arrays
    ns = len(spins)
    K = cross_matrix(axis)
    R = rotation_matrix(axis, 2 * np.pi * dn @ q)                 # (nb, 3, 3)
    JR = np.einsum("bxy,byz->bxz", J, R)
    left = spins[a, None] * d[a]
    right = spins[b, None] * d[b]
    bond = np.einsum("bx,bxy,by->b", left, JR, right)
    energy = (bond.sum() - np.sum(spins[:, None] * h * d)) / ns
    slope = np.einsum("bx,bxy,yz,bz->b", left, JR, K, right)    # dE_bond / dphi
    gradient = 2 * np.pi * (slope @ dn) / ns
    if not directions_gradient:
        return float(energy), gradient
    grad_d = -spins[:, None] * h
    np.add.at(grad_d, a, spins[a, None] * np.einsum("bxy,by->bx", JR, right))
    np.add.at(grad_d, b, spins[b, None] * np.einsum("bxy,bx->by", JR, left))
    return float(energy), gradient, grad_d / ns


def spiral_energy(model: SpinModel, structure: IncommensurateStructure,
                  conditions: Optional[ExternalConditions] = None) -> float:
    """Classical energy per site (E0) of the single-Q state, for any model.

    For a model that is not U(1) symmetric about the axis this is still the
    energy of the given single-Q configuration (averaged over the infinite
    lattice only when the phases equidistribute), but that configuration is in
    general not stationary.
    """
    validate_spiral(structure, model)
    arrays = _bond_arrays(model)
    d = np.array([structure.directions[s] for s in model.site_ids])
    return _energy_and_gradient(arrays, _fields(model, conditions), structure.rotation_axis,
                                structure.wave_vector, d)[0]


def spiral_energy_gradient(model: SpinModel, structure: IncommensurateStructure,
                           conditions: Optional[ExternalConditions] = None) -> np.ndarray:
    """``dE/dq`` per site at fixed ``d_a`` (E0 per reciprocal-lattice unit), analytic."""
    validate_spiral(structure, model)
    arrays = _bond_arrays(model)
    d = np.array([structure.directions[s] for s in model.site_ids])
    return _energy_and_gradient(arrays, _fields(model, conditions), structure.rotation_axis,
                                structure.wave_vector, d)[1]


def refine_spiral(model: SpinModel, structure: IncommensurateStructure,
                  conditions: Optional[ExternalConditions] = None,
                  fix_wave_vector: bool = False, gtol: float = 1e-12,
                  maxiter: int = 2000) -> IncommensurateStructure:
    """Minimize the classical energy over ``q`` and the cones and phases ``d_a``.

    The axis is kept: it is fixed by the symmetry of the model. The returned
    state records the final energy and gradient in its provenance. A local
    minimizer: start from the Luttinger-Tisza spiral
    (:meth:`IncommensurateStructure.from_lt`) or a classical search.

    Raises
    ------
    SpiralSymmetryError
        If the model is not U(1) symmetric about the axis.
    """
    validate_spiral(structure, model)
    violations = symmetry_violations(model, structure.rotation_axis, conditions)
    if violations:
        raise SpiralSymmetryError(violations)
    n = structure.rotation_axis
    e1 = np.eye(3)[int(np.argmin(np.abs(n)))]
    e1 = e1 - (e1 @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    arrays = _bond_arrays(model)
    h = _fields(model, conditions)
    d0 = np.array([structure.directions[s] for s in model.site_ids])
    theta0 = np.arccos(np.clip(d0 @ n, -1, 1))
    phi0 = np.arctan2(d0 @ e2, d0 @ e1)

    def unpack(x):
        q = structure.wave_vector if fix_wave_vector else x[:2]
        angles = x if fix_wave_vector else x[2:]
        theta, phi = angles[0::2], angles[1::2]
        d = (np.cos(theta)[:, None] * n + np.sin(theta)[:, None]
             * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
        return q, d

    def objective(x):
        q, d = unpack(x)
        energy, grad_q, grad_d = _energy_and_gradient(arrays, h, n, q, d, True)
        angles = x if fix_wave_vector else x[2:]
        theta, phi = angles[0::2], angles[1::2]
        plane = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
        d_theta = -np.sin(theta)[:, None] * n + np.cos(theta)[:, None] * plane
        d_phi = np.sin(theta)[:, None] * (-np.sin(phi)[:, None] * e1 + np.cos(phi)[:, None] * e2)
        grad_angles = np.column_stack([np.sum(grad_d * d_theta, axis=1),
                                       np.sum(grad_d * d_phi, axis=1)]).ravel()
        return energy, grad_angles if fix_wave_vector else np.r_[grad_q, grad_angles]

    angles = np.column_stack([theta0, phi0]).ravel()
    x0 = angles if fix_wave_vector else np.r_[structure.wave_vector, angles]
    res = minimize(objective, x0, jac=True, method="BFGS",
                   options={"gtol": gtol, "maxiter": maxiter})
    q, d = unpack(res.x)
    q = np.mod(q, 1.0)
    energy, gradient = _energy_and_gradient(arrays, h, n, q, d)
    provenance = {**dict(structure.provenance), "refined": True, "energy": energy,
                  "wave_vector_gradient": gradient.tolist(), "converged": bool(res.success)}
    return IncommensurateStructure(structure.model_ref, q, n,
                                   {s: d[i] / np.linalg.norm(d[i])
                                    for i, s in enumerate(model.site_ids)}, provenance)


def spiral_fingerprint(structure: IncommensurateStructure) -> str:
    """SHA-256 of the model reference, ``q``, axis and directions (rounded to 1e-12)."""
    payload = json.dumps({"model_ref": structure.model_ref,
                          "wave_vector": np.round(structure.wave_vector, 12).tolist(),
                          "axis": np.round(structure.rotation_axis, 12).tolist(),
                          "directions": {s: np.round(v, 12).tolist()
                                         for s, v in sorted(structure.directions.items())}},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class SpiralLSWTResult:
    """LSWT of a single-Q spiral.

    Attributes
    ----------
    header : ResultHeader
        Method ``"lswt-spiral"``, the lab-frame model and the spiral.
    structure : IncommensurateStructure
    wave_vector : (2,) array
        Cartesian ``Q``.
    rotating : LSWTResult
        LSWT of the rotating-frame model on the primitive cell; its momenta
        are rotating-frame momenta ``k`` and its bands are ``omega(k)``. Its
        ``extra["frame"]`` is ``"rotating"``, so lab-frame observables refuse it.
    """

    header: ResultHeader
    structure: IncommensurateStructure
    wave_vector: np.ndarray
    rotating: LSWTResult
    rotating_model: SpinModel = field(repr=False)

    @property
    def classical_energy(self) -> float:
        return self.rotating.classical_energy

    @property
    def zero_point_energy(self) -> float:
        return self.rotating.zero_point_energy

    @property
    def ground_state_energy(self) -> float:
        return self.rotating.ground_state_energy

    @property
    def k_points(self) -> np.ndarray:
        return self.rotating.k_points

    @property
    def boson_numbers(self) -> np.ndarray:
        return self.rotating.boson_numbers

    @property
    def thermal(self):
        return self.rotating.thermal

    def bands(self) -> np.ndarray:
        """Magnon energies ``omega(k)`` (nk, Ns) at the rotating-frame momenta, ascending."""
        return self.rotating.bands()

    def ordered_moments(self) -> np.ndarray:
        """Ordered moment ``S_a - <n_a>`` of every site (the same in every cell)."""
        return self.rotating.ordered_moments()

    def to_json_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        body = self.rotating.to_json_dict(include_arrays)["lswt"]
        spiral = {"wave_vector": self.structure.wave_vector,
                  "cartesian_wave_vector": self.wave_vector,
                  "rotation_axis": self.structure.rotation_axis,
                  "directions": dict(self.structure.directions),
                  "frame": "momenta and modes in the rotating frame"}
        return to_jsonable({"header": self.header, "spiral": spiral, "lswt": body})


def solve_spiral_lswt(model: SpinModel, structure: IncommensurateStructure,
                      conditions: Optional[ExternalConditions] = None,
                      geometry: Optional[CalculationGeometry] = None,
                      settings: LSWTSettings = LSWTSettings()) -> SpiralLSWTResult:
    """Linear spin-wave theory about a single-Q spiral (rotating frame, D34).

    Parameters
    ----------
    model : SpinModel
        Must be U(1) symmetric about ``structure.rotation_axis`` (see the
        module docstring).
    structure : IncommensurateStructure
        Reference spiral; it should be a classical stationary state (torques
        and ``dE/dq`` vanish).
    conditions : ExternalConditions, optional
        The field must be along the axis.
    geometry : CalculationGeometry, optional
        Thermodynamic limit only: an incommensurate spiral does not fit a torus.
    settings : LSWTSettings
        As for :func:`~spintoolkit.methods.lswt.run.solve_lswt`; the mesh is of
        the primitive reciprocal cell.

    Returns
    -------
    SpiralLSWTResult

    Raises
    ------
    SpiralSymmetryError
        If the model or the field breaks the rotational symmetry about the axis.
    LSWTError
        If ``H(k)`` has a zero or negative mode (as in ``solve_lswt``); at a
        ``q`` that does not minimize the classical energy the spiral is
        unstable and negative modes appear.
    """
    conditions = conditions or ExternalConditions()
    geometry = geometry or CalculationGeometry.thermodynamic_limit()
    if geometry.kind != "thermodynamic_limit":
        raise NotImplementedError("spiral LSWT is formulated in the thermodynamic limit; for a "
                                  "torus use a commensurate SpinState and solve_lswt")
    rotated, state = rotating_frame_model(model, structure, conditions)
    gradient = spiral_energy_gradient(model, structure, conditions)
    gradient_norm = float(np.linalg.norm(gradient))
    if gradient_norm > settings.stationarity_tolerance:
        warnings.warn(f"the wave vector is not a classical extremum (|dE/dq| = "
                      f"{gradient_norm:.3g} E0): the spiral is not stationary against a change "
                      "of pitch and LSWT is expected to show negative modes", UserWarning,
                      stacklevel=2)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="reference state is not stationary")
        rotating = solve_lswt(rotated, state, conditions, geometry, settings)
    torque = rotating.header.diagnostics["max_torque"]
    if torque > settings.stationarity_tolerance:
        warnings.warn(f"the spiral is not stationary (max torque {torque:.3g} E0): linear "
                      "boson terms remain and LSWT is not a consistent expansion", UserWarning,
                      stacklevel=2)
    rotating = replace(rotating, extra={**rotating.extra, "frame": ROTATING_FRAME,
                                        "wave_vector": structure.cartesian_wave_vector(model),
                                        "rotation_axis": structure.rotation_axis})
    diagnostics = {**rotating.header.diagnostics,
                   "frame": "rotating (Toth and Lake 2015)",
                   "wave_vector_gradient": gradient.tolist(),
                   "stationary": bool(torque <= settings.stationarity_tolerance
                                      and gradient_norm <= settings.stationarity_tolerance),
                   "rotating_model_ref": rotated.fingerprint()}
    header = ResultHeader.build(
        "lswt-spiral", model, None, geometry, conditions, settings.as_dict(),
        "energies per site of the model (E0); boson numbers per site; momenta in the "
        "rotating frame", diagnostics)
    header = replace(header, state_ref=spiral_fingerprint(structure))
    return SpiralLSWTResult(header, structure, structure.cartesian_wave_vector(model),
                            rotating, rotated)
