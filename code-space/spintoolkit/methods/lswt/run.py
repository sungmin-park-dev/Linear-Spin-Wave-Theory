"""LSWT on the common model types: ``solve_lswt(model, state, conditions, geometry, settings)``.

The model and state are converted with
:func:`~spintoolkit.system.conversion.to_spin_system` and the existing
``LSWTHamiltonian`` builds the bosonic ``H(k)`` of the magnetic cell (Fourier
convention D13). The result keeps everything the diagonalization produced, so
later observables of the same calculation reuse it without diagonalizing again:
the Hamiltonian actually diagonalized, the Colpa eigenvalues and the
paraunitary eigenvectors at every k.

Regularization (default ``"none"``): ``H(k)`` must be positive definite at
every k. A k where the lowest eigenvalue of ``H(k)`` is at most
``zero_mode_tolerance`` times its largest (a zero mode such as a Goldstone mode
at the zone centre, or an instability) raises :class:`LSWTError` naming the k.
Round-off can make an exact zero mode slightly positive, so a successful
Cholesky factorization alone is not accepted; use the default shifted
mesh, which avoids the zone centre, or request ``"MAGSWT"`` or
``"k-dependent"`` explicitly. The shift applied is always recorded.

Energies are per site in the energy unit E0 of the model (D20).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional, Tuple
import warnings

import numpy as np

from spintoolkit.definitions.defaults import (
    LSWT_DEFAULT_MESH, LSWT_STATIONARITY_TOLERANCE, LSWT_ZERO_MODE_TOLERANCE)
from spintoolkit.methods.classical import classical_energy, torques
from spintoolkit.methods.lswt.diagonalization import Diagonalizer
from spintoolkit.methods.lswt.hamiltonian import LSWTHamiltonian
from spintoolkit.methods.result import ResultHeader, to_jsonable
from spintoolkit.observables.bose_statistics import compute_static_magnon_kernel
from spintoolkit.states.spin_state import SpinState, validate_spin_state
from spintoolkit.system.cluster import allowed_momenta, expand_on_torus
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.conversion import to_spin_system
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import SpinModel

REGULARIZATIONS = ("none", "MAGSWT", "k-dependent")


class LSWTError(ValueError):
    """The LSWT Hamiltonian cannot be diagonalized as requested."""


def require_lab_frame(result, observable: str) -> None:
    """Refuse a rotating-frame spiral result for an observable that assumes lab-frame spins.

    The rotating-frame result of a spiral (D34) has the right energies and
    magnon bands, but its spins are the rotating-frame spins: lab-frame
    correlations mix ``k`` with ``k +- Q``.

    Raises
    ------
    ValueError
        If ``result.extra["frame"]`` is ``"rotating"``.
    """
    if getattr(result, "extra", {}).get("frame") == "rotating":
        raise ValueError(f"{observable} needs lab-frame spins, but this LSWT result is in the "
                         "rotating frame of a spiral (D34); use spiral_structure_factor for "
                         "the spin structure factor. Other lab-frame observables of spirals "
                         "are not implemented.")


@dataclass(frozen=True)
class LSWTSettings:
    """Method settings of :func:`solve_lswt`.

    Parameters
    ----------
    mesh : (int, int)
        Thermodynamic limit: ``N1 x N2`` uniform mesh of the magnetic
        reciprocal cell.
    shift : bool
        Offset the mesh by half a step, so that it avoids the zone centre.
    k_points : (m, 2) array_like, optional
        Explicit Cartesian momenta with equal weights, replacing the mesh.
    regularization : {"none", "MAGSWT", "k-dependent"}
    stationarity_tolerance : float
        Largest torque (E0) accepted without a warning.
    zero_mode_tolerance : float
        Without regularization, ``min eig H(k) <= zero_mode_tolerance *
        max |eig H(k)|`` is a zero mode or an instability.
    gapless : bool, optional
        The user's decision after inspecting zero-mode candidates (4b); used
        only for finite-temperature quantities, recorded in the header.
    """

    mesh: Tuple[int, int] = LSWT_DEFAULT_MESH
    shift: bool = True
    k_points: Optional[Any] = None
    regularization: str = "none"
    stationarity_tolerance: float = LSWT_STATIONARITY_TOLERANCE
    zero_mode_tolerance: float = LSWT_ZERO_MODE_TOLERANCE
    gapless: Optional[bool] = None

    def __post_init__(self):
        if self.regularization not in REGULARIZATIONS:
            raise ValueError(f"regularization must be one of {REGULARIZATIONS}")
        mesh = tuple(int(n) for n in self.mesh)
        if len(mesh) != 2 or min(mesh) < 1:
            raise ValueError("mesh must be two positive integers")
        object.__setattr__(self, "mesh", mesh)
        if self.k_points is not None:
            k = np.array(self.k_points, dtype=float).reshape(-1, 2)
            k.setflags(write=False)
            object.__setattr__(self, "k_points", k)

    def as_dict(self) -> Dict[str, Any]:
        return {"mesh": list(self.mesh), "shift": self.shift,
                "k_points": None if self.k_points is None else "explicit",
                "regularization": self.regularization,
                "stationarity_tolerance": self.stationarity_tolerance,
                "zero_mode_tolerance": self.zero_mode_tolerance, "gapless": self.gapless}


@dataclass(frozen=True)
class LSWTResult:
    """LSWT result body with the common header.

    Array axes: ``k`` over momenta, ``2 Ns`` over the Nambu components
    ``(a_1 .. a_Ns, a_1^dagger .. a_Ns^dagger)`` of the magnetic cell, whose
    sites are ``site_keys`` (model site, canonical cell) in this order.

    Attributes
    ----------
    header : ResultHeader
    site_keys : tuple
    spins : (Ns,) array
    directions : (Ns, 3) array
    k_points : (nk, 2) array
        Cartesian momenta.
    fractional : (nk, 2) array or None
        Momenta in the magnetic reciprocal basis (None for explicit k).
    weights : (nk,) array
        Integration weights, summing to one.
    hamiltonians : (nk, 2Ns, 2Ns) complex array
        ``H(k)`` as diagonalized (including any regularization shift).
    eigenvalues : (nk, 2Ns) array
        Colpa order: particle energies (descending), then the energies of the
        hole components (also positive).
    eigenvectors : (nk, 2Ns, 2Ns) complex array
        Paraunitary ``T`` with ``T^dagger H T = diag(|eigenvalues|)`` and the
        columns in the order of ``eigenvalues``.
    regularization_shift : (nk,) array
        Uniform on-site shift added at each k (zero without regularization).
    linear_terms : float
        Largest linear boson coefficient (zero at a stationary state).
    classical_energy, zero_point_energy, ground_state_energy : float
        Per site; ``ground_state_energy = classical + zero_point``.
    boson_numbers : (Ns,) array
        ``<a_i^dagger a_i>`` at zero temperature.
    lattice, magnetic_lattice : (2, 2) arrays
        Primitive and magnetic lattice vectors (rows).
    positions : (Ns, 2) array
        Cartesian positions of the magnetic-cell sites.
    local_frames : (Ns, 3, 3) array
        Rotation ``R_i`` of the Hamiltonian's local frame: columns are the
        local x, y and z axes (z along the spin); spin deviations are
        ``delta S_i = sqrt(S_i/2) [(x_i - i y_i) a_i + (x_i + i y_i) a_i^dagger]``.
    thermal : ThermalResult or None
        Finite-temperature quantities at ``conditions.temperature`` (4b).
    hamiltonian_at : callable
        ``hamiltonian_at(k)`` returns the unregularized ``H(k)`` for momenta
        ``(m, 2)``, for zero-mode scans and later observables. Not serialized.
    hamiltonian_derivatives_at : callable
        ``hamiltonian_derivatives_at(k)`` returns ``(dH/dk_x, dH/dk_y)``, each
        ``(m, 2Ns, 2Ns)``, analytic in the Fourier convention of ``H(k)``
        (D13); for Berry curvature (stage 5). Not serialized.
    """

    header: ResultHeader
    site_keys: Tuple
    spins: np.ndarray
    directions: np.ndarray
    k_points: np.ndarray
    fractional: Optional[np.ndarray]
    weights: np.ndarray
    hamiltonians: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    regularization_shift: np.ndarray
    linear_terms: float
    classical_energy: float
    zero_point_energy: float
    ground_state_energy: float
    boson_numbers: np.ndarray
    lattice: Optional[np.ndarray] = None
    magnetic_lattice: Optional[np.ndarray] = None
    thermal: Optional[Any] = None
    positions: Optional[np.ndarray] = None
    local_frames: Optional[np.ndarray] = None
    hamiltonian_at: Optional[Callable] = field(default=None, repr=False, compare=False)
    extra: Dict[str, Any] = field(default_factory=dict)
    hamiltonian_derivatives_at: Optional[Callable] = field(default=None, repr=False,
                                                           compare=False)

    @property
    def num_sites(self) -> int:
        return len(self.site_keys)

    def bands(self) -> np.ndarray:
        """Magnon energies ``(nk, Ns)``, ascending at each k."""
        return np.sort(self.eigenvalues[:, :self.num_sites], axis=1)

    def ordered_moments(self) -> np.ndarray:
        """Ordered moment ``S_i - <n_i>`` of every magnetic site at zero temperature."""
        return self.spins - self.boson_numbers

    def to_json_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        body = {"site_keys": [[s, list(c)] for s, c in self.site_keys],
                "spins": self.spins, "directions": self.directions,
                "k_points": self.k_points, "fractional": self.fractional,
                "weights": self.weights, "bands": self.bands(),
                "classical_energy": self.classical_energy,
                "zero_point_energy": self.zero_point_energy,
                "ground_state_energy": self.ground_state_energy,
                "boson_numbers": self.boson_numbers,
                "ordered_moments": self.ordered_moments(),
                "regularization_shift": self.regularization_shift,
                "linear_terms": self.linear_terms}
        if include_arrays:
            body.update({"hamiltonians": self.hamiltonians, "eigenvalues": self.eigenvalues,
                         "eigenvectors": self.eigenvectors})
        return to_jsonable({"header": self.header, "lswt": body})


def _momenta(model: SpinModel, state: SpinState, geometry: CalculationGeometry,
             settings: LSWTSettings):
    magnetic = state.magnetic_lattice(model)
    if settings.k_points is not None:
        k = np.array(settings.k_points)
        return k, None
    if geometry.kind == "finite_torus":
        expand_on_torus(model, geometry)          # same torus rules for every method (D23)
        _, k_all = allowed_momenta(model, geometry)
        p = np.mod(k_all @ magnetic.T / (2 * np.pi), 1.0)
        p[np.isclose(p, 1.0, rtol=0, atol=1e-9)] = 0.0
        _, unique = np.unique(np.round(p, 9), axis=0, return_index=True)
        unique = np.sort(unique)
        return k_all[unique], p[unique]
    n1, n2 = settings.mesh
    offset = 0.5 if settings.shift else 0.0
    grid = np.array([((i + offset) / n1, (j + offset) / n2) for i in range(n1) for j in range(n2)])
    k = 2 * np.pi * grid @ np.linalg.inv(magnetic).T
    return k, grid


def _diagonalize(H: np.ndarray, regularization: str, k: np.ndarray, zero_mode_tolerance: float):
    nk, n2, _ = H.shape
    ns = n2 // 2
    J = np.diag(np.r_[np.ones(ns), -np.ones(ns)])
    if regularization == "none":
        energies, vectors, failed, lowest = [], [], [], []
        for i, Hk in enumerate(H):
            w = np.linalg.eigvalsh(Hk)
            try:
                if w[0] <= zero_mode_tolerance * np.max(np.abs(w)):
                    raise np.linalg.LinAlgError
                K = np.linalg.cholesky(Hk)
            except np.linalg.LinAlgError:
                failed.append(i)
                lowest.append(float(w[0]))
                continue
            E, T = Diagonalizer.Colpa(K, J)
            energies.append(E)
            vectors.append(T)
        if failed:
            where = ", ".join(f"k={np.round(k[i], 6).tolist()} (min eig {e:.3g})"
                              for i, e in zip(failed[:5], lowest[:5]))
            raise LSWTError(
                f"H(k) has a zero or negative mode at {len(failed)} of {nk} momenta: {where}. "
                "A zero mode (e.g. Goldstone at the zone centre) needs a mesh that avoids it "
                "(shift=True) or an explicit regularization ('MAGSWT', 'k-dependent'); "
                "a negative eigenvalue means the reference state is unstable.")
        return H, np.array(energies), np.array(vectors), np.zeros(nk)
    original = H.copy()
    reg, bose_E, para_T, imag_E, mu = Diagonalizer.diagonalize_w_reg(
        H.copy(), regularization, paraunitary=True)
    if any(t is None for t in para_T):
        bad = [i for i, t in enumerate(para_T) if t is None]
        raise LSWTError(f"Colpa diagonalization failed at {len(bad)} momenta even with "
                        f"regularization '{regularization}' (first k={np.round(k[bad[0]], 6).tolist()})")
    shift = np.real(np.trace(reg - original, axis1=1, axis2=2)) / n2
    return np.asarray(reg), np.array(bose_E), np.array(para_T), shift


def solve_lswt(model: SpinModel, state: SpinState,
               conditions: Optional[ExternalConditions] = None,
               geometry: Optional[CalculationGeometry] = None,
               settings: LSWTSettings = LSWTSettings()) -> LSWTResult:
    """Linear spin-wave theory about ``state``.

    Parameters
    ----------
    model : SpinModel
    state : SpinState
        Reference classical state; it should be stationary (torques vanish).
    conditions : ExternalConditions, optional
    geometry : CalculationGeometry, optional
        Thermodynamic limit (default; momenta from ``settings.mesh``) or a
        finite torus (its momenta, reduced to the magnetic cell).
        A positive ``conditions.temperature`` (dimensionless ``k_B T / E0``)
        adds :attr:`LSWTResult.thermal`; zero-mode candidates must then be
        resolved with ``settings.gapless`` (see
        :func:`spintoolkit.observables.thermal.thermal_quantities`).
    settings : LSWTSettings

    Returns
    -------
    LSWTResult

    Raises
    ------
    LSWTError
        If ``H(k)`` cannot be diagonalized (see the module docstring).
    """
    conditions = conditions or ExternalConditions()
    geometry = geometry or CalculationGeometry.thermodynamic_limit()
    validate_spin_state(state, model, geometry)
    k, fractional = _momenta(model, state, geometry, settings)

    system = to_spin_system(model, state, conditions)
    data = system.to_legacy_dict("simple")
    hamiltonian = LSWTHamiltonian(data["Spin info"], data["Couplings"])
    H, linear = hamiltonian.Quadratic_Bose_Hamiltonian(k, angles=system.get_angles_flat())
    linear_max = float(max((abs(v) for v in linear.values()), default=0.0))
    H, E, T, shift = _diagonalize(np.asarray(H, dtype=complex), settings.regularization, k,
                                  settings.zero_mode_tolerance)

    ns = H.shape[1] // 2
    weights = np.full(len(k), 1.0 / len(k))
    zero_point = weights @ (np.sum(E[:, :ns], axis=1) / 2
                            - np.real(np.trace(H, axis1=1, axis2=2)) / 4) / ns
    kernel = compute_static_magnon_kernel(E[0], Temperature=0, Ns=ns)
    occupation = np.einsum("kin,n,kin->ki", T, kernel, T.conj()).real[:, ns:]
    boson_numbers = weights @ occupation

    e_cl = classical_energy(model, state, conditions)
    torque = max(float(np.linalg.norm(t)) for t in torques(model, state, conditions).values())
    keys = tuple((site.id, cell) for site in model.sites for cell in state.cells)
    diagnostics = {
        "max_torque": torque, "max_linear_term": linear_max,
        "stationary": torque <= settings.stationarity_tolerance,
        "min_hamiltonian_eigenvalue": float(np.min(np.linalg.eigvalsh(H))),
        "min_magnon_energy": float(np.min(E[:, :ns])),
        "regularization": settings.regularization,
        "max_regularization_shift": float(np.max(np.abs(shift), initial=0.0)),
        "momenta": ("explicit" if settings.k_points is not None else
                    "finite torus" if geometry.kind == "finite_torus" else
                    f"{settings.mesh[0]}x{settings.mesh[1]} mesh of the magnetic reciprocal cell, "
                    f"{'half-step shifted' if settings.shift else 'zone-centred'}"),
        "num_k": len(k)}
    if not diagnostics["stationary"]:
        warnings.warn(f"reference state is not stationary (max torque {torque:.3g} E0): linear "
                      "boson terms remain and LSWT is not a consistent expansion", UserWarning,
                      stacklevel=2)
    header = ResultHeader.build(
        "lswt", model, state, geometry, conditions, settings.as_dict(),
        "energies per site of the model (E0); boson numbers per magnetic site", diagnostics)
    angles = system.get_angles_flat()

    def hamiltonian_at(momenta):
        return np.asarray(hamiltonian.Quadratic_Bose_Hamiltonian(
            np.atleast_2d(np.asarray(momenta, dtype=float)), angles=angles)[0])

    def hamiltonian_derivatives_at(momenta):
        momenta = np.atleast_2d(np.asarray(momenta, dtype=float))
        hamiltonian.Quadratic_Bose_Hamiltonian(momenta[:1], angles=angles)   # local frames
        dx, dy = hamiltonian.partial_derivatives_of_Hk(momenta)
        return np.asarray(dx), np.asarray(dy)

    result = LSWTResult(header, keys, np.array([model.site(s).spin for s, _ in keys]),
                        np.array([state.direction(s, c) for s, c in keys]), k, fractional, weights,
                        H, E, T, shift, linear_max, float(e_cl), float(zero_point),
                        float(e_cl + zero_point), boson_numbers,
                        np.asarray(model.lattice, dtype=float), state.magnetic_lattice(model),
                        None, np.array([model.cartesian_position(s, c) for s, c in keys]),
                        np.array(list(hamiltonian.get_rmat_dict(angles=angles).values())),
                        hamiltonian_at, hamiltonian_derivatives_at=hamiltonian_derivatives_at)
    if conditions.temperature > 0:
        from spintoolkit.observables.thermal import thermal_quantities

        result = replace(result, thermal=thermal_quantities(
            result, [conditions.temperature], gapless=settings.gapless))
    return result
