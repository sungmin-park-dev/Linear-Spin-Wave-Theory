"""LSWT on the common model types: ``solve_lswt(model, state, conditions, geometry, settings)``.

:class:`~spintoolkit.methods.lswt.quadratic.QuadraticBoseHamiltonian` builds
the bosonic ``H(k)`` of the magnetic cell directly from the model and state
(Fourier convention D13; D36 replaced the conversion to the former
``SpinSystem``, with identical matrices). The result keeps everything the diagonalization produced, so
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
(``"MAGSWT"`` here is the uniform zero-mode shift of the legacy code, not the
augmented theory below; since D61 it never acts on an unstable state.)

Augmentation (D62, opt-in): ``augmentation="minimal"`` solves the minimally
augmented Hamiltonian ``H + mu sum_i a_i^dagger a_i`` of
:mod:`~spintoolkit.methods.lswt.augmentation` (MAGSWT) about a classically
unstable stationary state. Only the energy is defined for the spin model, as
an upper estimate; the boson numbers are NaN and observables refuse the result
(:func:`require_unaugmented`).
Passing the mesh is not a stability test: an instability between mesh points,
in particular a negative stiffness next to a Goldstone mode, goes unnoticed;
:func:`~spintoolkit.methods.stability.local_stability` tests continuous ``k`` (D51).

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
from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.methods.result import ResultHeader, to_jsonable
from spintoolkit.observables.bose_statistics import compute_static_magnon_kernel
from spintoolkit.states.spin_state import SpinState, validate_spin_state
from spintoolkit.system.cluster import allowed_momenta, expand_on_torus
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import SpinModel

REGULARIZATIONS = ("none", "MAGSWT", "k-dependent")


class LSWTError(ValueError):
    """The LSWT Hamiltonian cannot be diagonalized as requested."""


def require_unaugmented(result, observable: str) -> None:
    """Refuse a MAGSWT result (D62) for anything but its ground-state energy.

    The bands, eigenvectors and boson numbers of an augmented result belong to
    ``H + mu sum_i a_i^dagger a_i``, not to the spin model.

    Raises
    ------
    ValueError
        If ``result.extra["augmentation"]`` is set.
    """
    augmentation = getattr(result, "extra", {}).get("augmentation")
    if augmentation is not None:
        raise ValueError(f"{observable} is not defined for a MAGSWT result "
                         f"({augmentation['label']}): its magnons are those of the augmented "
                         "Hamiltonian, not of the spin model (D62); only the ground-state "
                         "energy is reported")


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
        Unless the set is closed under ``k -> -k`` modulo the magnetic
        reciprocal lattice, only the bands are defined: the zero-point
        energy and the boson numbers are NaN.
    regularization : {"none", "MAGSWT", "k-dependent"}
        A regularization only lifts zero modes (e.g. a Goldstone mode on the
        mesh). A negative eigenvalue of ``H(k)`` below ``-zero_mode_tolerance *
        max |eig H(k)|`` means the reference state is unstable and raises
        :class:`LSWTError` with any regularization: no energies are reported
        for an unstable state.
    stationarity_tolerance : float
        Largest torque (E0) accepted without a warning.
    zero_mode_tolerance : float
        Without regularization, ``min eig H(k) <= zero_mode_tolerance *
        max |eig H(k)|`` is a zero mode or an instability.
    gapless : bool, optional
        The user's decision after inspecting zero-mode candidates (4b); used
        only for finite-temperature quantities, recorded in the header.
    augmentation : None, "minimal" or float
        MAGSWT (D62). ``None`` (default): plain LSWT, an unstable state raises
        :class:`LSWTError`. ``"minimal"``: add the smallest shift
        ``mu sum_i a_i^dagger a_i`` that makes ``H(k)`` positive semidefinite
        at every continuous ``k``. A float ``mu >= 0`` uses that shift (it
        must be at least the minimal one). Needs ``regularization="none"``
        and zero temperature; the result is an upper estimate of the energy.
    """

    mesh: Tuple[int, int] = LSWT_DEFAULT_MESH
    shift: bool = True
    k_points: Optional[Any] = None
    regularization: str = "none"
    stationarity_tolerance: float = LSWT_STATIONARITY_TOLERANCE
    zero_mode_tolerance: float = LSWT_ZERO_MODE_TOLERANCE
    gapless: Optional[bool] = None
    augmentation: Optional[Any] = None

    def __post_init__(self):
        if self.regularization not in REGULARIZATIONS:
            raise ValueError(f"regularization must be one of {REGULARIZATIONS}")
        if self.augmentation is not None:
            if isinstance(self.augmentation, str):
                if self.augmentation != "minimal":
                    raise ValueError("augmentation must be None, 'minimal' or a number >= 0")
            elif not np.isfinite(float(self.augmentation)) or float(self.augmentation) < 0:
                raise ValueError("augmentation must be None, 'minimal' or a number >= 0")
            else:
                object.__setattr__(self, "augmentation", float(self.augmentation))
            if self.regularization != "none":
                raise ValueError("augmentation replaces regularization; use regularization='none'")
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
                "zero_mode_tolerance": self.zero_mode_tolerance, "gapless": self.gapless,
                "augmentation": self.augmentation}


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
        Per site; ``ground_state_energy = classical + zero_point``. The
        zero-point and ground-state energies are NaN when the reference state
        is not stationary (linear boson terms remain, so the expansion has no
        harmonic order), or when explicit ``k_points`` are not closed under
        ``k -> -k`` (the formula relabels the hole momentum).
    boson_numbers : (Ns,) array
        ``<a_i^dagger a_i>`` at zero temperature; NaN when the reference state
        is not stationary or explicit ``k_points`` are not closed under
        ``k -> -k``.
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
        """Ordered moment ``S_i - <n_i>`` of every magnetic site at zero temperature.

        NaN where ``<n_i> > S_i``: the moment would reverse, which is outside
        the spin-wave expansion.
        """
        moments = self.spins - self.boson_numbers
        return np.where(moments < 0, np.nan, moments)

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


def _closed_under_inversion(k: np.ndarray, magnetic: np.ndarray, tol: float = 1e-9) -> bool:
    """Whether every -k of the set equals some k up to a magnetic reciprocal vector.

    The zero-point energy and the boson numbers of :func:`solve_lswt` relabel
    the hole momentum -k as k, which needs such a set (research note *LSWT
    paraunitary diagonalization*). The default mesh and the finite torus are
    closed; explicit ``k_points`` need not be.
    """
    p = np.asarray(k, dtype=float) @ np.asarray(magnetic, dtype=float).T / (2 * np.pi)
    d = (-p)[:, None, :] - p[None, :, :]
    d -= np.round(d)
    return bool(np.all(np.any(np.all(np.abs(d) < tol, axis=-1), axis=1)))


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
    lowest = np.array([np.linalg.eigvalsh(Hk)[[0, -1]] for Hk in H])
    negative = lowest[:, 0] < -zero_mode_tolerance * np.max(np.abs(lowest), axis=1)
    if np.any(negative):
        i = int(np.flatnonzero(negative)[0])
        raise LSWTError(
            f"H(k) has a negative eigenvalue at {int(negative.sum())} of {nk} momenta (first "
            f"k={np.round(k[i], 6).tolist()}, min eig {lowest[i, 0]:.3g}): the reference state "
            f"is unstable, and regularization '{regularization}' only lifts zero modes.")
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
    if settings.augmentation is not None and conditions.temperature > 0:
        raise ValueError("finite-temperature quantities are not defined for a MAGSWT result "
                         "(D62): the thermal magnons would be those of the augmented Hamiltonian")
    geometry = geometry or CalculationGeometry.thermodynamic_limit()
    validate_spin_state(state, model, geometry)
    k, fractional = _momenta(model, state, geometry, settings)

    hamiltonian = QuadraticBoseHamiltonian(model, state, conditions)
    H = hamiltonian.at(k)
    linear_max = float(np.max(np.abs(hamiltonian.linear_terms), initial=0.0))
    augmentation = None
    if settings.augmentation is not None:
        from spintoolkit.methods.lswt.augmentation import find_augmentation

        augmentation = find_augmentation(hamiltonian, state.magnetic_lattice(model), k,
                                         settings.augmentation)
        H = H + augmentation.mu * np.eye(H.shape[1])
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
    if augmentation is not None:
        # The boson numbers are those of the augmented Hamiltonian; their mean is
        # dE/dmu, which bounds the energy offset of the padding above mu_min.
        diagnostics["augmentation"] = augmentation.as_dict()
        diagnostics["augmentation_padding_energy"] = float(
            (augmentation.mu - augmentation.mu_min) * np.mean(boson_numbers))
        diagnostics["energy_bound"] = augmentation.label
        boson_numbers = np.full(len(boson_numbers), np.nan)
    if settings.k_points is not None:
        diagnostics["inversion_closed"] = _closed_under_inversion(k, state.magnetic_lattice(model))
        if not diagnostics["inversion_closed"]:
            # Band paths are the usual open sets: only the vacuum quantities are
            # undefined there, so no warning (diagnostics["inversion_closed"]).
            zero_point = float("nan")
            boson_numbers = np.full(len(boson_numbers), np.nan)
    if not diagnostics["stationary"]:
        warnings.warn(f"reference state is not stationary (max torque {torque:.3g} E0): linear "
                      "boson terms remain and LSWT is not a consistent expansion; the zero-point "
                      "energy and the boson numbers are NaN", UserWarning, stacklevel=2)
        zero_point = float("nan")
        boson_numbers = np.full(len(boson_numbers), np.nan)
    header = ResultHeader.build(
        "lswt", model, state, geometry, conditions, settings.as_dict(),
        "energies per site of the model (E0); boson numbers per magnetic site", diagnostics)
    hamiltonian_at = hamiltonian.at
    hamiltonian_derivatives_at = hamiltonian.derivatives_at

    result = LSWTResult(header, keys, np.array([model.site(s).spin for s, _ in keys]),
                        np.array([state.direction(s, c) for s, c in keys]), k, fractional, weights,
                        H, E, T, shift, linear_max, float(e_cl), float(zero_point),
                        float(e_cl + zero_point), boson_numbers,
                        np.asarray(model.lattice, dtype=float), state.magnetic_lattice(model),
                        None, np.array([model.cartesian_position(s, c) for s, c in keys]),
                        hamiltonian.local_frames,
                        hamiltonian_at, hamiltonian_derivatives_at=hamiltonian_derivatives_at)
    if augmentation is not None:
        result.extra["augmentation"] = augmentation.as_dict()
    if conditions.temperature > 0:
        from spintoolkit.observables.thermal import thermal_quantities

        result = replace(result, thermal=thermal_quantities(
            result, [conditions.temperature], gapless=settings.gapless))
    return result
