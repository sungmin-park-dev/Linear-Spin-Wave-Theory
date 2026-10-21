"""Nonlinear spin waves on the common model types: ``solve_nlswt(model, state, ...)`` (D46).

Energies through order ``S^0`` (relative ``1/S^2``) and magnon energies through
order ``S^0`` (relative ``1/S``) about a classical stationary state; see
:mod:`spintoolkit.methods.nlswt.engine` for the formulas and
:mod:`spintoolkit.methods.nlswt.expansion` for the Holstein-Primakoff
truncation. Everything is perturbation theory about the LSWT vacuum, so the
expansion is defined only where LSWT is: every mode must be gapped on the
momenta used (Goldstone modes at the zone centre are avoided by the mesh), and
the tadpole must have no component along a zero mode of ``H(k=0)``.

Energies are per site in the energy unit E0 of the model (D20).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.methods.lswt.run import LSWTError
from spintoolkit.methods.nlswt.engine import (NonlinearEnergies, NonlinearSpinWaves,
                                              bogoliubov_mesh, hilbert_transform)
from spintoolkit.methods.nlswt.expansion import expand_model
from spintoolkit.methods.result import ResultHeader, freeze_fields, to_jsonable
from spintoolkit.states.spin_state import SpinState, validate_spin_state
from spintoolkit.system.cluster import allowed_momenta, expand_on_torus
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import SpinModel


@dataclass(frozen=True)
class NLSWTSettings:
    """Method settings of :func:`solve_nlswt`.

    Parameters
    ----------
    mesh : (int, int)
        Thermodynamic limit: ``N1 x N2`` mesh of the magnetic reciprocal cell,
        offset by one third of a step, ``((n1 + 1/3)/N1, (n2 + 1/3)/N2)``. This
        mesh is closed under ``q1 + q2 + q3 = 0`` (needed by three-magnon sums)
        and never contains the zone centre.
    zero_mode_tolerance : float
        ``min eig H(k) <= zero_mode_tolerance * max |eig H(k)|`` on the mesh
        raises (zero mode or instability).
    kernel_tolerance : float
        Relative size of a zero mode of ``H(k=0)`` in the tadpole.
    """

    mesh: Tuple[int, int] = (24, 24)
    zero_mode_tolerance: float = 1e-9
    kernel_tolerance: float = 1e-9

    def __post_init__(self):
        mesh = tuple(int(n) for n in self.mesh)
        if len(mesh) != 2 or min(mesh) < 1:
            raise ValueError("mesh must be two positive integers")
        object.__setattr__(self, "mesh", mesh)

    def as_dict(self) -> Dict[str, Any]:
        return {"mesh": list(self.mesh), "mesh_offset": "1/3 step",
                "zero_mode_tolerance": self.zero_mode_tolerance,
                "kernel_tolerance": self.kernel_tolerance}


@dataclass(frozen=True, eq=False)
class NLSWTResult:
    """Ground-state energy through order S^0 with the common header.

    Attributes
    ----------
    header : ResultHeader
    energies : NonlinearEnergies
        ``classical``, ``zero_point``, ``hartree_fock``, ``cubic``,
        ``tadpole`` per site; ``total`` and ``order_s0``.
    k_points : (nk, 2) array
        Momenta of all loop sums.
    solver : NonlinearSpinWaves
        The perturbation theory, for magnon energies at other momenta.
    lattice : (2, 2) array
        Primitive lattice of the model (rows), for named momentum paths.
    """

    header: ResultHeader
    energies: NonlinearEnergies
    k_points: np.ndarray
    solver: NonlinearSpinWaves = field(repr=False, compare=False)
    lattice: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        freeze_fields(self)

    def to_json_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Header, the energy pieces per site, the loop momenta and the lattice.

        The perturbation theory (``solver``) is not serialized; magnon energies
        at other momenta need the result object.
        """
        e = self.energies
        body = {"classical": e.classical, "zero_point": e.zero_point,
                "hartree_fock": e.hartree_fock, "cubic": e.cubic, "tadpole": e.tadpole,
                "order_s0": e.order_s0, "total": e.total, "condensate": e.condensate,
                "k_points": self.k_points, "lattice": self.lattice}
        return to_jsonable({"header": self.header, "nlswt": body})

    @property
    def ground_state_energy(self) -> float:
        """``E_cl + E_zp + E_(S^0)`` per site."""
        return self.energies.total

    def magnon_energies(self, k_points, eta: float = 0.0, *,
                        broadening: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Magnon energies through order S^0 (on-shell 1/S correction).

        Parameters
        ----------
        k_points : (m, 2) array_like
            Cartesian momenta.
        eta : float
            Broadening in ``Sigma_3(k, w + i eta)``; zero is exact on a finite
            torus, a small positive value regularizes the principal value where
            ``w`` crosses the two-magnon continuum on a mesh.
        broadening : float, optional
            Deprecated name of ``eta`` (the name used by the other spectral
            functions); removed in 0.3.

        Returns
        -------
        dict
            ``lswt`` (m, Ns), ``hartree_fock_tadpole`` (m, Ns), ``cubic``
            (m, Ns, complex), ``total`` = lswt + static + Re cubic, and
            ``decay_rate`` = -Im cubic (the quasiparticle half width at order
            S^0). ``total`` is NaN where it is negative: a magnon of negative
            energy is outside the 1/S expansion, and where the shift is a
            near-cancellation the mesh cannot resolve (D69: near a Goldstone
            mode folded to the zone centre, momenta off the grid
            ``(j1/N1, j2/N2)`` of the magnetic reciprocal cell, with a warning). In the thermodynamic limit
            ``decay_rate`` at ``eta = 0`` is NaN (a mesh sum of delta
            functions does not give the rate; use :meth:`decay_rates`).
        """
        if broadening is not None:
            warnings.warn("magnon_energies(broadening=) is deprecated; use eta=", DeprecationWarning,
                          stacklevel=2)
            eta = broadening
        lswt, static, cubic = self.solver.magnon_energies(k_points, eta)
        total = _physical_energies(lswt + static + cubic.real, lswt)
        if self.header.geometry.get("kind") != "finite_torus":
            k = np.atleast_2d(np.asarray(k_points, dtype=float))
            mesh = tuple(self.header.settings["mesh"])
            unsafe = _cancellation_limited(lswt, static, cubic.real, min(mesh))
            unsafe &= ~_on_mesh_grid(k, self.solver.mesh.reciprocal, mesh)[:, None]
            total = _flag_cancellation(total, unsafe, "magnon_energies", min(mesh))
        rate = -cubic.imag
        if eta == 0 and self.header.geometry.get("kind") != "finite_torus":
            rate = np.full(rate.shape, np.nan)
        return {"lswt": lswt, "hartree_fock_tadpole": static, "cubic": cubic,
                "total": total, "decay_rate": rate}


    def decay_rates(self, k_points, grid: Tuple[int, int] = (64, 64)) -> Dict[str, np.ndarray]:
        """Spontaneous two-magnon decay rates in the Born approximation (D57).

        ``Gamma_n(k) = -Im Sigma_3^nn(k, w_n(k) + i0)``, the golden rule with
        the cubic decay vertex and the LSWT energies, integrated over the
        magnetic zone by linear interpolation on triangles of an ``N1 x N2``
        grid (no broadening). It is the half width of the magnon peak at order
        S^0 (relative 1/S). Unlike ``magnon_energies(..., broadening)``, the
        result does not depend on an artificial ``eta``.

        Parameters
        ----------
        k_points : (m, 2) array_like
            Cartesian momenta; ``k`` and ``k - q`` must avoid zero modes on the grid.
        grid : (N1, N2)
            Integration grid, offset by one third of a step.

        Returns
        -------
        dict
            ``lswt`` (m, Ns) and ``decay_rate`` (m, Ns).

        Notes
        -----
        The rate is exactly zero where ``w_n(k)`` lies below the two-magnon
        continuum, and also where the vertex vanishes by symmetry although
        the decay is kinematically allowed. Near the edge of the continuum the
        Born approximation fails (threshold singularities, Zhitomirsky and
        Chernyshev, RMP 85, 219 (2013)), and a decay region much narrower than
        a grid step is missed; refine ``grid`` there.
        """
        k = np.atleast_2d(np.asarray(k_points, dtype=float))
        pairs = [self.solver.decay_rate(ki, grid) for ki in k]
        return {"lswt": np.array([p[0] for p in pairs]),
                "decay_rate": np.array([p[1] for p in pairs])}

    def spectral_function(self, k_points, omega, eta: float, mode: str = "on_shell",
                          grid: Tuple[int, int] = (64, 64),
                          levels: int = 400) -> Dict[str, np.ndarray]:
        """Magnon spectral functions dressed by the order-S^0 self-energy (D58).

        The self-energy of band n is ``dE_n(k) + Sigma_3^nn(k, w)``: the
        Hartree-Fock + tadpole shift and the cubic self-energy, whose
        frequency dependence comes from the two-magnon densities of
        :meth:`NonlinearSpinWaves.cubic_spectral_densities` through an exact
        Hilbert transform (no broadening inside the self-energy).

        ``mode="on_shell"`` (default) is the strict expansion to relative order
        1/S: a Lorentzian at ``E_n = w_n + dE_n + Re Sigma_3(w_n)`` with half
        width ``Gamma_n + eta``, ``Gamma_n = -Im Sigma_3(w_n)`` (D57).
        ``mode="dyson"`` resums the self-energy,
        ``A_n(k, w) = -(1/pi) Im 1 / (w + i eta - w_n - dE_n - Sigma_3^nn(k, w))``,
        which gives line shapes, weight moved into the continuum and the
        pinning of a peak at a continuum edge, but is not controlled in 1/S:
        where the corrections are large (S = 1/2) it can produce spurious poles.
        Its ``negative_frequency_weight`` is the check: for a bosonic magnon
        the weight below w = 0 is ``-sum |<m|b|0>|^2 <= 0``, and a positive
        value means that the resummation has produced an unphysical pole.

        Parameters
        ----------
        k_points : (m, 2) array_like
            Cartesian momenta, away from zero modes.
        omega : (nw,) array_like
        eta : float
            Instrumental Lorentzian half width, positive.
        mode : "on_shell" or "dyson"
        grid, levels :
            Triangle grid of the zone and number of energies of the densities.

        Returns
        -------
        dict
            ``lswt``, ``hartree_fock_tadpole``, ``energy`` (on-shell 1/S energy)
            and ``decay_rate``, each (m, Ns); ``spectral`` (m, Ns, nw); with
            ``mode="dyson"`` also ``self_energy`` (m, Ns, nw) and
            ``negative_frequency_weight`` (m, Ns), the integral of ``A`` below
            zero over the given ``omega`` grid (NaN when the grid has fewer
            than two negative frequencies: not checked). Where the on-shell
            energy is negative (outside the 1/S expansion) ``energy`` and the
            on-shell ``spectral`` of that band are NaN; so are ``energy`` and
            ``spectral`` (both modes) where the shift is a near-cancellation
            the mesh cannot resolve (D69, with a warning).

        Notes
        -----
        The self-energy is band diagonal (off-diagonal elements change the
        energies at higher order); within an exactly degenerate group of
        bands the diagonal elements depend on the basis, and their mean (the
        basis-independent trace) is used for every band of the group.
        """
        if eta <= 0:
            raise ValueError("eta must be positive")
        if mode not in ("on_shell", "dyson"):
            raise ValueError("mode must be 'on_shell' or 'dyson'")
        k = np.atleast_2d(np.asarray(k_points, dtype=float))
        w = np.asarray(omega, dtype=float)
        ns = self.solver.ns
        E_all = np.zeros((len(k), ns))
        static = np.zeros((len(k), ns))
        on_shell = np.zeros((len(k), ns), dtype=complex)
        sigma = np.zeros((len(k), ns, len(w)), dtype=complex)
        for i, ki in enumerate(k):
            E, x, rho_dec, rho_src = self.solver.cubic_spectral_densities(ki, grid, levels)
            _, T = self.solver._bogoliubov(ki.reshape(1, 2))
            dH = self.solver.static_hamiltonian(ki.reshape(1, 2))
            static[i] = np.einsum("an,ab,bn->n", T[0].conj(), dH[0], T[0])[:ns].real
            E_all[i] = E

            def cubic(n, freq):
                # -int rho_src(x) / (w + x + i0) is the conjugate of the transform at -w:
                # real for w > 0, it carries the negative-frequency (anomalous) weight
                return (hilbert_transform(x, rho_dec[n], freq)
                        + np.conj(hilbert_transform(x, rho_src[n], -np.asarray(freq))))

            for n in range(ns):
                on_shell[i, n] = cubic(n, [E[n]])[0]
                if mode == "dyson":
                    sigma[i, n] = cubic(n, w)
            for group in _degenerate_groups(E):
                if len(group) > 1:
                    static[i, group] = static[i, group].mean()
                    on_shell[i, group] = on_shell[i, group].mean()
                    sigma[i, group] = sigma[i, group].mean(axis=0)
        energy = _physical_energies(E_all + static + on_shell.real, E_all)
        unsafe = np.zeros(energy.shape, dtype=bool)
        if self.header.geometry.get("kind") != "finite_torus":
            mesh_size = min(self.header.settings["mesh"])
            unsafe = _cancellation_limited(E_all, static, on_shell.real, mesh_size)
            energy = _flag_cancellation(energy, unsafe, "spectral_function", mesh_size)
        rate = -on_shell.imag
        out = {"lswt": E_all, "hartree_fock_tadpole": static, "energy": energy,
               "decay_rate": rate}
        if mode == "on_shell":
            width = rate[:, :, None] + eta
            out["spectral"] = width / np.pi / ((w[None, None, :] - energy[:, :, None]) ** 2
                                               + width ** 2)
            return out
        denominator = (w[None, None, :] + 1j * eta - E_all[:, :, None] - static[:, :, None]
                       - sigma)
        spectral = -np.imag(1.0 / denominator) / np.pi
        spectral[unsafe] = np.nan
        below = w < 0
        weight = (np.trapz if hasattr(np, "trapz") else np.trapezoid)
        out.update(spectral=spectral, self_energy=sigma,
                   negative_frequency_weight=(weight(spectral[:, :, below], w[below], axis=-1)
                                              if below.sum() > 1 else np.full((len(k), ns), np.nan)))
        return out

#: The order-S^0 shift is computed only where its two parts, the Hartree-Fock +
#: tadpole term and Re Sigma_3, are each at most this fraction of the smallest
#: mesh size times the LSWT energy (D69).
CANCELLATION_FACTOR = 0.25


def _cancellation_limited(lswt: np.ndarray, static: np.ndarray, cubic: np.ndarray,
                          mesh_size: int) -> np.ndarray:
    """Bands whose 1/S shift is a near-cancellation that the mesh cannot resolve (D69).

    Near a Goldstone mode folded into the magnetic zone centre of a noncollinear
    state (the k = +-Q modes of the 120 degree state), the Hartree-Fock + tadpole
    term and Re Sigma_3 each grow like 1/|k| while their sum vanishes like |k|.
    Both are mesh sums, and their mismatch, of relative size ~1/N, is then
    multiplied by the ratio ``rho = max(|static|, |Re Sigma_3|) / w``. Measured on
    the triangular antiferromagnet (data-space/verification/261021-theory-reliability),
    ``rho > N / 4`` gives errors from 10% to several times the shift.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.maximum(np.abs(static), np.abs(cubic)) / np.abs(lswt)
    return ~(rho <= CANCELLATION_FACTOR * mesh_size)


def _on_mesh_grid(k: np.ndarray, reciprocal: np.ndarray, mesh) -> np.ndarray:
    """Momenta on the grid ``(j1 / N1, j2 / N2)`` of the magnetic reciprocal cell.

    For these, every ``k - q`` of the cubic mesh sum lies on the mesh (offset by
    -1/3 step), so the sum and the Hartree-Fock averages are one consistent
    finite set and the Goldstone cancellation is exact.
    """
    f = k @ np.linalg.inv(reciprocal) * np.asarray(mesh, dtype=float)
    return np.all(np.abs(f - np.rint(f)) < 1e-7, axis=1)


def _flag_cancellation(energy: np.ndarray, unsafe: np.ndarray, where: str,
                       mesh_size: int) -> np.ndarray:
    if not np.any(unsafe):
        return energy
    warnings.warn(
        f"{where}: the 1/S shift of {int(unsafe.sum())} band(s) is a cancellation of terms "
        f"more than {CANCELLATION_FACTOR} x {mesh_size} times the LSWT energy (near a "
        "Goldstone mode folded to the zone centre); the mesh cannot resolve it and the energy "
        "is NaN. Options: momenta on the mesh grid (multiples of 1/N of the magnetic "
        "reciprocal vectors, magnon_energies only), a finer mesh, or momenta further from "
        "the zero mode.", UserWarning, stacklevel=3)
    return np.where(unsafe, np.nan, energy)


def _physical_energies(energy: np.ndarray, lswt: np.ndarray) -> np.ndarray:
    """``energy`` with NaN where it is negative beyond round-off of the LSWT scale."""
    scale = max(float(np.max(np.abs(lswt), initial=0.0)), 1.0)
    return np.where(energy < -1e-10 * scale, np.nan, energy)


def _degenerate_groups(energies: np.ndarray, tolerance: float = 1e-8):
    """Index groups of equal energies (relative tolerance)."""
    order = np.argsort(energies)
    scale = max(float(np.max(np.abs(energies))), 1.0)
    groups, current = [], [int(order[0])]
    for a, b in zip(order[:-1], order[1:]):
        if energies[b] - energies[a] <= tolerance * scale:
            current.append(int(b))
        else:
            groups.append(current)
            current = [int(b)]
    groups.append(current)
    return groups


def _resolution(fractional: np.ndarray, limit: int = 4096) -> int:
    for n in range(1, limit + 1):
        if np.allclose(fractional * n, np.rint(fractional * n), atol=1e-7):
            return n
    raise LSWTError("momenta are not on a rational grid")


def _momenta(model, state, geometry, settings):
    magnetic = state.magnetic_lattice(model)
    reciprocal = 2 * np.pi * np.linalg.inv(magnetic).T
    if geometry.kind == "finite_torus":
        expand_on_torus(model, geometry)
        _, k_all = allowed_momenta(model, geometry)
        p = np.mod(k_all @ magnetic.T / (2 * np.pi), 1.0)
        p[np.isclose(p, 1.0, rtol=0, atol=1e-9)] = 0.0
        _, unique = np.unique(np.round(p, 9), axis=0, return_index=True)
        unique = np.sort(unique)
        grid = p[unique]
    else:
        n1, n2 = settings.mesh
        grid = np.array([((i + 1 / 3) / n1, (j + 1 / 3) / n2) for i in range(n1) for j in range(n2)])
    return grid @ reciprocal, grid, reciprocal


def solve_nlswt(model: SpinModel, state: SpinState,
                conditions: Optional[ExternalConditions] = None,
                geometry: Optional[CalculationGeometry] = None,
                settings: NLSWTSettings = NLSWTSettings()) -> NLSWTResult:
    """Interacting spin waves to order S^0 about a classical stationary state.

    Parameters
    ----------
    model : SpinModel
        Bilinear and Zeeman terms (onsite terms raise).
    state : SpinState
        Classical stationary state (linear boson terms must vanish).
    conditions : ExternalConditions, optional
        Field; temperature must be zero.
    geometry : CalculationGeometry, optional
        Thermodynamic limit (mesh of ``settings``) or a finite torus (its
        momenta; exact 1/S perturbation theory of that cluster, which needs
        every mode, including k = 0, to be gapped).
    settings : NLSWTSettings

    Returns
    -------
    NLSWTResult

    Raises
    ------
    LSWTError
        Zero or negative modes on the momenta, a non-stationary state, or a
        tadpole along a zero mode of ``H(0)`` (order by disorder: the state
        is not stationary at order S, so the order-S^0 energy is undefined).
    """
    conditions = conditions or ExternalConditions()
    if conditions.temperature > 0:
        raise ValueError("solve_nlswt is a zero-temperature method")
    geometry = geometry or CalculationGeometry.thermodynamic_limit()
    validate_spin_state(state, model, geometry)
    k, grid, reciprocal = _momenta(model, state, geometry, settings)
    quadratic = QuadraticBoseHamiltonian(model, state, conditions)
    expansion = expand_model(model, state, conditions)
    resolution = _resolution(np.mod(grid, 1.0))
    mesh = bogoliubov_mesh(quadratic.at, k, grid, reciprocal, settings.zero_mode_tolerance,
                           resolution)
    solver = NonlinearSpinWaves(expansion, mesh, quadratic.at, settings.kernel_tolerance)
    energies = solver.energies()
    diagnostics = {"num_k": len(k), "max_linear_term": solver.max_linear,
                   "momenta": ("finite torus" if geometry.kind == "finite_torus" else
                               f"{settings.mesh[0]}x{settings.mesh[1]} mesh offset by 1/3 step"),
                   "tadpole_norm": float(np.linalg.norm(solver.tadpole()[2]))}
    header = ResultHeader.build(
        "nlswt", model, state, geometry, conditions, settings.as_dict(),
        "energies per site of the model (E0)", diagnostics)
    return NLSWTResult(header, energies, k, solver, np.asarray(model.lattice, dtype=float))
