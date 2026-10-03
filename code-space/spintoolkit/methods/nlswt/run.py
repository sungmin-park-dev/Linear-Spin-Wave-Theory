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
from typing import Any, Dict, Optional, Tuple

import numpy as np

from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.methods.lswt.run import LSWTError
from spintoolkit.methods.nlswt.engine import NonlinearEnergies, NonlinearSpinWaves, bogoliubov_mesh
from spintoolkit.methods.nlswt.expansion import expand_model
from spintoolkit.methods.result import ResultHeader
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


@dataclass(frozen=True)
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
    """

    header: ResultHeader
    energies: NonlinearEnergies
    k_points: np.ndarray
    solver: NonlinearSpinWaves = field(repr=False, compare=False)

    @property
    def ground_state_energy(self) -> float:
        """``E_cl + E_zp + E_(S^0)`` per site."""
        return self.energies.total

    def magnon_energies(self, k_points, broadening: float = 0.0) -> Dict[str, np.ndarray]:
        """Magnon energies through order S^0 (on-shell 1/S correction).

        Parameters
        ----------
        k_points : (m, 2) array_like
            Cartesian momenta.
        broadening : float
            ``eta`` in ``Sigma_3(k, w + i eta)``; zero is exact on a finite
            torus, a small positive value regularizes the principal value where
            ``w`` crosses the two-magnon continuum on a mesh.

        Returns
        -------
        dict
            ``lswt`` (m, Ns), ``hartree_fock_tadpole`` (m, Ns), ``cubic``
            (m, Ns, complex), ``total`` = lswt + static + Re cubic, and
            ``decay_rate`` = -Im cubic (meaningful with a broadening; the
            quasiparticle half width at order S^0).
        """
        lswt, static, cubic = self.solver.magnon_energies(k_points, broadening)
        return {"lswt": lswt, "hartree_fock_tadpole": static, "cubic": cubic,
                "total": lswt + static + cubic.real, "decay_rate": -cubic.imag}


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
    return NLSWTResult(header, energies, k, solver)
