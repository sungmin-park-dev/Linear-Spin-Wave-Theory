"""Berry curvature and Chern numbers of LSWT magnon bands (stage 5a, D29).

Input is an :class:`~spintoolkit.methods.lswt.run.LSWTResult`; its stored
paraunitary eigenvectors are reused, and only ``dH/dk`` is evaluated
(:attr:`LSWTResult.hamiltonian_derivatives_at`).

Conventions (D29):

- Gauge: ``H(k)`` is the Bloch Hamiltonian of the Fourier convention D13 (full
  positions), i.e. the matrix of the one-magnon states
  ``a_k^dagger = N^-1/2 sum_r exp(i k . r) a_r^dagger``. The curvature
  ``Omega_n(k)`` is reported in this gauge. Pointwise values depend on the
  gauge (a cell gauge adds a curl of the sublattice-weighted positions);
  Chern numbers and the thermal Hall conductivity do not.
- Sign: ``Omega_n = dA_y/dk_x - dA_x/dk_y`` with ``A_n = i <u_n| eta grad |u_n>``;
  in Kubo form (Shindou et al., PRB 87, 174427 (2013); Matsumoto and
  Murakami, PRB 84, 184406 (2011))

      Omega_n = -2 Im sum_{m != n} eta_n eta_m (T^+ dH_x T)_nm (T^+ dH_y T)_mn
                / (lambda_n - lambda_m)^2,

  ``lambda = eta * E`` the signed BdG energies. ``C_n = (1/2 pi) int_BZ Omega_n``.
  Curvature carries the square of the model's length unit.
- Bands are the particle bands in ascending energy at each k. A band whose
  signed BdG separation from any other mode is at most ``band_gap_cutoff``
  (E0) at some k has undefined curvature there (NaN), and its Chern number is
  NaN. The cutoff is a numerical policy, not a physical gap or an error bound.

:func:`chern_numbers_fhs` computes the Chern numbers independently from the
eigenvectors alone by lattice link variables (Fukui, Hatsugai and Suzuki,
JPSJ 74, 1674 (2005)), with the paraunitary inner product. Neither is proof
on its own: a gap that closes between mesh points (e.g. Dirac points) passes
the mesh checks, the FHS sum is then still an integer but may be wrong, and
the Kubo sum is not an integer on a coarse mesh. :func:`chern_numbers`
accepts a Chern number only when the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import warnings

from spintoolkit.definitions.defaults import (
    TOPOLOGY_BAND_GAP_CUTOFF, TOPOLOGY_CHERN_AGREEMENT, TOPOLOGY_MIN_LINK_OVERLAP)
from spintoolkit.observables.topology import compute_berry_curvature


class TopologyError(ValueError):
    """The result cannot give a well-defined topological quantity."""


@dataclass(frozen=True)
class BerryCurvature:
    """Berry curvature of the particle bands on the momenta of an LSWT result.

    Attributes
    ----------
    k_points : (nk, 2) array
    weights : (nk,) array
    energies : (nk, Ns) array
        Magnon energies, ascending at each k (E0).
    curvature : (nk, Ns) array
        ``Omega_n(k)`` in the same band order (length unit squared); NaN where
        the band is not separated by more than ``band_gap_cutoff``.
    level_spacing : (nk, Ns) array
        Smallest separation of each band from every other signed BdG mode.
    band_gap_cutoff : float
    cell_area : float
        Area of the magnetic cell.
    full_zone : bool
        The momenta form a complete uniform mesh of the magnetic Brillouin zone.
    """

    k_points: np.ndarray
    weights: np.ndarray
    energies: np.ndarray
    curvature: np.ndarray
    level_spacing: np.ndarray
    band_gap_cutoff: float
    cell_area: float
    full_zone: bool

    def chern_numbers(self) -> np.ndarray:
        """Kubo Chern numbers ``(2 pi / A_cell) <Omega_n>`` (not rounded); NaN if undefined."""
        if not self.full_zone:
            raise TopologyError("Chern numbers need a complete uniform mesh of the magnetic "
                                "Brillouin zone (LSWTSettings.mesh, no explicit k-points)")
        return 2 * np.pi / self.cell_area * (self.weights @ self.curvature)


def _check_result(result) -> None:
    if result.hamiltonian_derivatives_at is None:
        raise TopologyError("the result has no Hamiltonian derivatives; use solve_lswt")
    regularization = result.header.settings.get("regularization", "none")
    if regularization == "k-dependent":
        raise TopologyError("a k-dependent regularization changes dH/dk; the curvature is "
                            "defined only without it")
    if np.any(result.regularization_shift != 0):
        raise TopologyError("the diagonalized H(k) carries a regularization shift; Berry "
                            "curvature needs regularization='none' and a gapped mesh")


def _mesh_indices(result) -> Optional[Tuple[np.ndarray, int, int]]:
    """Grid indices ``(nk, 2)`` and shape when the momenta form a complete uniform mesh."""
    f = result.fractional
    if f is None:
        return None
    f = np.mod(np.asarray(f, dtype=float), 1.0)
    axes = []
    for column in f.T:
        values = np.unique(np.round(column, 9))
        n = len(values)
        if not np.allclose(np.diff(values), 1.0 / n, rtol=0, atol=1e-8):
            return None
        axes.append((values, n))
    (v1, n1), (v2, n2) = axes
    if n1 * n2 != len(f):
        return None
    index = np.column_stack([np.searchsorted(v1, np.round(f[:, 0], 9)),
                             np.searchsorted(v2, np.round(f[:, 1], 9))])
    if len(np.unique(index, axis=0)) != len(f):
        return None
    return index, n1, n2


def _particle_order(energies: np.ndarray, ns: int) -> np.ndarray:
    return np.argsort(energies[:, :ns], axis=1)


def berry_curvature(result, band_gap_cutoff: float = TOPOLOGY_BAND_GAP_CUTOFF) -> BerryCurvature:
    """Berry curvature of every particle band at the momenta of ``result``.

    Parameters
    ----------
    result : LSWTResult
        Without regularization shift.
    band_gap_cutoff : float
        Absolute E0 threshold on the signed BdG separation (see the module
        docstring).

    Returns
    -------
    BerryCurvature
    """
    _check_result(result)
    ns = result.num_sites
    dx, dy = result.hamiltonian_derivatives_at(result.k_points)
    order = _particle_order(result.eigenvalues, ns)
    curvature = np.empty((len(result.k_points), ns))
    spacing = np.empty_like(curvature)
    for i, (E, T) in enumerate(zip(result.eigenvalues, result.eigenvectors)):
        omega, gap = compute_berry_curvature(E, T, [dx[i], dy[i]], num_sl=ns,
                                             band_gap_cutoff=band_gap_cutoff)
        curvature[i], spacing[i] = omega[order[i]], gap[order[i]]
    energies = np.take_along_axis(result.eigenvalues[:, :ns], order, axis=1)
    return BerryCurvature(result.k_points, result.weights, energies, curvature, spacing,
                          float(band_gap_cutoff), float(abs(np.linalg.det(result.magnetic_lattice))),
                          _mesh_indices(result) is not None)


def zone_gauge(result) -> Tuple[int, np.ndarray]:
    """Sign ``s`` with ``H(k + G) = D(G) H(k) D(G)^dagger``, ``D(G) = diag(e^{i s G.r}, e^{i s G.r})``.

    Both Nambu blocks take the same phase: ``a_(k+G)`` and ``a_-(k+G)^dagger``
    change by the same factor per site. Returns the sign and the magnetic
    reciprocal vectors (rows).
    """
    reciprocal = 2 * np.pi * np.linalg.inv(result.magnetic_lattice).T
    k = np.array([[0.137, -0.291]]) @ reciprocal
    for sign in (1, -1):
        ok = True
        for G in reciprocal:
            D = _gauge_matrix(result.positions, G, sign)
            lhs = result.hamiltonian_at(k + G)[0]
            rhs = D @ result.hamiltonian_at(k)[0] @ D.conj().T
            ok &= np.allclose(lhs, rhs, rtol=0, atol=1e-12 * max(1.0, np.max(np.abs(lhs))))
        if ok:
            return sign, reciprocal
    raise TopologyError("H(k + G) is not a gauge transform of H(k); cannot link across the "
                        "zone boundary")


def _gauge_matrix(positions, G, sign) -> np.ndarray:
    phase = np.exp(1j * sign * (positions @ G))
    return np.diag(np.r_[phase, phase])


def chern_numbers_fhs(result, band_gap_cutoff: float = TOPOLOGY_BAND_GAP_CUTOFF,
                      min_link_overlap: float = TOPOLOGY_MIN_LINK_OVERLAP) -> np.ndarray:
    """Chern numbers from lattice link variables of the stored eigenvectors.

    The momenta must form a complete uniform mesh; links that cross the zone
    boundary use the gauge matrix of :func:`zone_gauge`. Link variables use the
    paraunitary product ``u^dagger eta u'``. The sum of plaquette phases is an
    integer for any mesh once each band is separated; a band that is not
    separated by more than ``band_gap_cutoff`` at some mesh point gives NaN.
    A link overlap ``|u^dagger eta u'|`` at or below ``min_link_overlap``
    (FHS admissibility) also gives NaN with a warning: the energy-ordered band
    changes character between neighbouring points, i.e. it crosses another
    band between mesh points or the mesh does not resolve it. The mesh check
    of ``band_gap_cutoff`` alone cannot see such a crossing, and the Kubo sum
    of :meth:`BerryCurvature.chern_numbers` does not detect it either.

    Returns
    -------
    (Ns,) array
        Chern numbers of the particle bands in ascending energy (float, exact
        integers up to round-off).
    """
    _check_result(result)
    mesh = _mesh_indices(result)
    if mesh is None:
        raise TopologyError("the FHS Chern number needs a complete uniform mesh of the magnetic "
                            "Brillouin zone")
    index, n1, n2 = mesh
    ns = result.num_sites
    sign, reciprocal = zone_gauge(result)
    grid = np.empty((n1, n2), dtype=int)
    grid[index[:, 0], index[:, 1]] = np.arange(len(index))
    order = _particle_order(result.eigenvalues, ns)
    eta = np.r_[np.ones(ns), -np.ones(ns)]
    signed = result.eigenvalues * eta
    spacing = np.array([[np.min(np.abs(np.delete(s, c) - s[c])) for c in o]
                        for s, o in zip(signed, order)])
    k = result.k_points
    steps = (reciprocal[0] / n1, reciprocal[1] / n2)

    def vector(i, j, band):
        """Eigenvector of grid point (i mod n1, j mod n2), carried to the unwrapped momentum."""
        p = grid[i % n1, j % n2]
        u = result.eigenvectors[p][:, order[p, band]]
        expected = k[grid[0, 0]] + i * steps[0] + j * steps[1]
        G = expected - k[p]
        if np.linalg.norm(G) > 1e-12 * max(1.0, np.linalg.norm(reciprocal)):
            u = _gauge_matrix(result.positions, G, sign) @ u
        return u

    offsets = (k[grid[0, 0]] + np.add.outer(np.arange(n1), np.zeros(n2))[..., None] * steps[0]
               + np.add.outer(np.zeros(n1), np.arange(n2))[..., None] * steps[1]
               - k[grid])
    coefficients = offsets @ np.linalg.inv(reciprocal)
    if not np.allclose(coefficients, np.rint(coefficients), atol=1e-8):
        raise TopologyError("mesh momenta are not a translated uniform grid")
    orientation = np.sign(np.linalg.det(reciprocal))

    chern = np.full(ns, np.nan)
    for band in range(ns):
        if np.any(spacing[:, band] <= band_gap_cutoff):
            continue
        total, weakest = 0.0, np.inf
        for i in range(n1):
            for j in range(n2):
                u00 = vector(i, j, band)
                u10 = vector(i + 1, j, band)
                u11 = vector(i + 1, j + 1, band)
                u01 = vector(i, j + 1, band)
                links = (np.vdot(u00, eta * u10), np.vdot(u10, eta * u11),
                         np.vdot(u11, eta * u01), np.vdot(u01, eta * u00))
                weakest = min(weakest, min(abs(x) for x in links))
                total += np.angle(np.prod(links))
        if weakest <= min_link_overlap:
            warnings.warn(f"band {band}: smallest link overlap {weakest:.3g} <= {min_link_overlap}; "
                          "the band crosses another band between mesh points or the mesh does not "
                          "resolve it, so its Chern number is undefined here (NaN)",
                          UserWarning, stacklevel=2)
            continue
        chern[band] = -orientation * total / (2 * np.pi)
    return chern


def chern_numbers(result, band_gap_cutoff: float = TOPOLOGY_BAND_GAP_CUTOFF,
                  tolerance: float = TOPOLOGY_CHERN_AGREEMENT,
                  curvature: Optional[BerryCurvature] = None) -> np.ndarray:
    """Chern numbers accepted when the Kubo integral and the FHS integer agree.

    Parameters
    ----------
    result : LSWTResult
        On a complete uniform mesh.
    band_gap_cutoff : float
    tolerance : float
        Largest accepted ``|C_Kubo - C_FHS|``.
    curvature : BerryCurvature, optional
        Reused if already computed with the same cutoff.

    Returns
    -------
    (Ns,) array
        The FHS integers where accepted, NaN elsewhere (with a warning naming
        both values: refine the mesh, or the gap closes).
    """
    curvature = curvature if curvature is not None else berry_curvature(result, band_gap_cutoff)
    kubo = curvature.chern_numbers()
    fhs = chern_numbers_fhs(result, band_gap_cutoff)
    accepted = np.where(np.abs(kubo - fhs) <= tolerance, np.rint(fhs), np.nan)
    disagree = np.isnan(accepted) & ~np.isnan(kubo) & ~np.isnan(fhs)
    if np.any(disagree):
        bands = np.flatnonzero(disagree).tolist()
        warnings.warn(f"bands {bands}: Kubo {np.round(kubo[disagree], 4).tolist()} and FHS "
                      f"{np.round(fhs[disagree], 4).tolist()} disagree by more than {tolerance}; "
                      "refine the mesh, or the gap closes between mesh points", UserWarning,
                      stacklevel=2)
    return accepted
