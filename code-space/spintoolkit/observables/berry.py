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
accepts a Chern number only when the two agree. A plaquette that encloses a
band touching (e.g. a Dirac point at D = 0) has phase exactly pi, where the
branch -pi or +pi is set by round-off; FHS returns NaN for such inadmissible
plaquettes instead of an arbitrary integer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import warnings

from spintoolkit.definitions.defaults import (
    TOPOLOGY_ADAPTIVE_ABSOLUTE, TOPOLOGY_ADAPTIVE_MAX_DEPTH, TOPOLOGY_ADAPTIVE_MAX_POINTS,
    TOPOLOGY_ADAPTIVE_RELATIVE, TOPOLOGY_BAND_GAP_CUTOFF, TOPOLOGY_CHERN_AGREEMENT,
    TOPOLOGY_MIN_LINK_OVERLAP, TOPOLOGY_PLAQUETTE_PHASE_MARGIN)
from spintoolkit.observables.topology import (
    c2_weight, c2_weight_derivative, compute_berry_curvature, curvature_pair_terms,
    weighted_curvature_sum)


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
    if getattr(result, "extra", {}).get("frame") == "rotating":
        raise TopologyError("Berry curvature, Chern numbers and thermal Hall of spiral magnons "
                            "(rotating-frame LSWT, D34) are not validated; they are not "
                            "computed rather than reported unchecked")
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
                      min_link_overlap: float = TOPOLOGY_MIN_LINK_OVERLAP,
                      plaquette_phase_margin: float = TOPOLOGY_PLAQUETTE_PHASE_MARGIN) -> np.ndarray:
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

    The integer is independent of the branch of each plaquette phase only
    when every plaquette phase satisfies ``|F| < pi`` (FHS admissibility). A plaquette phase within ``plaquette_phase_margin`` of
    ``+-pi`` gives NaN with a warning: the plaquette encloses a band touching
    (e.g. a Dirac point, Berry phase pi) or the mesh does not resolve the
    curvature, and the branch, hence the integer, would be set by round-off.

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
        total, weakest, largest = 0.0, np.inf, 0.0
        for i in range(n1):
            for j in range(n2):
                u00 = vector(i, j, band)
                u10 = vector(i + 1, j, band)
                u11 = vector(i + 1, j + 1, band)
                u01 = vector(i, j + 1, band)
                links = (np.vdot(u00, eta * u10), np.vdot(u10, eta * u11),
                         np.vdot(u11, eta * u01), np.vdot(u01, eta * u00))
                weakest = min(weakest, min(abs(x) for x in links))
                phase = np.angle(np.prod(links))
                largest = max(largest, abs(phase))
                total += phase
        if weakest <= min_link_overlap:
            warnings.warn(f"band {band}: smallest link overlap {weakest:.3g} <= {min_link_overlap}; "
                          "the band crosses another band between mesh points or the mesh does not "
                          "resolve it, so its Chern number is undefined here (NaN)",
                          UserWarning, stacklevel=2)
            continue
        if largest >= np.pi - plaquette_phase_margin:
            warnings.warn(f"band {band}: a plaquette phase is {largest:.12g}, within "
                          f"{plaquette_phase_margin} of pi (FHS admissibility); the plaquette "
                          "encloses a band touching or the mesh does not resolve the curvature, "
                          "so its Chern number is undefined here (NaN)", UserWarning, stacklevel=2)
            continue
        chern[band] = -orientation * total / (2 * np.pi)
    return chern


def chern_numbers(result, band_gap_cutoff: float = TOPOLOGY_BAND_GAP_CUTOFF,
                  tolerance: float = TOPOLOGY_CHERN_AGREEMENT,
                  curvature: Optional[BerryCurvature] = None) -> np.ndarray:
    """Chern numbers accepted when the Kubo integral and the FHS integer agree.

    The two calculations fail in opposite ways (D31). With a gap, the FHS sum
    of plaquette phases gives the correct integer already on coarse meshes,
    but it always returns an integer: where the gap closes (e.g. Dirac points)
    it returns an arbitrary one. The Kubo integral samples the curvature, so
    it converges slowly where the curvature is concentrated near a small gap,
    and it departs from an integer where the gap closes or the mesh is too
    coarse. Agreement accepts the integer; otherwise the result is NaN, never
    a wrong integer in the tested models (a rejection means: refine the mesh,
    or the gap closes). Both use the same mesh, so this is not a proof; a
    second mesh (N and 2N) gives the same result when the answer is settled.

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


# ---------------------------------------------------------------------------
# Thermal Hall conductivity (stage 5b)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThermalHall:
    """Magnon thermal Hall conductivity on a temperature grid (D29).

    ``kappa_xy^2D / T`` per layer in units of ``k_B^2 / hbar`` (dimensionless;
    independent of the energy and length units). In SI:
    ``kappa^2D [W/K] = kappa_over_t * k_B^2 T / hbar`` with
    ``T = t E0 / k_B``; divide by the layer spacing for a 3D conductivity.

    Attributes
    ----------
    temperatures : (nt,) array
        ``t = k_B T / E0``.
    kappa_over_t : (nt,) array
        Pair form (see :func:`thermal_hall`); defined for degenerate and
        crossing bands.
    kappa_over_t_band_sum : (nt,) array
        ``sum_n c2 Omega_n`` with the per-band curvature of
        :func:`berry_curvature`; NaN if some band is not separated at some k.
        Equal to the pair form when all bands are separated.
    gapless : bool
    decision : str
        "scan" or "user" (as in :class:`~spintoolkit.observables.thermal.ThermalResult`).
    zero_modes : dict
    band_gap_cutoff : float
    cell_area : float
    integration : dict or None
        Adaptive integration (5d): estimated error per temperature, points
        used, convergence, cells stopped at the depth limit. None for the
        stored mesh.
    """

    temperatures: np.ndarray
    kappa_over_t: np.ndarray
    kappa_over_t_band_sum: np.ndarray
    gapless: bool
    decision: str
    zero_modes: dict
    band_gap_cutoff: float
    cell_area: float
    integration: Optional[dict] = None


@dataclass(frozen=True)
class AdaptiveIntegration:
    """Settings of the adaptive k integration of :func:`thermal_hall` (5d).

    The magnetic Brillouin zone, in reciprocal-basis fractions ``[0, 1)^2``,
    starts as the cells of the result's mesh. Each cell carries the midpoint
    values of its four quarters; their difference from its own midpoint value
    is its error estimate, and the Richardson combination of the two
    (midpoint error ~ h^2) is its value. The cells holding half of the total error
    are split (Doerfler marking) until the total error is below
    ``max(absolute_tolerance, relative_tolerance * |kappa / T|)`` at every
    temperature, the evaluation budget is spent, or only cells at the depth
    limit remain. A cell at the depth limit is not split further (e.g. at a
    zero mode) and its error stays in the estimate.

    Parameters
    ----------
    relative_tolerance, absolute_tolerance : float
        On ``kappa / T`` in units of ``k_B^2 / hbar``.
    max_points : int
        Evaluation budget (k points).
    max_depth : int
        Halvings of an initial cell.
    """

    relative_tolerance: float = TOPOLOGY_ADAPTIVE_RELATIVE
    absolute_tolerance: float = TOPOLOGY_ADAPTIVE_ABSOLUTE
    max_points: int = TOPOLOGY_ADAPTIVE_MAX_POINTS
    max_depth: int = TOPOLOGY_ADAPTIVE_MAX_DEPTH

    def __post_init__(self):
        if not (self.relative_tolerance >= 0 and self.absolute_tolerance >= 0
                and self.relative_tolerance + self.absolute_tolerance > 0):
            raise ValueError("tolerances must be non-negative and not both zero")
        if self.max_points < 1 or self.max_depth < 0:
            raise ValueError("max_points must be positive and max_depth non-negative")


def thermal_hall(result, temperatures, zero_modes=None, gapless: Optional[bool] = None,
                 band_gap_cutoff: float = TOPOLOGY_BAND_GAP_CUTOFF,
                 curvature: Optional[BerryCurvature] = None,
                 integration: Optional[AdaptiveIntegration] = None) -> ThermalHall:
    """Magnon thermal Hall conductivity ``kappa_xy / T`` (Matsumoto and Murakami).

    ``kappa^2D / T = -(k_B^2 / hbar) (1 / A_cell) < sum_n c2(rho_n) Omega_n >_k``
    over the particle bands. It is evaluated in pair form: the curvature
    terms between two particle bands enter with the weight difference
    ``(c2_n - c2_m) / (E_n - E_m)``, so exactly degenerate bands contribute
    zero, a degenerate group contributes ``c2 Tr F``, and nearly degenerate
    bands with large opposite curvatures cancel before the k sum. The response
    is defined for degenerate and crossing bands; only a particle-hole
    degeneracy (a zero mode on the mesh) leaves it undefined (NaN).

    Parameters
    ----------
    result : LSWTResult
        On a complete uniform mesh, without regularization shift.
    temperatures : sequence of float
        ``t = k_B T / E0 >= 0``.
    zero_modes, gapless
        As in :func:`~spintoolkit.observables.thermal.thermal_quantities` (D25):
        candidates stop the calculation until ``gapless`` is given; with zero
        modes the result is computed with a warning, and its convergence
        near the zero modes must be checked on finer meshes.
    band_gap_cutoff : float
    curvature : BerryCurvature, optional
        Reused for the band-sum comparison.
    integration : AdaptiveIntegration, optional
        Integrate adaptively (5d) instead of on the stored mesh; H(k) and
        dH/dk are evaluated at new momenta, and ``kappa_over_t_band_sum`` is
        NaN. Needed where the curvature concentrates near small gaps.

    Returns
    -------
    ThermalHall
    """
    from spintoolkit.observables.thermal import ZeroModeCandidateError
    from spintoolkit.observables.zero_modes import scan_zero_modes

    _check_result(result)
    if _mesh_indices(result) is None:
        raise TopologyError("the thermal Hall integral needs a complete uniform mesh of the "
                            "magnetic Brillouin zone")
    t = np.asarray(temperatures, dtype=float).ravel()
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise ValueError("temperatures must be finite and non-negative")
    report = zero_modes if zero_modes is not None else scan_zero_modes(result)
    if gapless is None:
        if report.has_candidates:
            raise ZeroModeCandidateError(report)
        gapless_flag, decision = report.has_zero, "scan"
    else:
        gapless_flag, decision = bool(gapless), "user"
    if gapless_flag:
        warnings.warn("gapless spectrum: kappa_xy includes the neighbourhood of the zero modes; "
                      "check its convergence on finer meshes", UserWarning, stacklevel=2)
    if integration is not None:
        area = float(abs(np.linalg.det(result.magnetic_lattice)))
        kappa, diagnostics = _adaptive_kappa(result, t, band_gap_cutoff, integration, area)
        return ThermalHall(t, kappa, np.full(len(t), np.nan), gapless_flag, decision,
                           report.to_dict(), float(band_gap_cutoff), area, diagnostics)
    curvature = curvature if curvature is not None else berry_curvature(result, band_gap_cutoff)
    area = curvature.cell_area
    ns = result.num_sites
    dx, dy = result.hamiltonian_derivatives_at(result.k_points)
    terms = [curvature_pair_terms(E, T, [Dx, Dy], ns, band_gap_cutoff=band_gap_cutoff)
             for E, T, Dx, Dy in zip(result.eigenvalues, result.eigenvectors, dx, dy)]
    kappa, band_sum = [], []
    for tk in t:
        values = np.array([weighted_curvature_sum(x, tk) for x in terms])
        kappa.append(-float(result.weights @ values) / area)
        band_sum.append(-float(result.weights @ np.sum(c2_weight(curvature.energies, tk)
                                                       * curvature.curvature, axis=1)) / area)
    return ThermalHall(t, np.array(kappa), np.array(band_sum), gapless_flag, decision,
                       report.to_dict(), float(band_gap_cutoff), area)


def _integrand(result, fractions, temperatures, band_gap_cutoff):
    """``sum_n c2 Omega_n`` (pair form) at fractional momenta, ``(m, nt)``; NaN where undefined.

    Also returns the smallest particle-band gap and particle energy met.
    """
    from spintoolkit.methods.lswt.diagonalization import Diagonalizer

    reciprocal = 2 * np.pi * np.linalg.inv(result.magnetic_lattice).T
    k = fractions @ reciprocal
    H = result.hamiltonian_at(k)
    dx, dy = result.hamiltonian_derivatives_at(k)
    ns = result.num_sites
    J = np.diag(np.r_[np.ones(ns), -np.ones(ns)])
    values = np.full((len(k), len(temperatures)), np.nan)
    smallest_gap, lowest = np.inf, np.inf
    for i in range(len(k)):
        try:
            E, T = Diagonalizer.Colpa(np.linalg.cholesky(H[i]), J)
        except np.linalg.LinAlgError:
            continue                                   # not positive definite: undefined
        terms = curvature_pair_terms(E, T, [dx[i], dy[i]], ns, band_gap_cutoff=band_gap_cutoff)
        values[i] = [weighted_curvature_sum(terms, tk) for tk in temperatures]
        particle = np.sort(E[:ns])
        lowest = min(lowest, particle[0])
        if ns > 1:
            smallest_gap = min(smallest_gap, float(np.min(np.diff(particle))))
    return values, smallest_gap, lowest


_QUARTERS = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]]) / 4.0


def _adaptive_kappa(result, temperatures, band_gap_cutoff, settings, area):
    """Adaptive midpoint cubature of ``<sum_n c2 Omega_n>`` over the zone (see AdaptiveIntegration)."""
    mesh = _mesh_indices(result)
    if mesh is None:
        raise TopologyError("adaptive integration starts from a complete uniform mesh")
    _, n1, n2 = mesh
    f = np.mod(np.asarray(result.fractional, dtype=float), 1.0)
    size = np.array([1.0 / n1, 1.0 / n2])
    nt = len(temperatures)
    stats = {"points": 0, "smallest_gap": np.inf, "lowest_energy": np.inf, "undefined": 0}

    def evaluate(points):
        values, gap, low = _integrand(result, points, temperatures, band_gap_cutoff)
        stats["points"] += len(points)
        stats["smallest_gap"] = min(stats["smallest_gap"], gap)
        stats["lowest_energy"] = min(stats["lowest_energy"], low)
        stats["undefined"] += int(np.sum(np.isnan(values[:, 0]))) if nt else 0
        return values

    def children_of(centres, sizes, own):
        """Quarter midpoints of each cell: values (m, 4, nt); estimate and error (m, nt)."""
        points = (centres[:, None, :] + _QUARTERS[None] * sizes[:, None, :]).reshape(-1, 2)
        quarters = evaluate(points).reshape(len(centres), 4, nt)
        weight = np.prod(sizes, axis=1)[:, None]
        fine = weight * quarters.mean(axis=1)
        coarse = weight * own
        # Richardson: the midpoint rule errs as h^2, so (4 fine - coarse) / 3 removes it.
        return fine + (fine - coarse) / 3, np.abs(fine - coarse)

    centres = f.copy()
    sizes = np.tile(size, (len(centres), 1))
    depth = np.zeros(len(centres), dtype=int)
    own = evaluate(centres)
    estimate, error = children_of(centres, sizes, own)
    converged, reason = False, ""
    while True:
        if np.any(np.isnan(estimate)):
            reason = "undefined integrand (a zero mode or an unstable point)"
            break
        total = estimate.sum(axis=0)
        tolerance = np.maximum(settings.absolute_tolerance,
                               settings.relative_tolerance * np.abs(total) / area) * area
        remaining = error.sum(axis=0)
        if np.all(remaining <= tolerance):
            converged, reason = True, "tolerance reached"
            break
        refinable = depth < settings.max_depth
        if not np.any(refinable):
            reason = "only cells at the depth limit remain"
            break
        score = np.max(error / tolerance, axis=1) * refinable
        order = np.argsort(score)[::-1]
        cumulative = np.cumsum(score[order])
        marked = order[:int(np.searchsorted(cumulative, 0.5 * cumulative[-1])) + 1]
        marked = marked[score[marked] > 0]
        cost = 20 * len(marked)                  # 4 new midpoints and 16 quarter points each
        if stats["points"] + cost > settings.max_points:
            marked = marked[:max(0, (settings.max_points - stats["points"]) // 20)]
            if len(marked) == 0:
                reason = "evaluation budget spent"
                break
        new_sizes = np.repeat(sizes[marked] / 2, 4, axis=0)
        new_centres = (centres[marked][:, None, :] + _QUARTERS[None] * sizes[marked][:, None, :]
                       ).reshape(-1, 2)
        new_own = evaluate(new_centres)
        new_estimate, new_error = children_of(new_centres, new_sizes, new_own)
        keep = np.ones(len(centres), dtype=bool)
        keep[marked] = False
        centres = np.vstack([centres[keep], new_centres])
        sizes = np.vstack([sizes[keep], new_sizes])
        depth = np.concatenate([depth[keep], np.repeat(depth[marked] + 1, 4)])
        estimate = np.vstack([estimate[keep], new_estimate])
        error = np.vstack([error[keep], new_error])
    total = estimate.sum(axis=0)
    kappa = -total / area
    at_limit = depth >= settings.max_depth
    diagnostics = {
        "method": "adaptive midpoint cubature, Doerfler marking",
        "settings": {"relative_tolerance": settings.relative_tolerance,
                     "absolute_tolerance": settings.absolute_tolerance,
                     "max_points": settings.max_points, "max_depth": settings.max_depth},
        "converged": converged, "stop_reason": reason, "points": stats["points"],
        "cells": len(centres), "max_depth_reached": int(depth.max(initial=0)),
        "error_estimate": (error.sum(axis=0) / area).tolist(),
        "error_at_depth_limit": (error[at_limit].sum(axis=0) / area).tolist(),
        "undefined_points": stats["undefined"],
        "largest_error_cells": [
            {"fraction": centres[i].tolist(), "depth": int(depth[i]),
             "error": (error[i] / area).tolist()}
            for i in np.argsort(np.max(error, axis=1))[::-1][:5]],
        "smallest_particle_gap": stats["smallest_gap"], "lowest_particle_energy": stats["lowest_energy"],
        "initial_mesh": [n1, n2]}
    if not converged:
        warnings.warn(f"adaptive thermal Hall integration not converged ({reason}); see "
                      "ThermalHall.integration", UserWarning, stacklevel=3)
    return kappa, diagnostics
