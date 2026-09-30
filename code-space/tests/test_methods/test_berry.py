"""Berry curvature, Chern numbers and thermal Hall conductivity of LSWT magnons (stage 5, D29).

1. Sign anchor: for the honeycomb ferromagnet with DM (Haldane magnons) the
   particle block of H(k) equals the Bloch matrix rebuilt from the ED
   one-magnon states on a torus, not its complex conjugate (which has the
   opposite Chern numbers); the ED Bloch states alone give the same Chern
   numbers.
2. Kubo curvature (dH/dk and the stored eigenvectors) equals the phase of a
   small plaquette of eigenvectors, pointwise, with and without pairing.
3. Chern numbers: Haldane +-1 (flips with D), Kitaev [111] polarized +-1
   (pairing terms), sum over particle bands zero, cell gauge = full-position
   gauge, Kubo converges to the FHS integers.
4. Null cases: D = 0, triangular-lattice Heisenberg; band crossings and
   degeneracies give NaN instead of a number, and a gap that closes between
   mesh points (where FHS alone returns a wrong integer) is rejected by the
   Kubo-FHS agreement.
5. Thermal Hall (5b): the Haldane value from an independent two-band
   curvature and a quadrature c2; pair form = band sum for separated bands;
   limits t -> 0 and t -> infinity; time reversal; zero for coplanar
   Heisenberg states, also with degenerate bands (Neel, 120 degrees) where
   the band sum is undefined; continuity as a gap closes; the existing SI
   routine on the same k data; zero-mode rule D25.
6. Adaptive k integration (5d): a narrow-gap Haldane case against a fine
   uniform two-band reference, error estimates bounding the actual error,
   stop reasons (budget, depth limit), coplanar zero.
"""

import warnings

import numpy as np
import pytest

from spintoolkit.methods.classical import refine_classical
from spintoolkit.methods.ed import EDSector, solve_ed
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.methods.lswt.diagonalization import Diagonalizer
from spintoolkit.models import neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.models.honeycomb import honeycomb_ferromagnet, kitaev_honeycomb
from spintoolkit.observables.berry import (
    AdaptiveIntegration, TopologyError, berry_curvature, c2_weight, c2_weight_derivative,
    chern_numbers, chern_numbers_fhs, thermal_hall, zone_gauge)
from spintoolkit.system.cluster import allowed_momenta, expand_on_torus
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry

N111 = np.ones(3) / np.sqrt(3)


def haldane(D=0.2, mesh=(12, 12)):
    model = honeycomb_ferromagnet(J=1.0, D=D)
    conditions = ExternalConditions(field=(0, 0, 0.3))
    return model, conditions, solve_lswt(model, polarized_state(model), conditions,
                                         settings=LSWTSettings(mesh=mesh))


def kitaev(h=1.0, mesh=(12, 12), sign=1):
    model = kitaev_honeycomb(K=-1.0)
    conditions = ExternalConditions(field=sign * h * N111)
    return solve_lswt(model, polarized_state(model, sign * N111), conditions,
                      settings=LSWTSettings(mesh=mesh))


def colpa(H):
    ns = len(H) // 2
    return Diagonalizer.Colpa(np.linalg.cholesky(H), np.diag(np.r_[np.ones(ns), -np.ones(ns)]))


def lowest_particle(H):
    E, T = colpa(H)
    ns = len(H) // 2
    return T[:, int(np.argmin(E[:ns]))]


def ed_bloch_matrices(model, conditions, L):
    """One-magnon Bloch matrices h(k)_ij = sum_s <s0(i)|H|s> exp(i k.(r_s - r_s0)) from ED."""
    geometry = CalculationGeometry.finite_torus([[L, 0], [0, L]])
    cluster = expand_on_torus(model, geometry)
    ed = solve_ed(model, geometry, conditions, EDSector(axis=(0, 0, 1), magnon_number=1),
                  return_vectors=True)
    block = ed.block(magnon_number=1)
    real_space = block.vectors @ np.diag(block.energies - ed.reference_energy) @ block.vectors.conj().T
    sublattice = np.array([model.site_ids.index(key[0]) for key in cluster.keys])
    first = [int(np.flatnonzero(sublattice == a)[0]) for a in range(model.num_sites)]
    _, k = allowed_momenta(model, geometry)
    h = np.zeros((len(k), model.num_sites, model.num_sites), dtype=complex)
    for n, kn in enumerate(k):
        for i, s0 in enumerate(first):
            phases = np.exp(1j * (cluster.positions - cluster.positions[s0]) @ kn)
            np.add.at(h[n, i], sublattice, real_space[s0] * phases)
    return k, h


def test_derivatives_match_finite_differences():
    for result in (haldane()[2], kitaev()):
        k = np.array([[0.31, -0.57], [1.2, 0.4]])
        dx, dy = result.hamiltonian_derivatives_at(k)
        step = 1e-6
        for derivative, e in ((dx, [step, 0]), (dy, [0, step])):
            numeric = (result.hamiltonian_at(k + e) - result.hamiltonian_at(k - e)) / (2 * step)
            assert np.max(np.abs(numeric - derivative)) < 1e-8 * np.max(np.abs(derivative))


def test_haldane_bloch_matrix_equals_the_ed_one_magnon_matrix():
    model, conditions, _ = haldane()
    k, h = ed_bloch_matrices(model, conditions, 4)
    result = solve_lswt(model, polarized_state(model), conditions, settings=LSWTSettings(k_points=k))
    block = result.hamiltonians[:, :2, :2]
    assert np.max(np.abs(result.hamiltonians[:, :2, 2:])) < 1e-15
    assert np.max(np.abs(h - block)) < 1e-12
    assert np.max(np.abs(h.conj() - block)) > 0.1       # the conjugate has the opposite Chern numbers


def test_ed_bloch_states_alone_give_the_haldane_chern_numbers():
    """FHS on the ED Bloch matrices in the cell gauge (periodic in k) of a 5 x 5 torus."""
    model, conditions, result = haldane(D=0.2)
    L = 5
    k, h = ed_bloch_matrices(model, conditions, L)
    positions = np.array([model.cartesian_position(s) for s in model.site_ids])
    reciprocal = 2 * np.pi * np.linalg.inv(model.lattice).T
    grid = {}
    for n, kn in enumerate(k):
        q = np.rint(np.linalg.solve(reciprocal.T, kn) * L).astype(int) % L
        cell_gauge = np.diag(np.exp(-1j * positions @ kn))
        grid[tuple(q)] = np.linalg.eigh(cell_gauge.conj().T @ h[n] @ cell_gauge)[1]
    chern = []
    for band in range(2):
        total = 0.0
        for i in range(L):
            for j in range(L):
                u = [grid[(i % L, j % L)], grid[((i + 1) % L, j % L)],
                     grid[((i + 1) % L, (j + 1) % L)], grid[(i % L, (j + 1) % L)]]
                u = [x[:, band] for x in u]
                total += np.angle(np.vdot(u[0], u[1]) * np.vdot(u[1], u[2])
                                  * np.vdot(u[2], u[3]) * np.vdot(u[3], u[0]))
        chern.append(-np.sign(np.linalg.det(reciprocal)) * total / (2 * np.pi))
    assert np.allclose(chern, [1, -1], atol=1e-10)
    assert np.allclose(chern_numbers_fhs(result), chern, atol=1e-10)


@pytest.mark.parametrize("which", ["haldane", "kitaev"])
def test_kubo_curvature_equals_the_phase_of_a_small_plaquette(which):
    result = haldane()[2] if which == "haldane" else kitaev()
    curvature = berry_curvature(result)
    ns = result.num_sites
    delta = 1e-4
    for n in (3, 40, 97):
        k = result.k_points[n]
        corners = k + delta * np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        u = [lowest_particle(H) for H in result.hamiltonian_at(corners)]
        eta = np.r_[np.ones(ns), -np.ones(ns)]
        phase = np.angle(np.prod([np.vdot(u[i], eta * u[(i + 1) % 4]) for i in range(4)]))
        plaquette = -phase / delta ** 2
        kubo = curvature.curvature[n, 0]
        assert abs(plaquette - kubo) < 1e-3 * max(1.0, abs(kubo))


def test_haldane_chern_numbers_flip_with_d_and_vanish_without_it():
    for D, expected in ((0.2, [1, -1]), (-0.2, [-1, 1])):
        result = haldane(D)[2]
        assert np.allclose(berry_curvature(result).chern_numbers(), expected, atol=1e-6)
        assert np.allclose(chern_numbers_fhs(result), expected, atol=1e-10)
    flat = berry_curvature(haldane(0.0)[2])
    assert np.max(np.abs(flat.curvature)) < 1e-12


def test_gap_closing_between_mesh_points_is_caught_by_kubo_and_fhs_disagreeing():
    """D = 0: Dirac points between the 12 x 12 mesh points. FHS still returns an integer (-1),
    the Kubo sum is 0; chern_numbers rejects the pair, and a finer mesh gives 0 in both."""
    coarse = haldane(0.0, (12, 12))[2]
    assert np.allclose(chern_numbers_fhs(coarse), [-1, 1], atol=1e-10)
    assert np.allclose(berry_curvature(coarse).chern_numbers(), 0, atol=1e-10)
    with pytest.warns(UserWarning, match="disagree"):
        assert np.all(np.isnan(chern_numbers(coarse)))
    assert np.allclose(chern_numbers(haldane(0.0, (24, 24))[2]), 0)
    assert np.array_equal(chern_numbers(haldane(0.2, (24, 24))[2]), [1, -1])


def test_kitaev_polarized_chern_numbers_with_pairing():
    coarse, fine = kitaev(mesh=(24, 24)), kitaev(mesh=(48, 48))
    assert np.max(np.abs(coarse.hamiltonians[:, :2, 2:])) > 0.1          # anomalous terms
    assert np.allclose(chern_numbers_fhs(coarse), [1, -1], atol=1e-10)
    kubo_coarse = berry_curvature(coarse).chern_numbers()
    kubo_fine = berry_curvature(fine).chern_numbers()
    assert np.allclose(kubo_fine, [1, -1], atol=1e-5)
    assert np.all(np.abs(kubo_fine - [1, -1]) < np.abs(kubo_coarse - [1, -1]))
    assert abs(np.sum(kubo_fine)) < 1e-12                              # sum over particle bands
    with pytest.warns(UserWarning, match="disagree"):                  # h = 2 needs a finer mesh
        assert np.all(np.isnan(chern_numbers(kitaev(h=2.0, mesh=(12, 12)))))
    assert np.array_equal(chern_numbers(kitaev(h=2.0, mesh=(24, 24))), [1, -1])
    reversed_field = chern_numbers_fhs(kitaev(mesh=(24, 24), sign=-1))
    assert np.allclose(reversed_field, [-1, 1], atol=1e-10)            # time reversal


def test_cell_gauge_gives_the_same_chern_numbers():
    """H_c(k) = D(k)^+ H(k) D(k) is periodic; FHS without boundary factors agrees."""
    result = kitaev(mesh=(12, 12))
    sign, reciprocal = zone_gauge(result)
    n, ns = 12, result.num_sites
    eta = np.r_[np.ones(ns), -np.ones(ns)]
    vectors = {}
    for i in range(n):
        for j in range(n):
            k = (i * reciprocal[0] + j * reciprocal[1]) / n
            phase = np.exp(1j * sign * result.positions @ k)
            D = np.diag(np.r_[phase, phase])
            vectors[i, j] = lowest_particle(D.conj().T @ result.hamiltonian_at(k)[0] @ D)
    total = 0.0
    for i in range(n):
        for j in range(n):
            u = [vectors[i, j], vectors[(i + 1) % n, j], vectors[(i + 1) % n, (j + 1) % n],
                 vectors[i, (j + 1) % n]]
            total += np.angle(np.prod([np.vdot(u[a], eta * u[(a + 1) % 4]) for a in range(4)]))
    cell = -np.sign(np.linalg.det(reciprocal)) * total / (2 * np.pi)
    assert abs(cell - chern_numbers_fhs(result)[0]) < 1e-10


def test_triangular_heisenberg_has_zero_chern_numbers():
    model = triangular_heisenberg()
    conditions = ExternalConditions(field=(0, 0, 1.0))
    state = refine_classical(model, state_120(model, ((1, 0, 0), (0, 0, 1))), conditions)
    result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(12, 12)))
    assert np.allclose(berry_curvature(result).chern_numbers(), 0, atol=1e-10)
    assert np.allclose(chern_numbers_fhs(result), 0, atol=1e-10)


def test_band_crossing_between_mesh_points_is_undefined_not_a_number():
    """Canted square antiferromagnet: the folded bands cross on lines between mesh points."""
    model = square_heisenberg()
    conditions = ExternalConditions(field=(0, 0, 2.0))
    state = refine_classical(model, neel_state(model, (1, 0, 0)), conditions)
    result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(8, 8)))
    assert np.max(np.abs(berry_curvature(result).curvature)) < 1e-12
    with pytest.warns(UserWarning, match="link overlap"):
        assert np.all(np.isnan(chern_numbers_fhs(result)))


def test_degenerate_bands_have_no_curvature():
    model = square_heisenberg()
    result = solve_lswt(model, neel_state(model), None, settings=LSWTSettings(mesh=(6, 6)))
    curvature = berry_curvature(result)
    assert np.all(np.isnan(curvature.curvature)) and np.all(np.isnan(curvature.chern_numbers()))
    assert np.all(np.isnan(chern_numbers_fhs(result)))


def test_inputs_that_cannot_give_a_chern_number_are_rejected():
    model, conditions, _ = haldane()
    explicit = solve_lswt(model, polarized_state(model), conditions,
                          settings=LSWTSettings(k_points=[[0.1, 0.2], [0.3, 0.1]]))
    with pytest.raises(TopologyError, match="complete uniform mesh"):
        berry_curvature(explicit).chern_numbers()
    with pytest.raises(TopologyError, match="complete uniform mesh"):
        chern_numbers_fhs(explicit)
    square = square_heisenberg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regularized = solve_lswt(square, neel_state(square), None, settings=LSWTSettings(
            mesh=(4, 4), shift=False, regularization="MAGSWT"))
    assert np.any(regularized.regularization_shift != 0)
    with pytest.raises(TopologyError, match="regularization"):
        berry_curvature(regularized)


# ---------------------------------------------------------------------------
# Thermal Hall (5b)
# ---------------------------------------------------------------------------

def c2_quadrature(energy, t):
    """c2 from its definition int_x^inf z^2 e^-z / (1 - e^-z)^2 dz, x = E / t."""
    from scipy.integrate import quad
    return quad(lambda z: z * z * np.exp(-z) / (-np.expm1(-z)) ** 2, energy / t, np.inf,
                epsabs=1e-13, epsrel=1e-12)[0]


def test_c2_weight_and_its_derivative():
    energies = np.array([1e-3, 0.05, 0.3, 1.0, 4.0])
    for t in (0.1, 1.0):
        assert np.allclose(c2_weight(energies, t), [c2_quadrature(e, t) for e in energies],
                           rtol=1e-9, atol=1e-14)
        step = 1e-6
        numeric = (c2_weight(energies + step, t) - c2_weight(energies - step, t)) / (2 * step)
        assert np.allclose(c2_weight_derivative(energies, t), numeric, rtol=1e-6)
    assert np.all(c2_weight(energies, 0.0) == 0) and c2_weight(np.array([1e4]), 1.0)[0] == 0


def test_haldane_thermal_hall_from_an_independent_two_band_calculation():
    """Omega_-/+ = +/-(1/2) d.(d_x d x d_y d) by finite differences of d(k); c2 by quadrature."""
    result = haldane(0.2, (12, 12))[2]
    pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]

    def d_vector(k):
        h = result.hamiltonian_at(k)[0][:2, :2]
        return np.real(np.trace(h)) / 2, np.array([np.real(np.trace(h @ p)) / 2 for p in pauli])

    step, ts = 1e-5, [0.1, 0.4, 2.0]
    reference = np.zeros(len(ts))
    for k in result.k_points:
        d0, d = d_vector(k)
        unit = lambda q: d_vector(q)[1] / np.linalg.norm(d_vector(q)[1])
        dx = (unit(k + [step, 0]) - unit(k - [step, 0])) / (2 * step)
        dy = (unit(k + [0, step]) - unit(k - [0, step])) / (2 * step)
        solid = unit(k) @ np.cross(dx, dy) / 2
        lower, upper = d0 - np.linalg.norm(d), d0 + np.linalg.norm(d)
        for i, t in enumerate(ts):
            reference[i] += c2_quadrature(lower, t) * solid - c2_quadrature(upper, t) * solid
    area = abs(np.linalg.det(result.magnetic_lattice))
    reference = -reference / len(result.k_points) / area
    kappa = thermal_hall(result, ts).kappa_over_t
    assert np.allclose(kappa, reference, rtol=1e-6)


def test_pair_form_equals_band_sum_and_limits():
    for result in (haldane(0.2, (24, 24))[2], kitaev(mesh=(24, 24))):
        ts = [0.0, 0.02, 0.05, 1.0, 1e3]
        hall = thermal_hall(result, ts)
        assert np.allclose(hall.kappa_over_t, hall.kappa_over_t_band_sum, rtol=1e-10, atol=1e-16)
        k = hall.kappa_over_t
        assert k[0] == 0 and abs(k[1]) < 1e-3 * abs(k[3])      # gapped: exponentially small
        assert abs(k[4]) < 1e-2 * abs(k[3])                       # sum of Chern numbers is zero
    forward, backward = (thermal_hall(kitaev(mesh=(24, 24), sign=s), [0.3]).kappa_over_t
                         for s in (1, -1))
    assert np.allclose(forward, -backward, rtol=1e-10)             # time reversal


def test_coplanar_heisenberg_states_have_zero_thermal_hall_even_with_degenerate_bands():
    """T x C2 about the spin-plane normal: Omega(-k) = -Omega(k). Neel and 120 degrees have
    degenerate or touching bands, so the band sum is undefined but the response is zero."""
    square, triangle = square_heisenberg(), triangular_heisenberg()
    with pytest.warns(UserWarning, match="gapless"):
        neel = thermal_hall(solve_lswt(square, neel_state(square), None,
                                       settings=LSWTSettings(mesh=(12, 12))), [0.1, 1.0])
    assert np.all(np.isnan(neel.kappa_over_t_band_sum)) and neel.gapless
    assert np.max(np.abs(neel.kappa_over_t)) < 1e-14
    with pytest.warns(UserWarning, match="gapless"):
        triangular = thermal_hall(solve_lswt(triangle, state_120(triangle), None,
                                             settings=LSWTSettings(mesh=(12, 12))), [0.1, 1.0])
    assert np.max(np.abs(triangular.kappa_over_t)) < 1e-14
    conditions = ExternalConditions(field=(0, 0, 1.0))
    state = refine_classical(triangle, state_120(triangle, ((1, 0, 0), (0, 0, 1))), conditions)
    canted = solve_lswt(triangle, state, conditions, settings=LSWTSettings(mesh=(12, 12)))
    with pytest.warns(UserWarning, match="gapless"):
        assert np.max(np.abs(thermal_hall(canted, [0.1, 1.0]).kappa_over_t)) < 1e-14


def test_thermal_hall_is_continuous_where_the_chern_numbers_jump():
    """Haldane D -> 0: C jumps from +-1 to undefined, kappa / T goes to zero linearly in D."""
    slope = [thermal_hall(haldane(D, (24, 24))[2], [0.3]).kappa_over_t[0] / D
             for D in (1e-3, 1e-2)]
    assert abs(slope[0] - slope[1]) < 0.1 * abs(slope[0])
    assert abs(thermal_hall(haldane(0.0, (24, 24))[2], [0.3]).kappa_over_t[0]) < 1e-15


def test_existing_si_routine_gives_the_same_kappa_on_the_same_k_data():
    """observables.topology.Topology (W/K per layer, meV and kelvin) on the stored diagonalization."""
    from types import SimpleNamespace

    from spintoolkit.definitions.constants import K_BOLTZMANN_MEV
    from spintoolkit.definitions import H_BAR_MEV
    from spintoolkit.observables.topology import Topology
    from tests.test_methods.test_thermal import nbcp_y

    model, state, conditions = nbcp_y({"JPD": 0.01})
    state = refine_classical(model, state, conditions)
    result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(12, 12)))
    dx, dy = result.hamiltonian_derivatives_at(result.k_points)
    k_data = {i: [[H, dx[i], dy[i]], [E, T], [True, None]]
              for i, (H, E, T) in enumerate(zip(result.hamiltonians, result.eigenvalues,
                                                result.eigenvectors))}
    parent = SimpleNamespace(Ns=result.num_sites, bz_data={"area": None},
                             system=SimpleNamespace(lattice_vectors=result.magnetic_lattice))
    kelvin = 0.5
    t = K_BOLTZMANN_MEV * kelvin                               # E0 = meV
    legacy = Topology(parent).compute_thermal_Hall(k_data, kelvin)[2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = thermal_hall(result, [t], gapless=True).kappa_over_t[0]
    si = ours * K_BOLTZMANN_MEV ** 2 * kelvin / H_BAR_MEV * 1.602176634e-22
    assert abs(si - legacy) < 1e-10 * abs(legacy)


def test_zero_mode_candidates_stop_the_thermal_hall_calculation():
    from spintoolkit.observables.thermal import ZeroModeCandidateError
    from tests.test_methods.test_thermal import square_model

    model = square_model(anisotropy=1e-7)
    result = solve_lswt(model, neel_state(model), None, settings=LSWTSettings(mesh=(8, 8)))
    with pytest.raises(ZeroModeCandidateError):
        thermal_hall(result, [0.1])
    assert thermal_hall(result, [0.1], gapless=False).decision == "user"


# ---------------------------------------------------------------------------
# Adaptive k integration (5d)
# ---------------------------------------------------------------------------

PAULI = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]


def haldane_reference(result, n, ts):
    """Uniform n x n midpoint mesh; lower-band curvature d.(d_x d x d_y d) / (2 |d|^3)."""
    reciprocal = 2 * np.pi * np.linalg.inv(result.magnetic_lattice).T
    grid = (np.arange(n) + 0.5) / n
    k = np.array(np.meshgrid(grid, grid, indexing="ij")).reshape(2, -1).T @ reciprocal
    h = result.hamiltonian_at(k)[:, :2, :2]
    dx, dy = result.hamiltonian_derivatives_at(k)

    def parts(m):
        return (np.real(np.trace(m, axis1=1, axis2=2)) / 2,
                np.stack([np.real(np.einsum("kij,ji->k", m, p)) / 2 for p in PAULI], axis=1))

    d0, d = parts(h)
    ddx, ddy = parts(dx[:, :2, :2])[1], parts(dy[:, :2, :2])[1]
    norm = np.linalg.norm(d, axis=1)
    lower = np.einsum("ki,ki->k", d, np.cross(ddx, ddy)) / (2 * norm ** 3)
    area = abs(np.linalg.det(result.magnetic_lattice))
    return np.array([-np.mean((c2_weight(d0 - norm, t) - c2_weight(d0 + norm, t)) * lower) / area
                     for t in ts])


def test_adaptive_integration_resolves_a_narrow_gap():
    """Haldane D = 0.01: the 12 x 12 mesh misses the curvature at K by 23 percent at t = 0.3."""
    result = haldane(0.01, (12, 12))[2]
    ts = [0.05, 0.3]
    reference = haldane_reference(result, 512, ts)
    coarse = thermal_hall(result, ts).kappa_over_t
    assert abs(coarse[1] / reference[1] - 1) > 0.2
    settings = AdaptiveIntegration(relative_tolerance=1e-2, max_points=40_000)
    hall = thermal_hall(result, ts, integration=settings)
    info = hall.integration
    assert info["converged"] and info["points"] <= 40_000
    actual = np.abs(hall.kappa_over_t - reference)
    assert np.all(actual <= np.array(info["error_estimate"]))
    assert np.all(actual <= 1e-2 * np.abs(reference))
    assert np.all(np.isnan(hall.kappa_over_t_band_sum))


def test_adaptive_integration_reports_why_it_stopped():
    result = haldane(0.01, (12, 12))[2]
    with pytest.warns(UserWarning, match="budget"):
        budget = thermal_hall(result, [0.3], integration=AdaptiveIntegration(
            relative_tolerance=1e-6, max_points=2_000))
    assert not budget.integration["converged"] and budget.integration["points"] <= 2_000
    with pytest.warns(UserWarning, match="depth limit"):
        shallow = thermal_hall(result, [0.3], integration=AdaptiveIntegration(
            relative_tolerance=1e-6, max_depth=0))
    assert shallow.integration["error_at_depth_limit"][0] > 0
    with pytest.raises(ValueError):
        AdaptiveIntegration(relative_tolerance=0, absolute_tolerance=0)


def test_adaptive_integration_keeps_the_coplanar_zero():
    triangle = triangular_heisenberg()
    result = solve_lswt(triangle, state_120(triangle), None, settings=LSWTSettings(mesh=(12, 12)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hall = thermal_hall(result, [0.1, 1.0], integration=AdaptiveIntegration(max_points=20_000))
    assert np.max(np.abs(hall.kappa_over_t)) < 1e-7
