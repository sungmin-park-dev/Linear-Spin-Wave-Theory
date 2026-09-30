"""Berry curvature and Chern numbers of LSWT magnons (stage 5a, D29).

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
    TopologyError, berry_curvature, chern_numbers, chern_numbers_fhs, zone_gauge)
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
