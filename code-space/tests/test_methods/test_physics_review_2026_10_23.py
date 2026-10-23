"""Regression tests for the physics errors found in the 2026-10-23 external review.

Each test reproduces a case where the package returned a wrong number without
a warning; the expected values come from closed forms or independent
calculations (supercell energies, decoupled layers, Bragg-point scaling).
"""

import warnings

import numpy as np
import pytest

import spintoolkit as stk
from spintoolkit.methods.classical import classical_energy
from spintoolkit.methods.lswt import spiral_energy
from spintoolkit.methods.phase_competition import compare_states
from spintoolkit.methods.nlswt.engine import triangle_delta_integral, triangle_density
from spintoolkit.observables.bands import _energies
from spintoolkit.observables.berry import chern_numbers
from spintoolkit.observables.two_magnon import two_magnon_continuum
from spintoolkit.states.incommensurate import IncommensurateStructure
from spintoolkit.system.exchange import xxz, xxz_with_soc
from spintoolkit.system.high_symmetry import high_symmetry_points, zone_boundary


def square(J=1.0, S=0.5, Jy=None, field=False):
    site = stk.Site("A", (0.0, 0.0), S)
    terms = [stk.Term.bilinear(("A", (0, 0)), ("A", (1, 0)), stk.heisenberg(J)),
             stk.Term.bilinear(("A", (0, 0)), ("A", (0, 1)), stk.heisenberg(J if Jy is None else Jy))]
    if field:
        terms.append(stk.Term.zeeman("A", np.eye(3)))
    return stk.SpinModel(np.eye(2), (site,), tuple(terms), {"model_id": "square"})


def neel(model):
    return stk.SpinState.from_function(
        model, [[1, 1], [1, -1]], lambda s, c: np.array([0, 0, 1.0]) * (-1) ** (c[0] + c[1]))


def test_exchange_helpers_keep_float_entries_for_integer_input():
    J = xxz_with_soc(1, 1, 0.2)
    assert J[0, 1] == J[1, 0] == pytest.approx(0.2)
    assert xxz(1, 2).dtype == float


@pytest.mark.parametrize("q", [(1 / 3, 0), (1 / 4, 1 / 3), (1 / 2, 0), (1 / 2, 1 / 2), (0, 0)])
def test_spiral_energy_is_the_lattice_average_without_u1_symmetry(q):
    J = np.diag([1.0, 0.3, 0.5])
    J[0, 1] = J[1, 0] = 0.2
    terms = [stk.Term.bilinear(("A", (0, 0)), ("A", (1, 0)), J),
             stk.Term.bilinear(("A", (0, 0)), ("A", (0, 1)), np.diag([0.4, -0.2, 0.1])),
             stk.Term.zeeman("A", np.eye(3)), stk.Term.onsite("A", np.diag([0.2, -0.1, 0.3]))]
    model = stk.SpinModel(np.eye(2), [stk.Site("A", (0, 0), 1.0)], terms, {"model_id": "aniso"})
    conditions = stk.ExternalConditions(field=(0.3, 0.1, 0.2))
    spiral = IncommensurateStructure(model.fingerprint(), list(q), [0, 0, 1], {"A": [0.6, 0, 0.8]})
    assert spiral_energy(model, spiral, conditions) == pytest.approx(
        classical_energy(model, spiral.to_spin_state(model), conditions), abs=1e-13)


def test_zero_mode_band_branch_keeps_nonreciprocal_particle_energies():
    # a zero mode and a band with w(k) = 2, w(-k) = 1
    energies, zero, unstable = _energies(np.diag([0.0, 2.0, 0.0, 1.0]), 1e-9)
    assert zero and not unstable
    np.testing.assert_allclose(energies, [0.0, 2.0])


def test_compare_states_does_not_rank_a_non_stationary_candidate():
    model = square(J=-1.0, field=True)
    conditions = stk.ExternalConditions(field=(0, 0, 0.5))
    tilt = stk.SpinState.from_function(model, np.eye(2, dtype=int),
                                       lambda s, c: np.array([np.sin(0.3), 0, np.cos(0.3)]))
    up = stk.SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: np.array([0, 0, 1.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reports = compare_states(model, {"tilt": tilt, "up": up}, conditions, refine=False)
    assert [r.name for r in reports] == ["up", "tilt"]
    assert reports[1].status == "not stationary" and np.isnan(reports[1].harmonic_energy)


def test_boson_number_above_spin_is_flagged():
    model = square(Jy=0.01)
    with pytest.warns(UserWarning, match="exceeds the spin"):
        result = stk.solve_lswt(model, neel(model), stk.ExternalConditions(),
                                settings=stk.LSWTSettings(mesh=(40, 40)))
    assert result.header.diagnostics["max_boson_number_per_spin"] > 1
    assert np.all(np.isnan(result.ordered_moments()))


def two_layers():
    """Uncoupled square AFMs with S = 1/2, J = 1 and S = 1, J = 1/2: equal JS, LSWT degenerate."""
    sites = (stk.Site("A", (0, 0), 0.5), stk.Site("B", (0.5, 0.5), 1.0))
    terms = []
    for d in [(1, 0), (0, 1)]:
        terms.append(stk.Term.bilinear(("A", (0, 0)), ("A", d), stk.heisenberg(1.0)))
        terms.append(stk.Term.bilinear(("B", (0, 0)), ("B", d), stk.heisenberg(0.5)))
    return stk.SpinModel(np.eye(2), sites, tuple(terms), {"model_id": "two_layers"})


def single_layer_energy(S, J, k):
    model = square(J=J, S=S)
    result = stk.solve_nlswt(model, neel(model), settings=stk.NLSWTSettings(mesh=(12, 12)))
    return result.magnon_energies(k)["total"][0, 0]


def test_degenerate_bands_use_degenerate_perturbation_theory():
    model = two_layers()
    k = np.array([[0.7, 0.2]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stk.solve_nlswt(model, neel(model), settings=stk.NLSWTSettings(mesh=(12, 12)))
        expected = np.sort([single_layer_energy(0.5, 1.0, k)] * 2
                           + [single_layer_energy(1.0, 0.5, k)] * 2)
        total = result.magnon_energies(k)["total"][0]
        spectral = result.spectral_function(k, np.linspace(0.1, 3, 5), eta=0.01)["energy"][0]
    np.testing.assert_allclose(np.sort(total), expected, atol=1e-10)
    np.testing.assert_allclose(np.sort(spectral), expected, atol=1e-6)


def test_longitudinal_continuum_diverges_at_the_neel_bragg_vector():
    model = square()
    result = stk.solve_lswt(model, neel(model), stk.ExternalConditions(),
                            settings=stk.LSWTSettings(mesh=(8, 8)))
    q = [[np.pi, np.pi], [0.0, 0.0], [0.7, 0.3]]
    with pytest.warns(UserWarning, match="diverge logarithmically"):
        static = two_magnon_continuum(result, q, mesh=(16, 16)).static()
    assert static[0, 2, 2] == np.inf
    assert static[1, 2, 2] == pytest.approx(0.0, abs=1e-12)      # total S^z is conserved
    assert np.isfinite(static[2, 2, 2]) and static[2, 2, 2] > 0


def test_flat_pair_energies_are_not_dropped():
    with pytest.warns(UserWarning, match="flat band"):
        rho = triangle_density(np.zeros((4, 4)), np.ones((4, 4)), [0.0, 1.0])
    assert np.all(np.isnan(rho))
    assert np.isnan(triangle_delta_integral(np.zeros((4, 4)), np.ones((4, 4))))
    assert triangle_delta_integral(np.ones((4, 4)), np.ones((4, 4))) == 0.0


def honeycomb_ferromagnet(J1, D):
    lattice = np.array([[1, 0], [0.5, np.sqrt(3) / 2]])
    sites = (stk.Site("A", (0, 0), 0.5), stk.Site("B", (1 / 3, 1 / 3), 0.5))
    terms = [stk.Term.bilinear(("A", (0, 0)), ("B", (0, 0)), stk.heisenberg(-J1)),
             stk.Term.bilinear(("A", (0, 0)), ("B", (-1, 0)), stk.heisenberg(-1.0)),
             stk.Term.bilinear(("A", (0, 0)), ("B", (0, -1)), stk.heisenberg(-1.0))]
    for d in [(1, 0), (-1, 1), (0, -1)]:
        terms.append(stk.Term.bilinear(("A", (0, 0)), ("A", d), stk.dzyaloshinskii_moriya([0, 0, D])))
        terms.append(stk.Term.bilinear(("B", (0, 0)), ("B", d), stk.dzyaloshinskii_moriya([0, 0, -D])))
    terms += [stk.Term.zeeman("A", np.eye(3)), stk.Term.zeeman("B", np.eye(3))]
    model = stk.SpinModel(lattice, sites, tuple(terms), {"model_id": "honeycomb"})
    state = stk.SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: np.array([0, 0, 1.0]))
    return model, state


@pytest.mark.parametrize("J1, D, mesh", [(2.0, 0.0, 8), (1.999, 0.001, 12)])
def test_chern_number_is_not_confirmed_across_an_unresolved_near_touching(J1, D, mesh):
    # J1 = 2: the two Dirac points merge (bands touch, C undefined); J1 = 1.999:
    # gap 2.5e-4 between mesh points, true C = (+1, -1), both mesh sums give 0.
    model, state = honeycomb_ferromagnet(J1, D)
    result = stk.solve_lswt(model, state, stk.ExternalConditions(field=(0, 0, 0.3)),
                            settings=stk.LSWTSettings(mesh=(mesh, mesh)))
    with pytest.warns(UserWarning, match="between mesh points"):
        chern = chern_numbers(result)
    assert np.all(np.isnan(chern))


def test_chern_number_of_a_resolved_gap_is_kept():
    model, state = honeycomb_ferromagnet(1.0, 0.1)
    result = stk.solve_lswt(model, state, stk.ExternalConditions(field=(0, 0, 0.3)),
                            settings=stk.LSWTSettings(mesh=(12, 12)))
    np.testing.assert_allclose(chern_numbers(result), [1.0, -1.0])


def test_brillouin_zone_of_a_skew_basis():
    corners = zone_boundary([[1, 0], [8, 1]])          # the square lattice
    np.testing.assert_allclose(np.sort(np.abs(corners), axis=0), np.full((4, 2), np.pi), atol=1e-12)
    points = high_symmetry_points([[1, 0], [8, 1]])
    for name, value in high_symmetry_points(np.eye(2)).items():
        assert np.linalg.norm(points[name]) == pytest.approx(np.linalg.norm(value))
