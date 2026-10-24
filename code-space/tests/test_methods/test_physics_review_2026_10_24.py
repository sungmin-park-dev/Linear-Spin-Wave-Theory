"""Regression tests for the second round of the 2026-10-23 external review (D75).

1. A vacuum with ``<n_i> > S_i`` (the moment would reverse) is outside the
   spin-wave expansion: every quantity built from it is NaN with a warning,
   the LSWT bands are kept.
2. Instabilities between mesh points are not certified as stable: a negative
   eigenvalue of ``H(k)`` found by the zero-mode scan, a frozen SCSWT vacuum
   past a gap closing at k = 0, a quartic saddle between sampled directions,
   a symmetry test that depends on the energy unit, and a selected state that
   is unstable at continuous k. The counterexamples come from closed forms.
"""

import warnings

import numpy as np
import pytest

import spintoolkit as stk
from spintoolkit.methods.classical import classical_energy
from spintoolkit.methods.luttinger_tisza import luttinger_tisza
from spintoolkit.methods.nlswt import SCSWTSettings, solve_scswt
from spintoolkit.methods.stability import local_stability
from spintoolkit.methods.state_selection import (LSWTHarmonicFreeEnergy, LSWTZeroPointEnergy,
                                                 select_on_manifold)
from spintoolkit.models import triangular_heisenberg
from spintoolkit.observables.berry import thermal_hall
from spintoolkit.observables.structure_factor import (bond_correlations, spin_correlation,
                                                      structure_factor)
from spintoolkit.observables.thermal import thermal_quantities
from spintoolkit.observables.two_magnon import two_magnon_continuum
from spintoolkit.observables.zero_modes import scan_zero_modes

from test_physics_review_2026_10_23 import honeycomb_ferromagnet, neel, square


@pytest.fixture(scope="module")
def quasi_1d():
    """Square Neel with J_y = 0.01 J, S = 1/2: <n> = 0.72 > S on a 40 x 40 mesh."""
    model = square(Jy=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stk.solve_lswt(model, neel(model), stk.ExternalConditions(),
                                settings=stk.LSWTSettings(mesh=(40, 40)))
    assert result.header.diagnostics["max_boson_number_per_spin"] > 1
    return model, result


def test_vacuum_observables_are_nan_when_bosons_exceed_spin(quasi_1d):
    model, result = quasi_1d
    with pytest.warns(UserWarning, match="exceed S"):
        factor = structure_factor(result, [[0.5, 0.3]])
    assert np.all(np.isnan(factor.weights)) and np.all(np.isfinite(factor.energies))
    with pytest.warns(UserWarning, match="exceed S"):
        correlation = spin_correlation(result, model, ("A", (0, 0)), ("A", (1, 0)))
    assert np.all(np.isnan(correlation["total"]))
    with pytest.warns(UserWarning, match="exceed S"):
        bonds = bond_correlations(result, model)
    assert all(np.isnan(b["correlation"]).all() for b in bonds["bonds"])
    assert bonds["energy"] == pytest.approx(result.ground_state_energy, abs=1e-12)
    with pytest.warns(UserWarning, match="exceed S"):
        static = two_magnon_continuum(result, [[0.5, 0.3]], mesh=(8, 8)).static()
    assert np.all(np.isnan(static))
    assert np.all(np.isfinite(result.bands()))            # classical oscillation frequencies


def test_nlswt_corrections_are_nan_when_bosons_exceed_spin():
    model = square(Jy=0.01)
    with pytest.warns(UserWarning, match="exceeds the spin"):
        result = stk.solve_nlswt(model, neel(model), settings=stk.NLSWTSettings(mesh=(24, 24)))
    assert result.header.diagnostics["max_boson_number_per_spin"] > 1
    assert np.isnan(result.energies.hartree_fock) and np.isnan(result.ground_state_energy)
    assert np.isfinite(result.energies.zero_point)
    with pytest.warns(UserWarning, match="exceed S"):
        out = result.magnon_energies([[0.4, 0.2]], eta=0.05)
    assert np.all(np.isfinite(out["lswt"])) and np.all(np.isnan(out["total"]))


def test_nlswt_boson_numbers_match_lswt():
    model = square()
    lswt = stk.solve_lswt(model, neel(model), stk.ExternalConditions(),
                          settings=stk.LSWTSettings(mesh=(24, 24)))
    nlswt = stk.solve_nlswt(model, neel(model), settings=stk.NLSWTSettings(mesh=(24, 24)))
    ratio = nlswt.header.diagnostics["max_boson_number_per_spin"]
    assert ratio == pytest.approx(lswt.header.diagnostics["max_boson_number_per_spin"], rel=2e-2)
    assert ratio < 1 and not nlswt.header.diagnostics["boson_numbers_exceed_spin"]


def test_thermal_hall_is_nan_where_thermal_bosons_exceed_spin():
    model, state = honeycomb_ferromagnet(1.0, 0.2)
    result = stk.solve_lswt(model, state, stk.ExternalConditions(field=(0, 0, 0.3)),
                            settings=stk.LSWTSettings(mesh=(16, 16)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beyond = thermal_quantities(result, [0.1, 3.0], gapless=False).beyond_lswt
    assert list(beyond) == [False, True]
    with pytest.warns(UserWarning, match="exceed S"):
        kappa = thermal_hall(result, [0.1, 3.0], gapless=False).kappa_over_t
    assert np.isfinite(kappa[0]) and np.isnan(kappa[1])


def test_harmonic_free_energy_is_nan_where_thermal_bosons_exceed_spin():
    site = stk.Site("A", (0.0, 0.0), 0.5)
    terms = [stk.Term.bilinear(("A", (0, 0)), ("A", d), stk.heisenberg(-1.0)) for d in [(1, 0), (0, 1)]]
    terms.append(stk.Term.zeeman("A", np.eye(3)))
    model = stk.SpinModel(np.eye(2), (site,), tuple(terms), {"model_id": "fm"})
    state = stk.SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: np.array([0, 0, 1.0]))
    conditions = stk.ExternalConditions(field=(0, 0, 0.2))
    assert np.isfinite(LSWTHarmonicFreeEnergy("Tetra", 12, 0.05)(model, state, conditions))
    hot = LSWTHarmonicFreeEnergy("Tetra", 12, 100.0)
    assert np.isnan(hot(model, state, conditions))
    assert hot.describe()["boson_numbers_above_spin_calls"] == 1


def test_bands_refuse_an_augmented_result():
    model = square(J=1.0, S=0.5, field=True)
    up = stk.SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: np.array([0, 0, 1.0]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stk.solve_lswt(model, up, stk.ExternalConditions(field=(0, 0, 1.0)),
                                settings=stk.LSWTSettings(mesh=(8, 8), augmentation="minimal"))
    assert result.extra["augmentation"]["mu"] > 0
    with pytest.raises(ValueError, match="MAGSWT"):
        result.bands()
    assert result.to_json_dict()["lswt"]["bands"] is None


def frustrated_chain(S=0.5):
    """J1 = 1, J2 = 0.7 along x, J = -1 along y: h_sat = S [J(0) - J(q)], cos q = -1/2.8."""
    site = stk.Site("A", (0.0, 0.0), S)
    terms = [stk.Term.bilinear(("A", (0, 0)), ("A", (1, 0)), stk.heisenberg(1.0)),
             stk.Term.bilinear(("A", (0, 0)), ("A", (2, 0)), stk.heisenberg(0.7)),
             stk.Term.bilinear(("A", (0, 0)), ("A", (0, 1)), stk.heisenberg(-1.0)),
             stk.Term.zeeman("A", np.eye(3))]
    model = stk.SpinModel(np.eye(2), (site,), tuple(terms), {"model_id": "chain"})
    J = lambda k: 2 * np.cos(k) + 1.4 * np.cos(2 * k) - 2       # noqa: E731
    h_sat = S * (J(0.0) - J(np.arccos(-1 / 2.8)))
    up = stk.SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: np.array([0, 0, 1.0]))
    return model, up, h_sat


def test_negative_eigenvalue_between_mesh_points_is_an_instability():
    model, up, h_sat = frustrated_chain()
    result = stk.solve_lswt(model, up, stk.ExternalConditions(field=(0, 0, h_sat - 0.01)),
                            settings=stk.LSWTSettings(mesh=(8, 8)))
    assert result.bands().min() > 0.05                     # the mesh sees no instability
    report = scan_zero_modes(result)
    assert report.has_unstable and not report.has_zero
    with pytest.warns(UserWarning, match="negative eigenvalue"):
        thermal = thermal_quantities(result, [0.1])
    assert np.isnan(thermal.free_energy[0]) and np.isnan(thermal.internal_energy[0])
    gapped = stk.solve_lswt(model, up, stk.ExternalConditions(field=(0, 0, h_sat + 0.01)),
                            settings=stk.LSWTSettings(mesh=(8, 8)))
    assert not scan_zero_modes(gapped).has_unstable


def test_frozen_scswt_vacuum_past_the_gap_closing_at_k0_is_nan():
    # triangular up-up-down, S = 1/2: the k = 0 gap closes at h_c1 = 1.26651 (D72), which the
    # 1/3-offset mesh does not contain
    model = triangular_heisenberg(J=1.0, S=0.5)
    directions = [(0, 0, -1.0), (0, 0, 1.0), (0, 0, 1.0)]
    state = stk.SpinState.from_function(
        model, [[1, 1], [-1, 2]], lambda site, cell: np.asarray(directions[(cell[0] - cell[1]) % 3]))
    settings = SCSWTSettings(mesh=(12, 12))
    ref = solve_scswt(model, state, stk.ExternalConditions(field=(0, 0, 1.5)), settings=settings)
    h_c1 = 1.2665145993202
    with pytest.warns(UserWarning, match="between the mesh points"):
        below = solve_scswt(model, state, stk.ExternalConditions(field=(0, 0, h_c1 - 1e-4)),
                            settings=settings, frozen=ref)
    assert np.isnan(below.ground_state_energy) and np.isnan(below.boson_numbers).all()
    above = solve_scswt(model, state, stk.ExternalConditions(field=(0, 0, h_c1 + 1e-3)),
                        settings=settings, frozen=ref)
    assert np.isfinite(above.ground_state_energy)


def test_quartic_saddle_between_sampled_directions_is_found():
    # Exact uniform energy change [4(q1-1)^2 + 16(q2-1)^2 - 16.4(q1-1)(q2-1)]/2 with
    # q_i = (1 + x_i^2)^(-1/2): zero quadratic term, quartic -(2/45) a^4 along x ~ (sqrt 2, 1)
    sites = (stk.Site("A", (0, 0), 1.0), stk.Site("B", (0.5, 0.5), 1.0))
    terms = [stk.Term.bilinear((s, (0, 0)), (s, d), -np.eye(3)) for s in "AB" for d in [(1, 0), (0, 1)]]
    terms += [stk.Term.bilinear(("A", (0, 0)), ("B", (0, 0)), np.diag([0, 0, 16.4])),
              stk.Term.onsite("A", np.diag([0, 40, 8.0])), stk.Term.onsite("B", np.diag([0, 40, 32.0])),
              stk.Term.zeeman("A", np.diag([0, 0, 8.4])), stk.Term.zeeman("B", np.diag([0, 0, 15.6]))]
    model = stk.SpinModel(np.eye(2), sites, tuple(terms), {"model_id": "narrow_quartic"})
    sign = {"A": -1.0, "B": 1.0}
    state = stk.SpinState.from_function(model, np.eye(2, dtype=int),
                                        lambda s, c: np.array([0, 0, sign[s]]))
    conditions = stk.ExternalConditions(field=(0, 0, 1.0))
    a = 0.05
    x = np.sqrt(2) * a * np.array([np.sqrt(2), 1]) / np.sqrt(3)
    tilted = stk.SpinState.from_function(
        model, np.eye(2, dtype=int),
        lambda s, c: np.array([x[int(s == "B")], 0, sign[s]]) / np.hypot(1, x[int(s == "B")]))
    change = classical_energy(model, tilted, conditions) - classical_energy(model, state, conditions)
    assert change / a ** 4 == pytest.approx(-2 / 45, rel=0.05)       # per spin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = local_stability(model, state, conditions)
    assert report.verdict == "unstable"


@pytest.mark.parametrize("c", [1.0, 1e-6, 1e-12])
def test_symmetry_classification_does_not_depend_on_the_energy_unit(c):
    # S = 1 square Neel at its spinodal (A_zz = -0.1, h = 0.9): quartic -81 a^4 / 67240 for
    # every rescaling c of all couplings
    site = stk.Site("A", (0, 0), 1.0)
    terms = [stk.Term.bilinear(("A", (0, 0)), ("A", d), c * np.eye(3)) for d in [(1, 0), (0, 1)]]
    terms += [stk.Term.onsite("A", c * np.diag([0, 0, -0.1])), stk.Term.zeeman("A", c * np.eye(3))]
    model = stk.SpinModel(np.eye(2), (site,), tuple(terms), {"model_id": "spinodal"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = local_stability(model, neel(model), stk.ExternalConditions(field=(0, 0, 0.9)))
    assert report.verdict == "unstable"


def test_luttinger_tisza_warns_when_bonds_alias_on_the_mesh():
    site = stk.Site("A", (0, 0), 1.0)
    bond = lambda d, J: stk.Term.bilinear(("A", (0, 0)), ("A", d), stk.heisenberg(J))  # noqa: E731
    terms = [bond((1, 0), -1), bond((48, 0), 2), bond((47, 0), -1), bond((49, 0), -1),
             bond((0, 1), -0.1)]
    model = stk.SpinModel(np.eye(2), (site,), tuple(terms), {"model_id": "long_bonds"})
    with pytest.warns(UserWarning, match="mesh"):
        coarse = luttinger_tisza(model)
    assert coarse.diagnostics["global_minimum_certified"] is False
    fine = luttinger_tisza(model, mesh=(200, 8))
    assert fine.lambda_min == pytest.approx(-3.0978591555, abs=1e-8)
    assert coarse.lambda_min > fine.lambda_min + 1          # the coarse search missed it


def test_selection_rejects_a_state_unstable_between_provider_momenta():
    # compass Jx = diag(-1, -a, 0), Jy = diag(-a, -1, 0) and j3 on (2, 0): the x state has
    # d_x(q) ~ S (a - 4 j3) q^2 < 0 for j3 > a / 4, invisible on the Tetra N = 4 mesh
    def model(j3, a=0.2):
        site = stk.Site("A", (0, 0), 0.5)
        terms = [stk.Term.bilinear(("A", (0, 0)), ("A", (1, 0)), np.diag([-1, -a, 0.0])),
                 stk.Term.bilinear(("A", (0, 0)), ("A", (0, 1)), np.diag([-a, -1, 0.0])),
                 stk.Term.bilinear(("A", (0, 0)), ("A", (2, 0)), j3 * np.eye(3))]
        return stk.SpinModel(np.eye(2), (site,), tuple(terms), {"model_id": "compass"})

    def select(j3):
        m = model(j3)
        state = stk.SpinState.from_function(m, np.eye(2, dtype=int), lambda s, c: np.array([1.0, 0, 0]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return select_on_manifold(m, state, None, LSWTZeroPointEnergy("Tetra", 4), axis=[0, 0, 1])

    assert select(0.03).verdict == "selected"
    unstable = select(0.0501)
    assert unstable.verdict == "not_soft" and "continuous momenta" in unstable.message
