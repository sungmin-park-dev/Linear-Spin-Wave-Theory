"""Undefined or non-physical quantities are NaN, not a filler number (2026-10-15 rule).

Each test sets up a case where the physical answer does not exist and checks
that NaN is returned, next to a case where the value exists and is reported.
"""

import warnings

import numpy as np
import pytest

from spintoolkit.methods.lswt.run import LSWTError, LSWTSettings, solve_lswt
from spintoolkit.methods.monte_carlo import _binned
from spintoolkit.methods.nlswt import NLSWTSettings, solve_nlswt
from spintoolkit.methods.state_selection import _harmonic_modes
from spintoolkit.models import neel_state, polarized_state, square_heisenberg
from spintoolkit.observables.structure_factor import structure_factor
from spintoolkit.observables.thermal import thermal_quantities
from spintoolkit.observables.two_magnon import two_magnon_continuum
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions


def test_harmonic_energy_of_a_landau_instability_is_nan():
    # Real frequencies but negative mode energies: H is not positive semidefinite.
    energy, unstable, omega = _harmonic_modes(np.diag([-0.3, 1.0, -0.3, 1.0]).astype(complex))
    assert np.isnan(energy) and unstable >= 1 and len(omega) == 0
    # A positive semidefinite block with an exact zero mode: zero-point energy exists.
    energy, unstable, omega = _harmonic_modes(np.diag([0.0, 1.0, 0.0, 1.0]).astype(complex))
    assert unstable == 0 and np.isfinite(energy) and np.isclose(omega.min(), 0.0)


def test_regularization_does_not_repair_an_unstable_state():
    ferro = square_heisenberg(J=-1.0)
    for regularization in ("MAGSWT", "k-dependent"):
        with pytest.raises(LSWTError, match="negative eigenvalue"):
            solve_lswt(ferro, neel_state(ferro),
                       settings=LSWTSettings(mesh=(4, 4), regularization=regularization))


def test_non_stationary_reference_has_no_energy():
    model = square_heisenberg(J=1.0)
    tilted = SpinState.from_function(model, np.eye(2, dtype=int),
                                     lambda s, c: np.array([0.6, 0.0, 0.8]))
    with pytest.warns(UserWarning, match="not stationary"):
        result = solve_lswt(model, tilted, ExternalConditions(field=(0, 0, 5.0)),
                            settings=LSWTSettings(mesh=(4, 4)))
    assert np.isfinite(result.classical_energy)
    assert np.isnan(result.zero_point_energy) and np.isnan(result.ground_state_energy)
    assert np.isnan(result.boson_numbers).all() and np.isnan(result.ordered_moments()).all()


def test_thermal_bragg_intensity_uses_thermal_moments_or_is_nan():
    # Gapped ferromagnet in a field: Bragg intensity at q = 0 from S - <n>(t).
    fm = square_heisenberg(J=-1.0)
    ferro = solve_lswt(fm, polarized_state(fm), ExternalConditions(field=(0, 0, 1.0)),
                       settings=LSWTSettings(mesh=(16, 16)))
    t = 0.4
    n = thermal_quantities(ferro, [t]).boson_numbers[0][0]
    sf = structure_factor(ferro, [[0.0, 0.0]], temperature=t)
    assert sf.elastic[0, 2, 2].real == pytest.approx((0.5 - n) ** 2, rel=1e-12)
    assert n > 1e-3                                    # differs from the t = 0 moment
    # Gapless Neel state: no order at t > 0 in two dimensions, the Bragg intensity is NaN.
    model = square_heisenberg(J=1.0)
    neel = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(12, 12)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cold = structure_factor(neel, [[np.pi, np.pi]])
        hot = structure_factor(neel, [[np.pi, np.pi]], temperature=0.3, gapless=True)
    assert np.isfinite(cold.elastic[0]).all() and np.isnan(hot.elastic[0]).all()


def test_jackknife_error_needs_at_least_two_blocks():
    values = np.random.default_rng(0).normal(size=40)
    assert np.isnan(_binned(values, np.mean, 1)[1])
    assert np.isnan(_binned(values, np.mean, 41)[1])
    assert np.isfinite(_binned(values, np.mean, 4)[1])


def test_left_out_two_magnon_pairs_have_no_energy():
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(8, 8)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        continuum = two_magnon_continuum(result, [[0.4, 0.2]], mesh=(8, 8), temperature=0.0)
        zone = two_magnon_continuum(result, [[0.0, 0.0]], mesh=(8, 8), temperature=0.0)
    for c in (continuum, zone):
        left_out = np.isnan(c.energies)
        assert np.all(c.weights[left_out] == 0)
        assert np.isfinite(c.spectrum(np.linspace(0, 5, 11), 0.1)).all()


def test_nlswt_decay_rate_needs_a_broadening_in_the_thermodynamic_limit():
    model = square_heisenberg(S=1.0)
    result = solve_nlswt(model, neel_state(model), settings=NLSWTSettings(mesh=(6, 6)))
    k = [[0.7, 0.2]]
    assert np.isnan(result.magnon_energies(k)["decay_rate"]).all()
    assert np.isfinite(result.magnon_energies(k, eta=0.05)["decay_rate"]).all()
    out = result.spectral_function(k, np.linspace(0.1, 3.0, 7), 0.05, mode="dyson",
                                   grid=(8, 8), levels=40)
    assert np.isnan(out["negative_frequency_weight"]).all()    # no negative frequencies given


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_legacy_solver_gives_no_energy_for_an_unstable_state():
    from spintoolkit import LSWTSolver, SpinSystem

    def chain(J):
        system = SpinSystem(lattice_vectors=[[2, 0], [0, 1]])
        system.add_site("A", [0, 0], spin=0.5, angles=[0, 0], magnetic_field=[0, 0, 0])
        system.add_site("B", [1, 0], spin=0.5, angles=[np.pi, 0], magnetic_field=[0, 0, 0])
        system.add_coupling("A", "B", J * np.eye(3), displacement=[0, 0])
        system.add_coupling("B", "A", J * np.eye(3), displacement=[1, 0])
        return system
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unstable = LSWTSolver(chain(-1.0), bz_type="simple").solve(N=4, regularization="MAGSWT")
        stable = LSWTSolver(chain(1.0), bz_type="simple").solve(N=4, regularization="MAGSWT")
    assert np.isnan(unstable.ground_state_energy) and np.isnan(unstable.eigenvalues).all()
    assert np.isfinite(stable.ground_state_energy)       # zero modes lifted, not an instability
