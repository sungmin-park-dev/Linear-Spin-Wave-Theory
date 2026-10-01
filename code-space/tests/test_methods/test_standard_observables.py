"""Density of states, spin components, energy-integrated S(Q) and M(h) (D45)."""

import warnings

import numpy as np
import pytest

from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.methods.magnetization import magnetization_curve
from spintoolkit.models import (neel_state, polarized_state, square_heisenberg, state_120,
                                triangular_heisenberg)
from spintoolkit.observables.bands import density_of_states
from spintoolkit.observables.neutron import correlation_path, static_slice
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.high_symmetry import high_symmetry_points


@pytest.fixture(scope="module")
def ferromagnet():
    """Triangular ferromagnet, S = 1/2, h = 0.5: w(k) = h + S (6 - 2 sum cos), band 0.5 ... 5.0."""
    model = triangular_heisenberg(J=-1.0)
    return solve_lswt(model, polarized_state(model), ExternalConditions(field=(0, 0, 0.5)),
                      settings=LSWTSettings(mesh=(48, 48)))


def test_density_of_states_integrates_to_one_inside_the_band(ferromagnet):
    omega = np.linspace(-1, 7, 4001)
    dos = density_of_states(ferromagnet, omega, 0.05)
    assert np.sum(dos.dos) * (omega[1] - omega[0]) == pytest.approx(1.0, abs=1e-6)
    outside = (omega < 0.3) | (omega > 5.2)
    assert np.max(dos.dos[outside]) < 1e-3


def test_ferromagnet_fluctuations_are_transverse_with_weight_S_over_2(ferromagnet):
    """|0> polarized along z: S^xx = S^yy = S/2 per magnon, S^zz = 0 (one-magnon)."""
    omega = np.linspace(0, 6, 1201)
    path = ("Γ", "K", "M")
    xx = correlation_path(ferromagnet, omega, 0.05, "xx", path, points=20)
    zz = correlation_path(ferromagnet, omega, 0.05, "zz", path, points=20)
    step = omega[1] - omega[0]
    np.testing.assert_allclose(xx.intensity.sum(axis=1) * step, 0.25, atol=2e-3)
    np.testing.assert_allclose(zz.intensity, 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        correlation_path(ferromagnet, omega, 0.05, "ab")


def test_static_slice_puts_the_ordered_moment_in_the_bragg_peaks():
    """Neel along x on the square lattice: Bragg weight at (pi, pi) is m^2 times the
    polarization factor 1 - Q_x^2 / Q^2 = 1/2, with m = S - <n> and g = 2 (moment = S)."""
    model = square_heisenberg()
    result = solve_lswt(model, neel_state(model, (1, 0, 0)), settings=LSWTSettings(mesh=(16, 16)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                   # zero modes at the Bragg vectors
        cut = static_slice(result, extent=4.0, points=9, g=2.0)
    m = result.ordered_moments()[0]
    at = np.flatnonzero(np.all(np.isclose(np.abs(cut.bragg_Q), np.pi), axis=1))
    assert len(at) == 4
    np.testing.assert_allclose(cut.bragg_intensity[at], 0.5 * m ** 2, rtol=1e-10)
    origin = np.flatnonzero(np.all(np.isclose(cut.bragg_Q, 0.0), axis=1))
    assert np.all(np.isnan(cut.bragg_intensity[origin]))  # Q = 0: polarization factor undefined
    assert cut.intensity.shape == (9, 9)


def test_magnetization_curve_limits_of_the_square_antiferromagnet():
    """Classical M = h / 8J up to h_sat = 8JS; at h = 0 and above saturation the 1/S
    correction vanishes (symmetry, and no zero-point motion of the polarized state)."""
    model = square_heisenberg(J=1.0, S=0.5)
    curve = magnetization_curve(model, neel_state(model, (1, 0, 0)), [0.0, 1.0, 2.0, 4.5],
                                k_density=24)
    np.testing.assert_allclose(curve.classical, [0.0, 0.125, 0.25, 0.5], atol=1e-7)
    assert abs(curve.harmonic[0]) < 1e-7
    assert curve.harmonic[-1] == pytest.approx(0.5, abs=1e-7)
    assert 0 < curve.harmonic[1] < curve.classical[1]          # quantum reduction at low field
    np.testing.assert_allclose(curve.moment_reduction[-1], 0.0, atol=1e-12)
    np.testing.assert_allclose(curve.ordered_moments[0], 0.5 - curve.moment_reduction[0])
    assert curve.moment_reduction[0][0] > curve.moment_reduction[2][0] > 0


def test_unstable_branch_gives_nan_not_a_number():
    """Neel along the field is a saddle: LSWT is unstable and M is NaN at harmonic order."""
    model = square_heisenberg(J=1.0, S=0.5)
    curve = magnetization_curve(model, neel_state(model, (0, 0, 1)), [1.0], k_density=12)
    assert np.isnan(curve.harmonic[0]) and np.all(np.isnan(curve.moment_reduction[0]))
