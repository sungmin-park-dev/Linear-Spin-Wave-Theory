"""Finite-temperature magnetization M(h, t) = -dF/dh (D49).

Temperatures are dimensionless, t = k_B T / E0. The square S = 1/2
antiferromagnet in a field along z is canted below h_sat = 4 and keeps a
Goldstone mode about z; above h_sat the polarized state is collinear and gapped.
"""

import warnings

import numpy as np
import pytest

from spintoolkit.methods.classical import refine_classical
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.methods.magnetization import magnetization_curve
from spintoolkit.models import neel_state, polarized_state, square_heisenberg
from spintoolkit.observables.thermal import thermal_quantities
from spintoolkit.system.conditions import ExternalConditions

T = np.array([0.0, 0.1, 0.3, 0.6])
CANTED_MESH = (17, 17)      # magnetization_curve mesh at k_density = 24 for the two-site cell


@pytest.fixture(scope="module")
def model():
    return square_heisenberg(J=1.0, S=0.5)


def moment_sum(model, state, h, t, mesh):
    """``ThermalResult.magnetization`` along z (g = 1 in this model)."""
    conditions = ExternalConditions(field=(0, 0, h))
    result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=mesh))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return thermal_quantities(result, t, gapless=False).magnetization[:, 2]


def test_zero_temperature_column_is_the_harmonic_magnetization(model):
    curve = magnetization_curve(model, neel_state(model, (1, 0, 0)), [1.0, 2.0], temperatures=T)
    assert curve.thermal.shape == (2, len(T))
    np.testing.assert_allclose(curve.thermal[:, 0], curve.harmonic, rtol=1e-12)
    # Canted: the moment sum misses the canting-angle shift already at t = 0.
    plain = moment_sum(model, curve.states[0], 1.0, [0.0], CANTED_MESH)[0]
    assert abs(plain - curve.harmonic[0]) > 0.05 * curve.harmonic[0]


def test_collinear_state_agrees_with_the_moment_sum(model):
    """Polarized along the field: no canting angle, so -dF/dh = S - <n>(t)."""
    curve = magnetization_curve(model, polarized_state(model), [4.5, 5.0], temperatures=T)
    for h, state, row in zip(curve.fields, curve.states, curve.thermal):
        np.testing.assert_allclose(row, moment_sum(model, state, h, T, (24, 24)), atol=1e-8)
    assert np.all(np.diff(curve.thermal, axis=1) < 0)


def test_maxwell_relation_dM_dt_equals_dS_dh(model):
    h, t, dt, dh = 1.0, 0.3, 1e-3, 1e-4
    curve = magnetization_curve(model, neel_state(model, (1, 0, 0)), [h],
                                temperatures=[t - dt, t + dt])
    dm_dt = (curve.thermal[0, 1] - curve.thermal[0, 0]) / (2 * dt)
    entropy = []
    for field in (h - dh, h + dh):
        conditions = ExternalConditions(field=(0, 0, field))
        state = refine_classical(model, curve.states[0], conditions)
        result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=CANTED_MESH))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            entropy.append(thermal_quantities(result, [t], gapless=False).entropy[0])
    # Agreement ~8e-7 relative, the O(dt^2) error of the difference.
    assert dm_dt == pytest.approx((entropy[1] - entropy[0]) / (2 * dh), rel=1e-5)


def test_goldstone_mode_leaves_the_magnetization_finite(model):
    """In 2D the boson numbers diverge with the mesh near a Goldstone mode; -dF/dh does not."""
    t = [0.3]
    coarse, fine = (magnetization_curve(model, neel_state(model, (1, 0, 0)), [1.0],
                                        k_density=k, temperatures=t) for k in (24, 48))
    assert fine.thermal[0, 0] == pytest.approx(coarse.thermal[0, 0], abs=1e-9)
    state = fine.states[0]
    drift = moment_sum(model, state, 1.0, t, (24, 24)) - moment_sum(model, state, 1.0, t, (48, 48))
    assert abs(drift[0]) > 1e-3


def test_unstable_branch_and_input_checks(model):
    curve = magnetization_curve(model, neel_state(model, (0, 0, 1)), [1.0], k_density=12,
                                temperatures=[0.0, 0.2])
    assert np.all(np.isnan(curve.thermal))
    default = magnetization_curve(model, polarized_state(model), [5.0], k_density=8)
    assert default.thermal.shape == (1, 0) and default.temperatures.shape == (0,)
    with pytest.raises(ValueError):
        magnetization_curve(model, polarized_state(model), [5.0], k_density=8,
                            temperatures=[-0.1])
