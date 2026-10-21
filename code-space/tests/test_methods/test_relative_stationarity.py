"""Stationarity tolerance is relative (D68): the verdict does not depend on the energy unit or S.

A Neel state tilted by a fixed small angle has the same relative torque
max|S x h| / max S|h| whatever the exchange scale, so ``solve_lswt`` calls it
stationary or not independently of the unit; the absolute torque scales with J.
"""

import warnings

import numpy as np
import pytest

from spintoolkit.methods.classical import stationarity
from spintoolkit.methods.lswt import solve_lswt
from spintoolkit.models import square_heisenberg
from spintoolkit.states.spin_state import SpinState


def tilted_neel(model, angle):
    def direction(site, cell):
        sign = (-1) ** (cell[0] + cell[1])
        tilt = angle if (cell[0] + cell[1]) % 2 == 0 else 0.0
        return sign * np.array([np.cos(tilt), np.sin(tilt), 0.0])
    return SpinState.from_function(model, [[1, 1], [1, -1]], direction)


@pytest.mark.parametrize("J, S", [(1.0, 0.5), (1e-6, 0.5), (1e4, 2.5)])
def test_relative_torque_is_scale_free(J, S):
    model = square_heisenberg(J=J, S=S)
    torque, relative = stationarity(model, tilted_neel(model, 1e-6))
    assert relative == pytest.approx(np.sin(1e-6), rel=1e-6)
    assert torque == pytest.approx(J * S * S * 4 * np.sin(1e-6), rel=1e-6)


@pytest.mark.parametrize("J", [1.0, 1e-9])
def test_verdict_does_not_depend_on_the_energy_unit(J):
    model = square_heisenberg(J=J)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off = solve_lswt(model, tilted_neel(model, 1e-4))
        on = solve_lswt(model, tilted_neel(model, 1e-10))
    assert not off.header.diagnostics["stationary"] and np.isnan(off.zero_point_energy)
    assert on.header.diagnostics["stationary"] and np.isfinite(on.zero_point_energy)
