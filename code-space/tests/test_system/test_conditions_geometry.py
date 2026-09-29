"""External conditions (dimensionless field and temperature) and calculation geometry."""

import numpy as np
import pytest

from spintoolkit.definitions import K_BOLTZMANN_MEV, MU_B_MEV_PER_T
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry


def test_conditions_are_dimensionless_values():
    conditions = ExternalConditions(field=(0, 0, 2.0), temperature=0.3)
    np.testing.assert_array_equal(conditions.field, [0, 0, 2.0])
    assert conditions.temperature == 0.3
    with pytest.raises(ValueError):
        conditions.field[2] = 1.0
    np.testing.assert_array_equal(ExternalConditions().field, [0, 0, 0])


def test_documented_physical_conversion_round_trips():
    """The user-side conversion stated in the docstrings: field = mu_B B / E0."""
    e0_mev, b_tesla, t_kelvin = 0.0779, 3.5, 1.2
    conditions = ExternalConditions(field=(0, 0, MU_B_MEV_PER_T * b_tesla / e0_mev),
                                    temperature=K_BOLTZMANN_MEV * t_kelvin / e0_mev)
    assert conditions.field[2] * e0_mev / MU_B_MEV_PER_T == pytest.approx(b_tesla)
    assert conditions.temperature * e0_mev / K_BOLTZMANN_MEV == pytest.approx(t_kelvin)


@pytest.mark.parametrize("kwargs", [dict(field=(0, 1)), dict(field=(0, 0, np.inf)),
                                    dict(temperature=-1.0), dict(temperature=np.nan)])
def test_invalid_conditions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ExternalConditions(**kwargs)


def test_geometry_kinds():
    assert CalculationGeometry.thermodynamic_limit().num_cells is None
    assert CalculationGeometry.finite_torus([[3, 0], [0, 4]]).num_cells == 12
    for cluster in ([[1, 1], [2, 2]], [[1.5, 0], [0, 2]]):
        with pytest.raises(ValueError):
            CalculationGeometry.finite_torus(cluster)
