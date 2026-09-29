"""External conditions (field, temperature units) and calculation geometry."""

import numpy as np
import pytest

from spintoolkit.definitions import K_BOLTZMANN_MEV, MU_B_MEV_PER_T
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import Units


def test_mev_model_uses_bohr_magneton_and_boltzmann_constant():
    conditions = ExternalConditions(B=(0, 0, 2.0), B_unit="T", T=3.0, T_unit="K")
    units = Units(energy="meV")
    np.testing.assert_allclose(conditions.zeeman_field(units), [0, 0, 2.0 * MU_B_MEV_PER_T])
    assert conditions.thermal_energy(units) == pytest.approx(3.0 * K_BOLTZMANN_MEV)


def test_relative_model_takes_energy_units_and_needs_a_scale_for_tesla():
    relative = ExternalConditions(B=(0, 0, 1.5), T=0.2)
    np.testing.assert_allclose(relative.zeeman_field(Units()), [0, 0, 1.5])
    assert relative.thermal_energy(Units()) == 0.2
    tesla = ExternalConditions(B=(0, 0, 1.0), B_unit="T")
    with pytest.raises(ValueError, match="energy_scale_meV"):
        tesla.zeeman_field(Units())
    scaled = Units(energy_scale_meV=0.5)
    np.testing.assert_allclose(tesla.zeeman_field(scaled), [0, 0, MU_B_MEV_PER_T / 0.5])


def test_mev_model_rejects_relative_field():
    with pytest.raises(ValueError, match="need B in T"):
        ExternalConditions(B=(0, 0, 1.0)).zeeman_field(Units(energy="meV"))


@pytest.mark.parametrize("kwargs", [dict(B=(0, 1)), dict(B_unit="gauss"),
                                    dict(T=-1.0), dict(T_unit="C")])
def test_invalid_conditions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ExternalConditions(**kwargs)


def test_geometry_kinds():
    assert CalculationGeometry.thermodynamic_limit().num_cells is None
    assert CalculationGeometry.finite_torus([[3, 0], [0, 4]]).num_cells == 12
    for cluster in ([[1, 1], [2, 2]], [[1.5, 0], [0, 2]]):
        with pytest.raises(ValueError):
            CalculationGeometry.finite_torus(cluster)
