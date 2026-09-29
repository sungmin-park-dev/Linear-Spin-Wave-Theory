"""Physical SI normalization against independent two-band boson curvature.

The executable example owns the analytic model and quadrature reference. These
checks do not reproduce the production c_2 function or its prefactor formula.
They certify full, equal-weight MBZ data with nondegenerate positive bands.
"""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spintoolkit.observables.thermodynamics import Thermodynamics
from spintoolkit.observables.topology import Topology

EXAMPLE = Path(__file__).resolve().parents[3] / "examples/thermal_hall_reference_check.py"
SPEC = importlib.util.spec_from_file_location("thermal_hall_reference", EXAMPLE)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
HALL = "Thermal Hall Conductance"


@pytest.fixture(scope="module")
def benchmark():
    return REFERENCE.model(n=12)


def hall_values(parent, data, temperature, spacing=None):
    topology = Topology(parent).compute_thermal_Hall(
        data, temperature, layer_spacing_m=spacing
    )[2]
    combined = Thermodynamics(parent).compute_thermodynamic_quantities_at_T(
        data, temperature, layer_spacing_m=spacing
    )[HALL]
    return np.array([topology, combined])


@pytest.mark.parametrize("temperature", [0.0, 0.8, 2.0, 5.0])
def test_both_paths_match_independent_si_integral(benchmark, temperature):
    parent, data, omega, modes = benchmark
    expected = REFERENCE.reference(omega, modes, 1.0, temperature)
    # SI and meV constants have slightly different published rounding.
    assert_allclose(hall_values(parent, data, temperature), expected, rtol=1e-7, atol=1e-28)


@pytest.mark.parametrize("spacing", [7e-10, 1.4e-9])
def test_bulk_response_divides_layer_response_by_spacing(benchmark, spacing):
    parent, data, omega, modes = benchmark
    expected = REFERENCE.reference(omega, modes, 1.0, 2.0) / spacing
    bulk = hall_values(parent, data, 2.0, spacing)
    assert_allclose(bulk, expected, rtol=1e-9, atol=0)
    assert_allclose(bulk * spacing, hall_values(parent, data, 2.0), rtol=1e-14, atol=0)


@pytest.mark.parametrize("cell", [
    [[3.7, 0], [0, 3.7]],
    [[2, 0.4], [0.3, 1.1]],
    [[-2, 0.4], [0.3, 1.1]],
    [[3.7e-10, 0], [0, 3.7e-10]],
])
def test_cartesian_length_units_and_cell_shape_cancel_in_integral(benchmark, cell):
    parent, data, omega, modes = REFERENCE.model(n=12, lattice_vectors=cell)
    area = abs(np.linalg.det(cell))
    expected = REFERENCE.reference(omega, modes, area, 2.0)
    actual = hall_values(parent, data, 2.0)
    assert_allclose(actual, expected, rtol=1e-9, atol=0)
    original_parent, original_data, _, _ = benchmark
    orientation = np.sign(np.linalg.det(cell))
    assert_allclose(actual, orientation * hall_values(original_parent, original_data, 2.0),
                    rtol=1e-12, atol=0)


def test_chirality_reverses_curvature_chern_and_hall(benchmark):
    parent, data, _, _ = benchmark
    reversed_parent, reversed_data, _, _ = REFERENCE.model(n=12, orientation=-1)
    positive = Topology(parent).compute_thermal_Hall(data, 2.0)
    negative = Topology(reversed_parent).compute_thermal_Hall(reversed_data, 2.0)
    for first, second in zip(positive, negative):
        assert_allclose(first, -second, rtol=1e-12, atol=1e-28)
    assert_allclose(hall_values(reversed_parent, reversed_data, 2.0),
                    -hall_values(parent, data, 2.0), rtol=1e-12, atol=0)


def test_temperature_sweep_propagates_layer_spacing(benchmark):
    parent, data, omega, modes = benchmark
    thermodynamics = Thermodynamics(parent)
    temperatures, layer = thermodynamics.get_thermodynamic_quantities(
        data, Temperature_range=(0, 2, 1)
    )
    _, bulk = thermodynamics.get_thermodynamic_quantities(
        data, Temperature_range=(0, 2, 1), layer_spacing_m=7e-10
    )
    expected = [REFERENCE.reference(omega, modes, 1.0, t) for t in temperatures]
    assert_allclose(layer[HALL], expected, rtol=1e-8, atol=1e-28)
    assert_allclose(bulk[HALL] * 7e-10, layer[HALL], rtol=1e-14, atol=1e-28)
    for key in layer.keys() - {HALL}:
        assert_allclose(layer[key], bulk[key], rtol=0, atol=0)


@pytest.mark.parametrize("temperature", [0.0, 2.0])
def test_detached_thermodynamics_reports_missing_geometry_only_for_hall(benchmark, temperature):
    parent, data, _, _ = benchmark
    attached = Thermodynamics(parent).compute_thermodynamic_quantities_at_T(data, temperature)
    detached = Thermodynamics().compute_thermodynamic_quantities_at_T(data, temperature)
    assert np.isnan(detached[HALL])
    for key in attached.keys() - {HALL}:
        assert_allclose(detached[key], attached[key], rtol=0, atol=0)
    geometry_missing = SimpleNamespace(Ns=2, bz_data=parent.bz_data)
    assert np.isnan(Topology(geometry_missing).compute_thermal_Hall(data, temperature)[2])


@pytest.mark.parametrize("invalid_exclude", [True, False])
@pytest.mark.parametrize("temperature", [0.0, 2.0])
def test_failed_samples_do_not_become_a_reweighted_hall_integral(benchmark, temperature,
                                                                invalid_exclude):
    parent, original, _, _ = benchmark
    data = copy.deepcopy(original)
    next(iter(data.values()))[2] = (False, None)
    curvature, chern, hall = Topology(parent).compute_thermal_Hall(data, temperature)
    assert np.isnan(curvature[0]).all()
    assert np.isnan(chern).all()
    assert np.isnan(hall)
    result = Thermodynamics(parent).compute_thermodynamic_quantities_at_T(
        data, temperature, invalid_exclude=invalid_exclude
    )
    assert np.isnan(result[HALL])
    assert np.isfinite(result["Internal Energy Density"])


def test_missing_derivatives_do_not_prevent_other_thermodynamics(benchmark):
    parent, original, _, _ = benchmark
    data = copy.deepcopy(original)
    for entry in data.values():
        entry[0] = entry[0][:1]
    result = Thermodynamics(parent).compute_thermodynamic_quantities_at_T(data, 2.0)
    expected = Thermodynamics(parent).compute_thermodynamic_quantities_at_T(original, 2.0)
    assert np.isnan(result[HALL])
    assert np.isnan(Topology(parent).compute_thermal_Hall(data, 2.0)[2])
    for key in result.keys() - {HALL}:
        assert_allclose(result[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("spacing", [0.0, -1.0, np.nan, np.inf])
def test_invalid_layer_spacing_rejected_by_every_entry_point(benchmark, spacing):
    parent, data, _, _ = benchmark
    with pytest.raises(ValueError, match="layer_spacing_m"):
        Topology(parent).compute_thermal_Hall(data, 2.0, layer_spacing_m=spacing)
    with pytest.raises(ValueError, match="layer_spacing_m"):
        Thermodynamics(parent).compute_thermodynamic_quantities_at_T(
            data, 2.0, layer_spacing_m=spacing
        )
    with pytest.raises(ValueError, match="layer_spacing_m"):
        Thermodynamics(parent).get_thermodynamic_quantities(data, layer_spacing_m=spacing)


@pytest.mark.parametrize("temperature", [-1.0, np.nan, np.inf])
def test_invalid_temperature_rejected_consistently(benchmark, temperature):
    parent, data, _, _ = benchmark
    with pytest.raises(ValueError, match="Temperature"):
        Topology(parent).compute_thermal_Hall(data, temperature)
    with pytest.raises(ValueError, match="Temperature"):
        Thermodynamics(parent).compute_thermodynamic_quantities_at_T(data, temperature)


@pytest.mark.parametrize("cell", [[[1, 0], [2, 0]], [[np.nan, 0], [0, 1]], [[1, 0, 0]]])
def test_malformed_geometry_rejected(benchmark, cell):
    original_parent, data, _, _ = benchmark
    parent = copy.deepcopy(original_parent)
    parent.system.lattice_vectors = cell
    with pytest.raises(ValueError, match="magnetic"):
        Topology(parent).compute_thermal_Hall(data, 2.0)
    with pytest.raises(ValueError, match="magnetic"):
        Thermodynamics(parent).compute_thermodynamic_quantities_at_T(data, 2.0)


def test_legacy_parent_uses_actual_lattice_setting(benchmark):
    parent, data, _, _ = benchmark
    legacy_parent = SimpleNamespace(
        Ns=2, bz_data=parent.bz_data,
        lattice_bz_settings=(parent.system.lattice_vectors, "simple"),
    )
    assert_allclose(hall_values(legacy_parent, data, 2.0),
                    hall_values(parent, data, 2.0), rtol=0, atol=0)


def test_empty_topology_integrals_are_unavailable(benchmark):
    parent, _, _, _ = benchmark
    curvature, chern, hall = Topology(parent).compute_thermal_Hall({}, 2.0)
    assert curvature.shape == (0, 2)
    assert np.isnan(chern).all()
    assert np.isnan(hall)


def test_adding_a_zero_curvature_band_does_not_dilute_hall(benchmark):
    """A decoupled flat third band contributes zero, not an extra 1/Ns."""
    from spintoolkit.methods.lswt.diagonalization import Diagonalizer

    parent, data, _, _ = benchmark
    # Retain the two dispersive bands and append an isolated 2 meV oscillator.
    blocks = np.array([entry[0] for entry in data.values()])
    expanded = np.zeros((len(data), 3, 6, 6), dtype=complex)
    indices = [0, 1, 3, 4]
    for i, target_i in enumerate(indices):
        for j, target_j in enumerate(indices):
            expanded[:, :, target_i, target_j] = blocks[:, :, i, j]
    expanded[:, 0, 2, 2] = 2.0
    expanded[:, 0, 5, 5] = 2.0
    extended_data, _ = Diagonalizer.get_K_data(
        np.zeros((len(data), 2)), expanded[:, 0], "No",
        k_indices=list(data), partial_derivative_Hk=[expanded[:, 1], expanded[:, 2]],
    )
    extended_parent = SimpleNamespace(Ns=3, bz_data=parent.bz_data, system=parent.system)
    assert_allclose(hall_values(extended_parent, extended_data, 2.0),
                    hall_values(parent, data, 2.0), rtol=1e-12, atol=0)
