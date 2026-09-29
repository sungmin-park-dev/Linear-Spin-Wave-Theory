"""Per-spin thermodynamics and per-sublattice occupation of gapped bosons.

The reference is the analytic two-mode partition function. Repeating uncoupled
dimers changes the magnetic cell, not any intensive observable. Thermal Hall
normalization and invalid-mode physics are outside this regression's scope.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spintoolkit import LSWTSolver, SpinSystem
from spintoolkit.definitions import K_BOLTZMANN_MEV as KB
from spintoolkit.observables.thermodynamics import Thermodynamics


SCALARS = ("Internal Energy Density", "Entropy Density", "Specific Heat Density",
           "Total Boson Number")


def _model(copies=1, paired=False):
    h1, h2 = 0.2, 0.45
    g = 0.035 * np.exp(0.43j) if paired else 0j
    j = g / np.sqrt(0.5 * 1.0)
    system = SpinSystem(lattice_vectors=[[copies, 0], [0, 1]])
    for i in range(copies):
        a, b = f"A{i}", f"B{i}"
        system.add_site(a, [i, 0], spin=0.5, angles=[0, 0], magnetic_field=[0, 0, h1])
        system.add_site(b, [i + 0.3, 0.2], spin=1, angles=[0, 0],
                        magnetic_field=[0, 0, h2])
        if paired:
            system.add_coupling(a, b, [[j.real, j.imag, 0], [j.imag, -j.real, 0], [0, 0, 0]],
                                displacement=[0.3, 0.2])
    solver = LSWTSolver(system, bz_type="simple")
    result = solver.solve(N=3, regularization="No")
    return solver, result.data["k_data"], h1, h2, g


def _reference(h1, h2, g, temperature, copies=1):
    root = np.sqrt((h1 + h2)**2 - 4 * abs(g)**2)
    modes = (root + np.array([h1 - h2, h2 - h1])) / 2
    vacuum = (root - h1 - h2) / 2
    n = np.zeros(2)
    entropy = specific_heat = thermal_free = 0
    if temperature > 0:
        x = modes / (KB * temperature)
        n = 1 / np.expm1(x)
        entropy = KB * np.mean((1 + n) * np.log1p(n) - n * np.log(n))
        specific_heat = KB * np.mean(x**2 * n * (1 + n))
        thermal_free = KB * temperature * np.mean(np.log1p(-np.exp(-x)))
    # a1 = cosh(r)*b1 + phase*sinh(r)*b2† (and the partner relation).
    v_squared = ((h1 + h2) / root - 1) / 2
    sites = (1 + v_squared) * n + v_squared * (n[::-1] + 1)
    return {
        "Internal Energy Density": vacuum / 2 + np.mean(modes * n),
        "Entropy Density": entropy,
        "Specific Heat Density": specific_heat,
        "Total Boson Number": sites.mean(),
        "Sublattice Boson Numbers": np.tile(sites, copies),
        "Free Energy Density": vacuum / 2 + thermal_free,
    }


@pytest.mark.parametrize("copies", [1, 2])
@pytest.mark.parametrize("temperature", [0, 1.2])
@pytest.mark.parametrize("paired", [False, True])
def test_individual_observables_match_partition_function_per_spin(copies, temperature, paired):
    solver, k_data, h1, h2, g = _model(copies, paired)
    reference = _reference(h1, h2, g, temperature, copies)
    thermodynamics = Thermodynamics(solver)
    actual = {
        "Internal Energy Density": thermodynamics.compute_internal_energy(k_data, temperature),
        "Entropy Density": thermodynamics.compute_entropy_density(k_data, temperature),
        "Specific Heat Density": thermodynamics.compute_specific_heat(k_data, temperature),
    }
    for key, value in actual.items():
        assert_allclose(value, reference[key], atol=1e-13, rtol=0, err_msg=key)
    sites, mean = thermodynamics.compute_boson_numbers(k_data, temperature)
    assert_allclose(sites, reference["Sublattice Boson Numbers"], atol=1e-13, rtol=0)
    assert_allclose(mean, reference["Total Boson Number"], atol=1e-13, rtol=0)


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("copies", [1, 2])
@pytest.mark.parametrize("temperature", [0, 1.2])
def test_combined_observables_count_sublattices_only_once(attached, copies, temperature):
    solver, k_data, h1, h2, g = _model(copies, paired=True)
    reference = _reference(h1, h2, g, temperature, copies)
    thermodynamics = Thermodynamics(solver if attached else None)
    combined = thermodynamics.compute_thermodynamic_quantities_at_T(k_data, temperature)
    for key in (*SCALARS, "Sublattice Boson Numbers"):
        assert_allclose(combined[key], reference[key], atol=1e-13, rtol=0, err_msg=key)
    assert_allclose(combined["Total Boson Number"],
                    np.mean(combined["Sublattice Boson Numbers"]), atol=1e-14)


@pytest.mark.parametrize("paired", [False, True])
def test_entropy_and_specific_heat_match_free_and_internal_energy_derivatives(paired):
    solver, k_data, _, _, _ = _model(copies=2, paired=paired)
    thermodynamics = Thermodynamics(solver)
    k_points = np.array([[0, 0], [0.37, -0.29], [-0.37, 0.29]])
    temperature, step = 1.2, 1e-4
    def free(t):
        value, _ = solver.Ham.compute_quantum_free_energy(k_points, T=t, reg_type="No")
        return value / (len(k_points) * solver.Ns)
    entropy = thermodynamics.compute_entropy_density(k_data, temperature)
    specific_heat = thermodynamics.compute_specific_heat(k_data, temperature)
    derivative_f = (free(temperature + step) - free(temperature - step)) / (2 * step)
    derivative_u = (thermodynamics.compute_internal_energy(k_data, temperature + step)
                    - thermodynamics.compute_internal_energy(k_data, temperature - step)) / (2 * step)
    assert_allclose(entropy, -derivative_f, atol=1e-9, rtol=0)
    assert_allclose(specific_heat, derivative_u, atol=1e-9, rtol=0)


def test_excluded_samples_use_valid_count_without_extra_sublattice_factor():
    solver, k_data, h1, h2, g = _model(copies=2, paired=True)
    thermodynamics = Thermodynamics(solver)
    # Mark known finite samples invalid to isolate denominator bookkeeping.
    # This does not model an actual Colpa failure or validate mode exclusion.
    flagged = {key: list(entry) for key, entry in k_data.items()}
    for key in list(flagged)[::2]:
        flagged[key][2] = (False, None)
    expected = _reference(h1, h2, g, 1.2, copies=2)
    combined = thermodynamics.compute_thermodynamic_quantities_at_T(flagged, 1.2)
    for key in (*SCALARS, "Sublattice Boson Numbers"):
        assert_allclose(combined[key], expected[key], atol=1e-13, rtol=0, err_msg=key)
    assert_allclose(thermodynamics.compute_entropy_density(flagged, 1.2),
                    expected["Entropy Density"], atol=1e-13)
    assert_allclose(thermodynamics.compute_specific_heat(flagged, 1.2),
                    expected["Specific Heat Density"], atol=1e-13)
    for entry in flagged.values():
        entry[2] = (False, None)
    combined = thermodynamics.compute_thermodynamic_quantities_at_T(flagged, 1.2)
    assert all(np.all(np.isnan(value)) for value in combined.values())


@pytest.mark.parametrize("attached", [False, True])
def test_temperature_sweep_preserves_intensive_values_and_sublattice_rows(attached):
    solver, k_data, h1, h2, g = _model(copies=2, paired=True)
    thermodynamics = Thermodynamics(solver if attached else None)
    temperatures, sweep = thermodynamics.get_thermodynamic_quantities(
        k_data, Temperature_range=(0, 1.2, 0.6))
    assert_allclose(temperatures, [0, 0.6, 1.2], atol=1e-14)
    for index, temperature in enumerate(temperatures):
        expected = _reference(h1, h2, g, temperature, copies=2)
        for key in SCALARS:
            assert_allclose(sweep[key][index], expected[key], atol=1e-13, rtol=0, err_msg=key)
        assert_allclose(sweep["Sublattice Boson Numbers"][:, index],
                        expected["Sublattice Boson Numbers"], atol=1e-13, rtol=0)
