"""Finite-temperature checks for stable quadratic bosons (meV and Kelvin).

Fock-space references below describe harmonic bosons, not finite-spin ED or
the validity of LSWT at high occupation. No Goldstone regularization is used.
"""

from decimal import Decimal, localcontext

import numpy as np
from numpy.testing import assert_allclose
import pytest

from lswt import LSWTSolver, SpinSystem
from lswt.definitions import K_BOLTZMANN_MEV as KB
from lswt.observables.thermodynamics import Thermodynamics
from lswt.methods.spin_wave.energy import EnergyFunction
from lswt.methods.spin_wave.hamiltonian import (
    LSWTHamiltonian, bosonic_free_energy, log_1_m_exp,
)


KPOINTS = np.array([[0, 0], [0.37, -0.29], [-0.37, 0.29]])


def _isolated(copies=1):
    fields = np.tile([0.2, 0.45], copies)
    system = SpinSystem(lattice_vectors=[[copies, 0], [0, 1]])
    for i, field in enumerate(fields):
        system.add_site(str(i), [i * 0.3, 0], spin=0.5, angles=[0, 0],
                        magnetic_field=[0, 0, field])
    return system, fields


def _hamiltonian(system):
    data = system.to_legacy_dict("simple")
    return LSWTHamiltonian(data["Spin info"], data["Couplings"])


@pytest.mark.parametrize("scalar", [False, True])
def test_free_energy_log_matches_high_precision_partition_function(scalar):
    """Exercise small, ordinary and exponentially small thermal corrections."""
    x = np.array(0.05) if scalar else np.array([1e-14, 1e-7, 0.05, 0.7, 2, 40])
    with localcontext() as context:
        context.prec = 80
        reference = np.array([
            float((1 - (-Decimal(str(value))).exp()).ln())
            for value in x.flat
        ]).reshape(x.shape)
    temperature = 1.2
    assert_allclose(log_1_m_exp(x * KB * temperature, temperature),
                    KB * temperature * reference, rtol=2e-14, atol=0)


@pytest.mark.parametrize("temperature", [0, 0.4, 1.2])
@pytest.mark.parametrize("copies", [1, 2])
def test_thermal_energies_are_scalar_band_sums_with_per_spin_normalization(temperature, copies):
    system, fields = _isolated(copies)
    ham = _hamiltonian(system)
    if temperature == 0:
        expected_u = expected_f = 0
    else:
        x = fields / (KB * temperature)
        expected_u = np.sum(fields / np.expm1(x))
        expected_f = KB * temperature * np.sum(np.log1p(-np.exp(-x)))
    u, _ = ham.compute_quantum_energy(KPOINTS, T=temperature, reg_type="No")
    f, _ = ham.compute_quantum_free_energy(KPOINTS, T=temperature, reg_type="No")
    assert np.ndim(u) == np.ndim(f) == 0
    assert_allclose(u, len(KPOINTS) * expected_u, atol=1e-14, rtol=1e-13)
    assert_allclose(f, len(KPOINTS) * expected_f, atol=1e-14, rtol=1e-13)
    energy = EnergyFunction(system.to_legacy_dict("simple"), N=3)
    assert_allclose(energy.quantum_free_energy_density_func(
        system.get_angles_flat(), reg_type="No", Temperature=temperature),
        expected_f / len(fields), atol=1e-14, rtol=1e-13)


def test_solver_uses_requested_temperature_and_resets_between_runs():
    system, fields = _isolated()
    solver = LSWTSolver(system, bz_type="simple")
    for temperature in [1.2, 0, 0.4]:
        occupation = (np.zeros_like(fields) if temperature == 0 else
                      1 / np.expm1(fields / (KB * temperature)))
        result = solver.solve(N=3, regularization="No", temperature=temperature)
        assert_allclose(list(result.data["boson_numbers"].values()), occupation,
                        atol=1e-14, rtol=1e-13)
        assert_allclose(result.data["average_boson_number"], occupation.mean(), atol=1e-14)
        # This field remains the T=0 energy, not U(T) or F(T).
        assert_allclose(result.ground_state_energy, -0.5 * fields.mean(), atol=1e-14)
        thermodynamics = Thermodynamics(solver)
        assert_allclose(thermodynamics.compute_internal_energy(
            result.data["k_data"], Temperature=temperature),
            np.mean(fields * occupation), atol=1e-14, rtol=1e-13)
    # The existing diagnostic entry point must also honor its temperature.
    solver.diagnosing_lswt(bz_type="simple", N=3, regularization="No", temperature=1.2)
    assert_allclose(list(solver.msl_average_boson_number.values()),
                    1 / np.expm1(fields / (KB * 1.2)), atol=1e-14)


def _pair_model(phase):
    h1, h2, g = 0.7, 1.1, 0.18 * np.exp(1j * phase)
    s1, s2 = 0.5, 1.0
    j = g / np.sqrt(s1 * s2)
    system = SpinSystem(lattice_vectors=[[1, 0], [0, 1]])
    system.add_site("A", [0, 0], spin=s1, angles=[0, 0], magnetic_field=[0, 0, h1])
    system.add_site("B", [0.35, 0.2], spin=s2, angles=[0, 0], magnetic_field=[0, 0, h2])
    system.add_coupling("A", "B", [[j.real, j.imag, 0], [j.imag, -j.real, 0], [0, 0, 0]],
                        displacement=[0.35, 0.2])
    return system, h1, h2, g


def _fock_thermal(h1, h2, g, temperature, cutoff):
    """Construct and thermally average the original two-mode boson matrix."""
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    first, second = np.kron(a, np.eye(cutoff)), np.kron(np.eye(cutoff), a)
    n1, n2 = first.T @ first, second.T @ second
    pair = first.T @ second.T
    matrix = h1 * n1 + h2 * n2 + g * pair + g.conjugate() * pair.T
    values, vectors = np.linalg.eigh(matrix)
    weights = np.exp(-(values - values[0]) / (KB * temperature))
    partition = weights.sum()
    probabilities = weights / partition
    u = values @ probabilities
    f = values[0] - KB * temperature * np.log(partition)
    occupation = [np.diag(n) @ (abs(vectors)**2 @ probabilities) for n in [n1, n2]]
    return np.array([u, f, *occupation])


@pytest.mark.parametrize("phase", [0, 0.61])
def test_paired_bosons_match_converged_thermal_fock_space(phase):
    system, h1, h2, g = _pair_model(phase)
    temperature = 2.0
    coarse = _fock_thermal(h1, h2, g, temperature, cutoff=10)
    fine = _fock_thermal(h1, h2, g, temperature, cutoff=14)
    assert_allclose(coarse, fine, atol=1e-10, rtol=0)
    ham = _hamiltonian(system)
    u, _ = ham.compute_quantum_energy(KPOINTS, T=temperature, reg_type="No")
    f, _ = ham.compute_quantum_free_energy(KPOINTS, T=temperature, reg_type="No")
    result = LSWTSolver(system, bz_type="simple").solve(
        N=3, regularization="No", temperature=temperature)
    actual = [u / len(KPOINTS), f / len(KPOINTS), *result.data["boson_numbers"].values()]
    assert_allclose(actual, fine, atol=1e-11, rtol=0)


@pytest.mark.parametrize("temperature", [0.8, 1.2])
def test_internal_energy_equals_free_energy_minus_temperature_derivative(temperature):
    system, _ = _isolated()
    ham = _hamiltonian(system)
    step = temperature * 1e-4
    def free(t):
        return ham.compute_quantum_free_energy(KPOINTS, T=t, reg_type="No")[0]
    derivative = (free(temperature + step) - free(temperature - step)) / (2 * step)
    u, _ = ham.compute_quantum_energy(KPOINTS, T=temperature, reg_type="No")
    assert_allclose(u, free(temperature) - temperature * derivative, atol=1e-9, rtol=0)


def test_free_energy_does_not_discard_zero_mode_divergence():
    # A zero-frequency unconstrained oscillator has a divergent partition sum.
    assert np.isneginf(bosonic_free_energy([0, 0.2], 1.2))
    assert np.isfinite(bosonic_free_energy([1e-18, 0.2], 1.2))
    assert_allclose(log_1_m_exp([0.2, 0.45], 0), 0, atol=0)


@pytest.mark.parametrize("temperature", [-1, np.nan, np.inf])
def test_thermal_entry_points_reject_invalid_temperature(temperature):
    system, _ = _isolated()
    ham = _hamiltonian(system)
    with pytest.raises(ValueError, match="[Tt]emperature"):
        ham.compute_quantum_energy(KPOINTS, T=temperature, reg_type="No")
    with pytest.raises(ValueError, match="[Tt]emperature"):
        ham.compute_quantum_free_energy(KPOINTS, T=temperature, reg_type="No")
    with pytest.raises(ValueError, match="[Tt]emperature"):
        LSWTSolver(system, bz_type="simple").solve(temperature=temperature)
