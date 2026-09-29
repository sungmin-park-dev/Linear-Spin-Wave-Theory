"""Check LSWT energy constants against analytic spectra and bosonic Fock space.

Each coupling represents one physical bond per magnetic unit cell. Energies
returned by LSWTSolver are per spin; LSWTHamiltonian returns an unnormalized
k-point sum. These tests use positive quadratic Hamiltonians, not unstable
states repaired by a substantial MAGSWT shift.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spintoolkit import LSWTSolver, SpinSystem
from spintoolkit.methods.lswt.energy import EnergyFunction
from spintoolkit.methods.lswt.hamiltonian import LSWTHamiltonian


def _energy_function(system, n=3):
    return EnergyFunction(system.to_legacy_dict("simple"), N=n)


def _hamiltonian(system):
    data = system.to_legacy_dict("simple")
    return LSWTHamiltonian(data["Spin info"], data["Couplings"])


@pytest.mark.parametrize("spins,fields", [([0.5], [0.8]), ([0.5, 1.0], [0.7, 1.3])],
                         ids=["one_spin", "unequal_spins"])
@pytest.mark.parametrize("regularization", ["No", "MAGSWT"])
def test_isolated_spins_have_no_zero_point_energy(spins, fields, regularization):
    """A polarized product state has exact E/N = -mean(h_i S_i), not Ecl+mean(h_i)."""
    system = SpinSystem(lattice_vectors=[[1, 0], [0, 1]])
    for index, (spin, field) in enumerate(zip(spins, fields)):
        system.add_site(str(index), [index / len(spins), 0], spin=spin,
                        angles=[0, 0], magnetic_field=[0, 0, field])
    expected = -np.mean(np.array(spins) * fields)
    energy = _energy_function(system)
    assert_allclose(energy.classical_energy_density_func(system.get_angles_flat()), expected)
    assert_allclose(energy.quantum_energy_density_func(system.get_angles_flat(),
                                                      reg_type=regularization), 0, atol=1e-14)
    result = LSWTSolver(system, bz_type="simple").solve(N=3, regularization=regularization)
    assert_allclose(result.ground_state_energy, expected, atol=1e-13, rtol=0)


@pytest.mark.parametrize("spin", [0.5, 1.5])
def test_same_sublattice_ferromagnet_counts_field_once_and_bonds_twice(spin):
    """Incident exchange bonds give 2JS per direction; the Zeeman coefficient is h."""
    field, jx, jy = 0.4, -0.2, -0.11
    system = SpinSystem(lattice_vectors=[[1, 0], [0, 1]])
    system.add_site("A", [0, 0], spin=spin, angles=[0, 0], magnetic_field=[0, 0, field])
    system.add_coupling("A", "A", jx * np.eye(3), displacement=[1, 0])
    system.add_coupling("A", "A", jy * np.eye(3), displacement=[0, 1])
    momenta = np.array([[0, 0], [0.37, -0.29], [-0.37, 0.29], [np.pi, np.pi]])
    ham = _hamiltonian(system)
    matrix, _ = ham.Quadratic_Bose_Hamiltonian(momenta)
    dispersion = (field + 2 * spin * jx * (np.cos(momenta[:, 0]) - 1)
                  + 2 * spin * jy * (np.cos(momenta[:, 1]) - 1))
    assert_allclose(matrix[:, 0, 0], dispersion, atol=1e-14)
    assert_allclose(matrix[:, 0, 1], 0, atol=1e-14)
    correction, _ = ham.compute_quantum_energy(momenta, reg_type="No")
    assert_allclose(correction, 0, atol=1e-13)
    expected = (jx + jy) * spin**2 - field * spin
    energy = _energy_function(system)
    assert_allclose(energy.classical_energy_density_func(system.get_angles_flat()), expected)
    result = LSWTSolver(system, bz_type="simple").solve(N=4, regularization="No")
    assert_allclose(result.ground_state_energy, expected, atol=1e-13, rtol=0)


def _paired_dimers(phase, copies=1):
    """Real spin exchange produces H2 = h1*n1+h2*n2+g*a1†*a2†+g* *a2*a1."""
    h1, h2 = 0.7, 1.1
    s1, s2 = 0.5, 1.0
    g = 0.18 * np.exp(1j * phase)
    j = g / np.sqrt(s1 * s2)
    exchange = np.array([[j.real, j.imag, 0], [j.imag, -j.real, 0], [0, 0, 0]])
    system = SpinSystem(lattice_vectors=[[copies, 0], [0, 1]])
    for index in range(copies):
        a, b = f"A{index}", f"B{index}"
        system.add_site(a, [index, 0], spin=s1, angles=[0, 0], magnetic_field=[0, 0, h1])
        system.add_site(b, [index + 0.35, 0.2], spin=s2, angles=[0, 0],
                        magnetic_field=[0, 0, h2])
        system.add_coupling(a, b, exchange, displacement=[0.35, 0.2])
    classical_per_spin = -(h1 * s1 + h2 * s2) / 2
    correction_per_cell = (np.sqrt((h1 + h2)**2 - 4 * abs(g)**2) - h1 - h2) / 2
    return system, h1, h2, g, classical_per_spin, correction_per_cell


def _fock_ground_energy(h1, h2, g, cutoff):
    """Diagonalize the quadratic BOSON Hamiltonian, not the full interacting spin model."""
    annihilation = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    first = np.kron(annihilation, np.eye(cutoff))
    second = np.kron(np.eye(cutoff), annihilation)
    creation = first.T @ second.T
    matrix = h1 * first.T @ first + h2 * second.T @ second
    matrix = matrix + g * creation + g.conjugate() * creation.T
    return np.linalg.eigvalsh(matrix)[0]


@pytest.mark.parametrize("phase", [0.0, 0.61], ids=["real_pair", "complex_pair"])
def test_pair_creation_energy_matches_converged_bosonic_fock_space(phase):
    """Nonzero vacuum energy fixes the half factors and trace subtraction independently."""
    system, h1, h2, g, classical, correction = _paired_dimers(phase)
    coarse = _fock_ground_energy(h1, h2, g, cutoff=7)
    fine = _fock_ground_energy(h1, h2, g, cutoff=11)
    assert_allclose(coarse, fine, atol=1e-10, rtol=0)
    assert_allclose(fine, correction, atol=1e-12, rtol=0)

    energy = _energy_function(system)
    angles = system.get_angles_flat()
    assert_allclose(energy.classical_energy_density_func(angles), classical, atol=1e-13)
    assert_allclose(energy.quantum_energy_density_func(angles, reg_type="No"),
                    fine / 2, atol=1e-13)
    assert_allclose(energy.quantum_free_energy_density_func(angles, reg_type="No", Temperature=0),
                    fine / 2, atol=1e-13)
    result = LSWTSolver(system, bz_type="simple").solve(N=3, regularization="No")
    assert_allclose(result.ground_state_energy, classical + fine / 2, atol=1e-13, rtol=0)


def test_energy_per_spin_is_invariant_under_unit_cell_replication():
    """Repeating independent dimers changes Ns and the number of bands, not energy density."""
    energies = []
    for copies in [1, 2]:
        system, _, _, _, classical, correction = _paired_dimers(0.61, copies=copies)
        result = LSWTSolver(system, bz_type="simple").solve(N=3, regularization="No")
        assert_allclose(result.ground_state_energy, classical + correction / 2, atol=1e-13, rtol=0)
        energies.append(result.ground_state_energy)
    assert_allclose(energies[0], energies[1], atol=1e-13)


def _dm_chain():
    spin, field, j, dm = 0.5, 0.8, -0.17, 0.037
    system = SpinSystem(lattice_vectors=[[1, 0], [0, 1]])
    system.add_site("A", [0, 0], spin=spin, angles=[0, 0], magnetic_field=[0, 0, field])
    exchange = j * np.eye(3) + np.array([[0, dm, 0], [-dm, 0, 0], [0, 0, 0]])
    system.add_coupling("A", "A", exchange, displacement=[1, 0])
    return _hamiltonian(system), spin, field, j, dm


def test_full_bdg_trace_identity_requires_momentum_pairing_for_nonreciprocal_bands():
    """Tr H(k)=2 Tr A(k) need not hold pointwise, even though A(k) is Hermitian."""
    ham, spin, field, j, dm = _dm_chain()
    kx = np.array([0.37, -0.37, 1.13, -1.13])
    momenta = np.column_stack([kx, np.zeros_like(kx)])
    matrix, _ = ham.Quadratic_Bose_Hamiltonian(momenta)
    # +D(S_i^x S_j^y - S_i^y S_j^x) gives hopping S(J-iD).
    dispersion = field + 2 * spin * j * (np.cos(kx) - 1) - 2 * spin * dm * np.sin(kx)
    assert_allclose(matrix[:, 0, 0], dispersion, atol=1e-14)
    full_trace = np.trace(matrix, axis1=1, axis2=2).real
    normal_trace = matrix[:, 0, 0].real
    assert np.max(np.abs(full_trace - 2 * normal_trace)) > 1e-3
    assert_allclose(full_trace.sum(), 2 * normal_trace.sum(), atol=1e-13)
    correction, _ = ham.compute_quantum_energy(momenta, reg_type="No")
    assert_allclose(correction, 0, atol=1e-13)


def test_same_sublattice_hopping_trace_uses_complete_fourier_sum():
    """A complete periodic grid removes nonzero-translation hopping from the mean trace."""
    ham, spin, field, j, _ = _dm_chain()
    kx = 2 * np.pi * np.arange(8) / 8
    momenta = np.column_stack([kx, np.zeros_like(kx)])
    matrix, _ = ham.Quadratic_Bose_Hamiltonian(momenta)
    onsite_coefficient = field - 2 * spin * j
    assert_allclose(matrix[:, 0, 0].mean(), onsite_coefficient, atol=1e-14)
    # An arbitrary subset is not the complete character sum used in the derivation.
    assert abs(matrix[:2, 0, 0].mean() - onsite_coefficient) > 1e-3
