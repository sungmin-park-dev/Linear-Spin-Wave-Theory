"""Regression for the archived B/B-dagger swap (issue 260802).

The reference extracts quadratic HP coefficients from finite spin-operator
matrix elements. It does not call the production rotation/circular-basis or
coupling routines. We use a_r = sum_k exp(-i k.r) a_k and the Nambu column
(a_k, a^dagger_-k), so pair creation belongs in the upper-right block.
"""

from importlib import import_module
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial.transform import Rotation

from spintoolkit.methods.lswt.hamiltonian import LSWTHamiltonian


KPOINTS = np.array([[0.0, 0.0], [0.37, -0.29], [-0.81, 0.46], [1.13, 0.67]])
ATOL = 5e-14


def _spin_matrices(spin, angles):
    """Return lab-frame spin matrices in the local |S,m> basis, m descending."""
    m = np.arange(spin, -spin - 1, -1)
    raising = np.zeros((len(m), len(m)), dtype=complex)
    for col in range(1, len(m)):
        raising[col - 1, col] = np.sqrt(spin * (spin + 1) - m[col] * (m[col] + 1))
    lowering = raising.conj().T
    local = np.array([(raising + lowering) / 2,
                      (raising - lowering) / (2j), np.diag(m)])
    theta, phi = angles
    rotation = (Rotation.from_rotvec([0, 0, phi]).as_matrix()
                @ Rotation.from_rotvec([0, theta, 0]).as_matrix())
    return np.einsum("ab,bij->aij", rotation, local)


@pytest.fixture(params=["real_pairing", "complex_pairing"])
def model(request):
    """Synthetic bonds exercise unequal spins and translated same-sublattice pairs.

    These fixtures test operator assembly, not a material ground state.
    A strong field along each local z axis keeps their quadratic matrices positive.
    """
    complex_pairing = request.param == "complex_pairing"
    spin_info = {}
    for name, spin, angles in zip(
        ["A", "B", "C"], [0.5, 1.0, 1.5],
        [[0.35, 0.2], [0.8, -0.7], [-0.55, 1.1]],
    ):
        if not complex_pairing:
            angles = [0.0, 0.0]
        theta, phi = angles
        direction = np.array([np.sin(theta) * np.cos(phi),
                              np.sin(theta) * np.sin(phi), np.cos(theta)])
        spin_info[name] = {"Spin": spin, "Angles": angles,
                           "Magnetic Field": 2.0 * direction}
    couplings = []
    for index, (left, right, delta) in enumerate([
        ("A", "B", [1.0, 0.0]), ("B", "C", [-0.5, 0.8]),
        ("C", "A", [-0.5, -0.8]), ("A", "B", [0.2, 1.1]),
        ("C", "C", [0.4, 1.2]),
    ]):
        exchange = np.diag([0.075, 0.042, -0.031])
        if complex_pairing:
            # Real Cartesian exchange, including symmetric SOC and antisymmetric terms.
            exchange += (index + 1) * np.array([
                [0.0, 0.006, -0.003], [0.002, 0.0, 0.004],
                [-0.001, 0.007, 0.0],
            ])
        couplings.append({"SpinI": left, "SpinJ": right,
                          "Exchange Matrix": exchange,
                          "Displacement": np.array(delta)})
    return spin_info, couplings, complex_pairing


def _spin_operator_reference(spin_info, couplings, kpoints):
    """Extract vacuum/one-magnon/pair matrix elements before Fourier transformation.

    For distinct physical sites, <1_i 1_j|H|0> is the HP pair-creation
    coefficient and <1_i|H|1_j> is the normal hopping coefficient. Diagonal
    one-magnon energies minus the vacuum energy give the local quadratic terms.
    A translated same-sublattice bond still joins two distinct physical spins.
    """
    ns = len(spin_info)
    indices = {name: index for index, name in enumerate(spin_info)}
    operators = {name: _spin_matrices(site["Spin"], site["Angles"])
                 for name, site in spin_info.items()}
    normal = np.zeros((2, len(kpoints), ns, ns), dtype=complex)
    pairing = np.zeros((len(kpoints), ns, ns), dtype=complex)
    for name, site in spin_info.items():
        field_term = -np.einsum("a,aij->ij", site["Magnetic Field"], operators[name])
        index = indices[name]
        normal[:, :, index, index] = field_term[1, 1] - field_term[0, 0]
    for bond in couplings:
        left, right = bond["SpinI"], bond["SpinJ"]
        i, j = indices[left], indices[right]
        size_right = operators[right].shape[-1]
        spin_hamiltonian = sum(
            bond["Exchange Matrix"][a, b] * np.kron(operators[left][a], operators[right][b])
            for a in range(3) for b in range(3)
        )
        vacuum = spin_hamiltonian[0, 0]
        normal[:, :, i, i] += spin_hamiltonian[size_right, size_right] - vacuum
        normal[:, :, j, j] += spin_hamiltonian[1, 1] - vacuum
        hopping = spin_hamiltonian[size_right, 1]
        creation = spin_hamiltonian[size_right + 1, 0]
        phase = np.exp(-1j * (kpoints @ bond["Displacement"]))
        for block, factor in zip(normal, [phase, phase.conj()]):
            block[:, i, j] += hopping * factor
            block[:, j, i] += (hopping * factor).conj()
        pairing[:, i, j] += creation * phase
        pairing[:, j, i] += creation * phase.conj()
    return np.concatenate([
        np.concatenate([normal[0], pairing], axis=2),
        np.concatenate([pairing.conj().transpose(0, 2, 1), normal[1].conj()], axis=2),
    ], axis=1)


def test_hamiltonian_matches_spin_operator_reference(model):
    """Complex pair creation must have the phase fixed by the spin Hamiltonian."""
    spin_info, couplings, _ = model
    momenta = np.concatenate([KPOINTS, -KPOINTS])
    actual, _ = LSWTHamiltonian(spin_info, couplings).Quadratic_Bose_Hamiltonian(momenta)
    expected = _spin_operator_reference(spin_info, couplings, momenta)
    assert_allclose(actual, expected, atol=ATOL, rtol=1e-13)


@pytest.mark.parametrize("axis", [0, 1], ids=["kx", "ky"])
def test_momentum_derivative_matches_spin_operator_reference(model, axis):
    """Protect the independently implemented Berry-curvature derivatives too."""
    spin_info, couplings, _ = model
    hamiltonian = LSWTHamiltonian(spin_info, couplings)
    hamiltonian.Quadratic_Bose_Hamiltonian(KPOINTS)
    actual = hamiltonian.partial_derivatives_of_Hk(KPOINTS)[axis]
    step = 1e-5
    offset = np.eye(2)[axis] * step
    expected = (_spin_operator_reference(spin_info, couplings, KPOINTS + offset)
                - _spin_operator_reference(spin_info, couplings, KPOINTS - offset)) / (2 * step)
    assert_allclose(actual, expected, atol=1e-9, rtol=1e-8)


def test_legacy_swap_is_detected_only_by_coefficient_reference(model, monkeypatch):
    """Load the preserved legacy implementation and isolate its anomalous-block swap."""
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "legacy"))
    legacy_class = import_module("modules.LinearSpinWaveTheory.lswt_Hamiltonian").LSWT_HAMILTONIAN
    spin_info, couplings, complex_pairing = model
    legacy = legacy_class(spin_info, couplings)
    old, _ = legacy.Quadratic_Bose_Hamiltonian(KPOINTS)
    old_minus_k, _ = legacy.Quadratic_Bose_Hamiltonian(-KPOINTS)
    expected = _spin_operator_reference(spin_info, couplings, KPOINTS)
    ns = len(spin_info)

    # Both generic constraints pass even for the incorrect complex coefficients.
    assert_allclose(old, old.conj().transpose(0, 2, 1), atol=ATOL, rtol=0)
    assert_allclose(old_minus_k[:, :ns, ns:], old[:, :ns, ns:].transpose(0, 2, 1),
                    atol=ATOL, rtol=0)
    assert_allclose(old[:, :ns, :ns], expected[:, :ns, :ns], atol=ATOL, rtol=1e-13)

    restored = old.copy()
    restored[:, :ns, ns:] = old[:, ns:, :ns]
    restored[:, ns:, :ns] = old[:, :ns, ns:]
    assert_allclose(restored, expected, atol=ATOL, rtol=1e-13)
    if complex_pairing:
        assert np.max(np.abs(old - expected)) > 1e-3
        with pytest.raises(AssertionError):
            assert_allclose(old, expected, atol=ATOL, rtol=1e-13)
        # The fixture also exposes an observable spectral difference without a shift.
        assert np.linalg.eigvalsh(expected).min() > 0
        assert np.linalg.eigvalsh(old).min() > 0
        metric = np.diag([1.0] * ns + [-1.0] * ns)
        old_spectrum = np.linalg.eigvals(metric @ old)
        spectrum = np.linalg.eigvals(metric @ expected)
        assert np.max(np.abs(spectrum.imag)) < 1e-12
        assert np.max(np.abs(old_spectrum.imag)) < 1e-12
        assert np.max(np.abs(np.sort(old_spectrum.real) - np.sort(spectrum.real))) > 1e-5
    else:
        assert_allclose(old, expected, atol=ATOL, rtol=1e-13)
