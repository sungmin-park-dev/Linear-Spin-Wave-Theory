"""Sparse bond storage of H(k) (D67): memory grows with the bonds, H(k) stays Hermitian.

Each bond contributes at most eight entries (four to P_b, four to Q_b), so
storage is independent of (2 Ns)^2 per bond; the equality with the former dense
assembly was checked to zero difference when the storage changed.
"""

import numpy as np

from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.models import square_heisenberg
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions


def test_bond_storage_is_eight_entries_per_bond_and_hermitian():
    model = square_heisenberg(J=-1.0)
    state = SpinState.from_function(model, [[8, 0], [0, 8]], lambda s, c: np.array([0, 0, 1.0]))
    H = QuadraticBoseHamiltonian(model, state, ExternalConditions(field=(0.1, 0.2, 0.3)))
    bonds = len(H._deltas)
    assert bonds == 2 * 64 and H._bonds.nnz <= 8 * bonds
    k = np.random.default_rng(0).normal(size=(4, 2))
    h = H.at(k)
    np.testing.assert_allclose(h, np.conj(np.swapaxes(h, 1, 2)), atol=1e-14)
    dx, _ = H.derivatives_at(k)
    step = 1e-6 * np.array([1.0, 0.0])
    np.testing.assert_allclose(dx, (H.at(k + step) - H.at(k - step)) / 2e-6, atol=1e-8)
