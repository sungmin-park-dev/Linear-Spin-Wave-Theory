"""Expansion of model terms on a finite torus (D23).

Duplicated bonds on small tori are summed; bonds that fold onto a single site
are rejected by every method (classical energy and ED here, LSWT comparisons
in test_ed.py through the same expansion).
"""

import numpy as np
import pytest

from spintoolkit.methods.classical import classical_energy
from spintoolkit.methods.ed import solve_ed
from spintoolkit.models import neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.system.cluster import ClusterError, allowed_momenta, expand_on_torus
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry


def torus(L):
    return CalculationGeometry.finite_torus(L)


def test_every_term_is_placed_once_per_cell():
    model = triangular_heisenberg(J=1.0)
    cluster = expand_on_torus(model, torus([[3, 0], [0, 3]]))
    assert cluster.num_sites == 9
    assert len(cluster.source) == 9 * len(model.terms_of_kind("bilinear"))
    assert np.all(cluster.source != cluster.target)


def test_small_torus_sums_duplicated_bonds():
    """On a 2 x 2 square torus the +x and -x neighbours coincide: two bonds per pair."""
    cluster = expand_on_torus(square_heisenberg(J=1.0), torus([[2, 0], [0, 2]]))
    pairs = {}
    for i, j in zip(cluster.source, cluster.target):
        key = tuple(sorted((int(i), int(j))))
        pairs[key] = pairs.get(key, 0) + 1
    assert len(pairs) == 4 and set(pairs.values()) == {2}


@pytest.mark.parametrize("L", [[[1, 0], [0, 1]], [[1, 0], [0, 3]], [[3, 0], [0, 1]]])
def test_bonds_folding_onto_one_site_are_rejected_by_every_method(L):
    model = square_heisenberg(J=1.0)
    with pytest.raises(ClusterError, match="self-interaction"):
        expand_on_torus(model, torus(L))
    with pytest.raises(ClusterError, match="self-interaction"):
        classical_energy(model, polarized_state(model), None, torus(L))
    with pytest.raises(ClusterError, match="self-interaction"):
        solve_ed(model, torus(L))


@pytest.mark.parametrize("model, state, L", [
    (square_heisenberg(J=1.0), neel_state, [[2, 0], [0, 2]]),
    (square_heisenberg(J=1.0), neel_state, [[4, 2], [0, 2]]),
    (triangular_heisenberg(J=1.0), state_120, [[3, 0], [0, 3]]),
    (triangular_heisenberg(J=1.0), state_120, [[1, 1], [-1, 2]]),
])
def test_torus_classical_energy_equals_thermodynamic_limit(model, state, L):
    s = state(model)
    field = ExternalConditions(field=(0, 0, 0.3))
    assert classical_energy(model, s, field, torus(L)) == pytest.approx(
        classical_energy(model, s, field), abs=1e-15)


def test_torus_must_be_commensurate_with_the_state():
    model = triangular_heisenberg(J=1.0)
    with pytest.raises(ValueError, match="not a lattice vector"):
        classical_energy(model, state_120(model), None, torus([[2, 0], [0, 2]]))


@pytest.mark.parametrize("L", [[[4, 0], [0, 3]], [[2, 1], [-1, 3]], [[3, 1], [0, 2]]])
def test_allowed_momenta_are_the_torus_reciprocal_lattice(L):
    model = triangular_heisenberg(J=1.0)
    q, k = allowed_momenta(model, torus(L))
    assert len(q) == abs(round(np.linalg.det(L)))
    assert len({tuple(np.round(x, 12)) for x in q}) == len(q)
    periods = np.asarray(L) @ model.lattice
    phases = np.exp(1j * k @ periods.T)
    np.testing.assert_allclose(phases, 1.0, atol=1e-12)
