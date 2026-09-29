"""Benchmark Hamiltonian builders and analytic reference configurations."""

import numpy as np
import pytest

from spintoolkit.models import (
    neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg,
)


@pytest.mark.parametrize("builder, bonds", [(square_heisenberg, 2), (triangular_heisenberg, 3)])
def test_builders_record_each_nearest_neighbour_bond_once(builder, bonds):
    model = builder(J=0.8, S=1.0, lattice_constant=2.0)
    bilinear = model.terms_of_kind("bilinear")
    assert model.site_ids == ("A",) and len(bilinear) == bonds
    for term in bilinear:
        (a, n1), (b, n2) = term.participants
        length = np.linalg.norm(model.cartesian_position(b, n2) - model.cartesian_position(a, n1))
        assert length == pytest.approx(2.0)
        np.testing.assert_array_equal(term.coefficient, 0.8 * np.eye(3))
    assert model.metadata["model_id"] == builder.__name__
    assert model.fingerprint() == builder(J=0.8, S=1.0, lattice_constant=2.0).fingerprint()


def test_g_tensor_options():
    assert not square_heisenberg(g=None).terms_of_kind("zeeman")
    g = np.diag([1.0, 1.0, 2.0])
    np.testing.assert_array_equal(
        square_heisenberg(g=g).terms_of_kind("zeeman")[0].coefficient, g)


def test_reference_configurations_put_neighbours_where_expected():
    square = square_heisenberg()
    neel = neel_state(square)
    for term in square.terms_of_kind("bilinear"):
        (_, n1), (_, n2) = term.participants
        for cell in neel.cells:
            shifted = (cell[0] + n2[0] - n1[0], cell[1] + n2[1] - n1[1])
            assert neel.direction("A", cell) @ neel.direction("A", shifted) == pytest.approx(-1)
    triangular = triangular_heisenberg()
    spiral = state_120(triangular)
    assert spiral.num_cells == 3
    for term in triangular.terms_of_kind("bilinear"):
        (_, n1), (_, n2) = term.participants
        for cell in spiral.cells:
            shifted = (cell[0] + n2[0] - n1[0], cell[1] + n2[1] - n1[1])
            assert spiral.direction("A", cell) @ spiral.direction("A", shifted) == pytest.approx(-0.5)


def test_reference_configuration_arguments():
    model = square_heisenberg()
    np.testing.assert_allclose(polarized_state(model, (0, 3, 4)).direction("A"), [0, 0.6, 0.8])
    with pytest.raises(ValueError):
        polarized_state(model, (0, 0, 0))
    with pytest.raises(ValueError):
        state_120(triangular_heisenberg(), plane=((1, 0, 0), (1, 1, 0)))
