"""SpinState construction, supercell bookkeeping and validation."""

import numpy as np
import pytest

from spintoolkit.models import (
    neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg,
)
from spintoolkit.states.spin_state import (
    SpinState, SpinStateError, reduce_cell, supercell_cells, validate_spin_state,
)
from spintoolkit.system.geometry import CalculationGeometry

UP = np.array([0.0, 0.0, 1.0])


@pytest.mark.parametrize("supercell", [[[1, 0], [0, 1]], [[1, 1], [1, -1]],
                                       [[2, 1], [1, 2]], [[1, 1], [-1, 2]], [[2, 0], [0, 2]]])
def test_supercell_cells_are_canonical_and_complete(supercell):
    M = np.array(supercell)
    cells = supercell_cells(M)
    assert len(cells) == abs(round(np.linalg.det(M)))
    for cell in cells:
        u = np.array(cell) @ np.linalg.inv(M)
        assert np.all(u > -1e-12) and np.all(u < 1 - 1e-12)
    rng = np.random.default_rng(0)
    for cell in rng.integers(-9, 10, size=(20, 2)):
        reduced = reduce_cell(cell, M)
        assert reduced in cells
        shift = (np.array(cell) - reduced) @ np.linalg.inv(M)
        np.testing.assert_allclose(shift, np.round(shift), atol=1e-12)


def test_direction_lookup_reduces_any_cell():
    model = square_heisenberg()
    state = neel_state(model)
    assert state.num_cells == 2
    np.testing.assert_array_equal(state.direction("A", (5, 2)), -UP)
    np.testing.assert_array_equal(state.direction("A", (-3, -1)), UP)


def test_singular_or_fractional_supercell_is_rejected():
    model = square_heisenberg()
    for supercell in ([[1, 1], [2, 2]], [[1.5, 0], [0, 1]]):
        with pytest.raises(SpinStateError):
            SpinState(model.fingerprint(), supercell, {("A", (0, 0)): UP})


def test_missing_noncanonical_and_nonunit_directions_are_reported():
    model = square_heisenberg()
    with pytest.raises(SpinStateError) as info:
        SpinState(model.fingerprint(), [[1, 1], [1, -1]],
                  {("A", (0, 0)): 2 * UP, ("A", (3, 0)): UP})
    found = info.value.violations
    assert any("not a unit vector" in v for v in found)
    assert any("not a canonical cell" in v for v in found)
    assert any("no direction for cells [(1, 0)]" in v for v in found)


def test_state_for_another_model_is_rejected():
    state = polarized_state(square_heisenberg(J=1.0))
    with pytest.raises(SpinStateError, match="different model"):
        validate_spin_state(state, square_heisenberg(J=2.0))


def test_torus_must_be_tiled_by_the_supercell():
    model = triangular_heisenberg()
    state = state_120(model)
    assert validate_spin_state(state, model, CalculationGeometry.finite_torus([[3, 0], [0, 3]])) == []
    with pytest.raises(SpinStateError, match="not a lattice vector"):
        validate_spin_state(state, model, CalculationGeometry.finite_torus([[4, 0], [0, 4]]))
    assert validate_spin_state(state, model, CalculationGeometry.thermodynamic_limit()) == []


def test_period_smaller_than_supercell_is_a_notice_not_an_error():
    model = square_heisenberg()
    folded = SpinState.from_function(model, [[2, 0], [0, 2]], lambda site, cell: UP)
    notices = validate_spin_state(folded, model)
    assert len(notices) == 1 and "smaller than the declared supercell" in notices[0]
    assert validate_spin_state(neel_state(model), model) == []


def test_state_is_immutable_and_copies_inputs():
    model = square_heisenberg()
    vector = UP.copy()
    state = SpinState(model.fingerprint(), np.eye(2, dtype=int), {("A", (0, 0)): vector})
    vector[2] = -1.0
    np.testing.assert_array_equal(state.direction("A"), UP)
    with pytest.raises(ValueError):
        state.directions[("A", (0, 0))][0] = 1.0
    with pytest.raises(TypeError):
        state.directions[("A", (0, 0))] = UP
