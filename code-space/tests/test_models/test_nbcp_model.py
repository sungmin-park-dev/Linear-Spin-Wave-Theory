"""NBCP as a common SpinModel: parameter sets, bonds and candidate states."""

import numpy as np
import pytest

from model import nbcp
from model.nbcp.model import LATTICE, SUPERCELLS, legacy_cells
from spintoolkit.methods.classical import classical_energy
from spintoolkit.system.conditions import ExternalConditions


def test_published_parameter_sets_build_valid_models():
    woodland = nbcp.build_published_model("woodland2025")
    np.testing.assert_allclose(woodland.terms_of_kind("zeeman")[0].coefficient,
                               np.diag([4.200, 4.200, 4.716]))
    assert "arXiv:2505.06398" in woodland.metadata["sources"][0]
    park = nbcp.build_published_model("park2026_fig4")
    np.testing.assert_array_equal(park.terms_of_kind("zeeman")[0].coefficient, np.eye(3))
    assert park.metadata["parameters"]["g"] is None
    assert woodland.fingerprint() != park.fingerprint()


def test_bonds_are_recorded_once_with_unit_and_sqrt3_lengths():
    model = nbcp.build_model({"Jxy": 0.07, "Jz": 0.12, "Kxy": 0.01})
    lengths = {}
    for term in model.terms_of_kind("bilinear"):
        (a, n1), (b, n2) = term.participants
        length = np.linalg.norm(model.cartesian_position(b, n2) - model.cartesian_position(a, n1))
        lengths.setdefault(term.label, []).append(length)
    np.testing.assert_allclose(lengths["NN"], [1.0] * 3)
    np.testing.assert_allclose(lengths["NNN"], [np.sqrt(3)] * 3)
    assert len(nbcp.build_model({"Jxy": 0.07, "Jz": 0.12}).terms_of_kind("bilinear")) == 3


@pytest.mark.parametrize("cell", list(SUPERCELLS))
def test_supercells_match_legacy_lattices_and_cover_every_sublattice(cell):
    mapping = legacy_cells(cell)
    assert len(set(mapping.values())) == len(mapping) == abs(round(np.linalg.det(SUPERCELLS[cell])))
    model = nbcp.build_model({"Jxy": 0.07, "Jz": 0.12})
    state = nbcp.candidate_state(model, cell, np.zeros(2 * len(mapping)))
    np.testing.assert_allclose(state.magnetic_lattice(model), SUPERCELLS[cell] @ LATTICE)


def test_three_msl_candidate_keeps_legacy_angle_order():
    model = nbcp.build_model({"Jxy": 0.07, "Jz": 0.12})
    angles = [np.pi / 2, 0.0, np.pi / 2, 2 * np.pi / 3, np.pi / 2, -2 * np.pi / 3]
    state = nbcp.candidate_state(model, "three_msl", angles)
    for (label, cell), (theta, phi) in zip(legacy_cells("three_msl").items(),
                                          np.reshape(angles, (-1, 2))):
        np.testing.assert_allclose(state.direction("Co", cell),
                                   [np.cos(phi), np.sin(phi), 0.0], atol=1e-15)
    xxz = classical_energy(model, state)
    assert xxz == pytest.approx(3 * 0.07 * 0.25 * np.cos(2 * np.pi / 3), abs=1e-15)


def test_field_as_zeeman_energy_or_through_g():
    h = np.array([0.0, 0.0, 0.2])
    with_g = nbcp.build_published_model("woodland2025")
    without_g = nbcp.build_model({"Jxy": 0.0779, "Jz": 0.1225})
    state_g = nbcp.candidate_state(with_g, "one_msl", [0.4, 0.7])
    state_h = nbcp.candidate_state(without_g, "one_msl", [0.4, 0.7])
    b = np.linalg.solve(with_g.terms_of_kind("zeeman")[0].coefficient.T, h)
    assert classical_energy(with_g, state_g, ExternalConditions(field=b)) == pytest.approx(
        classical_energy(without_g, state_h, ExternalConditions(field=h)), abs=1e-15)
