"""NBCP extraction regressions against the archived unit-cell implementation.

The archived DM rotation and quantum Hamiltonian have known differences from
the current implementation. Legacy comparisons here cover zero-DM model data
and classical energy only; quantum correctness has separate solver tests.
"""

from importlib import import_module
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from spintoolkit.methods.lswt.energy import EnergyFunction
from model import nbcp


CELLS = [
    ("one_msl", 2, "Hex_60"),
    ("two_msl", 4, "Tetra"),
    ("three_msl", 6, "Hex_30"),
    ("four_msl", 8, "Hex_60"),
]
CONFIG = {"Jxy": 0.076, "Jz": 0.125, "h": (0.03, -0.04, 0.2)}


@pytest.fixture
def legacy(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "legacy"))
    cells = import_module("modules.SpinSystem.nbcp_unitcells")
    energy = import_module("modules.Tools.analysis_tools")
    return cells.NBCP_UNIT_CELL, energy.Create_Energy_Function


@pytest.mark.parametrize("name,num_angles,bz_type", CELLS)
@pytest.mark.parametrize("parameters", [
    {},
    {"JPD": 0.013, "JGamma": -0.021},
    {"JPD": 0.013, "JGamma": -0.021,
     "Kxy": 0.004, "Kz": -0.005, "KPD": 0.002, "KGamma": -0.003},
], ids=["xxz", "nn_soc", "nn_and_nnn_soc"])
def test_model_and_classical_energy_match_legacy(
    legacy, name, num_angles, bz_type, parameters,
):
    legacy_cells, legacy_energy = legacy
    config = {**CONFIG, **parameters}
    angles = np.linspace(-1.1, 2.3, num_angles)
    reference = getattr(legacy_cells(config), "spin_system_data_" + name)(angles)
    system = getattr(nbcp, name)(
        config, angles,
        nbcp.make_nn_exchange_matrices(config),
        nbcp.make_nnn_exchange_matrices(config),
    )
    actual = system.to_legacy_dict(bz_type)

    assert list(actual["Spin info"]) == list(reference["Spin info"])
    for label, expected in reference["Spin info"].items():
        for key, value in expected.items():
            assert_array_equal(actual["Spin info"][label][key], value)
    assert actual["Lattice/BZ setting"][1] == reference["Lattice/BZ setting"][1]
    assert_array_equal(actual["Lattice/BZ setting"][0], reference["Lattice/BZ setting"][0])
    assert len(actual["Couplings"]) == len(reference["Couplings"])
    for coupling, expected in zip(actual["Couplings"], reference["Couplings"]):
        for key, value in expected.items():
            assert_array_equal(coupling[key], value)

    current_energy = EnergyFunction(actual, N=3)
    reference_energy = legacy_energy(reference, N=3)
    # Exercise both the supplied spin state and angle overrides.
    for trial in [angles, angles + 0.17, np.zeros(num_angles)]:
        assert_allclose(
            current_energy.classical_energy_density_func(trial),
            reference_energy.classical_energy_density_func(trial),
            rtol=0, atol=1e-15,
        )


@pytest.mark.parametrize("name,num_angles,bz_type", CELLS)
def test_optional_bonds_remain_explicit(name, num_angles, bz_type):
    config = {**CONFIG, "Kxy": 0.01}
    builder = getattr(nbcp, name)
    angles = np.zeros(num_angles)
    assert not builder(config, angles).couplings
    nnn_only = builder(config, angles, Exch_K=nbcp.make_nnn_exchange_matrices(config))
    assert len(nnn_only.couplings) == 3 * num_angles // 2
    assert nbcp.make_nnn_exchange_matrices(CONFIG) is None


def test_old_example_imports_preserve_builder_interface():
    example = import_module("examples.nbcp_ground_state")
    for name in nbcp.__all__:
        assert getattr(example, name) is getattr(nbcp, name)
    for phase, (name, num_angles, bz_type) in zip(example.PHASES.values(), CELLS):
        assert phase["builder"] is getattr(nbcp, name)
        assert phase["num_angles"] == num_angles
        assert phase["bz_type"] == bz_type
