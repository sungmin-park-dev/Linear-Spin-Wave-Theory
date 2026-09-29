"""Check the classical+quantum branch of SpinOptimizer.find_minimum.

The branch starts from the classical DE optimum and keeps an L-BFGS-B
refinement of E_cl + E_qm only when it lowers the total energy evaluated at
the DE angles. Otherwise the DE angles are returned with method 'DE'.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import OptimizeResult

from spintoolkit.methods.lswt.energy import EnergyFunction
from spintoolkit.methods.optimization import SpinOptimizer
from model import nbcp


CONFIG = {"Jxy": 0.076, "Jz": 0.125, "h": (0.03, -0.04, 0.2)}


def _energy_function(cell="one_msl", bz_type="Hex_60", h=CONFIG["h"]):
    config = {**CONFIG, "h": h}
    num_angles = {"one_msl": 2, "two_msl": 4}[cell]
    system = getattr(nbcp, cell)(config, np.zeros(num_angles),
                                 nbcp.make_nn_exchange_matrices(config))
    return EnergyFunction(system.to_legacy_dict(bz_type), N=3)


@pytest.mark.parametrize("angle_setting", [[None, None], [None, 0.25]],
                         ids=["all_free", "phi_fixed"])
def test_quantum_falls_back_to_de_when_bfgs_does_not_improve(angle_setting):
    """A BFGS result that is not lower keeps the DE angles and total energy."""
    cef = _energy_function()
    optimizer = SpinOptimizer()
    num_free = angle_setting.count(None)
    optimizer.find_optimum_w_BFGS_from_DE = lambda *args, **kwargs: OptimizeResult(
        x=np.zeros(num_free), fun=np.inf)

    best, classical = optimizer.find_minimum(cef, "quantum", angle_setting)

    assert best["method"] == "DE"
    assert len(best["angles"]) == 2 * cef.num_SL
    assert_allclose(best["angles"], classical["angles"])
    for full, fixed in zip(best["angles"], angle_setting):
        if fixed is not None:
            assert full == fixed
    assert best["E_cl"] == classical["E_cl"]
    assert_allclose(best["E_qm"], cef.quantum_energy_density_func(best["angles"]))
    assert_allclose(best["energy"], best["E_cl"] + best["E_qm"])
    assert best["MAGSWT"] is not None


def test_quantum_result_is_not_above_de_total_energy():
    """The reported state is self-consistent and not worse than DE in E_cl + E_qm."""
    cef = _energy_function()
    np.random.seed(0)

    best, classical = SpinOptimizer().find_minimum(cef, "quantum", [None, None])

    de_total = (classical["E_cl"]
                + cef.quantum_energy_density_func(classical["angles"]))
    assert best["method"] in ("DE", "DE+BFGS")
    assert len(best["angles"]) == 2 * cef.num_SL
    assert best["energy"] <= de_total
    assert_allclose(best["E_cl"], cef.classical_energy_density_func(best["angles"]))
    assert_allclose(best["E_qm"], cef.quantum_energy_density_func(best["angles"]))
    assert_allclose(best["energy"], best["E_cl"] + best["E_qm"])


def test_bfgs_refinement_does_not_mutate_initial_points():
    """Random restarts perturb a copy, not the caller's starting array."""
    init_points = np.array([0.1, -0.2, 0.3])
    original = init_points.copy()
    np.random.seed(0)

    SpinOptimizer().find_optimum_w_BFGS_from_DE(
        lambda x: float(np.sum(x ** 2)), [(-np.pi, np.pi)] * 3, init_points)

    assert_allclose(init_points, original, rtol=0, atol=0)


def test_quantum_keeps_classical_result_at_de_optimum():
    """cl_result angles still reproduce the DE classical energy after refinement."""
    cef = _energy_function("two_msl", "Tetra", h=(0.0, 0.0, 0.0))
    np.random.seed(0)

    _, classical = SpinOptimizer().find_minimum(cef, "quantum", [None] * 4)

    assert_allclose(cef.classical_energy_density_func(classical["angles"]),
                    classical["E_cl"], rtol=0, atol=1e-10)
