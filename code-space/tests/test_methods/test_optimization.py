"""Check SpinOptimizer.find_minimum argument handling.

The unconstrained ``quantum`` method (E_cl + E_qm minimized off the classical
manifold) was removed (D18), and so was the MAGSWT grid search (D27);
zero-point selection is done on the classical manifold by
``state_selection.select_on_manifold`` (D17).
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spintoolkit.methods.lswt.energy import EnergyFunction
from spintoolkit.methods.optimization import SpinOptimizer
from model import nbcp


CONFIG = {"Jxy": 0.076, "Jz": 0.125, "h": (0.03, -0.04, 0.2)}


def _energy_function():
    system = nbcp.one_msl(CONFIG, np.zeros(2), nbcp.make_nn_exchange_matrices(CONFIG))
    return EnergyFunction(system.to_legacy_dict("Hex_60"), N=3)


def test_angle_setting_none_matches_all_free_list():
    """angle_setting=None means all angles free, as the docstring states."""
    cef = _energy_function()
    optimizer = SpinOptimizer()
    results = [optimizer.find_minimum(cef, "classical", angle_setting)
               for angle_setting in (None, [None, None])]

    (best_none, classical_none), (best_list, classical_list) = results
    assert best_none["method"] == best_list["method"]
    assert_allclose(best_none["angles"], best_list["angles"])
    assert_allclose(best_none["energy"], best_list["energy"])
    assert_allclose(classical_none["angles"], classical_list["angles"])


@pytest.mark.parametrize("method", ["quantum", "classical+quantum", "MAGSWT", "magswt"])
def test_removed_methods_point_to_manifold_selection(method):
    with pytest.raises(ValueError, match="select_on_manifold"):
        SpinOptimizer().find_minimum(_energy_function(), method)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown opt_method"):
        SpinOptimizer().find_minimum(_energy_function(), "annealing")
