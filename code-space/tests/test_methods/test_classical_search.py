"""Global classical search on the common model (stage 6c, D30, D32).

1. Benchmarks: square Neel, triangular 120 degrees on the sqrt3 x sqrt3 cell,
   polarized square ferromagnet in a field; stationary after refinement.
2. The former SpinOptimizer (same differential-evolution settings) on NBCP
   candidate cells: the same search energy and the same refined energy.
3. The former search classes raise DeprecationWarning; the new one does not.
"""

import warnings

import numpy as np
import pytest

from model import nbcp
from model.nbcp.model import legacy_cells
from spintoolkit.methods.classical import (
    ClassicalSearchResult, classical_energy, classical_search, refine_classical, torques)
from spintoolkit.models import square_heisenberg, triangular_heisenberg
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.conversion import to_spin_system

NBCP_CONFIG = {"Jxy": 0.076, "Jz": 0.125, "JGamma": 0.1}
NBCP_FIELD = ExternalConditions(field=(0, 0, 0.376418))


def max_torque(model, state, conditions=None):
    return max(np.linalg.norm(t) for t in torques(model, state, conditions).values())


@pytest.mark.parametrize("model, cell, conditions, expected", [
    (square_heisenberg(J=1.0), [[1, 1], [1, -1]], None, -0.5),
    (triangular_heisenberg(J=1.0), [[1, 1], [-1, 2]], None, -0.375),
    (square_heisenberg(J=1.0), [[1, 0], [0, 1]], ExternalConditions(field=(0, 0, 5.0)), -2.0),
])
def test_benchmark_minima(model, cell, conditions, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = classical_search(model, cell, conditions)
    assert isinstance(result, ClassicalSearchResult)
    assert result.energy == pytest.approx(expected, abs=1e-12)
    assert result.search_energy >= result.energy - 1e-12
    assert max_torque(model, result.state, conditions) < 1e-10
    assert result.settings["seed"] == 42 and result.evaluations > 0


@pytest.mark.parametrize("cell, bz_type", [("two_msl", "Tetra"), ("three_msl", "Hex_30")])
def test_same_result_as_the_former_spin_optimizer(cell, bz_type):
    from spintoolkit.methods.lswt.energy import EnergyFunction
    from spintoolkit.methods.optimization import SpinOptimizer

    model = nbcp.build_model(NBCP_CONFIG)
    new = classical_search(model, nbcp.SUPERCELLS[cell], NBCP_FIELD)
    ns = len(legacy_cells(cell))
    start = nbcp.candidate_state(model, cell, np.zeros(2 * ns))
    with pytest.warns(DeprecationWarning):
        energy = EnergyFunction(to_spin_system(model, start, NBCP_FIELD).to_legacy_dict(bz_type),
                                N=2, update_args=True)
        optimizer = SpinOptimizer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, legacy = optimizer.find_minimum(energy, "classical", [None] * (2 * ns))
    assert new.search_energy == pytest.approx(legacy["E_cl"], abs=1e-13)
    refined = refine_classical(model, nbcp.candidate_state(model, cell, legacy["angles"]), NBCP_FIELD)
    assert new.energy == pytest.approx(classical_energy(model, refined, NBCP_FIELD), abs=1e-14)
