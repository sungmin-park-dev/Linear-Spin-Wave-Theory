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


# ---------------------------------------------------------------------------
# Adaptive restarts with the supercell Luttinger-Tisza bound (D70)
# ---------------------------------------------------------------------------

def test_supercell_bound():
    from spintoolkit.methods.classical import supercell_lt_bound
    model = triangular_heisenberg(J=1.0)
    assert supercell_lt_bound(model, [[1, 1], [-1, 2]]) == pytest.approx(-0.375, abs=1e-12)
    # the 120-degree momenta do not fit the 1 x 2 cell: the best collinear state, -J/4
    assert supercell_lt_bound(model, [[1, 0], [0, 2]]) == pytest.approx(-0.25, abs=1e-12)
    field = ExternalConditions(field=(0, 0, 1.0))
    assert supercell_lt_bound(square_heisenberg(J=1.0), [[1, 0], [0, 1]], field) is None


def test_adaptive_stops_when_the_bound_is_reached():
    result = classical_search(triangular_heisenberg(J=1.0), [[1, 1], [-1, 2]])
    assert result.settings["stop"] == "lt_bound_reached" and result.settings["bound_reached"]
    assert len(result.settings["runs"]) == 1


def test_fixed_restarts_and_validation():
    result = classical_search(square_heisenberg(J=1.0), [[1, 1], [1, -1]], restarts=2)
    assert [r["seed"] for r in result.settings["runs"]] == [42, 43]
    for bad in ("many", 0, 1.5):
        with pytest.raises(ValueError):
            classical_search(square_heisenberg(J=1.0), [[1, 1], [1, -1]], restarts=bad)


def test_adaptive_escapes_a_locally_stable_trap():
    """BCAO, J3 = 0.45, Jz+- = 0.5, period-6 stripe cell: seed 42 alone ends in a locally
    stable up-up-down state at -0.303614, above the LT bound -0.315466; later seeds find
    -0.304193 (theory-reliability benchmarks, 2026-10-21)."""
    from spintoolkit.models import bcao_minimal
    result = classical_search(bcao_minimal(J3=0.45, Jzpm3=0.5), [[1, 0], [0, 6]], maxiter=300)
    assert result.settings["runs"][0]["energy"] == pytest.approx(-0.303614, abs=1e-6)
    assert result.energy == pytest.approx(-0.304193, abs=1e-6)
    assert not result.settings["bound_reached"] and result.settings["stop"] == "no_lower_energy"
