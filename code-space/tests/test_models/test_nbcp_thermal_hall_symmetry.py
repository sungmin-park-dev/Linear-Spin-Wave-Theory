"""Pure XXZ NBCP magnons carry no thermal Hall current; bond-dependent terms allow one.

For a coplanar state whose plane contains the field axis z, T times the pi
spin rotation about the plane normal is an antiunitary symmetry of the XXZ
Hamiltonian in a field along z that fixes the state, so kappa_xy = 0. J_PD
breaks it.
"""

import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from model import nbcp
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods.classical import refine_classical
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.observables.berry import berry_curvature, thermal_hall
from spintoolkit.system.conditions import ExternalConditions

SCAN = (Path(__file__).resolve().parents[3]
        / "data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json")


def _y_state_result(extra):
    model = nbcp.build_model({"Jxy": 0.075, "Jz": 0.125, **extra})
    theta = np.array(json.loads(SCAN.read_text())["states"]["Y"]["theta"])
    conditions = ExternalConditions(field=(0, 0, 4.645 * MU_B_MEV_PER_T * 0.2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = refine_classical(model, nbcp.candidate_state(
            model, "three_msl", np.column_stack([theta, np.zeros(3)]).ravel()), conditions)
        return solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(9, 9)))


@pytest.mark.skipif(not SCAN.exists(), reason="NBCP Y-state scan not available")
def test_pure_xxz_y_state_has_no_berry_curvature_and_no_thermal_hall():
    result = _y_state_result({})
    assert np.nanmax(np.abs(berry_curvature(result).curvature)) < 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = thermal_hall(result, [0.02, 0.05], gapless=True).kappa_over_t
    np.testing.assert_allclose(kappa, 0.0, atol=1e-12)


@pytest.mark.skipif(not SCAN.exists(), reason="NBCP Y-state scan not available")
def test_pseudo_dipolar_term_breaks_the_symmetry():
    result = _y_state_result({"JPD": 0.01})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kappa = thermal_hall(result, [0.05], gapless=True).kappa_over_t
    assert abs(kappa[0]) > 1e-4
