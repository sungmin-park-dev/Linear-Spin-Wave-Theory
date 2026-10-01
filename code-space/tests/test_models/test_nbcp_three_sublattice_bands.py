"""NBCP three-sublattice phase below the transverse critical field (arXiv:2505.06398 Table 1).

The classical reference (one spin along B, two canted towards +c and -c)
matches the closed form cos(beta) = (h / S - 3 Jxy) / (3 (Jxy + Jz)); LSWT
about it is stable with three gapped modes, K folds onto Gamma, and the gap
closes continuously as B approaches the classical critical field from below.
"""

import warnings

import numpy as np
import pytest

from model import nbcp
from model.nbcp.model import PARAMETER_SETS, build_published_model
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods.classical import refine_classical
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.observables.bands import band_structure
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.high_symmetry import high_symmetry_points

PARAMS = PARAMETER_SETS["woodland2025"]["parameters"]
G_AB = PARAMETER_SETS["woodland2025"]["g"][1, 1]
START = [np.pi / 2, np.pi / 2, 0.3, np.pi / 2, np.pi - 0.3, np.pi / 2]   # A || y, B and C near +-c


def _ground(field_t):
    model = build_published_model("woodland2025")
    conditions = ExternalConditions(field=(0.0, MU_B_MEV_PER_T * field_t, 0.0))
    state = refine_classical(model, nbcp.candidate_state(model, "three_msl", START), conditions)
    return model, conditions, state


def _lowest_gap(field_t):
    model, conditions, state = _ground(field_t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(6, 6)))
        K = high_symmetry_points(model.lattice)["K"]
        bands = band_structure(result, [-1.5 * K, -K, 0 * K, K, 1.5 * K], points=121)
    return bands


@pytest.mark.parametrize("field_t", [0.7, 1.0, 1.4])
def test_classical_state_matches_the_closed_form(field_t):
    _, _, state = _ground(field_t)
    d = np.array(list(state.directions.values()))
    along = int(np.argmax(d[:, 1]))
    others = np.delete(d, along, axis=0)
    h = G_AB * MU_B_MEV_PER_T * field_t
    cos_beta = (2 * h - 3 * PARAMS["Jxy"]) / (3 * (PARAMS["Jxy"] + PARAMS["Jz"]))   # S = 1/2
    assert d[along, 1] == pytest.approx(1, abs=1e-9)
    assert np.allclose(others[:, 1], cos_beta, atol=1e-8)
    assert others[0, 2] == pytest.approx(-others[1, 2], abs=1e-8)
    assert np.allclose(d[:, 0], 0, atol=1e-9)


def test_three_gapped_modes_and_k_folds_onto_gamma():
    bands = _lowest_gap(1.0)
    assert bands.energies.shape[1] == 3 and not np.isnan(bands.energies).any()
    rows = [int(np.argmin(np.abs(bands.distance - bands.label_distances[i]))) for i in (2, 3)]
    assert np.allclose(bands.energies[rows[0]], bands.energies[rows[1]], atol=1e-10)
    assert np.min(bands.energies) > 0.02


def test_gap_closes_towards_the_classical_critical_field():
    gaps = [np.min(_lowest_gap(b).energies) for b in (1.0, 1.5, 1.7)]
    assert gaps[0] > gaps[1] > gaps[2] > 0
    assert gaps[2] < 0.1 * gaps[0]
