"""NBCP polarized phase in a transverse field reproduces arXiv:2505.06398 Eq. (5)."""

import warnings

import numpy as np
import pytest

from model.nbcp.model import LATTICE, PARAMETER_SETS, build_published_model
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import polarized_state
from spintoolkit.observables.bands import band_structure
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.high_symmetry import high_symmetry_points

PARAMS = PARAMETER_SETS["woodland2025"]["parameters"]
G_AB = PARAMETER_SETS["woodland2025"]["g"][1, 1]


def _eq5(k, field_t):
    deltas = np.array([LATTICE[0], LATTICE[1], LATTICE[0] + LATTICE[1]])
    gamma = np.cos(k @ deltas.T).sum(axis=1)
    a = G_AB * MU_B_MEV_PER_T * field_t - 3 * PARAMS["Jxy"] + (PARAMS["Jz"] + PARAMS["Jxy"]) / 2 * gamma
    b = (PARAMS["Jz"] - PARAMS["Jxy"]) / 2 * gamma
    w2 = a ** 2 - b ** 2
    return np.where(w2 >= 0, np.sqrt(np.clip(w2, 0, None)), np.nan)      # 2S = 1


def _bands(field_t):
    model = build_published_model("woodland2025")
    conditions = ExternalConditions(field=(0.0, MU_B_MEV_PER_T * field_t, 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solve_lswt(model, polarized_state(model, (0, 1, 0)), conditions,
                            settings=LSWTSettings(mesh=(6, 6), regularization="MAGSWT"))
        K = high_symmetry_points(model.lattice)["K"]
        return band_structure(result, [-1.5 * K, -K, 0 * K, K, 1.5 * K], points=150)


@pytest.mark.parametrize("field_t", [3.5, 1.741, 1.7])
def test_bands_match_eq5_and_are_undefined_exactly_where_it_is_imaginary(field_t):
    bands = _bands(field_t)
    reference = _eq5(bands.k_points, field_t)
    computed = bands.energies[:, 0]
    np.testing.assert_array_equal(np.isnan(computed), np.isnan(reference))
    np.testing.assert_allclose(computed[~np.isnan(computed)], reference[~np.isnan(reference)], atol=1e-12)


def test_gap_at_3p5_tesla_and_classical_critical_field():
    bands = _bands(3.5)
    assert np.nanmin(bands.energies) == pytest.approx(0.4657, abs=1e-4)        # paper: ~0.46 meV
    b_c = (3 * PARAMS["Jxy"] + 1.5 * PARAMS["Jz"]) / (G_AB * MU_B_MEV_PER_T)   # paper: 1.72 T
    assert b_c == pytest.approx(1.717, abs=1e-3)
    assert not np.isnan(_bands(b_c + 1e-4).energies).any()
    assert np.isnan(_bands(b_c - 1e-3).energies).any()
