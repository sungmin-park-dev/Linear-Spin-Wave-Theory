"""Minimally augmented spin waves (MAGSWT, D62) and the stationary-point search (D63).

Literature comparison: Chernyshev, arXiv:2509.08877, Eqs. (25)-(27): for the
honeycomb J1-J3 XXZ model (J1 = -1) the minimal shifts of the FM and zigzag
states are ``mu = 3S |gbar_Qmax| - A`` with ``A = 3S(1 - J3)`` (FM) and
``3S(1/3 + J3)`` (ZZ), independent of the XXZ anisotropies. Values are
dimensionless (E0 = |J1|).
"""

import warnings

import numpy as np
import pytest

from spintoolkit import (ExternalConditions, LSWTSettings, SpinState, compare_states,
                         find_stationary_state, solve_lswt)
from spintoolkit.methods.classical import classical_energy, torques
from spintoolkit.methods.lswt.run import LSWTError
from spintoolkit.models import bcao_minimal, polarized_state
from spintoolkit.observables.bands import band_structure

S = 0.5


def sp_gbar(J3):
    """|gbar_Qmax| in the classical spiral range, arXiv:2509.08877 Eq. (27)."""
    return np.sqrt((1 + 2 * J3) * (1 - J3) ** 3 / (J3 * (1 - 2 * J3))) / 3


def zigzag(model):
    return SpinState.from_function(model, [[1, 0], [0, 2]],
                                   lambda site, cell: np.array([(-1) ** cell[1], 0.0, 0.0]))


def stripes(model, signs, axis=(1.0, 0.0, 0.0)):
    a = np.array(axis)
    return SpinState.from_function(model, [[1, 0], [0, len(signs)]],
                                   lambda site, cell: signs[cell[1] % len(signs)] * a)


def magswt(model, state, augmentation="minimal", **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return solve_lswt(model, state, settings=LSWTSettings(augmentation=augmentation),
                          **kwargs)


@pytest.mark.parametrize("J3, gbar", [(0.30, sp_gbar(0.30)), (0.35, sp_gbar(0.35)),
                                      (0.60, 1 / 3 + 0.60)])
def test_minimal_shift_matches_chernyshev_eq25(J3, gbar):
    model = bcao_minimal(J3=J3, Delta1=0.0, Delta3=1.0, g=None)
    fm = magswt(model, polarized_state(model, (1, 0, 0))).extra["augmentation"]
    zz = magswt(model, zigzag(model)).extra["augmentation"]
    assert fm["mu_min"] == pytest.approx(3 * S * (gbar - (1 - J3)), abs=1e-7)
    assert zz["mu_min"] == pytest.approx(max(0.0, 3 * S * (gbar - (1 / 3 + J3))), abs=1e-7)
    # Collinear XXZ states: the real-spectrum criterion of the paper gives the same shift.
    assert fm["mu_real_spectrum"] == pytest.approx(fm["mu_min"], abs=1e-7)


def test_minimal_shift_independent_of_xxz_anisotropy():
    J3 = 0.35
    mus = []
    for d1, d3 in ((0.0, 1.0), (0.5, 0.5), (0.0, 0.0)):
        model = bcao_minimal(J3=J3, Delta1=d1, Delta3=d3, g=None)
        mus.append(magswt(model, polarized_state(model, (1, 0, 0))).extra["augmentation"]["mu_min"])
    assert np.allclose(mus, 3 * S * (sp_gbar(J3) - (1 - J3)), atol=1e-7)


def test_stable_state_is_unchanged():
    model = bcao_minimal(J3=0.1, Delta1=0.0, Delta3=1.0, g=None)
    state = polarized_state(model, (1, 0, 0))
    plain = solve_lswt(model, state)
    augmented = magswt(model, state)
    assert augmented.extra["augmentation"]["mu"] == 0.0
    assert augmented.ground_state_energy == plain.ground_state_energy
    assert np.array_equal(augmented.eigenvalues, plain.eigenvalues)


def test_unstable_state_energy_is_upper_estimate_monotonic_in_mu():
    model = bcao_minimal(J3=0.35, Delta1=0.0, Delta3=1.0, g=None)
    state = polarized_state(model, (1, 0, 0))
    with pytest.raises(LSWTError):
        solve_lswt(model, state)                         # D61 unchanged by default
    minimal = magswt(model, state)
    mu = minimal.extra["augmentation"]["mu_min"]
    assert "upper bound" in minimal.header.diagnostics["energy_bound"]
    assert abs(minimal.header.diagnostics["augmentation_padding_energy"]) < 1e-9
    energies = [magswt(model, state, m).ground_state_energy for m in (mu * 1.01, 2 * mu, 1e4)]
    assert minimal.ground_state_energy < energies[0] < energies[1] < energies[2]
    # mu -> infinity: the quantum correction vanishes as O(1/mu).
    assert energies[2] == pytest.approx(minimal.classical_energy, abs=1e-3)
    with pytest.raises(ValueError, match="below the minimal shift"):
        magswt(model, state, 0.5 * mu)


def test_augmented_result_reports_energy_only():
    model = bcao_minimal(J3=0.35, Delta1=0.0, Delta3=1.0, g=None)
    state = polarized_state(model, (1, 0, 0))
    result = magswt(model, state)
    assert np.all(np.isnan(result.boson_numbers))
    with pytest.raises(ValueError, match="MAGSWT"):
        band_structure(result)
    with pytest.raises(ValueError, match="finite-temperature"):
        magswt(model, state, conditions=ExternalConditions(temperature=0.1))


@pytest.mark.parametrize("bad", ["max", -1.0, float("nan")])
def test_augmentation_setting_validation(bad):
    with pytest.raises(ValueError):
        LSWTSettings(augmentation=bad)
    with pytest.raises(ValueError, match="regularization"):
        LSWTSettings(augmentation="minimal", regularization="MAGSWT")


def test_symmetric_patterns_are_not_stationary_with_jzpm3():
    model = bcao_minimal(J3=0.45, Jzpm3=0.6)
    for signs in ([1, 1, -1], [1, 1, -1, -1]):              # up-up-down, double zigzag
        seed = stripes(model, signs)
        assert max(np.linalg.norm(t) for t in torques(model, seed).values()) > 0.1
        found = find_stationary_state(model, seed)
        assert found.converged and found.max_torque <= 1e-10
        assert found.max_rotation_from_seed > 0.1          # canted out of the plane
        assert found.energy < classical_energy(model, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error")                  # no non-stationary warning
            magswt(model, found.state)


def test_stationary_search_keeps_a_saddle():
    model = bcao_minimal(J3=0.45, Jzpm3=0.0)
    uud = stripes(model, [1, 1, -1])
    exact = find_stationary_state(model, uud)
    assert exact.steps == 0 and exact.saddle_index >= 1
    rng = np.random.default_rng(3)
    moved = {k: v + 0.03 * rng.normal(size=3) for k, v in uud.directions.items()}
    kicked = SpinState(uud.model_ref, uud.supercell,
                       {k: v / np.linalg.norm(v) for k, v in moved.items()}, {})
    found = find_stationary_state(model, kicked)
    assert found.converged
    assert found.saddle_index == exact.saddle_index        # not slid down to a minimum
    assert found.energy == pytest.approx(exact.energy, abs=1e-12)


def test_compare_states_ranks_augmented_candidates():
    model = bcao_minimal(J3=0.35, Delta1=0.0, Delta3=1.0, g=None)
    candidates = {"FM": polarized_state(model, (1, 0, 0)), "ZZ": zigzag(model)}
    plain = {r.name: r for r in compare_states(model, candidates, local_stability=False)}
    assert plain["FM"].status == "unstable" and np.isnan(plain["FM"].harmonic_energy)
    reports = compare_states(model, candidates, refine="stationary", local_stability=False,
                             augmentation="minimal")
    by_name = {r.name: r for r in reports}
    assert by_name["FM"].status == "augmented"
    assert by_name["FM"].augmentation_mu > 0
    assert np.isfinite(by_name["FM"].harmonic_energy)
    assert np.isnan(by_name["FM"].lowest_magnon)
    assert reports[0].harmonic_energy <= reports[1].harmonic_energy
