"""Zero-mode scan and finite-temperature LSWT quantities (stage 4b, D25).

Temperatures are dimensionless, t = k_B T / E0.
"""

import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from model import nbcp
from spintoolkit.definitions.constants import K_BOLTZMANN_MEV, MU_B_MEV_PER_T
from spintoolkit.methods.ed import EDSector, solve_ed
from spintoolkit.methods.lswt import LSWTSettings, LSWTSolver, solve_lswt
from spintoolkit.models import neel_state, polarized_state, square_heisenberg, triangular_heisenberg
from spintoolkit.observables.thermal import ZeroModeCandidateError, thermal_quantities
from spintoolkit.observables.thermodynamics import Thermodynamics
from spintoolkit.observables.zero_modes import CANDIDATE, GAPPED, ZERO, scan_zero_modes
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.conversion import to_spin_system
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import Site, SpinModel, Term

ROOT = Path(__file__).resolve().parents[3]
FIELD = ExternalConditions(field=(0, 0, 5.0))


def square_model(anisotropy=0.0, J2=0.0):
    """Square S = 1/2 antiferromagnet: XXZ with easy-axis anisotropy, optional J2."""
    J = np.diag([1.0, 1.0, 1.0 + anisotropy])
    terms = [Term.bilinear(("A", (0, 0)), ("A", (1, 0)), J),
             Term.bilinear(("A", (0, 0)), ("A", (0, 1)), J)]
    if J2:
        terms += [Term.bilinear(("A", (0, 0)), ("A", (1, 1)), J2 * np.eye(3)),
                  Term.bilinear(("A", (0, 0)), ("A", (1, -1)), J2 * np.eye(3))]
    return SpinModel(np.eye(2), [Site("A", (0, 0), 0.5)], terms + [Term.zeeman("A", np.eye(3))],
                     {"model_id": f"square_xxz_{anisotropy}_{J2}"})


def nbcp_y(extra):
    scan = json.loads((ROOT / "data-space/verification/260912-pseudo-goldstone/scan-N12-P72.json")
                      .read_text())
    model = nbcp.build_model({"Jxy": 0.075, "Jz": 0.125, **extra})
    theta = np.array(scan["states"]["Y"]["theta"])
    state = nbcp.candidate_state(model, "three_msl", np.column_stack([theta, np.zeros(3)]).ravel())
    return model, state, ExternalConditions(field=(0, 0, 4.645 * MU_B_MEV_PER_T * 0.2))


# ---------------------------------------------------------------------------
# Zero-mode scan
# ---------------------------------------------------------------------------

def test_neel_goldstone_mode_at_zone_centre():
    model = square_heisenberg(J=1.0)
    state = neel_state(model)
    report = scan_zero_modes(solve_lswt(model, state), model, state)
    assert report.has_zero and not report.has_candidates
    centre = report.candidates[0]
    assert np.allclose(np.mod(centre.fractional + 0.5, 1) - 0.5, 0, atol=1e-6)
    assert centre.origin == "goldstone"
    assert report.zone_centre["broken_symmetries"] == 2


@pytest.mark.parametrize("anisotropy, expected", [(1e-2, GAPPED), (1e-5, GAPPED),
                                                  (1e-9, CANDIDATE), (0.0, ZERO)])
def test_small_gap_moves_from_gapped_to_candidate_to_zero(anisotropy, expected):
    model = square_model(anisotropy)
    report = scan_zero_modes(solve_lswt(model, neel_state(model)))
    lowest = report.global_minimum
    assert lowest.classification == expected
    assert np.allclose(np.mod(lowest.fractional + 0.5, 1) - 0.5, 0, atol=1e-6)
    if expected == GAPPED:
        # LSWT gap of the easy-axis antiferromagnet: 4JS sqrt((1+d)^2 - 1)
        assert lowest.gap == pytest.approx(2 * np.sqrt((1 + anisotropy) ** 2 - 1), rel=1e-3)


def test_candidates_require_the_users_decision():
    model = square_model(1e-9)
    result = solve_lswt(model, neel_state(model))
    with pytest.raises(ZeroModeCandidateError, match="gapless=True"):
        thermal_quantities(result, [0.1])
    with pytest.warns(UserWarning, match="gapless spectrum"):
        as_zero = thermal_quantities(result, [0.1], gapless=True)
    assert as_zero.decision == "user" and np.all(np.isnan(as_zero.moments))
    as_gap = thermal_quantities(result, [0.1], gapless=False)
    assert np.all(np.isfinite(as_gap.moments))
    with pytest.raises(ZeroModeCandidateError):
        solve_lswt(model, neel_state(model), ExternalConditions(temperature=0.1))
    settled = solve_lswt(model, neel_state(model), ExternalConditions(temperature=0.1),
                         settings=LSWTSettings(gapless=False))
    assert settled.thermal.decision == "user" and settled.header.settings["gapless"] is False


def test_line_of_zero_modes_is_flagged():
    """J1-J2 square at J2 = J1/2: the Neel spectrum vanishes on lines, not only at k = 0."""
    model = square_model(J2=0.5)
    state = neel_state(model)
    result = solve_lswt(model, state, settings=LSWTSettings(regularization="MAGSWT"))
    report = scan_zero_modes(result, model, state)
    off_centre = [c for c in report.candidates
                  if np.linalg.norm(np.mod(c.fractional + 0.5, 1) - 0.5) > 1e-3]
    assert off_centre and all(c.classification == ZERO for c in off_centre)
    assert any(c.line_directions for c in off_centre)
    assert all(c.origin.startswith("not a uniform rotation") for c in off_centre)


def test_nbcp_accidental_zero_mode():
    model, state, conditions = nbcp_y({"JPD": 0.01})
    report = scan_zero_modes(solve_lswt(model, state, conditions), model, state, conditions)
    assert report.has_zero
    assert report.zone_centre == {"symmetry_axes": 0, "broken_symmetries": 0,
                                  "classical_flat_directions": 1, "origin": "accidental"}
    model, state, conditions = nbcp_y({})
    report = scan_zero_modes(solve_lswt(model, state, conditions), model, state, conditions)
    assert report.zone_centre["origin"] == "goldstone"


def test_polarized_state_is_gapped_by_h_minus_h_sat():
    model = square_heisenberg(J=1.0)
    report = scan_zero_modes(solve_lswt(model, polarized_state(model), FIELD,
                                        settings=LSWTSettings(mesh=(16, 16), shift=False)))
    assert not report.candidates
    assert report.global_minimum.gap == pytest.approx(5.0 - 4.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Finite temperature
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore:boson numbers exceed S")
def test_polarized_ferromagnet_matches_direct_formulas():
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, polarized_state(model), FIELD, settings=LSWTSettings(mesh=(24, 24)))
    t = np.array([0.1, 0.5, 1.0, 3.0])
    thermal = thermal_quantities(result, t)
    w = result.bands()[:, 0]
    base = result.classical_energy + result.zero_point_energy
    for i, ti in enumerate(t):
        n = 1 / np.expm1(w / ti)
        assert thermal.internal_energy[i] == pytest.approx(base + np.mean(w * n), abs=1e-13)
        assert thermal.free_energy[i] == pytest.approx(
            base + ti * np.mean(np.log(-np.expm1(-w / ti))), abs=1e-13)
        assert thermal.entropy[i] == pytest.approx(
            np.mean((1 + n) * np.log1p(n) - n * np.log(n)), abs=1e-13)
        assert thermal.specific_heat[i] == pytest.approx(
            np.mean((w / ti) ** 2 * n * (n + 1)), abs=1e-13)
        assert thermal.moments[i][0] == pytest.approx(0.5 - np.mean(n), abs=1e-13)
    np.testing.assert_allclose(thermal.magnetization[:, :2], 0, atol=1e-15)


def test_thermodynamic_identities():
    model = triangular_heisenberg(J=1.0)
    result = solve_lswt(model, polarized_state(model), FIELD, settings=LSWTSettings(mesh=(18, 18)))
    t, h = 0.8, 1e-4
    f = thermal_quantities(result, [t - h, t, t + h])
    assert f.internal_energy[1] == pytest.approx(f.free_energy[1] + t * f.entropy[1], abs=1e-12)
    assert f.entropy[1] == pytest.approx(-(f.free_energy[2] - f.free_energy[0]) / (2 * h), rel=1e-7)
    assert f.specific_heat[1] == pytest.approx(
        (f.internal_energy[2] - f.internal_energy[0]) / (2 * h), rel=1e-7)
    assert f.specific_heat[1] == pytest.approx(t * (f.entropy[2] - f.entropy[0]) / (2 * h), rel=1e-7)


def test_low_and_high_temperature_limits():
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, polarized_state(model), FIELD, settings=LSWTSettings(mesh=(16, 16)))
    with pytest.warns(UserWarning, match="exceed S"):
        hot = thermal_quantities(result, [1e4])
    assert hot.specific_heat[0] == pytest.approx(1.0, abs=1e-6)       # one mode per site
    assert hot.beyond_lswt[0]
    cold = thermal_quantities(result, [0.05, 0.1])
    ratio = cold.specific_heat[0] / cold.specific_heat[1]
    assert ratio < np.exp(-1.0 / 0.05 + 1.0 / 0.1) * 10                # activated, gap 1


@pytest.mark.parametrize("model, L", [(square_heisenberg(J=1.0), [[2, 0], [0, 3]]),
                                      (triangular_heisenberg(J=1.0), [[3, 0], [0, 3]])],
                         ids=["square_2x3", "triangular_3x3"])
def test_low_temperature_free_energy_agrees_with_ed_up_to_two_magnons(model, L):
    """Z_ED and the free-boson Z share the 0- and 1-magnon terms exactly."""
    geometry = CalculationGeometry.finite_torus(L)
    ed = solve_ed(model, geometry, FIELD, EDSector(axis=(0, 0, 1), magnon_number="all"))
    energies = ed.energies()
    lswt = solve_lswt(model, polarized_state(model), FIELD, geometry)
    gap = lswt.bands().min()
    for t in (gap / 10, gap / 5):
        f_ed = (energies.min() - t * np.log(np.sum(np.exp(-(energies - energies.min()) / t)))) / ed.num_sites
        difference = abs(f_ed - thermal_quantities(lswt, [t]).free_energy[0])
        assert difference < np.exp(-2 * gap / t)
        assert difference < 1e-3 * np.exp(-gap / t)


def test_neel_at_finite_temperature():
    model = square_heisenberg(J=1.0)
    results = {n: solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(n, n)))
               for n in (32, 64)}
    with pytest.warns(UserWarning, match="gapless spectrum"):
        thermal = {n: thermal_quantities(r, [0.0, 0.3]) for n, r in results.items()}
    assert thermal[64].gapless and thermal[64].decision == "scan"
    assert np.all(np.isfinite(thermal[64].moments[0])) and np.all(np.isnan(thermal[64].moments[1]))
    assert thermal[64].internal_energy[1] == pytest.approx(thermal[32].internal_energy[1], abs=2e-4)
    assert thermal[64].specific_heat[1] == pytest.approx(thermal[32].specific_heat[1], abs=2e-3)


def test_matches_existing_kelvin_thermodynamics_on_nbcp():
    model = nbcp.build_model({"Jxy": 0.075, "Jz": 0.125, "JPD": 0.01})
    state = nbcp.candidate_state(model, "one_msl", [0.0, 0.0])
    conditions = ExternalConditions(field=(0, 0, 2.0))
    system = to_spin_system(model, state, conditions)
    solver = LSWTSolver(system, bz_type="Hex_60")
    solver.diagnosing_lswt(bz_type="Hex_60", N=6)
    from spintoolkit.system.brillouin_zone import BrillouinZone
    _, k_points, _ = BrillouinZone(system.to_legacy_dict("Hex_60")["Lattice/BZ setting"],
                                   bz_type="Hex_60").get_full(6)
    k_data, _ = solver.Ham.solve_k_Hamiltonian(k_points, regularization="MAGSWT")
    legacy = Thermodynamics(solver)
    result = solve_lswt(model, state, conditions,
                        settings=LSWTSettings(k_points=k_points, regularization="MAGSWT"))
    for kelvin in (0.5, 2.0):
        t = kelvin * K_BOLTZMANN_MEV
        new = thermal_quantities(result, [t])
        assert new.internal_energy[0] - result.classical_energy == pytest.approx(
            legacy.compute_internal_energy(k_data, Temperature=kelvin), abs=1e-14)
        assert new.entropy[0] * K_BOLTZMANN_MEV == pytest.approx(
            legacy.compute_entropy_density(k_data, kelvin), abs=1e-14)
        assert new.specific_heat[0] * K_BOLTZMANN_MEV == pytest.approx(
            legacy.compute_specific_heat(k_data, kelvin), abs=1e-14)
