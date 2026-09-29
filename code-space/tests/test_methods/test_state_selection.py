"""Zero-point state selection on the classical manifold (D17, D19).

NBCP Y (B = 0.2 T) and V (B = 1.4 T) states with a small J_PD or J_Gamma have
a classically flat orbit about z that E_qm splits. With J_Gamma the Y-state
splitting is only ~1e-12 meV per spin, so these tests check that the fit
resolves it far above its residual. References are the independent scans in
data-space/verification/260912-pseudo-goldstone (N = 48, 72 orbit points).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from model import nbcp
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods import state_selection as sel
from spintoolkit.methods.lswt.energy import EnergyFunction
from spintoolkit.methods.optimization import SpinOptimizer
from spintoolkit.models import polarized_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.conversion import to_spin_system
from spintoolkit.system.model import SpinModel, Term

ROOT = Path(__file__).resolve().parents[3]
SCAN = json.loads((ROOT / "data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json")
                  .read_text())
G_Z = 4.645
MESH_N = 6
PHI0 = 0.3
PHASES = {"Y": 0.2, "V": 1.4}
COUPLINGS = {"PD": {"JPD": 0.01}, "Gamma": {"JGamma": 0.01}}


def reference(phase, coupling):
    pd, gamma = (0.01, 0) if coupling == "PD" else (0, 0.01)
    for scan in SCAN["scans"]:
        if scan["phase"] == phase and scan["JPD_meV"] == pd and scan["JGamma_meV"] == gamma:
            return scan["current"]
    raise KeyError((phase, coupling))


def setup(phase, extra=None, phi0=PHI0, rotation=np.eye(3), noise=0.0):
    model = nbcp.build_model({"Jxy": 0.075, "Jz": 0.125, **(extra or {})})
    if not np.allclose(rotation, np.eye(3)):
        terms = [Term.bilinear(*t.participants, rotation @ t.coefficient @ rotation.T, t.label)
                 for t in model.terms_of_kind("bilinear")] + list(model.terms_of_kind("zeeman"))
        model = SpinModel(model.lattice, model.sites, terms, {"model_id": "rotated"})
    field = rotation @ [0.0, 0.0, G_Z * MU_B_MEV_PER_T * PHASES[phase]]
    theta = np.array(SCAN["states"][phase]["theta"])
    angles = np.column_stack([theta, np.full(3, phi0)]).ravel()
    angles += noise * np.random.default_rng(1).standard_normal(6)
    directions = nbcp.candidate_state(model, "three_msl", angles).directions
    state = nbcp.candidate_state(model, "three_msl", np.zeros(6))
    state = type(state)(state.model_ref, state.supercell,
                        {k: rotation @ v for k, v in directions.items()}, {"origin": "test"})
    return model, state, ExternalConditions(field=field)


def select(model, state, conditions, mode="physics", axis=None, bz_type="Hex_30"):
    return sel.select_on_manifold(model, state, conditions,
                                  sel.lswt_zero_point_energy(bz_type, MESH_N), axis=axis,
                                  criteria=sel.SelectionCriteria(mode=mode))


def common_azimuth(model, state, phase):
    """Rotation angle about z that best maps the phi = 0 reference configuration onto ``state``."""
    reference = setup(phase, phi0=0.0)[1]
    cross = dot = 0.0
    for key, r in reference.directions.items():
        s = state.directions[key]
        cross += r[0] * s[1] - r[1] * s[0]
        dot += r[0] * s[0] + r[1] * s[1]
    return float(np.arctan2(cross, dot))


@pytest.mark.parametrize("mode", sel.MODES)
@pytest.mark.parametrize("coupling", COUPLINGS)
@pytest.mark.parametrize("phase", PHASES)
def test_y_and_v_orbits_are_selected_and_match_the_reference_scans(phase, coupling, mode):
    model, state, conditions = setup(phase, COUPLINGS[coupling], noise=1e-3)
    result = select(model, state, conditions, mode)
    assert result.verdict == sel.SELECTED
    np.testing.assert_allclose(result.axis, [0, 0, 1], atol=1e-12)
    orbit = result.diagnostics["orbit"]
    ref = reference(phase, coupling)
    harmonic = 3 if (phase, coupling) == ("V", "Gamma") else 6
    assert orbit["E_qm_dominant_harmonic"] == harmonic
    ref_amplitude = np.hypot(ref["cos_coefficients"][harmonic - 1],
                             ref["sin_coefficients"][harmonic - 1])
    assert orbit["E_qm_amplitude"] == pytest.approx(ref_amplitude, rel=2e-2)
    assert result.diagnostics["C_qm"] == pytest.approx(ref["curvature_meV_per_spin"], rel=2e-2)
    # The ~1e-12 J_Gamma splitting of Y must be resolved, not reported as noise.
    assert orbit["E_qm_amplitude"] > 100 * orbit["E_qm_resolution_threshold"]
    period = 2 * np.pi / harmonic
    offset = np.mod(common_azimuth(model, result.state, phase) - ref["phi_min"] + period / 2,
                    period) - period / 2
    # Position resolution of the fit: residual / (m A); ~1e-5 rad for the Y J_Gamma case.
    resolution = orbit["E_qm_fit"]["residual"] / (harmonic * orbit["E_qm_amplitude"])
    assert abs(offset) < max(1e-6, 10 * resolution)
    assert len(result.diagnostics["equivalent_minima"]) == harmonic
    assert result.diagnostics["orbit"]["E_cl_span"] < 1e-15


@pytest.mark.parametrize("mode", sel.MODES)
@pytest.mark.parametrize("phase", PHASES)
def test_exact_u1_orbit_has_no_selection(phase, mode):
    result = select(*setup(phase), mode)
    assert result.verdict == sel.NO_SELECTION
    np.testing.assert_allclose(result.axis, [0, 0, 1], atol=1e-12)
    if mode == "physics":
        assert result.diagnostics["exact_symmetry"]
        assert "E_qm" not in result.diagnostics["orbit"]


def test_axis_follows_a_rotation_of_the_whole_problem():
    R = sel.rotation_matrix([1, 2, 0], 0.7)
    plain = select(*setup("Y", {"JPD": 0.01}))
    rotated = select(*setup("Y", {"JPD": 0.01}, rotation=R))
    assert rotated.verdict == sel.SELECTED
    axis = R @ [0, 0, 1]
    assert min(np.linalg.norm(rotated.axis - axis), np.linalg.norm(rotated.axis + axis)) < 1e-12
    assert rotated.diagnostics["orbit"]["E_qm_amplitude"] == pytest.approx(
        plain.diagnostics["orbit"]["E_qm_amplitude"], rel=1e-9)


def test_tilted_field_is_a_competition_in_physics_mode_and_pinned_in_fixed_mode():
    model, state, _ = setup("Y", {"JPD": 0.01})
    h = G_Z * MU_B_MEV_PER_T * 0.2
    conditions = ExternalConditions(field=[0.3 * h, 0, h])
    physics = select(model, state, conditions, "physics")
    assert physics.verdict == sel.COMPETITION
    assert set(physics.candidates) == {"classical", "quantum"}
    assert physics.state is not physics.candidates["quantum"]    # no automatic pick
    ratio = physics.diagnostics["classical_to_quantum"]["ratio"]
    assert 0.1 < ratio < 10
    assert select(model, state, conditions, "fixed").verdict == sel.NO_DEGENERACY


@pytest.mark.parametrize("mode", sel.MODES)
def test_polarized_nbcp_state_has_no_degeneracy(mode):
    model, state, _ = setup("Y")
    conditions = ExternalConditions(field=[0, 0, 20 * G_Z * MU_B_MEV_PER_T * 0.2])
    result = select(model, state, conditions, mode)
    assert result.verdict == sel.NO_DEGENERACY
    assert result.diagnostics["generator_rank"] == 2


@pytest.mark.parametrize("mode", sel.MODES)
def test_triangular_120_needs_an_axis_and_then_has_no_selection(mode):
    model = triangular_heisenberg(J=1.0)
    state = state_120(model)
    result = select(model, state, None, mode, bz_type="Hex_60")
    assert result.verdict == sel.AXIS_REQUIRED
    assert result.diagnostics["flat_rotation_count"] == 3
    with_axis = select(model, state, None, mode, axis=[0, 0, 1], bz_type="Hex_60")
    assert with_axis.verdict == sel.NO_SELECTION


@pytest.mark.parametrize("mode", sel.MODES)
def test_square_polarized_state_is_rank_deficient_and_not_degenerate(mode):
    model = square_heisenberg(J=1.0)
    conditions = ExternalConditions(field=[0, 0, 5.0])
    result = select(model, polarized_state(model), conditions, mode, bz_type="Tetra")
    assert result.verdict == sel.NO_DEGENERACY
    assert result.diagnostics["generator_rank"] == 2
    about_field = select(model, polarized_state(model), conditions, mode, axis=[0, 0, 1],
                         bz_type="Tetra")
    assert about_field.verdict == sel.NO_SELECTION
    assert "invariant" in about_field.message


def test_orbit_search_reproduces_the_magswt_grid_and_improves_off_grid():
    for phi0, same in [(0.0, True), (0.3, False)]:
        model, state, conditions = setup("V", {"JPD": 0.01}, phi0=phi0)
        result = select(model, state, conditions)
        orbit_energy = result.diagnostics["E_cl_selected"] + result.diagnostics["E_qm_selected"]
        system = to_spin_system(model, state, conditions)
        energy = EnergyFunction(system.to_legacy_dict("Hex_30"), N=MESH_N)
        angles = system.get_angles_flat()
        grid = SpinOptimizer.magswt_for_nbcp(energy, angles,
                                             energy.classical_energy_density_func(angles))
        if same:
            assert grid["energy"] == pytest.approx(orbit_energy, abs=1e-14)
        else:
            assert grid["energy"] - orbit_energy > 1e-6


def test_fit_harmonics_recovers_coefficients():
    phis = np.arange(36) * 2 * np.pi / 36
    values = 0.2 + 1e-12 * np.cos(6 * phis - 0.4) + 3e-7 * np.sin(3 * phis)
    fit = sel.fit_harmonics(phis, values, 12)
    assert fit["amplitudes"][5] == pytest.approx(1e-12, rel=1e-3)
    assert fit["amplitudes"][2] == pytest.approx(3e-7, rel=1e-9)
    assert fit["residual"] < 1e-16


def test_rotation_symmetry_of_terms():
    model = nbcp.build_model({"Jxy": 0.075, "Jz": 0.125})
    field = ExternalConditions(field=[0, 0, 0.05])
    assert sel.rotation_symmetry(model, field, [0, 0, 1], 1e-12)[0]
    assert not sel.rotation_symmetry(model, field, [1, 0, 0], 1e-12)[0]
    gamma = nbcp.build_model({"Jxy": 0.075, "Jz": 0.125, "JGamma": 0.01})
    symmetric, violation = sel.rotation_symmetry(gamma, field, [0, 0, 1], 1e-12)
    assert not symmetric and violation["bilinear"] > 1e-3


def test_criteria_validation():
    with pytest.raises(ValueError, match="mode"):
        sel.SelectionCriteria(mode="auto")
    with pytest.raises(ValueError, match="orbit_points"):
        sel.SelectionCriteria(orbit_points=20, max_harmonic=12)
