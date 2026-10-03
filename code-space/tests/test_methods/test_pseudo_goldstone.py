"""Pseudo-Goldstone gap to next order in 1/S (D47)."""

import numpy as np
import pytest

from model import nbcp
from spintoolkit.methods.nlswt import PseudoGoldstoneSettings, pseudo_goldstone_gap, rotate_state
from spintoolkit.methods.nlswt.run import solve_nlswt, NLSWTSettings
from spintoolkit.system.conditions import ExternalConditions

S, J, JZ, H = 0.5, 0.075, 0.125, 0.05377406697774


def _y_state(jpd):
    model = nbcp.build_model({'Jxy': J, 'Jz': JZ, 'JPD': jpd})
    t = np.arccos((H + 3 * S * JZ) / (3 * S * (J + JZ)))
    angles = np.column_stack([[t, -t, np.pi], np.zeros(3)]).ravel()
    return model, nbcp.candidate_state(model, 'three_msl', angles), ExternalConditions(field=[0, 0, H])


def test_self_energy_matrix_holds_the_band_self_energies():
    model, state, cond = _y_state(0.010)
    solver = solve_nlswt(model, state, cond, settings=NLSWTSettings(mesh=(6, 6))).solver
    k = np.array([0.31, -0.17])
    E, T = solver._bogoliubov(k)
    for n in range(3):
        tilde = T[0].conj().T @ solver.cubic_self_energy_matrix(k, E[0, n] + 0j) @ T[0]
        assert tilde[n, n] == pytest.approx(solver.cubic_self_energy(k, n, E[0, n]), abs=1e-12)
    sigma = solver.cubic_self_energy_matrix(np.zeros(2), 0j)
    assert np.allclose(sigma, sigma.conj().T, atol=1e-14)


def test_rotate_state_rotates_every_spin():
    model, state, _ = _y_state(0.010)
    rotated = rotate_state(model, state, (0, 0, 1), np.pi / 2)
    for cell in state.cells:
        x, y, z = state.direction('Co', cell)
        assert np.allclose(rotated.direction('Co', cell), [-y, x, z])


def test_exact_symmetry_has_no_gap_at_either_order():
    model, state, cond = _y_state(0.0)          # XXZ in a z field: rotation about z is a symmetry
    r = pseudo_goldstone_gap(model, state, (0, 0, 1), cond, PseudoGoldstoneSettings(mesh=(6, 6)))
    assert r.gap_squared == (0.0, 0.0)
    assert abs(r.curvature['sigma_xx']) < 1e-12


def test_leading_order_is_the_curvature_over_chi_relation():
    model, state, cond = _y_state(0.010)
    r = pseudo_goldstone_gap(model, state, (0, 0, 1), cond, PseudoGoldstoneSettings(mesh=(6, 6)))
    assert r.header.diagnostics['ward_identity_relative_error'] < 1e-4
    chi_per_spin = 10 / 9      # classical chi_z of this Y state (scan-N48-P72.json in 260912-pseudo-goldstone)
    c_phi_per_spin = r.curvature['zero_point'] / 3
    assert r.gap_squared[0] == pytest.approx(c_phi_per_spin / chi_per_spin, rel=2e-3)
    assert r.relative_correction < -1                 # the two-loop curvature dominates (D47)
