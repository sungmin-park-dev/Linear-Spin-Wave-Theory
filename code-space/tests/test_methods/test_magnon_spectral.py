"""Dressed magnon spectral functions and the interacting structure factor (D58)."""

import numpy as np
import pytest

from spintoolkit.methods.lswt.run import solve_lswt
from spintoolkit.methods.nlswt import NLSWTSettings, solve_nlswt
from spintoolkit.methods.nlswt.engine import hilbert_transform, triangle_density
from spintoolkit.models.heisenberg import (neel_state, square_heisenberg, state_120,
                                           triangular_heisenberg)
from spintoolkit.observables.interacting_spectrum import dressed_structure_factor

K_DECAY = (0.4, 0.3)        # fractional momenta of the 120-degree magnetic zone
K_EDGE = (0.2, 0.1)


def _triangular(S, mesh=12):
    model = triangular_heisenberg(S=S)
    state = state_120(model)
    result = solve_nlswt(model, state, settings=NLSWTSettings(mesh=(mesh, mesh)))
    return model, state, result, result.solver.mesh.reciprocal


def test_hilbert_transform_of_a_semicircle():
    # rho = (2/pi) sqrt(1 - x^2): int rho / (w - x + i0) = 2 (w - sqrt(w^2 - 1)) for |w| > 1,
    # 2 w - 2 i sqrt(1 - w^2) inside
    x = np.linspace(-1, 1, 4001)
    rho = 2 / np.pi * np.sqrt(np.clip(1 - x ** 2, 0, None))
    w = np.array([-1.7, -0.4, 0.3, 1.5])
    exact = np.where(np.abs(w) > 1, 2 * (w - np.sign(w) * np.sqrt(np.abs(w ** 2 - 1))),
                     2 * w - 2j * np.sqrt(np.clip(1 - w ** 2, 0, None)))
    assert np.allclose(hilbert_transform(x, rho, w), exact, atol=2e-3)


def test_triangle_density_integrates_to_the_mean_weight():
    n = 48
    u = (np.arange(n) + 1 / 3) / n
    U, V = np.meshgrid(u, u, indexing="ij")
    f = np.cos(2 * np.pi * U) + np.cos(2 * np.pi * V)
    g = 1 + np.sin(2 * np.pi * U) ** 2
    x = np.linspace(-2.1, 2.1, 2001)
    rho = triangle_density(f, g, x)
    assert np.sum(rho) * (x[1] - x[0]) == pytest.approx(np.mean(g), rel=1e-3)


def test_on_shell_rate_and_shift_match_the_order_one_over_s_results():
    _, _, result, reciprocal = _triangular(1.0)
    k = np.array([K_DECAY]) @ reciprocal
    spectral = result.spectral_function(k, np.linspace(0, 4, 5), 0.01, grid=(48, 48))
    # the same triangle integral, read off the density sampled on 400 energies
    assert np.allclose(spectral["decay_rate"][0], result.decay_rates(k, (48, 48))["decay_rate"][0],
                       rtol=0.01, atol=1e-12)
    # band below the continuum: the mesh principal value converges to the same shift
    lswt = spectral["lswt"][0]
    n = int(np.argmin(lswt))
    mesh = result.magnon_energies(k)
    assert spectral["energy"][0, n] == pytest.approx(mesh["total"][0, n], abs=0.02)


def test_dyson_sum_rule_at_large_spin_and_flag_at_spin_half():
    for S, unphysical in ((5.0, False), (0.5, True)):
        _, _, result, reciprocal = _triangular(S)
        k = np.array([K_DECAY, K_EDGE]) @ reciprocal
        w = np.linspace(-(8 * S + 4), 8 * S + 4, 40001)
        out = result.spectral_function(k, w, 0.02, mode="dyson", grid=(48, 48), levels=400)
        total = np.sum(out["spectral"], axis=-1) * (w[1] - w[0])
        if unphysical:
            # resummation of a large S^0 self-energy: a pole with positive weight below w = 0
            assert out["negative_frequency_weight"].max() > 0.1
        else:
            # Lorentzian tails beyond the window remove about 2 eta / (pi * 40) ~ 3e-4
            assert np.allclose(total, 1.0, atol=0.01)
            assert out["negative_frequency_weight"].max() <= 0.0


def test_no_cubic_vertex_gives_a_shifted_lorentzian():
    model = square_heisenberg(S=1.0)
    result = solve_nlswt(model, neel_state(model), settings=NLSWTSettings(mesh=(6, 6)))
    k = np.array([[0.7, 0.2]])
    w = np.linspace(0, 6, 3001)
    out = result.spectral_function(k, w, 0.05, mode="dyson", grid=(16, 16), levels=100)
    shell = result.spectral_function(k, w, 0.05, grid=(16, 16), levels=100)
    assert np.all(out["decay_rate"] == 0.0)
    assert np.allclose(out["spectral"], shell["spectral"], atol=1e-10)


def test_dressed_structure_factor_conserves_the_one_magnon_weight_on_shell():
    model, state, result, reciprocal = _triangular(1.0)
    lswt = solve_lswt(model, state)
    q = np.array([K_DECAY]) @ reciprocal
    w = np.linspace(-20, 30, 20001)
    out = dressed_structure_factor(lswt, result, q, w, 0.02, grid=(32, 32), levels=300,
                                   two_magnon_mesh=(12, 12))
    dw = w[1] - w[0]
    # Lorentzian tails beyond the grid: weight / pi * 2 * width / span
    assert np.sum(out["one_magnon"]) * dw == pytest.approx(np.sum(out["lswt"]) * dw, rel=0.02)
    assert np.all(out["two_magnon"] >= 0)
    assert np.allclose(out["total"], out["one_magnon"] + out["two_magnon"])
