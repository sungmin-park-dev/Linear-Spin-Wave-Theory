"""Spontaneous two-magnon decay rates in the Born approximation (D57).

``Gamma_n(k) = -Im Sigma_3^nn(k, w_n(k) + i0)`` by triangle integration of
the golden rule; the cubic vertex itself is checked against exact
diagonalization in test_nlswt.py.
"""

import numpy as np
import pytest

from spintoolkit.methods.lswt.run import solve_lswt
from spintoolkit.methods.nlswt import NLSWTSettings, solve_nlswt
from spintoolkit.methods.nlswt.engine import triangle_delta_integral
from spintoolkit.models.heisenberg import (neel_state, square_heisenberg, state_120,
                                           triangular_heisenberg)
from spintoolkit.observables.two_magnon import two_magnon_threshold
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions

DIAGONAL = np.array([1.0, 1.0]) / np.sqrt(2)


def _canted(model, fraction):
    """Square-lattice antiferromagnet in a field h = fraction * h_c, h_c = 8 J S."""
    h = 8.0 * model.sites[0].spin * fraction
    z, x = fraction, np.sqrt(1 - fraction ** 2)
    state = SpinState.from_function(
        model, [[1, 1], [1, -1]],
        lambda site, cell: np.array([x, 0, z]) if (cell[0] + cell[1]) % 2 == 0
        else np.array([-x, 0, z]), {"origin": "test"})
    return state, ExternalConditions(field=(0, 0, h))


def test_triangle_integral_matches_analytic_density_of_states():
    # int du dv delta(cos 2 pi u - c) = 1 / (pi sqrt(1 - c^2))
    n = 64
    u = (np.arange(n) + 1 / 3) / n
    U, _ = np.meshgrid(u, u, indexing="ij")
    value = triangle_delta_integral(np.cos(2 * np.pi * U) - 0.3, np.ones((n, n)))
    assert value == pytest.approx(1 / (np.pi * np.sqrt(0.91)), rel=5e-3)


def test_rate_is_the_zero_broadening_limit_of_the_self_energy():
    model = square_heisenberg(S=1.0)
    state, conditions = _canted(model, 0.9)
    result = solve_nlswt(model, state, conditions, settings=NLSWTSettings(mesh=(24, 24)))
    k = np.array([0.3, 0.1]) @ result.solver.mesh.reciprocal
    rates = result.decay_rates([k], grid=(64, 64))
    E, gamma = rates["lswt"][0], rates["decay_rate"][0]
    assert np.all(gamma > 0.2)
    # -Im Sigma_3(w + i eta) on the loop mesh: a broadened version of the same sum
    lorentzian = [-result.solver.cubic_self_energy(k, n, E[n] + 0.1j).imag for n in range(2)]
    assert np.allclose(lorentzian, gamma, rtol=0.1)


def test_no_cubic_vertex_no_decay():
    model = square_heisenberg(S=1.0)
    result = solve_nlswt(model, neel_state(model), settings=NLSWTSettings(mesh=(6, 6)))
    rates = result.decay_rates([[0.7, 0.2], [1.5, 1.5]], grid=(16, 16))
    assert np.all(rates["decay_rate"] == 0.0)


@pytest.mark.parametrize("fraction, decays", [(0.7, False), (0.74, False), (0.9, True)])
def test_square_antiferromagnet_decays_only_above_threshold_field(fraction, decays):
    # The Goldstone branch turns convex along the diagonal at h = 0.755929 h_c (found
    # numerically from the LSWT dispersion; equal to 2/sqrt 7 to 1e-6), so decays start there
    # (consistent with h* ~ 0.76 h_c, Zhitomirsky and Chernyshev, PRL 82, 4536 (1999)).
    model = square_heisenberg(S=1.0)
    state, conditions = _canted(model, fraction)
    result = solve_nlswt(model, state, conditions, settings=NLSWTSettings(mesh=(6, 6)))
    k = np.array([0.6, 1.5])[:, None] * DIAGONAL
    gamma = result.decay_rates(k, grid=(48, 48))["decay_rate"]
    if decays:
        assert gamma.max() > 0.1
    else:
        assert gamma.max() < 1e-20


def test_hidden_symmetry_forbids_kinematically_allowed_decay():
    # In the two-site cell of the canted state the upper (folded) band lies above the
    # two-magnon edge, but a translation times a pi spin rotation about the field
    # conserves the unfolded momentum: the vertex vanishes.
    model = square_heisenberg(S=1.0)
    state, conditions = _canted(model, 0.74)
    k = 0.6 * DIAGONAL
    edge = two_magnon_threshold(solve_lswt(model, state, conditions), [k], grid=(24, 24))[0]
    result = solve_nlswt(model, state, conditions, settings=NLSWTSettings(mesh=(6, 6)))
    rates = result.decay_rates([k], grid=(32, 32))
    upper = int(np.argmax(rates["lswt"][0]))
    assert rates["lswt"][0, upper] > edge + 1.0
    assert rates["decay_rate"][0, upper] < 1e-20


def test_triangular_rates_vanish_below_the_continuum_and_converge():
    model = triangular_heisenberg(S=1.0)
    state = state_120(model)
    result = solve_nlswt(model, state, settings=NLSWTSettings(mesh=(6, 6)))
    k = np.array([0.4, 0.3]) @ result.solver.mesh.reciprocal
    coarse = result.decay_rates([k], grid=(48, 48))
    fine = result.decay_rates([k], grid=(96, 96))
    edge = two_magnon_threshold(solve_lswt(model, state), [k], grid=(24, 24))[0]
    E, gamma = fine["lswt"][0], fine["decay_rate"][0]
    assert np.all(gamma[E < edge - 1e-6] == 0.0)
    assert np.any(gamma > 0.04)
    assert np.allclose(coarse["decay_rate"][0], gamma, rtol=0.05, atol=1e-12)
