"""Nonlinear spin waves to order S^0 (D46).

The decisive check is exactness on a finite torus: perturbation theory in 1/S
on the cluster momenta must reproduce the 1/S expansion of exact
diagonalization at large S, including the cubic and tadpole terms of a
noncollinear state in a field.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.methods.lswt.run import LSWTError
from spintoolkit.methods.nlswt import (NLSWTSettings, NonlinearExpansionError, expand_model,
                                       solve_nlswt)
from spintoolkit.methods.nlswt.engine import nambu_matrices
from spintoolkit.models.heisenberg import (neel_state, square_heisenberg, state_120,
                                           triangular_heisenberg)
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import Site, SpinModel, Term


def test_quadratic_order_reproduces_lswt_hamiltonian():
    model = triangular_heisenberg(S=1.5)
    state = state_120(model)
    conditions = ExternalConditions(field=(0.1, 0.2, 0.3))
    k = np.random.default_rng(0).normal(size=(6, 2))
    expansion = expand_model(model, state, conditions)
    H = QuadraticBoseHamiltonian(model, state, conditions).at(k)
    assert np.allclose(nambu_matrices(expansion.orders[2], k, expansion.num_sites), H, atol=1e-12)


def test_square_heisenberg_oguchi_constant():
    # E/N = -2 J S^2 - 0.315895 J S - 0.012474 J (Hamer, Zheng, Arndt 1992); collinear: no H_3
    model = square_heisenberg(S=1.0)
    result = solve_nlswt(model, neel_state(model), settings=NLSWTSettings(mesh=(24, 24)))
    e = result.energies
    assert e.classical == pytest.approx(-2.0)
    assert e.zero_point == pytest.approx(-0.315895, abs=5e-5)
    assert e.cubic == 0.0 and e.tadpole == 0.0
    assert e.hartree_fock == pytest.approx(-0.012474, abs=1e-5)


def test_triangular_heisenberg_order_s0_energy():
    # Chernyshev and Zhitomirsky, PRB 79, 144416 (2009), Eq. (42):
    # E/N = -(3/2) J S^2 [1 + 0.436824/(2S) + 0.02141/(2S)^2]; the S^0 term is -0.0080288 J.
    # The mesh converges to -0.008012 here (0.2% below the published integral).
    model = triangular_heisenberg(S=1.0)
    e = solve_nlswt(model, state_120(model), settings=NLSWTSettings(mesh=(12, 12))).energies
    assert e.zero_point == pytest.approx(-1.5 * 0.436824 / 2, rel=2e-4)
    assert e.order_s0 == pytest.approx(-0.0080288, rel=5e-3)
    assert abs(e.tadpole) < 1e-6
    assert e.cubic < 0 < e.hartree_fock


# ---- exactness on a two-spin torus -----------------------------------------------------

def _spin_matrices(S):
    m = S - np.arange(int(round(2 * S + 1)))
    plus = np.diag(np.sqrt(S * (S + 1) - m[1:] * (m[1:] + 1)), 1)
    return [(plus + plus.T) / 2, (plus - plus.T) / 2j, np.diag(m)]


def _cluster():
    rng = np.random.default_rng(2)
    offsets = [(0, 0), (-1, 0), (0, 1)]
    exchange = [np.eye(3) * rng.uniform(0.5, 1.5) + 0.6 * rng.normal(size=(3, 3)) for _ in offsets]
    h = 0.5 * rng.normal(size=3)
    J = sum(exchange)

    def energy(x):
        u = np.array([[np.sin(x[0]) * np.cos(x[1]), np.sin(x[0]) * np.sin(x[1]), np.cos(x[0])],
                      [np.sin(x[2]) * np.cos(x[3]), np.sin(x[2]) * np.sin(x[3]), np.cos(x[2])]])
        return u[0] @ J @ u[1] - h @ (u[0] + u[1]), u

    starts = np.random.default_rng(1).uniform(0, 2 * np.pi, (30, 4))
    best = min((minimize(lambda x: energy(x)[0], x0, method="BFGS") for x0 in starts),
               key=lambda r: r.fun)
    u = energy(best.x)[1]
    for _ in range(5000):                       # align each spin with its local field
        new = np.array([h - J @ u[1], h - J.T @ u[0]])
        u = new / np.linalg.norm(new, axis=1)[:, None]
    return offsets, exchange, h, u


def _ed_levels(S, J, h, count):
    ops = _spin_matrices(S)
    eye = np.eye(len(ops[0]))
    H = sum(J[a, b] * np.kron(ops[a], ops[b]) for a in range(3) for b in range(3))
    H = H - S * sum(h[a] * (np.kron(ops[a], eye) + np.kron(eye, ops[a])) for a in range(3))
    return np.linalg.eigvalsh(H)[:count]


def test_finite_torus_is_exact_large_s_perturbation_theory():
    offsets, exchange, h, u = _cluster()
    J = sum(exchange)

    def solve(S):
        terms = [Term.bilinear(("A", (0, 0)), ("B", off), Jb) for off, Jb in zip(offsets, exchange)]
        terms += [Term.zeeman(s, np.eye(3)) for s in "AB"]
        model = SpinModel(np.eye(2), [Site("A", (0, 0), S), Site("B", (0.5, 0.1), S)], terms,
                          {"model_id": "two_spin_cluster"})
        state = SpinState.from_function(model, np.eye(2, dtype=int),
                                        lambda s, c: u[0] if s == "A" else u[1], {"origin": "test"})
        return solve_nlswt(model, state, ExternalConditions(field=S * h),
                           CalculationGeometry.finite_torus(np.eye(2, dtype=int)))

    result = solve(1.0)
    e = result.energies
    assert abs(e.cubic) > 1e-3 and abs(e.tadpole) > 1e-4      # noncollinear in a field
    magnons = result.magnon_energies(np.zeros((1, 2)))
    w = magnons["lswt"][0]
    correction = magnons["hartree_fock_tadpole"][0] + magnons["cubic"][0].real

    spins = np.arange(8, 25, 2) / 1.0
    residual, gaps = [], []
    for S in spins:
        levels = _ed_levels(S, J, h, 6)
        residual.append(levels[0] / 2 - e.classical * S ** 2 - e.zero_point * S)
        excitations = levels[1:] - levels[0]
        gaps.append([excitations[np.argmin(abs(excitations - (w[n] * S + correction[n])))]
                     - w[n] * S for n in range(2)])
    A = np.vander(1 / spins, 6, increasing=True)
    e0 = np.linalg.lstsq(A, np.array(residual), rcond=None)[0][0]
    assert e0 == pytest.approx(e.order_s0, abs=1e-6)
    A = np.vander(1 / spins, 4, increasing=True)
    for n in range(2):
        delta = np.linalg.lstsq(A, np.array(gaps)[:, n], rcond=None)[0][0]
        assert delta == pytest.approx(correction[n], abs=2e-4)


def test_refuses_onsite_and_non_stationary_states():
    model = square_heisenberg(S=1.0)
    with pytest.raises(LSWTError, match="not stationary"):
        solve_nlswt(model, neel_state(model, (0, 0, 1)), ExternalConditions(field=(0.3, 0, 0)),
                    settings=NLSWTSettings(mesh=(6, 6)))
    onsite = SpinModel(model.lattice, model.sites,
                       list(model.terms) + [Term.onsite("A", np.diag([0, 0, -0.1]))],
                       {"model_id": "with_onsite"})
    with pytest.raises(NonlinearExpansionError, match="onsite"):
        solve_nlswt(onsite, neel_state(onsite), settings=NLSWTSettings(mesh=(6, 6)))
