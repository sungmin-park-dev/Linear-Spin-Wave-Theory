"""Dynamical structure factor through order S^0 (D59).

The decisive check is exactness on a finite torus: the order-S^0 weights of
every one- and two-magnon state of a noncollinear two-spin cluster in a field
(cubic vertex and tadpole present) must equal the 1/S expansion of exact
diagonalization at large S. The thermodynamic-limit evaluation is checked
against the same formulas on a mesh where no magnon decays, against D56 and
D58, and against the exact sum rule of the collinear antiferromagnet.
"""

import warnings

import numpy as np
import pytest

from spintoolkit.methods.lswt.run import solve_lswt
from spintoolkit.methods.nlswt import NLSWTSettings, solve_nlswt
from spintoolkit.methods.nlswt.structure import _diagonal, spin_operator
from spintoolkit.models.heisenberg import (neel_state, polarized_state, square_heisenberg,
                                           state_120, triangular_heisenberg)
from spintoolkit.observables.interacting_spectrum import (structure_factor_s0,
                                                          structure_factor_states)
from spintoolkit.observables.two_magnon import two_magnon_continuum
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import Site, SpinModel, Term

from test_magnon_decay import _canted
from test_nlswt import _cluster, _spin_matrices

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

POSITIONS = np.array([[0.0, 0.0], [0.5, 0.1]])


def _cluster_result(S):
    offsets, exchange, h, u = _cluster()
    terms = [Term.bilinear(("A", (0, 0)), ("B", off), Jb) for off, Jb in zip(offsets, exchange)]
    terms += [Term.zeeman(s, np.eye(3)) for s in "AB"]
    model = SpinModel(np.eye(2), [Site("A", (0, 0), S), Site("B", tuple(POSITIONS[1]), S)], terms,
                      {"model_id": "two_spin_cluster"})
    state = SpinState.from_function(model, np.eye(2, dtype=int),
                                    lambda s, c: u[0] if s == "A" else u[1], {"origin": "test"})
    return solve_nlswt(model, state, ExternalConditions(field=S * h),
                       CalculationGeometry.finite_torus(np.eye(2, dtype=int)))


def _ed_weights(S, qs, count=40):
    """Excitation energies and ``<0|S^a(q)|f><f|S^b(q)^+|0>`` of the two-spin cluster."""
    _, exchange, h, _ = _cluster()
    J = sum(exchange)
    ops = _spin_matrices(S)
    eye = np.eye(len(ops[0]))
    H = sum(J[a, b] * np.kron(ops[a], ops[b]) for a in range(3) for b in range(3))
    H = H - S * sum(h[a] * (np.kron(ops[a], eye) + np.kron(eye, ops[a])) for a in range(3))
    w, V = np.linalg.eigh(H)
    V = V[:, :count]
    out = []
    for q in qs:
        phase = np.exp(-1j * POSITIONS @ q)
        Sq = [(phase[0] * np.kron(ops[a], eye) + phase[1] * np.kron(eye, ops[a])) / np.sqrt(2)
              for a in range(3)]
        amp = np.array([V.conj().T @ (Sq[a].conj().T @ V[:, 0]) for a in range(3)])
        out.append((w[:count] - w[0], np.einsum("af,bf->fab", amp.conj(), amp)))
    return out


def test_finite_torus_weights_are_the_large_s_expansion_of_exact_diagonalization():
    qs = [np.zeros(2), np.array([2 * np.pi, 0.0])]          # q = 0 and a reciprocal vector
    states = structure_factor_states(_cluster_result(1.0), qs)
    spins = np.arange(10, 32, 3) / 1.0
    ed = [_ed_weights(S, qs) for S in spins]
    A = np.vander(1 / spins, 5, increasing=True)
    fit = np.linalg.pinv(A)
    for iq in range(len(qs)):
        E1 = states["one_magnon_energy"][iq]
        W1, L1 = states["one_magnon_weight"][iq], states["lswt_weight"][iq]
        assert np.max(np.abs(W1 - L1)) > 0.03                  # sizeable order-S^0 corrections
        for n in range(2):
            series = []
            for S, per_q in zip(spins, ed):
                ex, W = per_q[iq]
                f = np.argmin(np.abs(ex[1:] - S * E1[n].real)) + 1
                series.append(W[f] / S)
            coef = np.einsum("ij,jab->iab", fit, np.array(series))
            assert np.max(np.abs(coef[0] - L1[n])) < 5e-6                 # LSWT, order S
            assert np.max(np.abs(coef[1] - (W1[n] - L1[n]))) < 3e-4       # order S^0
        for EP, WP in zip(states["two_magnon_energy"][iq], states["two_magnon_weight"][iq]):
            series = []
            for S, per_q in zip(spins, ed):
                ex, W = per_q[iq]
                near = np.flatnonzero(np.abs(ex - EP * S) < 5)     # pair energy shifts are O(1)
                f = near[np.argmax(np.einsum("faa->f", W[near]).real)]
                series.append(W[f])
            coef = np.einsum("ij,jab->iab", fit, np.array(series))
            assert np.max(np.abs(coef[0] - WP)) < 1e-4


def test_square_antiferromagnet_obeys_the_total_moment_sum_rule_exactly():
    # sum over q of all inelastic weight + Bragg = S(S+1). With <n> = delta (no 1/S correction
    # for the Neel state: the Hartree-Fock terms only rescale H(k)), the order-S^0 one-magnon
    # correction averaged over the zone must be -delta^2 - <(a^+ a)^2>_c.
    model = square_heisenberg(S=1.0)
    result = solve_nlswt(model, neel_state(model), settings=NLSWTSettings(mesh=(8, 8)))
    solver = result.solver
    origin = tuple(float(x) for x in solver.expansion.positions[0])
    n = solver.contract((0, True, origin), (0, False, origin)).real
    pair = solver.contract((0, False, origin), (0, False, origin))
    lswt, correction, count = 0.0, 0.0, 0
    for G in (np.zeros(2), np.array([np.pi, np.pi])):
        for q in solver.mesh.k + G:
            A, dA, _ = spin_operator(solver, q)
            _, T = _diagonal(solver, q)
            a, da = (A @ T)[:, :solver.ns], (dA @ T)[:, :solver.ns]
            lswt += np.sum(np.abs(a) ** 2)
            correction += 2 * np.sum((da * a.conj()).real)
            count += 1
    assert lswt / count == pytest.approx(1 + 2 * n, abs=1e-12)
    assert correction / count == pytest.approx(-n ** 2 - (abs(pair) ** 2 + n * (n + 1)), abs=1e-12)


def test_ferromagnet_has_no_corrections():
    model = square_heisenberg(J=-1.0, S=1.0)
    result = solve_nlswt(model, polarized_state(model), ExternalConditions(field=(0, 0, 0.5)),
                         settings=NLSWTSettings(mesh=(9, 9)))
    w = np.linspace(0.01, 10, 500)
    out = structure_factor_s0(result, [[0.7, 0.2]], w, 0.05, grid=(18, 18))
    assert np.allclose(out["weight"], out["lswt_weight"], atol=1e-12)
    assert np.allclose(out["energy"].real, out["lswt_energy"], atol=1e-12)
    assert np.max(np.abs(out["continuum"])) < 1e-12


def test_neel_continuum_is_the_longitudinal_two_magnon_continuum():
    model = square_heisenberg(S=1.0)
    result = solve_nlswt(model, neel_state(model), settings=NLSWTSettings(mesh=(16, 16)))
    q = np.array([[1.1, 0.3]])
    w = np.linspace(0.001, 12, 6000)
    out = structure_factor_s0(result, q, w, 0.05, grid=(64, 64), levels=600)
    static = two_magnon_continuum(solve_lswt(model, neel_state(model)), q, mesh=(64, 64)).static()
    assert np.sum(out["continuum"][0]) * (w[1] - w[0]) == pytest.approx(np.trace(static[0]).real,
                                                                         rel=0.03)
    # collinear: real poles, weights reduced by the operator corrections
    assert np.all(np.abs(out["energy"].imag) < 1e-12)
    ratio = np.einsum("qnaa->qn", out["weight"]).real / np.einsum("qnaa->qn", out["lswt_weight"]).real
    assert np.all((0.7 < ratio) & (ratio < 0.95))


def test_thermodynamic_limit_matches_the_mesh_sum_where_nothing_decays():
    # canted square antiferromagnet at h = 0.5 h_c: cubic vertices but no decay at this q
    model = square_heisenberg(S=1.0)
    state, conditions = _canted(model, 0.5)
    q = np.array([[1.1, 0.3]])
    w = np.linspace(0.001, 30, 12000)
    result = solve_nlswt(model, state, conditions, settings=NLSWTSettings(mesh=(24, 24)))
    out = structure_factor_s0(result, q, w, 0.05, grid=(64, 64), levels=600)
    states = structure_factor_states(result, q)
    assert np.allclose(out["energy"][0].imag, 0.0, atol=1e-12)
    assert np.allclose(out["energy"][0].real, states["one_magnon_energy"][0].real, atol=0.02)
    assert np.allclose(np.einsum("naa->n", out["weight"][0]).real,
                       np.einsum("naa->n", states["one_magnon_weight"][0]).real, rtol=0.02)
    continuum = np.sum(out["continuum"][0]) * (w[1] - w[0])
    two = np.einsum("paa->", states["two_magnon_weight"][0]).real
    assert continuum == pytest.approx(two, rel=0.02)


def test_decaying_magnons_take_the_energies_and_widths_of_d58():
    model = triangular_heisenberg(S=5.0)
    result = solve_nlswt(model, state_120(model), settings=NLSWTSettings(mesh=(9, 9)))
    q = np.array([[1.3, 0.4]])
    w = np.linspace(0.01, 45, 4000)
    out = structure_factor_s0(result, q, w, 0.15, grid=(36, 36))
    dressed = result.spectral_function(q, w, 0.15, grid=(36, 36))
    assert np.allclose(out["energy"][0].real, dressed["energy"][0], atol=1e-8)
    assert np.allclose(-out["energy"][0].imag, dressed["decay_rate"][0], atol=1e-8)
    assert np.max(dressed["decay_rate"][0]) > 0.03                 # two bands decay here
    total = np.einsum("naa->n", out["weight"][0]).real
    bare = np.einsum("naa->n", out["lswt_weight"][0]).real
    assert np.all((0.8 < total / bare) & (total / bare < 0.95))     # relative 1/S at S = 5


def test_input_checks():
    model = square_heisenberg(S=1.0)
    result = solve_nlswt(model, neel_state(model), settings=NLSWTSettings(mesh=(6, 6)))
    with pytest.raises(ValueError, match="eta"):
        structure_factor_s0(result, [[1.0, 0.2]], [1.0], 0.0)
    with pytest.raises(ValueError, match="zero mode"):
        structure_factor_s0(result, [[0.0, 0.0]], [1.0], 0.1, grid=(6, 6))
