"""Luttinger-Tisza diagnostic (stage 6b, D30).

1. J(q): the Fourier sum over the momenta of a magnetic supercell equals the
   classical energy of any state on it (random directions, several cells).
2. Benchmarks: square FM and Neel, triangular 120 degrees (K, sqrt3 x sqrt3,
   -3JS^2/2), J1-J2 square (Neel, stripe, degenerate lines at J2 = J1/2),
   honeycomb FM with DM (zone centre), Kitaev (flat band, -|K|S^2/2).
3. The bound is below the classical energy; where the single-q strong
   constraint holds, the constructed state is valid and reaches it; the NBCP
   XXZ model fails the single-q constraint.
"""

import numpy as np
import pytest

from model import nbcp
from spintoolkit.methods.classical import classical_energy
from spintoolkit.methods.luttinger_tisza import _supercell, lt_matrix, luttinger_tisza
from spintoolkit.models import (
    honeycomb_ferromagnet, kitaev_honeycomb, square_heisenberg, state_120, triangular_heisenberg)
from spintoolkit.states.spin_state import SpinState, validate_spin_state
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import Site, SpinModel, Term


def unit(rng):
    v = rng.standard_normal(3)
    return v / np.linalg.norm(v)


def j1j2(ratio):
    J = np.eye(3)
    terms = ([Term.bilinear(("A", (0, 0)), ("A", o), J) for o in [(1, 0), (0, 1)]]
             + [Term.bilinear(("A", (0, 0)), ("A", o), ratio * J) for o in [(1, 1), (1, -1)]])
    return SpinModel(np.eye(2), [Site("A", (0, 0), 0.5)], terms, {"model_id": f"j1j2_{ratio}"})


def fourier_energy(model, state):
    """Energy per site from J(q) on the momenta of the state's supercell."""
    M = np.asarray(state.supercell, dtype=float)
    magnetic = M @ model.lattice
    reciprocal = 2 * np.pi * np.linalg.inv(magnetic).T
    primitive = 2 * np.pi * np.linalg.inv(model.lattice).T
    n_cells = int(round(abs(np.linalg.det(M))))
    cells = list(state.cells)
    q_all = []
    for i in range(-n_cells, n_cells + 1):
        for j in range(-n_cells, n_cells + 1):
            q = i * reciprocal[0] + j * reciprocal[1]
            f = np.mod(np.linalg.solve(primitive.T, q) + 1e-12, 1.0)
            if not any(np.allclose(f, g, atol=1e-9) for g, _ in q_all):
                q_all.append((f, q))
    assert len(q_all) == n_cells
    ns = model.num_sites
    total = 0.0
    for _, q in q_all:
        n = np.zeros(3 * ns, dtype=complex)
        for a, site in enumerate(model.site_ids):
            n[3 * a:3 * a + 3] = sum(np.exp(-1j * q @ model.cartesian_position(site, c))
                                     * state.direction(site, c) for c in cells) / np.sqrt(n_cells)
        total += np.real(np.conj(n) @ lt_matrix(model, q)[0] @ n)
    return total / (n_cells * ns)


@pytest.mark.parametrize("model, cell", [
    (triangular_heisenberg(), [[1, 1], [-1, 2]]),
    (j1j2(0.4), [[2, 0], [0, 2]]),
    (honeycomb_ferromagnet(D=0.3), [[2, 1], [0, 1]]),
    (kitaev_honeycomb(K=0.7), [[1, 0], [1, 2]]),
])
def test_lt_matrix_reproduces_the_classical_energy_of_any_commensurate_state(model, cell):
    rng = np.random.default_rng(3)
    state = SpinState.from_function(model, cell, lambda site, c: unit(rng), {})
    assert abs(fourier_energy(model, state) - classical_energy(model, state, None)) < 1e-12


def test_ferromagnet_and_neel_on_the_square_lattice():
    ferro = luttinger_tisza(square_heisenberg(J=-1.0), mesh=(12, 12))
    neel = luttinger_tisza(square_heisenberg(J=1.0), mesh=(12, 12))
    assert ferro.lambda_min == pytest.approx(-0.5) and neel.lambda_min == pytest.approx(-0.5)
    assert ferro.minima[0].fraction == ("0", "0")
    assert neel.minima[0].fraction == ("1/2", "1/2")
    assert abs(round(np.linalg.det(neel.minima[0].supercell))) == 2
    assert ferro.strong_constraint and neel.strong_constraint


def test_triangular_antiferromagnet_gives_the_120_degree_cell_and_energy():
    model = triangular_heisenberg(J=1.0, S=0.5)
    report = luttinger_tisza(model, mesh=(24, 24))
    assert report.lambda_min == pytest.approx(-3 * 0.25 / 2, abs=1e-12)
    assert len(report.minima) == 1 and not report.extended_degeneracy
    q = report.minima[0]
    assert q.commensurate and abs(round(np.linalg.det(q.supercell))) == 3
    assert q.strong_constraint and q.multiplicity == 3
    validate_spin_state(q.state, model, CalculationGeometry.thermodynamic_limit())
    assert q.state_energy == pytest.approx(report.lambda_min, abs=1e-12)
    assert classical_energy(model, state_120(model), None) == pytest.approx(q.state_energy, abs=1e-12)


def test_j1_j2_square_neel_stripe_and_degenerate_lines():
    neel, stripe, line = (luttinger_tisza(j1j2(r), mesh=(24, 24)) for r in (0.3, 0.7, 0.5))
    assert neel.lambda_min == pytest.approx(0.25 * (-2 + 2 * 0.3))
    assert neel.minima[0].fraction == ("1/2", "1/2")
    assert stripe.lambda_min == pytest.approx(0.25 * (-2 * 0.7))
    assert {m.fraction for m in stripe.minima} == {("1/2", "0"), ("0", "1/2")}
    assert not neel.extended_degeneracy and not stripe.extended_degeneracy
    assert line.extended_degeneracy and line.lambda_min == pytest.approx(-0.25)


def test_honeycomb_ferromagnet_and_kitaev():
    honeycomb = luttinger_tisza(honeycomb_ferromagnet(J=1.0, D=0.2), mesh=(12, 12))
    assert honeycomb.lambda_min == pytest.approx(-1.5 * 0.25)
    assert honeycomb.minima[0].fraction == ("0", "0") and honeycomb.strong_constraint
    kitaev = luttinger_tisza(kitaev_honeycomb(K=-1.0), mesh=(12, 12))
    assert kitaev.lambda_min == pytest.approx(-0.25 / 2)
    assert kitaev.extended_degeneracy and kitaev.near_minimal_fraction == pytest.approx(1.0)


def test_bound_lies_below_the_classical_energy_and_nbcp_fails_the_single_q_constraint():
    model = nbcp.build_model({"Jxy": 0.075, "Jz": 0.125})
    report = luttinger_tisza(model, mesh=(24, 24))
    assert not report.strong_constraint
    assert report.minima[0].fraction == ("1/3", "1/3") and report.minima[0].state is None
    rng = np.random.default_rng(0)
    for _ in range(5):
        state = SpinState.from_function(model, [[1, 1], [-1, 2]],
                                        lambda site, c: unit(rng), {})
        assert classical_energy(model, state, None) >= report.lambda_min - 1e-12


def test_quarter_wave_vector_accepts_the_up_up_down_down_state():
    """4q* in G but 2q* not: only Re(u.u) = 0 is required, so the Ising uudd state
    (not a spiral) reaches the bound; the former u.u = 0 test rejected it."""
    ising = np.diag([0.2, 0.2, 1.0])
    terms = [Term.bilinear(("A", (0, 0)), ("A", (2, 0)), ising),
             Term.bilinear(("A", (0, 0)), ("A", (0, 1)), -ising)]
    model = SpinModel(np.eye(2), [Site("A", (0, 0), 0.5)], terms, {"model_id": "quarter_ising"})
    report = luttinger_tisza(model, mesh=(24, 24))
    assert report.lambda_min == pytest.approx(-0.5)
    minimum = report.minima[0]
    assert minimum.fraction in {("1/4", "0"), ("3/4", "0")} and minimum.multiplicity == 1
    assert minimum.strong_constraint and minimum.state_energy == pytest.approx(-0.5)
    directions = np.array(list(minimum.state.directions.values()))
    assert np.allclose(np.abs(directions[:, 2]), 1.0)
    assert sorted(np.sign(directions[:, 2])) == [-1, -1, 1, 1]


def test_supercell_is_the_smallest_cell_holding_the_wave_vector():
    from fractions import Fraction
    for f in [(Fraction(1, 3), Fraction(1, 3)), (Fraction(1, 2), Fraction(0)),
              (Fraction(1, 4), Fraction(3, 4)), (Fraction(2, 5), Fraction(1, 5))]:
        cell = _supercell(list(f))
        q = np.array([float(x) for x in f])
        assert np.allclose(np.mod(cell @ q + 1e-12, 1.0), 0, atol=1e-9)
        index = int(np.lcm.reduce([x.denominator for x in f]))
        index //= int(np.gcd.reduce([int(x * index) for x in f] + [index]))
        assert abs(round(np.linalg.det(cell))) == index
