"""Check the cubic and quartic boson terms of the research note *LSWT nonlinear spin waves*.

The note writes the three- and four-boson terms of a bilinear spin Hamiltonian
per link with the circular components ``J~^{pq} = (e_I^p)^T J e_J^q``:

    H_3 = - sum_q [ sqrt(S_J) J~^{0q} n_I a_J^q + sqrt(S_I) J~^{q0} a_I^q n_J ],
    H_4 = J~^{00} n_I n_J
          - 1/4 sum_{p,q} J~^{pq} [ sqrt(S_J/S_I) A_I^p a_J^q + sqrt(S_I/S_J) a_I^p A_J^q ],

with ``a^- = a``, ``a^+ = a^dagger``, ``A^- = a^dagger a a`` and
``A^+ = a^dagger a^dagger a``; at a stationary configuration the cubic
Holstein-Primakoff corrections times the local field cancel. This script

1. builds H_3 and H_4 from these formulas for a two-sublattice model with
   random nonsymmetric exchange on inter- and same-sublattice links, spins
   S = 1 and 3/2, a random field and a stationary configuration, and compares
   them monomial by monomial with ``expand_model`` (D46);
2. checks that the linear (tadpole) coefficient eta_mu equals the derivative
   of the LSWT zero-point energy per cell under a uniform rotation of the
   classical directions, dE_zp / d alpha_mu^*, on a 12 x 12 mesh.

Run with ``python examples/verification/lswt_nonlinear_vertices_check.py``.
"""

from collections import defaultdict

import numpy as np

from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian, local_frame
from spintoolkit.methods.nlswt import NLSWTSettings, expand_model, solve_nlswt
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Site, SpinModel, Term


def build_case(seed=5):
    rng = np.random.default_rng(seed)
    spin = {"A": 1.0, "B": 1.5}
    pos = {"A": np.array([0.0, 0.0]), "B": np.array([0.5, 0.1])}
    offsets = [(0, 0), (-1, 0), (0, 1)]
    exchange = [np.eye(3) * rng.uniform(0.5, 1.5) + 0.6 * rng.normal(size=(3, 3)) for _ in offsets]
    bonds = [("A", (0, 0), "B", off, J) for off, J in zip(offsets, exchange)]
    bonds += [("A", (0, 0), "A", (1, 0), 0.4 * rng.normal(size=(3, 3))),
              ("B", (0, 0), "B", (0, 1), 0.4 * rng.normal(size=(3, 3)))]
    h = 0.7 * rng.normal(size=3)
    terms = [Term.bilinear((a, c1), (b, c2), J) for a, c1, b, c2, J in bonds]
    terms += [Term.zeeman(s, np.eye(3)) for s in "AB"]
    model = SpinModel(np.eye(2), [Site("A", (0, 0), spin["A"]), Site("B", (0.5, 0.1), spin["B"])],
                      terms, {"model_id": "nonlinear_vertex_check"})

    def gradient(u):                       # dE_cl / dS_I = -H_I
        g = {"A": -h.copy(), "B": -h.copy()}
        for a, _, b, _, J in bonds:
            g[a] += J @ (spin[b] * u[b])
            g[b] += J.T @ (spin[a] * u[a])
        return g

    u = {s: v / np.linalg.norm(v) for s, v in zip("AB", rng.normal(size=(2, 3)))}
    for _ in range(20000):                 # align each spin against its energy gradient
        g = gradient(u)
        u = {s: -g[s] / np.linalg.norm(g[s]) for s in "AB"}
    g = gradient(u)
    torque = max(np.linalg.norm(np.cross(u[s], g[s])) for s in "AB")
    return model, bonds, spin, pos, h, u, torque


def make_state(model, directions):
    return SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: directions[s],
                                   {"origin": "nonlinear_vertex_check"})


def circular_basis(direction):
    R = local_frame(direction)
    return {"+": (R[:, 0] + 1j * R[:, 1]) / np.sqrt(2), "-": (R[:, 0] - 1j * R[:, 1]) / np.sqrt(2),
            "0": R[:, 2]}


def canonical(monomial, home):
    """Monomial up to operator order and a common lattice translation."""
    best = None
    for site, _, r in monomial:
        shift = home[site] - np.asarray(r)
        ops = tuple(sorted((s, d, (round(x[0] + shift[0], 9) + 0.0, round(x[1] + shift[1], 9) + 0.0))
                           for s, d, x in monomial))
        best = ops if best is None or ops < best else best
    return best


def note_vertices(bonds, spin, pos, u, home):
    index = {"A": 0, "B": 1}
    cubic, quartic = defaultdict(complex), defaultdict(complex)

    def op(s, dagger, r):
        return (index[s], dagger, (float(r[0]), float(r[1])))

    for a, c1, b, c2, J in bonds:
        rI, rJ = pos[a] + np.array(c1), pos[b] + np.array(c2)
        eI, eJ = circular_basis(u[a]), circular_basis(u[b])
        SI, SJ = spin[a], spin[b]
        Jt = {(p, q): eI[p] @ J @ eJ[q] for p in "+-0" for q in "+-0"}
        lin_I = {"-": (op(a, False, rI),), "+": (op(a, True, rI),)}
        lin_J = {"-": (op(b, False, rJ),), "+": (op(b, True, rJ),)}
        cub_I = {"-": (op(a, True, rI), op(a, False, rI), op(a, False, rI)),
                 "+": (op(a, True, rI), op(a, True, rI), op(a, False, rI))}
        cub_J = {"-": (op(b, True, rJ), op(b, False, rJ), op(b, False, rJ)),
                 "+": (op(b, True, rJ), op(b, True, rJ), op(b, False, rJ))}
        n_I, n_J = (op(a, True, rI), op(a, False, rI)), (op(b, True, rJ), op(b, False, rJ))
        for q in "+-":
            cubic[canonical(n_I + lin_J[q], home)] -= np.sqrt(SJ) * Jt["0", q]
            cubic[canonical(lin_I[q] + n_J, home)] -= np.sqrt(SI) * Jt[q, "0"]
        quartic[canonical(n_I + n_J, home)] += Jt["0", "0"]
        for p in "+-":
            for q in "+-":
                quartic[canonical(cub_I[p] + lin_J[q], home)] -= 0.25 * np.sqrt(SJ / SI) * Jt[p, q]
                quartic[canonical(lin_I[p] + cub_J[q], home)] -= 0.25 * np.sqrt(SI / SJ) * Jt[p, q]
    return cubic, quartic


def check_vertices():
    model, bonds, spin, pos, h, u, torque = build_case()
    assert torque < 1e-12
    expansion = expand_model(model, make_state(model, u), ExternalConditions(field=h))
    home = expansion.positions
    cubic, quartic = note_vertices(bonds, spin, pos, u, home)
    worst = 0.0
    for n, mine in ((3, cubic), (4, quartic)):
        ref = defaultdict(complex)
        for c, m in expansion.orders[n]:
            ref[canonical(m, home)] += c
        worst = max(worst, max(abs(ref[k] - mine[k]) for k in set(ref) | set(mine)))
    print(f"1. cubic and quartic terms vs package expansion: max deviation {worst:.1e}")
    return worst


def check_tadpole_gradient(mesh=12, eps=1e-5):
    model, _, spin, _, h, u, _ = build_case()
    conditions = ExternalConditions(field=h)
    result = solve_nlswt(model, make_state(model, u), conditions,
                         settings=NLSWTSettings(mesh=(mesh, mesh)))
    eta = result.solver.linear_terms()[:2]          # coefficients of a_mu^dagger
    k = result.k_points
    metric = np.diag([1, 1, -1, -1])

    def zero_point(directions):
        H = QuadraticBoseHamiltonian(model, make_state(model, directions), conditions).at(k)
        total = 0.0
        for Hk in H:
            w = np.linalg.eigvals(metric @ Hk).real
            total += 0.5 * np.sort(w)[2:].sum() - 0.25 * np.trace(Hk).real
        return total / len(k)

    worst = 0.0
    for mu, s in enumerate("AB"):
        R = local_frame(u[s])
        for axis, part in ((0, eta[mu].real), (1, eta[mu].imag)):
            values = []
            for sign in (1, -1):
                d = dict(u)
                v = R[:, 2] + sign * np.sqrt(2 / spin[s]) * eps * R[:, axis]   # alpha = eps or i eps
                d[s] = v / np.linalg.norm(v)
                values.append(zero_point(d))
            worst = max(worst, abs((values[0] - values[1]) / (2 * eps) - 2 * part))
    print(f"2. tadpole coefficient vs dE_zp/d alpha* ({mesh}x{mesh} mesh): max deviation {worst:.1e}")
    return worst


if __name__ == "__main__":
    assert check_vertices() < 1e-13
    assert check_tadpole_gradient() < 1e-8
    print("all checks passed")
