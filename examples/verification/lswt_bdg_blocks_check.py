"""Check the explicit BdG blocks of the research note *LSWT boson Hamiltonian*.

The note writes the normal and anomalous blocks of the LSWT BdG matrix in terms
of the circular components ``J~^{pq} = (e_I^p)^T J (e_J^q)`` of each exchange
matrix, the local number-operator coefficient and the single-ion rule (D37).
This script implements those formulas directly, without the package builder,
and compares them with ``QuadraticBoseHamiltonian`` (D36):

1. A two-sublattice cell with random nonsymmetric exchange on inter- and
   same-sublattice bonds, random single-ion matrices (S = 1 and 3/2), random
   g-tensors, a random field and a random, non-stationary spin configuration.
   The blocks must agree entry by entry at random momenta.
2. The full-position convention: ``A_{k+G} = U^dag A_k U`` and
   ``B_{k+G} = U^dag B_k U`` with ``U = diag exp(i G . delta_mu)``, that is
   ``H_{k+G} = Lambda_G H_k Lambda_G^dag`` with ``Lambda_G = 1_2 (x) U^dag``.
3. The square-lattice Neel antiferromagnet: the blocks equal the closed forms
   ``A = 4JS 1`` and ``B = -4JS gamma_k sigma_x`` of the note's example, and
   the zero-point correction with the trace subtraction must approach the
   series result
   ``Delta E_zp / N = -0.315894842 J S`` (Zheng and Hamer, PRB 47, 7961, 1993).
4. The classical Hessian: for a random configuration of the model of case 1
   on a periodic 3 x 3 cluster, the second-order change of the classical
   energy (with the coherent-state single-ion value) under a deformation with
   complex amplitudes alpha equals the BdG quadratic form with a -> alpha.
5. A single-Q spiral in the rotating frame: for the square-lattice Heisenberg
   model with J1 and J2 along x and ferromagnetic Jy (a spiral along x), the
   blocks with rotating-frame exchange J R_z(Q . Delta) give
   omega_k^2 = S^2 (J_k - J_Q)((J_{k+Q} + J_{k-Q})/2 - J_Q), with
   J_k = sum_Delta J_Delta exp(i k . Delta), at the minimizing pitch;
   a non-minimizing pitch has imaginary modes with the same omega_k^2 < 0.

Run with ``python examples/verification/lswt_bdg_blocks_check.py``.
"""

import numpy as np

from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Site, SpinModel, Term

EX, EY, EZ = np.eye(3)


def rotation(n):
    """R(theta, phi) of the note *LSWT classical order*; third column n."""
    theta = np.arccos(np.clip(n[2], -1, 1))
    phi = np.arctan2(n[1], n[0]) if np.hypot(n[0], n[1]) > 0 else 0.0
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    return np.array([[ct * cp, -sp, st * cp], [ct * sp, cp, st * sp], [-st, 0, ct]])


def circular_basis(n):
    """(e^+, e^-, e^0) in laboratory coordinates."""
    R = rotation(n)
    return {"+": R @ (EX + 1j * EY) / np.sqrt(2), "-": R @ (EX - 1j * EY) / np.sqrt(2), "0": R @ EZ}


def note_blocks(sites, bonds, onsite, fields, k):
    """A_k and B_k from the note's formulas.

    sites  : list of (spin, direction)
    bonds  : list of (mu, nu, Delta, J), one entry per link of the set L
    onsite : list of (mu, D)
    fields : list of h_mu (coefficient of -h . S)
    """
    ns = len(sites)
    A = np.zeros((ns, ns), complex)
    B = np.zeros((ns, ns), complex)
    e = [circular_basis(n) for _, n in sites]
    S = [s for s, _ in sites]
    for mu in range(ns):
        A[mu, mu] += fields[mu] @ e[mu]["0"]                    # h~^0
    for mu, nu, delta, J in bonds:
        Jt = {p + q: e[mu][p] @ J @ e[nu][q] for p in "+-0" for q in "+-0"}
        t = {pq: np.sqrt(S[mu] * S[nu]) * Jt[pq] for pq in ("+-", "-+", "++")}
        ph = np.exp(1j * k @ delta)
        A[mu, nu] += t["+-"] * ph
        A[nu, mu] += t["-+"] / ph
        B[mu, nu] += t["++"] * ph
        B[nu, mu] += t["++"] / ph
        A[mu, mu] -= S[nu] * Jt["00"].real
        A[nu, nu] -= S[mu] * Jt["00"].real
    for mu, D in onsite:
        kappa = 1 - 1 / (2 * S[mu])
        Dt = {p + q: e[mu][p] @ (kappa * D) @ e[mu][q] for p in "+-0" for q in "+-0"}
        A[mu, mu] += S[mu] * (Dt["+-"] + Dt["-+"]) - 2 * S[mu] * Dt["00"].real
        B[mu, mu] += 2 * S[mu] * Dt["++"]
    return A, B


def note_bdg(sites, bonds, onsite, fields, k):
    A, B = note_blocks(sites, bonds, onsite, fields, k)
    Am, _ = note_blocks(sites, bonds, onsite, fields, -k)
    return np.block([[A, B], [B.conj().T, Am.conj()]])


def random_case(rng):
    lattice = np.array([[1.0, 0.0], [0.4, 0.9]])
    spins = {"a": 1.0, "b": 1.5}
    sites = [Site("a", (0.0, 0.0), spins["a"]), Site("b", (0.35, 0.6), spins["b"])]
    links = [("a", (0, 0), "b", (0, 0)), ("a", (0, 0), "b", (1, 0)), ("b", (0, 0), "a", (0, 1)),
             ("a", (0, 0), "a", (1, 0)), ("b", (0, 0), "b", (1, 1))]
    terms, bonds = [], []
    for s, ns_, t, nt in links:
        J = rng.normal(size=(3, 3))
        terms.append(Term.bilinear((s, ns_), (t, nt), J))
        bonds.append((s, ns_, t, nt, J))
    onsite = {}
    for s in spins:
        D = rng.normal(size=(3, 3))
        D = D + D.T
        g = rng.normal(size=(3, 3)) + 2 * np.eye(3)
        terms += [Term.onsite(s, D), Term.zeeman(s, g)]
        onsite[s] = (D, g)
    model = SpinModel(lattice, sites, terms, metadata={"model_id": "bdg-check"})
    directions = {s: (v := rng.normal(size=3)) / np.linalg.norm(v) for s in spins}
    state = SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: directions[s])
    field = rng.normal(size=3) * 0.3
    order = list(spins)
    note_sites = [(spins[s], directions[s]) for s in order]
    note_bonds = [(order.index(s), order.index(t), model.cartesian_position(t, nt) - model.cartesian_position(s, ns_), J)
                  for s, ns_, t, nt, J in bonds]
    note_onsite = [(order.index(s), onsite[s][0]) for s in order]
    note_fields = [onsite[s][1].T @ field for s in order]                # h = g^T (mu_B B)
    native = QuadraticBoseHamiltonian(model, state, ExternalConditions(field=field))
    assert [key[0] for key in native.site_keys] == order
    return native, (note_sites, note_bonds, note_onsite, note_fields)


def check_blocks():
    rng = np.random.default_rng(20261012)
    worst = 0.0
    for trial in range(5):
        native, note = random_case(rng)
        for k in rng.uniform(-4, 4, size=(6, 2)):
            diff = np.abs(native.at(k[None])[0] - note_bdg(*note, k)).max()
            worst = max(worst, diff)
    print(f"1. random two-sublattice models: max |H_package - H_note| = {worst:.2e}")
    return worst


def check_reciprocal_shift():
    rng = np.random.default_rng(5)
    _, note = random_case(rng)
    lattice = np.array([[1.0, 0.0], [0.4, 0.9]])
    deltas = np.array([[0.0, 0.0], [0.35, 0.6]]) @ lattice
    G = 2 * np.pi * np.linalg.inv(lattice) @ np.array([1, -2])   # 1 b1 - 2 b2
    worst = 0.0
    for k in rng.uniform(-3, 3, size=(4, 2)):
        U = np.diag(np.exp(1j * deltas @ G))
        A, B = note_blocks(*note, k)
        A2, B2 = note_blocks(*note, k + G)
        worst = max(worst, np.abs(A2 - U.conj().T @ A @ U).max(), np.abs(B2 - U.conj().T @ B @ U).max())
    print(f"2. reciprocal shift: max deviation from U^dag X U = {worst:.2e}")
    return worst


def check_classical_hessian(L=3, eps=1e-4):
    rng = np.random.default_rng(11)
    lattice = np.array([[1.0, 0.0], [0.4, 0.9]])
    note_sites, note_bonds, note_onsite, note_fields = random_case(rng)[1]
    ns = len(note_sites)
    frac = [np.array([0.0, 0.0]), np.array([0.35, 0.6])]
    cells = [(i, j) for i in range(L) for j in range(L)]
    pos = {(c, m): (np.array(c) + frac[m]) @ lattice for c in cells for m in range(ns)}
    # links of the cluster: each bond of the cell, translated, with the target cell found from Delta
    links = []
    for mu, nu, delta, J in note_bonds:
        for c in cells:
            target = pos[(c, mu)] + delta
            n = np.linalg.solve(lattice.T, target) - frac[nu]
            cn = (int(round(n[0])) % L, int(round(n[1])) % L)
            links.append(((c, mu), (cn, nu), J))
    e = {(c, m): circular_basis(note_sites[m][1]) for c in cells for m in range(ns)}
    spin = [S for S, _ in note_sites]

    def energy(alpha):
        vec = {}
        for key in pos:
            S, a = spin[key[1]], alpha[key]
            vec[key] = ((S - abs(a) ** 2) * e[key]["0"]
                        + np.sqrt(S) * (e[key]["-"] * a + e[key]["+"] * np.conj(a))).real
        E = sum(vec[I] @ J @ vec[K] for I, K, J in links)
        for mu, D in note_onsite:
            S = spin[mu]
            for c in cells:
                n = vec[(c, mu)] / np.linalg.norm(vec[(c, mu)])
                E += S * (S - 0.5) * n @ D @ n
        E -= sum(note_fields[key[1]] @ vec[key] for key in pos)
        return E

    alpha = {key: complex(*rng.normal(size=2)) for key in pos}
    second = (energy({k: eps * v for k, v in alpha.items()}) + energy({k: -eps * v for k, v in alpha.items()})
              - 2 * energy({k: 0j for k in alpha})) / (2 * eps ** 2)
    recip = 2 * np.pi * np.linalg.inv(lattice)                 # columns b1, b2
    quad = 0.0
    for m1 in range(L):
        for m2 in range(L):
            k = recip @ np.array([m1, m2]) / L
            ak = np.array([sum(np.exp(-1j * k @ pos[(c, m)]) * alpha[(c, m)] for c in cells) for m in range(ns)]) / L
            amk = np.array([sum(np.exp(1j * k @ pos[(c, m)]) * alpha[(c, m)] for c in cells) for m in range(ns)]) / L
            A, B = note_blocks(note_sites, note_bonds, note_onsite, note_fields, k)
            quad += (ak.conj() @ A @ ak + 0.5 * (ak.conj() @ B @ amk.conj() + np.conj(ak.conj() @ B @ amk.conj()))).real
    rel = abs(second - quad) / abs(quad)
    print(f"4. classical Hessian: second variation {second:.8f}, BdG form {quad:.8f}, relative difference {rel:.1e}")
    return rel


def rz(phi):
    c, s_ = np.cos(phi), np.sin(phi)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])


def check_spiral(S=1.0, J1=1.0, J2=0.7, Jy=-0.5):
    neighbors = [((1, 0), J1), ((2, 0), J2), ((0, 1), Jy)]                # one per pair
    def Jk(k):
        return sum(2 * J * np.cos(k @ np.array(d, float)) for d, J in neighbors)
    Qstar = np.array([np.arccos(-J1 / (4 * J2)), 0.0])                    # global minimum of J_k
    rng = np.random.default_rng(3)
    worst = 0.0
    negative = False
    for Q in (Qstar, Qstar + np.array([0.3, -0.1])):
        bonds = [(0, 0, np.array(d, float), J * np.eye(3) @ rz(Q @ np.array(d, float))) for d, J in neighbors]
        for k in np.vstack([rng.uniform(-np.pi, np.pi, size=(20, 2)), Qstar, Qstar + [0.1, 0.05]]):
            H = note_bdg([(S, EX)], bonds, [], [np.zeros(3)], k)
            w = np.linalg.eigvals(np.diag([1, -1]) @ H)
            w2 = S ** 2 * (Jk(k) - Jk(Q)) * ((Jk(k + Q) + Jk(k - Q)) / 2 - Jk(Q))
            if np.allclose(Q, Qstar):
                assert w2 >= -1e-12
                worst = max(worst, abs(np.max(w.real) ** 2 - w2))      # omega^2: zero modes
            elif w2 < 0:
                negative = True
                worst = max(worst, abs(np.sort(w.imag)[-1] ** 2 + w2))
    print(f"5. J1-J2-Jy spiral: max deviation from closed form {worst:.1e}; "
          f"non-minimizing pitch has imaginary modes: {negative}")
    return worst, negative


def check_square_zero_point(L=200, S=0.5):
    lattice = np.array([[1.0, 1.0], [1.0, -1.0]])        # sqrt2 x sqrt2 Neel cell
    sites = [(S, EZ), (S, -EZ)]
    J = np.eye(3)
    a_to_b = [np.array(d, float) for d in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    bonds = [(0, 1, d, J) for d in a_to_b]
    recip = 2 * np.pi * np.linalg.inv(lattice).T            # rows b1, b2
    total, count = 0.0, 0
    for i in range(L):
        for j in range(L):
            k = ((i + 0.5) / L) * recip[0] + ((j + 0.5) / L) * recip[1]
            H = note_bdg(sites, bonds, [], [np.zeros(3)] * 2, k)
            A, B = note_blocks(sites, bonds, [], [np.zeros(3)] * 2, k)
            gamma = 0.5 * (np.cos(k[0]) + np.cos(k[1]))
            assert np.allclose(A, 4 * S * np.eye(2), atol=1e-13)
            assert np.allclose(B, -4 * S * gamma * np.array([[0, 1], [1, 0]]), atol=1e-13)
            w = np.linalg.eigvals(np.diag([1, 1, -1, -1]) @ H).real
            total += 0.5 * (np.sort(w)[2:].sum() - np.trace(A).real)
            count += 1
    per_site = total / (count * 2) / S
    print(f"3. square Neel: Delta E_zp / (N J S) = {per_site:.6f} (series -0.315895, mesh {L}x{L})")
    return per_site


if __name__ == "__main__":
    assert check_blocks() < 1e-12
    assert check_reciprocal_shift() < 1e-12
    assert abs(check_square_zero_point() + 0.315894842) < 2e-4
    assert check_classical_hessian() < 1e-6
    worst, negative = check_spiral()
    assert worst < 1e-12 and negative
    print("all checks passed")
