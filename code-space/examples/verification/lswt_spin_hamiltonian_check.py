"""Check the claims of the research note *LSWT spin Hamiltonian*.

The note fixes the bilinear Hamiltonian, the link counting, the single-ion
term, the Zeeman convention and the coherent-state reference energy. This
script tests each statement that can be computed, with exact matrices for
small spins and with the package:

1. Spin-coherent state: ``<S^a S^b + S^b S^a>/2 = S(S - 1/2) n_a n_b +
   (S/2) delta_ab`` for S = 1/2 ... 5/2, the value behind the note's
   ``E_cl = <Phi|H|Phi>`` and the single-ion rule (D37).
2. Spin 1/2: ``S^a S^b + S^b S^a = delta_ab / 2``, so a real symmetric
   single-ion matrix gives ``tr(D)/4`` times the identity; for S = 1,
   ``-A (S^x)^2`` is lowest in the ``|m_x| = 1`` sector.
3. A real, nonsymmetric bond matrix gives a Hermitian bond operator, and
   the sum over links equals half the ordered-pair sum with
   ``J_JI = J_IJ^T`` (three spins 1/2 and 1 with random matrices).
4. A local frame needs ``det R = +1``: rotated components satisfy the spin
   algebra ``[S~x, S~y] = i S~z`` only for a proper rotation.
5. Zeeman convention: the Hamiltonian ``H(B) = H0 - mu_B B^T g S`` equals the
   electron Hamiltonian ``H0 + mu_B B^T g S`` at ``-B``. On a three-site
   cluster with DM exchange and anisotropic g the two have the same spectrum
   at the same B, and the moment ``mu_B g <S>`` of the convention equals the
   electron moment ``-mu_B g <S>_e`` at finite temperature.
6. Package: ``Term.zeeman`` contracts the field with the first g index
   (``h = mu_B g^T B``), a field favours spins parallel to it, and the
   classical energy of ``Term.onsite`` is the coherent-state value of case 1.

Run with ``python code-space/examples/verification/lswt_spin_hamiltonian_check.py``.
"""

import numpy as np

from spintoolkit.methods.classical import classical_energy
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Site, SpinModel, Term

TOL = 1e-10


def spin_matrices(S):
    """Return (Sx, Sy, Sz) for spin S in the S^z basis m = S, ..., -S."""
    m = np.arange(S, -S - 1, -1)
    sp = np.diag(np.sqrt(S * (S + 1) - m[1:] * (m[1:] + 1)), 1)
    return np.array([(sp + sp.T) / 2, (sp - sp.T) / 2j, np.diag(m)])


def embed(op, site, dims):
    """Operator on one site of a product space."""
    out = np.array([[1.0]])
    for i, d in enumerate(dims):
        out = np.kron(out, op if i == site else np.eye(d))
    return out


def coherent_state(S, n):
    """Spin-coherent state |S, n>: highest-weight state of n . S."""
    w, v = np.linalg.eigh(np.tensordot(n, spin_matrices(S), 1))
    return v[:, np.argmax(w)]


def check_coherent(rng):
    worst = 0.0
    for S in (0.5, 1.0, 1.5, 2.0, 2.5):
        ops = spin_matrices(S)
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        psi = coherent_state(S, n)
        sym = np.array([[psi.conj() @ (ops[a] @ ops[b] + ops[b] @ ops[a]) @ psi / 2
                         for b in range(3)] for a in range(3)])
        expect = S * (S - 0.5) * np.outer(n, n) + S / 2 * np.eye(3)
        worst = max(worst, np.max(np.abs(sym - expect)),
                    np.max(np.abs([psi.conj() @ o @ psi for o in ops] - S * n)))
    return worst


def check_single_ion(rng):
    ops = spin_matrices(0.5)
    anti = max(np.max(np.abs(ops[a] @ ops[b] + ops[b] @ ops[a] - (a == b) * np.eye(2) / 2))
               for a in range(3) for b in range(3))
    D = rng.normal(size=(3, 3))
    D = D + D.T
    term = np.einsum("ab,aij,bjk->ik", D, ops, ops)
    constant = np.max(np.abs(term - np.trace(D) / 4 * np.eye(2)))
    sx = spin_matrices(1.0)[0]
    w, v = np.linalg.eigh(-1.0 * sx @ sx)
    # The lowest level is the degenerate m_x = +-1 pair, so test <(S^x)^2>.
    easy_axis = float(np.sqrt((v[:, 0].conj() @ sx @ sx @ v[:, 0]).real))
    return max(anti, constant), easy_axis


def check_links(rng):
    spins = (0.5, 1.0, 0.5)
    dims = [int(2 * s + 1) for s in spins]
    S = [[embed(o, i, dims) for o in spin_matrices(s)] for i, s in enumerate(spins)]
    links = {(0, 1): rng.normal(size=(3, 3)), (1, 2): rng.normal(size=(3, 3)),
             (2, 0): rng.normal(size=(3, 3))}

    def bond(i, j, J):
        return sum(S[i][a] @ S[j][b] * J[a, b] for a in range(3) for b in range(3))

    link_sum = sum(bond(i, j, J) for (i, j), J in links.items())
    full = {}
    for (i, j), J in links.items():
        full[(i, j)], full[(j, i)] = J, J.T
    pair_sum = 0.5 * sum(bond(i, j, J) for (i, j), J in full.items())
    hermitian = max(np.max(np.abs(bond(i, j, J) - bond(i, j, J).conj().T))
                    for (i, j), J in links.items())
    return hermitian, np.max(np.abs(link_sum - pair_sum))


def check_handedness(rng):
    ops = spin_matrices(1.0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    out = {}
    for name, R in (("proper", q * np.sign(np.linalg.det(q))),
                    ("improper", -q * np.sign(np.linalg.det(q)))):
        local = np.tensordot(R.T, ops, 1)         # S~ = R^T S
        out[name] = np.max(np.abs(local[0] @ local[1] - local[1] @ local[0] - 1j * local[2]))
    return out


def check_time_reversal(rng):
    spins = (0.5, 1.0, 0.5)
    dims = [int(2 * s + 1) for s in spins]
    S = [np.array([embed(o, i, dims) for o in spin_matrices(s)]) for i, s in enumerate(spins)]
    H0 = 0
    for i, j in ((0, 1), (1, 2), (2, 0)):
        d = rng.normal(size=3)
        J = np.eye(3) + 0.3 * rng.normal(size=(3, 3))
        J = (J + J.T) / 2 + np.array([[0, d[2], -d[1]], [-d[2], 0, d[0]], [d[1], -d[0], 0]])
        H0 = H0 + np.einsum("ab,aij,bjk->ik", J, S[i], S[j])
    gs = [np.diag([1.9, 2.1, 2.3]) + 0.1 * rng.normal(size=(3, 3)) for _ in spins]
    B = rng.normal(size=3)

    def zeeman(field):
        return sum(np.einsum("a,ab,bij->ij", field, g, Si) for g, Si in zip(gs, S))

    H_conv = H0 - zeeman(B)                       # this note, mu_B = 1
    H_elec = H0 + zeeman(B)                       # electron sign
    spectrum = np.max(np.abs(np.linalg.eigvalsh(H_conv) - np.linalg.eigvalsh(H_elec)))

    def moment(H, sign, T=0.7):
        w, v = np.linalg.eigh(H)
        p = np.exp(-(w - w[0]) / T)
        rho = (v * (p / p.sum())) @ v.conj().T
        return sign * sum(g @ np.real(np.einsum("aij,ji->a", Si, rho)) for g, Si in zip(gs, S))

    return spectrum, np.max(np.abs(moment(H_conv, +1) - moment(H_elec, -1)))


def single_site_model(S, g=None, D=None):
    terms = [Term.zeeman("a", np.eye(3) if g is None else g)]
    if D is not None:
        terms.append(Term.onsite("a", D))
    return SpinModel(np.eye(2), [Site("a", (0.0, 0.0), S)], terms,
                     metadata={"model_id": "spin-hamiltonian-check"})


def package_energy(model, n, field):
    state = SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: n)
    return classical_energy(model, state, ExternalConditions(field=field))


def check_package(rng):
    S = 1.5
    g = rng.normal(size=(3, 3)) + 2 * np.eye(3)
    B = rng.normal(size=3)
    worst_zeeman = 0.0
    for _ in range(5):
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        energy = package_energy(single_site_model(S, g), n, B)
        worst_zeeman = max(worst_zeeman, abs(energy - (-(g.T @ B) @ (S * n))))
    h = np.array([0.0, 0.0, 0.7])
    parallel = package_energy(single_site_model(S), np.array([0, 0, 1.0]), h)
    antiparallel = package_energy(single_site_model(S), np.array([0, 0, -1.0]), h)
    D = rng.normal(size=(3, 3))
    D = D + D.T
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    onsite = package_energy(single_site_model(S, D=D), n, np.zeros(3))
    expect = S * (S - 0.5) * n @ D @ n + S / 2 * np.trace(D)
    # The constant (S/2) tr D does not change the state; compare differences too.
    n2 = rng.normal(size=3)
    n2 /= np.linalg.norm(n2)
    onsite2 = package_energy(single_site_model(S, D=D), n2, np.zeros(3))
    expect2 = S * (S - 0.5) * n2 @ D @ n2 + S / 2 * np.trace(D)
    return (worst_zeeman, parallel < antiparallel,
            abs(onsite - expect), abs((onsite - onsite2) - (expect - expect2)))


def main():
    rng = np.random.default_rng(20261020)
    ok = True

    def report(name, passed, detail):
        nonlocal ok
        ok &= bool(passed)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    err = check_coherent(rng)
    report("1 coherent-state moments", err < TOL, f"max error {err:.1e}")
    err, easy = check_single_ion(rng)
    report("2 spin-1/2 single-ion constant", err < TOL, f"max error {err:.1e}")
    report("2 S=1 easy axis -A(S^x)^2", abs(easy - 1) < TOL, f"|m_x| of lowest level {easy:.3f}")
    herm, links = check_links(rng)
    report("3 bond Hermiticity", herm < TOL, f"max |O - O^dag| {herm:.1e}")
    report("3 link sum = half ordered-pair sum", links < TOL, f"max error {links:.1e}")
    hand = check_handedness(rng)
    report("4 proper rotation keeps spin algebra", hand["proper"] < TOL,
           f"error {hand['proper']:.1e}")
    report("4 improper rotation breaks it", hand["improper"] > 0.1,
           f"error {hand['improper']:.2f}")
    spec, mom = check_time_reversal(rng)
    report("5 same spectrum as electron sign", spec < TOL, f"max error {spec:.1e}")
    report("5 same thermal moment", mom < TOL, f"max error {mom:.1e}")
    zee, parallel, onsite_abs, onsite_diff = check_package(rng)
    report("6 package h = mu_B g^T B", zee < TOL, f"max error {zee:.1e}")
    report("6 field favours parallel spin", parallel, "E(n || h) < E(n || -h)")
    report("6 package single-ion = coherent value (differences)", onsite_diff < TOL,
           f"error {onsite_diff:.1e}")
    report("6 package single-ion incl. (S/2) tr D", onsite_abs < TOL, f"error {onsite_abs:.1e}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
