"""Check the Goldstone-point statement of the research note *LSWT magnon topology*.

The note (section Goldstone Points) proves that the Chern number of a band that
reaches zero energy at an isolated momentum k0 is an integer, equal to the
gapped limit for every regularization, when the zero mode is either

(a) a single linear Goldstone mode: ``ker H_k0`` is one-dimensional and the
    dispersion is linear in every direction, or
(b) quadratic with a two-dimensional kernel, and the pairing of the zero mode
    with its hole partner small compared with its normal part near k0 (for
    example ``B_k = 0`` near k0),

and that the statement fails for a quadratic zero mode whose anomalous block
vanishes at k0 but not near it. This script checks the three cases:

1. Case (a) without threefold symmetry: the canted honeycomb antiferromagnet
   of ``lswt_bdg_blocks_check.py`` check 6 with the three nearest-neighbour
   bonds scaled by (1, 1.3, 0.8) and different DM couplings on the two
   sublattices (0.2 and 0.05). The curvature vanishes at Gamma, the loop
   Berry phase goes to zero, the Chern numbers on a mesh avoiding Gamma are
   (+1, -1), and three different gap-opening anisotropies give the same
   numbers.
2. Case (b) with a regularization that breaks the U(1) symmetry: the
   honeycomb ferromagnet of check 6 with a field along z and the bond
   anisotropy ``K (-S^x S^x + S^y S^y)``; the Chern numbers stay (+1, -1).
3. The counterexample: a square-lattice ferromagnet along z with a
   pseudo-dipolar anisotropy ``K (S . e)(S . e)`` on the first-neighbour
   bonds (e along the bond) and ``K/2`` on the second-neighbour bonds, plus
   ``S^z S^z`` terms that make every uniform direction degenerate. Then
   ``A_Gamma = B_Gamma = 0`` and ``B_k`` is proportional to ``(k_x + i k_y)^2``.
   The curvature is bounded, but the loop Berry phase tends to the constant
   ``-oint sinh^2(theta) dphi`` predicted from the k^2 blocks, the integral
   avoiding Gamma is not an integer, and a small field gives Chern number 0.

Run with ``python code-space/examples/verification/lswt_goldstone_chern_check.py``.
"""

import warnings

import numpy as np

from spintoolkit.methods.classical import classical_search
from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.models.heisenberg import SQUARE_LATTICE
from spintoolkit.models.honeycomb import honeycomb_ferromagnet
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Site, SpinModel, Term


def _colpa(H):
    """Return the signed eigenvalues and the paraunitary matrix of ``H``."""
    ns = H.shape[0] // 2
    sig = np.diag([1.0] * ns + [-1.0] * ns)
    K = np.linalg.cholesky(H)
    w, V = np.linalg.eigh(K.conj().T @ sig @ K)
    order = np.r_[np.argsort(-w)[:ns][::-1], np.argsort(w)[:ns][::-1]]
    w, V = w[order], V[:, order]
    return w, np.linalg.solve(K.conj().T, V) @ np.diag(np.sqrt(np.abs(w)))


def _curvature(Hq, k):
    """Kubo-form Berry curvature of the particle bands at ``k``."""
    H = Hq.at(k[None])[0]
    ns = H.shape[0] // 2
    dx, dy = [d[0] for d in Hq.derivatives_at(k[None])]
    lam, T = _colpa(H)
    X, Y = T.conj().T @ dx @ T, T.conj().T @ dy @ T
    s = np.r_[np.ones(ns), -np.ones(ns)]
    om = np.zeros(ns)
    for n in range(ns):
        for m in range(2 * ns):
            if m != n:
                om[n] += -2 * np.imag(s[n] * s[m] * X[n, m] * Y[m, n]) / (lam[n] - lam[m]) ** 2
    return lam[:ns], om


def _loop_phase(Hq, r, points=600):
    """Berry phase of the lowest particle band around a circle of radius ``r`` about Gamma."""
    ns = Hq.at(np.zeros((1, 2)))[0].shape[0] // 2
    sig = np.diag([1.0] * ns + [-1.0] * ns)
    vectors = [_colpa(Hq.at((r * np.array([np.cos(t), np.sin(t)]))[None])[0])[1][:, 0]
               for t in np.linspace(0, 2 * np.pi, points, endpoint=False)]
    product = np.prod([vectors[i].conj() @ sig @ vectors[(i + 1) % points] for i in range(points)])
    return float(-np.angle(product))


def _chern(Hq, model, L=60):
    """Chern numbers from the Kubo curvature on an L x L mesh that avoids Gamma."""
    recip = 2 * np.pi * np.linalg.inv(model.lattice)
    area = abs(np.linalg.det(recip))
    total = 0.0
    for i in range(L):
        for j in range(L):
            total = total + _curvature(Hq, recip @ ((np.array([i, j]) + 0.5) / L))[1]
    return total * area / L**2 / (2 * np.pi)


def _exponent(Hq, angle=0.3):
    u = np.array([np.cos(angle), np.sin(angle)])
    return np.log10(abs(_curvature(Hq, 1e-2 * u)[1][0] / _curvature(Hq, 1e-3 * u)[1][0]))


def distorted_canted_model(extra=None, scales=(1.0, 1.3, 0.8), D_A=0.2, D_B=0.05, S=0.5):
    """Canted honeycomb antiferromagnet without threefold symmetry; ``extra`` is added to each NN bond."""
    base = honeycomb_ferromagnet(J=-1.0, D=D_A, S=S)
    terms, bond = [], 0
    for t in base.terms:
        if t.label == "NN":
            coupling = scales[bond] * t.coefficient + (0 if extra is None else extra)
            terms.append(Term.bilinear(*t.participants, coupling, label="NN"))
            bond += 1
        elif t.label == "DM" and t.participants[0][0] == "B":
            terms.append(Term.bilinear(*t.participants, t.coefficient * D_B / D_A, label="DM"))
        else:
            terms.append(t)
    return SpinModel(base.lattice, list(base.sites), terms, {"model_id": "distorted_canted"})


def pseudo_dipolar_ferromagnet(J=1.0, K=0.3, S=0.5):
    """Square-lattice ferromagnet with a compensated pseudo-dipolar anisotropy (one site per cell)."""
    zz = np.diag([0.0, 0.0, 1.0])
    terms = []
    for offset, e in (((1, 0), (1, 0, 0)), ((0, 1), (0, 1, 0))):
        terms.append(Term.bilinear(("A", (0, 0)), ("A", offset),
                                   -J * np.eye(3) + K * np.outer(e, e) + K / 2 * zz, label="NN"))
    for offset, d in (((1, 1), (1, 1, 0)), ((1, -1), (1, -1, 0))):
        d = np.array(d) / np.sqrt(2)
        terms.append(Term.bilinear(("A", (0, 0)), ("A", offset),
                                   K / 2 * np.outer(d, d) + K / 4 * zz, label="NNN"))
    terms.append(Term.zeeman("A", np.eye(3)))
    return SpinModel(SQUARE_LATTICE, [Site("A", (0.0, 0.0), S)], terms,
                     {"model_id": "pseudo_dipolar_ferromagnet"})


def check_linear_goldstone(h=1.0):
    """Case 1: linear Goldstone mode, no C3."""
    conditions = ExternalConditions(field=(0, 0, h))
    model = distorted_canted_model()
    state = classical_search(model, [[1, 0], [0, 1]], conditions).state
    Hq = QuadraticBoseHamiltonian(model, state, conditions)
    assert np.abs(Hq.linear_terms).max() < 1e-12
    kernel = np.sum(np.linalg.eigvalsh(Hq.at(np.zeros((1, 2)))[0]) < 1e-10)
    loop = {r: _loop_phase(Hq, r) for r in (1e-1, 1e-2, 1e-3)}
    chern = _chern(Hq, model)
    print(f"1. distorted canted: dim ker H_Gamma = {kernel}, Omega_1 ~ |k|^{_exponent(Hq):.2f}, "
          f"loop phase {loop[1e-1]:.1e}, {loop[1e-2]:.1e}, {loop[1e-3]:.1e} (r = 0.1, 0.01, 0.001), "
          f"Chern = {np.round(chern, 5)}")
    gapped = {}
    xy = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]) / 2
    for name, extra in (("-K SxSx", np.diag([-1.0, 0.0, 0.0])),
                        ("-K SySy", np.diag([0.0, -1.0, 0.0])),
                        ("-K (SxSy+SySx)/2", -xy)):
        for K in (0.05, 0.005):
            reg = distorted_canted_model(extra=K * extra)
            reg_state = classical_search(reg, [[1, 0], [0, 1]], conditions).state
            Hr = QuadraticBoseHamiltonian(reg, reg_state, conditions)
            assert np.abs(Hr.linear_terms).max() < 1e-12
            sig = np.diag([1.0, 1.0, -1.0, -1.0])
            gap = np.sort(np.abs(np.linalg.eigvals(sig @ Hr.at(np.zeros((1, 2)))[0]).real))[0]
            gapped[(name, K)] = (gap, _chern(Hr, reg))
            print(f"1. gapped by {name}, K = {K}: Gamma gap {gap:.3f}, "
                  f"Chern = {np.round(gapped[(name, K)][1], 5)}")
    return kernel, loop, chern, gapped


def check_number_conserving_regularization():
    """Case 2: honeycomb ferromagnet, gap opened by a term that breaks U(1)."""
    out = {}
    for h, K in ((0.05, 0.02), (0.002, 0.0008)):
        base = honeycomb_ferromagnet(J=1.0, D=0.2, S=0.5)
        terms = [Term.bilinear(*t.participants, t.coefficient + np.diag([-K, K, 0.0]), label="NN")
                 if t.label == "NN" else t for t in base.terms]
        model = SpinModel(base.lattice, list(base.sites), terms, {"model_id": "fm_u1_broken"})
        state = SpinState.from_function(model, np.eye(2, dtype=int), lambda site, c: (0, 0, 1))
        Hq = QuadraticBoseHamiltonian(model, state, ExternalConditions(field=(0, 0, h)))
        assert np.abs(Hq.linear_terms).max() < 1e-12
        out[(h, K)] = _chern(Hq, model)
        print(f"2. ferromagnet, h = {h}, K = {K}: Chern = {np.round(out[(h, K)], 5)}")
    return out


def check_counterexample(r_pred=1e-3, points=2000):
    """Case 3: quadratic zero mode with B_Gamma = 0 but B_k ~ k_+^2."""
    model = pseudo_dipolar_ferromagnet()
    state = SpinState.from_function(model, np.eye(2, dtype=int), lambda site, c: (0, 0, 1))
    Hq = QuadraticBoseHamiltonian(model, state, ExternalConditions(field=(0, 0, 0.0)))
    assert np.abs(Hq.linear_terms).max() < 1e-12
    H0 = Hq.at(np.zeros((1, 2)))[0]
    grid = np.linspace(-np.pi, np.pi, 41)
    lowest = min(np.linalg.eigvalsh(Hq.at(np.array([[a, b]]))[0]).min() for a in grid for b in grid)
    # prediction from the k^2 blocks: gamma = -oint sinh^2(theta) dphi, tanh 2 theta = |B| / A
    angles = np.linspace(0, 2 * np.pi, points + 1)
    blocks = Hq.at(r_pred * np.c_[np.cos(angles), np.sin(angles)])
    A, B = blocks[:, 0, 0].real, blocks[:, 0, 1]
    sinh2 = (1 / np.sqrt(1 - (np.abs(B) / A) ** 2) - 1) / 2
    phi = np.unwrap(np.angle(B))
    predicted = -np.sum((sinh2[1:] + sinh2[:-1]) / 2 * np.diff(phi))
    winding = (phi[-1] - phi[0]) / (2 * np.pi)
    loop = {r: _loop_phase(Hq, r) for r in (1e-1, 1e-2, 1e-3)}
    omega = [_curvature(Hq, r * np.array([np.cos(0.3), np.sin(0.3)]))[1][0] for r in (1e-2, 1e-3)]
    chern = _chern(Hq, model, L=80)
    gapped = {h: _chern(QuadraticBoseHamiltonian(model, state, ExternalConditions(field=(0, 0, h))),
                        model, L=80) for h in (0.05, 0.01)}
    print(f"3. pseudo-dipolar ferromagnet: max |H_Gamma| = {np.abs(H0).max():.1e}, "
          f"min eigenvalue of H_k = {lowest:.1e}, winding of arg B = {winding:+.2f}")
    print(f"3. Omega_1 = {omega[0]:.4f}, {omega[1]:.4f} at |k| = 0.01, 0.001; loop phase "
          f"{loop[1e-1]:+.5f}, {loop[1e-2]:+.5f}, {loop[1e-3]:+.5f} (r = 0.1, 0.01, 0.001), "
          f"predicted {predicted:+.5f}")
    print(f"3. integral avoiding Gamma = {chern[0]:.5f} (-gamma / 2 pi = {-predicted / (2 * np.pi):.5f}); "
          f"gapped by h = 0.05, 0.01: {gapped[0.05][0]:.1e}, {gapped[0.01][0]:.1e}")
    return H0, lowest, winding, omega, loop, predicted, chern, gapped


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    kernel, loop, chern, gapped = check_linear_goldstone()
    assert kernel == 1
    assert abs(loop[1e-3]) < 1e-9 and abs(loop[1e-1]) < 1e-5
    assert np.allclose(chern, [1.0, -1.0], atol=1e-4)
    for gap, c in gapped.values():
        assert gap > 0.1 and np.allclose(c, [1.0, -1.0], atol=1e-3)
    for c in check_number_conserving_regularization().values():
        assert np.allclose(c, [1.0, -1.0], atol=1e-4)
    H0, lowest, winding, omega, loop, predicted, chern, gapped = check_counterexample()
    assert np.abs(H0).max() < 1e-12 and lowest > -1e-12 and abs(winding - 2) < 1e-6
    assert abs(omega[0] - omega[1]) < 1e-3                       # bounded curvature
    assert abs(loop[1e-3] - predicted) < 1e-4 and abs(predicted) > 0.1
    assert abs(chern[0] + predicted / (2 * np.pi)) < 1e-3        # not an integer
    assert all(abs(c[0]) < 1e-4 for c in gapped.values())
    print("all checks passed")
