"""Exact n-magnon sector above saturation for the S=1 U(1) core.

A site carries k magnons (k=0: m=+1, k=1: m=0, k=2: m=-1). Energies are
measured from the polarized state without the Zeeman term, so the n-magnon
energy at field h is E_n + n h.
"""
import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def basis(N, n):
    """All occupation patterns as sorted tuples of (site, k) with sum k = n."""
    out = []
    for n2 in range(n // 2 + 1):
        n1 = n - 2 * n2
        for doubles in itertools.combinations(range(N), n2):
            rest = [s for s in range(N) if s not in doubles] if n2 else range(N)
            for singles in itertools.combinations(rest, n1):
                occ = {s: 2 for s in doubles}
                occ.update({s: 1 for s in singles})
                out.append(tuple(sorted(occ.items())))
    return out


def hamiltonian(J, D, n):
    N = J.shape[0]
    states = basis(N, n)
    index = {s: i for i, s in enumerate(states)}
    nbr = [np.nonzero(J[i])[0] for i in range(N)]
    Js = J.sum(1)
    rows, cols, vals = [], [], []
    for a, s in enumerate(states):
        occ = dict(s)
        # diagonal: sum_{i<j} J (m_i m_j - 1) - D sum (m_i^2 - 1)
        diag = 0.0
        for i, k in occ.items():
            mi = 1 - k
            diag += -D * (mi * mi - 1)
            for j in nbr[i]:
                mj = 1 - occ.get(j, 0)
                # each bond with at least one occupied site; avoid double count
                if j in occ and j < i:
                    continue
                diag += J[i, j] * (mi * mj - 1)
        rows.append(a); cols.append(a); vals.append(diag)
        # hopping: one magnon moves i -> j (k_i -1, k_j +1), amplitude J_ij
        for i, k in occ.items():
            for j in nbr[i]:
                kj = occ.get(j, 0)
                if kj == 2:
                    continue
                new = dict(occ)
                if k == 1:
                    del new[i]
                else:
                    new[i] = 1
                new[j] = kj + 1
                b = index[tuple(sorted(new.items()))]
                rows.append(b); cols.append(a); vals.append(J[i, j])
    return sp.coo_matrix((vals, (rows, cols)), shape=(len(states),) * 2).tocsr()


def lowest(H):
    if H.shape[0] < 600:
        return np.linalg.eigvalsh(H.toarray())[0]
    return spla.eigsh(H, k=1, which="SA", tol=1e-9)[0][0]
