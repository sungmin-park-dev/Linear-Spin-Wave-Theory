"""Exact one- and two-magnon problem above saturation for the S=1 honeycomb
J1-J3-D model with field along the easy axis z (U(1) core of Na3Ni2BiO6).

H = sum_{i<j} J_ij S_i.S_j + B1 sum_<ij>_1 (S_i.S_j)^2 - D sum_i (S_i^z)^2 - h sum_i S_i^z

All energies below are measured from the fully polarized state with the
Zeeman part removed: a state with n magnons has energy E_n + n h.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

A1 = np.array([1.0, 0.0])
A2 = np.array([0.5, np.sqrt(3) / 2])
BASIS = [np.zeros(2), (A1 + A2) / 3]


def lattice(L, J1, J3, B1=0.0):
    """Coupling matrices on an L x L torus (2 L^2 sites).

    Bonds are generated from the A sublattice; on small tori several images
    of one bond add up, the same matrix is used by every solver below.
    """
    N = 2 * L * L
    site = lambda c1, c2, s: 2 * ((c1 % L) * L + (c2 % L)) + s
    d1, d3 = 1 / np.sqrt(3), 2 / np.sqrt(3)
    J = np.zeros((N, N))
    Bq = np.zeros((N, N))
    offs = []
    for n1 in range(-3, 4):
        for n2 in range(-3, 4):
            r = n1 * A1 + n2 * A2 + BASIS[1] - BASIS[0]
            d = np.linalg.norm(r)
            if abs(d - d1) < 1e-6:
                offs.append((n1, n2, J1, B1))
            elif abs(d - d3) < 1e-6:
                offs.append((n1, n2, J3, 0.0))
    assert len(offs) == 6
    for c1 in range(L):
        for c2 in range(L):
            i = site(c1, c2, 0)
            for n1, n2, j, b in offs:
                k = site(c1 + n1, c2 + n2, 1)
                J[i, k] += j; J[k, i] += j
                Bq[i, k] += b; Bq[k, i] += b
    return J, Bq


def one_magnon(J, D):
    """Single-magnon matrix (state: site i in m=0)."""
    Js = J.sum(1)
    return np.diag(D - Js) + J


def two_magnon(J, D, Bq=None):
    """Sparse two-magnon matrix. Basis: pairs |0_i 0_j> (i<j), then |-1_i>.

    Biquadratic B (S_i.S_j)^2 is only supported as zero here (B1 enters via
    `two_magnon_ed` checks later)."""
    if Bq is not None and np.any(Bq):
        raise NotImplementedError
    N = J.shape[0]
    Js = J.sum(1)
    iu, ju = np.triu_indices(N, 1)
    P = len(iu)
    pidx = lambda i, j: i * N - i * (i + 1) // 2 + (j - i - 1)
    rows, cols, vals = [], [], []
    # diagonal
    rows.append(np.arange(P)); cols.append(np.arange(P))
    vals.append(2 * D - Js[iu] - Js[ju] + J[iu, ju])
    rows.append(P + np.arange(N)); cols.append(P + np.arange(N)); vals.append(-2 * Js)
    nz = [np.nonzero(J[i])[0] for i in range(N)]
    deg = max(len(x) for x in nz)
    nbr = np.full((N, deg), -1); amp = np.zeros((N, deg))
    for i, x in enumerate(nz):
        nbr[i, :len(x)] = x; amp[i, :len(x)] = J[i, x]
    p = np.arange(P)
    for mover, stay in ((iu, ju), (ju, iu)):
        for s in range(deg):
            k = nbr[mover, s]; a = amp[mover, s]
            ok = (k >= 0) & (k != stay) & (a != 0)
            k, a, st, pp = k[ok], a[ok], stay[ok], p[ok]
            lo, hi = np.minimum(k, st), np.maximum(k, st)
            rows.append(pidx(lo, hi)); cols.append(pp); vals.append(a)
    # pair (i,j) bonded <-> |-1_i>, |-1_j>
    bonded = J[iu, ju] != 0
    for site_ in (iu, ju):
        r = P + site_[bonded]; c = p[bonded]; v = J[iu, ju][bonded]
        rows += [r, c]; cols += [c, r]; vals += [v, v]
    H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(P + N, P + N)).tocsr()
    return H


def lowest(H, k=1):
    if H.shape[0] < 400:
        return np.linalg.eigvalsh(H.toarray() if sp.issparse(H) else H)[:k]
    return np.sort(spla.eigsh(H, k=k, which="SA", tol=1e-10)[0])


# ---------------------------------------------------------------- ED check
def spin1_ops():
    sz = np.diag([1.0, 0.0, -1.0])
    sp_ = np.zeros((3, 3)); sp_[0, 1] = sp_[1, 2] = np.sqrt(2)
    return sz, sp_, sp_.T


def full_ed_sector(J, D, Bq, nmag):
    """Brute-force ED in the sector with nmag magnons (sum_i (1 - m_i) = nmag)."""
    N = J.shape[0]
    states = [s for s in np.ndindex(*(3,) * N) if sum(s) == nmag]  # local index 0:+1,1:0,2:-1
    index = {s: n for n, s in enumerate(states)}
    sz, spl, smi = spin1_ops()
    H = np.zeros((len(states), len(states)))
    m = np.array([1, 0, -1])
    for n, s in enumerate(states):
        mz = m[list(s)]
        H[n, n] += -D * np.sum(mz ** 2)
        for i in range(N):
            for j in range(i + 1, N):
                if J[i, j] == 0 and Bq[i, j] == 0:
                    continue
                # two-site operator S.S and (S.S)^2 in local 9-dim space
                SS = two_site_SS()
                op = J[i, j] * SS + Bq[i, j] * SS @ SS
                a = s[i] * 3 + s[j]
                for b in np.nonzero(np.abs(op[:, a]) > 1e-14)[0]:
                    t = list(s); t[i], t[j] = divmod(b, 3)
                    H[index[tuple(t)], n] += op[b, a]
    return H


_SS = None
def two_site_SS():
    global _SS
    if _SS is None:
        sz, spl, smi = spin1_ops()
        I = np.eye(3)
        _SS = np.kron(sz, sz) + 0.5 * (np.kron(spl, smi) + np.kron(smi, spl))
    return _SS
