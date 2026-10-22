"""Site-based NLCE for the uniform susceptibility chi_zz, chi_xx (per site) of the S=1 honeycomb
J1-J3-D model (J1 nearest, J3 third neighbours, -D Sz^2), h = 0.
Every connected cluster S (connectivity through J1 or J3 bonds) contributes a weight
W(S) = P(S) - sum_{S' < S} W(S'), P(S) = beta <M_a^2>_S (extensive), and per site
chi_a = sum_{S containing site 0} W(S)/|S|  (all sites equivalent).
Usage: python nlce_chi.py J3 D nmax   -> prints chi_zz, chi_xx and partial sums per order on a T grid (JSON)."""
import sys, json, itertools
import numpy as np

S1 = np.sqrt(2)
SX = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / S1
SYI = np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]]) / S1   # i*Sy is real: Sy = -i*SYI
SZ = np.diag([1.0, 0, -1])

# honeycomb sites (n1, n2, s), s=0 A, s=1 B
J1_OFF = [(0, 0), (-1, 0), (0, -1)]           # A(n) - B(n+off)
J3_OFF = [(-1, 1), (1, -1), (-1, -1)]


def neighbours(site):
    n1, n2, s = site
    out = []
    for offs, lab in ((J1_OFF, 1), (J3_OFF, 3)):
        for d1, d2 in offs:
            if s == 0:
                out.append(((n1 + d1, n2 + d2, 1), lab))
            else:
                out.append(((n1 - d1, n2 - d2, 0), lab))
    return out


def check_geometry():
    a1, a2 = np.array([1.0, 0]), np.array([0.5, np.sqrt(3) / 2])
    pos = lambda s: s[0] * a1 + s[1] * a2 + (s[2] / 3) * (a1 + a2)
    for site in [(0, 0, 0), (0, 0, 1)]:
        for nb, lab in neighbours(site):
            d = np.linalg.norm(pos(nb) - pos(site))
            want = 1 / np.sqrt(3) if lab == 1 else 2 / np.sqrt(3)
            assert abs(d - want) < 1e-9, (site, nb, lab, d)


def canon(sites):
    m1 = min(s[0] for s in sites); m2 = min(s[1] for s in sites)
    return tuple(sorted((s[0] - m1, s[1] - m2, s[2]) for s in sites))


def connected(sites):
    sites = set(sites); start = next(iter(sites)); seen = {start}; stack = [start]
    while stack:
        x = stack.pop()
        for nb, _ in neighbours(x):
            if nb in sites and nb not in seen:
                seen.add(nb); stack.append(nb)
    return len(seen) == len(sites)


def clusters_containing_origin(nmax):
    """all connected site sets of size <= nmax containing (0,0,0)"""
    origin = (0, 0, 0)
    found = {1: {frozenset([origin])}}
    for n in range(2, nmax + 1):
        new = set()
        for c in found[n - 1]:
            for x in c:
                for nb, _ in neighbours(x):
                    if nb not in c:
                        new.add(c | {nb})
        found[n] = new
    return found


def op(o, i, n):
    out = np.eye(1)
    for k in range(n):
        out = np.kron(out, o if k == i else np.eye(3))
    return out


def cluster_P(sites, J3, D, betas):
    sites = list(sites); n = len(sites); idx = {s: k for k, s in enumerate(sites)}
    Sz = [op(SZ, k, n) for k in range(n)]; Sx = [op(SX, k, n) for k in range(n)]; Syi = [op(SYI, k, n) for k in range(n)]
    H = np.zeros((3 ** n, 3 ** n))
    for s in sites:
        for nb, lab in neighbours(s):
            if nb in idx and idx[nb] > idx[s]:
                J = -1.0 if lab == 1 else J3
                i, j = idx[s], idx[nb]
                H += J * (Sz[i] @ Sz[j] + Sx[i] @ Sx[j] - Syi[i] @ Syi[j])   # SySy = -(iSy)(iSy)
    for k in range(n):
        H -= D * Sz[k] @ Sz[k]
    w, v = np.linalg.eigh(H)
    Mz = sum(Sz); Mx = sum(Sx)
    mz2 = np.einsum("ij,ij->j", v, (Mz @ Mz) @ v); mx2 = np.einsum("ij,ij->j", v, (Mx @ Mx) @ v)
    out = []
    for b in betas:
        p = np.exp(-b * (w - w.min())); p /= p.sum()
        out.append((b * (p @ mz2), b * (p @ mx2)))
    return np.array(out)   # shape (nT, 2)


def main(J3, D, nmax, Ts):
    check_geometry()
    betas = 1 / np.asarray(Ts)
    found = clusters_containing_origin(nmax)
    memoP, memoW = {}, {}

    def W(sites):
        key = canon(sites)
        if key in memoW:
            return memoW[key]
        if key not in memoP:
            memoP[key] = cluster_P(sites, J3, D, betas)
        w = memoP[key].copy()
        sl = list(sites)
        for r in range(1, len(sl)):
            for sub in itertools.combinations(sl, r):
                if connected(sub):
                    w -= W(frozenset(sub))
        memoW[key] = w
        return w

    partial = []
    total = np.zeros((len(Ts), 2))
    for n in range(1, nmax + 1):
        for c in found[n]:
            total += W(c) / n
        partial.append(total.copy())
    return dict(J3=J3, D=D, nmax=nmax, T=list(map(float, Ts)),
                chi_zz=[[float(x) for x in p[:, 0]] for p in partial],
                chi_xx=[[float(x) for x in p[:, 1]] for p in partial],
                n_clusters=[len(found[n]) for n in range(1, nmax + 1)])


if __name__ == "__main__":
    J3, D, nmax = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    Ts = np.round(np.concatenate([np.arange(1.0, 4.0, 0.25), np.arange(4.0, 20.01, 0.5)]), 4)
    print(json.dumps(main(J3, D, nmax, Ts)))
