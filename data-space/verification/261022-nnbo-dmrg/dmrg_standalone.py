"""Standalone cylinder iDMRG: S=1 honeycomb J1-J3-D, field along z, chain arrangements.
Usage: python dmrg_standalone.py J3 D W pattern h chi   -> prints one JSON line."""
import sys, json, time, warnings
warnings.simplefilter("ignore")
import numpy as np
from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg

A1 = np.array([1.0, 0.0]); A2 = np.array([0.5, np.sqrt(3) / 2])
FRAC = {"A": np.array([0.0, 0.0]), "B": np.array([1 / 3, 1 / 3])}
def cart(s, c):
    f = FRAC[s] + np.asarray(c, float); return f[0] * A1 + f[1] * A2
def pairs(dist, tol=1e-6):
    seen, out = set(), []
    for s in "AB":
        for t in "AB":
            for n1 in range(-3, 4):
                for n2 in range(-3, 4):
                    if abs(np.linalg.norm(cart(t, (n1, n2)) - cart(s, (0, 0))) - dist) < tol:
                        a, b = sorted([(s, 0, 0), (t, n1, n2)])
                        canon = ((a[0], 0, 0), (b[0], b[1] - a[1], b[2] - a[2]))
                        if canon not in seen:
                            seen.add(canon); out.append(canon)
    return out

class ChainCylinder(CouplingMPOModel):
    def init_sites(self, p):
        return SpinSite(S=1.0, conserve="Sz")
    def init_lattice(self, p):
        W, Lx = p["W"], p["Lx"]
        return Lattice([Lx], [self.init_sites(p)] * (2 * W),
                       positions=np.array([[0.0, i] for i in range(2 * W)]),
                       basis=[[1.0, 0.0]], bc="periodic", bc_MPS="infinite")
    def init_terms(self, p):
        W = p["W"]
        idx = lambda s, i: 2 * (i % W) + (0 if s == "A" else 1)
        dJ = {1: p.get("dJ1", 0.0), 3: p.get("dJ3", 0.0)}  # XXZ: J^zz = J + dJ, J^xy = J
        for J, dist, n in ((p["J1"], 1 / np.sqrt(3), 1), (p["J3"], 2 / np.sqrt(3), 3)):
            for (s, _, _), (t, n1, n2) in pairs(dist):
                for i in range(W):
                    u1, u2 = idx(s, i), idx(t, i + n1)
                    self.add_coupling(J + dJ[n], u1, "Sz", u2, "Sz", [n2])
                    self.add_coupling(0.5 * J, u1, "Sp", u2, "Sm", [n2], plus_hc=True)
        for u in range(2 * W):
            self.add_onsite(-p["D"], u, "Sz Sz"); self.add_onsite(-p["h"], u, "Sz")

def run(J3, D, W, pattern, h, chi, J1=-1.0):
    Lx = max(2, len(pattern))
    M = ChainCylinder(dict(J1=J1, J3=J3, D=D, h=h, W=W, Lx=Lx))
    st = []
    for x in range(Lx):
        st += ["up" if pattern[x % len(pattern)] == "u" else "down"] * (2 * W)
    psi = MPS.from_product_state(M.lat.mps_sites(), st, bc="infinite")
    t = time.time()
    info = dmrg.run(psi, M, dict(mixer=True, max_E_err=1e-11, max_S_err=1e-7, max_sweeps=80,
                                 min_sweeps=6, trunc_params=dict(chi_max=chi, svd_min=1e-12)))
    sz = psi.expectation_value("Sz")
    return dict(J3=J3, D=D, W=W, pattern=pattern, h=h, chi=chi, E=float(info["E"]),
                M=float(np.mean(sz)), szmin=float(np.min(sz)),
                Smax=float(np.max(psi.entanglement_entropy())), chi_used=int(max(psi.chi)),
                time=time.time() - t)

if __name__ == "__main__":
    a = sys.argv
    print(json.dumps(run(float(a[1]), float(a[2]), int(a[3]), a[4], float(a[5]), int(a[6]))), flush=True)
