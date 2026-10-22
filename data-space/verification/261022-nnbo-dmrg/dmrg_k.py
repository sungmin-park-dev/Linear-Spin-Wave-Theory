"""J1-J3-D model plus NN Kitaev term K sum_<ij>_gamma S_i^g S_j^g, field h along z, no Sz conservation.
Same cylinder geometry as dmrg_standalone.py.  Bond labels: the two intrachain J1 bonds
(A-B at 30 deg and A-B at 150 deg, mirror images) are X and Y; the interchain J1 bond (vertical) is Z.
Cubic axes in the global frame (z = field direction):
  EX = (1/sqrt2, 1/sqrt6, 1/sqrt3), EY = (-1/sqrt2, 1/sqrt6, 1/sqrt3), EZ = (0, -2/sqrt6, 1/sqrt3).
(The in-plane projection of each axis is parallel to its bond; this differs from the edge-sharing
convention, projection perpendicular to bond, by a uniform 90 deg rotation about z, which commutes
with the rest of H, so energies and M are identical.)
Optional Gamma term on the same NN bonds: Gamma (S^a S^b + S^b S^a) on a gamma bond, (a, b) the other
two cubic axes.  High-T Curie-Weiss anisotropy (checked by ED, ed_theta.py):
theta_c* - theta_ab = D/2 - 2 Gamma (K drops out), so Gamma < 0 is easy-axis along c*.
Usage: python dmrg_k.py J3 D K h W pattern chi [Gamma]   -> prints one JSON line."""
import sys, json, time, warnings
warnings.simplefilter("ignore")
import numpy as np
from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Lattice
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from dmrg_standalone import pairs, cart

AX = {"X": np.array([1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(3)]),
      "Y": np.array([-1 / np.sqrt(2), 1 / np.sqrt(6), 1 / np.sqrt(3)]),
      "Z": np.array([0.0, -2 / np.sqrt(6), 1 / np.sqrt(3)])}


OTHER = {"X": ("Y", "Z"), "Y": ("Z", "X"), "Z": ("X", "Y")}


def bond_label(v):
    if abs(v[0]) < 1e-6:
        return "Z"
    return "X" if v[0] * v[1] > 0 else "Y"


def proj_ops(e):
    """e.S = e_z Sz + c Sp + c* Sm with c = (e_x - i e_y)/2."""
    c = (e[0] - 1j * e[1]) / 2
    return [(e[2], "Sz"), (c, "Sp"), (np.conj(c), "Sm")]


class KitaevCylinder(CouplingMPOModel):
    def init_sites(self, p):
        return SpinSite(S=1.0, conserve=p.get("conserve", None))

    def init_lattice(self, p):
        W, Lx = p["W"], p["Lx"]
        return Lattice([Lx], [self.init_sites(p)] * (2 * W),
                       positions=np.array([[0.0, i] for i in range(2 * W)]),
                       basis=[[1.0, 0.0]], bc="periodic", bc_MPS="infinite")

    def init_terms(self, p):
        W, K, G = p["W"], p["K"], p.get("G", 0.0)
        idx = lambda s, i: 2 * (i % W) + (0 if s == "A" else 1)
        for J, dist in ((p["J1"], 1 / np.sqrt(3)), (p["J3"], 2 / np.sqrt(3))):
            for (s, _, _), (t, n1, n2) in pairs(dist):
                lab = bond_label(cart(t, (n1, n2)) - cart(s, (0, 0)))
                for i in range(W):
                    u1, u2 = idx(s, i), idx(t, i + n1)
                    self.add_coupling(J, u1, "Sz", u2, "Sz", [n2])
                    self.add_coupling(0.5 * J, u1, "Sp", u2, "Sm", [n2], plus_hc=True)
                    if K != 0 and dist < 0.6:  # Kitaev on NN (J1) bonds only
                        for a, oa in proj_ops(AX[lab]):
                            for b, ob in proj_ops(AX[lab]):
                                if abs(a * b) > 1e-14:
                                    self.add_coupling(K * a * b, u1, oa, u2, ob, [n2])
                    if G != 0 and dist < 0.6:  # Gamma on NN (J1) bonds only
                        ea, eb = (AX[c] for c in OTHER[lab])
                        for x, y in ((ea, eb), (eb, ea)):
                            for a, oa in proj_ops(x):
                                for b, ob in proj_ops(y):
                                    if abs(a * b) > 1e-14:
                                        self.add_coupling(G * a * b, u1, oa, u2, ob, [n2])
        for u in range(2 * W):
            self.add_onsite(-p["D"], u, "Sz Sz"); self.add_onsite(-p["h"], u, "Sz")


def run(J3, D, K, h, W, pattern, chi, G=0.0, J1=-1.0):
    Lx = max(2, len(pattern))
    M = KitaevCylinder(dict(J1=J1, J3=J3, D=D, K=K, G=G, h=h, W=W, Lx=Lx))
    st = []
    for x in range(Lx):
        st += ["up" if pattern[x % len(pattern)] == "u" else "down"] * (2 * W)
    psi = MPS.from_product_state(M.lat.mps_sites(), st, bc="infinite", dtype=complex)
    t = time.time()
    info = dmrg.run(psi, M, dict(mixer=True, max_E_err=1e-11, max_S_err=1e-7, max_sweeps=80,
                                 min_sweeps=6, trunc_params=dict(chi_max=chi, svd_min=1e-12)))
    sx, sy, sz = (np.real(psi.expectation_value(o)) for o in ("Sx", "Sy", "Sz"))
    stats = info["sweep_statistics"]
    return dict(J3=J3, D=D, K=K, G=G, h=h, W=W, chi=chi, init=pattern, E_per_site=float(np.real(info["E"])),
                M=float(np.mean(sz)), szmin=float(np.min(sz)),
                Mperp_max=float(np.max(np.hypot(sx, sy))), Mx=float(np.mean(sx)), My=float(np.mean(sy)),
                trunc_err=float(stats["max_trunc_err"][-1]),
                Smax=float(np.max(psi.entanglement_entropy())), chi_used=int(max(psi.chi)),
                n_sweeps=int(len(stats["sweep"])), time=time.time() - t)


if __name__ == "__main__":
    a = sys.argv
    print(json.dumps(run(float(a[1]), float(a[2]), float(a[3]), float(a[4]), int(a[5]), a[6], int(a[7]), *map(float, a[8:9]))),
          flush=True)
