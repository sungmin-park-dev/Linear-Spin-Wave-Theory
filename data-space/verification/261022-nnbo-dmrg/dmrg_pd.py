"""Phase-diagram point: same model/algorithm as dmrg_standalone.py, run at h=0.
Sz is conserved, so E(h) = E(0) - M h per site for each fixed-pattern sector.
Usage: python dmrg_pd.py J3 D W pattern chi [init [dJ1]]   (dJ1: XXZ J1^zz - J1^xy)   -> prints one JSON line.
init: "prod" (default, pattern product state) or "randN": the unit cell's site states
shuffled with seed N (same total Sz, i.e. same sector), or "axial": chains along the cylinder
axis (X-Z zigzag chains; chain index = circumference index i), chain i gets pattern[i % p].
Pattern letters: u, d, 0 (Sz=0 chain, seed for a canted chain), h (up/0 alternating chain).
"prod" puts the chains around the circumference (rings of length W); "axial" needs p | W."""
import sys, json, time, warnings
warnings.simplefilter("ignore")
import numpy as np
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from dmrg_standalone import ChainCylinder


def run(J3, D, W, pattern, chi, init="prod", dJ1=0.0, J1=-1.0):
    Lx = 2 if init == "axial" else max(2, len(pattern))
    if init == "axial" and W % len(pattern):
        raise ValueError("axial init needs len(pattern) | W")
    M = ChainCylinder(dict(J1=J1, J3=J3, D=D, dJ1=dJ1, h=0.0, W=W, Lx=Lx))
    st = []
    chain = {"u": ["up"] * (2 * W), "d": ["down"] * (2 * W), "0": ["0.0"] * (2 * W),
             "h": ["up", "0.0"] * W}  # '0': Sz=0 (canted-seed) chain, 'h': half-polarized chain
    for x in range(Lx):
        st += chain[pattern[x % len(pattern)]]
    if init == "axial":
        st = ["up" if pattern[(u // 2) % len(pattern)] == "u" else "down"
              for x in range(Lx) for u in range(2 * W)]
    if init.startswith("rand"):
        np.random.default_rng(int(init[4:])).shuffle(st)
    psi = MPS.from_product_state(M.lat.mps_sites(), st, bc="infinite")
    t = time.time()
    info = dmrg.run(psi, M, dict(mixer=True, max_E_err=1e-11, max_S_err=1e-7, max_sweeps=80,
                                 min_sweeps=6, trunc_params=dict(chi_max=chi, svd_min=1e-12)))
    sz = psi.expectation_value("Sz")
    stats = info["sweep_statistics"]
    return dict(J3=J3, D=D, dJ1=dJ1, W=W, chi=chi, pattern=pattern, init=init, E_per_site=float(info["E"]),
                M=float(np.mean(sz)), szmin=float(np.min(sz)),
                trunc_err=float(stats["max_trunc_err"][-1]),
                trunc_err_max_all=float(np.max(stats["max_trunc_err"])),
                chain_sz=[round(float(c), 4) for c in np.asarray(sz).reshape(Lx, 2 * W).mean(axis=1)],
                site_sz=[round(float(c), 3) for c in sz],
                Smax=float(np.max(psi.entanglement_entropy())), chi_used=int(max(psi.chi)),
                n_sweeps=int(len(stats["sweep"])), time=time.time() - t)


if __name__ == "__main__":
    a = sys.argv
    print(json.dumps(run(float(a[1]), float(a[2]), int(a[3]), a[4], int(a[5]), *a[6:7], *map(float, a[7:8]))), flush=True)
