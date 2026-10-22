"""Fixed total-Sz sector on a long unit cell (Lx chains), random product-state start.
Same model/algorithm as dmrg_standalone.py at h=0; E(h) = E(0) - M h.
Usage: python dmrg_sector.py J3 D W Lx n_down seed chi   -> prints one JSON line.
n_down sites of 2*W*Lx start 'down', the rest 'up', positions shuffled with seed.
seed < 0: 'uniform' start in the same sector -- 2*n_down sites in state Sz=0 spread evenly over
the cell, the rest 'up' (no down chains; a seed for canted / uniformly magnetized states)."""
import sys, json, time, warnings
warnings.simplefilter("ignore")
import numpy as np
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from dmrg_standalone import ChainCylinder


def run(J3, D, W, Lx, n_down, seed, chi, J1=-1.0):
    M = ChainCylinder(dict(J1=J1, J3=J3, D=D, h=0.0, W=W, Lx=Lx))
    N = 2 * W * Lx
    if seed < 0:
        n0 = 2 * n_down
        st = ["0.0" if (i + 1) * n0 // N > i * n0 // N else "up" for i in range(N)]
    else:
        st = ["down"] * n_down + ["up"] * (N - n_down)
        np.random.default_rng(seed).shuffle(st)
    psi = MPS.from_product_state(M.lat.mps_sites(), st, bc="infinite")
    t = time.time()
    info = dmrg.run(psi, M, dict(mixer=True, max_E_err=1e-11, max_S_err=1e-7, max_sweeps=80,
                                 min_sweeps=6, trunc_params=dict(chi_max=chi, svd_min=1e-12)))
    sz = psi.expectation_value("Sz")
    stats = info["sweep_statistics"]
    chain_sz = sz.reshape(Lx, 2 * W).mean(axis=1)  # mean Sz per chain (fixed x)
    return dict(J3=J3, D=D, W=W, Lx=Lx, chi=chi, n_down=n_down, seed=seed, E_per_site=float(info["E"]),
                M=float(np.mean(sz)), M_exact=(N - 2 * n_down) / N, szmin=float(np.min(sz)),
                chain_sz=[round(float(c), 4) for c in chain_sz],
                trunc_err=float(stats["max_trunc_err"][-1]),
                Smax=float(np.max(psi.entanglement_entropy())), chi_used=int(max(psi.chi)),
                n_sweeps=int(len(stats["sweep"])), time=time.time() - t)


if __name__ == "__main__":
    a = sys.argv
    print(json.dumps(run(float(a[1]), float(a[2]), int(a[3]), int(a[4]), int(a[5]), int(a[6]), int(a[7]))),
          flush=True)
