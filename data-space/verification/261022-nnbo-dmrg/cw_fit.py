"""Curie-Weiss fits of NLCE chi(T): 1/chi_a = (T - theta_a)/C_a fitted on temperature windows.
Usage: python cw_fit.py nlce/chi_*.json"""
import sys, json
import numpy as np

WINDOWS = [(2, 6), (3, 8), (4, 10), (5, 15), (8, 20)]
for f in sys.argv[1:]:
    r = json.load(open(f))
    T = np.array(r["T"]); n = len(r["chi_zz"])
    print(f"\n{f}: J3={r['J3']} D={r['D']} nmax={r['nmax']}  exact high-T: dtheta = D/2 = {r['D'] / 2:.4f}")
    # convergence check: last two orders at T = 2, 3, 5
    for Tc in (2.0, 3.0, 5.0):
        k = int(np.argmin(abs(T - Tc)))
        dz = abs(r["chi_zz"][-1][k] - r["chi_zz"][-2][k]) / r["chi_zz"][-1][k]
        print(f"   rel. change n-1 -> n at T={T[k]:.2f}: {dz:.1e}")
    for lo, hi in WINDOWS:
        m = (T >= lo) & (T <= hi)
        res = {}
        for a in ("zz", "xx"):
            chi = np.array(r["chi_" + a][-1])[m]
            slope, icpt = np.polyfit(T[m], 1 / chi, 1)
            C = 1 / slope; theta = -icpt * C
            res[a] = (theta, C)
        tz, tx = res["zz"][0], res["xx"][0]
        tavg = (tz + 2 * tx) / 3
        print(f"  window T/|J1| in [{lo},{hi}]: theta_c*={tz:.4f}  theta_ab={tx:.4f}  dtheta={tz - tx:.4f}"
              f"  ({(tz - tx) / (r['D'] / 2):.2f} x D/2)  theta_avg={tavg:.4f}  C_z={res['zz'][1]:.3f} C_x={res['xx'][1]:.3f}")
