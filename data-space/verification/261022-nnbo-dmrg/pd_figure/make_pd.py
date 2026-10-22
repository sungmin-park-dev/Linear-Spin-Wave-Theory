"""(J3, D) map from W=3 iDMRG: nature of the lowest M=1/3 state (collinear vs canted),
h_c1/h_sat and 1/3-plateau width/h_sat, with the experimental values (2 K) as contour levels.
Run from ~/nnbo-dmrg:  python pd_figure/make_pd.py
Writes pd_figure/pd_points.csv, pd_figure/pd_points.json, pd_figure/pd_map.png."""
import os, sys, json, csv
sys.path.insert(0, os.getcwd())
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mh_hull import load, mh

EXP_RATIO = 4.7 / 8.3      # h_c1 / h_sat (2 K)
EXP_WIDTH = 2.2 / 8.3      # flat plateau 5.2-7.4 T, width / h_sat

rows = []
for (J3, D), d in sorted(load().items()):
    d.setdefault(1.0, dict(E_per_site=-D - 1.5 + 1.5 * J3, label="u(exact)", chi=0, trunc_err=0.0, szmin=1.0))
    m13 = d.get(round(1 / 3, 6))
    if m13 is None or 0.0 not in {round(M, 6) for M in d if abs(M) < 1e-6} and min(abs(M) for M in d) > 1e-6:
        continue
    hull, steps = mh(d)
    hsat, hc1 = steps[-1][0], steps[1][0]
    p = [(lo, hi) for lo, hi, M in steps if abs(M - 1 / 3) < 1e-4]
    lo, hi = p[0] if p else (np.nan, np.nan)
    collinear = m13["szmin"] < 0
    rows.append(dict(J3=J3, D=D, state="collinear" if collinear else "canted", h_c1=hc1, h_sat=hsat,
                     ratio=hc1 / hsat, plateau_lo=lo, plateau_hi=hi,
                     width_over_hsat=(hi - lo) / hsat if collinear else np.nan,
                     hi_over_hsat=hi / hsat if collinear else np.nan,
                     m13_label=m13["label"], m13_szmin=m13["szmin"],
                     hull=[[round(a, 5), round(M, 5)] for a, _, M in steps]))

out = os.path.join("pd_figure", "pd_points")
with open(out + ".json", "w") as f:
    json.dump(rows, f, indent=1, default=float)
with open(out + ".csv", "w", newline="") as f:
    w = csv.writer(f)
    keys = ["J3", "D", "state", "h_c1", "h_sat", "ratio", "plateau_lo", "plateau_hi", "width_over_hsat",
            "hi_over_hsat", "m13_label", "m13_szmin"]
    w.writerow(keys)
    for r in rows:
        w.writerow([f"{r[k]:.5f}" if isinstance(r[k], float) else r[k] for k in keys])

J = np.array([r["J3"] for r in rows]); Dv = np.array([r["D"] for r in rows])
R = np.array([r["ratio"] for r in rows]); Wd = np.array([r["width_over_hsat"] for r in rows])
col = np.array([r["state"] == "collinear" for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, Z, lev, name in ((axes[0], R, EXP_RATIO, "h_c1 / h_sat"), (axes[1], Wd, EXP_WIDTH, "1/3 plateau width / h_sat")):
    ok = np.isfinite(Z)
    if ok.sum() >= 4:
        try:
            cs = ax.tricontour(J[ok], Dv[ok], Z[ok], levels=8, cmap="viridis", linewidths=0.8)
            ax.clabel(cs, fontsize=7, fmt="%.2f")
            ax.tricontour(J[ok], Dv[ok], Z[ok], levels=[lev], colors="red", linewidths=2)
        except Exception as e:
            print("contour skipped:", e)
    ax.scatter(J[col], Dv[col], marker="s", c="tab:blue", label="M=1/3 collinear (↑↑↓ chains)", zorder=3)
    ax.scatter(J[~col], Dv[~col], marker="o", c="tab:orange", label="M=1/3 canted / uniform", zorder=3)
    for x, y, z in zip(J, Dv, Z):
        if np.isfinite(z):
            ax.annotate(f"{z:.2f}", (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7)
    ax.set_xlabel("J3 / |J1|"); ax.set_title(f"{name}  (red: experiment {lev:.3f})", fontsize=10)
axes[0].set_ylabel("D / |J1|"); axes[0].set_yscale("log"); axes[0].legend(fontsize=7, loc="upper left")
fig.suptitle("S=1 honeycomb J1-J3-D, h || z: W=3 iDMRG (chi 128-256), lowest E0 per M over ring/axial/random starts",
             fontsize=9)
fig.tight_layout(); fig.savefig(out.replace("pd_points", "pd_map") + ".png", dpi=150)
print(f"{len(rows)} points -> pd_figure/pd_map.png, pd_points.csv/json")
for r in rows:
    print(f"J3={r['J3']:.2f} D={r['D']:<5} {r['state']:9s} ratio={r['ratio']:.3f} width/hs={r['width_over_hsat']:.3f} "
          f"hi/hs={r['hi_over_hsat']:.3f} h_c1={r['h_c1']:.4f} h_sat={r['h_sat']:.4f} [{r['m13_label']}]")
