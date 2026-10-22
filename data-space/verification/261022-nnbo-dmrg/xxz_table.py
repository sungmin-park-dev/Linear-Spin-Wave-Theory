"""XXZ test table: per (D, dJ1), lowest E0 per M over patterns / random / axial starts (W=3),
lower convex hull -> M(h); saturated state exact.  Delta theta = D/2 - 2 dJ1 (high-T, |J1| units).
Usage: python xxz_table.py [file.jsonl]"""
import sys, json
from collections import defaultdict
from mh_hull import mh

f = sys.argv[1] if len(sys.argv) > 1 else "dmrg_xxz_J0.40.jsonl"
J3 = 0.40
pts = defaultdict(dict)
for line in open(f):
    if not line.strip():
        continue
    r = json.loads(line)
    r["label"] = f"{r['pattern']}[{'ring' if r['init'] == 'prod' else r['init']}]"
    d = pts[(r["D"], r["dJ1"])]
    M = round(r["M"], 6)
    if M not in d or r["E_per_site"] < d[M]["E_per_site"]:
        d[M] = r
print("D     dJ1   | dTheta | h_c1    h_sat   | h_c1/h_sat | M=1/3 [lo,hi]      w/h_sat hi/h_sat | M=1/3 state      | steps")
for (D, dJ), d in sorted(pts.items()):
    d[1.0] = dict(E_per_site=-D + 1.5 * (-1 + dJ) + 1.5 * J3, label="u(exact)", szmin=1.0)
    hull, steps = mh(d)
    hsat = steps[-1][0]; hc1 = steps[1][0]
    p = [(lo, hi) for lo, hi, M in steps if abs(M - 1 / 3) < 1e-4]
    lo, hi = p[0] if p else (float("nan"),) * 2
    r = d.get(round(1 / 3, 6), {"label": "-", "szmin": float("nan")})
    print(f"{D:<5} {dJ:+.2f} | {D / 2 - 2 * dJ:.3f}  | {hc1:.4f}  {hsat:.4f} | {hc1 / hsat:.3f}      | "
          f"[{lo:.4f},{hi:.4f}] {(hi - lo) / hsat:.3f}   {hi / hsat:.3f}    | {r['label']:12s} {r['szmin']:+.2f} | "
          + " ".join(f"{M:.3f}@{a:.4f}" for a, _, M in steps[1:]))
    for M in sorted(d):
        if M == 1.0:
            continue
        x = d[M]
        print(f"      M={M:.4f} E0={x['E_per_site']:.8f} {x['label']} trunc={x.get('trunc_err', 0):.1e} szmin={x['szmin']:+.3f}")
