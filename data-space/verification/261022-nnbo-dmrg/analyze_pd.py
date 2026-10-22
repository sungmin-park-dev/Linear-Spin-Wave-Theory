"""Lower envelope of E(h) = E(0) - M h over fixed-pattern sectors, per (J3, D).
Usage: python analyze_pd.py file.jsonl [hmax]"""
import sys, json
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
hmax = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
groups = defaultdict(list)
best = {}
for r in rows:  # keep the lowest energy per (J3, D, pattern) over initial states
    k = (r["J3"], r["D"], r["pattern"])
    r.setdefault("init", "prod")
    if k in best:
        other = best[k]
        r["dE_init"] = other["dE_init"] = abs(r["E_per_site"] - other["E_per_site"])
    if k not in best or r["E_per_site"] < best[k]["E_per_site"]:
        best[k] = r
for r in best.values():
    groups[(r["J3"], r["D"])].append(r)

for (J3, D), rs in sorted(groups.items()):
    print(f"\nJ3={J3} D={D}  ({len(rs)} patterns)")
    for r in sorted(rs, key=lambda r: r["M"]):
        print(f"  {r['pattern']:7s} M={r['M']:+.4f} E0={r['E_per_site']:.8f} "
              f"[{r['init']}{', dE_init=%.1e' % r['dE_init'] if 'dE_init' in r else ''}] trunc={r['trunc_err']:.1e} szmin={r['szmin']:+.3f} chi={r['chi_used']} t={r['time']:.0f}s")
    # walk the lower envelope from h=0 upward
    h, seq = 0.0, []
    cur = min(rs, key=lambda r: (r["E_per_site"], -r["M"]))
    seq.append((0.0, cur))
    while True:
        best = None
        for r in rs:
            if r["M"] > cur["M"] + 1e-9:
                hc = (r["E_per_site"] - cur["E_per_site"]) / (r["M"] - cur["M"])
                if hc >= h - 1e-12 and (best is None or hc < best[0] - 1e-12
                                        or (abs(hc - best[0]) <= 1e-12 and r["M"] > best[1]["M"])):
                    best = (hc, r)
        if best is None or best[0] > hmax:
            break
        h, cur = best
        seq.append((h, cur))
    print("  envelope: " + "  ->  ".join(f"[h>{hc:.4f}] {r['pattern']}(M={r['M']:.3f})" for hc, r in seq))
    skipped = sorted({r["pattern"] for r in rs} - {r["pattern"] for _, r in seq})
    print(f"  never stable (within these candidates): {skipped}")
