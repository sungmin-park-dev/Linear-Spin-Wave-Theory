"""M(h) from the lower convex hull of E0(M) per (J3, D), W from env W (default 3), combining pattern and sector runs.
For each M the lowest E0 over all runs (patterns, initial states, chi) is used.
Usage: python mh_hull.py [--plot out.png] [J3 ...]"""
import sys, os, json, glob
from collections import defaultdict

WANT_W = int(os.environ.get("W", "3"))
FILES = sorted(glob.glob("dmrg_pd_J*.jsonl") + glob.glob("dmrg_pd_smallD2.jsonl") +
               glob.glob("dmrg_sector_J*.jsonl"))


def load():
    pts = defaultdict(dict)  # (J3, D) -> {M: row}
    for f in FILES:
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("W") != WANT_W:
                continue
            Mk = round(r.get("M_exact", r["M"]), 6)
            label = r.get("pattern") or f"Lx{r['Lx']}nd{r['n_down']}" + ("uni" if r['seed'] < 0 else f"rnd{r['seed']}")
            if r.get("pattern"):
                init = r.get("init", "prod")
                label += "[ring]" if init == "prod" else f"[{init}]"
            r["label"] = label
            d = pts[(r["J3"], r["D"])]
            if Mk not in d or r["E_per_site"] < d[Mk]["E_per_site"]:
                d[Mk] = r
    return pts


def lower_hull(points):
    """points: sorted list of (M, E). Returns lower convex hull vertices."""
    hull = []
    for p in points:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def mh(d):
    """M(h) steps: list of (h_lo, h_hi, M) with h = slope dE0/dM between hull vertices."""
    pts = sorted((M, r["E_per_site"]) for M, r in d.items() if M >= -1e-9)
    hull = lower_hull(pts)
    steps, hs = [], [0.0]
    for (M1, E1), (M2, E2) in zip(hull, hull[1:]):
        hs.append((E2 - E1) / (M2 - M1))
    hs.append(float("inf"))
    for i, (M, _) in enumerate(hull):
        steps.append((hs[i], hs[i + 1], M))
    return hull, steps


if __name__ == "__main__":
    args = sys.argv[1:]
    plot = None
    if "--plot" in args:
        plot = args[args.index("--plot") + 1]
        del args[args.index("--plot"):args.index("--plot") + 2]
    pts = load()
    keys = sorted(k for k in pts if not args or str(k[0]) in args or f"{k[0]:.2f}" in args)
    for k in keys:
        d = pts[k]
        hull, steps = mh(d)
        hullM = {round(M, 6) for M, _ in hull}
        print(f"\nJ3={k[0]} D={k[1]}: {len(d)} sectors")
        for M in sorted(d):
            r = d[M]
            print(f"  M={M:.4f} E0={r['E_per_site']:.8f} {'*' if M in hullM else ' '} {r['label']} "
                  f"chi={r['chi']} trunc={r['trunc_err']:.1e} szmin={r['szmin']:+.3f}")
        print("  M(h): " + "  ".join(f"[{lo:.4f},{hi:.4f}) M={M:.3f}" for lo, hi, M in steps))
        hsat = steps[-1][0]
        print(f"  h_sat={hsat:.4f}  h_c1(first rise)={steps[1][0]:.4f}  ratio={steps[1][0] / hsat:.3f}")
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for k in keys:
            _, steps = mh(pts[k])
            hsat = steps[-1][0]
            xs, ys = [], []
            for lo, hi, M in steps:
                hi = min(hi, 1.15 * hsat)
                xs += [lo, hi]; ys += [M, M]
            ax.plot(xs, ys, label=f"J3={k[0]}, D={k[1]}")
        ax.set_xlabel("h / |J1|"); ax.set_ylabel("M (per site, S=1)")
        ax.set_title("W=3 iDMRG: M(h) from lower convex hull of E0(M)")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plot, dpi=150)
        print("saved", plot)
