import os, sys, warnings, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); warnings.simplefilter('ignore')
import numpy as np, spintoolkit as stk
from spintoolkit import SpinModel, Site, Term, SpinState, ExternalConditions, IncommensurateStructure
from convert import to_magnopy
S, J1, h = 1.0, -1.0, 0.5
J2 = 1 / (4 * np.cos(2 * np.pi / 5))
I = np.eye(3)
m = SpinModel(np.eye(2), [Site("A", [0, 0], S)],
              [Term.bilinear(("A", (0, 0)), ("A", (1, 0)), J1 * I),
               Term.bilinear(("A", (0, 0)), ("A", (0, 1)), J1 * I),
               Term.bilinear(("A", (0, 0)), ("A", (2, 0)), J2 * I),
               Term.zeeman("A", I)], metadata={"model_id": "j1j2-chain"})
q = 2 * np.pi / 5
Jxy = J1 * np.cos(q) + J1 + J2 * np.cos(2 * q); Jzz = 2 * J1 + J2
ct = h / (2 * S * (Jzz - Jxy)); st_ = np.sqrt(1 - ct ** 2)
cond = ExternalConditions(field=(0, 0, h))
struct = IncommensurateStructure(m.fingerprint(), (0.2, 0.0), (0, 0, 1), {"A": (st_, 0, ct)})
rng = np.random.default_rng(2); ks = rng.uniform(-np.pi, np.pi, (200, 2))
Q = np.array([q, 0])
allk = np.concatenate([ks + n * Q for n in range(5)])
sp = stk.solve_spiral_lswt(m, struct, cond, settings=stk.LSWTSettings(k_points=allk))
wrot = sp.rotating.eigenvalues[:, 0].reshape(5, -1).T          # (nk, 5)
best = None
for sgn in (+1, -1):
    state = SpinState.from_function(m, [[5, 0], [0, 1]],
        lambda s, c: np.array([st_ * np.cos(sgn * q * c[0]), st_ * np.sin(sgn * q * c[0]), ct]))
    l, keys, A2 = to_magnopy(m, state, (0, 0, h))
    mag = np.array([np.sort(l.omega([k[0], k[1], 0]).real) for k in ks])
    d = np.max(np.abs(np.sort(wrot, 1) - mag))
    r_lab = stk.solve_lswt(m, state, cond, settings=stk.LSWTSettings(mesh=(10, 50)))
    print(f"sense {sgn:+d}: max |spiral(rotating frame) - magnopy 5-site supercell| = {d:.2e}")
    print(f"   ZPE spiral {stk.solve_spiral_lswt(m, struct, cond, settings=stk.LSWTSettings(mesh=(50,50))).zero_point_energy:.8f}"
          f"  stk 5-site supercell {r_lab.zero_point_energy:.8f}")
