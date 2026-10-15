import os, sys, warnings, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); warnings.simplefilter('ignore')
import numpy as np, spintoolkit as stk
from spintoolkit import SpinModel, Site, Term, SpinState, ExternalConditions
from spintoolkit.models import square_heisenberg, neel_state, triangular_heisenberg, state_120, polarized_state
from spintoolkit.models.honeycomb import honeycomb_ferromagnet
from convert import to_magnopy

rng = np.random.default_rng(1)

def compare(name, model, state, field=(0, 0, 0), mesh=(36, 36), nk=300):
    cond = ExternalConditions(field=field)
    l, keys, A2 = to_magnopy(model, state, field)
    B = 2 * np.pi * np.linalg.inv(A2).T          # reciprocal rows of magnetic cell
    ks = rng.uniform(-1, 1, (nk, 2)) @ B
    r = stk.solve_lswt(model, state, cond, settings=stk.LSWTSettings(k_points=ks))
    ns = len(keys)
    ours = np.sort(r.eigenvalues[:, :ns], axis=1)
    theirs = np.array([np.sort(l.omega([k[0], k[1], 0]).real) for k in ks])
    imag = max(np.abs(l.omega([k[0], k[1], 0]).imag).max() for k in ks[:20])
    # zero-point energy on the same mesh
    rm = stk.solve_lswt(model, state, cond, settings=stk.LSWTSettings(mesh=mesh))
    fr = (np.stack(np.meshgrid(np.arange(mesh[0]), np.arange(mesh[1]), indexing='ij'), -1).reshape(-1, 2) + 0.5) / np.array(mesh)
    kk = fr @ B
    delta = np.mean([0.5*np.sum(l.omega([k[0], k[1], 0]).real) + l.delta([k[0], k[1], 0]).real for k in kk])
    zpe_m = (l.E_2() + delta) / ns
    out = dict(model=name, sites=ns, band_max_abs_diff=float(np.max(np.abs(ours - theirs))),
               band_scale=float(np.max(ours)), magnopy_imag=float(imag),
               zpe_stk=float(rm.zero_point_energy), zpe_magnopy=float(zpe_m))
    print(json.dumps(out))
    return out

res = []
m = square_heisenberg(J=1.0, S=0.5)
res.append(compare("square AFM S=1/2, Neel", m, neel_state(m)))

h = 2.0; th = np.arccos(h / (8 * 0.5))
st = SpinState.from_function(m, [[1, 1], [-1, 1]],
        lambda s, c: np.array([(-1) ** (c[0] + c[1]) * np.sin(th), 0, np.cos(th)]))
res.append(compare("square AFM S=1/2, canted in h=0.5 h_sat", m, st, field=(0, 0, h)))

m = triangular_heisenberg(J=1.0, S=0.5)
res.append(compare("triangular AFM S=1/2, 120 deg", m, state_120(m)))

m = honeycomb_ferromagnet(J=1.0, D=0.1, S=0.5)
res.append(compare("honeycomb FM + 2nd-nbr DM (D=0.1)", m, polarized_state(m)))

# anisotropic exchange + single-ion, S=1: square lattice XXZ-like FM with off-diagonal term
lat = np.eye(2)
Jm = -np.array([[1.0, 0.15, 0], [0.15, 0.8, 0], [0, 0, 1.2]])
for S in (1.0, 2.5):
    mm = SpinModel(lat, [Site("A", [0, 0], S)],
                   [Term.bilinear(("A", (0, 0)), ("A", (1, 0)), Jm),
                    Term.bilinear(("A", (0, 0)), ("A", (0, 1)), Jm.T),
                    Term.onsite("A", np.diag([0, 0, -0.3])),
                    Term.zeeman("A", np.eye(3))], metadata={"model_id": "aniso"})
    res.append(compare(f"square FM, anisotropic J + single-ion D(Sz)^2, S={S}", mm,
                       polarized_state(mm), field=(0, 0, 0.2)))
    mr = SpinModel(lat, [Site("A", [0, 0], S)],
                   [Term.bilinear(("A", (0, 0)), ("A", (1, 0)), Jm),
                    Term.bilinear(("A", (0, 0)), ("A", (0, 1)), Jm.T),
                    Term.zeeman("A", np.eye(3))], metadata={"model_id": "aniso0"})
    res.append(compare(f"same, single-ion removed, S={S}", mr, polarized_state(mr), field=(0, 0, 0.2)))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json'), 'w'), indent=1)
