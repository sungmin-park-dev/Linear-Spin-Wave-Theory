"""Harmonic (classical + LSWT zero-point) phase diagram of the NBCP model.

Panels as in arXiv:2601.20963 Fig. 1: (h, J_PD) at J_Gamma = 0 and
(h, J_Gamma) at J_PD = 0, J = 0.075, J_z = 0.125 meV (J/J_z = 0.6), S = 1/2.
Candidates: 1-, 2-, 3- and 4-site (2x2) cells. The 3-site state is
classically degenerate along the global-z orbit, so it is evaluated on the
orbit and the zero-point-selected stable angle is used.
"""
import sys, json, warnings
from multiprocessing import Pool
W = 'pd-wt'
sys.path[:0] = [W + '/code-space', W]
import numpy as np
from model.nbcp.model import SUPERCELLS, build_model
from spintoolkit.methods.classical import classical_energy, classical_search, refine_classical
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.methods.lswt.run import LSWTError
from spintoolkit.methods.phase_competition import compare_states, magnetic_mesh
from spintoolkit.observables.texture import skyrmion_charge
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.states.spin_state import SpinState

K_DENSITY = 18
N_PHI = 12
FIELDS = np.round(np.arange(0.0, 0.5401, 0.02), 4)
COUPLINGS = np.round(np.arange(0.0, 0.03001, 0.0025), 5)


def Rz(p):
    c, s = np.cos(p), np.sin(p)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotate(state, p):
    return SpinState(state.model_ref, state.supercell,
                     {k: Rz(p) @ v for k, v in state.directions.items()}, state.provenance)


def lowest(model, cell, cond, previous):
    best = refine_classical(model, classical_search(model, SUPERCELLS[cell], cond).state, cond)
    if previous is not None:
        cont = refine_classical(model, previous, cond)
        if classical_energy(model, cont, cond) < classical_energy(model, best, cond) - 1e-12:
            best = cont
    return best


def three_site(model, state, cond):
    mesh = magnetic_mesh(state, K_DENSITY)
    e_cl = float(classical_energy(model, state, cond))
    rows = []
    for p in np.arange(N_PHI) * (np.pi / 3) / N_PHI:
        s = rotate(state, p)
        try:
            r = solve_lswt(model, s, cond, settings=LSWTSettings(mesh=mesh))
            rows.append((p, float(r.zero_point_energy), s))
        except LSWTError:
            rows.append((p, None, s))
    stable = [r for r in rows if r[1] is not None]
    if not stable:
        return {'status': 'unstable', 'classical': e_cl, 'harmonic': None,
                'stable_fraction': 0.0, 'state': state}
    p, ezp, s = min(stable, key=lambda r: r[1])
    return {'status': 'stable', 'classical': e_cl, 'harmonic': e_cl + ezp, 'phi': p,
            'stable_fraction': len(stable) / N_PHI, 'state': s}


def label_three(state):
    n = np.array(list(state.directions.values()))
    if np.all(np.abs(n[:, 2]) > 0.999):
        return 'P' if np.all(n[:, 2] > 0) else 'UUD'
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        if abs(n[j, 2] - n[k, 2]) < 0.02:
            return 'V' if np.linalg.norm(n[j] - n[k]) < 0.05 else 'Y'
    return 'Psi'


def label(name, state):
    n = np.array(list(state.directions.values()))
    if np.all(n[:, 2] > 0.999):
        return 'P'
    if name == 'three_msl':
        return label_three(state)
    if name == 'four_msl':
        q = skyrmion_charge(None, state) if False else None
    return {'one_msl': 'canted-1', 'two_msl': '2-site', 'four_msl': '4-site'}[name]


def sweep(args):
    axis, J = args
    params = {'Jxy': 0.075, 'Jz': 0.125, 'JPD': J if axis == 'PD' else 0.0,
              'JGamma': J if axis == 'Gamma' else 0.0}
    model = build_model(params)
    prev = {c: None for c in SUPERCELLS}
    out = []
    for h in FIELDS:
        cond = ExternalConditions(field=(0, 0, float(h)))
        states = {c: lowest(model, c, cond, prev[c]) for c in SUPERCELLS}
        prev.update(states)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reps = compare_states(model, {c: states[c] for c in ('one_msl', 'two_msl', 'four_msl')},
                                  cond, k_density=K_DENSITY, refine=False)
            t = three_site(model, states['three_msl'], cond)
        cands = []
        for r in reps:
            d = r.to_dict()
            lab = label(r.name, r.state)
            if r.name == 'four_msl' and d['skyrmion']['integer'] not in (None, 0):
                lab = 'SkX'
            cands.append({'cell': r.name, 'nsites': len(r.state.directions), 'label': lab,
                          'status': r.status, 'classical': d['classical_energy'],
                          'harmonic': d['harmonic_energy'], 'Q': d['skyrmion']['integer'],
                          'mz': d['magnetization'][2]})
        n3 = np.array(list(t['state'].directions.values()))
        cands.append({'cell': 'three_msl', 'nsites': 3, 'label': label_three(t['state']),
                      'status': t['status'], 'classical': t['classical'], 'harmonic': t['harmonic'],
                      'stable_fraction': t['stable_fraction'], 'phi': t.get('phi'),
                      'mz': float(0.5 * n3[:, 2].mean()), 'dirs': n3.round(6).tolist()})
        out.append({'axis': axis, 'J': float(J), 'h': float(h), 'candidates': cands,
                    'classical_winner': pick(cands, 'classical'), 'harmonic_winner': pick(cands, 'harmonic')})
        print(axis, J, h, out[-1]['harmonic_winner'], file=sys.stderr, flush=True)
    return out


def pick(cands, key, tol=1e-7):
    pool = [c for c in cands if c[key] is not None and (key == 'classical' or c['status'] == 'stable')]
    if not pool:
        return None
    e0 = min(c[key] for c in pool)
    tied = [c for c in pool if c[key] < e0 + tol]
    best = min(tied, key=lambda c: (c['nsites'] if c['label'] not in ('P',) else 0, c[key]))
    return {'label': best['label'], 'cell': best['cell']}


if __name__ == '__main__':
    tasks = [(a, J) for a in ('PD', 'Gamma') for J in COUPLINGS]
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        tasks = [('PD', 0.01)]
    with Pool(4) as pool:
        res = pool.map(sweep, tasks, chunksize=1)
    pts = [p for r in res for p in r]
    json.dump({'fields': FIELDS.tolist(), 'couplings': COUPLINGS.tolist(), 'points': pts},
              open('pd_scan.json' if len(sys.argv) == 1 else 'pd_test.json', 'w'), indent=1)
