"""Larger magnetic cells for the LSWT-unstable (hatched) region of the NBCP harmonic phase diagram.

The 2026-10-01 harmonic phase diagram
(`data-space/verification/261001-nbcp-harmonic-phase-diagram/`) compared
1-, 2-, 3- and 4-site cells; where every candidate was LSWT-unstable the
cell was left hatched. This script revisits each hatched point, and two
reference points inside Y and V, with larger commensurate cells:
3 x 3 (9 sites), 2 sqrt3 x 2 sqrt3 (12), 4 x 4 (16), 3 sqrt3 x 3 sqrt3 (27)
and 6 x 6 (36), besides the original cells.

On each cell the classical minimum is the lowest of random starts and of
the best states of the smaller cells it contains (tiled), all refined with
`refine_classical`, so a larger cell can only lower the energy. The lowest
state over all cells is then tested for LSWT stability with
`compare_states` (mesh density 18, as before). If its classical energy is
flat under a common rotation about z, twelve rotation angles in [0, pi/3)
are tested and the stable angle of lowest zero-point energy is kept, as for
the three-site states. The state is described by its cell, magnetization,
skyrmion number and the strongest Fourier components of its in-plane and
longitudinal spin parts.

Classical energies and harmonic stability only; cells up to 36 sites; an
incommensurate modulation is detected only through its commensurate
approximants.

Run from the repository root (about 30 minutes on four cores):
    python examples/nbcp_hatched_region_search.py
"""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

from model.nbcp.model import LATTICE, SUPERCELLS, build_model
from spintoolkit.methods.classical import classical_energy, refine_classical
from spintoolkit.methods.phase_competition import compare_states
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions

SCANS = ROOT / 'data-space/verification/261001-nbcp-harmonic-phase-diagram'
OUT = ROOT / 'data-space/verification/261005-nbcp-hatched-region'
CELLS = {
    '1': [[1, 0], [0, 1]], '2': SUPERCELLS['two_msl'].tolist(), '3': SUPERCELLS['three_msl'].tolist(),
    '4': SUPERCELLS['four_msl'].tolist(), '9': [[3, 0], [0, 3]], '12': [[4, 2], [2, 4]],
    '16': [[4, 0], [0, 4]], '27': [[6, 3], [3, 6]], '36': [[6, 0], [0, 6]],
}
#: smaller cells whose states tile each cell
CONTAINS = {'4': ['2'], '9': ['3'], '12': ['2', '3', '4'], '16': ['2', '4'], '27': ['3', '9'],
            '36': ['2', '3', '4', '9', '12']}
STARTS = 12
K_DENSITY = 18
N_PHI = 12
Y_FIELD = 4.645 * 0.05788381806 * 0.2       # 0.2 T as Zeeman energy, meV
V_FIELD = 4.645 * 0.05788381806 * 1.4       # 1.4 T


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hatched_points():
    points = []
    for name in ('scan-J0-0.03.json', 'scan-JGamma-0.035-0.10.json', 'scan-JPD-0.035-0.10.json'):
        for p in json.loads((SCANS / name).read_text())['points']:
            if p['h'] > 0 and p['harmonic_winner'] is None:
                best = min(p['candidates'], key=lambda c: c['classical'])
                points.append({'axis': p['axis'], 'J': p['J'], 'h': p['h'], 'kind': 'hatched',
                               'old_classical_winner': p['classical_winner'], 'old_classical_energy': best['classical']})
    return points


def rotate(state, angle):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return SpinState(state.model_ref, state.supercell, {k: tuple(R @ np.asarray(v)) for k, v in state.directions.items()},
                     state.provenance)


def tiled(model, cell, small):
    return SpinState.from_function(model, cell, lambda site, c: small.directions[(site, small.reduce_cell(c))],
                                   {'origin': 'tiled'})


def random_state(model, cell, rng):
    def direction(site, c):
        v = rng.normal(size=3)
        return tuple(v / np.linalg.norm(v))
    return SpinState.from_function(model, cell, direction, {'origin': 'random'})


def fourier(model, state):
    """Strongest Fourier components (in primitive reciprocal units) of the in-plane and z spin parts."""
    keys = list(state.directions)
    n = np.array([state.directions[k] for k in keys])
    positions = np.array([np.asarray(c) for _, c in keys], float)      # primitive cell coordinates (one site per cell)
    inv = np.linalg.inv(np.asarray(state.supercell, float))
    ks = {tuple(np.round((np.array([a, b]) @ inv.T) % 1, 6)) for a in range(-6, 7) for b in range(-6, 7)}
    rows = []
    for q in ks:
        phase = np.exp(2j * np.pi * positions @ np.array(q))
        perp = np.abs(phase @ (n[:, 0] + 1j * n[:, 1])) ** 2 + np.abs(phase @ (n[:, 0] - 1j * n[:, 1])) ** 2
        par = np.abs(phase @ (n[:, 2] - n[:, 2].mean() * (np.linalg.norm(q) < 1e-9))) ** 2
        rows.append((q, perp / (2 * len(n) ** 2), par / len(n) ** 2))
    perp = sorted(rows, key=lambda r: -r[1])[:3]
    par = sorted(rows, key=lambda r: -r[2])[:3]
    return {'in_plane': [[list(r[0]), round(float(r[1]), 4)] for r in perp],
            'longitudinal': [[list(r[0]), round(float(r[2]), 4)] for r in par]}


def search_point(point):
    t0 = time.monotonic()
    params = {'Jxy': 0.075, 'Jz': 0.125, 'JPD': point['J'] if point['axis'] == 'PD' else 0.0,
              'JGamma': point['J'] if point['axis'] == 'Gamma' else 0.0}
    model = build_model(params)
    cond = ExternalConditions(field=(0, 0, float(point['h'])))
    rng = np.random.default_rng(int(1e6 * point['J']) + int(1e3 * point['h']))
    best_state, energies = {}, {}
    for name, cell in CELLS.items():
        starts = [random_state(model, cell, rng) for _ in range(STARTS)]
        starts += [tiled(model, cell, best_state[s]) for s in CONTAINS.get(name, [])]
        found = None
        for start in starts:
            s = refine_classical(model, start, cond)
            e = float(classical_energy(model, s, cond))
            if found is None or e < found[0] - 1e-12:
                found = (e, s)
        energies[name], best_state[name] = found
    order = sorted(energies, key=lambda k: (round(energies[k], 9), len(best_state[k].directions)))
    winner = order[0]
    state = best_state[winner]
    e_cl = energies[winner]
    flat = abs(float(classical_energy(model, rotate(state, 0.37), cond)) - e_cl) < 1e-10
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        candidates = [state] if not flat else [rotate(state, a) for a in np.arange(N_PHI) * (np.pi / 3) / N_PHI]
        reports = compare_states(model, {f'r{i}': s for i, s in enumerate(candidates)}, cond, k_density=K_DENSITY, refine=False)
    stable = [r for r in reports if r.status == 'stable']
    pick = min(stable, key=lambda r: r.to_dict()['harmonic_energy']) if stable else reports[0]
    d = pick.to_dict()
    n = np.array(list(pick.state.directions.values()))
    out = dict(point)
    out.update({
        'cell_energies_meV_per_site': energies, 'winner_cell': winner, 'winner_sites': len(state.directions),
        'classical_energy': e_cl, 'gain_over_old_classical_meV_per_site': point['old_classical_energy'] - e_cl,
        'orbit_flat': flat, 'stable_angles': len(stable), 'tested_angles': len(candidates),
        'lswt_status': 'stable' if stable else 'unstable', 'harmonic_energy': d['harmonic_energy'],
        'skyrmion_Q': d['skyrmion']['integer'], 'mz_per_spin': float(0.5 * n[:, 2].mean()),
        'fourier': fourier(model, pick.state), 'directions': n.round(5).tolist(),
        'seconds': time.monotonic() - t0})
    return out


def main():
    start = time.monotonic()
    points = hatched_points()
    points += [{'axis': 'PD', 'J': 0.010, 'h': round(Y_FIELD, 6), 'kind': 'reference Y (0.2 T)',
                'old_classical_winner': {'label': 'Y', 'cell': 'three_msl'}, 'old_classical_energy': None},
               {'axis': 'PD', 'J': 0.010, 'h': round(V_FIELD, 6), 'kind': 'reference V (1.4 T)',
                'old_classical_winner': {'label': 'V', 'cell': 'three_msl'}, 'old_classical_energy': None}]
    for p in points:
        if p['old_classical_energy'] is None:             # reference points: the 3-site energy is the baseline
            model = build_model({'Jxy': 0.075, 'Jz': 0.125, 'JPD': p['J']})
            cond = ExternalConditions(field=(0, 0, p['h']))
            rng = np.random.default_rng(0)
            p['old_classical_energy'] = min(float(classical_energy(model, refine_classical(model, random_state(model, CELLS['3'], rng), cond), cond))
                                            for _ in range(STARTS))
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        results = list(pool.map(search_point, points))
    for r in results:
        print('%-5s J=%.4f h=%.3f %-20s winner %2s-site  gain %.2e  LSWT %-8s (%d/%d angles)  Q=%s mz=%.3f  in-plane q %s' % (
            r['axis'], r['J'], r['h'], r['kind'], r['winner_sites'], r['gain_over_old_classical_meV_per_site'],
            r['lswt_status'], r['stable_angles'], r['tested_angles'], r['skyrmion_Q'], r['mz_per_spin'],
            r['fourier']['in_plane'][0][0]), flush=True)
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'NBCP nearest-neighbour model, J = 0.075, J_z = 0.125 meV, S = 1/2, field along z as Zeeman energy h '
                 '(meV); classical minima on cells up to 36 sites and LSWT stability of the lowest; no quantum '
                 'corrections beyond harmonic order.',
        'cells': CELLS, 'starts_per_cell': STARTS, 'k_density': K_DENSITY, 'results': results,
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__)] + sorted(SCANS.glob('scan-*.json'))},
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'hatched-region-check.json').write_text(json.dumps(record, indent=1))
    for r in results:
        assert r['gain_over_old_classical_meV_per_site'] > -1e-9        # larger cells never raise the minimum


if __name__ == '__main__':
    main()
