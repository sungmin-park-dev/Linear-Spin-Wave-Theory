"""Does a four-site skyrmion crystal win anywhere in the NBCP model? (D42)

The target paper (arXiv:2601.20963) reports a four-site SkX phase; the NBCP
note (chapter 5) keeps it as a candidate because the draft's solid angle used
an absolute value. This script scans the model of the note
(J = 0.075 meV, J_z = 0.125 meV, field along c given as the Zeeman energy
h = g_z mu_B B) over J_PD, J_Gamma and h, and at every point

1. finds the classical minimum on the one-, two-, three- and four-site
   magnetic cells (differential evolution, continued from the state of the
   previous field, lower of the two kept);
2. computes the signed Berg-Luescher skyrmion number of each minimum
   (``observables.texture``);
3. compares the cells at harmonic order (``methods.phase_competition``):
   classical energy, LSWT stability on a matched mesh, and E_cl + E_zp.

The four-site cell contains the one- and two-site cells, so a four-site
minimum that only reproduces them ties with them; a four-site state counts as
a winner only if it is lower than every other cell by more than ``MARGIN``.
It is a skyrmion crystal if its integer skyrmion number is nonzero. A winner
among these cells is not a global ground state (larger cells and
incommensurate states are not included).

Usage
-----
    python examples/nbcp_skyrmion_competition.py            # scan and summary
    python examples/nbcp_skyrmion_competition.py --summarize  # summary of the saved scan
"""

import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from model.nbcp.model import SUPERCELLS, build_model
from spintoolkit.methods.classical import classical_energy, classical_search, refine_classical
from spintoolkit.methods.phase_competition import compare_states
from spintoolkit.system.conditions import ExternalConditions

OUT = ROOT / 'data-space/verification/261001-nbcp-skyrmion-competition'
BASE = {'Jxy': 0.075, 'Jz': 0.125}
JPD = (0.0, 0.005, 0.010, 0.020, 0.030)
JGAMMA = (0.0, 0.005, 0.010, 0.020)
FIELDS = tuple(np.round(np.linspace(0.0, 0.55, 12), 4))
CELLS = ('one_msl', 'two_msl', 'three_msl', 'four_msl')
K_DENSITY = 18
MARGIN = 1e-7                  # meV per spin


def lowest(model, cell, conditions, previous):
    best = classical_search(model, SUPERCELLS[cell], conditions).state
    if previous is not None:
        continued = refine_classical(model, previous, conditions)
        if classical_energy(model, continued, conditions) < classical_energy(model, best, conditions):
            best = continued
    return best


def main():
    points = []
    for jpd in JPD:
        for jg in JGAMMA:
            model = build_model({**BASE, 'JPD': jpd, 'JGamma': jg})
            previous = {cell: None for cell in CELLS}
            for h in FIELDS:
                conditions = ExternalConditions(field=(0.0, 0.0, float(h)))
                states = {}
                for cell in CELLS:
                    states[cell] = lowest(model, cell, conditions, previous[cell])
                    previous[cell] = states[cell]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    reports = compare_states(model, states, conditions, k_density=K_DENSITY,
                                             refine=False)
                by_name = {r.name: r for r in reports}
                four = by_name['four_msl']
                others = min(by_name[c].classical_energy for c in CELLS if c != 'four_msl')
                stable = [r for r in reports if r.status == 'stable']
                points.append({
                    'JPD': jpd, 'JGamma': jg, 'h_meV': float(h),
                    'classical_winner': min(reports, key=lambda r: r.classical_energy).name,
                    'harmonic_winner': stable[0].name if stable else None,
                    'four_site_Q': four.skyrmion.to_dict()['charge'],
                    'four_site_minus_best_other_classical_meV': four.classical_energy - others,
                    'candidates': [r.to_dict() for r in reports]})
                print(json.dumps({k: points[-1][k] for k in list(points[-1])[:7]}), file=sys.stderr)
    OUT.mkdir(parents=True, exist_ok=True)
    write(points)


def four_site(point):
    return next(c for c in point['candidates'] if c['name'] == 'four_msl')


def harmonic_margin(point):
    """E_cl + E_zp of the four-site state minus the lowest other stable cell (None if undefined)."""
    four = four_site(point)['harmonic_energy']
    others = [c['harmonic_energy'] for c in point['candidates']
              if c['name'] != 'four_msl' and c['harmonic_energy'] is not None]
    return None if four is None or not others else four - min(others)


def summarize(points):
    rows = []
    for p in points:
        Q = four_site(p)['skyrmion']['integer']
        rows.append({'JPD': p['JPD'], 'JGamma': p['JGamma'], 'h_meV': p['h_meV'], 'Q': Q,
                     'classical_margin_meV': p['four_site_minus_best_other_classical_meV'],
                     'harmonic_margin_meV': harmonic_margin(p),
                     'four_site_status': four_site(p)['status'],
                     'harmonic_winner': p['harmonic_winner'],
                     'stable_cells': [c['name'] for c in p['candidates'] if c['status'] == 'stable']})
    skx = [r for r in rows if r['Q'] not in (None, 0)]
    classical = [r for r in skx if r['classical_margin_meV'] < -MARGIN]
    # A state with Q != 0 cannot tie with the smaller cells, so the lowest stable
    # candidate is a strict winner; it may also be the only stable one.
    harmonic = [r for r in skx if r['harmonic_winner'] == 'four_msl']
    nontopological = [r for r in rows if r['Q'] == 0 and r['classical_margin_meV'] < -MARGIN]
    by_jpd = {}
    for r in skx:
        key = str(r['JPD'])
        by_jpd[key] = min(by_jpd.get(key, np.inf), r['classical_margin_meV'])
    return {'grid': {'JPD': JPD, 'JGamma': JGAMMA, 'h_meV': list(map(float, FIELDS))},
            'base': BASE, 'k_density': K_DENSITY, 'margin_meV': MARGIN,
            'four_site_Q_values': sorted({r['Q'] for r in rows if r['Q'] is not None}),
            'skx_points': len(skx),
            'skx_classical_winners': classical, 'skx_harmonic_winners': harmonic,
            'nontopological_four_site_winners': len(nontopological),
            'lowest_skx_classical_margin_by_JPD_meV': by_jpd}


def write(points):
    summary = summarize(points)
    (OUT / 'report.json').write_text(json.dumps({'summary': summary, 'points': points},
                                                indent=1) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    if '--summarize' in sys.argv:
        write(json.loads((OUT / 'report.json').read_text())['points'])
    else:
        main()
