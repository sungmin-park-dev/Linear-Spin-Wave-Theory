"""Check that the phase theory accounts for the cutoff dependence of the Y pinning.

`nbcp_angular_thermal_split.py` removes the soft modes with |k| < Lambda from
the harmonic thermal free energy and finds that the sixfold amplitude a6 of
the Y state depends on Lambda. This script tests whether a phase theory with
the relaxed, angle-dependent stiffness rho(phi) and susceptibility chi(phi)
regenerates exactly that part. The phase theory gives the soft frequency

    eps(k, phi) = k sqrt(khat . rho(phi) . khat / chi(phi)),

so (i) the long-wavelength sixfold coefficient of <ln eps_soft> must equal
that of <(1/2) ln(khat . rho . khat / chi)>, and (ii) the Bose free energy of
these frequencies below Lambda must reproduce the LSWT change of a6. Both use
the relaxed reduction of `nbcp_y_soc_conditions.py` (Y, 0.2 T,
J_PD = 0.010 meV). The check is harmonic; it does not include the
Debye-Waller suppression from phase fluctuations or vortices.

Run from the repository root after `nbcp_angular_thermal_split.py`:
    python examples/nbcp_matching_cutoff_check.py
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT), str(ROOT / 'legacy')]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lswt-mpl-cache')

import numpy as np

from examples.nbcp_y_soc_conditions import background, reduction
from examples.pseudo_goldstone_comparison import mesh

SPLIT = ROOT / 'data-space/verification/261001-angular-thermal-split/angular-thermal-split.json'
OUT = ROOT / 'data-space/verification/261002-matching-cutoff'
JPD = 0.010
N_PHI = 72
N_MESH = 48
CUTOFFS = [0.25, 0.5]
AREA = np.sqrt(3) / 2


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cos6(phi, values):
    centered = np.asarray(values) - np.mean(values)
    return float(2 / len(phi) * np.sum(centered * np.cos(6 * phi))), \
        float(2 / len(phi) * np.sum(centered * np.sin(6 * phi)))


def main():
    start = time.monotonic()
    phi = 2 * np.pi * np.arange(N_PHI) / N_PHI
    reduced = [reduction(background(JPD, 0.0, p)) for p in phi]
    stiffness = np.array([r[4] for r in reduced])          # per magnetic cell
    chi = np.array([r[5] for r in reduced])
    directions = np.linspace(0, np.pi, 360, endpoint=False)
    khat = np.column_stack([np.cos(directions), np.sin(directions)])
    log_velocity = [0.5 * np.mean(np.log(np.einsum('ki,ij,kj->k', khat, c, khat) / x))
                    for c, x in zip(stiffness, chi)]
    b6, b6_sin = cos6(phi, log_velocity)
    sqrt_det = np.sqrt(np.linalg.det(stiffness / (3 * AREA)))

    split = json.loads(SPLIT.read_text())
    y = next(r for r in split['results'] if r['phase'] == 'Y')
    shells = [s for s in y['shells'] if s['k_max'] <= 0.25]
    # b(k) = b0 - a k^2 through the rms momenta of the two innermost shells.
    k_rms = [np.sqrt((s['k_max']**4 - s['k_min']**4) / (2 * (s['k_max']**2 - s['k_min']**2)))
             for s in shells]
    (b0, slope) = np.linalg.solve(np.column_stack([np.ones(2), -np.square(k_rms)]),
                                  [s['b6_ln_eps_soft'] for s in shells])

    points = mesh(N_MESH)
    kabs = np.linalg.norm(points, axis=1)
    rows = []
    for row in y['temperatures']:
        T = row['T_meV']
        if T == 0:
            continue
        entry = {'T_meV': T, 'a6_ratio_cutoff_0': row['a6_cutoff_0.0'] / y['temperatures'][0]['a6_cutoff_0.0']}
        for cutoff in CUTOFFS:
            k = points[kabs < cutoff]
            free = []
            for c, x in zip(stiffness, chi):
                eps = np.sqrt(np.einsum('ki,ij,kj->k', k, c, k) / x)
                free.append(T * np.sum(np.log1p(-np.exp(-eps / T))) / len(kabs) / 3)
            entry[f'delta_a6_phase_theory_{cutoff}'] = cos6(phi, free)[0]
            entry[f'delta_a6_lswt_{cutoff}'] = row[f'a6_cutoff_{cutoff}'] - row['a6_cutoff_0.0']
        rows.append(entry)
        print('T=%.4f ratio(cutoff 0)=%.3f ' % (T, entry['a6_ratio_cutoff_0']) + ' '.join(
            'L%.2f: theory %.3e LSWT %.3e' % (c, entry[f'delta_a6_phase_theory_{c}'],
                                              entry[f'delta_a6_lswt_{c}']) for c in CUTOFFS))
    print('chi span %.1e; sqrt det rho mean %.5f meV, cos6 %.2f%%; b6 theory %.6f, LSWT k->0 %.6f'
          % (np.ptp(chi), sqrt_det.mean(), 100 * cos6(phi, sqrt_det)[0] / sqrt_det.mean(), b6, b0))
    assert np.ptp(chi) < 1e-12 and abs(b6_sin) < 1e-12
    assert abs(b6 / b0 - 1) < 0.01
    assert all(abs(r['delta_a6_phase_theory_0.25'] / r['delta_a6_lswt_0.25'] - 1) < 0.1 for r in rows)

    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Y (0.2 T), J_PD = 0.010 meV, J_Gamma = 0; phase theory with relaxed rho(phi), '
                 'chi(phi) against the harmonic LSWT cutoff split. T in meV; momenta in inverse '
                 'bond lengths; a6 per spin with f = f0 - a6 cos(6 phi).',
        'chi_span_per_cell': float(np.ptp(chi)),
        'sqrt_det_rho_meV': {'mean': float(sqrt_det.mean()),
                             'cos6_relative': cos6(phi, sqrt_det)[0] / float(sqrt_det.mean())},
        'b6_phase_theory': b6, 'b6_lswt_k_to_0': float(b0), 'b6_lswt_k2_slope': float(slope),
        'temperatures': rows,
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__), SPLIT, ROOT / 'examples/nbcp_y_soc_conditions.py']},
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'matching-cutoff-check.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
