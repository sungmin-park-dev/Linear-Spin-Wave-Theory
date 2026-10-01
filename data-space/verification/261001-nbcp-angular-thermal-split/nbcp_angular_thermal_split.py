"""Split the harmonic thermal sixfold pinning of NBCP Y and V by momentum.

Section 7 of the NBCP note requires the matching cutoff between the
eliminated fast modes and the retained phase theory to be recorded. This
diagnostic evaluates the harmonic (1/S) angular free energy

    f(phi, T) = E_zp(phi) + (k_B T / 3) <sum_bands ln[1 - exp(-eps/k_B T)]>_k

per spin on the fixed classical Y (0.2 T) and V (1.4 T) orbits at
J_PD = 0.010 meV, J_Gamma = 0. It removes the modes with |k| < Lambda from
the thermal sum for several Lambda and reports the sixfold amplitude. It
also reports, shell by shell, the sixfold harmonic of <ln eps_soft>, the
angular anisotropy of the soft-branch velocity. A k-independent value
marks the long-wavelength regime that a phase theory with an
angle-dependent stiffness tensor can represent. This is a harmonic
calculation, not a thermal phase diagram or an RG trajectory.

Run from the repository root:
    python examples/nbcp_angular_thermal_split.py
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

from examples.pseudo_goldstone_comparison import (METRIC, build, mesh, phase_state,
                                                  spin_angles)

OUT = ROOT / 'data-space/verification/261001-angular-thermal-split'
N_MESH = 48
N_PHI = 36
JPD = 0.010
TEMPERATURES = [0.0, 0.0025, 0.005, 0.01, 0.015, 0.02]
CUTOFFS = [0.0, 0.25, 0.5, 1.0]
SHELLS = [(0.0, 0.15), (0.15, 0.25), (0.25, 0.4), (0.4, 0.6), (0.6, 0.8),
          (0.8, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 3.5)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cos6(phi, values):
    """Coefficient c of c*cos(6 phi) and the residual sin(6 phi) part."""
    centered = np.asarray(values) - np.mean(values)
    c = 2 / len(phi) * np.exp(-1j * 6 * np.outer(1, phi))[0] @ centered
    return float(c.real), float(c.imag)


def bands(state, points, phi):
    data, current, _ = build(state, JPD, 0.0)
    eps, ezp = [], []
    for p in phi:
        mats, torque = current.Quadratic_Bose_Hamiltonian(
            points, angles=spin_angles(state['theta'], p))
        assert max(abs(v) for v in torque.values()) < 2e-12
        chol = np.linalg.cholesky(mats)
        eig = np.linalg.eigvalsh(chol.conj().transpose(0, 2, 1) @ METRIC @ chol)
        assert np.all(eig[:, :3] < 0) and np.all(eig[:, 3:] > 0)
        trace = np.trace(mats, axis1=1, axis2=2).real
        eps.append(eig[:, 3:])
        ezp.append(float(np.mean(eig[:, 3:].sum(axis=1) / 2 - trace / 4) / 3))
    return np.array(eps), np.array(ezp)


def analyse(phase, points, phi):
    state = phase_state(phase)
    eps, ezp = bands(state, points, phi)
    kabs = np.linalg.norm(points, axis=1)
    rows = []
    for T in TEMPERATURES:
        per_mode = (T * np.log1p(-np.exp(-eps / T)) if T > 0 else np.zeros_like(eps))
        row = {'T_meV': T}
        for cutoff in CUTOFFS:
            keep = kabs >= cutoff
            f = ezp + per_mode[:, keep].sum(axis=-1).sum(axis=1) / len(kabs) / 3
            row[f'a6_cutoff_{cutoff}'] = -cos6(phi, f)[0]
            row[f'sin6_residual_cutoff_{cutoff}'] = cos6(phi, f)[1]
        soft = per_mode[..., 0].mean(axis=1) / 3
        row['a6_thermal_soft_branch_all_k'] = -cos6(phi, soft)[0]
        row['a6_thermal_all_bands_all_k'] = -cos6(phi, per_mode.sum(-1).mean(1) / 3)[0]
        rows.append(row)
    shells = []
    for lo, hi in SHELLS:
        m = (kabs >= lo) & (kabs < hi)
        shells.append({'k_min': lo, 'k_max': hi, 'modes': int(m.sum()),
                       'b6_ln_eps_soft': cos6(phi, np.log(eps[:, m, 0]).mean(1))[0],
                       'mean_eps_soft_meV': float(eps[:, m, 0].mean())})
    return {'phase': phase, 'B_T': state['B_T'], 'theta': state['theta'],
            'Ezp_meV_per_spin': ezp.tolist(), 'temperatures': rows, 'shells': shells,
            'mode_fraction_below_cutoff': {str(c): float(np.mean(kabs < c)) for c in CUTOFFS}}


def main():
    start = time.monotonic()
    points = mesh(N_MESH)
    phi = np.arange(N_PHI) * 2 * np.pi / N_PHI
    results = [analyse(phase, points, phi) for phase in ('Y', 'V')]
    for r in results:
        print(r['phase'])
        for row in r['temperatures']:
            print('  T=%.4f ' % row['T_meV'] + ' '.join(
                'L%.2f:%+.3e' % (c, row[f'a6_cutoff_{c}']) for c in CUTOFFS)
                + ' soft:%+.3e' % row['a6_thermal_soft_branch_all_k'])
        for s in r['shells']:
            print('  shell %.2f-%.2f n=%5d b6=%+.3e <eps>=%.4f' % (
                s['k_min'], s['k_max'], s['modes'], s['b6_ln_eps_soft'],
                s['mean_eps_soft_meV']))
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Harmonic (1/S) angular free energy on the fixed classical Y (0.2 T) and '
                 'V (1.4 T) orbits; S=1/2, J=0.075 meV, Jz=0.125 meV, J_PD=0.010 meV, '
                 'J_Gamma=0, g_z=4.645. T in meV. a6 > 0 means minima at phi = 0 mod pi/3 '
                 '(the zero-point minima); b6 is the cos(6 phi) coefficient of the shell '
                 'average of ln eps_soft. Momenta in inverse units of the bond length.',
        'convention': 'f(phi) = f0 - a6 cos(6 phi) + ...; modes with |k| < cutoff are '
                      'removed from the thermal sum only; the zero-point term keeps all modes.',
        'quadrature': f'{N_MESH}x{N_MESH} midpoints with six rotated copies',
        'results': results,
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__), ROOT / 'examples/pseudo_goldstone_comparison.py']},
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'angular-thermal-split.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
