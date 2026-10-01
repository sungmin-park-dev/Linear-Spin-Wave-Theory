"""Test the SOC order of the V angular harmonics near the pure-PD axis.

The spin-rotation selection rule of the NBCP note (Section 6.3) predicts that
the threefold V harmonic is odd in J_Gamma and starts at J_PD * J_Gamma,
while on the pure-Gamma axis it starts at J_Gamma**3. This diagnostic
evaluates the leading T=0 vacuum energy along the fixed classical V orbit at
B = 1.4 T for mixed couplings and extracts unconstrained Fourier harmonics.
It also reports the higher harmonics of the saved pure-axis scans. It is a
leading semiclassical calculation, not a thermal or all-orders result.

Run from the repository root:
    python examples/nbcp_v_mixed_soc.py
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

from examples.pseudo_goldstone_comparison import (build, classical_from_bonds, mesh,
                                                  phase_state, spin_angles,
                                                  vacuum_energy)

OUT = ROOT / 'data-space/verification/261001-v-mixed-soc'
SAVED = ROOT / 'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json'
N_MESH = 24
N_PHI = 72
PD_VALUES = [0.0, 0.005, 0.010]
GAMMA_VALUES = [0.00025, 0.0005, 0.001, 0.002]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harmonics(phi, energy):
    """Unconstrained discrete Fourier amplitudes and phases below Nyquist."""
    centered = np.asarray(energy) - np.mean(energy)
    n = np.arange(1, len(phi) // 2)
    c = 2 / len(phi) * np.exp(-1j * np.outer(n, phi)) @ centered
    return np.abs(c), np.angle(c)


def curvature_at_minimum(phi, energy):
    """Curvature of the resolved Fourier series at its sampled-grid minimum."""
    centered = np.asarray(energy) - np.mean(energy)
    n = np.arange(1, len(phi) // 2)
    c = 2 / len(phi) * np.exp(-1j * np.outer(n, phi)) @ centered
    grid = np.linspace(0, 2 * np.pi, 20001)[:-1]
    series = np.real(np.exp(1j * np.outer(grid, n)) @ c)
    for _ in range(3):
        k = np.argmin(series)
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        grid = np.linspace(lo, hi, 2001)
        series = np.real(np.exp(1j * np.outer(grid, n)) @ c)
    x = grid[np.argmin(series)]
    return float(np.real(np.exp(1j * x * n) @ (-(n ** 2) * c))), float(x)


def scan(state, pd, gamma, points, phi):
    data, current, _ = build(state, pd, gamma)
    classical = [classical_from_bonds(data, spin_angles(state['theta'], p)) for p in phi]
    assert np.ptp(classical) < 1e-13
    energy = []
    for p in phi:
        mats, torque = current.Quadratic_Bose_Hamiltonian(
            points, angles=spin_angles(state['theta'], p))
        assert max(abs(v) for v in torque.values()) < 2e-12
        energy.append(vacuum_energy(mats)[0])
    amp, phase = harmonics(phi, energy)
    curvature, phi_min = curvature_at_minimum(phi, energy)
    # Reflection phi -> pi/3 - phi about pi/6; requires N_PHI divisible by 12.
    k0 = len(phi) // 12
    mirror = np.asarray(energy)[(2 * k0 - np.arange(len(phi))) % len(phi)]
    reflection = float(np.max(np.abs(np.asarray(energy) - mirror)))
    return {'JPD_meV': pd, 'JGamma_meV': gamma, 'N': N_MESH, 'n_phi': len(phi),
            'Ezp_meV_per_spin': energy,
            'amplitudes_meV_per_spin': amp.tolist(), 'phases_rad': phase.tolist(),
            'curvature_meV_per_spin': curvature, 'phi_min': phi_min,
            'reflection_residual_about_pi_over_6_meV_per_spin': reflection}


def saved_pure_axis():
    """Higher harmonics of the saved N48 V scans on the pure axes."""
    saved = json.loads(SAVED.read_text())
    rows = []
    for record in saved['scans']:
        cur = record['current']
        if record['phase'] != 'V' or 'cos_coefficients' not in cur:
            continue
        a, b = np.array(cur['cos_coefficients']), np.array(cur['sin_coefficients'])
        lam = np.hypot(a, b)
        rows.append({'JPD_meV': record['JPD_meV'], 'JGamma_meV': record['JGamma_meV'],
                     'lambda_n_meV_per_spin': {str(n): float(lam[n - 1]) for n in (3, 6, 9, 12, 18)},
                     'curvature_meV_per_spin': cur['curvature_meV_per_spin'],
                     'gap_meV': cur['gap_with_susceptibility_meV']})
    return rows


def main():
    start = time.monotonic()
    state = phase_state('V')
    points = mesh(N_MESH)
    phi = np.arange(N_PHI) * 2 * np.pi / N_PHI
    scans = []
    for pd in PD_VALUES:
        for gamma in GAMMA_VALUES + ([-0.001] if pd == 0.010 else []):
            record = scan(state, pd, gamma, points, phi)
            scans.append(record)
            amp = record['amplitudes_meV_per_spin']
            print(f"PD={pd:.4f} G={gamma:+.5f} l3={amp[2]:.4e} l6={amp[5]:.4e} "
                  f"phase3={record['phases_rad'][2]:+.4f}", flush=True)
    summary = []
    for pd in PD_VALUES[1:]:
        rows = [s for s in scans if s['JPD_meV'] == pd and s['JGamma_meV'] > 0]
        g = np.array([s['JGamma_meV'] for s in rows])
        l3 = np.array([s['amplitudes_meV_per_spin'][2] for s in rows])
        slope = np.polyfit(np.log(g), np.log(l3), 1)[0]
        summary.append({'JPD_meV': pd, 'loglog_slope_lambda3_vs_JGamma': float(slope),
                        'lambda3_over_JPD_JGamma': (l3 / (pd * g)).tolist()})
    pure = [s for s in scans if s['JPD_meV'] == 0.0]
    g = np.array([s['JGamma_meV'] for s in pure])
    l3 = np.array([s['amplitudes_meV_per_spin'][2] for s in pure])
    summary.append({'JPD_meV': 0.0, 'loglog_slope_lambda3_vs_JGamma':
                    float(np.polyfit(np.log(g), np.log(l3), 1)[0])})
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Leading T=0 vacuum energy on the fixed classical V orbit at B=1.4 T; '
                 'S=1/2, J=0.075 meV, Jz=0.125 meV, g_z=4.645. Unconstrained Fourier '
                 'amplitudes; no thermal or quantum-relaxation correction.',
        'state': {k: state[k] for k in ('B_T', 'h_meV', 'theta', 'chi_per_spin_per_meV')},
        'quadrature': f'{N_MESH}x{N_MESH} midpoints with six rotated copies',
        'scans': scans, 'summary': summary,
        'saved_pure_axis_N48': saved_pure_axis(),
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__), ROOT / 'examples/pseudo_goldstone_comparison.py', SAVED]},
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'v-mixed-soc-check.json').write_text(json.dumps(result, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
