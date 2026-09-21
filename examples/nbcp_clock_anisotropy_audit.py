"""Audit PD/Gamma clock harmonics using saved T=0 Y/V energy scans.

This does not calculate a finite-temperature phase diagram. It evaluates
unconstrained Fourier coefficients and checks exact pi-rotation identities
using Cartesian HP matrices at fresh, non-symmetrized momentum points.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lswt-mpl-cache')

from examples.pseudo_goldstone_comparison import (
    RECIPROCAL, build, direct_hp, spin_angles, vacuum_energy,
)


def summarize(record):
    """Extract harmonics without enforcing either threefold or sixfold order."""
    angles = np.asarray(record['phi'])
    energy = np.asarray(record['current']['Ezp_meV_per_spin'])
    if not np.all(np.isfinite(energy)):
        raise ValueError('The selected angular scan contains unresolved energies.')
    count = len(energy)
    if count % 6 or not np.allclose(angles, 2*np.pi*np.arange(count)/count):
        raise ValueError('Expected a uniform angular grid without a duplicate endpoint.')
    centered = energy-energy.mean()
    harmonics = np.arange(1, count//2)
    complex_coefficients = (2/count) * (
        np.exp(-1j*harmonics[:, None]*angles[None, :]) @ centered
    )
    amplitudes = abs(complex_coefficients)
    return {
        'phase': record['phase'], 'N': record['N'], 'n_phi': count,
        'JPD_meV': record['JPD_meV'], 'JGamma_meV': record['JGamma_meV'],
        'input_curvature_status': record['current']['status'],
        'energy_range_meV_per_spin': float(np.ptp(energy)),
        'harmonics': harmonics.tolist(),
        'cos_coefficients_meV_per_spin': complex_coefficients.real.tolist(),
        'sin_coefficients_meV_per_spin': (-complex_coefficients.imag).tolist(),
        'amplitudes_meV_per_spin': amplitudes.tolist(),
        'A3_meV_per_spin': float(amplitudes[2]),
        'A6_meV_per_spin': float(amplitudes[5]),
        'dominant_sampled_harmonic': int(harmonics[np.argmax(amplitudes)]),
        'max_pi_shift_residual_meV_per_spin': float(np.max(abs(energy-np.roll(energy, count//2)))),
        'max_pi_over_3_shift_residual_meV_per_spin': float(np.max(abs(energy-np.roll(energy, count//6)))),
        'max_2pi_over_3_shift_residual_meV_per_spin': float(np.max(abs(energy-np.roll(energy, count//3)))),
    }


def pi_rotation_checks(states):
    """Check exact identities at arbitrary angles and fresh k points.

    The Hamiltonian builder and bond geometry are shared with the comparison;
    the Cartesian HP assembly does not use its production local-frame transform.
    No sixfold momentum averaging or angular Fourier fit is used here.
    """
    points = np.random.default_rng(260916).uniform(-.5, .5, (128, 2)) @ RECIPROCAL
    rotation = np.diag([-1., -1., 1.])
    result = []
    for phase, state in states.items():
        for axis in ['PD', 'Gamma']:
            pd, gamma = (.01, 0.) if axis == 'PD' else (0., .01)
            data, _, _ = build(state, pd, gamma)
            flipped, _, _ = build(state, pd, -gamma)
            bonds = [b['Exchange Matrix'] for b in data['Couplings']]
            opposite = [b['Exchange Matrix'] for b in flipped['Couplings']]
            phi = .173
            h0 = direct_hp(data, points, spin_angles(state['theta'], phi))
            hpi = direct_hp(data, points, spin_angles(state['theta'], phi+np.pi))
            hm = direct_hp(flipped, points, spin_angles(state['theta'], phi))
            covariant_error = float(np.max(abs(hpi-hm)))
            # A matrix-identity check; this tolerance is not a band-gap cutoff
            # or a bound on the physical approximation.
            if not np.allclose(hpi, hm, rtol=2e-13, atol=2e-14):
                raise AssertionError('Pi rotation failed the Gamma-sign covariance identity.')
            result.append({
                'phase': phase, 'axis': axis, 'phi_rad': phi, 'n_k': len(points),
                'bond_pi_invariance_error_meV': float(max(np.max(abs(rotation @ j @ rotation-j)) for j in bonds)),
                'bond_gamma_sign_covariance_error_meV': float(max(np.max(abs(rotation @ j @ rotation-jm)) for j, jm in zip(bonds, opposite))),
                'HP_pi_invariance_error_meV': float(np.max(abs(hpi-h0))),
                'HP_gamma_sign_covariance_error_meV': covariant_error,
                'unaveraged_energy_pi_difference_meV_per_spin': float(vacuum_energy(hpi)[0]-vacuum_energy(h0)[0]),
            })
    return result


def plot_v_comparison(records, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), layout='constrained')
    for ax, axis, color in zip(axes, ['PD', 'Gamma'], ['#126b78', '#ae583d']):
        record = next(r for r in records if r['phase'] == 'V' and r[f'J{axis}_meV'] == .01
                      and r[f"J{'Gamma' if axis == 'PD' else 'PD'}_meV"] == 0)
        angle = np.degrees(record['phi'])
        energy = np.asarray(record['current']['Ezp_meV_per_spin'])
        delta = (energy-energy.min())*1e6  # 1 meV = 10^6 neV
        ax.plot(np.r_[angle, 360], np.r_[delta, delta[0]], color=color, lw=1.8)
        ax.set(xlabel=r'Common spin rotation $\phi$ (degrees)',
               ylabel=r'$e_{\rm sw}(\phi)-\min e_{\rm sw}$ (neV/spin)',
               xlim=(0, 360), ylim=(0, float(delta.max())*1.08))
        ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
        ax.set_title(r'$J_{\rm PD}=0.010$, $J_\Gamma=0$ meV: sixfold' if axis == 'PD'
                     else r'$J_{\rm PD}=0$, $J_\Gamma=0.010$ meV: threefold', fontsize=10)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(r'V state at $B=1.4$ T: the pure-PD axis retains an extra symmetry', fontsize=12)
    for suffix in ['png', 'svg']:
        fig.savefig(output/f'v-clock-anisotropy.{suffix}', dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data-space/verification/260916-clock-anisotropy')
    args = parser.parse_args()
    source_dir = ROOT/'data-space/verification/260912-pseudo-goldstone'
    summaries, inputs = [], {}
    for n in [12, 24, 48]:
        source = source_dir/f'scan-N{n}-P72.json'
        report = json.loads(source.read_text())
        inputs[source.relative_to(ROOT).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
        for record in report['scans']:
            if (record['JPD_meV'], record['JGamma_meV']) in [(.01, 0.), (0., .01)]:
                summaries.append(summarize(record))
    checks = pi_rotation_checks(report['states'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'T=0 clock-harmonic diagnostic and exact pi-rotation checks; no thermal phase calculation',
        'input_sha256': inputs,
        'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__).resolve(), ROOT/'examples/pseudo_goldstone_comparison.py',
                                 ROOT/'examples/nbcp_ground_state.py', ROOT/'code-space/lswt/core/exchange.py',
                                 ROOT/'code-space/lswt/core/spin_system.py']},
        'parameters': {key: report[key] for key in ['S', 'J_meV', 'Jz_meV', 'temperature_K', 'g_z']},
        'states': report['states'],
        'fourier_convention': 'E(phi)-mean = sum_n [a_n cos(n phi)+b_n sin(n phi)]; A_n=hypot(a_n,b_n)',
        'summaries': summaries,
        'pi_rotation_checks': checks,
        'limitations': ['Representative SOC values only; finite angular sampling can alias higher harmonics.',
                        'The saved scans use rotationally symmetrized momentum quadrature.',
                        'Fresh HP checks use unsymmetrized k points, but share bond geometry and state inputs.',
                        'No finite-temperature stiffness, vortex fugacity, domain walls or BKT transition are computed.'],
    }
    (args.output_dir/'clock-anisotropy-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    plot_v_comparison(report['scans'], args.output_dir)
    for row in summaries:
        if row['N'] == 48:
            axis = 'PD' if row['JPD_meV'] else 'Gamma'
            print(f"{row['phase']} {axis}: A3={row['A3_meV_per_spin']:.8g}, A6={row['A6_meV_per_spin']:.8g} meV/spin")
    print(args.output_dir)


if __name__ == '__main__':
    main()
