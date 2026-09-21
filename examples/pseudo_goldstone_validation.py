"""Check representative Y/V curvatures by independent local finite differences."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pseudo_goldstone_comparison import ROOT, build, mesh, spin_angles, vacuum_energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'data-space/verification/260912-pseudo-goldstone')
    args = parser.parse_args()
    reference = json.loads((args.directory/'scan-N24-P72.json').read_text())
    result = {'description': 'Current Hamiltonian, T=0, local five-point curvature, no shifts. '
                             'The finest derivative step can be less accurate for tiny angular signals.',
              'measurements': []}
    for row in reference['scans']:
        if max(row['JPD_meV'], row['JGamma_meV']) != .01:
            continue
        state = reference['states'][row['phase']]
        _, ham, _ = build(state, row['JPD_meV'], row['JGamma_meV'])
        phi = row['current']['phi_min']
        for n in [24, 48, 96]:
            points = mesh(n)
            cache = {}
            for offset in [0., -.16, -.08, -.04, -.02, -.01, .01, .02, .04, .08, .16]:
                matrices, _ = ham.Quadratic_Bose_Hamiltonian(points, angles=spin_angles(state['theta'], phi+offset))
                cache[offset] = vacuum_energy(matrices)[0]
            for step in [.08, .04, .02, .01]:
                energies = np.array([cache[i*step] for i in [-2, -1, 0, 1, 2]])
                c = float(np.dot(energies-energies[2], [-1, 16, -30, 16, -1])/(12*step*step))
                assert c > 0
                result['measurements'].append({'phase': state['phase'], 'JPD_meV': row['JPD_meV'],
                    'JGamma_meV': row['JGamma_meV'], 'N': n, 'step_rad': step,
                    'curvature_meV_per_spin': c,
                    'gap_meV': float(np.sqrt(c/state['chi_per_spin_per_meV']))})
            print(row['phase'], row['JPD_meV'], row['JGamma_meV'], n, flush=True)
    # Compare two practical steps, and mesh 48 -> 96, at the same step.
    comparisons = []
    for phase in ['Y', 'V']:
        for axis in ['PD', 'Gamma']:
            def get(n, step):
                return next(v for v in result['measurements'] if v['phase'] == phase and
                            v[f'J{axis}_meV'] == .01 and v['N'] == n and v['step_rad'] == step)
            mesh_change = get(96, .04)['gap_meV']/get(48, .04)['gap_meV']-1
            step_change = get(96, .02)['gap_meV']/get(96, .04)['gap_meV']-1
            assert abs(mesh_change) < .005
            assert abs(step_change) < .005
            comparisons.append({'phase': phase, 'axis': axis, 'mesh_48_to_96_relative_gap_change': mesh_change,
                                'step_004_to_002_relative_gap_change': step_change})
    result['comparisons'] = comparisons
    result['source_sha256'] = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                              for name in ['examples/pseudo_goldstone_validation.py',
                                           'examples/pseudo_goldstone_comparison.py',
                                           'code-space/lswt/solvers/hamiltonian.py']}
    (args.directory/'curvature-validation.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
