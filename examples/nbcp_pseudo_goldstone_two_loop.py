"""Next-order (two-loop) pseudo-Goldstone gap of the NBCP Y and V states (D47).

``pseudo_goldstone_gap`` about the classical Y (0.2 T) and V (1.4 T)
three-sublattice states with J_PD = 0.010 meV at the orientation phi = 0
selected by the zero-point energy. The gap is expanded as
Delta^2 = A + B (A: leading order, the C_phi / chi_z relation; B: next order
in 1/S, evaluated at S = 1/2). Recorded:

1. mesh convergence (N = 12, 24, 36; point-group-averaged meshes);
2. the J_PD dependence of the ratio of the two-loop to the one-loop
   curvature along the rotation (both start at J_PD^3);
3. an infrared diagnostic: the curvatures with a local pinning field that
   gaps the soft branch by about the physical gap.

Run from the repository root:
    python examples/nbcp_pseudo_goldstone_two_loop.py
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from examples.nbcp_nonlinear_spin_waves import SCAN, stationary
from model import nbcp
from spintoolkit.methods.nlswt import PseudoGoldstoneSettings, pseudo_goldstone_gap
from spintoolkit.system.conditions import ExternalConditions

OUT = ROOT / 'data-space/verification/261003-nbcp-two-loop-gap'
S, J, JZ = 0.5, 0.075, 0.125
MESHES = [12, 24, 36]
JPD_SCAN = [0.0025, 0.005]
PINNING = [1e-4, 3e-4]


def run(phase, jpd, n, pinning=0.0):
    scan = json.loads(SCAN.read_text())
    h = scan['states'][phase]['h_meV']
    theta = stationary(phase, h, np.array(scan['states'][phase]['theta']))
    model = nbcp.build_model({'Jxy': J, 'Jz': JZ, 'JPD': jpd})
    state = nbcp.candidate_state(model, 'three_msl', np.column_stack([theta, np.zeros(3)]).ravel())
    start = time.monotonic()
    r = pseudo_goldstone_gap(model, state, (0, 0, 1), ExternalConditions(field=[0, 0, h]),
                             PseudoGoldstoneSettings(mesh=(n, n), pinning=pinning))
    row = {'phase': phase, 'B_T': scan['states'][phase]['B_T'], 'h_meV': h, 'JPD_meV': jpd,
           'mesh': n, 'pinning_meV': pinning,
           'A_meV2': r.gap_squared[0], 'B_meV2': r.gap_squared[1],
           'leading_gap_meV': r.leading_gap if pinning == 0 else None,
           'B_over_A': r.relative_correction if pinning == 0 else None,
           'curvature': {k: float(v) for k, v in r.curvature.items()},
           'two_loop_over_one_loop_curvature': float(r.curvature['order_s0'] / r.curvature['zero_point']),
           'components': {k: float(v) for k, v in r.components.items()},
           'ward_identity_relative_error': float(r.header.diagnostics['ward_identity_relative_error']),
           'seconds': time.monotonic() - start}
    print(json.dumps(row), flush=True)
    return row


def main():
    start = time.monotonic()
    rows = []
    for phase in ['Y', 'V']:
        for n in MESHES:
            rows.append(run(phase, 0.010, n))
        for jpd in JPD_SCAN:
            rows.append(run(phase, jpd, 12))
        for lam in PINNING:
            rows.append(run(phase, 0.010, 24, lam))
    for phase in ['Y', 'V']:
        conv = [r['B_over_A'] for r in rows if r['phase'] == phase and r['pinning_meV'] == 0
                and r['JPD_meV'] == 0.010 and r['mesh'] in MESHES[-2:]]
        assert abs(conv[1] - conv[0]) < 0.05 * abs(conv[1]), (phase, conv)
        assert all(r['ward_identity_relative_error'] < 1e-4 for r in rows if r['phase'] == phase)
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'NBCP Y (0.2 T) and V (1.4 T), S = 1/2, J = 0.075, J_z = 0.125 meV, J_Gamma = 0, '
                 'phi = 0, rotation about z. Delta^2 = A + B: A leading order (C_phi / chi_z), '
                 'B next order in 1/S at S = 1/2 (two loops). Curvatures per magnetic cell; '
                 'zero_point and order_s0 are d^2/dphi^2 of E_zp and of the constrained '
                 'order-S^0 energy (meV/rad^2).',
        'rows': rows,
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'nbcp-two-loop-gap.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
