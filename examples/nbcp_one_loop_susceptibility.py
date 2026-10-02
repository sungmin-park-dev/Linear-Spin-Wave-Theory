"""Size of the next order in 1/S for the NBCP Y and V gap: one-loop chi_z.

The leading pseudo-Goldstone gap is Delta^2 = C_phi / chi_z with C_phi the
zero-point curvature along the global-z orbit and chi_z the classical relaxed
susceptibility (NBCP note, Appendix A). The next order needs C_phi at two
loops, which requires interacting spin-wave vertices not implemented here,
and chi_z at one loop, which this script evaluates:

    m(h) = -d(E_cl + E_zp)/dh,   chi(h) = -d^2(E_cl + E_zp)/dh^2,

with E_zp computed by LSWT on the classical stationary state at each field
(Y at 0.2 T, V at 1.4 T, J_PD = 0.010 meV, phi = 0). The ratio of the
one-loop to the classical chi is an indicator of how well the 1/S series for
the gap is controlled; it is not a 1/S^2 gap.

Run from the repository root:
    python examples/nbcp_one_loop_susceptibility.py
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
from scipy.optimize import root

from examples.pseudo_goldstone_comparison import (GZ, J, JZ, MU_B, S, build,
                                                  classical_derivatives, mesh,
                                                  spin_angles, vacuum_energy)

OUT = ROOT / 'data-space/verification/261002-one-loop-susceptibility'
JPD = 0.010
STEPS = [4e-3, 2e-3, 1e-3]
MESHES = [24, 36]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stationary(phase, h, guess):
    if phase == 'Y':
        t = np.arccos((h + 3 * S * JZ) / (3 * S * (J + JZ)))
        return np.array([t, -t, np.pi])
    theta = root(lambda t: classical_derivatives(t, h)[1], guess, tol=1e-13).x
    assert np.max(abs(classical_derivatives(theta, h)[1])) < 1e-12
    return theta


def energies(phase, h, guess, points):
    theta = stationary(phase, h, guess)
    data, ham, _ = build({'theta': theta, 'h_meV': h}, JPD, 0.0)
    matrices, _ = ham.Quadratic_Bose_Hamiltonian(points, angles=spin_angles(theta, 0.))
    return classical_derivatives(theta, h)[0] / 3, vacuum_energy(matrices)[0]


def main():
    start = time.monotonic()
    rows = []
    for phase, field in [('Y', 0.2), ('V', 1.4)]:
        h = GZ * MU_B * field
        guess = stationary(phase, h, np.array([.4, .4, -1.2])) if phase == 'V' else None
        for n in MESHES:
            points = mesh(n)
            for dh in STEPS:
                e = np.array([energies(phase, h + s * dh, guess, points) for s in (-1, 0, 1)])
                chi = -(e[2] - 2 * e[1] + e[0]) / dh**2
                m = -(e[2] - e[0]) / (2 * dh)
                rows.append({'phase': phase, 'B_T': field, 'mesh': n, 'dh_meV': dh,
                             'chi_classical': float(chi[0]), 'chi_one_loop': float(chi[1]),
                             'ratio': float(chi[1] / chi[0]),
                             'm_classical': float(m[0]), 'm_one_loop': float(m[1])})
                print('%s mesh %d dh %.0e: chi %.4f + %.4f (ratio %.3f), m %.4f %+.4f' % (
                    phase, n, dh, chi[0], chi[1], chi[1] / chi[0], m[0], m[1]))
    for phase in ['Y', 'V']:
        ratios = [r['ratio'] for r in rows if r['phase'] == phase]
        assert np.ptp(ratios) < 0.01, (phase, ratios)

    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'One-loop correction to chi_z = -d^2 E/dh^2 per spin (meV^-1) and to m_z per '
                 'spin, E_zp on the classical stationary state; S = 1/2, J = 0.075 meV, '
                 'J_z = 0.125 meV, J_PD = 0.010 meV, J_Gamma = 0, phi = 0, g_z = 4.645.',
        'rows': rows,
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__), ROOT / 'examples/pseudo_goldstone_comparison.py']},
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'one-loop-susceptibility.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
