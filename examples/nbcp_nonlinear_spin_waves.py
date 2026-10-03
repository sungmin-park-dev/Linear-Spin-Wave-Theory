"""Order-S^0 (two-loop) energy, magnetization and chi_z of the NBCP Y and V states (D46).

Interacting spin-wave theory (``solve_nlswt``) about the classical Y (0.2 T)
and V (1.4 T) three-sublattice states with J_PD = 0.010 meV, at the
orientation phi = 0 selected by the zero-point energy (the tadpole along the
global rotation then vanishes). For each state the energy per spin is split
into E_cl (S^2), E_zp (S^1) and the order-S^0 part (Hartree-Fock + cubic +
tadpole); m_z = -dE/dh and chi_z = -d^2E/dh^2 are central differences over
the field with the classical state re-solved at each field, so each order
of m and chi is the full derivative of that order of E (moment reduction and
canting-angle shifts included).

The gap relation Delta^2 = C_phi / chi_z holds at leading order in 1/S only,
so these numbers are not a next-order gap; chi at two loops is reported as
the size of the next terms of the series.

Run from the repository root:
    python examples/nbcp_nonlinear_spin_waves.py
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np
from scipy.optimize import root

from model import nbcp
from spintoolkit.methods.nlswt import NLSWTSettings, solve_nlswt
from spintoolkit.system.conditions import ExternalConditions

OUT = ROOT / 'data-space/verification/261003-nbcp-nonlinear-spin-waves'
SCAN = ROOT / 'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json'
S, J, JZ, JPD = 0.5, 0.075, 0.125, 0.010
MESHES = [12, 24]
STEPS = [2e-3, 1e-3]
PAIRS = [(0, 1), (1, 2), (2, 0)]


def classical_gradient(theta, h):
    """d E_cl / d theta per cell for spins in a vertical plane (three-sublattice XXZ)."""
    st, ct = np.sin(theta), np.cos(theta)
    grad = h * S * st
    for i, j in PAIRS:
        grad[i] += 3 * S * S * (J * ct[i] * st[j] - JZ * st[i] * ct[j])
        grad[j] += 3 * S * S * (J * ct[j] * st[i] - JZ * st[j] * ct[i])
    return grad


def stationary(phase, h, guess):
    if phase == 'Y':
        t = np.arccos((h + 3 * S * JZ) / (3 * S * (J + JZ)))
        return np.array([t, -t, np.pi])
    theta = root(lambda t: classical_gradient(t, h), guess, tol=1e-14).x
    assert np.max(np.abs(classical_gradient(theta, h))) < 1e-13
    return theta


def energies(model, phase, h, guess, n):
    theta = stationary(phase, h, guess)
    state = nbcp.candidate_state(model, 'three_msl', np.column_stack([theta, np.zeros(3)]).ravel())
    e = solve_nlswt(model, state, ExternalConditions(field=[0, 0, h]),
                    settings=NLSWTSettings(mesh=(n, n))).energies
    return e


def main():
    start = time.monotonic()
    scan = json.loads(SCAN.read_text())
    model = nbcp.build_model({'Jxy': J, 'Jz': JZ, 'JPD': JPD})
    rows = []
    for phase in ['Y', 'V']:
        h0 = scan['states'][phase]['h_meV']
        guess = np.array(scan['states'][phase]['theta'])
        for n in MESHES:
            for dh in STEPS:
                es = [energies(model, phase, h0 + s * dh, guess, n) for s in (-1, 0, 1)]
                E = np.array([[e.classical, e.zero_point, e.order_s0] for e in es])
                m = -(E[2] - E[0]) / (2 * dh)
                chi = -(E[2] - 2 * E[1] + E[0]) / dh ** 2
                e = es[1]
                row = {'phase': phase, 'B_T': scan['states'][phase]['B_T'], 'h_meV': h0,
                       'mesh': n, 'dh_meV': dh,
                       'E_per_spin': {'classical': e.classical, 'zero_point': e.zero_point,
                                      'hartree_fock': e.hartree_fock, 'cubic': e.cubic,
                                      'tadpole': e.tadpole, 'order_s0': e.order_s0},
                       'm_orders': m.tolist(), 'chi_orders': chi.tolist(),
                       'chi_ratios': (chi / chi[0]).tolist()}
                rows.append(row)
                print('%s N=%d dh=%.0e: E %.6f %+.6f %+.6f | m %.4f %+.4f %+.4f | chi %.4f %+.4f %+.4f'
                      % ((phase, n, dh) + tuple(E[1]) + tuple(m) + tuple(chi)), flush=True)
    for phase in ['Y', 'V']:
        chi2 = [r['chi_ratios'][2] for r in rows if r['phase'] == phase]
        assert np.ptp(chi2) < 0.03, (phase, chi2)
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'NBCP Y (0.2 T) and V (1.4 T), S = 1/2, J = 0.075, J_z = 0.125, J_PD = 0.010 meV, '
                 'J_Gamma = 0, phi = 0, field h = g_z mu_B B along z; energies per spin (meV), '
                 'm per spin, chi per spin (1/meV); orders: classical, one loop (E_zp), two loop '
                 '(Hartree-Fock + cubic + tadpole, order S^0).',
        'rows': rows,
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'nbcp-nonlinear-spin-waves.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
