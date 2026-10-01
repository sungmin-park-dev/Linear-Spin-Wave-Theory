"""Check the in-plane C2 x time-reversal map behind the NBCP angular reflection.

Section 9 of the NBCP note reports that the leading T=0 V energy along the
classical orbit is symmetric under phi -> pi/3 - phi. This script checks
the operation that gives it. Let C2 rotate by pi about the bond-1 axis
(model x axis) and let T reverse every spin. Their product keeps the
longitudinal field, maps the in-plane displacement (x, y) -> (x, -y) and
maps each spin by S -> diag(-1, 1, 1) S, so the azimuth goes to pi - phi
and the polar angle is unchanged. The script verifies, for mixed J_PD and
J_Gamma:

1. each nearest-neighbor exchange matrix satisfies R J_d R^T = J_{d'}, with
   d' the mirrored displacement and R = diag(1, -1, -1);
2. the mirror maps every three-sublattice site to the same sublattice;
3. the leading vacuum energy satisfies E(phi) = E(pi - phi) and
   E(phi) = E(phi + 2 pi / 3) on the Y and V orbits.

As a control, E(-phi) - E(phi) is also reported. Reflection about phi = 0 is
not implied by this operation. It fails for V once J_Gamma produces a
threefold harmonic; for Y it holds because inversion already halves the
period to pi, which together with 2 pi / 3 leaves period pi / 3.

Run from the repository root:
    python examples/nbcp_c2t_reflection_check.py
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT), str(ROOT / 'legacy')]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lswt-mpl-cache')

import numpy as np

from examples.pseudo_goldstone_comparison import (LATTICE, build, mesh, phase_state,
                                                  spin_angles, vacuum_energy)
from model.nbcp import make_nn_exchange_matrices
from model.nbcp.unit_cells import DISP_NN

OUT = ROOT / 'data-space/verification/261001-c2t-reflection'
COUPLINGS = [(0.010, 0.002), (0.005, 0.001), (0.010, -0.001)]
N_MESH = 12
PHI = [0.1, 0.37, 0.9, 1.3]
R_SPIN = np.diag([1., -1., -1.])
MIRROR = np.diag([1., -1.])
SITES = {'A': [0.5, np.sqrt(3) / 2], 'B': [-0.5, np.sqrt(3) / 2], 'C': [0., 0.]}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bond_residual(pd, gamma):
    mats = make_nn_exchange_matrices({'Jxy': 0.075, 'Jz': 0.125, 'JPD': pd,
                                      'JGamma': gamma})
    disp = [np.array(d) for d in DISP_NN]
    worst = 0.0
    for J, d in zip(mats, disp):
        image = MIRROR @ d
        target = [k for k, e in enumerate(disp)
                  if np.allclose(e, image) or np.allclose(e, -image)]
        assert len(target) == 1
        Jt = mats[target[0]]
        if np.allclose(disp[target[0]], -image):
            Jt = Jt.T
        worst = max(worst, float(np.max(np.abs(R_SPIN @ J @ R_SPIN.T - Jt))))
    return worst


def sublattice_map():
    """Sublattice of each mirrored site, found modulo the lattice vectors."""
    inv = np.linalg.inv(LATTICE.T)
    result = {}
    for name, r in SITES.items():
        image = MIRROR @ np.array(r)
        for other, s in SITES.items():
            n = inv @ (image - np.array(s))
            if np.allclose(n, np.round(n), atol=1e-12):
                result[name] = other
    return result


def energies(state, pd, gamma, points):
    _, current, _ = build(state, pd, gamma)

    def energy(phi):
        mats, torque = current.Quadratic_Bose_Hamiltonian(
            points, angles=spin_angles(state['theta'], phi))
        assert max(abs(v) for v in torque.values()) < 2e-12
        return vacuum_energy(mats)[0]

    rows = []
    for phi in PHI:
        e = energy(phi)
        rows.append({'phi': phi, 'E': e,
                     'E_pi_minus_phi_minus_E': energy(np.pi - phi) - e,
                     'E_phi_plus_2pi_over_3_minus_E': energy(phi + 2 * np.pi / 3) - e,
                     'E_pi_over_3_minus_phi_minus_E': energy(np.pi / 3 - phi) - e,
                     'control_E_minus_phi_minus_E': energy(-phi) - e})
    return rows


def main():
    points = mesh(N_MESH)
    sub = sublattice_map()
    print('sublattice map under the mirror:', sub)
    assert sub == {'A': 'A', 'B': 'B', 'C': 'C'}
    records = []
    for pd, gamma in COUPLINGS:
        res = bond_residual(pd, gamma)
        print(f'PD={pd} G={gamma:+} bond residual {res:.1e}')
        assert res < 1e-15
        for phase in ('Y', 'V'):
            rows = energies(phase_state(phase), pd, gamma, points)
            worst = {k: max(abs(r[k]) for r in rows) for k in
                     ('E_pi_minus_phi_minus_E', 'E_phi_plus_2pi_over_3_minus_E',
                      'E_pi_over_3_minus_phi_minus_E', 'control_E_minus_phi_minus_E')}
            print(f'  {phase}: ' + ', '.join(f'{k} {v:.1e}' for k, v in worst.items()))
            records.append({'phase': phase, 'JPD_meV': pd, 'JGamma_meV': gamma,
                            'bond_residual': res, 'rows': rows, 'max_abs': worst})
    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'C2 about the bond-1 axis times time reversal on the NBCP nearest-neighbor '
                 'model with XXZ, PD and Gamma exchange and a longitudinal field; leading '
                 'T=0 vacuum energy on the fixed classical Y (0.2 T) and V (1.4 T) orbits.',
        'spin_map': 'S -> -R S with R = diag(1,-1,-1), i.e. azimuth phi -> pi - phi',
        'site_map': '(x, y) -> (x, -y); sublattices preserved',
        'sublattice_map': sub,
        'quadrature': f'{N_MESH}x{N_MESH} midpoints with six rotated copies',
        'records': records,
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__), ROOT / 'examples/pseudo_goldstone_comparison.py',
                           ROOT / 'model/nbcp/exchange.py', ROOT / 'model/nbcp/unit_cells.py']},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'c2t-reflection-check.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
