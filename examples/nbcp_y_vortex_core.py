"""Classical vortex-core energy of the NBCP Y state at 0.2 T.

A hexagonal cluster of radius R (in bond lengths) carries the Y state. Its two
outer rings are fixed to the winding ansatz

    n_i = R_z(m arg(r_i - r0) + phi0) n_Y[colour(i)],

and every other spin is relaxed on the sphere. The vortex energy is measured
against the uniform state (m = 0) with the same fixed rings, and is fitted to

    E_v(R) = E_core + kappa ln R,   kappa = pi sqrt(det rho),

with rho the relaxed stiffness of `nbcp_y_soc_conditions.reduction`. The
bonds and sublattice colours follow the LSWT kernel of that module: each
stored displacement d joins colour c at r to colour c + 1 at r - d.

At J_PD != 0 the windings m = +1 and m = -1 are inequivalent. For m = +1 the
energy contains a term growing linearly with R whose sign depends on phi0, a
boundary contribution of a total-derivative gradient term; the script reports
the phi0 average, which removes it, and the phi0 spread. This is classical
only: no zero-point or thermal core free energy.

Run from the repository root:
    python examples/nbcp_y_vortex_core.py
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
from scipy.optimize import minimize

from examples.nbcp_y_soc_conditions import background, reduction
from examples.nbcp_y_stiffness import AREA, DELTAS, S

OUT = ROOT / 'data-space/verification/261002-y-vortex-core'
A1, A2 = np.array([1., 0.]), np.array([.5, np.sqrt(3) / 2])
SIZES = [16, 24, 32, 48]
FIT_SIZES = [24, 32, 48]
OFFSETS = np.arange(12) * np.pi / 6
CORE = np.array([.5, np.sqrt(3) / 6])     # centre of an elementary triangle


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Cluster:
    """Hexagonal Y cluster with fixed outer rings."""

    def __init__(self, pd, radius, fixed_rings=2):
        bg = background(pd, 0.0, 0.)
        self.h, self.normals, exchanges = bg[0], bg[2], bg[4]
        cells = np.array([(i, j) for i in range(-radius, radius + 1)
                          for j in range(-radius, radius + 1) if abs(i + j) <= radius])
        self.r = cells[:, :1] * A1 + cells[:, 1:] * A2
        self.color = (2 - cells[:, 0] - 2 * cells[:, 1]) % 3
        ring = np.max(np.abs(np.column_stack([cells, cells.sum(1)])), axis=1)
        self.free = ring <= radius - fixed_rings
        index = {tuple(c): k for k, c in enumerate(cells)}
        basis = np.linalg.inv(np.column_stack([A1, A2]))
        bi, bj, bJ = [], [], []
        for k in range(len(cells)):
            for d, exchange in zip(DELTAS, exchanges):
                other = index.get(tuple(np.rint(basis @ (self.r[k] - d)).astype(int)))
                if other is not None:
                    assert self.color[other] == (self.color[k] + 1) % 3
                    bi.append(k), bj.append(other), bJ.append(exchange)
        self.bi, self.bj, self.bJ = np.array(bi), np.array(bj), np.array(bJ)

    def ansatz(self, winding, offset, core=CORE):
        rel = self.r - core
        angle = winding * np.arctan2(rel[:, 1], rel[:, 0]) + offset
        base = self.normals[self.color]
        c, s = np.cos(angle), np.sin(angle)
        return np.column_stack([c * base[:, 0] - s * base[:, 1],
                                s * base[:, 0] + c * base[:, 1], base[:, 2]])

    def energy_gradient(self, spins):
        field_j = np.einsum('bxy,by->bx', self.bJ, spins[self.bj])
        energy = S * S * np.sum(spins[self.bi] * field_j) - self.h * S * spins[:, 2].sum()
        grad = np.zeros_like(spins)
        np.add.at(grad, self.bi, S * S * field_j)
        np.add.at(grad, self.bj, S * S * np.einsum('bxy,bx->by', self.bJ, spins[self.bi]))
        grad[:, 2] -= self.h * S
        return energy, grad

    def relax(self, spins):
        spins, free = spins.copy(), self.free

        def objective(x):
            v = x.reshape(-1, 3)
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            u = v / norm
            spins[free] = u
            energy, grad = self.energy_gradient(spins)
            g = grad[free]
            return energy, ((g - np.sum(g * u, 1, keepdims=True) * u) / norm).ravel()

        result = minimize(objective, spins[free].ravel(), jac=True, method='L-BFGS-B',
                          options={'maxiter': 20000, 'maxcor': 30, 'ftol': 1e-16, 'gtol': 1e-12})
        v = result.x.reshape(-1, 3)
        spins[free] = v / np.linalg.norm(v, axis=1, keepdims=True)
        energy, grad = self.energy_gradient(spins)
        g, u = grad[free], spins[free]
        torque = float(np.max(np.linalg.norm(g - np.sum(g * u, 1, keepdims=True) * u, axis=1)))
        return float(energy), spins, torque


def vortex_energy(cluster, winding, offset, core=CORE):
    reference, _, t0 = cluster.relax(cluster.ansatz(0, offset))
    energy, spins, t1 = cluster.relax(cluster.ansatz(winding, offset, core))
    return energy - reference, max(t0, t1)


def fit(sizes, energies):
    design = np.column_stack([np.ones(len(sizes)), np.log(sizes)])
    (core, kappa), *_ = np.linalg.lstsq(design, energies, rcond=None)
    return float(core), float(kappa)


def main():
    start = time.monotonic()
    cases = []
    for pd in [0.0, 0.010]:
        rho = reduction(background(pd, 0.0, 0.))[4] / (3 * AREA)
        kappa_pred = float(np.pi * np.sqrt(np.linalg.det(rho)))
        for winding in [1, -1]:
            offsets = OFFSETS if pd else OFFSETS[:3]
            table, torque = {}, 0.
            for radius in SIZES:
                cluster = Cluster(pd, radius)
                values = []
                for offset in offsets:
                    e, t = vortex_energy(cluster, winding, offset)
                    values.append(e)
                    torque = max(torque, t)
                table[radius] = values
            means = np.array([np.mean(table[r]) for r in FIT_SIZES])
            core_fit, kappa_fit = fit(FIT_SIZES, means)
            spread = {r: float(np.ptp(table[r])) for r in SIZES}
            harmonic2 = {r: float(2 * np.mean(np.array(table[r]) * np.cos(2 * offsets)))
                         for r in SIZES} if pd else None
            case = {'JPD_meV': pd, 'winding': winding, 'offsets_rad': offsets.tolist(),
                    'energy_meV': {str(r): v for r, v in table.items()},
                    'offset_spread_meV': {str(r): v for r, v in spread.items()},
                    'offset_cos2_coefficient_meV': None if harmonic2 is None
                    else {str(r): v for r, v in harmonic2.items()},
                    'kappa_fit_meV': kappa_fit, 'kappa_predicted_meV': kappa_pred,
                    'core_fit_meV': core_fit,
                    'core_with_predicted_kappa_meV': float(means[-1] - kappa_pred * np.log(FIT_SIZES[-1])),
                    'max_torque_meV': torque}
            cases.append(case)
            print('J_PD=%.3f m=%+d kappa fit %.5f pred %.5f core %.5f / %.5f spread(48) %.1e' % (
                pd, winding, kappa_fit, kappa_pred, core_fit,
                case['core_with_predicted_kappa_meV'], spread[48]))
    # Core position: triangle centres and sites of each colour at R = 24.
    cluster = Cluster(0.010, 24)
    positions = {'triangle_up': CORE, 'triangle_down': np.array([1., np.sqrt(3) / 3])}
    for colour in range(3):
        k = np.argmin(np.linalg.norm(cluster.r, axis=1) + 10 * (cluster.color != colour))
        positions[f'site_colour_{colour}'] = cluster.r[k]
    position_check = {name: vortex_energy(cluster, -1, 0., core)[0] for name, core in positions.items()}
    print('core position check (J_PD=0.010, m=-1, R=24):', position_check)

    zero, = [c for c in cases if c['JPD_meV'] == 0 and c['winding'] == 1]
    zero_minus, = [c for c in cases if c['JPD_meV'] == 0 and c['winding'] == -1]
    minus, = [c for c in cases if c['JPD_meV'] == 0.010 and c['winding'] == -1]
    assert max(abs(a - b) for r in SIZES for a, b in zip(zero['energy_meV'][str(r)],
                                                         zero_minus['energy_meV'][str(r)])) < 1e-9
    assert abs(zero['kappa_fit_meV'] / zero['kappa_predicted_meV'] - 1) < 0.02
    assert abs(minus['kappa_fit_meV'] / minus['kappa_predicted_meV'] - 1) < 0.08
    assert np.ptp(list(position_check.values())) < 1e-4

    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Classical vortex in the Y state, 0.2 T, J = 0.075 meV, J_z = 0.125 meV, '
                 'S = 1/2, J_Gamma = 0; hexagonal clusters with two fixed outer rings; energies '
                 'in meV relative to the uniform state with the same rings; R in bond lengths.',
        'fit': 'E(R) = core + kappa ln R over R = %s, using the offset average' % FIT_SIZES,
        'cases': cases, 'core_position_check_meV': position_check,
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__), ROOT / 'examples/nbcp_y_soc_conditions.py',
                           ROOT / 'examples/nbcp_y_stiffness.py']},
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'vortex-core-check.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
