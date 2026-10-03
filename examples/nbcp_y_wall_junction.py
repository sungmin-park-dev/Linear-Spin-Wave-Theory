"""Classical density walls, their phase structure and Z3 junctions in the NBCP Y state.

Three checks at 0.2 T, J = 0.075 meV, J_z = 0.125 meV, S = 1/2, J_Gamma = 0,
for J_PD = 0 and 0.010 meV. All spins move on the full sphere; there is no
quantum angular potential and no thermal free energy.

1. Periodic strips without fixed slabs (`nbcp_y_density_wall.DensityWall`
   geometry, W = 1) containing three walls of one kind. Domain d is the
   reference Y state with its sublattice spins cyclically permuted d times.
   A "forward" wall takes domain d to d + 1 when crossed along the strip
   axis, a "backward" wall takes d to d - 1. With three walls of one kind the
   walls are equivalent by translation, so E / (3 l) is the tension of that
   wall. A strip with one wall of each kind gives their average, which is the
   quantity of the earlier two-wall scan.
2. The phase across a wall. The transverse angle phi of domain d is the
   in-plane angle of its sublattice d (sublattice d + 1 points opposite).
   For J_PD != 0 the walls select an absolute angle phi*. Every bulk layer
   is held at phi* + u by a penalty and the layers within a margin of each
   wall relax; the energy against the margin separates the wall's own
   pinning from the bulk twist. For J_PD = 0 phi is free, and a strip with
   fixed slabs gives the phase jump across a single forward wall.
3. Honeycomb networks of forward walls on a torus spanned by s(a1 + a2) and
   s(2 a2 - a1), with hexagon centres at s a1 and s a2: three hexagonal
   domains, nine walls normal to a1, a2 and a2 - a1, and six Z3 junctions.
   Every wall is forward. Several initial phase patterns are relaxed; a few spins at each
   hexagon centre have their z components held, so that the network cannot
   coarsen away. Around each junction the change of phi within the domains
   (the bulk winding) is measured on a circle of radius 0.2 s.

Run from the repository root:
    python examples/nbcp_y_wall_junction.py
"""

from concurrent.futures import ProcessPoolExecutor
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
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np

from examples.nbcp_y_density_wall import DensityWall
from examples.nbcp_y_soc_conditions import background
from examples.nbcp_y_stiffness import DELTAS, rz
from examples.nbcp_y_vortex_core import A1, A2, Cluster

OUT = ROOT / 'data-space/verification/261003-y-wall-junction'
JPDS = [0.0, 0.010]
LATTICE = np.column_stack([A1, A2])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_phase(n):
    """Domain label, phi and amplitude per layer from the staggered in-plane estimators."""
    c = (n[..., 0] + 1j * n[..., 1]).mean(axis=1)
    estimators = np.array([(c[:, d] - c[:, (d + 1) % 3]) / 2 for d in range(3)])
    best = np.argmax(abs(estimators), axis=0)
    picked = estimators[best, np.arange(len(best))]
    return best, np.angle(picked), abs(picked)


# --------------------------------------------------------------------------
# Strips
# --------------------------------------------------------------------------

class Strip(DensityWall):
    """Periodic strip of magnetic cells with domains in the given order and no fixed slabs."""

    def __init__(self, pd, length, order, axis=0):
        super().__init__(pd, 0., length, axis=axis)
        self.free[:] = True
        self.order = order
        self.held = None

    def seed_domains(self, noise=0.01):
        k = len(self.order)
        n = np.zeros(self.fixed.shape)
        for a, d in enumerate(self.order):
            n[a * self.length // k:(a + 1) * self.length // k] = np.roll(self.bg[2], d, axis=0)
        n += np.random.default_rng(3).normal(scale=noise, size=n.shape)
        return (n / np.linalg.norm(n, axis=-1, keepdims=True)).ravel()

    def hold(self, phi, lam=20.0, margin=6):
        """Penalty (lam/2) sum_l c_l^2 with c_l = Im[z_d exp(-i phi)] on each bulk layer l.

        Every layer farther than `margin` layers from a wall is held, so the bulk
        cannot twist internally; the phase relaxes only within `margin` of each wall.
        """
        k = len(self.order)
        v = np.zeros(self.fixed.shape)
        for a, d in enumerate(self.order):
            rows = slice(a * self.length // k + margin, (a + 1) * self.length // k - margin)
            for sub, sign in [(d % 3, 1), ((d + 1) % 3, -1)]:
                v[rows, :, sub, 0] -= sign * np.sin(phi) / 2
                v[rows, :, sub, 1] += sign * np.cos(phi) / 2
        self.held, self.lam = v, lam

    def energy_spins(self, n):
        energy, grad = super().energy_spins(n)
        if getattr(self, 'held', None) is None:
            return energy, grad
        c = np.sum(self.held * n, axis=(1, 2, 3))
        return energy + 0.5 * self.lam * float(c @ c), grad + self.lam * c[:, None, None, None] * self.held

    def penalty(self, n):
        if self.held is None:
            return 0.
        c = np.sum(self.held * n, axis=(1, 2, 3))
        return 0.5 * self.lam * float(c @ c)


def strip_case(job):
    pd, length, kind, axis = job
    order = {'forward': (0, 1, 2), 'backward': (0, 2, 1), 'one_each': (0, 1)}[kind]
    strip = Strip(pd, length, order, axis)
    row, n = strip.solve(start=strip.seed_domains())
    best, phase, _ = strip_phase(n)
    first = best == order[0]
    out = {'JPD_meV': pd, 'L_cells': length, 'walls': kind, 'axis': axis,
           'tension_meV_per_a': row['excess_energy_meV'] / (len(order) * row['wall_length_a']),
           'bulk_phase_domain0_rad': float(np.angle(np.mean(np.exp(1j * phase[first])))),
           'max_projected_force_meV': row['max_projected_force_meV']}
    if kind == 'forward':
        k = len(order)
        centres = [(a + 0.5) * length // k for a in range(k)]
        out['domain_phases_rad'] = [float(phase[int(c)]) for c in centres]
    return out


def pinning_response(pd, length=96, axis=0, margins=(1, 2, 3, 4, 6, 10), lam=50.0, u=0.2):
    """Energy of three forward walls when every bulk layer is held at phi* + u.

    Layers within `margin` of a wall relax freely. If the wall fixed the phase at its
    core with stiffness k_w and the bulk had stiffness rho_n along the wall normal,
    the energy per wall length would be k u^2 / 2 with 1/k = 1/k_w + d / (2 rho_n),
    d being the free distance on each side. The returned fit tests this form.
    """
    strip = Strip(pd, length, (0, 1, 2), axis)
    row, n = strip.solve(start=strip.seed_domains())
    base, ell = row['excess_energy_meV'], row['wall_length_a']
    _, phase, _ = strip_phase(n)
    phistar = float(np.angle(np.mean(np.exp(1j * phase[:length // 3]))))
    rows = []
    for margin in margins:
        for target in (u, np.pi / 2):
            strip.hold(phistar + target, lam=lam, margin=margin)
            r, m = strip.solve(start=n.copy().ravel())
            _, p, _ = strip_phase(m)
            held = float(np.angle(np.mean(np.exp(1j * (p[margin:length // 3 - margin] - phistar)))))
            rows.append({'margin_layers': margin, 'free_distance_a': margin * strip.spacing,
                         'u_target_rad': float(target), 'u_rad': held,
                         'energy_meV_per_a': (r['excess_energy_meV'] - strip.penalty(m) - base) / (3 * ell),
                         'max_projected_force_meV': r['max_projected_force_meV']})
            strip.held = None
    small = [r for r in rows if r['u_target_rad'] == u]
    d = np.array([r['free_distance_a'] for r in small])
    compliance = np.array([r['u_rad'] ** 2 / (2 * r['energy_meV_per_a']) for r in small])
    fit = d >= 4
    slope, intercept = np.polyfit(d[fit], compliance[fit], 1)
    return {'JPD_meV': pd, 'L_cells': length, 'axis': axis, 'penalty': lam, 'phi_star_rad': phistar,
            'rows': rows, 'compliance_1_over_k_a_per_meV': compliance.tolist(),
            'fit_d_min_a': 4.0, 'fit_slope_per_meV': float(slope), 'fit_intercept_a_per_meV': float(intercept),
            'rho_normal_meV': float(1 / (2 * slope)),
            'note': 'Energies per wall length. The u = pi/2 rows can relax to other wall branches when the '
                    'free distance is large; they bound the cost of a quarter-turn mismatch held near the core.'}


def forward_jump(pd, length, axis=0):
    """Phase jump across the forward wall of a slab-fixed strip (two walls, slabs at phi = 0)."""
    wall = DensityWall(pd, 0., length, axis=axis)
    row, n = wall.solve(noise=.01)
    best, phase, _ = strip_phase(n)
    layers = np.arange(2, length // 2)
    b = best[layers]
    ia, ib = layers[b == 0], layers[b == 1]
    centre = (ia.max() + ib.min()) / 2
    a_side, b_side = ia[ia < centre - 8], ib[ib > centre + 8]
    fa = np.polyval(np.polyfit(a_side, np.unwrap(phase[a_side]), 1), centre)
    fb = np.polyval(np.polyfit(b_side, np.unwrap(phase[b_side]), 1), centre)
    return {'JPD_meV': pd, 'L_cells': length, 'axis': axis,
            'jump_rad': float(np.angle(np.exp(1j * (fb - fa)))),
            'max_projected_force_meV': row['max_projected_force_meV']}


# --------------------------------------------------------------------------
# Honeycomb networks
# --------------------------------------------------------------------------

class Network(Cluster):
    """Torus spanned by s(a1 + a2) and s(2 a2 - a1) with a honeycomb of walls."""

    def __init__(self, pd, s):
        bg = background(pd, 0.0, 0.)
        self.h, self.normals, exchanges = bg[0], bg[2], bg[4]
        self.s = s
        self.M = np.array([[s, -s], [s, 2 * s]])          # columns: T1, T2 in (i, j)
        self.Minv = np.linalg.inv(self.M)
        cells = [(i, j) for i in range(-s, 2 * s + 1) for j in range(0, 3 * s + 1)
                 if all(-1e-9 <= f < 1 - 1e-9 for f in self.Minv @ (i, j))]
        cells = np.array(cells)
        assert len(cells) == 3 * s * s
        self.index = {self.key(c): k for k, c in enumerate(cells)}
        self.r = cells @ LATTICE.T
        self.color = (2 - cells[:, 0] - 2 * cells[:, 1]) % 3
        self.free = np.ones(len(cells), bool)
        basis = np.linalg.inv(LATTICE)
        bi, bj, bJ = [], [], []
        for k in range(len(cells)):
            for d, exchange in zip(DELTAS, exchanges):
                other = self.index[self.key(np.rint(basis @ (self.r[k] - d)).astype(int))]
                assert self.color[other] == (self.color[k] + 1) % 3
                bi.append(k), bj.append(other), bJ.append(exchange)
        self.bi, self.bj, self.bJ = np.array(bi), np.array(bj), np.array(bJ)
        self.T = self.M.T @ LATTICE.T                     # rows: T1, T2
        neighbour = lambda shift: np.array([self.index[self.key(c + shift)] for c in cells])
        tri = np.stack([np.arange(len(cells)), neighbour((1, 0)), neighbour((0, 1))], 1)
        self.tri = np.take_along_axis(tri, np.argsort(self.color[tri], 1), 1)   # columns by colour
        self.tri_pos = self.r + (A1 + A2) / 3
        self.pin = None

    def key(self, c):
        f = self.Minv @ np.asarray(c, float)
        return tuple(np.rint(self.M @ (f - np.floor(f + 1e-9))).astype(int))

    def minimum_image(self, point, positions=None):
        x = (self.r if positions is None else positions) - point
        f = x @ np.linalg.inv(self.T)
        f -= np.round(f)
        candidates = np.stack([(f + (a, b)) @ self.T for a in (-1, 0, 1) for b in (-1, 0, 1)])
        pick = np.argmin(np.linalg.norm(candidates, axis=2), axis=0)
        return candidates[pick, np.arange(len(x))]

    def centres(self):
        return [k * self.s * A1 for k in range(3)]

    def junctions(self):
        third = self.s * (A1 + A2) / 3
        return [third + k * self.s * A1 for k in range(3)] + [2 * third + k * self.s * A1 for k in range(3)]

    def domains(self, chirality=-1):
        """Hexagon k gets domain chirality * k mod 3; chirality -1 makes every wall forward."""
        distance = np.stack([np.linalg.norm(self.minimum_image(p), axis=1) for p in self.centres()], 1)
        return (chirality * np.argmin(distance, 1)) % 3

    def spins(self, domain, phi):
        base = self.normals[(self.color - domain) % 3]
        c, s = np.cos(phi), np.sin(phi)
        return np.column_stack([c * base[:, 0] - s * base[:, 1], s * base[:, 0] + c * base[:, 1], base[:, 2]])

    def hold_centres(self, domain, radius=2.5, lam=5.0):
        mask = np.zeros(len(self.r), bool)
        for p in self.centres():
            mask |= np.linalg.norm(self.minimum_image(p), axis=1) < radius
        self.pin = (np.where(mask)[0], self.normals[(self.color - domain) % 3][mask, 2], lam)

    def penalty(self, spins):
        idx, target, lam = self.pin
        return 0.5 * lam * float(np.sum((spins[idx, 2] - target) ** 2))

    def energy_gradient(self, spins):
        energy, grad = Cluster.energy_gradient(self, spins)
        if self.pin is None:
            return energy, grad
        idx, target, lam = self.pin
        dz = spins[idx, 2] - target
        grad = grad.copy()
        grad[idx, 2] += lam * dz
        return energy + 0.5 * lam * float(dz @ dz), grad

    def winding(self, spins, centre, radius, width=0.8, threshold=0.45):
        """(total, bulk) phase winding / 2 pi on a circle; bulk keeps steps inside one domain."""
        c = spins[self.tri][:, :, 0] + 1j * spins[self.tri][:, :, 1]
        z = np.stack([(c[:, d] - c[:, (d + 1) % 3]) / 2 for d in range(3)], 1)
        best = np.argmax(abs(z), 1)
        picked = z[np.arange(len(z)), best]
        rel = self.minimum_image(centre, self.tri_pos)
        on = np.abs(np.linalg.norm(rel, axis=1) - radius) < width
        order = np.argsort(np.arctan2(rel[on, 1], rel[on, 0]))
        b, p, a = best[on][order], np.angle(picked[on][order]), abs(picked[on][order])
        steps = np.angle(np.exp(1j * (np.roll(p, -1) - p)))
        inside = (b == np.roll(b, -1)) & (a > threshold) & (np.roll(a, -1) > threshold)
        return float(steps.sum() / (2 * np.pi)), float(steps[inside].sum() / (2 * np.pi))


SEEDS = {'uniform': np.zeros(3), 'steps_2pi/3': 2 * np.pi / 3 * np.arange(3),
         **{f'random_{k}': np.random.default_rng(k).uniform(0, 2 * np.pi, 3) for k in range(4)}}


def network_case(job):
    pd, s, seed = job
    net = Network(pd, s)
    domain = net.domains()
    reference, _, _ = net.relax(net.spins(np.zeros(len(net.r), int), np.zeros(len(net.r))))
    net.hold_centres(domain)
    start = time.monotonic()
    energy, spins, torque = net.relax(net.spins(domain, SEEDS[seed][domain]))
    penalty = net.penalty(spins)
    windings = [net.winding(spins, j, 0.2 * s) for j in net.junctions()]
    wall_length = 9 * s / np.sqrt(3)
    return {'JPD_meV': pd, 's': s, 'seed': seed, 'excess_energy_meV': energy - penalty - reference,
            'per_wall_length_meV_per_a': (energy - penalty - reference) / wall_length,
            'hold_penalty_meV': penalty, 'max_torque_meV': torque,
            'total_winding': [round(w[0], 3) for w in windings],
            'bulk_winding': [round(w[1], 3) for w in windings], 'seconds': time.monotonic() - start}


def main():
    start = time.monotonic()
    strip_jobs = [(pd, length, kind, axis) for pd in JPDS for axis in (0, 1)
                  for kind in ('forward', 'backward') for length in (96, 192)]
    strip_jobs += [(pd, 96, 'one_each', axis) for pd in JPDS for axis in (0, 1)]
    network_jobs = [(pd, s, seed) for pd in JPDS for s in (24, 48, 72) for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        strips = list(pool.map(strip_case, strip_jobs))
        pinning = list(pool.map(pinning_response, [JPDS[1], JPDS[1]], [96, 96], [0, 1]))
        jumps = list(pool.map(forward_jump, [0.0, 0.0, 0.0], [128, 256, 512]))
        networks = list(pool.map(network_case, network_jobs))
    for r in strips:
        print('strip J_PD=%.3f L=%d axis %d %-8s sigma %.6f phi0 %+.3f' % (
            r['JPD_meV'], r['L_cells'], r['axis'], r['walls'], r['tension_meV_per_a'], r['bulk_phase_domain0_rad']))
    for p in pinning:
        print('pinning axis %d phi* %+.3f 1/k %s, intercept %.0f, rho_n %.5f' % (
            p['axis'], p['phi_star_rad'], [round(c) for c in p['compliance_1_over_k_a_per_meV']],
            p['fit_intercept_a_per_meV'], p['rho_normal_meV']))
    for p in pinning:
        print('  quarter turn:', [(r['margin_layers'], '%.2e' % r['energy_meV_per_a']) for r in p['rows']
                                  if r['u_target_rad'] > 1])
    print('forward-wall jump at J_PD = 0:', [(j['L_cells'], round(j['jump_rad'], 3)) for j in jumps])

    summary = []
    for pd in JPDS:
        for s in (24, 48, 72):
            runs = sorted([r for r in networks if r['JPD_meV'] == pd and r['s'] == s],
                          key=lambda r: r['excess_energy_meV'])
            lowest = runs[0]
            fractional = [r for r in runs if max(abs(b) for b in r['bulk_winding']) > 0.3]
            unwound = [r for r in runs if max(abs(b) for b in r['bulk_winding']) < 0.05]
            summary.append({'JPD_meV': pd, 's': s, 'lowest_seed': lowest['seed'],
                            'lowest_per_wall_length_meV_per_a': lowest['per_wall_length_meV_per_a'],
                            'lowest_max_abs_bulk_winding': max(abs(b) for b in lowest['bulk_winding']),
                            'gap_to_lowest_with_bulk_winding_above_0.3_meV':
                                fractional[0]['excess_energy_meV'] - lowest['excess_energy_meV'] if fractional else None,
                            'gap_to_lowest_unwound_meV':
                                unwound[0]['excess_energy_meV'] - lowest['excess_energy_meV'] if unwound else None})
            print('network J_PD=%.3f s=%d lowest %s %.6f meV/a, max |bulk winding| %.2f, gap to |q|>0.3 %s, '
                  'gap to unwound %s' % (pd, s, lowest['seed'], lowest['per_wall_length_meV_per_a'],
                                         summary[-1]['lowest_max_abs_bulk_winding'],
                                         summary[-1]['gap_to_lowest_with_bulk_winding_above_0.3_meV'],
                                         summary[-1]['gap_to_lowest_unwound_meV']))

    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Classical Y state at 0.2 T, J = 0.075 meV, J_z = 0.125 meV, S = 1/2, J_Gamma = 0; full-sphere '
                 'relaxations; no quantum angular potential or thermal free energy. Lengths in bond lengths a.',
        'conventions': 'Domain d = reference Y with sublattice spins cyclically permuted d times (np.roll); phi of '
                       'domain d is the in-plane angle of sublattice d. Forward wall: d -> d + 1 crossing along '
                       'the strip axis (+60 or -60 degrees for axis 0 or 1).',
        'strips': strips, 'pinning_potential': pinning, 'forward_wall_jump_JPD0': jumps,
        'networks': networks, 'network_summary': summary,
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__), ROOT / 'examples/nbcp_y_density_wall.py',
                           ROOT / 'examples/nbcp_y_vortex_core.py', ROOT / 'examples/nbcp_y_soc_conditions.py',
                           ROOT / 'examples/nbcp_y_stiffness.py']},
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'wall-junction-check.json').write_text(json.dumps(record, indent=1))

    tension = {(r['JPD_meV'], r['walls'], r['axis'], r['L_cells']): r['tension_meV_per_a'] for r in strips}
    for pd in JPDS:
        for axis in (0, 1):
            f, b = tension[(pd, 'forward', axis, 192)], tension[(pd, 'backward', axis, 192)]
            # At J_PD = 0 three forward walls on a ring cannot all take their preferred jump,
            # and the residual twist costs O(1/L).
            assert abs(tension[(pd, 'forward', axis, 96)] - f) < (2e-5 if pd == 0 else 1e-6)
            assert abs(tension[(pd, 'one_each', axis, 96)] - (f + b) / 2) < 1e-4      # the earlier average
            assert f < b < 2 * f                                     # backward walls are costlier but do not split
    for p in pinning:
        c = p['compliance_1_over_k_a_per_meV']
        assert all(np.diff(c) > 0)                               # softer as the free distance grows
        assert abs(p['fit_intercept_a_per_meV']) < 0.15 * c[-1]  # no resolvable compliance of the wall itself
    gaps = [abs(abs(j['jump_rad']) - np.pi) for j in jumps]
    assert gaps[0] > gaps[1] > gaps[2]                       # the J_PD = 0 forward jump approaches pi
    assert max(r['max_projected_force_meV'] for r in strips) < 1e-6
    assert max(r['max_torque_meV'] for r in networks) < 1e-5
    pure, pinned = [[row for row in summary if row['JPD_meV'] == pd] for pd in JPDS]
    assert pure[0]['lowest_max_abs_bulk_winding'] < pure[2]['lowest_max_abs_bulk_winding']   # grows with s
    assert pure[2]['gap_to_lowest_unwound_meV'] > 0
    assert all(row['lowest_max_abs_bulk_winding'] < 0.05 for row in pinned[1:])
    assert pinned[1]['gap_to_lowest_with_bulk_winding_above_0.3_meV'] < \
        pinned[2]['gap_to_lowest_with_bulk_winding_above_0.3_meV']


if __name__ == '__main__':
    main()
