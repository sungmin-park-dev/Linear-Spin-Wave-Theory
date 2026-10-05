"""Vortex-wall interaction and harmonic wall free energy in the classical NBCP Y state.

Two checks at 0.2 T, J = 0.075 meV, J_z = 0.125 meV, S = 1/2, J_Gamma = 0, for
J_PD = 0 and 0.010 meV. Classical spins only: no quantum angular potential, no
anharmonic or quantum free energy.

1. Vortex-wall interaction. A periodic torus of L x W magnetic cells holds
   three forward walls (`nbcp_y_wall_junction.Strip` with width W). A
   vortex-antivortex pair is placed in the middle domain at distance d from
   one wall, split by half the torus period along the wall. Each core is held
   by a ring of spins (1.8 < r < 3.2 bond lengths) whose in-plane vectors are
   pulled toward the pair ansatz with a free common rotation per core; the
   winding is measured afterwards on a circle of radius 5. The energy against
   the wall-only torus is compared with the continuum image solution for a
   strip with fixed phase (J_PD != 0) or free phase (J_PD = 0) on both walls,
   using the relaxed stiffness of the middle domain.
2. Wall free energy. The tangent-space Hessian of a W = 1 strip with three
   forward walls is block-diagonal in the Bloch momentum k along the wall.
   The classical harmonic free energy per wall length is
   F_wall(T) = sigma + T f1 with
   f1 = (1 / (2 n_w l N_k)) sum_k [sum ln lambda_wall(k) - sum ln lambda_bulk(k)],
   against a uniform strip at the phase of the first domain. f1 is
   extrapolated in N_k and in the strip length.

Run from the repository root:
    python examples/nbcp_y_vortex_wall.py
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
from scipy.optimize import minimize

from examples.nbcp_y_density_wall import DensityWall
from examples.nbcp_y_soc_conditions import background, reduction
from examples.nbcp_y_stiffness import AREA, S, rz
from examples.nbcp_y_wall_junction import Strip, strip_phase

OUT = ROOT / 'data-space/verification/261005-y-vortex-wall'
JPDS = [0.0, 0.010]
T_HAT = np.array([np.sqrt(3) / 2, -0.5])      # along the wall: lattice[1] / |lattice[1]|
N_HAT = np.array([0.5, np.sqrt(3) / 2])       # wall normal, toward increasing layer index
DISTANCES = [2.25, 3.75, 5.25, 6.75, 9.75, 12.75, 15.75, 19.75, 24.0]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rot_inplane(n, angle):
    c, s = np.cos(angle), np.sin(angle)
    out = n.copy()
    out[..., 0] = c * n[..., 0] - s * n[..., 1]
    out[..., 1] = s * n[..., 0] + c * n[..., 1]
    return out


# --------------------------------------------------------------------------
# 1. Vortex pair near a wall on a torus
# --------------------------------------------------------------------------

class Torus(Strip):
    """L x W torus with three forward walls and optional held vortex cores."""

    def __init__(self, pd, length, width):
        DensityWall.__init__(self, pd, 0., length, width=width, axis=0)
        self.free[:] = True
        self.order, self.held, self.hold = (0, 1, 2), None, None
        lattice = np.array(self.bg[7].lattice_vectors)
        positions = np.array([self.bg[6]['Spin info'][k]['Position'] for k in 'ABC'])
        a, b, s = np.meshgrid(np.arange(length), np.arange(width), np.arange(3), indexing='ij')
        self.r = a[..., None] * lattice[0] + b[..., None] * lattice[1] + positions[s]
        self.u, self.v = self.r @ N_HAT, self.r @ T_HAT
        self.Lv = width * np.linalg.norm(lattice[1])
        self.layer = a

    def base_state(self):
        """Wall-only state: the relaxed W = 1 strip broadcast over the width."""
        strip = Strip(self.pd, self.length, (0, 1, 2), 0)
        row, n = strip.solve(start=strip.seed_domains())
        return np.broadcast_to(n, self.fixed.shape).copy(), row

    def polar(self, zeta):
        dv = (self.v - zeta.real + self.Lv / 2) % self.Lv - self.Lv / 2
        du = self.u - zeta.imag
        return np.hypot(dv, du).ravel(), np.arctan2(du, dv).ravel()

    def pair_phase(self, z1, z2, layers):
        """Phase of a +1 chain at z1 and a -1 chain at z2 (period Lv along the wall), zero outside `layers`."""
        z = self.v + 1j * self.u
        theta = np.angle(np.sin(np.pi * (z - z1) / self.Lv)) - np.angle(np.sin(np.pi * (z - z2) / self.Lv))
        inside = (self.layer >= layers[0]) & (self.layer < layers[1])
        return np.where(inside, theta, 0.)

    def set_hold(self, base, theta, cores, lam, r1=1.8, r2=3.2):
        self.hold = []
        flat = base.reshape(-1, 3)
        for zeta in cores:
            idx = np.where((self.polar(zeta)[0] > r1) & (self.polar(zeta)[0] < r2))[0]
            b = flat[idx]
            self.hold.append((idx, np.arctan2(b[:, 1], b[:, 0]) + theta.ravel()[idx], np.hypot(b[:, 0], b[:, 1]), lam))

    def penalty_terms(self, n, psis):
        """Penalty energy, its gradient on the spins and on the core rotations psi."""
        flat = n.reshape(-1, 3)
        energy, grad, gpsi = 0., np.zeros_like(flat), np.zeros(len(psis))
        for k, (idx, beta0, amplitude, lam) in enumerate(self.hold):
            beta = beta0 + psis[k]
            dx = flat[idx, 0] - amplitude * np.cos(beta)
            dy = flat[idx, 1] - amplitude * np.sin(beta)
            energy += 0.5 * lam * float(dx @ dx + dy @ dy)
            grad[idx, 0] += lam * dx
            grad[idx, 1] += lam * dy
            gpsi[k] = lam * float(np.sum(amplitude * (dx * np.sin(beta) - dy * np.cos(beta))))
        return energy, grad.reshape(n.shape), gpsi

    def relax_held(self, start, psi0):
        size = 3 * int(self.free.sum())

        def objective(x):
            n, norms = self.unpack(x[:size])
            energy, grad = self.energy_spins(n)
            pen, gpen, gpsi = self.penalty_terms(n, x[size:])
            grad = grad + gpen
            unit, force = n[self.free], grad[self.free]
            projected = (force - unit * np.sum(force * unit, axis=-1, keepdims=True)) / norms
            return energy - self.reference + pen, np.concatenate([projected.ravel(), gpsi])

        x0 = np.concatenate([start[self.free].ravel(), psi0])
        result = minimize(objective, x0, jac=True, method='L-BFGS-B',
                          options={'maxiter': 40000, 'maxcor': 30, 'ftol': 1e-16, 'gtol': 1e-11, 'maxls': 40})
        n, _ = self.unpack(result.x[:size])
        energy, grad = self.energy_spins(n)
        tangent = np.linalg.norm((grad - n * np.sum(grad * n, axis=-1, keepdims=True)).reshape(-1, 3), axis=1)
        held = np.zeros(len(tangent), bool)
        for idx, *_ in self.hold:
            held[idx] = True
        return n, energy - self.reference, self.penalty_terms(n, result.x[size:])[0], float(tangent[~held].max())

    def winding(self, n, base, zeta, radius=5.0, width=0.9):
        """In-plane rotation against the wall-only state, wound once around zeta, divided by 2 pi."""
        dist, angle = self.polar(zeta)
        flat, ref = n.reshape(-1, 3), base.reshape(-1, 3)
        on = np.where((np.abs(dist - radius) < width) & (np.hypot(ref[:, 0], ref[:, 1]) > 0.3))[0]
        rotation = np.angle((flat[on, 0] + 1j * flat[on, 1]) * np.conj(ref[on, 0] + 1j * ref[on, 1]))
        p = rotation[np.argsort(angle[on])]
        return float(np.sum(np.angle(np.exp(1j * (np.roll(p, -1) - p)))) / (2 * np.pi))


def pair_case(job):
    pd, length, width, d, order, seed, lam = job
    torus = Torus(pd, length, width)
    base, _ = torus.base_state()
    e_wall = torus.energy_spins(base)[0] - torus.reference
    lo, hi = length // 3, 2 * length // 3
    u_wall = 0.5 * (torus.u[lo - 1].mean() + torus.u[lo].mean())

    def triangle_centre(u, v):
        k = np.argmin(np.hypot(torus.u - u, (torus.v - v + torus.Lv / 2) % torus.Lv - torus.Lv / 2))
        r = torus.r.reshape(-1, 2)[k] + np.array([0.5, np.sqrt(3) / 6])
        return complex(r @ T_HAT, r @ N_HAT)

    z1, z2 = triangle_centre(u_wall + d, 0.25 * torus.Lv), triangle_centre(u_wall + d, 0.75 * torus.Lv)
    if order < 0:
        z1, z2 = z2, z1
    theta = torus.pair_phase(z1, z2, (lo, hi))
    start = rot_inplane(base, theta)
    torus.set_hold(base, theta, (z1, z2), lam)
    psi0 = np.zeros(2)
    if seed:
        rng = np.random.default_rng(seed)
        start = start + rng.normal(scale=0.08, size=start.shape)
        start /= np.linalg.norm(start, axis=-1, keepdims=True)
        psi0 = rng.uniform(-0.5, 0.5, 2)
    n, energy, penalty, torque = torus.relax_held(start, psi0)
    return {'JPD_meV': pd, 'L_cells': length, 'W_cells': width, 'order': order, 'seed': seed, 'hold_lambda': lam,
            'distance_a': float(z1.imag - u_wall), 'separation_a': float(torus.Lv / 2),
            'domain_width_a': float((hi - lo) * torus.spacing),
            'energy_meV': float(energy - e_wall), 'hold_penalty_meV': penalty, 'max_unheld_torque_meV': torque,
            'windings': [round(torus.winding(n, base, z), 3) for z in (z1, z2)]}


def image_energy(d, w, period, s, rho_uu, rho_vv, fixed_phase, images=60):
    """d-dependent pair energy / kappa for a strip of width w, fixed (Dirichlet) or free (Neumann) phase on both edges."""
    rb = np.sqrt(rho_uu * rho_vv)
    su, sv = np.sqrt(rb / rho_uu), np.sqrt(rb / rho_vv)
    charges = [(1, 0.0, d), (-1, s, d)]
    total = 0.
    for qi, vi, ui in charges:
        for qj, vj, uj in charges:
            for m in range(-images, images + 1):
                for u_image, sign in ((uj + 2 * m * w, 1), (-uj + 2 * m * w, 1 if fixed_phase else -1)):
                    if m == 0 and u_image == uj and (qi, vi) == (qj, vj):
                        continue
                    z = (vi - vj) * sv + 1j * (ui - u_image) * su
                    total -= qi * qj * sign * np.log(abs(np.sin(np.pi * z / (period * sv))))
    return total


def middle_domain_stiffness(pd, length=96):
    """Stiffness tensor (v, u components) of the relaxed middle domain and its orbit angle."""
    strip = Strip(pd, length, (0, 1, 2), 0)
    _, n = strip.solve(start=strip.seed_domains())
    mid = n[length // 2, 0]
    angles = np.linspace(-np.pi, np.pi, 7201)
    normals = background(pd, 0., 0.)[2]
    error = [np.abs(mid - np.roll(normals, 1, axis=0) @ rz(a).T).max() for a in angles]
    phi = float(angles[int(np.argmin(error))])
    rho = reduction(background(pd, 0., phi))[4] / (3 * AREA)
    basis = np.stack([T_HAT, N_HAT])
    return phi, float(min(error)), basis @ rho @ basis.T


# --------------------------------------------------------------------------
# 2. Harmonic wall free energy
# --------------------------------------------------------------------------

def tangent_basis(n):
    ref = np.where(np.abs(n[..., 2:3]) < 0.9, np.array([0., 0., 1.]), np.array([1., 0., 0.]))
    e1 = np.cross(ref, n)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    return e1, np.cross(n, e1)


def hessian_k(strip, n, k):
    """Tangent Hessian (energy = u^dag H u / 2) of a W = 1 strip at Bloch momentum k per cell along the wall."""
    length = strip.length
    spins = n[:, 0]
    _, g = strip.energy_spins(n)
    frames = np.stack(tangent_basis(spins), -1)
    H = np.zeros((6 * length, 6 * length), complex)
    for a in range(length):
        for i in range(3):
            p = 6 * a + 2 * i
            H[p:p + 2, p:p + 2] -= np.eye(2) * float(spins[a, i] @ g[a, 0, i])
    for i, j, (da, db), exchange in strip.links:
        for a in range(length):
            p, q = 6 * a + 2 * i, 6 * ((a + da) % length) + 2 * j
            block = S * S * frames[a, i].T @ exchange @ frames[(a + da) % length, j] * np.exp(1j * k * db)
            H[p:p + 2, q:q + 2] += block
            H[q:q + 2, p:p + 2] += block.conj().T
    return H


def hessian_check():
    """Second derivative along a Bloch mode against finite differences (k = 0 with W = 1, k = pi with W = 2)."""
    rows = []
    rng = np.random.default_rng(1)
    for width, k in [(1, 0.0), (2, np.pi)]:
        strip = Strip(0.010, 8, (0, 1), 0)
        n1 = strip.seed_domains(noise=0.2).reshape(strip.fixed.shape)
        wide = Strip.__new__(Strip)
        DensityWall.__init__(wide, 0.010, 0., 8, width=width, axis=0)
        wide.free[:], wide.held = True, None
        n = np.concatenate([n1] * width, axis=1)
        e1, e2 = tangent_basis(n)
        e0 = wide.energy_spins(n)[0]
        c = rng.normal(size=(8, 3, 2))
        mode = np.stack([c * np.cos(k * b) for b in range(width)], axis=1)

        def energy(eps):
            v = n + eps * (mode[..., :1] * e1 + mode[..., 1:] * e2)
            return wide.energy_spins(v / np.linalg.norm(v, axis=-1, keepdims=True))[0]
        step = 1e-4
        fd = (energy(step) + energy(-step) - 2 * e0) / step ** 2
        analytic = float(np.real(c.ravel() @ hessian_k(strip, n1, k) @ c.ravel())) * width
        rows.append({'width': width, 'k': k, 'finite_difference': fd, 'analytic': analytic,
                     'relative_error': abs(fd - analytic) / abs(analytic)})
    return rows


def wall_free_energy(job):
    pd, length = job
    wall = Strip(pd, length, (0, 1, 2), 0)
    row, nw = wall.solve(start=wall.seed_domains())
    best, phase, _ = strip_phase(nw)
    phi0 = float(np.angle(np.mean(np.exp(1j * phase[best == 0]))))
    bulk = Strip(pd, length, (0,), 0)
    nb = bulk.seed_domains(noise=0.).reshape(bulk.fixed.shape)
    nb = nb @ rz(phi0 - float(np.mean(strip_phase(nb)[1]))).T
    rowb, nb = bulk.solve(start=nb.ravel())
    ell = row['wall_length_a']
    out = {'JPD_meV': pd, 'L_cells': length, 'tension_meV_per_a': row['excess_energy_meV'] / (3 * ell),
           'max_projected_force_meV': max(row['max_projected_force_meV'], rowb['max_projected_force_meV']), 'Nk': {}}
    for nk in (32, 64, 128):
        total, lowest_wall, lowest_bulk = 0., np.inf, np.inf
        for k in 2 * np.pi * (np.arange(nk) + 0.5) / nk:
            lw = np.linalg.eigvalsh(hessian_k(wall, nw, k))
            lb = np.linalg.eigvalsh(hessian_k(bulk, nb, k))
            lowest_wall, lowest_bulk = min(lowest_wall, lw[0]), min(lowest_bulk, lb[0])
            total += np.sum(np.log(lw)) - np.sum(np.log(lb))
        out['Nk'][str(nk)] = {'f1_per_a': 0.5 * total / (3 * ell * nk),
                              'lowest_eigenvalue_wall': float(lowest_wall), 'lowest_eigenvalue_bulk': float(lowest_bulk)}
    return out


def extrapolate(rows):
    """Linear extrapolation in 1/N_k at each length, then in 1/L from the two longest strips."""
    at_length = {}
    for r in rows:
        f64, f128 = r['Nk']['64']['f1_per_a'], r['Nk']['128']['f1_per_a']
        at_length[r['L_cells']] = 2 * f128 - f64
    lengths = sorted(at_length)
    a, b = lengths[-2], lengths[-1]
    limit = (b * at_length[b] - a * at_length[a]) / (b - a)
    return {'f1_infinite_Nk': {str(k): v for k, v in at_length.items()}, 'f1_extrapolated_per_a': limit,
            'f1_uncertainty_per_a': abs(limit - at_length[b])}


def main():
    start = time.monotonic()
    check = hessian_check()
    fe_jobs = [(pd, length) for pd in JPDS for length in (48, 96, 192)]
    pair_jobs = [(pd, 96, 48, d, order, seed, 0.05) for pd in JPDS for d in DISTANCES for order in (1, -1) for seed in (0, 1, 2)]
    pair_jobs += [(0.010, 96, 48, d, order, 0, 0.2) for d in (3.75, 9.75, 19.75) for order in (1, -1)]
    pair_jobs += [(0.010, 144, 72, d, order, 0, 0.05) for d in (3.75, 9.75, 19.75, 30.0) for order in (1, -1)]
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        free_energy = list(pool.map(wall_free_energy, fe_jobs))
        pairs = list(pool.map(pair_case, pair_jobs))
        stiffness = dict(zip(JPDS, pool.map(middle_domain_stiffness, JPDS)))

    summary_fe = {}
    for pd in JPDS:
        rows = [r for r in free_energy if r['JPD_meV'] == pd]
        summary_fe[str(pd)] = {'tension_meV_per_a': rows[-1]['tension_meV_per_a'], **extrapolate(rows)}
        print('J_PD=%.3f sigma %.5f f1 %.4f +- %.4f per a' % (pd, rows[-1]['tension_meV_per_a'],
              summary_fe[str(pd)]['f1_extrapolated_per_a'], summary_fe[str(pd)]['f1_uncertainty_per_a']))

    # Lowest energy at each distance over both core orders and all starts; starts that end on
    # different branches (same order) are listed.
    groups = {}
    for p in pairs:
        if p['hold_lambda'] == 0.05:
            groups.setdefault((p['JPD_meV'], p['L_cells'], round(p['distance_a'], 2)), []).append(p)
    lowest = {key: min(rows, key=lambda p: p['energy_meV']) for key, rows in groups.items()}
    starts = {}
    for p in pairs:
        if p['hold_lambda'] == 0.05:
            starts.setdefault((p['JPD_meV'], p['L_cells'], round(p['distance_a'], 2), p['order']), []).append(p['energy_meV'])
    branches = [{'JPD_meV': k[0], 'L_cells': k[1], 'distance_a': k[2], 'order': k[3],
                 'energies_meV': sorted({round(e, 7) for e in v})}
                for k, v in sorted(starts.items()) if np.ptp(v) > 1e-6]
    comparison = []
    for pd in JPDS:
        phi, match, rho = stiffness[pd]
        kappa = np.pi * np.sqrt(np.linalg.det(rho))
        for length, ref_d in [(96, 19.75), (144, 30.75)]:
            rows = sorted([v for k, v in lowest.items() if k[0] == pd and k[1] == length], key=lambda p: p['distance_a'])
            if not rows:
                continue
            ref = min(rows, key=lambda p: abs(p['distance_a'] - ref_d))
            w, s, period = ref['domain_width_a'], ref['separation_a'], 2 * ref['separation_a']
            model_ref = image_energy(ref['distance_a'], w, period, s, rho[1, 1], rho[0, 0], pd > 0)
            for p in rows:
                model = kappa * (image_energy(p['distance_a'], w, period, s, rho[1, 1], rho[0, 0], pd > 0) - model_ref)
                comparison.append({'JPD_meV': pd, 'L_cells': length, 'distance_a': p['distance_a'],
                                   'energy_minus_reference_meV': p['energy_meV'] - ref['energy_meV'],
                                   'image_model_meV': float(model), 'boundary': 'fixed phase' if pd > 0 else 'free phase',
                                   'windings': p['windings']})
                print('J_PD=%.3f L=%d d=%6.2f  dE %+.5f  image %+.5f  windings %s' % (
                    pd, length, p['distance_a'], p['energy_meV'] - ref['energy_meV'], model, p['windings']))
    for b in branches:
        print('branches', b)

    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Classical Y state at 0.2 T, J = 0.075 meV, J_z = 0.125 meV, S = 1/2, J_Gamma = 0; full-sphere '
                 'relaxations and classical harmonic fluctuations; no quantum angular potential, no anharmonic or '
                 'quantum free energy. Lengths in bond lengths a.',
        'conventions': 'Three forward walls on a torus of L x W magnetic cells (axis 0 of nbcp_y_density_wall). '
                       'The pair sits in the middle domain; distance from the centre of the wall between layers '
                       'L/3 - 1 and L/3; order +1 puts the +1 core first along the wall. Cores held by a ring '
                       '1.8 < r < 3.2 with penalty lambda/2 |n_perp - target|^2 and a free rotation per core.',
        'hessian_check': check, 'wall_free_energy': free_energy, 'wall_free_energy_summary': summary_fe,
        'middle_domain_stiffness': {str(pd): {'orbit_angle_rad': v[0], 'match_error': v[1],
                                              'rho_vv_vu_uv_uu_meV': v[2].ravel().tolist()} for pd, v in stiffness.items()},
        'pairs': pairs, 'branches': branches, 'image_comparison': comparison,
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__), ROOT / 'examples/nbcp_y_wall_junction.py', ROOT / 'examples/nbcp_y_density_wall.py',
                           ROOT / 'examples/nbcp_y_soc_conditions.py', ROOT / 'examples/nbcp_y_stiffness.py']},
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'vortex-wall-check.json').write_text(json.dumps(record, indent=1))

    assert max(r['relative_error'] for r in check) < 1e-6
    for r in free_energy:
        assert all(v['lowest_eigenvalue_wall'] > 0 for v in r['Nk'].values())   # straight walls are stable to ripples
    assert summary_fe['0.0']['f1_extrapolated_per_a'] < 0 < summary_fe['0.01']['f1_extrapolated_per_a']
    core = [p for p in pairs if p['L_cells'] == 96 and p['hold_lambda'] == 0.05 and 6 < p['distance_a'] < 18]
    assert all(p['windings'] == [1.0, -1.0] for p in core)
    assert not [b for b in branches if b['L_cells'] == 96 and 6 < b['distance_a'] < 18]   # one branch away from contact and centre
    fixed = {round(c['distance_a'], 2): c for c in comparison if c['JPD_meV'] == 0.010 and c['L_cells'] == 96}
    free = {round(c['distance_a'], 2): c for c in comparison if c['JPD_meV'] == 0.0 and c['L_cells'] == 96}
    assert fixed[7.75]['energy_minus_reference_meV'] > 0 > free[7.75]['energy_minus_reference_meV']   # repelled / attracted


if __name__ == '__main__':
    main()
