"""Classical vortex pairs of the NBCP Y state at 0.2 T.

A vortex (m = +1) and an antivortex (m = -1) are held at separation d on an
L x L torus. The spins within 1.6 bond lengths of each core are fixed to the
relaxed core of an isolated vortex (`nbcp_y_vortex_core.Cluster`, R = 32)
whose far-field angle matches the pair; all other spins are relaxed on the
sphere. On a torus every total-derivative gradient term sums to zero, so the
pair energy carries no boundary term. At J_PD != 0 the far-field angle phi0
is not fixed by the cores alone, because the classical energy of uniform
states does not depend on phi0; a quadratic penalty holds the mean in-plane
angle of the spins farther than L/4 from the pair at phi0.

Measured, per pair and against the uniform state with the same fixed spins:

    E(d)       = 2 mu + 2 kappa ln d + c (d/L)^2 + a/d^2,
    Dlogdet(d) - b6 C6(d) = g0 + g1 ln d + ...,

where Dlogdet is the log-determinant difference of the tangent-space Hessians
of the relaxed spins. The classical harmonic free energy of the pair is
E + (T/2) Dlogdet, so g1/4 = d kappa/dT is the harmonic thermal change of the
log coefficient. At J_PD != 0 the log-determinant of a uniform state depends
on its angle, g(phi) = g0 + b6 cos(6 phi) per spin (the harmonic thermal
clock term). For the dipolar far field of a pair, sum_i g(phi_i) grows like
d^2 ln(L/d) and would bias g1; it belongs to the clock term of the phase
theory, not to the stiffness, so the local sum C6 = sum_i [cos 6 phi_i]
(relative to the uniform state) is removed with b6 from the LSWT bands.
d is in bond lengths and energies in meV.

Supporting checks:
- the continuum log coefficient kappa_m(alpha) of a vortex with winding m and
  far-field angle alpha, from the relaxed stiffness rho(phi) of
  `nbcp_y_soc_conditions.reduction`, minimised over angular distortions;
- the harmonic stiffness change under a uniform twist on hexagonal clusters
  at J_PD = 0, compared with g1;
- the lattice (Peierls-Nabarro) dependence of the vortex energy on the core
  position, from clusters whose fixed rings follow the core.

Classical only: no zero-point core energy.

Run from the repository root:
    python examples/nbcp_y_vortex_pair.py
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
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from examples.nbcp_y_soc_conditions import background, reduction
from examples.nbcp_y_stiffness import AREA, DELTAS, S
from examples.nbcp_y_vortex_core import A1, A2, CORE, Cluster

OUT = ROOT / 'data-space/verification/261002-y-vortex-pair'
LATTICE = np.column_stack([A1, A2])
BASIS = np.linalg.inv(LATTICE)
PAIR_STEP = A1 + A2                  # a superlattice vector: keeps the colours
PIN_RADIUS = 1.6
KS = [4, 6, 8, 10, 13, 16, 20]
JPD = 0.010
# Far-field angles phi0 for a pair along PAIR_STEP (30 degrees). The +1 core
# then has alpha = phi0 - 210 degrees: 120 degrees is the soft orientation,
# 30 degrees the stiff one, the others lie between.
OFFSETS = {'soft': 2 * np.pi / 3, 'stiff': np.pi / 6, 'mid_0': 0.0, 'mid_90': np.pi / 2}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Lattices
# --------------------------------------------------------------------------

class Torus(Cluster):
    """L x L periodic Y state with the bond convention of `Cluster`."""

    def __init__(self, pd, length):
        assert length % 6 == 0
        bg = background(pd, 0.0, 0.)
        self.h, self.normals, exchanges = bg[0], bg[2], bg[4]
        cells = np.array([(i, j) for i in range(length) for j in range(length)])
        self.L = length
        self.r = cells @ LATTICE.T
        self.color = (2 - cells[:, 0] - 2 * cells[:, 1]) % 3
        self.free = np.ones(len(cells), bool)
        self.cvec = None
        bi, bj, bJ = [], [], []
        for k in range(len(cells)):
            for d, exchange in zip(DELTAS, exchanges):
                n = np.rint(BASIS @ (self.r[k] - d)).astype(int) % length
                other = n[0] * length + n[1]
                assert self.color[other] == (self.color[k] + 1) % 3
                bi.append(k), bj.append(other), bJ.append(exchange)
        self.bi, self.bj, self.bJ = np.array(bi), np.array(bj), np.array(bJ)
        self.centre = (length // 2) * (A1 + A2)

    def minimum_image(self, point):
        frac = (self.r - point) @ BASIS.T
        frac -= self.L * np.round(frac / self.L)
        return frac @ LATTICE.T

    def hold_far_angle(self, phi0, lam=1.0):
        """Penalty (lam/2) c^2 on c = sum_far (z x e(phi0)) . n over A and B sites."""
        far = (np.linalg.norm(self.minimum_image(self.centre), axis=1) > self.L / 4) & (self.color != 2)
        e = self.normals[self.color].copy()
        e[:, 2] = 0
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-300)
        c, s = np.cos(phi0), np.sin(phi0)
        e = np.column_stack([c * e[:, 0] - s * e[:, 1], s * e[:, 0] + c * e[:, 1], np.zeros(len(e))])
        self.cvec = np.where(far[:, None], np.cross([0., 0., 1.], e), 0.)
        self.lam = lam

    def energy_gradient(self, spins):
        energy, grad = super().energy_gradient(spins)
        if self.cvec is None:
            return energy, grad
        c = float(np.sum(self.cvec * spins))
        return energy + 0.5 * self.lam * c * c, grad + self.lam * c * self.cvec

    def cos6_sum(self, spins, mask):
        """(3/2) sum of cos(6 phi_i) over free A and B sites: the local-angle sum
        per spin of a smooth texture, phi_i measured from the phi = 0 state."""
        use = mask & (self.color != 2)
        base = self.normals[self.color[use]]
        a = np.arctan2(spins[use, 1], spins[use, 0]) - np.arctan2(base[:, 1], base[:, 0])
        return 1.5 * float(np.sum(np.cos(6 * a)))

    def far_angle(self, spins):
        rel = self.minimum_image(self.centre)
        far = (np.linalg.norm(rel, axis=1) > self.L / 4) & (self.color != 2)
        base = self.normals[self.color]
        a = np.arctan2(spins[far, 1], spins[far, 0]) - np.arctan2(base[far, 1], base[far, 0])
        return float(np.angle(np.mean(np.exp(1j * a))))


def rotated(cluster, angle):
    base = cluster.normals[cluster.color]
    c, s = np.cos(angle), np.sin(angle)
    return np.column_stack([c * base[:, 0] - s * base[:, 1], s * base[:, 0] + c * base[:, 1], base[:, 2]])


def with_free(cluster, mask, function):
    old = cluster.free.copy()
    cluster.free = mask
    try:
        return function()
    finally:
        cluster.free = old


# --------------------------------------------------------------------------
# Harmonic fluctuations
# --------------------------------------------------------------------------

def tangent(spins):
    ref = np.where(np.abs(spins[:, 2:3]) < 0.9, [[0., 0., 1.]], [[1., 0., 0.]])
    e1 = np.cross(spins, ref)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return e1, np.cross(spins, e1)


def hessian(cluster, spins):
    """Tangent-space Hessian of the exchange and Zeeman energy over free spins."""
    free = np.where(cluster.free)[0]
    index = -np.ones(len(spins), int)
    index[free] = np.arange(len(free))
    frames = np.stack(tangent(spins), 1)
    _, grad = Cluster.energy_gradient(cluster, spins)
    diagonal = -np.sum(spins * grad, 1)
    keep = (index[cluster.bi] >= 0) & (index[cluster.bj] >= 0)
    bi, bj = cluster.bi[keep], cluster.bj[keep]
    block = S * S * np.einsum('bax,bxy,bcy->bac', frames[bi], cluster.bJ[keep], frames[bj])
    rows, cols, vals = [], [], []
    for a in range(2):
        for c in range(2):
            rows += [2 * index[bi] + a, 2 * index[bj] + c]
            cols += [2 * index[bj] + c, 2 * index[bi] + a]
            vals += [block[:, a, c], block[:, a, c]]
        rows.append(2 * np.arange(len(free)) + a)
        cols.append(2 * np.arange(len(free)) + a)
        vals.append(diagonal[free])
    size = 2 * len(free)
    return sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(size, size)), frames[free]


def log_det(cluster, spins):
    """log|det| of the Hessian, including the far-angle penalty as a rank-one term."""
    matrix, frames = hessian(cluster, spins)
    lu = splu(matrix, permc_spec='MMD_AT_PLUS_A')
    diagonal = lu.U.diagonal()
    value = float(np.sum(np.log(np.abs(diagonal))))
    if getattr(cluster, 'cvec', None) is not None:
        v = np.einsum('iax,ix->ia', frames, cluster.cvec[cluster.free]).ravel()
        value += float(np.log(abs(1 + cluster.lam * v @ lu.solve(v))))
    return value


# --------------------------------------------------------------------------
# Pairs
# --------------------------------------------------------------------------

_ISOLATED = {}


def isolated_core(pd, winding, alpha, core):
    key = (pd, winding, round(alpha % (2 * np.pi), 10), tuple(np.round(core, 10)))
    if key not in _ISOLATED:
        cluster = Cluster(pd, 32)
        _ISOLATED[key] = (cluster.r, cluster.relax(cluster.ansatz(winding, alpha, core))[1])
    return _ISOLATED[key]


def pair(torus, pd, k, offset, core=CORE, pin_radius=PIN_RADIUS):
    """Pair along PAIR_STEP; `core` is the core position within the unit cell
    (an up-triangle centre, or a site with pin_radius < 1 to hold one spin)."""
    r1 = torus.centre + core - (k // 2) * PAIR_STEP
    r2 = r1 + k * PAIR_STEP
    mid = (r1 + r2) / 2
    rel = torus.minimum_image(mid)
    angle = np.full(len(torus.r), offset)
    for winding, position in [(1, r1), (-1, r2)]:
        d = rel - (position - mid)
        angle += winding * np.arctan2(d[:, 1], d[:, 0])
    spins = rotated(torus, angle)
    uniform = rotated(torus, np.full(len(torus.r), offset))
    pinned = np.zeros(len(torus.r), bool)
    for winding, position, other in [(1, r1, r2), (-1, r2, r1)]:
        alpha = offset - winding * np.arctan2(*(position - other)[::-1])
        positions, core_spins = isolated_core(pd, winding, alpha, core)
        for i in np.where(np.linalg.norm(torus.r - position, axis=1) < pin_radius)[0]:
            target = torus.r[i] - (position - core)
            j = np.argmin(np.linalg.norm(positions - target, axis=1))
            assert np.linalg.norm(positions[j] - target) < 1e-9
            spins[i] = core_spins[j]
            pinned[i] = True
    mask = ~pinned
    e1, s1, t1 = with_free(torus, mask, lambda: torus.relax(spins))
    e0, s0, t0 = with_free(torus, mask, lambda: torus.relax(uniform))
    g1 = with_free(torus, mask, lambda: log_det(torus, s1))
    g0 = with_free(torus, mask, lambda: log_det(torus, s0))
    return {'k': k, 'd': float(k * np.linalg.norm(PAIR_STEP)), 'E_meV': e1 - e0, 'dlogdet': g1 - g0,
            'cos6_sum': torus.cos6_sum(s1, mask) - torus.cos6_sum(s0, mask),
            'far_angle_rad': torus.far_angle(s1), 'max_torque_meV': max(t0, t1)}


SINGLE_SPIN_CORE = -A1               # a colour-0 site: its spin points along +z in the core


def series(job):
    pd, length, name, offset, hold = job
    start = time.monotonic()
    torus = Torus(pd, length)
    if hold:
        torus.hold_far_angle(offset)
    if name == 'single_spin':
        rows = [pair(torus, pd, k, offset, SINGLE_SPIN_CORE, 0.1) for k in KS if k >= 8]
    else:
        rows = [pair(torus, pd, k, offset) for k in KS]
    return {'JPD_meV': pd, 'L': length, 'orientation': name, 'phi0_rad': offset, 'far_angle_held': hold,
            'rows': rows, 'seconds': time.monotonic() - start}


def fit(rows, length, kappa=None, b6=0.0):
    """Fit E and Dlogdet; b6 removes the local sixfold term b6 * cos6_sum from Dlogdet."""
    d = np.array([r['d'] for r in rows])
    energy = np.array([r['E_meV'] for r in rows])
    logdet = np.array([r['dlogdet'] - b6 * r['cos6_sum'] for r in rows])
    corrections = [(d / length) ** 2, 1 / d ** 2]
    out = {}
    if kappa is not None:
        x = np.column_stack([np.ones_like(d)] + corrections)
        coefficients, *_ = np.linalg.lstsq(x, energy - 2 * kappa * np.log(d), rcond=None)
        out['two_mu_at_model_kappa_meV'] = float(coefficients[0])
    x = np.column_stack([np.ones_like(d), 2 * np.log(d)] + corrections)
    coefficients, *_ = np.linalg.lstsq(x, energy, rcond=None)
    out['two_mu_meV'], out['kappa_meV'] = float(coefficients[0]), float(coefficients[1])
    out['energy_fit_residual_meV'] = float(np.max(np.abs(x @ coefficients - energy)))
    x = np.column_stack([np.ones_like(d), np.log(d)] + corrections)
    coefficients, *_ = np.linalg.lstsq(x, logdet, rcond=None)
    out['logdet_constant'], out['logdet_slope'] = float(coefficients[0]), float(coefficients[1])
    out['dkappa_dT'] = float(coefficients[1] / 4)
    out['logdet_fit_residual'] = float(np.max(np.abs(x @ coefficients - logdet)))
    return out


# --------------------------------------------------------------------------
# Continuum and lattice checks
# --------------------------------------------------------------------------

def continuum_kappa(pd, windings=(1, -1), alphas=np.arange(12) * np.pi / 12, modes=8):
    """min_g (1/2) int dtheta (m + g')^2 e_theta . rho(m theta + alpha + g) . e_theta."""
    phis = np.arange(24) * np.pi / 12
    rho = np.array([reduction(background(pd, 0.0, p))[4] / (3 * AREA) for p in phis])
    coefficients = np.fft.rfft(rho, axis=0) / len(phis)
    theta = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    e_theta = np.column_stack([-np.sin(theta), np.cos(theta)])
    harmonics = np.arange(1, modes + 1)
    cos, sin = np.cos(np.outer(theta, harmonics)), np.sin(np.outer(theta, harmonics))

    def rho_of(phi):
        value = np.real(coefficients[0])[None] * np.ones((len(phi), 1, 1))
        for n in range(1, len(coefficients) - 1):
            value = value + 2 * np.real(coefficients[n][None] * np.exp(1j * n * phi)[:, None, None])
        return value

    def kappa(winding, alpha):
        def energy(x):
            a, b = x[:modes], x[modes:]
            g = cos @ a + sin @ b
            dg = (-sin * harmonics) @ a + (cos * harmonics) @ b
            local = np.einsum('ti,tij,tj->t', e_theta, rho_of(winding * theta + alpha + g), e_theta)
            return np.pi * np.mean((winding + dg) ** 2 * local)
        return float(minimize(energy, np.zeros(2 * modes), method='BFGS').fun)

    return {str(m): [kappa(m, a) for a in alphas] for m in windings}, alphas.tolist()


def logdet_cos6(meshes=(24, 36), angles=36):
    """b6 of the uniform-state log-determinant per spin, g(phi) = g0 + b6 cos(6 phi).

    For classical spins ln det H = 2 sum_modes ln eps + const, so per spin
    g = (2/3) <sum_bands ln eps_k>; the LSWT bands are those of
    `nbcp_angular_thermal_split` on the same Y state."""
    from examples.nbcp_angular_thermal_split import JPD as SPLIT_JPD, bands
    from examples.pseudo_goldstone_comparison import mesh, phase_state
    assert SPLIT_JPD == JPD
    phi = np.arange(angles) * 2 * np.pi / angles
    values = []
    for n in meshes:
        eps, _ = bands(phase_state('Y'), mesh(n), phi)
        g = (2 / 3) * np.log(eps).sum(-1).mean(1)
        values.append(float(np.real(2 / angles * np.exp(-6j * phi) @ (g - g.mean()))))
    assert abs(values[0] / values[-1] - 1) < 0.01, values
    return values[-1]


def twist_softening(pd, radius, step=0.01):
    """-(1/2) d^2 logdet / d^2 E for a relaxed uniform twist on a hexagonal cluster."""
    cluster = Cluster(pd, radius)
    energies, logdets = [], []
    for q in (-step, 0., step):
        energy, spins, _ = cluster.relax(rotated(cluster, q * cluster.r[:, 0]))
        energies.append(energy)
        logdets.append(log_det(cluster, spins))
    second = lambda v: (v[0] - 2 * v[1] + v[2]) / step ** 2
    return -0.5 * second(logdets) / second(energies)


def core_position_dependence(pd, radii=(8, 12, 16, 24)):
    """Vortex energy for cores at different points, fixed rings following the core.

    A small seeded perturbation keeps the relaxation off symmetric stationary
    points of the core structure."""
    points = {'up_triangle': CORE, 'down_triangle': np.array([1., np.sqrt(3) / 3]),
              'bond_centre': np.array([.75, np.sqrt(3) / 4]), 'site_colour_1': np.array([1., 0.]),
              'site_colour_2': np.array([0., 0.])}
    out = {}
    for radius in radii:
        cluster = Cluster(pd, radius)
        noise = 1e-3 * np.random.default_rng(radius).normal(size=(len(cluster.r), 3))

        def relaxed(spins):
            spins = spins + noise
            return cluster.relax(spins / np.linalg.norm(spins, axis=1, keepdims=True))[0]

        reference = relaxed(cluster.ansatz(0, 0.))
        energies = {name: relaxed(cluster.ansatz(1, 0., p)) - reference for name, p in points.items()}
        out[str(radius)] = {name: e - energies['up_triangle'] for name, e in energies.items()}
    return out


def main():
    start = time.monotonic()
    jobs = [(0.0, length, name, 0.0, False) for length in (96, 144) for name in ('isotropic', 'single_spin')]
    jobs += [(JPD, 96, name, offset, True) for name, offset in OFFSETS.items()]
    jobs += [(JPD, 144, name, OFFSETS[name], True) for name in ['soft', 'stiff']]
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        results = list(pool.map(series, jobs))

    kappa_model = {pd: continuum_kappa(pd) for pd in [0.0, JPD]}
    b6 = {0.0: 0.0, JPD: logdet_cos6()}
    cases = []
    for pd in [0.0, JPD]:
        windings, alphas = kappa_model[pd]
        for name in sorted({r['orientation'] for r in results if r['JPD_meV'] == pd}):
            runs = [r for r in results if r['JPD_meV'] == pd and r['orientation'] == name]
            offset = runs[0]['phi0_rad']
            alpha = (offset - np.arctan2(PAIR_STEP[1], PAIR_STEP[0]) - np.pi) % np.pi
            kappa_plus = float(np.interp(alpha, alphas + [np.pi], windings['1'] + windings['1'][:1]))
            model = 0.5 * (kappa_plus + float(np.mean(windings['-1'])))
            rows = [dict(row, L=run['L']) for run in runs for row in run['rows']]
            d = np.array([r['d'] for r in rows])
            length = np.array([r['L'] for r in rows])
            fitted = fit(rows, length, model, b6[pd])
            raw = fit(rows, length, model)
            fitted['dkappa_dT_without_sixfold_removal'] = raw['dkappa_dT']
            fitted['logdet_fit_residual_without_sixfold_removal'] = raw['logdet_fit_residual']
            if len(runs) > 1:
                fitted['dkappa_dT_by_L'] = {str(run['L']): fit(run['rows'], run['L'], model, b6[pd])['dkappa_dT']
                                            for run in runs}
                fitted['kappa_by_L_meV'] = {str(run['L']): fit(run['rows'], run['L'], model)['kappa_meV'] for run in runs}
            cases.append({'JPD_meV': pd, 'orientation': name, 'phi0_rad': offset,
                          'alpha_plus_rad': float(alpha), 'kappa_pair_model_meV': model, **fitted,
                          'max_far_angle_shift_rad': float(max(abs(np.angle(np.exp(1j * (r['far_angle_rad'] - offset))))
                                                               for r in rows))})
            print('J_PD=%.3f %-7s alpha+=%5.1f deg  kappa fit %.5f model %.5f  2mu %.5f (at model kappa %.5f)'
                  '  dkappa/dT %.3f (without sixfold removal %.3f)' % (
                      pd, name, np.degrees(alpha), fitted['kappa_meV'], model, fitted['two_mu_meV'],
                      fitted['two_mu_at_model_kappa_meV'], fitted['dkappa_dT'], raw['dkappa_dT']))

    # Held-core entropy s = -g0/4 at a common log slope: two ways of holding the core.
    slope = next(c for c in cases if c['orientation'] == 'isotropic')['logdet_slope']
    entropy = {}
    for name in ('isotropic', 'single_spin'):
        rows = [dict(row, L=run['L']) for run in results if run['JPD_meV'] == 0.0
                and run['orientation'] == name for row in run['rows']]
        d = np.array([r['d'] for r in rows])
        length = np.array([r['L'] for r in rows])
        x = np.column_stack([np.ones_like(d), (d / length) ** 2, 1 / d ** 2])
        g = np.array([r['dlogdet'] for r in rows]) - slope * np.log(d)
        entropy[name] = float(-np.linalg.lstsq(x, g, rcond=None)[0][0] / 4)
    print('held-core entropy per core (12 spins held, one spin held):', entropy)

    twist = {str(radius): twist_softening(0.0, radius) for radius in (48, 64, 96)}
    inverse = np.array([1 / int(r) for r in twist])
    slope, intercept = np.polyfit(inverse[1:], np.array(list(twist.values()))[1:], 1)
    twist_rate = float(intercept)
    rho0 = reduction(background(0.0, 0.0, 0.))[4] / (3 * AREA)
    kappa0 = float(np.pi * np.sqrt(np.linalg.det(rho0)))
    position = core_position_dependence(0.0)
    print('twist softening rate', twist, 'extrapolated', twist_rate, '-> dkappa/dT', -twist_rate * kappa0)
    print('core position dependence', position)

    radii = np.array([int(r) for r in position])
    lattice_term = {}
    for name in position[str(radii[0])]:
        values = np.array([position[str(r)][name] for r in radii])
        (_, lattice_term[name]), *_ = np.linalg.lstsq(np.column_stack([1 / radii ** 2, np.ones(len(radii))]),
                                                       values, rcond=None)
    print('R-independent part of the core-position dependence', lattice_term)

    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Classical vortex-antivortex pairs in the Y state, 0.2 T, J = 0.075 meV, J_z = 0.125 meV, '
                 'S = 1/2, J_Gamma = 0, on L x L tori; spins within %.1f bond lengths of each core fixed to '
                 'the relaxed isolated core; separation d along a1 + a2 in bond lengths; energies in meV '
                 'relative to the uniform state with the same fixed spins.' % PIN_RADIUS,
        'fit': 'E = 2 mu + 2 kappa ln d + c (d/L)^2 + a/d^2; Dlogdet - b6 cos6_sum = g0 + g1 ln d + c\' (d/L)^2 '
               '+ a\'/d^2; dkappa_dT = g1/4 is the classical harmonic change of the pair log coefficient. '
               'b6 cos6_sum is the local sixfold (thermal clock) part of Dlogdet, which belongs to the '
               'clock term of the phase theory and grows like d^2 ln(L/d) for a dipolar far field.',
        'logdet_cos6_b6_per_spin': b6[JPD],
        'series': results, 'cases': cases,
        'continuum_kappa_meV': {str(pd): {'alpha_rad': v[1], 'by_winding': v[0]} for pd, v in kappa_model.items()},
        'twist_softening_per_meV': {'by_radius': twist, 'extrapolated_1_over_R': twist_rate,
                                    'kappa_meV': kappa0},
        'held_core_entropy_per_core': {'held_spins_12': entropy['isotropic'], 'held_spins_1': entropy['single_spin'],
                                       'note': 'J_PD = 0, -g0/4 at the common log slope of the 12-spin series'},
        'core_position_dependence_meV': position,
        'core_position_R_independent_meV': {k: float(v) for k, v in lattice_term.items()},
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__), ROOT / 'examples/nbcp_y_vortex_core.py',
                           ROOT / 'examples/nbcp_y_soc_conditions.py', ROOT / 'examples/nbcp_y_stiffness.py']},
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'vortex-pair-check.json').write_text(json.dumps(record, indent=1))

    iso, = [c for c in cases if c['orientation'] == 'isotropic']
    soft, = [c for c in cases if c['orientation'] == 'soft']
    stiff, = [c for c in cases if c['orientation'] == 'stiff']
    assert abs(iso['kappa_meV'] / kappa0 - 1) < 0.03
    assert abs(iso['dkappa_dT'] / (-twist_rate * kappa0) - 1) < 0.05
    for case in (soft, stiff):
        assert abs(case['kappa_meV'] / case['kappa_pair_model_meV'] - 1) < 0.05, case
    for case in (soft, stiff):
        rates = list(case['dkappa_dT_by_L'].values())
        assert abs(rates[0] - rates[1]) < 0.1, case          # L-independent once the sixfold part is removed
        assert case['logdet_fit_residual'] < 0.5 * case['logdet_fit_residual_without_sixfold_removal'], case
    assert max(abs(v) for v in lattice_term.values()) < 5e-6
    assert min(row['E_meV'] for run in results for row in run['rows']) > 0.03   # no pair annihilated
    single, = [c for c in cases if c['orientation'] == 'single_spin']
    assert abs(single['two_mu_at_model_kappa_meV'] - iso['two_mu_at_model_kappa_meV']) < 5e-4


if __name__ == '__main__':
    main()
