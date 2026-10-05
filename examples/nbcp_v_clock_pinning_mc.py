"""Finite-temperature sequence of the V phase-only model with sixfold and threefold pinning.

Near the PD axis the V angular potential per spin is
    v(phi) = -lambda6 cos(6 phi) - lambda3 sin(3 phi),   r = lambda3 / (4 lambda6),
with the threefold minima at maxima of the sixfold term (NBCP note, V phase).
For r < 1 there are six degenerate minima in three pairs; at r = 1 the pairs
merge at the three threefold minima.

The phase-only model on a square lattice of spacing a0 (the matching length,
a0 = 1/Lambda) is
    E = -J sum_<ij> cos(phi_i - phi_j) - sum_i [h6 cos(6 phi_i) + h3 sin(3 phi_i)],
with J = sqrt(det rho) of V and h_n = lambda_n (a0^2 / a_spin) times the
Debye-Waller factor exp(-n^2 <phi^2>_{k > Lambda} / 2) of
`nbcp_v_matching_debye_waller.py`. lambda6 is the V amplitude at
J_PD = 0.010 meV; lambda3 = 4 r lambda6 (r = 1 near J_Gamma = 0.0022 meV).
Both are leading order in 1/S, lambda3 is taken at T = 0 and the lattice
vortex core is that of the XY model, not a matched V core.

Metropolis updates with replica exchange on L x L lattices, from a cold start; a hot
start on the largest lattice checks equilibration. Order parameters:
m1 = |<exp(i phi)>| (one of the three pairs, or one of six minima, is
selected), I = |<cos 3 phi>| (reflection within a pair; at r = 0 the
sublattice of the six minima), m6 = |<exp(6 i phi)>|, and the helicity
modulus Y. The effective exponent eta from m1 ~ L^(-eta/2) between L = 16
and 32 identifies an algebraic regime (1/9 < eta < 1/4 with Y > 2T/pi).

Run from the repository root (about 30 minutes on four cores):
    python examples/nbcp_v_clock_pinning_mc.py
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

import numpy as np

INPUT = ROOT / 'data-space/verification/261005-v-matching-debye-waller/v-matching-debye-waller.json'
OUT = ROOT / 'data-space/verification/261005-v-clock-pinning-mc'
A_SPIN = np.sqrt(3) / 2
RATIOS = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5]
SIZES = [16, 24, 32]
TEMPS = list(np.geomspace(0.0003, 0.004, 28))
SWEEPS, THERM = 12000, 4000


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def debye_waller_table(cutoffs):
    """Thermal D(T) of V above each cutoff on a 96 x 96 mesh (sixfold exponent)."""
    import examples.nbcp_v_matching_debye_waller as dw
    from examples.pseudo_goldstone_comparison import mesh
    dw.CUTOFFS = list(cutoffs)
    dw.TEMPERATURES = [0.0003, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004]
    result = dw.debye_waller('V', mesh(96))
    return {str(c): [[r['T_meV'], r[f'D_thermal_above_{c}']] for r in result['temperatures']] for c in cutoffs}


def interpolate(table, T):
    t = np.array([x[0] for x in table])
    d = np.array([max(x[1], 1e-12) for x in table])
    return float(np.exp(np.interp(np.log(T), np.log(t), np.log(d))))


def simulate(job):
    """Replica-exchange Metropolis for one (a0, r, L, scale, seed)."""
    a0, r, L, scale, seed, J, lam6, dtable = job
    rng = np.random.default_rng(seed)
    T = np.array(TEMPS)
    R = len(T)
    cell = a0 ** 2 / A_SPIN
    D = np.array([interpolate(dtable, t) for t in T])
    h6 = (scale * lam6 * cell * np.exp(-D))[:, None, None]
    h3 = (4 * r * scale * lam6 * cell * np.exp(-D / 4))[:, None, None]
    if seed == 0:
        # cold start: every replica in a global minimum of its single-site potential
        grid = np.linspace(0, 2 * np.pi, 7200, endpoint=False)
        start = [grid[np.argmin(-h6[k, 0, 0] * np.cos(6 * grid) - h3[k, 0, 0] * np.sin(3 * grid))] for k in range(R)]
        phi = np.broadcast_to(np.array(start)[:, None, None], (R, L, L)).copy()
    else:
        phi = rng.uniform(0, 2 * np.pi, (R, L, L))       # hot start, the equilibration check
    step = np.ones((R, 1, 1))
    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    masks = [(ii + jj) % 2 == c for c in (0, 1)]

    def site_energy(p, q):
        neighbours = (np.roll(p, 1, 1), np.roll(p, -1, 1), np.roll(p, 1, 2), np.roll(p, -1, 2))
        return -J * sum(np.cos(q - n) for n in neighbours) - h6 * np.cos(6 * q) - h3 * np.sin(3 * q)

    def parts(p):
        bonds = -(np.cos(p - np.roll(p, 1, -2)) + np.cos(p - np.roll(p, 1, -1))).sum((-2, -1))
        return bonds, -np.cos(6 * p).sum((-2, -1)), -np.sin(3 * p).sum((-2, -1))

    samples = {k: [] for k in ('E', 'm1', 'I', 'm6', 'Y')}
    accepted, tries = np.zeros(R), 0
    swaps, swap_tries = np.zeros(R - 1), np.zeros(R - 1)
    for sweep in range(SWEEPS):
        for mask in masks:
            new = phi + step * rng.uniform(-1, 1, phi.shape)
            dE = site_energy(phi, new) - site_energy(phi, phi)
            accept = mask & (rng.random(phi.shape) < np.exp(-np.clip(dE / T[:, None, None], -60, 60)))
            phi = np.where(accept, new, phi)
            accepted += accept.sum((1, 2)) / (L * L / 2)
            tries += 1
        if sweep < THERM and sweep % 50 == 49:
            step = np.clip(step * np.clip(accepted / tries / 0.4, 0.7, 1.3)[:, None, None], 0.02, np.pi)
            accepted[:], tries = 0, 0
        if sweep % 2 == 0:
            b, c6, c3 = parts(phi)
            for k in range(sweep % 4 // 2, R - 1, 2):
                # field strengths depend on T, so both configurations are evaluated at both temperatures
                e = lambda i, t: J * b[i] + h6[t, 0, 0] * c6[i] + h3[t, 0, 0] * c3[i]
                log_p = (e(k, k) - e(k + 1, k)) / T[k] + (e(k + 1, k + 1) - e(k, k + 1)) / T[k + 1]
                swap_tries[k] += 1
                if np.log(rng.random()) < log_p:
                    phi[[k, k + 1]] = phi[[k + 1, k]]
                    b[[k, k + 1]], c6[[k, k + 1]], c3[[k, k + 1]] = b[[k + 1, k]], c6[[k + 1, k]], c3[[k + 1, k]]
                    swaps[k] += 1
        if sweep >= THERM and sweep % 5 == 0:
            b, c6, c3 = parts(phi)
            samples['E'].append((J * b + h6[:, 0, 0] * c6 + h3[:, 0, 0] * c3) / (L * L))
            samples['m1'].append(np.abs(np.exp(1j * phi).mean((1, 2))))
            samples['I'].append(np.cos(3 * phi).mean((1, 2)))
            samples['m6'].append(np.abs(np.exp(6j * phi).mean((1, 2))))
            dx = phi - np.roll(phi, 1, 1)
            samples['Y'].append((J * np.cos(dx).sum((1, 2)) - J ** 2 / T * np.sin(dx).sum((1, 2)) ** 2) / (L * L))
    s = {k: np.array(v) for k, v in samples.items()}
    rows = []
    for k, t in enumerate(T):
        m1, I = s['m1'][:, k], s['I'][:, k]
        rows.append({'T_meV': float(t), 'K': J / float(t), 'h6_meV': float(h6[k, 0, 0]), 'h3_meV': float(h3[k, 0, 0]),
                     'E_per_site_meV': float(s['E'][:, k].mean()),
                     'm1': float(m1.mean()), 'U1': float(1 - np.mean(m1 ** 4) / (2 * np.mean(m1 ** 2) ** 2)),
                     'I': float(np.abs(I).mean()), 'UI': float(1 - np.mean(I ** 4) / (3 * np.mean(I ** 2) ** 2)),
                     'm6': float(s['m6'][:, k].mean()),
                     'helicity_over_2T_over_pi': float(s['Y'][:, k].mean() * np.pi / (2 * t))})
    return {'a0': a0, 'r': r, 'L': L, 'scale': scale, 'seed': seed, 'rows': rows,
            'swap_rates': (swaps / np.maximum(swap_tries, 1)).tolist()}


def analyse(runs, a0, scale):
    """Per r: effective eta from m1(16) and m1(32), the algebraic regime and the ordering temperatures."""
    out = []
    for r in RATIOS:
        by_size = {run['L']: run for run in runs if run['a0'] == a0 and run['r'] == r and run['scale'] == scale and run['seed'] == 0}
        if 16 not in by_size or 32 not in by_size:
            continue
        rows = []
        for small, large in zip(by_size[16]['rows'], by_size[32]['rows']):
            eta = -2 * np.log(large['m1'] / small['m1']) / np.log(2)
            rows.append({'T_meV': small['T_meV'], 'eta_eff': float(eta), 'helicity_over_2T_over_pi_L32': large['helicity_over_2T_over_pi'],
                         'm1_L32': large['m1'], 'I_L32': large['I'], 'UI_L16': small['UI'], 'UI_L32': large['UI']})
        algebraic = [x['T_meV'] for x in rows if 1 / 9 - 0.03 < x['eta_eff'] < 0.25 + 0.03 and x['helicity_over_2T_over_pi_L32'] > 1]
        ordered = [x['T_meV'] for x in rows if x['eta_eff'] < 1 / 9 - 0.03]
        ising = [x['T_meV'] for x in rows if x['UI_L32'] > 0.5 and x['UI_L16'] > 0.5]
        out.append({'r': r, 'J_Gamma_meV_at_JPD_0.010': None, 'rows': rows,
                    'algebraic_T_range_meV': [min(algebraic), max(algebraic)] if algebraic else None,
                    'pair_or_minimum_order_below_meV': max(ordered) if ordered else None,
                    'reflection_order_below_meV': max(ising) if ising else None})
    return out


def check(summary, seeds):
    """Conclusions recorded in the V research note."""
    base = {item['r']: item for item in summary['a0=10.0,scale=1.0']}
    assert base[0.0]['algebraic_T_range_meV'] is not None                     # window on the pure-PD axis
    assert all(base[r]['algebraic_T_range_meV'] is None for r in (0.05, 0.1, 0.25, 0.5, 1.0, 1.5))
    assert base[0.05]['reflection_order_below_meV'] < base[0.05]['pair_or_minimum_order_below_meV']
    assert base[1.0]['reflection_order_below_meV'] is None                    # three minima: no pair-internal choice
    for key in ('a0=5.0,scale=1.0', 'a0=10.0,scale=0.3'):
        varied = {item['r']: item for item in summary[key]}
        assert varied[0.0]['algebraic_T_range_meV'] is not None and varied[0.5]['algebraic_T_range_meV'] is None
    cold_hot = {(s['r'], s['scale']): s for s in seeds}
    assert cold_hot[(0.0, 1.0)]['max_m1_difference'] < 0.05                  # equilibrated where the window is claimed


def main():
    start = time.monotonic()
    inputs = json.loads(INPUT.read_text())
    J = inputs['v_matching']['sqrt_det_rho_meV']['mean']
    lam6 = inputs['v_matching']['a6_zero_meV_per_spin']
    dtables = debye_waller_table([0.1, 0.2])
    jobs = [(10.0, r, L, 1.0, 0, J, lam6, dtables['0.1']) for r in RATIOS for L in SIZES]
    jobs += [(5.0, r, L, 1.0, 0, J, lam6, dtables['0.2']) for r in (0.0, 0.5) for L in (16, 32)]       # matching length
    jobs += [(10.0, r, L, s, 0, J, lam6, dtables['0.1']) for s in (0.3, 3.0) for r in (0.0, 0.5) for L in (16, 32)]   # amplitude
    jobs += [(10.0, r, 32, 1.0, 1, J, lam6, dtables['0.1']) for r in (0.0, 0.5, 1.0)]                  # hot start
    jobs += [(10.0, r, 32, 3.0, 1, J, lam6, dtables['0.1']) for r in (0.0, 0.5)]
    jobs.sort(key=lambda j: -j[2])
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as pool:
        runs = list(pool.map(simulate, jobs))
    lam3_per_JGamma = 1.83 * 0.010           # lambda3 / J_Gamma at J_PD = 0.010 meV (V table), meV/spin per meV
    summary = {}
    for a0, scale in [(10.0, 1.0), (5.0, 1.0), (10.0, 0.3), (10.0, 3.0)]:
        result = analyse(runs, a0, scale)
        for item in result:
            item['J_Gamma_meV_at_JPD_0.010'] = 4 * item['r'] * lam6 / lam3_per_JGamma
        summary[f'a0={a0},scale={scale}'] = result
    seeds = []
    for r, scale in [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0), (0.0, 3.0), (0.5, 3.0)]:
        cold, hot = sorted([run for run in runs if run['a0'] == 10.0 and run['r'] == r and run['L'] == 32 and run['scale'] == scale],
                           key=lambda run: run['seed'])
        diff = [{'T_meV': x['T_meV'], 'm1': abs(x['m1'] - y['m1']), 'I': abs(x['I'] - y['I'])} for x, y in zip(cold['rows'], hot['rows'])]
        seeds.append({'r': r, 'scale': scale, 'max_m1_difference': max(d['m1'] for d in diff),
                      'max_I_difference': max(d['I'] for d in diff),
                      'temperatures_with_m1_difference_above_0.05': [d['T_meV'] for d in diff if d['m1'] > 0.05]})

    for key, result in summary.items():
        for item in result:
            print('%s r=%.2f (J_Gamma %.4f) algebraic %s, pair/minimum order below %s, reflection order below %s' % (
                key, item['r'], item['J_Gamma_meV_at_JPD_0.010'], item['algebraic_T_range_meV'],
                item['pair_or_minimum_order_below_meV'], item['reflection_order_below_meV']))
    print('seed check', seeds)

    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Phase-only V model at 1.4 T, J_PD = 0.010 meV, near the PD axis; inputs leading order in 1/S; lambda3 at '
                 'T = 0; XY lattice vortex core (not a matched V core). T in meV.',
        'model': 'E = -J sum cos(phi_i - phi_j) - sum [h6 cos 6 phi + h3 sin 3 phi]; J = sqrt(det rho_V); '
                 'h6 = scale lambda6 a0^2/a_spin exp(-D), h3 = 4 r scale lambda6 a0^2/a_spin exp(-D/4), '
                 'D = thermal Debye-Waller exponent above Lambda = 1/a0.',
        'J_meV': J, 'lambda6_meV_per_spin': lam6, 'debye_waller': dtables, 'temperatures': TEMPS,
        'sweeps': SWEEPS, 'thermalisation': THERM, 'summary': summary, 'seed_check': seeds, 'runs': runs,
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__), INPUT, ROOT / 'examples/nbcp_v_matching_debye_waller.py']},
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'v-clock-pinning-mc.json').write_text(json.dumps(record, indent=1))
    check(summary, seeds)


if __name__ == '__main__':
    main()
