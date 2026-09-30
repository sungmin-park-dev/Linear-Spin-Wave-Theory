"""Verify stage 5d: adaptive k integration of the magnon thermal Hall conductivity.

1. Haldane magnons with narrow gaps (D = 0.01, 0.003, 0.001, h = 0.3):
   adaptive result against uniform two-band references (N = 256 ... 2048),
   error estimate against the actual error, points against the uniform mesh.
2. A smooth case (D = 0.2), where the uniform midpoint rule on the periodic
   zone is already exponentially accurate, and Kitaev [111] (h = 2).
3. NBCP Y (0.2 T) and V (1.4 T) with J_PD or J_Gamma = 0.01 meV: adaptive
   result with the default settings against the uniform 192 x 192 values of
   the stage-5b record; the integrand near the accidental zero mode at Gamma.
4. Coplanar zero: triangular 120 degrees stays zero.

Usage
-----
    python examples/lswt_stage5d_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from model import nbcp
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods.classical import refine_classical
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import state_120, triangular_heisenberg
from spintoolkit.observables.berry import AdaptiveIntegration, _integrand, thermal_hall
from spintoolkit.system.conditions import ExternalConditions
from tests.test_methods.test_berry import haldane, haldane_reference, kitaev

SCAN = ROOT / 'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json'
STAGE5B = ROOT / 'docs/development/verification/stage5b-thermal-hall-2026-09-30.json'


def adaptive(result, ts, gapless=None, **kwargs):
    start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hall = thermal_hall(result, ts, gapless=gapless, integration=AdaptiveIntegration(**kwargs))
    info = dict(hall.integration)
    info['seconds'] = round(time.time() - start, 1)
    info['warnings'] = [str(w.message)[:90] for w in caught]
    return hall.kappa_over_t, info


def haldane_cases():
    rows, ts = [], [0.05, 0.3]
    for D in (0.2, 0.01, 0.003, 0.001):
        result = haldane(D, (12, 12))[2]
        references = {n: haldane_reference_chunked(result, n, ts).tolist() for n in (256, 512, 1024, 2048)}
        kappa, info = adaptive(result, ts)
        reference = np.array(references[2048])
        rows.append({'D': D, 'temperatures': ts, 'uniform_reference': references,
                     'uniform_12': thermal_hall(result, ts).kappa_over_t.tolist(),
                     'adaptive': kappa.tolist(), 'actual_error': np.abs(kappa - reference).tolist(),
                     'error_estimate': info['error_estimate'], 'points': info['points'],
                     'converged': info['converged'], 'max_depth_reached': info['max_depth_reached'],
                     'seconds': info['seconds']})
    return rows


def haldane_reference_chunked(result, n, ts, chunk=512):
    """The test reference on an n x n mesh, evaluated in row blocks to bound memory."""
    if n <= chunk:
        return haldane_reference(result, n, ts)
    reciprocal = 2 * np.pi * np.linalg.inv(result.magnetic_lattice).T
    total = np.zeros(len(ts))
    grid = (np.arange(n) + 0.5) / n
    for start in range(0, n, chunk // 4):
        rows = grid[start:start + chunk // 4]
        fractions = np.array(np.meshgrid(rows, grid, indexing='ij')).reshape(2, -1).T
        total += _two_band_sum(result, fractions @ reciprocal, ts)
    return -total / n ** 2 / abs(np.linalg.det(result.magnetic_lattice))


def _two_band_sum(result, k, ts):
    from spintoolkit.observables.topology import c2_weight
    from tests.test_methods.test_berry import PAULI
    h = result.hamiltonian_at(k)[:, :2, :2]
    dx, dy = result.hamiltonian_derivatives_at(k)

    def parts(m):
        return (np.real(np.trace(m, axis1=1, axis2=2)) / 2,
                np.stack([np.real(np.einsum('kij,ji->k', m, p)) / 2 for p in PAULI], axis=1))

    d0, d = parts(h)
    ddx, ddy = parts(dx[:, :2, :2])[1], parts(dy[:, :2, :2])[1]
    norm = np.linalg.norm(d, axis=1)
    lower = np.einsum('ki,ki->k', d, np.cross(ddx, ddy)) / (2 * norm ** 3)
    return np.array([np.sum((c2_weight(d0 - norm, t) - c2_weight(d0 + norm, t)) * lower) for t in ts])


def kitaev_case():
    ts = [0.05, 0.3]
    uniform = {n: thermal_hall(kitaev(2.0, (n, n)), ts).kappa_over_t.tolist() for n in (48, 96, 192)}
    kappa, info = adaptive(kitaev(2.0, (12, 12)), ts)
    return {'h': 2.0, 'uniform': uniform, 'adaptive': kappa.tolist(),
            'error_estimate': info['error_estimate'], 'points': info['points'],
            'converged': info['converged'], 'seconds': info['seconds']}


def nbcp_cases():
    scan = json.loads(SCAN.read_text())
    uniform = {}
    for row in json.loads(STAGE5B.read_text())['report']['nbcp_convergence']:
        uniform.setdefault((row['phase'], json.dumps(row['couplings'])), {})[row['mesh']] = row['kappa_over_t']
    rows, ts = [], [0.01, 0.02, 0.05, 0.1]
    for phase, field, extra in (('Y', 0.2, {'JPD': 0.01}), ('Y', 0.2, {'JGamma': 0.01}),
                                ('V', 1.4, {'JPD': 0.01}), ('V', 1.4, {'JGamma': 0.01})):
        model = nbcp.build_model({'Jxy': 0.075, 'Jz': 0.125, **extra})
        theta = np.array(scan['states'][phase]['theta'])
        conditions = ExternalConditions(field=(0, 0, 4.645 * MU_B_MEV_PER_T * field))
        state = refine_classical(model, nbcp.candidate_state(
            model, 'three_msl', np.column_stack([theta, np.zeros(3)]).ravel()), conditions)
        result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(24, 24)))
        kappa, info = adaptive(result, ts, gapless=True)
        near_gamma = [{'fraction': q, 'integrand_t0.05': float(_integrand(
            result, np.array([[q, 0.37 * q]]), [0.05], 1e-8)[0][0, 0])} for q in (1e-2, 1e-3, 1e-4)]
        rows.append({'phase': phase, 'field_T': field, 'couplings': extra, 'temperatures_meV': ts,
                     'adaptive': kappa.tolist(), 'error_estimate': info['error_estimate'],
                     'points': info['points'], 'converged': info['converged'],
                     'stop_reason': info['stop_reason'], 'max_depth_reached': info['max_depth_reached'],
                     'largest_error_cells': info['largest_error_cells'],
                     'smallest_particle_gap_meV': info['smallest_particle_gap'],
                     'uniform_stage5b': uniform[(phase, json.dumps(extra))],
                     'integrand_toward_gamma': near_gamma, 'seconds': info['seconds']})
    return rows


def coplanar_zero():
    triangle = triangular_heisenberg()
    result = solve_lswt(triangle, state_120(triangle), None, settings=LSWTSettings(mesh=(12, 12)))
    kappa, info = adaptive(result, [0.1, 1.0], max_points=20_000)
    return {'kappa_over_t': kappa.tolist(), 'points': info['points'], 'converged': info['converged']}


def main():
    report = {'haldane': haldane_cases(), 'kitaev': kitaev_case(), 'nbcp': nbcp_cases(),
              'coplanar_zero': coplanar_zero()}
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()
