"""Verify stage 5a: Berry curvature and Chern numbers of LSWT magnons (D29).

1. Sign anchor: honeycomb ferromagnet with DM (Haldane magnons); the particle
   block of H(k) against the Bloch matrix rebuilt from ED one-magnon states
   (4 x 4 torus), and the FHS Chern numbers of the ED Bloch states (5 x 5).
2. dH/dk against central differences; Kubo curvature against small-plaquette
   phases of the eigenvectors.
3. Chern numbers: Haldane (D = +-0.2, 0), Kitaev [111] polarized (h = 0.5, 1,
   2; pairing terms; reversed field), Kubo convergence to the FHS integers.
4. Null and undefined cases: triangular Heisenberg at h = 1, canted square
   antiferromagnet (bands cross between mesh points), Neel (degenerate), and
   D = 0 (Dirac points between mesh points: FHS alone returns a wrong integer
   on the 12 x 12 mesh; chern_numbers rejects it by the Kubo-FHS agreement).

Usage
-----
    python examples/lswt_stage5a_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from spintoolkit.methods.classical import refine_classical
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.observables.berry import berry_curvature, chern_numbers, chern_numbers_fhs
from spintoolkit.system.conditions import ExternalConditions
from tests.test_methods.test_berry import (
    ed_bloch_matrices, haldane, kitaev, lowest_particle, test_ed_bloch_states_alone_give_the_haldane_chern_numbers)


def sign_anchor():
    model, conditions, _ = haldane()
    k, h = ed_bloch_matrices(model, conditions, 4)
    result = solve_lswt(model, polarized_state(model), conditions, settings=LSWTSettings(k_points=k))
    block = result.hamiltonians[:, :2, :2]
    test_ed_bloch_states_alone_give_the_haldane_chern_numbers()     # asserts ED FHS = (+1, -1)
    return {'num_momenta': len(k), 'max_abs_ed_minus_lswt': float(np.max(np.abs(h - block))),
            'max_abs_conjugate_minus_lswt': float(np.max(np.abs(h.conj() - block))),
            'ed_fhs_chern_5x5': [1, -1]}


def pointwise():
    rows = []
    for name, result in (('haldane D=0.2', haldane()[2]), ('kitaev h=1', kitaev())):
        k = np.array([[0.31, -0.57], [1.2, 0.4]])
        dx, dy = result.hamiltonian_derivatives_at(k)
        step = 1e-6
        fd = max(float(np.max(np.abs((result.hamiltonian_at(k + e) - result.hamiltonian_at(k - e))
                                     / (2 * step) - d)) / np.max(np.abs(d)))
                 for d, e in ((dx, [step, 0]), (dy, [0, step])))
        curvature = berry_curvature(result)
        ns, delta = result.num_sites, 1e-4
        eta = np.r_[np.ones(ns), -np.ones(ns)]
        worst = 0.0
        for n in range(0, len(result.k_points), 7):
            corners = result.k_points[n] + delta * np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
            u = [lowest_particle(H) for H in result.hamiltonian_at(corners)]
            phase = np.angle(np.prod([np.vdot(u[i], eta * u[(i + 1) % 4]) for i in range(4)]))
            kubo = curvature.curvature[n, 0]
            worst = max(worst, abs(-phase / delta ** 2 - kubo) / max(1.0, abs(kubo)))
        rows.append({'case': name, 'derivative_relative_error': fd,
                     'plaquette_vs_kubo_relative_error_max': worst,
                     'points': len(range(0, len(result.k_points), 7)), 'plaquette_step': delta})
    return rows


def chern_table():
    rows = []
    for D in (0.2, -0.2, 0.0):
        for n in (12, 24):
            result = haldane(D, (n, n))[2]
            b = berry_curvature(result)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                accepted = chern_numbers(result, curvature=b)
            rows.append({'model': 'honeycomb FM + NNN DM', 'D': D, 'h': 0.3, 'mesh': n,
                         'kubo': b.chern_numbers().tolist(), 'fhs': chern_numbers_fhs(result).tolist(),
                         'accepted': [None if np.isnan(x) else x for x in accepted],
                         'max_abs_curvature': float(np.nanmax(np.abs(b.curvature))),
                         'min_level_spacing': float(np.min(b.level_spacing))})
    for h in (0.5, 1.0, 2.0):
        for n in (12, 24, 48, 96):
            t = time.time()
            result = kitaev(h, (n, n))
            b = berry_curvature(result)
            accepted = None
            if n <= 48:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    accepted = [None if np.isnan(x) else x for x in chern_numbers(result, curvature=b)]
            rows.append({'model': 'Kitaev K = -1, [111] polarized', 'h': h, 'mesh': n,
                         'kubo': b.chern_numbers().tolist(), 'accepted': accepted,
                         'fhs': chern_numbers_fhs(result).tolist() if n <= 48 else None,
                         'max_pairing': float(np.max(np.abs(result.hamiltonians[:, :2, 2:]))),
                         'max_abs_curvature': float(np.nanmax(np.abs(b.curvature))),
                         'min_level_spacing': float(np.min(b.level_spacing)),
                         'seconds': round(time.time() - t, 2)})
    reversed_field = kitaev(1.0, (24, 24), sign=-1)
    rows.append({'model': 'Kitaev K = -1, -[111] polarized (time reversal)', 'h': 1.0, 'mesh': 24,
                 'kubo': berry_curvature(reversed_field).chern_numbers().tolist(),
                 'fhs': chern_numbers_fhs(reversed_field).tolist()})
    return rows


def null_and_undefined():
    out = {}
    model = triangular_heisenberg()
    conditions = ExternalConditions(field=(0, 0, 1.0))
    state = refine_classical(model, state_120(model, ((1, 0, 0), (0, 0, 1))), conditions)
    result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(24, 24)))
    b = berry_curvature(result)
    out['triangular_heisenberg_h1'] = {'kubo': b.chern_numbers().tolist(),
                                       'fhs': chern_numbers_fhs(result).tolist(),
                                       'max_abs_curvature': float(np.nanmax(np.abs(b.curvature))),
                                       'note': 'pointwise curvature nonzero, integrals zero'}
    square = square_heisenberg()
    conditions = ExternalConditions(field=(0, 0, 2.0))
    canted = refine_classical(square, neel_state(square, (1, 0, 0)), conditions)
    result = solve_lswt(square, canted, conditions, settings=LSWTSettings(mesh=(24, 24)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fhs = chern_numbers_fhs(result)
    b = berry_curvature(result)
    out['canted_square_h2'] = {'kubo': b.chern_numbers().tolist(), 'fhs': [None if np.isnan(x) else x for x in fhs],
                               'max_abs_curvature': float(np.nanmax(np.abs(b.curvature))),
                               'warnings': [str(w.message)[:90] for w in caught],
                               'note': 'folded bands cross on lines between mesh points; FHS link overlap 1e-16'}
    neel = solve_lswt(square, neel_state(square), None, settings=LSWTSettings(mesh=(12, 12)))
    b = berry_curvature(neel)
    out['square_neel'] = {'nan_fraction': float(np.mean(np.isnan(b.curvature))),
                          'min_level_spacing': float(np.min(b.level_spacing))}
    return out


def main():
    report = {'sign_anchor': sign_anchor(), 'pointwise': pointwise(), 'chern': chern_table(),
              'null_and_undefined': null_and_undefined()}
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()
