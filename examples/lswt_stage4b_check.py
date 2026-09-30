"""Verify stage 4b: zero-mode scan and dimensionless finite-temperature LSWT.

1. Zero-mode scan: Neel (Goldstone), easy-axis gap sweep (gapped, candidate,
   zero), J1-J2 square at J2 = J1/2 (lines), triangular 120, NBCP Y with and
   without J_PD (Goldstone versus accidental), polarized states (gapped).
2. Low-temperature free energy against full-spectrum ED on small tori.
3. Neel at finite temperature: finite energies, divergent moments reported.
4. Temperature scan of a polarized triangular ferromagnet (U, S, C, moment).

Usage
-----
    python examples/lswt_stage4b_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from spintoolkit.methods.ed import EDSector, solve_ed
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.observables.thermal import thermal_quantities
from spintoolkit.observables.zero_modes import scan_zero_modes
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from tests.test_methods.test_thermal import nbcp_y, square_model

FIELD = ExternalConditions(field=(0, 0, 5.0))


def scan_row(label, model, state, conditions=None, regularization='none'):
    started = time.time()
    result = solve_lswt(model, state, conditions, settings=LSWTSettings(regularization=regularization))
    report = scan_zero_modes(result, model, state, conditions)
    return {'case': label, 'regularization': regularization, 'has_zero': report.has_zero,
            'has_candidates': report.has_candidates, 'summary': report.summary(),
            'global_minimum': {'fractional': report.global_minimum.fractional.tolist(),
                               'lowest': report.global_minimum.lowest,
                               'gap': report.global_minimum.gap,
                               'classification': report.global_minimum.classification},
            'zone_centre': report.zone_centre, 'num_candidates': len(report.candidates),
            'seconds': round(time.time() - started, 2)}


def scans():
    square, triangular = square_heisenberg(J=1.0), triangular_heisenberg(J=1.0)
    rows = [scan_row('square Neel', square, neel_state(square)),
            scan_row('square polarized, h = 5', square, polarized_state(square), FIELD)]
    for d in (1e-2, 1e-5, 1e-7, 1e-9, 0.0):
        m = square_model(d)
        row = scan_row(f'easy-axis XXZ, anisotropy {d:g}', m, neel_state(m))
        row['analytic_gap'] = 2 * np.sqrt((1 + d) ** 2 - 1)
        rows.append(row)
    m = square_model(J2=0.5)
    rows.append(scan_row('J1-J2 square, J2 = J1/2 (Neel)', m, neel_state(m), regularization='MAGSWT'))
    rows.append(scan_row('triangular 120', triangular, state_120(triangular)))
    for extra in ({}, {'JPD': 0.01}, {'JGamma': 0.01}):
        rows.append(scan_row(f'NBCP Y {extra or "XXZ"}', *nbcp_y(extra)))
    return rows


def ed_low_temperature():
    rows = []
    for label, model, L in [('square 2x3', square_heisenberg(J=1.0), [[2, 0], [0, 3]]),
                            ('triangular 3x3', triangular_heisenberg(J=1.0), [[3, 0], [0, 3]])]:
        geometry = CalculationGeometry.finite_torus(L)
        energies = solve_ed(model, geometry, FIELD,
                            EDSector(axis=(0, 0, 1), magnon_number='all')).energies()
        lswt = solve_lswt(model, polarized_state(model), FIELD, geometry)
        gap = float(lswt.bands().min())
        for ratio in (20, 10, 5, 2.5):
            t = gap / ratio
            n = len(lswt.site_keys) * abs(round(np.linalg.det(L)))
            f_ed = (energies.min() - t * np.log(np.sum(np.exp(-(energies - energies.min()) / t)))) / n
            difference = f_ed - thermal_quantities(lswt, [t]).free_energy[0]
            rows.append({'case': label, 'gap': gap, 't': t, 'F_ed_minus_F_lswt': difference,
                         'exp_minus_gap_over_t': float(np.exp(-gap / t)),
                         'exp_minus_2gap_over_t': float(np.exp(-2 * gap / t))})
    return rows


def neel_finite_temperature():
    model = square_heisenberg(J=1.0)
    rows = []
    for n in (16, 32, 64):
        result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(n, n)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            th = thermal_quantities(result, [0.0, 0.1, 0.3])
        rows.append({'N': n, 't': th.temperatures.tolist(), 'U': th.internal_energy.tolist(),
                     'C': th.specific_heat.tolist(), 'S': th.entropy.tolist(),
                     'moment': th.moments[:, 0].tolist(), 'gapless': th.gapless})
    return rows


def triangular_scan():
    model = triangular_heisenberg(J=1.0)
    result = solve_lswt(model, polarized_state(model), ExternalConditions(field=(0, 0, 5.0)))
    t = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
    th = thermal_quantities(result, t)
    return {'h': 5.0, 'h_sat': 4.5, 't': t.tolist(), 'F': th.free_energy.tolist(),
            'U': th.internal_energy.tolist(), 'S': th.entropy.tolist(),
            'C': th.specific_heat.tolist(), 'moment': th.moments[:, 0].tolist(),
            'beyond_lswt': th.beyond_lswt.tolist()}


def main():
    report = {'zero_mode_scans': scans(), 'ed_low_temperature': ed_low_temperature(),
              'neel_finite_temperature': neel_finite_temperature(),
              'triangular_polarized_scan': triangular_scan()}
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()
