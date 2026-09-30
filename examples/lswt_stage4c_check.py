"""Verify stage 4c: LSWT structure factor and bond correlations.

1. One-magnon weights against ED on tori (U(1)-polarized states).
2. Neel transverse weights against the analytic formula.
3. Sum rules on the extended mesh (square Neel, triangular 120).
4. Bond correlations: energy against E_GS (square, triangular, NBCP Y with J_PD).
5. (The comparison with the removed observables/correlations.py is kept in the
   stage-4c record.)
6. NBCP Y and V: structure factor along a path through the zone centre.

Usage
-----
    python examples/lswt_stage4c_check.py > report.json
"""

import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import neel_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.observables.structure_factor import bond_correlations, structure_factor
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from tests.test_methods.test_ed import CASES
from tests.test_methods.test_structure_factor import ed_weights, extended_momenta
from tests.test_methods.test_thermal import nbcp_y


def ed_comparison():
    rows = []
    for name in ['square_S1/2', 'square_DM', 'honeycomb_two_sites', 'triangular_S3/2_nondiagonal',
                 'nbcp_xxz']:
        model, L, h, _ = CASES[name]
        k, energies, weights = ed_weights(model, L, h)
        state = SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: np.array([0, 0, 1.0]))
        lswt = solve_lswt(model, state, ExternalConditions(field=(0, 0, h)), CalculationGeometry.finite_torus(L))
        sf = structure_factor(lswt, k)
        error = 0.0
        for i, W in enumerate(weights):
            for n, w in enumerate(sf.energies[i]):
                if w > 0:
                    same = np.abs(energies - w) < 1e-9
                    error = max(error, float(np.max(np.abs(sf.weights[i, n, :2, :2] - W[same].sum(axis=0)))))
            error = max(error, float(np.max(np.abs(sf.weights[i].sum(axis=0)[:2, :2] - W.sum(axis=0)))))
        rows.append({'case': name, 'cluster': L, 'num_momenta': len(k), 'max_weight_difference': error})
    return rows


def neel_analytic():
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(8, 8)))
    q = np.random.default_rng(3).uniform(-np.pi, np.pi, (20, 2))
    sf = structure_factor(result, q)
    errors = []
    for i, qi in enumerate(q):
        gamma = 0.5 * (np.cos(qi[0]) + np.cos(qi[1]))
        particle = sf.energies[i] > 0
        weight = np.real(sf.weights[i][particle][:, 0, 0] + sf.weights[i][particle][:, 1, 1]).sum()
        errors.append(abs(weight - 0.5 * np.sqrt((1 - gamma) / (1 + gamma))))
    return {'num_q': len(q), 'max_weight_error': float(max(errors))}


def sum_rules():
    rows = []
    for label, model, state in [('square Neel', square_heisenberg(J=1.0), neel_state),
                                ('triangular 120', triangular_heisenberg(J=1.0), state_120)]:
        result = solve_lswt(model, state(model), settings=LSWTSettings(mesh=(8, 8)))
        q, bragg = extended_momenta(result)
        sf = structure_factor(result, q)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            elastic = float(np.sum(np.real(np.trace(structure_factor(result, bragg).elastic, axis1=1, axis2=2))))
        n, S = result.boson_numbers, result.spins
        inelastic = float(np.mean(sf.trace().sum(axis=1)))
        rows.append({'case': label, 'inelastic': inelastic, 'expected_inelastic': float(np.mean(S * (1 + 2 * n))),
                     'elastic': elastic, 'expected_elastic': float(np.mean((S - n) ** 2)),
                     'total': inelastic + elastic, 'S(S+1)': float(np.mean(S * (S + 1))),
                     'total_minus_S(S+1)': inelastic + elastic - float(np.mean(S * (S + 1))),
                     'predicted_<n^2>': float(np.mean(n ** 2))})
    return rows


def bond_energy():
    rows = []
    for label in ['square', 'triangular', 'NBCP Y, J_PD']:
        if label == 'square':
            model = square_heisenberg(J=1.0)
            result = solve_lswt(model, neel_state(model))
        elif label == 'triangular':
            model = triangular_heisenberg(J=1.0)
            result = solve_lswt(model, state_120(model))
        else:
            model, state, conditions = nbcp_y({'JPD': 0.01})
            result = solve_lswt(model, state, conditions)
        energy = bond_correlations(result, model)['energy']
        rows.append({'case': label, 'bond_energy': energy, 'ground_state_energy': result.ground_state_energy,
                     'difference': energy - result.ground_state_energy})
    return rows


def nbcp_paths():
    rows = []
    for phase, extra in [('Y', {'JPD': 0.01}), ('Y', {'JGamma': 0.01})]:
        model, state, conditions = nbcp_y(extra)
        result = solve_lswt(model, state, conditions)
        path = np.linspace([-np.pi, 0], [np.pi, 0], 41)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            sf = structure_factor(result, path)
        rows.append({'phase': phase, 'couplings': extra, 'num_q': len(path),
                     'zero_mode_momenta': path[sf.zero_mode].tolist(),
                     'bragg_momenta': path[sf.bragg].tolist(),
                     'trace_weights_at_q_pi_over_2': sf.trace()[30].tolist(),
                     'warnings': [str(w.message)[:80] for w in caught]})
    return rows


def main():
    report = {'ed': ed_comparison(), 'neel_analytic': neel_analytic(), 'sum_rules': sum_rules(),
              'bond_energy': bond_energy(),
              'nbcp_paths': nbcp_paths()}
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()
