"""Verify the stage-2b zero-point state selection (D17, D19) on NBCP and benchmarks.

1. NBCP Y (B = 0.2 T) and V (B = 1.4 T) with J_PD or J_Gamma = 0.010 meV at
   mesh N = 6 and 12, both criteria modes, starting 1e-3 rad away from the
   reference configuration: verdict, axis, dominant harmonic, amplitude and
   curvature against the independent scans of
   data-space/verification/260912-pseudo-goldstone (N = 48), fit residual and
   resolution margin, and the MAGSWT regularization value.
2. The criteria cases of examples/nbcp_y_orbit_criteria_check.py from
   differential-evolution starts (three seeds): exact U(1), J_PD, J_Gamma,
   J_PD with the whole problem rotated, and the controls tilted field and
   polarized state (20 h).
3. Benchmarks: triangular Heisenberg 120-degree state at zero field and the
   polarized square Heisenberg state above saturation.
4. Energy landscape along the orbit (orbit_energy_landscape). The comparison with
   the removed MAGSWT grid search is kept in the stage-2b record.

Usage
-----
    python examples/nbcp_state_selection_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np
from scipy.optimize import differential_evolution

from model import nbcp
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods import state_selection as sel
from spintoolkit.methods.lswt.energy import EnergyFunction
from spintoolkit.models import polarized_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.conversion import site_label, to_spin_system
from spintoolkit.system.model import SpinModel, Term

SCAN = ROOT / 'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json'
G_Z = 4.645
J, JZ = 0.075, 0.125
PHASES = {'Y': 0.2, 'V': 1.4}
COUPLINGS = {'PD': {'JPD': 0.010}, 'Gamma': {'JGamma': 0.010}}
PHI0 = 0.3
DE_SEEDS = [42, 7, 2026]


def zeeman(field_T):
    return G_Z * MU_B_MEV_PER_T * field_T


def rotated(model, R):
    terms = [Term.bilinear(*t.participants, R @ t.coefficient @ R.T, t.label)
             for t in model.terms_of_kind('bilinear')] + list(model.terms_of_kind('zeeman'))
    return SpinModel(model.lattice, model.sites, terms, {**model.metadata, 'model_id': 'rotated'})


def common_azimuth(model, state, theta):
    """Rotation angle about z that best maps the phi = 0 reference configuration onto ``state``."""
    reference = nbcp.candidate_state(model, 'three_msl',
                                     np.column_stack([theta, np.zeros(3)]).ravel())
    cross = dot = 0.0
    for key, r in reference.directions.items():
        s = state.directions[key]
        cross += r[0] * s[1] - r[1] * s[0]
        dot += r[0] * s[0] + r[1] * s[1]
    return float(np.arctan2(cross, dot))


def summary(result, started):
    d = result.diagnostics
    orbit = d.get('orbit', {})
    fit = orbit.get('E_qm_fit', {})
    provider = d.get('quantum_energy_provider', {})
    return {'verdict': result.verdict, 'message': result.message,
            'axis': None if result.axis is None else result.axis.tolist(),
            'phi': result.phi, 'max_torque_before': d['refinement']['max_torque_before'],
            'max_torque_after': d['max_torque'], 'state_accuracy': d['state_accuracy'],
            'curvature_floor': d['curvature_floor'], 'gap_ratio': d['gap_ratio'],
            'null_count': d['null_count'], 'generator_rank': d['generator_rank'],
            'generator_min_relative_singular_value': min(d['generator_relative_singular_values']),
            'flat_rotation_count': d['flat_rotation_count'], 'C_cl': d.get('C_cl'),
            'C_qm': d.get('C_qm'), 'exact_symmetry': d.get('exact_symmetry'),
            'E_cl_span': orbit.get('E_cl_span'), 'E_qm_span': orbit.get('E_qm_span'),
            'E_qm_amplitude': orbit.get('E_qm_amplitude'),
            'E_qm_dominant_harmonic': orbit.get('E_qm_dominant_harmonic'),
            'E_qm_fit_residual': fit.get('residual'),
            'E_qm_resolution_threshold': orbit.get('E_qm_resolution_threshold'),
            'equivalent_minima': len(d.get('equivalent_minima', [])),
            'pinning_screen': d.get('pinning_screen'),
            'classical_to_quantum': d.get('classical_to_quantum'),
            'regularization': [provider.get('regularization_min'), provider.get('regularization_max')],
            'seconds': round(time.time() - started, 2)}


def nbcp_cases():
    scan = json.loads(SCAN.read_text())
    rows = []
    for phase, field_T in PHASES.items():
        theta = np.array(scan['states'][phase]['theta'])
        for coupling, extra in COUPLINGS.items():
            pd, gamma = extra.get('JPD', 0), extra.get('JGamma', 0)
            ref = next(s['current'] for s in scan['scans'] if s['phase'] == phase
                       and s['JPD_meV'] == pd and s['JGamma_meV'] == gamma)
            harmonic = 3 if (phase, coupling) == ('V', 'Gamma') else 6
            ref_amplitude = float(np.hypot(ref['cos_coefficients'][harmonic - 1],
                                           ref['sin_coefficients'][harmonic - 1]))
            model = nbcp.build_model({'Jxy': J, 'Jz': JZ, **extra})
            conditions = ExternalConditions(field=[0, 0, zeeman(field_T)])
            angles = np.column_stack([theta, np.full(3, PHI0)]).ravel()
            angles += 1e-3 * np.random.default_rng(1).standard_normal(6)
            state = nbcp.candidate_state(model, 'three_msl', angles)
            for N in (6, 12):
                for mode in sel.MODES:
                    started = time.time()
                    result = sel.select_on_manifold(
                        model, state, conditions, sel.lswt_zero_point_energy('Hex_30', N),
                        criteria=sel.SelectionCriteria(mode=mode))
                    row = {'phase': phase, 'coupling': coupling, 'field_T': field_T, 'N': N,
                           'mode': mode, **summary(result, started),
                           'reference_amplitude_N48': ref_amplitude,
                           'reference_curvature_N48': ref['curvature_meV_per_spin'],
                           'reference_phi_min_N48': ref['phi_min']}
                    if result.verdict == sel.SELECTED:
                        period = 2 * np.pi / harmonic
                        offset = np.mod(common_azimuth(model, result.state, theta) - ref['phi_min'] + period / 2,
                                        period) - period / 2
                        row.update({
                            'amplitude_over_reference': row['E_qm_amplitude'] / ref_amplitude,
                            'curvature_over_reference': row['C_qm'] / ref['curvature_meV_per_spin'],
                            'phi_offset_from_reference_mod_period': float(offset),
                            'phi_resolution': row['E_qm_fit_residual'] / (harmonic * row['E_qm_amplitude']),
                            'resolution_margin': row['E_qm_amplitude'] / row['E_qm_resolution_threshold']})
                    rows.append(row)
    return rows


def de_state(model, conditions, seed):
    """Differential-evolution minimum with the SpinOptimizer settings except the seed."""
    template = nbcp.candidate_state(model, 'three_msl', np.zeros(6))
    system = to_spin_system(model, template, conditions)
    energy = EnergyFunction(system.to_legacy_dict('Hex_30'), N=2)
    result = differential_evolution(
        energy.classical_energy_density_func, [(-np.pi, np.pi)] * 6, strategy='best1bin',
        popsize=18, tol=1e-9, mutation=(0.5, 0.9), recombination=0.8, maxiter=800,
        polish=True, updating='immediate', seed=seed)
    index = {s.label: i for i, s in enumerate(system.sites)}
    x = result.x.reshape(-1, 2)
    directions = {}
    for cell in template.cells:
        theta, phi = x[index[site_label('Co', cell, template.num_cells)]]
        directions[('Co', cell)] = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                                    np.cos(theta)]
    return SpinState(template.model_ref, template.supercell, directions,
                     {'origin': 'differential_evolution', 'seed': seed})


def criteria_cases():
    h = zeeman(0.2)
    tilt = sel.rotation_matrix([1, 2, 0], 0.7)
    cases = [('exact U(1)', {}, [0, 0, h], np.eye(3), True),
             ('J_PD = 0.010', {'JPD': .010}, [0, 0, h], np.eye(3), True),
             ('J_Gamma = 0.010', {'JGamma': .010}, [0, 0, h], np.eye(3), True),
             ('J_PD = 0.010, rotated', {'JPD': .010}, [0, 0, h], tilt, True),
             ('control: tilted field (0.3h, 0, h)', {'JPD': .010}, [.3 * h, 0, h], np.eye(3), False),
             ('control: polarized, field 20h', {}, [0, 0, 20 * h], np.eye(3), False)]
    rows = []
    for label, extra, field, R, degenerate in cases:
        model = rotated(nbcp.build_model({'Jxy': J, 'Jz': JZ, **extra}), R)
        conditions = ExternalConditions(field=R @ np.asarray(field, float))
        expected = R @ [0., 0., 1.]
        for seed in DE_SEEDS:
            state = de_state(model, conditions, seed)
            for mode in sel.MODES:
                started = time.time()
                result = sel.select_on_manifold(model, state, conditions,
                                                sel.lswt_zero_point_energy('Hex_30', 6),
                                                criteria=sel.SelectionCriteria(mode=mode))
                row = {'case': label, 'expected_degenerate': degenerate, 'seed': seed,
                       'mode': mode, **summary(result, started)}
                if degenerate and result.axis is not None:
                    row['axis_error'] = float(min(np.linalg.norm(result.axis - expected),
                                                  np.linalg.norm(result.axis + expected)))
                rows.append(row)
    return rows


def benchmark_cases():
    rows = []
    triangular = triangular_heisenberg(J=1.0)
    square = square_heisenberg(J=1.0)
    for mode in sel.MODES:
        criteria = sel.SelectionCriteria(mode=mode)
        for label, model, state, conditions, bz, axis in [
                ('triangular 120, h = 0', triangular, state_120(triangular), None, 'Hex_60', None),
                ('triangular 120, h = 0, axis z', triangular, state_120(triangular), None,
                 'Hex_60', [0, 0, 1]),
                ('square polarized, h = 5 > h_sat = 4', square, polarized_state(square),
                 ExternalConditions(field=[0, 0, 5.0]), 'Tetra', None),
                ('square polarized, axis z', square, polarized_state(square),
                 ExternalConditions(field=[0, 0, 5.0]), 'Tetra', [0, 0, 1])]:
            started = time.time()
            result = sel.select_on_manifold(model, state, conditions,
                                            sel.lswt_zero_point_energy(bz, 6), axis=axis,
                                            criteria=criteria)
            rows.append({'case': label, 'mode': mode, **summary(result, started)})
    return rows


def landscape_cases():
    """Energies along the orbit (replaces the removed MAGSWT grid search, D27)."""
    scan = json.loads(SCAN.read_text())
    rows = []
    phis = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    for phase, field_T in PHASES.items():
        theta = np.array(scan['states'][phase]['theta'])
        for coupling, extra in COUPLINGS.items():
            model = nbcp.build_model({'Jxy': J, 'Jz': JZ, **extra})
            conditions = ExternalConditions(field=[0, 0, zeeman(field_T)])
            state = nbcp.candidate_state(model, 'three_msl',
                                         np.column_stack([theta, np.full(3, PHI0)]).ravel())
            landscape = sel.orbit_energy_landscape(model, state, conditions,
                                                   sel.lswt_zero_point_energy('Hex_30', 6), phis=phis)
            selected = sel.select_on_manifold(model, state, conditions,
                                              sel.lswt_zero_point_energy('Hex_30', 6))
            rows.append({'phase': phase, 'coupling': coupling, 'num_phi': len(phis),
                         'E_cl_span': landscape.diagnostics['E_cl_span'],
                         'E_qm_span': float(np.ptp(landscape.quantum)),
                         'sampled_min_minus_selected': float(landscape.quantum.min()
                                                             - selected.diagnostics['E_qm_selected'])})
    return rows


def main():
    report = {'J_meV': J, 'Jz_meV': JZ, 'g_z': G_Z, 'phi0': PHI0, 'reference': str(SCAN.relative_to(ROOT)),
              'criteria_defaults': {k: v for k, v in vars(sel.SelectionCriteria()).items()},
              'nbcp': nbcp_cases(), 'criteria': criteria_cases(),
              'benchmarks': benchmark_cases(), 'landscape': landscape_cases()}
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()
