"""Verify stage 4a: LSWT on the common model types.

1. Regression: ``solve_lswt`` against the existing ``LSWTSolver`` on the NBCP
   One to Four MSL cells (same momenta, MAGSWT).
2. Mesh convergence of the square Neel and triangular 120-degree ground-state
   energy and moment reduction (no regularization, half-step shifted mesh),
   against the analytic LSWT values.
3. Approximate comparison with numerically exact results (D23): square
   S = 1/2 QMC (Sandvik, arXiv:2601.20189) and triangular DMRG moment
   (White and Chernyshev, arXiv:0705.2746).
4. NBCP Y and V states from the stage-2b selection: LSWT without
   regularization on the shifted mesh.

Usage
-----
    python examples/lswt_stage4a_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from model import nbcp
from model.nbcp.model import legacy_cells
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods.lswt import LSWTError, LSWTSettings, LSWTSolver, solve_lswt
from spintoolkit.models import neel_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.system.brillouin_zone import BrillouinZone
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.conversion import to_spin_system

BZ = {'one_msl': 'Hex_60', 'two_msl': 'Tetra', 'three_msl': 'Hex_30', 'four_msl': 'Hex_60'}
MESHES = [8, 16, 32, 64, 128]
LITERATURE = {
    'square_lswt': {'energy': -0.6579, 'delta_s': 0.1966,
                    'source': 'Manousakis, Rev. Mod. Phys. 63, 1 (1991); analytic LSWT integrals'},
    'square_qmc': {'energy': -0.669441857, 'moment': 0.307447,
                   'source': 'Sandvik, arXiv:2601.20189 (SSE QMC, extrapolated)'},
    'triangular_lswt': {'energy_coefficient': 0.436824, 'delta_s': 0.2613032,
                        'source': 'Chernyshev and Zhitomirsky, PRB 79, 144416 (2009), Eqs. (42), (18)'},
    'triangular_dmrg': {'moment': 0.205, 'moment_error': 0.015,
                        'source': 'White and Chernyshev, PRL 99, 127004 (2007), arXiv:0705.2746'},
}


def regression():
    rows = []
    for family in [{}, {'JPD': 0.013, 'JGamma': -0.021}]:
        for cell in BZ:
            n = len(legacy_cells(cell))
            angles = np.column_stack([np.linspace(0.3, 2.6, n), np.linspace(-2.0, 2.5, n)]).ravel()
            model = nbcp.build_model({'Jxy': 0.076, 'Jz': 0.125, **family})
            state = nbcp.candidate_state(model, cell, angles)
            conditions = ExternalConditions(field=[0.03, -0.04, 0.2])
            system = to_spin_system(model, state, conditions)
            legacy = LSWTSolver(system, bz_type=BZ[cell]).solve(N=6, regularization='MAGSWT')
            bz = BrillouinZone(system.to_legacy_dict(BZ[cell])['Lattice/BZ setting'], bz_type=BZ[cell])
            _, k_points, _ = bz.get_full(6)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                new = solve_lswt(model, state, conditions,
                                 settings=LSWTSettings(k_points=k_points, regularization='MAGSWT'))
            by_key = {tuple(map(float, k)): e for k, e in zip(k_points, new.eigenvalues[:, :new.num_sites])}
            bands = np.array([by_key[key] for key in sorted(legacy.data['k_data'])])
            rows.append({'cell': cell, 'family': 'soc' if family else 'xxz', 'num_k': len(k_points),
                         'energy_difference': abs(new.ground_state_energy - legacy.ground_state_energy),
                         'band_difference': float(np.max(np.abs(bands - legacy.eigenvalues))),
                         'boson_number_difference': float(np.max(np.abs(
                             new.boson_numbers - np.array(list(legacy.data['boson_numbers'].values())))))})
    return rows


def convergence(model, state, name):
    rows = []
    for n in MESHES:
        started = time.time()
        r = solve_lswt(model, state, settings=LSWTSettings(mesh=(n, n)))
        rows.append({'N': n, 'energy': r.ground_state_energy, 'delta_s': float(np.mean(r.boson_numbers)),
                     'min_magnon_energy': r.header.diagnostics['min_magnon_energy'],
                     'seconds': round(time.time() - started, 3)})
    extrapolated = 2 * rows[-1]['delta_s'] - rows[-2]['delta_s']
    return {'case': name, 'meshes': rows, 'delta_s_extrapolated_1_over_N': extrapolated}


def nbcp_states():
    scan = json.loads((ROOT / 'data-space/verification/260912-pseudo-goldstone/scan-N12-P72.json').read_text())
    rows = []
    for phase, field_T in [('Y', 0.2), ('V', 1.4)]:
        theta = np.array(scan['states'][phase]['theta'])
        for extra in [{}, {'JPD': 0.01}, {'JGamma': 0.01}]:
            model = nbcp.build_model({'Jxy': 0.075, 'Jz': 0.125, **extra})
            state = nbcp.candidate_state(model, 'three_msl', np.column_stack([theta, np.zeros(3)]).ravel())
            conditions = ExternalConditions(field=[0, 0, 4.645 * MU_B_MEV_PER_T * field_T])
            row = {'phase': phase, 'couplings': extra or 'xxz'}
            try:
                solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(12, 12), shift=False))
                row['zone_centred_mesh'] = 'no zero mode'
            except LSWTError as error:
                row['zone_centred_mesh'] = str(error)[:90]
            r = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(24, 24)))
            row.update({'energy': r.ground_state_energy, 'zero_point': r.zero_point_energy,
                        'moments': r.ordered_moments().tolist(),
                        'min_magnon_energy': r.header.diagnostics['min_magnon_energy'],
                        'max_torque': r.header.diagnostics['max_torque']})
            rows.append(row)
    return rows


def main():
    square, triangular = square_heisenberg(J=1.0), triangular_heisenberg(J=1.0)
    sq = convergence(square, neel_state(square), 'square Neel, S = 1/2')
    tri = convergence(triangular, state_120(triangular), 'triangular 120, S = 1/2')
    e_sq, m_sq = sq['meshes'][-1]['energy'], 0.5 - sq['delta_s_extrapolated_1_over_N']
    m_tri = 0.5 - tri['delta_s_extrapolated_1_over_N']
    approximate = {
        'square_energy': {'lswt': e_sq, 'qmc': LITERATURE['square_qmc']['energy'],
                          'relative_difference': (e_sq - LITERATURE['square_qmc']['energy'])
                          / abs(LITERATURE['square_qmc']['energy'])},
        'square_moment': {'lswt': m_sq, 'qmc': LITERATURE['square_qmc']['moment'],
                          'relative_difference': (m_sq - LITERATURE['square_qmc']['moment'])
                          / LITERATURE['square_qmc']['moment']},
        'triangular_moment': {'lswt': m_tri, 'dmrg': LITERATURE['triangular_dmrg']['moment'],
                              'dmrg_error': LITERATURE['triangular_dmrg']['moment_error'],
                              'relative_difference': (m_tri - 0.205) / 0.205},
    }
    report = {'literature': LITERATURE, 'regression': regression(), 'convergence': [sq, tri],
              'approximate_comparison': approximate, 'nbcp': nbcp_states()}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
