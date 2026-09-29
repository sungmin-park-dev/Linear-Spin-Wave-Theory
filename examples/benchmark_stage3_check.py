"""Verify the stage-3 torus expansion and minimal ED (D10, D23).

Runs the cases of code-space/tests/test_methods/test_ed.py and
test_system/test_cluster.py and records the numbers:

1. one-magnon ED energies against the LSWT bands at every torus momentum
   (U(1) about the field), with the momentum-sign negative control;
2. general Hamiltonians without symmetry against a Kronecker-product build;
3. sector and momentum blocks against the full spectrum;
4. the 4 x 4 square S = 1/2 Heisenberg ground state;
5. torus classical energies against the thermodynamic limit and the
   rejection of bonds that fold onto one site.

Usage
-----
    python examples/benchmark_stage3_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from spintoolkit.methods.classical import classical_energy
from spintoolkit.methods.ed import EDSector, SectorError, solve_ed
from spintoolkit.models import neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.cluster import ClusterError, allowed_momenta
from spintoolkit.system.conditions import ExternalConditions
from tests.test_methods.test_ed import (
    CASES, kronecker_spectrum, lswt_bands, nbcp, random_model, torus)


def magnon_cases():
    rows = []
    for name, (model, L, h, axis) in CASES.items():
        axis = np.asarray(axis, dtype=float)
        conditions = ExternalConditions(field=h * axis)
        state = SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: axis)
        _, k = allowed_momenta(model, torus(L))
        started = time.time()
        result = solve_ed(model, torus(L), conditions,
                          EDSector(axis=tuple(axis), magnon_number=1, momenta='all'))
        seconds = time.time() - started
        plus, minus = lswt_bands(model, state, conditions, k), lswt_bands(model, state, conditions, -k)
        error = max(np.max(np.abs(result.excitations(b) - plus[b.momentum_index])) for b in result.blocks)
        flipped = max(np.max(np.abs(result.excitations(b) - minus[b.momentum_index])) for b in result.blocks)
        classical = classical_energy(model, state, conditions, torus(L)) * result.num_sites
        rows.append({'case': name, 'cluster': L, 'num_sites': result.num_sites, 'h': h,
                     'axis': axis.tolist(), 'max_abs_ed_minus_lswt': float(error),
                     'max_abs_ed_minus_lswt_at_minus_k': float(flipped),
                     'reference_minus_classical': float(result.reference_energy - classical),
                     'min_one_magnon_gap': float(min(np.min(result.excitations(b)) for b in result.blocks)),
                     'hermiticity': result.diagnostics['hermiticity'], 'seconds': round(seconds, 3)})
    return rows


def general_cases():
    rows = []
    for name, model, L, conditions in [
            ('random, two cells', random_model(), [[1, 1], [-1, 1]],
             ExternalConditions(field=(0.3, -0.2, 0.5))),
            ('random, three cells', random_model(), [[1, 1], [-1, 2]],
             ExternalConditions(field=(0.3, -0.2, 0.5))),
            ('NBCP with J_PD, J_Gamma, D_z, 3 x 3', nbcp.build_model(
                {'Jxy': 0.075, 'Jz': 0.125, 'JGamma': 0.03, 'JPD': 0.02, 'Dz': 0.01}),
             [[3, 0], [0, 3]], ExternalConditions(field=(0.02, 0, 0.05)))]:
        reference = kronecker_spectrum(model, L, conditions)
        row = {'case': name, 'cluster': L, 'dimension': len(reference)}
        for label, sector in [('full', EDSector()), ('momenta', EDSector(momenta='all')),
                              ('rotated_axis_frame', EDSector(axis=(0.3, -0.5, 0.8)))]:
            result = solve_ed(model, torus(L), conditions, sector)
            row[label] = float(np.max(np.abs(result.energies() - reference)))
        rows.append(row)
    return rows


def other_checks():
    square, triangular = square_heisenberg(J=1.0), triangular_heisenberg(J=1.0)
    full = solve_ed(square, torus([[2, 0], [0, 3]]))
    blocks = solve_ed(square, torus([[2, 0], [0, 3]]),
                      sector=EDSector(axis=(1, 1, 0), magnon_number='all', momenta='all'))
    started = time.time()
    ground = solve_ed(square, torus([[4, 0], [0, 4]]),
                      sector=EDSector(axis=(0, 0, 1), magnon_number=8, momenta=(0, 10)),
                      num_eigenvalues=1)
    seconds = time.time() - started
    try:
        solve_ed(nbcp.build_model({'Jxy': 0.075, 'Jz': 0.125, 'JGamma': 0.01}), torus([[3, 0], [0, 3]]),
                 ExternalConditions(field=(0, 0, 1.0)), EDSector(axis=(0, 0, 1), magnon_number=1))
        broken = 'not rejected'
    except SectorError as error:
        broken = str(error)
    folded = {}
    for L in ([[1, 0], [0, 1]], [[1, 0], [0, 3]]):
        messages = []
        for call in (lambda: classical_energy(square, polarized_state(square), None, torus(L)),
                     lambda: solve_ed(square, torus(L))):
            try:
                call()
                messages.append('not rejected')
            except ClusterError as error:
                messages.append(str(error)[:60])
        folded[str(L)] = messages
    torus_energy = {}
    for label, model, state, L in [('square Neel 2x2', square, neel_state, [[2, 0], [0, 2]]),
                                   ('triangular 120 3x3', triangular, state_120, [[3, 0], [0, 3]])]:
        s = state(model)
        torus_energy[label] = float(classical_energy(model, s, None, torus(L))
                                    - classical_energy(model, s))
    return {'sector_union_vs_full': float(np.max(np.abs(blocks.energies() - full.energies()))),
            'square_4x4': {'E0_per_site': float(min(b.energies[0] for b in ground.blocks) / 16),
                           'literature': -0.7017802, 'block_dimensions': [b.dimension for b in ground.blocks],
                           'solver': [b.solver for b in ground.blocks],
                           'residual': max(b.residual for b in ground.blocks), 'seconds': round(seconds, 3)},
            'broken_u1_sector': broken, 'folded_bonds': folded,
            'torus_minus_thermodynamic_classical_energy': torus_energy}


def main():
    report = {'magnon': magnon_cases(), 'general': general_cases(), 'other': other_checks()}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
