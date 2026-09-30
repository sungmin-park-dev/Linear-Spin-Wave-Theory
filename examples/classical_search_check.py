"""Verify stage 6c: the global classical search on the common model (D32).

1. Benchmarks (square Neel, triangular 120 degrees, polarized square in a
   field) against their analytic energies.
2. The legacy SpinOptimizer searches stored in a regression snapshot (NBCP
   families xxz and nn_soc on the One-Four MSL cells, tilted field): the new
   search energy and the refined energy against the stored differential-
   evolution energy and the refined stored angles.
3. The NBCP example configuration (J_Gamma = 0.1 meV, 0.376418 meV field).

Usage
-----
    python examples/classical_search_check.py SNAPSHOT.npz > report.json
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
from spintoolkit.methods.classical import classical_energy, classical_search, refine_classical
from spintoolkit.models import square_heisenberg, triangular_heisenberg
from spintoolkit.system.conditions import ExternalConditions

CELL_KEYS = {'one_msl': 'one_msl', 'two_msl': 'two_msl', 'three_msl': 'three_msl',
             'four_msl': 'four_msl'}


def benchmarks():
    rows = []
    for name, model, cell, conditions, expected in (
            ('square Neel', square_heisenberg(), [[1, 1], [1, -1]], None, -0.5),
            ('triangular 120', triangular_heisenberg(), [[1, 1], [-1, 2]], None, -0.375),
            ('square polarized, h = 5', square_heisenberg(), [[1, 0], [0, 1]],
             ExternalConditions(field=(0, 0, 5.0)), -2.0)):
        start = time.time()
        result = classical_search(model, cell, conditions)
        rows.append({'case': name, 'energy': result.energy, 'expected': expected,
                     'difference': result.energy - expected, 'evaluations': result.evaluations,
                     'seconds': round(time.time() - start, 2)})
    return rows


def snapshot_searches(path):
    sys.path.insert(0, str(ROOT / 'examples'))
    from package_regression_snapshot import BASE, CELLS, FAMILIES, SEARCHES
    stored = np.load(path)
    rows = []
    for family, method in SEARCHES:
        config = {**BASE, **FAMILIES[family]}
        parameters = {k: v for k, v in config.items() if k != 'h'}
        model = nbcp.build_model(parameters)
        conditions = ExternalConditions(field=config['h'])
        for name, num_angles, bz_type in CELLS:
            key = f'search/{family}/{method}/{name}/classical'
            legacy_energy = float(stored[key + '/E_cl'])
            legacy_state = refine_classical(
                model, nbcp.candidate_state(model, name, stored[key + '/angles']), conditions)
            start = time.time()
            result = classical_search(model, nbcp.SUPERCELLS[name], conditions)
            refined_legacy = classical_energy(model, legacy_state, conditions)
            rows.append({'family': family, 'cell': name, 'legacy_search_energy': legacy_energy,
                         'new_search_energy': result.search_energy,
                         'legacy_refined_energy': refined_legacy, 'new_energy': result.energy,
                         'refined_difference': result.energy - refined_legacy,
                         'seconds': round(time.time() - start, 2)})
    return rows


def example_configuration():
    model = nbcp.build_model({'Jxy': 0.076, 'Jz': 0.125, 'JGamma': 0.1})
    conditions = ExternalConditions(field=(0, 0, 0.376418))
    return {name: classical_search(model, nbcp.SUPERCELLS[name], conditions).energy
            for name in CELL_KEYS}


def main():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        report = {'benchmarks': benchmarks(), 'snapshot_searches': snapshot_searches(sys.argv[1]),
                  'example_configuration': example_configuration()}
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()
