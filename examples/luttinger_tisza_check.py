"""Verify stage 6b: the Luttinger-Tisza diagnostic (D30).

Reports, for the benchmark models and NBCP (zero field), lambda_min, the
minima with their fractions, supercells, eigenspace dimensions and the
single-q strong-constraint result, and the Fourier energy identity on random
commensurate states.

Usage
-----
    python examples/luttinger_tisza_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from model import nbcp
from spintoolkit.methods.classical import classical_energy
from spintoolkit.methods.luttinger_tisza import luttinger_tisza
from spintoolkit.models import honeycomb_ferromagnet, kitaev_honeycomb, square_heisenberg, triangular_heisenberg
from spintoolkit.states.spin_state import SpinState
from tests.test_methods.test_luttinger_tisza import fourier_energy, j1j2, unit

CASES = [
    ('square FM (J = -1)', square_heisenberg(J=-1.0), -0.5),
    ('square Neel (J = 1)', square_heisenberg(J=1.0), -0.5),
    ('triangular AFM (J = 1)', triangular_heisenberg(J=1.0), -0.375),
    ('J1-J2 square, J2/J1 = 0.3', j1j2(0.3), -0.35),
    ('J1-J2 square, J2/J1 = 0.5', j1j2(0.5), -0.25),
    ('J1-J2 square, J2/J1 = 0.7', j1j2(0.7), -0.35),
    ('honeycomb FM + DM (D = 0.2)', honeycomb_ferromagnet(J=1.0, D=0.2), -0.375),
    ('Kitaev (K = -1)', kitaev_honeycomb(K=-1.0), -0.125),
    ('NBCP XXZ (Jxy 0.075, Jz 0.125 meV)', nbcp.build_model({'Jxy': 0.075, 'Jz': 0.125}), None),
]


def main():
    rows = []
    for name, model, expected in CASES:
        start = time.time()
        report = luttinger_tisza(model, mesh=(48, 48))
        rows.append({
            'model': name, 'lambda_min': report.lambda_min, 'expected': expected,
            'extended_degeneracy': report.extended_degeneracy,
            'near_minimal_fraction': report.near_minimal_fraction,
            'strong_constraint': report.strong_constraint,
            'minima': [{'fractional': m.fractional.tolist(), 'fraction': m.fraction,
                        'multiplicity': m.multiplicity,
                        'supercell': None if m.supercell is None else m.supercell.tolist(),
                        'strong_constraint': m.strong_constraint,
                        'strong_residual': m.strong_residual, 'state_energy': m.state_energy}
                       for m in report.minima[:4]],
            'seconds': round(time.time() - start, 2)})
    rng = np.random.default_rng(3)
    identity = []
    for name, model, cell in (('triangular', triangular_heisenberg(), [[1, 1], [-1, 2]]),
                              ('J1-J2 0.4', j1j2(0.4), [[2, 0], [0, 2]]),
                              ('honeycomb FM + DM', honeycomb_ferromagnet(D=0.3), [[2, 1], [0, 1]]),
                              ('Kitaev K = 0.7', kitaev_honeycomb(K=0.7), [[1, 0], [1, 2]]),
                              ('NBCP', nbcp.build_model({'Jxy': 0.075, 'Jz': 0.125, 'JPD': 0.01}),
                               [[2, 1], [1, 2]])):
        state = SpinState.from_function(model, cell, lambda site, c: unit(rng), {})
        identity.append({'model': name, 'supercell': cell,
                         'difference': abs(fourier_energy(model, state)
                                           - classical_energy(model, state, None))})
    print(json.dumps({'cases': rows, 'fourier_identity': identity}, indent=2, default=float))


if __name__ == '__main__':
    main()
