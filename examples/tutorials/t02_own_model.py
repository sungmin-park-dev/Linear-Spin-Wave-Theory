"""Tutorial 2: define your own model, find its classical state, compare candidates.

Companion script to ``docs/tutorials/02-own-model.md``. The spin-1/2
square-lattice J1-J2 Heisenberg model, energies in units of J1. Run from the
repository root:

    python examples/tutorials/t02_own_model.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code-space')]

import numpy as np

import spintoolkit as stk
from spintoolkit.methods.classical import classical_search
from spintoolkit.methods.phase_competition import compare_states


def square_j1j2(J2, S=0.5):
    """H = J1 sum_<ij> S_i . S_j + J2 sum_<<ij>> S_i . S_j - h . sum_i S_i (J1 = 1)."""
    site = stk.Site('A', (0.0, 0.0), S)
    terms = [stk.Term.bilinear(('A', (0, 0)), ('A', d), np.eye(3), label='J1')
             for d in [(1, 0), (0, 1)]]
    terms += [stk.Term.bilinear(('A', (0, 0)), ('A', d), J2 * np.eye(3), label='J2')
              for d in [(1, 1), (1, -1)]]
    terms.append(stk.Term.zeeman('A', np.eye(3)))
    return stk.SpinModel(np.eye(2), (site,), tuple(terms),
                         {'model_id': 'square_j1j2', 'parameters': {'J2': J2, 'S': S}})


def candidates(model):
    neel = stk.SpinState.from_function(
        model, [[1, 1], [1, -1]], lambda site, cell: np.array([1.0, 0, 0]) * (-1) ** (cell[0] + cell[1]))
    stripe = stk.SpinState.from_function(
        model, np.diag([2, 1]), lambda site, cell: np.array([1.0, 0, 0]) * (-1) ** cell[0])
    return {'Neel': neel, 'stripe': stripe}


def main():
    # 1. Search the classical ground state on a 2 x 2 supercell without a guess.
    model = square_j1j2(J2=0.0)
    search = classical_search(model, np.diag([2, 2]))
    print(f'classical energy {search.energy:.4f} J1 per spin')
    for key, direction in search.state.directions.items():
        print(key, np.round(direction, 3))

    # 2. LSWT on the state that was found.
    result = stk.solve_lswt(model, search.state, settings=stk.LSWTSettings(mesh=(48, 48)))
    print(f'E_gs = {result.ground_state_energy:.4f} J1 per spin, '
          f'<S> = {result.ordered_moments().mean():.4f}')

    # 3. Neel against stripe at two values of J2.
    for J2 in (0.2, 0.8):
        model = square_j1j2(J2)
        for report in compare_states(model, candidates(model), k_density=48):
            print(f'J2 = {J2}: {report.name:6s} {report.status:8s} '
                  f'E_cl = {report.classical_energy:.4f}  E_cl + E_zp = {report.harmonic_energy:.4f}')


if __name__ == '__main__':
    main()
