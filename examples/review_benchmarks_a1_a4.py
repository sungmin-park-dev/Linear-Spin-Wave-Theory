"""Benchmarks of the batch-review items A1-A4 against literature and exact results.

Prints the tables behind ``code-space/tests/test_methods/test_literature_benchmarks.py``
with the larger meshes and spins that are too slow for the test suite:

A1  square antiferromagnet: package M(h)/h against the closed-form canted
    dispersion, its h -> 0 limit against chi = 1/8 - 0.034447/S (Hamer, Zheng
    and Oitmaa, PRB 50, 6877 (1994)), and the reduced-moment formula.
A2  easy-axis ferromagnet in a transverse field, S = 1, 3/2, 2: ED on the
    3 x 3 torus against LSWT with (1 - 1/2S) A and with the bare A.
A3  kagome ferromagnet with DM (Chern numbers) and the in-plane field
    Kitaev-Gamma model (sign structure of kappa_xy).
A4  triangular 120 degree state: neutron weights against the closed form.

Run from the repository root (S = 2 ED takes about five minutes):

    python examples/review_benchmarks_a1_a4.py [--quick]
"""

from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT / 'code-space' / 'tests' / 'test_methods')]

import numpy as np

from spintoolkit.methods.classical import refine_classical
from spintoolkit.methods.ed import EDSector, solve_ed
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.methods.magnetization import magnetization_curve
from spintoolkit.models import neel_state, polarized_state, square_heisenberg, state_120
from spintoolkit.models import triangular_heisenberg
from spintoolkit.observables.berry import chern_numbers, thermal_hall
from spintoolkit.observables.neutron import neutron_intensity
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry

from test_literature_benchmarks import (easy_axis_ferromagnet, kagome_ferromagnet, kitaev_gamma,
                                        square_canted_magnetization)

warnings.filterwarnings('ignore')
QUICK = '--quick' in sys.argv


def a1():
    print('A1  square antiferromagnet, M/h at 1/S order')
    for S in (0.5, 1.0):
        model = square_heisenberg(J=1.0, S=S)
        fields = [0.02, 0.05, 0.1, 0.2]
        curve = magnetization_curve(model, neel_state(model, (1, 0, 0)), fields,
                                    k_density=48 if QUICK else 96)
        closed = [square_canted_magnetization(h, S, n=1600) / h for h in fields]
        chi = [square_canted_magnetization(h, S, n=1600) / h for h in (0.005, 0.01)]
        print(f'  S = {S}: h = {fields}')
        print(f'    package     {np.round(curve.harmonic / fields, 5)}')
        print(f'    closed form {np.round(closed, 5)}')
        print(f'    h -> 0: {2 * chi[0] - chi[1]:.5f}, Hamer et al. {0.125 - 0.034447 / S:.5f}, '
              f'(S - <n>)/8S {(S - 0.1966) / (8 * S):.5f}, classical 0.12500')


def a2():
    print('A2  -sum S.S - 0.5 sum (S^z)^2 - h sum S^x, h = h_flop / 2, 3 x 3 torus, per site')
    geometry = CalculationGeometry.finite_torus([[3, 0], [0, 3]])
    A = np.diag([0.0, 0.0, -0.5])
    for S in ((1.0, 1.5) if QUICK else (1.0, 1.5, 2.0)):
        h = 0.25 * (2 * S - 1)
        conditions = ExternalConditions(field=(h, 0, 0))
        start = time.time()
        ed = solve_ed(easy_axis_ferromagnet(S, A), geometry, conditions,
                      sector=EDSector(momenta='all'), num_eigenvalues=1)
        exact = min(block.energies[0] for block in ed.blocks) / 9
        energies = []
        for coefficient in (A, A / (1 - 1 / (2 * S))):
            model = easy_axis_ferromagnet(S, coefficient)
            state = refine_classical(model, polarized_state(model, (0.3, 0, 1)), conditions)
            energy = solve_lswt(model, state, conditions, geometry).ground_state_energy
            n = next(iter(state.directions.values()))
            if coefficient is not A:
                energy += -(S / 2) * np.trace(coefficient) + (S / 2) * (np.trace(A) - n @ A @ n)
            energies.append(energy)
        print(f'  S = {S}: ED {exact:.5f}  (1-1/2S)A {energies[0]:.5f} ({energies[0] - exact:+.5f})'
              f'  bare A {energies[1]:.5f} ({energies[1] - exact:+.5f})  [{time.time() - start:.0f} s]')


def a3():
    print('A3  Chern numbers and kappa_xy / T (units k_B^2 / hbar)')
    for D in (0.2, -0.2):
        model = kagome_ferromagnet(1.0, D, 0.5)
        result = solve_lswt(model, polarized_state(model), ExternalConditions(field=(0, 0, 0.3)),
                            settings=LSWTSettings(mesh=(48, 48)))
        print(f'  kagome D = {D:+}: C = {chern_numbers(result)}, '
              f'kappa/T(T = 0.3, 1) = {np.round(thermal_hall(result, [0.3, 1.0]).kappa_over_t, 5)}')
    a, b = np.array([1, 1, -2]) / np.sqrt(6), np.array([1, -1, 0]) / np.sqrt(2)
    for G in (-0.3, 0.3):
        model = kitaev_gamma(-1.0, G)
        for name, d in (('a', a), ('-a', -a), ('b', b)):
            conditions = ExternalConditions(field=tuple(3.0 * d))
            state = refine_classical(model, polarized_state(model, d), conditions)
            for mesh in ((48,) if QUICK else (48, 96)):
                result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(mesh, mesh)))
                print(f'  K = -1, Gamma = {G:+}, h = 3 {name:>2}, mesh {mesh}: C = {chern_numbers(result)}, '
                      f'kappa/T(T = 0.5, 1) = {thermal_hall(result, [0.5, 1.0]).kappa_over_t}')


def a4():
    print('A4  triangular 120 degrees, S = 1/2: neutron modes against the closed form')
    model = triangular_heisenberg(J=1.0, S=0.5)
    result = solve_lswt(model, state_120(model), settings=LSWTSettings(mesh=(24, 24)))
    K = np.array([4 * np.pi / 3, 0.0])
    bragg = neutron_intensity(result, [[*K, 0.0]])
    m = result.ordered_moments()[0]
    print(f'  elastic at K {bragg.elastic[0]:.6f}, m^2/4 = {m * m / 4:.6f} (m = S - <n> = {m:.4f}); '
          f'inelastic at K: {bragg.intensities[0]}')
    gamma_point = neutron_intensity(result, [[0.0, 4 * np.pi / np.sqrt(3), 0.0]])
    print(f'  inelastic at the nuclear Bragg vector (0, 4pi/sqrt3): {gamma_point.intensities[0]} '
          '(closed-form limit 0)')


if __name__ == '__main__':
    for part in (a1, a2, a3, a4):
        part()
