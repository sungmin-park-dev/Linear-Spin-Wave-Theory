"""Tutorial 5: Berry curvature, Chern numbers and magnon thermal Hall effect.

Companion script to ``docs/tutorials/05-topology.md``. Honeycomb ferromagnet
(J = 1, S = 1/2) with next-nearest-neighbour DM interaction D in a field
h = 0.1 along z. Run from the repository root:

    python examples/tutorials/t05_topology.py
"""

from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code-space')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import spintoolkit as stk
from spintoolkit.models import honeycomb_ferromagnet, polarized_state
from spintoolkit.observables.bands import band_structure
from spintoolkit.observables.berry import berry_curvature, chern_numbers, thermal_hall
from spintoolkit.visualization import plot_berry_curvature, plot_thermal_hall


def solve(D, h=0.1, mesh=48):
    model = honeycomb_ferromagnet(J=1.0, D=D, S=0.5)
    return stk.solve_lswt(model, polarized_state(model), stk.ExternalConditions(field=(0, 0, h)),
                          settings=stk.LSWTSettings(mesh=(mesh, mesh)))


def main(out_dir=None):
    D, h, S = 0.1, 0.1, 0.5
    result = solve(D)

    # Bands at K: 3JS + h -+ 3 sqrt(3) D S.
    at_K = band_structure(result, ('Γ', 'K'), points=60).energies[-1]
    print('bands at K:', np.round(at_K, 4),
          ' closed form:', np.round([3 * S + h - 3 * np.sqrt(3) * D * S,
                                     3 * S + h + 3 * np.sqrt(3) * D * S], 4))

    # Chern numbers: accepted only when the Kubo integral and the FHS integer agree.
    print('Chern numbers (lower, upper):', chern_numbers(result))
    for mesh in (24, 96):
        print(f'  mesh {mesh}:', chern_numbers(solve(D, mesh=mesh)))

    # Without DM the bands touch at K (Dirac point): the Chern number is undefined.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        print('D = 0:', chern_numbers(solve(0.0)))

    # Thermal Hall conductivity kappa_xy / T in units of k_B^2 / hbar per layer.
    temperatures = np.array([0.1, 0.3, 0.5, 1.0])
    hall = thermal_hall(result, temperatures)
    for t, k in zip(temperatures, hall.kappa_over_t):
        print(f'T = {t}: kappa_xy / T = {k:.4f}')
    print('D -> -D:', thermal_hall(solve(-D), temperatures).kappa_over_t.round(4))

    if out_dir is not None:
        curvature = berry_curvature(result)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        plot_berry_curvature(curvature, result.magnetic_lattice, 0, axes[0])
        plot_thermal_hall(thermal_hall(result, np.linspace(0.05, 1.5, 30)), axes[1])
        fig.tight_layout()
        fig.savefig(Path(out_dir) / 't05-topology.png', dpi=120)
    return result


if __name__ == '__main__':
    main(ROOT / 'data-space/tutorials')
