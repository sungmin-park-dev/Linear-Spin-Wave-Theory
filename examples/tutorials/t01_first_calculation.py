"""Tutorial 1: linear spin-wave theory of the triangular-lattice antiferromagnet.

Companion script to ``docs/tutorials/01-first-calculation.md``. Energies are
in units of J. Run from the repository root:

    python examples/tutorials/t01_first_calculation.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code-space')]

import matplotlib
matplotlib.use('Agg')
import numpy as np

import spintoolkit as stk
from spintoolkit.models import state_120, triangular_heisenberg
from spintoolkit.observables.bands import band_structure
from spintoolkit.visualization import plot_bands


def main(out_dir=None):
    # 1. The model: S = 1/2 Heisenberg antiferromagnet, J = 1.
    model = triangular_heisenberg(J=1.0, S=0.5)
    print(model.lattice)                    # primitive vectors (rows)
    print([site.id for site in model.sites], len(model.terms))

    # 2. The ordered state: 120-degree order on a three-site magnetic cell.
    state = state_120(model)
    print(state.supercell)
    for key, direction in state.directions.items():
        print(key, np.round(direction, 3))

    # 3. Linear spin-wave theory on a 24 x 24 mesh of the magnetic zone.
    result = stk.solve_lswt(model, state, settings=stk.LSWTSettings(mesh=(24, 24)))
    print(f'E_cl  = {result.classical_energy:.4f} J per spin')
    print(f'E_zp  = {result.zero_point_energy:.4f} J per spin')
    print(f'E_gs  = {result.ground_state_energy:.4f} J per spin')
    print(f'<S>   = {result.ordered_moments().mean():.4f}')

    # 4. Bands along Gamma-K-M-Gamma of the primitive zone.
    bands = band_structure(result, ('Γ', 'K', 'M', 'Γ'))
    at_M = band_structure(result, ('Γ', 'M'), points=50).energies[-1]
    print('magnon energies at M:', np.round(at_M, 4))

    # 5. Convergence of the ordered moment with the mesh (Goldstone modes).
    for n in (24, 48, 96):
        fine = stk.solve_lswt(model, state, settings=stk.LSWTSettings(mesh=(n, n)))
        print(f'mesh {n:3d}: E_gs = {fine.ground_state_energy:.6f}  '
              f'<S> = {fine.ordered_moments().mean():.4f}')

    # 6. Figure.
    if out_dir is not None:
        ax = plot_bands(bands)
        ax.set_title('triangular AFM, S = 1/2, 120-degree state')
        ax.figure.tight_layout()
        ax.figure.savefig(Path(out_dir) / 't01-bands.png', dpi=120)
    return result


if __name__ == '__main__':
    main(ROOT / 'data-space/tutorials')
