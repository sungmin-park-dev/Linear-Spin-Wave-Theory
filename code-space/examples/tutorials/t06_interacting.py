"""Tutorial 6: interacting spin waves of the triangular antiferromagnet (1/S).

Companion script to ``docs/tutorials/06-interacting.md``. The Heisenberg
antiferromagnet on the triangular lattice (J = 1) in its 120-degree state:
ground-state energy through order S^0, magnon energies and decay rates at
S = 1/2, and the dynamical structure factor through order S^0 at S = 5/2.
Run from the repository root:

    python code-space/examples/tutorials/t06_interacting.py
"""

from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'code-space')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import spintoolkit as stk
from spintoolkit.models import state_120, triangular_heisenberg
from spintoolkit.observables import interacting_path
from spintoolkit.visualization import plot_intensity_path, plot_spectrum_cut


def energies(S):
    model = triangular_heisenberg(J=1.0, S=S)
    result = stk.solve_nlswt(model, state_120(model), settings=stk.NLSWTSettings(mesh=(24, 24)))
    e = result.energies
    print(f'S = {S}: E_cl = {e.classical:.6f}  E_zp = {e.zero_point:.6f}  '
          f'E_(S^0) = {e.order_s0:.6f}  total = {e.total:.6f}')
    return result


def magnons(result, k):
    shifted = result.magnon_energies(k)
    decay = result.decay_rates(k, grid=(96, 96))
    print('LSWT     ', np.round(shifted['lswt'][0], 4))
    print('1/S      ', np.round(shifted['total'][0], 4))
    print('decay    ', np.round(decay['decay_rate'][0], 4))
    return shifted, decay


def main(out_dir=None, points=24, grid=(36, 36)):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        half = energies(0.5)
        k = np.array([[2 * np.pi / 3, 0.0]])          # halfway from Gamma to K
        print(f'magnons at k = {np.round(k[0], 4)} (S = 1/2):')
        magnons(half, k)

        large = energies(2.5)
        omega = np.linspace(0.0, 17.5, 350)
        path = interacting_path(large, omega, eta=0.25, path=('Γ', 'K', 'M', 'Γ'),
                                points=points, grid=grid)
    dw = omega[1] - omega[0]
    defined = np.isfinite(path.intensity[:, 0])
    print(f'undefined path points (Goldstone modes): {int(np.sum(~defined))}')
    flagged = np.flatnonzero(path.negative_weight > 0.01)
    print(f'path points where the expansion is not controlled: {flagged.tolist()}')
    i = int(np.argmin(np.abs(path.distance - 0.5 * path.label_distances[1])))
    total = np.sum(path.intensity[i]) * dw
    lswt = np.sum(path.lswt[i]) * dw
    line = np.sum(path.one_magnon[i]) * dw
    print(f'q = {np.round(path.Q[i], 3)}: total {total:.3f}  one magnon {line:.3f}  '
          f'continuum {total - line:.3f}  (LSWT {lswt:.3f})')

    if out_dir is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), width_ratios=[1.4, 1])
        plot_intensity_path(path, axes[0], intensity_label=r'$S(\mathbf{q},\omega)$')
        plot_spectrum_cut(path, i, axes[1])
        fig.tight_layout()
        fig.savefig(Path(out_dir) / 't06-interacting.png', dpi=120)
    return half, large, path


if __name__ == '__main__':
    main(ROOT / 'data-space/tutorials')
