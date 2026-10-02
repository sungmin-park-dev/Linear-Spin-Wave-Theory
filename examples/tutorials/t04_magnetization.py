"""Tutorial 4: magnetization curve M(h) at classical and 1/S order, checked by ED.

Companion script to ``docs/tutorials/04-magnetization.md``. The spin-1/2
square-lattice Heisenberg antiferromagnet (J = 1, g = 1) in a field along z;
saturation at h = 8JS = 4. Run from the repository root:

    python examples/tutorials/t04_magnetization.py
"""

from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'code-space')]

import matplotlib
matplotlib.use('Agg')
import numpy as np

from spintoolkit.methods.ed import EDSector, solve_ed
from spintoolkit.methods.magnetization import magnetization_curve
from spintoolkit.models import neel_state, square_heisenberg
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.visualization import plot_magnetization_curve


def lswt_curve(model, fields):
    # The Neel state is started perpendicular to the field and cants towards it.
    return magnetization_curve(model, neel_state(model, (1, 0, 0)), fields, k_density=48)


def ed_sector_energies(model, cluster):
    """Lowest zero-field energy of every total S^z = N S - n sector on a torus."""
    geometry = CalculationGeometry.finite_torus(cluster)
    N = geometry.num_cells * len(model.sites)
    energies = {}
    for n in range(N // 2 + 1):                  # S^z >= 0 suffices at h >= 0
        sector = EDSector(axis=(0, 0, 1), magnon_number=n, momenta='all')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = solve_ed(model, geometry, sector=sector, num_eigenvalues=1)
        energies[N * 0.5 - n] = min(b.energies[0] for b in result.blocks if len(b.energies))
    return N, energies


def ed_magnetization(N, energies, fields):
    """Ground-state S^z / N at each field from E(S^z) - h S^z (g = 1, field along z)."""
    sz = np.array(sorted(energies))
    e = np.array([energies[s] for s in sz])
    return np.array([sz[np.argmin(e - h * sz)] / N for h in fields])


def main(out_dir=None):
    model = square_heisenberg(J=1.0, S=0.5)          # g = 1, so M is in units of mu_B
    fields = np.linspace(0.0, 4.4, 45)
    curve = lswt_curve(model, fields)

    low = fields[1]
    print(f'chi at h = {low}: classical {curve.classical[1] / low:.4f}, '
          f'1/S {curve.harmonic[1] / low:.4f}')
    for h in (1.0, 2.0, 3.0):
        i = int(np.argmin(abs(fields - h)))
        print(f'h = {h}: classical {curve.classical[i]:.4f}  1/S {curve.harmonic[i]:.4f}  '
              f'S - <n> = {curve.ordered_moments[i].mean():.4f}')

    start = time.time()
    N, energies = ed_sector_energies(model, [[4, 0], [0, 4]])
    dense = np.linspace(0.0, 4.4, 441)
    m_ed = ed_magnetization(N, energies, dense)
    print(f'ED on {N} sites took {time.time() - start:.1f} s')
    # Compare at the centres of the ED plateaus, where the staircase is least
    # affected by the finite size.
    for value in (0.125, 0.25, 0.375):
        h = dense[m_ed == value]
        centre = 0.5 * (h[0] + h[-1])
        i = int(np.argmin(abs(fields - centre)))
        print(f'ED plateau M = {value:.3f} centred at h = {centre:.2f}: '
              f'classical {curve.classical[i]:.3f}  1/S {curve.harmonic[i]:.3f}')

    if out_dir is not None:
        axes = plot_magnetization_curve(curve)
        axes[0].step(dense, m_ed, where='mid', color='k', lw=0.8, label=f'ED, {N} sites')
        axes[0].legend(fontsize=8)
        axes[0].figure.tight_layout()
        axes[0].figure.savefig(Path(out_dir) / 't04-magnetization.png', dpi=120)
    return curve, (dense, m_ed)


if __name__ == '__main__':
    main(ROOT / 'data-space/tutorials')
