"""Tutorial 3: inelastic neutron intensity of the square-lattice antiferromagnet.

Companion script to ``docs/tutorials/03-neutron.md``. Spin-1/2 Heisenberg
antiferromagnet (J = 1), Neel order along z, g = 2, point form factor. Run
from the repository root:

    python examples/tutorials/t03_neutron.py
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
from spintoolkit.models import neel_state, square_heisenberg
from spintoolkit.observables.neutron import neutron_intensity, neutron_path, powder_average
from spintoolkit.visualization import plot_intensity_path, plot_powder


def closed_form(q, J=1.0, S=0.5):
    """Harmonic-order energy and transverse weight S^xx(q) of the Neel state."""
    gamma = 0.5 * (np.cos(q[0]) + np.cos(q[1]))
    return 4 * J * S * np.sqrt(1 - gamma ** 2), 0.5 * S * np.sqrt((1 - gamma) / (1 + gamma))


def main(out_dir=None):
    model = square_heisenberg(J=1.0, S=0.5)
    result = stk.solve_lswt(model, neel_state(model), settings=stk.LSWTSettings(mesh=(24, 24)))

    pi = np.pi
    Q = np.array([[pi, 0], [pi / 2, pi / 2], [pi / 2, 0], [0.9 * pi, 0.9 * pi]])
    spectrum = neutron_intensity(result, Q, g=2.0)
    for q, energies, weights in zip(Q, spectrum.energies, spectrum.intensities):
        positive = energies > 0
        omega, weight = closed_form(q)
        print(f'Q = ({q[0] / pi:.2f}, {q[1] / pi:.2f}) pi: '
              f'omega = {energies[positive].max():.4f} (closed form {omega:.4f}), '
              f'I = {weights[positive].sum():.4f} (closed form {weight:.4f})')

    # The Goldstone mode at the ordering vector: inelastic weight undefined, Bragg peak finite.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bragg = neutron_intensity(result, [[pi, pi]], g=2.0)
    print('at (pi, pi): inelastic', bragg.intensities[0], ' elastic', round(bragg.elastic[0], 4),
          ' <S>^2 =', round(result.ordered_moments().mean() ** 2, 4))

    if out_dir is not None:
        omega = np.linspace(0, 2.5, 251)
        X, M = (pi, 0.0), (pi, pi)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')               # NaN weight at the Bragg vector
            path = neutron_path(result, omega, 0.05, path=[(0, 0), X, M, (0, 0)], points=300, g=2.0)
            powder = powder_average(result, np.linspace(0.05, 8, 120), omega, 0.05,
                                    num_directions=400, g=2.0)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        plot_intensity_path(path, axes[0])
        plot_powder(np.linspace(0.05, 8, 120), omega, powder, axes[1])
        fig.tight_layout()
        fig.savefig(Path(out_dir) / 't03-neutron.png', dpi=120)
    return spectrum


if __name__ == '__main__':
    main(ROOT / 'data-space/tutorials')
