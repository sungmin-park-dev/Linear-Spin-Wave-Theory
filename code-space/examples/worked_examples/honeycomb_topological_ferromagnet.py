"""Worked example: honeycomb ferromagnet with second-neighbour DM interaction.

Companion script to the research note *LSWT example: honeycomb topological
magnon bands* (``workbench/topics/worked-examples/lswt-example-honeycomb-topological-ferromagnet``).
The model is ``H = -J sum_<ij> S_i . S_j + D sum_<<ij>> nu_ij z . (S_i x S_j)``
with ``J = 1``, ``D = 0.1`` and ``S = 1/2`` about the state polarized along
+z; energies are in units of J. The script compares the package with the
closed forms of the note:

- two bands ``omega_k = 3 J S -+ S sqrt(J^2 |f_k|^2 + 4 D^2 s_k^2)``,
- the gap ``6 sqrt(3) D S`` at K,
- Chern numbers of the two bands for D = +0.1 and D = -0.1, and the
  analytic count ``C_- = sum_(+-K) sgn(eta d^z) / 2`` from the Dirac expansion,
  where eta is the determinant of q -> (d^x, d^y) and d^z = 2 D S s_K.

Run from the repository root:

    python code-space/examples/worked_examples/honeycomb_topological_ferromagnet.py
"""

from _common import quiet

import numpy as np

import spintoolkit as stk
from spintoolkit.models import honeycomb_ferromagnet, polarized_state
from spintoolkit.observables.bands import band_structure
from spintoolkit.observables.berry import chern_numbers

J, D, S = 1.0, 0.1, 0.5
A1, A2 = np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])
TAU = (A1 + A2) / 3                                  # position of B in the cell
NEAREST = (TAU, TAU - A1, TAU - A2)                  # A -> B bond vectors
SECOND = (A1, A2 - A1, -A2)                          # 120 degrees apart, sum zero
K = np.array([4 * np.pi / 3, 0.0])


def bands(k, D=D):
    f = sum(np.exp(1j * k @ d) for d in NEAREST)
    s = sum(np.sin(k @ b) for b in SECOND)
    split = S * np.sqrt(J ** 2 * np.abs(f) ** 2 + 4 * D ** 2 * s ** 2)
    return np.stack([3 * J * S - split, 3 * J * S + split], -1)


def dirac_count(D, h=1e-6):
    """Lower-band Chern number from the massive Dirac points at +-K."""
    total = 0.0
    for corner in (K, -K):
        def d_xy(q):
            f = sum(np.exp(1j * (corner + q) @ d) for d in NEAREST)
            return np.array([-J * S * f.real, J * S * f.imag])   # d^x - i d^y = -J S f
        jacobian = np.column_stack([(d_xy(h * e) - d_xy(-h * e)) / (2 * h) for e in np.eye(2)])
        mass = 2 * D * S * sum(np.sin(corner @ b) for b in SECOND)
        total += 0.5 * np.sign(np.linalg.det(jacobian) * mass)
    return total


def main():
    quiet()
    values = {'chern': {}}
    for d in (D, -D):
        model = honeycomb_ferromagnet(J=J, D=d, S=S)
        result = stk.solve_lswt(model, polarized_state(model), settings=stk.LSWTSettings(mesh=(30, 30)))
        values['chern'][d] = [float(c) for c in chern_numbers(result)]
        values.setdefault('dirac_count', {})[d] = dirac_count(d)
        if d == D:
            M = np.array([np.pi, np.pi / np.sqrt(3)])
            path = band_structure(result, [(0, 0), tuple(K), tuple(M), (0, 0)], points=150)
            values['band_error'] = float(np.max(np.abs(path.energies - bands(path.k_points))))
            at_k = band_structure(result, [(0, 0), tuple(K)], points=60).energies[-1]
            values['bands_K'] = [float(e) for e in at_k]
            values['gap_K'] = float(at_k[1] - at_k[0])
            values['gap_K_closed'] = 6 * np.sqrt(3) * D * S
            values['top_gamma'] = float(bands(np.zeros(2))[1])
    v = values
    print(f"bands Gamma-K-M-Gamma: max deviation {v['band_error']:.1e}")
    print(f"omega(K) = {np.round(v['bands_K'], 4)}, gap {v['gap_K']:.4f} vs 6 sqrt(3) D S = "
          f"{v['gap_K_closed']:.4f}; omega_+(Gamma) = {v['top_gamma']:.4f}")
    for d, c in v['chern'].items():
        print(f"D = {d:+.1f}: Chern numbers (lower, upper) = {c}; "
              f"Dirac count C_- = {v['dirac_count'][d]:+.0f}")
    return values


if __name__ == '__main__':
    main()
