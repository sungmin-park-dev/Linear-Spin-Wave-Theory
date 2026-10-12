"""Worked example: honeycomb ferromagnet with second-neighbour DM interaction.

Companion script to the research note *LSWT example: honeycomb topological
magnon bands* (``workbench/notes/lswt-example-honeycomb-topological-ferromagnet``).
The model is ``H = -J sum_<ij> S_i . S_j + D sum_<<ij>> nu_ij z . (S_i x S_j)``
with ``J = 1``, ``D = 0.1`` and ``S = 1/2`` about the state polarized along
+z; energies are in units of J. The script compares the package with the
closed forms of the note:

- two bands ``omega_k = 3 J S -+ S sqrt(J^2 |f_k|^2 + 4 D^2 s_k^2)``,
- the gap ``6 sqrt(3) D S`` at K,
- Chern numbers of the two bands for D = +0.1 and D = -0.1.

Run from the repository root:

    python examples/worked_examples/honeycomb_topological_ferromagnet.py
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


def main():
    quiet()
    values = {'chern': {}}
    for d in (D, -D):
        model = honeycomb_ferromagnet(J=J, D=d, S=S)
        result = stk.solve_lswt(model, polarized_state(model), settings=stk.LSWTSettings(mesh=(30, 30)))
        values['chern'][d] = [float(c) for c in chern_numbers(result)]
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
        print(f"D = {d:+.1f}: Chern numbers (lower, upper) = {c}")
    return values


if __name__ == '__main__':
    main()
