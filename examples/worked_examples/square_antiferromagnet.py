"""Worked example: square-lattice Heisenberg antiferromagnet.

Companion script to the research note *LSWT example: square-lattice
antiferromagnet* (``workbench/notes/lswt-example-square-antiferromagnet``).
The model is ``H = J sum_<ij> S_i . S_j`` with ``J = 1`` about the Neel state;
energies are in units of J. The script compares the package with the closed
forms of the note:

- dispersion ``omega_k = 4 J S sqrt(1 - gamma_k^2)`` along Gamma-X-M-Gamma,
- ground-state energy ``-2 J S (S + c)`` with ``c = 1 - <sqrt(1 - gamma_k^2)>``,
- moment reduction ``Delta S = <1 / sqrt(1 - gamma_k^2)> / 2 - 1 / 2``,
- spin-wave velocity ``2 sqrt(2) J S``,

for S = 1/2 and S = 1. The moment converges as 1/n with the mesh size n
because of the Goldstone mode; the package value on the 96 x 96 mesh is
extrapolated from the meshes 48 and 96 in the same way as the closed form.

Run from the repository root:

    python examples/worked_examples/square_antiferromagnet.py
"""

from _common import converged_average, quiet

import numpy as np

import spintoolkit as stk
from spintoolkit.models import neel_state, square_heisenberg
from spintoolkit.observables.bands import band_structure

J = 1.0
SPINS = (0.5, 1.0)
MESH = 96


def gamma(x1, x2):
    return 0.5 * (np.cos(x1) + np.cos(x2))


def dispersion(k, S):
    return 4 * J * S * np.sqrt(np.clip(1 - gamma(k[..., 0], k[..., 1]) ** 2, 0, None))


def constants():
    c = converged_average(lambda x1, x2: 1 - np.sqrt(1 - gamma(x1, x2) ** 2))
    reduction = converged_average(lambda x1, x2: 0.5 / np.sqrt(1 - gamma(x1, x2) ** 2) - 0.5)
    return c, reduction


def main():
    quiet()
    c, reduction = constants()
    print(f"c = {c:.6f}, Delta S = {reduction:.6f}")
    values = {'c': c, 'moment_reduction': reduction, 'spins': {}}
    for S in SPINS:
        model = square_heisenberg(J=J, S=S)
        result = stk.solve_lswt(model, neel_state(model), settings=stk.LSWTSettings(mesh=(MESH, MESH)))
        path = [(0, 0), (np.pi, 0), (np.pi, np.pi), (0, 0)]
        bands = band_structure(result, path, points=120)
        closed = dispersion(bands.k_points, S)
        band_error = float(np.max(np.abs(bands.energies - closed[:, None])))   # two folded copies
        q = 1e-3
        velocity = float(band_structure(result, [(0, 0), (q, 0)], points=2).energies[-1, 0] / q)
        half = stk.solve_lswt(model, neel_state(model),
                              settings=stk.LSWTSettings(mesh=(MESH // 2, MESH // 2)))
        moment = float(result.ordered_moments().mean())
        values['spins'][S] = {
            'classical_energy': result.classical_energy,
            'ground_state_energy': result.ground_state_energy,
            'ground_state_closed': -2 * J * S * (S + c),
            'moment': moment,
            'moment_extrapolated': 2 * moment - float(half.ordered_moments().mean()),
            'moment_closed': S - reduction,
            'band_error': band_error,
            'zone_boundary': float(dispersion(np.array([np.pi, 0]), S)),
            'velocity': velocity,
            'velocity_closed': 2 * np.sqrt(2) * J * S,
        }
        v = values['spins'][S]
        print(f"S = {S}: E_gs = {v['ground_state_energy']:.6f} (package), "
              f"{v['ground_state_closed']:.6f} (closed form)")
        print(f"  <S> = {v['moment']:.4f} ({MESH} x {MESH} mesh), {v['moment_extrapolated']:.4f} "
              f"(meshes {MESH // 2} and {MESH} extrapolated), {v['moment_closed']:.4f} (closed form)")
        print(f"  bands: max deviation {band_error:.1e}; omega(X) = {v['zone_boundary']:.4f}; "
              f"velocity {velocity:.4f} vs {v['velocity_closed']:.4f}")
    return values


if __name__ == '__main__':
    main()
