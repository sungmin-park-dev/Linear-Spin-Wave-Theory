"""Worked example: triangular-lattice Heisenberg antiferromagnet, 120-degree order.

Companion script to the research note *LSWT example: triangular-lattice
antiferromagnet* (``workbench/topics/worked-examples/lswt-example-triangular-antiferromagnet``).
The model is ``H = J sum_<ij> S_i . S_j`` with ``J = 1`` and ``S = 1/2`` about
the coplanar 120-degree state on the three-site magnetic cell; energies are in
units of J. The script compares the package with the closed forms of the note:

- dispersion ``omega_k = 3 J S sqrt((1 - gamma_k)(1 + 2 gamma_k))``; on the
  magnetic cell the package returns the three folded copies ``omega_k``,
  ``omega_{k+Q}`` and ``omega_{k-Q}`` with ``Q = K = (4 pi / 3, 0)``,
- ground-state energy ``-3/2 J S^2 - 3/2 J S + 3/2 J S <sqrt(...)>``,
- moment reduction ``<(1 + gamma_k / 2) / sqrt(...)> / 2 - 1 / 2``,
- velocities of the Goldstone modes at Gamma and K.

Run from the repository root:

    python code-space/examples/worked_examples/triangular_antiferromagnet.py
"""

from _common import converged_average, quiet

import numpy as np

import spintoolkit as stk
from spintoolkit.models import state_120, triangular_heisenberg
from spintoolkit.observables.bands import band_structure

J, S = 1.0, 0.5
MESH = 96
A1, A2 = np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])
Q = np.array([4 * np.pi / 3, 0.0])


def gamma_fractional(x1, x2):
    """gamma_k with x_i = k . a_i; the third neighbour is a2 - a1."""
    return (np.cos(x1) + np.cos(x2) + np.cos(x2 - x1)) / 3


def dispersion(k):
    g = gamma_fractional(k @ A1, k @ A2)
    return 3 * J * S * np.sqrt(np.clip((1 - g) * (1 + 2 * g), 0, None))


def constants():
    root = lambda x1, x2: np.sqrt(np.abs((1 - gamma_fractional(x1, x2)) * (1 + 2 * gamma_fractional(x1, x2))))
    mean_root = converged_average(root)
    reduction = converged_average(lambda x1, x2: 0.5 * (1 + gamma_fractional(x1, x2) / 2)
                                  / root(x1, x2) - 0.5)
    return mean_root, reduction


def slopes(result, q=1e-3):
    """The three magnon energies at momenta +-q along x, divided by q and averaged.

    Gamma and +-K are all reciprocal vectors of the magnetic cell, so the three
    folded bands start at zero energy at Gamma: the copies omega_{k+-Q} carry the
    Goldstone modes at +-K (the two lower slopes, averaged by the caller), omega_k
    the one at Gamma.
    The average over +-q removes the O(q) correction from the cubic term of
    gamma_k about K.
    """
    ends = [band_structure(result, [(0.0, 0.0), (x, 0.0)], points=2).energies[-1] for x in (q, -q)]
    return (ends[0] + ends[1]) / (2 * q)


def main():
    quiet()
    mean_root, reduction = constants()
    model = triangular_heisenberg(J=J, S=S)
    state = state_120(model)
    result = stk.solve_lswt(model, state, settings=stk.LSWTSettings(mesh=(MESH, MESH)))
    half = stk.solve_lswt(model, state, settings=stk.LSWTSettings(mesh=(MESH // 2, MESH // 2)))
    M = np.array([np.pi, np.pi / np.sqrt(3)])
    bands = band_structure(result, [(0, 0), tuple(Q), tuple(M), (0, 0)], points=150)
    k = bands.k_points
    folded = np.sort(np.stack([dispersion(k), dispersion(k + Q), dispersion(k - Q)], 1), 1)
    moment = float(result.ordered_moments().mean())
    values = {
        'mean_root': mean_root,
        'moment_reduction': reduction,
        'classical_energy': result.classical_energy,
        'ground_state_energy': result.ground_state_energy,
        'ground_state_closed': -1.5 * J * S ** 2 - 1.5 * J * S + 1.5 * J * S * mean_root,
        'moment': moment,
        'moment_extrapolated': 2 * moment - float(half.ordered_moments().mean()),
        'moment_closed': S - reduction,
        'band_error': float(np.max(np.abs(bands.energies - folded))),
        'omega_M': float(dispersion(M)),
        'velocity_gamma': float(slopes(result)[2]),
        'velocity_gamma_closed': 3 * np.sqrt(3) / 2 * J * S,
        'velocity_K': float(slopes(result)[:2].mean()),
        'velocity_K_closed': 3 * np.sqrt(3) / (2 * np.sqrt(2)) * J * S,
    }
    v = values
    print(f"<sqrt((1 - gamma)(1 + 2 gamma))> = {mean_root:.6f}, Delta S = {reduction:.6f}")
    print(f"E_cl = {v['classical_energy']:.4f}, E_gs = {v['ground_state_energy']:.6f} (package), "
          f"{v['ground_state_closed']:.6f} (closed form)")
    print(f"<S> = {moment:.4f} ({MESH} x {MESH} mesh), {v['moment_extrapolated']:.4f} "
          f"(meshes {MESH // 2} and {MESH} extrapolated), {v['moment_closed']:.4f} (closed form)")
    print(f"bands Gamma-K-M-Gamma: max deviation {v['band_error']:.1e}; omega(M) = {v['omega_M']:.4f}")
    print(f"velocity at Gamma {v['velocity_gamma']:.4f} vs {v['velocity_gamma_closed']:.4f}, "
          f"at K {v['velocity_K']:.4f} vs {v['velocity_K_closed']:.4f}")
    return values


if __name__ == '__main__':
    main()
