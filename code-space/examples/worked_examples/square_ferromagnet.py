"""Worked example: square-lattice Heisenberg ferromagnet in a field.

Companion script to the research note *LSWT example: square-lattice
ferromagnet* (``workbench/topics/worked-examples/lswt-example-square-ferromagnet``). The model
is ``H = -J sum_<ij> S_i . S_j - h sum_i S_i^z`` with ``J = 1``, ``S = 1`` and
``h = 0.1``; energies are in units of J. The script compares the package with
the closed forms of the note:

- classical energy ``-2 J S^2 - h S`` and zero zero-point energy,
- dispersion ``omega_k = h + 4 J S (1 - gamma_k)`` along Gamma-X-M-Gamma,
- thermal reduction of the moment, lattice sum and its low-temperature form
  ``-(T / 4 pi J S) ln(1 - exp(-h / T))``.

Run from the repository root:

    python code-space/examples/worked_examples/square_ferromagnet.py
"""

from _common import quiet, zone_average

import numpy as np

import spintoolkit as stk
from spintoolkit.models import polarized_state, square_heisenberg
from spintoolkit.observables.bands import band_structure
from spintoolkit.observables.thermal import thermal_quantities
from spintoolkit.system.conditions import ExternalConditions

J, S, H = 1.0, 1.0, 0.1
TEMPERATURES = (0.05, 0.1)


def dispersion(k):
    gamma = 0.5 * (np.cos(k[..., 0]) + np.cos(k[..., 1]))
    return H + 4 * J * S * (1 - gamma)


def moment_reduction(t, n=96):
    """Lattice sum <n_B(omega_k)> on the mesh of the package calculation."""
    return zone_average(lambda x1, x2: 1 / np.expm1(dispersion(np.stack([x1, x2], -1)) / t), n)


def moment_reduction_low_t(t):
    return -t / (4 * np.pi * J * S) * np.log(1 - np.exp(-H / t))


def main():
    quiet()
    model = square_heisenberg(J=-J, S=S)               # the package sign: J < 0 is ferromagnetic
    state = polarized_state(model, (0, 0, 1))
    conditions = ExternalConditions(field=[0, 0, H])
    result = stk.solve_lswt(model, state, conditions, settings=stk.LSWTSettings(mesh=(96, 96)))
    path = [(0, 0), (np.pi, 0), (np.pi, np.pi), (0, 0)]
    bands = band_structure(result, path, points=120)
    band_error = float(np.max(np.abs(bands.energies[:, 0] - dispersion(bands.k_points))))
    thermal = thermal_quantities(result, TEMPERATURES, gapless=False)
    values = {
        'classical_energy': result.classical_energy,
        'zero_point_energy': result.zero_point_energy,
        'band_error': band_error,
        'gap': float(dispersion(np.zeros(2))),
        'zone_corner': float(dispersion(np.array([np.pi, np.pi]))),
        'moment_reduction': [S - m for m in thermal.moments[:, 0]],
        'moment_reduction_sum': [moment_reduction(t) for t in TEMPERATURES],
        'moment_reduction_low_t': [moment_reduction_low_t(t) for t in TEMPERATURES],
    }
    print(f"E_cl = {values['classical_energy']:.4f} J per spin, E_zp = {values['zero_point_energy']:.1e}")
    print(f"dispersion along Gamma-X-M-Gamma: max deviation from closed form {band_error:.1e}")
    print(f"omega(Gamma) = {values['gap']:.4f}, omega(M) = {values['zone_corner']:.4f}")
    for t, a, b, c in zip(TEMPERATURES, *(values[k] for k in
                          ('moment_reduction', 'moment_reduction_sum', 'moment_reduction_low_t'))):
        print(f"T = {t}: S - <S^z> = {a:.3e} (package), {b:.3e} (lattice sum), {c:.3e} (low T)")
    return values


if __name__ == '__main__':
    main()
