"""Verify stage 7: band and spin-configuration figures draw the stored numbers.

1. Square Neel and triangular 120 degrees: every plotted band line equals
   ``BandStructure.energies`` exactly, zero-mode markers sit at the marked
   momenta at E = 0.
2. An unstable reference state: the antiferromagnetic square model polarized
   by a field below saturation (h = 3 < h_sat = 2 z J S = 4). Its dispersion
   h - 2 z J S (1 - gamma_k)/2 = h - 2 (1 - gamma_k) is negative where
   gamma_k < -1/2 (around M); there the energies are NaN, stay gaps and are
   shaded, never interpolated. The stable part is checked against the formula.
3. Spin configuration of the 120 degree state from (SpinModel, SpinState).

Figures are written to ``docs/development/figures/``; the JSON report to stdout.

Usage
-----
    python examples/visualization_stage7_check.py > report.json
"""

import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.observables.bands import band_structure
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.visualization import plot_bands, plot_spin_configuration

FIGURES = ROOT / 'docs/development/figures'


def drawn_equals_stored(ax, bands):
    lines = [l for l in ax.get_lines() if l.get_linestyle() == '-']
    same = len(lines) == bands.energies.shape[1] and all(
        np.array_equal(l.get_ydata(), bands.energies[:, n], equal_nan=True)
        and np.array_equal(l.get_xdata(), bands.distance) for n, l in enumerate(lines))
    markers = [l for l in ax.get_lines() if l.get_label() == 'zero mode']
    marked = markers[0].get_xdata().tolist() if markers else []
    return {'lines_equal_stored_energies': bool(same), 'bands': len(lines),
            'zero_mode_markers': len(marked),
            'markers_at_marked_momenta': bool(np.array_equal(marked, bands.distance[bands.zero_modes])),
            'unstable_points': int(np.isnan(bands.energies).any(axis=1).sum()),
            'shaded_intervals': len(ax.patches)}


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    report = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    square = square_heisenberg()
    bands = band_structure(solve_lswt(square, neel_state(square), None, settings=LSWTSettings(mesh=(4, 4))),
                           ('Γ', 'X', 'M', 'Γ'), points=300)
    report['square_neel'] = drawn_equals_stored(plot_bands(bands, axes[0]), bands)
    axes[0].set_title('square Néel, J = 1, S = 1/2')

    tri = triangular_heisenberg()
    bands = band_structure(solve_lswt(tri, state_120(tri), None, settings=LSWTSettings(mesh=(6, 6))),
                           ('Γ', 'K', 'M', 'Γ'), points=300)
    report['triangular_120'] = drawn_equals_stored(plot_bands(bands, axes[1]), bands)
    axes[1].set_title('triangular 120° (folded, 3 bands)')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = solve_lswt(square, polarized_state(square), ExternalConditions(field=(0, 0, 3.0)),
                            settings=LSWTSettings(mesh=(4, 4), regularization='MAGSWT'))
        bands = band_structure(result, ('Γ', 'X', 'M', 'Γ'), points=300)
    entry = drawn_equals_stored(plot_bands(bands, axes[2]), bands)
    gamma = 0.5 * (np.cos(bands.k_points[:, 0]) + np.cos(bands.k_points[:, 1]))
    omega = 3.0 - 2 * (1 - gamma)
    stable = ~np.isnan(bands.energies[:, 0])
    entry['max_deviation_from_formula_where_stable'] = float(np.max(np.abs(bands.energies[stable, 0] - omega[stable])))
    entry['nan_exactly_where_formula_negative'] = bool(np.array_equal(~stable, omega < -1e-12))
    entry['warnings'] = sorted({str(w.message)[:80] for w in caught})
    report['polarized_below_saturation_h3'] = entry
    axes[2].set_title('unstable: polarized AFM, h = 3 < 4')
    fig.tight_layout()
    fig.savefig(FIGURES / 'stage7-bands.png', dpi=150)

    fig, _ = plot_spin_configuration(tri, state_120(tri), n_repeat=1, figsize=(5, 5),
                                     title='triangular 120°')
    fig.savefig(FIGURES / 'stage7-spin-configuration.png', dpi=150)
    report['figures'] = ['docs/development/figures/stage7-bands.png',
                         'docs/development/figures/stage7-spin-configuration.png']
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
