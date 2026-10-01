"""NBCP magnon bands in the transverse-field polarized phase vs arXiv:2505.06398.

Woodland et al. (arXiv:2505.06398) fit the nearest-neighbour XXZ model to
neutron spectra (Table 1: Jz = 0.1225 meV, Jxy = 0.0779 meV, g_ab = 4.200,
g_c = 4.716) and give the linear-spin-wave dispersion of the state polarized
by a field B along b* (perpendicular to the Ising axis c), their Eq. (5):

    hbar omega(Q) = 2S sqrt(A^2 - B^2),
    A = g_ab mu_B B - 3 Jxy + (Jz + Jxy)/2 gamma(Q),
    B = (Jz - Jxy)/2 gamma(Q),
    gamma(Q) = sum over the three bond directions of cos(Q . delta).

This script computes the same bands with the toolkit (SpinModel ->
solve_lswt -> band_structure -> plot_bands) and checks:

1. the bands against Eq. (5) on the path M-K'-Γ-K-M (the straight line
   through Γ and K used in the paper's Figs. 4 and 6),
2. the gap at 3.5 T against the quoted ~0.46 meV,
3. the classical critical field: the polarized state is stable only for
   g_ab mu_B B > 3 Jxy + 3 Jz / 2 (A = |B| at K), which the paper quotes as
   B_C^cl = 1.72 T, against the field where the toolkit first finds an
   unstable momentum,
4. 1.7 T, where the measured spectra were taken: 1.7 T is below B_C^cl, so
   the polarized reference state is classically unstable near K at the
   nominal field. The paper describes the 1.7 T data with an empirical
   offset B_y -> B_y + dB, dB = 0.041 T, that parameterizes quantum
   renormalization of the dispersion just above B_C (Table 1); it is not a
   model parameter. At 1.741 T the LSWT spectrum is stable with a small gap
   at K. Both are shown.

Energies are in meV (E0 of this model). Figure:
``data-space/verification/261001-nbcp-transverse-bands/``; JSON report to stdout.

Usage
-----
    python examples/nbcp_magnon_bands.py > report.json
"""

from dataclasses import replace
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

from model.nbcp.model import LATTICE, PARAMETER_SETS, build_published_model
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import polarized_state
from spintoolkit.observables.bands import band_structure
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.high_symmetry import high_symmetry_points
from spintoolkit.visualization import plot_bands

OUT = ROOT / 'data-space/verification/261001-nbcp-transverse-bands'
FIGURE = OUT / 'transverse-polarized-bands.png'
PARAMS = PARAMETER_SETS['woodland2025']['parameters']
G_AB = float(PARAMETER_SETS['woodland2025']['g'][1, 1])
S = 0.5
FIELD_OFFSET_T = 0.041
PAPER_GAP_35T = 0.46


def gamma(k):
    deltas = np.array([LATTICE[0], LATTICE[1], LATTICE[0] + LATTICE[1]])
    return np.cos(k @ deltas.T).sum(axis=1)


def eq5(k, field_t):
    jxy, jz = PARAMS['Jxy'], PARAMS['Jz']
    g = gamma(k)
    a = G_AB * MU_B_MEV_PER_T * field_t - 3 * jxy + (jz + jxy) / 2 * g
    b = (jz - jxy) / 2 * g
    w2 = a ** 2 - b ** 2
    return np.where(w2 >= 0, 2 * S * np.sqrt(np.clip(w2, 0, None)), np.nan)


def bands_at(model, field_t, points=301):
    conditions = ExternalConditions(field=(0.0, MU_B_MEV_PER_T * field_t, 0.0))
    state = polarized_state(model, (0, 1, 0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        # MAGSWT only lets solve_lswt finish on an unstable mesh; band_structure
        # always uses the unregularized H(k), so unstable momenta stay NaN.
        result = solve_lswt(model, state, conditions,
                            settings=LSWTSettings(mesh=(12, 12), regularization='MAGSWT'))
        K = high_symmetry_points(model.lattice)['K']
        bands = band_structure(result, [-1.5 * K, -K, 0 * K, K, 1.5 * K], points=points)
    bands = replace(bands, labels=['M', "K'", 'Γ', 'K', 'M'])
    messages = sorted({str(w.message)[:90] for w in caught if 'unstable' in str(w.message)})
    return result, bands, messages


def check(bands, field_t):
    ref = eq5(bands.k_points, field_t)
    computed = bands.energies[:, 0]
    stable = ~np.isnan(computed)
    i_k = int(np.argmin(np.abs(bands.distance - bands.label_distances[3])))
    return {'field_T': field_t,
            'max_deviation_from_eq5_meV': float(np.nanmax(np.abs(computed[stable] - ref[stable])))
            if stable.any() else None,
            'unstable_points': int((~stable).sum()), 'points': len(stable),
            'nan_exactly_where_eq5_imaginary': bool(np.array_equal(~stable, np.isnan(ref))),
            'energy_at_K_meV': None if np.isnan(computed[i_k]) else float(computed[i_k]),
            'energy_at_Gamma_meV': float(computed[int(np.argmin(np.abs(bands.distance - bands.label_distances[2])))]),
            'minimum_on_path_meV': float(np.nanmin(computed)) if stable.any() else None}


def critical_field(model):
    """Smallest field (bisection, 1e-6 T) at which no momentum of the path is unstable."""
    lo, hi = 1.0, 3.0
    while hi - lo > 1e-6:
        mid = 0.5 * (lo + hi)
        _, bands, _ = bands_at(model, mid, points=121)
        lo, hi = (mid, hi) if np.isnan(bands.energies).any() else (lo, mid)
    return hi


def main():
    model = build_published_model('woodland2025')
    report = {'parameters': {**PARAMS, 'g_ab': G_AB, 'S': S},
              'source': 'arXiv:2505.06398 Table 1, Eq. (5), Figs. 4 and 6 (path M-K\'-Γ-K-M)'}
    analytic_bc = (3 * PARAMS['Jxy'] + 1.5 * PARAMS['Jz']) / (G_AB * MU_B_MEV_PER_T)
    report['critical_field'] = {'analytic_T': analytic_bc, 'paper_classical_T': 1.72,
                                'toolkit_first_stable_T': critical_field(model),
                                'paper_measured_T': 1.65}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    cases = [(3.5, '3.5 T'), (1.7 + FIELD_OFFSET_T, '1.7 T + δB (empirical, 0.041 T)'), (1.7, '1.7 T nominal')]
    report['cases'] = []
    for ax, (field_t, title) in zip(axes, cases):
        _, bands, messages = bands_at(model, field_t)
        entry = check(bands, field_t)
        entry['warnings'] = messages
        report['cases'].append(entry)
        plot_bands(bands, ax, energy_label='E (meV)')
        ax.plot(bands.distance, eq5(bands.k_points, field_t), 'k--', lw=0.9,
                label='Eq. (5), arXiv:2505.06398')
        ax.set_title(f'B ∥ b*, {title}')
        if ax is axes[0]:
            ax.legend(loc='upper center', fontsize=8)
    report['cases'][0]['paper_gap_meV'] = PAPER_GAP_35T
    axes[0].set_ylim(0, 1.1)
    fig.suptitle('NBCP polarized phase, transverse field: toolkit LSWT (solid) vs paper Eq. (5) (dashed)',
                 fontsize=10)
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=150)
    report['figure'] = str(FIGURE.relative_to(ROOT))
    text = json.dumps(report, indent=1, ensure_ascii=False)
    (OUT / 'report.json').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
