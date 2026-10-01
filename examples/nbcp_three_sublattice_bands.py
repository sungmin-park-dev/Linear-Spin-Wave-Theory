"""NBCP magnon bands in the three-sublattice phase below the transverse critical field.

Woodland et al. (arXiv:2505.06398) measured the spectra at 0.7, 1.0 and 1.4 T
along b* (below B_C ~ 1.65 T) and compare them with LSWT for a
three-sublattice coplanar structure (their Fig. 6). This script computes
those bands with the toolkit for the published XXZ parameters (Table 1) and
checks the classical reference state before the bands:

1. Classical state. A multi-start search in the sqrt3 x sqrt3 cell finds two
   stationary types: (I) one spin along B and two canted symmetrically
   towards +c and -c in the b*-c plane, and (II) two equal spins and one
   different. (I) is lower at every field checked and matches the closed form
   cos(beta) = (h / S - 3 Jxy) / (3 (Jxy + Jz)), h = g_ab mu_B B, obtained by
   minimizing E = S^2 [6 Jxy cos(beta) + 3 (Jxy cos^2(beta) - Jz sin^2(beta))]
   - h S (1 + 2 cos(beta)) per magnetic cell. beta -> 0 at
   g_ab mu_B B = 3 Jxy + 3 Jz / 2, the classical critical field of the
   polarized phase (1.717 T). (II) is a saddle: LSWT about it is unstable.
   The energy difference is small (1e-5 meV per cell at 1.5 T), so quantum
   corrections beyond LSWT could matter there; within LSWT the reference is (I).
2. Hessian. With the field along b* no continuous symmetry is left, and the
   classical Hessian of (I) has no zero eigenvalue (no accidental
   degeneracy), so LSWT has no zero mode.
3. Bands. LSWT about (I) is stable on the path at 0.7, 1.0 and 1.4 T. The
   lowest gap (at Gamma and at K, which folds onto Gamma in the magnetic
   zone) closes continuously as B -> B_C^cl from below, as the polarized
   phase gap closes from above (``nbcp_magnon_bands.py``).

The paper's data are not in the repository, so the bands are not compared
with the measurement here. Energies in meV. Figure in
``data-space/verification/261001-nbcp-three-sublattice-bands/``; JSON report
to stdout.

Usage
-----
    python examples/nbcp_three_sublattice_bands.py > report.json
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

from model import nbcp
from model.nbcp.model import PARAMETER_SETS, build_published_model
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods.classical import classical_energy, refine_classical
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.observables.bands import band_structure
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.high_symmetry import high_symmetry_points
from spintoolkit.visualization import plot_bands

OUT = ROOT / 'data-space/verification/261001-nbcp-three-sublattice-bands'
PARAMS = PARAMETER_SETS['woodland2025']['parameters']
G_AB = float(PARAMETER_SETS['woodland2025']['g'][1, 1])
S = 0.5
FIELDS = (0.7, 1.0, 1.4)
SCAN = (0.25, 0.7, 1.0, 1.4, 1.5, 1.6, 1.65, 1.68, 1.70, 1.71)
STARTS = 30


def conditions_at(field_t):
    return ExternalConditions(field=(0.0, MU_B_MEV_PER_T * field_t, 0.0))


def analytic_cos_beta(field_t):
    h = G_AB * MU_B_MEV_PER_T * field_t
    return (h / S - 3 * PARAMS['Jxy']) / (3 * (PARAMS['Jxy'] + PARAMS['Jz']))


def stationary_states(model, field_t, seed=0):
    """Refined states from random starts, grouped by energy (lowest first)."""
    rng = np.random.default_rng(seed)
    conditions = conditions_at(field_t)
    found = {}
    for _ in range(STARTS):
        angles = np.column_stack([np.arccos(rng.uniform(-1, 1, 3)), rng.uniform(0, 2 * np.pi, 3)])
        state = refine_classical(model, nbcp.candidate_state(model, 'three_msl', angles.ravel()),
                                 conditions)
        energy = classical_energy(model, state, conditions)
        found.setdefault(round(energy, 11), state)
    return [(e, found[e]) for e in sorted(found)]


def describe(state):
    d = np.array(list(state.directions.values()))
    along = int(np.argmax(d[:, 1]))
    others = np.delete(d, along, axis=0)
    one_along_field = bool(abs(d[along, 1] - 1) < 1e-8 and np.allclose(others[0, 1], others[1, 1])
                           and abs(others[0, 2] + others[1, 2]) < 1e-8 and np.allclose(d[:, 0], 0))
    return {'S_y': d[:, 1].round(6).tolist(), 'S_z': d[:, 2].round(6).tolist(),
            'type': 'I (one spin along B)' if one_along_field else 'other',
            'cos_beta': float(others[0, 1]) if one_along_field else None}


def hessian_min_eigenvalue(model, state, conditions, step=1e-4):
    keys = list(state.directions)
    d0 = np.array([state.directions[k] for k in keys])
    frames = []
    for d in d0:
        e1 = np.cross(d, [1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        frames.append((e1, np.cross(d, e1)))

    def energy(x):
        dirs = {}
        for i, k in enumerate(keys):
            v = d0[i] + x[2 * i] * frames[i][0] + x[2 * i + 1] * frames[i][1]
            dirs[k] = v / np.linalg.norm(v)
        return classical_energy(model, replace(state, directions=dirs), conditions)

    n = 2 * len(keys)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = np.eye(n)[i] * step, np.eye(n)[j] * step
            H[i, j] = (energy(ei + ej) - energy(ei - ej) - energy(ej - ei) + energy(-ei - ej)) / (4 * step ** 2)
    return float(np.linalg.eigvalsh(H)[0])


def bands_of(model, state, field_t, points=301, regularization=None):
    """Bands on M-K'-Γ-K-M. MAGSWT only lets an unstable state finish; band_structure
    uses the unregularized H(k), so unstable momenta are NaN."""
    settings = LSWTSettings(mesh=(12, 12)) if regularization is None else LSWTSettings(
        mesh=(12, 12), regularization=regularization)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = solve_lswt(model, state, conditions_at(field_t), settings=settings)
        K = high_symmetry_points(model.lattice)['K']
        bands = band_structure(result, [-1.5 * K, -K, 0 * K, K, 1.5 * K], points=points)
    return replace(bands, labels=['M', "K'", 'Γ', 'K', 'M'])


def at_label(bands, i):
    row = bands.energies[int(np.argmin(np.abs(bands.distance - bands.label_distances[i])))]
    return [None if np.isnan(x) else float(x) for x in row]


def main():
    model = build_published_model('woodland2025')
    report = {'parameters': {**PARAMS, 'g_ab': G_AB, 'S': S},
              'source': 'arXiv:2505.06398 Table 1; fields of Fig. 6 (0.7, 1.0, 1.4 T || b*)',
              'classical_critical_field_T': (3 * PARAMS['Jxy'] + 1.5 * PARAMS['Jz']) / (G_AB * MU_B_MEV_PER_T),
              'scan': []}
    references = {}
    for field_t in SCAN:
        states = stationary_states(model, field_t)
        energy, state = states[0]
        references[field_t] = state
        bands = bands_of(model, state, field_t, points=151)
        entry = {'field_T': field_t, 'ground': describe(state),
                 'analytic_cos_beta': analytic_cos_beta(field_t),
                 'hessian_min_eigenvalue': hessian_min_eigenvalue(model, state, conditions_at(field_t)),
                 'unstable_points': int(np.isnan(bands.energies).any(axis=1).sum()),
                 'lowest_energy_on_path_meV': float(np.nanmin(bands.energies)),
                 'other_stationary_states': []}
        for e, other in states[1:]:
            other_bands = bands_of(model, other, field_t, points=151, regularization='MAGSWT')
            entry['other_stationary_states'].append(
                {'energy_above_ground_meV_per_cell': e - energy, **describe(other),
                 'lswt_unstable_points': int(np.isnan(other_bands.energies).any(axis=1).sum())})
        report['scan'].append(entry)
    fig, axes = plt.subplots(1, len(FIELDS), figsize=(13, 3.9), sharey=True)
    report['bands'] = []
    for ax, field_t in zip(axes, FIELDS):
        bands = bands_of(model, references[field_t], field_t)
        report['bands'].append({'field_T': field_t, 'Gamma_meV': at_label(bands, 2),
                                'K_meV': at_label(bands, 3), 'M_meV': at_label(bands, 4)})
        plot_bands(bands, ax, energy_label='E (meV)')
        ax.set_title(f'B ∥ b*, {field_t} T')
    axes[0].set_ylim(0, 0.5)
    fig.suptitle('NBCP three-sublattice phase (one spin ∥ B, two canted ±c), LSWT, '
                 'arXiv:2505.06398 Table 1 parameters', fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    figure = OUT / 'three-sublattice-bands.png'
    fig.savefig(figure, dpi=150)
    report['figure'] = str(figure.relative_to(ROOT))
    text = json.dumps(report, indent=1, ensure_ascii=False)
    (OUT / 'report.json').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
