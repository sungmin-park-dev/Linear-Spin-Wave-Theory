"""Verify stage 4d: high-symmetry points and magnon band structure (D33).

1. Analytic dispersions: square Neel, square ferromagnet in a field,
   triangular 120 degrees (folded single-q dispersion); zero modes at the
   Goldstone vertices.
2. NBCP Y (0.2 T) and V (1.4 T) with J_PD or J_Gamma = 0.01 meV, and the
   example configuration (Four MSL): magnon energies at the vertices of the
   primitive path Γ-K-M-Γ, the lowest energy on the path and the zero-mode
   points. Numbers only; figures belong to the visualization stage.

Usage
-----
    python examples/lswt_stage4d_check.py > report.json
"""

import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from model import nbcp
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods.classical import classical_search, refine_classical
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.observables.bands import band_structure
from spintoolkit.system.conditions import ExternalConditions

SCAN = ROOT / 'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json'
TRIANGULAR = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
S = 0.5


def deviations(bands, analytic):
    d = np.max(np.abs(bands.energies - analytic), axis=1)
    return {'max_away_from_zero_modes': float(np.max(d[~bands.zero_modes])),
            'max_at_zero_modes': float(np.max(d[bands.zero_modes], initial=0.0)),
            'zero_mode_points': int(bands.zero_modes.sum()), 'points': len(d)}


def analytic_checks():
    out = {}
    model = square_heisenberg()
    bands = band_structure(solve_lswt(model, neel_state(model), None, settings=LSWTSettings(mesh=(4, 4))),
                           ('Γ', 'X', 'M', 'Γ'), points=300)
    gamma = 0.5 * (np.cos(bands.k_points[:, 0]) + np.cos(bands.k_points[:, 1]))
    out['square_neel'] = deviations(bands, 4 * S * np.sqrt(1 - gamma ** 2)[:, None])
    ferro = square_heisenberg(J=-1.0)
    bands = band_structure(solve_lswt(ferro, polarized_state(ferro), ExternalConditions(field=(0, 0, 0.3)),
                                      settings=LSWTSettings(mesh=(4, 4))), ('Γ', 'X', 'M', 'Γ'), points=300)
    gamma = 0.5 * (np.cos(bands.k_points[:, 0]) + np.cos(bands.k_points[:, 1]))
    out['square_ferromagnet_h0.3'] = deviations(bands, (4 * S * (1 - gamma) + 0.3)[:, None])
    tri = triangular_heisenberg()
    bands = band_structure(solve_lswt(tri, state_120(tri), None, settings=LSWTSettings(mesh=(6, 6))),
                           ('Γ', 'K', 'M', 'Γ'), points=300)

    def omega(k):
        g = (np.cos(k @ TRIANGULAR[0]) + np.cos(k @ TRIANGULAR[1]) + np.cos(k @ (TRIANGULAR[1] - TRIANGULAR[0]))) / 3
        return 3 * S * np.sqrt(np.clip((1 - g) * (1 + 2 * g), 0, None))

    Q = np.array([4 * np.pi / 3, 0])
    k = bands.k_points
    out['triangular_120'] = deviations(
        bands, np.sort(np.column_stack([omega(k), omega(k + Q), omega(k - Q)]), axis=1))
    return out


def vertex_summary(bands):
    rows = {}
    for label, distance in zip(bands.labels, bands.label_distances):
        i = int(np.argmin(np.abs(bands.distance - distance)))
        rows.setdefault(label, bands.energies[i].tolist())
    return {'vertices': rows, 'lowest_on_path': float(np.nanmin(bands.energies)),
            'zero_mode_points': [bands.labels[j] for j, d in enumerate(bands.label_distances)
                                 if bands.zero_modes[int(np.argmin(np.abs(bands.distance - d)))]]}


def nbcp_cases():
    scan = json.loads(SCAN.read_text())
    out = []
    for phase, field, extra in (('Y', 0.2, {'JPD': 0.01}), ('Y', 0.2, {'JGamma': 0.01}),
                                ('V', 1.4, {'JPD': 0.01}), ('V', 1.4, {'JGamma': 0.01})):
        model = nbcp.build_model({'Jxy': 0.075, 'Jz': 0.125, **extra})
        theta = np.array(scan['states'][phase]['theta'])
        conditions = ExternalConditions(field=(0, 0, 4.645 * MU_B_MEV_PER_T * field))
        state = refine_classical(model, nbcp.candidate_state(
            model, 'three_msl', np.column_stack([theta, np.zeros(3)]).ravel()), conditions)
        result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(12, 12)))
        out.append({'phase': phase, 'field_T': field, 'couplings': extra,
                    **vertex_summary(band_structure(result, ('Γ', 'K', 'M', 'Γ'), points=300))})
    model = nbcp.build_model({'Jxy': 0.076, 'Jz': 0.125, 'JGamma': 0.1})
    conditions = ExternalConditions(field=(0, 0, 0.376418))
    ground = classical_search(model, nbcp.SUPERCELLS['four_msl'], conditions)
    result = solve_lswt(model, ground.state, conditions, settings=LSWTSettings(mesh=(12, 12)))
    out.append({'phase': 'Four MSL (example configuration)', 'field_meV': 0.376418,
                'couplings': {'Jxy': 0.076, 'Jz': 0.125, 'JGamma': 0.1},
                **vertex_summary(band_structure(result, ('Γ', 'K', 'M', 'Γ'), points=300))})
    return out


def main():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        report = {'analytic': analytic_checks(), 'nbcp': nbcp_cases()}
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()
