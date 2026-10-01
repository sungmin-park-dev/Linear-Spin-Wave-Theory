"""When can LSWT magnons of NBCP carry a thermal Hall current?

Symmetry argument. For the nearest-neighbour XXZ model with the field along
c (z), let C2(n) be the pi spin rotation about the in-plane axis n normal to
the plane of a coplanar state whose plane contains z (Y, V, up-up-down). C2(n)
leaves the XXZ exchange invariant and reverses h S^z; time reversal T reverses
it back and reverses the spins, which C2(n) restores. So T C2(n) is an
antiunitary symmetry of the Hamiltonian that fixes the state. It maps k to -k,
so Omega_n(k) = -Omega_n(-k) with E_n(k) = E_n(-k), and kappa_xy = 0 exactly.
In a transverse field the polarized state is collinear and kappa_xy = 0 for
the same reason. The bond-dependent terms J_PD and J_Gamma are not invariant
under C2(n), so they are the minimal model ingredients that allow a magnon
thermal Hall effect within LSWT.

Numerical check (this script). Y at 0.2 T and V at 1.4 T (field || c,
Jxy = 0.075, Jz = 0.125 meV, g_z = 4.645) with J_PD = J_Gamma = 0, with
J_PD = 0.01 meV and with J_Gamma = 0.01 meV; Woodland et al. polarized state
at 3.5 T || b*. Reported: the rank of the spin directions (1 collinear, 2 coplanar,
3 noncoplanar), max |Omega| on the mesh and kappa_xy / T at t = k_B T / E0 =
0.02 and 0.05 (E0 = 1 meV, about 0.23 K and 0.58 K) on a uniform 24 x 24
mesh. The nonzero values are not converged (stage 5d: uniform meshes are off
by percent near the accidental zero modes); only zero vs nonzero is the claim.
The values are those of the full-position Bloch convention returned by
``thermal_hall`` (the physical one); the cell convention, which drops the
site positions, would give other nonzero values here, even of opposite sign
(issue note 260802, 2026-10-01).

Usage
-----
    python examples/nbcp_thermal_hall_applicability.py > report.json
"""

import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from model import nbcp
from model.nbcp.model import build_published_model
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods.classical import refine_classical
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import polarized_state
from spintoolkit.observables.berry import berry_curvature, thermal_hall
from spintoolkit.system.conditions import ExternalConditions

SCAN = ROOT / 'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json'
OUT = ROOT / 'data-space/verification/261001-nbcp-thermal-hall-symmetry'
TEMPERATURES = (0.02, 0.05)
MESH = (24, 24)


def summary(state, result):
    directions = np.array(list(state.directions.values()))
    curvature = berry_curvature(result).curvature
    hall = thermal_hall(result, TEMPERATURES, gapless=True)
    sv = np.linalg.svd(directions, compute_uv=False)
    return {'direction_rank': int(np.sum(sv > 1e-9 * sv[0])),     # 1 collinear, 2 coplanar, 3 noncoplanar
            'max_abs_curvature': float(np.nanmax(np.abs(curvature))),
            'kappa_over_t': [float(x) for x in hall.kappa_over_t], 'temperatures': list(TEMPERATURES)}


def main():
    scan = json.loads(SCAN.read_text())
    report = {'mesh': list(MESH), 'cases': []}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for phase, field_t in (('Y', 0.2), ('V', 1.4)):
            for extra in ({}, {'JPD': 0.01}, {'JGamma': 0.01}):
                model = nbcp.build_model({'Jxy': 0.075, 'Jz': 0.125, **extra})
                theta = np.array(scan['states'][phase]['theta'])
                conditions = ExternalConditions(field=(0, 0, 4.645 * MU_B_MEV_PER_T * field_t))
                state = refine_classical(model, nbcp.candidate_state(
                    model, 'three_msl', np.column_stack([theta, np.zeros(3)]).ravel()), conditions)
                result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=MESH))
                report['cases'].append({'state': phase, 'field': f'{field_t} T || c',
                                        'couplings': {'Jxy': 0.075, 'Jz': 0.125, **extra},
                                        **summary(state, result)})
        model = build_published_model('woodland2025')
        conditions = ExternalConditions(field=(0, MU_B_MEV_PER_T * 3.5, 0))
        state = polarized_state(model, (0, 1, 0))
        result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=MESH))
        report['cases'].append({'state': 'polarized', 'field': '3.5 T || b*',
                                'couplings': 'woodland2025', **summary(state, result)})
    text = json.dumps(report, indent=1, ensure_ascii=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
