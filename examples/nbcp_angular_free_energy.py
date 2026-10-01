"""Angular free energy f(phi, T) of the NBCP Y and V states (D39).

For the Y (B = 0.2 T) and V (B = 1.4 T) three-sublattice states with
J_PD = 0.010 meV, evaluates along the relaxed orbit about z

    f(phi, T) = E_cl(phi) + E_zp(phi) + (T/N) sum ln(1 - exp(-eps/T))

with ``LSWTHarmonicFreeEnergy`` (harmonic, 1/S order) and fits its sixfold
harmonic. ``soft_cutoff = 0`` is the full harmonic free energy; a nonzero
cutoff leaves the lowest mode at |k| < cutoff to the phase theory (clock RG).
The cutoff is a matching choice, so several are reported (0.25 is about
the long-wavelength range of the phase theory; 1.0 goes beyond it). Energies and T in meV
(1 K = 0.08617 meV).

Usage
-----
    python examples/nbcp_angular_free_energy.py > report.json
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from model import nbcp
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods import state_selection as sel
from spintoolkit.system.conditions import ExternalConditions

SCAN = ROOT / 'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json'
G_Z, J, JZ, JPD = 4.645, 0.075, 0.125, 0.010
PHASES = {'Y': 0.2, 'V': 1.4}
TEMPERATURES = [0.0, 0.0025, 0.005, 0.01, 0.015, 0.02]
CUTOFFS = [0.0, 0.25, 1.0]
MESH_N = 24
PHIS = np.linspace(0.0, 2 * np.pi, 36, endpoint=False)


def main():
    scan = json.loads(SCAN.read_text())
    model = nbcp.build_model({'Jxy': J, 'Jz': JZ, 'JPD': JPD})
    report = {'model': {'J': J, 'Jz': JZ, 'JPD': JPD, 'g_z': G_Z}, 'mesh_N': MESH_N,
              'orbit_points': len(PHIS), 'units': 'meV per spin; T in meV', 'cases': []}
    for phase, field_T in PHASES.items():
        conditions = ExternalConditions(field=[0, 0, G_Z * MU_B_MEV_PER_T * field_T])
        theta = np.array(scan['states'][phase]['theta'])
        state = nbcp.candidate_state(model, 'three_msl',
                                     np.column_stack([theta, np.full(3, 0.3)]).ravel())
        for cutoff in CUTOFFS:
            for T in TEMPERATURES:
                if T == 0.0 and cutoff > 0:
                    continue
                provider = (sel.lswt_zero_point_energy('Hex_30', MESH_N) if T == 0.0 else
                            sel.LSWTHarmonicFreeEnergy('Hex_30', MESH_N, T, soft_cutoff=cutoff))
                landscape = sel.orbit_energy_landscape(model, state, conditions, provider, phis=PHIS)
                total = landscape.classical + landscape.quantum
                fit = sel.fit_harmonics(landscape.phi, total, 12)
                a6, b6 = fit['cos'][5], fit['sin'][5]
                described = provider.describe()
                report['cases'].append({
                    'phase': phase, 'field_T': field_T, 'T_meV': T, 'soft_cutoff': cutoff,
                    'sixfold_amplitude': float(np.hypot(a6, b6)),
                    'sixfold_minimum_phi': float(np.mod(np.arctan2(-b6, -a6) / 6, np.pi / 3)),
                    'E_cl_span': float(np.ptp(landscape.classical)),
                    'fit_residual': fit['residual'],
                    'undefined_calls': described.get('undefined_calls', 0),
                    'max_unstable_pairs': described.get('max_unstable_pairs'),
                    'soft_modes_left_out': described.get('soft_modes_left_out')})
                print(phase, cutoff, T, report['cases'][-1]['sixfold_amplitude'], file=sys.stderr)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
