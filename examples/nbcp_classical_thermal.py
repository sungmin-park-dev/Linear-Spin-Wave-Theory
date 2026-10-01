"""Classical finite-temperature scan of the NBCP Y state (D40).

Classical spins of length 1/2 with the NBCP nearest-neighbour XXZ model
(J = 0.075, J_z = 0.125, J_PD = 0.010 meV, g_z = 4.645) at B = 0.2 T on L x L
tori, sampled by Monte Carlo (``spintoolkit.methods.monte_carlo``), annealed
from the Y state downwards in T. Measured per spin:

- energy and specific heat;
- density order: the longitudinal three-sublattice amplitude m_K^z, with the
  Binder ratio U = 1 - <|m|^4> / (2 <|m|^2>^2) of this complex order;
- phase sector: the transverse amplitudes m_K^+- = m_K^x +- i m_K^y, the
  helicity modulus of the U(1)-symmetric part of the exchange (twist about z
  along x; J_PD is left out of the stiffness, see D40), compared with the
  universal BKT value 2T/pi, and the sixfold clock order
  <cos 6 phi> = <Re psi^3 / |psi|^3> with psi = m_K^+ conj(m_K^-), which
  is invariant under lattice translations and turns as exp(2 i a) under a
  spin rotation by a about z.

These are classical results: the T = 0 sixfold pinning of NBCP is a quantum
zero-point effect (zero classically for J_PD), so the classical clock order
tests only thermal order by disorder. T in meV (1 K = 0.08617 meV).

Usage
-----
    python examples/nbcp_classical_thermal.py > report.json
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from model import nbcp
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods.dynamics import ClassicalTorus
from spintoolkit.methods.monte_carlo import MonteCarlo, thermal_averages
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry

SCAN = ROOT / 'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json'
G_Z, J, JZ, JPD, FIELD_T = 4.645, 0.075, 0.125, 0.010, 0.2
SIZES = [12, 18, 24]
TEMPERATURES = [0.03, 0.025, 0.02, 0.0175, 0.015, 0.0125, 0.01, 0.008, 0.006, 0.005, 0.004,
                0.0035, 0.003, 0.0025, 0.002, 0.0015, 0.001]
THERMALIZATION, MEASUREMENTS, SWEEPS_BETWEEN = 1000, 3000, 2


def binder(values):
    a2 = np.mean(np.abs(values) ** 2)
    return 1 - np.mean(np.abs(values) ** 4) / (2 * a2 ** 2)


def main():
    scan = json.loads(SCAN.read_text())
    model = nbcp.build_model({'Jxy': J, 'Jz': JZ, 'JPD': JPD})
    conditions = ExternalConditions(field=[0, 0, G_Z * MU_B_MEV_PER_T * FIELD_T])
    theta = np.array(scan['states']['Y']['theta'])
    state = nbcp.candidate_state(model, 'three_msl', np.column_stack([theta, np.zeros(3)]).ravel())
    A = np.asarray(model.lattice, dtype=float)
    B = 2 * np.pi * np.linalg.inv(A).T
    K = (B[0] + B[1]) / 3
    report = {'model': {'J': J, 'Jz': JZ, 'JPD': JPD, 'g_z': G_Z, 'field_T': FIELD_T, 'S': 0.5},
              'sweeps': {'thermalization': THERMALIZATION, 'measurements': MEASUREMENTS,
                         'between': SWEEPS_BETWEEN},
              'units': 'meV per spin; T in meV', 'rows': []}
    for L in SIZES:
        torus = ClassicalTorus(model, CalculationGeometry.finite_torus([[L, 0], [0, L]]), conditions)
        spins = torus.spins_from_state(state)
        for T in TEMPERATURES:
            sampler = MonteCarlo(torus, T, seed=L * 1000 + int(T * 1e5))
            spins = sampler.tune(spins)
            averages, spins = thermal_averages(
                sampler, spins, THERMALIZATION, MEASUREMENTS, SWEEPS_BETWEEN,
                momenta={'K': K}, twists={'z_x': ((0, 0, 1), (1, 0))}, u1_projection=True)
            m = averages.series['m_K']
            plus, minus, density = m[:, 0] + 1j * m[:, 1], m[:, 0] - 1j * m[:, 1], m[:, 2]
            psi = plus * np.conj(minus)
            clock = np.real(psi ** 3) / np.maximum(np.abs(psi) ** 3, 1e-300)
            row = {'L': L, 'T_meV': T, 'acceptance': averages.acceptance,
                   'energy': [averages.energy, averages.energy_error],
                   'specific_heat': [averages.specific_heat, averages.specific_heat_error],
                   'density_order': float(np.mean(np.abs(density))),
                   'density_binder': float(binder(density)),
                   'transverse_order': float(np.mean(np.sqrt(np.abs(plus) ** 2 + np.abs(minus) ** 2) / np.sqrt(2))),
                   'helicity_u1': list(averages.helicity['z_x']),
                   'bkt_line_2T_over_pi': 2 * T / np.pi,
                   'clock_cos6': [float(np.mean(clock)), float(np.std(clock) / np.sqrt(len(clock) / 20))]}
            report['rows'].append(row)
            print(L, T, round(row['density_order'], 4), round(row['density_binder'], 3),
                  round(row['helicity_u1'][0], 5), round(2 * T / np.pi, 5),
                  round(row['clock_cos6'][0], 3), file=sys.stderr)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
