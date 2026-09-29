"""Collect numbers for the open D17 criteria on the NBCP Y state.

1. Null-space test from DE results. Starting from differential-evolution
   minima (three seeds), compare a fixed relative tolerance, a spectral-gap
   ratio, and the same fixed tolerance after a classical L-BFGS-B polish.
   Controls: a tilted field that lifts the classical degeneracy, and a
   field of 20 h, far above saturation, whose polarized state is invariant
   under R_z, so one rotation generator vanishes.
2. Mesh dependence of the zero-point selection. The sixfold amplitude of
   E_qm(phi) along the Y orbit, from the EnergyFunction route at small N,
   against the converged lambda_6 recorded by examples/nbcp_y_angular_matching.py.

Usage
-----
    python examples/nbcp_y_orbit_criteria_check.py
"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np
from scipy.optimize import differential_evolution, minimize

from examples.nbcp_y_orbit_axis_check import (
    FIELD, hessian, rotation, tangent_frames, to_angles, to_vectors)
from examples.nbcp_y_stiffness import J, JZ, frame
from model.nbcp import make_nn_exchange_matrices, three_msl
from spintoolkit.methods.lswt.energy import EnergyFunction

REFERENCE = ROOT / 'data-space/verification/260918-y-angular-matching/angular-matching-check.json'
DE_SEEDS = [42, 7, 2026]
FIXED_TOL = 1e-6
GAP_RATIO = 1e-3
RANK_TOL = 1e-4  # generator singular values below this x max count as zero
ORBIT_SAMPLES = 36
MESHES = [3, 6, 9, 12]


def energy_function(pd, gamma, field, N=3, R=np.eye(3)):
    cfg = dict(Jxy=J, Jz=JZ, JPD=pd, JGamma=gamma, h=tuple(R @ np.asarray(field, float)))
    exchanges = [R @ M @ R.T for M in make_nn_exchange_matrices(cfg)]
    system = three_msl(cfg, np.zeros(6), exchanges)
    return EnergyFunction(system.to_legacy_dict('Hex_30'), N=N)


def de_minimum(cef, seed):
    """Same settings as SpinOptimizer.find_optimum_w_DE except the seed."""
    result = differential_evolution(
        cef.classical_energy_density_func, [(-np.pi, np.pi)] * 6,
        strategy='best1bin', popsize=18, tol=1e-9, mutation=(0.5, 0.9),
        recombination=0.8, maxiter=800, polish=True, updating='immediate', seed=seed)
    return to_vectors(result.x), result.fun


def polish(cef, m0):
    """Classical L-BFGS-B in tangent coordinates, then back to unit vectors."""
    frames = tangent_frames(m0)

    def unit(x):
        m = np.array([v + x[2*i] * f[0] + x[2*i+1] * f[1]
                      for i, (v, f) in enumerate(zip(m0, frames))])
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    result = minimize(lambda x: cef.classical_energy_density_func(to_angles(unit(x))),
                      np.zeros(6), method='L-BFGS-B', options={'ftol': 1e-16, 'gtol': 1e-13})
    return unit(result.x)


def rotation_block(H, m0, frames):
    """Smallest |eigenvalue| of H restricted to the span of n x S and its axis."""
    G = np.array([[np.cross(axis, v) @ f[k] for axis in np.eye(3)]
                  for v, f in zip(m0, frames) for k in range(2)])
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    keep = s > RANK_TOL * s.max()
    Q = U[:, keep]
    w, y = np.linalg.eigh(Q.T @ H @ Q)
    k = np.argmin(abs(w))
    axis = Vt[keep].T @ (y[:, k] / s[keep])
    return float(w[k]), axis / np.linalg.norm(axis), int(keep.sum()), float(s.min() / s.max())


def diagnose(cef, m0, expected_axis=None):
    H, grad, frames = hessian(cef, m0)
    w = np.linalg.eigvalsh(H)
    order = np.argsort(abs(w))
    w_min, w_next = abs(w[order[0]]), abs(w[order[1]])
    rot_value, axis, rank, smallest = rotation_block(H, m0, frames)
    row = {'max_abs_gradient': float(np.max(abs(grad))), 'eig_min_abs': float(w_min),
           'eig_next_abs': float(w_next), 'gap_ratio': float(w_min / w_next),
           'null_by_fixed_tol': bool(w_min < FIXED_TOL * np.max(abs(w))),
           'null_by_gap_ratio': bool(w_min / w_next < GAP_RATIO),
           'rotation_generator_rank': rank, 'generator_smallest_relative_singular_value': smallest,
           'rotation_block_min_abs': float(abs(rot_value))}
    if expected_axis is not None:
        axis *= np.sign(axis @ expected_axis)
        row['axis_error'] = float(np.linalg.norm(axis - expected_axis))
    return row


def null_space_cases():
    h, *_ = frame(FIELD)
    tilt = rotation([1, 2, 0], .7)
    cases = [('exact U(1)', 0., 0., [0, 0, h], np.eye(3), True),
             ('J_PD = 0.010', .010, 0., [0, 0, h], np.eye(3), True),
             ('J_Gamma = 0.010', 0., .010, [0, 0, h], np.eye(3), True),
             ('J_PD = 0.010, rotated', .010, 0., [0, 0, h], tilt, True),
             ('control: tilted field (0.3h, 0, h)', .010, 0., [.3*h, 0, h], np.eye(3), False),
             ('control: polarized, field 20h', 0., 0., [0, 0, 20*h], np.eye(3), False)]
    rows = []
    for label, pd, gamma, field, R, degenerate in cases:
        cef = energy_function(pd, gamma, field, R=R)
        expected = R @ [0., 0., 1.] if degenerate else None
        for seed in DE_SEEDS:
            m0, e_de = de_minimum(cef, seed)
            raw = diagnose(cef, m0, expected)
            polished_state = polish(cef, m0)
            polished = diagnose(cef, polished_state, expected)
            rows.append({'case': label, 'expected_degenerate': degenerate, 'seed': seed,
                         'E_cl_DE': float(e_de),
                         'E_cl_polished': float(cef.classical_energy_density_func(
                             to_angles(polished_state))),
                         'from_DE': raw, 'after_polish': polished})
    return rows


def sixfold_amplitude(cef, phis):
    h, c, *_ = frame(FIELD)
    theta = [np.arccos(c), -np.arccos(c), np.pi]
    energies = np.array([cef.quantum_energy_density_func(
        np.column_stack([theta, np.full(3, phi)]).ravel()) for phi in phis])
    a6 = 2 * np.mean(energies * np.exp(-6j * phis))
    phi_min = np.mod(np.angle(-a6) / 6, np.pi / 3)
    return {'amplitude': float(abs(a6)),
            'phi_min_distance_to_zero_mod_pi_over_three': float(min(phi_min, np.pi / 3 - phi_min)),
            'peak_to_peak': float(np.ptp(energies))}


def mesh_cases():
    reference = json.loads(REFERENCE.read_text())
    lambda6 = {(row['JPD_meV'], row['JGamma_meV']): row['summaries'][-1]
               for row in reference['potential']}
    phis = np.arange(ORBIT_SAMPLES) * 2 * np.pi / ORBIT_SAMPLES
    rows = []
    for pd, gamma in [(.010, 0.), (0., .010), (.005, 0.), (0., .005)]:
        ref = lambda6[(pd, gamma)]
        for N in MESHES:
            h, *_ = frame(FIELD)
            result = sixfold_amplitude(energy_function(pd, gamma, [0, 0, h], N=N), phis)
            result.update({'JPD_meV': pd, 'JGamma_meV': gamma, 'N': N,
                           'reference_lambda6_N96': ref['lambda6_meV_per_spin'],
                           'reference_phi_min_distance_to_zero_mod_pi_over_three': float(min(
                               ref['phi_min_rad_mod_pi_over_three'],
                               np.pi / 3 - ref['phi_min_rad_mod_pi_over_three'])),
                           'amplitude_over_reference': result['amplitude'] / ref['lambda6_meV_per_spin']})
            rows.append(result)
    return rows


def main():
    report = {'field_T': FIELD, 'J_meV': J, 'Jz_meV': JZ, 'de_seeds': DE_SEEDS,
              'fixed_tolerance': FIXED_TOL, 'gap_ratio_threshold': GAP_RATIO, 'rank_tolerance': RANK_TOL,
              'orbit_samples': ORBIT_SAMPLES, 'reference': str(REFERENCE.relative_to(ROOT)),
              'null_space': null_space_cases(), 'mesh': mesh_cases()}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
