"""Check that the classical Hessian null space fixes the Y-state orbit axis.

For the NBCP three-sublattice Y state, compute the Hessian of the classical
energy in local tangent coordinates, find the global rotation axis n that lies
in its null space, and compare the classical and zero-point energies along the
orbit R_n(phi) with those along a common azimuth shift R_z(phi). The rotated
case turns the exchange matrices, field and spins by one rotation R, so the
expected axis is R z. Tangent coordinates avoid the spurious azimuthal zero
mode of the polar site C (theta = pi).

Usage
-----
    python examples/nbcp_y_orbit_axis_check.py
"""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lswt-mpl-cache')

import numpy as np

from examples.nbcp_y_stiffness import J, JZ, frame
from model.nbcp import make_nn_exchange_matrices, three_msl
from spintoolkit.methods.lswt.energy import EnergyFunction
from spintoolkit.methods.optimization import SpinOptimizer

FIELD = .2
MESH_N = 6
PHI0 = .3
STEP = 1e-4
NULL_TOL = 1e-6
ORBIT = np.linspace(0, 2 * np.pi, 25)[:-1]


def rotation(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def to_angles(m):
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return np.column_stack([np.arccos(np.clip(m[:, 2], -1, 1)),
                            np.arctan2(m[:, 1], m[:, 0])]).ravel()


def to_vectors(angles):
    theta, phi = np.asarray(angles).reshape(-1, 2).T
    return np.column_stack([np.sin(theta) * np.cos(phi),
                            np.sin(theta) * np.sin(phi), np.cos(theta)])


def build(pd, gamma, R):
    """Energy function and Y-state directions of the problem rotated by R."""
    h, c, *_ = frame(FIELD)
    cfg = dict(Jxy=J, Jz=JZ, JPD=pd, JGamma=gamma, h=tuple(R @ [0., 0., h]))
    exchanges = [R @ M @ R.T for M in make_nn_exchange_matrices(cfg)]
    theta = [np.arccos(c), -np.arccos(c), np.pi]
    m0 = to_vectors(np.column_stack([theta, np.full(3, PHI0)]).ravel()) @ R.T
    system = three_msl(cfg, to_angles(m0), exchanges)
    return EnergyFunction(system.to_legacy_dict('Hex_30'), N=MESH_N), m0


def tangent_frames(m):
    frames = []
    for v in m:
        a = np.array([1., 0, 0]) if abs(v[0]) < .9 else np.array([0, 1., 0])
        u = np.cross(v, a) / np.linalg.norm(np.cross(v, a))
        frames.append((u, np.cross(v, u)))
    return frames


def hessian(cef, m0):
    frames = tangent_frames(m0)

    def energy(x):
        m = np.array([v + x[2*i] * f[0] + x[2*i+1] * f[1]
                      for i, (v, f) in enumerate(zip(m0, frames))])
        return cef.classical_energy_density_func(to_angles(m))

    n = 2 * len(m0)
    H, grad = np.zeros((n, n)), np.zeros(n)
    for i in range(n):
        ei = np.eye(n)[i] * STEP
        grad[i] = (energy(ei) - energy(-ei)) / (2 * STEP)
        for j in range(i, n):
            ej = np.eye(n)[j] * STEP
            H[i, j] = H[j, i] = (energy(ei + ej) - energy(ei - ej)
                                 - energy(-ei + ej) + energy(-ei - ej)) / (4 * STEP**2)
    return H, grad, frames


def detect_axis(H, m0, frames, reference):
    """Rotation axis n minimizing |H G n| / |G n| and the null-space diagnostics."""
    G = np.array([[np.cross(axis, v) @ f[k] for axis in np.eye(3)]
                  for v, f in zip(m0, frames) for k in range(2)])
    values, vectors = np.linalg.eig(np.linalg.solve(G.T @ G, G.T @ H @ H @ G))
    k = np.argmin(values.real)
    axis = vectors[:, k].real / np.linalg.norm(vectors[:, k].real)
    axis *= np.sign(axis @ reference)
    w, V = np.linalg.eigh(H)
    null = V[:, abs(w) < NULL_TOL * np.max(abs(w))]
    Q, _ = np.linalg.qr(G)
    outside = [float(np.linalg.norm(v - Q @ (Q.T @ v))) for v in null.T]
    return axis, float(np.sqrt(abs(values[k].real))), w, null.shape[1], outside


def orbit_spread(cef, m0, axis):
    angles = [to_angles(m0 @ rotation(axis, a).T) for a in ORBIT]
    e_cl = [cef.classical_energy_density_func(x) for x in angles]
    e_qm = np.array([cef.quantum_energy_density_func(x) for x in angles])
    minima = ORBIT[np.isclose(e_qm, e_qm.min(), rtol=0, atol=1e-3 * np.ptp(e_qm) + 1e-18)]
    return {'E_cl_spread': float(np.ptp(e_cl)), 'E_qm_spread': float(np.ptp(e_qm)),
            'E_qm_minimum_offsets_rad': [float(x) for x in minima]}


def case(label, pd, gamma, R, from_de=False):
    cef, m0 = build(pd, gamma, R)
    if from_de:
        optimizer = SpinOptimizer()
        bounds, _, e_cl, _ = optimizer.wrapping_by_angles(cef, [None] * 6)
        m0 = to_vectors(optimizer.find_optimum_w_DE(e_cl, bounds).x)
    H, grad, frames = hessian(cef, m0)
    expected = R @ [0., 0., 1.]
    axis, residual, w, null_dim, outside = detect_axis(H, m0, frames, expected)
    row = {'case': label, 'JPD_meV': pd, 'JGamma_meV': gamma, 'start': 'DE' if from_de else 'Y',
           'max_abs_gradient': float(np.max(abs(grad))), 'hessian_eigenvalues': w.tolist(),
           'null_dimension': null_dim, 'null_norm_outside_rotation_span': outside,
           'detected_axis': axis.tolist(), 'expected_axis': expected.tolist(),
           'axis_error': float(np.linalg.norm(axis - expected)), 'axis_residual': residual}
    if not from_de:
        row['orbit_detected_axis'] = orbit_spread(cef, m0, axis)
        row['orbit_common_azimuth_shift'] = orbit_spread(cef, m0, [0., 0., 1.])
    return row


def main():
    tilt = rotation([1, 2, 0], .7)
    rows = [case('exact U(1)', 0., 0., np.eye(3)),
            case('J_PD accidental', .010, 0., np.eye(3)),
            case('J_Gamma accidental', 0., .010, np.eye(3)),
            case('J_PD, whole problem rotated', .010, 0., tilt),
            case('J_PD from DE', .010, 0., np.eye(3), from_de=True),
            case('J_PD rotated from DE', .010, 0., tilt, from_de=True)]
    report = {'field_T': FIELD, 'J_meV': J, 'Jz_meV': JZ, 'mesh_N': MESH_N,
              'orbit_points': len(ORBIT), 'hessian_step': STEP, 'null_tolerance': NULL_TOL,
              'rotation': {'axis': [1, 2, 0], 'angle_rad': .7}, 'cases': rows}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
