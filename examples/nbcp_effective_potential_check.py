"""Check the one-loop effective potential on the relaxed soft path (proposal for D28).

For NBCP Y (0.2 T, J_PD = 0.01 meV) and V (1.4 T, J_Gamma = 0.01 meV) with the
field tilted by an in-plane component ``tilt * h``:

1. Relaxed soft path: at each rotation angle phi about the orbit axis n, the
   classical energy is minimized over every tangent direction except the orbit
   tangent (Newton steps with the analytic Hessian). Only the torque along
   the orbit remains; a torque transverse to the spins does not change H2, so
   dropping the linear term is the constrained one-loop calculation.
2. Gamma(phi) = E_cl + E_zp on that path, against the rigid orbit R_n(phi)
   used now, and against the D17 selection in the limit tilt -> 0.
3. Adiabatic ratio: curvature of Gamma at its minimum (per unit tangent
   displacement) over the smallest hard-mode stiffness of the classical Hessian.

Usage
-----
    python examples/nbcp_effective_potential_check.py > report.json
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
from spintoolkit.methods.classical import classical_energy, tangent_expansion
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions

SCAN = ROOT / 'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json'
G_Z = 4.645
AXIS = np.array([0.0, 0.0, 1.0])
PHIS = np.linspace(0, 2 * np.pi, 72, endpoint=False)
MESH_N = 6
CASES = [('Y', 0.2, {'JPD': 0.01}, [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3]),
         ('V', 1.4, {'JGamma': 0.01}, [0.0, 1e-2, 3e-2, 0.1])]


def reference_state(model, phase):
    theta = np.array(json.loads(SCAN.read_text())['states'][phase]['theta'])
    return nbcp.candidate_state(model, 'three_msl', np.column_stack([theta, np.zeros(3)]).ravel())


def rotation_angle(reference, state, axis):
    """Least-squares rotation angle about axis mapping reference onto state."""
    cross = dot = 0.0
    for key, r in reference.directions.items():
        s = state.directions[key]
        r_perp, s_perp = r - (r @ axis) * axis, s - (s @ axis) * axis
        cross += np.cross(r_perp, s_perp) @ axis
        dot += r_perp @ s_perp
    return float(np.arctan2(cross, dot))


def moved(state, expansion, x):
    directions = expansion.directions + np.einsum('ia,iax->ix', x.reshape(-1, 2), expansion.frames)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    return SpinState(state.model_ref, state.supercell, dict(zip(expansion.keys, directions)),
                     state.provenance)


def relax_on_path(model, state, conditions, axis, steps=30):
    """Minimize E_cl over tangent directions orthogonal to the orbit tangent."""
    for _ in range(steps):
        ex = tangent_expansion(model, state, conditions)
        generators = sel._generators(ex)
        tangent = generators @ axis
        tangent /= np.linalg.norm(tangent)
        _, _, vt = np.linalg.svd(tangent[None, :])
        P = vt[1:].T                                              # complement of the tangent
        gP = P.T @ ex.gradient
        if np.max(np.abs(gP)) < 1e-15:
            break
        HP = P.T @ ex.hessian @ P
        w, v = np.linalg.eigh(HP)
        step = -v @ ((v.T @ gP) / np.maximum(w, 1e-3 * np.max(np.abs(w))))
        state = moved(state, ex, P @ step)
    ex = tangent_expansion(model, state, conditions)
    generators = sel._generators(ex)
    raw = generators @ axis
    tangent = raw / np.linalg.norm(raw)
    _, _, vt = np.linalg.svd(tangent[None, :])
    P = vt[1:].T
    hard = float(np.min(np.linalg.eigvalsh(P.T @ ex.hessian @ P)))
    return state, {'transverse_gradient': float(np.max(np.abs(P.T @ ex.gradient))),
                   'orbit_torque': float(tangent @ ex.gradient),
                   'hard_stiffness': hard, 'tangent_norm': float(np.linalg.norm(raw))}


def fit_minimum(phis, values, max_harmonic=12):
    fit = sel.fit_harmonics(np.asarray(phis), np.asarray(values), max_harmonic)
    phi = sel._fit_minimum(fit, len(phis))
    return phi, float(sel._series(fit, phi, 2)), fit['residual']


def wrap(x):
    return float(np.mod(x + np.pi, 2 * np.pi) - np.pi)


def case(phase, field_T, extra, tilt):
    model = nbcp.build_model({'Jxy': 0.075, 'Jz': 0.125, **extra})
    h = G_Z * MU_B_MEV_PER_T * field_T
    conditions = ExternalConditions(field=[tilt * h, 0, h])
    quantum = sel.lswt_zero_point_energy('Hex_30', MESH_N)
    ref = reference_state(model, phase)
    rigid_cl, rigid_qm, path_phi, path_cl, path_qm, info = [], [], [], [], [], []
    for phi in PHIS:
        rigid = sel.rotate_state(ref, AXIS, phi)
        rigid_cl.append(classical_energy(model, rigid, conditions))
        rigid_qm.append(quantum(model, rigid, conditions))
        relaxed, d = relax_on_path(model, rigid, conditions, AXIS)
        path_phi.append(rotation_angle(ref, relaxed, AXIS))
        path_cl.append(classical_energy(model, relaxed, conditions))
        path_qm.append(quantum(model, relaxed, conditions))
        info.append(d)
    path_phi = np.unwrap(np.array(path_phi))
    order = np.argsort(np.mod(path_phi, 2 * np.pi))
    p = np.mod(path_phi, 2 * np.pi)[order]
    cl, qm = np.array(path_cl)[order], np.array(path_qm)[order]
    gamma = cl + qm
    phi_gamma, c_gamma, residual = fit_minimum(p, gamma)
    phi_cl, _, _ = fit_minimum(p, cl)
    phi_qm, c_qm, _ = fit_minimum(p, qm)
    rigid_total = np.array(rigid_cl) + np.array(rigid_qm)
    phi_rigid, _, _ = fit_minimum(PHIS, rigid_total)
    i_min = int(np.argmin(np.abs(np.mod(p - phi_gamma + np.pi, 2 * np.pi) - np.pi)))
    at_min = [info[j] for j in order][i_min]
    c_unit = c_gamma / at_min['tangent_norm'] ** 2
    return {
        'phase': phase, 'field_T': field_T, 'couplings': extra, 'tilt': tilt,
        'tilt_degrees': float(np.degrees(np.arctan(tilt))), 'num_phi': len(PHIS), 'mesh_N': MESH_N,
        'max_transverse_gradient_on_path': max(d['transverse_gradient'] for d in info),
        'phi_min_gamma': phi_gamma, 'phi_min_classical_path': phi_cl, 'phi_min_quantum_path': phi_qm,
        'phi_min_rigid_orbit': phi_rigid,
        'gamma_minus_quantum_min_rad': wrap(phi_gamma - phi_qm),
        'gamma_minus_classical_min_rad': wrap(phi_gamma - phi_cl),
        'rigid_minus_relaxed_min_rad': wrap(phi_rigid - phi_gamma),
        'span_classical_path': float(np.ptp(cl)), 'span_quantum_path': float(np.ptp(qm)),
        'span_classical_rigid': float(np.ptp(rigid_cl)),
        'relaxation_energy_max': float(np.max(np.array(rigid_cl)[order] - cl)),
        'gamma_fit_residual': residual, 'gamma_curvature_per_rad2': c_gamma,
        'quantum_curvature_per_rad2': c_qm,
        'hard_stiffness_at_min': at_min['hard_stiffness'],
        'adiabatic_ratio': c_unit / at_min['hard_stiffness'],
    }


def d17_reference(phase, field_T, extra):
    model = nbcp.build_model({'Jxy': 0.075, 'Jz': 0.125, **extra})
    conditions = ExternalConditions(field=[0, 0, G_Z * MU_B_MEV_PER_T * field_T])
    ref = reference_state(model, phase)
    result = sel.select_on_manifold(model, ref, conditions, sel.lswt_zero_point_energy('Hex_30', MESH_N))
    return {'verdict': result.verdict, 'phi': result.phi, 'absolute_phi': rotation_angle(ref, result.state, AXIS)}


def main():
    report = {'cases': [], 'd17': {}}
    for phase, field_T, extra, tilts in CASES:
        report['d17'][phase] = d17_reference(phase, field_T, extra)
        for tilt in tilts:
            report['cases'].append(case(phase, field_T, extra, tilt))
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()
