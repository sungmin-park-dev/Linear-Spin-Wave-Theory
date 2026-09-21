"""Verify the relaxed canonical partner of the Y/V common-z orbit.

This independent review script checks explicit phase formulas against field
response, constrained classical minimization, coordinate changes, and the
six-dimensional weak-pinning dynamics. It does not recompute quantum gaps.
"""

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.pseudo_goldstone_comparison import (
    J, JZ, S, build, classical_derivatives, classical_from_bonds, phase_state,
    spin_angles,
)


def bond_hessian(data, angles, step=3e-4):
    """Polar Hessian per cell from the actual Cartesian bond energy."""
    basis = np.eye(6)[::2]

    def energy(t):
        return 3 * classical_from_bonds(data, t)

    center = energy(angles)
    result = np.empty((3, 3))
    for i, ei in enumerate(basis):
        result[i, i] = (-energy(angles + 2*step*ei)
                        + 16*energy(angles + step*ei) - 30*center
                        + 16*energy(angles - step*ei)
                        - energy(angles - 2*step*ei)) / (12*step**2)
        for j in range(i):
            ej = basis[j]
            result[i, j] = result[j, i] = (
                energy(angles + step*(ei+ej)) + energy(angles - step*(ei+ej))
                - energy(angles + step*(ei-ej)) - energy(angles - step*(ei-ej))
            ) / (4*step**2)
    return result


def check(phase):
    state = phase_state(phase)
    theta, h = np.array(state['theta']), state['h_meV']
    _, _, A, B = classical_derivatives(theta, h)
    w = -S*np.sin(theta)
    response = np.linalg.solve(A, w)
    chi = w @ response / 3
    partner = response / chi  # d(theta)/d(m_z per spin)
    assert np.isclose(w @ partner / 3, 1)
    assert np.isclose(partner @ A @ partner / 3, 1/chi)

    if phase == 'Y':
        sine = np.sin(theta[0])
        derivative = 1/(3*S*(J+JZ)*sine)
        explicit = np.array([-derivative, derivative, 0.])
        explicit_chi = 2/(9*(J+JZ))
        explicit_details = {'chi_formula': '2 / (9 * (J + Jz))'}
    else:
        a, b, c = A[0, 0]+A[0, 1], A[0, 2], A[2, 2]
        determinant = a*c-2*b*b
        st, su = np.sin(theta[0]), np.sin(theta[2])
        dt = -S*(c*st-b*su)/determinant
        du = -S*(a*su-2*b*st)/determinant
        explicit = np.array([dt, dt, du])
        explicit_chi = S*S*(2*c*st*st-4*b*st*su+a*su*su)/(3*determinant)
        explicit_details = {'a_meV': a, 'b_meV': b, 'c_meV': c,
                            'determinant_meV_squared': determinant}
    assert np.allclose(response, explicit, rtol=1e-12, atol=1e-12)
    assert np.isclose(chi, explicit_chi, rtol=1e-12)

    field_checks = []
    for dh in [1e-4, 3e-5, 1e-5]:
        plus = root(lambda x: classical_derivatives(x, h+dh)[1], theta, tol=1e-11)
        minus = root(lambda x: classical_derivatives(x, h-dh)[1], theta, tol=1e-11)
        for result, shifted_h in [(plus, h+dh), (minus, h-dh)]:
            assert max(abs(classical_derivatives(result.x, shifted_h)[1])) < 1e-11
        derivative = (plus.x-minus.x)/(2*dh)
        relative = np.linalg.norm(derivative-response)/np.linalg.norm(response)
        assert relative < 3e-5
        field_checks.append({'dh_meV': dh, 'relative_tangent_error': float(relative)})

    # Independently minimize the full nonlinear classical energy at fixed m_z
    # through its constrained stationarity equations, with a Lagrange multiplier.
    m0 = S*np.cos(theta).sum()/3
    constraint_checks = []
    for dm in [1e-4, 1e-5]:
        def equations(z):
            x, multiplier = z[:3], z[3]
            _, gradient, _, _ = classical_derivatives(x, h)
            wx = -S*np.sin(x)
            return np.r_[gradient-multiplier*wx, S*np.cos(x).sum()/3-m0-dm]

        result = root(equations, np.r_[theta+partner*dm, dm/chi], tol=1e-11)
        assert max(abs(equations(result.x))) < 1e-11
        tangent_error = np.linalg.norm((result.x[:3]-theta)/dm-partner)/np.linalg.norm(partner)
        e0 = classical_derivatives(theta, h)[0]/3
        e1 = classical_derivatives(result.x[:3], h)[0]/3
        stiffness = 2*(e1-e0)/(dm*dm)
        assert tangent_error < .003
        assert abs(stiffness*chi-1) < .003
        constraint_checks.append({'dm_per_spin': dm, 'relative_tangent_error': float(tangent_error),
                                  'energy_curvature_meV': float(stiffness),
                                  'expected_curvature_meV': float(1/chi)})

    # Spherical-coordinate relabeling of any subset must preserve chi. The
    # arbitrary common-polar path is a different path after such relabeling.
    data, _, _ = build(state, .005, .005)
    original = spin_angles(theta, .271)
    chart_checks = []
    for signs in itertools.product([-1., 1.], repeat=3):
        D = np.diag(signs)
        alternate = original.copy()
        for i, sign in enumerate(signs):
            if sign < 0:
                alternate[2*i] *= -1
                alternate[2*i+1] += np.pi
        actual = bond_hessian(data, alternate)
        target = D @ A @ D
        error = float(np.max(abs(actual-target)))
        wp = D @ w
        actual_chi = wp @ np.linalg.solve(actual, wp)/3
        assert error < 3e-8
        assert abs(actual_chi/chi-1) < 2e-6
        chart_checks.append({'polar_signs': list(signs), 'hessian_error_meV': error,
                             'relative_chi_error': float(actual_chi/chi-1)})

    g = np.sin(theta)
    poisson = np.block([[np.zeros((3, 3)), np.eye(3)],
                        [-np.eye(3), np.zeros((3, 3))]])/S
    dynamics_checks = []
    for curvature_cell in [1e-7, 1e-9]:
        pin = curvature_cell*np.outer(g, g)/(g@g)**2
        hessian = np.block([[A, np.zeros((3, 3))], [np.zeros((3, 3)), B+pin]])
        values, vectors = np.linalg.eig(poisson @ hessian)
        index = min(np.flatnonzero(values.imag > 0), key=lambda k: values[k].imag)
        omega, mode = values[index].imag, vectors[:, index]
        q = g @ mode[3:] / (g@g)
        measured_response = mode[:3] / (1j*omega*q)
        tangent_error = np.linalg.norm(measured_response-response)/np.linalg.norm(response)
        frequency_error = omega/np.sqrt(curvature_cell/(3*chi))-1
        assert abs(frequency_error) < 1e-5
        assert tangent_error < 1e-5
        dynamics_checks.append({'pinning_curvature_meV_per_cell': curvature_cell,
                                'relative_frequency_error': float(frequency_error),
                                'relative_partner_error': float(tangent_error)})

    trial = np.ones(3)
    berry, stiffness = w @ trial/3, trial @ A @ trial/3
    trial_chi = berry*berry/stiffness
    result = {'phase': phase, 'h_meV': h, 'theta_rad': theta.tolist(),
              'polar_hessian_meV_per_cell': A.tolist(), 'w_per_cell': w.tolist(),
              'response_per_meV': response.tolist(), 'canonical_partner_per_dm': partner.tolist(),
              'chi_per_spin_per_meV': float(chi), 'explicit_formula': explicit_details,
              'uniform_polar_Berry_coefficient': float(berry),
              'uniform_polar_stiffness_meV_per_spin': float(stiffness),
              'uniform_trial_chi_per_meV': float(trial_chi),
              'field_checks': field_checks, 'constraint_checks': constraint_checks,
              'coordinate_chart_checks': chart_checks, 'weak_pinning_checks': dynamics_checks}
    if phase == 'Y':
        assert abs(berry) < 1e-15
        result['uniform_trial_status'] = 'zero Berry pairing analytically; not a canonical pair'
    else:
        result['uniform_trial_status'] = 'canonicalizable, but not the relaxed low-energy partner'
        result['Berry_corrected_uniform_gap_ratio'] = float(np.sqrt(chi/trial_chi))
    print(phase, json.dumps({key: result[key] for key in [
        'chi_per_spin_per_meV', 'response_per_meV', 'uniform_trial_status']}))
    return result


def main():
    results = [check(phase) for phase in ['Y', 'V']]
    sources = [Path(__file__), ROOT/'examples/pseudo_goldstone_comparison.py',
               ROOT/'examples/nbcp_ground_state.py']
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': __doc__, 'S': S, 'J_meV': J, 'Jz_meV': JZ, 'results': results,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sources}}
    output = ROOT/'data-space/verification/260912-pseudo-goldstone/conjugate-mode-validation-260915.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(output)


if __name__ == '__main__':
    main()
