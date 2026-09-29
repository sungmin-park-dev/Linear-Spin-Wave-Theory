"""Compare Y/V SOC-induced pseudo-Goldstone gaps with archived gap code.

This is a T=0, leading-semiclassical comparison, not a production gap API.
Only nearest-neighbor XXZ, PD and Gamma exchange and a longitudinal field
are included. The three-sublattice classical orbit is a common lab-z
rotation. No onsite shift or replacement of imaginary frequencies is used.

Run with the project's scientific Python environment, for example:
    python examples/pseudo_goldstone_comparison.py --mesh 24 --nphi 72
"""

import argparse
import ast
import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lswt-mpl-cache')
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT), str(ROOT / 'legacy')]

import numpy as np
from scipy.optimize import minimize_scalar, root

from model.nbcp import make_nn_exchange_matrices, three_msl
from spintoolkit.methods.lswt.hamiltonian import LSWTHamiltonian
from modules.LinearSpinWaveTheory.lswt_Hamiltonian import LSWT_HAMILTONIAN

S, J, JZ = 0.5, 0.075, 0.125
GZ, MU_B = 4.645, 0.05788381806
METRIC = np.diag([1.] * 3 + [-1.] * 3)
LATTICE = np.array([[1.5, np.sqrt(3)/2], [1.5, -np.sqrt(3)/2]])
RECIPROCAL = 2*np.pi*np.linalg.inv(LATTICE).T
PAIRS = [(0, 1), (1, 2), (2, 0)]


def rotation_z(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def spin_angles(theta, phi):
    return np.column_stack((theta, np.full(3, phi))).ravel()


def classical_derivatives(theta, h):
    """Energy, gradient and polar Hessian per magnetic cell (three spins)."""
    st, ct = np.sin(theta), np.cos(theta)
    energy = -h*S*np.sum(ct)
    grad = h*S*st.copy()
    hx = np.diag(h*S*ct)
    hy = np.diag(h*S*ct)
    for i, j in PAIRS:
        bond = 3*S*S*(J*st[i]*st[j] + JZ*ct[i]*ct[j])
        energy += bond
        grad[i] += 3*S*S*(J*ct[i]*st[j] - JZ*st[i]*ct[j])
        grad[j] += 3*S*S*(J*ct[j]*st[i] - JZ*st[j]*ct[i])
        hx[i, i] -= bond
        hx[j, j] -= bond
        hy[i, i] -= bond
        hy[j, j] -= bond
        hx[i, j] = hx[j, i] = 3*S*S*(J*ct[i]*ct[j] + JZ*st[i]*st[j])
        hy[i, j] = hy[j, i] = 3*S*S*J
    return float(energy), grad, hx, hy


def phase_state(phase):
    field = {'Y': 0.2, 'V': 1.4}[phase]
    h = GZ*MU_B*field
    if phase == 'Y':
        t = np.arccos((h+3*S*JZ)/(3*S*(J+JZ)))
        theta = np.array([t, -t, np.pi])
    else:
        theta = root(lambda t: classical_derivatives(t, h)[1],
                     [.4, .4, -1.2], tol=1e-12).x
    _, grad, hx, hy = classical_derivatives(theta, h)
    assert np.max(abs(grad)) < 2e-12
    assert np.linalg.eigvalsh(hx).min() > 1e-8
    g = np.sin(theta)
    # w is d(M_z per magnetic cell)/d(theta_alpha).
    w = -S*g
    response = np.linalg.solve(hx, w)
    chi_cell = w @ response
    dh = 1e-5
    tm = root(lambda t: classical_derivatives(t, h-dh)[1], theta, tol=1e-12).x
    tp = root(lambda t: classical_derivatives(t, h+dh)[1], theta, tol=1e-12).x
    chi_fd = S*(np.cos(tp).sum()-np.cos(tm).sum())/(2*dh)
    assert abs(chi_fd/chi_cell-1) < 1e-6
    assert np.max(abs(hy @ g)) < 2e-12
    # Independent six-dimensional classical dynamics with a weak pinning
    # potential. Its small-C frequency must be sqrt(C_cell / chi_cell).
    poisson = np.block([[np.zeros((3, 3)), np.eye(3)],
                        [-np.eye(3), np.zeros((3, 3))]]) / S
    dynamics_checks = []
    for ccell in [1e-7, 1e-8, 1e-9]:
        pin = ccell*np.outer(g, g)/(g@g)**2
        stiffness = np.block([[hx, np.zeros((3, 3))],
                              [np.zeros((3, 3)), hy+pin]])
        eig = np.linalg.eigvals(poisson @ stiffness)
        assert max(abs(eig.real)) < 1e-10
        frequency = np.sort(eig.imag[eig.imag > 0])[0]
        prediction = np.sqrt(ccell/chi_cell)
        dynamics_checks.append({'pinning_curvature_meV_per_cell': ccell,
                                'frequency_meV': float(frequency),
                                'prediction_meV': float(prediction),
                                'relative_error': float(frequency/prediction-1)})
    assert abs(dynamics_checks[-1]['relative_error']) < 1e-5
    return {'phase': phase, 'B_T': field, 'h_meV': h, 'theta': theta.tolist(),
            'chi_per_spin_per_meV': float(chi_cell/3),
            'chi_finite_difference_per_spin_per_meV': float(chi_fd/3),
            'dtheta_dh': response.tolist(),
            'polar_hessian_per_cell': hx.tolist(),
            'max_classical_torque': float(max(abs(grad))),
            'old_uniform_theta_curvature_per_spin': float(hx.sum()/3),
            'old_pair_Berry_coefficient_per_spin': float(S*g.sum()/3),
            'weak_pinning_dynamics': dynamics_checks}


def build(state, pd, gamma):
    cfg = {'Jxy': J, 'Jz': JZ, 'JPD': pd, 'JGamma': gamma,
           'h': (0., 0., state['h_meV'])}
    system = three_msl(cfg, angles=spin_angles(state['theta'], 0),
                       Exch_J=make_nn_exchange_matrices(cfg), Exch_K=None)
    data = system.to_legacy_dict('Hex_30')
    return data, LSWTHamiltonian(data['Spin info'], data['Couplings']), \
        LSWT_HAMILTONIAN(data['Spin info'], data['Couplings'])


def classical_from_bonds(data, angles):
    """Independent classical energy per spin using actual bond matrices."""
    vectors = {}
    for i, name in enumerate(data['Spin info']):
        t, p = angles[2*i:2*i+2]
        vectors[name] = S*np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
    en = sum(vectors[b['SpinI']] @ b['Exchange Matrix'] @ vectors[b['SpinJ']]
             for b in data['Couplings'])
    en -= sum(vectors[name] @ info['Magnetic Field'] for name, info in data['Spin info'].items())
    return float(en/3)


def mesh(n):
    """Midpoint quadrature on one reciprocal cell, with six rotated copies."""
    q = (np.arange(n)+0.5)/n - 0.5
    points = np.stack(np.meshgrid(q, q, indexing='ij'), axis=-1).reshape(-1, 2) @ RECIPROCAL
    return np.concatenate([points @ rotation_z(m*np.pi/3)[:2, :2].T for m in range(6)])


def vacuum_energy(matrices):
    """Unshifted positive physical bands only; reject unstable backgrounds."""
    chol = np.linalg.cholesky(matrices)
    eig = np.linalg.eigvalsh(chol.conj().transpose(0, 2, 1) @ METRIC @ chol)
    assert np.all(eig[:, :3] < 0) and np.all(eig[:, 3:] > 0)
    trace = np.trace(matrices, axis1=1, axis2=2).real
    return float(np.mean(eig[:, 3:].sum(axis=1)/2-trace/4)/3), float(eig[:, 3:].min())


def direct_hp(data, points, angles):
    """Cartesian HP construction independent of production coupling transforms."""
    vectors, normals = [], []
    for t, p in np.reshape(angles, (3, 2)):
        e1 = np.array([np.cos(t)*np.cos(p), np.cos(t)*np.sin(p), -np.sin(t)])
        e2 = np.array([-np.sin(p), np.cos(p), 0.])
        normals.append(np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)]))
        vectors.append(np.sqrt(S/2)*(e1-1j*e2))
    aa = np.zeros((len(points), 3, 3), complex)
    am, bb = np.zeros_like(aa), np.zeros_like(aa)
    for i, info in enumerate(data['Spin info'].values()):
        aa[:, i, i] = am[:, i, i] = info['Magnetic Field'] @ normals[i]
    names = {'A': 0, 'B': 1, 'C': 2}
    for bond in data['Couplings']:
        i, j = names[bond['SpinI']], names[bond['SpinJ']]
        ex = bond['Exchange Matrix']
        longitudinal = S*(normals[i] @ ex @ normals[j])
        for mat in [aa, am]:
            mat[:, i, i] -= longitudinal
            mat[:, j, j] -= longitudinal
        hop = vectors[i].conj() @ ex @ vectors[j]
        pair = vectors[i].conj() @ ex @ vectors[j].conj()
        phase = np.exp(-1j*(points @ bond['Displacement']))
        for mat, pp in [(aa, phase), (am, phase.conj())]:
            mat[:, i, j] += hop*pp
            mat[:, j, i] += (hop*pp).conj()
        bb[:, i, j] += pair*phase
        bb[:, j, i] += pair*phase.conj()
    return np.concatenate([np.concatenate([aa, bb], axis=2),
                           np.concatenate([bb.conj().transpose(0, 2, 1),
                                           am.transpose(0, 2, 1)], axis=2)], axis=1)


def angular_fit(phi, energy, exact_u1=False):
    """Fit all harmonics through 18, without imposing expected symmetries."""
    m = np.arange(1, 19)
    design = np.column_stack([np.ones(len(phi)), np.cos(phi[:, None]*m), np.sin(phi[:, None]*m)])
    coeff = np.linalg.lstsq(design, energy-np.mean(energy), rcond=None)[0]
    residual = np.max(abs(design @ coeff-(energy-np.mean(energy))))
    a, b = coeff[1:19], coeff[19:]

    def evaluate(x, order=0):
        if order == 0:
            return np.cos(np.asarray(x)[..., None]*m) @ a + np.sin(np.asarray(x)[..., None]*m) @ b
        return np.cos(np.asarray(x)[..., None]*m) @ (-m*m*a) + np.sin(np.asarray(x)[..., None]*m) @ (-m*m*b)

    grid = np.linspace(0, 2*np.pi, 7201)[:-1]
    p0 = grid[np.argmin(evaluate(grid))]
    minimum = minimize_scalar(evaluate, bounds=(p0-.005, p0+.005), method='bounded',
                              options={'xatol': 1e-12}).x % (2*np.pi)
    curvature = float(evaluate(minimum, 2))
    numerical_floor = 18**2*(10*residual+100*np.finfo(float).eps*max(abs(energy)))
    status = 'resolved' if curvature > numerical_floor else 'below_numerical_resolution'
    if exact_u1:
        status, curvature, minimum = 'exact_U1', 0., 0.
    return {'phi_min': float(minimum), 'curvature_meV_per_spin': curvature,
            'curvature_noise_estimate': float(numerical_floor), 'status': status,
            'fit_max_residual_meV_per_spin': float(residual),
            'cos_coefficients': a.tolist(), 'sin_coefficients': b.tolist()}


def scan(state, pd, gamma, n, nphi):
    start = time.monotonic()
    data, current, legacy = build(state, pd, gamma)
    points, phi = mesh(n), np.arange(nphi)*2*np.pi/nphi
    record = {'phase': state['phase'], 'JPD_meV': pd, 'JGamma_meV': gamma,
              'N': n, 'n_k': len(points), 'n_phi': nphi, 'phi': phi.tolist()}
    ec = [classical_from_bonds(data, spin_angles(state['theta'], p)) for p in phi]
    record['Ecl_meV_per_spin'] = ec
    record['classical_energy_range'] = float(np.ptp(ec))
    assert record['classical_energy_range'] < 1e-13
    for label, ham in [('current', current), ('legacy', legacy)]:
        energy, min_frequency, max_torque = [], np.inf, 0.
        for p in phi:
            mats, torque = ham.Quadratic_Bose_Hamiltonian(points, angles=spin_angles(state['theta'], p))
            if label == 'current':
                max_torque = max(max_torque, max(abs(v) for v in torque.values()))
            try:
                en, freq = vacuum_energy(mats)
            except np.linalg.LinAlgError:
                record[label] = {'status': 'unstable_sampled_hessian', 'phi_failed': float(p),
                                 'min_matrix_eigenvalue': float(np.linalg.eigvalsh(mats).min())}
                break
            energy.append(en)
            min_frequency = min(min_frequency, freq)
        else:
            fit = angular_fit(phi, np.array(energy), exact_u1=(pd == gamma == 0))
            c = fit['curvature_meV_per_spin']
            valid = fit['status'] in ('resolved', 'exact_U1')
            gap = float(np.sqrt(c/state['chi_per_spin_per_meV'])) if valid else None
            old = float(S*np.sqrt(state['old_uniform_theta_curvature_per_spin']*c)) if valid else None
            record[label] = dict(fit, Ezp_meV_per_spin=energy, gap_with_susceptibility_meV=gap,
                                 gap_with_old_uniform_theta_and_times_S_meV=old,
                                 min_sampled_frequency_meV=float(min_frequency),
                                 max_linear_term=float(max_torque))
            if label == 'current':
                assert max_torque < 2e-12
    record['seconds'] = time.monotonic()-start
    print(json.dumps({k: record[k] for k in ['phase', 'JPD_meV', 'JGamma_meV', 'N', 'seconds']}),
          record['current']['status'], record['current'].get('gap_with_susceptibility_meV'), flush=True)
    return record


def verify(state, pd=.005, gamma=.005):
    data, ham, legacy = build(state, pd, gamma)
    points = np.random.default_rng(912).normal(size=(19, 2))
    angles = spin_angles(state['theta'], .271)
    mats, _ = ham.Quadratic_Bose_Hamiltonian(points, angles=angles)
    old, _ = legacy.Quadratic_Bose_Hamiltonian(points, angles=angles)
    reference = direct_hp(data, points, angles)
    error = float(np.max(abs(mats-reference)))
    assert error < 1e-12
    # Same physical spins, another spherical-coordinate representation.
    alternate = angles.copy()
    alternate[0] *= -1
    alternate[1] += np.pi
    changed, _ = ham.Quadratic_Bose_Hamiltonian(points, angles=alternate)
    e1, _ = vacuum_energy(mats)
    e2, _ = vacuum_energy(changed)
    assert abs(e1-e2) < 1e-13
    uniform = np.tile([1., 0.], 3)
    dt = 1e-4

    def old_stiffness(a):
        return (classical_from_bonds(data, a+dt*uniform)+classical_from_bonds(data, a-dt*uniform)
                -2*classical_from_bonds(data, a))/(dt*dt)

    return {'phase': state['phase'], 'current_vs_independent_HP_max': error,
            'legacy_vs_independent_HP_max': float(np.max(abs(old-reference))),
            'coordinate_relabel_energy_error': float(abs(e1-e2)),
            'old_stiffness_original': old_stiffness(angles),
            'old_stiffness_same_spins_other_coordinates': old_stiffness(alternate)}


def replay_legacy(state, record, out):
    """Execute the archived class verbatim with only its k-grid matched."""
    from modules.Tools.analysis_tools import Create_Energy_Function
    data, _, _ = build(state, record['JPD_meV'], record['JGamma_meV'])
    phi = record['legacy'].get('phi_min', 0.)
    for info in data['Spin info'].values():
        info['Angles'] = (info['Angles'][0], phi)
    points = mesh(12)

    def factory(spin_sys_data, N, update_args):
        energy = Create_Energy_Function(spin_sys_data, N=N, update_args=update_args)
        energy.k_points = points
        energy.num_k_points = len(points)
        return energy

    source = ROOT/'legacy/scripts/4_Pseudo_Gap.py'
    parsed = ast.parse(source.read_text())
    node = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == 'Compute_Pseudo_Gap')
    namespace = {'np': np, 'Create_Energy_Function': factory,
                 'DEFAULT_D_THETA': 1e-3*np.pi, 'DEFAULT_D_PHI': np.pi/20}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        obj = namespace['Compute_Pseudo_Gap'](data, Temperature=0)
        gap3 = obj.calculate_pseudo_gap(N=12)
        gap5 = obj.calculate_pseudo_gap(N=12, use_five_point=True)
    filename = f"legacy-replay-{state['phase']}-pd{record['JPD_meV']}-gamma{record['JGamma_meV']}.txt"
    (out/filename).write_text(output.getvalue())
    return {'phase': state['phase'], 'JPD_meV': record['JPD_meV'], 'JGamma_meV': record['JGamma_meV'],
            'phi_min_legacy': phi, 'N': 12, 'gap_3point_meV': float(gap3), 'gap_5point_meV': float(gap5),
            'log': filename, 'scope': 'Archived class unchanged; classical state, angle origin and midpoint k-grid supplied for comparison.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=int, default=24)
    parser.add_argument('--nphi', type=int, default=72)
    parser.add_argument('--couplings', type=float, nargs='+', default=[0, .005, .01, .015, .02])
    parser.add_argument('--phases', nargs='+', default=['Y', 'V'])
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'data-space/verification/260912-pseudo-goldstone')
    args = parser.parse_args()
    assert args.nphi >= 48 and args.nphi % 6 == 0
    args.output.mkdir(parents=True, exist_ok=True)
    states = {phase: phase_state(phase) for phase in args.phases}
    checks = [verify(state) for state in states.values()]
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': __doc__, 'S': S, 'J_meV': J, 'Jz_meV': JZ, 'temperature_K': 0,
              'g_z': GZ, 'mu_B_meV_per_T': MU_B, 'states': states, 'checks': checks, 'scans': []}
    output = args.output/f'scan-N{args.mesh}-P{args.nphi}.json'
    for state in states.values():
        for axis in ['PD', 'Gamma']:
            for value in args.couplings:
                if axis == 'Gamma' and value == 0:
                    continue
                report['scans'].append(scan(state, value if axis == 'PD' else 0,
                                            value if axis == 'Gamma' else 0, args.mesh, args.nphi))
                output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    if args.replay:
        report['legacy_replays'] = []
        for state in states.values():
            for row in report['scans']:
                if row['phase'] == state['phase'] and max(row['JPD_meV'], row['JGamma_meV']) == .01:
                    report['legacy_replays'].append(replay_legacy(state, row, args.output))
    sources = [str(Path(__file__).relative_to(ROOT)), 'examples/nbcp_ground_state.py',
               'model/__init__.py', 'model/nbcp/__init__.py',
               'model/nbcp/exchange.py', 'model/nbcp/unit_cells.py',
               'code-space/spintoolkit/methods/lswt/hamiltonian.py', 'code-space/spintoolkit/system/exchange.py',
               'legacy/modules/LinearSpinWaveTheory/lswt_Hamiltonian.py',
               'legacy/scripts/4_Pseudo_Gap.py', 'legacy/scripts/2_U_symmetry_YV.py']
    report['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(output)


if __name__ == '__main__':
    main()
