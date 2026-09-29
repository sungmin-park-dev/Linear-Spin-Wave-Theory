"""Validate the classical zero-SOC Y stiffness against energy and LSWT.

This diagnostic uses physical bond displacements in units of the NN distance.
It computes neither the zero-point correction to stiffness nor a thermal phase.
Run nbcp_y_stiffness.wl first for the independent symbolic checks.
"""

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lswt-mpl-cache')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

from model.nbcp import make_nn_exchange_matrices, three_msl
from lswt.methods.spin_wave.hamiltonian import LSWTHamiltonian

OUT = ROOT / 'data-space/verification/260917-y-stiffness'
S, J, JZ = .5, .075, .125
GZ, MU_B = 4.645, .05788381806
AREA = np.sqrt(3) / 2  # a=1; area per physical spin
DELTAS = np.array([[1., 0.], [-.5, np.sqrt(3)/2], [-.5, -np.sqrt(3)/2]])
PAIRS = [(0, 1), (1, 2), (2, 0)]
METRIC = np.diag([1.] * 3 + [-1.] * 3)
POISSON = np.block([[np.zeros((3, 3)), np.eye(3)],
                    [-np.eye(3), np.zeros((3, 3))]]) / S
# Remove only the global azimuth: y_A=y_B, with y_C free at the polar site.
GAUGE = np.zeros((6, 5))
GAUGE[:3, :3] = np.eye(3)
GAUGE[3, 3] = GAUGE[4, 3] = 1 / np.sqrt(2)
GAUGE[5, 4] = 1.


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def frame(field):
    h = GZ * MU_B * field
    c = (h + 3*S*JZ) / (3*S*(J+JZ))
    assert 0 < h < 3*S*J
    s = np.sqrt(1-c*c)
    normals = np.array([[s, 0., c], [-s, 0., c], [0., 0., -1.]])
    polar = np.array([[c, 0., -s], [c, 0., s], [-1., 0., 0.]])
    azimuth = np.tile([0., 1., 0.], (3, 1))
    return h, c, normals, polar, azimuth


def local_spins(z, field):
    _, _, normals, polar, azimuth = frame(field)
    tangent = GAUGE @ z
    x, y = tangent[:3], tangent[3:]
    return normals * np.sqrt(1-x*x-y*y)[:, None] + polar*x[:, None] + azimuth*y[:, None]


def energy(z, q, field):
    """Exact fixed-length classical energy per spin on a twisted branch."""
    h = GZ * MU_B * field
    n = local_spins(z, field)
    exchange = np.diag([J, J, JZ])
    summed = sum(rz(q @ d) for d in DELTAS)
    return (S*S*sum(n[i] @ exchange @ summed @ n[j] for i, j in PAIRS)
            - h*S*n[:, 2].sum()) / 3


def gradient(z, q, field):
    """Complex-step differentiation of the analytic fixed-length energy."""
    return np.array([energy(z.astype(complex) + 1e-25j*np.eye(5)[i], q, field).imag/1e-25
                     for i in range(5)])


def relaxed(q, field):
    result = root(lambda z: gradient(z, q, field), np.zeros(5), tol=1e-10)
    residual = float(np.max(np.abs(gradient(result.x, q, field))))
    assert residual < 2e-12, (field, q, result.message, residual)
    assert np.max(np.abs(result.x)) < .02, (field, q, result.x)
    return float(energy(result.x, q, field)), result.x, residual


def tangent_hessian(q, field):
    """Cartesian constrained Hessian per magnetic cell, independently assembled."""
    h, _, normals, polar, azimuth = frame(field)
    basis = np.stack([polar, azimuth], axis=-1)
    matrix = np.zeros((6, 6), complex)
    exchange = np.diag([J, J, JZ])
    for i in range(3):
        matrix[i, i] = matrix[i+3, i+3] = h*S*normals[i, 2]
    for i, j in PAIRS:
        ii, jj = [i, i+3], [j, j+3]
        for d in DELTAS:
            longitudinal = S*S*normals[i] @ exchange @ normals[j]
            matrix[ii, ii] -= longitudinal
            matrix[jj, jj] -= longitudinal
            block = S*S * (basis[i].T @ exchange @ basis[j]) * np.exp(-1j*q@d)
            matrix[np.ix_(ii, jj)] += block
            matrix[np.ix_(jj, ii)] += block.conj().T
    assert np.max(np.abs(matrix-matrix.conj().T)) < 1e-14
    return matrix


def torus_energy(z, q, field, length=6):
    """Separate site/color enumeration; twist applied across periodic boundaries."""
    n = local_spins(z, field)
    exchange = np.diag([J, J, JZ])
    primitives = np.array([[1., 0.], [.5, np.sqrt(3)/2]])
    directions = [(1, 0), (0, 1), (-1, 1)]
    total = 0.
    for u in range(length):
        for v in range(length):
            i = (u + 2*v) % 3
            total -= GZ*MU_B*field*S*n[i, 2]
            for du, dv in directions:
                uu, vv = (u+du) % length, (v+dv) % length
                j = (uu + 2*vv) % 3
                displacement = np.array([du, dv]) @ primitives
                total += S*S*n[i] @ exchange @ rz(q@displacement) @ n[j]
    return float(total / length**2)


def positive_spectrum(matrix):
    values = np.linalg.eigvals(matrix)
    assert np.max(np.abs(values.imag)) < 1e-10
    positive = np.sort(values.real[values.real > 0])
    assert len(positive) == 3
    return positive


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    symbolic = json.loads((OUT/'symbolic-check.json').read_text())
    assert all(symbolic['checks'].values())
    assert symbolic['source_sha256'] == hashlib.sha256((ROOT/'examples/nbcp_y_stiffness.wl').read_bytes()).hexdigest()
    fields = [.05, .1, .2, .3, .4]
    steps = [.03, .01, .003, .001]
    directions = [0., np.pi/7, np.pi/3, np.pi/2]
    results = []
    max_spectrum_error = 0.
    max_torus_error = 0.
    rng = np.random.default_rng(917)
    for field in fields:
        h, c, _, _, _ = frame(field)
        theta = np.array([np.arccos(c), -np.arccos(c), np.pi])
        angles = np.column_stack([theta, np.zeros(3)]).ravel()
        cfg = {'Jxy': J, 'Jz': JZ, 'JPD': 0., 'JGamma': 0., 'h': (0., 0., h)}
        system = three_msl(cfg, angles=angles, Exch_J=make_nn_exchange_matrices(cfg))
        data = system.to_legacy_dict('Hex_30')  # data adapter only, no legacy solver
        ham = LSWTHamiltonian(data['Spin info'], data['Couplings'])
        rho = J*S*S*(1-c*c) / np.sqrt(3)
        chi = 2 / (9*(J+JZ))
        slope = np.sqrt(AREA*rho/chi)
        uniform = tangent_hessian(np.zeros(2), field)
        hard_values = np.linalg.eigvalsh(GAUGE.T @ uniform @ GAUGE)
        assert hard_values.min() > 0
        e0 = float(energy(np.zeros(5), np.zeros(2), field))
        # Check the 3-site bond count using a separately enumerated 36-site torus.
        for _ in range(3):
            z = rng.normal(scale=.005, size=5)
            q = rng.normal(scale=.02, size=2)
            error = abs(float(energy(z, q, field))-torus_energy(z, q, field))
            max_torus_error = max(max_torus_error, error)
            assert error < 2e-15
        samples = []
        for direction in directions:
            unit = np.array([np.cos(direction), np.sin(direction)])
            for step in steps:
                q = step * unit
                plus, zp, rp = relaxed(q, field)
                minus, zm, rm = relaxed(-q, field)
                rho_relaxed = (plus+minus-2*e0) / (AREA*step**2)
                rho_frozen = (energy(np.zeros(5), q, field)+energy(np.zeros(5), -q, field)-2*e0)/(AREA*step**2)
                mats, torque = ham.Quadratic_Bose_Hamiltonian(q[None, :], angles=angles)
                assert max(abs(v) for v in torque.values()) < 1e-12
                spectrum = positive_spectrum(METRIC @ mats[0])
                independent = positive_spectrum(1j*POISSON @ tangent_hessian(q, field))
                max_spectrum_error = max(max_spectrum_error, float(np.max(abs(spectrum-independent))))
                rho_wave = chi*(spectrum[0]/step)**2 / AREA
                samples.append({'direction_rad': direction, 'qa': step,
                                'rho_frozen_meV': float(rho_frozen),
                                'rho_relaxed_meV': float(rho_relaxed),
                                'rho_from_LSWT_meV': float(rho_wave),
                                'epsilon_meV': float(spectrum[0]),
                                'energy_slope_meV_a': float(spectrum[0]/step),
                                'internal_displacement_norm': float(np.linalg.norm(zp)),
                                'displacement_over_qa_squared': float(np.linalg.norm(zp)/step**2),
                                'max_relaxed_gradient': max(rp, rm),
                                'relaxed_energy_gain_meV_per_spin': float(energy(np.zeros(5), q, field)-plus)})
        finest = [r for r in samples if r['qa'] == min(steps)]
        static_error = max(abs(r['rho_relaxed_meV']/rho-1) for r in finest)
        wave_error = max(abs(r['rho_from_LSWT_meV']/rho-1) for r in finest)
        assert static_error < 1e-5, (field, static_error)
        assert wave_error < 1e-5, (field, wave_error)
        results.append({'B_T': field, 'h_meV': h, 'cos_t': c,
                        'rho_classical_meV': rho, 'chi_per_spin_meV_inverse': chi,
                        'energy_slope_meV_a': slope,
                        'hard_uniform_eigenvalues_per_cell_meV': hard_values.tolist(),
                        'finest_static_relative_error_max': static_error,
                        'finest_LSWT_relative_error_max': wave_error,
                        'samples': samples})
    assert max_spectrum_error < 1e-10
    inputs = [Path(__file__), ROOT/'examples/nbcp_y_stiffness.wl',
              ROOT/'examples/nbcp_ground_state.py',
              ROOT/'model/__init__.py', ROOT/'model/nbcp/__init__.py',
              ROOT/'model/nbcp/exchange.py', ROOT/'model/nbcp/unit_cells.py',
              ROOT/'code-space/lswt/methods/spin_wave/hamiltonian.py']
    report = {'scope': 'T=0 classical zero-SOC Y stiffness and harmonic dynamics only; no renormalized thermal stiffness',
              'parameters': {'S': S, 'J_meV': J, 'Jz_meV': JZ, 'gz': GZ, 'muB_meV_per_T': MU_B,
                             'a': 1., 'area_per_spin_a_squared': AREA},
              'definitions': {'rho': 'energy per area = rho |grad phi|^2 / 2; rho in meV',
                              'q': 'physical Cartesian wavevector, inverse NN distance; no 2pi rescaling',
                              'relaxation': 'five regular tangent variables; global azimuth gauge removed, axial spin allowed to tilt'},
              'symbolic_checks': symbolic['checks'],
              'independent_torus_energy_max_absolute_error_meV_per_spin': max_torus_error,
              'independent_dynamics_spectrum_max_absolute_error_meV': max_spectrum_error,
              'results': results,
              'inputs_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              'limitations': ['No quantum zero-point correction to stiffness.', 'No finite-temperature or vortex calculation.',
                              'No global comparison with competing phases.', 'SOC and V are not treated.',
                              'The single-mode reduction excludes additional soft-mode endpoints.']}
    (OUT/'stiffness-check.json').write_text(json.dumps(report, indent=2)+'\n')
    plot(results)
    print(json.dumps({'B_T': [.05,.1,.2,.3,.4],
                      'rho_meV': [r['rho_classical_meV'] for r in results],
                      'max_finest_static_relative_error': max(r['finest_static_relative_error_max'] for r in results),
                      'max_finest_LSWT_relative_error': max(r['finest_LSWT_relative_error_max'] for r in results),
                      'independent_spectrum_max_error_meV': max_spectrum_error,
                      'independent_torus_max_error_meV_per_spin': max_torus_error}, indent=2))


def plot(results):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout='constrained')
    field = np.linspace(.02, .414, 160)
    c = (GZ*MU_B*field + 3*S*JZ)/(3*S*(J+JZ))
    axes[0].plot(field, 1e3*J*S*S*(1-c*c)/np.sqrt(3), color='#126b89', label='Analytical classical result')
    for key, marker, label in [('rho_relaxed_meV', 'o', 'Relaxed energy difference'),
                                ('rho_from_LSWT_meV', 'x', 'LSWT dispersion')]:
        values = [next(s[key] for s in r['samples'] if s['qa']==.001 and s['direction_rad']==0.) for r in results]
        axes[0].plot([r['B_T'] for r in results], 1e3*np.array(values), marker, fillstyle='none', label=label)
    axes[0].set(xlabel='Magnetic field B (T)', ylabel=r'Classical stiffness $\rho_s$ ($10^{-3}$ meV)', title='Y state, zero PD and Gamma')
    axes[0].legend(fontsize=8)
    selected = next(r for r in results if r['B_T']==.2)
    rows = [r for r in selected['samples'] if r['direction_rad']==0.]
    for key, marker, label in [('rho_frozen_meV','s','Fixed internal spins'),
                               ('rho_relaxed_meV','o','Relaxed internal spins'),
                               ('rho_from_LSWT_meV','x','LSWT dispersion')]:
        axes[1].loglog([r['qa'] for r in rows], [abs(r[key]/selected['rho_classical_meV']-1) for r in rows],
                       marker+'-', label=label)
    axes[1].set(xlabel=r'Twist or momentum $qa$', ylabel=r'Relative error in $\rho_s$', title='Convergence at B = 0.2 T')
    axes[1].legend(fontsize=8)
    fig.savefig(OUT/'y-stiffness-validation.png', dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    main()
