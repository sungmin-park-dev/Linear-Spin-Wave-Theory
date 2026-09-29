"""Match the Y angular stiffness and leading vacuum potential at B=0.2 T.

This diagnostic preserves the current microscopic Hamiltonian. It combines
classical gradient coefficients with a leading T=0 vacuum potential, not a
quantum-renormalized stiffness or thermal RG initial condition. Numerical
resolution estimates below are empirical diagnostics, not certified bounds.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lswt-mpl-cache')

import numpy as np
from scipy.optimize import minimize_scalar

from examples.nbcp_y_soc_conditions import background, reduction, static_kernel
from examples.nbcp_y_stability import COUPLINGS, TO_NAMBU, batch_kernel
from examples.nbcp_y_stiffness import AREA, J, JZ, METRIC, S
from examples.pseudo_goldstone_comparison import mesh, vacuum_energy
from spintoolkit.methods.lswt.hamiltonian import LSWTHamiltonian

OUT = ROOT / 'data-space/verification/260918-y-angular-matching'
OLD = ROOT / 'data-space/verification/260912-pseudo-goldstone'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fourier(values):
    """Return unconstrained cosine and sine coefficients below Nyquist."""
    values = np.asarray(values)
    centered = values - values.mean(axis=0)
    coeff = 2 * np.fft.rfft(centered, axis=0) / len(values)
    return coeff[1:-1].real, -coeff[1:-1].imag


def evaluate(phi, a, b, order=0):
    modes = np.arange(1, len(a) + 1)
    angles = np.asarray(phi)[..., None] * modes
    if order == 2:
        a, b = -modes**2 * a, -modes**2 * b
    return np.cos(angles) @ a + np.sin(angles) @ b


def tensor_error(tensors, reference):
    """Max relative error in directional gradient energy, relative to exact rho."""
    values, vectors = np.linalg.eigh(tensors)
    invroot = (vectors * (1 / np.sqrt(values))[:, None, :]) @ vectors.transpose(0, 2, 1)
    residual = invroot @ (reference - tensors) @ invroot
    return float(np.max(abs(np.linalg.eigvalsh(residual))))


def stiffness(pd, gamma):
    phi = 2 * np.pi * np.arange(576) / 576
    tensors = np.array([reduction(background(pd, gamma, p))[4] / (3 * AREA) for p in phi])
    values, vectors = np.linalg.eigh(tensors)
    a, b = fourier(tensors)
    mean = tensors.mean(axis=0)
    alpha = np.pi / 3
    rotation = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    covariance = float(np.max(abs(np.roll(tensors, -96, axis=0) - rotation @ tensors @ rotation.T)))
    assert covariance < 2e-15
    assert np.max(abs(a[4:])) < 2e-15 and np.max(abs(b[4:])) < 2e-15
    if pd == gamma == 0:
        axes = None  # An isotropic tensor has no defined principal direction.
    else:
        axes = (np.arctan2(vectors[:, 1, 0], vectors[:, 0, 0]) % np.pi).tolist()
    convergence = []
    for count in [72, 144, 288]:
        coarse = tensors[::576 // count]
        ac, bc = fourier(coarse)
        convergence.append({'P': count,
            'rho_n1_to_n4_max_coefficient_difference_meV': float(max(np.max(abs(ac[:4]-a[:4])), np.max(abs(bc[:4]-b[:4])))),
            'eigenvalue_extrema_max_difference_meV': float(max(np.max(abs(np.linalg.eigvalsh(coarse).min(axis=0)-values.min(axis=0))), np.max(abs(np.linalg.eigvalsh(coarse).max(axis=0)-values.max(axis=0))))),
            'sqrt_determinant_extrema_max_difference_meV': float(max(abs(np.sqrt(np.linalg.det(coarse)).min()-np.sqrt(np.linalg.det(tensors)).min()), abs(np.sqrt(np.linalg.det(coarse)).max()-np.sqrt(np.linalg.det(tensors)).max())))})
    check = []
    for p in [.137, .419]:
        bg = background(pd, gamma, p)
        g, _, _, _, c, chi, _ = reduction(bg)
        ham = LSWTHamiltonian(bg[6]['Spin info'], bg[6]['Couplings'])
        for angle in [.23, 1.13]:
            direction = np.array([np.cos(angle), np.sin(angle)])
            for step in [.004, .002, .001]:
                k = step * direction
                exact = direction @ c @ direction / (3*AREA)
                static = static_kernel(k, bg, g)[0] / (3*AREA*step**2)
                matrices, _ = ham.Quadratic_Bose_Hamiltonian(np.array([k, -k]), angles=bg[5])
                spectra = np.linalg.eigvals(METRIC @ matrices)
                assert np.max(abs(spectra.imag)) < 1e-10
                eps = np.sort(spectra.real, axis=1)[:, 3]
                assert np.all(eps > 0)
                paired = chi * eps.prod() / (3*AREA*step**2)
                check.append({'phi': p, 'direction_rad': angle, 'ka': step,
                    'static_relative_error': float(abs(static/exact-1)),
                    'production_paired_relative_error': float(abs(paired/exact-1))})
    assert max(r['static_relative_error'] for r in check if r['ka'] == .001) < 3e-6
    assert max(r['production_paired_relative_error'] for r in check if r['ka'] == .001) < 3e-6
    return {'JPD_meV': pd, 'JGamma_meV': gamma, 'phi': phi.tolist(),
        'rho_tensor_meV': tensors.tolist(), 'eigenvalues_meV': values.tolist(),
        'determinant_meV_squared': np.linalg.det(tensors).tolist(),
        'minor_principal_axis_rad_mod_pi': axes, 'angular_mean_tensor_meV': mean.tolist(),
        'component_harmonics': list(range(1, 5)), 'component_cos_meV': a[:4].tolist(),
        'component_sin_meV': b[:4].tolist(), 'max_component_amplitude_above_n4_meV': float(np.max(np.hypot(a[4:], b[4:]))),
        'sixty_degree_tensor_covariance_max_meV': covariance,
        'eigenvalue_min_meV': values.min(axis=0).tolist(), 'eigenvalue_max_meV': values.max(axis=0).tolist(),
        'sqrt_det_min_meV': float(np.sqrt(np.linalg.det(tensors)).min()),
        'sqrt_det_max_meV': float(np.sqrt(np.linalg.det(tensors)).max()),
        'constant_angular_mean_max_relative_directional_error': tensor_error(tensors, mean),
        'constant_phi_zero_max_relative_directional_error': tensor_error(tensors, tensors[0]),
        'angular_convergence': convergence, 'finite_k_checks': check}


def energy_at(points, pd, gamma, phi):
    bg = background(pd, gamma, phi)
    matrices = TO_NAMBU.conj().T @ batch_kernel(points, bg) @ TO_NAMBU
    return vacuum_energy(matrices)[0]


def new_scan(pd, gamma, n, count, old=None):
    """Compute a full circle; reuse saved even angles when doubling P=72."""
    tag = f'scan-pd{pd:g}-gamma{gamma:g}-N{n}-P{count}.json'
    path = OUT / tag
    dependencies = [Path(__file__), ROOT/'examples/nbcp_y_stability.py',
                    ROOT/'examples/nbcp_y_soc_conditions.py', ROOT/'examples/nbcp_y_stiffness.py',
                    ROOT/'examples/pseudo_goldstone_comparison.py', ROOT/'examples/nbcp_ground_state.py',
                    ROOT/'model/__init__.py', ROOT/'model/nbcp/__init__.py',
                    ROOT/'model/nbcp/exchange.py', ROOT/'model/nbcp/unit_cells.py',
                    ROOT/'code-space/spintoolkit/methods/lswt/hamiltonian.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in dependencies}
    if path.exists():
        row = json.loads(path.read_text())
        if row['source_sha256'] == hashes:
            return row
    points = mesh(n)
    if old is not None:
        replay_error = abs(energy_at(points, pd, gamma, 0.)-old['energy_meV_per_spin'][0])
        assert replay_error < 1e-16
    phi = 2*np.pi*np.arange(count)/count
    energy = np.empty(count)
    start = time.monotonic()
    for i, p in enumerate(phi):
        if old is not None and i % 2 == 0:
            energy[i] = old['energy_meV_per_spin'][i//2]
        else:
            energy[i] = energy_at(points, pd, gamma, p)
        if i % 24 == 0:
            print(f'{tag}: {i}/{count}', flush=True)
    row = {'JPD_meV': pd, 'JGamma_meV': gamma, 'N': n, 'P': count,
        'phi': phi.tolist(), 'energy_meV_per_spin': energy.tolist(),
        'quadrature': 'Midpoint reciprocal-cell grid and six rotated copies; no angular symmetry imposed.',
        'source': str(path.relative_to(ROOT)),
        'reused_even_angles_from': old.get('source') if old is not None else None,
        'saved_angle_replay_error_meV_per_spin': replay_error if old is not None else None,
        'wall_seconds': time.monotonic()-start, 'source_sha256': hashes}
    path.write_text(json.dumps(row, indent=2)+'\n')
    return row


def potential_summary(scan):
    en = np.array(scan['energy_meV_per_spin'])
    a, b = fourier(en)
    amplitude = np.hypot(a, b)
    modes = np.arange(1, len(a)+1)
    forbidden = modes % 6 != 0
    # This intentionally conservative floor does not claim statistical coverage.
    floor = max(100*np.finfo(float).eps*np.max(abs(en)), np.max(amplitude[forbidden]))
    resolved = amplitude > floor
    ar, br = np.where(resolved, a, 0), np.where(resolved, b, 0)
    phi = np.linspace(0, 2*np.pi, 14401)[:-1]
    p0 = phi[np.argmin(evaluate(phi, ar, br))]
    result = minimize_scalar(lambda x: float(evaluate(x, ar, br)), bounds=(p0-.002, p0+.002),
                             method='bounded', options={'xatol': 1e-13})
    minimum = float(result.x % (np.pi/3))
    curvature = float(evaluate(minimum, ar, br, 2))
    a6, b6 = np.zeros_like(a), np.zeros_like(b)
    a6[5], b6[5] = a[5], b[5]
    full = evaluate(phi, a, b)
    single = evaluate(phi, a6, b6)
    higher = (modes != 6) & resolved
    return {'N': scan['N'], 'P': scan['P'], 'harmonics': modes.tolist(),
        'cos_meV_per_spin': a.tolist(), 'sin_meV_per_spin': b.tolist(),
        'amplitudes_meV_per_spin': amplitude.tolist(),
        'lambda6_meV_per_spin': float(amplitude[5]),
        'lambda12_over_lambda6': float(amplitude[11]/amplitude[5]),
        'lambda18_over_lambda6': float(amplitude[17]/amplitude[5]),
        'max_forbidden_amplitude_meV_per_spin': float(np.max(amplitude[forbidden])),
        'empirical_coefficient_floor_meV_per_spin': float(floor),
        'resolved_modes_under_empirical_floor': modes[resolved].tolist(),
        'higher_harmonic_weighted_curvature_ratio': float(np.sum(modes[higher]**2*amplitude[higher])/(36*amplitude[5])),
        'sixth_only_max_energy_error_over_lambda6': float(np.max(abs(full-single))/amplitude[5]),
        'phi_min_rad_mod_pi_over_three': minimum,
        'resolved_curvature_at_min_meV_per_spin': curvature,
        'sixth_only_min_curvature_relative_error': float(abs(36*amplitude[5]/curvature-1)),
        'conservative_unresolved_curvature_diagnostic_meV_per_spin': float(np.sum(modes[~resolved]**2)*floor),
        'max_sixty_degree_shift_residual_meV_per_spin': float(np.max(abs(en-np.roll(en, len(en)//6)))),
        'nyquist_amplitude_meV_per_spin': float(abs(np.fft.rfft(en-en.mean())[-1])/len(en))}


def independent_energy_checks():
    """Compare Cartesian and production matrices and independent dynamics sums."""
    points = mesh(12)
    rows = []
    for pd, gamma in COUPLINGS:
        for phi in [.173, .437]:
            bg = background(pd, gamma, phi)
            tangent = TO_NAMBU.conj().T @ batch_kernel(points, bg) @ TO_NAMBU
            ham = LSWTHamiltonian(bg[6]['Spin info'], bg[6]['Couplings'])
            production, torques = ham.Quadratic_Bose_Hamiltonian(points, angles=bg[5])
            matrix_error = float(np.max(abs(tangent-production)))
            e1 = vacuum_energy(tangent)[0]
            e2 = vacuum_energy(production)[0]
            eigen = np.linalg.eigvals(METRIC @ production)
            assert np.max(abs(eigen.imag)) < 1e-11
            positive = np.sort(eigen.real, axis=1)[:, 3:]
            e3 = np.mean(positive.sum(axis=1)/2 - np.trace(production, axis1=1, axis2=2).real/4)/3
            assert matrix_error < 1e-13 and abs(e1-e3) < 1e-14
            rows.append({'JPD_meV': pd, 'JGamma_meV': gamma, 'phi': phi,
                'matrix_max_difference_meV': matrix_error, 'energy_production_difference_meV_per_spin': abs(e1-e2),
                'energy_direct_eig_difference_meV_per_spin': float(abs(e1-e3)),
                'max_torque': float(max(abs(v) for v in torques.values()))})
    p = 2*np.pi*np.arange(144)/144
    a, b = fourier(3*np.cos(6*p+.23)+.04*np.sin(12*p))
    assert np.max(abs(evaluate(p, a, b)-(3*np.cos(6*p+.23)+.04*np.sin(12*p)))) < 1e-12
    return rows


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), layout='constrained')
    for col, pair in enumerate([(.01, 0.), (0., .01), (.01, .01)]):
        st = next(x for x in report['stiffness'] if (x['JPD_meV'], x['JGamma_meV']) == pair)
        en = next(x for x in report['potential'] if (x['JPD_meV'], x['JGamma_meV']) == pair)
        phi = np.degrees(st['phi']); values = 1e3*np.array(st['eigenvalues_meV'])
        mask = phi <= 60
        ax = axes[0, col]
        ax.plot(phi[mask], values[mask, 0], label=r'$\rho_-$', color='#186c79')
        ax.plot(phi[mask], values[mask, 1], label=r'$\rho_+$', color='#b75b40')
        ax.plot(phi[mask], 1e3*np.sqrt(st['determinant_meV_squared'])[mask], '--', color='#555555', label=r'$\sqrt{\det\rho}$')
        ax.set(title=['Pure PD', 'Pure Gamma', 'Mixed PD + Gamma'][col], ylabel='Classical stiffness (10$^{-3}$ meV)', xlim=(0, 60))
        ax.legend(fontsize=8, loc='best')
        scan = en['finest_angular_scan']; p = np.array(scan['phi']); e = np.array(scan['energy_meV_per_spin'])
        unit = 1e12 if col == 1 else 1e6
        summary = en['summaries'][-2]  # N48/P144, preceding N96/P72.
        a6, b6 = summary['cos_meV_per_spin'][5], summary['sin_meV_per_spin'][5]
        smooth = np.linspace(0, np.pi/3, 401)
        single = a6*np.cos(6*smooth)+b6*np.sin(6*smooth)
        ax = axes[1, col]; mask = np.degrees(p) <= 60
        ax.plot(np.degrees(p[mask]), unit*(e[mask]-e.mean()), 'o', ms=3.5, color='#186c79', label='All computed harmonics')
        ax.plot(np.degrees(smooth), unit*single, '-', color='#b75b40', lw=1.2, label='Sixth harmonic')
        ax.set(xlabel=r'Common spin angle $\phi$ (degrees)', ylabel='Centered energy ('+('10$^{-12}$ meV/spin)' if col == 1 else 'neV/spin)'), xlim=(0, 60))
        ax.legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([0, 15, 30, 45, 60]); ax.grid(alpha=.18); ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Y at 0.2 T: angle-dependent classical stiffness and leading vacuum potential\nActive SOC = 0.010 meV; energy panels use different absolute scales', fontsize=11)
    for suffix in ['png', 'svg']:
        fig.savefig(OUT/f'y-angular-matching.{suffix}', dpi=220)
    plt.close(fig)


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    energy_checks = independent_energy_checks()
    stiff = [stiffness(pd, gamma) for pd, gamma in COUPLINGS]
    old = {}
    for n in [12, 24, 48]:
        path = OLD/f'scan-N{n}-P72.json'
        for scan in json.loads(path.read_text())['scans']:
            pair = (scan['JPD_meV'], scan['JGamma_meV'])
            if scan['phase'] == 'Y' and pair in COUPLINGS:
                old[pair, n] = {'N': n, 'P': 72, 'phi': scan['phi'],
                    'energy_meV_per_spin': scan['current']['Ezp_meV_per_spin'],
                    'source': str(path.relative_to(ROOT)), 'source_sha256': digest(path)}
    potentials = []
    for pd, gamma in COUPLINGS[1:]:
        pair = (pd, gamma)
        scans = [old[pair, n] for n in [12, 24, 48] if (pair, n) in old]
        if not scans:
            scans = [new_scan(pd, gamma, 24, 72), new_scan(pd, gamma, 48, 72)]
        finest = new_scan(pd, gamma, 48, 144, scans[-1])
        scans += [finest, new_scan(pd, gamma, 96, 72)]
        summaries = [potential_summary(scan) for scan in scans]
        angular = np.array(summaries[-2]['amplitudes_meV_per_spin'][:35])-np.array(summaries[-3]['amplitudes_meV_per_spin'][:35])
        momentum = np.array(summaries[-1]['amplitudes_meV_per_spin'])-np.array(summaries[-3]['amplitudes_meV_per_spin'])
        potentials.append({'JPD_meV': pd, 'JGamma_meV': gamma, 'summaries': summaries,
            'finest_angular_scan': finest,
            'P72_to_144_N48_max_amplitude_change_meV_per_spin': float(np.max(abs(angular))),
            'N48_to_96_P72_lambda6_change_meV_per_spin': float(momentum[5]),
            'N48_to_96_P72_lambda6_relative_change': float(momentum[5]/summaries[-1]['lambda6_meV_per_spin']),
            'N48_to_96_P72_lambda12_change_meV_per_spin': float(momentum[11])})
    mixed_checks = []
    for coupling in [.005, .01]:
        def coefficient(pair):
            row = next(x for x in potentials if (x['JPD_meV'], x['JGamma_meV']) == pair)['summaries'][-1]
            return np.array(row['cos_meV_per_spin'])+1j*np.array(row['sin_meV_per_spin'])
        mixed = coefficient((coupling, coupling))
        additive = coefficient((coupling, 0.)) + coefficient((0., coupling))
        mixed_checks.append({'coupling_meV': coupling,
            'sixth_coefficient_nonadditivity_meV_per_spin': float(abs(mixed[5]-additive[5])),
            'relative_to_mixed_lambda6': float(abs(mixed[5]-additive[5])/abs(mixed[5]))})
    inputs = [Path(__file__), ROOT/'examples/nbcp_y_soc_conditions.py', ROOT/'examples/nbcp_y_stiffness.py',
        ROOT/'examples/nbcp_y_stability.py', ROOT/'examples/pseudo_goldstone_comparison.py',
        ROOT/'examples/nbcp_ground_state.py',
        ROOT/'model/__init__.py', ROOT/'model/nbcp/__init__.py',
        ROOT/'model/nbcp/exchange.py', ROOT/'model/nbcp/unit_cells.py',
        ROOT/'code-space/spintoolkit/methods/lswt/hamiltonian.py',
        ROOT/'data-space/verification/260917-y-stability/stability-check.json',
        OLD/'curvature-validation.json'] + [OLD/f'scan-N{n}-P72.json' for n in [12, 24, 48]]
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'T=0 classical gradient / leading vacuum-potential matching on the fixed classical Y orbit; human physics review pending.',
        'parameters': {'B_T': .2, 'J_meV': J, 'Jz_meV': JZ, 'S': S, 'area_per_spin_a_squared': AREA},
        'fourier_convention': 'e-mean = sum [a_n cos(n phi)+b_n sin(n phi)], lambda_n=hypot(a_n,b_n), units meV/spin.',
        'resolution_convention': 'max(100 eps max|e|, largest forbidden harmonic); diagnostic only, not a rigorous error bound. Report mesh/angle changes separately.',
        'gradient_error_convention': 'sup_phi,q |q^T(rho_reference-rho(phi))q|/[q^T rho(phi)q], sampled in phi, all q via generalized eigenvalues.',
        'independent_energy_checks': energy_checks, 'stiffness': stiff, 'potential': potentials,
        'mixed_SOC_nonadditivity': mixed_checks,
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in inputs},
        'wall_seconds': time.monotonic()-start,
        'python_peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'limitations': ['No quantum gradient correction or quantum hard-coordinate reoptimization.',
            'Finite angular/momentum grids do not bound all unresolved harmonics.',
            'Six rotated momentum grids preserve lattice quadrature symmetry; no harmonic selection imposed in Fourier extraction.',
            'Pure-Gamma higher harmonics lie below the empirical resolution; extra displayed digits are computational diagnostics only.',
            'No matched thermal stiffness/potential, matching scale, vortex fugacity, defect optimization or finite-size thermal study.',
            'Current code momentum convention retained; laboratory sign of odd-in-k response unresolved.']}
    (OUT/'angular-matching-check.json').write_text(json.dumps(report, indent=2)+'\n')
    plot(report)
    print(json.dumps({'wall_seconds': report['wall_seconds'], 'peak_RSS_MiB': report['python_peak_RSS_MiB'],
        'stiffness_max_relative_errors': [x['constant_angular_mean_max_relative_directional_error'] for x in stiff],
        'mixed_SOC_nonadditivity': mixed_checks}, indent=2), flush=True)


if __name__ == '__main__':
    main()
