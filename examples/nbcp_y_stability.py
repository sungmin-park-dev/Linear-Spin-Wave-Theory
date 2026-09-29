"""Full magnetic-zone harmonic stability screen of the classical Y orbit.

This is a diagnostic, not a phase diagram or a quantum/thermal matching.
Static eigenvalues refer to dimensionless orthonormal spin tangents and have
units of meV per magnetic cell. They are not magnon energies. The numerical
thresholds below are adjustable diagnostics, not eigenvalue error bounds.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT), str(ROOT / 'legacy')]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lswt-mpl-cache')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from examples.nbcp_y_soc_conditions import background, kernel, reduction
from examples.nbcp_y_stiffness import AREA, DELTAS, J, JZ, METRIC, PAIRS, POISSON, S
from spintoolkit.methods.lswt.hamiltonian import LSWTHamiltonian
from modules.LinearSpinWaveTheory.lswt_Hamiltonian import LSWT_HAMILTONIAN

OUT = ROOT / 'data-space/verification/260917-y-stability'
LATTICE = np.array([[1.5, np.sqrt(3)/2], [1.5, -np.sqrt(3)/2]])
RECIPROCAL = 2*np.pi*np.linalg.inv(LATTICE).T
TRANSLATIONS = np.array([(i, j) for i in [-1, 0, 1] for j in [-1, 0, 1]]) @ RECIPROCAL
IDENTITY = np.eye(3)
TO_NAMBU = np.block([[IDENTITY, IDENTITY], [-1j*IDENTITY, 1j*IDENTITY]]) / np.sqrt(2*S)
COUPLINGS = [(0., 0.), (.005, 0.), (0., .005), (.005, .005),
             (.010, 0.), (0., .010), (.010, .010)]


def batch_kernel(points, bg):
    """Vectorized independent Cartesian Hessian, with fixed-length correction."""
    h, _, normals, basis, exchanges, *_ = bg
    matrix = np.zeros((len(points), 6, 6), complex)
    for i in range(3):
        matrix[:, i, i] = matrix[:, i+3, i+3] = h*S*normals[i, 2]
    for i, j in PAIRS:
        for d, exchange in zip(DELTAS, exchanges):
            longitudinal = S*S*normals[i] @ exchange @ normals[j]
            for index in [i, i+3, j, j+3]:
                matrix[:, index, index] -= longitudinal
            block = S*S*(basis[i].T @ exchange @ basis[j])
            phase = np.exp(-1j*points @ d)
            for a, ia in enumerate([i, i+3]):
                for b, jb in enumerate([j, j+3]):
                    matrix[:, ia, jb] += block[a, b]*phase
                    matrix[:, jb, ia] += (block[a, b]*phase).conj()
    return matrix


def mesh(n):
    """Gamma-inclusive nested grid on a complete reciprocal primitive cell."""
    f = np.arange(n)/n - .5
    uv = np.stack(np.meshgrid(f, f, indexing='ij'), axis=-1).reshape(-1, 2)
    return uv, uv @ RECIPROCAL


def gamma_distance(points):
    return np.linalg.norm(points[:, None, :] - TRANSLATIONS[None, :, :], axis=-1).min(axis=1)


def inspect_mesh(n, bg, static_tol, imag_tol):
    uv, points = mesh(n)
    matrices = batch_kernel(points, bg)
    eigenvalues = np.linalg.eigvalsh(matrices)
    dynamic = np.linalg.eigvals(1j*POISSON @ matrices)
    distance = gamma_distance(points)
    nongamma = distance > 1e-10
    imag = np.abs(dynamic.imag).max(axis=1)
    # Do not turn unstable complex frequencies into magnon energies.
    valid = (eigenvalues[:, 0] >= -static_tol) & (imag <= imag_tol) & nongamma
    positive_count = (dynamic.real > imag_tol).sum(axis=1)
    valid &= positive_count == 3
    sorted_real = np.sort(dynamic.real, axis=1)
    minima = {}
    for index in [0, 1]:
        loc = int(np.argmin(eigenvalues[:, index]))
        minima[f'static_lambda{index+1}'] = {
            'value_meV_per_cell': float(eigenvalues[loc, index]),
            'reciprocal_fraction': uv[loc].tolist(), 'k': points[loc].tolist()}
    for radius in [.05, .1, .2]:
        mask = distance >= radius
        indices = np.flatnonzero(mask)
        loc = int(indices[np.argmin(eigenvalues[mask, 0])])
        minima[f'static_lambda1_outside_{radius}'] = {
            'value_meV_per_cell': float(eigenvalues[loc, 0]), 'k': points[loc].tolist()}
    # At Gamma the two zero algebraic roots occupy positions 2 and 3;
    # position 4 is still the second nonnegative branch. Do not discard its
    # finite hard energy just because the soft pair is numerically defective.
    energy_valid = valid | ((~nongamma) & (eigenvalues[:, 0] >= -static_tol))
    min_second = float(sorted_real[energy_valid, 4].min()) if energy_valid.any() else None
    row = {'N': n, 'n_k': n*n, 'negative_static_points': int((eigenvalues[:, 0] < -static_tol).sum()),
           'complex_dynamic_points_excluding_gamma': int(((imag > imag_tol) & nongamma).sum()),
           'max_imaginary_energy_excluding_gamma_meV': float(imag[nongamma].max()),
           'gamma_imaginary_roundoff_meV': float(imag[~nongamma].max()),
           'minima': minima, 'minimum_second_positive_energy_meV': min_second,
           'invalid_positive_count_at_static_stable_nongamma_points': int((nongamma &
                (eigenvalues[:, 0] >= -static_tol) & (imag <= imag_tol) & (positive_count != 3)).sum())}
    return row, (uv, points, eigenvalues)


def refine_static(bg, grid, index):
    """Multistart search beyond the mesh; not a certified global bound."""
    uv, _, values = grid
    seeds = []
    for loc in np.argsort(values[:, index]):
        p = uv[loc]
        if all(np.linalg.norm(((p-s+.5) % 1)-.5) > .12 for s in seeds):
            seeds.append(p)
        if len(seeds) == 6:
            break
    def objective(fraction):
        return float(np.linalg.eigvalsh(kernel(fraction @ RECIPROCAL, bg))[index])
    results = []
    for seed in seeds:
        result = minimize(objective, seed, method='Nelder-Mead',
                          options={'xatol': 2e-9, 'fatol': 1e-14, 'maxiter': 350})
        fraction = (result.x+.5) % 1 - .5
        results.append({'value_meV_per_cell': objective(fraction),
                        'reciprocal_fraction': fraction.tolist(),
                        'k': (fraction @ RECIPROCAL).tolist(), 'success': bool(result.success),
                        'evaluations': int(result.nfev)})
    return {'minimum': min(results, key=lambda x: x['value_meV_per_cell']), 'starts': results}


def phase_projector_check(bg):
    """Follow the static low-mode projector only in a specified small-k disk.

    The separated lowest static eigenvalue defines a projector, insensitive
    to arbitrary eigenvector phases. This does not assign a global magnon
    branch identity across crossings.
    """
    g, hard, _, _, c, chi, drift = reduction(bg)
    unit = g / np.linalg.norm(g)
    steps = np.array([.001, .003, .01, .03, .06, .1])
    rows = []
    for angle in np.arange(12)*np.pi/6:
        previous = unit.astype(complex)
        for step in steps:
            q = step*np.array([np.cos(angle), np.sin(angle)])
            values, vectors = np.linalg.eigh(kernel(q, bg))
            vector = vectors[:, 0]
            rows.append({'ka': float(step), 'direction_rad': float(angle),
                         'overlap_with_global_phase': float(abs(unit.conj() @ vector)**2),
                         'overlap_with_previous_projector': float(abs(previous.conj() @ vector)**2),
                         'static_separation_meV_per_cell': float(values[1]-values[0])})
            previous = vector
    return {'rho_tensor_meV': (c/(3*AREA)).tolist(),
            'rho_eigenvalues_meV': np.linalg.eigvalsh(c/(3*AREA)).tolist(),
            'uniform_hard_eigenvalues_meV_per_cell': np.linalg.eigvalsh(hard).tolist(),
            'chi_per_spin_meV_inverse': chi/3, 'drift_meV_a': drift.tolist(),
            'minimum_neighbor_projector_overlap': min(x['overlap_with_previous_projector'] for x in rows),
            'minimum_global_phase_overlap': min(x['overlap_with_global_phase'] for x in rows),
            'minimum_static_separation_in_disk_meV_per_cell': min(x['static_separation_meV_per_cell'] for x in rows),
            'samples': rows}


def refine_second_energy(bg, grid):
    """Refine the second ordered frequency without assigning band identity."""
    uv, points, _ = grid
    energies = np.sort(np.linalg.eigvals(1j*POISSON @ batch_kernel(points, bg)).real, axis=1)[:, 4]
    seeds = []
    for loc in np.argsort(energies):
        p = uv[loc]
        if all(np.linalg.norm(((p-s+.5) % 1)-.5) > .05 for s in seeds):
            seeds.append(p)
        if len(seeds) == 4:
            break
    def objective(fraction):
        values = np.linalg.eigvals(1j*POISSON @ kernel(fraction @ RECIPROCAL, bg))
        return float(np.sort(values.real)[4])
    results = []
    for seed in seeds:
        result = minimize(objective, seed, method='Nelder-Mead',
                          options={'xatol': 2e-9, 'fatol': 1e-14, 'maxiter': 350})
        fraction = (result.x+.5) % 1 - .5
        k = fraction @ RECIPROCAL
        static = np.linalg.eigvalsh(kernel(k, bg))
        dynamic = np.linalg.eigvals(1j*POISSON @ kernel(k, bg))
        assert static[0] > -1e-10 and np.max(abs(dynamic.imag)) < 1e-8
        results.append({'value_meV': objective(fraction), 'reciprocal_fraction': fraction.tolist(),
                        'k': k.tolist(), 'success': bool(result.success), 'evaluations': int(result.nfev)})
    return {'minimum': min(results, key=lambda x: x['value_meV']), 'starts': results}


def validation(static_tol, imag_tol):
    rng = np.random.default_rng(9172026)
    checks = []
    for pd, gamma, phi in [(0., 0., 0.), (.01, 0., .173), (0., .01, .31), (.01, .01, .47)]:
        bg = background(pd, gamma, phi)
        points = rng.uniform(-.5, .5, (19, 2)) @ RECIPROCAL
        actual = batch_kernel(points, bg)
        original = np.array([kernel(q, bg) for q in points])
        transformed = TO_NAMBU.conj().T @ actual @ TO_NAMBU
        ham = LSWTHamiltonian(bg[6]['Spin info'], bg[6]['Couplings'])
        production, torques = ham.Quadratic_Bose_Hamiltonian(points, angles=bg[5])
        archived = LSWT_HAMILTONIAN(bg[6]['Spin info'], bg[6]['Couplings'])
        legacy, _ = archived.Quadratic_Bose_Hamiltonian(points, angles=bg[5])
        periodic = np.linalg.eigvalsh(batch_kernel(points+RECIPROCAL[0], bg))
        row = {'JPD': pd, 'JGamma': gamma, 'phi': phi,
               'vectorized_vs_scalar_kernel_max': float(np.max(abs(actual-original))),
               'tangent_to_production_matrix_max': float(np.max(abs(transformed-production))),
               'legacy_to_tangent_matrix_max': float(np.max(abs(transformed-legacy))),
               'reciprocal_periodicity_max': float(np.max(abs(periodic-np.linalg.eigvalsh(actual)))),
               'max_torque': float(max(abs(v) for v in torques.values()))}
        for key in ['vectorized_vs_scalar_kernel_max', 'tangent_to_production_matrix_max',
                    'reciprocal_periodicity_max', 'max_torque']:
            assert row[key] < 1e-12, row
        checks.append(row)
    # A deliberately large PD coefficient tests the negative-curvature detector.
    control, _ = inspect_mesh(24, background(.08, 0., .173), static_tol, imag_tol)
    assert control['negative_static_points'] > 0
    _, symmetry_points = mesh(12)
    symmetry_errors = []
    for pd, gamma in COUPLINGS:
        first = np.linalg.eigvalsh(batch_kernel(symmetry_points, background(pd, gamma, .173)))
        rotated = np.linalg.eigvalsh(batch_kernel(symmetry_points, background(pd, gamma, .173+np.pi/3)))
        error = float(np.max(abs(np.sort(first.ravel())-np.sort(rotated.ravel()))))
        assert error < 1e-12
        symmetry_errors.append(error)
    return {'matrix_checks': checks, 'sixfold_static_spectral_set_errors': symmetry_errors,
            'unstable_control_JPD_meV': .08,
            'unstable_control': control,
            'legacy_status': 'Differences retained as the documented anomalous-block legacy discrepancy; archive is not the acceptance oracle.'}


def saved_angles(pd, gamma):
    source = ROOT / 'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json'
    scans = json.loads(source.read_text())['scans']
    matches = [x for x in scans if x['phase'] == 'Y' and abs(x['JPD_meV']-pd) < 1e-12
               and abs(x['JGamma_meV']-gamma) < 1e-12]
    if not matches or (pd == 0 and gamma == 0):
        return None
    current = matches[0]['current']
    if current.get('status') != 'resolved':
        return None
    return {'phi_rad': float(current['phi_min']), 'source': str(source.relative_to(ROOT)),
            'status': 'Saved leading-semiclassical orbit minimum; no hard-coordinate quantum relaxation.'}


def make_plot(report):
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)
    for pd, gamma in COUPLINGS:
        rows = [r for r in report['cases'] if r['JPD_meV'] == pd and r['JGamma_meV'] == gamma
                and r['angle_source'] == 'uniform_orbit_sample']
        x = np.array([r['phi_rad'] for r in rows])*180/np.pi
        y0 = [1e3*r['local_phase']['rho_eigenvalues_meV'][0] for r in rows]
        y1 = [1e3*r['refined_static_lambda2']['minimum']['value_meV_per_cell'] for r in rows]
        y2 = [1e3*r['refined_second_energy']['minimum']['value_meV'] for r in rows]
        label = f'({pd:.3f}, {gamma:.3f})'
        for ax, y in zip(axes, [y0, y1, y2]):
            ax.plot(x, y, '.-', ms=3, lw=1.1, label=label)
    axes[0].set_ylabel(r'Minimum stiffness eigenvalue ($\mu$eV)')
    axes[1].set_ylabel(r'Refined $\min_k\lambda_2$ ($\mu$eV/cell)')
    axes[1].set_ylim(0, 25)
    axes[2].set_ylabel(r'Refined $\min_k\varepsilon_2$ ($\mu$eV)')
    for ax in axes:
        ax.set_xlabel(r'Background angle $\phi$ (degrees)')
        ax.grid(alpha=.2)
    axes[2].legend(title=r'$(J_{\rm PD},J_\Gamma)$ in meV', fontsize=7, title_fontsize=8)
    fig.suptitle('Y at 0.2 T: classical harmonic screen; no quantum pinning inserted', fontsize=11)
    fig.savefig(OUT/'y-stability-screen.png', dpi=180)
    fig.savefig(OUT/'y-stability-screen.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meshes', type=int, nargs='+', default=[24, 48, 96])
    parser.add_argument('--angles', type=int, default=12, help='Angles across a 60-degree symmetry sector')
    parser.add_argument('--static-tol', type=float, default=1e-10, help='Negative-curvature threshold in meV/cell')
    parser.add_argument('--imag-tol', type=float, default=1e-8, help='Imaginary-frequency threshold in meV')
    args = parser.parse_args()
    assert all(n >= 6 and n % 2 == 0 for n in args.meshes)
    assert args.static_tol > 0 and args.imag_tol > 0 and args.angles > 0
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    checks = validation(args.static_tol, args.imag_tol)
    cases = []
    for pd, gamma in COUPLINGS:
        angles = [(phi, 'uniform_orbit_sample') for phi in np.arange(args.angles)*np.pi/(3*args.angles)]
        selected = saved_angles(pd, gamma)
        if selected:
            angles.append((selected['phi_rad'], 'saved_quantum_selected_angle'))
        for phi, origin in angles:
            bg = background(pd, gamma, phi)
            row = {'B_T': .2, 'JPD_meV': pd, 'JGamma_meV': gamma, 'phi_rad': float(phi), 'angle_source': origin}
            row['local_phase'] = phase_projector_check(bg)
            row['meshes'] = []
            for n in args.meshes:
                record, grid = inspect_mesh(n, bg, args.static_tol, args.imag_tol)
                row['meshes'].append(record)
            row['refined_static_lambda1'] = refine_static(bg, grid, 0)
            row['refined_static_lambda2'] = refine_static(bg, grid, 1)
            row['refined_second_energy'] = refine_second_energy(bg, grid)
            k0 = np.linalg.eigvalsh(kernel(np.zeros(2), bg))
            row['gamma_static_eigenvalues_meV_per_cell'] = k0.tolist()
            row['gamma_static_nullity_at_tolerance'] = int((abs(k0) <= args.static_tol).sum())
            assert row['gamma_static_nullity_at_tolerance'] == 1
            if selected and origin == 'saved_quantum_selected_angle':
                row['saved_selection'] = selected
            row['screen_passed'] = (all(m['negative_static_points'] == 0 and
                m['complex_dynamic_points_excluding_gamma'] == 0 and
                m['invalid_positive_count_at_static_stable_nongamma_points'] == 0 for m in row['meshes'])
                and row['refined_static_lambda1']['minimum']['value_meV_per_cell'] >= -args.static_tol)
            cases.append(row)
        print(json.dumps({'couplings_meV': [pd, gamma], 'angles': len(angles),
                          'elapsed_seconds': round(time.perf_counter()-start, 2),
                          'screen_passed': all(x['screen_passed'] for x in cases[-len(angles):])}), flush=True)
    inputs = [Path(__file__), ROOT/'examples/nbcp_y_soc_conditions.py', ROOT/'examples/nbcp_y_stiffness.py',
              ROOT/'examples/nbcp_ground_state.py',
              ROOT/'model/__init__.py', ROOT/'model/nbcp/__init__.py',
              ROOT/'model/nbcp/exchange.py', ROOT/'model/nbcp/unit_cells.py',
              ROOT/'code-space/spintoolkit/methods/lswt/hamiltonian.py',
              ROOT/'code-space/spintoolkit/system/exchange.py', ROOT/'legacy/modules/LinearSpinWaveTheory/lswt_Hamiltonian.py',
              ROOT/'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json']
    report = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Full-zone numerical stability screen of the classical Y orbit at B=0.2 T; not a global phase or thermal calculation.',
              'parameters': {'J_meV': J, 'Jz_meV': JZ, 'S': S, 'B_T': .2, 'a': 1., 'temperature_K': 0.},
              'sampling': {'meshes': args.meshes, 'orbit_angles_per_60_degrees': args.angles,
                           'reciprocal_rows': RECIPROCAL.tolist(), 'origin_included': True},
              'thresholds': {'static_negative_meV_per_cell': args.static_tol, 'imaginary_energy_meV': args.imag_tol,
                             'meaning': 'User-adjustable numerical diagnostics; not rigorous error bounds or physical degeneracy definitions.'},
              'validation': checks, 'cases': cases,
              'elapsed_seconds': time.perf_counter()-start,
              'peak_process_rss_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == 'darwin' else 1024),
              'limitations': ['Finite grids, finite angular sampling and local multistart refinement are not certified global minima.',
                              'Second sorted positive frequency is a multiplicity diagnostic, not an identified hard branch across crossings.',
                              'Phase projector continuity is checked only for ka<=0.1 in 12 directions; no global branch labeling.',
                              'Static Hessian eigenvalues are not physical excitation energies.',
                              'No competing-phase comparison, quantum hard-coordinate relaxation, self-energy, or finite-temperature calculation.',
                              'Stored bond directions use current code coordinates; odd-in-k physical orientation remains open.',
                              'SOC values are test parameters, not measured material estimates.'],
              'inputs_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (OUT/'stability-check.json').write_text(json.dumps(report, indent=2)+'\n')
    make_plot(report)
    print(json.dumps({'cases': len(cases), 'all_screen_passed': all(x['screen_passed'] for x in cases),
                      'elapsed_seconds': report['elapsed_seconds'], 'peak_process_rss_MiB': report['peak_process_rss_MiB']}, indent=2))


if __name__ == '__main__':
    main()
