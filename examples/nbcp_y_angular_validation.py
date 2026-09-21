"""Cross-check angular matching with fresh momenta and local energy differences.

These are independent numerical routes within the same microscopic model,
not an independent many-body benchmark or human physics acceptance.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code-space'), str(ROOT)]

import numpy as np

from examples.nbcp_y_angular_matching import OUT, energy_at, evaluate
from examples.nbcp_y_soc_conditions import background, reduction
from examples.nbcp_y_stability import COUPLINGS, RECIPROCAL, TO_NAMBU, batch_kernel
from examples.nbcp_y_stiffness import AREA, METRIC
from examples.pseudo_goldstone_comparison import mesh


def main():
    source = OUT/'angular-matching-check.json'
    report = json.loads(source.read_text())
    rng = np.random.default_rng(260918)
    points = rng.uniform(-.4, .4, (37, 2)) @ RECIPROCAL
    alpha = np.pi/3
    rotation = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    covariance = []
    for pd, gamma in COUPLINGS:
        spectra = []
        for q, phi in [(points, .173), (points@rotation.T, .173+alpha)]:
            h = TO_NAMBU.conj().T @ batch_kernel(q, background(pd, gamma, phi)) @ TO_NAMBU
            e = np.linalg.eigvals(METRIC @ h)
            assert np.max(abs(e.imag)) < 1e-11
            spectra.append(np.sort(e.real, axis=1))
        error = float(np.max(abs(spectra[0]-spectra[1])))
        assert error < 1e-12
        covariance.append({'JPD_meV': pd, 'JGamma_meV': gamma,
            'fresh_unsymmetrized_momenta': len(points), 'sixty_degree_dynamic_spectral_error_meV': error})
    tensor_checks = []
    for row in report['stiffness']:
        pd, gamma = row['JPD_meV'], row['JGamma_meV']
        mean = np.array(row['angular_mean_tensor_meV'])
        u = -row['component_cos_meV'][1][0][0]
        v = row['component_cos_meV'][3][0][0]
        for phi in rng.uniform(0, 2*np.pi, 7):
            z = -u*np.exp(2j*phi)+v*np.exp(-4j*phi)
            prediction = mean + np.array([[z.real, z.imag], [z.imag, -z.real]])
            actual = reduction(background(pd, gamma, phi))[4]/(3*AREA)
            error = float(np.max(abs(actual-prediction)))
            assert error < 2e-15
            tensor_checks.append({'JPD_meV': pd, 'JGamma_meV': gamma, 'phi': float(phi),
                'off_grid_tensor_formula_error_meV': error})
    # Fresh finite differences avoid the Fourier differentiation route.
    points = mesh(96)
    curvature = []
    for row in report['potential']:
        pd, gamma = row['JPD_meV'], row['JGamma_meV']
        summary = row['summaries'][-1]
        phi = summary['phi_min_rad_mod_pi_over_three']
        e0 = energy_at(points, pd, gamma, phi)
        checks = []
        for step in [.08, .04, .02]:
            ep, em, epp, emm = [energy_at(points, pd, gamma, phi+offset*step) for offset in [1, -1, 2, -2]]
            derivative = (-epp+16*ep-30*e0+16*em-emm)/(12*step**2)
            target = summary['resolved_curvature_at_min_meV_per_spin']
            # Propagated absolute-energy roundoff proxy, not a rigorous bound.
            floor = 64/(12*step**2)*100*np.finfo(float).eps*abs(e0)
            checks.append({'step_rad': step, 'five_point_curvature_meV_per_spin': float(derivative),
                'relative_difference_from_resolved_fourier': float(abs(derivative/target-1)),
                'absolute_energy_roundoff_diagnostic_meV_per_spin': float(floor)})
        if pd:
            assert checks[-1]['relative_difference_from_resolved_fourier'] < 1e-5
        curvature.append({'JPD_meV': pd, 'JGamma_meV': gamma,
            'resolved_fourier_curvature_meV_per_spin': summary['resolved_curvature_at_min_meV_per_spin'],
            'status': 'Finite-difference corroboration; tiny pure-Gamma differences remain noise-sensitive.',
            'checks': checks})
        print(f'Curvature checks complete: PD={pd:g}, Gamma={gamma:g}', flush=True)
    phi = np.linspace(0, 2*np.pi, 47)
    a = np.zeros(18); b = np.zeros(18); a[5] = 3; b[11] = .04
    derivative = evaluate(phi, a, b, 2)
    assert np.max(abs(derivative-(-108*np.cos(6*phi)-5.76*np.sin(12*phi)))) < 1e-12
    paths = [source, Path(__file__), ROOT/'examples/nbcp_y_angular_matching.py',
             ROOT/'examples/nbcp_y_soc_conditions.py', ROOT/'examples/nbcp_y_stability.py']
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Same-model checks of full-circle tensor representation, unsymmetrized spectral covariance and local potential curvature.',
        'spectral_covariance': covariance, 'tensor_formula': tensor_checks,
        'five_point_curvature': curvature, 'synthetic_fourier_second_derivative': 'passed',
        'inputs_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'independent-check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'max_spectral_covariance_error_meV': max(x['sixty_degree_dynamic_spectral_error_meV'] for x in covariance),
        'max_tensor_formula_error_meV': max(x['off_grid_tensor_formula_error_meV'] for x in tensor_checks)}, indent=2))


if __name__ == '__main__':
    main()
