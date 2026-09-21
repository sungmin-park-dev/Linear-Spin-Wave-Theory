"""Check the clock-RG normalization and its limited matching to saved Y data.

This checks algebra and reads existing classical results. It does not integrate
an NBCP thermal RG trajectory or estimate transition temperatures.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data-space/verification/260917-clock-rg'


def main():
    K, q, cutoff, dl, p, a0, r = sp.symbols('K q Lambda dl p a0 r', positive=True)
    y, yv, rho_x, rho_y, angle = sp.symbols('y yv rho_x rho_y angle', positive=True)
    shell = sp.integrate(1/(2*sp.pi*K*q), (q, cutoff*sp.exp(-dl), cutoff))
    x_p = p*p/(4*sp.pi*K)
    pinning_step = y*sp.exp(2*dl-p*p*shell/2)
    pinning_flow = sp.diff(pinning_step, dl).subs(dl, 0)
    radial = sp.integrate(r**3*(a0/r)**(2*x_p), (r, a0, a0*sp.exp(dl)))
    shell_second_moment = sp.simplify(sp.diff(radial, dl).subs(dl, 0)*sp.integrate(sp.cos(angle)**2, (angle, 0, 2*sp.pi)))
    delta_K = sp.simplify(2*y*y*p*p/(8*a0**4)*shell_second_moment)
    dual_K = 1/(4*sp.pi**2*K)
    vortex_screen = sp.simplify(4*sp.pi**2*delta_K.subs({p:1, y:2*yv}))
    scale = (rho_x/rho_y)**sp.Rational(1, 4)
    L = sp.diag(scale, 1/scale)
    transformed = sp.simplify(L.inv()*sp.diag(rho_x, rho_y)*L.inv().T)
    eta = 1/(2*sp.pi*K)
    checks = {
        'shell_variance': sp.simplify(shell-dl/(2*sp.pi*K)) == 0,
        'pinning_linear_flow': sp.simplify(pinning_flow-(2-x_p)*y) == 0,
        'dipole_second_moment': sp.simplify(shell_second_moment-sp.pi*a0**4) == 0,
        'pinning_stiffness_feedback': sp.simplify(delta_K-sp.pi*p*p*y*y/4) == 0,
        'dual_vertex_is_vortex': sp.simplify(1/(4*sp.pi*dual_K)-sp.pi*K) == 0,
        'vortex_screening_convention': sp.simplify(vortex_screen-4*sp.pi**3*yv*yv) == 0,
        'area_preserving_map': sp.simplify(L.det()-1) == 0,
        'isotropic_transformed_tensor': transformed == sp.sqrt(rho_x*rho_y)*sp.eye(2),
        'sixfold_pinning_marginality': sp.simplify((2-x_p).subs({p:6,K:9/(2*sp.pi)})) == 0,
        'unit_vortex_marginality': sp.simplify((2-sp.pi*K).subs(K,2/sp.pi)) == 0,
        'sixfold_lower_exponent': eta.subs(K,9/(2*sp.pi)) == sp.Rational(1,9),
        'upper_exponent': eta.subs(K,2/sp.pi) == sp.Rational(1,4),
    }
    assert all(checks.values()), checks
    soc_path = ROOT/'data-space/verification/260917-y-soc-conditions/soc-conditions-check.json'
    soc = json.loads(soc_path.read_text())
    local = []
    for case in soc['cases']:
        rho = np.asarray(case['rho_tensor_meV'])
        assert np.linalg.eigvalsh(rho).min() > 0
        if case['phi0_rad'] == 0:
            local.append({key: case[key] for key in ['JPD_meV','JGamma_meV','phi0_rad','rho_tensor_meV']})
            local[-1]['sqrt_det_rho_meV'] = float(np.sqrt(np.linalg.det(rho)))
    report = {
        'scope':'Clock RG algebra in the declared dipole fugacity convention; local Y matching is not a thermal RG test',
        'sympy_version':sp.__version__, 'checks':checks,
        'fugacity_convention':'Action cosine coefficient y_p; electric charge fugacity y_p/2; vortex fugacity y_v per unit charge; dual cosine coefficient 2*y_v.',
        'local_Y_phi0_examples':local,
        'thermal_RG_trajectory_computed':False,
        'missing_inputs':['Matched thermal stiffness and angle-dependent gradient terms',
                          'Thermal clock potential', 'Vortex core fugacity',
                          'Density-order and full-model stability across the temperature range'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'Y_data_sha256':hashlib.sha256(soc_path.read_bytes()).hexdigest(),
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'clock-rg-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
