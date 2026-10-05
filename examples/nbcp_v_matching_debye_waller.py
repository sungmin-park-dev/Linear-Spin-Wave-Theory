"""V-phase matching inputs and the Debye-Waller factor of the Y and V sixfold pinning.

Continues `nbcp_matching_cutoff_check.py` (Y) at J_PD = 0.010 meV, J_Gamma = 0.

1. V stiffness. The relaxed reduction of `nbcp_y_soc_conditions.reduction` is
   written for any coplanar three-sublattice state whose soft coordinate is
   the common laboratory-z rotation g = (0, 0, 0, sin theta_a): the gauge keeps
   the polar components and the azimuthal components orthogonal to g. It
   reproduces the Y reduction exactly and gives rho(phi), chi(phi) for V at
   1.4 T; chi is checked against `pseudo_goldstone_comparison.phase_state`.
2. V matching. As for Y, the phase theory eps = k sqrt(khat.rho(phi).khat/chi)
   is compared with the LSWT soft branch: the sixfold coefficient of
   <ln eps_soft> at k -> 0, and the sixfold part of the Bose free energy of the
   modes below Lambda (LSWT recomputed on a 96 x 96 mesh). The O(T) ratio
   a6(T)/a6(0) with all modes kept is read from the angular split.
3. Debye-Waller factor. Phase fluctuations above the matching cutoff multiply
   the sixfold coefficient by exp(-D) with D = 18 <phi^2>_{k > Lambda}. Per
   mode the phase variance is (eps / 2 kappa) coth(eps / 2T), where kappa(k)
   is the relaxed static phase stiffness (Schur complement of the static
   kernel) and eps the LSWT soft branch. D is split into its thermal part
   (eps / kappa) n_B(eps), reported against the classical estimate T / kappa,
   and its zero-point part eps / (2 kappa). Meshes of 96 and 144 points per side.

Leading order in 1/S throughout; no anharmonic terms, vortices or quantum
angular potential beyond harmonic order.

Run from the repository root:
    python examples/nbcp_v_matching_debye_waller.py
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT), str(ROOT / 'legacy')]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lswt-mpl-cache')

import numpy as np
from scipy.linalg import null_space

from examples import nbcp_y_soc_conditions as soc
from examples.nbcp_y_stiffness import AREA, J, JZ
from examples.pseudo_goldstone_comparison import GZ, METRIC, MU_B, build, mesh, phase_state, spin_angles
from model.nbcp import make_nn_exchange_matrices

SPLIT = ROOT / 'data-space/verification/261001-angular-thermal-split/angular-thermal-split.json'
OUT = ROOT / 'data-space/verification/261005-v-matching-debye-waller'
JPD = 0.010
N_PHI = 72
OMEGA = np.linalg.inv(soc.POISSON)
TEMPERATURES = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004]
CUTOFFS = [0.1, 0.15, 0.25]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cos6(phi, values):
    centered = np.asarray(values) - np.mean(values)
    return float(2 / len(phi) * np.sum(centered * np.cos(6 * phi))), \
        float(2 / len(phi) * np.sum(centered * np.sin(6 * phi)))


def v_background(pd, phi, theta, field=1.4):
    """Background tuple in the layout of `nbcp_y_soc_conditions.background` for a coplanar state."""
    h = GZ * MU_B * field
    st, ct = np.sin(theta), np.cos(theta)
    cp, sp = np.cos(phi), np.sin(phi)
    normals = np.column_stack([st * cp, st * sp, ct])
    polar = np.column_stack([ct * cp, ct * sp, -st])
    azimuth = np.tile([-sp, cp, 0.], (3, 1))
    exchanges = make_nn_exchange_matrices(dict(Jxy=J, Jz=JZ, JPD=pd, JGamma=0., h=(0., 0., h)))
    return h, None, normals, np.stack([polar, azimuth], axis=-1), exchanges


def reduce(bg, theta):
    """Relaxed stiffness C (per magnetic cell) and susceptibility chi of the common z rotation."""
    g = np.concatenate([np.zeros(3), np.sin(theta)])
    gauge = np.zeros((6, 5))
    gauge[:3, :3] = np.eye(3)
    gauge[3:, 3:] = null_space(g[None, 3:])
    K0 = soc.kernel(np.zeros(2), bg)
    H = gauge.T @ K0 @ gauge
    L = np.column_stack([-1j * gauge.T @ soc.kernel(np.zeros(2), bg, (i,)) @ g for i in range(2)])
    assert np.max(abs(L.imag)) < 1e-12
    L = L.real
    D = np.array([[np.real(g @ soc.kernel(np.zeros(2), bg, (i, j)) @ g) / 2 for j in range(2)] for i in range(2)])
    C = D - np.real(L.T @ np.linalg.solve(H, L))
    b = gauge.T @ OMEGA @ g
    return {'g': g, 'gauge': gauge, 'C': C, 'chi': float(np.real(b @ np.linalg.solve(H, b))),
            'zero_mode_residual': float(np.max(abs(K0 @ g))), 'min_eig_H': float(np.linalg.eigvalsh(H).min())}


def static_stiffness(q, bg, g, gauge):
    K = soc.kernel(q, bg)
    z = -np.linalg.solve(gauge.T @ K @ gauge, gauge.T @ K @ g)
    v = g + gauge @ z
    return float(np.real(v.conj() @ K @ v))


def lswt_bands(state, points, phi):
    _, current, _ = build(state, JPD, 0.0)
    mats, torque = current.Quadratic_Bose_Hamiltonian(points, angles=spin_angles(state['theta'], phi))
    assert max(abs(v) for v in torque.values()) < 2e-12
    chol = np.linalg.cholesky(mats)
    eig = np.linalg.eigvalsh(chol.conj().transpose(0, 2, 1) @ METRIC @ chol)
    return eig[:, 3:]


def debye_waller(phase, points):
    state = phase_state(phase)
    theta = np.asarray(state['theta'])
    bg = soc.background(JPD, 0.0, 0.0) if phase == 'Y' else v_background(JPD, 0.0, theta)
    red = reduce(bg, theta)
    kappa = np.array([static_stiffness(q, bg, red['g'], red['gauge']) for q in points])
    eps = lswt_bands(state, points, 0.0)[:, 0]
    kabs = np.linalg.norm(points, axis=1)
    small = kabs < 0.1
    rho = red['C'] / (3 * AREA)
    sqrt_det = float(np.sqrt(np.linalg.det(rho)))
    checks = {'kappa_over_qCq_small_k': [float(x) for x in
                                         (lambda r: (r.min(), r.max()))(kappa[small] / np.einsum('ki,ij,kj->k', points[small], red['C'], points[small]))],
              'eps_over_phase_theory_small_k': [float(x) for x in
                                                (lambda r: (r.min(), r.max()))(eps[small] / np.sqrt(kappa[small] / red['chi']))]}
    zero_point = eps / (2 * kappa)
    out = {'phase': phase, 'sqrt_det_rho_meV': sqrt_det, 'checks': checks,
           'D_zero_point_all_k': 18 * float(zero_point.mean()),
           'D_zero_point_below': {str(c): 18 * float(np.sum(zero_point[kabs < c]) / len(kabs)) for c in CUTOFFS},
           'window_meV': {'bare_BKT_edge_pi_rho_over_2': np.pi * sqrt_det / 2,
                          'sixfold_relevance_K_9_over_2pi': 2 * np.pi * sqrt_det / 9},
           'temperatures': []}
    for T in TEMPERATURES:
        thermal = eps / kappa / np.expm1(eps / T)
        classical = T / kappa
        row = {'T_meV': T, 'K': sqrt_det / T, 'sixfold_eigenvalue_2_minus_9_over_piK': 2 - 9 * T / (np.pi * sqrt_det)}
        for c in [0.0] + CUTOFFS:
            above = kabs > c
            row[f'D_thermal_above_{c}'] = 18 * float(np.sum(thermal[above]) / len(kabs))
            row[f'D_classical_above_{c}'] = 18 * float(np.sum(classical[above]) / len(kabs))
        out['temperatures'].append(row)
    return out


def v_matching():
    state = phase_state('V')
    theta = np.asarray(state['theta'])
    phi = 2 * np.pi * np.arange(N_PHI) / N_PHI
    reduced = [reduce(v_background(JPD, p, theta), theta) for p in phi]
    C = np.array([r['C'] for r in reduced])
    chi = np.array([r['chi'] for r in reduced])
    khat = np.column_stack([np.cos(np.linspace(0, np.pi, 360, endpoint=False)),
                            np.sin(np.linspace(0, np.pi, 360, endpoint=False))])
    b6, b6_sin = cos6(phi, [0.5 * np.mean(np.log(np.einsum('ki,ij,kj->k', khat, c, khat) / x)) for c, x in zip(C, chi)])
    sqrt_det = np.sqrt(np.linalg.det(C / (3 * AREA)))
    zero_jpd = reduce(v_background(0.0, 0.0, theta), theta)

    split = json.loads(SPLIT.read_text())
    v = next(r for r in split['results'] if r['phase'] == 'V')
    shells = [s for s in v['shells'] if s['k_max'] <= 0.25]
    k_rms = [np.sqrt((s['k_max']**4 - s['k_min']**4) / (2 * (s['k_max']**2 - s['k_min']**2))) for s in shells]
    b0, slope = np.linalg.solve(np.column_stack([np.ones(2), -np.square(k_rms)]), [s['b6_ln_eps_soft'] for s in shells])
    a60 = v['temperatures'][0]['a6_cutoff_0.0']
    ratios = [{'T_meV': r['T_meV'], 'a6_ratio_all_modes': r['a6_cutoff_0.0'] / a60}
              for r in v['temperatures'] if r['T_meV'] > 0]

    # Cutoff split on a finer mesh, 36 angles: LSWT (all bands) against the phase theory.
    phi36 = phi[::2]
    points = mesh(96)
    kabs = np.linalg.norm(points, axis=1)
    eps = np.array([lswt_bands(state, points, p) for p in phi36])
    split_rows = []
    for T in [0.0025, 0.005, 0.01]:
        per_mode = T * np.log1p(-np.exp(-eps / T))
        row = {'T_meV': T}
        for cut in CUTOFFS:
            keep = kabs < cut
            lswt = cos6(phi36, per_mode[:, keep].sum(-1).sum(1) / len(kabs) / 3)[0]
            theory = []
            for c, x in zip(C[::2], chi[::2]):
                e = np.sqrt(np.einsum('ki,ij,kj->k', points[keep], c, points[keep]) / x)
                theory.append(T * np.sum(np.log1p(-np.exp(-e / T))) / len(kabs) / 3)
            row[f'removed_cos6_lswt_{cut}'] = lswt
            row[f'removed_cos6_phase_theory_{cut}'] = cos6(phi36, theory)[0]
            row[f'ratio_{cut}'] = cos6(phi36, theory)[0] / lswt
        split_rows.append(row)
    return {'theta': theta.tolist(), 'chi_cell': float(chi.mean()), 'chi_span': float(np.ptp(chi)),
            'chi_phase_state_per_spin': state['chi_per_spin_per_meV'],
            'zero_mode_residual': max(r['zero_mode_residual'] for r in reduced),
            'min_eig_H': min(r['min_eig_H'] for r in reduced),
            'sqrt_det_rho_meV': {'mean': float(sqrt_det.mean()), 'cos6_relative': cos6(phi, sqrt_det)[0] / float(sqrt_det.mean()),
                                 'min': float(sqrt_det.min()), 'max': float(sqrt_det.max())},
            'sqrt_det_rho_JPD0_meV': float(np.sqrt(np.linalg.det(zero_jpd['C'] / (3 * AREA)))),
            'rho_phi0_meV': (C[0] / (3 * AREA)).tolist(),
            'b6_phase_theory': b6, 'b6_sin_residual': b6_sin, 'b6_lswt_k_to_0': float(b0), 'b6_lswt_k2_slope': float(slope),
            'a6_zero_meV_per_spin': a60, 'a6_ratios': ratios, 'cutoff_split': split_rows}


def main():
    start = time.monotonic()
    # The general reduction reproduces the Y-specific one.
    bg = soc.background(JPD, 0.0, 0.3)
    ref = soc.reduction(bg)
    general = reduce(bg, np.array([np.arccos(bg[1]), -np.arccos(bg[1]), np.pi]))
    y_check = {'max_C_difference': float(np.abs(general['C'] - ref[4]).max()), 'chi_difference': general['chi'] - ref[5]}

    v = v_matching()
    # The sharp cutoff makes D above 0.1 move by about 15 % between meshes; the spread of
    # the 96 and 144 meshes is kept as its uncertainty.
    coarse = [debye_waller(phase, mesh(96)) for phase in ('Y', 'V')]
    dw = [debye_waller(phase, mesh(144)) for phase in ('Y', 'V')]
    for d, c in zip(dw, coarse):
        for row, other in zip(d['temperatures'], c['temperatures']):
            row['mesh_spread'] = {str(cut): abs(row[f'D_thermal_above_{cut}'] - other[f'D_thermal_above_{cut}'])
                                  for cut in CUTOFFS}

    print('Y check', y_check)
    print('V: chi/3 %.6f (phase_state %.6f), sqrt det rho %.6f meV (J_PD=0: %.6f), cos6 %.3f, b6 theory %.5f LSWT %.5f'
          % (v['chi_cell'] / 3, v['chi_phase_state_per_spin'], v['sqrt_det_rho_meV']['mean'], v['sqrt_det_rho_JPD0_meV'],
             v['sqrt_det_rho_meV']['cos6_relative'], v['b6_phase_theory'], v['b6_lswt_k_to_0']))
    for r in v['cutoff_split']:
        print('  T=%.4f ' % r['T_meV'] + ' '.join('L%.2f ratio %.3f' % (c, r[f'ratio_{c}']) for c in CUTOFFS))
    print('  a6(T)/a6(0):', [(r['T_meV'], round(r['a6_ratio_all_modes'], 3)) for r in v['a6_ratios']])
    for d in dw:
        print('%s sqrt det %.5f window %s D0 %.1f checks %s' % (d['phase'], d['sqrt_det_rho_meV'], d['window_meV'],
                                                               d['D_zero_point_all_k'], d['checks']))
        for r in d['temperatures']:
            print('  T=%.4f K=%.2f thermal D>0.1 %.3f(%.3f) D>0.15 %.3f D>0.25 %.3f | classical D>0.25 %.2f' % (
                r['T_meV'], r['K'], r['D_thermal_above_0.1'], r['mesh_spread']['0.1'], r['D_thermal_above_0.15'],
                r['D_thermal_above_0.25'], r['D_classical_above_0.25']))

    record = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Y (0.2 T) and V (1.4 T), S = 1/2, J = 0.075 meV, J_z = 0.125 meV, J_PD = 0.010 meV, J_Gamma = 0. '
                 'Leading order in 1/S; harmonic thermal fluctuations. T in meV, momenta in inverse bond lengths, '
                 'stiffness per area in meV, a6 per spin with f = f0 - a6 cos(6 phi).',
        'debye_waller_definition': 'D = 18 <phi^2>_{|k| > Lambda}; per mode <|phi_k|^2> = (eps/2 kappa) coth(eps/2T), '
                                   'thermal part (eps/kappa) n_B(eps), zero-point part eps/(2 kappa), classical T/kappa; '
                                   'kappa = relaxed static phase stiffness per magnetic cell, eps = LSWT soft branch.',
        'y_reduction_check': y_check, 'v_matching': v, 'debye_waller': dw,
        'inputs_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
                          [Path(__file__), SPLIT, ROOT / 'examples/nbcp_y_soc_conditions.py',
                           ROOT / 'examples/pseudo_goldstone_comparison.py']},
        'seconds': time.monotonic() - start,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'v-matching-debye-waller.json').write_text(json.dumps(record, indent=1))

    assert y_check['max_C_difference'] < 1e-14 and abs(y_check['chi_difference']) < 1e-12
    assert abs(v['chi_cell'] / 3 / v['chi_phase_state_per_spin'] - 1) < 1e-8 and v['chi_span'] < 1e-12
    assert v['zero_mode_residual'] < 1e-12 and v['min_eig_H'] > 0
    assert abs(v['b6_phase_theory'] / v['b6_lswt_k_to_0'] - 1) < 0.03
    assert all(abs(r['ratio_0.1'] - 1) < 0.06 for r in v['cutoff_split'])        # V phase regime: Lambda ~ 0.1
    assert all(r['ratio_0.25'] > 1.15 for r in v['cutoff_split'])                # and not 0.25
    for d in dw:
        assert all(abs(x - 1) < 0.03 for x in d['checks']['kappa_over_qCq_small_k'])
        assert all(abs(x - 1) < 0.01 for x in d['checks']['eps_over_phase_theory_small_k'])
        window = [r for r in d['temperatures'] if r['T_meV'] <= 0.003]
        assert all(r['D_thermal_above_0.25'] < 0.25 for r in window)            # small above the matching cutoff
        assert all(r['D_classical_above_0.25'] > 1 for r in window)             # the classical estimate is not
        assert d['D_zero_point_all_k'] > 10


if __name__ == '__main__':
    main()
