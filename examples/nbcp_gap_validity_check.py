"""Check the validity conditions of the NBCP pseudo-Goldstone gap relation.

The note uses Delta^2 = C_phi / chi_z at T = 0 and leading order in 1/S
(Appendix A). This script checks three things that relation assumes.

1. Coordinate dependence and the tadpole. The one-loop energy for uniform
   (k = 0) orientations of the three-sublattice cell is
   lam^2 E_cl(x) + lam E_1(x), with E_1 the spin-wave zero-point energy at
   background x (linear terms dropped). Along the exact global-z orbit its
   curvature is C_phi. Along the straight tangent line in linear local
   coordinates (the parametrization of Rau et al., Eq. 7) the background
   leaves the classical manifold at second order, so the curvature picks
   up grad E_1 times the polar bending of the orbit. The vacuum shift
   -H^+ grad E_1 / lam (the tadpole of Lin and Shi, arXiv:2505.07229)
   combined with the classical third derivative along the line must
   cancel that term at order lam. Second derivatives of E_1 off the orbit
   are not used: they grow with the k mesh (recorded), because the
   gapless branch near Gamma turns soft off the classical manifold.
2. Mode separation and stability. Hard k = 0 frequencies from the 6x6
   classical dynamics are compared with the k = 0 spin-wave spectrum, the
   number of classical zero modes is counted, the 6x6 dynamics with the
   leading-order pinning is solved without the single-mode reduction, and
   the gap is compared with the hard modes and with the lowest mode away
   from Gamma. Near the coupling where the background fails, the failing
   orbit angle and wavevector are recorded.
3. Finite temperature. The harmonic free energy
   F = E_cl + E_1 + T sum ln(1 - exp(-omega/T)) per spin is computed along
   the exact orbit. Its curvature at the selected angle gives
   Delta(T) = sqrt(C_phi(T) / chi_z). Thermal corrections to chi_z are
   subleading in 1/S and are not included. A second column removes the
   lowest branch for |k| < 0.25 from the thermal sum, since those modes are
   the phase field that the clock RG treats separately.

Run from the repository root:
    python examples/nbcp_gap_validity_check.py
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

from examples.pseudo_goldstone_comparison import (METRIC, S, angular_fit, build,
                                                  classical_from_bonds, mesh,
                                                  phase_state, rotation_z,
                                                  spin_angles, vacuum_energy)

OUT = ROOT / 'data-space/verification/261001-gap-validity'
N_UNIFORM = 24
N_THERMAL = 48
N_PHI = 72
LAMBDAS = [1., 4., 16., 64., 256.]
TEMPERATURES = [0., 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]
STEP = 2e-3
STEP_CLASSICAL = 1e-4
STEP_QUANTUM = 1e-5
CASES = [('Y', 0.010, 0.0), ('V', 0.010, 0.0), ('Y', 0.005, 0.0),
         ('V', 0.005, 0.0), ('V', 0.0, 0.010)]
THERMAL_CASES = [('Y', 0.010, 0.0), ('V', 0.010, 0.0), ('Y', 0.005, 0.0),
                 ('V', 0.005, 0.0), ('V', 0.0, 0.010), ('V', 0.010, 0.005)]
SEPARATION_CASES = [('Y', pd, 0.0) for pd in (0.005, 0.010, 0.0125, 0.014)] + \
                   [('V', pd, 0.0) for pd in (0.005, 0.010, 0.0115)] + \
                   [('Y', 0.0, 0.010), ('V', 0.0, 0.010)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Uniform:
    """Uniform three-sublattice orientations in gnomonic tangent coordinates."""

    def __init__(self, state, pd, gamma, n):
        self.data, self.ham, _ = build(state, pd, gamma)
        self.points = mesh(n)
        self.normals, self.e1, self.e2 = [], [], []
        for t in state['theta']:
            self.normals.append(np.array([np.sin(t), 0., np.cos(t)]))
            self.e1.append(np.array([np.cos(t), 0., -np.sin(t)]))
            self.e2.append(np.array([0., 1., 0.]))

    def directions(self, x):
        u, v = x[:3], x[3:]
        out = []
        for a in range(3):
            p = self.normals[a] + u[a]*self.e1[a] + v[a]*self.e2[a]
            out.append(p/np.linalg.norm(p))
        return out

    @staticmethod
    def angles(directions):
        return np.ravel([[np.arccos(np.clip(m[2], -1, 1)), np.arctan2(m[1], m[0])]
                         for m in directions])

    def classical(self, x):
        return 3*classical_from_bonds(self.data, self.angles(self.directions(x)))

    def zero_point(self, x):
        mats, _ = self.ham.Quadratic_Bose_Hamiltonian(
            self.points, angles=self.angles(self.directions(x)))
        return 3*vacuum_energy(mats)[0]

    def berry(self, x):
        """Spin length times the gnomonic area factor, per sublattice."""
        u, v = x[:3], x[3:]
        return S*(1 + u*u + v*v)**-1.5


def hessian(f, x, h=STEP):
    n = len(x)
    f0 = f(x)
    grad = np.zeros(n)
    hess = np.zeros((n, n))
    for i in range(n):
        ei = np.eye(n)[i]*h
        fp, fm = f(x+ei), f(x-ei)
        grad[i] = (fp-fm)/(2*h)
        hess[i, i] = (fp-2*f0+fm)/h**2
        for j in range(i):
            ej = np.eye(n)[j]*h
            hess[i, j] = hess[j, i] = (f(x+ei+ej)-f(x+ei-ej)-f(x-ei+ej)+f(x-ei-ej))/(4*h*h)
    return f0, grad, hess


def frequencies(hess, berry):
    """Berry-form dynamics in (u_a, v_a) pairs; returns sorted positive omega."""
    form = np.zeros((6, 6))
    for a in range(3):
        form[a, 3+a] = berry[a]
        form[3+a, a] = -berry[a]
    eig = np.linalg.eigvals(np.linalg.solve(form, hess))
    return np.sort(np.abs(eig.imag))[::2], float(np.max(abs(eig.real)))


def orbit_curvature(uniform, scale=1.):
    """Second derivative of E_1 per spin along the exact global-z orbit."""
    base = uniform.directions(np.zeros(6))

    def energy(phi):
        dirs = [rotation_z(phi) @ m for m in base]
        mats, _ = uniform.ham.Quadratic_Bose_Hamiltonian(uniform.points, angles=Uniform.angles(dirs))
        return vacuum_energy(mats)[0]

    h = 0.02
    values = [energy(k*h) for k in (-2, -1, 0, 1, 2)]
    return scale*(-values[0]+16*values[1]-30*values[2]+16*values[3]-values[4])/(12*h*h)


def gradient(f, x, h):
    return np.array([(f(x+e*h)-f(x-e*h))/(2*h) for e in np.eye(len(x))])


def uniform_check(phase, pd, gamma):
    state = phase_state(phase)
    uni = Uniform(state, pd, gamma, N_UNIFORM)
    phi0 = selected_angle(state, pd, gamma)
    rot = rotation_z(phi0)
    uni.normals = [rot @ m for m in uni.normals]
    uni.e1 = [rot @ m for m in uni.e1]
    uni.e2 = [rot @ m for m in uni.e2]
    chi = state['chi_per_spin_per_meV']
    c_phi = orbit_curvature(uni)
    prediction = np.sqrt(c_phi/chi)
    zero = np.zeros(6)
    # Classical 6x6 dynamics must reproduce the k = 0 spin-wave spectrum.
    _, gcl, hcl = hessian(uni.classical, zero, STEP_CLASSICAL)
    classical_freq, _ = frequencies(hcl, uni.berry(zero))
    mats, _ = uni.ham.Quadratic_Bose_Hamiltonian(np.zeros((1, 2)), angles=uni.angles(uni.directions(zero)))
    lswt = np.sort(np.abs(np.linalg.eigvals(METRIC @ mats[0]).real))[::2]
    # Orbit tangent and bending in gnomonic coordinates (signed, per sublattice).
    zhat = np.array([0, 0, 1.])
    weight = np.array([np.cross(zhat, n) @ e2 for n, e2 in zip(uni.normals, uni.e2)])
    sin_t = np.array([-e1[2] for e1 in uni.e1])
    cos_t = np.array([n[2] for n in uni.normals])
    tangent = np.concatenate([np.zeros(3), weight])
    # Only first derivatives of E_1 off the orbit are used: its second
    # derivatives in hard directions are infrared sensitive (see below).
    g1 = gradient(uni.zero_point, zero, STEP_QUANTUM)
    s = 1e-3
    line = [uni.zero_point(k*s*tangent) for k in (-2, -1, 0, 1, 2)]
    c_line = (-line[0]+16*line[1]-30*line[2]+16*line[3]-line[4])/(12*s*s)/3
    # Polar bending of the orbit away from the straight tangent line:
    # line - orbit = (phi^2/2) sin(theta) cos(theta) e1 per sublattice.
    bend = float(np.sum(g1[:3]*sin_t*cos_t)/3)
    # Tadpole: the zero-point gradient shifts the hard coordinates by
    # -H^+ g / lam; with the classical third derivative along the line this
    # changes the soft curvature by -(d_s^2 grad E_cl) . H^+ g at order lam.
    q = 1e-2
    third = (gradient(uni.classical, q*tangent, STEP_CLASSICAL)
             - 2*gradient(uni.classical, zero, STEP_CLASSICAL)
             + gradient(uni.classical, -q*tangent, STEP_CLASSICAL))/(q*q)
    w, v = np.linalg.eigh(hcl)
    keep = w > 1e-3*w.max()
    pinv = (v[:, keep]/w[keep]) @ v[:, keep].T
    tadpole = float(-(third @ pinv @ g1)/3)
    # Off-orbit second derivative of E_1 in a hard direction versus mesh.
    infrared = []
    for n in (12, 24, 48):
        probe = Uniform(state, pd, gamma, n)
        probe.normals, probe.e1, probe.e2 = uni.normals, uni.e1, uni.e2
        e = np.eye(6)[0]
        h = 1e-6
        vals = [probe.zero_point(k*h*e) for k in (-1, 0, 1)]
        infrared.append({'N': n, 'd2E1_du_A2_per_cell': (vals[0]-2*vals[1]+vals[2])/h**2})
    # Single-mode reduction: full 6x6 classical dynamics plus the
    # leading-order pinning C_phi along the orbit, versus sqrt(C_phi/chi).
    pin = 3*c_phi*np.outer(tangent, tangent)/(tangent @ tangent)**2
    pinned, _ = frequencies(hcl + pin, uni.berry(zero))
    record = {'phase': phase, 'JPD_meV': pd, 'JGamma_meV': gamma, 'phi_selected': phi0,
              'N': N_UNIFORM, 'chi_per_spin_per_meV': chi,
              'six_mode_pinned_frequencies_meV': pinned.tolist(),
              'six_mode_over_prediction': float(pinned[0]/prediction),
              'C_phi_orbit_meV_per_spin': c_phi, 'prediction_meV': float(prediction),
              'C_straight_line_meV_per_spin': float(c_line),
              'orbit_bending_term_meV_per_spin': bend,
              'line_minus_orbit_minus_bending_meV_per_spin': float(c_line - c_phi - bend),
              'tadpole_term_meV_per_spin': tadpole,
              'line_plus_tadpole_over_orbit': float((c_line + tadpole)/c_phi),
              'line_over_orbit': float(c_line/c_phi),
              'classical_zero_modes': int(np.sum(classical_freq < 1e-3*classical_freq.max())),
              'classical_uniform_frequencies_meV': classical_freq.tolist(),
              'lswt_gamma_frequencies_meV': lswt.tolist(),
              'max_gamma_frequency_mismatch_meV': float(np.max(abs(classical_freq[1:]-lswt[1:]))),
              'zero_point_gradient_per_cell': g1.tolist(),
              'classical_gradient_per_cell': gcl.tolist(),
              'offorbit_hessian_infrared': infrared}
    print(f"{phase} PD={pd} G={gamma}: C={c_phi:.4e} line/orbit={c_line/c_phi:.4f} "
          f"bend-check={c_line-c_phi-bend:.2e} (line+tadpole)/orbit={(c_line+tadpole)/c_phi:.4f} "
          f"gamma-mismatch={record['max_gamma_frequency_mismatch_meV']:.1e} "
          f"6x6/pred={record['six_mode_over_prediction']:.4f} "
          f"IR={[round(r['d2E1_du_A2_per_cell'], 4) for r in infrared]}", flush=True)
    return record


def selected_angle(state, pd, gamma):
    if pd == gamma == 0:
        return 0.
    try:
        return _selected_angle(state, pd, gamma)
    except (np.linalg.LinAlgError, AssertionError):
        # Unstable background: no zero-point energy exists. Report the
        # sixfold PD minimum phi = 0 for the stability diagnostics only.
        return 0.


def _selected_angle(state, pd, gamma):
    data, ham, _ = build(state, pd, gamma)
    points = mesh(12)
    phi = np.arange(N_PHI)*2*np.pi/N_PHI
    energy = []
    for p in phi:
        mats, _ = ham.Quadratic_Bose_Hamiltonian(points, angles=spin_angles(state['theta'], p))
        energy.append(vacuum_energy(mats)[0])
    return angular_fit(phi, np.array(energy))['phi_min']


def bands(ham, theta, phi, points):
    mats, _ = ham.Quadratic_Bose_Hamiltonian(points, angles=spin_angles(theta, phi))
    eig = np.linalg.eigvals(METRIC[None] @ mats)
    return mats, np.sort(eig.real, axis=1)[:, 3:], float(np.max(abs(eig.imag)))


def separation_check(phase, pd, gamma):
    state = phase_state(phase)
    data, ham, _ = build(state, pd, gamma)
    points = mesh(N_THERMAL)
    phi0 = selected_angle(state, pd, gamma)
    mats, omega, imag = bands(ham, state['theta'], phi0, points)
    kabs = np.linalg.norm(points, axis=1)
    lowest = omega[:, 0]
    unstable = bool(np.linalg.eigvalsh(mats).min() < 0)
    away = kabs > 0.5
    k_low = points[away][np.argmin(lowest[away])]
    record = {'phase': phase, 'JPD_meV': pd, 'JGamma_meV': gamma, 'phi_selected': phi0,
              'unstable_background': unstable,
              'min_matrix_eigenvalue': float(np.linalg.eigvalsh(mats).min()),
              'max_imaginary_frequency_meV': imag,
              'lowest_mode_away_from_gamma_meV': float(lowest[away].min()),
              'k_of_lowest_away_from_gamma': k_low.tolist(),
              'second_band_min_meV': float(omega[:, 1].min())}
    if not unstable:
        uni = Uniform(state, pd, gamma, 24)
        rot = rotation_z(phi0)
        uni.normals = [rot @ m for m in uni.normals]
        uni.e1 = [rot @ m for m in uni.e1]
        uni.e2 = [rot @ m for m in uni.e2]
        c_phi = orbit_curvature(uni)
        gap = float(np.sqrt(max(c_phi, 0)/state['chi_per_spin_per_meV']))
        gmat, _ = ham.Quadratic_Bose_Hamiltonian(np.zeros((1, 2)), angles=spin_angles(state['theta'], phi0))
        gamma_modes = np.sort(np.abs(np.linalg.eigvals(METRIC @ gmat[0]).real))[::2]
        record.update({'gap_meV': gap,
                       'gamma_hard_modes_meV': gamma_modes[1:].tolist(),
                       'gap_over_lowest_hard_gamma_mode': gap/gamma_modes[1],
                       'gap_over_lowest_mode_away_from_gamma': gap/float(lowest[away].min())})
    print(json.dumps({k: record[k] for k in record if k not in ('gamma_hard_modes_meV',)}), flush=True)
    return record


def threshold_probe(phase, pd, gamma, nphi=36):
    """Where along the orbit and at which k the background first fails."""
    state = phase_state(phase)
    data, ham, _ = build(state, pd, gamma)
    points = mesh(N_THERMAL)
    worst = None
    for p in np.arange(nphi)*2*np.pi/nphi:
        mats, _ = ham.Quadratic_Bose_Hamiltonian(points, angles=spin_angles(state['theta'], p))
        ev = np.linalg.eigvalsh(mats).min(axis=1)
        i = int(np.argmin(ev))
        if worst is None or ev[i] < worst['min_matrix_eigenvalue']:
            worst = {'phi': float(p), 'min_matrix_eigenvalue': float(ev[i]),
                     'k': points[i].tolist(), 'abs_k': float(np.linalg.norm(points[i]))}
    record = {'phase': phase, 'JPD_meV': pd, 'JGamma_meV': gamma, 'worst_on_orbit': worst}
    print(json.dumps(record), flush=True)
    return record


def thermal_check(phase, pd, gamma, cutoff=0.25):
    """Harmonic free-energy curvature; also with soft-branch modes |k| < cutoff
    removed from the thermal sum (they belong to the phase field theory)."""
    state = phase_state(phase)
    data, ham, _ = build(state, pd, gamma)
    points = mesh(N_THERMAL)
    inner = np.linalg.norm(points, axis=1) < cutoff
    phi = np.arange(N_PHI)*2*np.pi/N_PHI
    free = {(t, c): [] for t in TEMPERATURES for c in (False, True)}
    min_omega = np.inf
    for p in phi:
        mats, _ = ham.Quadratic_Bose_Hamiltonian(points, angles=spin_angles(state['theta'], p))
        e0, wmin = vacuum_energy(mats)
        chol = np.linalg.cholesky(mats)
        omega = np.linalg.eigvalsh(chol.conj().transpose(0, 2, 1) @ METRIC @ chol)[:, 3:]
        min_omega = min(min_omega, wmin)
        for t in TEMPERATURES:
            if t == 0:
                terms = np.zeros_like(omega)
            else:
                terms = t*np.log1p(-np.exp(-omega/t))
            full = terms.sum(axis=1)
            cut = full.copy()
            cut[inner] -= terms[inner, 0]
            free[(t, False)].append(e0 + float(np.mean(full))/3)
            free[(t, True)].append(e0 + float(np.mean(cut))/3)
    chi = state['chi_per_spin_per_meV']
    rows = []
    for t in TEMPERATURES:
        row = {'T_meV': t}
        for c, label in ((False, 'all_modes'), (True, f'soft_branch_cut_{cutoff}')):
            fit = angular_fit(phi, np.array(free[(t, c)]))
            curv = fit['curvature_meV_per_spin']
            row[label] = {'phi_min': fit['phi_min'], 'curvature_meV_per_spin': curv,
                          'status': fit['status'],
                          'gap_meV': float(np.sqrt(curv/chi)) if curv > 0 else None}
        rows.append(row)
    for label in ('all_modes', f'soft_branch_cut_{cutoff}'):
        g0 = rows[0][label]['gap_meV']
        for r in rows:
            g = r[label]['gap_meV']
            r[label]['gap_over_T0'] = g/g0 if g and g0 else None
    print(f"{phase} PD={pd} G={gamma}: " + ' '.join(
        f"T={r['T_meV']}:{r['all_modes']['gap_over_T0']:.3f}/{r[f'soft_branch_cut_{cutoff}']['gap_over_T0']:.3f}"
        for r in rows), flush=True)
    return {'phase': phase, 'JPD_meV': pd, 'JGamma_meV': gamma, 'N': N_THERMAL, 'n_phi': N_PHI,
            'soft_branch_cutoff': cutoff, 'modes_below_cutoff_fraction': float(np.mean(inner)),
            'min_sampled_frequency_meV': float(min_omega), 'rows': rows}


def main():
    start = time.monotonic()
    record = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': __doc__, 'S': S, 'finite_difference_step_rad': STEP,
              'uniform': [uniform_check(*c) for c in CASES]}
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / 'gap-validity-check.json'
    out.write_text(json.dumps(record, indent=1))
    record['separation'] = [separation_check(*c) for c in SEPARATION_CASES]
    out.write_text(json.dumps(record, indent=1))
    record['thresholds'] = [threshold_probe(*c) for c in
                            [('Y', 0.014, 0.0), ('Y', 0.015, 0.0), ('V', 0.0115, 0.0), ('V', 0.0125, 0.0)]]
    out.write_text(json.dumps(record, indent=1))
    record['thermal'] = [thermal_check(*c) for c in THERMAL_CASES]
    record['seconds'] = time.monotonic()-start
    record['inputs_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in
                               [Path(__file__), ROOT / 'examples/pseudo_goldstone_comparison.py',
                                ROOT / 'model/nbcp/exchange.py']}
    out.write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
