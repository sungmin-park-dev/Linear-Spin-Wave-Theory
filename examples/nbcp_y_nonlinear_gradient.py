"""Constrained nonlinear classical test of the Y phase-gradient reduction.

The phase is arg(n_A^+ - n_B^+) in each magnetic cell. All five remaining
local coordinates relax. No vacuum pinning, finite-T model, or defects are
inserted. A regular local chart restricts this diagnostic to the Y branch.
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
sys.path[:0] = [str(ROOT/'code-space'), str(ROOT)]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lswt-mpl-cache')

import numpy as np
from scipy.optimize import minimize

from examples.nbcp_y_soc_conditions import background, kernel, reduction
from examples.nbcp_y_stiffness import AREA, GAUGE, S

OUT = ROOT/'data-space/verification/260918-y-nonlinear-gradient'
PAIRS = [(.005, 0.), (.010, 0.), (.010, .010)]


class ConstrainedWave:
    """Periodic magnetic-cell torus with a prescribed staggered phase."""

    def __init__(self, pd, gamma, length, amplitude, phi0, mode=(1, 1)):
        self.bg = background(pd, gamma, phi0)
        self.pd, self.gamma = pd, gamma
        self.length, self.amplitude, self.phi0 = length, amplitude, phi0
        self.mode = np.array(mode)
        self.lattice = np.asarray(self.bg[7].lattice_vectors)
        self.positions = np.array([self.bg[6]['Spin info'][s]['Position'] for s in ['A', 'B', 'C']])
        self.reciprocal = 2*np.pi*np.linalg.inv(self.lattice).T
        self.q = self.mode @ self.reciprocal / length
        indices = np.stack(np.meshgrid(np.arange(length), np.arange(length), indexing='ij'), axis=-1)
        self.wave = 2*np.pi*np.einsum('...i,i->...', indices, self.mode)/length
        self.phi = phi0 + amplitude*np.cos(self.wave)
        base = background(pd, gamma, 0.)
        c, s = np.cos(self.phi), np.sin(self.phi)
        self.normals = np.empty((length, length, 3, 3))
        self.basis = np.empty((length, length, 3, 3, 2))
        for dest, original in [(self.normals, base[2]), (self.basis[..., 0], base[3][..., 0]),
                               (self.basis[..., 1], base[3][..., 1])]:
            dest[..., 0] = c[..., None]*original[:, 0]-s[..., None]*original[:, 1]
            dest[..., 1] = s[..., None]*original[:, 0]+c[..., None]*original[:, 1]
            dest[..., 2] = original[:, 2]
        self.links = []
        for bond in self.bg[6]['Couplings']:
            i, j = ['A', 'B', 'C'].index(bond['SpinI']), ['A', 'B', 'C'].index(bond['SpinJ'])
            delta = (self.positions[i]-self.positions[j]-bond['Displacement']) @ np.linalg.inv(self.lattice)
            assert np.max(abs(delta-np.rint(delta))) < 1e-12
            shift = tuple(np.rint(delta).astype(int))
            exchange = np.asarray(bond['Exchange Matrix'])
            reference = self.bg[2][i] @ exchange @ self.bg[2][j]
            self.links.append((i, j, shift, exchange, reference))

    def spins(self, flat):
        z = flat.reshape(self.length, self.length, 5)
        tangent = z @ GAUGE.T
        xy = np.stack([tangent[..., :3], tangent[..., 3:]], axis=-1)
        root = np.sqrt(1-np.sum(xy*xy, axis=-1))
        n = self.normals*root[..., None]+np.einsum('...iax,...ix->...ia', self.basis, xy)
        return n, xy, root

    def energy_gradient(self, flat):
        n, xy, root = self.spins(flat)
        # Subtract the uniform background bond by bond to reduce cancellation.
        energy = -self.bg[0]*S*np.sum(n[..., 2]-self.bg[2][:, 2])
        grad = np.zeros_like(n); grad[..., 2] = -self.bg[0]*S
        for i, j, shift, exchange, reference in self.links:
            neighbor = np.roll(n[..., j, :], tuple(-x for x in shift), axis=(0, 1))
            field = neighbor @ exchange.T
            energy += S*S*np.sum(np.sum(n[..., i, :]*field, axis=-1)-reference)
            grad[..., i, :] += S*S*field
            grad[..., j, :] += S*S*np.roll(n[..., i, :] @ exchange, shift, axis=(0, 1))
        radial = np.sum(grad*self.normals, axis=-1)/root
        chart = np.einsum('...ia,...iax->...ix', grad, self.basis)-radial[..., None]*xy
        six = np.concatenate([chart[..., 0], chart[..., 1]], axis=-1)
        return float(energy), (six @ GAUGE).ravel()

    def harmonic(self):
        g = reduction(self.bg)[0]
        # Cell-coordinate Fourier amplitudes, retaining the physical-site phase factors.
        phases = np.exp(1j*self.positions @ self.q)
        u = np.diag(np.r_[phases, phases])
        kc = u @ kernel(self.q, self.bg) @ u.conj().T
        hard = GAUGE.T @ kc @ GAUGE
        response = -np.linalg.solve(hard, GAUGE.T @ kc @ g)
        vector = g+GAUGE @ response
        kappa = float(np.real(vector.conj() @ kc @ vector))
        initial = self.amplitude*np.real(np.exp(1j*self.wave)[..., None]*response)
        return kappa, initial.ravel()

    def continuum(self):
        # Integrate the full angle-dependent leading gradient term over a period.
        phases = np.linspace(0, 2*np.pi, 2048, endpoint=False)
        bg = background(self.pd, self.gamma, 0.)
        rho0 = reduction(bg)[4]/(3*AREA)
        rho30 = reduction(background(self.pd, self.gamma, np.pi/6))[4]/(3*AREA)
        mean = np.trace(rho0)/2
        # z(0)=-r2+r4, Re z(pi/6)=-(r2+r4)/2.
        z0 = (rho0[0, 0]-rho0[1, 1])/2
        z30 = (rho30[0, 0]-rho30[1, 1])/2
        r2, r4 = -z30-z0/2, -z30+z0/2
        phi = self.phi0+self.amplitude*np.cos(phases)
        z = -r2*np.exp(2j*phi)+r4*np.exp(-4j*phi)
        qx, qy = self.q
        directional = mean*(qx*qx+qy*qy)+z.real*(qx*qx-qy*qy)+2*z.imag*qx*qy
        energy_per_spin = AREA/2*np.mean(directional*(self.amplitude*np.sin(phases))**2)
        fixed = self.amplitude**2*(self.q @ reduction(self.bg)[4] @ self.q)/12
        return float(energy_per_spin), float(fixed)

    def solve(self, start=None):
        before = time.monotonic()
        kappa, linear = self.harmonic()
        if start is None:
            start = linear
        result = minimize(self.energy_gradient, start, jac=True, method='L-BFGS-B',
            bounds=[(-.6, .6)]*len(start), options={'gtol': 3e-10, 'ftol': 1e-15,
                'maxiter': 1500, 'maxls': 40, 'maxcor': 12})
        en, gradient = self.energy_gradient(result.x)
        n, xy, roots = self.spins(result.x)
        staggered = (n[..., 0, 0]-n[..., 1, 0])+1j*(n[..., 0, 1]-n[..., 1, 1])
        phase_error = float(np.max(abs(np.angle(staggered*np.exp(-1j*self.phi)))))
        continuum, fixed = self.continuum()
        per_spin = en/(3*self.length**2)
        row = {'JPD_meV': self.pd, 'JGamma_meV': self.gamma, 'L_cells': self.length,
            'spins': 3*self.length**2, 'mode': self.mode.tolist(), 'q': self.q.tolist(),
            'wavelength_a': float(2*np.pi/np.linalg.norm(self.q)), 'amplitude_rad': self.amplitude,
            'phi0_rad': self.phi0, 'energy_meV_per_spin': per_spin,
            'continuum_meV_per_spin': continuum, 'frozen_phi0_meV_per_spin': fixed,
            'finite_q_harmonic_meV_per_spin': self.amplitude**2*kappa/12,
            'continuum_relative_error': (continuum-per_spin)/per_spin,
            'frozen_relative_error': (fixed-per_spin)/per_spin,
            'nonlinear_to_harmonic_relative_change': per_spin/(self.amplitude**2*kappa/12)-1,
            'max_phase_constraint_error_rad': phase_error,
            'max_spin_norm_error': float(np.max(abs(np.linalg.norm(n, axis=-1)-1))),
            'minimum_staggered_transverse_magnitude': float(abs(staggered).min()),
            'max_internal_tangent': float(np.sqrt(np.sum(xy*xy, axis=-1)).max()),
            'rms_internal_tangent': float(np.sqrt(np.mean(np.sum(xy*xy, axis=-1)))),
            'max_longitudinal_spin_change': float(S*np.max(abs(n[..., 2]-self.bg[2][:, 2]))),
            'max_chart_variable': float(np.max(abs(result.x))),
            'max_gradient_meV': float(np.max(abs(gradient))), 'optimizer_success': bool(result.success),
            'optimizer_message': str(result.message), 'iterations': result.nit,
            'energy_evaluations': result.nfev, 'wall_seconds': time.monotonic()-before}
        assert phase_error < 1e-12 and row['max_spin_norm_error'] < 1e-12
        assert row['max_chart_variable'] < .59, row
        assert row['max_gradient_meV'] < 2e-8, row
        return row, result.x


def checks():
    rng = np.random.default_rng(91826)
    system = ConstrainedWave(.01, .01, 8, .3, .173)
    x = rng.normal(scale=.025, size=8*8*5)
    e, g = system.energy_gradient(x); v = rng.normal(size=x.size)
    step = 1e-6
    fd = (system.energy_gradient(x+step*v)[0]-system.energy_gradient(x-step*v)[0])/(2*step)
    error = abs(fd-g@v)/max(1, abs(fd)); assert error < 1e-8
    n = system.spins(x)[0]
    # Independent explicit directed-bond enumeration, counted once per bond.
    direct = -system.bg[0]*S*np.sum(n[..., 2]-system.bg[2][:, 2])
    for i, j, shift, exchange, reference in system.links:
        for a in range(8):
            for b in range(8):
                neighbor = n[(a+shift[0]) % 8, (b+shift[1]) % 8, j]
                direct += S*S*(n[a, b, i] @ exchange @ neighbor-reference)
    assert abs(direct-e) < 1e-12
    harmonic = []
    for pd, gamma in PAIRS:
        for mode in [(1, 1), (1, -1)]:
            test = ConstrainedWave(pd, gamma, 16, .0005, .173, mode)
            row, _ = test.solve()
            assert abs(row['nonlinear_to_harmonic_relative_change']) < 2e-5, row
            harmonic.append(row)
    return {'directional_derivative_relative_error': error,
        'explicit_bond_energy_difference_meV': abs(direct-e), 'tiny_amplitude_harmonic': harmonic}


def minimum_checks():
    """Perturb all cells and enlarge a torus at fixed wavelength."""
    rng = np.random.default_rng(91827)
    rows = []
    for pd, gamma in PAIRS:
        system = ConstrainedWave(pd, gamma, 32, 1., np.pi/6, (1, -1))
        reference, initial = system.solve()
        starts = []
        for _ in range(2):
            candidate, _ = system.solve(initial+rng.normal(scale=.035, size=initial.size))
            starts.append({'energy_difference_meV_per_spin': candidate['energy_meV_per_spin']-reference['energy_meV_per_spin'],
                          'max_gradient_meV': candidate['max_gradient_meV']})
        assert max(abs(x['energy_difference_meV_per_spin']) for x in starts) < 1e-11
        small = ConstrainedWave(pd, gamma, 16, .5, np.pi/6, (1, 1))
        small_row, x = small.solve()
        large = ConstrainedWave(pd, gamma, 32, .5, np.pi/6, (2, 2))
        tiled = np.tile(x.reshape(16, 16, 5), (2, 2, 1)).ravel()
        large_row, _ = large.solve(tiled+rng.normal(scale=.02, size=tiled.size))
        difference = large_row['energy_meV_per_spin']-small_row['energy_meV_per_spin']
        assert abs(difference) < 1e-11
        rows.append({'JPD_meV': pd, 'JGamma_meV': gamma, 'random_full_cell_starts': starts,
                     'fixed_wavelength_L16_to_L32_energy_difference_meV_per_spin': difference})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    start = time.monotonic(); validation = checks()
    cases = []
    for pd, gamma in PAIRS:
        for mode in [(1, 1), (1, -1)]:
            for phi0 in [0., np.pi/6]:
                for amplitude in ([.02, .5] if args.pilot else [.02, .2, .5, 1.]):
                    lengths = [8, 32] if args.pilot else [8, 16, 32, 64]
                    if not args.pilot and amplitude in [.02, 1.]:
                        lengths = lengths + [128]
                    for length in lengths:
                        row, _ = ConstrainedWave(pd, gamma, length, amplitude, phi0, mode).solve()
                        cases.append(row)
                    print(f'PD={pd:g} Gamma={gamma:g} mode={mode} phi0={phi0:.3f} A={amplitude:g}: '
                          f'longest continuum error={cases[-1]["continuum_relative_error"]:.4g}', flush=True)
    multistart = minimum_checks() if not args.pilot else []
    result = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Classical nonlinear constrained smooth phase waves on the Y branch, B=0.2 T; no defects, quantum potential or thermal matching.',
        'phase_definition': 'arg(n_A^+ - n_B^+) in each magnetic cell; same prescribed phi for its reference frames. Five regular tangent coordinates relax.',
        'chart': 'n_i=n0_i sqrt(1-x_i^2-y_i^2)+e1_i x_i+e2_i y_i; x=GAUGE z, hence y_A=y_B; all z bounded by +/-0.6, no bound may be active.',
        'cell_embedding': 'Neighbor at R+ri-d, with original stored d; no production convention changed. Cell-phase Fourier kernel U Kphysical U^dagger.',
        'validation': validation, 'minimum_checks': multistart, 'cases': cases, 'wall_seconds': time.monotonic()-start,
        'peak_RSS_MiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        'inputs_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), ROOT/'examples/nbcp_y_soc_conditions.py', ROOT/'examples/nbcp_y_stiffness.py', ROOT/'examples/nbcp_ground_state.py',
            ROOT/'model/__init__.py', ROOT/'model/nbcp/__init__.py',
            ROOT/'model/nbcp/exchange.py', ROOT/'model/nbcp/unit_cells.py']},
        'limitations': ['Local constrained minima in a regular Y chart, not a proof of the global constrained minimum.',
            'The cell phase and physical-site phase differ at finite q; the finite-q harmonic comparison uses the identical cell constraint.',
            'No density walls, vortex cores, quantum stiffness corrections, thermal coefficients or RG trajectory.']}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/('pilot.json' if args.pilot else 'nonlinear-gradient-check.json')).write_text(json.dumps(result, indent=2)+'\n')
    if not args.pilot:
        plot(result)
    print(json.dumps({'cases': len(cases), 'wall_seconds': result['wall_seconds'], 'RSS_MiB': result['peak_RSS_MiB']}, indent=2))


def plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout='constrained')
    for amplitude, color in zip([.02, .2, .5, 1.], ['#606a73', '#44846c', '#18718a', '#b65a3b']):
        lengths = sorted(set(r['L_cells'] for r in report['cases'] if r['amplitude_rad'] == amplitude))
        rows = [[r for r in report['cases'] if r['L_cells'] == length and r['amplitude_rad'] == amplitude] for length in lengths]
        axes[0].loglog(lengths, [100*max(abs(x['continuum_relative_error']) for x in rr) for rr in rows], 'o-', color=color, label=f'A = {amplitude:g} rad')
        axes[1].loglog(lengths, [100*max(abs(x['frozen_relative_error']) for x in rr) for rr in rows], 'o-', color=color)
        axes[2].loglog(lengths, [max(x['max_internal_tangent'] for x in rr) for rr in rows], 'o-', color=color)
    axes[0].set_ylabel('Maximum energy error (%)')
    axes[0].set_title('Angle-dependent stiffness')
    axes[0].axhline(1, ls=':', color='#777777', lw=1)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Maximum energy error (%)'); axes[1].set_title('Stiffness frozen at background angle')
    axes[2].set_ylabel('Maximum internal tangent magnitude'); axes[2].set_title('Relaxation away from the local Y orbit')
    for ax in axes:
        ax.set_xlabel('Magnetic cells per side L'); ax.grid(alpha=.18, which='both')
        ax.set_xticks([8, 16, 32, 64, 128], labels=['8', '16', '32', '64', '128'])
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Nonlinear classical Y phase waves at 0.2 T\nMaxima over three SOC pairs, two directions and two background angles', fontsize=11)
    for suffix in ['png', 'svg']:
        fig.savefig(OUT/f'nonlinear-gradient-convergence.{suffix}', dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    main()
