"""Render Y/V orbit and gap figures from saved data; never write the manuscript."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/lswt-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'current': '#126b89', 'legacy': '#be593b', 'formula': '#82858d'}


def select(report, phase, axis, value):
    return next(r for r in report['scans'] if r['phase'] == phase and
                abs(r['JPD_meV']-(value if axis == 'PD' else 0)) < 1e-12 and
                abs(r['JGamma_meV']-(value if axis == 'Gamma' else 0)) < 1e-12)


def orbit_figure(report, out):
    fig = plt.figure(figsize=(8.2, 6.0), layout='constrained')
    for row, phase in enumerate(['Y', 'V']):
        state = report['states'][phase]
        record = select(report, phase, 'Gamma', .01)
        ax = fig.add_subplot(2, 3, row*3+1, projection='3d')
        colors = ['#ce594e', '#398359', '#4765a4']
        p = np.linspace(0, 2*np.pi, 201)
        for label, t, color in zip(['A', 'B', 'C'], state['theta'], colors):
            if phase == 'V' and label == 'B':
                continue
            if phase == 'V' and label == 'A':
                label = 'A=B'
            xx, yy = np.sin(t)*np.cos(p), np.sin(t)*np.sin(p)
            ax.plot(xx, yy, np.full_like(p, np.cos(t)), color=color, alpha=.7)
            for angle, alpha in [(0, 1), (np.pi/2, .3)]:
                vec = np.array([np.sin(t)*np.cos(angle), np.sin(t)*np.sin(angle), np.cos(t)])
                ax.quiver(0, 0, 0, *vec, color=color, alpha=alpha, arrow_length_ratio=.12)
            ax.text(np.sin(t)*1.13, 0, np.cos(t)*1.13, label, color=color, fontsize=9)
        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel='x', ylabel='y', zlabel='z')
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])
        ax.tick_params(labelsize=7, pad=0)
        ax.xaxis.labelpad = -4
        ax.yaxis.labelpad = -4
        ax.zaxis.labelpad = -4
        ax.view_init(20, -60)
        ax.set_title(f'{phase}: B = {state["B_T"]:.1f} T')
        if phase == 'V':
            ax.text2D(.02, .93, 'A and B overlap', transform=ax.transAxes, fontsize=8)
        ax = fig.add_subplot(2, 3, row*3+2)
        ec = np.asarray(record['Ecl_meV_per_spin'])
        angle = np.degrees(record['phi'])
        ax.plot(angle, (ec-ec.mean())*1e15, color=COLORS['current'], linewidth=1)
        ax.set(xlabel=r'$\phi$ (degrees)', ylabel=r'$e_{cl}-\bar e_{cl}$ ($10^{-15}$ meV/spin)',
               title='Classical energy', xlim=(0, 360), ylim=(-.2, .2))
        ax.set_xticks([0, 120, 240, 360])
        ax.text(.04, .9, f'Roundoff only\nRange: {np.ptp(ec):.1e} meV/spin', transform=ax.transAxes, fontsize=8)
        ax = fig.add_subplot(2, 3, row*3+3)
        # Keep absolute units while resolving the very different scales.
        factor, unit = (1e12, r'$10^{-12}$ meV/spin') if phase == 'Y' else (1e6, r'$10^{-6}$ meV/spin')
        for label in ['current', 'legacy']:
            values = np.asarray(record[label]['Ezp_meV_per_spin'])
            ax.plot(np.r_[angle, 360], np.r_[values-values.min(), values[0]-values.min()]*factor,
                    label='Current Hamiltonian' if label == 'current' else 'Archived Hamiltonian',
                    color=COLORS[label], linestyle='-' if label == 'current' else '--')
        ax.set(xlabel=r'$\phi$ (degrees)', ylabel=f'$e_{{zp}}-e_{{zp,min}}$ ({unit})',
               title='Zero-point energy', xlim=(0, 360))
        ax.set_xticks([0, 120, 240, 360])
    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=2, fontsize=9, frameon=False)
    for axis in fig.axes:
        if getattr(axis, 'name', '') != '3d':
            axis.grid(alpha=.2)
    for suffix in ['png', 'svg']:
        fig.savefig(out/f'yv-equal-energy-orbits.{suffix}', dpi=300)
    plt.close(fig)


def gap_figure(report, out):
    fig, axs = plt.subplots(2, 2, figsize=(8.2, 6.4), layout='constrained')
    for i, phase in enumerate(['Y', 'V']):
        for j, axis in enumerate(['PD', 'Gamma']):
            ax = axs[i, j]
            rows = sorted([r for r in report['scans'] if r['phase'] == phase and
                           r['JGamma_meV' if axis == 'PD' else 'JPD_meV'] == 0],
                          key=lambda r: r[f'J{axis}_meV'])
            x = np.array([r[f'J{axis}_meV'] for r in rows])*1000
            for label, impl, key, style in [
                ('Current H + canonical response', 'current', 'gap_with_susceptibility_meV', '-o'),
                ('Current H + old gap formula', 'current', 'gap_with_old_uniform_theta_and_times_S_meV', ':s'),
                ('Archived H + old gap formula', 'legacy', 'gap_with_old_uniform_theta_and_times_S_meV', '--^')]:
                y = [r[impl].get(key) for r in rows]
                color = COLORS['formula'] if 'Current H + old' in label else COLORS[impl]
                ax.plot(x, [v*1000 if v is not None else np.nan for v in y], style,
                        color=color, markersize=4, linewidth=1.8, label=label)
            invalid = [k for k, r in enumerate(rows) if r['current']['status'] == 'unstable_sampled_hessian']
            if invalid:
                first = x[min(invalid)]
                ax.axvspan(first, x.max()+.5, color='#edcebf', alpha=.35)
                ax.text(.99, .97, 'Sampled instability\nGap not assigned', ha='right', va='top',
                        transform=ax.transAxes, fontsize=8, color='#934123')
            unresolved = [r for r in rows if r['current']['status'] == 'below_numerical_resolution']
            if unresolved:
                ax.text(.04, .45, 'Smallest nonzero point:\nbelow numerical resolution', transform=ax.transAxes,
                        fontsize=8, color='#555555')
            state = report['states'][phase]
            coupling = r'J_{PD}' if axis == 'PD' else r'J_\Gamma'
            zero = r'$J_\Gamma=0$' if axis == 'PD' else r'$J_{PD}=0$'
            ax.set(title=f'{phase}: B = {state["B_T"]:.1f} T; '+zero,
                   xlabel=rf'${coupling}$ ($\mu$eV)', ylabel=r'Gap ($\mu$eV)', xlim=(-.5, x.max()+.5))
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.2)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=1, fontsize=9, frameon=False)
    for suffix in ['png', 'svg']:
        fig.savefig(out/f'yv-pseudo-gap-comparison.{suffix}', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT/'data-space/verification/260912-pseudo-goldstone')
    parser.add_argument('--base-mesh', type=int, default=48)
    args = parser.parse_args()
    report = json.loads((args.directory/f'scan-N{args.base_mesh}-P72.json').read_text())
    plt.rcParams.update({'font.size': 9.5, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
    orbit_figure(report, args.directory)
    gap_figure(report, args.directory)
    print(args.directory)


if __name__ == '__main__':
    main()
