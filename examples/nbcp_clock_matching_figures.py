"""Plot the four separately activated SOC cases from the saved N=48 scans.

No diagonalization is rerun. Classical energy is constant on the orbit, so
semiclassical angular energy differences are evaluated from its zero-point part.
The old orbit illustration and all original calculation arrays are preserved.
"""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/lswt-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json'
OUT = ROOT/'data-space/verification/260917-clock-matching'


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    data = json.loads(SOURCE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.titlesize':11,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'svg.fonttype':'none', 'savefig.facecolor':'white'})
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), layout='constrained')
    panels = []
    for row, phase in enumerate(['Y', 'V']):
        for col, interaction in enumerate(['PD', 'Gamma']):
            pd, gamma = (.01, 0.) if interaction == 'PD' else (0., .01)
            matches = [r for r in data['scans'] if r['phase']==phase
                       and np.isclose(r['JPD_meV'], pd, rtol=0, atol=1e-14)
                       and np.isclose(r['JGamma_meV'], gamma, rtol=0, atol=1e-14)]
            assert len(matches)==1
            record = matches[0]
            assert record['current']['status']=='resolved'
            phi = np.array(record['phi'])
            energy = np.array(record['current']['Ezp_meV_per_spin'])
            classical = np.array(record['Ecl_meV_per_spin'])
            assert len(phi)==72 and np.isfinite(energy).all()
            assert np.allclose(phi, 2*np.pi*np.arange(72)/72)
            assert np.ptp(classical)<1e-14
            delta = energy-energy.min()
            exponent = int(np.floor(np.log10(delta.max())))
            factor = 10.**(-exponent)
            ax = axes[row,col]
            degree = np.degrees(phi)
            x = np.r_[degree,360.]
            plotted = np.r_[delta,delta[0]]*factor
            ax.plot(x, plotted, color=('#126b89' if col==0 else '#b35d32'),
                    linewidth=1.5, marker='o', markersize=2.0)
            coupling = r'$J_{\rm PD}=0.010,\ J_\Gamma=0$' if col==0 else r'$J_{\rm PD}=0,\ J_\Gamma=0.010$'
            title = f'({"abcd"[2*row+col]}) {phase} - '+('PD only' if col==0 else r'$\Gamma$ only')
            ax.set_title(title+f'; B = {data["states"][phase]["B_T"]:.1f} T\n'+coupling+' meV', pad=10)
            ax.set(xlabel=r'$\phi$ (degrees)',
                   ylabel=rf'$\Delta e(\phi)$ ($10^{{{exponent}}}$ meV/spin)',
                   xlim=(0,360), ylim=(-.035*plotted.max(),1.08*plotted.max()))
            ax.set_xticks(np.arange(0,361,60));ax.grid(alpha=.18)
            harmonics=np.arange(1,36)
            coeff=2/len(phi)*(np.exp(-1j*harmonics[:,None]*phi)@(energy-energy.mean()))
            dominant=int(harmonics[np.argmax(abs(coeff))])
            panels.append({'panel':'abcd'[2*row+col], 'phase':phase, 'axis':interaction,
                           'B_T':data['states'][phase]['B_T'], 'JPD_meV':pd, 'JGamma_meV':gamma,
                           'status':record['current']['status'], 'n_phi':len(phi), 'base_mesh_N':48,
                           'energy_range_meV_per_spin':float(np.ptp(energy)),
                           'classical_roundoff_range_meV_per_spin':float(np.ptp(classical)),
                           'axis_exponent_meV_per_spin':exponent, 'dominant_sampled_harmonic':dominant,
                           'phi_deg_closed':x.tolist(), 'delta_energy_meV_per_spin_closed':np.r_[delta,delta[0]].tolist()})
    fig.suptitle(r'Angular energy on the common-$z$ orbit: four separate SOC cases', fontsize=13)
    for suffix in ['png','svg']:
        fig.savefig(OUT/f'yv-angular-energy-four-cases.{suffix}', dpi=300)
    plt.close(fig)
    assert [p['dominant_sampled_harmonic'] for p in panels]==[6,6,6,3]
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==source_hash
    manifest={'scope':'Replot existing T=0 current-Hamiltonian arrays; no new physics scan',
              'definition':'Delta e = e_sw(phi)-min_phi e_sw = e_zp(phi)-min_phi e_zp, using analytically constant classical energy.',
              'normalization':'Energy per physical spin; independent explicitly labeled absolute scales, no peak normalization.',
              'source':str(SOURCE.relative_to(ROOT)), 'source_sha256':source_hash,
              'plotter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'panels':panels,
              'outputs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('yv-angular-energy-four-cases.*')}}
    (OUT/'four-case-figure.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in p.items() if not isinstance(v,list)} for p in panels],indent=2))


if __name__=='__main__':
    main()
