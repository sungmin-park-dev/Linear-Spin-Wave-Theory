"""Plot saved classical density-wall diagnostics without rerunning physics."""

import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/lswt-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data-space/verification/260918-y-density-wall'


def main():
    scan=json.loads((OUT/'density-wall-check.json').read_text())
    validation=json.loads((OUT/'density-wall-validation.json').read_text())
    rows=scan['cases']+validation['cases']
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    colors=['#606a73','#44846c','#18718a','#b65a3b']
    for (pd,gamma),color in zip([(0.,0.),(.005,0.),(.01,0.),(.01,.01)],colors):
        selected=[r for r in rows if r['JPD_meV']==pd and r['JGamma_meV']==gamma and r['axis']==0 and r['translated_domain']==1 and r['W_cells']==1]
        lengths=[32,64,128,256,512]
        best=[min((r for r in selected if r['L_cells']==l),key=lambda x:x['average_tension_meV_per_a']) for l in lengths]
        label=f'({pd:g}, {gamma:g})'
        axes[0,0].plot(1/np.array(lengths),[r['average_tension_meV_per_a'] for r in best],'o-',color=color,label=label)
        phases=sorted((r for r in scan['cases'] if r['JPD_meV']==pd and r['JGamma_meV']==gamma and r['L_cells']==128),key=lambda x:x['offset_rad'])
        axes[0,1].plot([r['offset_rad']/np.pi for r in phases],[r['average_tension_meV_per_a'] for r in phases],'o-',color=color)
        extra=[r for r in selected if r.get('check')=='phase_twist_seed' and r['L_cells']==128]
        axes[0,1].scatter([r['offset_rad']/np.pi for r in extra],[r['average_tension_meV_per_a'] for r in extra],marker='x',s=65,color=color,zorder=5)
        core=best[2];profile=np.array(core['density_z_profile'])
        order=profile @ np.exp(2j*np.pi*np.arange(3)/3)
        center=np.argmax(np.linalg.norm(np.diff(profile[:64],axis=0),axis=1))+.5
        x=(np.arange(128)-center)*1.5
        axes[1,0].plot(np.arange(128)/128,abs(order)/abs(order[0]),color=color,label=label)
        if pd==gamma==.01:
            x=(np.arange(128)-np.argmin(abs(order)))*1.5
            for i,c in enumerate(['#44846c','#18718a','#b65a3b']):
                axes[1,1].plot(x,profile[:,i],color=c,label='ABC'[i])
    axes[0,0].set(xlabel='Inverse cell length 1/L',ylabel='Average tension (meV/a)',title='Lowest sampled branch; two periodic walls')
    axes[0,0].legend(title='(JPD, JGamma) in meV',fontsize=8)
    axes[0,1].set(xlabel='Pinned domain phase offset / pi',ylabel='Average tension (meV/a)',title='L = 128; crosses: broad-twist starts')
    axes[1,0].set(xlabel='Cell coordinate / L',ylabel='Density amplitude / bulk amplitude',title='Both density-wall cores at L = 128',xlim=(0,1))
    axes[1,1].set(xlabel='Distance from minimum density amplitude (a)',ylabel='Unit-spin longitudinal component',title='Mixed SOC: sublattice profile at one core',xlim=(-12,12));axes[1,1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Classical Y density walls at 0.2 T; local minima, no thermal free energy')
    for suffix in ['png','svg']:
        fig.savefig(OUT/f'density-wall-diagnostic.{suffix}',dpi=200)
    plt.close(fig)


if __name__=='__main__':
    main()
