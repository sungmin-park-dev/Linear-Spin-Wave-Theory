"""Periodic two-wall classical Y density-domain diagnostic at B=0.2 T.

All unpinned spins vary on the full sphere. No quantum pinning or thermal
free energy is inserted. The two-wall energy gives an average tension.
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

from examples.nbcp_y_soc_conditions import background
from examples.nbcp_y_stiffness import S, rz

OUT = ROOT/'data-space/verification/260918-y-density-wall'


class DensityWall:
    """Two translated Y domains pinned in a periodic magnetic-cell torus."""

    def __init__(self, pd, gamma, length, width=1, offset=0., domain=1, axis=0):
        assert length % 4 == 0
        self.bg = background(pd, gamma, 0.)
        self.pd, self.gamma, self.length, self.width = pd, gamma, length, width
        self.offset, self.domain, self.axis = offset, domain, axis
        lattice = np.array(self.bg[7].lattice_vectors)
        self.period = float(np.linalg.norm(lattice[1-axis]))
        self.spacing = float(abs(np.linalg.det(lattice))/self.period)
        self.left = self.bg[2].copy()
        self.right = np.roll(self.left, domain, axis=0) @ rz(offset).T
        positions = np.array([self.bg[6]['Spin info'][s]['Position'] for s in ['A', 'B', 'C']])
        self.links = []
        for bond in self.bg[6]['Couplings']:
            i, j = ['A','B','C'].index(bond['SpinI']), ['A','B','C'].index(bond['SpinJ'])
            delta = (positions[i]-positions[j]-bond['Displacement']) @ np.linalg.inv(lattice)
            assert np.max(abs(delta-np.rint(delta))) < 1e-12
            shift = tuple(np.rint(delta).astype(int)[[axis, 1-axis]])
            exchange = np.asarray(bond['Exchange Matrix'])
            self.links.append((i, j, shift, exchange))
        self.free = np.ones((length, width, 3), bool)
        self.fixed = np.zeros((length, width, 3, 3))
        for layer in [0, 1]:
            self.free[layer] = False
            self.fixed[layer] = self.left
            self.free[layer+length//2] = False
            self.fixed[layer+length//2] = self.right
        uniform = np.broadcast_to(self.left, self.fixed.shape).copy()
        self.reference, _ = self.energy_spins(uniform)

    def energy_spins(self, n):
        energy = -self.bg[0]*S*np.sum(n[...,2])
        grad = np.zeros_like(n); grad[...,2] = -self.bg[0]*S
        for i,j,shift,exchange in self.links:
            neighbor = np.roll(n[...,j,:], tuple(-k for k in shift), axis=(0,1))
            field = neighbor @ exchange.T
            energy += S*S*np.sum(n[...,i,:]*field)
            grad[...,i,:] += S*S*field
            grad[...,j,:] += S*S*np.roll(n[...,i,:] @ exchange, shift, axis=(0,1))
        return float(energy), grad

    def unpack(self, flat):
        v = flat.reshape(-1,3)
        norms = np.linalg.norm(v, axis=-1, keepdims=True)
        n = self.fixed.copy(); n[self.free] = v/norms
        return n, norms

    def objective(self, flat):
        n, norms = self.unpack(flat)
        energy, grad = self.energy_spins(n)
        unit = n[self.free]; force = grad[self.free]
        projected = (force-unit*np.sum(force*unit,axis=-1,keepdims=True))/norms
        return energy-self.reference, projected.ravel()

    def seed(self, scale=1.5, noise=0., rng=None):
        t = np.arange(self.length)
        # Two separated interfaces; pinned slabs lie at the extrema.
        fraction = (1-np.tanh(np.cos(2*np.pi*(t-.5)/self.length)*self.length/(2*np.pi*scale)))/2
        n = (1-fraction[:,None,None,None])*self.left+fraction[:,None,None,None]*self.right
        n = np.broadcast_to(n,self.fixed.shape).copy()
        if noise:
            n += (rng or np.random.default_rng(918)).normal(scale=noise,size=n.shape)
        n /= np.linalg.norm(n,axis=-1,keepdims=True)
        return n[self.free].ravel()

    def solve(self, start=None, scale=1.5, noise=0., maxiter=5000):
        then = time.monotonic()
        x = self.seed(scale,noise) if start is None else start
        opt = minimize(self.objective,x,jac=True,method='L-BFGS-B',
            options={'gtol':1e-10,'ftol':1e-15,'maxiter':maxiter,'maxls':40,'maxcor':15})
        n,_ = self.unpack(opt.x)
        energy,g = self.energy_spins(n)
        tangent = g-n*np.sum(g*n,axis=-1,keepdims=True)
        residual = float(np.max(abs(tangent[self.free])))
        density = np.mean(n[...,2],axis=1) @ np.exp(2j*np.pi*np.arange(3)/3)
        # Density width is a participation length of the change in longitudinal
        # three-sublattice vector, not a fitted continuum correlation length.
        profile = np.mean(n[...,2],axis=1)
        slopes = np.linalg.norm(np.roll(profile,-1,axis=0)-profile,axis=-1)
        participation = self.spacing*np.sum(slopes)**2/(2*np.sum(slopes**2))
        row={'JPD_meV':self.pd,'JGamma_meV':self.gamma,'L_cells':self.length,
            'W_cells':self.width,'axis':self.axis,'translated_domain':self.domain,
            'offset_rad':self.offset,'spins':3*self.length*self.width,
            'wall_length_a':self.period*self.width,'wall_separation_a':self.spacing*self.length/2,
            'excess_energy_meV':energy-self.reference,
            'average_tension_meV_per_a':(energy-self.reference)/(2*self.period*self.width),
            'density_participation_width_a':float(participation),
            'min_density_amplitude_ratio':float(abs(density).min()/abs(density[0])),
            'max_projected_force_meV':residual,'optimizer_success':bool(opt.success),
            'optimizer_message':str(opt.message),'iterations':int(opt.nit),
            'wall_seconds':time.monotonic()-then,
            'density_z_profile':profile.tolist(),
            'density_complex_profile':np.column_stack([density.real,density.imag]).tolist(),
            'spin_profile':np.mean(n,axis=1).tolist(),
            'max_transverse_variation':float(np.max(abs(n-n.mean(axis=1,keepdims=True)))),
            'max_norm_error':float(np.max(abs(np.linalg.norm(n,axis=-1)-1)))}
        return row,n


def checks():
    system=DensityWall(.01,.01,16,3,.37)
    rng=np.random.default_rng(919)
    x=system.seed(noise=.13,rng=rng);v=rng.normal(size=x.size)
    e,g=system.objective(x);step=1e-6
    fd=(system.objective(x+step*v)[0]-system.objective(x-step*v)[0])/(2*step)
    relative=abs(fd-g@v)/max(abs(fd),abs(g@v),1e-12)
    assert relative<1e-6
    n,_=system.unpack(x)
    direct=-system.bg[0]*S*np.sum(n[...,2])
    for i,j,(da,db),exchange in system.links:
        for a in range(system.length):
            for b in range(system.width):
                direct+=S*S*(n[a,b,i] @ exchange @ n[(a+da)%system.length,(b+db)%system.width,j])
    exact=system.energy_spins(n)[0]
    assert abs(direct-exact)<1e-11
    domain_checks=[]
    for pd,gamma in [(0.,0.),(.005,0.),(.01,0.),(.01,.01)]:
        sys=DensityWall(pd,gamma,16,2)
        for domain in [0,1,2]:
            uniform=np.broadcast_to(np.roll(sys.left,domain,axis=0)@rz(.31).T,sys.fixed.shape).copy()
            en,gr=sys.energy_spins(uniform)
            tang=gr-uniform*np.sum(gr*uniform,axis=-1,keepdims=True)
            domain_checks.append({'pair':[pd,gamma],'domain':domain,
                'energy_difference_meV':en-sys.reference,'max_torque_meV':float(abs(tang).max())})
    assert max(abs(r['energy_difference_meV']) for r in domain_checks)<1e-11
    assert max(r['max_torque_meV'] for r in domain_checks)<1e-12
    return {'gradient_relative_error':relative,'explicit_bond_difference_meV':abs(direct-exact),
            'uniform_domains':domain_checks}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    then=time.monotonic();validation=checks();rows=[]
    for pd,gamma in [(0.,0.),(.005,0.),(.01,0.),(.01,.01)]:
        for length in ([32,64] if args.pilot else [32,64,128]):
            for offset in np.linspace(0,2*np.pi,4 if args.pilot else 8,endpoint=False):
                row,n=DensityWall(pd,gamma,length,offset=offset).solve(noise=.01)
                rows.append(row)
                print(pd,gamma,length,round(offset,3),row['average_tension_meV_per_a'],row['max_projected_force_meV'],flush=True)
    result={'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Classical periodic two-density-wall local minimization at B=0.2 T; no quantum potential or finite temperature.',
            'geometry':'Periodic magnetic a1 by a2 torus. Two layers pinned to each domain at opposite sides. Two interfaces; reported tension is their average.',
            'phase_convention':'Right boundary equals cyclic permutation of left sublattice spins followed by common laboratory z rotation through offset_rad.',
            'validation':validation,'cases':rows,'wall_seconds':time.monotonic()-then,
            'peak_RSS_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            'inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [Path(__file__),ROOT/'examples/nbcp_y_soc_conditions.py',ROOT/'examples/nbcp_y_stiffness.py',ROOT/'examples/nbcp_ground_state.py',
              ROOT/'model/__init__.py', ROOT/'model/nbcp/__init__.py',
              ROOT/'model/nbcp/exchange.py', ROOT/'model/nbcp/unit_cells.py']}}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/('pilot.json' if args.pilot else 'density-wall-check.json')).write_text(json.dumps(result,indent=2)+'\n')
    print('Seconds',result['wall_seconds'],flush=True)


if __name__=='__main__':
    main()
