"""Refine classical density-wall local minima and finite-size diagnostics."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code-space')]

import numpy as np

from examples.nbcp_y_density_wall import DensityWall, OUT


def main():
    then=time.monotonic();rng=np.random.default_rng(91918)
    initial=json.loads((OUT/'density-wall-check.json').read_text())
    rows=[]
    for pd,gamma in [(0.,0.),(.005,0.),(.01,0.),(.01,.01)]:
        candidates=[r for r in initial['cases'] if r['JPD_meV']==pd and r['JGamma_meV']==gamma and r['L_cells']==128]
        chosen=min(candidates,key=lambda r:r['average_tension_meV_per_a'])
        offset=chosen['offset_rad']
        system=DensityWall(pd,gamma,128,offset=offset)
        best=None;best_n=None
        for scale,noise in [(1.5,.01),(5.,.05),(.5,.3),(10.,.3)]:
            row,n=system.solve(scale=scale,noise=noise)
            row.update(check='multistart',seed_scale=scale,seed_noise=noise)
            rows.append(row)
            if best is None or row['average_tension_meV_per_a']<best['average_tension_meV_per_a']:
                best,best_n=row,n
        for width in [4,8,16]:
            sys=DensityWall(pd,gamma,128,width,offset=offset)
            tiled=np.tile(best_n,(1,width,1,1))
            tiled+=rng.normal(scale=.05,size=tiled.shape)
            row,n=sys.solve(start=tiled[sys.free].ravel())
            row['check']='transverse_size';rows.append(row)
            print('width',pd,gamma,width,row['average_tension_meV_per_a'],row['max_projected_force_meV'],row['max_transverse_variation'],flush=True)
        for length in [256,512]:
            for angle in [offset,(offset+np.pi)%(2*np.pi)]:
                row,n=DensityWall(pd,gamma,length,offset=angle).solve(noise=.01)
                row['check']='wall_separation';rows.append(row)
                print('length',pd,gamma,length,angle,row['average_tension_meV_per_a'],flush=True)
        for domain,axis in [(2,0),(1,1),(2,1)]:
            row,n=DensityWall(pd,gamma,128,offset=offset,domain=domain,axis=axis).solve(noise=.05)
            row['check']='domain_orientation';rows.append(row)
    # Move an imposed pi offset into a broad twist within the right domain.
    # This tests whether a high-energy direct interpolation was metastable.
    for pd,gamma in [(0.,0.),(.005,0.),(.01,0.),(.01,.01)]:
        candidates=[r for r in initial['cases'] if r['JPD_meV']==pd and r['JGamma_meV']==gamma and r['L_cells']==128]
        offset=min(candidates,key=lambda r:r['average_tension_meV_per_a'])['offset_rad']
        for length in [128,512]:
            source=DensityWall(pd,gamma,length,offset=offset)
            base,n=source.solve(noise=.01)
            target=DensityWall(pd,gamma,length,offset=(offset+np.pi)%(2*np.pi))
            coordinate=(np.arange(length)-.5)/length
            angle=np.where((coordinate>.25)&(coordinate<.75),np.pi*np.sin(2*np.pi*(coordinate-.25))**2,0.)
            c,s=np.cos(angle)[:,None,None],np.sin(angle)[:,None,None]
            rotated=n.copy()
            rotated[...,0]=c*n[...,0]-s*n[...,1]
            rotated[...,1]=s*n[...,0]+c*n[...,1]
            row,_=target.solve(start=rotated[target.free].ravel())
            row['check']='phase_twist_seed';row['base_tension_meV_per_a']=base['average_tension_meV_per_a']
            rows.append(row)
            print('twist',pd,gamma,length,row['average_tension_meV_per_a'],flush=True)
        # A third translated density domain is allowed in both initial cores.
        system=DensityWall(pd,gamma,128,offset=offset)
        seed=system.seed().reshape(-1,3)
        n,_=system.unpack(seed.ravel())
        third=np.roll(system.left,2,axis=0)
        for center in [32,96]:
            n[center-4:center+4]=third
        n+=rng.normal(scale=.03,size=n.shape)
        row,_=system.solve(start=n[system.free].ravel())
        row['check']='third_domain_seed';rows.append(row)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'cases':rows,
        'wall_seconds':time.monotonic()-then,
        'scope':'Finite-size and local-minimum diagnostics; no global-minimum proof or finite-temperature inference.',
        'inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'examples/nbcp_y_density_wall.py',OUT/'density-wall-check.json']}}
    (OUT/'density-wall-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Seconds',report['wall_seconds'],flush=True)


if __name__=='__main__':
    main()
