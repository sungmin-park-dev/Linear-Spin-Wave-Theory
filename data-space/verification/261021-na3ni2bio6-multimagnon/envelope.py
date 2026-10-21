"""Magnetization curve of the U(1) core in the Sz-product-state limit:
lower convex hull of e(m) over all Sz-product states periodic in supercells
of <= nmax unit cells. Units |J1| = 1 (J1 = -1)."""
import numpy as np, itertools
from math import gcd
from prodsearch import offsets, supercells, Jsc
_cache={}
def configs(Ns):
    if Ns not in _cache:
        _cache[Ns]=np.array(list(itertools.product((1,0,-1),repeat=Ns)),dtype=float)
    return _cache[Ns]
def hull_curve(J3,D,nmax=5,J1=-1.0):
    offs=offsets(J1,J3); best={}
    for n in range(1,nmax+1):
        Ns=2*n; M=configs(Ns); Mt=M.sum(1).astype(int)
        for a,b,c in supercells(n):
            J=Jsc(a,b,c,offs)
            E=0.5*np.einsum('ci,ij,cj->c',M,J,M)-D*(M*M).sum(1)
            for mt in np.unique(Mt):
                sel=Mt==mt; k=np.argmin(np.where(sel,E,np.inf))
                g=gcd(abs(int(mt)),Ns) or Ns; key=(int(mt)//g,Ns//g)
                e=E[k]/Ns
                if key not in best or e<best[key][0]-1e-12: best[key]=(e,M[k].astype(int),(a,b,c))
    pts=sorted(((k[0]/k[1],v[0],k,v) for k,v in best.items() if k[0]>=0),key=lambda t:t[0])
    # lower convex hull in (m, e), m from 0..1
    hull=[]
    for p in pts:
        while len(hull)>=2:
            (m1,e1,*_),(m2,e2,*_)=hull[-2],hull[-1]
            if (e2-e1)*(p[0]-m1) >= (p[1]-e1)*(m2-m1)-1e-12: hull.pop()
            else: break
        hull.append(p)
    # transition fields between hull vertices
    out=[]
    for i,(m,e,k,v) in enumerate(hull):
        hlo=0 if i==0 else (e-hull[i-1][1])/(m-hull[i-1][0])
        hhi=np.inf if i==len(hull)-1 else (hull[i+1][1]-e)/(hull[i+1][0]-m)
        if hhi>hlo+1e-9 and hhi>0: out.append((m,max(hlo,0),hhi,v[1],v[2]))
    return out
if __name__=="__main__":
    import sys
    J3,D=float(sys.argv[1]),float(sys.argv[2])
    for m,lo,hi,cfg,cell in hull_curve(J3,D):
        print(f"m={m:.3f}  h in [{lo:.4f},{hi:.4f}]  cell={cell} cfg={cfg}")
