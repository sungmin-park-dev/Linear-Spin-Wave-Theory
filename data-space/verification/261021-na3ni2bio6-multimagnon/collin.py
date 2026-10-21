import numpy as np, sys
import envelope as ev
from prodsearch import offsets, supercells, Jsc
import itertools
from math import gcd
J3,D=float(sys.argv[1]),float(sys.argv[2])
offs=ev.offsets(-1.0,J3); best={}
for n in range(1,7):
    Ns=2*n; M=ev.configs(Ns); Mt=M.sum(1).astype(int)
    for a,b,c in supercells(n):
        J=Jsc(a,b,c,offs); E=0.5*np.einsum('ci,ij,cj->c',M,J,M)-D*(M*M).sum(1)
        for mt in np.unique(Mt):
            sel=Mt==mt; k=np.argmin(np.where(sel,E,np.inf)); g=gcd(abs(int(mt)),Ns) or Ns; key=(int(mt)//g,Ns//g)
            if key not in best or E[k]/Ns<best[key][0]-1e-12: best[key]=(E[k]/Ns,M[k].astype(int),(a,b,c))
e0=best[(0,1)][0]; e1=best[(1,1)][0]; hs=e1-e0
print("zz->pol h =",hs)
for k,v in sorted(best.items(),key=lambda t:t[0][0]/t[0][1]):
    m=k[0]/k[1]
    if m<0: continue
    line=e0+hs*m
    print(f"m={k[0]}/{k[1]}  e-line={v[0]-line:+.5f}  cell={v[2]} cfg={v[1]}")
