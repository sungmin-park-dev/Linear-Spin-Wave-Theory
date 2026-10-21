"""SU(3) product-state (site-factorized S=1) variational search on small supercells.
Energy: sum J_ij <S_i>.<S_j> - D sum <Sz^2> - h sum <Sz>. Returns h_sat: largest h at
which some product state beats the polarized state (lower bound for the collapse field)."""
import numpy as np, sys
from scipy.optimize import minimize
from prodsearch import offsets, supercells, Jsc
sz=np.diag([1.,0,-1]); spl=np.zeros((3,3)); spl[0,1]=spl[1,2]=np.sqrt(2)
sx=(spl+spl.T)/2; sy=(spl-spl.T)/(2j)
rng=np.random.default_rng(0)
def energy(x,J,D,h,Ns):
    c=(x[:3*Ns]+1j*x[3*Ns:]).reshape(Ns,3); c/=np.linalg.norm(c,axis=1,keepdims=True)
    S=np.stack([np.einsum('ni,ij,nj->n',c.conj(),o,c).real for o in (sx,sy,sz)],1)
    q=np.einsum('ni,ij,nj->n',c.conj(),sz@sz,c).real
    return 0.5*np.einsum('ia,ij,ja->',S,J,S)-D*q.sum()-h*S[:,2].sum()
def beats(J1,J3,D,h,nmax=4,tries=6):
    offs=offsets(J1,J3)
    for n in range(1,nmax+1):
        for a,b,c in supercells(n):
            J=Jsc(a,b,c,offs); Ns=2*n; Ep=0.5*J.sum()-D*Ns-h*Ns
            for t in range(tries):
                r=minimize(energy,rng.normal(size=6*Ns),args=(J,D,h,Ns),method='L-BFGS-B')
                if r.fun<Ep-1e-7*Ns: return True
    return False
J1,J3,D=map(float,sys.argv[1:4]); lo,hi=float(sys.argv[4]),float(sys.argv[5])
for it in range(12):
    h=(lo+hi)/2
    if beats(J1,J3,D,h): lo=h
    else: hi=h
print(f"J3={J3} D={D} h_sat(SU3 product, <=8 sites) in [{lo:.4f},{hi:.4f}]")
