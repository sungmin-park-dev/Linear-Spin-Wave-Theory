"""Exact search over all Sz-product states (m_i in {1,0,-1}) periodic in every
supercell of up to nmax unit cells (2 sites each). Returns the largest field at
which some state beats the polarized one: h_sat = max (E_pol - E)/n_mag."""
import numpy as np, itertools, sys
A1=np.array([1.0,0]);A2=np.array([.5,np.sqrt(3)/2]);dB=(A1+A2)/3
def offsets(J1,J3):
    out=[]
    for n1 in range(-3,4):
        for n2 in range(-3,4):
            d=np.linalg.norm(n1*A1+n2*A2+dB)
            if abs(d-1/np.sqrt(3))<1e-6: out.append((n1,n2,J1))
            if abs(d-2/np.sqrt(3))<1e-6: out.append((n1,n2,J3))
    return out
def supercells(n):
    for a in range(1,n+1):
        if n%a: continue
        c=n//a
        for b in range(a): yield a,b,c
def Jsc(a,b,c,offs):
    n=a*c; idx={}
    cells=[(x,y) for y in range(c) for x in range(a)]
    for k,cl in enumerate(cells): idx[cl]=k
    def red(x,y):
        k=y//c; y-=k*c; x-=k*b; return idx[(x%a,y)]
    J=np.zeros((2*n,2*n))
    for (x,y) in cells:
        i=2*idx[(x,y)]
        for n1,n2,j in offs:
            k=2*red(x+n1,y+n2)+1
            J[i,k]+=j; J[k,i]+=j
    return J
def search(J1,J3,D,nmax):
    offs=offsets(J1,J3); best=(-np.inf,None)
    for n in range(1,nmax+1):
        Ns=2*n
        M=np.array(list(itertools.product((1,0,-1),repeat=Ns)),dtype=float)
        nm=(1-M).sum(1); ok=nm>0
        for a,b,c in supercells(n):
            J=Jsc(a,b,c,offs)
            E=0.5*np.einsum('ci,ij,cj->c',M,J,M)-D*(M*M).sum(1)
            Ep=0.5*J.sum()-D*Ns
            r=np.where(ok,(Ep-E)/np.where(ok,nm,1),-np.inf)
            k=r.argmax()
            if r[k]>best[0]+1e-12: best=(r[k],M[k].copy(),(a,b,c))
    return best
if __name__=="__main__":
    J1,J3,D,nmax=map(float,sys.argv[1:5])
    h,m,cell=search(J1,J3,D,int(nmax))
    print(f"h_sat={h:.4f} cell={cell} m={m.astype(int)} M/N={m.mean():.3f}")
