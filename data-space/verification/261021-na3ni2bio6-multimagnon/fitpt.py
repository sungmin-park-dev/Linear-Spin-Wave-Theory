import numpy as np
from twomag import lattice, one_magnon, two_magnon, lowest
r=0.493; hs=3*r-1
for D in (0.2,0.4,0.6,1.25):
    J,_=lattice(24,-1.0,r); e1=np.linalg.eigvalsh(one_magnon(J,D))[0]; E2=lowest(two_magnon(J,D))[0]
    print(f"r={r} D={D} h_s={hs:.3f} h1={-e1:.4f} h2={-E2/2:.4f} bind={2*e1-E2:.4f} h2/h_s={-E2/2/hs:.3f}",flush=True)
