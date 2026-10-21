import numpy as np
from twomag import lattice, one_magnon, two_magnon, lowest
def run(L,J3,D):
    J,_=lattice(L,-1.0,J3); e1=np.linalg.eigvalsh(one_magnon(J,D))[0]; E2=lowest(two_magnon(J,D))[0]
    print(f"L={L} J3={J3} D={D} h1={-e1:.4f} h2={-E2/2:.4f} bind={2*e1-E2:.4f}",flush=True)
for L in (24,30): run(L,0.37,0.0); run(L,0.5,0.21); run(L,1.0,0.21)
for D in (0.05,0.1,0.15,0.2): run(24,0.37,D)
