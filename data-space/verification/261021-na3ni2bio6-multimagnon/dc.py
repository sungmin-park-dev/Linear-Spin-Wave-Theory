import numpy as np
from twomag import lattice, one_magnon, two_magnon, lowest
from prodsearch import search
for J3 in (0.32,0.34,0.37,0.40,0.45,0.5,0.6):
    J,_=lattice(24,-1.0,J3); e1=np.linalg.eigvalsh(one_magnon(J,0.0))[0]; E2=lowest(two_magnon(J,0.0))[0]
    hp,m,cell=search(-1.0,J3,0.0,5)
    print(f"J3={J3} h1(0)={-e1:.4f} h2(0)={-E2/2:.4f} bind={2*e1-E2:.4f} h_prod={hp:.4f} cell={cell} Dc~{-E2/2-hp:.4f}",flush=True)
