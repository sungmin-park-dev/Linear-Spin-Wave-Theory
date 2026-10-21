import numpy as np, sys
from twomag import lattice, one_magnon, two_magnon, lowest
from prodsearch import search
L=18
J3s=[0.26,0.3,0.37,0.5,0.7,1.0]; Ds=[0.0,0.21,0.5,1.0,3.0]
print("J3/|J1|  D/|J1|   h1      h2      h_prod   h2/h_prod  bind")
for J3 in J3s:
    for D in Ds:
        J,_=lattice(L,-1.0,J3)
        e1=np.linalg.eigvalsh(one_magnon(J,D))[0]; E2=lowest(two_magnon(J,D))[0]
        hp=search(-1.0,J3,D,5)[0]
        h1,h2=-e1,-E2/2
        print(f"{J3:6.2f} {D:7.2f} {h1:8.4f} {h2:8.4f} {hp:8.4f} {h2/hp if hp>0 else np.nan:8.3f} {2*e1-E2:8.4f}",flush=True)
