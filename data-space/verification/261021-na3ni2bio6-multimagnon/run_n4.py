import numpy as np, time, sys
from twomag import lattice
from nmag import hamiltonian, lowest
J3,D=float(sys.argv[1]),float(sys.argv[2]); L=6
J,_=lattice(L,-1.0,J3); t=time.time()
H=hamiltonian(J,D,4); E=lowest(H)
print(f"L=6 J3={J3} D={D} n=4 dim={H.shape[0]} h_4={-E/4:.4f} ({time.time()-t:.0f}s)",flush=True)
