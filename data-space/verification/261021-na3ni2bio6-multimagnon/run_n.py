import numpy as np, time, sys
from twomag import lattice
from nmag import hamiltonian, lowest
pts=[(0.37,0.0),(0.37,0.15),(0.40,0.15),(0.49,0.2),(0.3705,0.212)]
plan={4:5,5:4,6:3}
L=int(sys.argv[1])
for J3,D in pts:
    J,_=lattice(L,-1.0,J3); row=[]
    for n in range(1,plan[L]+1):
        t=time.time(); E=lowest(hamiltonian(J,D,n)); row.append(-E/n)
    print(f"L={L} J3={J3} D={D} h_n=-E_n/n: "+" ".join(f"{x:.4f}" for x in row)+f"  best n={1+int(np.argmax(row))}",flush=True)
