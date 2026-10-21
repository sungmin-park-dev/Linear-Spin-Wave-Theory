import numpy as np, time
from twomag import *
gmu=2.16*0.05788
sets={'Valenti':(-3.687,1.366,0.781),'Shangguan':(-1,1,3)}
for name,(J1,J3,D) in sets.items():
    for L in (6,12,18,24):
        t=time.time()
        J,_=lattice(L,J1,J3)
        e1=np.linalg.eigvalsh(one_magnon(J,D))[0]
        e2=lowest(two_magnon(J,D),2)
        h1,h2=-e1,-e2[0]/2
        s=f"{name:9s} L={L:2d} emin={e1:.5f} E2={e2[0]:.5f} bind={2*e1-e2[0]:.5f} h1={h1:.4f} h2={h2:.4f}"
        if name=='Valenti': s+=f"  B1={h1/gmu:.2f}T B2={h2/gmu:.2f}T"
        print(s, f"{time.time()-t:.1f}s", flush=True)
