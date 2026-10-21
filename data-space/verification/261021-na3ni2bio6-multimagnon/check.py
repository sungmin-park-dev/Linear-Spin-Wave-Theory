import numpy as np
from twomag import *
for (L,J1,J3,D) in [(2,-3.687,1.366,0.781),(2,-1,1,3),(2,0.7,-0.3,-0.4)]:
    J,Bq=lattice(L,J1,J3)
    E0=np.linalg.eigvalsh(full_ed_sector(J,D,Bq,0))[0]
    e1=np.linalg.eigvalsh(full_ed_sector(J,D,Bq,1))-E0
    e2=np.linalg.eigvalsh(full_ed_sector(J,D,Bq,2))-E0
    a1=np.linalg.eigvalsh(one_magnon(J,D))
    a2=np.linalg.eigvalsh(two_magnon(J,D).toarray())
    print(L,J1,J3,D, np.abs(e1-a1).max(), np.abs(e2-a2).max(), len(e2))
