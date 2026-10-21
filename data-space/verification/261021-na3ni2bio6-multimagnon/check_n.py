import numpy as np
from twomag import lattice, full_ed_sector, two_magnon
from nmag import hamiltonian
for J1,J3,D in [(-1,0.37,0.1),(-1,0.49,0.7)]:
    J,Bq=lattice(2,J1,J3)
    E0=np.linalg.eigvalsh(full_ed_sector(J,D,Bq,0))[0]
    for n in (1,2,3,4):
        a=np.linalg.eigvalsh(full_ed_sector(J,D,Bq,n))-E0
        b=np.linalg.eigvalsh(hamiltonian(J,D,n).toarray())
        print(J3,D,n,len(a),np.abs(a-b).max())
J,_=lattice(6,-1,0.37)
a=np.linalg.eigvalsh(two_magnon(J,0.1).toarray())[:5]; b=np.linalg.eigvalsh(hamiltonian(J,0.1,2).toarray())[:5]
print("L6 n2",np.abs(a-b).max())
