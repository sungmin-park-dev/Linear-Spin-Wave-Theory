"""Zero-field classical ground states of the BCAO minimal model (D63 comparison).

Compares with Maksimov, arXiv:2308.10672, Figs. 1(a), 2(c), 3(a): along
``Jzpm3`` the sequence zigzag -> up-up-down -> double zigzag. Each row is the
lowest classical energy per site found by ``classical_search`` on stripe
supercells ``[[1, 0], [0, n]]`` (n = 1, 2, 3, 4, 6: FM, zigzag, UUD, double
zigzag, triple zigzag families), Delta1 = Delta3 = 0, S = 1/2, J1 = -1.

Usage: python bcao_zero_field_classical.py > data-space/verification/261020-bcao-zero-field/scan.txt
"""

from spintoolkit import classical_search
from spintoolkit.models import bcao_minimal

SCAN = {0.34: (0.1, 0.2, 0.3, 0.5), 0.45: (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1)}

if __name__ == "__main__":
    print("J3  Jzpm3  [(E per site, stripe period), ...] lowest first")
    for J3, values in SCAN.items():
        for Jz in values:
            model = bcao_minimal(J3=J3, Jzpm3=Jz)
            rows = sorted((classical_search(model, [[1, 0], [0, n]], maxiter=300).energy, n)
                          for n in (1, 2, 3, 4, 6))
            print(J3, Jz, [(round(e, 5), n) for e, n in rows], flush=True)
