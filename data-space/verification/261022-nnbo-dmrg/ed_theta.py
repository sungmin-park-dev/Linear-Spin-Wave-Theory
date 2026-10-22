"""High-T Curie-Weiss check by full ED on a 2x2 honeycomb torus (8 S=1 sites, every site z=3).
H = J1 sum S.S + K sum S^g S^g + Gamma sum (S^a S^b + S^b S^a) - D sum Sz^2 (no J3, no field).
theta_a = lim T->inf T (T chi_aa / C - 1), C = S(S+1)/3 per site.
Prediction: theta_z = -(2/3)(3 J1 + K + 2 Gamma) + D/3, theta_x = -(2/3)(3 J1 + K - Gamma) - D/6,
so theta_z - theta_x = D/2 - 2 Gamma.
Usage: python ed_theta.py J1 K Gamma D"""
import sys
import numpy as np
import scipy.sparse as sp
from dmrg_k import AX

s2 = np.sqrt(2)
Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / s2
Sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / s2
Sz = np.diag([1.0, 0, -1])
N = 8


def op(o, i):
    return sp.kron(sp.kron(sp.identity(3 ** i), sp.csr_matrix(o)), sp.identity(3 ** (N - 1 - i)), format="csr")


S = [[op(o, i) for o in (Sx, Sy, Sz)] for i in range(N)]
idx = {("A", a, b): 2 * (2 * a + b) for a in range(2) for b in range(2)}
idx.update({("B", a, b): 2 * (2 * a + b) + 1 for a in range(2) for b in range(2)})
bonds = []
for a in range(2):
    for b in range(2):
        i = idx[("A", a, b)]
        bonds += [(i, idx[("B", a, b)], "X"), (i, idx[("B", (a - 1) % 2, b)], "Y"),
                  (i, idx[("B", a, (b - 1) % 2)], "Z")]
OTHER = {"X": ("Y", "Z"), "Y": ("Z", "X"), "Z": ("X", "Y")}


def proj(i, e):
    return e[0] * S[i][0] + e[1] * S[i][1] + e[2] * S[i][2]


def hamiltonian(J1, K, G, D):
    H = sp.csr_matrix((3 ** N, 3 ** N), dtype=complex)
    for i, j, g in bonds:
        H += J1 * sum(S[i][a] @ S[j][a] for a in range(3))
        H += K * proj(i, AX[g]) @ proj(j, AX[g])
        a, b = OTHER[g]
        H += G * (proj(i, AX[a]) @ proj(j, AX[b]) + proj(i, AX[b]) @ proj(j, AX[a]))
    for i in range(N):
        H -= D * S[i][2] @ S[i][2]
    return H


if __name__ == "__main__":
    J1, K, G, D = map(float, sys.argv[1:5])
    w, v = np.linalg.eigh(hamiltonian(J1, K, G, D).toarray())
    Mz = sum(S[i][2] for i in range(N)); Mx = sum(S[i][0] for i in range(N))
    Mz2 = np.real(np.einsum("ji,ji->i", v.conj(), (Mz @ Mz) @ v))
    Mx2 = np.real(np.einsum("ji,ji->i", v.conj(), (Mx @ Mx) @ v))
    C = 2 / 3
    th = {}
    for T in (20.0, 40.0, 80.0, 160.0):
        p = np.exp(-(w - w.min()) / T); p /= p.sum()
        th[T] = [T * (T * (p @ M2) / N / T / C - 1) for M2 in (Mz2, Mx2)]
    # Richardson extrapolation in 1/T from the two largest temperatures
    tz = 2 * th[160.0][0] - th[80.0][0]; tx = 2 * th[160.0][1] - th[80.0][1]
    for T, (a, b) in th.items():
        print(f"T={T:6.1f}: theta_z={a:+.5f} theta_x={b:+.5f} diff={a - b:+.5f}")
    pz = -(2 / 3) * (3 * J1 + K + 2 * G) + D / 3; px = -(2 / 3) * (3 * J1 + K - G) - D / 6
    print(f"extrapolated: theta_z={tz:+.5f} theta_x={tx:+.5f} diff={tz - tx:+.5f}")
    print(f"prediction  : theta_z={pz:+.5f} theta_x={px:+.5f} diff={pz - px:+.5f}  (D/2 - 2G = {D / 2 - 2 * G:+.5f})")
