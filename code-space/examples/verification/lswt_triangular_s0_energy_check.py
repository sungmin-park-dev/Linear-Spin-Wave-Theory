"""Independent evaluation of the S^0 ground-state energy of the triangular antiferromagnet.

Chernyshev and Zhitomirsky, PRB 79, 144416 (2009), write the energy of the
120-degree state of the nearest-neighbour Heisenberg model as Eq. (39),

    E/N = -(3/2) J S^2 [1 + I_2/S + (I_4 + 2 I_3)/(2S)^2],

with the Hartree-Fock constant (Eq. 40) built from c_l = sum_k gamma_k^l / omega_k,

    I_4 = (c_0 + c_1 - 2 c_2 - 1)^2 - (3/2)(c_1 - c_2)^2,

and the source-vertex integral (Eq. 41)

    I_3 = sum_{k,q} Gamma2(q, -k-q; k)^2 / (omega_k + omega_q + omega_{k+q}),

where gamma_k = (cos kx + 2 cos(kx/2) cos(sqrt3 ky/2))/3,
gamma-bar_k = (sin kx - 2 sin(kx/2) cos(sqrt3 ky/2))/3,
omega_k = sqrt((1 - gamma_k)(1 + 2 gamma_k)), A_k = 1 + gamma_k/2, and the
Bogolyubov parameters u_k = sqrt((A_k/omega_k + 1)/2),
v_k = sign(gamma_k) sqrt((A_k/omega_k - 1)/2) (so that sum u v = (3/4) c_1, their A1).
The vertex is their Eq. (22). They quote I_4 = -0.254293, I_3 = 0.13785(1) and
I_4 + 2 I_3 = 0.02141 (Eq. 42). Nothing from spintoolkit enters these formulas.

Quadrature. Both I_4 and I_3 converge only as 1/N on an N x N mesh (1/omega
singularities at the Goldstone points), but their 1/N terms cancel in the sum.
On the mesh p = (n + 1/3)/N in the reciprocal basis (N divisible by 3), k, q and
-k-q all lie on the same mesh, so the sum is the S^0 energy of one twisted
cluster and converges like N^-3. The c_l alone are extrapolated in 1/N from
N = 1200 to 4800 to give I_4 in the thermodynamic limit, and I_3 follows from
the converged sum.

Usage
-----
    python code-space/examples/verification/lswt_triangular_s0_energy_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

LATTICE = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
RECIPROCAL = 2 * np.pi * np.linalg.inv(LATTICE).T          # rows b1, b2


def functions(k):
    kx, ky = k[..., 0], k[..., 1]
    c = np.cos(np.sqrt(3) * ky / 2)
    gamma = (np.cos(kx) + 2 * np.cos(kx / 2) * c) / 3
    gamma_bar = (np.sin(kx) - 2 * np.sin(kx / 2) * c) / 3
    omega = np.sqrt(np.clip((1 - gamma) * (1 + 2 * gamma), 0, None))
    a = 1 + gamma / 2
    u = np.sqrt((a / omega + 1) / 2)
    v = np.sign(gamma) * np.sqrt((a / omega - 1) / 2)
    return gamma, gamma_bar, omega, u, v


def mesh(N, shift=1 / 3):
    p = (np.arange(N) + shift) / N
    return np.stack(np.meshgrid(p, p, indexing="ij"), -1).reshape(-1, 2) @ RECIPROCAL


def hartree_fock_constant(c):
    c0, c1, c2 = c
    return (c0 + c1 - 2 * c2 - 1) ** 2 - 1.5 * (c1 - c2) ** 2


def cluster_sum(N):
    """(I_4, I_3) on one N x N twisted cluster."""
    k = mesh(N)
    g, gb, w, u, v = functions(k)
    i4 = hartree_fock_constant([np.mean(g ** l / w) for l in range(3)])
    total = 0.0
    for a in range(len(k)):
        g2, gb2, w2, u2, v2 = functions(-k[a] - k)          # momentum 2 = -k - q
        gb3, w3, u3, v3 = gb[a], w[a], u[a], v[a]            # momentum 3 = k; 1 = q runs
        vertex = (gb * (u + v) * (u2 * v3 + v2 * u3) + gb2 * (u2 + v2) * (u * v3 + v * u3)
                  + gb3 * (u3 + v3) * (u * v2 + v * u2))
        total += np.sum(vertex ** 2 / (w + w2 + w3))
    return i4, total / len(k) ** 2


def package_value(n):
    from spintoolkit import NLSWTSettings, solve_nlswt
    from spintoolkit.models import state_120, triangular_heisenberg
    model = triangular_heisenberg(S=1.0)
    e = solve_nlswt(model, state_120(model), settings=NLSWTSettings(mesh=(n, n))).energies
    return {"magnetic_mesh": n, "order_s0": e.order_s0, "hartree_fock": e.hartree_fock,
            "cubic": e.cubic}


if __name__ == "__main__":
    c_rows = []
    for N in (1200, 2400, 4800):
        g, _, w, _, _ = functions(mesh(N))
        c_rows.append([float(np.mean(g ** l / w)) for l in range(3)])
    c_rows = np.array(c_rows)
    c_limit = 2 * c_rows[-1] - c_rows[-2]                   # Richardson in 1/N
    i4 = hartree_fock_constant(c_limit)

    clusters = []
    for N in (24, 48, 72, 96, 144):
        start = time.perf_counter()
        i4_N, i3_N = cluster_sum(N)
        coefficient = i4_N + 2 * i3_N
        clusters.append({"N": N, "I4": i4_N, "I3": i3_N, "I4_plus_2I3": coefficient,
                         "order_s0_energy": -1.5 * coefficient / 4,
                         "seconds": round(time.perf_counter() - start, 1)})
    c96, c144 = clusters[-2]["I4_plus_2I3"], clusters[-1]["I4_plus_2I3"]
    coefficient = c144 + (c144 - c96) * 96 ** 3 / (144 ** 3 - 96 ** 3)   # N^-3 tail
    report = {
        "c_l": {"N": [1200, 2400, 4800], "values": c_rows.tolist(),
                "limit": c_limit.tolist(),
                "published": [1.5747334, -0.1042539, 0.3444458]},
        "I4": {"limit": i4, "published": -0.254293},
        "clusters": clusters,
        "I4_plus_2I3": {"limit": coefficient, "tail_estimate": coefficient - c144,
                        "published": 0.02141},
        "I3": {"limit": (coefficient - i4) / 2, "published": "0.13785(1)"},
        "order_s0_energy": {"limit": -1.5 * coefficient / 4,
                            "published": -1.5 * 0.02141 / 4},
        "package": [package_value(n) for n in (12, 24)],
    }
    print(json.dumps(report, indent=2))
