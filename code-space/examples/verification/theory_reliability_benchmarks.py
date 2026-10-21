"""Tables behind the theory-reliability benchmarks of 2026-10-21 (D69).

1. Triangular antiferromagnet, S = 1/2: ratio of the on-shell 1/S magnon energy
   to LSWT near the zone centre of the magnetic cell, on and off the mesh grid,
   with rho = max(|HF + tadpole|, |Re Sigma_3|) / w (Chernyshev and Zhitomirsky,
   PRB 79, 144416 (2009), Eq. (56): v0 ratio 0.88512, vQ ratio 1.08285).
2. BCAO minimal model, zero field: lowest classical energy per site on stripe
   cells [[1, 0], [0, n]] (n = 1, 2, 3, 4, 6) for J3 = 0.2 and 0.45, from
   ``classical_search`` with three seeds (Maksimov, arXiv:2308.10672, Fig. 3:
   J3 = 0.2 canted triple zigzag for 0.4 < Jz+- < 0.6; J3 = 0.45 up-up-down to
   double zigzag "around 0.9").

The closed-form and literature checks are tests in
``code-space/tests/test_methods/test_reliability_benchmarks.py``.

Usage: python theory_reliability_benchmarks.py > data-space/verification/261021-theory-reliability/tables.txt
"""

import warnings

import numpy as np

from spintoolkit import classical_search
from spintoolkit.methods.nlswt.run import NLSWTSettings, solve_nlswt
from spintoolkit.models import bcao_minimal, state_120, triangular_heisenberg


def triangular_table(meshes=(24, 48)):
    model = triangular_heisenberg(J=1.0, S=0.5)
    state = state_120(model)
    b = 2 * np.pi * np.linalg.inv(state.magnetic_lattice(model)).T
    print("# 1. triangular S=1/2: total/lswt per band [Gamma mode, +-Q modes]; rho per band")
    for n in meshes:
        result = solve_nlswt(model, state, settings=NLSWTSettings(mesh=(n, n)))
        rows = [("on grid", j / n * b[0]) for j in (1, 2, 3)]
        rows += [("off grid", f * b[0] + 0.37 * f * b[1]) for f in (0.02, 0.045, 0.08, 0.11, 0.15, 0.3)]
        for label, k in rows:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lswt, static, cubic = result.solver.magnon_energies(k[None])
            ratio = (lswt + static + cubic.real)[0] / lswt[0]
            rho = np.maximum(np.abs(static), np.abs(cubic.real))[0] / lswt[0]
            print(f"N={n:3d} {label:8s} |k|={np.linalg.norm(k):.4f} ratio={np.round(ratio, 4)} "
                  f"rho={np.round(rho, 1)} rho>N/4={rho > 0.25 * n}", flush=True)


def bcao_table():
    print("# 2. BCAO zero field: (E per site, stripe period n) lowest first; seeds 0, 1, 2")
    for J3, values in {0.2: (0.3, 0.4, 0.5, 0.6, 0.7), 0.45: (0.4, 0.5, 0.6, 0.7, 0.9, 1.0)}.items():
        for jzp in values:
            model = bcao_minimal(J3=J3, Jzpm3=jzp)
            rows = []
            for n in (1, 2, 3, 4, 6):
                energies = [classical_search(model, [[1, 0], [0, n]], maxiter=300, seed=s).energy
                            for s in (0, 1, 2)]
                rows.append((round(min(energies), 5), n, round(max(energies) - min(energies), 5)))
            print(J3, jzp, sorted(rows), flush=True)


if __name__ == "__main__":
    triangular_table()
    bcao_table()
