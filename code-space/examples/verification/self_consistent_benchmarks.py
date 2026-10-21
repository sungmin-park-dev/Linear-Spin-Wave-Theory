"""Tables behind the self-consistent one-loop checks of 2026-10-21 (D72).

1. Exact diagonalization at large S on finite tori against the 1/S expansion:
   a. polarized BCAO in an in-plane field (3 x 1 torus, Delta and Jz+-; one
      loop with Hartree-Fock, cubic and tadpole terms);
   b. polarized XYZ ferromagnet in a field (2 x 2 torus; self-consistent and
      one loop).
   Columns ``(E_ED - E) * S``: bounded as S grows when the energy is right
   through order S^0.
2. Triangular up-up-down plateau (S^z_tot conserving, no cubic vertex):
   self-consistent boson numbers, ordered moments and gap-closing fields
   against S, with Chubukov and Golosov's one-loop limit 3JS - 0.5JS/(2S),
   3JS + 1.3JS/(2S) (columns ``(3S - h_c1) 2S / S`` and ``(h_c2 - 3S) 2S / S``).

Tests: ``code-space/tests/test_methods/test_self_consistent.py`` and B11 in
``test_reliability_benchmarks.py``.

Usage: python self_consistent_benchmarks.py > data-space/verification/261021-self-consistent/tables.txt
"""

import numpy as np

from spintoolkit import ExternalConditions, SpinModel, SpinState, Site, Term
from spintoolkit.methods.ed import solve_ed
from spintoolkit.methods.nlswt import SCSWTSettings, solve_nlswt, solve_scswt
from spintoolkit.models import bcao_minimal, polarized_state, triangular_heisenberg
from spintoolkit.system.geometry import CalculationGeometry

NAMBU3 = np.diag([1, 1, 1, -1, -1, -1])


def bcao_table():
    print("# 1a. polarized BCAO, J3=0.45, Delta1=0.3, Delta3=0.4, Jz+-=0.8, h = 3S along x, 3x1 torus")
    print("# S   E_ED/site     E_1L/site    (E_ED-E_1L)*S   HF          cubic       tadpole")
    geometry = CalculationGeometry.finite_torus([[3, 0], [0, 1]])
    for S in (1.0, 1.5, 2.0, 2.5, 3.0):
        model = bcao_minimal(J3=0.45, Delta1=0.3, Delta3=0.4, Jzpm3=0.8, S=S)
        conditions = ExternalConditions(field=(3.0 * S, 0, 0))
        exact = solve_ed(model, geometry, conditions, num_eigenvalues=2).energies()[0] / 6
        e = solve_nlswt(model, polarized_state(model, (1, 0, 0)), conditions, geometry=geometry).energies
        print(f"{S:4.1f} {exact:13.8f} {e.total:13.8f} {(exact - e.total) * S:12.4e} "
              f"{e.hartree_fock:11.3e} {e.cubic:11.3e} {e.tadpole:11.3e}")


def xyz_table():
    print("\n# 1b. square XYZ ferromagnet J = diag(0.6, -0.4, 0.3), h = 5S along z, 2x2 torus")
    print("# S   (E_ED-E_SC)*S  (E_ED-E_1L)*S  (gap_ED-gap_SC)*S  (gap_ED-gap_1L)*S  iterations")
    geometry = CalculationGeometry.finite_torus([[2, 0], [0, 2]])
    for S in (1, 2, 4, 8):
        terms = [Term.bilinear(("A", (0, 0)), ("A", d), np.diag([0.6, -0.4, 0.3]))
                 for d in [(1, 0), (0, 1)]] + [Term.zeeman("A", np.eye(3))]
        model = SpinModel(np.eye(2), (Site("A", (0.0, 0.0), S),), tuple(terms), {"model_id": "xyz"})
        conditions = ExternalConditions(field=(0, 0, 5.0 * S))
        state = polarized_state(model, (0, 0, 1))
        exact = solve_ed(model, geometry, conditions, num_eigenvalues=4).energies()
        sc = solve_scswt(model, state, conditions, geometry=geometry)
        nl = solve_nlswt(model, state, conditions, geometry=geometry)
        gap_sc = np.nanmin(sc.magnon_energies(sc.k_points))
        gap_nl = np.nanmin(nl.magnon_energies(nl.k_points)["total"])
        gap = exact[1] - exact[0]
        print(f"{S:4d} {(exact[0] / 4 - sc.ground_state_energy) * S:13.5f} "
              f"{(exact[0] / 4 - nl.energies.total) * S:13.5f} {(gap - gap_sc) * S:16.4f} "
              f"{(gap - gap_nl) * S:17.4f} {sc.header.diagnostics['iterations']:8d}")


def uud_table(mesh=24):
    print(f"\n# 2. triangular UUD at h = 3JS, {mesh}x{mesh} mesh offset by 1/3 step")
    print("# S     n_down    n_up      |<Sz>|_down |<Sz>|_up  h_c1      h_c2      "
          "(3S-h_c1)2S/S (h_c2-3S)2S/S iterations")
    directions = [(0, 0, -1.0), (0, 0, 1.0), (0, 0, 1.0)]
    for S in (0.5, 1.0, 1.5, 2.5, 5.0, 20.0, 100.0):
        model = triangular_heisenberg(J=1.0, S=S)
        state = SpinState.from_function(model, [[1, 1], [-1, 2]],
                                        lambda site, cell: np.asarray(directions[(cell[0] - cell[1]) % 3]))
        result = solve_scswt(model, state, ExternalConditions(field=(0, 0, 3 * S)),
                             settings=SCSWTSettings(mesh=(mesh, mesh)))
        H = result.hamiltonian(np.zeros((1, 2)))[0]
        w = np.sort(np.linalg.eigvals(NAMBU3 @ H).real)[3:]   # field slopes (+1, -1, +1)
        h_c1, h_c2 = 3 * S - min(w[0], w[2]), 3 * S + w[1]
        n = result.boson_numbers
        m = result.ordered_moments()
        print(f"{S:6.1f} {n[0]:9.6f} {n[1]:9.6f} {m[0]:10.6f} {m[1]:10.6f} {h_c1:9.5f} {h_c2:9.5f} "
              f"{(3 * S - h_c1) * 2:12.5f} {(h_c2 - 3 * S) * 2:12.5f} "
              f"{result.header.diagnostics['iterations']:6d}")


if __name__ == "__main__":
    bcao_table()
    xyz_table()
    uud_table()
