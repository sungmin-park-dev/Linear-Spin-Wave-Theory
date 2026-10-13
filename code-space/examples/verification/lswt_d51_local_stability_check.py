"""Independent checks of the continuous-k local stability test (D51).

1. Closed forms for the polarized state of the square ferromagnet J1 = -1 with
   an antiferromagnetic third neighbour J3 along x (spin S). A flat spiral
   with wave vector q along x has the classical energy per spin

       E(q) = S^2 (-cos q - 1 + J3 cos 2q),

   so ``E(q) - E(0) = S^2 (1/2 - 2 J3) q^2 + O(q^4)`` and the state is
   unstable for J3 > 1/4 with the minimum at cos q* = 1 / (4 J3). For a
   collinear ferromagnet H(k) = A(k) (identity) with A(k) = 2 (E(k) - E(0)) / S
   (the magnon energy),
   so the lowest eigenvalue found by local_stability must equal
   2 (E(q) - E(0)) / S at its momentum and rho_xx = S (1 - 4 J3). Nothing
   from spintoolkit enters these formulas; ``classical_energy`` of the
   spiral on a 100-site cell is printed beside them.
2. Analytic Goldstone stiffnesses: square ferromagnet and Neel state, rho = |J| S.
3. Verdicts and run times for NBCP three-sublattice states whose stability was
   checked by relaxing long supercells in nbcp-spin-supersolid (2026-10-13,
   data-space/verification/261013-classical-phase-boundaries/longcell_check.py).

Usage
-----
    python code-space/examples/verification/lswt_d51_local_stability_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from model import nbcp
from spintoolkit import solve_lswt
from spintoolkit.methods.classical import classical_energy, classical_search
from spintoolkit.methods.stability import local_stability
from spintoolkit.models import neel_state, polarized_state, square_heisenberg
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Site, SpinModel, Term

S = 0.5


def ferro_with_j3(J3):
    terms = [Term.bilinear(("A", (0, 0)), ("A", (1, 0)), -np.eye(3)),
             Term.bilinear(("A", (0, 0)), ("A", (0, 1)), -np.eye(3)),
             Term.bilinear(("A", (0, 0)), ("A", (2, 0)), J3 * np.eye(3))]
    return SpinModel(np.eye(2), [Site("A", (0, 0), S)], terms, {"model_id": f"fm_j3_{J3}"})


def spiral_energy(J3, q):
    return S ** 2 * (-np.cos(q) - 1 + J3 * np.cos(2 * q))


def ferro_checks():
    out = []
    for J3, mesh in [(0.24, 24), (0.2505, 24), (0.3, 24), (0.3, 4)]:
        model = ferro_with_j3(J3)
        state = polarized_state(model)
        start = time.perf_counter()
        report = local_stability(model, state, mesh=mesh)
        elapsed = time.perf_counter() - start
        q = 2 * np.pi * report.lowest_fractional[0]
        row = {"J3": J3, "mesh": mesh, "verdict": report.verdict, "seconds": round(elapsed, 3),
               "rho": report.zero_modes[0].stiffness_min,
               "rho_closed_form": S * (1 - 4 * J3),
               "lowest_eigenvalue": report.lowest * report.scale,
               "closed_form_at_same_q": 2 * (spiral_energy(J3, q) - spiral_energy(J3, 0)) / S,
               "lowest_q_x": float(report.lowest_fractional[0])}
        if J3 > 0.25:
            q_star = np.arccos(1 / (4 * J3))
            row["q_star_closed_form"] = float(q_star / (2 * np.pi))
            cell = 100
            q_cell = 2 * np.pi * round(cell * q_star / (2 * np.pi)) / cell
            spiral = SpinState.from_function(
                model, [[cell, 0], [0, 1]],
                lambda site, c: (np.cos(q_cell * c[0]), np.sin(q_cell * c[0]), 0.0),
                {"origin": "flat spiral"})
            row["spiral_100_cell"] = {
                "q_x": q_cell / (2 * np.pi),
                "classical_energy_gain": classical_energy(model, spiral)
                - classical_energy(model, state),
                "closed_form_gain": spiral_energy(J3, q_cell) - spiral_energy(J3, 0)}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    solve_lswt(model, state)
                    row["solve_lswt_default_mesh"] = "no error (mesh misses the instability)"
                except Exception as error:                      # noqa: BLE001
                    row["solve_lswt_default_mesh"] = f"raised: {str(error).splitlines()[0]}"
        out.append(row)
    return out


def goldstone_checks():
    out = []
    for name, J, builder in [("square ferromagnet", -1.0, polarized_state),
                             ("square Neel", 1.0, neel_state)]:
        model = square_heisenberg(J=J)
        report = local_stability(model, builder(model))
        out.append({"case": name, "verdict": report.verdict,
                    "rho": report.zero_modes[0].stiffness_min, "closed_form": abs(J) * S})
    return out


def nbcp_checks():
    cases = [("V, J_PD = J_Gamma = 0", {}, 0.40, "stable (no unstable k)"),
             ("V, J_Gamma = 0.04", {"JGamma": 0.04}, 0.44, "lower on 51 spins, gain 1.1e-5"),
             ("V, J_Gamma = 0.035", {"JGamma": 0.035}, 0.42, "lower on 96 spins, gain 1.7e-7"),
             ("Y, J_PD = 0.0125", {"JPD": 0.0125}, 0.02, "lower on 81 spins, gain 1.2e-6"),
             ("Y, J_PD = 0.01", {"JPD": 0.01}, 0.05, "stable over the orbit (phase scan)")]
    out = []
    for name, couplings, h, longcell in cases:
        model = nbcp.build_model({"Jxy": 0.076, "Jz": 0.125, **couplings})
        conditions = ExternalConditions(field=(0, 0, h))
        state = classical_search(model, nbcp.SUPERCELLS["three_msl"], conditions).state
        start = time.perf_counter()
        report = local_stability(model, state, conditions)
        out.append({"case": name, "h_meV": h, "verdict": report.verdict,
                    "seconds": round(time.perf_counter() - start, 3),
                    "zero_mode": None if not report.zero_modes else {
                        "origin": report.zero_modes[0].origin,
                        "rho_min_meV_a2": report.zero_modes[0].stiffness_min},
                    "lowest_relative": report.lowest, "long_cell_check": longcell})
    return out


if __name__ == "__main__":
    print(json.dumps({"ferromagnet_j3": ferro_checks(), "goldstone": goldstone_checks(),
                      "nbcp": nbcp_checks()}, indent=2, default=float))
