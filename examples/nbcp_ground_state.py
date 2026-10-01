"""NBCP ground state optimization example.

Finds the classical ground state among four candidate magnetic unit cells
(One/Two/Three/Four MSL) for a triangular lattice antiferromagnet with
bond-angle dependent exchange interactions by the global classical search on
the common model (classical_search, D32), then selects among degenerate
classical states by the zero-point energy on the classical manifold
(select_on_manifold, D17/D28). The former MAGSWT grid search was removed
(D27); the former SpinOptimizer/EnergyFunction search is deprecated (D30).

Parameters match legacy/scripts/modified_do_it.py for benchmarking.
"""

from pathlib import Path
import sys

# Support both repository imports and direct execution of this example.
if __package__ in (None, ""):
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(_ROOT / "code-space"), str(_ROOT)]

import numpy as np

# Re-export the existing builder names for callers of this example module.
from model.nbcp.exchange import make_nn_exchange_matrices, make_nnn_exchange_matrices
from model.nbcp.unit_cells import (
    DEFAULT_SPIN, DISP_NN, DISP_NNN,
    one_msl, two_msl, three_msl, four_msl,
)
from model import nbcp as nbcp_model
from spintoolkit.methods.classical import classical_search
from spintoolkit.system.conditions import ExternalConditions


# ======================================================================
# NBCP configuration  (same as legacy modified_do_it.py)
# ======================================================================

NBCP_CONFIG = {
    "Jxy": 0.076,
    "Jz": 0.125,
    "JGamma": 0.1,
    "JPD": 0.00,
    "Kxy": 0.0,
    "Kz": 0.00,
    "KPD": 0.00,
    "KGamma": 0.0,
    "h": (0.00, 0.00, 0.376418),
}

# ======================================================================
# Phase search: optimize all 4 MSL structures, pick lowest energy
# ======================================================================

PHASES = {
    "One MSL":   {"builder": one_msl,   "num_angles": 2, "bz_type": "Hex_60"},
    "Two MSL":   {"builder": two_msl,   "num_angles": 4, "bz_type": "Tetra"},
    "Three MSL": {"builder": three_msl, "num_angles": 6, "bz_type": "Hex_30"},
    "Four MSL":  {"builder": four_msl,  "num_angles": 8, "bz_type": "Hex_60"},
}


def find_ground_state(config, verbose=False):
    """Classical ground state among the four candidate cells.

    Parameters
    ----------
    config : dict
        NBCP couplings (meV) and the Zeeman field ``h`` (meV, g = I).
    verbose : bool

    Returns
    -------
    best : dict
        ``phase_name``, ``result`` (ClassicalSearchResult), ``model``,
        ``conditions`` and ``bz_type`` of the lowest cell.
    results : dict
        ClassicalSearchResult of every cell.
    """
    parameters = {k: v for k, v in config.items() if k != "h"}
    model = nbcp_model.build_model(parameters)
    conditions = ExternalConditions(field=config["h"])
    results = {}
    for phase_name in PHASES:
        results[phase_name] = classical_search(
            model, nbcp_model.SUPERCELLS[CELL_KEYS[phase_name]], conditions)
        if verbose:
            print(f"  {phase_name:9s}  E_cl = {results[phase_name].energy:.9f} meV per site")
    best = min(results, key=lambda name: results[name].energy)
    return ({"phase_name": best, "result": results[best], "model": model,
             "conditions": conditions, "bz_type": PHASES[best]["bz_type"]}, results)


CELL_KEYS = {"One MSL": "one_msl", "Two MSL": "two_msl",
             "Three MSL": "three_msl", "Four MSL": "four_msl"}


def select_zero_point(ground, N=20):
    """Zero-point selection on the classical manifold of the classical ground state.

    Runs :func:`select_on_manifold` with the LSWT zero-point energy on the
    Brillouin-zone type of the ground-state cell.
    """
    from spintoolkit.methods.state_selection import lswt_zero_point_energy, select_on_manifold

    return select_on_manifold(ground["model"], ground["result"].state, ground["conditions"],
                              lswt_zero_point_energy(ground["bz_type"], N))


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from spintoolkit.visualization.spin_plotter import plot_spin_configuration

    print("Classical search per candidate cell:")
    ground, _ = find_ground_state(NBCP_CONFIG, verbose=True)
    print(f"Classical ground state: {ground['phase_name']} "
          f"(E_cl = {ground['result'].energy:.9f} meV per site)")

    selection = select_zero_point(ground, N=20)
    print(f"Zero-point selection on the classical manifold: {selection.verdict}")
    print(f"  {selection.message}")

    fig, ax = plot_spin_configuration(
        ground["model"], ground["result"].state, n_repeat=1, figsize=(8, 8),
        title=f"NBCP {ground['phase_name']} Classical Ground State "
              f"(E_cl = {ground['result'].energy:.6f})",
    )
    plt.show()
