"""NBCP ground state optimization example.

Finds the classical ground state among four candidate magnetic unit cells
(One/Two/Three/Four MSL) for a triangular lattice antiferromagnet with
bond-angle dependent exchange interactions, then selects among degenerate
classical states by the zero-point energy on the classical manifold
(select_on_manifold, D17). The former MAGSWT grid search was removed (D27).

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
from spintoolkit.methods.lswt.energy import EnergyFunction
from spintoolkit.methods.optimization import SpinOptimizer


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


def find_ground_state(config, opt_method="classical", N=20,
                      angles_setting=None, verbose=True):
    """Search all 4 MSL phases for the ground state.

    Parameters
    ----------
    config : dict
        NBCP configuration dictionary.
    opt_method : str
        "classical" (the only search; zero-point selection: select_zero_point).
    N : int
        BZ mesh density.
    angles_setting : dict or None
        Per-phase angle constraints, e.g.
        {"One MSL": (None, 0), "Two MSL": (None, None, None, None), ...}.
    verbose : bool
        Print progress.

    Returns
    -------
    opt_result : dict
        Best result: phase_name, energy, angles, system, bz_type, MAGSWT.
    cls_result : dict
        Best classical result: phase_name, energy, angles, system, bz_type.
    all_results : dict
        Results for all phases.
    """
    Exch_J = make_nn_exchange_matrices(config)
    Exch_K = make_nnn_exchange_matrices(config)

    if verbose:
        print("=" * 60)
        print("NBCP Ground State Search")
        print("=" * 60)
        for key, val in config.items():
            print(f"  {key}: {val}")
        print(f"  opt_method: {opt_method}, N: {N}")
        print("=" * 60)

        print("\nNearest-neighbor exchange matrices:")
        for i, J in enumerate(Exch_J):
            print(f"  Bond {i} (phi={i*120}deg):\n{J}\n")

    optimizer = SpinOptimizer()
    all_results = {}

    opt_best_E = np.inf
    cls_best_E = np.inf
    opt_result = None
    cls_result = None

    for phase_name, phase_info in PHASES.items():
        builder = phase_info["builder"]
        bz_type = phase_info["bz_type"]

        # Get angle setting for this phase
        if angles_setting and phase_name in angles_setting:
            a_setting = angles_setting[phase_name]
        else:
            a_setting = None

        if verbose:
            print(f"\n--- {phase_name} ---")

        # Build SpinSystem with random initial angles, convert to legacy dict
        system = builder(config, angles=None, Exch_J=Exch_J, Exch_K=Exch_K)
        spin_sys_data = system.to_legacy_dict(bz_type)

        # Create energy function (still uses legacy dict)
        cef = EnergyFunction(spin_sys_data, N=N, update_args=True)

        # Optimize
        phase_opt, phase_cls = optimizer.find_minimum(
            cef, opt_method, a_setting, verbose=verbose,
        )

        all_results[phase_name] = phase_opt

        # Rebuild SpinSystem with optimized angles
        opt_system = builder(config, angles=tuple(phase_opt["angles"]),
                             Exch_J=Exch_J, Exch_K=Exch_K)
        cls_system = builder(config, angles=tuple(phase_cls["angles"]),
                             Exch_J=Exch_J, Exch_K=Exch_K)

        if phase_opt["energy"] < opt_best_E:
            opt_best_E = phase_opt["energy"]
            opt_result = {
                "phase_name": phase_name,
                "energy": phase_opt["energy"],
                "angles": phase_opt["angles"],
                "system": opt_system,
                "bz_type": bz_type,
                "MAGSWT": phase_opt["MAGSWT"],
                "E_cl": phase_opt["E_cl"],
                "E_qm": phase_opt["E_qm"],
            }

        if phase_cls["E_cl"] < cls_best_E:
            cls_best_E = phase_cls["E_cl"]
            cls_result = {
                "phase_name": phase_name,
                "energy": phase_cls["E_cl"],
                "angles": phase_cls["angles"],
                "system": cls_system,
                "bz_type": bz_type,
            }

    if verbose:
        print("\n" + "=" * 60)
        print(f"Classical ground state: {cls_result['phase_name']}")
        print(f"  E_cl = {cls_result['energy']:.6f}")
        print(f"  angles = {np.round(cls_result['angles'], 4)}")
        print(f"\n{opt_method} ground state: {opt_result['phase_name']}")
        print(f"  E_tot = {opt_result['energy']:.6f}")
        print(f"  E_cl  = {opt_result['E_cl']:.6f}")
        print(f"  E_qm  = {opt_result['E_qm']:.6f}")
        print(f"  MAGSWT (mu) = {opt_result['MAGSWT']:.2e}")
        print(f"  angles = {np.round(opt_result['angles'], 4)}")
        print("=" * 60)

    return opt_result, cls_result, all_results


CELL_KEYS = {"One MSL": "one_msl", "Two MSL": "two_msl",
             "Three MSL": "three_msl", "Four MSL": "four_msl"}


def select_zero_point(config, classical_result, N=20):
    """Zero-point selection on the classical manifold of the classical ground state.

    Converts the legacy angles to the common model (Zeeman energy h as the
    field with g = I) and runs :func:`select_on_manifold` with the LSWT
    zero-point energy on the same Brillouin-zone type.
    """
    from model import nbcp
    from spintoolkit.methods.state_selection import lswt_zero_point_energy, select_on_manifold
    from spintoolkit.system.conditions import ExternalConditions

    parameters = {k: v for k, v in config.items() if k != "h"}
    model = nbcp.build_model(parameters)
    state = nbcp.candidate_state(model, CELL_KEYS[classical_result["phase_name"]],
                                 classical_result["angles"])
    return select_on_manifold(model, state, ExternalConditions(field=config["h"]),
                              lswt_zero_point_energy(classical_result["bz_type"], N))


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from spintoolkit.visualization.spin_plotter import plot_spin_configuration

    angles_setting = {
        "One MSL":   (None, 0),
        "Two MSL":   (None, None, None, None),
        "Three MSL": (None, 0, None, 0, None, 0),
        "Four MSL":  (None, None, None, None, None, None, None, None),
    }

    opt_result, cls_result, all_results = find_ground_state(
        NBCP_CONFIG,
        opt_method="classical",
        N=20,
        angles_setting=angles_setting,
        verbose=True,
    )

    selection = select_zero_point(NBCP_CONFIG, cls_result, N=20)
    print(f"Zero-point selection on the classical manifold: {selection.verdict}")
    print(f"  {selection.message}")

    # Plot classical ground state spin configuration
    cls_system = cls_result["system"]
    cls_phase = cls_result["phase_name"]
    E_cl = cls_result["energy"]

    fig, ax = plot_spin_configuration(
        cls_system, n_repeat=1, figsize=(8, 8),
        title=f"NBCP {cls_phase} Classical Ground State  (E_cl = {E_cl:.6f})",
    )
    plt.show()
