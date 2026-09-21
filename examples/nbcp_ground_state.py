"""NBCP ground state optimization example.

Finds the classical + MAGSWT ground state among four candidate
magnetic unit cells (One/Two/Three/Four MSL) for a triangular lattice
antiferromagnet with bond-angle dependent exchange interactions.

Parameters match legacy/scripts/modified_do_it.py for benchmarking.
"""

import numpy as np

from lswt.core.exchange import bond_angle_exchange, nnn_exchange
from lswt.core.spin_system import SpinSystem
from lswt.solvers.energy import EnergyFunction
from lswt.solvers.optimizer import SpinOptimizer


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

DEFAULT_SPIN = 1 / 2


# ======================================================================
# Exchange matrices from config
# ======================================================================

def make_nn_exchange_matrices(config):
    """Build 3 nearest-neighbor exchange matrices for bond angles 0, 2pi/3, 4pi/3."""
    Jx = Jy = config["Jxy"]
    Jz = config["Jz"]
    Jpd = config.get("JPD", 0.0)
    Gamma = config.get("JGamma", 0.0)
    Dx = config.get("Dx", 0.0)
    Dy = config.get("Dy", 0.0)
    Dz = config.get("Dz", 0.0)

    nn_angles = [0, 2 * np.pi / 3, 4 * np.pi / 3]
    return [bond_angle_exchange(phi, Jx, Jy, Jz, Jpd, Gamma, Dx, Dy, Dz)
            for phi in nn_angles]


def make_nnn_exchange_matrices(config):
    """Build 3 next-nearest-neighbor exchange matrices for bond angles pi/2, 7pi/6, 11pi/6."""
    Kx = Ky = config.get("Kxy", 0.0)
    Kz = config.get("Kz", 0.0)
    Kpd = config.get("KPD", 0.0)
    KGamma = config.get("KGamma", 0.0)

    if all(v == 0 for v in (Kx, Ky, Kz, Kpd, KGamma)):
        return None

    nnn_angles = [np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6]
    return [nnn_exchange(phi, Kx, Ky, Kz, Kpd, KGamma) for phi in nnn_angles]


# ======================================================================
# Displacement vectors
# ======================================================================

# Nearest-neighbor displacements (lattice_constant = 1)
DISP_NN = [
    (1.0, 0.0),
    (-0.5, np.sqrt(3) / 2),
    (-0.5, -np.sqrt(3) / 2),
]

# Next-nearest-neighbor displacements
_d_nnn = np.sqrt(3)
DISP_NNN = [
    (0.0, _d_nnn),
    (-np.sqrt(3) / 2 * _d_nnn, -0.5 * _d_nnn),
    (+np.sqrt(3) / 2 * _d_nnn, -0.5 * _d_nnn),
]


# ======================================================================
# Unit cell builders  (return SpinSystem via builder pattern)
# ======================================================================

def one_msl(config, angles=None, Exch_J=None, Exch_K=None):
    """One magnetic sublattice unit cell (1 site).

    Returns
    -------
    SpinSystem
    """
    if angles is None:
        angles = np.pi * (2 * np.random.rand(2) - 1)
    theta_a, phi_a = angles

    system = SpinSystem(lattice_vectors=[[0.5, +np.sqrt(3) / 2],
                                         [0.5, -np.sqrt(3) / 2]])
    system.add_site("A", [0, 0], DEFAULT_SPIN, [theta_a, phi_a], config["h"])

    if Exch_J is not None:
        for J, d in zip(Exch_J, DISP_NN):
            system.add_coupling("A", "A", J, d)
    if Exch_K is not None:
        for K, d in zip(Exch_K, DISP_NNN):
            system.add_coupling("A", "A", K, d)

    return system


def two_msl(config, angles=None, Exch_J=None, Exch_K=None):
    """Two magnetic sublattice unit cell (2 sites).

    Returns
    -------
    SpinSystem
    """
    if angles is None:
        angles = np.pi * (2 * np.random.rand(4) - 1)
    theta_a, phi_a, theta_b, phi_b = angles

    system = SpinSystem(lattice_vectors=[[1.0, 0.0],
                                         [0.0, np.sqrt(3)]])
    system.add_site("A", [0, 0], DEFAULT_SPIN, [theta_a, phi_a], config["h"])
    system.add_site("B", [0.5, np.sqrt(3) / 2], DEFAULT_SPIN, [theta_b, phi_b], config["h"])

    if Exch_J is not None:
        for lj, J, d in zip(["A", "B", "B"], Exch_J, DISP_NN):
            system.add_coupling("A", lj, J, d)
        for lj, J, d in zip(["B", "A", "A"], Exch_J, DISP_NN):
            system.add_coupling("B", lj, J, d)
    if Exch_K is not None:
        for lj, K, d in zip(["A", "B", "B"], Exch_K, DISP_NNN):
            system.add_coupling("A", lj, K, d)
        for lj, K, d in zip(["B", "A", "A"], Exch_K, DISP_NNN):
            system.add_coupling("B", lj, K, d)

    return system


def three_msl(config, angles=None, Exch_J=None, Exch_K=None):
    """Three magnetic sublattice unit cell (3 sites).

    Returns
    -------
    SpinSystem
    """
    if angles is None:
        angles = np.pi * (2 * np.random.rand(6) - 1)
    theta_a, phi_a, theta_b, phi_b, theta_c, phi_c = angles

    system = SpinSystem(lattice_vectors=[[1.5, +np.sqrt(3) / 2],
                                         [1.5, -np.sqrt(3) / 2]])
    system.add_site("A", [0.5, np.sqrt(3) / 2], DEFAULT_SPIN, [theta_a, phi_a], config["h"])
    system.add_site("B", [-0.5, np.sqrt(3) / 2], DEFAULT_SPIN, [theta_b, phi_b], config["h"])
    system.add_site("C", [0, 0], DEFAULT_SPIN, [theta_c, phi_c], config["h"])

    if Exch_J is not None:
        for J, d in zip(Exch_J, DISP_NN):
            system.add_coupling("A", "B", J, d)
        for J, d in zip(Exch_J, DISP_NN):
            system.add_coupling("B", "C", J, d)
        for J, d in zip(Exch_J, DISP_NN):
            system.add_coupling("C", "A", J, d)
    if Exch_K is not None:
        for K, d in zip(Exch_K, DISP_NNN):
            system.add_coupling("A", "A", K, d)
        for K, d in zip(Exch_K, DISP_NNN):
            system.add_coupling("B", "B", K, d)
        for K, d in zip(Exch_K, DISP_NNN):
            system.add_coupling("C", "C", K, d)

    return system


def four_msl(config, angles=None, Exch_J=None, Exch_K=None):
    """Four magnetic sublattice unit cell (4 sites).

    Returns
    -------
    SpinSystem
    """
    if angles is None:
        angles = np.pi * (2 * np.random.rand(8) - 1)
    theta_a, phi_a, theta_b, phi_b, theta_c, phi_c, theta_d, phi_d = angles

    system = SpinSystem(lattice_vectors=[[1.0, +np.sqrt(3)],
                                         [1.0, -np.sqrt(3)]])
    system.add_site("A", [1, 0], DEFAULT_SPIN, [theta_a, phi_a], config["h"])
    system.add_site("B", [0.5, np.sqrt(3) / 2], DEFAULT_SPIN, [theta_b, phi_b], config["h"])
    system.add_site("C", [-0.5, np.sqrt(3) / 2], DEFAULT_SPIN, [theta_c, phi_c], config["h"])
    system.add_site("D", [0, 0], DEFAULT_SPIN, [theta_d, phi_d], config["h"])

    if Exch_J is not None:
        for lj, J, d in zip(["D", "B", "C"], Exch_J, DISP_NN):
            system.add_coupling("A", lj, J, d)
        for lj, J, d in zip(["C", "A", "D"], Exch_J, DISP_NN):
            system.add_coupling("B", lj, J, d)
        for lj, J, d in zip(["B", "D", "A"], Exch_J, DISP_NN):
            system.add_coupling("C", lj, J, d)
        for lj, J, d in zip(["A", "C", "B"], Exch_J, DISP_NN):
            system.add_coupling("D", lj, J, d)
    if Exch_K is not None:
        for lj, K, d in zip(["D", "B", "C"], Exch_K, DISP_NNN):
            system.add_coupling("A", lj, K, d)
        for lj, K, d in zip(["C", "A", "D"], Exch_K, DISP_NNN):
            system.add_coupling("B", lj, K, d)
        for lj, K, d in zip(["B", "D", "A"], Exch_K, DISP_NNN):
            system.add_coupling("C", lj, K, d)
        for lj, K, d in zip(["A", "C", "B"], Exch_K, DISP_NNN):
            system.add_coupling("D", lj, K, d)

    return system


# ======================================================================
# Phase search: optimize all 4 MSL structures, pick lowest energy
# ======================================================================

PHASES = {
    "One MSL":   {"builder": one_msl,   "num_angles": 2, "bz_type": "Hex_60"},
    "Two MSL":   {"builder": two_msl,   "num_angles": 4, "bz_type": "Tetra"},
    "Three MSL": {"builder": three_msl, "num_angles": 6, "bz_type": "Hex_30"},
    "Four MSL":  {"builder": four_msl,  "num_angles": 8, "bz_type": "Hex_60"},
}


def find_ground_state(config, opt_method="MAGSWT", N=20,
                      angles_setting=None, verbose=True):
    """Search all 4 MSL phases for the ground state.

    Parameters
    ----------
    config : dict
        NBCP configuration dictionary.
    opt_method : str
        "classical" or "MAGSWT".
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


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from lswt.visualization.spin_plotter import plot_spin_configuration

    angles_setting = {
        "One MSL":   (None, 0),
        "Two MSL":   (None, None, None, None),
        "Three MSL": (None, 0, None, 0, None, 0),
        "Four MSL":  (None, None, None, None, None, None, None, None),
    }

    opt_result, cls_result, all_results = find_ground_state(
        NBCP_CONFIG,
        opt_method="MAGSWT",
        N=20,
        angles_setting=angles_setting,
        verbose=True,
    )

    # Plot classical ground state spin configuration
    cls_system = cls_result["system"]
    cls_phase = cls_result["phase_name"]
    E_cl = cls_result["energy"]

    fig, ax = plot_spin_configuration(
        cls_system, n_repeat=1, figsize=(8, 8),
        title=f"NBCP {cls_phase} Classical Ground State  (E_cl = {E_cl:.6f})",
    )
    plt.show()
