"""NBCP ground state optimization example.

Finds the classical + MAGSWT ground state among four candidate
magnetic unit cells (One/Two/Three/Four MSL) for a triangular lattice
antiferromagnet with bond-angle dependent exchange interactions.

Parameters match legacy/scripts/modified_do_it.py for benchmarking.
"""

import numpy as np

from lswt.core.exchange import bond_angle_exchange, nnn_exchange
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
# Unit cell builders  (return legacy dict format for EnergyFunction)
# ======================================================================

def _make_couplings(spin_i, spin_j_list, exch_matrices, displacements):
    """Helper: zip spin labels, exchange matrices, displacements into coupling dicts."""
    return [{"SpinI": spin_i, "SpinJ": spin_j,
             "Exchange Matrix": J, "Displacement": d}
            for spin_j, J, d in zip(spin_j_list, exch_matrices, displacements)]


def one_msl(config, angles=None, Exch_J=None, Exch_K=None):
    """One magnetic sublattice unit cell (1 site, BZ: Hex_60)."""
    if angles is None:
        angles = np.pi * (2 * np.random.rand(2) - 1)
    theta_a, phi_a = angles

    spin_info = {
        "A": {"Position": (0, 0), "Spin": DEFAULT_SPIN,
              "Angles": (theta_a, phi_a), "Magnetic Field": config["h"]},
    }

    lattice_vectors = (np.array([0.5, +np.sqrt(3) / 2]),
                       np.array([0.5, -np.sqrt(3) / 2]))

    couplings = []
    if Exch_J is not None:
        couplings += _make_couplings("A", ["A"] * 3, Exch_J, DISP_NN)
    if Exch_K is not None:
        couplings += _make_couplings("A", ["A"] * 3, Exch_K, DISP_NNN)

    return {"Spin info": spin_info, "Couplings": couplings,
            "Lattice/BZ setting": (lattice_vectors, "Hex_60")}


def two_msl(config, angles=None, Exch_J=None, Exch_K=None):
    """Two magnetic sublattice unit cell (2 sites, BZ: Tetra)."""
    if angles is None:
        angles = np.pi * (2 * np.random.rand(4) - 1)
    theta_a, phi_a, theta_b, phi_b = angles

    spin_info = {
        "A": {"Position": (0, 0), "Spin": DEFAULT_SPIN,
              "Angles": (theta_a, phi_a), "Magnetic Field": config["h"]},
        "B": {"Position": (0.5, np.sqrt(3) / 2), "Spin": DEFAULT_SPIN,
              "Angles": (theta_b, phi_b), "Magnetic Field": config["h"]},
    }

    lattice_vectors = (np.array([1.0, 0.0]),
                       np.array([0.0, np.sqrt(3)]))

    couplings = []
    if Exch_J is not None:
        couplings += _make_couplings("A", ["A", "B", "B"], Exch_J, DISP_NN)
        couplings += _make_couplings("B", ["B", "A", "A"], Exch_J, DISP_NN)
    if Exch_K is not None:
        couplings += _make_couplings("A", ["A", "B", "B"], Exch_K, DISP_NNN)
        couplings += _make_couplings("B", ["B", "A", "A"], Exch_K, DISP_NNN)

    return {"Spin info": spin_info, "Couplings": couplings,
            "Lattice/BZ setting": (lattice_vectors, "Tetra")}


def three_msl(config, angles=None, Exch_J=None, Exch_K=None):
    """Three magnetic sublattice unit cell (3 sites, BZ: Hex_30)."""
    if angles is None:
        angles = np.pi * (2 * np.random.rand(6) - 1)
    theta_a, phi_a, theta_b, phi_b, theta_c, phi_c = angles

    spin_info = {
        "A": {"Position": (0.5, np.sqrt(3) / 2), "Spin": DEFAULT_SPIN,
              "Angles": (theta_a, phi_a), "Magnetic Field": config["h"]},
        "B": {"Position": (-0.5, np.sqrt(3) / 2), "Spin": DEFAULT_SPIN,
              "Angles": (theta_b, phi_b), "Magnetic Field": config["h"]},
        "C": {"Position": (0, 0), "Spin": DEFAULT_SPIN,
              "Angles": (theta_c, phi_c), "Magnetic Field": config["h"]},
    }

    lattice_vectors = (np.array([1.5, +np.sqrt(3) / 2]),
                       np.array([1.5, -np.sqrt(3) / 2]))

    couplings = []
    if Exch_J is not None:
        couplings += _make_couplings("A", ["B"] * 3, Exch_J, DISP_NN)
        couplings += _make_couplings("B", ["C"] * 3, Exch_J, DISP_NN)
        couplings += _make_couplings("C", ["A"] * 3, Exch_J, DISP_NN)
    if Exch_K is not None:
        couplings += _make_couplings("A", ["A"] * 3, Exch_K, DISP_NNN)
        couplings += _make_couplings("B", ["B"] * 3, Exch_K, DISP_NNN)
        couplings += _make_couplings("C", ["C"] * 3, Exch_K, DISP_NNN)

    return {"Spin info": spin_info, "Couplings": couplings,
            "Lattice/BZ setting": (lattice_vectors, "Hex_30")}


def four_msl(config, angles=None, Exch_J=None, Exch_K=None):
    """Four magnetic sublattice unit cell (4 sites, BZ: Hex_60)."""
    if angles is None:
        angles = np.pi * (2 * np.random.rand(8) - 1)
    theta_a, phi_a, theta_b, phi_b, theta_c, phi_c, theta_d, phi_d = angles

    spin_info = {
        "A": {"Position": (1, 0), "Spin": DEFAULT_SPIN,
              "Angles": (theta_a, phi_a), "Magnetic Field": config["h"]},
        "B": {"Position": (0.5, np.sqrt(3) / 2), "Spin": DEFAULT_SPIN,
              "Angles": (theta_b, phi_b), "Magnetic Field": config["h"]},
        "C": {"Position": (-0.5, np.sqrt(3) / 2), "Spin": DEFAULT_SPIN,
              "Angles": (theta_c, phi_c), "Magnetic Field": config["h"]},
        "D": {"Position": (0, 0), "Spin": DEFAULT_SPIN,
              "Angles": (theta_d, phi_d), "Magnetic Field": config["h"]},
    }

    lattice_vectors = (np.array([1.0, +np.sqrt(3)]),
                       np.array([1.0, -np.sqrt(3)]))

    couplings = []
    if Exch_J is not None:
        couplings += _make_couplings("A", ["D", "B", "C"], Exch_J, DISP_NN)
        couplings += _make_couplings("B", ["C", "A", "D"], Exch_J, DISP_NN)
        couplings += _make_couplings("C", ["B", "D", "A"], Exch_J, DISP_NN)
        couplings += _make_couplings("D", ["A", "C", "B"], Exch_J, DISP_NN)
    if Exch_K is not None:
        couplings += _make_couplings("A", ["D", "B", "C"], Exch_K, DISP_NNN)
        couplings += _make_couplings("B", ["C", "A", "D"], Exch_K, DISP_NNN)
        couplings += _make_couplings("C", ["B", "D", "A"], Exch_K, DISP_NNN)
        couplings += _make_couplings("D", ["A", "C", "B"], Exch_K, DISP_NNN)

    return {"Spin info": spin_info, "Couplings": couplings,
            "Lattice/BZ setting": (lattice_vectors, "Hex_60")}


# ======================================================================
# Phase search: optimize all 4 MSL structures, pick lowest energy
# ======================================================================

PHASES = {
    "One MSL":   {"builder": one_msl,   "num_angles": 2},
    "Two MSL":   {"builder": two_msl,   "num_angles": 4},
    "Three MSL": {"builder": three_msl, "num_angles": 6},
    "Four MSL":  {"builder": four_msl,  "num_angles": 8},
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
        Best result: phase_name, energy, angles, spin_sys_data, MAGSWT.
    cls_result : dict
        Best classical result: phase_name, energy, angles, spin_sys_data.
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
            print(f"  Bond {i} (phi={i*120}°):\n{J}\n")

    optimizer = SpinOptimizer()
    all_results = {}

    opt_best_E = np.inf
    cls_best_E = np.inf
    opt_result = None
    cls_result = None

    for phase_name, phase_info in PHASES.items():
        builder = phase_info["builder"]
        num_angles = phase_info["num_angles"]

        # Get angle setting for this phase
        if angles_setting and phase_name in angles_setting:
            a_setting = angles_setting[phase_name]
        else:
            a_setting = None

        if verbose:
            print(f"\n--- {phase_name} ---")

        # Build spin system with random initial angles
        spin_sys_data = builder(config, angles=None, Exch_J=Exch_J, Exch_K=Exch_K)

        # Create energy function
        cef = EnergyFunction(spin_sys_data, N=N, update_args=True)

        # Optimize
        phase_opt, phase_cls = optimizer.find_minimum(
            cef, opt_method, a_setting, verbose=verbose,
        )

        all_results[phase_name] = phase_opt

        # Rebuild spin_sys_data with optimized angles
        opt_data = builder(config, angles=tuple(phase_opt["angles"]),
                           Exch_J=Exch_J, Exch_K=Exch_K)
        cls_data = builder(config, angles=tuple(phase_cls["angles"]),
                           Exch_J=Exch_J, Exch_K=Exch_K)

        if phase_opt["energy"] < opt_best_E:
            opt_best_E = phase_opt["energy"]
            opt_result = {
                "phase_name": phase_name,
                "energy": phase_opt["energy"],
                "angles": phase_opt["angles"],
                "spin_sys_data": opt_data,
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
                "spin_sys_data": cls_data,
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
