"""NBCP candidate magnetic unit cells, independent of the solver.

Coordinates and bond displacements use a nearest-neighbor distance of one.
The existing builders use spin 1/2 and accept explicit exchange matrices.
Omitting Exch_J or Exch_K omits that set of bonds; it does not infer exchanges
from config. The field config["h"] is the Zeeman energy vector, not tesla.
"""

import numpy as np

from spintoolkit.system.spin_system import SpinSystem


DEFAULT_SPIN = 1 / 2


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
