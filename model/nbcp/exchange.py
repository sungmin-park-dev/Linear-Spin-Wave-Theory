"""NBCP bond exchange matrices assembled from model parameters.

Parameters use the same energy unit as the caller's SpinSystem. This module
selects NBCP bond angles; the general exchange formulas live in lswt.system.
"""

import numpy as np

from spintoolkit.system.exchange import bond_angle_exchange, nnn_exchange


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
