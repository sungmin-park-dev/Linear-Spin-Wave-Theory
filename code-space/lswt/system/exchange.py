"""Convenience functions for constructing 3x3 exchange matrices.

All functions return np.ndarray of shape (3, 3) representing the bilinear
spin interaction: H = sum_<ij> S_i^T J_ij S_j.

These can be combined with standard arithmetic:
    J = heisenberg(1.0) + dzyaloshinskii_moriya([0, 0, 0.3])
"""

import numpy as np


def heisenberg(J: float) -> np.ndarray:
    """Isotropic Heisenberg exchange: J * I_3.

    Parameters
    ----------
    J : float
        Exchange coupling strength.

    Returns
    -------
    np.ndarray
        3x3 diagonal matrix.
    """
    return J * np.eye(3)


def xxz(Jxy: float, Jz: float) -> np.ndarray:
    """XXZ anisotropic exchange: diag(Jxy, Jxy, Jz).

    Parameters
    ----------
    Jxy : float
        In-plane exchange coupling.
    Jz : float
        Out-of-plane exchange coupling.

    Returns
    -------
    np.ndarray
        3x3 diagonal matrix.
    """
    return np.diag([Jxy, Jxy, Jz])


def xxz_with_soc(Jxy: float, Jz: float, Gamma: float) -> np.ndarray:
    """XXZ exchange with symmetric off-diagonal spin-orbit coupling.

    The matrix has the form::

        [[Jxy,   Gamma, 0   ],
         [Gamma, Jxy,   0   ],
         [0,     0,     Jz  ]]

    Parameters
    ----------
    Jxy : float
        In-plane exchange coupling.
    Jz : float
        Out-of-plane exchange coupling.
    Gamma : float
        Symmetric off-diagonal coupling (spin-orbit origin).

    Returns
    -------
    np.ndarray
        3x3 exchange matrix.
    """
    J = np.diag([Jxy, Jxy, Jz])
    J[0, 1] = Gamma
    J[1, 0] = Gamma
    return J


def dzyaloshinskii_moriya(D) -> np.ndarray:
    """Antisymmetric Dzyaloshinskii-Moriya interaction.

    Constructs the antisymmetric part of the exchange matrix from
    the DM vector D = (Dx, Dy, Dz):

        J_DM = [[  0,  Dz, -Dy],
                [-Dz,   0,  Dx],
                [ Dy, -Dx,   0]]

    such that S_i^T J_DM S_j = D . (S_i x S_j).

    Parameters
    ----------
    D : array_like
        DM vector of length 3.

    Returns
    -------
    np.ndarray
        3x3 antisymmetric matrix.
    """
    D = np.asarray(D, dtype=float)
    if D.shape != (3,):
        raise ValueError(f"DM vector must have length 3, got shape {D.shape}")
    return np.array([
        [0, D[2], -D[1]],
        [-D[2], 0, D[0]],
        [D[1], -D[0], 0],
    ])


def bond_angle_exchange(phi: float, Jx: float, Jy: float, Jz: float,
                        Jpd: float = 0.0, Gamma: float = 0.0,
                        Dx: float = 0.0, Dy: float = 0.0,
                        Dz: float = 0.0) -> np.ndarray:
    """Bond-angle dependent exchange matrix for triangular lattice systems.

    Constructs a 3x3 exchange matrix that depends on the bond direction
    angle phi. Includes XXZ exchange, pseudo-dipolar (PD), symmetric
    off-diagonal (Gamma), and Dzyaloshinskii-Moriya (DM) interactions,
    all rotated according to the bond angle.

    The matrix form is::

        [[Jx + 2*Jpd*cos(phi),   Dz - 2*Jpd*sin(phi),  -Dy' - Gamma*sin(phi)],
         [-Dz - 2*Jpd*sin(phi),  Jy - 2*Jpd*cos(phi),   Dx' + Gamma*cos(phi)],
         [ Dy' - Gamma*sin(phi), -Dx' + Gamma*cos(phi),  Jz                  ]]

    where Dx', Dy' are the DM components rotated by phi.

    Parameters
    ----------
    phi : float
        Bond direction angle in radians. For a triangular lattice with
        three nearest-neighbor bonds, use phi = 0, 2*pi/3, 4*pi/3.
    Jx : float
        Exchange coupling along x.
    Jy : float
        Exchange coupling along y.
    Jz : float
        Exchange coupling along z.
    Jpd : float, optional
        Pseudo-dipolar interaction strength (default: 0).
    Gamma : float, optional
        Symmetric off-diagonal coupling strength (default: 0).
    Dx : float, optional
        DM vector x-component (default: 0).
    Dy : float, optional
        DM vector y-component (default: 0).
    Dz : float, optional
        DM vector z-component (default: 0).

    Returns
    -------
    np.ndarray
        3x3 exchange matrix.
    """
    # Rotate DM components by bond angle
    Dx_rot = Dx * np.cos(phi) - Dy * np.sin(phi)
    Dy_rot = Dx * np.sin(phi) + Dy * np.cos(phi)

    return np.array([
        [Jx + 2 * Jpd * np.cos(phi),
         Dz - 2 * Jpd * np.sin(phi),
         -Dy_rot - Gamma * np.sin(phi)],
        [-Dz - 2 * Jpd * np.sin(phi),
         Jy - 2 * Jpd * np.cos(phi),
         Dx_rot + Gamma * np.cos(phi)],
        [Dy_rot - Gamma * np.sin(phi),
         -Dx_rot + Gamma * np.cos(phi),
         Jz],
    ])


def nnn_exchange(phi: float, Kx: float, Ky: float, Kz: float,
                 Kpd: float = 0.0, KGamma: float = 0.0) -> np.ndarray:
    """Next-nearest-neighbor bond-angle dependent exchange matrix.

    Parameters
    ----------
    phi : float
        Bond direction angle in radians. For triangular lattice NNN bonds,
        use phi = pi/2, 7*pi/6, 11*pi/6.
    Kx : float
        Exchange coupling along x.
    Ky : float
        Exchange coupling along y.
    Kz : float
        Exchange coupling along z.
    Kpd : float, optional
        Pseudo-dipolar interaction strength (default: 0).
    KGamma : float, optional
        Symmetric off-diagonal coupling strength (default: 0).

    Returns
    -------
    np.ndarray
        3x3 exchange matrix.
    """
    return np.array([
        [Kx - 2 * Kpd * np.cos(phi),
         -2 * Kpd * np.sin(phi),
         KGamma * np.sin(phi)],
        [-2 * Kpd * np.sin(phi),
         Ky + 2 * Kpd * np.cos(phi),
         KGamma * np.cos(phi)],
        [KGamma * np.sin(phi),
         KGamma * np.cos(phi),
         Kz],
    ])


def kitaev(K: float, bond: str) -> np.ndarray:
    """Bond-dependent Kitaev interaction.

    Parameters
    ----------
    K : float
        Kitaev coupling strength.
    bond : str
        Bond type: 'x', 'y', or 'z'.

    Returns
    -------
    np.ndarray
        3x3 matrix with only one nonzero diagonal element.
    """
    idx = {'x': 0, 'y': 1, 'z': 2}
    if bond not in idx:
        raise ValueError(f"bond must be 'x', 'y', or 'z', got '{bond}'")
    J = np.zeros((3, 3))
    J[idx[bond], idx[bond]] = K
    return J
