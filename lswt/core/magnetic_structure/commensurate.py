"""
Commensurate magnetic structures with finite supercells.

This module implements magnetic ordering patterns that repeat with a finite
period, forming a magnetic supercell.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from .base import AbstractMagneticStructure


class CommensurateStructure(AbstractMagneticStructure):
    """
    Commensurate magnetic structure with finite magnetic supercell.

    This represents magnetic orderings where spins repeat with a finite period,
    such as:
    - Ferromagnetic (all spins parallel)
    - Simple antiferromagnetic (2-sublattice)
    - 120° structure on triangular lattice (3-sublattice)
    - √3 × √3 structures
    - Any q-vector that is a rational fraction of reciprocal lattice

    The structure is defined by:
    1. Magnetic supercell size (n1, n2)
    2. Spin angles (θ, φ) for each magnetic sublattice

    Parameters
    ----------
    num_basis_sites : int
        Number of atoms in the crystallographic unit cell
    magnetic_supercell : Tuple[int, int]
        Size of magnetic unit cell (n1, n2) in units of primitive vectors
    angles : np.ndarray, shape (num_magnetic_sublattices, 2)
        Spin angles (θ, φ) for each magnetic sublattice
        θ: polar angle (0 to π)
        φ: azimuthal angle (0 to 2π)
    spin_magnitude : float, optional
        Magnitude of spin (default: 0.5 for S=1/2)

    Attributes
    ----------
    num_magnetic_sublattices : int
        Total number of magnetic sublattices = num_basis_sites * n1 * n2

    Examples
    --------
    Ferromagnetic structure (all spins up in z):
    >>> structure = CommensurateStructure(
    ...     num_basis_sites=1,
    ...     magnetic_supercell=(1, 1),
    ...     angles=np.array([[0.0, 0.0]])  # θ=0 → z-direction
    ... )

    120° structure on triangular lattice (3-sublattice, spins in xy-plane):
    >>> angles_120 = np.array([
    ...     [np.pi/2, 0.0],           # θ=π/2, φ=0
    ...     [np.pi/2, 2*np.pi/3],     # θ=π/2, φ=2π/3
    ...     [np.pi/2, 4*np.pi/3]      # θ=π/2, φ=4π/3
    ... ])
    >>> structure = CommensurateStructure(
    ...     num_basis_sites=1,
    ...     magnetic_supercell=(1, 1),  # Actually 3-sublattice in magnetic cell
    ...     angles=angles_120
    ... )

    Notes
    -----
    The mapping from (site_index, unit_cell) to magnetic sublattice is:
        sublattice_idx = site_index + num_basis * (n1 + n2 * supercell[0])

    This assumes a standard ordering of sublattices.
    """

    def __init__(self,
                 num_basis_sites: int,
                 magnetic_supercell: Tuple[int, int],
                 angles: np.ndarray,
                 spin_magnitude: float = 0.5):
        """
        Initialize commensurate magnetic structure.

        Parameters
        ----------
        num_basis_sites : int
            Number of sites in crystallographic basis
        magnetic_supercell : Tuple[int, int]
            Magnetic supercell size (n1, n2)
        angles : np.ndarray, shape (N_mag, 2)
            Spin angles (θ, φ) for each magnetic sublattice
        spin_magnitude : float
            Spin quantum number S
        """
        # Validation
        if num_basis_sites < 1:
            raise ValueError("num_basis_sites must be at least 1")

        if magnetic_supercell[0] < 1 or magnetic_supercell[1] < 1:
            raise ValueError("Magnetic supercell dimensions must be at least 1")

        if spin_magnitude <= 0:
            raise ValueError("Spin magnitude must be positive")

        # Store parameters
        self.num_basis_sites = num_basis_sites
        self.magnetic_supercell = magnetic_supercell
        self.spin_magnitude = spin_magnitude

        # Calculate number of magnetic sublattices
        n1, n2 = magnetic_supercell
        self.num_magnetic_sublattices = num_basis_sites * n1 * n2

        # Validate angles shape
        if angles.shape[0] != self.num_magnetic_sublattices:
            raise ValueError(
                f"angles shape mismatch: expected ({self.num_magnetic_sublattices}, 2), "
                f"got {angles.shape}. "
                f"num_basis={num_basis_sites}, supercell={magnetic_supercell} "
                f"→ {self.num_magnetic_sublattices} magnetic sublattices"
            )

        if angles.shape[1] != 2:
            raise ValueError(f"angles must have shape (N, 2) for (θ, φ), got {angles.shape}")

        # Store angles
        self.angles = np.array(angles, dtype=float)

    def get_spin_direction(self,
                          site_index: int,
                          unit_cell: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Get spin direction at specified site and unit cell.

        Parameters
        ----------
        site_index : int
            Index within crystallographic basis (0 to num_basis_sites-1)
        unit_cell : Tuple[int, int]
            Unit cell coordinates (n1, n2)

        Returns
        -------
        spin : np.ndarray, shape (3,)
            Unit vector (Sx, Sy, Sz)
        """
        # Map to magnetic sublattice
        mag_idx = self._get_magnetic_sublattice_index(site_index, unit_cell)

        # Get angles
        theta, phi = self.angles[mag_idx]

        # Convert to Cartesian (unit vector)
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

    def _get_magnetic_sublattice_index(self,
                                      site_index: int,
                                      unit_cell: Tuple[int, int]) -> int:
        """
        Map (site_index, unit_cell) to magnetic sublattice index.

        Parameters
        ----------
        site_index : int
            Crystallographic basis index
        unit_cell : Tuple[int, int]
            Unit cell (n1, n2), will be wrapped to magnetic supercell

        Returns
        -------
        mag_idx : int
            Index into self.angles array (0 to num_magnetic_sublattices-1)
        """
        # Wrap unit cell to magnetic supercell
        n1_mag, n2_mag = self.magnetic_supercell
        n1_wrapped = unit_cell[0] % n1_mag
        n2_wrapped = unit_cell[1] % n2_mag

        # Linear index
        mag_idx = site_index + self.num_basis_sites * (n1_wrapped + n2_wrapped * n1_mag)

        return mag_idx

    def get_magnetic_unit_cell_size(self) -> Tuple[int, int]:
        """
        Get magnetic supercell size.

        Returns
        -------
        size : Tuple[int, int]
            (n1, n2) magnetic supercell dimensions
        """
        return self.magnetic_supercell

    def get_num_magnetic_sublattices(self) -> int:
        """
        Get number of magnetic sublattices.

        Returns
        -------
        num : int
            Number of inequivalent magnetic sites
        """
        return self.num_magnetic_sublattices

    def get_optimization_parameters(self) -> np.ndarray:
        """
        Get angles for optimization.

        Returns
        -------
        params : np.ndarray, shape (2 * num_magnetic_sublattices,)
            Flattened array [θ₁, φ₁, θ₂, φ₂, ...]
        """
        return self.angles.flatten()

    def set_optimization_parameters(self, params: np.ndarray) -> None:
        """
        Set angles from optimization result.

        Parameters
        ----------
        params : np.ndarray
            Flattened angles [θ₁, φ₁, θ₂, φ₂, ...]
        """
        if len(params) != 2 * self.num_magnetic_sublattices:
            raise ValueError(
                f"Expected {2 * self.num_magnetic_sublattices} parameters, "
                f"got {len(params)}"
            )

        self.angles = params.reshape((-1, 2))

    def to_dict(self) -> Dict:
        """
        Serialize to dictionary.

        Returns
        -------
        data : Dict
            Dictionary with keys:
            - 'type': 'commensurate'
            - 'num_basis_sites': int
            - 'magnetic_supercell': [n1, n2]
            - 'angles': list of [θ, φ] pairs
            - 'spin_magnitude': float
        """
        return {
            'type': 'commensurate',
            'num_basis_sites': self.num_basis_sites,
            'magnetic_supercell': list(self.magnetic_supercell),
            'angles': self.angles.tolist(),
            'spin_magnitude': self.spin_magnitude
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CommensurateStructure':
        """
        Reconstruct from dictionary.

        Parameters
        ----------
        data : Dict
            Dictionary from to_dict()

        Returns
        -------
        structure : CommensurateStructure
            Reconstructed object
        """
        if data.get('type') != 'commensurate':
            raise ValueError(f"Expected type 'commensurate', got '{data.get('type')}'")

        return cls(
            num_basis_sites=data['num_basis_sites'],
            magnetic_supercell=tuple(data['magnetic_supercell']),
            angles=np.array(data['angles']),
            spin_magnitude=data.get('spin_magnitude', 0.5)
        )

    def get_all_spin_directions(self) -> np.ndarray:
        """
        Get spin directions for all magnetic sublattices.

        Returns
        -------
        spins : np.ndarray, shape (num_magnetic_sublattices, 3)
            Unit vectors for all sublattices
        """
        spins = np.zeros((self.num_magnetic_sublattices, 3))

        for i in range(self.num_magnetic_sublattices):
            theta, phi = self.angles[i]
            spins[i] = [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ]

        return spins

    def get_total_magnetization(self) -> np.ndarray:
        """
        Calculate total magnetization vector.

        Returns
        -------
        M : np.ndarray, shape (3,)
            Sum of all spin directions (not normalized)

        Notes
        -----
        For antiferromagnetic structures, this should be close to zero.
        For ferromagnetic, this equals the number of sublattices.
        """
        spins = self.get_all_spin_directions()
        return np.sum(spins, axis=0)

    def __repr__(self) -> str:
        """String representation."""
        n1, n2 = self.magnetic_supercell
        return (f"CommensurateStructure(supercell=({n1}, {n2}), "
                f"sublattices={self.num_magnetic_sublattices}, "
                f"S={self.spin_magnitude})")
