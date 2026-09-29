"""
Abstract base classes for magnetic structures.

This module defines how spin configurations are represented, separate from
the underlying crystallographic lattice geometry.

Key Distinction
---------------
Commensurate vs Incommensurate structures fundamentally differ:

Commensurate:
    - Finite magnetic supercell
    - Discrete spin angles per sublattice
    - q-vector is rational fraction of reciprocal lattice
    - Example: 120° structure on triangular lattice

Incommensurate:
    - Infinite lattice (no supercell)
    - Continuous spiral/helix modulation
    - q-vector is irrational
    - Example: spiral with q = (0.4, 0.4)

This architectural distinction affects:
    - Optimization (angles vs q-vector)
    - Visualization (discrete arrows vs continuous field)
    - Brillouin zone (folded vs extended)
    - Hamiltonian size (finite vs infinite with truncation)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Union


class AbstractMagneticStructure(ABC):
    """
    Abstract base class for magnetic ordering patterns.

    This represents HOW spins are arranged, independent of:
    - The underlying lattice geometry (handled by AbstractLattice)
    - Exchange interactions (handled by Interaction classes)

    Subclasses implement either:
    - CommensurateStructure: Finite supercell with discrete angles
    - IncommensurateStructure: Infinite spiral/helix with q-vector

    The interface is designed to work for both types while exposing
    the fundamental differences through methods like
    `get_magnetic_unit_cell_size()` which returns None for incommensurate.
    """

    @abstractmethod
    def get_spin_direction(self,
                          site_index: int,
                          unit_cell: Tuple[int, int] = (0, 0)) -> np.ndarray:
        """
        Get the spin direction at a specific site and unit cell.

        Parameters
        ----------
        site_index : int
            Index of the basis site (0, 1, 2, ... for multi-atom basis)
        unit_cell : Tuple[int, int], optional
            Which unit cell (n1, n2) in lattice vector coordinates.
            Default is (0, 0) for the origin unit cell.

        Returns
        -------
        spin : np.ndarray, shape (3,)
            Unit vector (Sx, Sy, Sz) giving spin direction

        Notes
        -----
        For commensurate structures:
            - Returns discrete spin from lookup table
            - Periodic with magnetic supercell

        For incommensurate structures:
            - Computes continuous spiral
            - Never exactly repeats
        """
        pass

    @abstractmethod
    def get_magnetic_unit_cell_size(self) -> Optional[Tuple[int, int]]:
        """
        Get the size of the magnetic unit cell.

        Returns
        -------
        size : Tuple[int, int] or None
            (n1, n2) = supercell size in units of primitive lattice vectors
            Returns None for incommensurate structures (no finite supercell)

        Examples
        --------
        Commensurate q=0 structure: (1, 1)
        Commensurate √3 × √3 structure: (√3, √3) → represented as integer supercell
        Incommensurate spiral: None
        """
        pass

    @abstractmethod
    def get_num_magnetic_sublattices(self) -> Optional[int]:
        """
        Get the number of magnetic sublattices.

        Returns
        -------
        num_sublattices : int or None
            For commensurate: number of inequivalent spins in magnetic unit cell
            For incommensurate: None (infinite sublattices)

        Notes
        -----
        For a commensurate structure with:
        - Crystallographic basis: Nb atoms
        - Magnetic supercell: (n1, n2)
        Then: num_magnetic_sublattices = Nb * n1 * n2

        Example: 3-sublattice 120° structure on triangular lattice
            Nb = 1 (triangular), supercell = (1, 1) → 1 sublattice? No!
            Actually need to specify the magnetic ordering...
            This depends on how we define the magnetic structure.
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict:
        """
        Serialize magnetic structure to dictionary.

        Returns
        -------
        data : Dict
            Dictionary representation, suitable for:
            - Saving to YAML configuration
            - Saving to results files
            - Reconstruction via from_dict()

        Notes
        -----
        Must include 'type' field to distinguish commensurate/incommensurate.
        """
        pass

    @abstractmethod
    def get_optimization_parameters(self) -> np.ndarray:
        """
        Get parameters for optimization.

        Returns
        -------
        params : np.ndarray
            For commensurate: angles [θ₁, φ₁, θ₂, φ₂, ...]
            For incommensurate: [qx, qy, θ_ref, φ_ref, ...]

        Notes
        -----
        These are the degrees of freedom for energy minimization.
        """
        pass

    @abstractmethod
    def set_optimization_parameters(self, params: np.ndarray) -> None:
        """
        Set parameters from optimization.

        Parameters
        ----------
        params : np.ndarray
            Optimized parameters in the same format as get_optimization_parameters()
        """
        pass

    def is_commensurate(self) -> bool:
        """
        Check if structure is commensurate.

        Returns
        -------
        is_commensurate : bool
            True if commensurate (finite supercell), False if incommensurate
        """
        return self.get_magnetic_unit_cell_size() is not None

    def __repr__(self) -> str:
        """String representation."""
        name = self.__class__.__name__
        if self.is_commensurate():
            size = self.get_magnetic_unit_cell_size()
            num_sl = self.get_num_magnetic_sublattices()
            return f"{name}(supercell={size}, sublattices={num_sl})"
        else:
            return f"{name}(incommensurate)"
