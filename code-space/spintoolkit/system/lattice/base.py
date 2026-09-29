"""
Abstract base class for crystallographic lattices.

This module defines the interface that all lattice types must implement.
Lattices are purely geometric objects - they contain NO magnetic information.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class Neighbor:
    """
    Information about a neighbor site.

    Attributes
    ----------
    source_site : int
        Index of the source atom in the basis
    target_site : int
        Index of the target atom in the basis
    displacement : np.ndarray
        Displacement vector in real space (includes lattice vector component)
    lattice_vector : Tuple[int, int]
        Which unit cell the neighbor is in (n1, n2)
    distance : float
        Distance between sites
    bond_type : str
        Label for this bond type (e.g., 'NN', 'NNN', 'NN_x', etc.)
    """
    source_site: int
    target_site: int
    displacement: np.ndarray
    lattice_vector: Tuple[int, int]
    distance: float
    bond_type: str


class AbstractLattice(ABC):
    """
    Abstract base class for crystallographic lattices.

    This represents the GEOMETRIC structure of the crystal lattice only.
    It contains NO information about:
    - Magnetic moments or spin configurations
    - Magnetic ordering patterns
    - Exchange interactions

    These magnetic properties are handled by:
    - MagneticStructure classes (spin configurations)
    - Interaction classes (exchange, anisotropy, DM)
    - SpinSystem class (combines lattice + magnetic structure + interactions)

    Design Philosophy
    -----------------
    Separation of concerns:
    - Lattice = crystallographic geometry (this class)
    - MagneticStructure = spin pattern (commensurate/incommensurate)
    - Interactions = physical couplings
    - SpinSystem = complete physical system

    This enables:
    - Reusing same lattice for different magnetic structures
    - Clear distinction between commensurate/incommensurate orderings
    - Easy testing of geometry independently from physics
    """

    @abstractmethod
    def get_primitive_vectors(self) -> np.ndarray:
        """
        Get primitive lattice vectors in real space.

        Returns
        -------
        vectors : np.ndarray, shape (2, 2)
            Primitive vectors [a1, a2] where each row is a vector.
            For 2D lattices: a1 = [a1x, a1y], a2 = [a2x, a2y]

        Notes
        -----
        Convention: right-handed coordinate system.
        The area of the unit cell is |a1 × a2|.

        Examples
        --------
        For a square lattice with a = 1:
            [[1.0, 0.0],
             [0.0, 1.0]]

        For a triangular lattice with a = 1:
            [[1.0, 0.0],
             [0.5, √3/2]]
        """
        pass

    @abstractmethod
    def get_basis_positions(self) -> List[np.ndarray]:
        """
        Get positions of atoms within the unit cell (basis).

        Returns
        -------
        positions : List[np.ndarray]
            List of basis positions in fractional coordinates.
            Each position is a 2D vector [x, y] in units of lattice vectors.

        Notes
        -----
        For a single-atom basis (e.g., triangular lattice), this returns [[0, 0]].
        For multi-atom basis (e.g., honeycomb), this returns positions of all
        atoms in the crystallographic unit cell.

        The number of basis atoms determines the size of the Hamiltonian
        for a commensurate magnetic structure with q=0.

        Examples
        --------
        Triangular lattice (1 atom):
            [[0.0, 0.0]]

        Honeycomb lattice (2 atoms):
            [[0.0, 0.0],
             [1/3, 1/3]]
        """
        pass

    @abstractmethod
    def get_neighbors(self,
                     order: int = 1,
                     max_distance: Optional[float] = None) -> List[Neighbor]:
        """
        Get neighbor information for atoms in the unit cell.

        Parameters
        ----------
        order : int, optional
            Neighbor order:
            - 1: nearest neighbors (NN)
            - 2: next-nearest neighbors (NNN)
            - 3: third-nearest neighbors
            - etc.
            Default is 1.
        max_distance : float, optional
            Maximum distance to consider. If None, use automatic
            distance cutoff based on order.

        Returns
        -------
        neighbors : List[Neighbor]
            List of all neighbor pairs within the specified order.
            Each Neighbor contains complete bonding information.

        Notes
        -----
        For a single-atom basis:
        - Each neighbor appears once (not duplicated with reverse direction)
        - Displacement vectors point from source to target

        For multi-atom basis:
        - Returns neighbors for ALL basis atoms
        - Includes both intra-cell and inter-cell neighbors

        The bond_type field helps distinguish different bond geometries:
        - Triangular lattice NN: all equivalent → 'NN'
        - Square lattice NN: may label as 'NN_x', 'NN_y'
        - Kagome lattice: different bond types even at same distance

        Examples
        --------
        Triangular lattice, order=1:
            3 nearest neighbors at distance a

        Triangular lattice, order=2:
            6 next-nearest neighbors at distance a√3
        """
        pass

    @abstractmethod
    def get_reciprocal_vectors(self) -> np.ndarray:
        """
        Get reciprocal lattice vectors.

        Returns
        -------
        vectors : np.ndarray, shape (2, 2)
            Reciprocal vectors [b1, b2] where each row is a vector.

        Notes
        -----
        Defined by: a_i · b_j = 2π δ_ij

        For 2D lattices:
            b1 = 2π * (a2_perp) / (a1 · a2_perp)
            b2 = 2π * (a1_perp) / (a2 · a1_perp)

        where a_perp is the perpendicular direction.

        Returns
        -------
        vectors : np.ndarray, shape (2, 2)
            Reciprocal lattice vectors in the same format as primitive vectors.
        """
        pass

    def get_high_symmetry_points(self) -> Dict[str, np.ndarray]:
        """
        Get high-symmetry points in the Brillouin zone.

        Returns
        -------
        points : Dict[str, np.ndarray]
            Dictionary mapping point labels to k-vectors.
            Common labels: 'Γ' (gamma), 'K', 'M', 'X', etc.

        Notes
        -----
        These points are specific to each lattice type:
        - Triangular: Γ, K, M
        - Square: Γ, X, M
        - Honeycomb: Γ, K, K', M

        k-vectors are in reciprocal space units (not fractional).

        Default implementation returns only Γ point.
        Subclasses should override for specific lattices.
        """
        return {'Γ': np.array([0.0, 0.0])}

    def get_unit_cell_area(self) -> float:
        """
        Calculate the area of the unit cell.

        Returns
        -------
        area : float
            Area in units of lattice_constant²

        Notes
        -----
        Computed as |a1 × a2| for 2D lattices.
        """
        a1, a2 = self.get_primitive_vectors()
        # 2D cross product: |a1_x * a2_y - a1_y * a2_x|
        return abs(a1[0] * a2[1] - a1[1] * a2[0])

    def get_lattice_constant(self) -> float:
        """
        Get the characteristic lattice constant.

        Returns
        -------
        a : float
            Lattice constant (typically length of a1)

        Notes
        -----
        For non-Bravais lattices or anisotropic lattices,
        this may need to be overridden.
        """
        a1, _ = self.get_primitive_vectors()
        return np.linalg.norm(a1)

    def real_to_fractional(self, position: np.ndarray) -> np.ndarray:
        """
        Convert real-space coordinates to fractional coordinates.

        Parameters
        ----------
        position : np.ndarray, shape (2,)
            Position in real space (x, y)

        Returns
        -------
        fractional : np.ndarray, shape (2,)
            Position in fractional coordinates (n1, n2)
            such that position = n1*a1 + n2*a2
        """
        a1, a2 = self.get_primitive_vectors()
        # Solve: position = n1*a1 + n2*a2
        A = np.column_stack([a1, a2])
        return np.linalg.solve(A, position)

    def fractional_to_real(self, fractional: np.ndarray) -> np.ndarray:
        """
        Convert fractional coordinates to real-space coordinates.

        Parameters
        ----------
        fractional : np.ndarray, shape (2,)
            Position in fractional coordinates (n1, n2)

        Returns
        -------
        position : np.ndarray, shape (2,)
            Position in real space: n1*a1 + n2*a2
        """
        a1, a2 = self.get_primitive_vectors()
        n1, n2 = fractional
        return n1 * a1 + n2 * a2

    def __repr__(self) -> str:
        """String representation of the lattice."""
        name = self.__class__.__name__
        a = self.get_lattice_constant()
        num_basis = len(self.get_basis_positions())
        return f"{name}(a={a:.3f}, basis_atoms={num_basis})"
