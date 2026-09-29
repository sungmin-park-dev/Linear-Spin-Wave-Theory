"""
Preset lattice implementations for common 2D magnetic systems.

This module provides concrete implementations of AbstractLattice for:
- Triangular lattice
- Square lattice (TODO)
- Honeycomb lattice (TODO)
- Kagome lattice (TODO)
"""

import numpy as np
from typing import List, Dict, Optional
from .base import AbstractLattice, Neighbor


class TriangularLattice(AbstractLattice):
    """
    Triangular (hexagonal) Bravais lattice.

    This is a single-atom basis lattice where each site has 6 nearest neighbors.
    The lattice is characterized by:
    - Coordination number: 6 (NN), 6 (NNN), 6 (3NN)
    - Angles between NN bonds: 60°
    - Crystal system: Hexagonal

    Geometry
    --------
    Primitive vectors (for lattice constant a):
        a1 = a * [1, 0]
        a2 = a * [1/2, √3/2]

    Nearest neighbors (NN) at distance a:
        δ₁ = a * [1, 0]
        δ₂ = a * [-1/2, √3/2]
        δ₃ = a * [-1/2, -√3/2]

    Next-nearest neighbors (NNN) at distance a√3:
        δ₁ = a * [0, √3]
        δ₂ = a * [-3/2, -√3/2]
        δ₃ = a * [3/2, -√3/2]

    High-symmetry points in Brillouin zone:
        Γ = [0, 0]
        K = (4π/3a) * [1, 0]  (corner of hexagonal BZ)
        M = (2π/√3a) * [1, 0]  (edge midpoint)

    Parameters
    ----------
    lattice_constant : float, optional
        Lattice constant 'a' (default: 1.0)

    Examples
    --------
    >>> lattice = TriangularLattice(lattice_constant=1.0)
    >>> a1, a2 = lattice.get_primitive_vectors()
    >>> print(f"a1 = {a1}, a2 = {a2}")
    a1 = [1.0, 0.0], a2 = [0.5, 0.866...]

    >>> neighbors = lattice.get_neighbors(order=1)
    >>> print(f"Number of NN bonds: {len(neighbors)}")
    Number of NN bonds: 3

    >>> neighbors = lattice.get_neighbors(order=2)
    >>> print(f"Number of NNN bonds: {len(neighbors)}")
    Number of NNN bonds: 6

    Notes
    -----
    This lattice is commonly used for frustrated magnets:
    - Antiferromagnets on triangular lattice cannot satisfy all bonds → frustration
    - Examples: NiGa₂S₄, Ba₃CoSb₂O₉, NBCP (Na₂BaCo(PO₄)₂)
    """

    def __init__(self, lattice_constant: float = 1.0):
        """
        Initialize triangular lattice.

        Parameters
        ----------
        lattice_constant : float, optional
            Lattice constant 'a', default is 1.0
        """
        if lattice_constant <= 0:
            raise ValueError("Lattice constant must be positive")

        self.lattice_constant = lattice_constant

    def get_primitive_vectors(self) -> np.ndarray:
        """
        Get primitive lattice vectors for triangular lattice.

        Returns
        -------
        vectors : np.ndarray, shape (2, 2)
            [a1, a2] where:
            a1 = a * [1, 0]
            a2 = a * [1/2, √3/2]
        """
        a = self.lattice_constant
        return np.array([
            [a,         0.0],
            [a / 2.0,   a * np.sqrt(3) / 2.0]
        ])

    def get_basis_positions(self) -> List[np.ndarray]:
        """
        Get basis positions (single atom at origin).

        Returns
        -------
        positions : List[np.ndarray]
            [[0, 0]] - single atom at origin
        """
        return [np.array([0.0, 0.0])]

    def get_neighbors(self,
                     order: int = 1,
                     max_distance: Optional[float] = None) -> List[Neighbor]:
        """
        Get neighbor information for triangular lattice.

        Parameters
        ----------
        order : int
            1 = nearest neighbors (NN) at distance a
            2 = next-nearest neighbors (NNN) at distance a√3
            3 = third neighbors at distance 2a
        max_distance : float, optional
            Not used for triangular lattice (order determines neighbors)

        Returns
        -------
        neighbors : List[Neighbor]
            List of neighbor bonds

        Notes
        -----
        For single-atom basis, source_site = target_site = 0.
        Each bond is listed once (not duplicated with reverse direction).
        """
        a = self.lattice_constant

        if order == 1:
            # Nearest neighbors at distance a
            # Three bonds in forward direction (the other 3 are reverse)
            return [
                Neighbor(
                    source_site=0,
                    target_site=0,
                    displacement=np.array([a, 0.0]),
                    lattice_vector=(1, 0),
                    distance=a,
                    bond_type='NN'
                ),
                Neighbor(
                    source_site=0,
                    target_site=0,
                    displacement=np.array([-a/2, a*np.sqrt(3)/2]),
                    lattice_vector=(0, 1),
                    distance=a,
                    bond_type='NN'
                ),
                Neighbor(
                    source_site=0,
                    target_site=0,
                    displacement=np.array([-a/2, -a*np.sqrt(3)/2]),
                    lattice_vector=(-1, 0),
                    distance=a,
                    bond_type='NN'
                ),
            ]

        elif order == 2:
            # Next-nearest neighbors at distance a√3
            dist_nnn = a * np.sqrt(3)
            return [
                Neighbor(
                    source_site=0,
                    target_site=0,
                    displacement=np.array([0.0, dist_nnn]),
                    lattice_vector=(1, 1),
                    distance=dist_nnn,
                    bond_type='NNN'
                ),
                Neighbor(
                    source_site=0,
                    target_site=0,
                    displacement=np.array([-np.sqrt(3)/2 * dist_nnn, -0.5 * dist_nnn]),
                    lattice_vector=(-1, 0),
                    distance=dist_nnn,
                    bond_type='NNN'
                ),
                Neighbor(
                    source_site=0,
                    target_site=0,
                    displacement=np.array([np.sqrt(3)/2 * dist_nnn, -0.5 * dist_nnn]),
                    lattice_vector=(0, -1),
                    distance=dist_nnn,
                    bond_type='NNN'
                ),
                Neighbor(
                    source_site=0,
                    target_site=0,
                    displacement=np.array([0.0, -dist_nnn]),
                    lattice_vector=(-1, -1),
                    distance=dist_nnn,
                    bond_type='NNN'
                ),
                Neighbor(
                    source_site=0,
                    target_site=0,
                    displacement=np.array([np.sqrt(3)/2 * dist_nnn, 0.5 * dist_nnn]),
                    lattice_vector=(1, 0),
                    distance=dist_nnn,
                    bond_type='NNN'
                ),
                Neighbor(
                    source_site=0,
                    target_site=0,
                    displacement=np.array([-np.sqrt(3)/2 * dist_nnn, 0.5 * dist_nnn]),
                    lattice_vector=(0, 1),
                    distance=dist_nnn,
                    bond_type='NNN'
                ),
            ]

        elif order == 3:
            # Third neighbors at distance 2a
            dist_3nn = 2 * a
            return [
                Neighbor(
                    source_site=0,
                    target_site=0,
                    displacement=np.array([2*a, 0.0]),
                    lattice_vector=(2, 0),
                    distance=dist_3nn,
                    bond_type='3NN'
                ),
                Neighbor(
                    source_site=0,
                    target_site=0,
                    displacement=np.array([-a, a*np.sqrt(3)]),
                    lattice_vector=(1, 2),
                    distance=dist_3nn,
                    bond_type='3NN'
                ),
                Neighbor(
                    source_site=0,
                    target_site=0,
                    displacement=np.array([-a, -a*np.sqrt(3)]),
                    lattice_vector=(-2, 0),
                    distance=dist_3nn,
                    bond_type='3NN'
                ),
            ]

        else:
            raise ValueError(f"Neighbor order {order} not implemented. "
                           f"Supported orders: 1 (NN), 2 (NNN), 3 (3NN)")

    def get_reciprocal_vectors(self) -> np.ndarray:
        """
        Get reciprocal lattice vectors.

        Returns
        -------
        vectors : np.ndarray, shape (2, 2)
            [b1, b2] where:
            b1 = (2π/a) * [1, -1/√3]
            b2 = (2π/a) * [0, 2/√3]

        Notes
        -----
        Satisfies: aᵢ · bⱼ = 2π δᵢⱼ
        """
        a = self.lattice_constant
        factor = 2 * np.pi / a

        return np.array([
            [factor,              -factor / np.sqrt(3)],
            [0.0,                 2 * factor / np.sqrt(3)]
        ])

    def get_high_symmetry_points(self) -> Dict[str, np.ndarray]:
        """
        Get high-symmetry points in the Brillouin zone.

        Returns
        -------
        points : Dict[str, np.ndarray]
            'Γ': Gamma point (zone center) [0, 0]
            'K': Corner of hexagonal BZ
            'M': Edge midpoint of hexagonal BZ

        Notes
        -----
        Standard path for band structure: Γ → K → M → Γ
        """
        a = self.lattice_constant

        # Reciprocal lattice constant
        b = 4 * np.pi / (a * np.sqrt(3))

        return {
            'Γ': np.array([0.0, 0.0]),
            'K': np.array([b * np.sqrt(3) / 2, b / 2]),
            'M': np.array([b * np.sqrt(3) / 2, 0.0]),
        }


class SquareLattice(AbstractLattice):
    """
    Square Bravais lattice (placeholder - to be implemented).

    TODO: Implement square lattice geometry
    - NN: 4 neighbors at distance a
    - NNN: 4 neighbors at distance a√2
    - High-symmetry points: Γ, X, M
    """

    def __init__(self, lattice_constant: float = 1.0):
        raise NotImplementedError("SquareLattice not yet implemented. "
                                "Will be added in Phase 1B.")

    def get_primitive_vectors(self) -> np.ndarray:
        raise NotImplementedError

    def get_basis_positions(self) -> List[np.ndarray]:
        raise NotImplementedError

    def get_neighbors(self, order: int = 1, max_distance: Optional[float] = None) -> List[Neighbor]:
        raise NotImplementedError

    def get_reciprocal_vectors(self) -> np.ndarray:
        raise NotImplementedError


class HoneycombLattice(AbstractLattice):
    """
    Honeycomb lattice (placeholder - to be implemented).

    TODO: Implement honeycomb lattice
    - Two-atom basis (A, B sublattices)
    - Each atom has 3 nearest neighbors
    - Important for graphene, transition metal dichalcogenides
    - High-symmetry points: Γ, K, K', M
    """

    def __init__(self, lattice_constant: float = 1.0):
        raise NotImplementedError("HoneycombLattice not yet implemented. "
                                "Will be added in Phase 1B.")

    def get_primitive_vectors(self) -> np.ndarray:
        raise NotImplementedError

    def get_basis_positions(self) -> List[np.ndarray]:
        raise NotImplementedError

    def get_neighbors(self, order: int = 1, max_distance: Optional[float] = None) -> List[Neighbor]:
        raise NotImplementedError

    def get_reciprocal_vectors(self) -> np.ndarray:
        raise NotImplementedError


# Lattice registry for config-based construction
LATTICE_REGISTRY = {
    'triangular': TriangularLattice,
    'square': SquareLattice,
    'honeycomb': HoneycombLattice,
}


def create_lattice(lattice_type: str, **kwargs) -> AbstractLattice:
    """
    Factory function to create lattices from string names.

    Parameters
    ----------
    lattice_type : str
        Type of lattice ('triangular', 'square', 'honeycomb')
    **kwargs
        Additional arguments passed to lattice constructor
        (e.g., lattice_constant=1.5)

    Returns
    -------
    lattice : AbstractLattice
        Instantiated lattice object

    Examples
    --------
    >>> lattice = create_lattice('triangular', lattice_constant=1.5)
    >>> isinstance(lattice, TriangularLattice)
    True

    Raises
    ------
    ValueError
        If lattice_type is not recognized
    """
    if lattice_type not in LATTICE_REGISTRY:
        available = ', '.join(LATTICE_REGISTRY.keys())
        raise ValueError(f"Unknown lattice type '{lattice_type}'. "
                        f"Available types: {available}")

    lattice_class = LATTICE_REGISTRY[lattice_type]
    return lattice_class(**kwargs)
