"""
Lattice geometry module.

This module provides abstract and concrete implementations of crystallographic
lattices. Lattices represent ONLY geometric structure - no magnetic information.

Available lattices:
- TriangularLattice: Single-atom hexagonal lattice
- SquareLattice: (TODO)
- HoneycombLattice: (TODO)
"""

from .base import AbstractLattice, Neighbor
from .presets import (
    TriangularLattice,
    SquareLattice,
    HoneycombLattice,
    LATTICE_REGISTRY,
    create_lattice
)

__all__ = [
    'AbstractLattice',
    'Neighbor',
    'TriangularLattice',
    'SquareLattice',
    'HoneycombLattice',
    'LATTICE_REGISTRY',
    'create_lattice',
]
