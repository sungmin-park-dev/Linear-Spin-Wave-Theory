"""
Core domain models for LSWT package.

This module contains the fundamental abstractions:
- Lattice: crystallographic geometry
- MagneticStructure: spin configurations (commensurate/incommensurate)
- SpinSystem: complete physical system

These are the building blocks used by solvers, physics modules, and analysis tools.
"""

from .lattice import (
    AbstractLattice,
    Neighbor,
    TriangularLattice,
    SquareLattice,
    HoneycombLattice,
    LATTICE_REGISTRY,
    create_lattice
)

from .magnetic_structure import (
    AbstractMagneticStructure,
    CommensurateStructure,
    IncommensurateStructure
)

from .spin_system import SpinSystem

__all__ = [
    # Lattice
    'AbstractLattice',
    'Neighbor',
    'TriangularLattice',
    'SquareLattice',
    'HoneycombLattice',
    'LATTICE_REGISTRY',
    'create_lattice',

    # Magnetic Structure
    'AbstractMagneticStructure',
    'CommensurateStructure',
    'IncommensurateStructure',

    # Spin System
    'SpinSystem',
]
