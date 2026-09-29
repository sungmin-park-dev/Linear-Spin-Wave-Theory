"""
Magnetic structure module.

This module provides abstractions for magnetic ordering patterns, separate from
crystallographic lattice geometry.

Available structures:
- CommensurateStructure: Finite supercell with discrete spin angles (implemented)
- IncommensurateStructure: Infinite spiral/helix (stub - Phase 5+)
"""

from .base import AbstractMagneticStructure
from .commensurate import CommensurateStructure
from .incommensurate import IncommensurateStructure

__all__ = [
    'AbstractMagneticStructure',
    'CommensurateStructure',
    'IncommensurateStructure',
]
