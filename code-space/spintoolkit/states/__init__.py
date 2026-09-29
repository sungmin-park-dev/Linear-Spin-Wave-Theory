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
from .spin_state import SpinState, SpinStateError, validate_spin_state

__all__ = [
    'AbstractMagneticStructure',
    'CommensurateStructure',
    'IncommensurateStructure',
    'SpinState',
    'SpinStateError',
    'validate_spin_state',
]
