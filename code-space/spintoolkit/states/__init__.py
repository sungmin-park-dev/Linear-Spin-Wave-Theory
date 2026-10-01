"""
Magnetic structure module.

This module provides abstractions for magnetic ordering patterns, separate from
crystallographic lattice geometry.

Available structures:
- SpinState: spin directions on an integer-matrix magnetic supercell (the
  common state of all methods; D16)
- IncommensurateStructure: single-Q spiral (planar or conical, any wave vector);
  LSWT in the rotating frame (D34)

The diagonal-supercell ``CommensurateStructure`` was removed (D30); SpinState
covers every commensurate cell, including non-diagonal ones such as sqrt(3) x sqrt(3).
"""

from .base import AbstractMagneticStructure
from .incommensurate import IncommensurateStructure
from .spin_state import SpinState, SpinStateError, validate_spin_state

__all__ = [
    'AbstractMagneticStructure',
    'IncommensurateStructure',
    'SpinState',
    'SpinStateError',
    'validate_spin_state',
]
