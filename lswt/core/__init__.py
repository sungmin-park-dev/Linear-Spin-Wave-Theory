"""Core domain models for LSWT calculations."""

from lswt.core.spin_system import SpinSystem
# Backward-compatible aliases
from lswt.core.spin_system import SpinSite, Coupling
from lswt.core import exchange
from lswt.core.brillouin_zone import BrillouinZone
from lswt.core.diagonalization import Diagonalizer

__all__ = [
    'SpinSystem', 'SpinSite', 'Coupling',
    'exchange', 'BrillouinZone', 'Diagonalizer',
]
