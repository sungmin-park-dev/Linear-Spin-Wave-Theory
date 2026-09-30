"""Standard benchmark models and their analytic reference configurations."""

from .heisenberg import (
    neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg,
)
from .honeycomb import honeycomb_ferromagnet, kitaev_honeycomb

__all__ = ['square_heisenberg', 'triangular_heisenberg', 'honeycomb_ferromagnet',
           'kitaev_honeycomb', 'polarized_state', 'neel_state', 'state_120']
