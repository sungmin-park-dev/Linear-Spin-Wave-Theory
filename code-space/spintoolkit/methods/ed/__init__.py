"""Exact diagonalization on finite tori (minimal verification tool, D10, D23).

The Hamiltonian is built for any model without assuming a symmetry; the
U(1) magnetization (n-magnon) sectors and translation momenta are optional
reductions.
"""

from .solver import EDBlock, EDResult, EDSector, SectorError, solve_ed

__all__ = ["EDBlock", "EDResult", "EDSector", "SectorError", "solve_ed"]
