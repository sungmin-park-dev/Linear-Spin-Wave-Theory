"""Abstract base class for spin system solvers.

All solvers (LSWT, real-space BdG, ED, DMRG, ...) share a common interface:

    solver = SomeSolver(system, **options)
    result = solver.solve()

This module defines that contract so that different methods can be compared
on the same SpinSystem with minimal code changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from lswt.core.spin_system import SpinSystem


@dataclass
class SolverResult:
    """Common container for solver output.

    Every solver must populate ``ground_state_energy`` and ``method``.
    Solver-specific data lives in ``data``, keyed by descriptive names.

    Parameters
    ----------
    ground_state_energy : float
        Ground state energy per site.
    eigenvalues : np.ndarray or None
        Energy spectrum. Shape depends on solver:
        - LSWT: (num_k, num_bands) magnon bands
        - Real-space BdG: (num_modes,) finite-size spectrum
        - ED: (dim_hilbert,) full spectrum
    method : str
        Solver identifier, e.g. 'LSWT', 'RealSpaceBdG', 'ED', 'DMRG'.
    spin_config : np.ndarray or None
        Optimized spin angles [theta_0, phi_0, ...], if the solver
        performed optimization. None if angles were fixed.
    data : dict
        Solver-specific output. Examples:
        - LSWT: {'k_data': ..., 'bz_data': ..., 'berry_curvature': ...}
        - ED: {'wavefunctions': ..., 'partition_function': ...}
        - DMRG: {'bond_dimensions': ..., 'entanglement': ...}
    """
    ground_state_energy: float
    method: str
    eigenvalues: Optional[np.ndarray] = None
    spin_config: Optional[np.ndarray] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        eig_shape = self.eigenvalues.shape if self.eigenvalues is not None else None
        return (
            f"SolverResult(method='{self.method}', "
            f"E0={self.ground_state_energy:.6f}, "
            f"eigenvalues.shape={eig_shape})"
        )


class AbstractSolver(ABC):
    """Base class for all spin system solvers.

    Subclasses must implement :meth:`solve`, which returns a
    :class:`SolverResult`.

    Parameters
    ----------
    system : SpinSystem
        The spin system to solve.

    Examples
    --------
    >>> system = SpinSystem(sites, couplings, lattice_vectors)
    >>> solver = LSWTSolver(system)
    >>> result = solver.solve()
    >>> print(result.ground_state_energy)
    """

    def __init__(self, system: SpinSystem):
        self.system = system

    @abstractmethod
    def solve(self, **kwargs) -> SolverResult:
        """Run the solver and return results.

        Returns
        -------
        result : SolverResult
            Solver output with at minimum ground_state_energy and method.
        """
        ...

    @property
    def num_sites(self) -> int:
        """Number of magnetic sites in the unit cell."""
        return self.system.num_sites

    def __repr__(self):
        return f"{self.__class__.__name__}(system={self.system})"
