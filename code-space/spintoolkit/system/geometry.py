"""Calculation geometry: size, shape and boundary of the computed system.

This is the computational lattice condition, not the crystal lattice itself
(which lives in :mod:`spintoolkit.system.lattice` and in
:attr:`SpinModel.lattice`). The 1st scope supports the thermodynamic limit of
the periodic system and finite periodic tori; cylinders and open clusters are
added with the ED and tensor-network solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

THERMODYNAMIC_LIMIT = "thermodynamic_limit"
FINITE_TORUS = "finite_torus"


def integer_matrix(value: Any, name: str) -> np.ndarray:
    """Return a read-only (2, 2) integer copy of an exactly-integer matrix.

    Raises
    ------
    ValueError
        If the matrix is not (2, 2), has non-integer entries, or is singular.
    """
    array = np.asarray(value, dtype=float)
    if array.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2, 2), got {array.shape}")
    if not np.all(np.isfinite(array)) or not np.all(array == np.round(array)):
        raise ValueError(f"{name} must contain exact integers, got {array.tolist()}")
    matrix = array.astype(int)
    if round(np.linalg.det(matrix)) == 0:
        raise ValueError(f"{name} is singular (det = 0)")
    matrix.setflags(write=False)
    return matrix


@dataclass(frozen=True)
class CalculationGeometry:
    """Size, shape and boundary condition of the computed system.

    Build instances with :meth:`thermodynamic_limit` or :meth:`finite_torus`.

    Parameters
    ----------
    kind : {"thermodynamic_limit", "finite_torus"}
    cluster : (2, 2) integer array, optional
        For a torus, rows are the cluster vectors in units of the primitive
        lattice vectors: the torus contains ``abs(det cluster)`` primitive cells.
    """

    kind: str
    cluster: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.kind == THERMODYNAMIC_LIMIT:
            if self.cluster is not None:
                raise ValueError("the thermodynamic limit takes no cluster")
        elif self.kind == FINITE_TORUS:
            object.__setattr__(self, "cluster", integer_matrix(self.cluster, "cluster"))
        else:
            raise ValueError(f"unknown geometry kind {self.kind!r}")

    @classmethod
    def thermodynamic_limit(cls) -> "CalculationGeometry":
        """Infinite periodic system (k-point grids are numerical settings)."""
        return cls(THERMODYNAMIC_LIMIT)

    @classmethod
    def finite_torus(cls, cluster: Any) -> "CalculationGeometry":
        """Finite periodic torus spanned by the rows of ``cluster``."""
        return cls(FINITE_TORUS, cluster)

    @property
    def num_cells(self) -> Optional[int]:
        """Number of primitive cells, or None in the thermodynamic limit."""
        if self.cluster is None:
            return None
        return abs(int(round(np.linalg.det(self.cluster))))
