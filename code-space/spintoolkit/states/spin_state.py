"""Classical spin configuration used by classical calculations and LSWT.

A :class:`SpinState` stores one unit direction per site of a magnetic
supercell. The supercell is an integer matrix ``M``: the rows of ``M @ A`` are
the magnetic lattice vectors. Directions are keyed by ``(site_id, cell)``,
where ``cell`` is the canonical representative of the cell modulo the
supercell, i.e. ``cell @ inv(M)`` lies in ``[0, 1)^2``. Directions are unit
vectors (angles are not stored: ``phi`` is undefined at the poles).

A state is not part of the model. It refers to the model by fingerprint and
must be commensurate with the calculation geometry; :func:`validate_spin_state`
checks both. Stationarity and stability depend on the model and the field and
are diagnosed by the calculation methods, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from spintoolkit.definitions.defaults import DEFAULT_TOLERANCE
from spintoolkit.system.geometry import CalculationGeometry, integer_matrix
from spintoolkit.system.model import SpinModel

SCHEMA_VERSION = 1

Cell = Tuple[int, int]
Key = Tuple[str, Cell]


class SpinStateError(ValueError):
    """Raised when a state is inconsistent; lists every violation found."""

    def __init__(self, violations: Sequence[str]):
        self.violations = list(violations)
        lines = "\n".join(f"  - {v}" for v in self.violations)
        super().__init__(f"{len(self.violations)} spin-state violation(s):\n{lines}")


def _adjugate(matrix: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    (a, b), (c, d) = matrix.tolist()
    return ((d, -b), (-c, a)), a * d - b * c


def reduce_cell(cell: Sequence[int], supercell: np.ndarray) -> Cell:
    """Canonical representative of ``cell`` modulo the supercell lattice.

    Uses exact integer arithmetic: ``cell @ inv(M) = cell @ adj(M) / det(M)``.
    """
    (adj0, adj1), det = _adjugate(supercell)
    c0, c1 = int(cell[0]), int(cell[1])
    v = (c0 * adj0[0] + c1 * adj1[0], c0 * adj0[1] + c1 * adj1[1])
    k = (v[0] // det, v[1] // det)
    m = supercell.tolist()
    return (c0 - k[0] * m[0][0] - k[1] * m[1][0],
            c1 - k[0] * m[0][1] - k[1] * m[1][1])


def supercell_cells(supercell: np.ndarray) -> Tuple[Cell, ...]:
    """All canonical cells of one magnetic supercell, sorted."""
    corners = np.array([[0, 0], supercell[0], supercell[1], supercell[0] + supercell[1]])
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    cells = {reduce_cell(c, supercell)
             for c in product(range(lo[0], hi[0] + 1), range(lo[1], hi[1] + 1))}
    return tuple(sorted(cells))


def _cell(value: Any) -> Tuple:
    entries = tuple(float(x) for x in value)
    if len(entries) == 2 and all(x.is_integer() for x in entries):
        return tuple(int(x) for x in entries)
    return entries


@dataclass(frozen=True)
class SpinState:
    """Unit spin directions on a magnetic supercell.

    Parameters
    ----------
    model_ref : str
        Fingerprint of the model this state belongs to
        (:meth:`SpinModel.fingerprint`).
    supercell : array_like, shape (2, 2)
        Integer matrix ``M`` with ``det M != 0``; rows of ``M @ A`` are the
        magnetic lattice vectors.
    directions : mapping of (site_id, cell) to array_like, shape (3,)
        One unit vector per site and canonical cell.
    provenance : mapping, optional
        Origin of the state, e.g. ``{"origin": "optimized",
        "selection_energy": "classical"}``.

    Raises
    ------
    SpinStateError
        If the supercell, the keys or the vectors are inconsistent.
    """

    model_ref: str
    supercell: np.ndarray
    directions: Mapping[Key, np.ndarray]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        try:
            supercell = integer_matrix(self.supercell, "supercell")
        except ValueError as error:
            raise SpinStateError([str(error)]) from None
        object.__setattr__(self, "supercell", supercell)
        directions = {}
        for (site, cell), vector in dict(self.directions).items():
            if np.iscomplexobj(vector):
                raise TypeError(f"direction of {(site, cell)} must be real")
            array = np.array(vector, dtype=float, copy=True)
            array.setflags(write=False)
            directions[(str(site), _cell(cell))] = array
        object.__setattr__(self, "directions", MappingProxyType(directions))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        violations = _intrinsic_violations(self)
        if violations:
            raise SpinStateError(violations)

    @classmethod
    def from_function(cls, model: SpinModel, supercell: Any,
                      direction: Callable[[str, Cell], Any],
                      provenance: Optional[Mapping[str, Any]] = None) -> "SpinState":
        """Build a state of ``model`` by evaluating ``direction(site_id, cell)``.

        ``direction`` must return unit vectors; they are not normalized here.
        """
        matrix = integer_matrix(supercell, "supercell")
        directions = {(site_id, cell): direction(site_id, cell)
                      for site_id in model.site_ids for cell in supercell_cells(matrix)}
        return cls(model.fingerprint(), matrix, directions, provenance or {})

    @property
    def num_cells(self) -> int:
        """Number of primitive cells in the magnetic supercell."""
        return abs(_adjugate(self.supercell)[1])

    @property
    def cells(self) -> Tuple[Cell, ...]:
        """Canonical cells of the supercell."""
        return supercell_cells(self.supercell)

    @property
    def site_ids(self) -> Tuple[str, ...]:
        """Site identifiers present in the state, sorted."""
        return tuple(sorted({site for site, _ in self.directions}))

    def reduce_cell(self, cell: Sequence[int]) -> Cell:
        """Canonical representative of an arbitrary cell."""
        return reduce_cell(cell, self.supercell)

    def direction(self, site_id: str, cell: Sequence[int] = (0, 0)) -> np.ndarray:
        """Unit direction of a site in any cell of the infinite lattice."""
        return self.directions[(site_id, self.reduce_cell(cell))]

    def magnetic_lattice(self, model: SpinModel) -> np.ndarray:
        """Cartesian magnetic lattice vectors (rows of ``M @ A``)."""
        return self.supercell @ model.lattice


def _intrinsic_violations(state: SpinState) -> List[str]:
    out = []
    cells = set(state.cells)
    per_site: Dict[str, set] = {}
    for (site, cell), vector in state.directions.items():
        if len(cell) != 2 or not all(isinstance(x, int) for x in cell):
            out.append(f"{site}: cell {cell} must be two exact integers")
            continue
        if cell not in cells:
            out.append(f"{site}: cell {cell} is not a canonical cell of the supercell "
                       f"(expected one of {sorted(cells)})")
        per_site.setdefault(site, set()).add(cell)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            out.append(f"{(site, cell)}: direction must be finite with shape (3,)")
        elif abs(np.linalg.norm(vector) - 1.0) > DEFAULT_TOLERANCE:
            out.append(f"{(site, cell)}: direction is not a unit vector "
                       f"(norm {np.linalg.norm(vector):.12g})")
    for site, present in per_site.items():
        missing = sorted(cells - present)
        if missing:
            out.append(f"{site}: no direction for cells {missing}")
    if not state.directions:
        out.append("state has no directions")
    return out


def validate_spin_state(state: SpinState, model: SpinModel,
                        geometry: Optional[CalculationGeometry] = None) -> List[str]:
    """Check a state against its model and, optionally, a calculation geometry.

    Parameters
    ----------
    state : SpinState
    model : SpinModel
    geometry : CalculationGeometry, optional
        If a finite torus, the supercell must tile the cluster.

    Returns
    -------
    notices : list of str
        Non-fatal findings, e.g. a period smaller than the declared supercell
        (folded bands make band-resolved Chern numbers ambiguous).

    Raises
    ------
    SpinStateError
        If the state belongs to another model, misses or adds sites, or does
        not fit the geometry.
    """
    violations = []
    if state.model_ref != model.fingerprint():
        violations.append("model_ref does not match the model fingerprint "
                          "(the state was built for a different model)")
    expected, present = set(model.site_ids), set(state.site_ids)
    if present - expected:
        violations.append(f"sites {sorted(present - expected)} are not in the model")
    if expected - present:
        violations.append(f"model sites {sorted(expected - present)} have no directions")
    if geometry is not None and geometry.cluster is not None:
        (adj0, adj1), det = _adjugate(state.supercell)
        for row in geometry.cluster.tolist():
            v = (row[0] * adj0[0] + row[1] * adj1[0], row[0] * adj0[1] + row[1] * adj1[1])
            if v[0] % det or v[1] % det:
                violations.append(f"cluster vector {row} is not a lattice vector of the "
                                  f"magnetic supercell {state.supercell.tolist()}")
    if violations:
        raise SpinStateError(violations)
    return _period_notices(state)


def _period_notices(state: SpinState) -> List[str]:
    cells = state.cells
    for shift in cells:
        if shift == (0, 0):
            continue
        if all(np.allclose(state.direction(site, (c[0] + shift[0], c[1] + shift[1])),
                           state.direction(site, c), atol=DEFAULT_TOLERANCE)
               for site in state.site_ids for c in cells):
            return [f"the state is periodic under cell shift {shift}, smaller than the "
                    f"declared supercell {state.supercell.tolist()}"]
    return []
