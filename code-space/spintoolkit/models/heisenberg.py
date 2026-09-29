"""Square- and triangular-lattice Heisenberg benchmark models.

Two kinds of builders live here:

- Hamiltonian builders (:func:`square_heisenberg`, :func:`triangular_heisenberg`)
  return a :class:`SpinModel`.
- Reference-configuration builders (:func:`polarized_state`, :func:`neel_state`,
  :func:`state_120`) return a :class:`SpinState` with an analytically known
  classical configuration. They do not solve the Hamiltonian, and a
  configuration is a ground state only under its stated conditions, e.g. the
  Neel state for ``J > 0`` at zero field, the polarized state above the
  saturation field.

All values are dimensionless in the unit of ``J``. With the default ``g = 1``
the field (``ExternalConditions.field``) equals the Zeeman energy ``h``
(saturation at ``h = 8JS`` on the square and ``9JS`` on the triangular lattice).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.model import Site, SpinModel, Term

SQUARE_LATTICE = np.array([[1.0, 0.0], [0.0, 1.0]])
TRIANGULAR_LATTICE = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def _g_tensor(g: Any) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    return g * np.eye(3) if g.ndim == 0 else g


def _heisenberg(model_id, lattice, offsets, J, S, g, lattice_constant):
    exchange = J * np.eye(3)
    terms = [Term.bilinear(("A", (0, 0)), ("A", offset), exchange, label="NN")
             for offset in offsets]
    if g is not None:
        terms.append(Term.zeeman("A", _g_tensor(g)))
    return SpinModel(
        lattice=lattice_constant * lattice,
        sites=[Site("A", (0.0, 0.0), S)],
        terms=terms,
        metadata={"model_id": model_id,
                  "parameters": {"J": J, "S": S, "g": None if g is None else np.asarray(g).tolist(),
                                 "lattice_constant": lattice_constant}},
    )


def square_heisenberg(J: float = 1.0, S: float = 0.5, g: Any = 1.0,
                      lattice_constant: float = 1.0) -> SpinModel:
    """Nearest-neighbour Heisenberg model ``J sum S_i . S_j`` on the square lattice.

    Parameters
    ----------
    J : float
        Exchange per bond; ``J > 0`` is antiferromagnetic.
    S : float
        Spin quantum number.
    g : float or (3, 3) array or None
        g-tensor of the zeeman term (scalar means isotropic); None omits the
        zeeman term, so the model does not couple to a field.
    lattice_constant : float
        Length of the primitive vectors.

    Notes
    -----
    One site per cell and two bond records per site, ``(1, 0)`` and ``(0, 1)``.
    """
    return _heisenberg("square_heisenberg", SQUARE_LATTICE, [(1, 0), (0, 1)],
                       J, S, g, lattice_constant)


def triangular_heisenberg(J: float = 1.0, S: float = 0.5, g: Any = 1.0,
                          lattice_constant: float = 1.0) -> SpinModel:
    """Nearest-neighbour Heisenberg model on the triangular lattice.

    Primitive vectors ``a1 = (1, 0)`` and ``a2 = (1/2, sqrt(3)/2)``; three bond
    records per site with offsets ``(1, 0)``, ``(0, 1)`` and ``(-1, 1)``.
    Parameters as in :func:`square_heisenberg`.
    """
    return _heisenberg("triangular_heisenberg", TRIANGULAR_LATTICE,
                       [(1, 0), (0, 1), (-1, 1)], J, S, g, lattice_constant)


def _unit(vector: Sequence[float]) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("direction must be non-zero")
    return vector / norm


def _require_one_site(model: SpinModel, name: str) -> str:
    if model.num_sites != 1:
        raise ValueError(f"{name} is defined for one-site primitive cells, "
                         f"got {model.num_sites} sites")
    return model.site_ids[0]


def polarized_state(model: SpinModel, direction: Sequence[float] = (0, 0, 1)) -> SpinState:
    """All spins along ``direction`` (normalized here) on the primitive cell."""
    n = _unit(direction)
    return SpinState.from_function(model, np.eye(2, dtype=int), lambda site, cell: n,
                                   {"origin": "analytic", "configuration": "polarized"})


def neel_state(model: SpinModel, direction: Sequence[float] = (0, 0, 1)) -> SpinState:
    """Checkerboard configuration with spins ``+n`` and ``-n``.

    The sign is ``(-1)**(n1 + n2)`` on a one-site cell with the supercell
    ``[[1, 1], [1, -1]]``. On the square lattice this is the Neel state, the
    classical ground state for ``J > 0`` at zero field.
    """
    _require_one_site(model, "neel_state")
    n = _unit(direction)
    return SpinState.from_function(
        model, [[1, 1], [1, -1]],
        lambda site, cell: n if (cell[0] + cell[1]) % 2 == 0 else -n,
        {"origin": "analytic", "configuration": "neel"})


def state_120(model: SpinModel,
              plane: Sequence[Sequence[float]] = ((1, 0, 0), (0, 1, 0))) -> SpinState:
    """Coplanar three-sublattice configuration with spins 120 degrees apart.

    Sublattice ``k = (n1 - n2) mod 3`` on a one-site cell with the
    ``sqrt(3) x sqrt(3)`` supercell ``[[1, 1], [-1, 2]]``; spins point along
    ``cos(2 pi k / 3) e1 + sin(2 pi k / 3) e2`` for the orthonormal ``plane``
    vectors ``(e1, e2)``. With the primitive vectors of
    :func:`triangular_heisenberg`, nearest neighbours lie on different
    sublattices; this is the classical ground state for ``J > 0`` at zero field.
    """
    _require_one_site(model, "state_120")
    e1, e2 = (np.asarray(v, dtype=float) for v in plane)
    if not (np.isclose(e1 @ e1, 1) and np.isclose(e2 @ e2, 1) and np.isclose(e1 @ e2, 0)):
        raise ValueError("plane must contain two orthonormal vectors")

    def direction(site, cell):
        angle = 2 * np.pi * ((cell[0] - cell[1]) % 3) / 3
        return np.cos(angle) * e1 + np.sin(angle) * e2

    return SpinState.from_function(model, [[1, 1], [-1, 2]], direction,
                                   {"origin": "analytic", "configuration": "120"})
