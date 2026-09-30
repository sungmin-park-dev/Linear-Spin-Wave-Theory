"""Honeycomb-lattice benchmark models for topological magnon quantities (stage 5).

- :func:`honeycomb_ferromagnet`: nearest-neighbour ferromagnet with a
  next-nearest-neighbour Dzyaloshinskii-Moriya interaction along z. The
  polarized state along +z conserves the magnon number, and the magnon bands
  are the Haldane model: Chern numbers +-1 for ``D != 0`` (Owerre, J. Phys.:
  Condens. Matter 28, 386001 (2016); Kim et al., PRL 117, 227201 (2016)).
- :func:`kitaev_honeycomb`: Kitaev model; the state polarized along [111] by a
  strong field has anomalous (pairing) terms and magnon bands with Chern
  numbers +-1 (McClarty et al., PRB 98, 060404(R) (2018)).

Primitive vectors ``a1 = (1, 0)``, ``a2 = (1/2, sqrt(3)/2)`` (next-nearest-
neighbour distance 1); site A at fractional ``(0, 0)``, B at ``(1/3, 1/3)``.
The nearest neighbours of A(0, 0) are B at cell offsets ``(0, 0)``,
``(-1, 0)`` and ``(0, -1)``. All values are dimensionless (D20).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spintoolkit.models.heisenberg import TRIANGULAR_LATTICE, _g_tensor
from spintoolkit.system.model import Site, SpinModel, Term

#: Cell offsets of the B neighbours of A(0, 0).
NEAREST_NEIGHBOUR_OFFSETS = ((0, 0), (-1, 0), (0, -1))
#: Next-nearest-neighbour offsets, 120 degrees apart and summing to zero.
NEXT_NEAREST_OFFSETS = ((1, 0), (-1, 1), (0, -1))


def _sites(S):
    return [Site("A", (0.0, 0.0), S), Site("B", (1 / 3, 1 / 3), S)]


def honeycomb_ferromagnet(J: float = 1.0, D: float = 0.1, S: float = 0.5,
                          g: Any = 1.0) -> SpinModel:
    """``-J sum_<ij> S_i . S_j + D sum_<<ij>> nu_ij z . (S_i x S_j)``.

    Parameters
    ----------
    J : float
        Nearest-neighbour exchange; ``J > 0`` is ferromagnetic.
    D : float
        Next-nearest-neighbour DM coupling. ``nu_ij = +1`` on A for the
        offsets :data:`NEXT_NEAREST_OFFSETS` and ``-1`` on B for the same
        offsets (opposite chirality on the two sublattices, as in the Haldane
        model).
    S : float
    g : float or (3, 3) array or None
        g-tensor of the zeeman term; None omits it.
    """
    heisenberg = -J * np.eye(3)
    dm = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    terms = [Term.bilinear(("A", (0, 0)), ("B", offset), heisenberg, label="NN")
             for offset in NEAREST_NEIGHBOUR_OFFSETS]
    for site, nu in (("A", 1.0), ("B", -1.0)):
        terms += [Term.bilinear((site, (0, 0)), (site, offset), nu * D * dm, label="DM")
                  for offset in NEXT_NEAREST_OFFSETS]
    if g is not None:
        terms += [Term.zeeman(site, _g_tensor(g)) for site in ("A", "B")]
    return SpinModel(TRIANGULAR_LATTICE, _sites(S), terms,
                     {"model_id": "honeycomb_ferromagnet_dm",
                      "parameters": {"J": J, "D": D, "S": S,
                                     "g": None if g is None else np.asarray(g).tolist()}})


def kitaev_honeycomb(K: float = -1.0, S: float = 0.5, g: Any = 1.0) -> SpinModel:
    """``K sum_<ij>_gamma S_i^gamma S_j^gamma``.

    The bond from A(0, 0) to B at offset ``(0, 0)``, ``(-1, 0)`` and
    ``(0, -1)`` is of type x, y and z; every site has one bond of each type.

    Parameters
    ----------
    K : float
        Kitaev coupling (``K < 0`` ferromagnetic).
    S : float
    g : float or (3, 3) array or None
    """
    terms = []
    for gamma, offset in enumerate(NEAREST_NEIGHBOUR_OFFSETS):
        coupling = np.zeros((3, 3))
        coupling[gamma, gamma] = K
        terms.append(Term.bilinear(("A", (0, 0)), ("B", offset), coupling,
                                   label="xyz"[gamma]))
    if g is not None:
        terms += [Term.zeeman(site, _g_tensor(g)) for site in ("A", "B")]
    return SpinModel(TRIANGULAR_LATTICE, _sites(S), terms,
                     {"model_id": "kitaev_honeycomb",
                      "parameters": {"K": K, "S": S,
                                     "g": None if g is None else np.asarray(g).tolist()}})
