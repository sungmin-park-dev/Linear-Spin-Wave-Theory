"""Honeycomb J1-J3 XXZ model with the bond-dependent third-neighbour term (BCAO).

The "minimal model" of BaCo2(AsO4)2 (Maksimov et al., PRB 106, 165131
(2022); field study Maksimov, arXiv:2308.10672, Eq. (1)):

    H = sum_<ij>_1 J1 (S^x S^x + S^y S^y + Delta1 S^z S^z)
      + sum_<ij>_3 [ J3 (S^x S^x + S^y S^y + Delta3 S^z S^z)
                     - Jzpm3 ((S^x_i S^z_j + S^z_i S^x_j) cos phi_a
                              + (S^y_i S^z_j + S^z_i S^y_j) sin phi_a) ]
      - sum_i mu_B B^T g S_i,

with x, y in the honeycomb plane and z normal to it. With ``Jzpm3 = 0`` it
is the J1-J3 XXZ model of Chernyshev, arXiv:2509.08877 Eq. (7).

Bond phases. ``(cos phi_a, sin phi_a)`` is the in-plane unit vector
perpendicular to bond ``a``, rotating with the bond under C3. This follows
from the cubic-to-crystallographic mapping (Maksimov Eq. (2)): with the
crystallographic z along cubic [111], the Z bond lies along [1,-1,0], and the
in-plane projection of the cubic z axis, which carries the S^z-S^perp cross
term of the Kitaev and Gamma couplings, is (-1,-1,2), perpendicular to that
bond. ``phi_a = 0`` is assigned to the bond along -y (offset ``(0, -1)``), so
the x axis, the field direction of Maksimov's Fig. 3, is perpendicular to a
bond. The overall sign of the three vectors is a convention: reversing it is
``Jzpm3 -> -Jzpm3``, which the C2 rotation about z (sites and spins) maps
back to the same model with the in-plane field reversed, and that rotation
is itself a symmetry, so energies do not depend on it. (Maksimov's Fig. 2(a)
draws the bond labels; it could not be read here, so the perpendicular
assignment rests on the derivation above.)

Lattice as in :mod:`~spintoolkit.models.honeycomb`: ``a1 = (1, 0)``,
``a2 = (1/2, sqrt(3)/2)``, A at fractional ``(0, 0)``, B at ``(1/3, 1/3)``;
the nearest-neighbour distance is ``1/sqrt(3)``. Third neighbours are
opposite across a hexagon, ``delta^(3)_a = -2 delta_a``, of the same bond type.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from spintoolkit.models.heisenberg import TRIANGULAR_LATTICE, _g_tensor
from spintoolkit.models.honeycomb import NEAREST_NEIGHBOUR_OFFSETS, _sites
from spintoolkit.system.model import SpinModel, Term

#: Cell offsets of the B third neighbours of A(0, 0), in the order of
#: :data:`~spintoolkit.models.honeycomb.NEAREST_NEIGHBOUR_OFFSETS`.
THIRD_NEIGHBOUR_OFFSETS = ((-1, -1), (1, -1), (-1, 1))
#: Bond phases phi_a for the bonds of NEAREST_NEIGHBOUR_OFFSETS (directions 30, 150, -90 deg).
BOND_PHASES = (2 * np.pi / 3, -2 * np.pi / 3, 0.0)


def _xxz(J: float, delta: float) -> np.ndarray:
    return J * np.diag([1.0, 1.0, delta])


def bcao_minimal(J1: float = -1.0, J3: float = 0.34, Delta1: float = 0.0,
                 Delta3: float = 0.0, Jzpm3: float = 0.0, S: float = 0.5,
                 g: Any = 1.0) -> SpinModel:
    """Honeycomb J1-J3 XXZ model with the bond-dependent ``Jzpm3`` term.

    Parameters
    ----------
    J1 : float
        Nearest-neighbour exchange (``J1 < 0`` ferromagnetic; BCAO ``J1 = -1``
        sets the energy unit).
    J3 : float
        Third-neighbour exchange (antiferromagnetic for ``J3 > 0``).
    Delta1, Delta3 : float
        XXZ anisotropies of the two couplings.
    Jzpm3 : float
        Bond-dependent third-neighbour coupling ``J^(3)_{z+-}``, with the sign of
        Maksimov Eq. (1) (it enters with a minus sign).
    S : float
    g : float or (3, 3) array or None
        g-tensor of the Zeeman term; None omits it. The field is set by
        :class:`~spintoolkit.system.conditions.ExternalConditions`.

    Returns
    -------
    SpinModel
    """
    terms = []
    for (offset, offset3, phi) in zip(NEAREST_NEIGHBOUR_OFFSETS, THIRD_NEIGHBOUR_OFFSETS,
                                      BOND_PHASES):
        terms.append(Term.bilinear(("A", (0, 0)), ("B", offset), _xxz(J1, Delta1), label="J1"))
        coupling = _xxz(J3, Delta3)
        n = np.array([np.cos(phi), np.sin(phi), 0.0])
        cross = np.outer(n, [0.0, 0.0, 1.0])
        coupling = coupling - Jzpm3 * (cross + cross.T)
        terms.append(Term.bilinear(("A", (0, 0)), ("B", offset3), coupling, label="J3"))
    if g is not None:
        terms += [Term.zeeman(site, _g_tensor(g)) for site in ("A", "B")]
    return SpinModel(TRIANGULAR_LATTICE, _sites(S), terms,
                     {"model_id": "bcao_minimal",
                      "parameters": {"J1": J1, "J3": J3, "Delta1": Delta1, "Delta3": Delta3,
                                     "Jzpm3": Jzpm3, "S": S,
                                     "g": None if g is None else np.asarray(g).tolist()}})
