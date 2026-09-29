"""Expansion of a model's terms on a finite periodic torus.

A :class:`SpinModel` records every bond once, from cell ``(0, 0)``. A finite
torus with integer cluster matrix ``L`` (rows of ``L @ A`` are the torus
periods) has one copy of every site per canonical cell; translating each term
to every cell and folding the endpoints back onto the torus gives the operator
list that exact diagonalization needs.

Rules (D23), the same for every method:

- Each (term, cell) pair is kept as its own bond. On a small torus two cells
  can produce the same pair of sites (e.g. the +x and -x neighbours of a 2 x 2
  square torus); those bonds are all kept, so the pair carries the sum of the
  couplings. This is the Hamiltonian of the periodic cluster, and the
  thermodynamic-limit formulas evaluated at the torus momenta reproduce it.
- A bond whose two endpoints fold onto the same site is a self-interaction
  ``S_i . J . S_i``, an on-site term that the ``bilinear`` kind does not
  describe consistently for classical spins, spin operators and LSWT. Such a
  torus is rejected by every method until an on-site quadratic kind exists.

Momenta allowed by the torus are ``k = 2 pi inv(A) q`` with ``L q`` integer;
they are returned both as fractional coordinates ``q`` (in the reciprocal
basis of the primitive lattice) and in Cartesian form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from spintoolkit.states.spin_state import reduce_cell, supercell_cells
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import BILINEAR, ZEEMAN, SpinModel

#: Term kinds that can be expanded on a torus.
SUPPORTED_KINDS = (BILINEAR, ZEEMAN)

Key = Tuple[str, Tuple[int, int]]


class ClusterError(ValueError):
    """The model cannot be expanded on the requested torus."""


@dataclass(frozen=True)
class TorusCluster:
    """A model expanded on a finite periodic torus.

    Attributes
    ----------
    model_ref : str
        Fingerprint of the model.
    cluster : (2, 2) int array
        Cluster matrix ``L``.
    keys : tuple of (site_id, cell)
        Torus sites, model site order within each canonical cell of ``L``.
    positions : (n, 2) array
        Cartesian positions of the torus sites (cell origin plus site offset).
    spins : (n,) array
        Spin length of each site.
    source, target : (m,) int arrays
        Site indices of each bond; ``source != target``.
    exchange : (m, 3, 3) array
        Coupling matrix of each bond, ``S_source . J . S_target``.
    bond_terms : (m,) int array
        Index of the bilinear term (in ``model.terms_of_kind("bilinear")``)
        each bond comes from.
    g_tensors : (n, 3, 3) array
        Sum of the zeeman ``g`` tensors of each site (zero if none).
    lattice : (2, 2) array
        Primitive lattice ``A`` of the model.
    """

    model_ref: str
    cluster: np.ndarray
    keys: Tuple[Key, ...]
    positions: np.ndarray
    spins: np.ndarray
    source: np.ndarray
    target: np.ndarray
    exchange: np.ndarray
    bond_terms: np.ndarray
    g_tensors: np.ndarray
    lattice: np.ndarray

    @property
    def num_sites(self) -> int:
        return len(self.keys)

    @property
    def num_cells(self) -> int:
        return int(round(abs(np.linalg.det(self.cluster))))

    def index(self) -> Dict[Key, int]:
        """Map ``(site_id, canonical cell)`` to the site index."""
        return {key: i for i, key in enumerate(self.keys)}

    def fields(self, conditions: Optional[ExternalConditions] = None) -> np.ndarray:
        """Zeeman field ``h_i = g_i^T b`` of every site, shape (n, 3)."""
        field = (conditions or ExternalConditions()).field
        return np.einsum("iab,a->ib", self.g_tensors, field)

    def translation(self, shift) -> np.ndarray:
        """Permutation ``p`` with ``p[i]`` the image of site ``i`` under a cell shift."""
        index = self.index()
        return np.array([index[(site, reduce_cell((cell[0] + shift[0], cell[1] + shift[1]),
                                                  self.cluster))]
                         for site, cell in self.keys], dtype=int)


def expand_on_torus(model: SpinModel, geometry: CalculationGeometry) -> TorusCluster:
    """Expand the terms of ``model`` on the finite torus of ``geometry``.

    Raises
    ------
    ClusterError
        If ``geometry`` is not a finite torus, if the model has term kinds that
        cannot be expanded, or if a bond folds onto a single site.
    """
    if geometry.kind != "finite_torus":
        raise ClusterError("expansion needs a finite_torus geometry")
    unsupported = sorted({t.kind for t in model.terms} - set(SUPPORTED_KINDS))
    if unsupported:
        raise ClusterError(f"term kinds {unsupported} cannot be expanded on a torus")
    L = geometry.cluster
    cells = supercell_cells(L)
    keys = tuple((site.id, cell) for cell in cells for site in model.sites)
    index = {key: i for i, key in enumerate(keys)}
    positions = np.array([model.cartesian_position(site, cell) for site, cell in keys])
    spins = np.array([model.site(site).spin for site, _ in keys])

    source, target, exchange, bond_terms = [], [], [], []
    folded: List[str] = []
    for t, term in enumerate(model.terms_of_kind(BILINEAR)):
        (a, n1), (b, n2) = term.participants
        for cell in cells:
            i = index[(a, reduce_cell((cell[0] + n1[0], cell[1] + n1[1]), L))]
            j = index[(b, reduce_cell((cell[0] + n2[0], cell[1] + n2[1]), L))]
            if i == j:
                folded.append(f"{term.label or t}: {a}{tuple(n1)} -> {b}{tuple(n2)}")
                break
            source.append(i)
            target.append(j)
            exchange.append(term.coefficient)
            bond_terms.append(t)
    if folded:
        raise ClusterError(
            "bonds fold onto a single site on this torus (self-interaction, D23): "
            + "; ".join(folded) + ". Use a torus larger than every bond.")

    g_tensors = np.zeros((len(keys), 3, 3))
    for term in model.terms_of_kind(ZEEMAN):
        site = term.participants[0][0]
        for cell in cells:
            g_tensors[index[(site, cell)]] += term.coefficient
    return TorusCluster(model.fingerprint(), L, keys, positions, spins,
                        np.array(source, dtype=int), np.array(target, dtype=int),
                        np.array(exchange, dtype=float).reshape(-1, 3, 3),
                        np.array(bond_terms, dtype=int), g_tensors,
                        np.asarray(model.lattice, dtype=float))


def allowed_momenta(model: SpinModel, geometry: CalculationGeometry
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Momenta of the finite torus.

    Returns
    -------
    fractional : (N_c, 2) array
        ``q`` in ``[0, 1)`` with ``L q`` integer; ``k . A_j = 2 pi q_j``.
    cartesian : (N_c, 2) array
        ``k = 2 pi inv(A) q``.
    """
    if geometry.kind != "finite_torus":
        raise ClusterError("allowed momenta need a finite_torus geometry")
    L = geometry.cluster
    inverse = np.linalg.inv(L)
    fractional = []
    for m in supercell_cells(L.T):
        q = np.mod(inverse @ np.asarray(m, dtype=float), 1.0)
        q[np.isclose(q, 1.0, rtol=0, atol=1e-12)] = 0.0
        fractional.append(q)
    fractional = np.array(fractional)
    cartesian = 2 * np.pi * fractional @ np.linalg.inv(np.asarray(model.lattice, float)).T
    return fractional, cartesian
