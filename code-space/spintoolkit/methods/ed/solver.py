"""Exact diagonalization of a model on a finite torus.

The Hamiltonian is built for any model without assuming a symmetry. Two
optional reductions split it into blocks:

- ``axis`` and ``magnon_number``: the total magnetization along ``axis`` is
  conserved when every rotation about ``axis`` leaves all terms invariant
  (U(1)); the block with ``n`` deviations from the state polarized along
  ``+axis`` is the ``n``-magnon sector. The spin frame is rotated so that
  ``axis`` becomes the quantization axis; this global rotation does not change
  the spectrum. Terms that would leave the sector are reported, and the
  calculation stops if they are not negligible.
- ``momenta``: translations of the torus by primitive cells. A block at
  momentum ``k`` holds the states with ``T_R |psi> = exp(-i k . R) |psi>``, the
  sign for which a one-magnon state ``a_k^dagger |0>`` of the toolkit Fourier
  convention (D13) carries momentum ``+k``.

Energies are total energies of the torus in the energy unit E0 (D23);
:meth:`EDResult.per_site` and :meth:`EDResult.excitations` derive the other
normalizations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from spintoolkit.definitions.defaults import (
    ED_DENSE_LIMIT, ED_LANCZOS_MIN_DIMENSION, ED_SYMMETRY_TOLERANCE)
from spintoolkit.methods.ed.basis import SectorBasis, TranslationOrbits, cluster_translations
from spintoolkit.methods.ed.hamiltonian import frame_rotation, matrix_elements, operator_terms
from spintoolkit.methods.result import ResultHeader, to_jsonable
from spintoolkit.system.cluster import TorusCluster, allowed_momenta, expand_on_torus
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import SpinModel


class SectorError(ValueError):
    """The requested sector is not conserved by the Hamiltonian."""


@dataclass(frozen=True)
class EDSector:
    """Blocks to diagonalize.

    Parameters
    ----------
    axis : (3,) array_like, optional
        Quantization axis. Required for ``magnon_number``.
    magnon_number : int or "all", optional
        Deviations from the state polarized along ``+axis``; "all" solves every
        sector. None diagonalizes the whole space without the U(1) reduction.
    momenta : "all" or sequence of int, optional
        Indices into :func:`~spintoolkit.system.cluster.allowed_momenta`; None
        does not use translations.
    """

    axis: Optional[Tuple[float, float, float]] = None
    magnon_number: Optional[Union[int, str]] = None
    momenta: Optional[Union[str, Tuple[int, ...]]] = None

    def __post_init__(self):
        if self.axis is not None:
            axis = np.asarray(self.axis, dtype=float)
            if axis.shape != (3,) or np.linalg.norm(axis) == 0:
                raise ValueError("axis must be a nonzero 3-vector")
            object.__setattr__(self, "axis", tuple(float(x) for x in axis / np.linalg.norm(axis)))
        if self.magnon_number is not None and self.axis is None:
            raise ValueError("magnon_number needs an axis")
        if self.magnon_number is not None and self.magnon_number != "all":
            if int(self.magnon_number) != self.magnon_number or self.magnon_number < 0:
                raise ValueError("magnon_number must be a non-negative integer or 'all'")
        if self.momenta is not None and self.momenta != "all":
            object.__setattr__(self, "momenta", tuple(int(i) for i in self.momenta))


@dataclass(frozen=True)
class EDBlock:
    """Eigenvalues of one block.

    Attributes
    ----------
    magnon_number : int or None
    momentum_index : int or None
    q : (2,) array or None
        Fractional momentum (``k . A_j = 2 pi q_j``).
    k : (2,) array or None
        Cartesian momentum.
    dimension : int
    energies : (m,) array
        Total energies of the torus, ascending.
    vectors : (dimension, m) array or None
        Eigenvectors in the block basis, when requested.
    solver : str
        "dense" or "lanczos".
    residual : float
        Largest ``|H v - E v|`` of the returned pairs.
    """

    magnon_number: Optional[int]
    momentum_index: Optional[int]
    q: Optional[np.ndarray]
    k: Optional[np.ndarray]
    dimension: int
    energies: np.ndarray
    vectors: Optional[np.ndarray]
    solver: str
    residual: float


@dataclass(frozen=True)
class EDResult:
    """Exact-diagonalization result body (the common header follows in stage 4).

    Attributes
    ----------
    method : str
        "ed".
    model_ref : str
    cluster : (2, 2) int array
    conditions : ExternalConditions
    sector : EDSector
    num_sites : int
    blocks : tuple of EDBlock
    reference_energy : float or None
        Energy of the product state polarized along ``+axis`` (an eigenstate
        when the magnetization is conserved); None without an axis.
    diagnostics : dict
    header : ResultHeader or None
        Common header (D14).
    """

    method: str
    model_ref: str
    cluster: np.ndarray
    conditions: ExternalConditions
    sector: EDSector
    num_sites: int
    blocks: Tuple[EDBlock, ...]
    reference_energy: Optional[float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    header: Optional[ResultHeader] = None

    def to_json_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """Header and blocks; eigenvectors only when ``include_arrays``."""
        blocks = []
        for b in self.blocks:
            item = {"magnon_number": b.magnon_number, "momentum_index": b.momentum_index,
                    "q": b.q, "k": b.k, "dimension": b.dimension, "energies": b.energies,
                    "solver": b.solver, "residual": b.residual}
            if include_arrays and b.vectors is not None:
                item["vectors"] = b.vectors
            blocks.append(item)
        return to_jsonable({"header": self.header,
                            "ed": {"sector": self.sector, "num_sites": self.num_sites,
                                   "reference_energy": self.reference_energy,
                                   "blocks": blocks}})

    def energies(self) -> np.ndarray:
        """All computed total energies, ascending."""
        return np.sort(np.concatenate([b.energies for b in self.blocks]))

    def per_site(self) -> np.ndarray:
        """All computed energies per site."""
        return self.energies() / self.num_sites

    def excitations(self, block: EDBlock) -> np.ndarray:
        """Energies of ``block`` relative to the polarized reference state."""
        if self.reference_energy is None:
            raise ValueError("excitations need an axis (polarized reference state)")
        return block.energies - self.reference_energy

    def block(self, magnon_number: Optional[int] = None,
              momentum_index: Optional[int] = None) -> EDBlock:
        """The block with the given labels."""
        for b in self.blocks:
            if b.magnon_number == magnon_number and b.momentum_index == momentum_index:
                return b
        raise KeyError((magnon_number, momentum_index))


def _diagonalize(matrix, num_eigenvalues, return_vectors, dense_limit):
    dimension = matrix.shape[0]
    if dimension == 0:
        return np.zeros(0), (np.zeros((0, 0)) if return_vectors else None), "dense", 0.0
    lanczos = (num_eigenvalues is not None and num_eigenvalues < dimension - 1
               and dimension > ED_LANCZOS_MIN_DIMENSION)
    if not lanczos:
        if dimension > dense_limit:
            raise ValueError(f"block dimension {dimension} exceeds dense_limit={dense_limit}; "
                             "give a small num_eigenvalues for Lanczos")
        dense = matrix.toarray()
        values, vectors = np.linalg.eigh(dense)
        if num_eigenvalues is not None:
            values, vectors = values[:num_eigenvalues], vectors[:, :num_eigenvalues]
        solver = "dense"
    else:
        values, vectors = eigsh(matrix, k=num_eigenvalues, which="SA")
        order = np.argsort(values)
        values, vectors = values[order], vectors[:, order]
        solver = "lanczos"
    residual = float(np.max(np.linalg.norm(matrix @ vectors - vectors * values, axis=0),
                            initial=0.0))
    return values, (vectors if return_vectors else None), solver, residual


def _hermiticity(matrix) -> float:
    difference = matrix - matrix.getH()
    return float(np.max(np.abs(difference.data), initial=0.0))


def solve_ed(model: SpinModel, geometry: CalculationGeometry,
             conditions: Optional[ExternalConditions] = None,
             sector: EDSector = EDSector(), num_eigenvalues: Optional[int] = None,
             return_vectors: bool = False, dense_limit: int = ED_DENSE_LIMIT,
             symmetry_tolerance: float = ED_SYMMETRY_TOLERANCE) -> EDResult:
    """Diagonalize ``model`` on the finite torus of ``geometry``.

    Parameters
    ----------
    model : SpinModel
    geometry : CalculationGeometry
        A finite torus; see :func:`~spintoolkit.system.cluster.expand_on_torus`.
    conditions : ExternalConditions, optional
    sector : EDSector
    num_eigenvalues : int, optional
        Lowest eigenvalues per block; None computes all (dense). Blocks larger
        than ``ED_LANCZOS_MIN_DIMENSION`` use Lanczos when this is small.
    return_vectors : bool
    dense_limit : int
        Largest block diagonalized densely.
    symmetry_tolerance : float
        Relative size below which sector-changing coefficients count as round-off.

    Returns
    -------
    EDResult

    Raises
    ------
    SectorError
        If the Hamiltonian changes the requested magnetization.
    """
    conditions = conditions or ExternalConditions()
    cluster: TorusCluster = expand_on_torus(model, geometry)
    fields = cluster.fields(conditions)
    rotation = np.eye(3) if sector.axis is None else frame_rotation(sector.axis)
    conserve = sector.magnon_number is not None
    terms = operator_terms(cluster.source, cluster.target, cluster.exchange, fields,
                           rotation, conserve=conserve, tolerance=symmetry_tolerance)
    if conserve:
        magnon_numbers = (range(int(np.rint(np.sum(2 * cluster.spins))) + 1)
                          if sector.magnon_number == "all" else [int(sector.magnon_number)])
    else:
        magnon_numbers = [None]
    q_all, k_all = allowed_momenta(model, geometry)
    if sector.momenta is None:
        momentum_indices = [None]
    elif sector.momenta == "all":
        momentum_indices = list(range(len(q_all)))
    else:
        momentum_indices = list(sector.momenta)
    if sector.momenta is not None:
        permutations, shifts = cluster_translations(cluster)

    diagnostics: Dict[str, Any] = {
        "num_sites": cluster.num_sites, "num_bonds": len(cluster.source),
        "frame_rotation": rotation.tolist(), "dropped_sector_coefficient": terms.dropped,
        "symmetry_tolerance": symmetry_tolerance, "hermiticity": 0.0}
    blocks = []
    for n in magnon_numbers:
        basis = SectorBasis(cluster.spins, n)
        columns, target_codes, values = matrix_elements(
            terms, basis.digits, basis.codes, basis.weights, cluster.spins)
        rows = basis.lookup(target_codes)
        leaving = (rows < 0) & (np.abs(values) > 0)
        if np.any(leaving):
            raise SectorError(
                f"the Hamiltonian changes the magnetization along {sector.axis} "
                f"(largest amplitude {np.max(np.abs(values[leaving])):.3g}); the "
                "magnon-number sector is not conserved")
        if sector.momenta is None:
            matrix = sparse.csr_matrix((values, (rows, columns)),
                                       shape=(basis.dimension, basis.dimension))
            diagnostics["hermiticity"] = max(diagnostics["hermiticity"], _hermiticity(matrix))
            energies, vectors, solver, residual = _diagonalize(
                matrix, num_eigenvalues, return_vectors, dense_limit)
            blocks.append(EDBlock(n, None, None, None, basis.dimension, energies, vectors,
                                  solver, residual))
            continue
        orbits = TranslationOrbits(basis, permutations, shifts)
        reps = orbits.representatives()
        for m in momentum_indices:
            q = q_all[m]
            reps_q = reps[orbits.compatible(reps, q, basis)]
            position = -np.ones(basis.dimension, dtype=np.int64)
            position[reps_q] = np.arange(len(reps_q))
            source_position = position[columns]
            use = source_position >= 0
            target = rows[use]
            target_rep = orbits.representative[target]
            row_position = position[target_rep]
            keep = row_position >= 0
            shift = shifts[orbits.shift_to_representative[target[keep]]]
            phase = np.exp(2j * np.pi * (shift @ q))
            norm = np.sqrt(orbits.stabilizer[target_rep[keep]]
                           / orbits.stabilizer[columns[use][keep]])
            data = values[use][keep] * phase * norm
            matrix = sparse.csr_matrix((data, (row_position[keep], source_position[use][keep])),
                                       shape=(len(reps_q), len(reps_q)))
            diagnostics["hermiticity"] = max(diagnostics["hermiticity"], _hermiticity(matrix))
            energies, vectors, solver, residual = _diagonalize(
                matrix, num_eigenvalues, return_vectors, dense_limit)
            blocks.append(EDBlock(n, m, q.copy(), k_all[m].copy(), len(reps_q), energies,
                                  vectors, solver, residual))

    reference = None
    if sector.axis is not None:
        # <polarized|H|polarized>: the diagonal entries of the all-zero-deviation state.
        up = np.zeros((1, cluster.num_sites), dtype=np.int64)
        weights = SectorBasis(cluster.spins, 0).weights
        _, codes, values = matrix_elements(terms, up, np.zeros(1, dtype=np.int64),
                                           weights, cluster.spins)
        reference = float(np.real(np.sum(values[codes == 0])))
    settings = {"sector": to_jsonable(sector), "num_eigenvalues": num_eigenvalues,
                "return_vectors": return_vectors, "dense_limit": dense_limit,
                "symmetry_tolerance": symmetry_tolerance}
    header = ResultHeader.build("ed", model, None, geometry, conditions, settings,
                                "total energies of the torus (E0)", diagnostics)
    return EDResult("ed", cluster.model_ref, cluster.cluster, conditions, sector,
                    cluster.num_sites, tuple(blocks), reference, diagnostics, header)
