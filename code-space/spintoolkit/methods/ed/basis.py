"""Product-state bases, magnetization sectors and translation orbits.

A basis state gives each site a deviation ``d_i = S_i - m_i`` from the fully
polarized state along the quantization axis (``d_i = 0 .. 2 S_i``). States are
encoded as mixed-radix integers ``code = sum_i d_i w_i`` with
``w_i = prod_{j < i} (2 S_j + 1)`` and kept sorted by code.

Without a conserved quantity the basis is the whole product space. With a
conserved total magnetization along the quantization axis, the sector with
``n`` deviations (the ``n``-magnon sector) is enumerated directly, so a small
``n`` stays cheap on large clusters.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


class SectorBasis:
    """Sorted product states with an optional fixed total deviation.

    Parameters
    ----------
    spins : sequence of float
        Spin length of each site.
    magnon_number : int, optional
        Total deviation ``sum_i d_i``; None enumerates the whole space.
    """

    def __init__(self, spins: Sequence[float], magnon_number: Optional[int] = None):
        self.spins = np.asarray(spins, dtype=float)
        self.dims = np.rint(2 * self.spins + 1).astype(np.int64)
        if not np.allclose(self.dims, 2 * self.spins + 1):
            raise ValueError("spin lengths must be multiples of 1/2")
        self.weights = np.concatenate([[1], np.cumprod(self.dims)[:-1]]).astype(np.int64)
        if np.prod(self.dims.astype(float)) >= 2.0 ** 62:
            raise ValueError("Hilbert space too large for integer state codes")
        self.magnon_number = magnon_number
        digits = _enumerate(self.dims, magnon_number)
        codes = digits @ self.weights
        order = np.argsort(codes)
        self.digits = digits[order]
        self.codes = codes[order]

    @property
    def dimension(self) -> int:
        return len(self.codes)

    def lookup(self, codes: np.ndarray) -> np.ndarray:
        """Index of each code in the basis, or -1 if absent."""
        position = np.searchsorted(self.codes, codes)
        position = np.minimum(position, len(self.codes) - 1)
        return np.where(self.codes[position] == codes, position, -1)

    def permuted_codes(self, permutation: np.ndarray) -> np.ndarray:
        """Codes of all states after moving site ``i`` to ``permutation[i]``."""
        moved = np.empty_like(self.digits)
        moved[:, permutation] = self.digits
        return moved @ self.weights


def _enumerate(dims: np.ndarray, total: Optional[int]) -> np.ndarray:
    """All digit rows with ``0 <= d_i < dims[i]`` and, optionally, ``sum d = total``."""
    n = len(dims)
    if total is not None:
        if total < 0 or total > int(np.sum(dims - 1)):
            return np.zeros((0, n), dtype=np.int64)
    remaining = np.concatenate([np.cumsum((dims - 1)[::-1])[::-1][1:], [0]])
    rows = np.zeros((1, 0), dtype=np.int64)
    sums = np.zeros(1, dtype=np.int64)
    for i, d in enumerate(dims):
        choice = np.arange(d, dtype=np.int64)
        new_sums = (sums[:, None] + choice[None, :]).ravel()
        new_rows = np.concatenate([np.repeat(rows, d, axis=0),
                                   np.tile(choice, len(rows))[:, None]], axis=1)
        if total is not None:
            keep = (new_sums <= total) & (new_sums + remaining[i] >= total)
            new_rows, new_sums = new_rows[keep], new_sums[keep]
        rows, sums = new_rows, new_sums
    return rows


class TranslationOrbits:
    """Translation orbits of the states of a basis.

    Parameters
    ----------
    basis : SectorBasis
    permutations : sequence of (n,) int arrays
        Site permutation of every translation of the torus (identity first).
    shifts : (N_c, 2) int array
        Cell shift of every translation.

    Attributes
    ----------
    representative : (D,) int array
        Basis index of each state's representative (smallest code in its orbit).
    shift_to_representative : (D,) int array
        Translation index ``t`` with ``T_t |state> = |representative>``.
    stabilizer : (D,) int array
        Number of translations that leave the state unchanged.
    """

    def __init__(self, basis: SectorBasis, permutations: Sequence[np.ndarray],
                 shifts: np.ndarray):
        self.shifts = np.asarray(shifts, dtype=np.int64)
        images = np.column_stack([basis.permuted_codes(p) for p in permutations])
        self.shift_to_representative = np.argmin(images, axis=1)
        smallest = images[np.arange(len(images)), self.shift_to_representative]
        self.representative = basis.lookup(smallest)
        if np.any(self.representative < 0):
            raise ValueError("translations do not preserve the basis")
        self.stabilizer = np.sum(images == basis.codes[:, None], axis=1)
        self.images = images

    def representatives(self) -> np.ndarray:
        """Basis indices of the orbit representatives."""
        return np.flatnonzero(self.representative == np.arange(len(self.representative)))

    def compatible(self, index: np.ndarray, q: np.ndarray, basis: SectorBasis,
                   tolerance: float = 1e-9) -> np.ndarray:
        """Representatives whose Bloch state at momentum ``q`` is nonzero."""
        own = basis.codes[index][:, None]
        stabilizing = self.images[index] == own
        phases = np.exp(2j * np.pi * (self.shifts @ q))
        return np.abs(stabilizing @ phases) > tolerance


def cluster_translations(cluster) -> Tuple[list, np.ndarray]:
    """Permutations and cell shifts of every translation of a :class:`TorusCluster`."""
    from spintoolkit.states.spin_state import supercell_cells

    shifts = np.array(supercell_cells(cluster.cluster), dtype=np.int64)
    return [cluster.translation(tuple(s)) for s in shifts], shifts
