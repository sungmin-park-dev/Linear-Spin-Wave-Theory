"""Spin-operator Hamiltonian of a torus cluster in a product-state basis.

With ladder operators ``s = (S^+, S^-, S^z)`` and ``S = U s``
(``S^x = (S^+ + S^-)/2``, ``S^y = (S^+ - S^-)/(2i)``) a bond is

    S_i^T J S_j = s_i^T (U^T J U) s_j,

and a zeeman term is ``-h^T S_i = -(U^T h) . s_i``. Every product of ladder
operators changes the total deviation by the sum of its steps
(``S^+``: -1, ``S^-``: +1, ``S^z``: 0), so the terms that change the
magnetization along the quantization axis are known before any state is
touched. In a fixed-magnetization sector those terms must vanish; their
largest coefficient is reported as the symmetry violation.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import numpy as np

#: ``S = U s`` with ``s = (S^+, S^-, S^z)``.
LADDER = np.array([[0.5, 0.5, 0.0], [-0.5j, 0.5j, 0.0], [0.0, 0.0, 1.0]])

#: Deviation step of ``S^+``, ``S^-`` and ``S^z``.
STEP = np.array([-1, 1, 0])


class OperatorTerms(NamedTuple):
    """Ladder-basis coefficients of a cluster Hamiltonian."""

    pairs: List[Tuple[int, int, int, int, complex]]   # (i, j, alpha, beta, c)
    singles: List[Tuple[int, int, complex]]            # (i, alpha, c)
    dropped: float                                     # largest dropped coefficient


def frame_rotation(axis) -> np.ndarray:
    """Rotation ``R`` with ``R @ axis = z`` (identity for ``axis = z``)."""
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    c = float(n @ z)
    if c > 1 - 1e-15:
        return np.eye(3)
    if c < -1 + 1e-15:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    return np.eye(3) + s * k + (1 - c) * k @ k


def operator_terms(source, target, exchange, fields, rotation=np.eye(3),
                   conserve: bool = False, tolerance: float = 0.0) -> OperatorTerms:
    """Ladder coefficients of the bonds and fields in the rotated frame.

    Parameters
    ----------
    source, target, exchange : arrays
        Bonds of a :class:`~spintoolkit.system.cluster.TorusCluster`.
    fields : (n, 3) array
        Zeeman fields ``h_i`` (the Hamiltonian contains ``-h_i . S_i``).
    rotation : (3, 3) array
        Spin frame rotation; the quantization axis is ``rotation.T @ z``.
    conserve : bool
        Drop coefficients that change the magnetization if their magnitude is
        at most ``tolerance`` times the largest coefficient; larger ones are
        kept, and the basis lookup then reports the broken sector.
    """
    R = np.asarray(rotation, dtype=float)
    pairs, singles = [], []
    scale = max([np.max(np.abs(J)) for J in exchange] + [np.max(np.abs(fields), initial=0.0)] + [0.0])
    dropped = 0.0
    for i, j, J in zip(source, target, exchange):
        C = LADDER.T @ (R @ J @ R.T) @ LADDER
        for a in range(3):
            for b in range(3):
                c = C[a, b]
                if c == 0:
                    continue
                if conserve and STEP[a] + STEP[b] != 0:
                    if abs(c) <= tolerance * scale:
                        dropped = max(dropped, abs(c))
                        continue
                pairs.append((int(i), int(j), a, b, complex(c)))
    for i, h in enumerate(np.asarray(fields, dtype=float)):
        c_vec = -(LADDER.T @ (R @ h))
        for a in range(3):
            c = c_vec[a]
            if c == 0:
                continue
            if conserve and STEP[a] != 0:
                if abs(c) <= tolerance * scale:
                    dropped = max(dropped, abs(c))
                    continue
            singles.append((i, a, complex(c)))
    return OperatorTerms(pairs, singles, dropped)


def _local_tables(spins: np.ndarray):
    """Matrix elements of S^+, S^-, S^z on deviation digits, padded to the largest dimension."""
    dmax = int(np.max(np.rint(2 * spins + 1)))
    tables = np.zeros((len(spins), 3, dmax))
    for i, S in enumerate(spins):
        d = np.arange(dmax)
        m = S - d
        valid = d <= 2 * S + 1e-9
        tables[i, 0] = np.where(valid & (d >= 1), np.sqrt(np.maximum(S * (S + 1) - m * (m + 1), 0)), 0)
        tables[i, 1] = np.where(valid & (d <= 2 * S - 1 + 1e-9),
                                np.sqrt(np.maximum(S * (S + 1) - m * (m - 1), 0)), 0)
        tables[i, 2] = np.where(valid, m, 0)
    return tables


def matrix_elements(terms: OperatorTerms, digits: np.ndarray, codes: np.ndarray,
                    weights: np.ndarray, spins: np.ndarray):
    """Nonzero ``<target|H|source>`` for the given source states.

    Returns
    -------
    columns : int array
        Row index into ``digits`` of the source state.
    target_codes : int array
    values : complex array
    """
    tables = _local_tables(spins)
    columns, targets, values = [], [], []
    rows = np.arange(len(codes))
    diagonal = np.zeros(len(codes), dtype=complex)

    for i, j, a, b, c in terms.pairs:
        amp = c * tables[i, a][digits[:, i]] * tables[j, b][digits[:, j]]
        keep = amp != 0
        if not np.any(keep):
            continue
        if STEP[a] == 0 and STEP[b] == 0:
            diagonal[keep] += amp[keep]
            continue
        columns.append(rows[keep])
        targets.append(codes[keep] + STEP[a] * weights[i] + STEP[b] * weights[j])
        values.append(amp[keep])
    for i, a, c in terms.singles:
        amp = c * tables[i, a][digits[:, i]]
        keep = amp != 0
        if not np.any(keep):
            continue
        if STEP[a] == 0:
            diagonal[keep] += amp[keep]
            continue
        columns.append(rows[keep])
        targets.append(codes[keep] + STEP[a] * weights[i])
        values.append(amp[keep])
    columns.append(rows)
    targets.append(codes)
    values.append(diagonal)
    return np.concatenate(columns), np.concatenate(targets), np.concatenate(values)
