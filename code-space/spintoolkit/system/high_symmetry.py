"""High-symmetry points of a two-dimensional Brillouin zone (stage 4d, D33).

The first Brillouin zone is the Wigner-Seitz cell of the reciprocal lattice
``b = 2 pi inv(A)^T`` (rows ``b_1, b_2``). Its corners and edge midpoints are
named by the shape of the reciprocal lattice:

- hexagonal (two shortest reciprocal vectors of equal length at 60 or 120
  degrees): ``Γ``, ``M`` (midpoint of the edge whose normal has the smallest
  angle in ``[0, 2 pi)`` from +x), ``K`` and ``K'`` (its corners, ``K`` first
  counter-clockwise);
- square and rectangular: ``Γ``, ``X = b_1 / 2``, ``Y = b_2 / 2`` and
  ``M = (b_1 + b_2) / 2`` (the corner);
- oblique: ``Γ``, the edge midpoints ``E1 ...`` and corners ``C1 ...`` in
  counter-clockwise order.

All points are Cartesian momenta. ``Γ`` is the zone centre. Other conventions
(e.g. a different ``K`` among the six corners) are equivalent by symmetry and
can be passed to :func:`~spintoolkit.observables.bands.band_structure` as
explicit coordinates.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

GAMMA = "Γ"


def reciprocal_lattice(lattice) -> np.ndarray:
    """Rows ``b_i`` with ``a_i . b_j = 2 pi delta_ij`` for the lattice rows ``a_i``."""
    return 2 * np.pi * np.linalg.inv(np.asarray(lattice, dtype=float)).T


def _wigner_seitz(reciprocal: np.ndarray):
    """Edges of the Wigner-Seitz cell: (normal vector G, midpoint G/2, two corners)."""
    candidates = [i * reciprocal[0] + j * reciprocal[1]
                  for i in range(-3, 4) for j in range(-3, 4) if (i, j) != (0, 0)]
    candidates.sort(key=lambda g: g @ g)
    candidates = candidates[:12]
    corners = []
    for a in range(len(candidates)):
        for b in range(a + 1, len(candidates)):
            G1, G2 = candidates[a], candidates[b]
            matrix = np.array([G1, G2])
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            point = np.linalg.solve(matrix, [G1 @ G1 / 2, G2 @ G2 / 2])
            if all(point @ G <= G @ G / 2 + 1e-9 for G in candidates):
                if not any(np.allclose(point, c, atol=1e-9) for c in corners):
                    corners.append(point)
    corners.sort(key=lambda c: np.arctan2(c[1], c[0]))
    edges = []
    for G in candidates:
        on = [c for c in corners if abs(c @ G - G @ G / 2) < 1e-9]
        if len(on) == 2:
            edges.append((G, G / 2, on))
    return edges, corners


def zone_boundary(lattice) -> np.ndarray:
    """Corners of the first Brillouin zone of ``lattice`` (rows), counter-clockwise, shape (nc, 2)."""
    _, corners = _wigner_seitz(reciprocal_lattice(lattice))
    return np.array(corners)


def high_symmetry_points(lattice) -> Dict[str, np.ndarray]:
    """Named high-symmetry points of the first Brillouin zone of ``lattice`` (rows).

    Parameters
    ----------
    lattice : (2, 2) array_like
        Real-space lattice vectors (rows), e.g. ``model.lattice`` or a
        magnetic lattice ``M @ model.lattice``.

    Returns
    -------
    dict
        Name to Cartesian momentum; see the module docstring for the names.
    """
    reciprocal = reciprocal_lattice(lattice)
    edges, corners = _wigner_seitz(reciprocal)
    edges.sort(key=lambda e: (round(float(e[0] @ e[0]), 9),
                              round(float(np.mod(np.arctan2(e[0][1], e[0][0]), 2 * np.pi)), 9)))
    g1 = edges[0][0]
    independent = [e[0] for e in edges
                   if abs(e[0][0] * g1[1] - e[0][1] * g1[0]) > 1e-9 * (e[0] @ e[0])]
    g2 = min(independent, key=lambda g: (round(float(g @ g), 9), abs(g @ g1)))
    points: Dict[str, np.ndarray] = {GAMMA: np.zeros(2)}
    n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
    cos = float(g1 @ g2) / (n1 * n2)
    if len(corners) == 6 and np.isclose(n1, n2) and np.isclose(abs(cos), 0.5):
        edge = edges[0]
        points["M"] = edge[1]
        first, second = sorted(edge[2], key=lambda c: np.arctan2(c[1], c[0]))
        points["K"], points["K'"] = first, second
        return points
    if len(corners) == 4 and np.isclose(cos, 0.0, atol=1e-9):
        points["X"] = reciprocal[0] / 2
        points["Y"] = reciprocal[1] / 2
        points["M"] = (reciprocal[0] + reciprocal[1]) / 2
        return points
    for i, (_, midpoint, _) in enumerate(sorted(edges, key=lambda e: np.arctan2(e[1][1], e[1][0])), 1):
        points[f"E{i}"] = midpoint
    for i, corner in enumerate(corners, 1):
        points[f"C{i}"] = corner
    return points
