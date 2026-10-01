"""Topological charge of a periodic spin texture (D42).

Convention
----------
The lattice is triangulated periodically (:func:`lattice_triangles`) and every
elementary triangle ``(i, j, k)`` is ordered counterclockwise seen from
``+z``. Its signed solid angle is the Berg-Luescher value (Berg and Luescher,
Nucl. Phys. B 190, 412 (1981))

    tan(Omega / 2) = n_i . (n_j x n_k) / (1 + n_i . n_j + n_j . n_k + n_k . n_i),

with ``Omega`` in ``(-2 pi, 2 pi)``. It is the area of the spherical triangle
spanned by the three unit vectors along the shortest geodesics, with the
orientation inherited from the lattice triangle. The skyrmion number of the
magnetic cell is

    Q = (1 / 4 pi) sum_triangles Omega,

an integer for every periodic texture on which all triangles are defined
(the triangles tile a torus, whose image on the sphere covers it an integer
number of times). It is the lattice version of
``(1 / 4 pi) int n . (d_x n x d_y n) dx dy``: a skyrmion with its core along
``-z`` in a ``+z`` background and vorticity +1 has Q = -1.

``Omega`` is undefined when ``n_i . (n_j x n_k) = 0`` and the denominator is
not positive: the three spins lie on a great circle and do not fit in one
hemisphere (e.g. the coplanar 120 degree state, where ``Omega = +-2 pi`` are
equally valid). Q is then undefined, not zero, and is reported as such. The
scalar chirality ``chi = sum n_i . (n_j x n_k)`` is always defined but is not
a topological invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Tolerance for the exceptional configuration of the Berg-Luescher formula.
SOLID_ANGLE_TOLERANCE = 1e-10

Vertex = Tuple[str, Tuple[int, int]]


def solid_angle(n1: np.ndarray, n2: np.ndarray, n3: np.ndarray,
                tolerance: float = SOLID_ANGLE_TOLERANCE) -> float:
    """Signed Berg-Luescher solid angle of three unit vectors; NaN where undefined."""
    triple = float(np.dot(n1, np.cross(n2, n3)))
    denominator = 1.0 + float(np.dot(n1, n2) + np.dot(n2, n3) + np.dot(n3, n1))
    if abs(triple) <= tolerance and denominator <= tolerance:
        return float("nan")
    return 2.0 * float(np.arctan2(triple, denominator))


def lattice_triangles(model, patch: int = 3) -> List[Tuple[Vertex, Vertex, Vertex]]:
    """Counterclockwise elementary triangles of one primitive cell of a periodic triangulation.

    The sites of a ``(2 patch + 1)^2`` patch of primitive cells are
    Delaunay-triangulated and every triangle whose centroid lies in the home
    cell ``[0, 1)^2`` (fractional) is kept, so that the translates tile the
    plane once. Vertices are ``(site_id, cell)``.

    Raises
    ------
    ValueError
        If the kept triangles do not cover exactly one cell area (a degenerate
        Delaunay choice that is not periodic); pass ``triangles`` explicitly
        to :func:`skyrmion_charge` in that case.
    """
    from scipy.spatial import Delaunay

    lattice = np.asarray(model.lattice, dtype=float)
    vertices: List[Vertex] = []
    points = []
    for c1 in range(-patch, patch + 1):
        for c2 in range(-patch, patch + 1):
            for site in model.sites:
                vertices.append((site.id, (c1, c2)))
                points.append(model.cartesian_position(site.id, (c1, c2)))
    points = np.array(points)
    inverse = np.linalg.inv(lattice)
    triangles = []
    for simplex in Delaunay(points).simplices:
        corners = points[simplex]
        centroid = corners.mean(axis=0) @ inverse
        if not np.all((centroid >= -1e-9) & (centroid < 1 - 1e-9)):
            continue
        a, b, c = corners
        cross = (b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0]
        if abs(cross) < 1e-12:
            continue
        order = simplex if cross > 0 else simplex[[0, 2, 1]]
        triangles.append(tuple(vertices[i] for i in order))
    area = sum(_area(model, t) for t in triangles)
    cell_area = abs(np.linalg.det(lattice))
    if not np.isclose(area, cell_area, rtol=1e-9):
        raise ValueError(f"Delaunay triangles cover {area:.6g} instead of the cell area "
                         f"{cell_area:.6g}; give the triangles explicitly")
    return triangles


def _area(model, triangle) -> float:
    a, b, c = (model.cartesian_position(site, cell) for site, cell in triangle)
    return 0.5 * float((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0])


@dataclass(frozen=True)
class SkyrmionCharge:
    """Topological charge of a periodic texture.

    Attributes
    ----------
    charge : float
        ``Q`` per magnetic cell; NaN if any triangle is undefined.
    integer : int or None
        ``round(Q)``, or None when Q is undefined or not within ``1e-6`` of
        an integer (which signals an inconsistent triangulation).
    per_triangle : (nt,) array
        Solid angles ``Omega`` of the triangles of the magnetic cell.
    chirality : float
        ``sum n_i . (n_j x n_k)`` over the same triangles.
    undefined : int
        Number of exceptional triangles.
    num_triangles, num_sites : int
    """

    charge: float
    integer: Optional[int]
    per_triangle: np.ndarray
    chirality: float
    undefined: int
    num_triangles: int
    num_sites: int

    def to_dict(self) -> Dict[str, Any]:
        return {"charge": None if np.isnan(self.charge) else self.charge,
                "integer": self.integer, "chirality": self.chirality,
                "undefined_triangles": self.undefined, "num_triangles": self.num_triangles,
                "num_sites": self.num_sites}


def skyrmion_charge(model, state, triangles: Optional[Sequence[Sequence[Vertex]]] = None,
                    tolerance: float = SOLID_ANGLE_TOLERANCE) -> SkyrmionCharge:
    """Skyrmion number of ``state`` per magnetic cell (Berg-Luescher).

    Parameters
    ----------
    model : SpinModel
    state : SpinState
    triangles : sequence of 3 vertices, optional
        Counterclockwise elementary triangles of one primitive cell, vertices
        ``(site_id, cell)``; default :func:`lattice_triangles`. They must tile
        the plane once under primitive translations.
    tolerance : float
        For the exceptional configuration (see module docstring).
    """
    triangles = lattice_triangles(model) if triangles is None else [tuple(t) for t in triangles]
    for t in triangles:
        if _area(model, t) <= 0:
            raise ValueError(f"triangle {t} is not counterclockwise")
    omegas, chi = [], 0.0
    for cell in state.cells:
        for triangle in triangles:
            n = [state.direction(site, (cell[0] + c[0], cell[1] + c[1])) for site, c in triangle]
            omegas.append(solid_angle(*n, tolerance=tolerance))
            chi += float(np.dot(n[0], np.cross(n[1], n[2])))
    omegas = np.array(omegas)
    undefined = int(np.isnan(omegas).sum())
    charge = float(omegas.sum() / (4 * np.pi)) if undefined == 0 else float("nan")
    integer = None
    if undefined == 0 and abs(charge - round(charge)) < 1e-6:
        integer = int(round(charge))
    return SkyrmionCharge(charge, integer, omegas, chi, undefined, len(omegas),
                          len(state.directions))
