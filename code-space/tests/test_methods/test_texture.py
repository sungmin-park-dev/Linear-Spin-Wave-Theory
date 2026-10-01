"""Skyrmion number of periodic textures (D42).

1. Solid angle: the octant spanned by x, y, z is +pi/2; reversing the order
   flips the sign; three coplanar spins at 120 degrees are exceptional.
2. Ferromagnet: Q = 0; coplanar 120 degree state: Q undefined (not zero).
3. A skyrmion (core -z, background +z) on 12 x 12 triangular and square
   supercells: Q = -1 for vorticity +1 and +1 for vorticity -1, as the
   continuum (w/2)(cos theta(0) - cos theta(inf)); the square lattice checks
   the degenerate Delaunay case.
4. Four-sublattice tetrahedral state: every triangle covers one face of the
   tetrahedron (|Omega| = pi, area 4 pi / 4) with one sign, so |Q| = 2 per
   2 x 2 cell; Q is unchanged by a global rotation and flips under time
   reversal (n -> -n).
"""

import numpy as np
import pytest

from spintoolkit.models import polarized_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.observables.texture import lattice_triangles, skyrmion_charge, solid_angle
from spintoolkit.states.spin_state import SpinState

X, Y, Z = np.eye(3)


def test_solid_angle():
    assert solid_angle(X, Y, Z) == pytest.approx(np.pi / 2, abs=1e-14)
    assert solid_angle(X, Z, Y) == pytest.approx(-np.pi / 2, abs=1e-14)
    a = [np.array([np.cos(t), np.sin(t), 0.0]) for t in (0, 2 * np.pi / 3, 4 * np.pi / 3)]
    assert np.isnan(solid_angle(*a))
    assert solid_angle(Z, Z, Z) == 0.0


def test_ferromagnet_and_coplanar_120():
    tri = triangular_heisenberg()
    assert len(lattice_triangles(tri)) == 2
    assert skyrmion_charge(tri, polarized_state(tri)).integer == 0
    q = skyrmion_charge(tri, state_120(tri))
    assert np.isnan(q.charge) and q.integer is None and q.undefined == q.num_triangles == 6
    assert q.chirality == 0.0


def skyrmion(model, L, vorticity):
    A = model.lattice
    M = np.diag([L, L])
    period = M @ A
    centre = (np.array([L / 2, L / 2]) + 0.25) @ A
    radius = L * np.linalg.norm(A[0]) / 4

    def direction(site, cell):
        r = np.asarray(cell, float) @ A - centre
        f = np.linalg.solve(period.T, r)
        r = (f - np.round(f)) @ period                     # nearest image
        theta = np.pi * max(0.0, 1 - np.linalg.norm(r) / radius)
        phi = vorticity * np.arctan2(r[1], r[0])
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    return SpinState.from_function(model, M, direction)


@pytest.mark.parametrize("model", [triangular_heisenberg(), square_heisenberg()],
                         ids=["triangular", "square"])
def test_skyrmion_sign_matches_continuum(model):
    assert skyrmion_charge(model, skyrmion(model, 12, +1)).integer == -1
    assert skyrmion_charge(model, skyrmion(model, 12, -1)).integer == +1


def tetrahedral(model, directions):
    return SpinState.from_function(model, np.diag([2, 2]),
                                   lambda site, c: directions[2 * (c[0] % 2) + (c[1] % 2)])


def test_tetrahedral_state():
    tri = triangular_heisenberg()
    t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    q = skyrmion_charge(tri, tetrahedral(tri, t))
    np.testing.assert_allclose(np.abs(q.per_triangle), np.pi, atol=1e-12)
    assert abs(q.integer) == 2 and q.num_triangles == 8
    R = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))[0]
    R *= np.sign(np.linalg.det(R))                              # proper rotation
    assert skyrmion_charge(tri, tetrahedral(tri, t @ R.T)).integer == q.integer
    assert skyrmion_charge(tri, tetrahedral(tri, -t)).integer == -q.integer


def test_rejects_clockwise_triangles():
    tri = triangular_heisenberg()
    (a, b, c), _ = lattice_triangles(tri)
    with pytest.raises(ValueError, match="counterclockwise"):
        skyrmion_charge(tri, polarized_state(tri), triangles=[(a, c, b)])
