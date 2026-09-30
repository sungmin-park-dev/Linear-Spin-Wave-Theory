"""High-symmetry points and magnon band structure (stage 4d, D33).

1. High-symmetry points of hexagonal, square, rectangular and magnetic
   (sqrt3 x sqrt3) zones: distances and the reciprocal relation.
2. Bands against analytic dispersions: square Neel 4JS sqrt(1 - gamma^2),
   ferromagnet in a field 4|J|S(1 - gamma) + h, triangular 120 degrees folded
   omega(k), omega(k +- Q) with omega = 3JS sqrt((1 - gamma)(1 + 2 gamma)).
3. Goldstone vertices are zero modes, not instabilities; an unstable state
   gives NaN with a warning; explicit vertices and the magnetic zone.
"""

import numpy as np
import pytest

from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import (
    neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg)
from spintoolkit.observables.bands import band_structure
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.high_symmetry import high_symmetry_points, reciprocal_lattice

TRIANGULAR = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
S = 0.5


def test_high_symmetry_points():
    hexagonal = high_symmetry_points(TRIANGULAR)
    assert np.allclose(hexagonal["K"], [4 * np.pi / 3, 0])
    assert np.linalg.norm(hexagonal["M"]) == pytest.approx(2 * np.pi / np.sqrt(3))
    assert np.linalg.norm(hexagonal["K'"]) == pytest.approx(4 * np.pi / 3)
    square = high_symmetry_points(np.eye(2))
    assert np.allclose([square["X"], square["Y"], square["M"]], [[np.pi, 0], [0, np.pi], [np.pi, np.pi]])
    rectangle = high_symmetry_points([[1, 0], [0, 2]])
    assert np.allclose([rectangle["X"], rectangle["Y"]], [[np.pi, 0], [0, np.pi / 2]])
    magnetic = high_symmetry_points(np.array([[1, 1], [-1, 2]]) @ TRIANGULAR)
    assert np.linalg.norm(magnetic["K"]) == pytest.approx(4 * np.pi / 3 / np.sqrt(3))
    for lattice in (TRIANGULAR, np.eye(2), [[1, 0], [0.3, 1.1]]):
        assert np.allclose(np.asarray(lattice) @ reciprocal_lattice(lattice).T, 2 * np.pi * np.eye(2))


def split(bands, analytic):
    """Largest deviation away from and at the zero-mode points."""
    deviation = np.max(np.abs(bands.energies - analytic), axis=1)
    return np.max(deviation[~bands.zero_modes]), np.max(deviation[bands.zero_modes], initial=0.0)


def test_square_neel_and_ferromagnet():
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, neel_state(model), None, settings=LSWTSettings(mesh=(4, 4)))
    bands = band_structure(result, ("Γ", "X", "M", "Γ"), points=120)
    gamma = 0.5 * (np.cos(bands.k_points[:, 0]) + np.cos(bands.k_points[:, 1]))
    away, at = split(bands, 4 * S * np.sqrt(1 - gamma ** 2)[:, None])
    assert away < 1e-12 and at < 1e-7
    assert list(np.flatnonzero(bands.zero_modes)) == [0, np.argmin(np.abs(bands.distance - bands.label_distances[2])), len(bands.distance) - 1]
    ferro = square_heisenberg(J=-1.0)
    result = solve_lswt(ferro, polarized_state(ferro), ExternalConditions(field=(0, 0, 0.3)),
                        settings=LSWTSettings(mesh=(4, 4)))
    bands = band_structure(result, ("Γ", "X", "M", "Γ"), points=120)
    gamma = 0.5 * (np.cos(bands.k_points[:, 0]) + np.cos(bands.k_points[:, 1]))
    assert np.max(np.abs(bands.energies[:, 0] - (4 * S * (1 - gamma) + 0.3))) < 1e-12
    assert not bands.zero_modes.any()


def test_triangular_120_degree_bands_are_the_folded_single_q_dispersion():
    model = triangular_heisenberg(J=1.0)
    result = solve_lswt(model, state_120(model), None, settings=LSWTSettings(mesh=(6, 6)))
    bands = band_structure(result, ("Γ", "K", "M", "Γ"), points=150)

    def omega(k):
        gamma = (np.cos(k @ TRIANGULAR[0]) + np.cos(k @ TRIANGULAR[1])
                 + np.cos(k @ (TRIANGULAR[1] - TRIANGULAR[0]))) / 3
        return 3 * S * np.sqrt(np.clip((1 - gamma) * (1 + 2 * gamma), 0, None))

    Q = np.array([4 * np.pi / 3, 0])
    k = bands.k_points
    analytic = np.sort(np.column_stack([omega(k), omega(k + Q), omega(k - Q)]), axis=1)
    away, at = split(bands, analytic)
    assert away < 1e-12 and at < 1e-7
    assert bands.labels == ["Γ", "K", "M", "Γ"] and bands.zero_modes.sum() == 3


def test_unstable_state_explicit_vertices_and_magnetic_zone():
    ferro = square_heisenberg(J=-1.0)
    result = solve_lswt(ferro, neel_state(ferro), None,
                        settings=LSWTSettings(mesh=(2, 2), regularization="MAGSWT"))
    with pytest.warns(UserWarning, match="not positive semidefinite"):
        bands = band_structure(result, ("Γ", "X"), points=10)
    assert np.isnan(bands.energies).any()
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, neel_state(model), None, settings=LSWTSettings(mesh=(4, 4)))
    explicit = band_structure(result, [(0.1, 0.2), (1.0, 0.4)], points=20)
    assert explicit.labels[0] == "(0.1, 0.2)" and np.allclose(explicit.k_points[-1], [1.0, 0.4])
    magnetic = band_structure(result, ("Γ", "X", "M", "Γ"), points=40, lattice="magnetic")
    assert magnetic.lattice == "magnetic"
    with pytest.raises(ValueError, match="unknown point"):
        band_structure(result, ("Γ", "K"))
