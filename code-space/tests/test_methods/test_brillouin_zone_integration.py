"""Full magnetic-BZ coverage and Chern/Hall integration invariants."""

import importlib.util
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from lswt import LSWTSolver, SpinSystem
from lswt.system.brillouin_zone import BrillouinZone
from lswt.observables.thermodynamics import Thermodynamics
from lswt.observables.topology import Topology

EXAMPLE = Path(__file__).resolve().parents[3] / "examples/thermal_hall_reference_check.py"
SPEC = importlib.util.spec_from_file_location("bz_reference", EXAMPLE)
REFERENCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REFERENCE)
TRIANGULAR = np.array([[0.5, np.sqrt(3) / 2], [0.5, -np.sqrt(3) / 2]])
CASES = [
    ("simple", [[1, 0], [0.5, np.sqrt(3) / 2]]),
    ("Hex_60", 2 * TRIANGULAR),
    ("Hex_30", [[1.5, np.sqrt(3) / 2], [1.5, -np.sqrt(3) / 2]]),
    ("Tetra", [[1, 0], [0, np.sqrt(3)]]),
    ("tetra", [[1, 1], [-2, 2]]),
    ("wigner_seitz", [[1, 0], [12.3, 0.7]]),
]


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("mode,cell", CASES)
def test_full_grid_reciprocity_area_and_fourier_moments(mode, cell, override):
    cell = np.asarray(cell)
    bz = BrillouinZone((cell, mode), bz_type=mode if override else None)
    info, points, indices = bz.get_full(8, print_idx=True)
    reciprocal = np.asarray(info["reciprocal_vectors"])
    assert_allclose(cell @ reciprocal.T, 2 * np.pi * np.eye(2), atol=1e-13)
    assert len(points) == 16**2
    assert len(indices) == len(points)
    assert_allclose(len(points) * info["area"], (2 * np.pi)**2 / abs(np.linalg.det(cell)),
                    rtol=1e-13)
    # Periodic characters integrate to zero over a full cell. This detects
    # missing boundary rows and overlapping boundary representatives.
    q = points @ cell.T
    for harmonic in ([1, 0], [0, 1], [1, 2], [-3, 1]):
        assert abs(np.mean(np.exp(1j * (q @ harmonic)))) < 2e-13
    fractions = np.mod(q / (2 * np.pi) + 0.123, 1)
    assert len(np.unique(np.round(fractions, 10), axis=0)) == len(points)
    if mode != "simple":
        polygon = np.asarray(info["BZ_corners"])
        polygon_area = abs(np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )) / 2
        assert_allclose(polygon_area, (2 * np.pi)**2 / abs(np.linalg.det(cell)), rtol=1e-12)


@pytest.mark.parametrize("cell", [
    [[1, 0], [0.5, np.sqrt(3) / 2]],
    [[1, 0], [12.3, 0.7]],
    [[0.7, 0.2], [-0.3, 1.4]],
    (TRIANGULAR * 3.7e-10).tolist(),
])
def test_wigner_seitz_points_are_inside_the_actual_polygon(cell):
    # Half-plane containment is independent of the nearest-translation code.
    bz = BrillouinZone((cell, "wigner_seitz"))
    info, points, _ = bz.get_full(7, center=(0.123, -0.213))
    polygon = np.asarray(info["BZ_corners"])
    scale = np.max(abs(polygon))
    polygon, points = polygon / scale, points / scale
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end - start
        displacement = points - start
        cross = edge[0] * displacement[:, 1] - edge[1] * displacement[:, 0]
        assert np.min(cross) > -1e-12


@pytest.mark.parametrize("mode,cell", CASES)
def test_generated_bz_matches_analytic_chern_and_independent_hall(mode, cell):
    info, points, _ = BrillouinZone((cell, mode), bz_type=mode).get_full(12)
    parent, data, omega, energies = REFERENCE.model(lattice_vectors=cell, k_points=points)
    parent.bz_data = info
    curvature, chern, hall = Topology(parent).compute_thermal_Hall(data, 2.0, bz_type=mode)
    area = abs(np.linalg.det(cell))
    expected_chern = np.sign(np.linalg.det(cell)) * np.array([1, -1])
    assert_allclose(curvature, omega, atol=1e-12)
    assert_allclose(chern, expected_chern, atol=3e-7, rtol=0)
    expected_hall = REFERENCE.reference(omega, energies, area, 2.0)
    assert_allclose(hall, expected_hall, rtol=1e-9, atol=0)
    combined = Thermodynamics(parent).compute_thermodynamic_quantities_at_T(data, 2.0)
    assert_allclose(combined["Thermal Hall Conductance"], hall, rtol=0, atol=0)


@pytest.mark.parametrize("mode", [None, "simple", "Hex_60", "Hex_30", "Tetra", "tetra", "wigner_seitz"])
def test_chern_is_not_divided_by_number_of_bands_based_on_option(mode):
    parent, data, _, _ = REFERENCE.model(n=24)
    _, chern, _ = Topology(parent).compute_thermal_Hall(data, 2.0, bz_type=mode)
    assert_allclose(chern, [1, -1], atol=3e-7, rtol=0)


def test_chern_converges_without_rounding_to_integer():
    errors = []
    for n in [4, 8, 16]:
        _, points, _ = BrillouinZone((TRIANGULAR, "Hex_60")).get_full(n)
        parent, data, _, _ = REFERENCE.model(lattice_vectors=TRIANGULAR, k_points=points)
        chern = Topology(parent).compute_thermal_Hall(data, 2.0)[1]
        errors.append(abs(chern[0] + 1))
    assert errors[0] > 1e-3
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 2e-9


@pytest.mark.parametrize("mode,cell", CASES)
def test_actual_solver_accepts_bz_modes_and_rejects_partial_integrals(mode, cell):
    system = SpinSystem(lattice_vectors=cell)
    system.add_site("A", [0, 0], spin=0.5, angles=[0, 0], magnetic_field=[0, 0, 0.8])
    solver = LSWTSolver(system, bz_type=mode)
    result = solver.solve(N=3, regularization="No")
    data = result.data["k_data"]
    assert len(data) == 36
    assert Topology(solver).compute_thermal_Hall(data, 2.0, bz_type=mode)[2] == 0
    partial = dict(list(data.items())[1:])
    curvature, chern, hall = Topology(solver).compute_thermal_Hall(partial, 2.0)
    assert np.isfinite(curvature).all()
    assert np.isnan(chern).all() and np.isnan(hall)
    thermodynamics = Thermodynamics(solver)
    combined = thermodynamics.compute_thermodynamic_quantities_at_T(partial, 2.0)
    assert np.isnan(combined["Thermal Hall Conductance"])
    assert np.isfinite(combined["Internal Energy Density"])
    _, sweep = thermodynamics.get_thermodynamic_quantities(partial, Temperature_range=(1, 2, 1))
    assert np.isnan(sweep["Thermal Hall Conductance"]).all()


@pytest.mark.parametrize("n", [0, -1, 1.5])
def test_invalid_grid_size_is_rejected(n):
    with pytest.raises(ValueError, match="positive integer"):
        BrillouinZone((TRIANGULAR, "Hex_60")).get_full(n)


def test_tetra_rejects_nonrectangular_lattice():
    with pytest.raises(ValueError, match="orthogonal"):
        BrillouinZone((TRIANGULAR, "Tetra"))
