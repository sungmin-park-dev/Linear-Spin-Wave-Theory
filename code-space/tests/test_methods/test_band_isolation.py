"""Band isolation, small resolved gaps, and propagation through public paths.

The two-level reference is analytic. Flat-band grids are periodic positive
bosonic models; they exercise validity reporting, not a material Chern number.
"""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose
import pytest

from spintoolkit.methods.lswt.diagonalization import Diagonalizer
from spintoolkit.observables.thermodynamics import Thermodynamics
from spintoolkit.observables.topology import Topology, compute_berry_curvature


def local_two_level(delta, scale=1.0):
    """Return eigenpairs and derivatives of the local Dirac model at k=0."""
    energies = np.tile([0.8 + delta, 0.8 - delta], 2) * scale
    derivatives = []
    for sigma in (np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])):
        matrix = np.zeros((4, 4), complex)
        matrix[:2, :2] = 0.1 * scale * sigma
        matrix[2:, 2:] = -0.1 * scale * sigma.T
        derivatives.append(matrix)
    return energies, np.eye(4), derivatives


def flat_grid(energies):
    """A constant positive Hamiltonian on a complete 2x2 reciprocal grid."""
    points = np.array([[0, 0], [0, np.pi], [np.pi, 0], [np.pi, np.pi]])
    matrix = np.tile(np.diag(np.tile(energies, 2)), (4, 1, 1))
    data, _ = Diagonalizer.get_K_data(
        points, matrix, "No", partial_derivative_Hk=[np.zeros_like(matrix)] * 2
    )
    parent = SimpleNamespace(
        Ns=len(energies), system=SimpleNamespace(lattice_vectors=np.eye(2)),
        bz_data={"area": np.pi**2}, _integration_k_keys=frozenset(data),
    )
    return parent, data


@pytest.mark.parametrize("delta", [0.0, 1e-15])
@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_unresolved_pair_is_nan_and_spacing_is_not_skipped(delta, scale):
    energies, vectors, derivatives = local_two_level(delta, scale)
    curvature, spacing = compute_berry_curvature(
        energies, vectors, derivatives, band_gap_cutoff=1e-8 * scale
    )
    assert np.isnan(curvature).all()
    assert_allclose(spacing, abs(energies[0] - energies[1]), rtol=0, atol=0)


@pytest.mark.parametrize("delta", [1e-2, 1e-4, 1e-6])
@pytest.mark.parametrize("scale", [1e-160, 1.0, 1e160])
def test_small_resolved_gap_matches_analytic_curvature_under_energy_rescaling(delta, scale):
    energies, vectors, derivatives = local_two_level(delta, scale)
    # Scaling the energy unit must not overflow/underflow a squared denominator.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        curvature, spacing = compute_berry_curvature(
            energies, vectors, derivatives, band_gap_cutoff=1e-8 * scale
        )
    assert_allclose(curvature, [-0.005 / delta**2, 0.005 / delta**2], rtol=1e-9)
    assert_allclose(spacing / scale, 2 * delta, rtol=1e-9)


def test_explicit_zero_cutoff_retains_a_nonzero_gap():
    curvature, _ = compute_berry_curvature(*local_two_level(1e-13), band_gap_cutoff=0)
    assert np.isfinite(curvature).all()


def test_all_other_bands_are_checked_even_if_not_adjacent_in_array():
    energies = np.tile([0.8, 1.5, 0.8], 2)
    curvature, spacing = compute_berry_curvature(
        energies, np.eye(6), [np.zeros((6, 6))] * 2
    )
    assert np.isnan(curvature[[0, 2]]).all()
    assert curvature[1] == 0
    assert_allclose(spacing, [0, 0.7, 0])


def test_single_band_reports_signed_particle_hole_separation():
    _, spacing = compute_berry_curvature(
        np.array([0.8, 0.8]), np.eye(2), [np.zeros((2, 2))] * 2
    )
    assert_allclose(spacing, [1.6])


def test_zero_mode_collision_is_unavailable():
    curvature, spacing = compute_berry_curvature(
        np.zeros(2), np.eye(2), [np.zeros((2, 2))] * 2
    )
    assert np.isnan(curvature).all()
    assert_allclose(spacing, 0)


def test_phase_choice_does_not_change_resolved_curvature():
    energies, vectors, derivatives = local_two_level(1e-4)
    vectors = vectors @ np.diag(np.exp(1j * np.array([0.7, -1.1, 0.2, 2.0])))
    curvature, _ = compute_berry_curvature(energies, vectors, derivatives)
    assert_allclose(curvature, [-5e5, 5e5], rtol=1e-10)


@pytest.mark.parametrize("temperature", [0.0, 2.0])
def test_isolated_band_chern_survives_but_hall_is_unavailable(temperature):
    parent, data = flat_grid([1.5, 0.8, 0.8])
    curvature, chern, hall = Topology(parent).compute_thermal_Hall(data, temperature)
    assert_allclose(curvature[:, 0], 0)
    assert np.isnan(curvature[:, 1:]).all()
    assert chern[0] == 0
    assert np.isnan(chern[1:]).all()
    assert np.isnan(hall)
    combined = Thermodynamics(parent).compute_thermodynamic_quantities_at_T(data, temperature)
    assert np.isnan(combined["Thermal Hall Conductance"])
    for key in combined.keys() - {"Thermal Hall Conductance"}:
        assert np.isfinite(combined[key]).all(), key


def test_unresolved_hall_propagates_through_temperature_sweep_and_3d_conversion():
    parent, data = flat_grid([0.8, 0.8])
    _, values = Thermodynamics(parent).get_thermodynamic_quantities(
        data, Temperature_range=(0, 2, 1), layer_spacing_m=7e-10
    )
    assert np.isnan(values["Thermal Hall Conductance"]).all()


def test_verbose_small_gap_warning_keeps_resolved_results():
    parent, data = flat_grid([0.8001, 0.7999])
    with pytest.warns(RuntimeWarning, match="mesh convergence"):
        _, chern, hall = Topology(parent).compute_thermal_Hall(data, 2.0, verbose=True)
    assert_allclose(chern, 0)
    assert hall == 0


def test_gapped_curvature_matches_preserved_legacy(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "legacy"))
    legacy = import_module("modules.LinearSpinWaveTheory.lswt_topology")
    arguments = local_two_level(1e-4)
    actual, _ = compute_berry_curvature(*arguments)
    historical, _ = legacy.compute_Berry_curvature(*arguments)
    assert_allclose(actual, historical, rtol=1e-14)


def test_default_cutoff_is_policy_and_can_be_lowered():
    arguments = local_two_level(1e-10)
    excluded, spacing = compute_berry_curvature(*arguments)
    allowed, _ = compute_berry_curvature(*arguments, band_gap_cutoff=1e-12)
    assert np.isnan(excluded).all()
    assert np.isfinite(allowed).all()
    assert (spacing > 0).all()


def test_cutoff_boundary_is_excluded_without_changing_the_gap():
    energies = np.array([2., 1., 2., 1.])
    arguments = energies, np.eye(4), [np.zeros((4, 4))] * 2
    excluded, spacing = compute_berry_curvature(*arguments, band_gap_cutoff=1)
    allowed, _ = compute_berry_curvature(*arguments, band_gap_cutoff=np.nextafter(1., 0.))
    assert np.isnan(excluded).all()
    assert_allclose(spacing, 1)
    assert_allclose(allowed, 0)


@pytest.mark.parametrize("temperature", [0., 2.])
def test_custom_cutoff_reaches_both_hall_paths_only(temperature):
    parent, data = flat_grid([1.5, 0.8000005, 0.7999995])
    topology = Topology(parent)
    _, chern, hall = topology.compute_thermal_Hall(
        data, temperature, band_gap_cutoff=1e-5, layer_spacing_m=7e-10
    )
    assert chern[0] == 0
    assert np.isnan(chern[1:]).all()
    assert np.isnan(hall)
    assert topology.compute_thermal_Hall(data, temperature, band_gap_cutoff=1e-7)[2] == 0
    thermodynamics = Thermodynamics(parent)
    strict = thermodynamics.compute_thermodynamic_quantities_at_T(
        data, temperature, band_gap_cutoff=1e-5, layer_spacing_m=7e-10
    )
    loose = thermodynamics.compute_thermodynamic_quantities_at_T(
        data, temperature, band_gap_cutoff=1e-7, layer_spacing_m=7e-10
    )
    assert np.isnan(strict['Thermal Hall Conductance'])
    assert loose['Thermal Hall Conductance'] == 0
    for key in strict.keys() - {'Thermal Hall Conductance'}:
        assert_allclose(strict[key], loose[key], rtol=0, atol=0)


def test_custom_cutoff_is_preserved_in_temperature_sweep():
    parent, data = flat_grid([0.8000005, 0.7999995])
    thermodynamics = Thermodynamics(parent)
    for cutoff, excluded in [(1e-5, True), (1e-7, False)]:
        _, result = thermodynamics.get_thermodynamic_quantities(
            data, Temperature_range=(0, 2, 1), band_gap_cutoff=cutoff,
            layer_spacing_m=7e-10,
        )
        hall = result['Thermal Hall Conductance']
        assert np.isnan(hall).all() if excluded else (hall == 0).all()


@pytest.mark.parametrize('cutoff', [-1., np.nan, np.inf, 1j, [1e-8]])
def test_invalid_cutoff_is_rejected_even_for_empty_public_data(cutoff):
    parent, _ = flat_grid([0.8, 0.6])
    thermodynamics = Thermodynamics(parent)
    for call in [
        lambda: compute_berry_curvature(*local_two_level(.1), band_gap_cutoff=cutoff),
        lambda: Topology(parent).compute_thermal_Hall({}, 2., band_gap_cutoff=cutoff),
        lambda: thermodynamics.compute_thermodynamic_quantities_at_T({}, 2., band_gap_cutoff=cutoff),
        lambda: thermodynamics.get_thermodynamic_quantities({}, band_gap_cutoff=cutoff),
    ]:
        with pytest.raises(ValueError, match='band_gap_cutoff'):
            call()
