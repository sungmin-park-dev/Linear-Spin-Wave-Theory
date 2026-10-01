"""Spectra, topology, texture and phase figures draw stored numbers and keep undefined values visible."""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import (honeycomb_ferromagnet, polarized_state, state_120,
                                triangular_heisenberg)
from spintoolkit.observables.berry import berry_curvature, thermal_hall
from spintoolkit.observables.neutron import neutron_path, neutron_slice, powder_average
from spintoolkit.observables.thermal import thermal_quantities
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.high_symmetry import high_symmetry_points, zone_boundary
from spintoolkit.visualization import (plot_berry_curvature, plot_intensity_path,
                                       plot_intensity_slice, plot_phase_diagram, plot_powder,
                                       plot_spin_texture, plot_thermal_hall, plot_thermodynamics)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def triangle():
    model = triangular_heisenberg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model, solve_lswt(model, state_120(model), settings=LSWTSettings(mesh=(6, 6)))


def test_zone_boundary_of_the_triangular_lattice_is_the_hexagon_through_K():
    corners = zone_boundary(triangular_heisenberg().lattice)
    K = high_symmetry_points(triangular_heisenberg().lattice)["K"]
    assert len(corners) == 6
    np.testing.assert_allclose(np.linalg.norm(corners, axis=1), np.linalg.norm(K))


def test_path_map_holds_the_magnon_at_M_and_keeps_K_undefined(triangle):
    """omega(M) = 3JS sqrt((1 - g)(1 + 2g)) with g = -1/3 is 1.0 J; K is a Bragg vector."""
    _, result = triangle
    omega = np.linspace(0, 2, 201)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spectrum = neutron_path(result, omega, 0.05, path=("K", "M"), points=10, g=2.0)
    assert np.all(np.isnan(spectrum.intensity[0]))                 # Goldstone weight at K
    assert omega[np.argmax(spectrum.intensity[-1])] == pytest.approx(1.0, abs=0.01)
    ax = plot_intensity_path(spectrum)
    data = ax.collections[0].get_array()
    assert np.ma.is_masked(data) and data.mask.sum() == len(omega)  # one undefined column
    assert [t.get_text() for t in ax.get_xticklabels()] == ["K", "M"]


def test_slice_and_powder_draw_their_arrays(triangle):
    _, result = triangle
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cut = neutron_slice(result, 1.0, 0.1, points=9, g=2.0)
        Q = np.linspace(0.5, 6, 5)
        omega = np.linspace(0, 2, 11)
        powder = powder_average(result, Q, omega, 0.1, num_directions=20, g=2.0)
    ax = plot_intensity_slice(cut)
    np.testing.assert_array_equal(np.ma.filled(ax.collections[0].get_array(), np.nan).ravel(),
                                  cut.intensity.ravel())
    ax = plot_powder(Q, omega, powder)
    assert ax.collections[0].get_array().shape == powder.T.shape


def test_energy_scale_needs_a_label(triangle):
    _, result = triangle
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spectrum = neutron_path(result, [0.5, 1.0], 0.1, path=("M", "Γ"), points=4)
    with pytest.raises(ValueError, match="energy_label"):
        plot_intensity_path(spectrum, energy_scale=2.0)


def test_berry_curvature_title_lists_the_chern_numbers():
    model = honeycomb_ferromagnet(J=1.0, D=0.1)
    result = solve_lswt(model, polarized_state(model), ExternalConditions(field=(0, 0, 0.1)),
                        settings=LSWTSettings(mesh=(12, 12)))
    curvature = berry_curvature(result)
    ax = plot_berry_curvature(curvature, result.magnetic_lattice, band=0)
    chern = curvature.chern_numbers()
    np.testing.assert_allclose(chern, [1, -1], atol=0.01)
    assert ax.get_title() == f"band 0; Chern numbers {chern[0]:+.3f}, {chern[1]:+.3f}"
    hall = thermal_hall(result, [0.2, 0.5])
    ax = plot_thermal_hall(hall)
    np.testing.assert_array_equal(ax.get_lines()[0].get_ydata(), hall.kappa_over_t)


def test_texture_reports_undefined_charge_for_the_120_degree_state():
    model = triangular_heisenberg()
    ax = plot_spin_texture(model, state_120(model))
    assert ax.get_title().startswith("Q undefined")
    ax = plot_spin_texture(model, polarized_state(model))
    assert ax.get_title() == "Q = 0 per magnetic cell"


def test_phase_diagram_keeps_undefined_points_separate():
    phases = [["A", None], ["B", "A"]]
    ax = plot_phase_diagram([0.0, 1.0], [0.0, 1.0], phases)
    np.testing.assert_array_equal(ax.collections[0].get_array().reshape(2, 2), [[1, 0], [2, 1]])
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["A", "B", "undefined"]
    with pytest.raises(ValueError, match="shape"):
        plot_phase_diagram([0.0], [0.0, 1.0], phases)


def test_thermodynamics_shades_where_lswt_breaks_down():
    model = triangular_heisenberg(J=-1.0)
    result = solve_lswt(model, polarized_state(model), ExternalConditions(field=(0, 0, 0.2)))
    thermal = thermal_quantities(result, np.linspace(0.05, 3, 30))
    assert thermal.beyond_lswt.any() and not thermal.beyond_lswt.all()
    axes = plot_thermodynamics(thermal, ("specific_heat", "magnetization"))
    np.testing.assert_array_equal(axes[0].get_lines()[0].get_ydata(), thermal.specific_heat)
    assert len(axes[1].get_lines()) == 3
    assert axes[0].patches                                             # shaded interval
    with pytest.raises(ValueError, match="unknown"):
        plot_thermodynamics(thermal, ("heat",))
