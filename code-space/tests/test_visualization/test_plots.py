"""Stage 7 figures draw the stored numbers and never hide instabilities."""

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import neel_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.observables.bands import BandStructure, band_structure
from spintoolkit.system.conversion import to_spin_system
from spintoolkit.visualization import plot_bands, plot_spin_configuration


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def neel_bands():
    model = square_heisenberg()
    result = solve_lswt(model, neel_state(model), None, settings=LSWTSettings(mesh=(4, 4)))
    return band_structure(result, ("Γ", "X", "M", "Γ"), points=120)


def _band_lines(ax):
    return [line for line in ax.get_lines() if line.get_linestyle() == "-"]


def test_lines_are_the_stored_energies_and_zero_modes_sit_at_zero(neel_bands):
    ax = plot_bands(neel_bands)
    lines = _band_lines(ax)
    assert len(lines) == neel_bands.energies.shape[1]
    for n, line in enumerate(lines):
        np.testing.assert_array_equal(line.get_xdata(), neel_bands.distance)
        np.testing.assert_array_equal(line.get_ydata(), neel_bands.energies[:, n])
    assert neel_bands.zero_modes.any()      # Goldstone vertices Γ and M
    (markers,) = [l for l in ax.get_lines() if l.get_label() == "zero mode"]
    np.testing.assert_array_equal(markers.get_xdata(), neel_bands.distance[neel_bands.zero_modes])
    np.testing.assert_array_equal(markers.get_ydata(), 0.0)
    assert [t.get_text() for t in ax.get_xticklabels()] == list(neel_bands.labels)


def test_display_scale_is_explicit(neel_bands):
    with pytest.raises(ValueError):
        plot_bands(neel_bands, energy_scale=2.0)
    with pytest.raises(ValueError):
        plot_bands(neel_bands, energy_scale=-1.0, energy_label="E (meV)")
    ax = plot_bands(neel_bands, energy_scale=2.0, energy_label="E (meV)")
    np.testing.assert_array_equal(_band_lines(ax)[0].get_ydata(), 2.0 * neel_bands.energies[:, 0])
    assert ax.get_ylabel() == "E (meV)"


def test_unstable_points_stay_gaps_and_are_shaded():
    distance = np.linspace(0.0, 1.0, 11)
    energies = np.column_stack([distance + 1.0, distance + 2.0])
    energies[4:7] = np.nan
    bands = BandStructure(np.zeros((11, 2)), distance, energies, ["A", "B"],
                          np.array([0.0, 1.0]), np.zeros(11, bool), "primitive")
    ax = plot_bands(bands)
    for line in _band_lines(ax):
        assert np.isnan(line.get_ydata()[4:7]).all()
    (span,) = ax.patches
    x = span.get_patch_transform().transform(span.get_path().vertices)[:, 0]   # Polygon or Rectangle
    assert np.isclose(x.min(), 0.35) and np.isclose(x.max(), 0.65)


def test_spin_configuration_draws_the_state_of_the_model():
    model = triangular_heisenberg()
    state = state_120(model)
    fig, ax = plot_spin_configuration(model, state, show_polar=False)
    system = to_spin_system(model, state)
    labels = {t.get_text() for t in ax.texts}
    assert {site.label for site in system.sites} <= labels
    with pytest.raises(TypeError):
        plot_spin_configuration(model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning):
            plot_spin_configuration(system, show_polar=False)
    with pytest.raises(TypeError):
        plot_spin_configuration(system, state)
