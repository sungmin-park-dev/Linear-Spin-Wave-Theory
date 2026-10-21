"""Zero modes on the k set, undefined thermal quantities and lab-frame spiral magnetization (D64).

Physics: at a Goldstone point with anomalous terms (B != 0) the boson number of
the mode diverges as the regularization shift goes to zero (n ~ mu^(-1/2)), so a
mesh sum that contains the point has no value; the 2D T = 0 integral itself is
finite and is reached by a mesh that avoids the point. A ferromagnet (B = 0) has
n = 0 at its zero mode. At t > 0 the 2D boson numbers of a gapless spectrum
diverge (Mermin-Wagner) and are +inf; quantities without a harmonic expansion
(non-stationary reference, k set not closed under k -> -k) are NaN.
Literature comparison: Takahashi, PRB 40, 2494 (1989) (square AFM, 2 S - 2 <n>
with <n> = 0.1966 for S = 1/2); Mermin and Wagner, PRL 17, 1133 (1966).
"""

import warnings

import numpy as np
import pytest

from spintoolkit.methods.lswt import LSWTSettings, solve_lswt, solve_spiral_lswt
from spintoolkit.models import neel_state, polarized_state, square_heisenberg
from spintoolkit.observables.berry import berry_curvature
from spintoolkit.observables.structure_factor import structure_factor
from spintoolkit.observables.thermal import thermal_quantities
from spintoolkit.observables.two_magnon import two_magnon_continuum
from spintoolkit.states.incommensurate import IncommensurateStructure
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Site, SpinModel, Term

S = 0.5


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function(*args, **kwargs)


@pytest.mark.parametrize("regularization", ["MAGSWT", "k-dependent"])
def test_regularized_goldstone_on_the_mesh_gives_no_boson_number(regularization):
    model = square_heisenberg(J=1.0)
    with pytest.warns(UserWarning, match="grow without bound"):
        result = solve_lswt(model, neel_state(model), settings=LSWTSettings(
            mesh=(16, 16), shift=False, regularization=regularization))
    assert result.header.diagnostics["divergent_zero_mode_occupation"]
    assert np.isnan(result.boson_numbers).all()
    assert np.isfinite(result.ground_state_energy)


def test_shifted_mesh_keeps_the_boson_number():
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(64, 64)))
    assert result.header.diagnostics["regularized_zero_modes"] == 0
    assert np.mean(result.boson_numbers) == pytest.approx(0.1966, abs=5e-3)   # 0.1926 at 64^2


def test_ferromagnet_zero_mode_keeps_zero_boson_number():
    model = square_heisenberg(J=-1.0)
    result = quiet(solve_lswt, model, polarized_state(model, (0, 0, 1)),
                   settings=LSWTSettings(mesh=(8, 8), shift=False, regularization="MAGSWT"))
    assert result.header.diagnostics["regularized_zero_modes"] >= 1
    assert not result.header.diagnostics["divergent_zero_mode_occupation"]
    np.testing.assert_allclose(result.boson_numbers, 0.0, atol=1e-10)


def test_gapless_boson_numbers_at_finite_temperature_are_infinite():
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(16, 16)))
    with pytest.warns(UserWarning, match=r"\+inf"):
        thermal = thermal_quantities(result, [0.0, 0.3])
    assert np.all(np.isfinite(thermal.boson_numbers[0]))
    assert np.all(np.isposinf(thermal.boson_numbers[1]))
    assert np.isnan(thermal.moments[1]).all() and np.isnan(thermal.magnetization[1]).all()
    assert np.isfinite(thermal.internal_energy[1]) and np.isfinite(thermal.free_energy[1])


def test_regularized_zero_mode_makes_free_energy_undefined_at_finite_temperature():
    model = square_heisenberg(J=1.0)
    result = quiet(solve_lswt, model, neel_state(model),
                   settings=LSWTSettings(mesh=(8, 8), shift=False, regularization="MAGSWT"))
    with pytest.warns(UserWarning, match="depend on the shift"):
        thermal = thermal_quantities(result, [0.0, 0.3], gapless=True)
    assert np.isnan(thermal.free_energy[1]) and np.isnan(thermal.entropy[1])
    assert np.isfinite(thermal.internal_energy[1]) and np.isfinite(thermal.specific_heat[1])
    # t = 0: the mesh sum is not defined; t > 0: the gapless 2D boson number diverges.
    assert np.isnan(thermal.boson_numbers[0]).all() and np.isposinf(thermal.boson_numbers[1]).all()


def tilted_state():
    model = square_heisenberg(J=1.0)
    state = SpinState.from_function(model, np.eye(2, dtype=int),
                                    lambda s, c: np.array([0.6, 0.0, 0.8]))
    result = quiet(solve_lswt, model, state, ExternalConditions(field=(0, 0, 5.0)),
                   settings=LSWTSettings(mesh=(4, 4)))
    assert not result.header.diagnostics["stationary"]
    return result


def test_thermal_quantities_of_a_non_stationary_state_are_nan():
    result = tilted_state()
    with pytest.warns(UserWarning, match="not stationary"):
        thermal = thermal_quantities(result, [0.0, 0.2])
    for column in (thermal.free_energy, thermal.internal_energy, thermal.entropy,
                   thermal.specific_heat, thermal.boson_numbers, thermal.magnetization):
        assert np.isnan(column).all()


def test_thermal_quantities_on_an_open_k_set_are_nan():
    model = square_heisenberg(J=1.0, S=S)
    field = ExternalConditions(field=(0, 0, 10.0))
    k = np.array([[0.3, 0.1], [1.0, 0.4]])
    result = solve_lswt(model, polarized_state(model, (0, 0, 1)), field,
                        settings=LSWTSettings(k_points=k))
    assert result.header.diagnostics["inversion_closed"] is False
    with pytest.warns(UserWarning, match="not closed"):
        thermal = thermal_quantities(result, [0.2])
    assert np.isnan(thermal.internal_energy).all() and np.isnan(thermal.specific_heat).all()


def test_fluctuation_observables_refuse_a_non_stationary_state():
    result = tilted_state()
    with pytest.raises(ValueError, match="not stationary"):
        structure_factor(result, [[0.3, 0.2]])
    with pytest.raises(ValueError, match="not stationary"):
        two_magnon_continuum(result, [[0.3, 0.2]], mesh=(4, 4))
    with pytest.warns(UserWarning, match="quadratic part only"):
        try:
            berry_curvature(result, [[0.3, 0.2]])
        except Exception:
            pass


def chain_model():
    J = np.eye(3)
    terms = [Term.bilinear(("A", (0, 0)), ("A", (1, 0)), J),
             Term.bilinear(("A", (0, 0)), ("A", (2, 0)), 0.5 * J),
             Term.bilinear(("A", (0, 0)), ("A", (0, 1)), -0.5 * J),
             Term.zeeman("A", np.eye(3))]
    return SpinModel(np.eye(2), [Site("A", (0, 0), S)], terms, {"model_id": "d64_chain"})


def test_conical_spiral_magnetization_is_the_lab_frame_one():
    model = chain_model()                       # q = 1/3, cone cos(theta) = 0.4 in h along z
    J0, JQ = 2.0, 2 * (np.cos(2 * np.pi / 3) + 0.5 * np.cos(4 * np.pi / 3) - 0.5)
    conditions = ExternalConditions(field=(0, 0, 0.4 * S * (J0 - JQ)))
    cone = IncommensurateStructure(model.fingerprint(), [1 / 3, 0.0], [0, 0, 1],
                                   {"A": [np.sqrt(1 - 0.16), 0, 0.4]})
    supercell = solve_lswt(model, cone.to_spin_state(model), conditions,
                           settings=LSWTSettings(mesh=(6, 6)))
    spiral = solve_spiral_lswt(model, cone, conditions, settings=LSWTSettings(mesh=(18, 6)))
    lab = quiet(thermal_quantities, supercell, [0.0], gapless=True).magnetization[0]
    rotating = quiet(thermal_quantities, spiral.rotating, [0.0], gapless=True).magnetization[0]
    np.testing.assert_allclose(rotating[:2], 0.0, atol=1e-12)
    assert rotating[2] == pytest.approx(lab[2], abs=1e-6)
    np.testing.assert_allclose(lab[:2], 0.0, atol=1e-10)


def test_off_diagonal_structure_factor_keeps_its_imaginary_part():
    # Ferromagnet along z: one magnon carries S^xy = +-i S^xx, a purely chiral weight
    # that the real part (the default) does not show (audit ARCH-01).
    model = square_heisenberg(J=-1.0)
    result = solve_lswt(model, polarized_state(model, (0, 0, 1)),
                        ExternalConditions(field=(0, 0, 0.5)), settings=LSWTSettings(mesh=(8, 8)))
    factor = structure_factor(result, [[0.7, 0.3]])
    w = np.linspace(0, 4, 401)
    xx = factor.spectrum(w, 0.05, component="xx")
    xy = factor.spectrum(w, 0.05, component="xy", part="complex")
    np.testing.assert_allclose(factor.spectrum(w, 0.05, component="xy"), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(xy), xx, atol=1e-12)
    np.testing.assert_allclose(factor.spectrum(w, 0.05, component="yx", part="imag"),
                               -xy.imag, atol=1e-12)
    with pytest.raises(ValueError, match="Cartesian component"):
        factor.spectrum(w, 0.05, part="imag")
