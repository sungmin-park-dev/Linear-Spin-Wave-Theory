"""Non-interacting longitudinal two-magnon continuum (D56, D60 at t > 0).

The equal-time sum is checked against an independent real-space Wick
contraction, <dn_r dn_r'>_c = <a_r^+ a_r'><a_r a_r'^+> + <a_r^+ a_r'^+><a_r a_r'>,
on the same torus as the momentum mesh, with thermal contractions at t > 0.
"""

import warnings

import numpy as np
import pytest

from spintoolkit import solve_lswt
from spintoolkit.methods.lswt.run import LSWTSettings
from spintoolkit.models import (neel_state, polarized_state, square_heisenberg, state_120,
                                triangular_heisenberg)
from spintoolkit.models.heisenberg import SQUARE_LATTICE
from spintoolkit.observables.two_magnon import two_magnon_continuum
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Site, SpinModel, Term

S = 0.5


def real_space_static(result, q, L, t=0.0):
    """(1/N) sum_{r r'} exp(-i q (r - r')) n_r^a n_r'^b <dn_r dn_r'>_c on an L x L torus."""
    ns = result.num_sites
    E = result.eigenvalues
    nb = np.zeros_like(E) if t == 0 else 1 / np.expm1(E / t)
    kernel = np.concatenate([1 + nb[:, :ns], nb[:, ns:]], axis=1)      # <phi phi^+>
    G = np.einsum("kmn,kn,kln->kml", result.eigenvectors, kernel, result.eigenvectors.conj())
    lattice = np.asarray(result.magnetic_lattice)
    positions = np.asarray(result.positions)
    n = result.directions
    out = np.zeros((len(q), 3, 3), dtype=complex)
    for R1 in range(L):
        for R2 in range(L):
            R = R1 * lattice[0] + R2 * lattice[1]
            for i in range(ns):
                for j in range(ns):
                    d = positions[i] - positions[j] - R
                    g = np.einsum("k,kmn->mn", result.weights * np.exp(1j * result.k_points @ d), G)
                    c = g[ns + i, ns + j] * g[i, j] + g[ns + i, j] * g[i, ns + j]
                    out += (np.exp(-1j * q @ d)[:, None, None]
                            * np.outer(n[i], n[j])[None] * c)
    return out / ns


def solved(model, state, conditions=None, L=8):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(L, L)))
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(result.magnetic_lattice)).T
    q = np.array([[0, 0], [1 / L, 0], [2 / L, 3 / L], [0.5, 0.5]]) @ reciprocal
    return result, q


def test_ferromagnet_has_no_two_magnon_weight():
    """The polarized ferromagnet is the exact vacuum: no pairs are created."""
    model = square_heisenberg(J=-1.0)
    result, q = solved(model, polarized_state(model), ExternalConditions(field=(0, 0, 0.1)))
    continuum = two_magnon_continuum(result, q, mesh=(8, 8))
    assert np.max(np.abs(continuum.static())) < 1e-25


@pytest.mark.parametrize("case", ["neel", "triangular"])
def test_equal_time_sum_equals_real_space_wick_contraction(case):
    if case == "neel":
        model, L = square_heisenberg(), 8
        state = neel_state(model)
    else:
        model, L = triangular_heisenberg(), 6
        state = state_120(model)
    result, q = solved(model, state, L=L)
    continuum = two_magnon_continuum(result, q, mesh=(L, L))
    assert np.all(continuum.excluded == 0)
    expected = real_space_static(result, q, L)
    assert np.max(np.abs(expected)) > 1e-2
    np.testing.assert_allclose(continuum.static(), expected, atol=1e-13)


def test_neel_pair_energies_are_sums_of_the_dispersion():
    """Square Neel: w(k) = 4 J S sqrt(1 - gamma_k^2), and each pair sits at w(k) + w(k + q)."""
    model = square_heisenberg()
    result, q = solved(model, neel_state(model))
    continuum = two_magnon_continuum(result, q[2:3], mesh=(8, 8))

    def w(k):
        gamma = 0.5 * (np.cos(k[:, 0]) + np.cos(k[:, 1]))
        return 4 * S * np.sqrt(1 - gamma ** 2)

    k = continuum.mesh
    expected = np.repeat(w(k) + w(-k - q[2]), 4)               # 2 x 2 band pairs per k
    np.testing.assert_allclose(continuum.energies[0], expected, atol=1e-12)
    # The longitudinal weight is along the staggered axis only.
    static = continuum.static()[0]
    assert abs(static[2, 2]) > 1e-3
    assert np.allclose(static[:2, :], 0, atol=1e-14) and np.allclose(static[:, :2], 0, atol=1e-14)


def test_spectrum_integrates_to_the_static_sum():
    model = square_heisenberg()
    result, q = solved(model, neel_state(model))
    continuum = two_magnon_continuum(result, q, mesh=(8, 8))
    omega = np.linspace(-2, 8, 4001)
    spectrum = continuum.spectrum(omega, 0.05, "gaussian", "zz")
    integral = spectrum.sum(axis=1) * (omega[1] - omega[0])
    np.testing.assert_allclose(integral, np.real(continuum.static()[:, 2, 2]), atol=1e-6)


def easy_axis(anisotropy=1.05):
    """Square XXZ antiferromagnet with an easy axis along z: a gapped Neel state."""
    exchange = np.diag([1.0, 1.0, anisotropy])
    terms = [Term.bilinear(("A", (0, 0)), ("A", off), exchange) for off in [(1, 0), (0, 1)]]
    terms.append(Term.zeeman("A", np.eye(3)))
    return SpinModel(SQUARE_LATTICE, [Site("A", (0.0, 0.0), S)], terms, {"model_id": "xxz"})


def gapped_neel(L=8):
    model = easy_axis()
    return solved(model, neel_state(model), ExternalConditions(field=(0, 0, 0.05)), L=L)


def test_finite_temperature_equal_time_sum_equals_thermal_wick_contraction():
    result, q = gapped_neel()
    for t in (0.3, 1.0):
        continuum = two_magnon_continuum(result, q, mesh=(8, 8), temperature=t)
        expected = real_space_static(result, q, 8, t)
        assert np.max(np.abs(expected - real_space_static(result, q, 8))) > 1e-2   # thermal part
        np.testing.assert_allclose(continuum.static(), expected, atol=1e-13)
        assert continuum.settings["gapless"] is False


def test_detailed_balance():
    """S(-q, -w) = exp(-w / t) S(q, w), integrated: sum W(-q) = sum W(q) exp(-e / t)."""
    result, q = gapped_neel()
    t = 0.5
    plus = two_magnon_continuum(result, q[1:3], mesh=(8, 8), temperature=t)
    minus = two_magnon_continuum(result, -q[1:3], mesh=(8, 8), temperature=t)
    zz = np.real(plus.weights[:, :, 2, 2])
    boltzmann = np.sum(zz * np.exp(-plus.energies / t), axis=1)
    np.testing.assert_allclose(np.real(minus.weights[:, :, 2, 2]).sum(axis=1), boltzmann,
                               rtol=1e-10)
    assert np.min(plus.energies) < 0 < np.max(plus.energies)                # all three channels


def test_low_temperature_limit_is_the_zero_temperature_continuum():
    result, q = gapped_neel()
    cold = two_magnon_continuum(result, q, mesh=(8, 8), temperature=1e-3)
    zero = two_magnon_continuum(result, q, mesh=(8, 8))
    np.testing.assert_allclose(cold.static(), zero.static(), atol=1e-12)


def test_ferromagnet_scatters_thermal_magnons():
    """Polarized ferromagnet: only b^+ b, S_zz(q, w) = <n_k (1 + n_k+q) delta(w - w_k+q + w_k)>."""
    model = square_heisenberg(J=-1.0)
    result, q = solved(model, polarized_state(model), ExternalConditions(field=(0, 0, 0.3)))
    t = 0.4
    continuum = two_magnon_continuum(result, q, mesh=(8, 8), temperature=t)

    def w(k):
        return 0.3 + 2 * S * (2 - np.cos(k[:, 0]) - np.cos(k[:, 1]))

    k = continuum.mesh
    nb = lambda e: 1 / np.expm1(e / t)
    for i, qi in enumerate(q):
        expected = np.mean(nb(w(k)) * (1 + nb(w(k + qi))))
        assert np.real(continuum.static()[i, 2, 2]) == pytest.approx(expected, rel=1e-12)
    assert np.max(np.abs(continuum.static()[:, :2, :])) < 1e-14


def test_gapless_spectrum_at_finite_temperature_is_undefined():
    model = square_heisenberg()
    result, q = solved(model, neel_state(model))
    with pytest.warns(UserWarning, match="gapless"):
        continuum = two_magnon_continuum(result, q, mesh=(8, 8), temperature=0.1)
    assert np.all(np.isnan(continuum.static()))
    with pytest.raises(ValueError, match="non-negative"):
        two_magnon_continuum(result, q, temperature=-0.1)
