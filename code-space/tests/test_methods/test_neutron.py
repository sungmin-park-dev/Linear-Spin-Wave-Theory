"""Unpolarized neutron intensity (D41) against exact statements.

1. With g = 2 and F = 1 at Q_z = 0 it is the polarization-projected spin
   structure factor; a scalar g and a form factor scale it by (g F / 2)^2.
2. Ferromagnet along z: one circular magnon with I = W (1 + Q_z^2 / Q^2),
   anisotropic g gives (g_x^2 (1 - Qx^2/Q^2) + g_y^2 (1 - Qy^2/Q^2)) W / 4,
   and the Bragg intensity is (g_z m / 2)^2 (1 - Q_z^2 / Q^2).
3. Spiral (rotating frame) and the commensurate supercell agree with an
   anisotropic g and Q_z != 0, inelastic and elastic.
4. Free spins in a field (flat band): the powder average of 1 + cos^2 theta
   is 4/3; the broadening conserves the integrated weight and its width.
5. Domains: the Neel state along y of a C4-symmetric compass model has
   I_y(Q) = I_x(R^{-1} Q) with R the fourfold rotation.
6. Tabulated form factors: F(0) = 1, j2 term vanishes at Q = 0, Co2+ value.
"""

import numpy as np
import pytest

from spintoolkit.methods.lswt import LSWTSettings, solve_lswt, solve_spiral_lswt
from spintoolkit.models import neel_state, polarized_state, triangular_heisenberg
from spintoolkit.observables.neutron import (
    FORM_FACTOR_COEFFICIENTS, FormFactor, domain_average, neutron_intensity, powder_average,
    sphere_directions)
from spintoolkit.observables.structure_factor import structure_factor
from spintoolkit.states.incommensurate import IncommensurateStructure
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Site, SpinModel, Term

Z = np.array([0.0, 0.0, 1.0])


def square_ferromagnet(S=1.0, J=-1.0, coupled=True):
    terms = [Term.zeeman("A", np.eye(3))]
    if coupled:
        terms += [Term.bilinear(("A", (0, 0)), ("A", o), J * np.eye(3)) for o in ((1, 0), (0, 1))]
    return SpinModel(np.eye(2), [Site("A", (0, 0), S)], terms, {"model_id": "fm"})


@pytest.fixture(scope="module")
def ferromagnet():
    model = square_ferromagnet()
    return solve_lswt(model, polarized_state(model), ExternalConditions(field=(0, 0, 0.4)),
                      settings=LSWTSettings(mesh=(4, 4)))


def test_reduces_to_spin_structure_factor_and_scales():
    model = triangular_heisenberg(J=1.0, S=0.5)
    from spintoolkit.models import state_120
    result = solve_lswt(model, state_120(model), settings=LSWTSettings(mesh=(6, 6)))
    q = np.random.default_rng(0).uniform(-5, 5, (12, 2))
    base = neutron_intensity(result, q)
    np.testing.assert_allclose(base.intensities, structure_factor(result, q).neutron(),
                               rtol=0, atol=1e-13)
    co = FormFactor.from_ion("Co2")
    scaled = neutron_intensity(result, q, g=4.2, form_factor=co, length_unit=5.3)
    factor = (4.2 / 2 * co(np.linalg.norm(q, axis=1) / 5.3)) ** 2
    np.testing.assert_allclose(scaled.intensities, factor[:, None] * base.intensities,
                               rtol=1e-12, atol=1e-14)


def test_ferromagnet_polarization_factor_and_anisotropic_g(ferromagnet):
    q = np.array([[0.3, 0.7], [1.1, -0.4], [np.pi, 0.5]])
    W = np.real(structure_factor(ferromagnet, q).weights[:, 0, 0, 0])       # S^xx of the magnon
    for qz in (0.0, 0.8, 2.5):
        Q = np.column_stack([q, np.full(len(q), qz)])
        I = neutron_intensity(ferromagnet, Q).intensities
        np.testing.assert_allclose(I[:, 0], W * (1 + qz ** 2 / np.sum(Q ** 2, 1)), rtol=1e-12)
        np.testing.assert_allclose(I[:, 1], 0, atol=1e-14)                    # T = 0: no hole
        gx, gy, gz = 1.7, 3.1, 5.0
        I = neutron_intensity(ferromagnet, Q, g=np.diag([gx, gy, gz])).intensities
        u = Q / np.linalg.norm(Q, axis=1)[:, None]
        expected = (gx ** 2 * (1 - u[:, 0] ** 2) + gy ** 2 * (1 - u[:, 1] ** 2)) * W / 4
        np.testing.assert_allclose(I[:, 0], expected, rtol=1e-12)
    # Bragg peak at G = (2 pi, 0) with Q_z: moment along z, factor 1 - Q_z^2 / Q^2.
    m = ferromagnet.spins[0] - ferromagnet.boson_numbers[0]
    for qz in (0.0, 3.0):
        Q = np.array([[2 * np.pi, 0.0, qz]])
        bragg = neutron_intensity(ferromagnet, Q, g=np.diag([2.0, 2.0, 4.0])).elastic[0]
        assert bragg == pytest.approx((4.0 / 2 * m) ** 2 * (1 - qz ** 2 / np.sum(Q ** 2)),
                                      rel=1e-12)
    assert neutron_intensity(ferromagnet, [[0.3, 0.2, 0.0]]).elastic[0] == 0


@pytest.mark.filterwarnings("ignore:H\\(.\\) has a zero mode")
def test_spiral_equals_supercell_with_anisotropic_g():
    model = triangular_heisenberg(J=1.0, S=0.5)
    spiral = IncommensurateStructure.planar(model, [1 / 3, 2 / 3], Z)
    supercell = solve_lswt(model, spiral.to_spin_state(model), settings=LSWTSettings(mesh=(6, 6)))
    # The same momenta in the primitive zone, so that the zero-point moment reductions agree.
    B = 2 * np.pi * np.linalg.inv(model.lattice).T
    shifts = [n1 * B[0] / 3 + n2 * B[1] / 3 for n1, n2 in ((0, 0), (1, 2), (2, 1))]
    k = np.concatenate([supercell.k_points + G for G in shifts])
    rotating = solve_spiral_lswt(model, spiral, settings=LSWTSettings(k_points=k))
    np.testing.assert_allclose(rotating.boson_numbers, supercell.boson_numbers.mean(), atol=1e-12)
    K = spiral.cartesian_wave_vector(model)
    q = np.vstack([np.random.default_rng(2).uniform(-6, 6, (10, 2)), K, K + B[0]])
    Q = np.column_stack([q, np.linspace(-2, 2, len(q))])
    g = np.array([[3.0, 0.4, 0.0], [0.4, 3.5, 0.2], [0.0, 0.2, 5.0]])
    kwargs = dict(g=g, form_factor=FormFactor.from_ion("Co2"), length_unit=5.3)
    a = neutron_intensity(rotating, Q, **kwargs)
    b = neutron_intensity(supercell, Q, **kwargs)
    omega = np.linspace(-3, 3, 121)
    np.testing.assert_allclose(a.broaden(omega, 0.1), b.broaden(omega, 0.1), atol=1e-10)
    np.testing.assert_allclose(a.elastic, b.elastic, atol=1e-12)
    assert np.all(a.elastic[-2:] > 0.01) and np.all(a.elastic[:-2] == 0)


def test_free_spins_powder_average_and_broadening():
    S, h = 1.0, 0.5
    model = square_ferromagnet(S, coupled=False)
    result = solve_lswt(model, polarized_state(model), ExternalConditions(field=(0, 0, h)),
                        settings=LSWTSettings(mesh=(2, 2)))
    W = S / 2                                                   # transverse weight per site
    omega = np.linspace(-2, 3, 5001)
    fwhm = 0.1
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    Qs = np.array([0.5, 1.5, 4.0])
    co = FormFactor.from_ion("Co2")
    powder = powder_average(result, Qs, omega, fwhm, num_directions=600, g=4.0,
                            form_factor=co, length_unit=5.0)
    peak = np.exp(-0.5 * ((omega - h) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    expected = (4 / 3) * W * (4.0 / 2 * co(Qs / 5.0))[:, None] ** 2 * peak[None]
    np.testing.assert_allclose(powder, expected, rtol=5e-3, atol=1e-12)
    # Broadening conserves the integrated intensity; the width may depend on energy.
    single = neutron_intensity(result, [[0.4, 0.0, 0.3]])
    for width in (0.1, lambda w: 0.05 + 0.1 * w):
        for shape in ("gaussian", "lorentzian"):
            spectrum = single.broaden(np.linspace(-100, 100, 100001), width, shape)
            total = np.nansum(single.intensities)
            assert np.sum(spectrum) * 2e-3 == pytest.approx(total, rel=2e-3 if shape ==
                                                            "lorentzian" else 1e-9)
    # Width of the Gaussian at the mode energy: peak height I / (sigma sqrt(2 pi)).
    height = single.broaden([h], lambda w: 0.05 + 0.1 * w)[0, 0]
    s = (0.05 + 0.1 * h) / (2 * np.sqrt(2 * np.log(2)))
    assert height == pytest.approx(single.intensities[0, 0] / (s * np.sqrt(2 * np.pi)), rel=1e-12)


def test_sphere_directions_are_balanced():
    d = sphere_directions(2000)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(d.mean(axis=0), 0, atol=2e-3)
    np.testing.assert_allclose(d.T @ d / len(d), np.eye(3) / 3, atol=2e-3)


def compass_model(J=1.0, K=0.3, S=1.0):
    """Square lattice, J S.S plus K S^x S^x on x bonds and K S^y S^y on y bonds (C4 with spins)."""
    terms = [Term.bilinear(("A", (0, 0)), ("A", (1, 0)), J * np.eye(3) + np.diag([K, 0, 0])),
             Term.bilinear(("A", (0, 0)), ("A", (0, 1)), J * np.eye(3) + np.diag([0, K, 0]))]
    return SpinModel(np.eye(2), [Site("A", (0, 0), S)], terms, {"model_id": "compass"})


def test_domain_average_uses_rotated_momenta():
    model = compass_model()
    # In-plane Neel directions are classically degenerate (zero mode at the zone
    # centre), so the results are built on generic momenta only.
    settings = LSWTSettings(k_points=np.random.default_rng(4).uniform(-3, 3, (6, 2)))
    along_x = solve_lswt(model, neel_state(model, (1, 0, 0)), settings=settings)
    along_y = solve_lswt(model, neel_state(model, (0, 1, 0)), settings=settings)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])       # C4 about z
    Q = np.column_stack([np.random.default_rng(3).uniform(-4, 4, (8, 2)), np.full(8, 0.7)])
    omega = np.linspace(0, 4, 81)
    direct = neutron_intensity(along_y, Q).broaden(omega, 0.2)
    rotated = domain_average(along_x, Q, omega, 0.2, [R])
    np.testing.assert_allclose(rotated, direct, atol=1e-10)
    # Equal two-domain average and weights.
    both = domain_average(along_x, Q, omega, 0.2, [np.eye(3), R], weights=[1, 3])
    plain = neutron_intensity(along_x, Q).broaden(omega, 0.2)
    np.testing.assert_allclose(both, 0.25 * plain + 0.75 * direct, atol=1e-10)
    # Not a lattice symmetry: rejected.
    c3 = np.array([[-0.5, -np.sqrt(3) / 2, 0], [np.sqrt(3) / 2, -0.5, 0], [0, 0, 1]])
    with pytest.raises(ValueError, match="lattice"):
        domain_average(along_x, Q, omega, 0.2, [c3])


def test_form_factor_table():
    for label in FORM_FACTOR_COEFFICIENTS:
        f = FormFactor.from_ion(label, j2_weight=0.5)
        assert f(0.0) == pytest.approx(1.0, abs=0.01), label
        assert f(0.0) == FormFactor.from_ion(label)(0.0)          # <j2>(0) = 0
    co = FormFactor.from_ion("Co2")
    s2 = (2.0 / (4 * np.pi)) ** 2
    expected = (0.4332 * np.exp(-14.3553 * s2) + 0.5857 * np.exp(-4.6077 * s2)
                - 0.0382 * np.exp(-0.1338 * s2) + 0.0179)
    assert co(2.0) == pytest.approx(expected, rel=1e-14)
    assert np.all(np.diff(co(np.linspace(0, 6, 50))) < 0)
    assert FormFactor.point()(3.0) == 1.0
    with pytest.raises(KeyError, match="no form factor"):
        FormFactor.from_ion("Co7")
