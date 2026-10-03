"""Literature and closed-form benchmarks of the batch-review items A1-A4.

A1 (D45) magnetization at 1/S order, M = -d(E_cl + E_zp)/dh:
    square-lattice antiferromagnet, against the closed-form canted-state
    dispersion (Zhitomirsky and Nikuni, PRB 57, 5013 (1998)) and the 1/S
    perpendicular susceptibility chi = 1/8 - 0.034447/S (Hamer, Zheng and
    Oitmaa, PRB 50, 6877 (1994); arXiv:cond-mat/9212020).
A2 (D37) single-ion coefficient (1 - 1/2S) A: easy-axis S = 1 ferromagnet in
    a transverse field against exact diagonalization on the same 3 x 3 torus.
A3 (D29) thermal Hall: kagome ferromagnet with Dzyaloshinskii-Moriya
    interaction from an independently built Bloch matrix, Chern numbers
    (+-1, 0, -+1) (Mook, Henk and Mertig, PRB 89, 134409 (2014)); in-plane
    field polarized Kitaev-Gamma model, kappa(a) = -kappa(-a) and kappa(b) = 0
    (Chern, Zhang and Kim, PRL 126, 147201 (2021)).
A4 (D41) neutron intensity: triangular 120 degree state against the
    rotating-frame closed form (Chernyshev and Zhitomirsky, PRB 79, 144416
    (2009)) with Q_z != 0, and the Bragg weight m^2/4 at K.

The scripts that produce the full tables are in ``examples/review_benchmarks_a1_a4.py``.
"""

import numpy as np
import pytest
from scipy.special import spence

from spintoolkit.methods.classical import refine_classical
from spintoolkit.methods.ed import EDSector, solve_ed
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.methods.magnetization import magnetization_curve
from spintoolkit.models import neel_state, polarized_state, square_heisenberg, state_120
from spintoolkit.models import triangular_heisenberg
from spintoolkit.models.heisenberg import TRIANGULAR_LATTICE
from spintoolkit.models.honeycomb import NEAREST_NEIGHBOUR_OFFSETS
from spintoolkit.observables.berry import chern_numbers, thermal_hall
from spintoolkit.observables.neutron import neutron_intensity
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import Site, SpinModel, Term

DM_Z = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


# ---------------------------------------------------------------- A1 -----

def square_canted_zero_point(h, S, n=400):
    """E_zp per site of the canted square antiferromagnet (J = g = 1), closed form.

    A_k = 4S(1 + sin^2(t) g_k), B_k = -4S cos^2(t) g_k, sin t = h / 8S, with t
    the canting angle out of the plane perpendicular to the field.
    """
    s2 = (h / (8 * S)) ** 2
    k = (np.arange(n) + 0.5) / n * 2 * np.pi
    kx, ky = np.meshgrid(k, k)
    gamma = 0.5 * (np.cos(kx) + np.cos(ky))
    A, B = 4 * S * (1 + s2 * gamma), -4 * S * (1 - s2) * gamma
    return 0.5 * np.mean(np.sqrt(A * A - B * B) - A)


def square_canted_magnetization(h, S, n=400, step=1e-4):
    return h / 8 - (square_canted_zero_point(h + step, S, n)
                    - square_canted_zero_point(h - step, S, n)) / (2 * step)


@pytest.mark.parametrize("S", [0.5, 1.0])
def test_square_magnetization_equals_closed_form_canted_dispersion(S):
    model = square_heisenberg(J=1.0, S=S)
    fields = [0.1, 0.2, 1.0, 3.0]
    curve = magnetization_curve(model, neel_state(model, (1, 0, 0)), fields, k_density=48)
    closed = [square_canted_magnetization(h, S) for h in fields]
    np.testing.assert_allclose(curve.harmonic, closed, rtol=5e-3)   # k mesh at low h


@pytest.mark.parametrize("S", [0.5, 1.0, 1.5])
def test_square_perpendicular_susceptibility_matches_hamer_zheng_oitmaa(S):
    # M/h has a correction linear in h at small field; extrapolate it away.
    chi = [square_canted_magnetization(h, S, n=800) / h for h in (0.005, 0.01)]
    limit = 2 * chi[0] - chi[1]
    assert limit == pytest.approx(0.125 - 0.034447 / S, abs=2e-4)
    # The reduced-moment formula (S - <n>) / (8S) with <n> = 0.1966 misses the
    # canting-angle shift and is off by much more.
    assert abs((S - 0.1966) / (8 * S) - limit) > 0.05 * limit


def test_magnetization_is_exactly_saturated_above_the_saturation_field():
    model = square_heisenberg(J=1.0, S=0.5)
    curve = magnetization_curve(model, polarized_state(model), [4.2, 5.0], k_density=12)
    np.testing.assert_allclose(curve.harmonic, 0.5, atol=1e-8)
    np.testing.assert_allclose(curve.classical, 0.5, atol=1e-8)


# ---------------------------------------------------------------- A2 -----

NN_TRIANGULAR = [(1, 0), (0, 1), (-1, 1)]


def easy_axis_ferromagnet(S, A):
    terms = [Term.bilinear(("A", (0, 0)), ("A", o), -np.eye(3)) for o in NN_TRIANGULAR]
    terms += [Term.zeeman("A", np.eye(3)), Term.onsite("A", A)]
    return SpinModel(TRIANGULAR_LATTICE, [Site("A", (0, 0), S)], terms, {"model_id": "fm_axis"})


def test_single_ion_coherent_state_rule_against_ed_in_a_transverse_field():
    """-sum S.S - 0.5 sum (S^z)^2 - h sum S^x at half the spin-flop field, S = 1, 3 x 3 torus.

    The coherent-state coefficient (1 - 1/2S) A reproduces ED to 3e-4 per site;
    the unrenormalized coefficient A with the Holstein-Primakoff constant
    (S/2)(tr A - n.A.n) is off by 1.6e-2 (2.2e-2 for S = 3/2, 2.4e-2 for S = 2).
    """
    S, A = 1.0, np.diag([0.0, 0.0, -0.5])
    h = 0.5 * 0.5 * (2 * S - 1)
    conditions = ExternalConditions(field=(h, 0, 0))
    geometry = CalculationGeometry.finite_torus([[3, 0], [0, 3]])
    exact = solve_ed(easy_axis_ferromagnet(S, A), geometry, conditions,
                     sector=EDSector(momenta="all"), num_eigenvalues=1)
    exact = min(block.energies[0] for block in exact.blocks) / 9

    def lswt(coefficient):
        model = easy_axis_ferromagnet(S, coefficient)
        state = refine_classical(model, polarized_state(model, (0.3, 0, 1)), conditions)
        result = solve_lswt(model, state, conditions, geometry)
        n = next(iter(state.directions.values()))
        return result.ground_state_energy, n

    renormalized, _ = lswt(A)
    bare_input = A / (1 - 1 / (2 * S))          # the package then uses A itself
    bare, n = lswt(bare_input)
    bare += -(S / 2) * np.trace(bare_input) + (S / 2) * (np.trace(A) - n @ A @ n)
    assert abs(renormalized - exact) < 5e-4
    assert abs(bare - exact) > 1e-2


# ---------------------------------------------------------------- A3 -----

KAGOME = np.array([[2.0, 0.0], [1.0, np.sqrt(3)]])
KAGOME_SITES = {"A": (0.0, 0.0), "B": (0.5, 0.0), "C": (0.0, 0.5)}
#: Counterclockwise nearest-neighbour bonds i -> j of the up and down triangles.
KAGOME_BONDS = [("A", (0, 0), "B", (0, 0)), ("B", (0, 0), "C", (0, 0)), ("C", (0, 0), "A", (0, 0)),
                ("A", (0, 0), "B", (-1, 0)), ("B", (-1, 0), "C", (0, -1)),
                ("C", (0, -1), "A", (0, 0))]


def kagome_ferromagnet(J, D, S):
    terms = [Term.bilinear((i, (0, 0)), (j, tuple(np.subtract(cj, ci))), -J * np.eye(3) + D * DM_Z)
             for i, ci, j, cj in KAGOME_BONDS]
    terms += [Term.zeeman(s, np.eye(3)) for s in KAGOME_SITES]
    return SpinModel(KAGOME, [Site(s, f, S) for s, f in KAGOME_SITES.items()], terms,
                     {"model_id": "kagome_fm_dm"})


def kagome_bloch(k, J, D, S, h):
    """Magnon Bloch matrix built by hand: hopping -JS - iDS on i -> j, full positions."""
    index = {s: n for n, s in enumerate(KAGOME_SITES)}
    H = (4 * J * S + h) * np.eye(3, dtype=complex)
    for i, ci, j, cj in KAGOME_BONDS:
        r = (np.add(KAGOME_SITES[j], cj) - np.add(KAGOME_SITES[i], ci)) @ KAGOME
        t = (-J * S - 1j * D * S) * np.exp(1j * k @ r)
        H[index[i], index[j]] += t
        H[index[j], index[i]] += np.conj(t)
    return H


def c2(energy, t):
    rho = 1 / np.expm1(energy / t)
    return (1 + rho) * np.log((1 + rho) / rho) ** 2 - np.log(rho) ** 2 - 2 * spence(1 + rho)


def test_kagome_thermal_hall_from_an_independent_bloch_matrix():
    J, D, S, h, n = 1.0, 0.2, 0.5, 0.3, 24
    ts = np.array([0.1, 0.3, 1.0, 3.0])
    model = kagome_ferromagnet(J, D, S)
    result = solve_lswt(model, polarized_state(model), ExternalConditions(field=(0, 0, h)),
                        settings=LSWTSettings(mesh=(n, n)))
    reciprocal = 2 * np.pi * np.linalg.inv(KAGOME).T
    reference, step = np.zeros(len(ts)), 1e-6
    for a in range(n):
        for b in range(n):
            k = ((a + 0.5) / n) * reciprocal[0] + ((b + 0.5) / n) * reciprocal[1]
            E, U = np.linalg.eigh(kagome_bloch(k, J, D, S, h))
            dx, dy = ((kagome_bloch(k + d, J, D, S, h) - kagome_bloch(k - d, J, D, S, h)) / (2 * step)
                      for d in (np.array([step, 0]), np.array([0, step])))
            X, Y = U.conj().T @ dx @ U, U.conj().T @ dy @ U
            for m in range(3):
                omega = sum(-2 * np.imag(X[m, l] * Y[l, m]) / (E[m] - E[l]) ** 2
                            for l in range(3) if l != m)
                reference += c2(E[m], ts) * omega
    reference = -reference / n ** 2 / abs(np.linalg.det(KAGOME))
    k_test = np.array([[0.3, 0.7]])
    lswt_bands = solve_lswt(model, polarized_state(model), ExternalConditions(field=(0, 0, h)),
                            settings=LSWTSettings(k_points=k_test)).bands()[0]
    np.testing.assert_allclose(lswt_bands, np.linalg.eigvalsh(kagome_bloch(k_test[0], J, D, S, h)),
                               atol=1e-12)
    np.testing.assert_allclose(thermal_hall(result, ts).kappa_over_t, reference, rtol=1e-6)


def test_kagome_chern_numbers_match_the_literature_and_flip_with_d():
    for D, expected in ((0.2, [1, 0, -1]), (-0.2, [-1, 0, 1])):
        model = kagome_ferromagnet(1.0, D, 0.5)
        result = solve_lswt(model, polarized_state(model), ExternalConditions(field=(0, 0, 0.3)),
                            settings=LSWTSettings(mesh=(24, 24)))
        np.testing.assert_array_equal(chern_numbers(result), expected)


def kitaev_gamma(K, G, S=0.5):
    terms = []
    for gamma, offset in enumerate(NEAREST_NEIGHBOUR_OFFSETS):
        M = np.zeros((3, 3))
        M[gamma, gamma] = K
        a, b = [x for x in range(3) if x != gamma]
        M[a, b] = M[b, a] = G
        terms.append(Term.bilinear(("A", (0, 0)), ("B", offset), M))
    terms += [Term.zeeman(s, np.eye(3)) for s in ("A", "B")]
    return SpinModel(TRIANGULAR_LATTICE, [Site("A", (0, 0), S), Site("B", (1 / 3, 1 / 3), S)],
                     terms, {"model_id": "kitaev_gamma"})


def test_in_plane_field_kitaev_gamma_sign_structure():
    """a = (1,1,-2)/sqrt6 and b = (1,-1,0)/sqrt2 (along the z bond). Time reversal gives
    kappa(-a) = -kappa(a); C2 about b with the mirror exchanging x and y bonds gives
    kappa(b) = 0. K = -1, Gamma = -0.3 keeps the two bands gapped (Gamma = +0.3 leaves
    a gap of 0.006 for a and a Dirac touching for b)."""
    model = kitaev_gamma(-1.0, -0.3)
    a, b = np.array([1, 1, -2]) / np.sqrt(6), np.array([1, -1, 0]) / np.sqrt(2)
    out = {}
    for name, d in (("a", a), ("-a", -a), ("b", b)):
        conditions = ExternalConditions(field=tuple(3.0 * d))
        state = refine_classical(model, polarized_state(model, d), conditions)
        result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(48, 48)))
        out[name] = (chern_numbers(result), thermal_hall(result, [0.5, 1.0]).kappa_over_t)
    np.testing.assert_array_equal(out["a"][0], [-1, 1])
    np.testing.assert_array_equal(out["-a"][0], [1, -1])
    np.testing.assert_array_equal(out["b"][0], [0, 0])
    assert np.all(np.abs(out["a"][1]) > 1e-3)
    np.testing.assert_allclose(out["-a"][1], -out["a"][1], rtol=1e-10)
    assert np.all(np.abs(out["b"][1]) < 1e-12)


# ---------------------------------------------------------------- A4 -----

def test_triangular_120_neutron_intensity_against_rotating_frame_closed_form():
    """Modes at w(Q) with S^{y'y'}(Q)(1 - Qz^2/Q^2) (out of plane) and at w(Q -+ K) with
    S^{x'x'}(Q -+ K)(1 + Qz^2/Q^2)/4 (in plane); g = 2, F = 1, per site."""
    S = 0.5
    model = triangular_heisenberg(J=1.0, S=S)
    result = solve_lswt(model, state_120(model), settings=LSWTSettings(mesh=(12, 12)))
    bonds = np.array([[1, 0], [0.5, np.sqrt(3) / 2], [-0.5, np.sqrt(3) / 2]])

    def gamma(k):
        return np.mean(np.cos(bonds @ k))

    def omega(k):
        return 3 * S * np.sqrt((1 - gamma(k)) * (1 + 2 * gamma(k)))

    def in_plane(k):
        return S / 2 * np.sqrt((1 + 2 * gamma(k)) / (1 - gamma(k)))

    def out_of_plane(k):
        return S / 2 * np.sqrt((1 - gamma(k)) / (1 + 2 * gamma(k)))

    K = np.array([4 * np.pi / 3, 0.0])
    rng = np.random.default_rng(1)
    Q = np.c_[rng.uniform(-6, 6, (10, 2)), rng.uniform(-3, 3, 10)]
    spectrum = neutron_intensity(result, Q)
    for q, energies, weights in zip(Q, spectrum.energies, spectrum.intensities):
        qz2 = q[2] ** 2 / (q @ q)
        closed = [(omega(q[:2]), out_of_plane(q[:2]) * (1 - qz2))]
        closed += [(omega(q[:2] + s * K), in_plane(q[:2] + s * K) * (1 + qz2) / 4) for s in (1, -1)]
        particle = energies > 0
        np.testing.assert_allclose(sorted(zip(energies[particle], weights[particle])),
                                   sorted(closed), rtol=0, atol=1e-12)
    with pytest.warns(UserWarning, match="zero mode"):
        bragg = neutron_intensity(result, [[*K, 0.0]])
    moment = result.ordered_moments()[0]
    assert bragg.elastic[0] == pytest.approx(moment ** 2 / 4, rel=1e-12)
    assert np.all(np.isnan(bragg.intensities))        # Goldstone modes at the Bragg vector
