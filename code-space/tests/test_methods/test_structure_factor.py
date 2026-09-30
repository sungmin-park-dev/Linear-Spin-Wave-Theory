"""LSWT structure factor and bond correlations (stage 4c, D26).

1. Exact: for a U(1)-polarized state the one-magnon weights equal the ED
   matrix elements |<n|S^b(q)^dagger|FM>|^2 on the same torus (DM, two-site
   basis, S = 3/2 included).
2. Neel: transverse weight S sqrt((1 - gamma)/(1 + gamma)) at w = 4JS sqrt(1 - gamma^2).
3. Sum rules on the extended mesh: inelastic average = <S (1 + 2 n)>, elastic
   (Parseval) = <|m|^2>, total = S(S+1) + <n^2>.
4. Full-position gauge: S(q + G_primitive) = S(q), S(q + G_magnetic) != S(q).
5. Detailed balance at t > 0; Bragg intensities; bond energy = E_GS.
"""

import numpy as np
import pytest

from model import nbcp
from spintoolkit.methods.ed import EDSector, solve_ed
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.observables.structure_factor import (
    bond_correlations, spin_correlation, structure_factor, to_ladder)
from spintoolkit.observables.thermal import ZeroModeCandidateError
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.cluster import allowed_momenta, expand_on_torus
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from tests.test_methods.test_ed import CASES
from tests.test_methods.test_thermal import nbcp_y, square_model


def ed_weights(model, L, h):
    """ED weights W^{ab} (a, b in x, y) and excitation energies of every one-magnon state."""
    geometry = CalculationGeometry.finite_torus(L)
    cluster = expand_on_torus(model, geometry)
    result = solve_ed(model, geometry, ExternalConditions(field=(0, 0, h)),
                      EDSector(axis=(0, 0, 1), magnon_number=1), return_vectors=True)
    block = result.blocks[0]
    energies, vectors = result.excitations(block), block.vectors
    _, k = allowed_momenta(model, geometry)
    out = []
    for q in k:
        # <i| S^x(q)^dagger |FM>, <i| S^y(q)^dagger |FM>: S^-_i |FM> = sqrt(2S) |i>
        vx = np.sqrt(2 * cluster.spins) / 2 * np.exp(1j * cluster.positions @ q) / np.sqrt(cluster.num_sites)
        amplitudes = np.array([vectors.conj().T @ vx, vectors.conj().T @ (1j * vx)])
        out.append(np.einsum("an,bn->nab", amplitudes.conj(), amplitudes))
    return k, energies, out


@pytest.mark.parametrize("name", ["square_S1/2", "square_DM", "honeycomb_two_sites",
                                  "triangular_S3/2_nondiagonal"])
def test_one_magnon_weights_equal_ed(name):
    model, L, h, axis = CASES[name]
    k, energies, weights = ed_weights(model, L, h)
    state = SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: np.array([0, 0, 1.0]))
    lswt = solve_lswt(model, state, ExternalConditions(field=(0, 0, h)),
                      CalculationGeometry.finite_torus(L))
    sf = structure_factor(lswt, k)
    for i, W in enumerate(weights):
        for n, w in enumerate(sf.energies[i]):
            if w <= 0:
                continue
            same = np.abs(energies - w) < 1e-9
            np.testing.assert_allclose(sf.weights[i, n, :2, :2], W[same].sum(axis=0), atol=1e-14)
        np.testing.assert_allclose(sf.weights[i].sum(axis=0)[:2, :2], W.sum(axis=0), atol=1e-14)
        np.testing.assert_allclose(sf.weights[i][:, 2, :], 0, atol=1e-15)


def test_neel_transverse_weights():
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(8, 8)))
    q = np.array([[0.3, 0.7], [1.1, -0.4], [2.0, 2.9], [-2.2, 0.5]])
    sf = structure_factor(result, q)
    for i, qi in enumerate(q):
        gamma = 0.5 * (np.cos(qi[0]) + np.cos(qi[1]))
        particle = sf.energies[i] > 0
        np.testing.assert_allclose(sf.energies[i][particle], 2 * np.sqrt(1 - gamma ** 2), atol=1e-12)
        transverse = np.real(sf.weights[i][particle][:, 0, 0] + sf.weights[i][particle][:, 1, 1]).sum()
        assert transverse == pytest.approx(0.5 * np.sqrt((1 - gamma) / (1 + gamma)), abs=1e-12)
        assert np.real(sf.weights[i][:, 2, 2]).sum() == pytest.approx(0, abs=1e-15)


def extended_momenta(result):
    """The mesh translated by every magnetic reciprocal vector modulo the primitive ones."""
    magnetic = 2 * np.pi * np.linalg.inv(result.magnetic_lattice).T
    reps = {}
    for m in np.ndindex(7, 7):
        G = (np.array(m) - 3) @ magnetic          # rows of magnetic are reciprocal vectors
        p = np.round(np.mod(G @ result.lattice.T / (2 * np.pi), 1.0), 9) % 1.0
        reps.setdefault(tuple(p), G)
    return np.concatenate([result.k_points + G for G in reps.values()]), list(reps.values())


@pytest.mark.parametrize("model, state", [
    (square_heisenberg(J=1.0), neel_state), (triangular_heisenberg(J=1.0), state_120)],
    ids=["square_neel", "triangular_120"])
def test_sum_rules_on_the_extended_mesh(model, state):
    result = solve_lswt(model, state(model), settings=LSWTSettings(mesh=(6, 6)))
    q, bragg_vectors = extended_momenta(result)
    sf = structure_factor(result, q)
    n = result.boson_numbers
    inelastic = np.mean(sf.trace().sum(axis=1))
    assert inelastic == pytest.approx(np.mean(result.spins * (1 + 2 * n)), abs=1e-13)
    with pytest.warns(UserWarning, match="zero mode"):
        at_bragg = structure_factor(result, bragg_vectors)       # Goldstone modes sit here
    assert np.all(at_bragg.zero_mode) and np.all(np.isnan(at_bragg.energies))
    elastic = np.sum(np.real(np.trace(at_bragg.elastic, axis1=1, axis2=2)))
    assert elastic == pytest.approx(np.mean((result.spins - n) ** 2), abs=1e-13)
    total = inelastic + elastic
    assert total == pytest.approx(np.mean(result.spins * (result.spins + 1) + n ** 2), abs=1e-13)


def test_periodic_only_under_primitive_reciprocal_vectors():
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(4, 4)))
    q = np.array([[0.37, -0.81]])
    base = structure_factor(result, q).static()
    shifted = structure_factor(result, q + [2 * np.pi, 0]).static()
    magnetic = structure_factor(result, q + [np.pi, np.pi]).static()
    np.testing.assert_allclose(shifted, base, atol=1e-13)
    assert np.max(np.abs(magnetic - base)) > 1e-2


def test_detailed_balance_at_finite_temperature():
    model, L, h, _ = CASES["square_DM"]
    state = polarized_state(model)
    result = solve_lswt(model, state, ExternalConditions(field=(0, 0, h)), settings=LSWTSettings(mesh=(6, 6)))
    t, q = 0.7, np.array([[0.4, -1.3]])
    plus, minus = structure_factor(result, q, t), structure_factor(result, -q, t)
    # S^{ab}(q, -w) = exp(-w/t) S^{ba}(-q, w), mode by mode
    for n, w in enumerate(plus.energies[0]):
        if w >= 0 or np.linalg.norm(plus.weights[0, n]) < 1e-14:
            continue
        m = np.argmin(np.abs(minus.energies[0] + w))
        np.testing.assert_allclose(plus.weights[0, n], np.exp(w / t) * minus.weights[0, m].T,
                                   atol=1e-13)


def test_bragg_intensities():
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(8, 8)))
    with pytest.warns(UserWarning, match="zero mode"):
        sf = structure_factor(result, [[np.pi, np.pi], [0.0, 0.0], [0.3, 0.2]])
    assert list(sf.zero_mode) == [True, True, False]
    m = 0.5 - result.boson_numbers[0]
    assert list(sf.bragg) == [True, True, False]
    assert np.real(sf.elastic[0, 2, 2]) == pytest.approx(m ** 2, abs=1e-14)
    np.testing.assert_allclose(sf.elastic[1], 0, atol=1e-15)


@pytest.mark.parametrize("case", ["square", "triangular", "nbcp_y_pd"])
def test_bond_correlations_reproduce_the_ground_state_energy(case):
    if case == "square":
        model = square_heisenberg(J=1.0)
        result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(10, 10)))
    elif case == "triangular":
        model = triangular_heisenberg(J=1.0)
        result = solve_lswt(model, state_120(model), settings=LSWTSettings(mesh=(10, 10)))
    else:
        model, state, conditions = nbcp_y({"JPD": 0.01})
        result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(10, 10)))
    bonds = bond_correlations(result, model)
    assert bonds["energy"] == pytest.approx(result.ground_state_energy, abs=1e-14)
    assert len(bonds["bonds"]) == len(model.terms_of_kind("bilinear")) * len(result.site_keys) // len(model.sites)


def test_finite_temperature_follows_the_zero_mode_policy():
    model = square_model(1e-9)
    result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(6, 6)))
    with pytest.raises(ZeroModeCandidateError):
        structure_factor(result, [[0.3, 0.2]], temperature=0.1)
    sf = structure_factor(result, [[0.3, 0.2]], temperature=0.1, gapless=False)
    assert sf.gapless is False and np.all(np.isfinite(sf.trace()))


def test_spectrum_and_json():
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(4, 4)))
    sf = structure_factor(result, [[0.3, 0.2], [1.0, 2.0]])
    omega = np.linspace(-1, 4, 4001)
    spectrum = sf.spectrum(omega, eta=0.02, shape="gaussian")
    np.testing.assert_allclose(np.trapezoid(spectrum, omega, axis=1), sf.trace().sum(axis=1), rtol=1e-6)
    assert np.all(np.isfinite(sf.neutron()))
    data = sf.to_json_dict()
    assert set(data) >= {"q", "energies", "weights", "elastic", "bragg"}


def test_real_space_correlations_are_the_fourier_pair_of_the_static_structure_factor():
    """On an N x N magnetic mesh the correlations sum back to S(k) at the mesh momenta."""
    model = triangular_heisenberg(J=1.0)
    state = state_120(model)
    result = solve_lswt(model, state, settings=LSWTSettings(mesh=(4, 4)))
    static = structure_factor(result, result.k_points).static()
    supercell = np.rint(result.magnetic_lattice @ np.linalg.inv(result.lattice)).astype(int)
    ns, k = result.num_sites, result.k_points[5]
    total = np.zeros((3, 3), dtype=complex)
    for site_i, cell_i in result.site_keys:
        for site_j, cell_j in result.site_keys:
            for R in np.ndindex(4, 4):
                shift = np.array(R) @ supercell
                cell = (cell_j[0] + shift[0], cell_j[1] + shift[1])
                c = spin_correlation(result, model, (site_i, cell_i), (site_j, cell))
                dr = model.cartesian_position(site_i, cell_i) - model.cartesian_position(site_j, cell)
                total += np.exp(-1j * k @ dr) * c["fluctuation"]
    np.testing.assert_allclose(total / ns, static[5], atol=1e-13)


def test_spin_correlation_symmetry_and_bond_consistency():
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(8, 8)))
    a = spin_correlation(result, model, ("A", (0, 0)), ("A", (2, 1)))
    b = spin_correlation(result, model, ("A", (2, 1)), ("A", (0, 0)))
    np.testing.assert_allclose(a["total"], b["total"].T, atol=1e-14)   # different sites commute
    bond = bond_correlations(result, model)["bonds"][0]
    first = spin_correlation(result, model, ("A", (0, 0)), ("A", (1, 0)))
    np.testing.assert_allclose(np.real(first["total"]), bond["correlation"], atol=1e-14)
    assert np.real(np.trace(first["total"])) < 0                     # antiferromagnetic bond


def test_ladder_components_of_a_polarized_ferromagnet():
    model = square_heisenberg(J=-1.0)
    result = solve_lswt(model, polarized_state(model), ExternalConditions(field=(0, 0, 1.0)),
                        settings=LSWTSettings(mesh=(6, 6)))
    sf = structure_factor(result, [[0.4, -0.9]])
    ladder = to_ladder(sf.weights[0].sum(axis=0))
    # only S^+ (S^+)^dagger = S^+ S^- connects |FM> to one-magnon states: weight 2S
    assert np.real(ladder[0, 0]) == pytest.approx(1.0, abs=1e-14)
    ladder[0, 0] = 0
    np.testing.assert_allclose(ladder, 0, atol=1e-14)
