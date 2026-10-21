"""Self-consistent one-loop spin waves, ``solve_scswt`` (D72).

1. Square antiferromagnet: the Hartree-Fock term rescales H(k) uniformly, so
   the vacuum does not change and the fixed point is the one-loop result.
2. Exact diagonalization at large S (anisotropic XYZ exchange, finite torus):
   the self-consistent energy and gap differ from the exact ones by O(1/S)
   (relative 1/S^2), as the one-loop ones do.
3. The fixed point is a stationary point of the Gaussian energy
   ``<H_0 + H_2 + H_4>`` over Bogoliubov vacua, which checks that the
   decoupling is the derivative of ``<H_4>``.
4. Triangular up-up-down plateau: the down sublattice loses exactly twice the
   moment of an up sublattice (S^z_tot conservation); the field moves the
   k = 0 gaps by -+(h - 3JS) with frozen and with self-consistent
   contractions alike, and at large S the gap-closing fields approach
   Chubukov and Golosov's one-loop result 3JS - 0.5JS/(2S), 3JS + 1.3JS/(2S).
5. States with cubic vertices are refused.
"""

import numpy as np
import pytest

from spintoolkit import ExternalConditions, SpinModel, Site, Term
from spintoolkit.methods.ed import solve_ed
from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.methods.lswt.run import LSWTError
from spintoolkit.methods.nlswt import SCSWTSettings, solve_nlswt, solve_scswt
from spintoolkit.methods.nlswt.engine import Contractions, bogoliubov_mesh, wick4
from spintoolkit.methods.nlswt.expansion import expand_model
from spintoolkit.methods.nlswt.run import NLSWTSettings, _momenta, _resolution
from spintoolkit.models import (bcao_minimal, neel_state, polarized_state, square_heisenberg,
                                state_120, triangular_heisenberg)
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.geometry import CalculationGeometry

NAMBU3 = np.diag([1, 1, 1, -1, -1, -1])


def xyz_square(S, J=(0.6, -0.4, 0.3)):
    terms = [Term.bilinear(("A", (0, 0)), ("A", d), np.diag(J)) for d in [(1, 0), (0, 1)]]
    terms.append(Term.zeeman("A", np.eye(3)))
    return SpinModel(np.eye(2), (Site("A", (0.0, 0.0), S),), tuple(terms), {"model_id": "xyz"})


def uud(S):
    model = triangular_heisenberg(J=1.0, S=S)
    directions = [(0, 0, -1.0), (0, 0, 1.0), (0, 0, 1.0)]
    state = SpinState.from_function(model, [[1, 1], [-1, 2]],
                                    lambda site, cell: np.asarray(directions[(cell[0] - cell[1]) % 3]))
    return model, state


def zone_centre_energies(result):
    H = result.hamiltonian(np.zeros((1, 2)))[0]
    return np.sort(np.linalg.eigvals(NAMBU3 @ H).real)[3:]


def test_square_antiferromagnet_fixed_point_is_the_one_loop_result():
    model = square_heisenberg(J=1.0, S=0.5)
    state = neel_state(model, (1, 0, 0))
    result = solve_scswt(model, state, settings=SCSWTSettings(mesh=(16, 16)))
    one_loop = solve_nlswt(model, state, settings=NLSWTSettings(mesh=(16, 16)))
    e = result.energies
    assert e.one_loop == pytest.approx(one_loop.energies.total, abs=1e-12)
    assert e.total == pytest.approx(e.one_loop, abs=1e-8)
    k = np.array([[0.3, 0.1], [0.8, 0.8], [3.0, 0.2]])
    ratio = result.magnon_energies(k)[:, 0] / one_loop.magnon_energies(k)["total"][:, 0]
    # exact uniformity holds for a mesh closed under k -> -k; the offset mesh leaves ~1e-6
    np.testing.assert_allclose(ratio, 1.0, atol=1e-5)
    assert result.header.diagnostics["converged"]


def test_large_spin_exact_diagonalization_agrees_through_relative_order_one_over_s():
    geometry = CalculationGeometry.finite_torus([[2, 0], [0, 2]])
    diff = {}
    for S in (2, 10):
        model = xyz_square(S)
        conditions = ExternalConditions(field=(0, 0, 5.0 * S))
        state = polarized_state(model, (0, 0, 1))
        exact = solve_ed(model, geometry, conditions, num_eigenvalues=4).energies()
        result = solve_scswt(model, state, conditions, geometry=geometry)
        gap = np.nanmin(result.magnon_energies(result.k_points))
        diff[S] = (exact[0] / 4 - result.ground_state_energy, exact[1] - exact[0] - gap)
    # an error at order S^0 (energy) or relative 1/S (gap) would keep the ratio near 1
    assert diff[2][0] / diff[10][0] > 2.0
    assert diff[2][1] / diff[10][1] > 2.0


def test_gaussian_energy_is_stationary_at_the_fixed_point():
    model = xyz_square(1.0, J=(0.7, 1.0, 1.4))
    state = neel_state(model, (0, 0, 1))
    conditions = ExternalConditions(field=(0, 0, 0.3))
    settings = SCSWTSettings(mesh=(12, 12))
    result = solve_scswt(model, state, conditions, settings=settings)
    expansion = expand_model(model, state, conditions)
    k, grid, reciprocal = _momenta(model, state, CalculationGeometry.thermodynamic_limit(), settings)
    resolution = _resolution(np.mod(grid, 1.0))
    X = np.zeros((4, 4), complex)
    X[:2, :2] = X[2:, 2:] = [[1, 0.3], [0.3, 1]]
    X[:2, 2:] = X[2:, :2] = [[0.2, 0.1], [0.1, -0.2]]

    def energy(base, eps):
        contract = Contractions(bogoliubov_mesh(lambda q: base(q) + eps * X, k, grid, reciprocal,
                                                1e-9, resolution))
        quadratic = sum(c * contract(m[0], m[1]) for c, m in expansion.orders[2])
        return float(np.real(quadratic + sum(c * wick4(contract, m) for c, m in expansion.orders[4])))

    lswt = QuadraticBoseHamiltonian(model, state, conditions).at
    slope = {name: (energy(base, 1e-3) - energy(base, -1e-3)) / 2e-3
             for name, base in [("sc", result.hamiltonian),
                                ("lswt", lambda q: np.asarray(lswt(q), complex))]}
    assert abs(slope["lswt"]) > 1e-3
    assert abs(slope["sc"]) < 1e-6
    assert energy(result.hamiltonian, 0) / 2 + result.energies.classical == pytest.approx(
        result.ground_state_energy, abs=1e-12)


def test_up_up_down_plateau():
    S = 0.5
    model, state = uud(S)
    settings = SCSWTSettings(mesh=(12, 12))
    ref = solve_scswt(model, state, ExternalConditions(field=(0, 0, 3 * S)), settings=settings)
    n = ref.boson_numbers
    assert n[0] == pytest.approx(2 * n[1], abs=1e-12) and n[1] == pytest.approx(n[2], abs=1e-12)
    w = zone_centre_energies(ref)
    h = 3 * S - 0.1
    frozen = solve_scswt(model, state, ExternalConditions(field=(0, 0, h)), settings=settings,
                         frozen=ref)
    again = solve_scswt(model, state, ExternalConditions(field=(0, 0, h)), settings=settings,
                        initial=ref)
    assert frozen.header.diagnostics["mode"] == "frozen"
    np.testing.assert_allclose(again._coefficients, ref._coefficients, atol=1e-9)
    np.testing.assert_allclose(again.boson_numbers, n, atol=1e-10)
    shifted = zone_centre_energies(frozen)
    np.testing.assert_allclose(np.sort(shifted), np.sort(w + np.array([-0.1, 0.1, -0.1])), atol=1e-10)
    # gap-closing fields at S = 1/2 (regression of this implementation, not a literature value)
    h_c1, h_c2 = 3 * S - min(w[0], w[2]), 3 * S + w[1]
    assert (h_c1, h_c2) == pytest.approx((1.2665, 2.0517), abs=1e-3)
    with pytest.warns(UserWarning, match="not positive definite"):
        beyond = solve_scswt(model, state, ExternalConditions(field=(0, 0, h_c1 - 0.05)),
                             settings=settings, frozen=ref)
    assert np.isnan(beyond.ground_state_energy) and np.isnan(beyond.boson_numbers).all()
    with pytest.warns(UserWarning, match="not positive definite"):
        out = beyond.magnon_energies(np.zeros((1, 2)))
    assert np.isnan(out).all()
    with pytest.raises(LSWTError, match="LSWT vacuum does not exist"):
        solve_scswt(model, state, ExternalConditions(field=(0, 0, h)), settings=settings)


@pytest.mark.parametrize("S", [20.0])
def test_up_up_down_edges_approach_chubukov_golosov_at_large_spin(S):
    model, state = uud(S)
    result = solve_scswt(model, state, ExternalConditions(field=(0, 0, 3 * S)),
                         settings=SCSWTSettings(mesh=(24, 24)))
    w = zone_centre_energies(result)
    assert min(w[0], w[2]) * 2 == pytest.approx(0.5, abs=0.01)
    assert w[1] * 2 == pytest.approx(1.3, abs=0.03)


def test_states_with_cubic_vertices_are_refused():
    model = triangular_heisenberg(J=1.0, S=0.5)
    with pytest.raises(LSWTError, match="three-boson vertices"):
        solve_scswt(model, state_120(model))
    bcao = bcao_minimal(J3=0.45, Delta1=0.3, Delta3=0.4, Jzpm3=0.8)
    with pytest.raises(LSWTError, match="three-boson vertices"):
        solve_scswt(bcao, polarized_state(bcao, (1, 0, 0)), ExternalConditions(field=(3.0, 0, 0)))


def test_argument_checks():
    model, state = uud(0.5)
    conditions = ExternalConditions(field=(0, 0, 1.5))
    settings = SCSWTSettings(mesh=(6, 6))
    ref = solve_scswt(model, state, conditions, settings=settings)
    with pytest.raises(ValueError, match="not both"):
        solve_scswt(model, state, conditions, settings=settings, frozen=ref, initial=ref)
    other, other_state = uud(1.0)
    with pytest.raises(ValueError, match="only the field may differ"):
        solve_scswt(other, other_state, ExternalConditions(field=(0, 0, 3.0)), settings=settings,
                    frozen=ref)
    with pytest.raises(LSWTError, match="did not converge"):
        solve_scswt(model, state, conditions, settings=SCSWTSettings(mesh=(6, 6), max_iterations=1))
    with pytest.raises(ValueError):
        SCSWTSettings(mixing=0.0)
    with pytest.raises(ValueError, match="zero-temperature"):
        solve_scswt(model, state, ExternalConditions(field=(0, 0, 1.5), temperature=0.1))
    data = ref.to_json_dict()
    assert data["scswt"]["total"] == pytest.approx(ref.ground_state_energy)


# ---- cubic vertices in mean field (D73) ---------------------------------------------------------

def canted_square(S, J=(1.0, 0.7, 0.85), h_over_s=3.0):
    """Square antiferromagnet with exchange diag(J) in a field along z; Neel along x canted to z."""
    terms = [Term.bilinear(("A", (0, 0)), ("A", d), np.diag(J)) for d in [(1, 0), (0, 1)]]
    terms.append(Term.zeeman("A", np.eye(3)))
    model = SpinModel(np.eye(2), (Site("A", (0.0, 0.0), S),), tuple(terms), {"model_id": "xyz"})
    c = h_over_s / (4 * (J[0] + J[2]))
    s = np.sqrt(1 - c * c)
    state = SpinState.from_function(model, [[1, 1], [1, -1]], lambda site, cell: np.array(
        [s * (-1) ** (cell[0] + cell[1]), 0.0, c]))
    return model, state, ExternalConditions(field=(0, 0, h_over_s * S))


MEAN_FIELD = SCSWTSettings(mesh=(12, 12), cubic="mean_field")


def test_mean_field_cubic_without_cubic_vertices_is_the_plain_scheme():
    model, state = uud(0.5)
    conditions = ExternalConditions(field=(0, 0, 1.5))
    plain = solve_scswt(model, state, conditions, settings=SCSWTSettings(mesh=(12, 12)))
    mean_field = solve_scswt(model, state, conditions, settings=MEAN_FIELD)
    assert mean_field.ground_state_energy == pytest.approx(plain.ground_state_energy, abs=1e-10)
    assert max(mean_field.header.diagnostics["angle_shifts"]) == 0.0
    assert mean_field.energies.three_magnon == 0.0
    np.testing.assert_allclose(mean_field.hamiltonian(plain.k_points),
                               plain.hamiltonian(plain.k_points), atol=1e-8)


def test_mean_field_cubic_reproduces_the_one_over_s_tadpole_at_large_spin():
    """Gapped canted state (no continuous symmetry): the renormalized angle is the 1/S condensate
    of solve_nlswt, and the Gaussian energy is the one-loop energy without the three-magnon
    second-order term, both up to relative order 1/S."""
    out = {}
    for S in (5.0, 20.0):
        model, state, conditions = canted_square(S)
        one_loop = solve_nlswt(model, state, conditions, settings=NLSWTSettings(mesh=(12, 12)))
        result = solve_scswt(model, state, conditions, settings=MEAN_FIELD)
        e = one_loop.energies
        tilt = np.sqrt(2 / S) * np.abs(e.condensate[:2])          # <s^x>/S of the condensate
        np.testing.assert_allclose(result.header.diagnostics["angle_shifts"], tilt, rtol=0.5 / S)
        gaussian = e.classical + e.zero_point + e.hartree_fock + e.tadpole
        out[S] = (result.ground_state_energy - gaussian, result.energies.three_magnon - e.cubic)
        assert result.header.diagnostics["spurious_gap"] is None
        assert np.isfinite(result.magnon_energies(result.k_points[:3])).all()
    # leftovers of order 1/S: a missing order-S^0 piece would keep the ratio near 1
    assert out[5.0][0] / out[20.0][0] == pytest.approx(4.0, rel=0.05)
    assert out[5.0][1] / out[20.0][1] == pytest.approx(4.0, rel=0.1)


def test_mean_field_cubic_goldstone_mode_is_flagged():
    """U(1) about the field: without Sigma_3 the Goldstone mode acquires a spurious gap."""
    model, state, conditions = canted_square(1.0, J=(1.0, 1.0, 1.0), h_over_s=4.0)
    with pytest.warns(UserWarning, match="zero mode"):
        result = solve_scswt(model, state, conditions, settings=MEAN_FIELD)
    assert result.header.diagnostics["k0_zero_modes"] == 1
    assert result.header.diagnostics["spurious_gap"] > 0.5
    with pytest.warns(UserWarning, match="spurious gap"):
        assert np.isnan(result.magnon_energies(np.zeros((1, 2)))).all()
    assert np.isfinite(result.ground_state_energy)


def test_mean_field_cubic_field_scan_and_arguments():
    model, state, conditions = canted_square(2.0)
    ref = solve_scswt(model, state, conditions, settings=MEAN_FIELD)
    _, moved_state, moved = canted_square(2.0, h_over_s=3.2)
    fresh = solve_scswt(model, moved_state, moved, settings=MEAN_FIELD)
    again = solve_scswt(model, moved_state, moved, settings=MEAN_FIELD, initial=ref)
    assert again.ground_state_energy == pytest.approx(fresh.ground_state_energy, abs=1e-10)
    # the start need not be classically stationary: the force contains H_1
    off = solve_scswt(model, state, moved, settings=MEAN_FIELD)
    assert off.ground_state_energy == pytest.approx(fresh.ground_state_energy, abs=1e-10)
    assert np.isnan(off.energies.one_loop) and np.isfinite(fresh.energies.one_loop)
    with pytest.raises(ValueError, match="initial="):
        solve_scswt(model, moved_state, moved, settings=MEAN_FIELD, frozen=ref)
    with pytest.raises(ValueError):
        SCSWTSettings(cubic="sigma3")
    assert again.state.direction("A") @ fresh.state.direction("A") == pytest.approx(1.0, abs=1e-12)
    data = again.to_json_dict()
    assert data["scswt"]["three_magnon"] == pytest.approx(again.energies.three_magnon)
