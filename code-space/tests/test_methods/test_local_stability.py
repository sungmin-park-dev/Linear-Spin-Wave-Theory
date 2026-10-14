"""Local stability over continuous momenta with zero-mode stiffness (D51).

Values are dimensionless (E0 = |J1|, lengths in the lattice constant).
"""

import numpy as np
import pytest

from model import nbcp
from spintoolkit import solve_lswt
from spintoolkit.methods.classical import classical_search
from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.methods.stability import (MARGINAL, NOT_STATIONARY, STABLE, UNSTABLE,
                                           local_stability)
from spintoolkit.models import (neel_state, polarized_state, square_heisenberg, state_120,
                                triangular_heisenberg)
from spintoolkit.states.incommensurate import IncommensurateStructure
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Site, SpinModel, Term

S = 0.5


def ferro_with_j3(J3):
    """Square ferromagnet J1 = -1 with an antiferromagnetic third neighbour J3 along x.

    Along x the classical energy is ``-cos q + J3 cos 2q`` per spin, so the
    polarized state has stiffness ``rho_xx = S (1 - 4 J3)`` and is unstable
    for ``J3 > 1/4`` with its minimum at ``cos q* = 1 / (4 J3)``.
    """
    terms = [Term.bilinear(("A", (0, 0)), ("A", (1, 0)), -np.eye(3)),
             Term.bilinear(("A", (0, 0)), ("A", (0, 1)), -np.eye(3)),
             Term.bilinear(("A", (0, 0)), ("A", (2, 0)), J3 * np.eye(3))]
    return SpinModel(np.eye(2), [Site("A", (0, 0), S)], terms,
                     {"model_id": f"square_fm_j3_{J3}"})


def test_ferromagnet_goldstone_stiffness_is_exact():
    model = square_heisenberg(J=-1.0)
    report = local_stability(model, polarized_state(model))
    assert report.verdict == STABLE and report.is_stable
    (zero,) = report.zero_modes
    assert zero.origin == "goldstone" and zero.nullity == 2 and zero.status == "positive"
    assert zero.stiffness_min == pytest.approx(S, abs=1e-12)        # omega = |J| S k^2


def test_neel_goldstone_stiffness_is_exact():
    model = square_heisenberg(J=1.0)
    report = local_stability(model, neel_state(model))
    assert report.verdict == STABLE
    (zero,) = report.zero_modes
    assert zero.origin == "goldstone" and zero.nullity == 2
    assert zero.stiffness_min == pytest.approx(S, abs=1e-12)        # A - |B| = J S k^2


def test_stiffness_equals_small_q_limit_of_lowest_eigenvalue():
    model = triangular_heisenberg()
    state = state_120(model)
    report = local_stability(model, state)
    assert report.verdict == STABLE
    (zero,) = report.zero_modes
    assert zero.nullity == 3 and zero.origin == "goldstone"
    hamiltonian = QuadraticBoseHamiltonian(model, state)
    angles = np.linspace(0, np.pi, 37)

    def scaled(q):
        return min(np.linalg.eigvalsh(hamiltonian.at(q * np.array([np.cos(t), np.sin(t)])))[0, 0]
                   for t in angles) / q ** 2

    # The degenerate Goldstone triplet splits at third order, lambda = rho q^2 - c |q|^3,
    # so the linear correction of lambda / q^2 is removed by Richardson extrapolation.
    q = 1e-3
    assert scaled(q) == pytest.approx(zero.stiffness_min, rel=1e-3)
    assert 2 * scaled(q / 2) - scaled(q) == pytest.approx(zero.stiffness_min, rel=1e-6)


def test_polarized_antiferromagnet_below_and_above_saturation():
    model = square_heisenberg()                                     # saturation at h = 8 J S = 4
    below = local_stability(model, polarized_state(model), ExternalConditions(field=(0, 0, 3.0)))
    assert below.verdict == UNSTABLE
    assert np.allclose(below.lowest_fractional, [0.5, 0.5], atol=1e-6)
    above = local_stability(model, polarized_state(model), ExternalConditions(field=(0, 0, 5.0)))
    assert above.verdict == STABLE and above.zero_modes == ()
    assert above.lowest > 0


def test_long_wavelength_instability_that_the_lswt_mesh_misses():
    stable = local_stability(ferro_with_j3(0.24), polarized_state(ferro_with_j3(0.24)))
    assert stable.verdict == STABLE
    assert stable.zero_modes[0].stiffness_min == pytest.approx(S * (1 - 4 * 0.24), abs=1e-12)

    model = ferro_with_j3(0.2505)
    state = polarized_state(model)
    report = local_stability(model, state)
    assert report.verdict == UNSTABLE
    (zero,) = report.zero_modes
    assert zero.status == "negative"
    assert zero.stiffness_min == pytest.approx(S * (1 - 4 * 0.2505), abs=1e-12)
    assert np.mod(zero.stiffness_direction + 1e-9, np.pi) < 1e-6    # along x
    # The default shifted LSWT mesh has no point inside the unstable pocket.
    result = solve_lswt(model, state)
    assert np.all(np.isfinite(result.eigenvalues))


def test_instability_between_mesh_points_is_found():
    model = ferro_with_j3(0.3)
    report = local_stability(model, polarized_state(model), mesh=4)
    assert report.verdict == UNSTABLE
    q_star = np.arccos(1 / (4 * 0.3)) / (2 * np.pi)
    assert min(abs(report.lowest_fractional[0] - q_star),
               abs(report.lowest_fractional[0] - (1 - q_star))) < 1e-4
    assert report.lowest < 0


def test_accidental_degeneracy_is_marginal():
    """Triangular Heisenberg in a field: classical degeneracy beyond the broken U(1)."""
    model = triangular_heisenberg()
    conditions = ExternalConditions(field=(0, 0, 1.0))
    state = classical_search(model, [[1, 1], [-1, 2]], conditions).state
    report = local_stability(model, state, conditions)
    assert report.verdict == MARGINAL and not report.is_stable
    (zero,) = report.zero_modes
    assert zero.origin == "accidental" and zero.nullity > 1 and zero.status == "positive"


def test_line_of_zero_modes_is_marginal():
    terms = [Term.bilinear(("A", (0, 0)), ("A", o), np.eye(3)) for o in [(1, 0), (0, 1)]]
    terms += [Term.bilinear(("A", (0, 0)), ("A", o), 0.5 * np.eye(3)) for o in [(1, 1), (1, -1)]]
    model = SpinModel(np.eye(2), [Site("A", (0, 0), S)], terms, {"model_id": "j1j2_half"})
    report = local_stability(model, neel_state(model))
    assert report.verdict == MARGINAL
    assert any("not isolated" in reason for reason in report.reasons)


def test_non_stationary_state_is_not_judged():
    model = square_heisenberg()
    state = SpinState.from_function(model, [[1, 1], [1, -1]],
                                    lambda site, cell: (0, 0, 1) if sum(cell) % 2 == 0
                                    else (1, 0, 0), {"origin": "test"})
    report = local_stability(model, state)
    assert report.verdict == NOT_STATIONARY and report.zero_modes == ()
    assert np.isnan(report.lowest)


@pytest.mark.parametrize("couplings, h, verdict", [
    ({}, 0.40, STABLE),                       # V at J_PD = J_Gamma = 0: U(1) Goldstone
    ({"JGamma": 0.04}, 0.44, UNSTABLE),       # negative stiffness of the orbit zero mode
    ({"JPD": 0.01}, 0.05, MARGINAL),          # Y: accidental orbit degeneracy
])
def test_nbcp_three_sublattice_states(couplings, h, verdict):
    """NBCP points checked independently by relaxing long supercells (2026-10-13)."""
    model = nbcp.build_model({"Jxy": 0.076, "Jz": 0.125, **couplings})
    conditions = ExternalConditions(field=(0, 0, h))
    state = classical_search(model, nbcp.SUPERCELLS["three_msl"], conditions).state
    report = local_stability(model, state, conditions)
    assert report.verdict == verdict
    assert report.to_dict()["verdict"] == verdict


def test_negative_mode_at_the_zone_centre():
    """Ferromagnet against the field: stationary, but every k is unstable, most at k = 0."""
    model = square_heisenberg(J=-1.0)
    report = local_stability(model, polarized_state(model), ExternalConditions(field=(0, 0, -1.0)))
    assert report.verdict == UNSTABLE and report.zero_modes == ()
    assert report.lowest * report.scale == pytest.approx(-1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Beyond harmonic order (D54)
# ---------------------------------------------------------------------------

def square_xxz(J, Jz, spin=S, onsite=None):
    """Square lattice ``J (SxSx + SySy) + Jz SzSz`` with a Zeeman term and an optional onsite term."""
    terms = [Term.bilinear(("A", (0, 0)), ("A", o), np.diag([J, J, Jz])) for o in [(1, 0), (0, 1)]]
    terms.append(Term.zeeman("A", np.eye(3)))
    if onsite is not None:
        terms.append(Term.onsite("A", onsite))
    return SpinModel(np.eye(2), [Site("A", (0, 0), spin)], terms,
                     {"model_id": f"square_xxz_{J}_{Jz}_{spin}"})


def amplitude_coefficient(report):
    """Leading coefficient c of the energy change per spin c a^4 at the smallest amplitude."""
    (check,) = report.higher_order
    a = check.amplitude / 2 ** 4
    return check.lowest * report.scale / a ** 4


def test_easy_plane_ferromagnet_at_its_critical_field_rises_at_quartic_order():
    """Uniform tilt: 2 S^2 delta cos^2(t) - h S cos(t) is flat at second order for h = 4 S delta
    and rises as (S^2 delta / 2) t^4."""
    delta = 0.2
    model = square_xxz(-1.0, -1.0 + delta)
    conditions = ExternalConditions(field=(0, 0, 4 * S * delta))
    harmonic = local_stability(model, polarized_state(model), conditions, higher_order=False)
    assert harmonic.verdict == MARGINAL and harmonic.higher_order == ()
    report = local_stability(model, polarized_state(model), conditions)
    assert report.verdict == STABLE and report.reasons == ()
    (check,) = report.higher_order
    assert check.kind == "amplitude" and check.outcome == "rises"
    assert check.order == pytest.approx(4.0, abs=1e-2)
    assert amplitude_coefficient(report) == pytest.approx(S ** 2 * delta / 2, rel=1e-3)


def test_saturated_antiferromagnet_folds_the_zone_corner_mode():
    """At h_sat = 8 J S the zero mode is at (pi, pi); folded to k = 0 the canting energy
    2 J S^2 cos(2t) - h S cos(t) rises as J S^2 t^4."""
    model = square_heisenberg()
    report = local_stability(model, polarized_state(model), ExternalConditions(field=(0, 0, 4.0)))
    assert report.verdict == STABLE
    (check,) = report.higher_order
    assert np.allclose(check.fractional, [[0.5, 0.5]]) and check.spins == 4
    assert amplitude_coefficient(report) == pytest.approx(S ** 2, rel=1e-3)


def test_single_ion_antiferromagnet_falls_at_its_spinodal():
    """Easy-axis single-ion anisotropy: the Neel state softens at h = 0.9 where the energy falls
    at quartic order (the spin flop is first order)."""
    model = square_xxz(1.0, 1.0, spin=1.0, onsite=np.diag([0, 0, -0.1]))
    report = local_stability(model, neel_state(model), ExternalConditions(field=(0, 0, 0.9)))
    assert report.verdict == UNSTABLE
    (check,) = report.higher_order
    assert check.outcome == "falls" and check.order == pytest.approx(4.0, abs=1e-2)
    assert any("falls at order" in reason for reason in report.reasons)


@pytest.mark.parametrize("case", ["xxz", "triangular", "nbcp"])
def test_classically_degenerate_manifolds_stay_marginal(case):
    """Exact classical degeneracies: the XXZ antiferromagnet at its bicritical field, the
    triangular antiferromagnet in a field and the NBCP Y orbit with J_PD."""
    if case == "xxz":
        model = square_xxz(1.0, 1.3)
        conditions = ExternalConditions(field=(0, 0, 4 * S * np.sqrt(1.3 ** 2 - 1)))
        state = neel_state(model)
    elif case == "triangular":
        model, conditions = triangular_heisenberg(), ExternalConditions(field=(0, 0, 1.0))
        state = classical_search(model, [[1, 1], [-1, 2]], conditions).state
    else:
        model = nbcp.build_model({"Jxy": 0.076, "Jz": 0.125, "JPD": 0.01})
        conditions = ExternalConditions(field=(0, 0, 0.05))
        state = classical_search(model, nbcp.SUPERCELLS["three_msl"], conditions).state
    report = local_stability(model, state, conditions)
    assert report.verdict == MARGINAL
    (check,) = report.higher_order
    assert check.kind == "amplitude" and check.outcome == "flat"
    assert any("classically degenerate" in reason for reason in report.reasons)


def test_flat_stiffness_rises_at_quartic_order_in_q():
    """J3 = 1/4: rho_xx = 0 and the energy along x rises as q^4 / 8 per spin, a minimum."""
    model = ferro_with_j3(0.25)
    report = local_stability(model, polarized_state(model))
    assert report.zero_modes[0].status == "flat"
    assert report.verdict == STABLE
    (check,) = report.higher_order
    assert check.kind == "momentum" and check.outcome == "rises"
    assert check.order == pytest.approx(4.0, abs=1e-2)
    assert local_stability(model, polarized_state(model), higher_order=False).verdict == MARGINAL


# ---------------------------------------------------------------------------
# Single-Q spirals in the rotating frame (D55)
# ---------------------------------------------------------------------------

def spiral_chain(Jz_ratio=1.0, J2=0.4):
    """Square lattice J1 = 1 and J2 along x (XXZ), ferromagnetic Jy = -0.5 along y."""
    xxz = np.diag([1.0, 1.0, Jz_ratio])
    terms = [Term.bilinear(("A", (0, 0)), ("A", (1, 0)), xxz),
             Term.bilinear(("A", (0, 0)), ("A", (2, 0)), J2 * xxz),
             Term.bilinear(("A", (0, 0)), ("A", (0, 1)), -0.5 * np.eye(3))]
    return SpinModel(np.eye(2), [Site("A", (0, 0), S)], terms,
                     {"model_id": f"spiral_chain_{Jz_ratio}_{J2}"})


Q_CHAIN = np.arccos(-1 / (4 * 0.4)) / (2 * np.pi)      # cos(2 pi q) = -J1 / (4 J2)


def test_triangular_120_spiral_matches_its_supercell():
    """The 120 deg state as a spiral: one Goldstone mode at k = 0 and the two lab rotations
    perpendicular to the axis at k = +-K of the rotating frame, all positive."""
    model = triangular_heisenberg()
    spiral = IncommensurateStructure.planar(model, [1 / 3, 2 / 3], (0, 0, 1))
    report = local_stability(model, spiral)
    assert report.verdict == STABLE
    momenta = sorted(tuple(z.fractional) for z in report.zero_modes)
    assert np.allclose(momenta, [(0, 0), (1 / 3, 2 / 3), (2 / 3, 1 / 3)], atol=1e-9)
    assert all(z.origin == "goldstone" and z.status == "positive" for z in report.zero_modes)
    assert local_stability(model, state_120(model)).verdict == STABLE


def test_incommensurate_spiral_at_the_energy_minimum_is_stable():
    model = spiral_chain()
    spiral = IncommensurateStructure.planar(model, [Q_CHAIN, 0.0], (0, 0, 1))
    report = local_stability(model, spiral)
    assert report.verdict == STABLE
    assert len(report.zero_modes) == 3                 # k = 0 and +-Q of the rotating frame


def test_spiral_with_the_wrong_pitch_has_a_linear_term():
    """Torques vanish for any pitch, but dE/dq != 0 appears as a linear term at the Goldstone
    modes +-Q, and the true pitch lowers the energy."""
    model = spiral_chain()
    report = local_stability(model, IncommensurateStructure.planar(model, [0.3, 0.0], (0, 0, 1)))
    assert report.verdict == UNSTABLE
    assert any("linear term" in reason for reason in report.reasons)


def test_spiral_against_its_anisotropy():
    """Easy plane about the axis: only the U(1) Goldstone mode is left and the spiral is stable.
    Easy axis: a spiral in the hard plane is unstable."""
    model = spiral_chain(Jz_ratio=0.9)
    spiral = IncommensurateStructure.planar(model, [Q_CHAIN, 0.0], (0, 0, 1))
    report = local_stability(model, spiral)
    assert report.verdict == STABLE and len(report.zero_modes) == 1
    model = spiral_chain(Jz_ratio=1.1)
    spiral = IncommensurateStructure.planar(model, [Q_CHAIN, 0.0], (0, 0, 1))
    assert local_stability(model, spiral).verdict == UNSTABLE


def test_conical_spiral_in_an_axial_field_is_stable():
    """Cone of the J1-J2 chain: cos(theta) = h / (S (J(0) - J(Q))); the field leaves only the
    U(1) Goldstone mode about the axis."""
    model = SpinModel(spiral_chain().lattice, spiral_chain().sites,
                      list(spiral_chain().terms) + [Term.zeeman("A", np.eye(3))],
                      {"model_id": "spiral_chain_field"})
    J0 = 2 * (1.0 + 0.4 - 0.5)
    JQ = 2 * (np.cos(2 * np.pi * Q_CHAIN) + 0.4 * np.cos(4 * np.pi * Q_CHAIN) - 0.5)
    cos_theta = 0.4
    conditions = ExternalConditions(field=(0, 0, cos_theta * S * (J0 - JQ)))
    cone = IncommensurateStructure(model.fingerprint(), [Q_CHAIN, 0.0], (0, 0, 1),
                                   {"A": [np.sqrt(1 - cos_theta ** 2), 0, cos_theta]})
    report = local_stability(model, cone, conditions)
    assert report.max_torque < 1e-12
    assert report.verdict == STABLE and len(report.zero_modes) == 1


def test_twisted_easy_plane_spiral_can_be_locally_stable():
    """Without the perpendicular rotations the spins cannot leave the plane locally, so a small
    change of pitch is a metastable twist; the report says the pitch is not an extremum."""
    model = spiral_chain(Jz_ratio=0.9)
    report = local_stability(model, IncommensurateStructure.planar(model, [Q_CHAIN + 0.02, 0.0],
                                                                   (0, 0, 1)))
    assert report.verdict == STABLE
    assert np.linalg.norm(report.settings["pitch_gradient"]) > 1e-3
    assert "twisted spiral" in report.summary()
    exact = local_stability(model, IncommensurateStructure.planar(model, [Q_CHAIN, 0.0], (0, 0, 1)))
    assert np.linalg.norm(exact.settings["pitch_gradient"]) < 1e-12
    assert "twisted" not in exact.summary()
