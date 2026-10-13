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
