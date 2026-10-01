"""Classical Monte Carlo and thermal observables (D40) against exact statements.

1. Twist derivatives equal finite differences of the twisted energy, and the
   T = 0 helicity modulus of the square ferromagnet is |J| S^2.
2. Overrelaxation conserves the energy exactly.
3. Single spins with a field and a single-ion term: the sampled <n_z> equals
   the quadrature of exp(-E/T).
4. High-temperature series of the square Heisenberg model:
   <e> = -(1/T) (1/N) sum_b |J_b|_F^2 S^4 / 9 + O(T^-3) (no odd loops).
5. Monte Carlo and Langevin dynamics (independent samplers of exp(-E/T))
   agree on the energy of an interacting model with an onsite term.
"""

import numpy as np
import pytest

from spintoolkit.methods.dynamics import ClassicalTorus, Langevin, evolve
from spintoolkit.methods.monte_carlo import (
    MonteCarlo, colour_classes, order_parameter, thermal_averages, twist_derivatives, u1_part)
from spintoolkit.models import polarized_state, state_120
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import Site, SpinModel, Term, onsite_renormalization

SQUARE = np.eye(2)
TRIANGULAR = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
NN_TRIANGULAR = ((1, 0), (0, 1), (-1, 1))
NN_SQUARE = ((1, 0), (0, 1))


def torus_of(model, L, field=(0, 0, 0)):
    return ClassicalTorus(model, CalculationGeometry.finite_torus([[L, 0], [0, L]]),
                          ExternalConditions(field=field))


def square_heisenberg(J, S=1.0):
    terms = [Term.bilinear(("A", (0, 0)), ("A", o), J * np.eye(3)) for o in NN_SQUARE]
    return SpinModel(SQUARE, [Site("A", (0, 0), S)], terms + [Term.zeeman("A", np.eye(3))],
                     {"model_id": "square"})


def triangular_easy_plane(S=1.0):
    terms = [Term.bilinear(("A", (0, 0)), ("A", o), np.diag([1.0, 1.0, 0.8])) for o in NN_TRIANGULAR]
    terms += [Term.zeeman("A", np.eye(3)), Term.onsite("A", np.diag([0.0, 0.0, 0.3]))]
    return SpinModel(TRIANGULAR, [Site("A", (0, 0), S)], terms, {"model_id": "tri"})


def rotation(axis, angle):
    n = np.asarray(axis, float) / np.linalg.norm(axis)
    G = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(angle) * G + (1 - np.cos(angle)) * G @ G


def test_colour_classes_have_no_internal_bond():
    torus = torus_of(triangular_easy_plane(), 6)
    classes = colour_classes(torus)
    colour = np.empty(torus.num_sites, int)
    for c, sites in enumerate(classes):
        colour[sites] = c
    assert len(classes) <= 4                        # greedy; the minimum is 3
    assert np.all(colour[torus.bond_source] != colour[torus.bond_target])


def test_twist_derivatives_and_zero_temperature_stiffness():
    torus = torus_of(triangular_easy_plane(), 6)
    spins = torus.random_spins(2)
    axis, direction = np.array([0.0, 0.0, 1.0]), np.array([0.6, 0.8])

    def twisted(delta):
        theta = torus.bond_vectors @ direction
        return sum(si @ J @ rotation(axis, delta * t) @ sj for si, J, sj, t in
                   zip(spins[torus.bond_source], torus.bond_exchange, spins[torus.bond_target], theta))

    first, second = twist_derivatives(torus, spins, axis, direction)
    h = 1e-4
    assert first == pytest.approx((twisted(h) - twisted(-h)) / (2 * h), rel=1e-6)
    assert second == pytest.approx((twisted(h) - 2 * twisted(0) + twisted(-h)) / h ** 2, rel=1e-5)

    J, S = -1.0, 1.5
    model = square_heisenberg(J, S)
    torus = torus_of(model, 4)
    ground = torus.spins_from_state(polarized_state(model, (1, 0, 0)))
    first, second = twist_derivatives(torus, ground, (0, 0, 1), (1, 0))
    assert first == pytest.approx(0, abs=1e-12)
    assert second / torus.num_sites == pytest.approx(abs(J) * S ** 2, rel=1e-12)


def test_twist_needs_u1_exchange_or_projection():
    """An exchange that breaks U(1) about the axis is rejected; the projection keeps
    exactly the U(1)-symmetric part (a model built from it gives the same derivatives)."""
    J = np.array([[1.0, 0.3, 0.1], [0.0, 0.6, -0.2], [0.1, -0.2, 0.8]])   # Dz-like part 0.15
    terms = [Term.bilinear(("A", (0, 0)), ("A", o), J) for o in NN_TRIANGULAR]
    model = SpinModel(TRIANGULAR, [Site("A", (0, 0), 1.0)], terms, {"model_id": "anisotropic"})
    torus = torus_of(model, 6)
    spins = torus.random_spins(4)
    with pytest.raises(ValueError, match="not U\\(1\\) symmetric"):
        twist_derivatives(torus, spins)
    J_bar = u1_part(J, (0, 0, 1))
    assert J_bar == pytest.approx(np.array([[0.8, 0.15, 0], [-0.15, 0.8, 0], [0, 0, 0.8]]))
    symmetric = SpinModel(TRIANGULAR, model.sites,
                          [Term.bilinear(("A", (0, 0)), ("A", o), J_bar) for o in NN_TRIANGULAR],
                          {"model_id": "u1"})
    np.testing.assert_allclose(twist_derivatives(torus, spins, u1_projection=True),
                               twist_derivatives(torus_of(symmetric, 6), spins), rtol=1e-12)


def test_overrelaxation_conserves_energy():
    model = square_heisenberg(1.0)
    torus = torus_of(model, 6, field=(0.1, 0.2, 0.3))
    sampler = MonteCarlo(torus, 1.0, seed=0)
    spins = torus.random_spins(0)
    before = torus.energy(spins)
    sampler._overrelax(spins)
    assert torus.energy(spins) == pytest.approx(before, abs=1e-13)
    np.testing.assert_allclose(np.linalg.norm(spins, axis=1), torus.lengths, atol=1e-13)


@pytest.mark.parametrize("S, h, D, T", [(1.0, 1.0, 0.0, 0.5), (1.5, 0.5, 1.0, 0.6)])
def test_single_spins_sample_boltzmann(S, h, D, T):
    terms = [Term.zeeman("A", np.eye(3))] + ([Term.onsite("A", np.diag([0, 0, D]))] if D else [])
    model = SpinModel(SQUARE, [Site("A", (0, 0), S)], terms, {"model_id": "free"})
    torus = torus_of(model, 16, field=(0, 0, h))
    sampler = MonteCarlo(torus, T, step=1.0, seed=3)
    averages, _ = thermal_averages(sampler, torus.random_spins(3), 200, 2000,
                                   momenta={"uniform": (0.0, 0.0)})
    m_z = np.mean(averages.series["m_uniform"][:, 2].real) / S
    z = np.linspace(-1, 1, 20001)
    weight = np.exp(-(-h * S * z + onsite_renormalization(S) * D * S ** 2 * z ** 2) / T)
    assert m_z == pytest.approx(np.sum(z * weight) / np.sum(weight), abs=0.005)


def test_high_temperature_series():
    J, S, T = 1.0, 1.0, 8.0
    model = square_heisenberg(J, S)
    torus = torus_of(model, 12)
    sampler = MonteCarlo(torus, T, step=2.0, seed=5)
    averages, _ = thermal_averages(sampler, torus.random_spins(5), 200, 4000)
    series = -(1 / T) * 2 * 3 * J ** 2 * S ** 4 / 9        # two bonds per site, |J|_F^2 = 3 J^2
    assert averages.energy == pytest.approx(series, abs=max(4 * averages.energy_error, 0.003))
    assert abs(averages.energy - series) < 0.05 * abs(series)


def test_monte_carlo_agrees_with_langevin():
    model = triangular_easy_plane()
    torus = torus_of(model, 6, field=(0, 0, 0.5))
    T = 0.6
    sampler = MonteCarlo(torus, T, step=0.8, seed=11)
    averages, _ = thermal_averages(sampler, torus.random_spins(11), 500, 4000)
    langevin = Langevin(dt=0.02, damping=0.5, temperature=T, seed=12)
    spins, _ = evolve(torus, torus.random_spins(12), langevin, 2000)
    _, trajectory = evolve(torus, spins, langevin, 40000, record_every=20)
    energies = np.array([torus.energy(s) for s in trajectory])
    blocks = np.array([b.mean() for b in np.array_split(energies, 20)])
    langevin_error = blocks.std(ddof=1) / np.sqrt(len(blocks))
    tolerance = 4 * np.hypot(averages.energy_error, langevin_error) + 0.01
    assert averages.energy == pytest.approx(energies.mean(), abs=tolerance)


def test_order_parameter_of_the_120_state():
    model = triangular_easy_plane()
    torus = torus_of(model, 6)
    spins = torus.spins_from_state(state_120(model))
    K = np.array([4 * np.pi / 3, 0.0])
    m = order_parameter(torus, spins, K)
    assert np.linalg.norm(m) == pytest.approx(1 / np.sqrt(2), abs=1e-12)   # |m_K| = S / sqrt(2)
