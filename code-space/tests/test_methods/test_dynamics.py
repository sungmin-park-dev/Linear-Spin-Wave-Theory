"""Classical spin dynamics (D38) against exact statements.

1. Implicit midpoint conserves every spin length and the energy (exactly, up
   to the fixed-point tolerance) and precesses a free spin at the exact
   midpoint frequency ``(2/dt) arctan(h dt/2)``.
2. Langevin dynamics samples ``exp(-E/T)``: single spins in a field (Langevin
   function) and with a single-ion term (quadrature), which also checks the
   coherent-state factor of the onsite term.
3. Small oscillations about a classical ground state have the LSWT
   frequencies of the same classical energy (onsite term included).
4. The structure factor obeys its sum rule and peaks at the LSWT dispersion.
"""

import numpy as np
import pytest

from spintoolkit.methods.classical import classical_energy, refine_classical
from spintoolkit.methods.dynamics import (
    ClassicalTorus, ImplicitMidpoint, Langevin, classical_structure_factor, evolve, thermal_samples)
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import polarized_state, state_120
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import Site, SpinModel, Term, onsite_renormalization

SQUARE = np.eye(2)
TRIANGULAR = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
NN_TRIANGULAR = ((1, 0), (0, 1), (-1, 1))
NN_SQUARE = ((1, 0), (0, 1))


def free_spins(S, onsite=None):
    terms = [Term.zeeman("A", np.eye(3))] + ([Term.onsite("A", onsite)] if onsite is not None else [])
    return SpinModel(SQUARE, [Site("A", (0, 0), S)], terms, {"model_id": "free"})


def easy_plane_triangular():
    terms = [Term.bilinear(("A", (0, 0)), ("A", o), np.eye(3)) for o in NN_TRIANGULAR]
    terms += [Term.zeeman("A", np.eye(3)), Term.onsite("A", np.diag([0.0, 0.0, 0.3]))]
    return SpinModel(TRIANGULAR, [Site("A", (0, 0), 1.0)], terms, {"model_id": "tri_easy_plane"})


def test_torus_energy_matches_classical_energy():
    model = easy_plane_triangular()
    geometry = CalculationGeometry.finite_torus([[3, 0], [0, 3]])
    conditions = ExternalConditions(field=(0.1, 0.0, 0.4))
    state = state_120(model)
    torus = ClassicalTorus(model, geometry, conditions)
    spins = torus.spins_from_state(state)
    assert torus.energy(spins) == pytest.approx(
        classical_energy(model, state, conditions, geometry), abs=1e-14)
    # Field = -dE/ds by finite differences on a random configuration.
    spins = torus.random_spins(1)
    h = torus.fields(spins)
    eps = 1e-6
    for i, a in [(0, 0), (4, 2), (7, 1)]:
        plus, minus = spins.copy(), spins.copy()
        plus[i, a] += eps
        minus[i, a] -= eps
        numeric = -(torus.energy(plus) - torus.energy(minus)) * torus.num_sites / (2 * eps)
        assert h[i, a] == pytest.approx(numeric, abs=1e-7)


def test_midpoint_conserves_length_and_energy_and_precesses_exactly():
    model = easy_plane_triangular()
    torus = ClassicalTorus(model, CalculationGeometry.finite_torus([[3, 0], [0, 3]]),
                           ExternalConditions(field=(0.0, 0.2, 0.5)))
    spins = torus.random_spins(3)
    final, trajectory = evolve(torus, spins, ImplicitMidpoint(0.05), 2000, record_every=100)
    energies = [torus.energy(s) for s in trajectory] + [torus.energy(final)]
    assert np.ptp(energies) < 1e-11
    np.testing.assert_allclose(np.linalg.norm(final, axis=1), torus.lengths, atol=1e-12)

    h, dt, steps = 1.3, 0.1, 400
    free = ClassicalTorus(free_spins(1.0), CalculationGeometry.finite_torus([[1, 0], [0, 1]]),
                          ExternalConditions(field=(0, 0, h)))
    s0 = np.array([[1.0, 0.0, 0.0]])
    s1, _ = evolve(free, s0, ImplicitMidpoint(dt), steps)
    angle = steps * 2 * np.arctan(h * dt / 2)
    # ds/dt = s x h with h along +z turns the spin clockwise about z.
    np.testing.assert_allclose(s1[0], [np.cos(angle), -np.sin(angle), 0.0], atol=1e-11)


def langevin_average(model, field, temperature, steps=6000, seed=7):
    torus = ClassicalTorus(model, CalculationGeometry.finite_torus([[16, 0], [0, 16]]),
                           ExternalConditions(field=field))
    langevin = Langevin(dt=0.02, damping=0.5, temperature=temperature, seed=seed)
    spins, _ = evolve(torus, torus.random_spins(seed), langevin, 1000)
    _, trajectory = evolve(torus, spins, langevin, steps, record_every=20)
    return np.mean(trajectory[..., 2] / torus.lengths[None, :])


def test_langevin_samples_the_langevin_function():
    S, h, T = 1.0, 1.0, 0.5
    x = S * h / T
    exact = 1 / np.tanh(x) - 1 / x
    assert langevin_average(free_spins(S), (0, 0, h), T) == pytest.approx(exact, abs=0.01)


def test_langevin_samples_single_ion_distribution():
    """E(n_z) = -h S n_z + kappa D S^2 n_z^2 + const; the measure is uniform in n_z."""
    S, h, D, T = 1.5, 0.5, 1.0, 0.6
    kappa = onsite_renormalization(S)
    z = np.linspace(-1, 1, 20001)
    weight = np.exp(-(-h * S * z + kappa * D * S ** 2 * z ** 2) / T)
    exact = np.sum(z * weight) / np.sum(weight)   # uniform grid
    sampled = langevin_average(free_spins(S, np.diag([0.0, 0.0, D])), (0, 0, h), T)
    assert sampled == pytest.approx(exact, abs=0.01)
    # The large-S (unrenormalized) value is clearly different.
    weight_large = np.exp(-(-h * S * z + D * S ** 2 * z ** 2) / T)
    assert abs(np.sum(z * weight_large) / np.sum(weight_large) - exact) > 0.04


def peak_frequencies(signal, dt, threshold):
    """Frequencies of the local maxima of a windowed power spectrum above ``threshold``."""
    power = np.abs(np.fft.rfft(np.hanning(len(signal))[:, None] * signal, axis=0)) ** 2
    power = power.sum(axis=1)
    w = 2 * np.pi * np.fft.rfftfreq(len(signal), d=dt)
    peaks = []
    for m in range(1, len(power) - 1):
        if power[m] > power[m - 1] and power[m] >= power[m + 1] and power[m] > threshold * power.max():
            # Quadratic interpolation of the log power around the maximum.
            a, b, c = np.log(power[m - 1:m + 2])
            peaks.append(w[m] + 0.5 * (a - c) / (a - 2 * b + c) * (w[1] - w[0]))
    return np.array(peaks)


def test_small_oscillations_have_lswt_frequencies():
    model = easy_plane_triangular()
    geometry = CalculationGeometry.finite_torus([[3, 0], [0, 3]])
    state = refine_classical(model, state_120(model))
    torus = ClassicalTorus(model, geometry)
    # All torus momenta (folded onto the magnetic zone); the magnetic zone centres
    # are moved off the Goldstone zero by 1e-4 so that H(k) stays positive definite
    # (the finite modes move by O(1e-8), far below the frequency resolution).
    k = torus.momenta.copy()
    magnetic = state.supercell @ model.lattice
    centre = np.all(np.abs(np.exp(1j * k @ magnetic.T) - 1) < 1e-9, axis=1)
    k[centre] += (1e-4, 0.0)
    lswt = solve_lswt(model, state, settings=LSWTSettings(k_points=k))
    bands = np.unique(np.round(lswt.bands().ravel(), 6))
    bands = bands[bands > 1e-2]                       # Goldstone mode has no oscillation
    ground = torus.spins_from_state(state)
    rng = np.random.default_rng(0)
    kick = 1e-4 * rng.normal(size=ground.shape)
    start = torus.project(ground + kick - np.sum(kick * ground, 1, keepdims=True) * ground)
    dt, steps = 0.05, 8000
    _, trajectory = evolve(torus, start, ImplicitMidpoint(dt), steps, record_every=1)
    # Out-of-plane components: every finite mode moves them, while the U(1) zero
    # mode (uniform rotation about z, excited by the kick) leaves them unchanged.
    out_of_plane = trajectory[:, :, 2] - trajectory[:, :, 2].mean(axis=0)
    observed = peak_frequencies(out_of_plane, dt, 1e-6)
    observed = 2 / dt * np.tan(observed * dt / 2)      # undo the midpoint frequency map
    resolution = 2 * np.pi / (steps * dt)
    assert len(observed) == len(bands)
    for w in bands:
        assert np.min(np.abs(observed - w)) < 0.05 * resolution, w


def test_structure_factor_sum_rule_and_lswt_dispersion():
    """Square-lattice ferromagnet, S = 1, easy-axis anisotropy, field along z, low T."""
    S, J, D, h = 1.0, -1.0, -0.2, 0.5
    terms = [Term.bilinear(("A", (0, 0)), ("A", o), J * np.eye(3)) for o in NN_SQUARE]
    terms += [Term.zeeman("A", np.eye(3)), Term.onsite("A", np.diag([0.0, 0.0, D]))]
    model = SpinModel(SQUARE, [Site("A", (0, 0), S)], terms, {"model_id": "fm_square"})
    geometry = CalculationGeometry.finite_torus([[4, 0], [0, 4]])
    conditions = ExternalConditions(field=(0, 0, h))
    torus = ClassicalTorus(model, geometry, conditions)
    langevin = Langevin(dt=0.02, damping=0.3, temperature=0.02, seed=1)
    samples = thermal_samples(torus, langevin, num_samples=4, thermalization_steps=1500,
                              decorrelation_steps=300, initial=torus.spins_from_state(
                                  polarized_state(model)))
    dt, steps = 0.05, 2048
    sqw = classical_structure_factor(torus, samples, dt, steps)
    # Sum rule: sum_w S(q, w) dw/2pi equals the window-weighted <|s_q|^2>/N.
    weights = np.hanning(steps)
    expected = np.zeros(len(torus.momenta))
    phases = np.exp(-1j * torus.momenta @ torus.positions.T)
    for sample in samples:
        _, traj = evolve(torus, sample, ImplicitMidpoint(dt), steps, record_every=1)
        s_q = np.einsum("qi,tia->tqa", phases, traj)
        expected += np.einsum("t,tqa->q", weights ** 2, np.abs(s_q) ** 2) / np.sum(weights ** 2)
    expected /= len(samples) * torus.num_sites
    np.testing.assert_allclose(np.trace(sqw.equal_time(), axis1=1, axis2=2).real, expected,
                               rtol=1e-10)
    # Transverse peak at the LSWT frequency of every torus momentum.
    lswt = solve_lswt(model, polarized_state(model), conditions, geometry)
    resolution = 2 * np.pi / (steps * dt)
    positive = sqw.frequencies > 0
    for q, k in enumerate(sqw.momenta):
        m = np.argmin(np.linalg.norm(np.mod(lswt.k_points - k + np.pi, 2 * np.pi) - np.pi, axis=1))
        transverse = np.real(sqw.intensity[q, :, 0, 0] + sqw.intensity[q, :, 1, 1])
        w_peak = sqw.frequencies[positive][np.argmax(transverse[positive])]
        w_peak = 2 / dt * np.tan(w_peak * dt / 2)
        assert abs(w_peak - lswt.bands()[m, 0]) <= resolution, (k, w_peak, lswt.bands()[m, 0])
