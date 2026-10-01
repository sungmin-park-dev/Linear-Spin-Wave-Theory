"""Harmonic free energy along the orbit (D39).

Checks against exact statements for the polarized square-lattice
antiferromagnet above saturation, where the LSWT energies are known in
closed form, ``eps(k) = h - 4 J S + 2 J S (cos kx + cos ky)``:

1. the thermal term equals ``T/N sum_k ln(1 - exp(-eps/T))`` on the mesh;
2. ``-dF/dT`` equals the Bose entropy of the same modes;
3. the soft cutoff leaves out the lowest mode at the momenta inside it;
4. an unstable reference state (field below saturation) gives ``nan``.
"""

import numpy as np
import pytest

from spintoolkit.methods import state_selection as sel
from spintoolkit.models import polarized_state, square_heisenberg
from spintoolkit.system.conditions import ExternalConditions

J, S, N = 1.0, 0.5, 8


def polarized(h):
    model = square_heisenberg(J=J, S=S)
    return model, polarized_state(model), ExternalConditions(field=(0, 0, h))


def mesh_energies(provider, h):
    k = next(iter(provider._cache.values()))
    k = k[np.linalg.norm(k, axis=1) > 1e-12]          # no axis: the zone centre is dropped
    return k, h - 4 * J * S + 2 * J * S * (np.cos(k[:, 0]) + np.cos(k[:, 1]))


def test_thermal_term_matches_closed_form():
    h, T = 4.5, 0.3
    model, state, conditions = polarized(h)
    zero_point = sel.lswt_zero_point_energy("Tetra", N)(model, state, conditions)
    provider = sel.LSWTHarmonicFreeEnergy("Tetra", N, T)
    free = provider(model, state, conditions)
    _, eps = mesh_energies(provider, h)
    exact = T * np.mean(np.log1p(-np.exp(-eps / T)))
    assert free - zero_point == pytest.approx(exact, rel=1e-12, abs=1e-15)
    assert provider.describe()["undefined_calls"] == 0


def test_entropy_is_minus_temperature_derivative():
    h, T, dT = 4.5, 0.4, 1e-5
    model, state, conditions = polarized(h)
    F = [sel.LSWTHarmonicFreeEnergy("Tetra", N, t)(model, state, conditions) for t in (T - dT, T + dT)]
    provider = sel.LSWTHarmonicFreeEnergy("Tetra", N, T)
    provider(model, state, conditions)
    _, eps = mesh_energies(provider, h)
    n = 1 / np.expm1(eps / T)
    entropy = np.mean((n + 1) * np.log1p(n) - n * np.log(n))
    assert -(F[1] - F[0]) / (2 * dT) == pytest.approx(entropy, rel=1e-7)


def test_soft_cutoff_leaves_out_lowest_mode():
    h, T, cutoff = 4.5, 0.3, 1.2
    model, state, conditions = polarized(h)
    full = sel.LSWTHarmonicFreeEnergy("Tetra", N, T)
    cut = sel.LSWTHarmonicFreeEnergy("Tetra", N, T, soft_cutoff=cutoff)
    difference = cut(model, state, conditions) - full(model, state, conditions)
    k, eps = mesh_energies(full, h)
    inside = np.linalg.norm(k, axis=1) < cutoff
    assert inside.any()
    assert difference == pytest.approx(-T * np.sum(np.log1p(-np.exp(-eps[inside] / T))) / len(k),
                                       rel=1e-12)
    assert cut.describe()["soft_modes_left_out"] == [int(inside.sum())]


def test_unstable_state_has_no_harmonic_free_energy():
    model, state, conditions = polarized(1.0)          # below saturation (8 J S = 4)
    provider = sel.LSWTHarmonicFreeEnergy("Tetra", N, 0.3)
    assert np.isnan(provider(model, state, conditions))
    assert provider.describe()["undefined_calls"] == 1
    with pytest.raises(ValueError):
        sel.LSWTHarmonicFreeEnergy("Tetra", N, 0.0)
