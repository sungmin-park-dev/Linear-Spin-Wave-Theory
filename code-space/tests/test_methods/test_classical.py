"""Classical energy, local fields and torques from the common SpinModel.

The same functions read the square, triangular and NBCP models without any
model-specific branch. The NBCP fixture is the one-site primitive cell built
from the existing exchange matrices; its classical energy must equal the
existing EnergyFunction on the One-MSL cell.
"""

import numpy as np
import pytest

from model import nbcp
from model.nbcp.unit_cells import DISP_NN, DISP_NNN
from spintoolkit.methods.classical import classical_energy, local_fields, torques
from spintoolkit.methods.lswt.energy import EnergyFunction
from spintoolkit.models import (
    neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg,
)
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Site, SpinModel, Term

NBCP_LATTICE = np.array([[0.5, np.sqrt(3) / 2], [0.5, -np.sqrt(3) / 2]])
NBCP_CONFIG = {"Jxy": 0.076, "Jz": 0.125, "JPD": 0.013, "JGamma": -0.021,
               "Kxy": 0.004, "Kz": -0.005, "KPD": 0.002, "KGamma": -0.003,
               "Dz": 0.004, "h": (0.03, -0.04, 0.2)}


def nbcp_primitive_model(config):
    """One-site NBCP fixture; offsets follow r_target = r_source - d (D13)."""
    inverse = np.linalg.inv(NBCP_LATTICE)
    terms = []
    for label, matrices, displacements in [
        ("NN", nbcp.make_nn_exchange_matrices(config), DISP_NN),
        ("NNN", nbcp.make_nnn_exchange_matrices(config), DISP_NNN),
    ]:
        for J, d in zip(matrices, displacements):
            offset = tuple(np.rint(-np.asarray(d) @ inverse).astype(int))
            terms.append(Term.bilinear(("A", (0, 0)), ("A", offset), J, label))
    terms.append(Term.zeeman("A", np.eye(3)))
    return SpinModel(NBCP_LATTICE, [Site("A", (0, 0), 0.5)], terms,
                     {"model_id": "nbcp_primitive_fixture", "energy_unit": "meV"})


def spherical(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def max_torque(model, state, conditions=None):
    return max(np.linalg.norm(t) for t in torques(model, state, conditions).values())


@pytest.mark.parametrize("J, S", [(1.0, 0.5), (0.7, 1.5)])
def test_one_consumer_reproduces_exact_classical_energies(J, S):
    cases = [
        (square_heisenberg(J, S), neel_state, -2 * J * S**2),
        (triangular_heisenberg(J, S), state_120, -1.5 * J * S**2),
    ]
    for model, configuration, expected in cases:
        state = configuration(model)
        assert classical_energy(model, state) == pytest.approx(expected, abs=1e-14)
        assert max_torque(model, state) < 1e-14


@pytest.mark.parametrize("builder, bonds_per_site", [(square_heisenberg, 2),
                                                    (triangular_heisenberg, 3)])
def test_polarized_state_in_field(builder, bonds_per_site):
    J, S, h = 1.0, 0.5, 3.2
    model = builder(J, S)
    state = polarized_state(model)
    field = ExternalConditions(field=(0, 0, h))
    expected = bonds_per_site * J * S**2 - h * S
    assert classical_energy(model, state, field) == pytest.approx(expected, abs=1e-14)
    assert max_torque(model, state, field) < 1e-14


def test_energy_per_site_does_not_depend_on_the_chosen_supercell():
    model = triangular_heisenberg(g=np.diag([1.0, 1.0, 2.0]))
    n = spherical(0.4, 1.3)
    small = SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: n)
    large = SpinState.from_function(model, [[2, 1], [1, 2]], lambda s, c: n)
    field = ExternalConditions(field=(0.3, -0.1, 0.8))
    assert classical_energy(model, small, field) == pytest.approx(
        classical_energy(model, large, field), abs=1e-14)


def test_local_fields_match_finite_differences():
    rng = np.random.default_rng(3)
    dm = np.array([[0, 0.3, -0.1], [-0.3, 0, 0.2], [0.1, -0.2, 0]])
    model = SpinModel(
        [[1.0, 0.0], [0.5, np.sqrt(3) / 2]],
        [Site("A", (0, 0), 0.5), Site("B", (0.5, 0.0), 1.0)],
        [Term.bilinear(("A", (0, 0)), ("B", (0, 0)), np.eye(3) + dm),
         Term.bilinear(("A", (0, 0)), ("A", (0, 1)), np.diag([0.5, -0.2, 0.9])),
         Term.bilinear(("B", (0, 0)), ("A", (1, 0)), rng.normal(size=(3, 3))),
         Term.zeeman("A", np.diag([1.0, 1.2, 2.0])),
         Term.zeeman("B", rng.normal(size=(3, 3)))],
        {"model_id": "random_two_site"})
    supercell = [[1, 1], [-1, 2]]
    raw = {(s, c): spherical(*rng.uniform(0, np.pi, 2) * (1, 2))
           for s in model.site_ids for c in SpinState.from_function(
               model, supercell, lambda s, c: np.array([0, 0, 1.0])).cells}
    state = SpinState(model.fingerprint(), supercell, raw)
    field = ExternalConditions(field=(0.4, -0.3, 0.7))
    fields = local_fields(model, state, field)
    total_sites = model.num_sites * state.num_cells
    eps = 1e-6
    for key, n in raw.items():
        t = np.cross(n, rng.normal(size=3))
        t /= np.linalg.norm(t)

        def energy(delta):
            moved = dict(raw)
            moved[key] = (n + delta * t) / np.linalg.norm(n + delta * t)
            return classical_energy(model, SpinState(model.fingerprint(), supercell, moved),
                                    field) * total_sites

        derivative = (energy(eps) - energy(-eps)) / (2 * eps)
        spin = model.site(key[0]).spin
        assert derivative == pytest.approx(-spin * fields[key] @ t, rel=1e-6, abs=1e-9)


def test_missing_zeeman_term_warns_only_with_a_field():
    model = square_heisenberg(g=None)
    state = polarized_state(model)
    with pytest.warns(UserWarning, match="no zeeman term"):
        classical_energy(model, state, ExternalConditions(field=(0, 0, 1.0)))
    classical_energy(model, state)


@pytest.mark.parametrize("angles", [(0.3, 1.1), (2.0, -0.7), (1.2, 2.9)])
def test_nbcp_primitive_fixture_matches_existing_classical_energy(angles):
    model = nbcp_primitive_model(NBCP_CONFIG)
    n = spherical(*angles)
    state = SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: n)
    # g = I in the fixture, so the field is the legacy Zeeman energy h (E0 = meV).
    field = ExternalConditions(field=NBCP_CONFIG["h"])
    system = nbcp.one_msl(NBCP_CONFIG, np.array(angles),
                          nbcp.make_nn_exchange_matrices(NBCP_CONFIG),
                          nbcp.make_nnn_exchange_matrices(NBCP_CONFIG))
    existing = EnergyFunction(system.to_legacy_dict("Hex_60"), N=3)
    assert classical_energy(model, state, field) == pytest.approx(
        float(existing.classical_energy_density_func(np.array(angles))), abs=1e-15)
