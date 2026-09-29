"""SpinModel/SpinState <-> SpinSystem conversion against the legacy NBCP builders.

Element-wise equality of the bosonic H(k) with the existing unit-cell builders
confirms the displacement reading r_target = r_source - d (D13) numerically.
"""

import numpy as np
import pytest

from model import nbcp
from model.nbcp.model import legacy_cells
from spintoolkit.methods.classical import classical_energy
from spintoolkit.methods.lswt.energy import EnergyFunction
from spintoolkit.methods.lswt.hamiltonian import LSWTHamiltonian
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.conversion import from_spin_system, site_label, to_spin_system
from spintoolkit.system.spin_system import SpinSystem

K_POINTS = np.array([[0.0, 0.0], [0.37, -0.29], [-0.81, 0.46], [1.13, 0.67]])
CELLS = ["one_msl", "two_msl", "three_msl", "four_msl"]
BZ = {"one_msl": "Hex_60", "two_msl": "Tetra", "three_msl": "Hex_30", "four_msl": "Hex_60"}
FIELD = np.array([0.03, -0.04, 0.2])
BASE = {"Jxy": 0.076, "Jz": 0.125}
FAMILIES = {
    "xxz": {},
    "nn_soc": {"JPD": 0.013, "JGamma": -0.021},
    "nn_nnn_soc": {"JPD": 0.013, "JGamma": -0.021,
                   "Kxy": 0.004, "Kz": -0.005, "KPD": 0.002, "KGamma": -0.003},
    "dm": {"Dx": 0.002, "Dy": -0.001, "Dz": 0.003},
    "zero_exchange": {"Jxy": 0.0, "Jz": 0.0},
}


def hamiltonian(system, bz_type):
    data = system.to_legacy_dict(bz_type)
    k_ham, _ = LSWTHamiltonian(data["Spin info"], data["Couplings"]).Quadratic_Bose_Hamiltonian(
        K_POINTS, angles=system.get_angles_flat())
    return np.asarray(k_ham)


def legacy_system(cell, config, angles):
    return getattr(nbcp, cell)({**config, "h": tuple(FIELD)}, np.asarray(angles),
                               nbcp.make_nn_exchange_matrices(config),
                               nbcp.make_nnn_exchange_matrices(config))


def legacy_order(cell, legacy, state, converted):
    """Indices of the converted sites in the legacy site order."""
    mapping = legacy_cells(cell)
    labels = [s.label for s in converted.sites]
    return [labels.index(site_label("Co", mapping[s.label], state.num_cells)) for s in legacy.sites]


def reorder(k_ham, order):
    n = len(order)
    index = np.concatenate([order, np.asarray(order) + n])
    return k_ham[:, index][:, :, index]


def canonical_angles(rng, cell):
    """theta in (0, pi): the converted angles reproduce the legacy local frames."""
    n = len(legacy_cells(cell))
    return np.column_stack([rng.uniform(0.2, np.pi - 0.2, n), rng.uniform(-3.0, 3.0, n)]).ravel()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cell", CELLS)
def test_converted_hamiltonian_equals_legacy_builder_elementwise(cell, family):
    config = {**BASE, **FAMILIES[family]}
    angles = canonical_angles(np.random.default_rng(7301), cell)
    model = nbcp.build_model(config)
    state = nbcp.candidate_state(model, cell, angles)
    converted = to_spin_system(model, state, ExternalConditions(field=FIELD))
    legacy = legacy_system(cell, config, angles)
    np.testing.assert_allclose(converted.lattice_vectors, legacy.lattice_vectors, atol=1e-15)
    order = legacy_order(cell, legacy, state, converted)
    np.testing.assert_allclose(reorder(hamiltonian(converted, BZ[cell]), order),
                               hamiltonian(legacy, BZ[cell]), rtol=0, atol=1e-14)


def test_opposite_displacement_reading_breaks_three_msl():
    """Negative control: storing d = r_target - r_source changes H(k) for Three MSL."""
    config = {**BASE, **FAMILIES["nn_soc"]}
    angles = canonical_angles(np.random.default_rng(1), "three_msl")
    model = nbcp.build_model(config)
    state = nbcp.candidate_state(model, "three_msl", angles)
    converted = to_spin_system(model, state, ExternalConditions(field=FIELD))
    flipped = SpinSystem(lattice_vectors=converted.lattice_vectors)
    for site in converted.sites:
        flipped.add_site(site.label, site.position, site.spin, site.angles, site.magnetic_field)
    for c in converted.couplings:
        flipped.add_coupling(converted.sites[c.site_i].label, converted.sites[c.site_j].label,
                             c.exchange_matrix, -c.displacement)
    legacy = legacy_system("three_msl", config, angles)
    order = legacy_order("three_msl", legacy, state, converted)
    difference = reorder(hamiltonian(flipped, "Hex_30"), order) - hamiltonian(legacy, "Hex_30")
    assert np.max(np.abs(difference)) > 1e-3


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cell", CELLS)
def test_round_trip_through_common_format_keeps_hamiltonian(cell, family):
    config = {**BASE, **FAMILIES[family]}
    legacy = legacy_system(cell, config, canonical_angles(np.random.default_rng(11), cell))
    model, state, conditions = from_spin_system(legacy)
    back = to_spin_system(model, state, conditions)
    np.testing.assert_allclose(hamiltonian(back, BZ[cell]), hamiltonian(legacy, BZ[cell]),
                               rtol=0, atol=1e-14)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cell", CELLS)
def test_energies_agree_for_any_angle_representation(cell, family):
    """Classical and zero-point energies are gauge invariant, so any angles work."""
    config = {**BASE, **FAMILIES[family]}
    n = len(legacy_cells(cell))
    angles = np.linspace(-1.1, 2.3, 2 * n)
    model = nbcp.build_model(config)
    state = nbcp.candidate_state(model, cell, angles)
    field = ExternalConditions(field=FIELD)
    legacy = EnergyFunction(legacy_system(cell, config, angles).to_legacy_dict(BZ[cell]), N=3)
    assert classical_energy(model, state, field) == pytest.approx(
        float(legacy.classical_energy_density_func(angles)), abs=1e-14)
    converted = to_spin_system(model, state, field)
    new = EnergyFunction(converted.to_legacy_dict(BZ[cell]), N=3)
    assert float(new.quantum_energy_density_func(converted.get_angles_flat())) == pytest.approx(
        float(legacy.quantum_energy_density_func(angles)), abs=1e-12)


def test_site_dependent_fields_are_rejected_with_an_explanation():
    system = SpinSystem(lattice_vectors=np.eye(2))
    system.add_site("A", [0, 0], 0.5, [0.1, 0.2], [0, 0, 0.1])
    system.add_site("B", [0.5, 0.5], 0.5, [0.1, 0.2], [0, 0, -0.1])
    with pytest.raises(ValueError, match="local_field"):
        from_spin_system(system)


def test_displacement_off_the_lattice_is_rejected():
    system = SpinSystem(lattice_vectors=np.eye(2))
    system.add_site("A", [0, 0], 0.5, [0.1, 0.2], [0, 0, 0])
    system.add_coupling("A", "A", np.eye(3), [0.5, 0.0])
    with pytest.raises(ValueError, match="D13"):
        from_spin_system(system)


def test_missing_zeeman_term_warns_when_field_is_applied():
    model = nbcp.build_model(BASE)
    no_zeeman = type(model)(model.lattice, model.sites, model.terms_of_kind("bilinear"),
                            {"model_id": "nbcp_without_zeeman"})
    state = nbcp.candidate_state(no_zeeman, "one_msl", [0.3, 0.4])
    with pytest.warns(UserWarning, match="no zeeman term"):
        to_spin_system(no_zeeman, state, ExternalConditions(field=FIELD))
