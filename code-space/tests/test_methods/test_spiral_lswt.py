"""Rotating-frame LSWT of single-Q spirals (D34).

1. Triangular 120 deg state as a spiral: omega(k) = 3JS sqrt((1 - g)(1 + 2g)) and,
   on the same momenta, energies, bands, boson numbers and the full lab-frame
   S^{ab}(q, w) equal ordinary LSWT on the sqrt(3) x sqrt(3) supercell.
2. Incommensurate J1-J2 spiral (square lattice, ferromagnetic interchain):
   Luttinger-Tisza -> spiral -> refined q with cos(2 pi q) = -J1 / (4 J2);
   omega(k) = S sqrt((J_k - J_Q) ((J_{k+Q} + J_{k-Q}) / 2 - J_Q)).
3. The same spiral on a two-site cell: identical energy and S(q, w) at the
   same Cartesian momenta (sublattice phase of the cell-index convention).
4. DM spiral (ferromagnetic J, D along the axis): q from tan(2 pi q) = D / |J|,
   and at q = 1/6 equality with the 6 x 1 supercell; a conical spiral in a
   field along the axis: cos(theta) = h / (S (J_0 - J_Q)) and supercell equality.
5. Refusals: models without U(1) symmetry about the axis, a field off the axis,
   a q off the classical minimum (negative modes), lab-frame observables given
   the rotating-frame result, collinear LT minima.
"""

import json

import numpy as np
import pytest

from spintoolkit.methods.luttinger_tisza import luttinger_tisza
from spintoolkit.methods.lswt import (
    LSWTError, LSWTSettings, SpiralSymmetryError, refine_spiral, rotating_frame_model,
    solve_lswt, solve_spiral_lswt, spiral_energy, spiral_energy_gradient)
from spintoolkit.methods.lswt.spiral import symmetry_violations
from spintoolkit.methods.classical import classical_energy, torques
from spintoolkit.models import square_heisenberg, triangular_heisenberg
from spintoolkit.observables.berry import TopologyError, berry_curvature
from spintoolkit.observables.structure_factor import (
    spin_correlation, spiral_structure_factor, structure_factor)
from spintoolkit.states.incommensurate import IncommensurateStructure, rotation_matrix
from spintoolkit.states.spin_state import SpinStateError
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Site, SpinModel, Term

S = 0.5
Z = np.array([0.0, 0.0, 1.0])


def reciprocal(lattice):
    return 2 * np.pi * np.linalg.inv(lattice).T


def chain_model(J1=1.0, J2=0.4, Jy=-0.5, D=0.0, spin=S, Jz_ratio=1.0, field_g=True):
    """Square lattice: J1 (+ DM along z) and J2 along x, ferromagnetic Jy along y."""
    xxz = np.diag([1.0, 1.0, Jz_ratio])
    dm = np.array([[0, D, 0], [-D, 0, 0], [0, 0, 0]])
    terms = [Term.bilinear(("A", (0, 0)), ("A", (1, 0)), J1 * xxz + dm, "J1"),
             Term.bilinear(("A", (0, 0)), ("A", (2, 0)), J2 * xxz, "J2"),
             Term.bilinear(("A", (0, 0)), ("A", (0, 1)), Jy * np.eye(3), "Jy")]
    if field_g:
        terms.append(Term.zeeman("A", np.eye(3)))
    return SpinModel(np.eye(2), [Site("A", (0, 0), spin)], terms, {"model_id": "chain"})


def chain_model_two_site(J1=1.0, J2=0.4, Jy=-0.5):
    """The same model on the cell (2, 0), (0, 1) with sites at x = 0 and x = 1."""
    J = np.eye(3)
    terms = [Term.bilinear(("A", (0, 0)), ("B", (0, 0)), J1 * J, "J1"),
             Term.bilinear(("B", (0, 0)), ("A", (1, 0)), J1 * J, "J1"),
             Term.bilinear(("A", (0, 0)), ("A", (1, 0)), J2 * J, "J2"),
             Term.bilinear(("B", (0, 0)), ("B", (1, 0)), J2 * J, "J2"),
             Term.bilinear(("A", (0, 0)), ("A", (0, 1)), Jy * J, "Jy"),
             Term.bilinear(("B", (0, 0)), ("B", (0, 1)), Jy * J, "Jy")]
    return SpinModel(np.diag([2.0, 1.0]), [Site("A", (0, 0), S), Site("B", (0.5, 0), S)], terms,
                     {"model_id": "chain2"})


def heisenberg_dispersion(k, Q, bonds):
    """S sqrt((J_k - J_Q)((J_{k+Q} + J_{k-Q})/2 - J_Q)), J_k = sum over bonds of 2 J cos(k . d)."""
    def Jk(p):
        return sum(2 * J * np.cos(p @ d) for d, J in bonds)
    return S * np.sqrt(np.clip((Jk(k) - Jk(Q[None]))
                               * ((Jk(k + Q) + Jk(k - Q)) / 2 - Jk(Q[None])), 0, None))


def unfolded(model, supercell_result):
    """Primitive-cell momenta equivalent to the magnetic mesh of a supercell result."""
    B = reciprocal(model.lattice)
    Bm = reciprocal(supercell_result.magnetic_lattice)
    reps = []
    for n1 in range(6):
        for n2 in range(6):
            G = n1 * Bm[0] + n2 * Bm[1]
            f = np.mod(G @ model.lattice.T / (2 * np.pi), 1.0)
            if not any(np.allclose(np.mod(f - g + 0.5, 1) - 0.5, 0, atol=1e-9) for g, _ in reps):
                reps.append((f, G))
    assert len(reps) == len(supercell_result.site_keys) // model.num_sites
    return np.concatenate([supercell_result.k_points + G for _, G in reps])


def assert_same_spectrum(a, b, omega=np.linspace(-4, 4, 161), eta=0.05):
    np.testing.assert_allclose(a.static(), b.static(), rtol=0, atol=1e-10)
    for component in ("xx", "yy", "zz", "neutron"):
        np.testing.assert_allclose(a.spectrum(omega, eta, "gaussian", component),
                                   b.spectrum(omega, eta, "gaussian", component),
                                   rtol=0, atol=1e-9)


# ----------------------------------------------------------------------------
# 1. Triangular 120 deg state
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def triangular():
    model = triangular_heisenberg(J=1.0, S=S)
    spiral = IncommensurateStructure.planar(model, [1 / 3, 2 / 3], Z)
    supercell = solve_lswt(model, spiral.to_spin_state(model), settings=LSWTSettings(mesh=(8, 8)))
    k = unfolded(model, supercell)
    result = solve_spiral_lswt(model, spiral, settings=LSWTSettings(k_points=k))
    return model, spiral, result, supercell


def test_triangular_dispersion_is_analytic(triangular):
    model, _, result, _ = triangular
    a = model.lattice
    k = result.k_points
    gamma = np.mean([np.cos(k @ d) for d in (a[0], a[1], a[1] - a[0])], axis=0)
    expected = 3 * S * np.sqrt((1 - gamma) * (1 + 2 * gamma))
    np.testing.assert_allclose(result.bands()[:, 0], expected, rtol=0, atol=1e-12)
    assert result.classical_energy == pytest.approx(-1.5 * S ** 2, abs=1e-14)


def test_triangular_equals_supercell_lswt(triangular):
    _, _, result, supercell = triangular
    assert result.ground_state_energy == pytest.approx(supercell.ground_state_energy, abs=1e-13)
    assert result.zero_point_energy == pytest.approx(supercell.zero_point_energy, abs=1e-13)
    np.testing.assert_allclose(np.sort(result.bands().ravel()),
                               np.sort(supercell.bands().ravel()), rtol=0, atol=1e-11)
    np.testing.assert_allclose(result.boson_numbers, supercell.boson_numbers.mean(), atol=1e-13)


@pytest.mark.parametrize("temperature", [0.0, 0.3])
def test_triangular_structure_factor_equals_supercell(triangular, temperature):
    model, _, result, supercell = triangular
    q = np.random.default_rng(1).uniform(-6, 6, (16, 2))
    a = spiral_structure_factor(result, q, temperature, gapless=True)
    b = structure_factor(supercell, q, temperature, gapless=True)
    assert a.energies.shape[1] == b.energies.shape[1]           # 3 branches x 2 = 2 x 3 sites
    assert_same_spectrum(a, b)


def test_triangular_bragg_peaks(triangular):
    model, spiral, result, supercell = triangular
    B = reciprocal(model.lattice)
    K = spiral.cartesian_wave_vector(model)
    q = np.array([K, -K, K + B[0], B[1], 0.3 * B[0]])
    a = spiral_structure_factor(result, q)
    b = structure_factor(supercell, q)
    np.testing.assert_array_equal(a.bragg, [True, True, True, True, False])
    np.testing.assert_array_equal(a.bragg, b.bragg)
    np.testing.assert_allclose(a.elastic, b.elastic, atol=1e-12)
    # Planar spiral: no uniform moment, Bragg weight only at +-Q, transverse to the axis.
    assert np.allclose(a.elastic[3], 0) and np.allclose(a.elastic[:3, 2, 2], 0)


# ----------------------------------------------------------------------------
# 2. Incommensurate J1-J2 spiral
# ----------------------------------------------------------------------------

J1, J2, JY = 1.0, 0.4, -0.5
Q_EXACT = np.arccos(-J1 / (4 * J2)) / (2 * np.pi)
BONDS = [(np.array([1.0, 0.0]), J1), (np.array([2.0, 0.0]), J2), (np.array([0.0, 1.0]), JY)]


@pytest.fixture(scope="module")
def incommensurate():
    model = chain_model(J1, J2, JY)
    report = luttinger_tisza(model, mesh=(24, 24))
    minimum = report.minima[0]
    assert not minimum.commensurate and minimum.strong_constraint
    spiral = refine_spiral(model, IncommensurateStructure.from_lt(model, minimum))
    return model, spiral, solve_spiral_lswt(model, spiral, settings=LSWTSettings(mesh=(16, 16)))


def test_lt_and_refinement_find_the_analytic_pitch(incommensurate):
    model, spiral, _ = incommensurate
    q = spiral.wave_vector
    # BFGS + Newton on the analytic gradient reaches its round-off (measured
    # max |grad| 1.7e-16, pitch error 0 on x86-64); 1e-12 leaves 1e4 of margin.
    assert spiral.provenance["converged"]
    assert spiral.provenance["max_gradient"] < 1e-12
    assert min(q[0], 1 - q[0]) == pytest.approx(Q_EXACT, abs=1e-12)
    assert q[1] == pytest.approx(0, abs=1e-12) or q[1] == pytest.approx(1, abs=1e-12)
    assert spiral_energy(model, spiral) == pytest.approx(
        S ** 2 * (J1 * np.cos(2 * np.pi * Q_EXACT) + J2 * np.cos(4 * np.pi * Q_EXACT) + JY),
        abs=1e-13)
    assert np.linalg.norm(spiral_energy_gradient(model, spiral)) < 1e-12
    assert spiral.commensurate_supercell() is None


def test_incommensurate_dispersion_is_analytic(incommensurate):
    model, spiral, result = incommensurate
    Q = spiral.cartesian_wave_vector(model)
    expected = heisenberg_dispersion(result.k_points, Q, BONDS)
    np.testing.assert_allclose(result.bands()[:, 0], expected, rtol=0, atol=1e-9)
    assert result.header.diagnostics["stationary"]
    assert result.header.method == "lswt-spiral"


def test_energy_functions_agree_with_rotating_frame_model(incommensurate):
    model, spiral, result = incommensurate
    rotated, state = rotating_frame_model(model, spiral)
    assert classical_energy(rotated, state) == pytest.approx(spiral_energy(model, spiral), abs=1e-14)
    assert max(np.linalg.norm(t) for t in torques(rotated, state).values()) < 1e-12
    assert result.classical_energy == pytest.approx(spiral_energy(model, spiral), abs=1e-14)


def test_two_site_cell_gives_the_same_physics(incommensurate):
    model, spiral, result = incommensurate
    q = min(spiral.wave_vector[0], 1 - spiral.wave_vector[0])
    model2 = chain_model_two_site(J1, J2, JY)
    spiral2 = IncommensurateStructure(
        model2.fingerprint(), [2 * q, 0.0], Z,
        {"A": [1.0, 0, 0], "B": rotation_matrix(Z, 2 * np.pi * q) @ [1.0, 0, 0]})
    mesh = solve_spiral_lswt(model2, spiral2, settings=LSWTSettings(mesh=(6, 12)))
    k_one = np.concatenate([mesh.k_points, mesh.k_points + [np.pi, 0.0]])
    one = solve_spiral_lswt(model, IncommensurateStructure.planar(model, [q, 0.0], Z),
                            settings=LSWTSettings(k_points=k_one))
    assert mesh.ground_state_energy == pytest.approx(one.ground_state_energy, abs=1e-13)
    momenta = np.random.default_rng(3).uniform(-5, 5, (12, 2))
    assert_same_spectrum(spiral_structure_factor(mesh, momenta),
                         spiral_structure_factor(one, momenta))


def test_json_output(incommensurate, tmp_path):
    _, spiral, result = incommensurate
    data = json.loads(json.dumps(result.to_json_dict()))
    assert data["header"]["method"] == "lswt-spiral"
    assert np.allclose(data["spiral"]["wave_vector"], spiral.wave_vector)
    assert "bands" in data["lswt"]


# ----------------------------------------------------------------------------
# 3. DM spiral and conical spiral
# ----------------------------------------------------------------------------

def test_dm_spiral_pitch_and_supercell():
    D = np.sqrt(3.0)                                    # tan(2 pi q) = D / |J| -> q = 1/6
    model = chain_model(J1=-1.0, J2=0.0, Jy=-1.0, D=D)
    start = IncommensurateStructure.planar(model, [0.1, 0.0], Z)
    spiral = refine_spiral(model, start)
    q = spiral.wave_vector[0]
    assert min(q, 1 - q) == pytest.approx(1 / 6, abs=1e-12)
    spiral = IncommensurateStructure.planar(model, [round(q * 6) / 6, 0.0], Z)
    supercell = solve_lswt(model, spiral.to_spin_state(model), settings=LSWTSettings(mesh=(4, 12)))
    assert supercell.header.diagnostics["stationary"]
    k = unfolded(model, supercell)
    result = solve_spiral_lswt(model, spiral, settings=LSWTSettings(k_points=k))
    assert result.ground_state_energy == pytest.approx(supercell.ground_state_energy, abs=1e-13)
    np.testing.assert_allclose(np.sort(result.bands().ravel()),
                               np.sort(supercell.bands().ravel()), atol=1e-11)
    momenta = np.random.default_rng(5).uniform(-4, 4, (10, 2))
    assert_same_spectrum(spiral_structure_factor(result, momenta),
                         structure_factor(supercell, momenta))


def test_conical_spiral_in_axial_field():
    model = chain_model(J1=1.0, J2=0.5, Jy=-0.5)       # cos(2 pi q) = -1/2 -> q = 1/3
    J0 = 2 * (1.0 + 0.5 - 0.5)
    JQ = 2 * (np.cos(2 * np.pi / 3) + 0.5 * np.cos(4 * np.pi / 3) - 0.5)
    h = 0.4 * S * (J0 - JQ)
    conditions = ExternalConditions(field=(0, 0, h))
    start = IncommensurateStructure(model.fingerprint(), [0.3, 0.0], Z,
                                    {"A": [np.sin(1.2), 0, np.cos(1.2)]})
    spiral = refine_spiral(model, start, conditions)
    assert min(spiral.wave_vector[0], 1 - spiral.wave_vector[0]) == pytest.approx(1 / 3, abs=1e-12)
    assert np.cos(spiral.cone_angles()["A"]) == pytest.approx(0.4, abs=1e-12)
    exact = IncommensurateStructure(model.fingerprint(), [1 / 3, 0.0], Z,
                                    {"A": [np.sqrt(1 - 0.16), 0, 0.4]})
    supercell = solve_lswt(model, exact.to_spin_state(model), conditions,
                           settings=LSWTSettings(mesh=(8, 8)))
    k = unfolded(model, supercell)
    result = solve_spiral_lswt(model, exact, conditions, settings=LSWTSettings(k_points=k))
    assert result.ground_state_energy == pytest.approx(supercell.ground_state_energy, abs=1e-13)
    np.testing.assert_allclose(np.sort(result.bands().ravel()),
                               np.sort(supercell.bands().ravel()), atol=1e-11)
    momenta = np.random.default_rng(7).uniform(-4, 4, (10, 2))
    a, b = spiral_structure_factor(result, momenta), structure_factor(supercell, momenta)
    assert_same_spectrum(a, b)
    # The uniform cone moment gives a Bragg peak at q = 0 along the axis.
    bragg = spiral_structure_factor(result, [[0.0, 0.0]])
    assert bragg.elastic[0, 2, 2].real > 0


# ----------------------------------------------------------------------------
# 4. Refusals
# ----------------------------------------------------------------------------

def test_rejects_exchange_without_axial_symmetry():
    model = chain_model(Jz_ratio=1.0)
    terms = list(model.terms)
    terms[0] = Term.bilinear(("A", (0, 0)), ("A", (1, 0)), np.diag([1.0, 0.9, 1.0]), "J1")
    broken = SpinModel(model.lattice, model.sites, terms, {"model_id": "broken"})
    spiral = IncommensurateStructure.planar(broken, [Q_EXACT, 0], Z)
    with pytest.raises(SpiralSymmetryError, match="J1"):
        solve_spiral_lswt(broken, spiral)
    with pytest.raises(SpiralSymmetryError):
        refine_spiral(broken, spiral)
    # diag(1, 0.9, 1) is XXZ about y: a spiral rotating in the xz plane is allowed.
    assert symmetry_violations(broken, [0, 1.0, 0]) == []


def test_xxz_with_axis_along_z_is_accepted():
    model = chain_model(Jz_ratio=0.7)
    q = np.arccos(-J1 / (4 * J2)) / (2 * np.pi)
    result = solve_spiral_lswt(model, IncommensurateStructure.planar(model, [q, 0], Z),
                               settings=LSWTSettings(mesh=(8, 8)))
    assert np.all(result.bands() > 0)


def test_rejects_field_off_the_axis():
    model = chain_model()
    spiral = IncommensurateStructure.planar(model, [Q_EXACT, 0], Z)
    with pytest.raises(SpiralSymmetryError, match="perpendicular"):
        solve_spiral_lswt(model, spiral, ExternalConditions(field=(0.1, 0, 0)))


def test_pitch_off_the_minimum_is_unstable():
    model = chain_model()
    spiral = IncommensurateStructure.planar(model, [Q_EXACT - 0.05, 0], Z)
    with pytest.warns(UserWarning, match="not a classical extremum"):
        with pytest.raises(LSWTError, match="negative"):
            solve_spiral_lswt(model, spiral, settings=LSWTSettings(mesh=(32, 16)))


def test_lab_frame_observables_refuse_the_rotating_result(incommensurate):
    model, _, result = incommensurate
    with pytest.raises(ValueError, match="rotating frame"):
        structure_factor(result.rotating, [[0.1, 0.2]])
    with pytest.raises(ValueError, match="rotating frame"):
        spin_correlation(result.rotating, model, ("A", (0, 0)), ("A", (1, 0)))
    with pytest.raises(TopologyError, match="not validated"):
        berry_curvature(result.rotating)


def test_structure_factor_refuses_half_reciprocal_wave_vector():
    model = chain_model(J1=1.0, J2=0.0, Jy=-0.5)
    result = solve_spiral_lswt(model, IncommensurateStructure.planar(model, [0.5, 0.0], Z),
                               settings=LSWTSettings(mesh=(8, 8)))
    with pytest.raises(NotImplementedError, match="2Q"):
        spiral_structure_factor(result, [[0.1, 0.1]])


def test_from_lt_rejects_collinear_minimum():
    model = square_heisenberg(J=1.0, S=S)
    report = luttinger_tisza(model, mesh=(12, 12))
    with pytest.raises(ValueError, match="not a spiral"):
        IncommensurateStructure.from_lt(model, report.minima[0])


def test_from_lt_triangular_is_the_120_state():
    model = triangular_heisenberg(J=1.0, S=S)
    report = luttinger_tisza(model, mesh=(12, 12))
    spiral = IncommensurateStructure.from_lt(model, report.minima[0])
    assert spiral_energy(model, spiral) == pytest.approx(-1.5 * S ** 2, abs=1e-12)
    assert spiral.commensurate_supercell() is not None


def test_state_validation():
    model = chain_model()
    with pytest.raises(SpinStateError):
        IncommensurateStructure(model.fingerprint(), [0.1, 0], [0, 0, 2.0], {"A": [1, 0, 0]})
    with pytest.raises(SpinStateError):
        IncommensurateStructure(model.fingerprint(), [0.1, 0], Z, {"A": [1, 1, 0]})
    other = IncommensurateStructure("x", [0.1, 0], Z, {"A": [1.0, 0, 0]})
    with pytest.raises(SpinStateError, match="fingerprint"):
        spiral_energy(model, other)
    with pytest.raises(SpinStateError, match="not commensurate"):
        IncommensurateStructure.planar(model, [Q_EXACT, 0], Z).to_spin_state(model)
