"""Single-ion (onsite) terms ``S^T A S`` across methods (D37).

Physics checks, each against an exact statement:

1. ``S = 1/2``: the operator is the constant ``tr(A)/4``; classical energy,
   torques and LSWT bands must not depend on ``A`` beyond that constant.
2. A single spin, exact diagonalization against LSWT for S = 1 .. 10 with a
   transverse anisotropy: the coherent-state rule (factor ``1 - 1/(2S)``)
   keeps the error of the gap and of the ground-state energy bounded, while
   the large-S rule misses the gap by an O(1) amount (the single-ion gap of
   ``D (S^z)^2`` is ``(2S - 1)|D|``, not ``2S|D|``).
3. U(1)-symmetric lattice model above saturation: LSWT one-magnon bands equal
   the exact one-magnon ED energies on a torus, onsite term included.
4. Internal consistency: bond-correlation energy equals the LSWT ground-state
   energy; analytic classical gradient equals finite differences; the LT
   bound stays below the classical energy.
"""

import numpy as np
import pytest

from spintoolkit.methods.classical import classical_energy, refine_classical, tangent_expansion, torques
from spintoolkit.methods.ed import EDSector, solve_ed
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.methods.luttinger_tisza import luttinger_tisza
from spintoolkit.models import polarized_state, state_120
from spintoolkit.observables.structure_factor import bond_correlations
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import Site, SpinModel, SpinModelError, Term, onsite_renormalization

SQUARE = np.eye(2)
TRIANGULAR = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
NN_TRIANGULAR = ((1, 0), (0, 1), (-1, 1))


def single_spin(S, A, g=True):
    terms = [Term.onsite("A", A)] + ([Term.zeeman("A", np.eye(3))] if g else [])
    return SpinModel(SQUARE, [Site("A", (0, 0), S)], terms, {"model_id": f"single_spin_{S}"})


def spin_matrices(S):
    m = np.arange(S, -S - 1, -1)
    plus = np.diag(np.sqrt(S * (S + 1) - m[1:] * (m[1:] + 1)), 1)
    return [(plus + plus.T) / 2, (plus - plus.T) / 2j, np.diag(m)]


def test_onsite_validation():
    with pytest.raises(SpinModelError, match="symmetric"):
        single_spin(1, [[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    with pytest.raises(SpinModelError, match="more than one onsite"):
        SpinModel(SQUARE, [Site("A", (0, 0), 1)],
                  [Term.onsite("A", np.eye(3)), Term.onsite("A", np.eye(3))], {"model_id": "x"})
    assert onsite_renormalization(0.5) == 0.0
    assert onsite_renormalization(1.0) == 0.5


def test_spin_half_onsite_is_a_constant():
    A = np.array([[0.4, 0.1, -0.2], [0.1, -0.3, 0.05], [-0.2, 0.05, 0.7]])

    def model(with_onsite):
        terms = [Term.bilinear(("A", (0, 0)), ("A", o), np.diag([1.0, 1.0, 0.8])) for o in NN_TRIANGULAR]
        terms += [Term.zeeman("A", np.eye(3))] + ([Term.onsite("A", A)] if with_onsite else [])
        return SpinModel(TRIANGULAR, [Site("A", (0, 0), 0.5)], terms, {"model_id": str(with_onsite)})

    conditions = ExternalConditions(field=(0.1, 0, 0.3))
    plain, anisotropic = model(False), model(True)
    for direction in ((0, 0, 1), (1, 1, 0.3)):
        e = [classical_energy(m, polarized_state(m, direction), conditions) for m in (plain, anisotropic)]
        assert e[1] - e[0] == pytest.approx(np.trace(A) / 4, abs=1e-14)
        t = [torques(m, polarized_state(m, direction), conditions) for m in (plain, anisotropic)]
        np.testing.assert_allclose(list(t[0].values()), list(t[1].values()), atol=1e-15)
    field = ExternalConditions(field=(0, 0, 9.0))
    k = np.random.default_rng(0).uniform(-3, 3, (6, 2))
    bands = [solve_lswt(m, polarized_state(m), field, settings=LSWTSettings(k_points=k)).bands()
             for m in (plain, anisotropic)]
    np.testing.assert_allclose(bands[0], bands[1], atol=1e-13)


@pytest.mark.parametrize("S", [1, 1.5, 2, 3, 5, 10])
def test_single_spin_against_exact_spectrum(S):
    A = np.diag([0.3, -0.1, -0.5])
    h = 2.0 * S
    conditions = ExternalConditions(field=(0, 0, h))
    model = single_spin(S, A)
    lswt = solve_lswt(model, polarized_state(model), conditions,
                      settings=LSWTSettings(k_points=[[0.0, 0.0]]))
    Sx, Sy, Sz = spin_matrices(S)
    ops = (Sx, Sy, Sz)
    H = sum(A[a, b] * ops[a] @ ops[b] for a in range(3) for b in range(3)) - h * Sz
    exact = np.linalg.eigvalsh(H)
    gap_error = (exact[1] - exact[0]) - lswt.bands()[0, 0]
    energy_error = exact[0] - lswt.ground_state_energy
    assert abs(gap_error) < 0.02              # large-S coefficients would miss by ~0.57
    assert abs(energy_error) < 0.01           # bounded in S (no O(S) double counting)
    # The ED module gives the same exact spectrum with the onsite operator.
    ed = solve_ed(model, CalculationGeometry.finite_torus([[1, 0], [0, 1]]), conditions,
                  num_eigenvalues=2)
    np.testing.assert_allclose(ed.blocks[0].energies[:2], exact[:2], atol=1e-10)


def test_one_magnon_bands_are_exact_with_onsite_anisotropy():
    S, Dxy, Dz = 1.0, 0.15, -0.4
    terms = [Term.bilinear(("A", (0, 0)), ("A", o), np.diag([0.6, 0.6, 1.0])) for o in NN_TRIANGULAR]
    terms += [Term.zeeman("A", np.eye(3)), Term.onsite("A", np.diag([Dxy, Dxy, Dz]))]
    model = SpinModel(TRIANGULAR, [Site("A", (0, 0), S)], terms, {"model_id": "xxz_onsite"})
    geometry = CalculationGeometry.finite_torus([[3, 0], [0, 3]])
    conditions = ExternalConditions(field=(0, 0, 12.0))
    lswt = solve_lswt(model, polarized_state(model), conditions, geometry)
    ed = solve_ed(model, geometry, conditions, EDSector(axis=(0, 0, 1), magnon_number=1))
    np.testing.assert_allclose(np.sort(lswt.bands().ravel()), ed.excitations(ed.blocks[0]),
                               rtol=0, atol=1e-11)
    # The polarized state is an exact eigenstate; its energy per site is the
    # coherent-state classical energy (zero-point energy vanishes).
    assert lswt.ground_state_energy * 9 == pytest.approx(ed.reference_energy, abs=1e-11)


def easy_plane_triangular_120():
    S = 1.0
    terms = [Term.bilinear(("A", (0, 0)), ("A", o), np.eye(3)) for o in NN_TRIANGULAR]
    terms += [Term.onsite("A", np.diag([0.0, 0.0, 0.3]))]
    model = SpinModel(TRIANGULAR, [Site("A", (0, 0), S)], terms, {"model_id": "tri_easy_plane"})
    return model, state_120(model)


def test_bond_correlation_energy_equals_lswt_energy():
    model, state = easy_plane_triangular_120()
    # Tilt the plane so that the onsite term has transverse and longitudinal parts.
    terms = list(model.terms[:-1]) + [Term.onsite("A", np.array([[0.1, 0.05, 0.0],
                                                                  [0.05, -0.05, 0.0],
                                                                  [0.0, 0.0, 0.3]]))]
    model = SpinModel(model.lattice, model.sites, terms, {"model_id": "tri_onsite"})
    state = refine_classical(model, state_120(model))
    assert tangent_expansion(model, state).max_torque < 1e-10
    result = solve_lswt(model, state, settings=LSWTSettings(mesh=(12, 12)))
    assert bond_correlations(result, model)["energy"] == pytest.approx(result.ground_state_energy,
                                                                       abs=1e-13)


def test_classical_gradient_matches_finite_differences():
    A = np.array([[0.2, 0.1, -0.3], [0.1, -0.4, 0.2], [-0.3, 0.2, 0.5]])
    model = single_spin(2.0, A)
    n = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    conditions = ExternalConditions(field=(0.2, 0.1, 0.4))

    def energy(direction):
        state = SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: direction)
        return classical_energy(model, state, conditions)

    state = SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: n)
    expansion = tangent_expansion(model, state, conditions)
    step = 1e-6
    for a, e in enumerate(expansion.frames[0]):
        numeric = (energy(n + step * e) - energy(n - step * e)) / (2 * step)
        assert expansion.gradient[a] == pytest.approx(numeric, abs=1e-7)
    assert expansion.energy == pytest.approx(energy(n), abs=1e-14)


def test_lt_bound_below_classical_energy_with_onsite():
    model, state = easy_plane_triangular_120()
    report = luttinger_tisza(model, mesh=(24, 24))
    assert report.lambda_min <= classical_energy(model, state) + 1e-12
    # Easy-plane anisotropy keeps the coplanar 120-degree state at the bound.
    assert report.lambda_min == pytest.approx(classical_energy(model, state), abs=1e-9)


def test_spiral_lswt_with_axial_onsite_equals_supercell():
    """Easy-plane anisotropy is axial about z: the rotating-frame LSWT of the 120-degree
    state must equal the supercell LSWT; an in-plane anisotropy must be rejected."""
    from spintoolkit.methods.lswt import solve_spiral_lswt
    from spintoolkit.methods.lswt.spiral import symmetry_violations
    from spintoolkit.states.incommensurate import IncommensurateStructure

    model, _ = easy_plane_triangular_120()
    spiral = IncommensurateStructure.planar(model, [1 / 3, 2 / 3], (0, 0, 1))
    supercell = solve_lswt(model, spiral.to_spin_state(model), settings=LSWTSettings(mesh=(6, 6)))
    # The same momenta: magnetic mesh plus the magnetic reciprocal vectors folding onto it.
    Bm = 2 * np.pi * np.linalg.inv(supercell.magnetic_lattice).T
    k = np.concatenate([supercell.k_points + G for G in (0 * Bm[0], Bm[0], 2 * Bm[0])])
    rotating = solve_spiral_lswt(model, spiral, settings=LSWTSettings(k_points=k))
    assert rotating.ground_state_energy == pytest.approx(supercell.ground_state_energy, abs=1e-12)
    np.testing.assert_allclose(np.sort(rotating.bands().ravel()), np.sort(supercell.bands().ravel()),
                               atol=1e-10)
    assert symmetry_violations(model, (0, 0, 1)) == []
    in_plane = SpinModel(model.lattice, model.sites,
                         list(model.terms[:-1]) + [Term.onsite("A", np.diag([0.3, 0.0, 0.0]))],
                         {"model_id": "tri_in_plane"})
    assert any("onsite" in v for v in symmetry_violations(in_plane, (0, 0, 1)))
