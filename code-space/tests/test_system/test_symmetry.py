"""Crystal symmetry: allowed couplings, symmetric orbits and model checks.

References for the allowed forms:
- flat triangular layer (6/mmm): bond-dependent XYZ exchange, three parameters;
- triangular layer of edge-sharing octahedra (CdI2 type, D3d sites, P-3m1):
  four parameters J, Delta, J_pm_pm, J_z_pm with no DM (inversion at the bond
  centre) and an axial g-tensor (Li et al., Sci. Rep. 5, 16419 (2015));
- flat kagome layer: DM along the normal only, uniform circulation around a
  triangle (Moriya rules; Elhajal, Canals and Lacroix, PRB 66, 014422 (2002)).
"""

import warnings

import numpy as np
import pytest

from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import polarized_state, triangular_heisenberg
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Site, SpinModel, Term
from spintoolkit.system.symmetry import (
    CrystalSymmetry, LayerCrystal, SymmetryError, close_group, find_symmetry,
    operation_from_rotation)

TRIANGULAR = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
ORIGIN = ("A", (0, 0))
NN = ("A", (1, 0))


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def flat_triangular():
    crystal = LayerCrystal(TRIANGULAR, [[0, 0]], ["Co"])
    return CrystalSymmetry(crystal, [Site("A", (0, 0), 0.5)])


def octahedral_triangular(height=0.6):
    """CdI2-type layer: anions above at (1/3, 1/3) and below at (2/3, 2/3)."""
    crystal = LayerCrystal(TRIANGULAR, [[0, 0], [1 / 3, 1 / 3], [2 / 3, 2 / 3]],
                           ["Yb", "O", "O"], [0.0, height, -height])
    return CrystalSymmetry(crystal, [Site("A", (0, 0), 0.5)])


def flat_kagome():
    positions = [(0, 0), (0.5, 0), (0, 0.5)]
    crystal = LayerCrystal(TRIANGULAR, positions, ["Cu"] * 3)
    return CrystalSymmetry(crystal, [Site(s, p, 0.5) for s, p in zip("ABC", positions)])


def span_contains(basis, matrix):
    coefficients = np.einsum("mab,ab->m", basis, matrix)
    return np.allclose(np.einsum("m,mab->ab", coefficients, basis), matrix)


def test_group_orders():
    assert flat_triangular().order == 24          # 6/mmm
    assert octahedral_triangular().order == 12    # -3m
    assert flat_kagome().order == 24


def test_magnetic_sites_only_overestimate_symmetry():
    model = triangular_heisenberg()
    assert len(find_symmetry(LayerCrystal.from_model(model))) == 24
    assert octahedral_triangular().order < 24


def test_flat_triangular_bond_has_xyz_form():
    basis = flat_triangular().allowed_exchange(ORIGIN, NN)
    assert len(basis) == 3
    for a in range(3):                             # bond along x: xx, yy, zz only
        assert span_contains(basis, np.diag(np.eye(3)[a]))
    assert all(np.allclose(b, b.T) for b in basis)


def test_octahedral_triangular_bond_has_four_parameters():
    symmetry = octahedral_triangular()
    basis = symmetry.allowed_exchange(ORIGIN, NN)
    assert len(basis) == 4
    assert all(np.allclose(b, b.T) for b in basis)     # inversion at the bond centre: no DM
    yz = np.zeros((3, 3))
    yz[1, 2] = yz[2, 1] = 1.0                          # J_z_pm couples z with the in-plane normal of the bond
    assert span_contains(basis, yz)
    xz = np.zeros((3, 3))
    xz[0, 2] = xz[2, 0] = 1.0
    assert not span_contains(basis, xz)
    g_basis = symmetry.allowed_g_tensor("A")
    assert len(g_basis) == 2
    assert span_contains(g_basis, np.diag([1.0, 1.0, 0.0]))
    assert span_contains(g_basis, np.diag([0.0, 0.0, 1.0]))


def test_flat_kagome_allows_only_normal_dm():
    symmetry = flat_kagome()
    basis = symmetry.allowed_exchange(("A", (0, 0)), ("B", (0, 0)))
    antisymmetric = [b - b.T for b in basis if not np.allclose(b, b.T)]
    assert len(basis) == 4
    assert len(antisymmetric) == 1
    D = antisymmetric[0]
    assert np.allclose(D[[0, 1, 2], [2, 2, 0]], 0)    # only D_z = J_xy - J_yx
    dm = np.array([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 0]])
    terms = symmetry.bilinear_terms(("A", (0, 0)), ("B", (0, 0)), np.eye(3) + 0.2 * dm)
    assert len(terms) == 6
    # The DM vector circulates uniformly: A->B, B->C and C->A carry the same D_z.
    by_bond = {(t.participants[0][0], t.participants[1][0], t.participants[1][1]): t.coefficient
               for t in terms}
    assert np.allclose(by_bond[("A", "B", (0, 0))], np.eye(3) + 0.2 * dm)
    assert np.allclose(by_bond[("B", "C", (0, 0))], np.eye(3) + 0.2 * dm)
    assert np.allclose(by_bond[("A", "C", (0, 0))].T, np.eye(3) + 0.2 * dm)


def test_triangular_orbit_rotates_the_exchange():
    J = np.diag([1.0, 0.7, 0.4])
    terms = flat_triangular().bilinear_terms(ORIGIN, NN, J, label="NN")
    assert len(terms) == 3
    lattice = TRIANGULAR
    for term in terms:
        (_, n1), (_, n2) = term.participants
        bond = (np.array(n2) - np.array(n1)) @ lattice
        R = rz(np.arctan2(bond[1], bond[0]))
        assert np.allclose(term.coefficient, R @ J @ R.T)
        assert term.label == "NN"


def test_symmetry_breaking_coefficients_are_rejected():
    symmetry = octahedral_triangular()
    J = np.eye(3)
    J[0, 2] = J[2, 0] = 0.1                            # forbidden J_xz on a bond along x
    with pytest.raises(SymmetryError, match="forbidden part"):
        symmetry.bilinear_terms(ORIGIN, NN, J)
    g = np.eye(3)
    g[0, 1] = 0.2
    with pytest.raises(SymmetryError, match="forbidden part"):
        symmetry.zeeman_terms("A", g)


def test_reversed_representative_gives_the_same_orbit():
    symmetry = flat_kagome()
    dm = np.array([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 0]])
    J = np.eye(3) + 0.3 * dm
    forward = symmetry.bilinear_terms(("A", (0, 0)), ("B", (0, 0)), J)
    backward = symmetry.bilinear_terms(("B", (0, 0)), ("A", (0, 0)), J.T)
    key = lambda t: (t.participants, t.coefficient.tobytes())
    assert sorted(map(key, forward)) == sorted(map(key, backward))
    basis_forward = symmetry.allowed_exchange(("A", (0, 0)), ("B", (0, 0)))
    basis_backward = symmetry.allowed_exchange(("B", (0, 0)), ("A", (0, 0)))
    assert all(span_contains(basis_backward, b.T) for b in basis_forward)


def test_check_model():
    assert flat_triangular().check_model(triangular_heisenberg()) == []
    assert octahedral_triangular().check_model(triangular_heisenberg()) == []
    broken = SpinModel(TRIANGULAR, [Site("A", (0, 0), 0.5)],
                       [Term.bilinear(ORIGIN, ("A", offset), np.diag([1.0, 0.5, 0.5]))
                        for offset in ((1, 0), (0, 1), (-1, 1))],
                       {"model_id": "same_matrix_on_every_bond"})
    assert flat_triangular().check_model(broken)
    missing = SpinModel(TRIANGULAR, [Site("A", (0, 0), 0.5)],
                        [Term.bilinear(ORIGIN, NN, np.eye(3))], {"model_id": "one_bond"})
    assert any("has no term" in v for v in flat_triangular().check_model(missing))


def test_generators_close_into_a_subgroup():
    inversion = operation_from_rotation(TRIANGULAR, -np.eye(3))
    c3 = operation_from_rotation(TRIANGULAR, rz(2 * np.pi / 3))
    group = close_group(TRIANGULAR, [inversion, c3])
    assert len(group) == 6                             # -3 (S6)
    crystal = octahedral_triangular().crystal
    symmetry = CrystalSymmetry(crystal, [Site("A", (0, 0), 0.5)], [inversion, c3])
    assert symmetry.order == 6
    # A smaller group allows more: -3 lets J_xz and J_yz mix on the bond.
    assert len(symmetry.allowed_exchange(ORIGIN, NN)) == 6
    with pytest.raises(SymmetryError):
        CrystalSymmetry(crystal, [Site("A", (0, 0), 0.5)],
                        [operation_from_rotation(TRIANGULAR, np.diag([1.0, 1.0, -1.0]))])


def test_site_outside_the_crystal_is_rejected():
    crystal = LayerCrystal(TRIANGULAR, [[0, 0]], ["Co"])
    with pytest.raises(SymmetryError, match="not an atom"):
        CrystalSymmetry(crystal, [Site("A", (0.5, 0), 0.5)])


def test_generated_model_has_c3_symmetric_magnons():
    """LSWT bands of a symmetric model about a C3-invariant state satisfy w(C3 k) = w(k)."""
    symmetry = octahedral_triangular()
    J = np.array([[-1.0, 0, 0], [0, -0.8, 0.15], [0, 0.15, -1.2]])
    terms = symmetry.bilinear_terms(ORIGIN, NN, J) + symmetry.zeeman_terms("A", np.diag([2.0, 2.0, 1.5]))
    model = SpinModel(TRIANGULAR, [Site("A", (0, 0), 1.0)], terms, {"model_id": "d3d_triangular"})
    assert symmetry.check_model(model) == []
    k = np.random.default_rng(0).uniform(-3, 3, size=(5, 2))
    rotated = k @ rz(2 * np.pi / 3)[:2, :2].T
    state = polarized_state(model, (0, 0, 1))
    conditions = ExternalConditions(field=(0, 0, 4.0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # the z state must be stationary
        bands = [solve_lswt(model, state, conditions,
                            settings=LSWTSettings(k_points=q)).bands() for q in (k, rotated)]
    assert np.allclose(bands[0], bands[1], atol=1e-10)
