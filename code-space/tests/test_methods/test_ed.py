"""Minimal exact diagonalization (stage 3, D10, D23).

1. General Hamiltonians without any symmetry against an independent
   Kronecker-product construction.
2. Sector and momentum reductions reproduce the full spectrum.
3. With U(1) symmetry about the field, the polarized state and the one-magnon
   states are exact eigenstates, and the one-magnon energies equal the LSWT
   bands at every torus momentum. A DM term makes omega(k) != omega(-k), so
   this also checks the momentum sign (D13).
4. The 4 x 4 square S = 1/2 Heisenberg ground state.
"""

from functools import reduce

import numpy as np
import pytest

from model import nbcp
from spintoolkit.methods.classical import classical_energy
from spintoolkit.methods.ed import EDSector, SectorError, solve_ed
from spintoolkit.methods.lswt.hamiltonian import LSWTHamiltonian
from spintoolkit.models import square_heisenberg, triangular_heisenberg
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.cluster import allowed_momenta, expand_on_torus
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.conversion import to_spin_system
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import Site, SpinModel, Term

HEX = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])


def torus(L):
    return CalculationGeometry.finite_torus(L)


def with_dm(model, D):
    K = np.array([[0, D, 0], [-D, 0, 0], [0, 0, 0.0]])
    terms = [Term.bilinear(*t.participants, t.coefficient + K, t.label)
             for t in model.terms_of_kind("bilinear")] + list(model.terms_of_kind("zeeman"))
    return SpinModel(model.lattice, model.sites, terms, {"model_id": f"dm_{D}"})


def with_spin(model, S):
    return SpinModel(model.lattice, [Site(s.id, s.position, S) for s in model.sites],
                     model.terms, {"model_id": f"spin_{S}"})


def honeycomb():
    """Two-site basis, DM on some bonds, different g-tensors per site."""
    dm = np.array([[0, 0.2, 0], [-0.2, 0, 0], [0, 0, 0.0]])
    return SpinModel(HEX, [Site("A", (0, 0), 0.5), Site("B", (1 / 3, 1 / 3), 0.5)], [
        Term.bilinear(("A", (0, 0)), ("B", (0, 0)), np.eye(3) + dm),
        Term.bilinear(("A", (0, 0)), ("B", (-1, 0)), np.eye(3)),
        Term.bilinear(("A", (0, 0)), ("B", (0, -1)), 0.7 * np.eye(3)),
        Term.bilinear(("A", (0, 0)), ("A", (1, 0)), dm / 2),
        Term.zeeman("A", np.diag([1, 1, 2.0])), Term.zeeman("B", np.diag([1, 1, 1.5]))],
        {"model_id": "honeycomb_test"})


def random_model(seed=5):
    """No symmetry: random asymmetric exchange, S = 1/2 and 1, random g."""
    rng = np.random.default_rng(seed)
    return SpinModel(HEX, [Site("A", (0, 0), 0.5), Site("B", (0.3, 0.4), 1.0)], [
        Term.bilinear(("A", (0, 0)), ("B", (0, 0)), rng.normal(size=(3, 3))),
        Term.bilinear(("B", (0, 0)), ("A", (1, 0)), rng.normal(size=(3, 3))),
        Term.bilinear(("A", (0, 0)), ("A", (0, 1)), rng.normal(size=(3, 3))),
        Term.zeeman("A", rng.normal(size=(3, 3))), Term.zeeman("B", np.eye(3))],
        {"model_id": "random_test"})


def spin_matrices(S):
    m = np.arange(S, -S - 1, -1)
    raising = np.zeros((len(m), len(m)))
    for a in range(1, len(m)):
        raising[a - 1, a] = np.sqrt(S * (S + 1) - m[a] * (m[a] + 1))
    return [(raising + raising.T) / 2, (raising - raising.T) / 2j, np.diag(m)]


def kronecker_spectrum(model, L, conditions):
    """Independent reference: dense Kronecker products of spin matrices."""
    cluster = expand_on_torus(model, torus(L))
    ops = [spin_matrices(S) for S in cluster.spins]
    dims = [len(o[2]) for o in ops]

    def embed(i, op):
        return reduce(np.kron, [op if j == i else np.eye(d) for j, d in enumerate(dims)])

    H = 0
    for i, j, J in zip(cluster.source, cluster.target, cluster.exchange):
        for a in range(3):
            for b in range(3):
                if J[a, b]:
                    H = H + J[a, b] * embed(i, ops[i][a]) @ embed(j, ops[j][b])
    for i, h in enumerate(cluster.fields(conditions)):
        for a in range(3):
            if h[a]:
                H = H - h[a] * embed(i, ops[i][a])
    return np.linalg.eigvalsh(H)


def lswt_bands(model, state, conditions, k_points):
    """Positive eigenvalues of sigma_3 H(k) from the existing LSWT, no regularization."""
    system = to_spin_system(model, state, conditions)
    data = system.to_legacy_dict("simple")
    H, _ = LSWTHamiltonian(data["Spin info"], data["Couplings"]).Quadratic_Bose_Hamiltonian(
        k_points, angles=system.get_angles_flat())
    H = np.asarray(H)
    n = H.shape[1] // 2
    sigma = np.diag(np.r_[np.ones(n), -np.ones(n)])
    return np.array([np.sort(np.linalg.eigvals(sigma @ Hk).real)[n:] for Hk in H])


@pytest.mark.parametrize("model, L, conditions", [
    (random_model(), [[1, 1], [-1, 1]], ExternalConditions(field=(0.3, -0.2, 0.5))),
    (random_model(), [[1, 1], [-1, 2]], ExternalConditions(field=(0.3, -0.2, 0.5))),
    (nbcp.build_model({"Jxy": 0.075, "Jz": 0.125, "JGamma": 0.03, "JPD": 0.02, "Dz": 0.01}),
     [[3, 0], [0, 3]], ExternalConditions(field=(0.02, 0, 0.05))),
], ids=["random_two_cells", "random_three_cells", "nbcp_soc"])
def test_general_hamiltonian_matches_kronecker_construction(model, L, conditions):
    reference = kronecker_spectrum(model, L, conditions)
    for sector in (EDSector(), EDSector(momenta="all"), EDSector(axis=(0.3, -0.5, 0.8))):
        result = solve_ed(model, torus(L), conditions, sector)
        np.testing.assert_allclose(result.energies(), reference, rtol=0, atol=1e-12)
        assert result.diagnostics["hermiticity"] < 1e-13


def test_magnetization_and_momentum_blocks_reproduce_the_full_spectrum():
    model = square_heisenberg(J=1.0)
    full = solve_ed(model, torus([[2, 0], [0, 3]]))
    blocks = solve_ed(model, torus([[2, 0], [0, 3]]),
                      sector=EDSector(axis=(1, 1, 0), magnon_number="all", momenta="all"))
    np.testing.assert_allclose(blocks.energies(), full.energies(), rtol=0, atol=1e-13)
    assert sum(b.dimension for b in blocks.blocks) == 2 ** 6


def test_broken_u1_sector_is_rejected():
    model = nbcp.build_model({"Jxy": 0.075, "Jz": 0.125, "JGamma": 0.01})
    with pytest.raises(SectorError, match="not conserved"):
        solve_ed(model, torus([[3, 0], [0, 3]]), ExternalConditions(field=(0, 0, 1.0)),
                 EDSector(axis=(0, 0, 1), magnon_number=1))


CASES = {
    "square_S1/2": (square_heisenberg(J=1.0), [[4, 0], [0, 4]], 5.0, (0, 0, 1)),
    "square_S1_2x3": (with_spin(square_heisenberg(J=1.0), 1.0), [[2, 0], [0, 3]], 9.0, (0, 0, 1)),
    "triangular_S1/2": (triangular_heisenberg(J=1.0), [[3, 0], [0, 3]], 5.0, (0, 0, 1)),
    "triangular_S3/2_nondiagonal": (with_spin(triangular_heisenberg(J=1.0), 1.5),
                                    [[2, 1], [-1, 3]], 14.0, (0, 0, 1)),
    "square_DM": (with_dm(square_heisenberg(J=1.0), 0.3), [[4, 0], [0, 4]], 6.0, (0, 0, 1)),
    "triangular_DM_minus_z": (with_dm(triangular_heisenberg(J=1.0), 0.3), [[3, 0], [0, 4]], 6.0,
                              (0, 0, -1)),
    "square_tilted_axis": (square_heisenberg(J=1.0), [[3, 0], [0, 3]], 5.0, (1 / 3, 2 / 3, 2 / 3)),
    "honeycomb_two_sites": (honeycomb(), [[3, 0], [0, 3]], 4.0, (0, 0, 1)),
    "nbcp_xxz": (nbcp.build_model({"Jxy": 0.075, "Jz": 0.125}), [[3, 0], [0, 3]], 1.0, (0, 0, 1)),
}


@pytest.mark.parametrize("name", CASES)
def test_one_magnon_energies_equal_lswt_at_every_torus_momentum(name):
    model, L, h, axis = CASES[name]
    axis = np.asarray(axis, dtype=float)
    conditions = ExternalConditions(field=h * axis)
    geometry = torus(L)
    expand_on_torus(model, geometry)            # the LSWT comparison obeys D23 too
    state = SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: axis)
    _, k = allowed_momenta(model, geometry)
    bands = lswt_bands(model, state, conditions, k)
    result = solve_ed(model, geometry, conditions,
                      EDSector(axis=tuple(axis), magnon_number=1, momenta="all"))
    for block in result.blocks:
        np.testing.assert_allclose(result.excitations(block), bands[block.momentum_index],
                                   rtol=0, atol=1e-12)
    # The polarized state: exact eigenstate with the classical energy.
    polarized = solve_ed(model, geometry, conditions,
                         EDSector(axis=tuple(axis), magnon_number=0))
    assert polarized.blocks[0].energies[0] == pytest.approx(result.reference_energy, abs=1e-13)
    assert result.reference_energy == pytest.approx(
        classical_energy(model, state, conditions, geometry) * result.num_sites, abs=1e-12)


@pytest.mark.parametrize("name", ["square_DM", "honeycomb_two_sites"])
def test_momentum_sign_is_fixed(name):
    """With DM, pairing ED at k with LSWT at -k fails: the sign convention is tested."""
    model, L, h, axis = CASES[name]
    conditions = ExternalConditions(field=h * np.asarray(axis, float))
    state = SpinState.from_function(model, np.eye(2, dtype=int), lambda s, c: np.asarray(axis, float))
    _, k = allowed_momenta(model, torus(L))
    flipped = lswt_bands(model, state, conditions, -k)
    result = solve_ed(model, torus(L), conditions, EDSector(axis=axis, magnon_number=1, momenta="all"))
    assert max(np.max(np.abs(result.excitations(b) - flipped[b.momentum_index]))
               for b in result.blocks) > 0.1


def test_square_4x4_ground_state():
    """E_0 / N = -0.7017802 J for the 16-site S = 1/2 Heisenberg antiferromagnet."""
    result = solve_ed(square_heisenberg(J=1.0), torus([[4, 0], [0, 4]]),
                      sector=EDSector(axis=(0, 0, 1), magnon_number=8, momenta=(0, 10)),
                      num_eigenvalues=1)
    ground = min(b.energies[0] for b in result.blocks)
    assert ground / 16 == pytest.approx(-0.7017802, abs=5e-8)
    assert all(b.solver == "lanczos" and b.residual < 1e-8 for b in result.blocks)


def test_result_normalizations():
    model = square_heisenberg(J=1.0)
    result = solve_ed(model, torus([[2, 0], [0, 2]]), ExternalConditions(field=(0, 0, 5.0)),
                      EDSector(axis=(0, 0, 1), magnon_number="all"))
    np.testing.assert_allclose(result.per_site(), result.energies() / 4)
    zero = result.block(magnon_number=0)
    assert result.excitations(zero)[0] == pytest.approx(0.0, abs=1e-14)
    with pytest.raises(ValueError, match="needs an axis"):
        EDSector(magnon_number=1)
