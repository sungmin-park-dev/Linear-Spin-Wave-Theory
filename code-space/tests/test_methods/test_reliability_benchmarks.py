"""Theory-reliability benchmarks of 2026-10-21 (D69): literature and independent constructions.

B1  BCAO minimal model (``bcao_minimal``) against the cubic extended
    Kitaev-Heisenberg parametrization, Maksimov et al., arXiv:2204.09695
    (PRB 106, 165131 (2022)), Eqs. (1)-(3).
B2  BCAO classical energies against an independent minimizer (no spintoolkit).
B3  Luttinger-Tisza of the honeycomb J1-J3 model (J1 < 0): ferromagnet up to
    J3 = |J1|/4, spiral, zigzag (M point) from J3 = 0.4|J1| (arXiv:2204.09695,
    citing Rastelli, Tassi and Reatto, Physica B+C 97, 1 (1979)).
B4  Polarized BCAO state in an in-plane field: LSWT bands against an
    independent Holstein-Primakoff Bloch matrix.
B5  Square antiferromagnet: the 1/S magnon spectrum is the LSWT one times
    Z_c = 1 + (1 - <sqrt(1 - gamma^2)>)/(2S) = 1 + 0.1579/(2S) (Oguchi,
    Phys. Rev. 117, 117 (1960)).
B6  Triangular antiferromagnet: XY limit energy through order S^0 and
    Goldstone velocities at order 1/S (Chernyshev and Zhitomirsky,
    PRB 79, 144416 (2009), arXiv:0901.4803, Eqs. (56), (92)).
B7  Honeycomb Heisenberg antiferromagnet: E/(N J S^2) = -3/2 - 0.31476/S and
    M/S = 1 - 0.2582/S (Zheng, Oitmaa and Hamer, PRB 44, 11869 (1991), as
    quoted in Farnell et al., arXiv:1811.04229).
B8  Triangular antiferromagnet in a field, 1/3 plateau: the one-loop gaps of
    the up-up-down state at h = 3JS close at h_c1 = 3JS - 0.5JS/(2S) and
    h_c2 = 3JS + 1.3JS/(2S) (Chubukov and Golosov, J. Phys. Condens. Matter 3,
    69 (1991), as quoted in Starykh, arXiv:1412.8482, Sec. III A; two digits).
B9  Order by disorder at low field: zero-point energy selects the coplanar Y
    state over the classically degenerate cone; the 1/S susceptibilities
    chi = (1 + c/(2S))/(9J) against the sums of Keselman et al.,
    arXiv:2512.02150, End Matter Eqs. (3), (4) (J2 = 0), evaluated here.
B10 Canted BCAO states in fields along any direction (zigzag and double
    zigzag supercells): LSWT bands against an independent Holstein-Primakoff
    matrix in local frames built from the terms and the directions only. The
    polarized state softens exactly at the classical saturation field of the
    zigzag (M point), closed forms in the test.

The full tables are produced by
``code-space/examples/verification/theory_reliability_benchmarks.py``.
"""

import warnings

import numpy as np
import pytest
from scipy.linalg import cholesky
from scipy.optimize import minimize

from spintoolkit import ExternalConditions, LSWTSettings, SpinModel, SpinState, Site, Term
from spintoolkit import classical_search, solve_lswt
from spintoolkit.methods.luttinger_tisza import luttinger_tisza
from spintoolkit.methods.nlswt.run import NLSWTSettings, solve_nlswt
from spintoolkit.models import bcao_minimal, neel_state, polarized_state, square_heisenberg
from spintoolkit.models import state_120, triangular_heisenberg
from spintoolkit.models.bcao import THIRD_NEIGHBOUR_OFFSETS
from spintoolkit.models.heisenberg import TRIANGULAR_LATTICE

SQRT2 = np.sqrt(2.0)
A1, A2 = TRIANGULAR_LATTICE
R_B = (A1 + A2) / 3


# ---------------------------------------------------------------- B1 -----

def cubic_parameters(J, delta, jpp, jzp):
    """arXiv:2204.09695 Eq. (3): crystallographic (J, Delta, J++, Jz+-) -> (J, K, Gamma, Gamma')."""
    return {"J": (2 * J + delta * J + 2 * jpp - SQRT2 * jzp) / 3,
            "K": -2 * jpp + SQRT2 * jzp,
            "G": (-J + delta * J - 4 * jpp - SQRT2 * jzp) / 3,
            "Gp": (-2 * J + 2 * delta * J + 4 * jpp + SQRT2 * jzp) / 6}


def cubic_matrix(p, gamma):
    """Exchange matrix of a gamma bond in cubic axes, arXiv:2204.09695 Eq. (1)."""
    a, b = [i for i in range(3) if i != gamma]
    M = p["J"] * np.eye(3)
    M[gamma, gamma] += p["K"]
    M[a, b] = M[b, a] = p["G"]
    M[gamma, a] = M[a, gamma] = M[gamma, b] = M[b, gamma] = p["Gp"]
    return M


@pytest.mark.parametrize("J3, delta3, jzp", [(0.34, 0.0, 0.2), (0.45, 0.4, 0.9), (1.3, -0.2, -0.7)])
def test_bcao_couplings_are_the_cubic_kitaev_model_in_crystal_axes(J3, delta3, jzp):
    # Crystal axes in cubic coordinates: z along [111]; x along the in-plane projection
    # (1, 1, -2)/sqrt6 of -z_cubic, perpendicular to the Z bond [1, -1, 0]. The opposite
    # x is Jz+- -> -Jz+-, the C2 rotation about z (see bcao.py).
    x = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
    z = np.ones(3) / np.sqrt(3)
    R = np.array([x, np.cross(z, x), z])
    bond_direction = {0: np.array([0, 1, -1]) / SQRT2, 1: np.array([-1, 0, 1]) / SQRT2,
                      2: np.array([1, -1, 0]) / SQRT2}
    model = bcao_minimal(J1=-1.0, J3=J3, Delta3=delta3, Jzpm3=jzp)
    third = [t for t in model.terms if t.label == "J3"]
    assert len(third) == 3
    for term, offset in zip(third, THIRD_NEIGHBOUR_OFFSETS):
        r = R_B + offset[0] * A1 + offset[1] * A2
        r = np.r_[r / np.linalg.norm(r), 0.0]
        gamma = [g for g, d in bond_direction.items() if abs(abs((R @ d) @ r) - 1) < 1e-9]
        assert len(gamma) == 1
        expected = R @ cubic_matrix(cubic_parameters(J3, delta3, 0.0, jzp), gamma[0]) @ R.T
        np.testing.assert_allclose(term.coefficient, expected, atol=1e-13)


# ---------------------------------------------------------------- B2 -----

def independent_bcao_energy(n, J3, jzp, starts=30, seed=0):
    """Lowest classical energy per site (S = 1/2) on the stripe cell [[1, 0], [0, n]].

    Bonds written out from Maksimov, arXiv:2308.10672 Eq. (1), Delta1 = Delta3 = 0.
    """
    nn = [((0, 0), 2 * np.pi / 3), ((-1, 0), -2 * np.pi / 3), ((0, -1), 0.0)]
    third = {(0, 0): (-1, -1), (-1, 0): (1, -1), (0, -1): (-1, 1)}
    bonds = []
    for offset, phi in nn:
        bonds.append((offset, -np.diag([1.0, 1.0, 0.0])))
        u = np.outer([np.cos(phi), np.sin(phi), 0.0], [0.0, 0.0, 1.0])
        bonds.append((third[offset], J3 * np.diag([1.0, 1.0, 0.0]) - jzp * (u + u.T)))
    i, j, M = [], [], []
    for c in range(n):
        for (d1, d2), Jb in bonds:
            i.append(c)
            j.append((c + d2) % n)          # cell (0, c) + (d1, d2) wraps to (0, (c + d2) mod n)
            M.append(Jb)
    i, j, M = np.array(i), np.array(j), np.array(M)

    def energy(x):
        th, ph = x[:2 * n], x[2 * n:]
        s = np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]
        return 0.25 * np.einsum("pa,pab,pb->", s[:n][i], M, s[n:][j]) / (2 * n)

    rng = np.random.default_rng(seed)
    return min(minimize(energy, np.r_[rng.uniform(0, np.pi, 2 * n), rng.uniform(0, 2 * np.pi, 2 * n)],
                        method="L-BFGS-B").fun for _ in range(starts))


@pytest.mark.parametrize("n, jzp, expected", [(4, 0.7, -0.341523), (3, 0.7, -0.333370)])
def test_bcao_classical_energies_match_an_independent_minimizer(n, jzp, expected):
    """J3 = 0.45: the canted double zigzag (n = 4) lies below the up-up-down state (n = 3)
    already at Jz+- = 0.7, where arXiv:2308.10672 Fig. 3 still shows up-up-down; both
    codes agree, so that boundary is not a package error."""
    model = bcao_minimal(J3=0.45, Jzpm3=jzp)
    found = classical_search(model, [[1, 0], [0, n]], maxiter=300, seed=1).energy
    assert independent_bcao_energy(n, 0.45, jzp) == pytest.approx(expected, abs=2e-6)
    assert found == pytest.approx(expected, abs=2e-6)


# ---------------------------------------------------------------- B3 -----

def honeycomb_lt_minimum(J3, n=240):
    """Independent LT: maximize |f(q)| = |J1 sum e^{iq.d} + J3 sum e^{-2iq.d}| (J1 = -1)."""
    deltas = [R_B + o[0] * A1 + o[1] * A2 for o in ((0, 0), (-1, 0), (0, -1))]
    B = 2 * np.pi * np.linalg.inv(np.array([A1, A2])).T
    u = np.arange(n) / n
    q = np.stack(np.meshgrid(u, u), -1).reshape(-1, 2) @ B
    f = (-sum(np.exp(1j * q @ d) for d in deltas) + J3 * sum(np.exp(-2j * q @ d) for d in deltas))
    return np.abs(f).max()


@pytest.mark.parametrize("J3, phase", [(0.2, "FM"), (0.24, "FM"), (0.3, "spiral"),
                                       (0.35, "spiral"), (0.41, "zigzag"), (0.45, "zigzag")])
def test_honeycomb_j1_j3_luttinger_tisza_phases(J3, phase):
    model = bcao_minimal(J1=-1.0, J3=J3, Delta1=1.0, Delta3=1.0, g=None)
    report = luttinger_tisza(model)
    # per-site LT energy -S^2 max|f| / 2 with S = 1/2
    assert report.lambda_min == pytest.approx(-honeycomb_lt_minimum(J3) / 8, rel=1e-4)
    q = np.mod(report.minima[0].fractional + 1e-9, 1.0)
    if phase == "FM":
        np.testing.assert_allclose(np.minimum(q, 1 - q), 0, atol=1e-6)
    elif phase == "zigzag":                        # an M point: 2q integer, q != 0
        np.testing.assert_allclose(2 * q, np.rint(2 * q), atol=1e-6)
        assert np.any(np.abs(q - 0.5) < 1e-6)
    else:
        assert not report.minima[0].commensurate


# ---------------------------------------------------------------- B4 -----

def independent_polarized_bands(model, S, h, n, k):
    """Holstein-Primakoff Bloch matrix of a ferromagnet along n, built from the terms."""
    e1 = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = e1 - n * (n @ e1)
    e1 /= np.linalg.norm(e1)
    u = e1 + 1j * np.cross(n, e1)
    positions = {"A": np.zeros(2), "B": R_B}
    index = {"A": 0, "B": 1}

    def blocks(kk):
        A = np.zeros((2, 2), complex)
        Bm = np.zeros((2, 2), complex)
        for term in model.terms:
            if term.kind != "bilinear":
                continue
            (si, ci), (sj, cj) = term.participants
            J, i, j = term.coefficient, index[si], index[sj]
            r = (positions[sj] + cj[0] * A1 + cj[1] * A2) - (positions[si] + ci[0] * A1 + ci[1] * A2)
            phase = np.exp(1j * kk @ r)
            A[i, j] += S / 2 * (u @ J @ u.conj()) * phase
            A[j, i] += S / 2 * (u.conj() @ J @ u) / phase
            Bm[i, j] += S / 2 * (u @ J @ u) * phase
            Bm[j, i] += S / 2 * (u @ J @ u) / phase
            A[i, i] -= S * (n @ J @ n)
            A[j, j] -= S * (n @ J @ n)
        return A + np.eye(2) * (h @ n), Bm

    A, Bm = blocks(k)
    Am, _ = blocks(-k)
    K = cholesky(np.block([[A, Bm], [Bm.conj().T, Am.T]]))
    w = np.linalg.eigvalsh(K @ np.diag([1, 1, -1, -1]) @ K.conj().T)
    return np.sort(w[w > 0])


@pytest.mark.parametrize("J3, d1, d3, jzp, h", [(0.34, 0.0, 0.0, 0.2, 0.4), (0.45, 0.4, 0.4, 0.9, 2.0),
                                               (0.34, 0.2, 0.6, -0.5, 1.5)])
def test_polarized_bcao_bands_match_an_independent_holstein_primakoff_matrix(J3, d1, d3, jzp, h):
    model = bcao_minimal(J3=J3, Delta1=d1, Delta3=d3, Jzpm3=jzp)
    field = np.array([h, 0.0, 0.0])
    k = np.random.default_rng(0).uniform(-6, 6, (6, 2))
    bands = solve_lswt(model, polarized_state(model, (1, 0, 0)), ExternalConditions(field=tuple(field)),
                       settings=LSWTSettings(k_points=k)).bands()
    own = [independent_polarized_bands(model, 0.5, field, np.array([1.0, 0, 0]), q) for q in k]
    np.testing.assert_allclose(np.sort(bands, axis=1), own, atol=1e-12)


# ---------------------------------------------------------------- B5 -----

@pytest.mark.parametrize("S", [0.5, 1.0])
def test_square_one_over_s_magnons_are_uniformly_renormalized(S):
    n = 800
    u = (np.arange(n) + 0.5) / n * 2 * np.pi
    gamma = 0.5 * np.add.outer(np.cos(u), np.cos(u))
    z_c = 1 + (1 - np.mean(np.sqrt(1 - gamma ** 2))) / (2 * S)
    assert (z_c - 1) * 2 * S == pytest.approx(0.1579, abs=1e-4)          # Oguchi 0.158
    model = square_heisenberg(J=1.0, S=S)
    result = solve_nlswt(model, neel_state(model, (1, 0, 0)), settings=NLSWTSettings(mesh=(24, 24)))
    k = np.array([[0.3, 0.1], [0.8, 0.8], [3.0, 0.2], [1.1, 2.4]])
    out = result.magnon_energies(k)
    np.testing.assert_allclose(out["total"] / out["lswt"], z_c, rtol=2e-5)


# ---------------------------------------------------------------- B6 -----

def triangular_xy(S):
    terms = [Term.bilinear(("A", (0, 0)), ("A", d), np.diag([1.0, 1.0, 0.0]))
             for d in [(1, 0), (0, 1), (-1, 1)]]
    return SpinModel(TRIANGULAR_LATTICE, (Site("A", (0, 0), S),), tuple(terms), {"model_id": "tri_xy"})


def test_triangular_xy_energy_through_order_s0_matches_chernyshev_zhitomirsky_eq92():
    S = 0.5
    model = triangular_xy(S)
    state = state_120(model)
    lswt = solve_lswt(model, state, settings=LSWTSettings(mesh=(48, 48)))
    assert lswt.ground_state_energy == pytest.approx(-1.5 * S ** 2 * (1 + 0.064515 / (2 * S)), abs=1e-6)
    e = solve_nlswt(model, state, settings=NLSWTSettings(mesh=(12, 12))).energies
    assert e.order_s0 == pytest.approx(-1.5 * S ** 2 * 0.013326 / (2 * S) ** 2, rel=3e-4)


def test_triangular_goldstone_velocities_at_order_one_over_s_on_the_mesh_grid():
    """Eq. (56): v0 (1 - 0.11488/2S), vQ (1 + 0.08285/2S). At k = b1/N (on the mesh grid)
    the ratios agree to 2 %; the remainder is the k^2 term and the mesh."""
    S = 0.5
    model = triangular_heisenberg(J=1.0, S=S)
    state = state_120(model)
    result = solve_nlswt(model, state, settings=NLSWTSettings(mesh=(24, 24)))
    b = 2 * np.pi * np.linalg.inv(state.magnetic_lattice(model)).T
    out = result.magnon_energies((b[0] / 24)[None])
    ratio = out["total"][0] / out["lswt"][0]          # bands: Gamma mode, then the two +-Q modes
    assert ratio[0] == pytest.approx(1 - 0.11488 / (2 * S), rel=0.02)
    np.testing.assert_allclose(ratio[1:], 1 + 0.08285 / (2 * S), rtol=0.025)


def test_unresolved_goldstone_cancellation_off_the_mesh_grid_is_nan():
    """D69: at k = 0.045 b1 + ... (off the grid) the +-Q shifts are each ~50 times the
    LSWT energy and cancel; the mesh gives 1.4 instead of ~1.08, so the result is NaN."""
    model = triangular_heisenberg(J=1.0, S=0.5)
    state = state_120(model)
    result = solve_nlswt(model, state, settings=NLSWTSettings(mesh=(24, 24)))
    b = 2 * np.pi * np.linalg.inv(state.magnetic_lattice(model)).T
    k = np.array([0.045 * b[0] + 0.0167 * b[1], 0.3 * b[0] + 0.111 * b[1]])
    with pytest.warns(UserWarning, match="cancellation"):
        out = result.magnon_energies(k)
    assert np.isfinite(out["total"][0, 0]) and np.all(np.isnan(out["total"][0, 1:]))
    assert np.all(np.isfinite(out["total"][1]))
    with pytest.warns(UserWarning, match="cancellation"):
        spectral = result.spectral_function(k[:1], np.linspace(0, 2, 5), 0.05, grid=(24, 24))
    assert np.all(np.isnan(spectral["energy"][0, 1:])) and np.all(np.isnan(spectral["spectral"][0, 1:]))


# ---------------------------------------------------------------- B7 -----

def honeycomb_neel(S):
    model = bcao_minimal(J1=1.0, J3=0.0, Delta1=1.0, S=S, g=None)
    state = SpinState.from_function(model, [[1, 0], [0, 1]],
                                    lambda s, c: np.array([1.0, 0, 0]) * (1 if s == "A" else -1))
    return model, state


def test_honeycomb_antiferromagnet_matches_spin_wave_literature():
    S = 0.5
    model, state = honeycomb_neel(S)
    lswt = solve_lswt(model, state, settings=LSWTSettings(mesh=(64, 64)))
    alpha = (lswt.ground_state_energy / S ** 2 + 1.5) * S
    assert alpha == pytest.approx(-0.31476, abs=2e-5)
    # the moment converges like 1/N (Goldstone modes); 0.2418 at N -> infinity
    assert S - lswt.ordered_moments()[0] == pytest.approx(0.2582, abs=1.5e-2)
    e = solve_nlswt(model, state, settings=NLSWTSettings(mesh=(24, 24))).energies
    assert e.total == pytest.approx(-0.5489, abs=1e-4)


# ---------------------------------------------------------------------------
# B8, B9: triangular antiferromagnet in a field along z
# ---------------------------------------------------------------------------

def _sublattice_state(model, directions):
    """Three-sublattice state on the sqrt3 x sqrt3 cell, sublattice (n1 - n2) mod 3."""
    return SpinState.from_function(model, [[1, 1], [-1, 2]],
                                   lambda site, cell: np.asarray(directions[(cell[0] - cell[1]) % 3]))


@pytest.mark.parametrize("S", [0.5, 1.0])
def test_triangular_plateau_critical_fields_match_chubukov_golosov(S):
    """Collinear: no cubic vertex; the Hartree-Fock shifts of the two soft modes at k -> 0
    are h_cl - h_c1 and h_c2 - h_cl (the field moves the two modes by +-(h - h_cl))."""
    model = triangular_heisenberg(J=1.0, S=S)
    state = _sublattice_state(model, [(0, 0, -1.0), (0, 0, 1.0), (0, 0, 1.0)])
    conditions = ExternalConditions(field=(0, 0, 3 * S))
    n = 24
    result = solve_nlswt(model, state, conditions, settings=NLSWTSettings(mesh=(n, n)))
    b = 2 * np.pi * np.linalg.inv(state.magnetic_lattice(model)).T
    lswt, static, cubic = result.solver.magnon_energies(np.array([b[0] / n, 2 * b[0] / n]))
    assert np.all(cubic == 0)
    rising, falling = np.argsort(lswt[0])[:2]           # gapless at k = 0; mode 1 is the lower
    shift = (4 * static[0] - static[1]) / 3              # k^2 extrapolation to k = 0
    h_c1, h_c2 = 3 * S - shift[rising], 3 * S + shift[falling]
    assert 3 * S - h_c1 == pytest.approx(0.5 * S / (2 * S), abs=0.01)
    assert h_c2 - 3 * S == pytest.approx(1.3 * S / (2 * S), abs=0.01)


def _triangular_susceptibility_sums(m=1200):
    b1 = 2 * np.pi * np.array([1, -1 / np.sqrt(3)])
    b2 = 2 * np.pi * np.array([0, 2 / np.sqrt(3)])
    f = (np.arange(m) + 1 / 3) / m
    k = (f[:, None, None] * b1 + f[None, :, None] * b2).reshape(-1, 2)

    def parts(q):
        g = (np.cos(q[:, 0]) + 2 * np.cos(q[:, 0] / 2) * np.cos(np.sqrt(3) * q[:, 1] / 2)) / 3
        a, b = 1 + g / 2, -1.5 * g
        return g, a, b, np.sqrt(np.maximum(a * a - b * b, 0))

    Q = np.array([4 * np.pi / 3, 0])
    g, a, b, E = parts(k)
    _, ap, bp, Ep = parts(k + Q)
    _, am, bm, Em = parts(k - Q)
    c_y = -np.mean((a - b) * g / E) + 3 * np.mean((ap - bp) * (am - bm) * g ** 2 / (Ep * Em * (Ep + Em)))
    c_cone = np.mean((a + b) * g / E)
    return c_y, c_cone


def test_zero_point_energy_selects_the_y_state_with_the_literature_susceptibility():
    S = 0.5
    model = triangular_heisenberg(J=1.0, S=S)
    fields = np.array([0.005, 0.01, 0.015])
    energies = {"Y": [], "cone": []}
    for h in fields:
        mz = h / (3 * S)
        t, p = np.arccos((mz + 1) / 2), np.arccos(mz / 3)
        states = {"Y": [(0, 0, -1.0), (np.sin(t), 0, np.cos(t)), (-np.sin(t), 0, np.cos(t))],
                  "cone": [(np.sin(p) * np.cos(a), np.sin(p) * np.sin(a), np.cos(p))
                           for a in 2 * np.pi * np.arange(3) / 3]}
        for name, directions in states.items():
            result = solve_lswt(model, _sublattice_state(model, directions),
                                ExternalConditions(field=(0, 0, h)), settings=LSWTSettings(mesh=(120, 120)))
            energies[name].append(result.ground_state_energy)
    fit = np.vstack([np.ones(3), fields ** 2, fields ** 4]).T
    c = {name: (-2 * np.linalg.lstsq(fit, np.array(e), rcond=None)[0][1] * 9 - 1) * 2 * S
         for name, e in energies.items()}
    c_y, c_cone = _triangular_susceptibility_sums()
    assert np.all(np.array(energies["Y"]) < np.array(energies["cone"]))
    assert c["Y"] == pytest.approx(c_y, abs=0.005)
    assert c["cone"] == pytest.approx(c_cone, abs=0.005)


# ---------------------------------------------------------------- B10 ----

def independent_canted_bands(model, state, field, k_points):
    """LSWT of a noncollinear supercell state from the terms and the directions only.

    Local frame per magnetic sublattice: n along the spin, u = e + i n x e; bond i -> j
    with exchange J adds sqrt(S_i S_j)/2 (u_i J u_j^*, u_i J u_j) to (A, B) and
    -S_j n_i J n_j, -S_i n_i J n_j to the diagonal; the field -b.S adds b.n_i (g = 1).
    """
    spins = {s.id: s.spin for s in model.sites}
    pos = {s.id: s.position @ model.lattice for s in model.sites}
    inverse = np.linalg.inv(np.asarray(state.supercell, dtype=float))
    keys = sorted(state.directions)
    index = {key: i for i, key in enumerate(keys)}
    n = {key: np.asarray(v) / np.linalg.norm(v) for key, v in state.directions.items()}
    u = {}
    for key, v in n.items():
        e = np.array([0, 0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0, 0])
        e = e - v * (v @ e)
        e /= np.linalg.norm(e)
        u[key] = e + 1j * np.cross(v, e)

    def sublattice(site, cell):
        for s, c in keys:
            x = (cell - np.asarray(c)) @ inverse
            if s == site and np.allclose(x, np.rint(x), atol=1e-9):
                return s, c
        raise RuntimeError("no sublattice")                      # pragma: no cover

    def blocks(k):
        A = np.diag([field @ n[key] for key in keys]).astype(complex)
        B = np.zeros_like(A)
        for term in model.terms_of_kind("bilinear"):
            (si, ci), (sj, cj) = term.participants
            J = np.asarray(term.coefficient)
            for s0, c0 in keys:
                if s0 != si:
                    continue
                cell_j = np.asarray(c0) + np.asarray(cj) - np.asarray(ci)
                ki, kj = (s0, c0), sublattice(sj, cell_j)
                r = pos[sj] + cell_j @ model.lattice - pos[si] - np.asarray(c0) @ model.lattice
                phase, amp = np.exp(1j * k @ r), np.sqrt(spins[si] * spins[sj]) / 2
                i, j = index[ki], index[kj]
                A[i, j] += amp * (u[ki] @ J @ u[kj].conj()) * phase
                A[j, i] += amp * (u[kj] @ J.T @ u[ki].conj()) / phase
                B[i, j] += amp * (u[ki] @ J @ u[kj]) * phase
                B[j, i] += amp * (u[kj] @ J.T @ u[ki]) / phase
                A[i, i] -= spins[sj] * (n[ki] @ J @ n[kj])
                A[j, j] -= spins[si] * (n[ki] @ J @ n[kj])
        return A, B

    ns, bands = len(keys), []
    for k in k_points:
        (A, B), (Am, _) = blocks(k), blocks(-k)
        K = cholesky(np.block([[A, B], [B.conj().T, Am.T]]))
        w = np.linalg.eigvalsh(K @ np.diag([1.0] * ns + [-1.0] * ns) @ K.conj().T)
        bands.append(np.sort(w[w > 0]))
    return np.array(bands)


@pytest.mark.parametrize("params, field, cell", [
    ((0.45, 0.1, 0.2, 0.2), (0.3, 0.0, 0.0), [[1, 0], [0, 2]]),
    ((0.45, 0.1, 0.2, 0.2), (0.0, 0.0, 0.5), [[1, 0], [0, 2]]),
    ((0.45, 0.1, 0.2, 0.2), (0.2, 0.1, 0.3), [[1, 0], [0, 2]]),
    ((0.45, 0.0, 0.0, 0.7), (0.1, 0.0, 0.0), [[1, 0], [0, 4]]),
])
def test_canted_bcao_bands_match_an_independent_holstein_primakoff_matrix(params, field, cell):
    J3, d1, d3, jzp = params
    model = bcao_minimal(J3=J3, Delta1=d1, Delta3=d3, Jzpm3=jzp)
    conditions = ExternalConditions(field=field)
    state = classical_search(model, cell, conditions, maxiter=300).state
    magnetization = np.mean(list(state.directions.values()), axis=0)
    assert 0.01 < np.linalg.norm(magnetization) < 0.99          # canted, not collinear
    k = np.random.default_rng(1).uniform(-6, 6, (5, 2))
    bands = solve_lswt(model, state, conditions, settings=LSWTSettings(k_points=k)).bands()
    np.testing.assert_allclose(bands, independent_canted_bands(model, state, np.array(field), k),
                               atol=1e-12)


@pytest.mark.parametrize("J3, d1, d3, jzp", [(0.45, 0.1, 0.2, 0.2), (0.6, -0.3, 0.5, -0.8)])
def test_polarized_bcao_softens_at_the_zigzag_saturation_field(J3, d1, d3, jzp):
    """J1 = -1, J3 > 0.4: the softest transverse mode of the polarized state is at M, where
    f1 = -1 and f3 = 3 (third neighbours at -2 x the first). Along z,
    h_c = S (|J1 - 3 J3| + 3 (J1 Delta1 + J3 Delta3)) for any Jz+-, which couples the
    transverse x, y only to the longitudinal z. In the plane Jz+- couples the two transverse
    components (y and z for a field along x), so h_ab = S (|J1 - 3 J3| - |3 J1 + 3 J3|) is
    checked at Jz+- = 0; at Jz+- = -0.8 the polarized state is unstable there."""
    from spintoolkit.methods.lswt.run import LSWTError

    S, J1 = 0.5, -1.0
    h_ab = S * (abs(J1 - 3 * J3) - abs(3 * J1 + 3 * J3))
    h_c = S * (abs(J1 - 3 * J3) + 3 * (J1 * d1 + J3 * d3))
    cases = [(jzp, (0.0, 0.0, 1.0), h_c), (0.0, (1.0, 0.0, 0.0), h_ab), (0.0, (0.0, 1.0, 0.0), h_ab)]
    for coupling, direction, h_s in cases:
        model = bcao_minimal(J1=J1, J3=J3, Delta1=d1, Delta3=d3, Jzpm3=coupling, S=S)
        state, d = polarized_state(model, direction), np.array(direction)
        settings = LSWTSettings(mesh=(24, 24), shift=False)
        above = solve_lswt(model, state, ExternalConditions(field=tuple(1.001 * h_s * d)), settings=settings)
        assert above.bands().min() > 0
        with pytest.raises(LSWTError):
            solve_lswt(model, state, ExternalConditions(field=tuple(0.999 * h_s * d)), settings=settings)
