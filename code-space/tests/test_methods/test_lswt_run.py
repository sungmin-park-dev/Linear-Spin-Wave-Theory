"""LSWT on the common model types (stage 4a).

1. The new entry point reproduces the existing LSWTSolver (same momenta, MAGSWT).
2. Square and triangular Heisenberg ground-state energies and moment
   reductions against the analytic LSWT values:
   square  E/N = -2JS(S+1) + 2JS <sqrt(1 - gamma_k^2)>, dS = <1/(2 sqrt(1-gamma^2))> - 1/2
   (independent quadrature below); triangular E/N = -3/2 J S^2 (1 + 0.436824/(2S)),
   dS = 0.2613032 (Chernyshev and Zhitomirsky, PRB 79, 144416 (2009), Eqs. (42), (18)).
3. Zero modes are reported instead of producing garbage (regularization "none").
4. Stored diagonalization data: paraunitarity and T^dagger H T = diag(E).
5. Finite torus: bands equal the stage-3 one-magnon ED energies.
6. Header provenance and JSON output.
"""

import numpy as np
import pytest

from model import nbcp
from model.nbcp.model import legacy_cells
from spintoolkit.methods.ed import EDSector, solve_ed
from spintoolkit.methods.lswt import LSWTError, LSWTSettings, LSWTSolver, solve_lswt
from spintoolkit.methods.result import load_json, save_json, state_fingerprint
from spintoolkit.models import neel_state, polarized_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.brillouin_zone import BrillouinZone
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.conversion import to_spin_system
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import Site, SpinModel

BZ = {"one_msl": "Hex_60", "two_msl": "Tetra", "three_msl": "Hex_30", "four_msl": "Hex_60"}
FIELD = np.array([0.03, -0.04, 0.2])


def with_spin(model, S):
    return SpinModel(model.lattice, [Site(s.id, s.position, S) for s in model.sites],
                     model.terms, {"model_id": f"spin_{S}"})


def square_reference(n=1000):
    """Analytic square-lattice LSWT integrals (J = 1) by midpoint quadrature.

    The moment integrand diverges like 1/|k| at the zone centre and corner, so
    the midpoint rule has an O(1/n) error; it is removed by extrapolating from
    n and 2n (the smooth energy integrand needs no extrapolation).
    """
    def integrals(m):
        k = (np.arange(m) + 0.5) * 2 * np.pi / m
        gamma = 0.5 * (np.cos(k)[:, None] + np.cos(k)[None, :])
        root = np.sqrt(1 - gamma ** 2)
        return np.mean(root), np.mean(0.5 / root) - 0.5
    (_, coarse), (omega, fine) = integrals(n), integrals(2 * n)
    return omega, 2 * fine - coarse


@pytest.mark.filterwarnings("ignore:reference state is not stationary")
@pytest.mark.parametrize("cell", list(BZ))
@pytest.mark.parametrize("family", [{}, {"JPD": 0.013, "JGamma": -0.021}])
def test_reproduces_existing_lswt_solver(cell, family):
    config = {"Jxy": 0.076, "Jz": 0.125, **family}
    n = len(legacy_cells(cell))
    angles = np.column_stack([np.linspace(0.3, 2.6, n), np.linspace(-2.0, 2.5, n)]).ravel()
    model = nbcp.build_model(config)
    state = nbcp.candidate_state(model, cell, angles)
    conditions = ExternalConditions(field=FIELD)
    system = to_spin_system(model, state, conditions)
    legacy = LSWTSolver(system, bz_type=BZ[cell]).solve(N=4, regularization="MAGSWT")
    bz = BrillouinZone(system.to_legacy_dict(BZ[cell])["Lattice/BZ setting"], bz_type=BZ[cell])
    _, k_points, _ = bz.get_full(4)
    new = solve_lswt(model, state, conditions,
                     settings=LSWTSettings(k_points=k_points, regularization="MAGSWT"))
    assert new.ground_state_energy == pytest.approx(legacy.ground_state_energy, abs=1e-13)
    keys = sorted(legacy.data["k_data"])
    by_key = {tuple(map(float, k)): e for k, e in zip(k_points, new.eigenvalues[:, :new.num_sites])}
    np.testing.assert_allclose(np.array([by_key[key] for key in keys]), legacy.eigenvalues,
                               rtol=0, atol=1e-12)
    # The boson numbers are ill-conditioned here: these test states are not classical
    # minima, MAGSWT shifts H(k) by up to ~0.3 and leaves magnon energies ~4e-5, so a
    # perturbation of H(k) at machine epsilon moves <n> by up to ~7e-10 (measured,
    # 2026-10-01). The native H(k) (D36) agrees with the old builder to ~2e-16 but
    # sums in a different order, so the bound reflects that conditioning.
    np.testing.assert_allclose(new.boson_numbers, list(legacy.data["boson_numbers"].values()),
                               rtol=0, atol=1e-8)


def test_square_neel_matches_analytic_lswt():
    model = square_heisenberg(J=1.0)
    omega, delta_s = square_reference()
    results = {n: solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(n, n)))
               for n in (32, 64)}
    assert results[64].ground_state_energy == pytest.approx(-1.5 + omega, abs=2e-6)
    assert results[64].ground_state_energy == pytest.approx(-0.6579, abs=1e-4)   # Manousakis 1991
    extrapolated = 2 * results[64].boson_numbers - results[32].boson_numbers     # dS ~ 1/N
    np.testing.assert_allclose(extrapolated, delta_s, atol=3e-4)
    assert delta_s == pytest.approx(0.1966, abs=1e-4)


@pytest.mark.parametrize("S", [0.5, 1.0])
def test_triangular_120_matches_analytic_lswt(S):
    model = with_spin(triangular_heisenberg(J=1.0), S)
    results = {n: solve_lswt(model, state_120(model), settings=LSWTSettings(mesh=(n, n)))
               for n in (32, 64)}
    assert results[64].ground_state_energy == pytest.approx(
        -1.5 * S ** 2 * (1 + 0.436824 / (2 * S)), abs=2e-6)
    extrapolated = 2 * results[64].boson_numbers - results[32].boson_numbers
    np.testing.assert_allclose(extrapolated, 0.2613032, atol=3e-4)


def test_zone_centre_zero_mode_is_reported():
    model = square_heisenberg(J=1.0)
    with pytest.raises(LSWTError, match="zero or negative mode"):
        solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(4, 4), shift=False))
    result = solve_lswt(model, neel_state(model),
                        settings=LSWTSettings(mesh=(4, 4), shift=False, regularization="MAGSWT"))
    assert result.header.diagnostics["max_regularization_shift"] >= 1e-9
    assert result.header.settings["regularization"] == "MAGSWT"


def test_stored_eigenvectors_are_paraunitary_and_diagonalize_h():
    model = triangular_heisenberg(J=1.0)
    result = solve_lswt(model, state_120(model), settings=LSWTSettings(mesh=(6, 6)))
    n = result.num_sites
    eta = np.diag(np.r_[np.ones(n), -np.ones(n)])
    for H, E, T in zip(result.hamiltonians, result.eigenvalues, result.eigenvectors):
        np.testing.assert_allclose(T.conj().T @ eta @ T, eta, atol=1e-10)
        np.testing.assert_allclose(T.conj().T @ H @ T, np.diag(np.abs(E)), atol=1e-10)


def test_torus_bands_equal_one_magnon_ed():
    model = square_heisenberg(J=1.0)
    geometry = CalculationGeometry.finite_torus([[4, 0], [0, 4]])
    conditions = ExternalConditions(field=(0, 0, 5.0))
    lswt = solve_lswt(model, polarized_state(model), conditions, geometry)
    ed = solve_ed(model, geometry, conditions, EDSector(axis=(0, 0, 1), magnon_number=1))
    assert len(lswt.k_points) == 16
    np.testing.assert_allclose(np.sort(lswt.bands().ravel()), ed.excitations(ed.blocks[0]),
                               rtol=0, atol=1e-12)
    neel = solve_lswt(model, neel_state(model), None, geometry, LSWTSettings(regularization="MAGSWT"))
    assert len(neel.k_points) == 8          # 16 cells / 2 cells per magnetic cell


def test_torus_that_folds_a_bond_is_rejected():
    model = square_heisenberg(J=1.0)
    with pytest.raises(ValueError, match="self-interaction"):
        solve_lswt(model, polarized_state(model), ExternalConditions(field=(0, 0, 5.0)),
                   CalculationGeometry.finite_torus([[1, 0], [0, 3]]))


def test_non_stationary_reference_warns():
    model = square_heisenberg(J=1.0)
    tilted = SpinState.from_function(model, np.eye(2, dtype=int),
                                     lambda s, c: np.array([0.6, 0.0, 0.8]))
    with pytest.warns(UserWarning, match="not stationary"):
        result = solve_lswt(model, tilted, ExternalConditions(field=(0, 0, 5.0)),
                            settings=LSWTSettings(mesh=(4, 4)))
    assert not result.header.diagnostics["stationary"]
    assert result.linear_terms > 0.1


def test_header_and_json(tmp_path):
    model = triangular_heisenberg(J=1.0)
    state = state_120(model)
    result = solve_lswt(model, state, settings=LSWTSettings(mesh=(4, 4)))
    header = result.header
    assert header.method == "lswt" and header.model_ref == model.fingerprint()
    assert header.state_ref == state_fingerprint(state) != state_fingerprint(neel_state(square_heisenberg()))
    assert header.geometry["kind"] == "thermodynamic_limit"
    assert header.diagnostics["stationary"] and header.code_version["spintoolkit"]
    summary = load_json(save_json(result, tmp_path / "lswt.json"))
    assert summary["lswt"]["ground_state_energy"] == pytest.approx(result.ground_state_energy)
    assert "eigenvectors" not in summary["lswt"]
    full = load_json(save_json(result, tmp_path / "full.json", include_arrays=True))
    np.testing.assert_allclose(np.array(full["lswt"]["eigenvectors"]["real"]),
                               result.eigenvectors.real)
    ed = solve_ed(model, CalculationGeometry.finite_torus([[3, 0], [0, 3]]),
                  sector=EDSector(axis=(0, 0, 1), magnon_number=1))
    ed_json = load_json(save_json(ed, tmp_path / "ed.json"))
    assert ed_json["header"]["method"] == "ed" and ed_json["header"]["state_ref"] is None
