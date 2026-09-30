"""NBCP LSWT Hamiltonian verification (Step 3).

Builds the LSWT Hamiltonian for the Four MSL ground state and checks:
1. Hermiticity: H†(k) = H(k)
2. Bosonic commutation constraint: B(-k) = B(k)^T
3. Eigenvalue physicality (real, positive after regularization)
4. Sensitivity of the matrix to a B/B† block swap

Coefficient correctness and the actual archived implementation are tested in
code-space/tests/test_solvers/test_hamiltonian_pairing.py. Structural constraints
alone do not distinguish the legacy swap from the correct construction.
"""

import numpy as np
from examples.nbcp_ground_state import (
    NBCP_CONFIG, PHASES,
    make_nn_exchange_matrices, make_nnn_exchange_matrices,
    four_msl, find_ground_state,
)
from spintoolkit.methods.lswt.hamiltonian import LSWTHamiltonian
from spintoolkit.system.brillouin_zone import BrillouinZone


# ======================================================================
# 1. Get Four MSL ground state
# ======================================================================

print("=" * 60)
print("Step 1: Finding NBCP Four MSL ground state")
print("=" * 60)

angles_setting = {
    "One MSL":   (None, 0),
    "Two MSL":   (None, None, None, None),
    "Three MSL": (None, 0, None, 0, None, 0),
    "Four MSL":  (None, None, None, None, None, None, None, None),
}

opt_result, cls_result, all_results = find_ground_state(
    NBCP_CONFIG, opt_method="classical", N=20,
    angles_setting=angles_setting, verbose=True,
)

# Use the optimized (classical) result
gs = opt_result
print(f"\nUsing: {gs['phase_name']}")
print(f"  angles = {np.round(gs['angles'], 6)}")


# ======================================================================
# 2. Build LSWT Hamiltonian
# ======================================================================

print("\n" + "=" * 60)
print("Step 2: Building LSWT Hamiltonian")
print("=" * 60)

system = gs["system"]
bz_type = gs["bz_type"]
legacy_data = system.to_legacy_dict(bz_type)

spin_info = legacy_data["Spin info"]
couplings = legacy_data["Couplings"]
lattice_bz = legacy_data["Lattice/BZ setting"]

ham = LSWTHamiltonian(spin_info, couplings)
Ns = ham.Ns

# Build BZ and get k-points
bz = BrillouinZone(lattice_bz, bz_type=bz_type)
bz_data, full_k_points, _ = bz.get_full(N=20)
n_kpts = len(full_k_points)
print(f"  Ns = {Ns}, k-points = {n_kpts}, bz_type = {bz_type}")

# Build Hamiltonian
H_k, linear_terms = ham.Quadratic_Bose_Hamiltonian(full_k_points)
print(f"  H(k) shape: {H_k.shape}")


# ======================================================================
# 3. Check Hermiticity
# ======================================================================

print("\n" + "=" * 60)
print("Step 3: Hermiticity check  H†(k) = H(k)")
print("=" * 60)

max_herm_err = 0.0
for idx in range(n_kpts):
    diff = H_k[idx] - H_k[idx].conj().T
    err = np.max(np.abs(diff))
    max_herm_err = max(max_herm_err, err)

print(f"  Max Hermiticity error: {max_herm_err:.2e}")
assert max_herm_err < 1e-12, f"Hermiticity violated! err = {max_herm_err}"
print("  PASSED")


# ======================================================================
# 4. Check B(-k) = B(k)^T  (bosonic commutation constraint)
# ======================================================================

print("\n" + "=" * 60)
print("Step 4: Bosonic constraint  B(-k) = B(k)^T")
print("=" * 60)

# Build H(-k)
H_mk, _ = ham.Quadratic_Bose_Hamiltonian(-full_k_points)

max_B_err = 0.0
for idx in range(n_kpts):
    B_k = H_k[idx, :Ns, Ns:]      # upper-right block
    B_mk = H_mk[idx, :Ns, Ns:]    # upper-right block of H(-k)
    err = np.max(np.abs(B_mk - B_k.T))
    max_B_err = max(max_B_err, err)

print(f"  Max B(-k) vs B(k)^T error: {max_B_err:.2e}")
assert max_B_err < 1e-12, f"Bosonic constraint violated! err = {max_B_err}"
print("  PASSED")


# ======================================================================
# 5. Check A†(k) = A(k)  (A block Hermiticity)
# ======================================================================

print("\n" + "=" * 60)
print("Step 5: A block Hermiticity  A†(k) = A(k)")
print("=" * 60)

max_A_err = 0.0
for idx in range(n_kpts):
    A_k = H_k[idx, :Ns, :Ns]
    err = np.max(np.abs(A_k - A_k.conj().T))
    max_A_err = max(max_A_err, err)

print(f"  Max A block Hermiticity error: {max_A_err:.2e}")
assert max_A_err < 1e-12, f"A block not Hermitian! err = {max_A_err}"
print("  PASSED")


# ======================================================================
# 6. Check whether exchanging B and B† changes the matrix
# ======================================================================

print("\n" + "=" * 60)
print("Step 6: B block structure (Γ = {:.3f})".format(NBCP_CONFIG["JGamma"]))
print("=" * 60)

# A complex B can still be Hermitian. Inspect the actual block-swap difference.
test_idx = n_kpts // 3  # pick a generic k-point
B_test = H_k[test_idx, :Ns, Ns:]
max_imag = np.max(np.abs(np.imag(B_test)))
is_symmetric = np.allclose(B_test, B_test.T, atol=1e-12)
swap_difference = np.max(np.abs(
    H_k[:, :Ns, Ns:] - H_k[:, :Ns, Ns:].conj().transpose(0, 2, 1)
))

print(f"  k-point: {full_k_points[test_idx]}")
print(f"  Max |Im(B)|: {max_imag:.6e}")
print(f"  B == B^T (symmetric): {is_symmetric}")
print(f"  Max |B - B†| over sampled k-points: {swap_difference:.6e}")
if swap_difference > 1e-10:
    print("  → Exchanging B and B† changes H; compare spectra separately.")
else:
    print("  → B = B† within tolerance on this grid; the swap is invisible here.")
print("  Coefficient verification: code-space/tests/test_solvers/test_hamiltonian_pairing.py")


# ======================================================================
# 7. Linear terms check (should vanish at equilibrium)
# ======================================================================

print("\n" + "=" * 60)
print("Step 7: Linear terms (should vanish at equilibrium)")
print("=" * 60)

max_linear = 0.0
for sl_name, val in linear_terms.items():
    mag = np.abs(val)
    max_linear = max(max_linear, mag)
    print(f"  {sl_name}: |linear| = {mag:.6e}")

print(f"  Max |linear term|: {max_linear:.6e}")
if max_linear < 1e-4:
    print("  → Near equilibrium (linear terms small)")
else:
    print("  → WARNING: Large linear terms — may not be at equilibrium")


# ======================================================================
# 8. Diagonalize and check eigenvalues
# ======================================================================

print("\n" + "=" * 60)
print("Step 8: Colpa diagonalization & eigenvalue check")
print("=" * 60)

k_data, mu_magswt = ham.solve_k_Hamiltonian(
    full_k_points, Berry_curvature=True, regularization="MAGSWT"
)

print(f"  MAGSWT chemical potential (μ): {mu_magswt:.6e}")

# Collect eigenvalues
all_evals = []
n_colpa_ok = 0
for k_key in sorted(k_data.keys()):
    _, eigen_data, colpa_data = k_data[k_key]
    evals, evec = eigen_data
    colpa_ok = colpa_data[0]
    if colpa_ok:
        n_colpa_ok += 1
    all_evals.append(evals[:Ns])

all_evals = np.array(all_evals)
print(f"  Colpa succeeded: {n_colpa_ok}/{n_kpts}")
print(f"  Eigenvalue shape: {all_evals.shape}")
print(f"  Min eigenvalue: {all_evals.min():.6f}")
print(f"  Max eigenvalue: {all_evals.max():.6f}")
print(f"  Mean eigenvalue: {all_evals.mean():.6f}")

# Check all positive
n_negative = np.sum(all_evals < -1e-10)
print(f"  Negative eigenvalues (< -1e-10): {n_negative}")
if n_negative == 0:
    print("  PASSED: All eigenvalues non-negative")
else:
    print("  WARNING: Negative eigenvalues found")

# Band summary
print("\n  Band summary (min / mean / max):")
for band in range(Ns):
    band_evals = all_evals[:, band]
    print(f"    Band {band}: {band_evals.min():.6f} / "
          f"{band_evals.mean():.6f} / {band_evals.max():.6f}")


# ======================================================================
# 9. Summary
# ======================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Phase: {gs['phase_name']}")
print(f"  E_cl  = {gs['E_cl']:.6f}")
print(f"  E_qm  = {gs['E_qm']:.6f}")
print(f"  E_tot = {gs['energy']:.6f}")
print(f"  MAGSWT μ = {mu_magswt:.6e}")
print(f"  Hermiticity: PASSED (err = {max_herm_err:.2e})")
print(f"  B(-k) = B(k)^T: PASSED (err = {max_B_err:.2e})")
print(f"  A Hermitian: PASSED (err = {max_A_err:.2e})")
print(f"  Colpa: {n_colpa_ok}/{n_kpts}")
print(f"  Eigenvalue range: [{all_evals.min():.6f}, {all_evals.max():.6f}]")
print("=" * 60)
