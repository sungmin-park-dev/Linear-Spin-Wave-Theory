# 1/S² for the pseudo-Goldstone gap: feasibility (2026-10-02)

Full 1/S²: Δ² = C_φ/χ needs C_φ at two loops (quartic and cubic spin-wave vertices, Hartree–Fock plus second-order self-energy) and χ at one loop. The package has no interacting spin-wave (nonlinear) solver; building one is a solver feature, not a small-cell calculation. Exact diagonalization cannot resolve λ6 ~ 1e-6 meV/spin against finite-size tower gaps ~ 1e-2 meV. Verdict: not tractable in this thread without that feature.

Size indicator (one_loop_susceptibility.py; E_zp(h) on the classical background, mesh 24, Δh = 2e-3 meV, J_PD = 0.010 meV):

| State | χ classical (meV⁻¹/spin) | one-loop correction | ratio | m classical | one-loop m |
|---|---|---|---|---|---|
| Y, 0.2 T | 1.111 | +0.408 | 0.37 | 0.101 | −0.038 |
| V, 1.4 T | 1.551 | +0.029 | 0.02 | 0.363 | −0.040 |

Reading: for Y the next order changes χ by ~37%, which alone would lower the gap by ~15% (Δ ∝ χ^{−1/2}); the two-loop C_φ is unknown and could compensate or add. For V the 1/S series for χ is well behaved. The Y gap should therefore be quoted with an O(20–40%) next-order uncertainty; the V gap is better controlled. This is a partial piece of the next order and is not a 1/S² result.
