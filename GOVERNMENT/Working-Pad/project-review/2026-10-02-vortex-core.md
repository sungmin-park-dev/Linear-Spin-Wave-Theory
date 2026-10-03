# Classical vortex cores of the NBCP Y supersolid (2026-10-02)

Model: J = 0.075, J_z = 0.125 meV, S = 1/2, 0.2 T (h = 0.0538 meV), J_Γ = 0. Hexagonal clusters R = 8–48 (up to 6 937 spins); two outer rings fixed to n_i = R_z(m arg(r_i − r0) + φ0) n_Y; every other spin relaxed on the sphere (L-BFGS, torque ≤ 1e-8 meV). Energy measured against the uniform state with the same fixed rings. Scripts: vortex.py, scan.py, single.py.

| J_PD (meV) | m | log coefficient (fit R = 24–48) | π√det ρ (relaxed reduction) | E_core (meV) |
|---|---|---|---|---|
| 0 | ±1 (equal to 1e-13) | 0.01189 | 0.01201 | 0.0071–0.0076 |
| 0.010 | −1 | 0.00719 | 0.00763 | 0.0036–0.0053 |
| 0.010 | +1 (average over φ0) | 0.00622 | 0.00763 | 0.009–0.015 (convention dependent) |

E_core range: fitted constant vs. constant with the predicted log coefficient. Core position (triangle centres, any sublattice site) changes nothing beyond 1e-5 meV: the core relaxes to the same place.

Findings
1. The vortex energy grows with the relaxed stiffness π√det ρ (1% at J_PD = 0, 6% for m = −1 at 0.010). This independently confirms that J_PD = 0.010 meV lowers √det ρ by 36% (0.00382 → 0.00243 meV): the bare upper window edge πρ/2 drops from 0.0060 to 0.0038 meV.
2. With J_PD the two windings are inequivalent (only the combined spin+lattice rotation survives; the spin-only mirror that maps m → −m at J_PD = 0 is broken). The m = +1 energy carries a cos 2φ0 term that grows ∝ R: a total-derivative (boundary) term f(φ)·∇φ that vanishes for uniform twists but not for a co-rotating winding. In a neutral pair it contributes only around the cores, so its split between "core" and "boundary" is a convention; only pair energies are convention free. Unconstrained pairs annihilate on relaxation and pinned pairs are dominated by the pinning, so a clean pair value is not available yet.
3. Core energies are comparable to the window temperatures: E_core/T ≈ 1–2 for m = −1 at T ≈ 0.0024–0.004 meV, so y_v = e^{−E_core/T} ≈ 0.1–0.3. The vortex fugacity is not small; the weak-coupling clock RG with bare K overestimates T_BKT, consistent with the classical MC T_BKT ≈ 0.0024 meV lying below even the J_PD-reduced bare edge 0.0038 meV (ratio 1.6; lattice XY ratio is 1.76).
4. Classical cores only; quantum S = 1/2 cores and the T dependence of E_core are not computed.
