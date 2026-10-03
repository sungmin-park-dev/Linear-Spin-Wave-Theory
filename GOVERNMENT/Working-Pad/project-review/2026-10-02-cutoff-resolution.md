# Finite-T matching cutoff: resolution (2026-10-02)

Question (NBCP note ch. 7): the harmonic thermal sixfold pinning of Y depends on the momentum cutoff Λ below which soft modes are left to the phase theory (Y, J_PD = 0.010 meV, 0.2 T: cutoff 0 lowers a6, cutoff 1.0 raises it).

Check (hydro_cutoff_check.py, uses examples/nbcp_y_soc_conditions.py relaxed reduction):
- Along the exact global-z orbit the relaxed classical susceptibility χ is angle independent (span 9e-16), while the relaxed stiffness tensor ρ(φ) varies: √det ρ = 0.00618 meV with a 2.0% cos6φ component.
- Phase-theory prediction for the soft-branch sixfold anisotropy, b6 = cos6 coefficient of ⟨½ ln(k̂·ρ(φ)·k̂/χ)⟩_k̂ = 0.005034. LSWT shells: 0.00489 (|k|<0.15), 0.00451 (0.15–0.25); k² extrapolation to k→0 gives 0.00504 (agreement 0.1%).
- Removing modes below Λ changes a6 by Δa6(Λ). Phase theory with ε = k√(k̂ρ(φ)k̂/χ) and Bose free energy reproduces the LSWT Δa6 to 6–7% at Λ = 0.25 for T = 0.0025–0.02 meV, but is 25–30% off at Λ = 0.5 (outside the hydrodynamic regime).

| T (meV) | Δa6(0.25) phase theory | Δa6(0.25) LSWT | Δa6(0.5) phase theory | Δa6(0.5) LSWT | a6(T)/a6(0), cutoff 0 |
|---|---|---|---|---|---|
| 0.0025 | 2.03e-8 | 1.91e-8 | 2.78e-8 | 2.44e-8 | 0.983 |
| 0.005 | 7.04e-8 | 6.57e-8 | 1.63e-7 | 1.30e-7 | 0.899 |
| 0.010 | 1.78e-7 | 1.66e-7 | 5.62e-7 | 4.30e-7 | 0.671 |
| 0.015 | 2.86e-7 | 2.66e-7 | 9.89e-7 | 7.48e-7 | 0.616 |
| 0.020 | 3.95e-7 | 3.67e-7 | 1.42e-6 | 1.07e-6 | 0.698 |

Reading
1. The cutoff dependence is not an ambiguity. The pinning removed from the matching by a cutoff Λ ≲ 0.25 is generated back, at the same order, by the angle dependence of the stiffness when the phase theory integrates its own modes. A matching that uses a cutoff must keep ρ(φ) in the phase theory; one that uses a constant stiffness must use cutoff 0. Both give the same long-wavelength pinning at O(T).
2. Cutoff 1.0 drops modes outside the regime a phase theory can represent, without a counterpart, so its sign change is an artefact of an inconsistent split.
3. Consistent harmonic answer for Y: thermal soft modes weaken the sixfold pinning, a6(T)/a6(0) = 0.90 (0.005 meV), 0.67 (0.010), 0.62 (0.015). This is the order-by-disorder entropy of the anisotropic stiffness opposing the zero-point selection.
4. Not included: the Debye–Waller suppression from phase fluctuations themselves (the RG eigenvalue 2 − 9T/πJ̃; relative order T/S beyond this), anharmonic terms, vortices. V not checked this way (the relaxed reduction is Y only); its LSWT shell b6 varies strongly with k (0.016, 0.013, 0.008), so its hydrodynamic regime is narrower.
5. For the paper: Eq. (3) uses the T = 0 g6 and a constant J̃. At the bare window temperatures (≲ 0.006 meV) the Y pinning is 10–15% weaker; this shifts the lock-in crossover, not the relevance of g6.
