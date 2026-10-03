# Y-state vortex pairs (classical), 2026-10-02

Script: examples/nbcp_y_vortex_pair.py (PR #30). Record: data-space/verification/261002-y-vortex-pair/vortex-pair-check.json.
Model: Y at 0.2 T, J=0.075, Jz=0.125 meV, S=1/2, J_Gamma=0. Tori L=96, 144; cores pinned (12 spins) to relaxed isolated core.

| quantity | J_PD=0 | J_PD=0.010 soft | J_PD=0.010 stiff |
|---|---|---|---|
| kappa_pair (meV), torus / continuum | 0.01209 / 0.01201 | 0.00679 / 0.00682 | 0.00825 / 0.00842 |
| 2 mu (meV) | 0.0162 | 0.0158 | 0.018-0.019 |
| d kappa / dT (harmonic, sixfold removed) | -1.37 (twist -1.38) | -0.21 to -0.26 | -0.95 to -1.08 |
| k_B T* = kappa/(2 - dkappa/dT) (meV) | 0.0036 (bare 0.0060) | 0.0030-0.0031 | 0.0027-0.0028 |

- No Peierls-Nabarro barrier: core-position dependence ~R^-2, remainder <= 3.4e-6 meV.
- Held-core entropy (J_PD=0): -0.38 (12 spins) vs +2.48 (one spin) -> core fugacity convention-dependent by ~17x.
- At J_PD != 0 the log-det has a thermal sixfold term b6 cos6phi per spin, b6 = -9.35e-5; for the pair's dipolar far field it grows like d^2 ln(L/d) and biased the raw rates (soft raw +0.17 was an artefact). Removed via local sum; per-L rates then agree.
- Classical MC T_BKT ~ 0.0024 meV is 10-25% below the harmonic edges; MC estimator omits J_PD. Consistency check, not derivation. The earlier "large fugacity" reading is withdrawn.
- Not done: quantum S=1/2 cores, wall-vortex composites.
