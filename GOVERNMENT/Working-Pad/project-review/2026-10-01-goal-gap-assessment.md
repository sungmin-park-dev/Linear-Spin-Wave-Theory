# Status against the two project goals (2026-10-01)

Basis: the repository on `main` after PR #21 (f49d6a0), checked directly. Tests: 589 pass, line coverage 91 %. The wheel builds (`pip wheel .`, 60 modules).

Short answer: **neither goal is complete.** The computational core is in place and verified against exact results. What is missing is three things:
- the comparison with experimental intensities;
- a closed phase competition, including the skyrmion phase;
- release preparation (single API, English docs, license, version).

## Goal 1. Checking and reproducing the NBCP paper

| Area | Status | Basis |
|---|---|---|
| Model, classical states, LSWT bands | Done | Polarized phase in transverse field matches arXiv:2505.06398 Eq. (5), the 0.46 meV gap and B_C^cl = 1.72 T; three-sublattice bands at 0.7–1.4 T (`data-space/verification/261001-*`) |
| Y/V pseudo-Goldstone pinning (zero-point) | Done | N = 48 scans, state selection D17/D28, C2′T reflection (PR #18) |
| Finite-T angular free energy, classical thermal order | Tools done (D38–D40) | Y scan: density order T_d ≈ 0.015 meV, stiffness T_BKT ≈ 0.0024 meV, no classical lock-in |
| **Comparison with measured intensities** | **Implemented 2026-10-01 (D41), PR pending** — was: Missing | Bands only. No intensity comparable with neutron data (form factor, g-tensor, out-of-plane momentum, resolution, domain average) |
| **Skyrmion phase (SkX)** | **Missing** | Note ch. 5: "candidate, not established". No signed topological charge, stationary texture, energy comparison or LSWT stability |
| **Global phase competition** | **Incomplete** | Note ch. 3 and 10: only an incomplete candidate set is compared. No unbiased search over larger supercells |
| Quantum/thermal matching, vortex cores | Open (NBCP note thread) | Note ch. 7 and 8, TASK-QUEUE #4 |
| Thermal Hall J_PD·J_Γ magnitudes | Open (TASK-QUEUE thread) | TASK-QUEUE #3 |

## Goal 2. Releasing the LSWT package

### Features (2D general LSWT, compared with SpinW and Sunny)

- Present:
  - arbitrary 2D lattice and bilinear exchange, g-tensor, onsite terms (S ≥ 1);
  - crystal symmetry and allowed exchange;
  - classical search and refinement, LT diagnostic;
  - commensurate supercell LSWT and single-Q spiral (rotating frame);
  - Colpa diagonalization, thermodynamics;
  - S(q, ω) with the in-plane polarization factor;
  - Berry curvature, Chern numbers, thermal Hall;
  - ED, TeNPy and NetKet connections;
  - classical dynamics and Monte Carlo.
- Missing and needed for release: **neutron intensity I(Q, ω)**:
  - magnetic form factor;
  - moment g·S;
  - out-of-plane Q_z polarization;
  - energy resolution, powder average and domain average.
- Missing, can come after release:
  - 1/S corrections and the two-magnon continuum;
  - SU(N) LSWT;
  - non-U(1) and multi-Q incommensurate states;
  - long-range dipolar interactions;
  - CIF input.

### Release readiness

| Item | Status | Needed |
|---|---|---|
| License | **No LICENSE file** (pyproject declares MIT) | Add the file |
| Version | pyproject `0.2.dev0` and `__init__` `0.2.0-dev` differ; the latter is not PEP 440 | Single source |
| Public API | **Two parallel paths**: the new `SpinModel`/`solve_lswt` and the deprecated `SpinSystem`/`LSWTSolver`, legacy dict input and the `lswt` alias package. The README and top-level quick start still use the old path | Fix the public API on the new path; keep the old one with a deprecation warning for 0.x and remove it in 1.0 (interface decision, needs your confirmation) |
| User documentation | README is Korean and out of date (says observables are not connected, and that LSWT takes SpinSystem). No English API reference or tutorials | English README, quick start, API docs, 3–4 tutorials |
| Examples | 50 scripts, mostly verification. NBCP examples depend on the repository's `model/nbcp`, which is not in the package | Ship the NBCP model in the package or document examples as repository-only; separate user examples |
| Dependencies | matplotlib and tqdm are required | Make plotting an optional extra |
| Tests and CI | 91 %; low in the old path (energy 56 %, brillouin_zone 59 %, spin_system 62 %, exchange 43 %). CI on Py 3.9 and 3.12, Linux | Add 3.10 and 3.11; cross-checks against SpinW or Sunny on standard models |
| Release process | No CHANGELOG, release workflow or tag | CHANGELOG, PyPI publishing workflow |
| Physics review | D35–D40, 10 docs/lswt drafts, spiral memo, NBCP note | Your review all at once after implementation; a release gate |

## Recommended order

1. **Neutron intensity I(Q, ω).** Needed for both goals: the NBCP intensity comparison (Woodland 2025) and a release requirement.
2. **Topological charge and phase competition.** Signed lattice skyrmion number, a stationary SkX texture, then supercell Monte Carlo annealing plus LSWT zero-point comparison over the candidate set. Closes NBCP ch. 3 and 5 on the code side; the note itself belongs to the NBCP thread.
3. **API consolidation and basic release items.** Public API on the new path with deprecations, LICENSE, a single version, optional plotting dependency. The API decision needs your confirmation; I would bring a proposal.
4. **English documentation, tutorials, NBCP model in the package, SpinW/Sunny cross-checks.**
5. **Your review of all drafts and decisions**, then fixes, then the 0.3 (or 1.0) PyPI release.

After release: 1/S corrections, SU(N), multi-Q.
