# Park et al. 2026 (arXiv:2601.20963v1): gaps and theoretical problems

Paper: S. Park, S.-M. Park, Y.-T. Oh, H.-Y. Lee, E.-G. Moon, "Spin-orbit-induced instability and finite-temperature stabilization of a triangular-lattice supersolid", arXiv:2601.20963v1 (cited as `park2026` in `docs/nbcp/references.bib`).

Date: 2026-10-01. Compared against `docs/nbcp/main.tex` on main (through PR #18) and today's results from other threads.

**How the paper was read.** This environment's proxy blocks direct downloads from arxiv.org. I read the HTML version through a summarizing web fetch, and asked it to quote the section headings, Eqs. (1)–(4), the RG flow, the Δ_PG formula and the captions of Figs. 3–4. The quotes below come from that fetch. I could not see the figures themselves, the supplementary material, or any text that is not quoted here. Please check the items marked "check" against your own copy.

Paper structure, as quoted: I Introduction; II Results (Model and symmetries; Ground state phase diagram; Instability of the spin-supersolid phases; Finite-temperature stabilization of the spin-supersolid phases); III Discussion.

## Skyrmion competition (relayed 2026-10-01 07:49Z from the roadmap thread; not yet on main, not re-checked here)
- At the Fig. 4 value J_PD = 0.010 meV the 4-site SkX never wins in cells up to 2x2; closest approach 0.92 μeV/spin above competitors.
- It wins only for J_PD ≥ 0.02 and J_Γ ≥ 0.01 meV with h ≤ 0.1 meV, where every 1-, 2- and 3-site state is LSWT-unstable.
- Signed charge Q = −2 per 2x2 cell for field along +c; the manuscript's absolute value drops the sign.
- Candidates only (limited cells). The paper's SkX parameters are not in the repo. Source: data-space/verification/261001-nbcp-skyrmion-competition/README.md on the roadmap branch. Fold into docs/nbcp §5 after it reaches main.

## Fig. 4(b) provenance check (2026-10-01 06:45Z)
- Repo cannot establish it. Git history starts 2026-09-29 (eight months after arXiv v1); no figure script, no Δ_PG-vs-J data, no plotting code in legacy/.
- legacy/scripts/4_Pseudo_Gap.py computes one point per run, writes nothing, and its __main__ uses Jxy=0.076, h=0.05 meV, J_Γ=0.02 — not the Fig. 4 parameters. A scan wrapper would be needed and is not archived.
- The paper prints (1/S)√det; the script multiplies by S. The two differ by 4×, so formula and script cannot both describe the figure.
- arXiv PDF/source downloads are blocked here, and the figure tick labels are not in the PDF text layer.
- Fingerprints at 0.010 meV (meV): correct Y/PD 0.0069, V/PD 0.0156, Y/Γ 7.3e-6, V/Γ 0.0020; legacy as archived 0.0024, 0.0047, 2.6e-7, 0.00020; legacy with 1/S 0.0098, 0.0189, 1.0e-6, 0.0008. The Γ axis separates them best (Y/Γ differs 28×).

## Fifth pass (2026-10-01): validity of our own gap relation (PR #22)
- Rau Eq. (7) implemented literally (linear local coordinates) for the global-z mode of Y/V gives a zero-point curvature 13–35 times the exact-orbit C_φ; the excess is the dropped tadpole (Lin–Shi), confirmed numerically (line + tadpole = orbit to 0.1–0.4%). The legacy gap code rotates the azimuth exactly, so its φφ curvature is orbit-type (inferred from code); its errors are the θ partner and ×S.
- Our Δ² = C_φ/χ_z: six-mode uniform dynamics agrees to 0.1%; Γ frequencies reproduced.
- Stability: orbit fails first at angular maxima, Y J_PD = 0.014, V 0.0115 meV. Fig. 4(b) values near those couplings are not reliable at leading order.
- V r = 1: soft mode of a C₂′T-breaking transition (physical at this order).
- Finite T (harmonic F): Y pinning weakens (Δ ratio 0.95 at 0.005 meV, 0.83 at 0.010), V strengthens.
- Not done: full one-loop self-energy (frequency dependence, next order in 1/S).

## Fourth pass (2026-10-01): finite-temperature pseudo-Goldstone literature

Sources (HTML read through the summarizing fetch; quotes as returned):
- X. Lin and T. Shi, "Pseudo-Goldstone Modes at Finite Temperature", arXiv:2505.07229v1. Its abstract names Na₂BaCo(PO₄)₂ and K₂Co(SeO₃)₂.
- S. Khatua, M. J. P. Gingras and J. G. Rau, "Pseudo-Goldstone modes and dynamical gap generation from order-by-thermal-disorder", arXiv:2301.11948.
- Park et al. do not cite either paper in the quoted text. Lin–Shi appeared eight months before Park et al. and names NBCP.

### What the two papers say
- Lin–Shi: "the curvature formula [Rau 2018] has only been justified for systems with collinear magnetic order." They add that the nonlinear dependence of the rotation on φ and θ (their O₂,₀, O₀,₂, O₁,₁ terms) "is entirely neglected" in Rau, so the tadpole self-energy Σ₀⁽²⁾ is omitted.
- Lin–Shi Eq. (9) normalizes the rotation angles by sublattice weights |χ_φ,α|, |χ_θ,α| from the zero-mode eigenvector. The 1/S then follows from the canonical commutator [O₁,₀, O₀,₁] = i, not from equal weights.
- Lin–Shi Eq. (11): at finite T the gap is the same curvature formula with the free energy F(φ,θ) in place of the ground-state energy. Eq. (7): F = E_g + T Σ ln(1 − e^{−S d_k/T}).
- Lin–Shi, triangular XXZ (κ > 1, zero field, accidental degeneracy, type-I mode): Δ(T) ≈ Δ₀ − cT; fit Δ/√S ≈ 0.523 − 0.5803 T/S at κ = 1.2. The cause is entropy from magnon scattering between bands.
- Khatua–Gingras–Rau: for collinear order with a single magnon band, order-by-thermal-disorder generates a gap Δ ∝ √T. Lin–Shi say this scaling does not carry over to noncollinear order.

### Consequences for Park et al.
- **User confirmation (2026-10-01 04:57Z): the error the user found is item 1 below.** The note already cites Lin–Shi and uses the canonically normalized, exact-orbit form; PR #20 (merged) adds the explicit statement to Appendix A.
1. **The gap formula is applied outside its justified range.**
   - Park et al. use Rau Eq. (1) for the noncollinear Y and V states. Lin–Shi state that this formula is justified only for collinear order.
   - The missing ingredients are the ones in our A2 finding. Lin–Shi's ingredients are canonical normalization with sublattice weights and the nonlinear part of the rotation. Ours is the sin ϑ_α Berry weight of a global z rotation.
   - Correction to my earlier wording ("Rau's formula holds for equal weights; 1/S comes from that normalization"): Rau says equal weighting "is not essential". The precise condition is that θ and φ be canonically conjugate, with the 1/S fixed by the commutator. Equal weights in a linear local-frame parametrization are one way to satisfy it for collinear order. For noncollinear order, the rotation must also be followed beyond linear order.
   - Our Δ² = C_φ/χ_z uses the zero-point energy along the exact rotated classical orbit, so the nonlinear part of the rotation is included in C_φ. I infer that it agrees with Lin–Shi's type-I formula Δ = S^{1/2}√(d̃ χ_φ†Σ̄₀χ_φ), but I have not checked this term by term.
2. **The finite-T argument uses T = 0 quantities.**
   - Δ_PG and the pinning g̃6 in Eq. (3) come from the T = 0 ground-state energy ε, but the paper's claim is about finite T.
   - By Lin–Shi Eq. (11), the finite-T pinning is the curvature of F(φ,T). For the triangular XXZ coplanar order, Lin–Shi find it falls linearly with T.
   - Our harmonic f(φ,T) shows the same sign. For Y at J_PD = 0.010 meV, the thermal term lowers the sixfold amplitude a6 from 1.44×10⁻⁶ to 0.96×10⁻⁶ meV per spin at T = 0.01 meV with no cutoff. At low T the harmonic term grows faster than linearly (−2.5×10⁻⁸ at 0.0025, −1.4×10⁻⁷ at 0.005, −4.7×10⁻⁷ at 0.01 meV), so Lin–Shi's linear term must come from interactions beyond harmonic order.
   - Effect on the window: linear RG irrelevance does not depend on the size of g6, so the window edges at leading order are unchanged. What changes is the lock-in crossover and the T = 0 Δ_PG used to describe finite T.
   - The Discussion links the persisting magnetocaloric response to the T = 0 SOC gap being washed out by KT physics. Lin–Shi provide a separate, microscopic channel that closes the gap at finite T, so the Discussion's attribution is incomplete.
3. **Check:** Park et al. do not define θ. If θ is the accidental XXZ direction (Lin–Shi's mode) and φ is the U(1) azimuth, then the two are not a conjugate pair in general, and the printed formula needs that justification as well.

## Third pass (2026-10-01): equation-level check of items 1 and 2 against the original papers

Sources:
- Park et al., arXiv:2601.20963v1 (HTML, quoted).
- Rau, McClarty and Moessner, PRL 121, 237201 (2018), arXiv:1805.00947, Eqs. (1)–(12) quoted from the PDF. This is the paper's Ref. [71].
- `legacy/scripts/4_Pseudo_Gap.py` (read in full).
- `examples/pseudo_goldstone_comparison.py` classical Hessians.

### Item 2: pseudo-Goldstone gap

1. **Rau's derivation.**
   - Rau Eq. (7) parametrizes each spin in its own local frame: S_α = S(φ x̂_α + θ ŷ_α + ẑ_α√(1−φ²−θ²)). It uses the same φ and θ on every sublattice ("the relative weight of the rotations not varying between sublattices").
   - The equations of motion, Rau Eq. (9), are dφ/dt = (1/S)∂ε/∂θ and dθ/dt = −(1/S)∂ε/∂φ, with ε per spin. Rau Eq. (8) defines ε = S²ε_cl + Sε_qu.
   - Eq. (1), Δ = (1/S)√(ε_θθ ε_φφ − ε_θφ²), follows only for this canonical, equal-weight pair.
2. **The paper's printed formula is Rau Eq. (1) verbatim, but the paper's φ is a global rotation U_z(φ)** (Fig. 4 caption). A global z rotation moves spin α in its local frame by sin ϑ_α:
   - Y at 0.2 T: |sin ϑ| = (0.594, 0.594, 0).
   - V at 1.4 T: |sin ϑ| = (0.398, 0.398, 0.940).

   The weights are not equal, so Rau's normalization does not apply. Generalizing Rau Eq. (9) to weights w_α = sin ϑ_α and a polar direction u gives the Lagrangian per spin L = b θ φ̇ − ε, with b = (S/N_s)Σ_α w_α u_α. The frequency is then **Δ² = ε_θθ ε_φφ / b²**.
   - This reduces to Rau Eq. (1) when w = u = 1, so b = S.
   - Minimizing over u (Cauchy–Schwarz) gives Δ² = C_φ/χ_z. This is the formula in the note, App. A.
   - The printed form is not invariant under θ → λθ (Δ ∝ λ), so it is meaningful only with b = S.
3. **Numbers** at J_PD = 0.010 meV, J_Γ = 0, using the current Hamiltonian and the N = 48 curvature C_φ:

   | Reading | Y gap (meV) | Y ratio | V gap (meV) | V ratio |
   |---|---|---|---|---|
   | Canonical √(C_φ/χ_z) | 0.006862 | 1 | 0.015647 | 1 |
   | Printed (1/S)√, θ = uniform signed polar offset (legacy θ) | 0.00208 | 0.30× (b = 0: not a frequency) | 0.01137 | 0.73× |
   | Printed (1/S)√, θ = Rau's uniform local-frame polar rotation | 0.00400 | 0.58× | 0.01415 | 0.90× |
   | Canonical with that unrelaxed u (Cauchy–Schwarz upper bound) | 0.01011 | 1.47× | 0.02445 | 1.56× |
   | Legacy code, S√(…) with uniform signed θ | 0.000520 | 0.076× | 0.002841 | 0.18× |

   The last row reproduces App. A's "Current/old" column exactly, so the reading of the legacy code is confirmed.
4. **Verdict.**
   - (a) Derived: Rau Eq. (1), applied to the paper's own φ (a global z rotation), lacks the Berry weight sin ϑ_α. It is not the "precise expression" of Ref. [71] for this geometry. The paper's deferral to Ref. [71] covers the general method, not this normalization.
   - (b) The legacy code differs from the printed formula by a further factor S² = 1/4: it multiplies by S instead of dividing.
   - (c) Fig. 4(b) is affected numerically only if it came from that code. That is still unconfirmed. If it did, the PD gaps shown are about 0.08–0.18× the canonical value with the current Hamiltonian, before the archived-Hamiltonian difference.

### Item 1: V threefold vs sixfold

1. **Model check.** Paper Eq. (2) H_PD and H_Γ match `bond_angle_exchange` element by element: xx = J + 2J_PD cos φ, yy = J − 2J_PD cos φ, xy = −2J_PD sin φ, xz = −J_Γ sin φ, yz = J_Γ cos φ. The S6 equation matches ch. 2.
2. **R_z(π).**
   - Under R_z(π), [S^x,S^y]_ij and {S^x,S^y}_ij are even and {S^y,S^z}_ij and {S^x,S^z}_ij are odd. So H_PD is invariant and H_Γ → −H_Γ, as derived.
   - **The paper's text scopes its Z3 statement to finite J_Γ**: "for finite J_Γ, quantum zero-point fluctuations lift the U(1) degeneracy" and "the least symmetric term, H_Γ, dictates the symmetry of the total Hamiltonian". So the Z3 claim is not an error. What is missing is an explicit statement that V is Z6 at J_Γ = 0, where the KT window is allowed.
3. **A correction to Fig. 3(c), "M_V ≅ Z3", for small J_Γ/J_PD (derived and numerical).**
   - With v = −λ6 cos 6φ − λ3 sin 3φ, the minima satisfy sin 3φ = r with r = λ3/(4λ6). For 0 < r < 1 there are 6 minima, related by the threefold rotation and C2′T.
   - Counted directly on the raw 72-point LSWT energies (no fit): 6 minima at r = 0.112, 0.224, 0.448, 0.586 and 0.905, and 3 minima at r = 1.17, 2.35 and 4.76 and on the pure-Γ axis.
   - r = 1 occurs at J_Γ/J_PD ≈ 0.09 (J_PD = 0.005 meV) and ≈ 0.22 (J_PD = 0.010 meV).
   - In this regime the ordered V state also breaks C2′T, so its manifold has 6 points, not Z3. The paper's no-KT conclusion is unchanged: for cos 3φ, 9T/(4πJ̃′) < 2 for all T < πJ̃′/2.
4. **Eqs. (3) and (4) and the RG.**
   - The forms −g6 cos 6φ and −g3 sin 3φ agree with the C2′T phase constraint (δ6 ∈ {0, π}, δ3 = ±π/2).
   - dg6/dl = (2 − 9T/(πJ̃)) g6 equals p²/(4πK) with K = J̃/T and p = 6.
   - The window 2πJ̃/9 < T < πJ̃/2 follows at bare stiffness.

## Re-check (2026-10-01, second pass): corrections to the first pass

The user asked for a second check in case we were wrong. Two classifications in the first pass were too strong.

- **A1 is downgraded to a scope omission (class C).** The physics is confirmed again:
  - R_z(π) leaves H_XXZ, H_PD and the longitudinal Zeeman term invariant and flips H_Γ.
  - Saved V scans at J_Γ = 0 give λ3 ≤ 1e-18 against λ6 = 1.0e-5.
  - However, the paper's text does not explicitly claim Z3 at J_Γ = 0. Its sentence about "the least symmetric term, H_Γ" presupposes J_Γ ≠ 0.
  - It is an error only if a figure applies Z3 on the PD axis, for example if Fig. 4(d) shows a J_Γ = 0 V curve read as threefold. I could not see the figure. **Check: which couplings are used in Fig. 4(c,d)?**
- **A2 is downgraded to a conditional implementation issue.**
  - The paper prints Δ_PG = (1/S)√(det) and adds "see Ref. [71] for the precise expression". Ref. [71] is Rau, McClarty and Moessner, PRL 121, 237201 (2018). So the printed formula is schematic and not a stated derivation.
  - The canonical-pair problem is real in `legacy/scripts/4_Pseudo_Gap.py`: it uses uniform polar offsets, the archived Hamiltonian, and multiplies by S. Fig. 4(b) is affected only if it came from that code. **Check: which code produced Fig. 4(b)?**
  - Taken literally, the printed normalization would still be off. With the natural Y coordinate θ = canting angle t, the relaxed partner direction is correct, but the 1/S prefactor gives Δ_printed/Δ_correct = (2/3) sin t = 0.396 at 0.2 T. The correct Berry weight is (S/3)Σ sinϑ_α u_α, not S.
- **B1's factor is conditional.** The "about 2.5×" assumes the paper's J̃ equals the continuum stiffness ρ_s. The paper does not define J̃ microscopically, so only the qualitative point stands: the window edges use bare, unrenormalized stiffness.
- **C1 is confirmed directly from the raw energies**, not only from the two-harmonic fit. Counting local minima of the 72-point V curves:
  - r = 0.112, 0.224, 0.448, 0.586 and 0.905: 6 minima, at sin 3φ = r. For example, r = 0.448 gives 10°, 50°, 130°, ….
  - r = 1.17, 2.35 and 4.76, and the pure-Γ axis: 3 minima.
- **Re-confirmed:**
  - Y has λ2 and λ3 at roundoff on both axes, so it is Z6.
  - The RG exponent is 9T/(πJ̃) = p²/(4πK) with K = J̃/T and p = 6, and the window edges follow.
  - The 6D weak-pinning dynamics check reproduces √(C_φ/χ_z) to 1e-5.

**Net after the re-check: no definite error in the paper's printed text is confirmed.** The concrete problems are:
1. the missing J_Γ = 0 qualification for V;
2. Fig. 4(b) magnitudes, if that figure came from the legacy gap code;
3. the effective-theory approximations B1–B3.

## Claims of the paper that the note confirms

| Paper | Claim | Status in the note |
|---|---|---|
| Eq. (2), the S6 transformation | Model and the combined lattice–spin S6 | Same bond matrices as `bond_angle_exchange` (PR #4 review); S6 is ch. 2 Eq. eq-nbcp-s6-transform |
| Sec. II "Instability" | SOC lifts U(1) by quantum zero-point fluctuations and opens a gap at T=0 | Confirmed. The classical energy is exactly flat along the orbit (span ≤1.4e-17 meV), so the pinning is purely zero-point. The classical Monte Carlo (MC) run by the roadmap thread finds ⟨cos6φ⟩ = 0 ± 0.02 down to 0.001 meV. |
| Fig. 3(b), Eq. (3) | Y is Z6 | Confirmed for any J_PD, J_Γ. S6 together with inversion gives period π/3 (ch. 8). The order of λ6 is fixed by the selection rule (ch. 6.3): J_PD³ on the PD axis and J_Γ⁶ on the Γ axis. |
| RG flow after Eq. (3) | dg6/dl = (2 − 9T/(πJ̃)) g6; window 2πJ̃/9 < T < πJ̃/2 | Correct for the lattice XY clock model at constant stiffness K = J̃/T (App. B). |
| Eq. (4) | V potential ∝ sin 3φ | Confirmed and now derived. C2′T forces δ3 = ±π/2, i.e. cos(3φ − δ3) = ±sin 3φ (PR #18, ch. 9.4). |
| Sec. II "Finite-T" | Z3 in V removes the KT window | Confirmed for J_Γ ≠ 0. The cos3φ eigenvalue 2 − 9/(4πK) > 0 for every K > 2/π. Near the PD axis λ3 ∝ J_PD·J_Γ is linear in J_Γ (PR #6), so the conclusion is robust there. |

## Issues, by class

### A. Items first classified as definite errors (see the re-check above: A1 is now class C, A2 is conditional)

**A1. V on the pure-PD axis is Z6, not Z3.**
- Where: Fig. 3(c) caption ("a three-fold (Z3) for V̄"); Eq. (4); Fig. 4(b), gray/green curves "Δ_PG as a function of J_PD at J_Γ = 0"; the conclusion "the threefold anisotropy eliminates any stable KT phase".
- Derivation (ch. 9.2, sec-nbcp-v-symmetry): the spin-only rotation U_π = R_z(π) leaves H_PD invariant and maps the V orbit point φ to φ + π. At J_Γ = 0 this halves the period from 2π/3 to π/3.
- Numerics (saved N=48 scans): at J_PD = 0.010 meV, J_Γ = 0, λ3 = 4×10⁻¹⁹ meV/spin, which is roundoff, while λ6 = 1.02×10⁻⁵.
- Consequence: on the PD axis the sixfold window 2πJ̃/9 < T < πJ̃/2 is symmetry-allowed for V as well. The paper's statement "V cannot restore supersolidity" holds only for J_Γ ≠ 0.
- The paper's sentence "the symmetry of the total Hamiltonian is dictated by its least symmetric term, H_Γ" is the source of this: it does not apply when J_Γ = 0.

**A2. The Δ_PG formula is not a canonical frequency in the physical azimuth.**
- Where: Sec. II "Instability", Δ_PG = (1/S)√[(∂²_θ ε)₀(∂²_φ ε)₀ − (∂_θ∂_φ ε)₀²]; plotted in Fig. 4(b).
- The paper's text does not define θ, and does not say whether ε is per spin or per cell. The formula is the equal-weight local-coordinate form of Rau et al. (arXiv:1805.00947, Eqs. 7–9). For the global azimuth of a three-sublattice state, the Berry weights are sin ϑ_α, so the 1/S prefactor and a common θ are not the conjugate pair (App. A, sec-interpretation).
- Y: a common polar increment u = (1,1,1) has Berry coefficient b = 0 in the signed chart (t, −t, π). It does not change m_z, so it cannot pair with φ at all. The relaxed partner is (−1, 1, 0), giving Δ² = C_φ/χ_z with χ_Y = 2/[9(J + J_z)] (App. A Eq. eq-lswt-yv-pg-y-gap).
- V: the uniform partner can be Berry-normalized, but its restricted susceptibility is 0.0068 meV⁻¹ instead of χ_V = 1.551 meV⁻¹. That overestimates the gap by 15.1× before any other factor.
- Numbers (App. A, table 3; J_PD or J_Γ = 0.010 meV, N = 48). The correct value uses the current Hamiltonian and the relaxed response. The archived value uses the archived Hamiltonian and the uniform-θ formula.

  | Case | Correct gap (meV) | Archived gap (meV) |
  |---|---|---|
  | Y/PD | 0.00686 | 0.00245 |
  | Y/Γ | 7.3×10⁻⁶ | 2.6×10⁻⁷ |
  | V/PD | 0.0156 | 0.00472 |
  | V/Γ | 0.00204 | 0.00020 |

- Check: this assumes Fig. 4(b) was produced by `legacy/scripts/4_Pseudo_Gap.py`. If it was, then (i) a uniform polar offset is used for θ, (ii) the archived Hamiltonian with the B/B† assembly issue is used, and (iii) the result is multiplied by S rather than divided, and negative determinants are returned as 0 (issue note 260918, items 2–5). If Fig. 4(b) came from another code, only the formula-level statement above applies.
- Consequence: the qualitative claim (a gap opens, scaling with SOC order) survives. The plotted magnitudes in Fig. 4(b) are off by factors of about 3 (PD) to 10–30 (Γ).

### B. Unjustified steps or approximations

**B1. Constant (bare) stiffness in the window formula.**
- Where: the RG after Eq. (3), g6(l) = e^{(2 − 9T/πJ̃)l} g6(0), and the window 2πJ̃/9 < T < πJ̃/2.
- The exponential solution holds only if K does not flow. Vortices renormalize K, and the window edges are set by the renormalized stiffness (App. B, eq-nbcp-clock-pinning-flow). The existence of a window for p = 6 is robust (José–Kadanoff–Kirkpatrick–Nelson). The quoted temperatures are not.
- Size (inferred, normalization to be checked): the zero-SOC Y stiffness is ρ_s = JS²(1 − c²)/√3 = 0.0038 meV, so the bare upper edge is πρ_s/2 ≈ 0.0060 meV. The roadmap thread's classical MC gives T_BKT ≈ 0.0024 meV and drifting down with L, about 2.5× lower.

**B2. No microscopic matching for J̃ and g6 (Eq. 3).**
- Where: Eq. (3) and its use at finite T.
- J̃ is not derived from the spin model. g6 is the T = 0 zero-point coefficient, used as the bare coupling at finite T.
- The note (ch. 7.4) treats Eq. (3) as a physically motivated effective-theory hypothesis, not a derivation. It lists the missing matching steps: stiffness tensor, f(φ,T) at a recorded cutoff, vortex-core fugacity, and the density-wall/vortex composites.
- Harmonic f(φ,T) result (my N = 48 split, with roadmap D39 for cross-check): thermal fluctuations lower the Y sixfold amplitude by 20–33% at T = 0.01 meV for a matching cutoff |k| ≤ 0.25. The soft-branch angular entropy opposes the zero-point selection. This does not change whether g6 is relevant in linear RG. It shifts the lock-in crossover scale, and it is cutoff-dependent.

**B3. Isotropic, angle-independent coupling in Eq. (3).**
- With J_PD = 0.010 meV the Y stiffness tensor is anisotropic: ε_ρ = 43% (ch. 8, angular matching).
- The KT criterion should use √det ρ at the matching scale. The size of the effect is quantitative, not qualitative.

### C. Missing pieces

**C1. V near the PD axis has six minima, not three.**
- Where: Fig. 3(c) and Eq. (4).
- With both harmonics, v = −λ6 cos 6φ − λ3 sin 3φ. Its minima satisfy sin 3φ = r with r = λ3/(4λ6). For r < 1 there are six degenerate minima, related by Z3 and C2′T, with unequal barriers. The Z3 picture with three minima holds only for r > 1.
- At J_PD = 0.005 and 0.010 meV, r = 1 falls at J_Γ/J_PD ≈ 0.09 and 0.22.
- At r = 1 the leading-order gap closes: C_φ = 36λ6(1 − r²). Higher-order terms set the true gap there.
- The paper's conclusion of no KT window still holds, because cos3φ is relevant.

**C2. The V gap is non-monotonic in mixed SOC.**
- Fig. 4(b) shows only the pure axes. Along a mixed line the leading-order V gap first falls to zero at r = 1, then reopens (ch. 9.4, Table tab-nbcp-v-mixed-soc).

**C3. Persistence of density order through the window is asserted, not computed.**
- The supersolid needs D_z ≠ 0 at T_BKT. Classical MC (Y, 0.2 T) supports this: T_d ≈ 0.015 meV ≫ T_BKT ≈ 0.0024 meV, with a phase-disordered density-ordered window in between.
- This is a classical result. Quantum S = 1/2 temperatures are not given by it.

**C4. Defects.**
- Vortex-core energies, density walls, and the composite defects between translation domains and angular minima are not discussed.
- The note's density-wall calculation (ch. 8) finds a finite wall tension. Vortex–wall coupling is the next calculation.

**C5. The Ψ phase was not checked here.**

## Short verdict

- The T = 0 statements and the Y finite-T scenario are qualitatively sound as an effective-theory argument.
- There are two concrete problems: the V = Z3 claim on the pure-PD axis (A1), and the gap formula and Fig. 4(b) magnitudes (A2).
- The finite-T window is a hypothesis with bare-stiffness temperatures (B1–B3). It is not a microscopic result.
- Update (10:25Z, after PR #20/#22): the confirmed error is the gap formula (Rau Eq. 1 on noncollinear Y/V), so Fig. 4(b) magnitudes are unreliable whatever code produced them. A1 is a scope overstatement only at J_Γ = 0. The Y window's existence follows from Z6 + JKKN; its temperatures and whether NBCP actually realizes it need quantum finite-T matching (pinning is purely zero-point, so classical MC cannot test lock-in). Skyrmion relay: at J_PD = 0.010 meV the 4-site SkX loses, so it does not compete with the supersolid at the argument's parameters.

## Evidence

- `docs/nbcp/chapters/09-v-phase.tex` (V symmetry, mixed SOC)
- `docs/nbcp/appendices/a-pseudo-goldstone-gap.tex` (gap)
- `docs/nbcp/appendices/b-clock-rg.tex` (RG)
- `data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json`
- `data-space/verification/261001-v-mixed-soc/`
- `data-space/verification/261001-c2t-reflection/`
- `GOVERNMENT/Working-Pad/issue-notes/open/260918-nbcp-physics-code-review.md` (legacy gap code)
- `/mnt/project-files/nbcp-thermal/` (f(φ,T) and classical MC from the roadmap thread; cutoff split)

## Update 2026-10-02 18:3xZ: classical vortex pairs (PR #30)
Pair energies on tori: mu = 0.0081 meV per core (J_PD=0); at J_PD=0.010 the pair coefficient is orientation-dependent (0.0068 soft, 0.0083 stiff); no lattice barrier. Harmonic edges T* 0.0027-0.0031 meV vs MC 0.0024 (10-25%); core free energy convention-dependent (~17x). Earlier large-fugacity reading withdrawn. Details: /mnt/project-files/nbcp-thermal/vortex-pair/2026-10-02-vortex-pair-summary.md
