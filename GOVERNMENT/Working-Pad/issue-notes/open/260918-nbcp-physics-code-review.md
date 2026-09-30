---
frontmatter-version: 1
title: NBCP physics-to-code review
section: working-pad/issue-notes
status: in-review
last-edited-by: claude
created: 2026-09-18
updated: 2026-09-30
---

# NBCP physics-to-code review

## Scope and current status

This record accompanies the user-approved nine-chapter reorganization of [the NBCP manuscript](../../../../docs/nbcp/main.tex). It owns the implementation mapping and open review questions; physical derivations remain in the manuscript. The current pass reads the implementation entry points and existing evidence. It does not rerun all physics calculations or certify the whole package. A saved numerical check, an independent physical derivation and human acceptance are distinct states.

## Claim-to-implementation map

| Physical object | Implementation | Existing evidence | Remaining review |
|---|---|---|---|
| Nearest-neighbor exchange and magnetic cells | [make_nn_exchange_matrices and cell builders](../../../../examples/nbcp_ground_state.py) | Model matrices and three-sublattice energy in manuscript Chapters 2–3; saved geometry audit | Independently reconstruct directed neighbors and bond multiplicities from the declared lattice; resolve displacement wording. |
| Classical energy and stationarity | [EnergyFunction.classical_energy_density_func](../../../../code-space/spintoolkit/methods/lswt/energy.py), [classical_from_bonds](../../../../examples/pseudo_goldstone_comparison.py) | Saved classical derivative and stationary-state comparisons | Track signs, spin factors and per-spin normalization across arbitrary cell sizes; do not infer global stability from stationarity. |
| Quadratic HP/BdG Hamiltonian and vacuum energy | [Quadratic_Bose_Hamiltonian and compute_quantum_energy](../../../../code-space/spintoolkit/methods/lswt/hamiltonian.py), [direct_hp and vacuum_energy](../../../../examples/pseudo_goldstone_comparison.py) | [Earlier zero-point normalization review](../closed/260910-zero-point-energy-normalization.md); matrix comparison in manuscript Appendix D | Review current generic conventions beyond the sampled NBCP backgrounds, including unstable modes and normal-ordering constants. |
| Pseudo-Goldstone gap and canonical response | [Comparison](../../../../examples/pseudo_goldstone_comparison.py), [conjugate-mode diagnostic](../../../../examples/pseudo_goldstone_conjugate_mode.py) | Appendix A, saved coordinate and susceptibility checks | Preserve the distinction between leading curvature matching and a full quantum self-energy calculation. |
| Classical Y stiffness and SOC reduction | [Y stiffness](../../../../examples/nbcp_y_stiffness.py), [SOC reduction](../../../../examples/nbcp_y_soc_conditions.py) | [SOC checks](../../../../data-space/verification/260917-y-soc-conditions/soc-conditions-check.json) and Appendix C | Audit physical-site/cell Fourier conventions and whether eliminated modes remain separated in the intended regime. |
| Full-zone stability and angular matching | [Stability](../../../../examples/nbcp_y_stability.py), [angular matching](../../../../examples/nbcp_y_angular_matching.py), [independent routes](../../../../examples/nbcp_y_angular_validation.py) | Saved stability and angular evidence linked in Appendix D | These share geometry and model inputs; same-model agreement is not independent material validation. |
| Nonlinear smooth waves and density walls | [Smooth-wave constraints](../../../../examples/nbcp_y_nonlinear_gradient.py), [wall minimization](../../../../examples/nbcp_y_density_wall.py), [wall refinements](../../../../examples/nbcp_y_density_wall_validation.py) | Chapter 7 and corresponding result JSON files | Check the physical constraint, sphere/chart coverage, two-wall subtraction and metastability separately from optimizer flags; no global-minimum proof. |
| Thermal supersolidity and clock RG | [Symbolic RG checks](../../../../examples/nbcp_clock_rg_checks.py); theory in Chapter 6 and Appendix B | Normalization identities and conditional scaling criteria | No microscopic thermal trajectory, vortex-core weight or finite-size transition determination has been computed. |
| Skyrmion and complete V matching | Four-sublattice builder; local V angular/gap diagnostics | Chapters 4 and 8 state the available evidence | SkX signed charge and competing-phase scan are unverified; V stiffness, defects and thermal matching remain open. |

## Priority convention issue: displacement direction

The current [SpinSystem.Coupling documentation](../../../../code-space/spintoolkit/system/spin_system.py) describes displacement as a vector from site i to site j. The existing NBCP [geometry_audit and torus_curvature](../../../../examples/nbcp_y_soc_conditions.py) use the reverse embedding, with the neighbor at r_i minus the stored displacement. The nonlinear wave and wall diagnostics follow that existing embedding. This is a confirmed wording/embedding discrepancy, not a conclusion that every spectrum or energy is wrong. Resolving its physical impact requires tracing the Fourier sign, basis positions and directed bonds together; no convention or production code is changed by this document reorganization.

### 2026-09-30: resolved by the toolkit convention D13

The displacement question is settled at the code level; manuscript wording is a separate edit.

- Convention (transfer contract D13, user decision 2026-09-29): a stored displacement is
  `d = r_source - r_target`; the target site sits at `r_source - d`, and the Hamiltonian phase is
  `exp(-i k . d)`. The `SpinSystem.Coupling`/`add_coupling` docstrings state this since stage 2.
  The earlier "from i to j" wording no longer exists in the code.
- Numerical confirmation: stage 2 compared the common-model Hamiltonian built with
  `r_target = r_source - d` against the existing NBCP builders element by element over 20 cases
  (largest difference 1.7e-16; `docs/development/verification/stage2-nbcp-connection-2026-09-29.json`).
- Momentum sign and conjugation: stage 3 matched the one-magnon ED spectrum at every torus momentum,
  with DM fixing the k sign; stage 5a matched the particle block of `H(k)` element by element to the
  Bloch matrix rebuilt from ED one-magnon states (2.7e-14; its complex conjugate differs by 2), so
  the code's `k` is the momentum of the physical Bloch state
  `a_k^dagger = N^-1/2 sum_r exp(i k . r) a_r^dagger`.
- Consequence: calculations that embed the neighbor at `r_i - d` (the NBCP geometry audit, torus
  curvature, smooth-wave and wall diagnostics) use the physical momentum. An odd-in-k response can
  be given a laboratory direction once the orientation of the model's `x` axis relative to the
  crystal axes is stated; that orientation is a material input, not a code convention.
- Remaining in this review: the manuscript sentences that call the sign unresolved (Chapter 8,
  "The unresolved displacement sign must be fixed ..."; Chapter 10, "The physical displacement
  convention must also be resolved ...") and the crystal-axis orientation.

## Independent review order

1. Fix the physical meaning of each coordinate and normalization before comparing numerical output.
2. Reconstruct bond counting and classical energies without reusing the production bond list; include distinct magnetic cells and a known limiting case.
3. Derive the quadratic kernel with the same explicit convention and compare matrix elements, partner structure and observables. Comparison only with the archive is insufficient.
4. Check the constrained nonlinear calculations against their stated variational problem, including branch and boundary restrictions.
5. Record each conclusion as implemented, numerically checked in a stated regime, independently supported, or unresolved. Human physics acceptance remains separate.

## Preserved legacy implementation findings

The following findings were moved from the former manuscript section without changing their scientific content. Source line locations refer to the hashes saved with the original gap scans, as stated in that record. Directional references such as “above” and “below” in the preserved text refer to the original comparison; the corresponding matrix results now appear in [Appendix D](../../../../docs/nbcp/research-note.md#sec-legacy).


1. `legacy/scripts/2_U_symmetry_YV.py::_get_angles` and `4_Pseudo_Gap.py::_get_E_cl_and_qm` use a common azimuthal increment. That direction is appropriate for this SOC-induced Y/V gap. It must not be replaced by the different internal zero-field XXZ orbit.

2. `4_Pseudo_Gap.py` also increments every polar angle by the same amount to compute the hard curvature. In the signed Y coordinates used here, its Berry pairing with the global-z orbit vanishes. In V this pair can be Berry-normalized, but its amplitude distribution is not the relaxed low-energy response. The detailed check above distinguishes these two failures.

3. The same script hardcodes `spin = 1/2` and returns `sqrt(hessian_det) * spin`. The input energies already contain physical spin factors. Neither multiplying by S nor simply changing this to division by S repairs an unnormalized, incorrect coordinate pair.

4. It imports `modules.Tools.analysis_tools.Create_Energy_Function`, which still uses the archived Hamiltonian with the B/B-dagger assembly issue. Fixes in `code-space/lswt` do not change this import. Matrix-level comparisons against independent Cartesian HP coefficients are saved below.

5. Negative determinants and energy-evaluation exceptions return 0.0. That conflates an invalid calculation with a physical gap closing. The input azimuth is not automatically minimized within the gap class. The zero mixed-curvature assignment is consistent with the leading type-I approximation under the conditions derived above; it is not a separately demonstrated source of the leading discrepancy.

6. The azimuthal energy plot divides by an angle-dependent energy range; on the exact-U(1) baseline that range is roundoff. Such normalized curves can visually amplify numerical noise. The figures in this audit retain absolute energy units.

Code locations in `legacy/scripts/4_Pseudo_Gap.py`: the archived energy-factory import is on line 7, uniform polar/azimuthal offsets on lines 55-58, exception-to-zero handling on lines 117-119, mixed-curvature assignment on line 135, and determinant/spin-factor handling on lines 149-157. These locators refer to the source hashes saved with the scans.



## Structural reorganization verification

The manuscript now has nine main chapters and four appendices, with Skyrmion and V evidence limits explicit. The original 87 display equations, numerical tables, figures and explicit semantic identifiers are preserved. Detailed legacy implementation findings were moved here; the corresponding manuscript anchor remains in Appendix D. The exporter change only expands the table of contents to two levels. Existing production, legacy, calculation scripts and numerical data were preserved against a pre-edit snapshot.

The [section disposition map](../../../../data-space/verification/260918-nbcp-restructure/section-mapping.json) and [document checks](../../../../data-space/verification/260918-nbcp-restructure/document-check.json) record the reorganization. This is structural and output verification; the implementation map above is an initial review record, not a completed independent physical audit.


## 2026-09-18: NBCP-only LaTeX source migration

The user approved limiting the proposed LaTeX authoring change to NBCP. The
[administrative source-authority decision](../../../Court-Precedents/2026-09-18-nbcp-latex-source-authority.md)
records that scope. The editable entry point is now
[main.tex](../../../../docs/nbcp/main.tex), with nine chapter files, four
appendices, a reference file, and separate presentation/metadata files.
The old Markdown is a navigation stub; its exact pre-migration content,
converter and style files are retained in
[the dated archive](../../../../docs/archive/nbcp/2026-09-18-markdown-source.zip).

The exporter now compiles TeX directly with XeLaTeX. Quarto is no longer a
build dependency. Figures remain owned by the existing calculation data;
PDF evidence links use repository-relative paths. The source map preserves
all 120 explicit Markdown IDs as native TeX labels (157 labels including
automatically named subsections). All 87 displayed equations, 879 total
inline/display math blocks, 18 tables and eight figure inputs were preserved.

[Migration checks](../../../../data-space/verification/260918-nbcp-tex-migration/document-check.json)
record a 51-page PDF, no undefined/duplicate references, no overfull or missing
glyph messages, and 49 pixel-identical pages. Page 2 retains identical text;
page 50 updates authoring and build provenance. All rendered pages were
inspected, with no clipping or overlap found. The previous source and exporter
were checked byte-for-byte against the archive. Production code, numerical
results and LSWT general theory are unchanged (237 protected existing files).

This is a source-format and presentation migration, not a new physics result
or acceptance. NBCP remains in review. The displacement/bond-count audit and
vortex/thermal matching remain separate open work.


## 2026-09-21: one-page Introduction revision

The user requested recent spin-supersolid research, representative materials,
the scope of this work, and an NBCP structure figure in at most one page.
[Section 1](../../../../docs/nbcp/chapters/01-introduction.tex) now contains
three material rows, the NBCP structure/texture panel from Gao et al. (2022),
and two scope items. The figure is credited under CC BY 4.0, with panel (b)
omitted through LaTeX clipping. The July 2026 K2Co(SeO3)2 result is explicitly
a preprint. Literature claims and unfinished thermal validation are separated.

[Sources and figure provenance](../../../../docs/nbcp/sources/260921-introduction-sources.md)
and [render checks](../../../../data-space/verification/260921-nbcp-introduction/document-check.json)
record the review. Section 1 occupies PDF page 3 only; Section 2 begins on page
4. No overfull/missing-glyph/undefined-reference messages occurred. The other
17 existing TeX/build inputs are unchanged, including the user's page break
after the contents. All 52 PDF pages were visually checked; the Introduction
was inspected at page size. No calculation or physics acceptance was added.


## 2026-09-21: captions and numeric literature citations

At the user's request, all nine figures now have bottom captions and all
19 tables have numbered top captions. Longtables repeat their column headers
on continuation pages without repeating the initial caption. Literature links
in the content now use native LaTeX citations, with 11 deduplicated numbered
bibliography entries in first-citation order. Original-paper locators and
source-role qualifications are retained; code/data/license links remain direct.
The convention is recorded in the NBCP editing guide. The Introduction remains
one page; only local layout spacing was adjusted. Displayed equations and
existing table data are unchanged. The check record is
[data-space/verification/260921-nbcp-captions/document-check.json](../../../../data-space/verification/260921-nbcp-captions/document-check.json).
No physics calculation or acceptance was added.


## 2026-09-21: bottom table captions and concise Section 1

The user clarified that prose shortening applies to Section 1 only. Its
material comparison, NBCP introduction, captions and scope are now concise,
while retaining the seven literature citations, figure attribution and open
thermal-validation boundary. All 19 table captions now appear below the
tables, superseding the top-caption convention above. Multipage tables place
the caption below the final segment and retain repeated column headers.
Other chapter and appendix prose is unchanged.

[Verification](../../../../data-space/verification/260921-nbcp-concise/document-check.json)
confirms a one-page Introduction on PDF page 3, 52 total pages, and no LaTeX
review messages. All pages were visually inspected; no caption overlap or
clipping was observed. This is an editorial change, not physics acceptance.


## 2026-09-21: Model Hamiltonian and TikZ bond diagram

At the user's request, Section 2 now presents the XXZ, PD and Gamma
Hamiltonians and the three symmetry remarks using the supplied paper excerpt
and Park et al., Section II, Eqs. (1)-(2), checked at
https://arxiv.org/html/2601.20963v1. The crystallographic reduction is qualified
as the unitary point group. A TikZ nearest-neighbor shell shows the three
unoriented bond families, phases, coordinate axes and perpendicular field.

The entire previous Section 2 is preserved as Appendix E, except for its
section heading and section label. Existing subsection and equation labels
are retained, including the exchange-matrix reference used by Appendix C.
No production code or numerical results were changed. A symbolic expansion
checks the equivalence of the displayed bilinears and original exchange
matrix; a numerical rotation checks the S6 bond-family mapping.

[Document checks](../../../../data-space/verification/260921-nbcp-model/document-check.json)
record the 54-page build without LaTeX review messages and visual inspection
of the changed pages. Section 2 occupies pages 4-5 and Appendix E pages 52-53.
The Introduction remains one page. This is source-grounded restructuring
and local algebraic verification, not new physics acceptance.


## 2026-09-21: supplied lattice source and expanded Appendix E

The user supplied a triangular-lattice TeX draft for the figure and appendix.
Section 2 now shows a repeated lattice patch, delta_1/2/3 vectors, bond phases
and perpendicular field. Appendix E spells out bond-centered inversion,
C2x constraints, parameter definitions and C3 covariance. The general-angle
rotation result was checked symbolically. Existing canonical exchange
matrices, model coefficients and production code are unchanged.

[Source integration record](../../../../docs/nbcp/sources/260921-triangular-lattice-integration.md)
links the verbatim reference snapshot and distinguishes the source's polar-like
inversion notation from the retained axial-spin rule. NNN derivations remain
outside this integration pending their existing symmetry/convention review.
[Render checks](../../../../data-space/verification/260921-nbcp-lattice/document-check.json)
record 55 pages with no LaTeX review messages. The changed figure and all
three Appendix E pages were visually checked; the short parameter table
remains together. This does not change physics acceptance.
