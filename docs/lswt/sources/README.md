# LSWT Source Map

This directory collects the source evidence used while rewriting the Linear Spin Wave Theory documentation in [`docs/lswt/`](../README.md). These files preserve provenance and support comparison; they do not own the project's current theory claims. Only user-reviewed theory Markdown in the numbered `docs/lswt/` directories can become the accepted LSWT theory canon. Source files are preserved here even when their format is editable.

## Reading Priority

| Priority | Directory | Role |
|---|---|---|
| 00 | [`00-primary-source/`](00-primary-source) | The strongest record of the original LSWT note, including review annotations |
| 01 | [`01-editable-notes/`](01-editable-notes) | Editable TeX lineages and their rendered snapshots |
| 02 | [`02-reference-papers/`](02-reference-papers) | External physics references used to check derivations and conventions |
| 03 | [`03-verification-notes/`](03-verification-notes) | Internal notes that compare formulas, basis conventions, and code |

The numeric prefixes define evidence-review order, not independent levels of theory canon. When sources disagree, do not merge them silently: return to the primary PDF, record the discrepancy, and leave the result `Unknown` until it is resolved by physics and mathematics review.

## Source Roles

| Material | Role and limitation |
|---|---|
| [`Linear_Spin_Wave_Theory___Note.pdf`](00-primary-source/Linear_Spin_Wave_Theory___Note.pdf) | Primary evidence. Use it to recover the original structure, equations, prose, and 33 visible `[REVIEW: ...]` annotations. |
| [`note.tex`](01-editable-notes/note.tex) | Earlier editable note in the original section order. |
| [`note_lswt_reviewed.tex`](01-editable-notes/note_lswt_reviewed.tex) | Editable transcription that retains the original section order and contains 33 `\REVIEW{...}` callouts. It assists comparison but does not override the primary PDF. |
| [`note_lswt_restructured.tex`](01-editable-notes/note_lswt_restructured.tex) | Structurally reorganized TeX reference containing the same 33 review callouts and additional appendix organization. It is not the canonical authoring surface. |
| PDFs in `01-editable-notes/` | Rendered snapshots of their corresponding TeX lineages. They are reference outputs, not editable masters. |
| [`1402.6069v4.pdf`](02-reference-papers/1402.6069v4.pdf) | External LSWT reference: J. Toth and B. Lake, *Linear spin wave theory for single-Q incommensurate magnetic structures*. |
| [`hamiltonian_convention.tex`](03-verification-notes/hamiltonian_convention.tex) | Internal derivation and comparison note for the Hamiltonian basis and legacy code. It is verification evidence, not theory authority. |

The exact production relationship between the 27-page primary PDF and `note_lswt_reviewed.pdf` is currently `Unknown`; visual and structural similarity does not establish that one file was generated directly from the other.

`PhysRevB.111.075167.pdf` is intentionally excluded from this directory. It is a writing-style reference rather than an LSWT physics source.

## Integrity Record

| File | Pages | SHA-256 |
|---|---:|---|
| `00-primary-source/Linear_Spin_Wave_Theory___Note.pdf` | 27 | `f00ac6c0bd33779702361a1b4237fe7962f000a2bd105c8c89ba37603ecd3503` |
| `01-editable-notes/note.tex` | — | `ce1874442cbfd385d69ae46b6d4859932d89617628968a01e300b36e706181eb` |
| `01-editable-notes/note.pdf` | 26 | `f846fe728cd08f1fb5040fa44a11cca448e607482741a21b0ea3188f841e6a50` |
| `01-editable-notes/note_lswt_reviewed.tex` | — | `bb9001229539933387660a85d2d0537426c1a5afe5104f22f5b7c36d8fab0924` |
| `01-editable-notes/note_lswt_reviewed.pdf` | 26 | `e76f88c7f644f738c68c00d95d031ef21e5c2975376f41d5028b62efc2d03eea` |
| `01-editable-notes/note_lswt_restructured.tex` | — | `8f4a1fc95927b486f28da8d415a7093f00233d4636c505338fda6209ad00f124` |
| `01-editable-notes/note_lswt_restructured.pdf` | 27 | `f5dd6e3bc3673c8a2e3a29551a95b809fa0e8c64f0fc8229f91fb0a3e99b5aa2` |
| `02-reference-papers/1402.6069v4.pdf` | 12 | `25e7b9fdae30bd03d607cec8a7b33868b491f99483252e4ab471fb20aa27ff42` |
| `03-verification-notes/hamiltonian_convention.tex` | — | `719d266aca2073aaa1c5328a1964fe8ba43522d8a41a7f856685395a4686c624` |
| `03-verification-notes/hamiltonian_convention.pdf` | 4 | `526d965e4b66fe05fce4ed506e46f524ffd57dc2f011b04f9e52c792f56eed0d` |

All three TeX notes use the shared [`Setting/citation.bib`](01-editable-notes/Setting/citation.bib). REVTeX may regenerate job-specific `*Notes.bib` control files during a build; these generated files are ignored and are not part of the source collection.

## Source-to-Documentation Map

This table is a navigation aid. It identifies the Markdown owner that should receive reviewed content; it does not assert that the content has already been transferred or accepted.

| Original note section | Current Markdown owner |
|---|---|
| Summary of Notation and Symbols | [`notation-and-conventions.md`](../00-foundations/notation-and-conventions.md) |
| Introduction to Spin Wave Theory: model and workflow | [`lswt-overview.md`](../00-foundations/lswt-overview.md) |
| Introduction to Spin Wave Theory: Hamiltonian, Eqs. (1)–(3) | [`bilinear-spin-hamiltonian.md`](../00-foundations/bilinear-spin-hamiltonian.md) |
| Spin to Boson Transformations | [`holstein-primakoff-expansion.md`](../01-derivation/holstein-primakoff-expansion.md) |
| Rotations for Spin Models | [`classical-order-and-local-frame.md`](../00-foundations/classical-order-and-local-frame.md) |
| Bosonic representations for spin Hamiltonian | [`real-space-boson-hamiltonian.md`](../01-derivation/real-space-boson-hamiltonian.md) |
| Momentum Space Representations | [`momentum-space-bdg-hamiltonian.md`](../01-derivation/momentum-space-bdg-hamiltonian.md) |
| Diagonalization of Quadratic Boson Hamiltonian; Paraunitary Diagonalization | [`paraunitary-diagonalization.md`](../01-derivation/paraunitary-diagonalization.md) |
| Physical Quantities in Linear Spin Wave Theory | [`magnon-observables.md`](../02-observables/magnon-observables.md) |
| Thermodynamics in Linear Spin Wave Theory | [`thermodynamics.md`](../02-observables/thermodynamics.md) and [`thermodynamic-derivations.md`](../04-appendices/thermodynamic-derivations.md) |
| Correlations in Linear Spin Wave Theory | [`spin-correlations.md`](../02-observables/spin-correlations.md) |
| Structure Factor; Spectral Function | [`structure-factor-and-spectral-function.md`](../02-observables/structure-factor-and-spectral-function.md) |
| Skyrmions, Topological Magnons, and Hall Effects | [`topological-magnon-quantities.md`](../02-observables/topological-magnon-quantities.md) |
| Example: Solving spin system with linear spin wave theory | [`worked-example.md`](../03-examples/worked-example.md) |
| Restructured TeX appendices: Luttinger–Tisza, paraunitarity proofs, thermodynamic derivations | [`04-appendices/`](../04-appendices) |

## Editing Boundary

- Preserve source files as evidence; do not edit them merely to match the current Markdown notation.
- Apply canonical notation and explanatory changes only in the appropriate `docs/lswt/` theory owner.
- Keep theory acceptance, code verification, and publication as separate review states.
- Record unresolved sign, index, normalization, and source-lineage questions explicitly instead of inferring an answer from file names.
