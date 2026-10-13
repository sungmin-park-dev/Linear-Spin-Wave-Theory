---
frontmatter-version: 1
title: Documentation Consolidation by Topic
section: issue-notes/closed
issue-type: structure
status: closed
last-edited-by: codex
created: 2026-09-16
updated: 2026-09-16
---

# Documentation Consolidation by Topic

## User-approved scope

The user approved combining `docs/` and `research-space/` under one document root, separating LSWT general theory from NBCP research, and distinguishing editable documents, sources, archives and generated outputs. This record updates the path clauses of the 2026-08-09 authoring-surface decision. It does not change the single-canon principle, document acceptance, physical claims or source-evidence roles.

The user also clarified that a well-written research note should be organized around its content and reasoning. The existing visual design was retained. This migration does not implement the separately identified TeX editability improvements or convert the NBCP note to a TeX master.

## Initial path mapping

| Previous path | Current path | Role |
|---|---|---|
| `docs/00-foundations/` through `docs/04-appendices/` | `docs/lswt/00-foundations/` through `docs/lswt/04-appendices/` | Editable LSWT theory Markdown |
| `docs/README.md` | `docs/lswt/README.md` | Existing LSWT reading order; a new root README routes between topics |
| `research-space/sources/` | `docs/lswt/sources/` | Preserved primary PDF, TeX lineages, references and verification evidence |
| `data-space/verification/260912-pseudo-goldstone/comparison-report.md` | `docs/nbcp/pseudo-goldstone/comparison-report.md` | Editable NBCP research note; still in review |
| `comparison-report.tex`, `comparison-report.pdf`, `report-export.json` in that data directory | `docs/build/nbcp/` | Generated outputs and export provenance |
| Calculation JSON, logs and original figures | Unchanged under `data-space/verification/` | Numerical evidence |
| Existing archived code and notes | Unchanged under `legacy/` | Historical comparison evidence |

`research-space/` was removed after its source collection was moved. The new `docs/archive/` explains the archive role and points to the existing legacy notes; no active document was retired during this operation. `docs/nbcp/sources/` indexes the research note's references and does not duplicate LSWT source files.

## Initial navigation and generation

- [Document index](../../../../docs/README.md) separates topics and file roles.
- [LSWT index](../../../../docs/lswt/README.md) retains the existing reading order.
- [NBCP index](../../../../docs/nbcp/README.md) identifies the editable research note and its evidence.
- Current instructions, path policies, theory frontmatter and active issue references use the new locations. Clickable links in historical records resolve to the moved files; historical prose retains the paths used at the time.
- The exporter reads the NBCP Markdown and the saved figures from their separate locations. Its temporary input uses local image paths, and the final build includes two PNG copies for standalone TeX compilation.
- Figure regeneration checks the moved Markdown before creating an initial report. It preserves the authored note and does not recreate a note in the data directory.

## Initial verification

- All 11 preserved LSWT source files other than the navigation README retain their SHA-256 hashes.
- All 22 saved data assets, 52 legacy files and 40 package/test files included in the pre-migration integrity snapshot remain unchanged.
- Display equations and review states in the 17 LSWT theory files and the NBCP note are unchanged. The NBCP numerical table rows are unchanged.
- Both modified Python scripts parse. Figure regeneration in a temporary data directory produces PNGs identical to the saved figures, preserves the authored Markdown and creates no duplicate report in data-space.
- Quarto and XeLaTeX regenerate the report at its new build location. The export manifest identifies the source, data and output directories and matches the source, exporter and output hashes.
- All 232 local Markdown links checked across documents, governance and root guidance resolve; template examples inside code fences are excluded. All 24 authored document frontmatter paths resolve to their containing directories.
- The PDF remains 11 pages. Raster comparison shows pages 1-10 are identical to the pre-migration preview; only the reproduction paths on page 11 changed. The rendered pages were visually checked. No overfull boxes, missing glyphs or unresolved references were reported; the previous sectioning, microtype and one underfull-box warning remain.

Directory organization and output checks do not establish physics acceptance. The open [LSWT documentation review](../open/260809-lswt-documentation-audit.md) and [pseudo-Goldstone issue](../open/260810-pseudo-goldstone-gap.md) continue to track their existing questions.

## Follow-up: one NBCP manuscript and one output folder

On 2026-09-16 the user requested merging duplicate content and choosing either Markdown or TeX as the authoring source. The final arrangement keeps **`docs/nbcp/research-note.md` as the only NBCP manuscript**. This follow-up supersedes the NBCP authoring and output paths in the initial mapping above; it does not change the LSWT evidence collection or its authoring rules.

| Previous item | Final owner or disposition |
|---|---|
| Main chapters in `docs/nbcp/research-note.md` | Same file; existing chapter order retained |
| `pseudo-goldstone/clock-anisotropy-review.md` | Full symmetry, numerical and RG discussion in chapter 3 |
| `pseudo-goldstone/comparison-report.md` | Complete derivation, comparisons and convergence evidence in Appendix A |
| Separate clock and gap source lists | Collected in the manuscript's References, with their source roles retained |
| `docs/nbcp/sources/README.md` | Navigation folded into the NBCP README |
| `docs/nbcp/typesetting/` | `examples/nbcp-note-style/`; presentation only, unchanged file bytes |
| Separate and integrated products under `docs/build/` | Replaced by `docs/nbcp/output/research-note.pdf` and `research-export.json`; old output tree removed |
| `examples/pseudo_goldstone_export.py` | Retired; `examples/nbcp_research_export.py` is the sole manuscript exporter |
| Initial report writer in `pseudo_goldstone_plot.py` | Removed; figure regeneration cannot recreate retired Markdown |

The [NBCP README](../../../../docs/nbcp/README.md) now identifies the only editable manuscript, the PDF, evidence and the regeneration command. The exporter retains no TeX, rendering Markdown or copied images. Those files exist only in a temporary directory during export. The ordinary LaTeX preamble retains the previous typography and command interfaces.

The [recovery ZIP](../../../../data-space/verification/260916-nbcp-integration/consolidation-before.zip) contains the pre-merge authored NBCP files, exporter and plot scripts, navigation and historical export manifests. It is a recovery snapshot, not an active authoring copy. Regenerable PDF/TeX/image products were not archived. The Overleaf audit remains source evidence and is not a full remote-project backup.

The [preservation check](../../../../data-space/verification/260916-nbcp-integration/consolidation-check.json) records source hashes, preservation and link checks. All 15 main-note, 16 gap-note and 12 clock-note display equations remain represented; the repeated identical clock-window equation has one owner, leaving 42 displays. All 67 table lines and the detailed notes' scientific paragraphs remain present. Existing data files remain unchanged, and regenerating the figures in a temporary directory produces byte-identical PNGs without creating or modifying Markdown. The research remains `in-review`; no new physics scan or acceptance is implied.

Final output verification: 21 PDF pages rendered and inspected; no overfull boxes, missing glyphs, unresolved references or duplicate labels reported. All 256 checked local links resolve. Source and figure hashes match the export manifest.
