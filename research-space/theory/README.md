---
frontmatter-version: 1
title: Linear Spin Wave Theory Documentation
section: theory
status: in-review
last-edited-by: codex
created: 2026-03-22
updated: 2026-08-01
---

# Linear Spin Wave Theory Documentation

This directory is the gateway for the project's spin-theory documentation.

The active LSWT canonical workspace is now [`lswt/`](lswt/). The older
`sections/` files remain as converted Markdown source material until their
contents are migrated into the canonical LSWT structure.

Here, `canonical workspace` means the active target for canonicalization. It
does not mean that the documents have been accepted as user-approved canon.

## Active Workspace

| Path | Role | Editing status |
|---|---|---|
| [`lswt/README.md`](lswt/README.md) | Goal, approved source authority, and editing rules | Active, in review |
| [`lswt/map-lswt.md`](lswt/map-lswt.md) | Navigation and document-role boundaries | Active, in review |
| [`lswt/current-sections-audit.md`](lswt/current-sections-audit.md) | Coverage, lifecycle, review issues, and open questions | Active audit |
| [`lswt/`](lswt/) subdirectories | Draft canonical LSWT theory documents | Active drafts and skeletons |
| [`common/README.md`](common/README.md) | Draft cross-solver convention candidate 영역 | In review; 2 drafts, accepted 0 |
| [`sections/`](sections/) | Earlier converted Markdown used for comparison | Source-only; preserve |
| [`notation.md`](notation.md) | Earlier converted notation summary | Source-only; preserve |

The legacy converted files still contain material that has not been migrated.
Do not delete, rename, or treat them as the current reading path until
`lswt/current-sections-audit.md` confirms that their coverage has been reviewed.

## Legacy Converted Sources

The links below describe the earlier converted Markdown layout. They are
retained for source comparison, not as the active canonical reading order.

### Core Documentation

1. **[Notation and Symbols](notation.md)**
   Summary of mathematical notation, custom LaTeX commands, and symbol definitions used throughout the documentation.

### Theory Sections

2. **[I. Introduction to Spin Wave Theory](sections/01_spin_wave_theory_intro.md)**
   Foundation of spin wave theory including:
   - Spin Hamiltonian formulation
   - Holstein-Primakoff transformation
   - Bogoliubov transformation
   - Momentum space formulation

3. **[II. Physical Quantities](sections/02_physical_quantities.md)**
   Overview of computable physical quantities in LSWT framework:
   - Thermodynamic quantities (partition function, free energy, entropy, specific heat)
   - Correlation functions (dynamic structure factor, spectral function)
   - Topological properties (Chern number, thermal Hall conductance)

4. **[III. Thermodynamics](sections/03_thermodynamics.md)**
   Detailed derivations of thermodynamic quantities:
   - Partition function
   - Internal energy
   - Free energy
   - Entropy
   - Specific heat
   - Boson number expectation values

5. **[IV. Correlations](sections/04_correlations.md)**
   Spin-spin correlation functions:
   - Real-time dynamical correlations
   - Dynamic structure factor
   - Spectral function
   - Equal-time correlations

6. **[V. Topology](sections/05_topology.md)**
   Topological aspects of magnon systems:
   - Skyrmion number
   - Chern number and Berry curvature
   - Thermal Hall conductance

7. **[VI. Worked Example](sections/06_worked_example.md)**
   Step-by-step example of solving a quadratic boson Hamiltonian using LSWT methods.

## Directory Structure

```
research-space/theory/
├── README.md                    # This gateway
├── lswt/                        # Active LSWT canonicalization workspace
├── common/                      # Provisional cross-solver candidates; accepted 0
├── notation.md                  # Legacy converted notation source
└── sections/                    # Legacy converted sections pending migration
    ├── 01_spin_wave_theory_intro.md
    ├── 02_physical_quantities.md
    ├── 03_thermodynamics.md
    ├── 04_correlations.md
    ├── 05_topology.md
    └── 06_worked_example.md
```

## Historical Conversion Process

The legacy converted Markdown was created through a conversion workflow whose
exact input provenance has not yet been verified. Its headings and section
order align with `legacy/research-notes/lswt/note_lswt_reviewed.tex`, while an
older README record named the restructured TeX as the input.

1. **Conversion source**: Unverified; compare the reviewed and restructured TeX
   before relying on converted text.
2. **Pandoc conversion**: LaTeX → Markdown with equation preservation.
3. **Refinement**: Script processing recorded for:
   - Simplifying equation references
   - Converting custom LaTeX commands (e.g., `\kvec` → `\mathbf{k}`)
   - Cleaning up formatting artifacts
   - Organizing into logical sections

This historical conversion record does not define current evidence authority.
Use the approved routing and active editing authority in
[`lswt/README.md`](lswt/README.md) when sources disagree.

## Related Files

- **Editable transcription**:
  [`../../legacy/research-notes/lswt/note_lswt_reviewed.tex`](../../legacy/research-notes/lswt/note_lswt_reviewed.tex)
- **Structural reference TeX**:
  [`../sources/lswt/note_lswt_restructured.tex`](../sources/lswt/note_lswt_restructured.tex)
- **Approved source authority**: [`lswt/README.md`](lswt/README.md)
- **Decision record**:
  [`../../GOVERNMENT/Court-Precedents/2026-08-01-lswt-markdown-source-authority.md`](../../GOVERNMENT/Court-Precedents/2026-08-01-lswt-markdown-source-authority.md)

## Usage for Developers

When working with the LSWT package:

1. Start from `lswt/map-lswt.md` for the theory-document structure.
2. Check `lswt/current-sections-audit.md` before relying on a draft equation.
3. Use the approved evidence routing in `lswt/README.md`; do not assume the
   restructured TeX, generated PDF, and primary PDF are interchangeable.
4. Treat documentation cleanup and theory-code verification as separate steps.

## Legacy Reader Notes (Unverified)

The material below is retained from the older gateway and has not yet been
checked claim-by-claim against the primary PDF. It is not part of the active
canonical reading path.

### Spin Wave Theory Basics
- Represents quantum spin fluctuations as bosonic excitations (magnons)
- Valid in the low-temperature limit where quantum fluctuations are small
- Provides analytical and numerical access to thermodynamic and dynamical properties

### Main Theoretical Steps
1. **Classical ground state**: Find spin configuration minimizing classical energy
2. **Local frame transformation**: Align local $z$-axis with classical spin direction
3. **Holstein-Primakoff**: Map spin operators to bosonic ladder operators
4. **Fourier transform**: Convert to momentum space
5. **Bogoliubov transformation**: Diagonalize quadratic Hamiltonian
6. **Physical quantities**: Calculate observables from magnon spectrum

### Applicability
- Frustrated magnets (triangular, kagome, honeycomb lattices)
- Magnetic skyrmion systems
- Topological magnon bands

### Draft Citation

If you use these theoretical notes or the LSWT package, please cite:

```bibtex
@misc{lswt_theory,
  author = {Park, Sung-Min},
  title = {Linear Spin Wave Theory: Theoretical Documentation},
  year = {2025},
  howpublished = {\url{https://github.com/...}},
}
```

*(Update with actual publication details when available)*

## 🤝 Contributing

Found an error or typo? Please:
1. Read the approved evidence routing and conflict rules in `lswt/README.md`.
2. Record unresolved source or physical conflicts in
   `lswt/current-sections-audit.md`.
3. Update an active `lswt/` document only after identifying its source and
   review status. Do not rewrite the legacy converted Markdown in place.

## 📧 Contact

**Author**: Sung-Min Park
**Email**: sungmin.park.0226@gmail.com

---

**Last Updated**: 2026-08-01
**Document Status**: Working gateway
