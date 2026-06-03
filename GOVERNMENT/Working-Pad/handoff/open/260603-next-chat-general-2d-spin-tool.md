---
frontmatter-version: 1
title: Next Chat - General 2D Spin-System Simulation Tool
section: handoff/open
status: draft
execution-status: pending
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-03
branch: migrate/government-structure
---

# Next Chat - General 2D Spin-System Simulation Tool

## Current State

- Branch: `migrate/government-structure`
- Latest committed baseline before this handoff: `8e88170 Create full government layer maps`
- Root structure is now organized as:
  - `GOVERNMENT/`
  - `code-space/`
  - `doc-space/`
  - `research-space/`
  - `data-space/`
  - `legacy/`
- `GOVERNMENT/` has full layer directories:
  - `User-Constitution/`
  - `Court-Precedents/`
  - `Agents-Bylaws/`
  - `Working-Pad/`
- AAD-style map files and Working-Pad lifecycle folders exist for current `GOVERNMENT` navigation.

## User Direction

The project should move from a narrow LSWT package toward a general-purpose simulation tool for two-dimensional spin systems.

Working interpretation:

- Keep LSWT as one solver capability, not the entire product identity.
- Generalize the domain model around 2D lattices, spin configurations, interactions, and solver workflows.
- Preserve physics correctness over UI/API convenience.
- Do not make major API or physics-semantics decisions without user confirmation.

## Known Open Issue

- `GOVERNMENT/Working-Pad/issue-notes/open/260602-commensurate-structure-test-failures.md`
  - `python -m pytest code-space/tests` currently has `CommensurateStructure` failures.
  - This should be handled before treating the test suite as reliable.

## Next Conversation Goal

Establish a concrete refactor plan for turning the codebase into a general 2D spin-system simulation tool.

The next conversation should start with a codebase audit, not immediate large refactors.

## Proposed Work Plan

### 1. Define Scope and Product Boundary

Clarify what "general 2D spin-system simulation tool" means for the first stable target:

- Supported lattice types
- Supported spin models and interactions
- Classical vs quantum workflows
- Required solvers
- Expected outputs: spectra, thermodynamics, correlations, topology, visualization, data export

Deliverable: short scope document under `GOVERNMENT/Working-Pad/idea-proposals/` or a user-approved `User-Constitution/` definition.

### 2. Audit Current Code Against Target

Inspect:

- `code-space/lswt/core/`
- `code-space/lswt/solvers/`
- `code-space/lswt/observables/`
- `doc-space/examples/`
- `legacy/`

Classify current modules as:

- Keep as general core
- Keep as LSWT-specific
- Move/rename
- Deprecate
- Needs physics review

Deliverable: audit issue or plan in `GOVERNMENT/Working-Pad/issue-notes/open/`.

### 3. Resolve Test Baseline

Handle the current `CommensurateStructure` failure before broad changes.

Required decision:

- Should magnetic sublattices be inferred from `angles`, or fixed by basis and supercell?
- How should 120-degree order be represented in the public API?

Deliverable: passing baseline tests or a user-approved test update plan.

### 4. Propose Architecture

Likely target separation:

- Generic 2D spin-system domain model
- Model/interactions layer
- Solver interfaces
- LSWT solver implementation
- Observables layer
- Visualization and examples

No package rename should be done before user approval.

### 5. Implement in Small Commits

After the plan is accepted:

- Fix test baseline
- Separate generic core from LSWT-specific code
- Update examples
- Update docs and maps
- Run tests and import checks after each step

## First Commands for Next Chat

```bash
git status --short --branch
find GOVERNMENT -maxdepth 5 -print | sort
python -m pytest code-space/tests
```

Expected pytest result at handoff time: `CommensurateStructure` failures remain unless fixed in a later commit.

## Do Not Do Without User Confirmation

- Rename the package from `lswt`.
- Remove legacy code.
- Redefine the physical meaning of magnetic supercells or sublattices.
- Treat `theory_code_verification_plan.md` as fully current; it is in `Agents-Bylaws/procedures/` but marked `needs-review`.
