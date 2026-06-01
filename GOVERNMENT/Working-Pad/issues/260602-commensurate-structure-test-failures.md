---
frontmatter-version: 1
title: CommensurateStructure pytest failures
section: issues
status: open
created: 2026-06-02
detected-during: government-space-migration
---

# CommensurateStructure pytest failures

## Summary

`python -m pytest code-space/tests` currently fails in `code-space/tests/test_core/test_magnetic_structure/test_commensurate.py`.

This is recorded as an existing unresolved code/test issue, not a blocker for the GOVERNMENT + `*-space` structure migration commit.

## Observed Result

- Test run: `34 passed, 5 failed`
- Failing area: `CommensurateStructure`
- Failing tests:
  - `TestCommensurateStructureCreation::test_120_degree_structure`
  - `TestCommensurateStructureSpinDirections::test_120_degree_spins_in_xy_plane`
  - `TestCommensurateStructureSpinDirections::test_120_degree_total_magnetization_zero`
  - `TestCommensurateStructureOptimization::test_get_optimization_parameters`
  - `TestCommensurateStructureOptimization::test_set_optimization_parameters`

## Failure Pattern

The implementation computes:

```text
num_magnetic_sublattices = num_basis_sites * magnetic_supercell[0] * magnetic_supercell[1]
```

For `num_basis_sites=1` and `magnetic_supercell=(1, 1)`, it expects `angles.shape == (1, 2)`.

Several tests pass `(2, 2)` or `(3, 2)` angles for the same `(1, 1)` supercell, so the constructor raises an angle shape mismatch before the assertions run.

## Required Follow-Up

Decide the intended model:

1. If 120-degree order is represented by three magnetic sublattices, update the tests to use a compatible magnetic supercell or basis count.
2. If the API should infer magnetic sublattices from the angle array, update `CommensurateStructure` and document the rule.

Do not resolve this inside the migration commit.
