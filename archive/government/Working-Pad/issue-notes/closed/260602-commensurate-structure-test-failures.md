---
frontmatter-version: 1
title: CommensurateStructure pytest failures
section: issue-notes/closed
issue-type: problem
last-edited-by: claude
status: closed
resolution: resolved
created: 2026-06-02
updated: 2026-09-30
closed: 2026-09-30
outcome: GOVERNMENT/Working-Pad/issue-notes/open/260930-lt-step-and-commensurate-structure-decision.md
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

## 해결 (2026-09-30)

사용자 승인 후 1번 모델로 해결했다. 클래스 코드는 바꾸지 않았다.

- **판정**: 클래스의 부격자 수 규칙(`num_basis_sites * n1 * n2`)은 인덱싱, 직렬화와
  통과하던 14개 테스트와 일관된다. 실패한 5개 테스트가 `(1, 1)` 셀에 각도 2–3개를 넣은
  것이 잘못이었다. 실패는 파일을 만든 `6a991a9`(Phase 1A)부터 있었다.
- **120° 물리**: 대각 초격자만 표현하므로 한 사이트 삼각격자의 √3×√3 셀은 담을 수 없다.
  3×1 셀은 `n2 mod 1`로 접혀 a₂ 방향 스핀이 평행하다(최근접 내적 {−0.5, 1.0}).
  3×3 대각 셀(부격자 9개, φ = 2π(n1 − n2)/3)은 모든 최근접 내적이 −0.5다.
- **테스트 수정** (`code-space/tests/test_states/test_commensurate.py`):
  - 120° 테스트를 두 표현으로 매개변수화했다: 3-site basis(`(1, 1)`, NBCP식 우회)와
    3×3 대각 셀.
  - 3×3 셀의 최근접 120° 확인과 3×1 셀이 120°가 아님을 확인하는 테스트를 추가했다.
  - optimization 테스트 두 개는 셀을 `(3, 1)`, `(2, 1)`로 바로잡았다.
- **docstring**: `commensurate.py`의 120° 예시를 3-site basis로 고치고, 대각 셀의 한계와
  정수 행렬 초격자를 쓰는 `SpinState`를 안내했다.
- **결과**: 이 파일 24 통과; 전체 `code-space/tests` 473 통과 / 0 실패(변경 전 463 / 5).

`closed/`로의 이동과 `map-issue-notes.md`·`TASK-QUEUE.md` 동기화는 보류했다. 두 파일은
기본 저장소에서 진행 중인 미커밋 구조 개편이 수정하고 있어, 그 작업이 커밋된 뒤 옮긴다.

후속 판단(LT 단계 구현, `CommensurateStructure` 정리)은
`issue-notes/open/260930-lt-step-and-commensurate-structure-decision.md`로 분리했다.
