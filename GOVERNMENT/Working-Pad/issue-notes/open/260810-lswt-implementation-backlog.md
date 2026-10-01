---
frontmatter-version: 1
title: LSWT implementation backlog migrated from AGENTS
section: issue-notes/open
issue-type: review
status: draft
last-edited-by: claude
created: 2026-08-10
updated: 2026-09-30
related:
  - GOVERNMENT/Working-Pad/issue-notes/closed/260802-code-space-audit-general-2d-spin-tool.md
  - GOVERNMENT/Working-Pad/issue-notes/closed/260810-agents-progress-history.md
must-read: GOVERNMENT/Agents-Bylaws/templates/issue-notes-template.md
---

# LSWT implementation backlog migrated from AGENTS

## 배경

`AGENTS.md`에 들어 있던 NBCP band-plot 진행 상태와 일반 구현 TODO는 에이전트 지침이 아니라 변하는 작업 상태다. 이 기록은 해당 항목을 Working-Pad로 옮기고, `TASK-QUEUE.md`가 활성 작업의 단일 인덱스가 되도록 하기 위해 작성했다.

## 문제 정의

여러 구현 과제가 완료 이력과 함께 `AGENTS.md`에 섞여 있어 우선순위, owner issue와 검증 경계를 추적하기 어려웠다. 아래 항목의 우선순위와 구체적인 착수 순서는 아직 결정되지 않았다.

## 본론

### 리뷰 대상 현황

NBCP band-plot 예제의 선행 작업 중 classical ground-state 최적화와 spin-configuration visualizer는 완료된 것으로 기록되어 있다. 2026-09-30 기준으로 LSWT Hamiltonian은 toolkit 2단계(기존 builder와 원소 단위 일치)와 3단계(ED 1-마그논과 일치), Colpa 대각화는 4a단계에서 검증되었다. 남은 단계는 band plot 작성과 결과 검토다.

그 밖에 `AGENTS.md`에서 이동한 미완료 항목은 다음과 같다.

| 항목 | 현재 기록 | 관련 owner 또는 검토 경계 |
|---|---|---|
| Band plotter와 interactive exchange viewer 포팅 | 미구현 | visualization backlog |
| 공개 배포 정리 | `.gitignore`, README 등이 미완료로 기록 | theory/code 검증과 별도 상태로 관리 |
| Real-space BdG solver | 장기 과제 | `AbstractSolver` 인터페이스 검토 |
| `IncommensurateStructure` | 2026-10-01 단일 Q 나선과 회전틀 LSWT 구현(D34); 다중 Q·비 U(1) 모델·나선 위상량은 후속 | `idea-proposals/2026-10-01-spiral-rotating-frame-lswt.md`, 사용자 물리 검토 |

2026-09-30 정리(사용자 승인): toolkit 0–5단계로 끝난 항목을 표에서 뺐다. `EnergyFunction`의 입력 형식과
observables–솔버 연결은 공통 경로(`classical_energy(model, state)`, `LSWTResult`를 쓰는 물리량)로 대체되었고,
`hamiltonian_at`은 `LSWTResult.hamiltonian_at`, k-data의 dataclass 전환은 `LSWTResult`(D24)로 구현되었다.
격자·자기 구조와 시스템 연결은 `260930-lt-step-and-commensurate-structure-decision.md`로 옮겼고, pytest 기준은
전체 통과(500개)다. 기존 경로의 정리 시점은 `docs/development` Beamer의 미결 사항에서 추적한다.

### 발견 사항

- 2026-10-01 NBCP 3부분격자 상(0.7, 1.0, 1.4 T ∥ b*, arXiv:2505.06398 Table 1·Fig. 6의 자기장) 밴드
  (`examples/nbcp_three_sublattice_bands.py`, `data-space/verification/261001-nbcp-three-sublattice-bands/`,
  `test_nbcp_three_sublattice_bands.py`).
  - **고전 기준 상태.** √3×√3 셀에서 무작위 시작 30개를 정련하면 두 정류 상태가 나온다. 상태 I은 스핀 하나가 B를 따르고
    두 스핀이 b*-c 평면에서 +c, −c 쪽으로 대칭 기운 상태다. 기울기는 닫힌 식 `cos beta = (h/S - 3 Jxy) / (3 (Jxy + Jz))`와
    1e-8 안에서 맞고, beta → 0이 편극상의 고전 임계장 1.717 T와 같다. 상태 II는 두 스핀이 같고 하나가 다르며, 모든 장에서
    상태 I보다 높고(1.5 T에서 셀당 1e-5 meV) LSWT가 불안정한 안장점이다.
  - **선택의 근거와 한계.** LSWT 기준은 상태 I이다. 두 상태의 에너지 차가 작으므로 LSWT를 넘는 양자 보정이 선택을 바꿀 수
    있는지는 열린 질문이다. 논문은 양자 요동이 이 구조를 고른다고 서술한다(원문 미대조, 요약 기반).
  - **안정성과 밴드.** 장이 b*를 따르면 연속 대칭이 없다. 고전 Hessian에 0 고유값이 없고(0.25 T에서 최소 1.7e-3), LSWT에
    영모드도 없다. 경로 M-K'-Γ-K-M에서 세 모드가 모두 안정하고, K가 자기 BZ의 Γ로 접힌다. 최저 갭은 0.7, 1.0, 1.4 T에서
    0.041, 0.031, 0.015 meV이고, B → B_C^cl 아래에서 연속으로 닫힌다(1.71 T에서 3e-4 meV). 이는 편극상 갭이 위에서 닫히는 것과
    맞는다.
  - **미완료.** 측정 데이터와의 비교는 하지 않았다. 논문의 측정 B_C ≈ 1.65 T와 고전값의 차이는 LSWT 밖의 효과다.
- 2026-09-11 후속: 스핀당 U/S/C와 sublattice 점유수의 정규화, 통합·온도 스캔의 Ns 추론을 신규 21개 테스트로 검증·수정했다. [정규화 종결 이슈](../closed/260911-thermodynamic-observable-normalization.md)에 해석해·legacy 대조를 기록하고 완료 항목을 표에서 제거했다. Thermal Hall, 적용되지 않는 `exclude_gamma` 및 Goldstone/invalid-mode 처방은 별도 검토 범위다.
- 2026-09-11 온도 전달, thermal energy의 band 합, free-energy logarithm 결함은 신규 17개 회귀 테스트로 수정·종결했다. [해당 이슈](../closed/260911-finite-temperature-energy-and-occupation.md)에 legacy 대조와 검증 범위를 기록했다.
- 2026-09-10 B/B† 구현 회귀 검증과 공개 솔버의 T=0 에너지 반환식 수정은 각각 종결 이슈에 기록했다. 이는 전체 Hamiltonian, Colpa 또는 유한온도 검증의 완료를 뜻하지 않는다.
- 일부 항목은 이미 code-space audit에 상세히 기록되어 있으므로 이 문서에서 내용을 중복 확정하지 않는다.
- “합의 완료”라고 적힌 API 항목도 승인 근거와 현재 코드 상태를 다시 확인한 뒤 착수해야 한다.
- NBCP band plot은 Hamiltonian과 diagonalization 검증보다 먼저 완료 상태로 만들 수 없다.
- Theory acceptance, code verification, example completion과 public release는 서로 다른 상태다.

### 권고

1. 사용자가 다음 구현 단위를 선택하면 해당 항목을 독립 issue 또는 handoff로 분리한다.
2. 구현 전 관련 theory owner와 기존 code audit을 다시 확인한다.
3. 구현, legacy 또는 독립 benchmark 검증, 사용자 확인을 같은 issue에서 추적한다.
4. 완료된 항목은 이 backlog에서 제거하고 해당 issue를 closed로 이동한다.

## 결론 / 미결 사항

표에 남은 항목은 우선순위 미정의 open backlog다. 최초 이동(2026-08-10)에서는 코드를 변경하거나 과거의 “완료/합의” 기록을 새로 검증하지 않았다. 2026-09-10~11 후속 구현 검증은 종결 이슈에서 추적하며, 여기에는 아직 남은 항목을 둔다.

## 참조

- `GOVERNMENT/Working-Pad/TASK-QUEUE.md` — 활성 작업 인덱스
- `GOVERNMENT/Working-Pad/issue-notes/closed/260802-code-space-audit-general-2d-spin-tool.md` — 현재 code-space 구조 진단
- `GOVERNMENT/Working-Pad/issue-notes/open/260602-commensurate-structure-test-failures.md` — magnetic-structure 검증 문제
- [T=0 energy assembly 수정 및 유한온도 검토 경계](../closed/260910-zero-point-energy-normalization.md)
- [유한온도 U/F 및 점유수 수정과 남은 정규화](../closed/260911-finite-temperature-energy-and-occupation.md)
- [열역학 observables 정규화 종결](../closed/260911-thermodynamic-observable-normalization.md)
- `examples/` — NBCP 실행 예제
- `code-space/spintoolkit/` — 현재 구현(2026-09-29 `lswt`에서 이름 변경)
