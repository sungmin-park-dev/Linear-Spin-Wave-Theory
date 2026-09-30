---
frontmatter-version: 1
title: Task Queue
section: working-pad
status: in-review
last-edited-by: claude
created: 2026-06-03
updated: 2026-09-30
---

# Task Queue

> 이 파일은 LSWT `GOVERNMENT/Working-Pad/`의 활성 작업 인덱스다.
> 상세 내용은 링크된 파일에 둔다.

## 현재 작업

> **2026-09-30 전체 멈춤(사용자 결정):** 10–11월 지원 기간에 하루 연구 시간(5시간)을 Emergence EB 대표작(3시간, v1 12/8)과 TN+NQS 한 트랙(2시간)에 모으기 위해, LSWT 작업은 EB v1 이후 또는 2026-10-10 월간 검토까지 멈춘다. 재개 시 아래 순위를 그대로 이어간다.

| 순위 | 유형 | 내용 | 상태 | 파일 |
|---|---|---|---|---|
| 1 | handoff | 솔버 seam 스파이크 (ED/TN/NQS로 XXZ 풀어 실측 검증) | pending — TN-Study에 findings draft v1(1D, Heisenberg점) 있음, 2D 4×4 단계 미완료라 LSWT 승격 전 | `handoff/open/260607-solver-seam-spike.md` |
| 2 | handoff | 범용 2D spin-system simulation tool 전환 작업 | 2단계(code-space audit) 완료, 3–4단계 대기 | `handoff/open/260603-next-chat-general-2d-spin-tool.md` |
| 3 | issue | code-space audit — 범용 2D spin-tool 대상 분류 | draft | `issue-notes/open/260802-code-space-audit-general-2d-spin-tool.md` |
| 4 | issue | `CommensurateStructure` pytest 실패 | draft | `issue-notes/open/260602-commensurate-structure-test-failures.md` |
| 5 | issue | `LSWTHamiltonian` B/B† 블록 치환 오류 | draft | `issue-notes/open/260802-hamiltonian-b-block-substitution-bug.md` |
| 6 | issue | Thermal Hall `real_space_volume` 단위/계수 오류 | draft | `issue-notes/open/260802-topology-thermal-hall-real-space-volume-bug.md` |
| 7 | idea | 범용 2D 스핀 시스템 도구 전환 노트 | draft | `idea-proposals/260603-general-2d-spin-tool-migration-note.md` |
| 8 | idea | 공통 SpinModel IR 표현법 | draft | `idea-proposals/2026-06-04-general-spin-model-ir.md` |
| 9 | idea | Project Knowledge Philosophy 사본 | draft | `idea-proposals/2026-05-30-project-knowledge-philosophy.md` |

## 동기화 규칙

- `handoff/open/`, `issue-notes/open/`, `vault-staging/`, `idea-proposals/`에 활성 항목을 추가하면 이 표에도 등록한다.
- 항목이 완료되거나 닫히면 이 표에서 제거하고 해당 파일을 `closed/`로 옮기거나 정본 위치로 승격한다.
- 이 표는 상세 내용을 반복하지 않는다. 상세 맥락은 대상 파일에 둔다.

## 읽기 순서

1. 이 파일
2. `handoff/open/`
3. `issue-notes/open/`
4. `idea-proposals/`
5. `vault-staging/`
