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
| 1 | issue | `docs/lswt/` LSWT 이론 문서 — 다음: local circular component 및 real-space H2 내용 검토 | in-review; 17개 문서 일괄 문체 교정 반영, 10개 skeleton 내용 보완과 사용자 검토 대기 | `issue-notes/open/260809-lswt-documentation-audit.md` |
| 2 | handoff | 솔버 seam 스파이크 (ED/TN/NQS로 XXZ 풀어 실측 검증) | pending — TN-Study에 findings draft v1(1D, Heisenberg점) 있음, 2D 4×4 단계 미완료라 LSWT 승격 전 | `handoff/open/260607-solver-seam-spike.md` |
| 3 | handoff | 범용 2D spin-system simulation tool 전환 작업 | 2단계(code-space audit) 완료, 3–4단계 대기 | `handoff/open/260603-next-chat-general-2d-spin-tool.md` |
| 4 | issue | code-space audit — 범용 2D spin-tool 대상 분류 | draft | `issue-notes/open/260802-code-space-audit-general-2d-spin-tool.md` |
| 5 | issue | `CommensurateStructure` pytest 실패 | draft | `issue-notes/open/260602-commensurate-structure-test-failures.md` |
| 6 | issue | Thermal Hall 면적·단위·반환 기준 | in-review; 조절 가능한 고정 cutoff·비용 확인, 107개 회귀 통과; 다음: NBCP cutoff·mesh 수렴 | `issue-notes/open/260802-topology-thermal-hall-real-space-volume-bug.md` |
| 7 | issue | NBCP 연구노트와 arXiv 열적 주장 검토 | in-review; Y angular·smooth-wave와 density wall 164개 local minima 대조 완료; 벽 폭 3–4a, 장력 양수·크기 수렴 및 metastability 확인; 다음: vortex core·wall 결합, 이후 thermal 검증; quantum/thermal matching 미완료 | `issue-notes/open/260810-pseudo-goldstone-gap.md` |
| 8 | issue | NBCP band plot와 LSWT 구현 backlog | draft; 우선순위 미정 | `issue-notes/open/260810-lswt-implementation-backlog.md` |
| 9 | idea | 범용 2D 스핀 시스템 도구 전환 노트 | draft | `idea-proposals/260603-general-2d-spin-tool-migration-note.md` |
| 10 | idea | 공통 SpinModel IR 표현법 | draft | `idea-proposals/2026-06-04-general-spin-model-ir.md` |
| 11 | idea | Project Knowledge Philosophy 사본 | draft | `idea-proposals/2026-05-30-project-knowledge-philosophy.md` |
| 12 | issue | NBCP 문서 구조와 코드 물리 구현 검토 | in-review; 9장·4부록 LaTeX 원본 전환 및 claim-to-code 대응 작성; 다음: displacement convention과 독립 bond-count 검토 | `issue-notes/open/260918-nbcp-physics-code-review.md` |
| 13 | idea | 모델 간 공통 전달 규약 — 필드·단위·검증·NBCP 변위 대응 | in-review; 공통 런타임 API 미구현 | `idea-proposals/2026-09-23-spin-model-transfer-contract.md` |
| 14 | development | 2D Spin-System Toolkit 개발 설계와 기능별 폴더 구성 | 기본 저장소 통합; 로컬 Beamer 검토 대기 | `../../docs/development/README.md` |

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
