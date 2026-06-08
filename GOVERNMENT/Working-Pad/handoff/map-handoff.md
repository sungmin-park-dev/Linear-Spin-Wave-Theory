---
frontmatter-version: 1
title: Map - handoff
section: handoff
status: in-review
last-edited-by: claude
created: 2026-06-03
updated: 2026-06-07
---

# Map - handoff

대화 또는 작업 세션 사이의 인수인계 문서.

- 진행 중인 인수인계는 `open/`에 둔다.
- 완료된 인수인계는 `closed/`로 이동한다.
- 새 대화는 최신 open handoff 문서를 먼저 읽고 시작한다.

## Open

| 파일 | 목적 | 상태 |
|---|---|---|
| `open/260603-next-chat-general-2d-spin-tool.md` | LSWT를 범용 2D spin-system simulation tool로 전환하기 위한 다음 대화 계획 | pending |
| `open/260607-solver-seam-spike.md` | XXZ를 ED/DMRG/NQS로 풀어 시스템↔솔버 seam 실측 검증 — Tensor-Network-Study의 Cluster_Ising 하네스 재사용(코드는 TN-study), seam findings·Bethe 노트는 LSWT research-space로 (codex) | pending |

## Closed

| 파일 | 목적 | 결과 |
|---|---|---|
| _없음_ | - | - |

## Agent Instructions

- 새 handoff 생성 시 `open/`에 파일을 만들고 이 map에 등록한다.
- 작업이 이어받아졌거나 더 이상 유효하지 않으면 `closed/`로 이동하고 map을 갱신한다.
- handoff는 현재 상태, 다음 목표, 알려진 리스크, 첫 실행 순서를 포함해야 한다.
