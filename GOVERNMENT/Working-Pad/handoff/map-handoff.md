---
frontmatter-version: 1
template-version: 1
title: Map - handoff
section: handoff
status: in-review
last-edited-by: claude
created: 2026-06-03
updated: 2026-09-30
must-read: GOVERNMENT/Agents-Bylaws/templates/map-template.md
---

# Map - handoff

대화 또는 작업 세션 사이의 인수인계 문서.

- 진행 중인 인수인계는 `open/`에 둔다.
- 완료된 인수인계는 `closed/`로 이동한다.
- 새 대화는 최신 open handoff 문서를 먼저 읽고 시작한다.

## 목차

### `open/` — 진행 중인 handoff

| 파일 | 목적 | 상태 |
|---|---|---|
| [[GOVERNMENT/Working-Pad/handoff/open/260607-solver-seam-spike\|260607-solver-seam-spike]] | XXZ를 ED/DMRG/NQS로 풀어 시스템↔솔버 seam 실측 검증. 세부 계획은 문서 본문에서 관리 | pending |

### `closed/` — 완료된 handoff

| 파일 | 목적 | 결과 |
|---|---|---|
| [[GOVERNMENT/Working-Pad/handoff/closed/260603-next-chat-general-2d-spin-tool\|260603-next-chat-general-2d-spin-tool]] | LSWT를 범용 2D spin-system simulation tool로 전환하기 위한 다음 대화 계획 | superseded; 전달 규약·1차 개발 계획·toolkit 0–5단계로 수행 |
| [[GOVERNMENT/Working-Pad/handoff/closed/260917-codex-to-codex-nbcp-angular-matching\|260917-codex-to-codex-nbcp-angular-matching]] | Y 전체 BZ 안정성 검사 이후 각도별 강성·potential matching 재개 | done; 계산·독립 수치 대조 완료, 사용자 물리 검토 및 후속 결함·thermal 계산은 별도 |

## 에이전트 지침

- 새 handoff 생성 시 `open/`에 파일을 만들고 이 map에 등록한다.
- 작업이 이어받아졌거나 더 이상 유효하지 않으면 `closed/`로 이동하고 map을 갱신한다.
- handoff는 현재 상태, 다음 목표, 알려진 리스크, 첫 실행 순서를 포함해야 한다.

## 참고 문서

- [[GOVERNMENT/Agents-Bylaws/templates/map-template|map-template]] — map 작성 기준
- [[GOVERNMENT/Agents-Bylaws/templates/handoff-template|handoff-template]] — handoff 작성 기준
