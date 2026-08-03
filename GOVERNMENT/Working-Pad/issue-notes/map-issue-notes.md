---
frontmatter-version: 1
template-version: 1
title: Map - issue-notes
section: issue-notes
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-30
must-read: GOVERNMENT/Agents-Bylaws/templates/map-template.md
---

# Map - issue-notes

LSWT 프로젝트의 미해결/종결 이슈와 논의 기록.

- 활성 이슈는 `open/`에 둔다.
- 해결된 이슈는 `closed/`로 이동하고 이 map의 종결 이슈 표에 기록한다.
- 문제·검증 실패·리뷰·열린 논의는 여기 기록한다. 새 방향 제안은 `idea-proposals/`에 둔다.

파일명 형식: `YYMMDD-{topic}.md`

## 목차

### `open/` — 열린 이슈

| 파일 | 유형 | 역할 | 상태 |
|---|---|---|---|
| [[GOVERNMENT/Working-Pad/issue-notes/open/260602-commensurate-structure-test-failures\|260602-commensurate-structure-test-failures]] | problem | `CommensurateStructure` pytest 실패 기록 | draft |

### `closed/` — 종결 이슈

| 파일 | 유형 | 역할 | 결과 |
|---|---|---|---|
| [[GOVERNMENT/Working-Pad/issue-notes/closed/260604-lswt-section-migration-record\|260604-lswt-section-migration-record]] | migration-record | 기존 Markdown/LaTeX source에서 새 LSWT 정본 파일로 가는 mapping 기록 | implemented |

## 에이전트 지침

- 새 이슈는 `open/`에 생성하고 열린 이슈 표와 `../TASK-QUEUE.md`에 등록한다.
- 이슈가 해결되면 파일 frontmatter `status`를 갱신하고 `closed/`로 이동한다.
- 해결 과정에서 재사용 가능한 절차가 생기면 사용자 확인 후 `GOVERNMENT/Agents-Bylaws/` 승격 후보로 기록한다.

## 참고 문서

- [[GOVERNMENT/Agents-Bylaws/templates/map-template|map-template]] — map 작성 기준
- [[GOVERNMENT/Agents-Bylaws/templates/issue-notes-template|issue-notes-template]] — issue note 작성 기준
- [[GOVERNMENT/Working-Pad/TASK-QUEUE|TASK-QUEUE]] — 활성 작업 큐
