---
frontmatter-version: 1
title: Directory Structure Guide
section: policies
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-03
---

# Directory Structure Guide

LSWT 프로젝트에서 `GOVERNMENT/` 운영 문서를 어디에 둘지 판단하는 기준이다.

## 레이어 개요

| 레이어 | 폴더 | 역할 | 에이전트 쓰기 권한 |
|---|---|---|---|
| 1 | `GOVERNMENT/User-Constitution/` | 장기 원칙, 프로젝트 정체성, 보호해야 할 정의 | 직접 수정 금지. `vault-staging/` 경유 |
| 2 | `GOVERNMENT/Court-Precedents/` | 사용자 승인된 결정 기록 | 직접 수정 금지. `vault-staging/` 경유 |
| 3 | `GOVERNMENT/Agents-Bylaws/` | 에이전트 절차, 정책, 템플릿 | 사용자 승인 범위 안에서 유지 |
| 4 | `GOVERNMENT/Working-Pad/` | 진행 중 작업, 논의, 임시 캡처 | 자유롭게 작성하되 map 갱신 |

## 배치 기준

**User-Constitution에 넣는 경우**

- 프로젝트 정체성, 장기 범위, 보호해야 할 원칙
- 예: 범용 2D spin-system tool의 제품 정의가 확정된 경우
- 기준: "1년 뒤에도 기준으로 남아야 하는가?"

**Court-Precedents에 넣는 경우**

- 사용자가 승인한 결정 기록
- 예: site/link/magnetic-structure convention의 최종 채택 기록
- 기준: "앞으로 같은 질문이 나오면 이 판단을 재사용해야 하는가?"

**Agents-Bylaws에 넣는 경우**

- 에이전트가 따라야 하는 절차, 정책, 템플릿
- 예: frontmatter policy, naming convention, theory-code verification procedure
- 기준: "작업 방식 자체를 규정하는가?"

**Working-Pad에 넣는 경우**

- 진행 중 이슈와 논의: `issue-notes/`
- 구조·방향 제안: `idea-proposals/`
- 정본 반영 대기: `vault-staging/`
- 분류 전 임시 캡처: `inbox/`
- 대화/작업 인수인계: `handoff/`

## Working-Pad 워크플로우

| 폴더 | 용도 | 종료 방식 |
|---|---|---|
| `handoff/open/` | 다음 세션으로 넘길 활성 작업 | 완료 후 `handoff/closed/` 이동 |
| `issue-notes/open/` | 열린 문제, 논의, 리뷰 | 해결 후 `issue-notes/closed/` 이동 |
| `idea-proposals/` | 아직 채택되지 않은 방향 제안 | 채택 시 `vault-staging/` 또는 정본 문서로 승격 |
| `vault-staging/` | 사용자 승인 후 정본 반영 대기 | 승인 후 대상 파일 반영 및 staging 파일 제거 |
| `inbox/` | 아직 분류하지 않은 임시 캡처 | 적절한 위치로 이동 후 제거 |

## 탐색 파일

- `map-*.md`: 에이전트용 디렉토리 인덱스. 폴더에 파일이 늘어나면 우선 작성한다.
- `README.md`: 사용자 오리엔테이션이 필요한 폴더에 둔다.
- `AGENTS.md`: 해당 하위 폴더에 추가 지침이 필요한 경우에만 둔다.

## 관련 문서

- `GOVERNMENT/Agents-Bylaws/policies/frontmatter-policy.md`
- `GOVERNMENT/Agents-Bylaws/policies/naming-convention.md`
