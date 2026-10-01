---
frontmatter-version: 1
title: Directory Structure Guide
section: policies
status: in-review
last-edited-by: claude
created: 2026-06-03
updated: 2026-10-01
---

# Directory Structure Guide

LSWT 프로젝트의 지식·코드·운영 문서를 어디에 둘지 판단하는 기준이다.

## 프로젝트 콘텐츠 루트

| 폴더 | 역할 | 편집 경계 |
|---|---|---|
| `docs/` | LSWT와 NBCP 문서의 통합 진입점 | `docs/README.md`에서 주제와 파일 역할을 안내 |
| `docs/development/` | Toolkit 개발 목표와 시스템 설계 | `main.tex`·`sections/`·`appendices/`가 편집 원본; 구현·검증 기록은 부록 |
| `docs/lswt/` | LSWT 일반 이론 Markdown | 번호 폴더의 사용자 승인 본문만 theory canon; 읽기 순서는 `docs/lswt/README.md` 소유 |
| `docs/nbcp/` | NBCP 연구 노트와 원문 대조 근거 | `main.tex`와 장·부록별 TeX에서 집필; Markdown은 navigation·검토 기록 |
| `docs/archive/` | 집필을 종료한 과거 문서 | 현재 편집 원본으로 사용하지 않음; 기존 `legacy/` 보존 자료는 이동하지 않음 |
| `docs/nbcp/output/` | 생성 PDF와 생성 기록 | 원본에서 재생성; 중간 TeX·Markdown·그림 사본은 임시 파일 |
| `examples/` | 실행 가능한 Python 예제와 예제 자산 | 이론 설명과 분리하고 코드 검증 상태를 따로 기록 |
| `docs/lswt/sources/` | 현재 이론 작업에서 참조하는 PDF·TeX 등 원자료 | Evidence/reference이며 직접 theory canon이 되지 않음 |
| `code-space/` | Python 패키지와 테스트 | Theory acceptance와 별도로 검증 |
| `legacy/` | 과거 코드와 연구 노트의 보존 영역 | 현재 정본이 아니며 출처·누락 대조에만 사용 |
| `model/<name>/` | 모델별 물리 정의·계산과 원시·중간 결과 | 공통 계산법은 `code-space/spintoolkit/`에 두고, 정돈된 결과만 `data-space/`로 승격 |
| `data-space/` | 검토하고 정돈한 계산 결과 데이터 | 문서나 코드의 source of truth로 사용하지 않음 |
| `workbench/` | research-workspace 앱이 관리하는 NBCP 유도·계산 과정 블록(`blocks/`)과 일지(`log/`) (2026-09-30 사용자 승인) | 과정 기록이며 NBCP 내용 원본이 아님. 결론은 사람이 `docs/nbcp/` 해당 장·부록에 옮겨 적는다(2026-09-18 NBCP LaTeX 원본 결정 유지). 컴파일 부산물 `.build/`는 git 제외. `STATUS.md`는 앱이 `research.yaml`의 `sources:`에 적힌 정본(TASK-QUEUE·검토 상태·참고문헌)과 블록·일지를 모아 쓰는 자동 요약으로 정본이 아니며 git 제외(2026-10-01 사용자 승인) |

`docs/`의 1단계는 주제(`development`, `lswt`, `nbcp`)와 보관 역할(`archive`)로 나눈다.
`docs/lswt/` 안에서는 `00-`, `01-`처럼 숫자 prefix로 큰 읽기 순서를 표현한다. 개별 파일명에는 숫자 prefix를 반복하지 않는다. 상세 규칙은
`naming-convention.md`를 따른다.

현재 경로는 사용자 승인 [2026-09-16 문서 통합 기록](../../Working-Pad/issue-notes/closed/260916-docs-topic-consolidation.md)을 따른다. 이전 결정의 경로 표기는 당시 기록으로 보존한다.

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
