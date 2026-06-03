---
frontmatter-version: 1
title: Naming Convention
section: policies
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-03
---

# Naming Convention

LSWT repo의 파일, 폴더, 운영 문서 이름 규칙이다. 언어·도구가 강제하는 공식 스타일은 이 문서보다 우선한다.

## 기본 원칙

- 이름만으로 독자, 범위, 역할이 드러나야 한다.
- 도메인 없는 범용명은 피한다: `misc`, `notes`, `utils`, `general`, `data`.
- 기존 코드 스타일과 충돌하면 같은 폴더의 관행을 우선한다.

## 케이스 규칙

| 대상 | 규칙 | 예 |
|---|---|---|
| Python 모듈 | snake_case | `spin_system.py` |
| Python 클래스 | PascalCase | `SpinSystem` |
| Python 함수/변수 | snake_case | `add_coupling` |
| 일반 Markdown 파일 | kebab-case | `directory-structure-guide.md` |
| GOVERNMENT 폴더 | AAD convention 유지 | `Working-Pad/`, `Agents-Bylaws/` |
| map 문서 | `map-*.md` | `map-working-pad.md` |
| 작업 문서 | `YYMMDD-{topic}.md` 또는 `YYYY-MM-DD-{topic}.md` | `260603-general-2d-spin-tool.md` |

## GOVERNMENT 이름

- `issue-notes/`: 열린 문제, 리뷰, 논의 기록.
- `vault-staging/`: 정본 레이어 반영 전 사용자 승인 대기.
- `idea-proposals/`: 방향 제안과 구조 제안.
- `handoff/`: 세션 간 인수인계.
- `inbox/`: 분류 전 임시 캡처.

`issues/`와 `staging/`은 새로 만들지 않는다. 기존 경로 참조가 나오면 각각 `issue-notes/`, `vault-staging/`으로 갱신한다.

## Theory 문서 이름

- LSWT 정본 문서는 `research-space/theory/lswt/` 아래에 둔다.
- 섹션 순서가 중요한 파일은 `00_`, `01_` 같은 numeric prefix를 허용한다.
- migration 또는 audit 문서는 목적을 명확히 쓴다.
  - 좋음: `section-migration-plan.md`
  - 피함: 실제 map이 아닌데 `*-map.md`로 부르는 것

## map guard text

`map-*.md` 파일은 가능하면 H1 아래에 짧은 guard text를 둔다.

```markdown
> 이 파일은 [folder] 디렉토리의 인덱스입니다.
> 새 항목 추가 시 이 표에 한 줄을 추가합니다.
```

## 관련 문서

- `GOVERNMENT/Agents-Bylaws/policies/frontmatter-policy.md`
- `GOVERNMENT/Agents-Bylaws/templates/map-template.md`
