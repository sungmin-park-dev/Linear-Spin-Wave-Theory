---
frontmatter-version: 1
title: Frontmatter Policy
section: policies
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-03
---

# Frontmatter Policy

LSWT `GOVERNMENT/` 운영 문서의 frontmatter 기준이다. AAD convention을 LSWT 경로 체계에 맞춘 사본이다.

## 적용 범위

- `GOVERNMENT/**/*.md`
- `research-space/theory/**/*.md` 중 정본 theory 문서로 승격되는 파일

Legacy 또는 원자료 파일은 frontmatter가 없어도 허용한다. 단, 운영 기준 문서로 편집할 때는 frontmatter를 추가한다.

## 공통 필드

| 필드 | 필수 | 설명 |
|---|---|---|
| `frontmatter-version` | 예 | 현재 `1` |
| `title` | 예 | 문서 제목 |
| `section` | 예 | `GOVERNMENT/` 내부 상대 영역. 예: `issue-notes/open`, `templates`, `policies` |
| `status` | 예 | `draft`, `in-review`, `accepted`, `closed` |
| `last-edited-by` | 예 | `user`, `claude`, `codex`, `system`, `unknown` |
| `created` | 예 | `YYYY-MM-DD` |
| `updated` | 권장 | `YYYY-MM-DD` |
| `must-read` | 조건부 | 편집 전 반드시 읽어야 하는 템플릿 또는 정책 |

## 상태 규칙

| 값 | 의미 | 주 사용처 |
|---|---|---|
| `draft` | 작성 중 | Working-Pad 초안 |
| `in-review` | 사용자 검토 필요 | 에이전트가 만든 정책, 템플릿, proposal |
| `accepted` | 사용자 승인됨 | 장기 기준, 확정 map, 결정 기록 |
| `closed` | 종결됨 | 완료된 issue-note 또는 handoff |

AI가 작성하거나 수정한 문서는 스스로 `accepted`로 만들지 않는다. 사용자가 확인한 뒤 `reviewed-by: user`, `reviewed-at: YYYY-MM-DD`를 붙이고 `accepted`로 바꾼다.

## `last-edited-by`

| 값 | 주체 |
|---|---|
| `user` | 사용자 |
| `claude` | Claude |
| `codex` | Codex |
| `system` | 자동화 스크립트 |
| `unknown` | 불명 |

## must-read 예시

```yaml
must-read: GOVERNMENT/Agents-Bylaws/templates/map-template.md
```

```yaml
must-read:
  - GOVERNMENT/Agents-Bylaws/templates/issue-notes-template.md
  - GOVERNMENT/Agents-Bylaws/policies/naming-convention.md
```

## 유형별 최소 예시

### map-*.md

```yaml
---
frontmatter-version: 1
title: Map - [folder-name]
section: [folder-name]
status: in-review
last-edited-by: codex
created: YYYY-MM-DD
updated: YYYY-MM-DD
must-read: GOVERNMENT/Agents-Bylaws/templates/map-template.md
---
```

### issue-notes/open

```yaml
---
frontmatter-version: 1
title: [Issue Title]
section: issue-notes/open
issue-type: problem
status: draft
last-edited-by: codex
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

### vault-staging

```yaml
---
frontmatter-version: 1
title: [Proposal Title]
section: vault-staging
status: in-review
last-edited-by: codex
created: YYYY-MM-DD
updated: YYYY-MM-DD
proposed-target-path: GOVERNMENT/...
---
```

## 관련 문서

- `GOVERNMENT/Agents-Bylaws/templates/map-template.md`
- `GOVERNMENT/Agents-Bylaws/policies/directory-structure-guide.md`
