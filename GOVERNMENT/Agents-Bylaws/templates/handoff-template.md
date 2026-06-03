---
frontmatter-version: 1
title: Template — handoff
section: templates
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-03
---

# Template — handoff

`GOVERNMENT/Working-Pad/handoff/` 하위 파일에 사용하는 템플릿.
상세 운영 기준은 `GOVERNMENT/Agents-Bylaws/policies/directory-structure-guide.md`를 참조한다.

---

## 디렉토리 구조

```
GOVERNMENT/Working-Pad/handoff/
  open/      ← execution-status: pending | in-progress
  closed/    ← execution-status: done | skipped
```

새 핸드오프는 `open/` 에 생성. 완료 후 `closed/` 로 이동.

---

## 파일명 규칙

```
YYMMDD-[from]-to-[to]-[topic].md
```

| 세그먼트 | 규칙 |
|---|---|
| `YYMMDD` | 작성 날짜 (예: `260521`) |
| `from` | 작성 주체: `claude` \| `codex` \| `user` |
| `to` | 수신 주체: `claude` \| `codex` \| `user` |
| `topic` | 주된 작업 내용을 2–4단어 kebab-case |

예시: `260521-claude-to-codex-checker-update.md`

---

## frontmatter

```yaml
---
title: "Handoff — [작업 제목]"
section: handoff
status: draft
from: claude          # claude | codex | user
to: codex             # claude | codex | user
execution-status: pending   # pending | in-progress | done | skipped
last-edited-by: claude
created: YYYY-MM-DD
---
```

> `frontmatter-version` 생략 가능. `status: draft` 고정. `execution-status` 가 `done` 또는 `skipped` 이면 `closed/` 로 이동한다.

---

## 본문 구조

```markdown
# Handoff — [작업 제목]

## 이번 세션 완료 사항

### 1. [작업 그룹 제목]

- [완료된 항목 — 파일 경로 포함]
- [완료된 항목]

### 2. [작업 그룹 제목]

- …

---

## 다음 세션 작업 목록

### A. [작업 제목]

**작업 내용**: [무엇을 해야 하는가]

**대상 파일**: `경로/파일명`

**참고**: [명세 위치, 관련 결정 등]

### B. [작업 제목]

…

---

## 참고 문서

- `[경로]` — [한 줄 설명]
```

---

## 작성 규칙 요약

- **완료 사항**: 파일 경로를 포함해 구체적으로. 완료 못 한 항목은 다음 세션 목록으로.
- **다음 세션 목록**: 각 항목은 이전 대화 맥락 없이 독립 실행 가능해야 한다. 의존 관계 명시.
- **참고 문서**: 오리엔테이션에 필요한 문서만. `AGENTS.md` 등 항상 로드되는 파일은 생략.

---

## 생성 후 처리

1. 파일 위치: `GOVERNMENT/Working-Pad/handoff/open/`
2. `GOVERNMENT/Working-Pad/TASK-QUEUE.md` 현재 작업 테이블에 행 추가.

## 완료 처리

1. frontmatter: `execution-status: done` (또는 `skipped`).
2. 파일을 `open/` → `closed/` 로 이동.
3. `GOVERNMENT/Working-Pad/TASK-QUEUE.md`에서 해당 행 삭제.
