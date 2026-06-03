---
frontmatter-version: 1
title: Template — idea-proposals
section: templates
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-03
---

# Template — idea-proposals

`GOVERNMENT/Working-Pad/idea-proposals/` 파일에 사용하는 템플릿.

> 아직 논의·검토 중인 아이디어·설계 제안에 사용한다.
> 내용이 확정되면 `vault-staging/`으로 이동. 문제·이슈 논의라면 `issue-notes/open/` 사용.

---

## 디렉토리 구조

```
GOVERNMENT/Working-Pad/idea-proposals/
  map-idea-proposals.md
  YYYY-MM-DD-[topic].md 또는 YYMMDD-[topic].md
```

---

## 라이프사이클

| 상태 | 의미 |
|---|---|
| `draft` | 작성 중 |
| `in-review` | 사용자 검토 중 |

채택 결정 → 구체화 후 `vault-staging/`으로 이동.
드롭/거부 → 삭제 (git 히스토리 보존).

---

## 파일명 규칙

```
YYYY-MM-DD-[brief-kebab-topic].md
```

---

## frontmatter

```yaml
---
frontmatter-version: 1
title: "[제안 제목]"
section: idea-proposals
status: draft                       # draft | in-review
author: claude                      # claude | codex | user
last-edited-by: claude
created: YYYY-MM-DD
# 선택:
# confidence: high | medium | low
# source_refs:
#   - conversation: [날짜 및 맥락 요약]
---
```

---

## 본문 구조

```markdown
# [제안 제목]

## Summary

[한 문단. 무엇을 왜 제안하는가.]

## Proposal

[제안 내용. 아이디어 수준이면 방향성, 구체적이면 Before/After 형식.]

## Rationale

[왜 이 방향인가. 고려한 대안이 있다면 포함.]

## Open Questions

[아직 결론 나지 않은 질문들. 없으면 섹션 생략.]
```

---

## 생성 후 처리

1. 파일 위치: `GOVERNMENT/Working-Pad/idea-proposals/`
2. `map-idea-proposals.md` 해당 토픽 표에 한 줄 추가.
3. `TASK-QUEUE.md` 현재 작업 테이블에 행 추가 (유형: `idea`).

## 채택 결정 시 처리

1. 내용을 구체화하여 `GOVERNMENT/Working-Pad/vault-staging/YYYY-MM-DD-[topic].md` 생성.
2. `map-idea-proposals.md` 표에서 행 삭제.
3. `TASK-QUEUE.md`에서 이 파일 행 삭제 (vault-staging 행으로 교체).
4. 이 파일 삭제.
