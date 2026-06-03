---
frontmatter-version: 1
title: Template — vault-staging
section: templates
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-03
---

# Template — vault-staging

`GOVERNMENT/Working-Pad/vault-staging/` 파일에 사용하는 템플릿.

> 내용이 확정됐고 `GOVERNMENT/User-Constitution/`, `GOVERNMENT/Court-Precedents/`, 또는 `GOVERNMENT/Agents-Bylaws/` 반영 대기 중인 것만 여기에 둔다.
> 아직 논의 중이면 `idea-proposals/open/`, 문제·이슈라면 `issue-notes/open/` 사용.

---

## 디렉토리 구조

```
GOVERNMENT/Working-Pad/vault-staging/   ← 플랫 (open/ 서브폴더 없음)
  map-vault-staging.md
  YYYY-MM-DD-[topic].md      ← 승인 대기 항목들
```

---

## 라이프사이클

| 상태 | 의미 |
|---|---|
| `proposed` | 초안 완성, 검토 전 |
| `in-review` | 사용자 검토 중 |

승인 완료 → 대상 파일 반영 후 이 파일 삭제.
거부 → 삭제 (git 히스토리 보존).

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
title: "[제목]"
section: vault-staging
status: draft
proposal-status: proposed           # proposed | in-review
author: claude                      # claude | codex | user
last-edited-by: claude
created: YYYY-MM-DD
proposed-target-path: GOVERNMENT/... # 변경 대상 파일 경로 (필수)
# 선택:
# confidence: high | medium | low
# source_refs:
#   - conversation: [날짜 및 맥락 요약]
---
```

---

## 본문 구조

```markdown
# [제목]

## Summary

[한 문단. 무엇을 왜 반영하는가.]

## Proposed change

**Before**: [기존 상태 또는 `—` (신규)]

**After**: [반영 후 상태]

## Rationale

[왜 이 방향인가.]

## Notes

[구현 시 주의할 점. 없으면 섹션 생략.]
```

---

## 생성 후 처리

1. 파일 위치: `GOVERNMENT/Working-Pad/vault-staging/`
2. `proposed-target-path` 필수 기입.
3. `map-vault-staging.md` 표에 한 줄 추가.
4. `TASK-QUEUE.md` 현재 작업 테이블에 행 추가 (유형: `vault`).

## 승인 완료 처리

1. 내용을 `proposed-target-path` 대상 파일에 반영.
2. `map-vault-staging.md` 표에서 행 삭제.
3. `TASK-QUEUE.md`에서 해당 행 삭제.
4. 이 파일 삭제.
