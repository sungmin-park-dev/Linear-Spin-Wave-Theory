---
frontmatter-version: 1
title: Decision - LSWT Docs Authoring Surface
section: decisions
decision-type: source-authority
status: accepted
last-edited-by: codex
created: 2026-08-09
updated: 2026-09-05
effective-date: 2026-08-09
scope: docs
reviewed-by: user
reviewed-at: 2026-08-09
supersedes-path-clauses-in:
  - GOVERNMENT/Court-Precedents/2026-08-01-lswt-markdown-source-authority.md
---

# Decision - LSWT Docs Authoring Surface

## Context

2026-08-01 결정은 Markdown을 유일한 LSWT 지식 정본 작성면으로 확정했지만
당시 경로는 `research-space/theory/lswt/`였다. 사용자는 2026-08-09 이론 설명을
더 얕고 독자 중심인 `docs/`로 올리고, 번호가 붙은 하위 폴더가 읽는 순서를
나타내도록 승인했다.

2026-09-05 사용자가 기존 기준 문서 보완을 승인함에 따라, staging에 남아 있던
경로 결정을 Court-Precedents에 기록했다. `reviewed-at`은 원래 경로 승인일이며,
`updated`는 이 기록을 정비한 날짜다. 이 결정은 이론 본문의 acceptance가 아니다.

## Decision

1. LSWT 이론의 active Markdown authoring surface는 `docs/`다.
2. `docs/00-foundations/`부터 `docs/04-appendices/`까지의 folder prefix는 큰
   읽기 순서를 나타내며 claim의 우선순위나 수식 번호가 아니다.
3. `docs/README.md`는 독자 navigation과 읽기 순서를 소유하지만 이론 정본에는
   포함하지 않는다.
4. `docs/**/*.md` 중 사용자가 `status: accepted`, `reviewed-by: user`,
   `reviewed-at`으로 승인한 content document만 현재 이론 정본이다.
5. 실행 가능한 Python 예제와 검증 asset은 root `examples/`에 둔다. 설명 문서는
   `docs/03-examples/`에 둔다.
6. 과거 converted Markdown, common candidate와 navigation snapshot은
   `legacy/research-notes/lswt/converted-markdown/`에서 source-only evidence로
   보존한다. 그 경로로 이동했다는 사실은 coverage나 acceptance를 뜻하지 않는다.
7. Operational audit는
   `GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md`가
   소유한다.
8. 원본 PDF, reviewed TeX, restructured TeX, generated output의 역할과 theory
   acceptance/code verification/web publication의 분리는 2026-08-01 결정을
   그대로 유지한다.
9. 원본과 TeX의 현재 보관 위치는
   [Source Inventory](../../docs/lswt/sources/README.md)가 안내한다.
   원본 PDF는 `research-space/sources/00-primary-source/`에, reviewed와
   restructured TeX는 `research-space/sources/01-editable-notes/`에 있다.
   2026-08-01 결정의 옛 경로는 당시 기록으로 보존하며 현재 파일 위치로
   해석하지 않는다.

## Consequences

- `research-space/`는 active theory Markdown이 아니라 source evidence를
  보관한다.
- Theory document frontmatter는 Quarto 예약 field인 `section` 대신
  string-valued `doc-path`를 사용한다.
- AGENTS, lifecycle, naming, writing-style와 active issue reference는 새 경로를
  사용한다.
- Semantic content consolidation은 구조 이동과 별개로 concept owner별 source
  review와 Human Physics and Mathematics Review를 계속 거친다.

## Related Documents

- [Previous Source Authority Decision](2026-08-01-lswt-markdown-source-authority.md): 단일 Markdown 정본과 근거·파생물의 역할을 정한 원래 결정. 이 후속 결정은 경로 조항만 대체한다.
- [Documentation Map](../../docs/lswt/README.md): 현재 읽는 순서와 작성 기준의 진입점.
- [Canonical Document Lifecycle](../Agents-Bylaws/procedures/lswt-canonical-document-lifecycle.md): 이론 내용의 작성, 검토와 acceptance 절차.
- [Decisions Map](map-decisions.md): 결정 기록의 탐색 인덱스.
