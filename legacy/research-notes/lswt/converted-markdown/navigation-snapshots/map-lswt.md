---
frontmatter-version: 1
template-version: 1
title: Map - lswt
section: theory/lswt
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-08-01
must-read: GOVERNMENT/Agents-Bylaws/templates/map-template.md
---

# Map - lswt

> 이 파일은 `research-space/theory/lswt/` 디렉토리의 인덱스입니다.
> 새 LSWT 정본 문서나 appendix를 추가하면 이 표도 갱신합니다.

LSWT 이론 문서를 정본화하기 위한 작업 공간. 파일명에는 순번이나 분류코드를
넣지 않고, 1단계 폴더와 이 map으로 문서의 역할과 읽는 순서를 관리한다.

## 목차

| 항목 | 역할 |
|---|---|
| [[research-space/theory/lswt/README\|README]] | 이 디렉토리의 목적, 승인된 source authority, 작업 원칙 |
| [[research-space/theory/lswt/map-lswt\|map-lswt]] | 이 navigation 파일 |
| [[research-space/theory/lswt/current-sections-audit\|current-sections-audit]] | 현재 workspace의 coverage, lifecycle, review issue snapshot |
| `foundations/` | LSWT 유도 전에 고정해야 하는 Hamiltonian, notation, local-frame convention |
| `derivation/` | spin operator에서 momentum-space BdG Hamiltonian과 diagonalization까지의 LSWT 본체 |
| `observables/` | diagonalization 이후 계산되는 물리량과 response function |
| `examples/` | worked example와 코드 검증으로 이어지는 사용 절차 |
| `appendices/` | 본문 흐름을 보조하는 증명과 유도 |

### Remarks

하위 폴더 내부 문서 목록과 읽기 순서는 각 폴더의 `map-*.md`에서 관리한다.
기존 Markdown/LaTeX source에서 새 LSWT 정본 파일로 가는 mapping 기록은
[[GOVERNMENT/Working-Pad/issue-notes/closed/260604-lswt-section-migration-record|260604-lswt-section-migration-record]]를 참조한다.

## 에이전트 지침

- `README.md`에는 목적과 작업 원칙을 둔다. 파일 목록은 이 map에 둔다.
- `theory/common/`은 cross-solver convention의 draft candidate 영역이다.
  사용자 승인 전 accepted canon이나 code contract로 사용하지 않는다.
- 본문 이식 중 code discrepancy나 common 후보가 보이면 해당 section의 작업 메모에 기록한다.
- 새 파일을 추가할 때는 먼저 어느 폴더의 논리 단위인지 확인하고, 순번형 파일명은 쓰지 않는다.

## 참고 문서

- [[GOVERNMENT/Agents-Bylaws/templates/map-template|map-template]] — map 작성 기준
- [[research-space/theory/lswt/README|README]] — LSWT theory 작업 원칙
- [[GOVERNMENT/User-Constitution/single-knowledge-canon|single-knowledge-canon]] — 단일 지식 정본 원칙
- [[GOVERNMENT/Court-Precedents/2026-08-01-lswt-markdown-source-authority|2026-08-01-lswt-markdown-source-authority]] — LSWT source authority 결정
- [[GOVERNMENT/Agents-Bylaws/procedures/lswt-canonical-document-lifecycle|lswt-canonical-document-lifecycle]] — 정본화·검증·출력 절차 초안
- [[GOVERNMENT/Working-Pad/issue-notes/closed/260604-lswt-section-migration-record|260604-lswt-section-migration-record]] — 기존 source에서 정본 파일로 가는 migration 기록
