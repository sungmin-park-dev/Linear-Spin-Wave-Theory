---
frontmatter-version: 1
template-version: 1
title: Map - issue-notes
section: issue-notes
status: in-review
last-edited-by: claude
created: 2026-06-03
updated: 2026-09-30
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
| [[GOVERNMENT/Working-Pad/issue-notes/open/260802-code-space-audit-general-2d-spin-tool\|260802-code-space-audit-general-2d-spin-tool]] | review | 범용 2D spin-tool 기준 code-space 분류와 gap 기록 | draft |
| [[GOVERNMENT/Working-Pad/issue-notes/open/260802-topology-thermal-hall-real-space-volume-bug\|260802-topology-thermal-hall-real-space-volume-bug]] | problem | 층당/3D SI κ·full BZ·Chern, 고정 cutoff; 구현 항목은 toolkit 5a–5d(D29)로 해결, NBCP 적용·이론 검토 남음 | in-review |
| [[GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit\|260809-lswt-documentation-audit]] | review | `docs/lswt/` coverage, legacy consolidation과 열린 이론 검토 추적 | in-review |
| [[GOVERNMENT/Working-Pad/issue-notes/open/260810-pseudo-goldstone-gap\|260810-pseudo-goldstone-gap]] | problem | SOC Y/V gap 및 PD-only V의 6회 이방성; 위상 강성·열적 주장 검토와 패키지 설계 대기 | in-review |
| [[GOVERNMENT/Working-Pad/issue-notes/open/260810-lswt-implementation-backlog\|260810-lswt-implementation-backlog]] | review | NBCP band plot와 LSWT 구현 backlog | draft |
| [[GOVERNMENT/Working-Pad/issue-notes/open/260918-nbcp-physics-code-review\|260918-nbcp-physics-code-review]] | review | NBCP 9장 구조, claim-to-code 대응과 독립 물리 구현 검토 | in-review |
| [[GOVERNMENT/Working-Pad/issue-notes/open/260930-lt-step-and-commensurate-structure-decision\|260930-lt-step-and-commensurate-structure-decision]] | discussion | 자기 셀 결정: LT 단계와 `CommensurateStructure` 유지·흡수·삭제 | draft; LSWT 재개 시 결정 |

### `closed/` — 종결 이슈

| 파일 | 유형 | 역할 | 결과 |
|---|---|---|---|
| [[GOVERNMENT/Working-Pad/issue-notes/closed/260602-commensurate-structure-test-failures\|260602-commensurate-structure-test-failures]] | problem | `CommensurateStructure` 테스트 5개 실패: 대각 셀과 각도 수 불일치(테스트 오류) | resolved (`3a2dad2`); 자기 셀 결정은 260930 |
| [260916-docs-topic-consolidation](closed/260916-docs-topic-consolidation.md) | structure | docs와 research-space 통합, LSWT·NBCP 분리와 경로·출력 검증 | resolved; 물리 검토 상태 유지 |
| [[GOVERNMENT/Working-Pad/issue-notes/closed/260604-lswt-section-migration-record\|260604-lswt-section-migration-record]] | review | 기존 Markdown/LaTeX source에서 새 LSWT 정본 파일로 가는 mapping 기록 | resolved |
| [[GOVERNMENT/Working-Pad/issue-notes/closed/260802-hamiltonian-b-block-substitution-bug\|260802-hamiltonian-b-block-substitution-bug]] | problem | Legacy B/B† 교환 원인, 기존 수정 이력과 독립 회귀 검증 | resolved; theory A1 별도 |
| [[GOVERNMENT/Working-Pad/issue-notes/closed/260810-agents-progress-history\|260810-agents-progress-history]] | review | `AGENTS.md`에서 제거한 완료 이력과 활성 항목 이동 기록 | superseded by Working-Pad |
| [[GOVERNMENT/Working-Pad/issue-notes/closed/260910-zero-point-energy-normalization\|260910-zero-point-energy-normalization]] | problem | 영점에너지 trace 계수 검증과 공개 솔버의 스핀당 에너지 수정 | resolved; theory A2/A5/A10/A11/A12 별도 |
| [[GOVERNMENT/Working-Pad/issue-notes/closed/260911-finite-temperature-energy-and-occupation\|260911-finite-temperature-energy-and-occupation]] | problem | 유한온도 전달, thermal energy 합산과 free-energy 계산 수정 | resolved; observables 정규화 별도 |
| [[GOVERNMENT/Working-Pad/issue-notes/closed/260911-thermodynamic-observable-normalization\|260911-thermodynamic-observable-normalization]] | problem | 스핀당 U/S/C, sublattice 점유수와 통합·스캔 함수의 정규화 | resolved; Thermal Hall·Goldstone 별도 |

## 에이전트 지침

- 새 이슈는 `open/`에 생성하고 열린 이슈 표와 `../TASK-QUEUE.md`에 등록한다.
- 이슈가 해결되면 파일 frontmatter `status`를 갱신하고 `closed/`로 이동한다.
- 해결 과정에서 재사용 가능한 절차가 생기면 사용자 확인 후 `GOVERNMENT/Agents-Bylaws/` 승격 후보로 기록한다.

## 참고 문서

- [[GOVERNMENT/Agents-Bylaws/templates/map-template|map-template]] — map 작성 기준
- [[GOVERNMENT/Agents-Bylaws/templates/issue-notes-template|issue-notes-template]] — issue note 작성 기준
- [[GOVERNMENT/Working-Pad/TASK-QUEUE|TASK-QUEUE]] — 활성 작업 큐
