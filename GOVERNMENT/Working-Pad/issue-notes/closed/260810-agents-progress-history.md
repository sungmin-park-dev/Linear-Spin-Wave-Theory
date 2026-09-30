---
frontmatter-version: 1
title: AGENTS progress history migration record
section: issue-notes/closed
issue-type: review
status: closed
resolution: superseded
outcome: GOVERNMENT/Working-Pad/TASK-QUEUE.md
last-edited-by: codex
created: 2026-08-10
updated: 2026-09-10
closed: 2026-08-10
must-read: GOVERNMENT/Agents-Bylaws/templates/issue-notes-template.md
---

# AGENTS progress history migration record

## 배경

`AGENTS.md`는 장기간 유지되는 에이전트 규칙과 source authority를 제공해야 하지만, 과거에는 완료 이력, 알려진 버그, 예제 진행률과 구현 backlog까지 함께 기록했다. 2026-08-10 사용자 결정에 따라 변하는 상태를 Working-Pad로 이동하고 `AGENTS.md`에서는 진행 이력을 제거했다.

## 문제 정의

에이전트 지침과 프로젝트 상태가 한 파일에 섞이면 완료 이력이 오래된 현재 상태처럼 읽히고, `TASK-QUEUE.md` 및 issue-note와 충돌할 수 있다. 이 문서는 제거된 완료 이력을 보존하고 활성 항목이 이동한 위치를 기록한다.

## 본론

### 완료 이력

#### 2026-08-09 이론 문서 구조 개편

- LSWT Markdown 작성면을 루트 `docs/`로 이동
- `00-foundations/`부터 `04-appendices/`까지 번호 prefix로 읽는 순서 부여
- 탐색 문서를 `docs/README.md` 하나로 통합
- 실행 예제를 루트 `examples/`로 분리
- 과거·중복 Markdown을 `legacy/research-notes/lswt/converted-markdown/`에 source-only로 보존

#### 2026-08-01 이론 지식 정본 결정

- LSWT 이론의 유일한 정본 작성면을 Markdown으로 확정
- 원본 PDF를 primary evidence, reviewed TeX를 editable transcription, restructured TeX를 structural reference로 구분
- 향후 generated TeX·PDF·HTML을 accepted Markdown의 단방향 파생물로 정의
- Theory acceptance, code verification와 web publication 상태를 분리

#### 2026-06-01 구조 마이그레이션

- 루트 폴더를 `GOVERNMENT/`, `code-space/`, 당시 `doc-space/`, `research-space/` 기준으로 재편
- `pyproject.toml` 패키지 탐색 경로를 `code-space`로 갱신
- AAD 철학 문서 사본을 `GOVERNMENT/Working-Pad/idea-proposals/`에 배치
- 오래된 이론-코드 검증 계획서를 `GOVERNMENT/Agents-Bylaws/procedures/`로 이동하고 `needs-review`로 표시
- `.DS_Store`, `__pycache__`, `*.egg-info` 생성물 정리 및 ignore 규칙 추가

#### 2026-05-31 사전 정리

- 코드, 지식, 운영 문서의 중간 분류 작업 수행
- `research-space/theory/latex_sections/` 삭제
- 당시 `research-space/sources/lswt/`를 정리하고 `note_lswt_restructured.tex`를 master로 기록했으나, 이 판단은 2026-08-01 source-authority 결정으로 superseded
- 이론-코드 검증 계획서 초안 작성
- AAD 마이그레이션 사전 작업 문서 사본 준비

#### 이전 구현 완료 기록

- 패키지 구조와 `pyproject.toml`
- Core: `SpinSystem`, exchange, `BrillouinZone`, `Diagonalizer`
- Solvers: `LSWTSolver`, `LSWTHamiltonian`, `SpinOptimizer`, `EnergyFunction`
- Observables: `bose_statistics`, thermodynamics, topology, correlations
- `AbstractSolver`와 `SolverResult` 공통 인터페이스
- Legacy 코드와 3661 k-point 고유값 비교에서 차이 0으로 기록된 검증
- Legacy 코드와 스크립트를 `legacy/`로 아카이빙
- 상수 통합, `utils/` 제거, `physics/`를 `observables/`로 변경
- `SpinSystem` nested Site/Coupling, label-first API와 solver-side `bz_type`
- `site()`와 `get_couplings()` 접근 메서드
- `add_site()`와 `add_coupling()` builder pattern
- `exchange.py`의 `bond_angle_exchange()`와 `nnn_exchange()`
- NBCP classical ground-state 최적화와 legacy bit-level 일치 기록
- Spin-configuration visualizer

### 활성 항목 이동 기록

2026-08-10 당시의 이동 경로를 보존한다. B/B† 이슈는 이후 2026-09-10
기존 수정 검증과 회귀 테스트 추가로 종결됐으며, 현재 기록은
[closed 이슈](260802-hamiltonian-b-block-substitution-bug.md)에 있다.

| 제거된 AGENTS 항목 | 현재 owner |
|---|---|
| `docs/` 경로 precedent 승격, notation/equation-ID/output pilot | `issue-notes/open/260809-lswt-documentation-audit.md`, `vault-staging/2026-08-09-lswt-docs-authoring-surface.md` |
| B/B† block bug | `issue-notes/open/260802-hamiltonian-b-block-substitution-bug.md` |
| Thermal Hall volume/unit bug | `issue-notes/open/260802-topology-thermal-hall-real-space-volume-bug.md` |
| Pseudo-Goldstone gap | `issue-notes/open/260810-pseudo-goldstone-gap.md` |
| NBCP band plot와 나머지 구현 TODO | `issue-notes/open/260810-lswt-implementation-backlog.md` |

## 결론 / 미결 사항

`AGENTS.md`의 progress section은 이 기록으로 대체되었다. 앞으로 완료 이력은 closed issue-note에, 미해결 문제는 open issue-note에, 활성 우선순위는 `TASK-QUEUE.md`에 기록한다.

이 문서는 과거의 완료 주장을 새로 검증하지 않는다. 특히 legacy 수치 일치와 API 합의 기록은 후속 작업에서 현재 checkout과 원 근거를 다시 확인해야 한다.

## 참조

- `AGENTS.md` — 안정적인 에이전트 지침과 source authority
- `GOVERNMENT/Working-Pad/TASK-QUEUE.md` — 활성 작업 인덱스
- `GOVERNMENT/Working-Pad/issue-notes/map-issue-notes.md` — open/closed 기록 map
- `GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md` — 현재 documentation 상태와 review 항목
