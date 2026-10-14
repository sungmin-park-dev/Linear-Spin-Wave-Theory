---
frontmatter-version: 1
title: Documentation
doc-path: docs
status: in-review
last-edited-by: codex
created: 2026-09-16
updated: 2026-09-23
---

# 문서 안내

`docs/`에는 우리가 쓴 글(패키지 튜토리얼, 내부 검증 자료)을 둔다. 원자료는 맨 위 [`sources/`](../sources/README.md), 더 고치지 않는 보존본은 맨 위 [`archive/`](../archive/README.md)에 있다. 실제 업무와 정본은 [`workbench/`](../workbench/)에서 관리한다. NBCP 연구는 2026-10-09 별도 저장소 `nbcp-spin-supersolid`로 옮겼다([결정](../workbench/decisions/2026-10-09-nbcp-repository-split.md)).

| 찾는 내용 | 위치 | 용도 |
|---|---|---|
| 패키지 사용법(영어 튜토리얼) | [tutorials/](tutorials/README.md) | 첫 계산부터 중성자·M(h)·위상까지 5편. 각 수치는 해석해·문헌·ED와 대조하고 테스트로 확인한다. |
| 내부 검증 자료 | [verification-notes/](verification-notes/) | 수식·basis convention과 코드의 대조 (옛 `docs/sources/03-verification-notes`) |
| LSWT 원자료 | [../sources/](../sources/README.md) | 원본 노트·TeX 전사본. 현재 이론 작업이 계속 참조한다. |
| LSWT 일반 이론 Markdown (보존본) | [../archive/2026-10-08-lswt-markdown/](../archive/2026-10-08-lswt-markdown/README.md) | 2026-10-08 상태로 보존. 더 고치지 않는다. |
| LSWT 이론 LaTeX 연구노트 (보존본) | [../archive/2026-10-11-lswt-latex-notes/](../archive/2026-10-11-lswt-latex-notes/) | 2026-10-11 Markdown 전환 직전 상태와 변환 도구. 더 고치지 않는다. |
| Toolkit 개발 설계 Beamer (보존본) | [../archive/2026-10-08-development-beamer/](../archive/2026-10-08-development-beamer/README.md) | 2026-10-08 상태로 보존. 더 고치지 않는다. |
| 그 밖에 집필을 종료한 문서 | [../archive/](../archive/README.md) | 이전 문서·스냅샷, legacy 코드, 옛 GOVERNMENT 기록. 옛 경로 → 새 경로 표 |

## 현재 편집할 파일

[2026-10-08 결정](../workbench/decisions/2026-10-08-lswt-development-research-notes.md)에 따라 내용의 편집 원본은 research-workspace 앱의 연구노트 [`workbench/topics/`](../workbench/topics/)(주제 폴더 안)다.

- 일반 이론(Markdown `note.md`, [2026-10-11 결정](../workbench/decisions/2026-10-11-lswt-notes-markdown.md)): `lswt-spin-hamiltonian`, `lswt-classical-order`, `lswt-boson-hamiltonian`, `lswt-paraunitary-diagonalization`, `lswt-magnon-thermodynamics`, `lswt-spin-correlations`, `lswt-magnon-topology`; 1/S 보정 `lswt-nonlinear-spin-waves`; 예제 노트 `lswt-example-*` 4개(계산은 `code-space/examples/worked_examples/`)
- 개발 설계와 결정 기록: `dev-design`, `dev-verification`, `dev-decisions`
- 계산 코드와 결과 데이터는 각각 [code-space/examples/](../code-space/examples/)와 [data-space/](../data-space/)에서 관리한다.

폴더 위치는 검토 완료 여부를 뜻하지 않는다. LSWT 이론의 acceptance와 코드 검증은 별도로 기록한다. 참고자료는 현재 주장을 확인하는 근거이며 본문과 동등한 편집 원본이 아니다.
