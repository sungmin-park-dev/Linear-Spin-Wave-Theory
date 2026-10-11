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

`docs/`는 보관하고 공개할 것만 둔다. 실제 업무와 정본은 [`workbench/`](../workbench/)에서 관리한다. NBCP 연구는 2026-10-09 별도 저장소 `nbcp-spin-supersolid`로 옮겼다([결정](../GOVERNMENT/Court-Precedents/2026-10-09-nbcp-repository-split.md)).

| 찾는 내용 | 위치 | 용도 |
|---|---|---|
| 패키지 사용법(영어 튜토리얼) | [tutorials/](tutorials/README.md) | 첫 계산부터 중성자·M(h)·위상까지 5편. 각 수치는 해석해·문헌·ED와 대조하고 테스트로 확인한다. |
| LSWT 원자료 | [sources/](sources/README.md) | 원본 노트·TeX 전사본·검증 근거. 현재 이론 작업이 계속 참조한다. |
| LSWT 일반 이론 Markdown (보존본) | [archive/2026-10-08-lswt-markdown/](archive/2026-10-08-lswt-markdown/README.md) | 2026-10-08 상태로 보존. 더 고치지 않는다. |
| Toolkit 개발 설계 Beamer (보존본) | [archive/2026-10-08-development-beamer/](archive/2026-10-08-development-beamer/README.md) | 2026-10-08 상태로 보존. 더 고치지 않는다. |
| 그 밖에 집필을 종료한 문서 | [archive/](archive/README.md) | 이전 문서·스냅샷 |

## 현재 편집할 파일

[2026-10-08 결정](../GOVERNMENT/Court-Precedents/2026-10-08-lswt-development-research-notes.md)에 따라 내용의 편집 원본은 research-workspace 앱의 연구노트 [`workbench/notes/`](../workbench/notes/)다.

- 일반 이론: `lswt-spin-hamiltonian`, `lswt-classical-order`, `lswt-boson-hamiltonian`, `lswt-paraunitary-diagonalization`, `lswt-magnon-thermodynamics`, `lswt-spin-correlations`, `lswt-magnon-topology`
- 개발 설계와 결정 기록: `dev-design`, `dev-verification`, `dev-decisions`
- 계산 코드와 결과 데이터는 각각 [examples/](../examples/)와 [data-space/](../data-space/)에서 관리한다.

폴더 위치는 검토 완료 여부를 뜻하지 않는다. LSWT 이론의 acceptance와 코드 검증은 별도로 기록한다. 참고자료는 현재 주장을 확인하는 근거이며 본문과 동등한 편집 원본이 아니다.
