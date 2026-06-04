# LSWT Theory Canonical Workspace

이 디렉토리는 Linear Spin Wave Theory 문서를 정본화하는 작업 위치다.
문서의 전체 목표와 작성 방향은 이 README에 두고, 파일 목록과 각 파일의
역할은 `map-lswt.md`에서 관리한다.

## Goal

1. 기존 tex 문서의 내용을 섹션별로 개별적인 markdown 파일로 누락없이 옮겨 작성한다. 
2. 작성된 마크다운 파일에서는 정돈된 이론적인 내용을 위주로 작성한다. 프로젝트의 방향성 및 계획 전략 등은 일체 포함하지 않는다.

## Editing Rules

- LSWT 정본 문서를 먼저 정리한다.
- 정본 본문에는 작업 계획이나 운영 메모를 넣지 않는다.
- LSWT 정리 과정에서 Monte Carlo, Tensor Network, Neural Quantum State도 공유해야 할 정의가 보이면 common 후보로 기록한다.
- 이론 convention은 코드 구현보다 먼저 문서에서 설명한다.
- 파일 목록과 각 파일의 역할은 `map-lswt.md`에서만 관리한다.
- 문서의 초안을 작성 기존 원본의 구조를 반영하여 시작한다. 예를 들어, 1.1 <section> 은 ## <section title> 으로 정확히 매핑 시킨다. 


## Current Focus

- `00_introduction.md`
- `01_from_spins_to_bosons.md`

## Source Material

| Category | Path | Contents |
|---|---|---|
| Primary source | `research-space/sources/lswt/note_lswt_restructured.tex` | 정본화 기준이 되는 최신 LaTeX source. LSWT 전체 흐름, 수식, review TODO를 포함한다. |
| Supporting source | `legacy/research-notes/lswt/note_lswt_reviewed.tex` | 과거 reviewed version. 현재 source와 비교해 누락된 설명이나 수정 이력을 확인할 때 사용한다. |
| Supporting source | `legacy/research-notes/lswt/hamiltonian_convention.tex` | Hamiltonian convention 관련 과거 정리. site/link counting, exchange convention 확인용 참고 자료다. |
| Supporting source | `legacy/research-notes/lswt/note.tex` | 이전 LSWT note 원본. 오래된 구조나 표현의 출처를 확인할 때만 보조로 사용한다. |
| Legacy converted Markdown | `research-space/theory/sections/` | 기존 Markdown 변환본. 새 정본 파일로 내용을 이식할 때 source material로만 사용한다. |
| Legacy converted Markdown | `research-space/theory/notation.md` | 기존 notation 요약. `appendices/A_notation.md`로 이식할 기준 자료다. |
