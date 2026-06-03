# LSWT Theory Canonical Workspace

이 디렉토리는 Linear Spin Wave Theory 문서를 앞으로 적극적으로 수정하고
정본화할 작업 위치다.

현재 단계에서는 아직 이론 내용을 rewrite하지 않는다. 먼저 기존 Markdown
문서와 LaTeX 기준 소스를 대조하고, 어떤 섹션을 어떤 정본 파일로 옮길지
계획을 확정한다.

## 기준 소스

정본화의 기준 LaTeX 소스:

- `research-space/sources/lswt/note_lswt_restructured.tex`

보조 참고 자료:

- `legacy/research-notes/lswt/note_lswt_reviewed.tex`
- `legacy/research-notes/lswt/hamiltonian_convention.tex`
- `legacy/research-notes/lswt/note.tex`

기존 Markdown 변환본:

- `research-space/theory/sections/`
- `research-space/theory/notation.md`

## 작업 원칙

- LSWT 문서를 먼저 정리한다.
- `common/` 문서는 먼저 만들지 않는다.
- LSWT 정리 과정에서 Monte Carlo, Tensor Network, Neural Quantum State도
  공유해야 할 정의가 보이면 common 후보로 기록한다.
- common 후보는 충분히 명확해진 뒤에만 `research-space/theory/common/`
  같은 별도 위치로 승격한다.
- 이론 convention은 코드 구현보다 먼저 문서에서 설명한다.

## 파일 탐색

파일 목록과 각 문서의 역할은 `map-lswt.md`에 둔다.

## 다음 단계

1. `section-migration-plan.md`를 검토한다.
2. 새 LSWT section skeleton을 기준으로 하나씩 본문을 이식한다.
3. 각 섹션을 `note_lswt_restructured.tex` 기준으로 정리한다.
4. 정리 중 발견되는 code discrepancy와 common 후보를 함께 기록한다.
