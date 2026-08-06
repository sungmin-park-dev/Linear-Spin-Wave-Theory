---
frontmatter-version: 1
title: LSWT Theory Canonical Workspace
section: theory/lswt
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-08-01
---

# LSWT Theory Canonical Workspace

이 디렉토리는 Linear Spin Wave Theory 문서를 정본화하는 작업 위치다.
여기서 `canonical workspace`는 정본을 만들기 위한 활성 작업 대상을 뜻하며,
사용자가 승인한 `status: accepted` 정본을 뜻하지 않는다. 현재 승인 정본은
없다.

문서의 목표와 approved evidence routing은 이 README에서 관리한다.
`map-lswt.md`는 navigation과 파일 역할 경계의 기준이고,
`current-sections-audit.md`는 특정 시점의 coverage와 review 상태를 기록하는
evidence snapshot이다. Audit의 파일 표는 navigation 목록을 대체하지 않는다.

Source authority의 승인 원문은
`GOVERNMENT/Court-Precedents/2026-08-01-lswt-markdown-source-authority.md`에
있다. 이 README는 그 결정을 현재 workspace에 적용하는 안내문이다.

## Goal

1. 원본 PDF의 내용을 논리 단위별 Markdown 파일로 누락 없이 옮긴다.
2. 정본 후보 본문에는 검증된 이론 내용만 두고, 작업 계획과 agent note는
   audit 또는 Working-Pad 문서로 분리한다.
3. 파일명에는 순번이나 분류 코드를 넣지 않고, 폴더와 map으로 문서의 역할과
   읽는 순서를 관리한다.

## Lifecycle

| Layer | Meaning |
|---|---|
| Primary evidence | 원본 내용을 판단할 때 가장 먼저 확인하는 자료 |
| Supporting source | 전사, 구조 비교, 수정 이력을 돕지만 primary evidence를 덮어쓰지 않는 자료 |
| Active working target | `research-space/theory/lswt/` 아래의 draft와 skeleton |
| Accepted canon | 사용자가 검토하고 `accepted`로 승인한 문서. 현재 없음 |
| Legacy/source-only | 이식 대조가 끝날 때까지 보존하지만 직접 정본으로 편집하지 않는 자료 |

## Editing Rules

- 정본 후보 본문에는 작업 계획, 구현 전략, review ledger를 넣지 않는다.
- 이론 convention은 코드 구현보다 먼저 문서에서 설명한다.
- 원본에 없는 claim을 보완 추론만으로 정본 후보 본문에 추가하지 않는다.
- Source 간 수식이 다르면 자동 선택하지 말고 audit에 `open`으로 남긴다.
- Review issue가 해결되었다고 표시하려면 근거와 사용자 확인이 필요하다.
- Monte Carlo, Tensor Network, Neural Quantum State도 공유할 정의는
  `common candidate`로만 기록한다. 현재 `theory/common/`을 승인 정본으로
  간주하지 않는다. Candidate 경계는 `../common/README.md`를 따른다.
- 기존 converted Markdown은 coverage 검증 전까지 삭제하거나 직접
  rewrite하지 않는다.
- 이론 claim은 `research-space/theory/lswt/`의 대응 Markdown에서만
  편집한다. Markdown과 TeX를 병렬 master로 유지하지 않는다.
- 생성된 TeX, PDF, HTML에서 발견한 의미·수식 오류는 Markdown으로 환류한
  뒤 다시 생성한다. 생성물을 직접 고쳐 지식 변경을 만들지 않는다.
- Theory acceptance, code verification, web publication은 별도 상태로
  기록한다.

## Approved Evidence Routing

> Approved: 2026-08-01. 이 표는 Markdown 정본화 과정에서 자료를 대조하는
> 순서다. 이 routing의 승인은 현재 draft 이론 문서를 `accepted`로 승인한
> 것이 아니다.

| Priority | Category | Path | Use |
|---:|---|---|---|
| 1 | Primary evidence | `/Users/david/Downloads/Linear_Spin_Wave_Theory___Note.pdf` | 사용자가 지정한 27-page 원본. 전체 흐름, 수식, review annotation을 판단하는 최우선 자료 |
| 2 | Editable transcription | `legacy/research-notes/lswt/note_lswt_reviewed.tex` | Section order와 review ID가 primary PDF에 대응하는 전사 보조 자료. 경로상 legacy이며 PDF를 덮어쓰지 않음 |
| 3 | Structural reference | `research-space/sources/lswt/note_lswt_restructured.tex` | 논리적 재구성과 section mapping 참고. Primary PDF와 구조가 다르고 현재 clean-build 기준 소스가 아님 |
| 4 | Historical snapshot | `research-space/sources/lswt/note_lswt_restructured.pdf` | Restructured TeX 계열의 과거 출력물. Primary evidence와 별개의 PDF |
| 5 | Historical source | `legacy/research-notes/lswt/hamiltonian_convention.tex` | Hamiltonian convention의 과거 정리와 수정 이력 확인 |
| 6 | Historical source | `legacy/research-notes/lswt/note.tex` | 더 오래된 LSWT note의 표현과 출처 확인 |
| 7 | Legacy converted Markdown | `research-space/theory/sections/`, `research-space/theory/notation.md` | 내용 이식과 누락 대조용 source-only 자료 |

`note_lswt_reviewed.tex`가 PDF와 대응하더라도 primary evidence는 PDF다.
`note_lswt_restructured.tex`의 분할 방식은 새 Markdown 구조를 설계하는 데
활용할 수 있지만, PDF와 다른 수식이나 section order를 정본으로 자동
승격하지 않는다.

## Active Editing Authority

- Unique canonical authoring surface:
  `research-space/theory/lswt/`의 이론 내용 Markdown
- Accepted canon: `status: accepted`, `reviewed-by: user`, `reviewed-at`을 모두
  가진 문서만 해당. 현재 0개
- Active editable LaTeX master: 없음
- Historical TeX master record: `note_lswt_restructured.tex` 지정은
  2026-08-01 결정으로 superseded
- Publication outputs: accepted Markdown에서 생성할 TeX, PDF, HTML

정본 작성면은 여러 논리 단위의 Markdown으로 구성되지만 하나의 corpus다.
README, map, audit, draft, `theory/common/` candidate는 이론 정본이 아니다.

## Evidence Review Order

1. Primary PDF의 해당 page와 review annotation을 확인한다.
2. `note_lswt_reviewed.tex`에서 전사 가능한 식과 문장을 찾는다.
3. Restructured TeX와 legacy Markdown에서 구조 개선안과 수정 이력을 대조한다.
4. 출처들이 일치하면 draft 본문으로 이식한다.
5. 출처들이 다르거나 물리적 의도가 불명확하면
   `current-sections-audit.md`에 `open`으로 기록하고 사용자 확인 전 본문을
   확정하지 않는다.

## Workspace Status

- 일부 내용이 작성된 draft: 7 files
- Frontmatter가 있는 draft skeleton: 10 files
- User-approved accepted canon: 0 files

## Next Canonicalization Step

1. `foundations/notation-and-conventions.md`에서 notation, 수식 ID와 인용
   계약을 정리한다.
2. `foundations/bilinear-spin-hamiltonian.md`와 함께 HTML·TeX·PDF 출력
   pilot을 수행한다.
3. Pilot 결과를 검토한 뒤 renderer와 Markdown 문법 범위를 확정한다.

파일별 coverage snapshot, review routing, 열린 질문은
`current-sections-audit.md`에서 확인한다.
