---
frontmatter-version: 1
title: LSWT Canonical Document Lifecycle
section: procedures
status: in-review
last-edited-by: codex
created: 2026-08-01
updated: 2026-08-01
must-read:
  - GOVERNMENT/User-Constitution/single-knowledge-canon.md
  - GOVERNMENT/Court-Precedents/2026-08-01-lswt-markdown-source-authority.md
---

# LSWT Canonical Document Lifecycle

LSWT 이론을 source evidence에서 Markdown 정본 후보로 이식하고, 코드 검증과
웹·PDF 출판으로 연결하는 절차다. Source authority는 이 절차가 아니라 위의
User-Constitution과 Court decision이 소유한다.

## Artifact Roles

| Artifact | Role | Direct content editing |
|---|---|---|
| Accepted LSWT theory Markdown | 유일한 현재 이론 정본 | 허용. 실질 변경 후 재검토 필요 |
| Draft/in-review Markdown | 정본 후보 | 허용 |
| 원본 PDF | Primary evidence | 금지 |
| `note_lswt_reviewed.tex` | Editable transcription | 이식 대조 외 편집 중지 |
| `note_lswt_restructured.tex` | Structural reference | 독립 master로 편집하지 않음 |
| Code and tests | Executable verification | 코드 절차에 따라 편집 |
| Generated TeX/PDF/HTML | 파생 출판물 | 금지 |
| Template, CSS, renderer config | 표현 계층 | 허용 |

README, map, audit, Working-Pad와 `theory/common/` candidate는 LSWT 이론
정본에 포함하지 않는다.

## Canonicalization Lifecycle

### 1. Source review

1. 원본 PDF의 해당 page, section, review annotation을 확인한다.
2. `note_lswt_reviewed.tex`에서 수식과 문장의 editable transcription을 찾는다.
3. `note_lswt_restructured.tex`와 legacy Markdown은 구조와 누락 대조에만 쓴다.
4. Source가 충돌하거나 물리적 의도가 불명확하면 audit에 `open` 또는
   `Unknown`으로 남긴다. 자동 병합하지 않는다.

### 2. Draft authoring

1. 지식 내용은 대응하는 `research-space/theory/lswt/` Markdown에만 작성한다.
2. 한 claim, 정의, 수식은 하나의 canonical Markdown 파일에서만 소유한다.
3. 작업 계획, review ledger, code discrepancy는 본문이 아니라 audit 또는
   Working-Pad에 둔다.
4. Source trace, 수식 ID, 인용키가 확정되지 않은 내용은 `accepted`로
   승격하지 않는다.

### 3. Markdown equation numbering and identity

1. Canonical Markdown의 display equation은 번호 없이 작성하는 것을 기본으로
   한다. 이후 참조되지 않는 수식에는 ID도 붙이지 않는다.
2. Markdown source에 `\tag{...}`를 넣거나, canonical cross-reference를
   `Eq. (3)`처럼 표시 번호로 고정하지 않는다.
3. 다른 문서나 코드 검증에서 다시 참조해야 하는 핵심 수식에만 semantic ID를
   부여한다. ID는 수식의 의미를 나타내며 표시 번호나 파일 내 순번을 포함하지
   않는다.
4. HTML, TeX와 PDF에 표시할 수식 번호는 renderer가 소유한다. 출력 순서가
   바뀌어도 Markdown의 semantic ID는 유지한다.
5. 원본 PDF의 equation number는 primary evidence의 위치를 가리키는 source
   locator로만 사용한다. Frontmatter의 `source-section`, source trace 또는
   audit에서는 `Source Eq. (3)`처럼 기록할 수 있지만 canonical equation의
   이름이나 cross-reference로 사용하지 않는다.
6. Semantic ID의 구체적인 prefix, Markdown 문법과 reference 문법은 출력
   pilot 전에 별도로 결정한다.

### 4. Human physics and mathematics review

이 gate에서는 사용자가 실제 독자의 입장에서 한 logical section씩 rendered
document를 읽고, 물리적 의미와 수학적 전개를 직접 판정한다. 전체 문서를 모두
작성할 때까지 기다리지 않으며, 각 section을 `accepted`로 바꾸기 전에 수행한다.

Review를 시작하기 전에 다음 자료를 한 묶음으로 준비한다.

- 검토할 canonical Markdown과 실제 독자용 rendered preview
- 대응하는 원본 PDF page·section과 editable transcription
- 원본에서 변경한 notation, convention과 수식의 목록
- 물리적 가정, summation convention과 적용 범위
- 해결되지 않은 `Unknown`, source review ID와 formula conflict
- 이미 확인된 theory-code discrepancy

사용자는 다음 항목을 확인한다.

- 각 문장과 수식의 물리적 의미가 맞는가
- 계수, 부호, index, conjugation과 normalization이 수학적으로 맞는가
- 모든 기호가 정의됐고 notation contract를 따르는가
- 가정, convention과 적용 범위가 충분히 명시됐는가
- source와 다르게 쓴 부분에 근거와 설명이 있는가
- 남은 항목을 승인 범위, 수정 필요 또는 `Unknown`으로 구분할 수 있는가

Review 결과는 `acceptance review로 진행 가능`, `revision required`, `Unknown`
중 하나로 기록한다. Section의 핵심 claim이나 수식에 영향을 주는 `Unknown`은
acceptance를 막는다. 범위 밖의 미해결 문제는 본문에서 한계를 명시하고 audit에
추적할 수 있다.

자동 lint, renderer 성공, TeX transcription 또는 코드의 수치 일치는 이 gate를
대신하지 않는다. Review 뒤 물리적 claim, 수식 또는 notation이 바뀌면 rendered
preview를 다시 만들고 이 gate를 반복한다. Foundations처럼 한 document group을
완료한 뒤에는 section 간 notation, 가정과 논리 흐름을 확인하는 integration
review를 추가로 수행한다.

### 5. Acceptance and status transition

문서를 `accepted`로 바꾸기 전에 다음을 확인한다.

- Human physics and mathematics review가 완료됐고 사용자가 명시적으로 승인했다.
- source coverage와 열린 review issue가 기록되어 있다.
- 물리적 convention과 notation이 사용자 검토 범위 안에서 확정됐다.
- 내부 링크, 수식, 인용과 문서 구조 검사가 통과했다.
- `reviewed-by: user`, `reviewed-at`이 있다.

실질적인 claim, 수식 또는 notation을 변경하면 해당 문서를 다시
`in-review`로 두고 사용자 재검토를 받는다.

### 6. Code verification

- Theory acceptance와 code verification은 별도 상태다.
- Accepted equation 또는 claim과 대응 코드·테스트를 명시적으로 연결한다.
- 코드 불일치는 audit 또는 issue에 기록하며, 테스트 통과만으로 theory를
  자동 승인하지 않는다.
- 이론 변경 후에는 관련 코드 검증을 다시 수행한다.

### 7. Derived outputs

```text
accepted Markdown manifest
        |---> generated HTML
        |---> generated TeX ---> generated PDF
        `---> code-verification mapping
```

- HTML, TeX와 PDF는 동일한 accepted Markdown manifest를 입력으로 사용한다.
- 생성물을 Markdown 변환 입력으로 다시 사용하거나 직접 수정하지 않는다.
- 의미·수식 수정은 Markdown에, 조판 수정은 template 또는 renderer에 반영한다.
- 공개 build에는 accepted 문서만 포함한다. Draft와 audit는 preview에서만
  확인한다.
- Renderer와 artifact 보관 경로는 출력 pilot을 통과한 뒤 확정한다.

## Initial Work Order

1. `foundations/notation-and-conventions.md`에서 notation contract를 정리한다.
2. Semantic equation ID, cross-reference와 citation key contract를 단계적으로
   결정한다.
3. `foundations/bilinear-spin-hamiltonian.md`와 함께 HTML·TeX·PDF pilot을 한다.
4. `notation-and-conventions.md`와 `bilinear-spin-hamiltonian.md`를 첫 Human
   physics and mathematics review 묶음으로 검토한다.
5. Pilot과 review 결과로 지원할 Markdown 문법과 renderer를 결정한다.
6. Foundations, derivation, observables, examples, appendices 순으로 이식한다.
7. 각 문서는 source review, 사용자 acceptance, code verification을 분리해
   진행한다.

현재 이론 문서는 모두 draft 또는 in-review이며 accepted canon은 0개다.
