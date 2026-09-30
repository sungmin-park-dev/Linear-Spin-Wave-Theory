---
frontmatter-version: 1
title: LSWT Canonical Document Lifecycle
section: procedures
status: in-review
last-edited-by: codex
created: 2026-08-01
updated: 2026-09-16
must-read:
  - GOVERNMENT/User-Constitution/single-knowledge-canon.md
  - GOVERNMENT/Court-Precedents/2026-08-01-lswt-markdown-source-authority.md
  - GOVERNMENT/Court-Precedents/2026-08-09-lswt-docs-authoring-surface.md
---

# LSWT Canonical Document Lifecycle

LSWT 이론을 source evidence에서 Markdown 정본 후보로 이식하고, 코드 검증과
웹·PDF 출판으로 연결하는 절차다. Source authority는 이 절차가 아니라 위의
User-Constitution과 Court decision이 소유한다.

NBCP 연구 노트의 내용 원본은 [2026-09-18 결정](../../Court-Precedents/2026-09-18-nbcp-latex-source-authority.md)에 따라 `docs/nbcp/main.tex`와 포함된 TeX 파일이다. 아래 Markdown 전용 작성·변환 조항은 LSWT 일반 이론에 적용한다. NBCP도 source trace, semantic label, 인간의 물리·수학 검토와 acceptance 구분은 유지하며, PDF는 TeX 원본에서 생성한다.

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

`docs/README.md`, 주제별 README, sources, NBCP 연구 노트, archive, build, audit, Working-Pad와
`legacy/research-notes/lswt/converted-markdown/`는 LSWT 이론 정본에 포함하지
않는다.

## Review Unit and Completion

한 작업 단위는 독자가 독립적으로 검토할 수 있는 하나의 정의, 주장 또는 유도 구간이다. 작업 전에 대상 파일·section, 필요한 선행 convention과 변경 종류를 정한다. 기존 audit에 이 범위, 수행한 검토, 남은 질문과 다음 행동을 기록하고 작업 큐에는 다음 행동만 요약한다.

- **문체·구조 교정:** 합의한 의미와 수식을 유지하면서 영어, 설명 순서와 연결을 다듬는다. 물리적 의미를 바꾸지 않는 교정마다 전체 원본이나 코드를 다시 검증할 필요는 없다.
- **원문 대조·내용 보완:** 해당 원본 구간과 현재 owner를 대조한다. Claim, 수식, notation 또는 적용 조건이 달라지면 근거와 영향을 기록하고 필요한 사용자 물리·수학 검토로 연결한다.

작업 단위의 편집 완료는 합의한 범위가 본문에 반영되고, 제외·보류한 내용의 위치와 이유가 기록되며, 관련 링크·기호·수식 참조와 실제 rendered preview를 확인한 상태를 뜻한다. 본문에 필요한 설명이 연결 대상의 skeleton에만 있으면 해당 의존성은 미완료로 기록한다. 의미에 영향을 주는 미해결 질문이 있는 단위는 수정 반영 여부와 별도로 검토 보류 상태를 남긴다.

문체 검토, 원문 대조, 사용자 물리·수학 검토와 코드 검증의 결과는 구분해 기록한다. 일부 section의 사용자 승인은 그 범위와 날짜를 audit에 남기며, 파일 전체가 아래 acceptance 조건을 충족하기 전에는 frontmatter를 `accepted`로 바꾸지 않는다. 기준 문서의 교정은 관련 규칙과 링크의 일관성을 확인하며, 이론 수식이나 코드의 검증 완료로 기록하지 않는다.

## Canonicalization Lifecycle

### 1. Source review

1. 원본 PDF의 해당 page, section, review annotation을 확인한다.
2. `note_lswt_reviewed.tex`에서 수식과 문장의 editable transcription을 찾는다.
3. `note_lswt_restructured.tex`와
   `legacy/research-notes/lswt/converted-markdown/`은 구조와 누락 대조에만
   쓴다.
4. Source가 충돌하거나 물리적 의도가 불명확하면 audit에 `open` 또는
   `Unknown`으로 남긴다. 자동 병합하지 않는다.

### 2. Draft authoring

1. LSWT 일반 이론은 대응하는 `docs/lswt/` 번호 폴더의 Markdown에만 작성한다. NBCP 연구 노트는 `docs/nbcp/`에서 별도로 운영한다.
2. 한 claim, 정의, 수식은 하나의 canonical Markdown 파일에서만 소유한다.
3. 작업 계획, review ledger, code discrepancy는 본문이 아니라 audit 또는
   Working-Pad에 둔다.
4. Source trace, 수식 ID, 인용키가 확정되지 않은 내용은 `accepted`로
   승격하지 않는다.
5. 원문 보존은 주장, 수식과 논리 연결의 추적 가능성을 뜻한다. 개념별 재배치는
   허용하되, 원문 구조를 존중한다는 이유로 현재 owner 구분을 되돌리거나
   간결성을 이유로 핵심 수식과 설명을 삭제하지 않는다.
6. 원문 내용은 대응 owner로 이식·이동하거나, 제외·보류 이유와 source 위치를
   audit에 남긴다. 원문에 존재한다는 사실만으로 미결정 범위를 자동 확장하지
   않는다. 원문 밖의 추가 claim은 확인된 외부 source 또는 내부 유도로 연결한다.
7. Overview는 흐름을 설명하는 핵심 관계식을 요약할 수 있다. 상세 정의,
   convention, 유도와 semantic equation ID는 해당 owner가 소유하며, 요약은
   owner와 일치시키고 링크한다.

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
6. Semantic equation ID는 Quarto native cross-reference 문법으로 작성한다.
   수식 block 바로 뒤에 `{#eq-lswt-<semantic-slug>}`를 붙이고, 본문에서는
   `@eq-lswt-<semantic-slug>`로 참조한다.
7. ID는 repository 전체에서 고유한 lowercase ASCII kebab-case로 작성한다.
   표시 번호, 원본 PDF의 equation number, 파일 내 순번과 underscore는 넣지
   않는다.
8. ID는 수식의 물리적 의미를 나타낸다. 파일 이동, section 재배치 또는 표시
   번호 변경만으로는 ID를 바꾸지 않으며, 수식의 물리적 정체가 달라질 때 새
   ID를 부여한다.
9. 하나의 ID는 하나의 logical equation block만 가리킨다. 독립적으로 참조할
   수식이 한 display block에 함께 있으면 block을 나누고 각각 ID를 부여한다.
10. Semantic equation reference가 있는 HTML, TeX와 PDF는 Quarto transform을
    거쳐 생성한다. Standalone Pandoc 변환은 이 cross-reference contract의
    지원 대상이 아니다.
11. GitHub와 일반 Obsidian source view에서 `{#eq-...}`와 `@eq-...`가 그대로
    보일 수 있다. 따라서 reference 주위의 문장은 semantic ID가 해석되지
    않아도 대상 수식의 의미를 파악할 수 있도록 작성한다.
12. 한국어 조사나 어미를 `@eq-...` 바로 뒤에 붙이지 않는다. Quarto가 이를
    ID의 일부로 해석할 수 있으므로, reference는 같은 source line의 설명 문구와
    colon 뒤에 독립된 token으로 배치한다. `@eq-...`로 source line을 시작하지
    않는다. Pandoc 계열 parser가 이를 example-list label로 해석할 수 있다.

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
- 이론의 claim, 수식 또는 convention이 바뀌면 영향을 받는 코드 검증 항목을
  기록하고 해당 검증을 다시 수행한다. 문체 교정에 legacy 수치 비교를 일괄
  요구하지 않으며, 코드 검증 대기 상태와 이론 사용자 검토 상태를 구분한다.

### 7. Derived outputs

```text
accepted Markdown manifest
        |---> generated HTML
        |---> generated TeX ---> generated PDF
        `---> code-verification mapping
```

- 공개 HTML, TeX와 PDF는 동일한 accepted Markdown manifest를 입력으로 사용한다.
- 사용자 검토용 preview는 draft/in-review Markdown에서도 생성한다. Preview
  생성과 열람은 theory acceptance 또는 공개 publication을 의미하지 않는다.
- 생성물을 Markdown 변환 입력으로 다시 사용하거나 직접 수정하지 않는다.
- 의미·수식 수정은 Markdown에, 조판 수정은 template 또는 renderer에 반영한다.
- 공개 build에는 accepted 문서만 포함한다. Draft와 audit는 preview에서만
  확인한다.
- 보관할 생성 출력은 해당 주제의 `output/`에 둔다. NBCP는 `docs/nbcp/output/`에 PDF와 생성 기록만 남기며 편집 원본과 계산 데이터를 복사하지 않는다.
- 일회성 renderer 검증과 중간 TeX·Markdown·그림 사본은 temporary output directory에서 처리한다. 보관할 preview만 해당 주제의 `output/`으로 복사한다.
- Quarto 검증이 source 옆에 `<document>_files/` resource directory 또는 중복
  `.gitignore`를 만들면 검증 직후 생성 여부와 내용을 확인하고 제거한다.
- Renderer가 만든 resource directory는 theory source나 publication artifact로
  취급하지 않는다. `docs/.gitignore`가 이러한 임시 `_files/` 경로를 공통으로
  제외한다.

## Work Order and Current State

문서 작업은 Foundations, derivation, observables, examples, appendices의 의존 관계를 따라 진행한다. 각 단위에서 필요한 notation을 먼저 대조하고, 이미 합의한 convention과 수식 ID 문법은 재사용한다. 새 renderer 또는 지원 문법을 도입할 때는 관련 문서로 출력 pilot을 수행한다.

진행 상태와 바로 다음 작업은 [Task Queue](../../Working-Pad/TASK-QUEUE.md), 문서별 coverage·열린 질문·사용자 검토 범위는 [Documentation Audit](../../Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md)가 소유한다. 이 절차에는 현재 문서 수나 완료 이력을 중복 기록하지 않는다.
