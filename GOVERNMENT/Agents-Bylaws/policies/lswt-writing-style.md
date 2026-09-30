---
frontmatter-version: 1
title: LSWT Writing Style Policy
section: policies
status: in-review
last-edited-by: codex
created: 2026-06-04
updated: 2026-09-16
---

# LSWT Writing Style Policy

이 문서는 `docs/lswt/`의 reader-facing LSWT 이론 노트를 영어 수학·물리 학술 문체로 작성하고 교정하기 위한 project-specific policy다. 이 문서는 문체와 source formatting만 소유하며, 물리적 claim, notation, source authority, acceptance와 publication lifecycle은 관련 owner document를 따른다.

## Purpose and Priority

이 정책은 `docs/lswt/00-foundations/`부터 `docs/lswt/04-appendices/`까지의 theory Markdown에 적용한다. `docs/README.md`, 주제별 README와 governance 문서는 navigation 또는 운영 목적에 맞게 작성하되, terminology와 Markdown formatting 규칙은 가능한 한 공유한다. 코드, issue note, Working-Pad와 사용자 대화는 적용 대상이 아니다.

아래에서 **Required**는 반드시 지킬 규칙, **Preferred**는 문맥에 따라 적용하는 학술 문체, **Allowed**는 에이전트가 과잉 교정하지 말아야 할 허용 범위를 뜻한다. `status: in-review`인 동안에는 이 구분을 current working guidance로 적용하되 accepted policy로 보고하지 않는다. 명시적인 사용자 결정과 project owner document는 이 정책보다 우선한다.

## Reader and Explanatory Depth

기본 독자는 양자역학과 선형대수를 알고 있지만 해당 LSWT 유도를 처음 따라가는 사람이다. 기본적인 대수 조작은 설명 없이 사용할 수 있으나, 이 노트가 선택한 기호, 기저, convention과 근사 조건을 이미 안다고 가정하지 않는다. 독자는 본문과 연결된 owner 문서만으로 계산을 따라갈 수 있어야 한다.

| Document role | Expected explanation |
|---|---|
| Overview | 이론의 대상, 적용 범위와 핵심 관계식을 간결하게 연결한다. 상세 정의와 계산은 해당 owner로 안내한다. |
| Notation or reference | 기호의 의미, 사용 범위와 convention을 빠르게 확인할 수 있게 한다. 실제 계산에 처음 도입할 때 필요한 설명은 해당 유도에서도 제공한다. |
| Derivation | 가정과 정의에서 결과까지의 논리적 연결을 보인다. 기저 변환, 연산자 순서, 합산 범위, 정규화와 근사 단계는 결과에 영향을 주는 중간 과정을 설명한다. |
| Observable | 물리량의 정의, 적용 조건과 계산식을 연결하고, normalization과 결과의 물리적 해석을 설명한다. |
| Worked example | 입력 모델과 가정, 계산 단계, 결과와 확인 방법을 따라갈 수 있게 제시한다. 일반 정의는 owner 문서를 사용한다. |
| Appendix | 본문에서 연결한 긴 증명이나 보조 유도를 완결한다. 본문은 결과를 이해하는 데 필요한 정의와 조건을 유지한다. |

이 구분은 설명 깊이의 기준이며 모든 문서에 같은 section 구조나 분량을 강제하지 않는다. 간결하게 쓰더라도 주장에 필요한 수식이나 논리 연결을 생략하지 않는다. 원문의 재배치, 제외·보류와 변경 근거 기록은 [LSWT Canonical Document Lifecycle](../procedures/lswt-canonical-document-lifecycle.md)의 draft authoring 절차를 따른다.

## Required Rules

### Consistent Narrative Tone

문체는 차분한 영어 이론 강의노트로 통일한다. 정의와 확인된 관계는 직접 서술하고, 가정은 `We assume ...`, convention 선택은 `We define ...` 또는 `We use ...`처럼 선택임을 드러낸다. `we`는 이런 선택을 설명할 때 사용하며 독자에게 말을 거는 구어체나 작업 지시형 문장은 쓰지 않는다.

문단은 구체적인 물리적 대상에서 출발해 정의·수식·조건·의미를 연결한다. 각 문서에 이 순서를 기계적으로 반복하지는 않되, `The following quantity is defined`처럼 대상이 빠진 도입과 수식 뒤의 단순한 기호 재진술을 피한다. 짧은 문장으로 쓰면서도 결과를 이해하는 데 필요한 원인, 조건과 논리 연결은 남긴다.

| Avoid | Preferred |
|---|---|
| `This document explains the local-frame convention.` | `The local quantization axis follows the classical spin direction.` |
| `The following quantity is defined.` | `The coefficient of the local boson-number operator receives contributions from the incident bonds and the rotated field.` |
| `The terms obviously vanish.` | `The linear terms vanish when the reference configuration is stationary under the fixed-spin-length constraint.` |

목차 수준의 문서는 범위와 예정된 주제를 명료하게 쓰되, 존재하지 않는 계산을 이미 유도한 것처럼 표현하지 않는다. 작성 수준과 필요한 작업은 audit에서 추적하며, 결과의 의미에 영향을 주는 미확정 convention은 해당 수식 가까이에 제한 조건으로 남긴다.

### Language and Mathematical Prose

- Reader-facing theory body, heading, table, caption과 reference description은 영어로 작성한다.
- Heading과 opening paragraph는 문서 또는 section의 구체적인 물리적 대상과 범위를 앞에서 밝힌다. Markdown heading에는 수동 section number를 붙이지 않는다.
- Prose paragraph는 하나의 논리적 역할을 수행하고 Markdown source에서 한 physical line으로 작성한다. Editor의 soft wrap을 사용한다.
- Display equation은 주변 문장의 문법적 일부로 작성한다. 새 기호, 가정, approximation과 적용 조건은 필요한 지점에서 정의한다.
- Matrix, vector, operator, index, imaginary unit와 energy notation은 [Notation and Conventions](../../../docs/lswt/00-foundations/notation-and-conventions.md)를 따른다.
- Claim의 강도는 근거 수준을 넘지 않아야 하며, regime과 approximation은 해당 claim 가까이에 둔다.
- Primary note에 없는 claim은 확인된 scholarly source 또는 documented derivation으로 뒷받침한다. 외부 논문은 `External Sources`, 내부 유도는 `Internal Documents`에 기록하며 citation을 추측하거나 생성하지 않는다.
- Package, API, implementation plan, task status, agent note와 unresolved review ledger는 theory body에 넣지 않는다.
- Source conflict나 물리·수학적 불확실성을 문체 교정으로 해결하지 않는다. `Unknown` 또는 open review item을 보존하고 Human Physics and Mathematics Review로 넘긴다.

## Preferred Academic Style

- 각 문단은 일반적으로 핵심 주장이나 설정으로 시작하고, 넓은 배경에서 해당 문서가 소유하는 정의·Hamiltonian·유도로 빠르게 이동한다.
- 구체적인 물리량이나 조건을 쓸 수 있을 때 `This document explains ...` 또는 `the present formulation` 같은 document-centered meta-prose를 피한다.
- Definition, assumption, derived result, numerical evidence, interpretation과 limitation을 문장 차원에서 구분한다.
- 수식 앞에서 그 식의 역할을 밝히고, 수식 뒤에서는 좌변과 우변을 반복하기보다 물리적 의미, 조건, 결과 또는 다음 유도 단계를 설명한다.
- 같은 대상을 여러 동의어로 바꾸어 부르지 않는다. 모호한 `this`, `it`, `these` 대신 필요한 경우 물리량이나 식의 이름을 반복한다.
- `show`, `prove`, `verify`, `suggest`처럼 근거 수준을 나타내는 동사를 구분한다. `clearly`, `obviously`, `trivially`로 조건이나 논증을 생략하지 않는다.
- 이 corpus는 전문적인 theory note이므로 journal의 IMRaD 구조, novelty claim 또는 literature-gap narrative를 강제하지 않는다.

## Allowed Variations

- `Introduction`, `Discussion`, `References`, 구체적인 `Scope`와 `Limitations`처럼 역할이 명확한 관습적 heading을 사용할 수 있다.
- `we`는 convention 선택이나 유도 순서를 안내할 때 사용할 수 있다. 모든 section을 `In this section, we ...`로 시작하지 않는다.
- Overview와 해석 중심 문단에는 수식이 없어도 된다. 반대로 밀접하게 연결된 정의나 유도식은 하나의 aligned display에 함께 둘 수 있다.
- Heading capitalization은 document group 안에서 일관되게 유지하면 된다. 특정 journal의 typography를 canonical Markdown에 강제하지 않는다.
- 관련 수식을 독립적으로 참조하지 않는다면 하나의 logical group으로 둘 수 있다. Semantic equation identity와 renderer 규칙은 lifecycle 문서가 소유한다.

## Headings

Heading은 구체적인 주제, 주장, 물리적 regime 또는 계산 대상을 가능한 한 앞에 둔다. 하위 heading은 상위 section의 주제를 좁혀야 한다.

| Vague or document-centered | Concrete and front-loaded |
|---|---|
| `Scope of the Present Formulation` | `Bilinear Spin Hamiltonians` |
| `Conditions for Applying LSWT` | `Stationarity, Stability, and Small-Fluctuation Conditions` |
| `From Spins to Magnons` | `Holstein–Primakoff Expansion of Local Spin Operators` |

## Terminology Quick Reference

| Meaning | Preferred form |
|---|---|
| Collective excitation | `spin wave` |
| General theory | `spin-wave theory (SWT)` on first use; then `SWT` |
| Harmonic theory | `linear spin-wave theory (LSWT)` on first use; then `LSWT` |
| Named mappings and interactions | `Holstein–Primakoff`, `Dzyaloshinskii–Moriya`, `Luttinger–Tisza` with an en dash |
| Spatial domains | `real space`, `momentum space` as nouns; `real-space`, `momentum-space` as modifiers |
| Reciprocal-space region | `Brillouin zone`; use `BZ` only after definition |
| Magnetic periodicity | `magnetic unit cell`, not `unitcell` |
| Ordering wave vector | `single-$Q$ order` |

`phase`, `state`, `reference configuration`, `mode`, `band`, `representation`과 `approximation`은 stylistic synonyms가 아니다. 예를 들어 HP는 bosonic representation, LSWT는 harmonic approximation, magnon mode는 고정된 momentum의 eigenmode, magnon band는 momentum에 걸쳐 이어진 branch를 뜻한다. 더 상세한 symbol과 convention은 notation owner가 소유한다.

## Reference Recording

각 theory document는 authoring provenance를 위해 다음 구조의 `## References`로 끝난다. 이는 최종 journal bibliography가 아니라 source ledger다.

```markdown
## References

### Internal Documents

- [Document title](relative/path.md): relationship to the current document.

### External Sources

- Author, paper title, journal, year, and DOI: claim or equation supported by the source.
```

- `Internal Documents`에는 관련 theory owner와 convention owner를 기록한다.
- `External Sources`에는 본문 claim 또는 equation을 실제로 지원하는 published scholarly source를 기록한다.
- Primary LSWT note는 frontmatter의 `source`와 `source-section`이 가리키므로 반복하지 않는다.
- 외부 source가 없으면 빈 subsection을 유지하거나 `- None.`으로 표시할 수 있다.

## Final Check

Theory note를 사용자 검토에 올리기 전에 다음을 확인한다.

- 문서의 목적, 물리적 대상과 적용 범위가 opening에서 드러나는가?
- Heading과 문단이 구체적이며 논리적 순서를 따르는가?
- 수식, 기호, 가정, approximation과 물리적 의미가 연결되는가?
- Terminology와 notation이 owner document와 일치하는가?
- Claim의 강도와 조건이 근거 수준에 맞는가?
- Code detail, 작업 기록과 unresolved ledger가 theory body에서 분리됐는가?
- 외부 claim이 확인된 source와 연결되는가?

이 검사는 문체 검토이며 Human Physics and Mathematics Review 또는 theory acceptance를 의미하지 않는다.

## Related Governance and Style References

- [AGENTS.md](../../../AGENTS.md)
- [Documentation Map](../../../docs/lswt/README.md)
- [Notation and Conventions](../../../docs/lswt/00-foundations/notation-and-conventions.md)
- [Source Inventory](../../../docs/lswt/sources/README.md)
- [Single Knowledge Canon](../../User-Constitution/single-knowledge-canon.md)
- [LSWT Canonical Document Lifecycle](../procedures/lswt-canonical-document-lifecycle.md)
- [APS Style Basics](https://journals.aps.org/authors/style-basics)
- [Physical Review Style and Notation Guide](https://publish.aps.org/files/styleguide-pr.pdf)
