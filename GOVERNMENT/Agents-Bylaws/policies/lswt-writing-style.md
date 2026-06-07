---
frontmatter-version: 1
title: LSWT Writing Style Policy
section: policies
status: in-review
last-edited-by: codex
created: 2026-06-04
updated: 2026-06-04
---

# LSWT Writing Style Policy

이 문서는 LSWT 정본 문서를 작성하고 교정할 때 따르는 문장 스타일 기준이다.
일반적인 글쓰기 skill이 아니라, 현재 LSWT theory 문서 작업에 적용하는 repo
policy다.

## Scope

적용 대상:

- `research-space/theory/lswt/**/*.md`
- LSWT theory 문서와 직접 연결되는 검증 메모, convention 메모
- LSWT theory 문서로 승격될 Working-Pad 초안

비적용 대상:

- 코드 주석과 docstring
- 운영 handoff, issue note, task queue
- 사용자와의 대화문

## Source Style

문체 기준은 두 자료를 함께 본다.

| Source | Role |
|---|---|
| `/Users/david/Downloads/Linear_Spin_Wave_Theory___Note.pdf` | 계산 노트형 유도 흐름, notation, review note |
| `/Users/david/Downloads/PhysRevB.111.075167.pdf` | 출판 논문형 claim control, 적용 조건, 증거 수준 구분 |

LSWT 정본 문서는 계산 노트의 명확성을 유지하되, claim의 범위와 증거 수준은
논문식으로 조심스럽게 관리한다.

## Sentence Principles

1. 각 파일은 먼저 그 파일이 고정할 물리적 대상이나 유도 단계를 밝힌다.
2. 넓은 배경 설명은 짧게 두고, 가능한 한 빨리 정의, Hamiltonian, 또는 유도
   대상에 들어간다.
3. 수식은 논리의 중심에 둔다. 다만 독자가 수식의 역할을 모를 때는 짧은
   안내문을 먼저 둔다.
4. 새 기호는 도입 직후 정의하되, `where`를 기본 템플릿으로 반복하지 않는다.
   `let`, `denote`, `defined as`, `with`, `Here`, appositive phrase, 표를
   상황에 맞게 쓴다.
5. claim에는 적용 조건을 함께 둔다. 예를 들어 `for commensurate ordered
   states`, `within LSWT`, `when the quadratic Hamiltonian is positive
   definite` 같은 제한을 생략하지 않는다.
6. 수식 뒤에는 그 식이 어떤 역할을 하는지 한 문장으로 설명한다. 감상적
   표현이나 과장된 의미 부여는 피한다.
7. 같은 전환문을 반복하지 않는다. `In this section`, `Note that`,
   `Therefore`, `Thus`는 실제 전환이 있을 때만 쓴다.
8. proof, derivation, conjecture, numerical evidence, review issue, open
   question을 문장 차원에서 구분한다.

## Preferred Flow

문단의 기본 흐름은 다음과 같다.

```text
purpose or setup -> equation or definition -> symbol explanation -> condition -> meaning or next step
```

이 흐름은 편집 원칙이지 문장 템플릿이 아니다. 모든 문단을 같은 구조로
반복하지 않는다.

## Definitions and Notation

좋은 정의 방식:

```text
Let I = (n, mu) denote a physical spin site, with n the magnetic unit-cell
index and mu the magnetic sublattice index.
```

```text
The bond displacement is defined as Delta_l = r_J - r_I.
```

피해야 할 방식:

```text
where I is ..., where n is ..., where mu is ...
```

`where`는 긴 equation 뒤에 핵심 기호를 한 번 정리할 때만 쓴다. 매 수식마다
반복하지 않는다.

## Claim Control

본문 claim은 다음처럼 범위를 분명히 둔다.

| Weak or risky | Preferred |
|---|---|
| `The Hamiltonian can be written in momentum space.` | `For commensurate ordered states, the quadratic Hamiltonian can be written in momentum space.` |
| `The zero-point energy is ...` | `With the convention used here, the zero-point correction is ...` |
| `This term vanishes.` | `This term vanishes when the classical configuration is an extremum of the constrained classical energy.` |

검증되지 않은 source note의 문장은 canonical claim으로 승격하지 않는다.

## LSWT-Specific Rules

- 원본 PDF에 없는 내용을 본문 claim으로 추가하지 않는다. 필요한 경우
  `작업 메모`, `검증 메모`, 또는 `Common 후보 메모`로 분리한다.
- Review note에서 지적된 오류나 불확실성은 본문에서 해결한 척하지 않는다.
  검증 전에는 명시적으로 열린 문제로 둔다.
- Convention 문제는 가능한 한
  `research-space/theory/lswt/foundations/notation-and-conventions.md`와
  연결한다.
- Code discrepancy는 이론 본문에 섞지 않고 `검증 메모`에 기록한 뒤,
  별도 code verification 작업에서 처리한다.
- 한국어 문서에서도 technical term은 필요한 경우 영어로 유지한다. 단,
  영어 term 나열 때문에 문장이 불완전해지지 않게 한다.

## Anti-Patterns

피해야 할 문장 습관:

- `where`를 매 수식마다 반복하는 것
- 모든 절을 `In this section, we ...`로 시작하는 것
- 결론이 아닌 곳에 `Therefore`를 붙이는 것
- `clearly`, `obviously`, `intuitively`로 검증되지 않은 단계를 넘기는 것
- review issue가 남은 식을 canonical equation처럼 쓰는 것
- 본문에 작업 계획, 구현 전략, 또는 agent note를 섞는 것

## Related Documents

- `research-space/theory/lswt/README.md`
- `research-space/theory/lswt/map-lswt.md`
- `research-space/theory/lswt/foundations/notation-and-conventions.md`
- `GOVERNMENT/Agents-Bylaws/policies/frontmatter-policy.md`
