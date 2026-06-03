---
frontmatter-version: 1
title: 범용 2D 스핀 시스템 도구 전환 노트
section: idea-proposals
status: draft
created: 2026-06-03
updated: 2026-06-03
last-edited-by: codex
---

# 범용 2D 스핀 시스템 도구 전환 노트

이 문서는 이 프로젝트를 LSWT 중심 패키지에서 범용 2D 스핀 시스템 시뮬레이션 도구로 옮겨가기 위해 함께 정리하는 작업 노트다.

아직 아키텍처를 확정하려는 문서가 아니다.
코드 마이그레이션을 시작하기 전에 작업 방향, 질문, 코멘트, 결정사항을 한곳에 모으기 위한 공유 문서다.

## 출발점

현재 작업 방향:

- LSWT는 첫 번째 solver 기능으로 유지하되, 프로젝트 전체 정체성이
  LSWT에만 묶이지 않도록 한다.
- 범용 도구는 2D lattice, spin configuration, interaction, solver workflow,
  물리적으로 의미 있는 output을 중심으로 잡는다.
- 물리 convention은 코드 주석이나 구현 계획에만 두지 않고, 이론 문서
  안에서 설명과 함께 정리한다.
- 큰 API 결정이나 물리 의미 결정은 refactoring 중에 암묵적으로 하지
  않는다.

성민 코멘트:

- 현재 장기 계획을 알려줄테니, 이에 따라 작업을 진행하는 것이 좋을 것 같아.
- 현재 장기 계획은 LSWT 외에 Monte Carlo simulation, Tensor-Network, Neural Quantum State에 대한 시뮬레이션 도구로 도구의 범위를 확장하려고해.
- 그중 LSWT 문서는 여러 시뮬레이션 도구가 들어왔을 때, 기능이 제대로 구현이 되었는지 검토할 수 있는 벤치마크로서 확실한 기능을 해야해.
- 따라서 현재 중요한 것은 두 가지 관점에서 작업을 진행할 필요가 있어.
  1. Spin-Wave Theory 이론 문서 정리
  2. Spin-Wave Theory 파일 구조 정리
    1. 이 중 격자가 가장 중요해.
    2. 격자 구조 상에서 정의된 스핀 시스템에 대해서 시뮬레이션을 할 때, Spin-Wave, Monte-Carlo, TN, 나아가 nqs 에 공통으로 사용될 수 있도록 공통 데이터 처리 방식 및 구조를 정의해야해.
    3. 그런데, 문제는 격자의 자유도가 꽤 큰거 같아서 이를 어디까지 포함해서 격자를 만들지가 고민이야. 일단 현재 spin-wave theory commensurate phase에 집중해 있는데 incommensurate까지 확장할 수 있을지, skrymion 등을 포함하게 할 수 있을지 등을 논의해야해.
    4. 이러한 격자의 컨벤션과 범위는 모두 문서를 읽으면서 정리하려고해.



## 왜 이론 문서가 중요한가

이번 전환에서 나오는 여러 질문은 단순한 구현 문제가 아니라 이론적 정의
문제다. Hamiltonian, Fourier convention, Brillouin-zone folding, 그리고
서로 다른 solver가 같은 system을 어떻게 해석하는지에 직접 영향을 준다.

따라서 이런 convention은 이론 문서 안에서 수식과 예제와 함께 설명하고,
코드와 테스트는 그 문서를 참조하도록 만드는 것이 좋다.

성민 코멘트:

-

결정:

- 우선 LSWT theory 문서를 정본화한다.
- `common/` convention 문서는 먼저 추상적으로 만들지 않고, LSWT 문서 정리
  과정에서 여러 solver가 공유해야 하는 정의가 확인될 때 추출한다.
- LSWT 문서 작업 중 프로젝트 범위나 `common/`에 들어갈 후보가 보이면
  Codex가 먼저 지적하고, 성민 확인 후 working note 또는 정본 문서에
  반영한다.

## 정리해야 할 Convention 질문들

아래 항목들은 최종 목차가 아니라 논의를 시작하기 위한 축이다.

### Site Indexing

질문: physical site, crystallographic unit cell, basis site, magnetic sublattice의
관계를 어떻게 정의할 것인가?

다듬어볼 수 있는 표현:

```text
physical site = (unit_cell, basis_site)
magnetic_sublattice = structure_map(unit_cell, basis_site)
```

성민 코멘트:

-

결정:

-

### Link And Coupling Counting

질문: Hamiltonian에서 bond/link를 어떻게 셀 것인가?

정리할 것:

- 각 physical link를 한 번만 넣을 것인지, symmetric pair를 모두 넣고
  `1/2` factor를 둘 것인지.
- `displacement`가 real-space displacement인지, fractional cell displacement인지.
- intra-cell coupling과 inter-cell coupling을 모호하지 않게 표현하는 방법.

성민 코멘트:

-

결정:

-

### Magnetic Structure

질문: magnetic sublattice의 개수는 무엇이 결정하는가?

현재 열린 문제:

- 현재 `CommensurateStructure` 구현은 basis 개수와 magnetic supercell로
  sublattice 개수를 계산한다.
- 일부 테스트는 angle array 자체가 magnetic sublattice 개수를 정할 수도
  있다는 식으로 작성되어 있다.

이 부분은 코드를 고치기 전에 theory/API convention으로 먼저 정리해야 한다.

성민 코멘트:

-

결정:

-

### Brillouin Zone

질문: crystallographic BZ와 magnetic BZ를 어떻게 구분할 것인가?

정리할 것:

- solver가 기본적으로 어느 BZ를 sampling해야 하는지.
- magnetic supercell이 BZ를 어떻게 folding하는지.
- folding 이후 band path와 high-symmetry point를 어떻게 정의하는지.

성민 코멘트:

-

결정:

-

### Solver Boundary

질문: generic `SpinSystem`에 들어갈 것과 LSWT 같은 특정 solver 내부에
들어갈 것을 어떻게 나눌 것인가?

논의할 초기 경계:

- Generic domain: lattice, site, interaction, magnetic structure, field,
  여러 solver가 공통으로 필요로 하는 metadata.
- LSWT-specific domain: Holstein-Primakoff expansion, local-frame rotation,
  bosonic BdG Hamiltonian, Colpa diagonalization, magnon observable.

성민 코멘트:

-

결정:

-

## 당분간의 작업 제약

이 노트가 더 명확해질 때까지:

- 패키지명을 `lswt`에서 바꾸지 않는다.
- legacy code를 제거하지 않는다.
- magnetic supercell 또는 magnetic sublattice 의미를 코드에서 먼저
  재정의하지 않는다.
- 현재 pytest failure를 단순한 우발적 실패로 보지 않는다. 실제 convention
  충돌을 드러내는 신호로 취급한다.
- theory/API convention 정리 방향이 잡히기 전에는 큰 refactor를 시작하지
  않는다.

성민 코멘트:

-

결정:

-

## 다음 작업

이 노트를 바탕으로 LSWT theory 문서를 어디에 둘지 먼저 결정한다.
그 다음 LSWT 문서 정리 과정에서 공통 convention 후보를 뽑는다.

후보 위치:

```text
research-space/theory/lswt/
```

성민 코멘트:

-

결정:

- LSWT 문서를 먼저 정리한다.
- 공통 convention은 LSWT 문서 정리 이후 `common/` 후보로 분리한다.
- LSWT를 정리하면서 발견되는 공통 후보를 누적 기록하고, 충분히 명확해진
  항목만 `common/`으로 승격한다.
- 적극적으로 수정할 LSWT 정본 문서는 `research-space/theory/lswt/`에 새로
  정리한다.
- LSWT 정본화 중 계속 대조할 원자료는 `research-space/sources/lswt/`에 둔다.
- 거의 보지 않을 과거 LaTeX/PDF 자료는 `legacy/research-notes/lswt/`에 둔다.
