---
frontmatter-version: 1
title: LT 단계 구현과 CommensurateStructure 정리 여부 판단
section: issue-notes/open
issue-type: discussion
status: draft
last-edited-by: claude
created: 2026-09-30
updated: 2026-09-30
related:
  - GOVERNMENT/Working-Pad/issue-notes/open/260602-commensurate-structure-test-failures.md
  - GOVERNMENT/Working-Pad/idea-proposals/2026-09-23-spin-model-transfer-contract.md
  - code-space/spintoolkit/states/commensurate.py
  - code-space/spintoolkit/states/spin_state.py
  - research-space/theory/lswt/appendices/luttinger-tisza-method.md
---

# LT 단계 구현과 CommensurateStructure 정리 여부 판단

## 배경

2026-09-30 `CommensurateStructure` 테스트 실패(260602)를 테스트 수정으로 해결하는 과정에서
두 질문이 나왔다. 사용자가 이 둘을 필요 여부부터 판단하는 의제로 올리기로 했다.

사용자가 제시한 자기 셀 결정 흐름은 다음과 같다. 고전 대각화(Luttinger–Tisza, LT)로
셀을 정할 수 있으면 정하고, 정할 수 없으면 셀 제약 없이 푼다.

1. $J(\mathbf q)$를 대각화해 최소 고유값의 $\mathbf q^*$를 찾는다.
2. LT 해가 강한 제약 $|\mathbf S_i| = S$를 만족하고 $\mathbf q^*$가 commensurate하면, 그 주기가
   자기 셀 $M$을 정한다. 예를 들어 삼각격자 AFM은 $\mathbf q^* = K$, 셀은 √3×√3이다.
3. LT가 실패하면 후보 셀이나 큰 셀에서 고전 에너지를 수치 최소화한다. 비등가 부격자,
   결합 의존 이방성, 자기장이 있으면 대체로 실패한다.
4. $\mathbf q^*$가 incommensurate하면 유한 셀이 없다. rotating-frame LSWT, 큰 클러스터,
   real-space BdG 등으로 간다.

## 문제 정의

다음 두 가지를 결정해야 한다.

1. 도구에 LT 단계(위 1–2번)를 구현할지, 구현한다면 어느 단계에 둘지.
2. `CommensurateStructure`를 유지할지, `SpinState`로 흡수할지, 삭제할지.

## 본론

### 현재 상태 (2026-09-30, `0e9247f` 기준)

- **LT 단계**: 코드에 없다. 이론 부록 `research-space/theory/lswt/appendices/luttinger-tisza-method.md`는
  본문 이식 전 뼈대(draft)다.
- **셀 결정**: 사용자나 모델이 후보 셀을 준다. NBCP는 One–Four MSL을 정수 행렬 $M$으로
  정의하고(전달 규약 D16), 셀마다 고전 탐색을 한 뒤 `select_on_manifold`로 고른다.
  3번 경로에 해당한다.
- **4번 경로**: `IncommensurateStructure`는 미구현(`NotImplementedError`)이고 BdG solver도 없다.
- **`SpinState`**: 정수 행렬 $M$ 초격자를 쓴다. 고전 계산(`methods/classical.py`),
  상태 선택, LSWT 공통 경로가 모두 이것을 쓴다. 삼각격자 120° 벤치마크
  `models/heisenberg.py::state_120`은 이미 √3×√3 셀 `[[1, 1], [-1, 2]]`로 만든다.
- **`CommensurateStructure`**: Phase 1A(`6a991a9`)의 대각 $(n_1, n_2)$ 셀 컨테이너다.
  계산 경로에 연결되어 있지 않다. 남은 참조는 다음과 같다.
  - `spintoolkit.states` export
  - `lswt` 호환 매핑과 `test_import_compatibility.py`
  - `spintoolkit/README.md` 예시
  - `examples/phase1a_demo.py`: `SpinSystem(lattice=...)` 호출로 이미 실행되지 않는다.
  - `doc-space/examples/phase1a_demo.py`: 기본 저장소의 미커밋 개편에서 삭제되었다.

### 질문 1 — LT 단계

- **구현 안 함**: 지금처럼 후보 셀을 명시한다. NBCP처럼 이방성이 강한 모델에서는
  LT가 대부분 실패하므로 얻는 것이 적다.
- **진단 도구로 구현**: $J(\mathbf q)$ 최소와 강한 제약 만족 여부만 보고한다. 후보 셀을
  제안하는 보조 수단이며, 선택 단계는 바꾸지 않는다. Heisenberg·Kitaev 벤치마크의
  기대 셀을 독립적으로 확인하는 용도도 있다.
- **파이프라인 1단계로 구현**: LT 성공 시 셀과 상태를 자동으로 정한다. 강한 제약 판정,
  퇴화 $\mathbf q^*$ 처리, incommensurate 분기 규약이 필요해 범위가 크다.

### 질문 2 — `CommensurateStructure`

- **유지**: 현상 유지. 대각 셀 한계와 `SpinState` 안내는 docstring에 적었다.
- **`SpinState`로 흡수**: 필요한 기능(각도 기반 최적화 파라미터 등)만 `SpinState` 쪽으로
  옮기고 이 클래스는 deprecated alias로 둔다.
- **삭제**: 테스트, export, 호환 매핑, README 예시, 고장 난 `examples/phase1a_demo.py`를
  함께 정리한다. 옛 pickle 복원 여부를 확인해야 한다.

## 결론 / 미결 사항

미결. LSWT 작업 전체 멈춤(2026-09-30 결정)에 따라 재개 시 판단한다. 두 질문은 독립적이다.
질문 2는 코드 정리이고, 질문 1은 기능 범위 결정이다.

설계 검토면은 `docs/development` Beamer의 "미결 사항" 쪽이다(검토본 29).

## 참조

- `GOVERNMENT/Working-Pad/issue-notes/open/260602-commensurate-structure-test-failures.md` — 계기가 된 테스트 실패와 해결
- `GOVERNMENT/Working-Pad/idea-proposals/2026-09-23-spin-model-transfer-contract.md` — D16 정수 행렬 초격자
- `code-space/spintoolkit/states/commensurate.py`, `code-space/spintoolkit/states/spin_state.py`
- `code-space/spintoolkit/models/heisenberg.py` — `state_120`
- `research-space/theory/lswt/appendices/luttinger-tisza-method.md` — LT 이론 부록(draft)
