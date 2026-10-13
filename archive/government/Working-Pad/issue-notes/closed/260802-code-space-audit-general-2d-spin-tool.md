---
frontmatter-version: 1
title: code-space audit against general 2D spin-system tool target
section: issue-notes/closed
issue-type: review
status: closed
resolution: superseded
outcome: docs/development/README.md
last-edited-by: claude
created: 2026-08-02
updated: 2026-09-30
closed: 2026-09-30
related: handoff/closed/260603-next-chat-general-2d-spin-tool.md
---

# code-space audit against general 2D spin-system tool target

`handoff/closed/260603-next-chat-general-2d-spin-tool.md`가 요청한 2단계
("Audit Current Code Against Target")의 결과다. 아직 리팩터는 하지 않았다 —
분류와 사실관계 확인만 했다. `code-space/lswt/`, `examples/`,
`legacy/`를 대상으로 했다.

## 분류 기준

핸드오프가 지정한 5개 범주: Keep as general core / Keep as LSWT-specific /
Move-rename / Deprecate / Needs physics review.

> 경로 안내(2026-09-29): 아래 절 제목은 2026-08-02 감사 당시의 구조다. 현재 대응 위치는
> `core/` → `spintoolkit/system/`·`states/`·`methods/lswt/diagonalization.py`,
> `solvers/` → `spintoolkit/methods/`(`optimization.py`, `lswt/`), `config.py` → `spintoolkit/definitions/`이다.

## `code-space/lswt/core/`

| 파일 | 분류 | 근거 |
|---|---|---|
| `spin_system.py` (`SpinSystem`) | **General core** | 이미 solver-agnostic 설계(CLAUDE.md 원칙). LSWT 전용 로직 없음 |
| `exchange.py` | **General core** | `heisenberg`/`xxz`/`dzyaloshinskii_moriya`/`kitaev`는 모델 범용. `bond_angle_exchange`/`nnn_exchange`는 NBCP 동기로 추가됐지만 함수 자체는 범용 3×3 행렬 빌더 |
| `brillouin_zone.py` (`BrillouinZone`) | **General core** | 순수 2D reciprocal-space 격자 기하. Bosonic BdG에 종속되지 않음 |
| `lattice/base.py`, `lattice/presets.py` | **General core, 미연결** | `AbstractLattice`, `TriangularLattice`/`SquareLattice`/`HoneycombLattice` 구현됨. `grep`로 확인: **`spin_system.py`는 이 모듈을 import하지 않는다** — `SpinSystem.lattice_vectors`는 그냥 raw array. CLAUDE.md 미완료 목록의 기존 항목과 일치 |
| `magnetic_structure/base.py`, `incommensurate.py` | **General core, 미연결** | 위와 동일하게 `SpinSystem`과 연결 안 됨 |
| `magnetic_structure/commensurate.py` | **Needs physics review** | 위 미연결 문제에 더해, `260602-commensurate-structure-test-failures.md`의 pytest 실패 5건 미해결. 120도 구조를 magnetic sublattice 몇 개로 표현할지 API 결정이 먼저 필요 — 이 audit에서 임의로 판단하지 않음 |
| `diagonalization.py` (`Diagonalizer`, Colpa) | **LSWT-specific** | Bosonic BdG 대각화는 LSWT류 방법 전용 수치 기법. ED/TN/NQS에는 해당 없음 |

## `code-space/lswt/solvers/`

| 파일 | 분류 | 근거 |
|---|---|---|
| `base.py` (`AbstractSolver`, `SolverResult`) | **General core** | 공통 솔버 인터페이스, 이미 이 목적으로 설계됨 |
| `hamiltonian.py` (`LSWTHamiltonian`) | **LSWT-specific, broader physics review pending** | 2026-09-10 B/B† 기존 수정과 운동량 미분을 독립 회귀 테스트로 확인했다. [종결 이슈](../closed/260802-hamiltonian-b-block-substitution-bug.md) 참조. 이론 A1과 다른 물리 범위는 별도 검토다. |
| `solver.py` (`LSWTSolver`) | **LSWT-specific** | 위 solver들을 orchestrate하는 진입점 |
| `energy.py` (`EnergyFunction`) | **Move/refactor 후보** | 분류 확인: `__init__(self, spin_sys_data, N, ...)`가 `SpinSystem`이 아니라 legacy dict를 받음. 고전 에너지 평가 자체는 모델에 상관없이 범용이지만, 현재 API가 legacy dict에 묶여 있음. CLAUDE.md 미완료 목록에 이미 있는 항목("`EnergyFunction` → `SpinSystem` 직접 수용") — 여기서는 재확인만 |
| `optimizer.py` (`SpinOptimizer`) | **Move/refactor 후보** | `EnergyFunction`과 같은 legacy dict 경로에 묶여 있어 같은 리팩터가 필요. 알고리즘(`scipy.optimize`) 자체는 범용 |

## `code-space/lswt/observables/`

| 파일 | 분류 | 근거 |
|---|---|---|
| `bose_statistics.py` | **General core 후보** | `SpinSystem`이나 `LSWTSolver`를 import하지 않음. `lswt.config` 상수만 사용하는 순수 함수 모음(Bose-Einstein distribution, kernel 함수). Bosonic quasiparticle을 다루는 어떤 방법에도 재사용 가능 |
| `thermodynamics.py`, `correlations.py` | **LSWT-specific, 미연결** | `grep` 확인: 세 파일 모두 `LSWTSolver`/`SolverResult`를 import하지 않음 — CLAUDE.md 미완료 목록의 "Observables 모듈 ↔ LSWTSolver 연결 (TODO 주석 상태)"과 일치. 코드는 있지만 실제 결과 파이프라인에 연결 안 됨 |
| `topology.py` | **LSWT-specific, needs physics review, 미연결** | 위와 같은 연결 문제 + CLAUDE.md 기록된 알려진 버그(Thermal Hall `real_space_volume`, `self.Ns` 곱셈 누락과 단위 변환 계수 오류) |

## `code-space/lswt/visualization/`

| 파일 | 분류 | 근거 |
|---|---|---|
| `spin_plotter.py` | **General core** | `SpinSystem` 객체를 직접 그리는 함수들. LSWT 결과(spectrum 등)가 아니라 시스템 정의 자체를 시각화하므로 범용 |

## `code-space/lswt/config.py`

**General core.** 물리 상수와 기본값. 모델·솔버에 안 묶임.

## `examples/`

| 파일 | 분류 | 근거 |
|---|---|---|
| `nbcp_ground_state.py`, `nbcp_hamiltonian_check.py`, `phase1a_demo.py` | **Keep as LSWT-specific** | NBCP 재현이 목적인 worked example. 셋 다 `system.to_legacy_dict()` 경유 — `EnergyFunction`/`optimizer.py` 리팩터가 먼저 끝나야 이 예제들도 갱신 가능 |
| `__pycache__/nbcp_ground_state.cpython-313.pyc` | **Deprecate/cleanup** | 커밋된 생성물. `.gitignore` 규칙 확인 필요 (별도로 사소한 정리, 이 audit에서 실행하지 않음) |

## `legacy/`

**Keep as-is, 재확인만.** `modules/`, `scripts/`, `research-notes/`는 `code-space`/`examples`
어디서도 import되지 않음 (이전 대화에서 확인된 사실 재확인, 이번에 추가 조사 안 함). 이름에
`-space` 접미사가 없는 것은 이미 알려진 사소한 불일치이며 우선순위 낮음.

## 종결 (2026-09-30, 사용자 승인)

toolkit 0–5단계의 공통 경로가 아래 핵심 발견 세 가지를 대체했다.

1. 격자·자기 구조가 `SpinSystem`에 연결되지 않은 문제 → 공통 자료형 `SpinModel`(격자·사이트·항),
   `SpinState`(자기 초격자와 방향), `CalculationGeometry`(1단계). 대각 셀 전용 `CommensurateStructure`의
   유지·흡수·삭제는 `issue-notes/open/260930-lt-step-and-commensurate-structure-decision.md`로 옮겼다.
2. Observables가 `LSWTSolver`와 연결되지 않은 문제 → `solve_lswt`의 `LSWTResult`가 대각화를 보관하고
   열역학(4b), 구조인자(4c), 위상량(5단계)이 그것을 재사용한다.
3. `EnergyFunction`/`SpinOptimizer`가 legacy dict를 받는 문제 → `classical_energy(model, state)`와 상태 선택의
   `LSWTZeroPointEnergy`(1·2b단계). 기존 경로(`LSWTSolver`, `EnergyFunction`/`SpinOptimizer`, SI Hall API)의
   정리 시점은 `docs/development` Beamer의 미결 사항에서 추적한다.

테스트 기준은 전체 통과(500개, 2026-09-30)로 정리되었다. 아래 본문은 당시 점검 기록으로 보존한다.

## 핵심 발견 — 우선순위용

1. **"General core"로 분류된 `lattice/`, `magnetic_structure/`가 `SpinSystem`과
   실제로 연결돼 있지 않다.** 코드는 존재하지만 `SpinSystem`은 그냥 raw
   `lattice_vectors` 배열을 받을 뿐, `TriangularLattice` 같은 preset이나
   `CommensurateStructure`를 사용하지 않는다. "범용 2D spin-tool"의 기반이 될
   추상화가 이미 구현됐는데 배선만 안 된 상태 — 이게 scope 정의(핸드오프 1단계)
   전에 먼저 눈에 띄는 가장 큰 구조적 간극이다.
2. **Observables 3개 모듈(`thermodynamics`, `correlations`, `topology`)이
   `LSWTSolver`와 연결 안 됨.** 계산 로직은 있지만 `solver.solve()` 결과에서
   자동으로 얻을 수 있는 경로가 없다.
3. **`EnergyFunction`/`SpinOptimizer`가 `SpinSystem`이 아니라 legacy dict를
   받는다.** 범용화하려면 이 경계부터 정리해야 한다.
4. 이 audit의 범용화 분류는 물리식이나 API 수정이 아니다. 2026-09-10
   `hamiltonian.py`의 B/B† 구현 이슈는 기존 수정 검증과 회귀 테스트 추가로
   종결했다. `commensurate.py` 테스트 실패, `topology.py` 이슈와 이론 A1은
   각자의 검토 범위에 남는다.

## 다음 단계 제안 (결정 아님)

핸드오프의 3단계("Resolve Test Baseline")와 4단계("Propose Architecture")로
넘어가기 전에, 위 "핵심 발견" 3개 중 어느 것부터 다룰지 성민 확인이 필요하다.
이 audit은 그 결정을 위한 입력이지 결정 자체가 아니다.
