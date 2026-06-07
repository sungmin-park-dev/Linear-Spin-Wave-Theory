---
frontmatter-version: 1
title: Roadmap - General 2D Spin Solver
section: working-pad
status: in-review
last-edited-by: claude
created: 2026-06-07
updated: 2026-06-07
---

# 로드맵 — 범용 2D 스핀 시스템 Solver

> 구현 착수 전 전체 경로. "무엇을 먼저/왜"만 정한다.
> 표기: [확정] / [열림 = 각 Phase 착수 시 결정]
> 개별 작업 plan은 각 Phase 착수 시 별도 작성한다.

## 0. 최종 목표와 원칙

- **최종 목표:** 임의의 2D 스핀 모델을 여러 방법(LSWT·ED·TN·NQS)으로 풀고
  관측량을 계산하는 **범용 solver**.
  척추: `[시스템 정의] → [여러 solver] → [관측량/출력]`
- **원칙:**
  1. **단일 물리 골든은 없다.** 방법마다 유효영역이 다르다.
     - **ED** = 유한 클러스터 엄밀 → 같은 작은 계의 **코드-정확성 기준**
       (단 finite-size effect, thermodynamic limit 아님).
     - **LSWT** = ordered phase(준고전) 한정 근사.
     - thermodynamic-limit 물리는 finite-size scaling / TN·NQS로 보완.
  2. [열림] 공통 모듈(IR): `SpinSystem` 점진 진화 vs 새 구조 — 미확정(Phase 3).
  3. 솔버 2개(LSWT + ED 예정) 작동 후 IR 일반화.
  4. 단계마다 검증 → 성민 확인 → 다음 단계.

## 1. 현재 상태 (검증 대상, 단언 아님)

- LSWT 파이프라인 정상작동 **미검증**(CLAUDE.md 주장에만 근거) → Phase 0에서 확인.
- legacy가 NBCP 두 논문 재현 기록: arXiv:2505.06398 + npj Quantum Materials 2022
  (DOI 10.1038/s41535-022-00500-3).
- Geometry/자기구조 계층 구현됐으나 `SpinSystem` 미연결, commensurate 테스트
  5개 실패, `IncommensurateStructure` 미구현.
- 이론 문서 정본화 진행 중(codex 트랙).

## 2. 단계별 로드맵

### Phase 0 — 정리 · 검증 · 안전망
- [확정] arXiv 참조 수정 + codex 문서 reorg 체크포인트 + IR 논의 기록.
- [확정] **새 패키지/독립 파일 정상작동 검증** — NBCP 예제 실제 실행으로 LSWT
  파이프라인 동작 확인.
- [확정] **최소 회귀 안전망** — 정상작동 확인 후 ordered phase에서 결정적 회귀
  테스트로 동결(물리 골든 아님, 리팩터 회귀망).
- [확정] commensurate 실패 5개 triage — `120° = 3 sublattice?` 물리 결정.

### Phase 1 — ED 정확성 기준 확립
- 작은 **ED 솔버**(`AbstractSolver` 상속), 유한 클러스터 엄밀.
  [열림] 자체 구현 vs 공개 패키지(QuSpin 등).
- LSWT를 ordered phase에서 ED와 대조 → LSWT 유효영역 명문화.

### Phase 2 — LSWT 제품화 (ordered phase 한정)
- 관측량(thermo/topology/correlations) → `LSWTSolver` 배선.
- NBCP 재현 end-to-end(밴드·Chern·thermal Hall·열역학) + 밴드 플로터 포팅 +
  결정적 예제.

### Phase 3 — 공통 모듈(IR) 일반화 (ED + LSWT 경험 기반)
- [열림] `SpinSystem` 진화 방식 확정 → `Term`/`Geometry` 점진 일반화.
- 3-way 정의 스파이크(SpinSystem / NetKet / TeNPy)로 Term operator 표현 확정.

### Phase 4 — 확장
- TN/NQS exporter(NetKet/TeNPy), Geometry/자기구조 완전 연결, 시각화, 공개 배포.

### Phase 5 — Incommensurate 확장 (후기 별도)
- `IncommensurateStructure` 구현 + LSWT incommensurate solver.
  (Phase 4와 순서 [열림])

## 3. 병렬 지원 트랙 (비게이팅)

- 이론 문서 정본화(codex): 각 Phase에 convention 공급. Phase 완료를
  게이트하지 않는다.

## 4. 열린 결정 모음 (각 Phase 착수 시 확정)

- 공통 모듈: `SpinSystem` 진화 vs 새 구조 (Phase 3).
- 두 번째 솔버 ED 적합성 / 자체구현 vs QuSpin (Phase 1).
- Term operator 표현 numeric vs named (Phase 3 스파이크).
- Phase 4 / Phase 5 순서, 공개 배포 시점/범위.

## 배경 메모

- 이 로드맵은 2026-06-05~07 성민과의 방향성 논의에서 도출됐다.
- 공통 SpinModel IR 논의 기록은 별도(추후 `idea-proposals/2026-06-04-general-spin-model-ir.md`에 review로 반영 예정).
