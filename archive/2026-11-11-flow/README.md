# 옛 흐름 점과 끝나지 않은 맡긴 일 (2026-11-11 보관)

2026-11-11에 사용자가 정해 옮긴 보존본이다. 더 고치지 않는다.

- 왜: 이 점과 일은 에이전트가 만들고 적은 기록이고, 사람이 판단하거나 승인한 것이 하나도 없다. 새 일은 앱의 "문제"(`workbench/problems/`) 아래에서 하므로, 화면과 에이전트 요약에 옛 기록이 섞이지 않게 옮겼다(research-workspace 결정 "흐름과 연구 프로젝트 재설계", 2026-11-11).
- 여기 적힌 상태("끝남 · 해결" 등)는 옮길 때 파일에 적혀 있던 값이다. 확인된 결과라는 뜻이 아니다.
- 노트(`workbench/topics/`, `workbench/notes/`)는 옮기지 않았다. 계산과 유도의 내용은 노트에 있다.
- 되살리려면 파일을 원래 자리로 옮긴다.

## 옛 경로 → 새 경로

| 옛 경로 | 새 경로 | 수 |
|---|---|---|
| `workbench/flow/<점>.yaml`, `layout.yaml` | `archive/2026-11-11-flow/flow/` | 67 |
| `workbench/comments/flow-<점>.md` (점의 기록) | `archive/2026-11-11-flow/comments/` | 2 |
| `workbench/tasks/<일>.md` (끝나지 않은 일) | `archive/2026-11-11-flow/tasks/` | 8 |

## 흐름 점 67개

### Worked Examples (20)

| 점 | 종류 | 적힌 상태 | 제목 | 근거 파일 |
|---|---|---|---|---|
| `materials-q` | 질문 | 진행 | 스핀파 이론과 spin-toolkit을 실제 물질의 상도·들뜸 문제에 적용해 논문이 될 메시지를 찾을 수 있는가 | `docs/reviews/2026-10-21-na3ni2bio6-nematic.md` |
| `nbcp-app-q` | 질문 | 진행 | NBCP의 Y·V 상은 U(1)을 깨는 비등방성 아래에서도 유한온도 spin supersolid로 남는가 | `data-space/verification/260912-pseudo-goldstone`<br>`workbench/decisions/2026-10-09-nbcp-repository-split.md` |
| `nbcp-z6-finding` | 명제 | 끝남 · 해결 | 순수 PD 축에서 V는 U_π 대칭 때문에 각도 조화가 6의 배수뿐이다 | `data-space/verification/260912-pseudo-goldstone` |
| `nbcp-rau-finding` | 명제 | 끝남 · 해결 | 논문은 공선 질서용 Rau gap 식을 비공선 Y·V에 적용했다 | `data-space/verification/260912-pseudo-goldstone` |
| `nbcp-one-over-s-finding` | 명제 | 끝남 · 해결 | S = 1/2에서 1/S 전개는 유사 Goldstone gap도 선택 방향도 정하지 못한다 | `workbench/topics/toolkit-development/dev-decisions/b-decisions-references.tex` |
| `worked-examples-calc` | 수치 계산 | 끝남 · 해결 | 표준 모형 네 개를 닫힌 식으로 풀고 노트의 숫자를 테스트로 묶는다 | `code-space/tests/test_worked_examples.py`<br>`workbench/topics/worked-examples/lswt-example-square-antiferromagnet/note.md`<br>`workbench/topics/worked-examples/lswt-example-triangular-antiferromagnet/note.md` |
| `honeycomb-sign-ana` | 이론 계산 | 끝남 · 해결 | 벌집 DM 강자성 마그논 Chern 수의 부호를 해석적으로 유도한다 | `workbench/topics/worked-examples/lswt-example-honeycomb-topological-ferromagnet/note.md` |
| `bcao-app-q` | 질문 | 끝남 · 해결 | BaCo₂(AsO₄)₂의 자기장 상도를 스핀파로 그리면 논문이 될 새 질문이 있는가 | `docs/reviews/2026-10-21-bcao-field-phase-diagram-survey.md`<br>`workbench/tasks/2026-10-21-theory-reliability.md` |
| `bcao-literature` | 문헌 | 끝남 · 해결 | BCAO 자기장 상도의 실험과 이론 선행 연구를 조사한다 | `docs/reviews/2026-10-21-bcao-field-phase-diagram-survey.md` |
| `bcao-maksimov-finding` | 명제 | 끝남 · 해결 | Maksimov 등의 J3 = 0.45 UUD → 이중 지그재그 경계(Jz± ≈ 0.9)는 같은 모형으로 재현되지 않는다 | `workbench/tasks/2026-10-21-theory-reliability.md` |
| `bcao-u1-finding` | 명제 | 끝남 · 해결 | c축 장에서 BCAO의 지그재그 계열 상태는 J±±에 고정되어 NBCP 같은 우연한 U(1)이 없다 | `docs/reviews/2026-10-21-bcao-field-phase-diagram-survey.md` |
| `bcao-verdict-finding` | 명제 | 끝남 · 해결 | BCAO 장 상도는 이미 정해진 갭 있는 공선 상들이라 새 논문 질문이 없어 접는다 | `docs/reviews/2026-10-21-bcao-field-phase-diagram-survey.md` |
| `nnbo-app-q` | 질문 | 진행 | Na₃Ni₂BiO₆의 c* 장 정체기·계단과 포화 직전 숨은 상(8.35–10.5 T)은 무엇인가 | `docs/reviews/2026-10-21-na3ni2bio6-nematic.md` |
| `nnbo-cascade-q` | 질문 | 미해결 | 사슬 배열 축퇴를 푸는 항이 1/3 정체기와 1/2·2/3 계단을 Kitaev 없이 설명하는가 | 없음 |
| `nnbo-chain-degeneracy-finding` | 명제 | 끝남 · 해결 | 곱 상태에서 포화장 h_s = J1 + 3J3의 사슬 배열은 거시적으로 축퇴한다 | `data-space/verification/261021-na3ni2bio6-multimagnon/collin.py` |
| `nnbo-multimagnon-finding` | 명제 | 끝남 · 해결 | 포화 직전에는 쌍보다 4마그논 덩어리가 먼저 불안정해져 쌍 네마틱이 생기지 않는다 | `data-space/verification/261021-na3ni2bio6-multimagnon/nmag.py` |
| `nnbo-multipolar-q` | 질문 | 멈춤 | J1–J3–D 육각격자에서 4마그논 결합 상태가 응축하는 다극자 상이 있는가 | 없음 |
| `nnbo-nematic-q` | 질문 | 끝남 · 해결 | 포화 직전 숨은 상은 두 마그논 응축으로 생긴 스핀 네마틱인가 | `docs/reviews/2026-10-21-na3ni2bio6-nematic.md` |
| `nnbo-refit-calc` | 수치 계산 | 끝남 · 해결 | S = 1 U(1) 핵심 모형을 실험 포화장과 Curie–Weiss 온도에 다시 맞춘다 | `data-space/verification/261021-na3ni2bio6-multimagnon/fitpt.py` |
| `nnbo-two-magnon-calc` | 수치 계산 | 끝남 · 해결 | 포화장 위 두 마그논 문제를 정확히 풀어 쌍이 1차 붕괴를 이기는 D 상한을 구한다 | `data-space/verification/261021-na3ni2bio6-multimagnon/twomag.py` |

### Observables from Spin Waves (14)

| 점 | 종류 | 적힌 상태 | 제목 | 근거 파일 |
|---|---|---|---|---|
| `thermal-hall-q` | 질문 | 진행 | thermal Hall 계산의 면적·단위·반환 기준이 맞는가 | `workbench/tasks/2026-08-02-thermal-hall-real-space-volume.md` |
| `topology-q` | 질문 | 진행 | Berry 곡률·Chern 수·열 Hall 전도도를 언제 정의할 수 있고 어떤 관례로 계산하는가 | 없음 |
| `fhs-kubo-criteria` | 명제 | 끝남 · 해결 | Chern 수는 FHS와 Kubo가 일치할 때만 받는다 (D31) | `workbench/topics/toolkit-development/dev-decisions/b-decisions-references.tex` |
| `bloch-convention` | 명제 | 끝남 · 해결 | thermal Hall은 full-position Bloch 규약에서만 물리적이다 | `docs/reviews/2026-10-01-lswt-draft-physics-review-guide.md`<br>`workbench/topics/spin-wave-observables/lswt-magnon-topology/note.md` |
| `thermal-m-q` | 질문 | 끝남 · 해결 | 유한 온도 자화를 기운 상태에서 어떻게 정의하는가 | 없음 |
| `goldstone-m-finding` | 명제 | 진행 | 2D Goldstone 상에서도 M = −∂F/∂h는 유한하다 | `workbench/topics/spin-wave-observables/lswt-magnon-thermodynamics/note.md` |
| `thermal-m-ana` | 이론 계산 | 끝남 · 해결 | 유한 온도 자화 M(h,t) = −∂F/∂h와 그 분해식 (D49) | `code-space/examples/verification/lswt_d49_thermal_magnetization_check.py`<br>`workbench/topics/spin-wave-observables/lswt-magnon-thermodynamics/note.md` |
| `goldstone-points-ana` | 이론 계산 | 끝남 · 해결 | Goldstone 점에서 Berry 곡률과 κ: B = 0과 B ≠ 0 | `code-space/examples/verification/lswt_bdg_blocks_check.py`<br>`workbench/topics/spin-wave-observables/lswt-magnon-topology/note.md` |
| `gapless-berry-q` | 질문 | 끝남 · 해결 | 갭이 없는데 Berry 곡률과 Chern 수를 계산할 수 있는가 | 없음 |
| `gapped-limit-finding` | 명제 | 끝남 · 해결 | Goldstone 띠의 Chern 수는 작은 갭을 연 극한으로 정의한다 | `code-space/examples/verification/lswt_bdg_blocks_check.py`<br>`workbench/topics/spin-wave-observables/lswt-magnon-topology/note.md` |
| `goldstone-chern-proof-q` | 질문 | 끝남 · 해결 | Goldstone 띠의 Chern 수가 정수라는 일반 증명이 있는가 | `code-space/examples/verification/lswt_goldstone_chern_check.py`<br>`workbench/topics/spin-wave-observables/lswt-magnon-topology/note.md` |
| `loop-phase-ana` | 이론 계산 | 끝남 · 해결 | 적분 가능성과 정수성은 다르다: Goldstone 점 둘레 Berry 위상 | `code-space/examples/verification/lswt_bdg_blocks_check.py`<br>`workbench/topics/spin-wave-observables/lswt-magnon-topology/note.md` |
| `two-magnon-calc` | 수치 계산 | 끝남 · 해결 | 종방향 두 마그논 연속체를 O(S⁰)에서 계산한다 (D56) | `code-space/tests/test_observables/test_two_magnon.py`<br>`workbench/topics/spin-wave-observables/lswt-spin-correlations/note.md` |
| `zero-mode-weight` | 명제 | 끝남 · 해결 | 영모드 운동량의 한 마그논 가중치는 방향 무관 극한이 있을 때만 값을 낸다 (D52) | `workbench/topics/spin-wave-observables/lswt-spin-correlations/note.md`<br>`workbench/topics/toolkit-development/dev-decisions/b-decisions-references.tex` |

### Fundamental Theory of Spin Wave Theory (17)

| 점 | 종류 | 적힌 상태 | 제목 | 근거 파일 |
|---|---|---|---|---|
| `project-q` | 질문 | 진행 | 2차원 스핀 모형의 LSWT 이론을 하나의 검증된 정본으로 정리할 수 있는가 | `workbench/decisions/2026-08-01-lswt-markdown-source-authority.md`<br>`workbench/research.yaml` |
| `canon-review-q` | 질문 | 진행 | 이론 노트를 원문·문헌과 대조하고 다시 유도해 하나의 정본으로 만들 수 있는가 | `workbench/tasks/2026-10-20-theory-note-review.md` |
| `drafts-check-ana` | 이론 계산 | 끝남 · 해결 | 이론 초안 10개를 독립 유도로 다시 검산해 오류 14건을 고친다 | `docs/reviews/2026-10-01-lswt-draft-physics-review-guide.md` |
| `onsite-rule-ana` | 이론 계산 | 끝남 · 해결 | 단일 이온 항은 결맞음 상태 값 (1−1/2S)A로 넣는다 (D37) | `workbench/topics/fundamental-theory-of-spin-wave-theory/lswt-boson-hamiltonian/note.md`<br>`workbench/topics/toolkit-development/dev-decisions/b-decisions-references.tex` |
| `spiral-frame-ana` | 이론 계산 | 끝남 · 해결 | 단일 Q 나선을 회전틀에서 원시 셀 LSWT로 푼다 (D34) | `workbench/topics/fundamental-theory-of-spin-wave-theory/lswt-classical-order/note.md`<br>`workbench/topics/toolkit-development/dev-decisions/b-decisions-references.tex` |
| `spiral-q` | 질문 | 끝남 · 해결 | 단일 Q 나선을 자기 초격자 없이 풀고 그 안정성을 판정할 수 있는가 | 없음 |
| `pre-move-fixes` | 이론 계산 | 끝남 · 해결 | 연구노트로 옮기기 전 점검에서 이론 노트 다섯 곳을 고친다 | `workbench/log/2026-10-08.md` |
| `boson-blocks-ana` | 이론 계산 | 끝남 · 해결 | 보존 해밀토니안을 원형 성분으로 완성하고 BdG 블록을 명시한다 | `code-space/examples/verification/lswt_bdg_blocks_check.py`<br>`workbench/topics/fundamental-theory-of-spin-wave-theory/lswt-boson-hamiltonian/note.md` |
| `pitch-finding` | 명제 | 끝남 · 해결 | 단일 Q 나선은 모든 Q에서 토크가 0이지만 전역 최소에서만 안정하다 | `code-space/examples/verification/lswt_bdg_blocks_check.py`<br>`workbench/topics/fundamental-theory-of-spin-wave-theory/lswt-classical-order/note.md` |
| `stability-hessian-ana` | 이론 계산 | 끝남 · 해결 | 고전 에너지의 2차 변분은 정확히 BdG 행렬이다 | `code-space/examples/verification/lswt_bdg_blocks_check.py`<br>`workbench/topics/fundamental-theory-of-spin-wave-theory/lswt-classical-order/note.md` |
| `stability-q` | 질문 | 끝남 · 해결 | 고전 기준 상태가 정류하고 국소적으로 안정한지 어떻게 판정하는가 | 없음 |
| `zero-modes-q` | 질문 | 진행 | 영모드(Goldstone 점)가 있을 때 각 LSWT 관측량은 정의되는가, 어떤 극한으로 정의하는가 | 없음 |
| `local-stability-calc` | 수치 계산 | 끝남 · 해결 | 연속 k와 영모드 강성으로 국소 안정성을 판정한다 (D51) | `code-space/examples/verification/lswt_d51_local_stability_check.py`<br>`workbench/topics/fundamental-theory-of-spin-wave-theory/lswt-classical-order/note.md` |
| `mesh-instability-q` | 질문 | 끝남 · 해결 | 유한 메시가 놓치는 장파장 불안정을 잡을 수 있는가 | 없음 |
| `spiral-harmonics-ana` | 이론 계산 | 끝남 · 해결 | 회전 조건이 깨지면 단일 Q 나선이 어떻게 무너지는가 | `code-space/examples/verification/lswt_spiral_harmonics_check.py`<br>`workbench/topics/fundamental-theory-of-spin-wave-theory/lswt-classical-order/note.md` |
| `higher-order-ana` | 이론 계산 | 끝남 · 해결 | 2차가 정하지 못한 영모드를 고차 항으로 판정한다 (D54) | `code-space/tests/test_methods/test_local_stability.py`<br>`workbench/topics/fundamental-theory-of-spin-wave-theory/lswt-classical-order/note.md` |
| `spiral-stability-calc` | 수치 계산 | 끝남 · 해결 | 단일 Q 나선의 국소 안정성을 회전틀에서 판정한다 (D55) | `code-space/tests/test_methods/test_local_stability.py`<br>`workbench/topics/toolkit-development/dev-decisions/b-decisions-references.tex` |

### Beyond Linear Spin Waves (9)

| 점 | 종류 | 적힌 상태 | 제목 | 근거 파일 |
|---|---|---|---|---|
| `nlswt-expansion` | 수치 계산 | 끝남 · 해결 | 비선형 스핀파 1/S 전개를 S⁰ 차수까지 구현하고 큰 S ED와 대조한다 (D46) | `workbench/topics/toolkit-development/dev-decisions/b-decisions-references.tex` |
| `pg-gap-finding` | 명제 | 끝남 · 해결 | 유사 Goldstone gap의 다음 차수는 S = 1/2에서 선도 차수보다 크다 (D47) | `workbench/topics/beyond-linear-spin-waves/lswt-nonlinear-spin-waves/note.md`<br>`workbench/topics/toolkit-development/dev-decisions/b-decisions-references.tex` |
| `zc-literature` | 문헌 | 끝남 · 해결 | Zheng–Hamer (1993) 원문으로 정사각 Z_c와 바닥 에너지를 대조한다 | 없음 |
| `nlswt-note-ana` | 이론 계산 | 끝남 · 해결 | 1/S 보정을 연구노트로 유도한다: 3·4차 항, 꼭짓점, tadpole | `code-space/examples/verification/lswt_nonlinear_vertices_check.py`<br>`workbench/topics/beyond-linear-spin-waves/lswt-nonlinear-spin-waves/note.md` |
| `nonlinear-q` | 질문 | 진행 | 1/S 보정(에너지 S⁰, 마그논 1/S, 붕괴와 두 마그논 연속체)을 유도하고 문헌과 맞출 수 있는가 | 없음 |
| `tadpole-finding` | 명제 | 진행 | tadpole 계수는 영점 에너지가 고전 방향에 주는 힘이다 | `code-space/examples/verification/lswt_nonlinear_vertices_check.py`<br>`workbench/topics/beyond-linear-spin-waves/lswt-nonlinear-spin-waves/note.md` |
| `tri-s0-q` | 질문 | 끝남 · 해결 | 삼각격자 S⁰ 바닥 에너지가 Chernyshev–Zhitomirsky 값과 왜 0.2% 다른가 | 없음 |
| `tri-s0-finding` | 명제 | 끝남 · 해결 | 0.2% 차이는 논문의 I₃ 수치에서 온다 | `code-space/examples/verification/lswt_triangular_s0_energy_check.py` |
| `decay-calc` | 수치 계산 | 끝남 · 해결 | 마그논 붕괴율을 넓힘 없이 황금률로 계산한다 (D57) | `workbench/topics/beyond-linear-spin-waves/lswt-nonlinear-spin-waves/note.md` |

### Toolkit Development (7)

| 점 | 종류 | 적힌 상태 | 제목 | 근거 파일 |
|---|---|---|---|---|
| `toolkit-q` | 질문 | 진행 | 정본 이론 위에 반고전 패키지(spin-toolkit)를 만들어 정본과 문헌으로 검증하고 공개할 수 있는가 | `workbench/decisions/2026-08-01-lswt-markdown-source-authority.md`<br>`workbench/research.yaml` |
| `state-selection-q` | 질문 | 끝남 · 해결 | 고전적으로 축퇴된 상태 중 무엇을 기준 상태로 고르는가 (영점 에너지, 1-loop) | `workbench/topics/toolkit-development/dev-decisions/b-decisions-references.tex` |
| `one-loop-potential` | 명제 | 끝남 · 해결 | 고전·양자 비교는 완화 경로 위 1-loop 유효 퍼텐셜의 최소로 한다 (D28) | `workbench/topics/toolkit-development/dev-decisions/b-decisions-references.tex` |
| `solver-validation-q` | 질문 | 끝남 · 해결 | 1/S 솔버와 유한 온도 자화를 큰 S의 ED와 문헌 값으로 검증했는가 | 없음 |
| `orbit-refine-calc` | 수치 계산 | 끝남 · 해결 | 약하게 고정된 궤도에서 고전 상태 정제가 시작점에 따라 갈리는 원인 (D50) | `workbench/topics/toolkit-development/dev-decisions/b-decisions-references.tex`<br>`workbench/topics/toolkit-development/dev-verification/a-implementation-verification.tex` |
| `scope-finding` | 명제 | 끝남 · 해결 | 패키지 범위는 2D 스핀 모형의 반고전 툴킷이다 (D48) | `workbench/decisions/2026-10-10-package-scope-semiclassical.md`<br>`workbench/tasks/2026-06-07-solver-seam-spike.md` |
| `release-audit-q` | 질문 | 진행 | 공개 전에 API·성능·정의 안 되는 값의 처리와 이론적 신뢰성을 확인했는가 | `workbench/tasks/2026-10-21-theory-reliability.md`<br>`workbench/topics/toolkit-development/dev-decisions/b-decisions-references.tex` |

## 맡긴 일 8개

| 일 | 적힌 상태 | 제목 | 만든 날 |
|---|---|---|---|
| `2026-08-02-thermal-hall-real-space-volume` | 진행 | Thermal Hall 면적·단위·반환 기준 | 2026-08-02 |
| `2026-08-09-lswt-documentation-audit` | 멈춤 | LSWT 이론 문서 점검 (범위·열린 물리 결정·검토 기록) | 2026-06-03 |
| `2026-08-10-lswt-implementation-backlog` | 멈춤 | LSWT 구현 남은 목록 | 2026-08-10 |
| `2026-10-13-batch-review` | 결과 · 판단 대기 | 결정과 문서 일괄 검토 | 2026-10-13 |
| `2026-10-15-handoff` | 멈춤 | 인계 (2026-10-15) — 새 대화에서 이어 가기 | 2026-10-15 |
| `2026-10-15-remaining-development` | 멈춤 | 남은 개발 정리 (2026-10-15) | 2026-10-15 |
| `2026-10-20-theory-note-review` | 진행 | 이론 노트 검토 (draft → accepted) | 2026-10-20 |
| `2026-10-21-theory-reliability` | 진행 | 패키지 이론적 신뢰성 확인 | 2026-10-21 |
