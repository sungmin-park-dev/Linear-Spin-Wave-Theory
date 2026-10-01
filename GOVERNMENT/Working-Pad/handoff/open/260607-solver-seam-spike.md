---
frontmatter-version: 1
title: Solver Seam Spike - XXZ multi-solver (reuse Tensor-Network-Study)
section: handoff/open
status: draft
execution-status: pending
last-edited-by: claude
created: 2026-06-07
updated: 2026-10-01
branch: main
cross-repo: ~/GitHub/Tensor-Network-Study
---

# Solver Seam Spike — XXZ 멀티솔버 (TN-Study 하네스 재사용)

## 목적 (Goal)

LSWT 코드를 "시스템 정의 ↔ 솔버 엔진"으로 재구성하기 전에, **각 솔버가 '시스템'에서
무엇을 읽는지를 실측 검증**한다(seam 지도 S1~S5). 작은 XXZ 모델을 ED·DMRG·NQS로
풀고 exact 1D XXZ(Bethe)와 대조한다.

- **그린필드 아님 — 기존 자산 재사용·개선:** `~/GitHub/Tensor-Network-Study`의
  `Projects/Cluster_Ising/main.py`가 이미 **Exact/ED/DMRG/NQS 멀티솔버 벤치마크
  하네스**(TeNPy+NetKet, 표·플롯·CLI). 이를 재사용해 XXZ 프로젝트를 만든다.
- **작업 위치 = Tensor-Network-Study** (그 리포의 publishable 학습 자산을 개선).
- **LSWT로 돌려보낼 산출물:** seam 결론(`findings.md`) + Bethe 정확해 노트
  (`bethe-ansatz-xxz.md`) → **LSWT `research-space/`**.

## 작업 레포 (codex) — 어디서 여나

**`~/GitHub/Tensor-Network-Study`에서 연다.** 코드·`PLAN.md`·하네스 재사용이 모두
거기고, 그 리포의 conventions를 따르며 **코드 커밋은 TN-study repo**로 간다.
LSWT는 **양 끝(bookend)만** 건드린다:
- 시작: LSWT의 이 핸드오프를 읽어 스펙 확보.
- 끝: `findings.md` + `bethe-ansatz-xxz.md`를 LSWT `research-space/`에 쓰고,
  이 핸드오프를 `closed/`로 이동 + `TASK-QUEUE.md` 갱신. (결론 커밋 = LSWT repo)
- 두 repo는 git이 별개 — **코드=TN-study / 결론=LSWT**.

## 기존 자산 재사용 (`~/GitHub/Tensor-Network-Study`)

- `Projects/Cluster_Ising/main.py` (634줄): `run_exact()`, `run_solver(name,…)`
  (ED/DMRG/NQS 디스패치), 결과표·플롯(λ/size sweep, NQS 수렴), argparse CLI.
  deps: `physics-tenpy` + numpy/scipy/matplotlib (+ NQS는 jax/netket).
  → **하네스(솔버 통합·플롯·CLI)를 재사용**, 모델만 XXZ로 교체 + geometry/뷰어/seam 추가.
- `src/{tn_core,models,observables}/`는 **빈 스켈레톤** — 채워야 하면 이 기회에.
- 참고: `Tutorials/Notebooks/TNTs-1..4.ipynb`, `~/GitHub/nqs-learning/NQS_Tutorial01_1D_TFIM.ipynb`,
  `~/GitHub/ITensor-Tutorial`.
- ※ TN-study는 별도 repo(고유 구조·conventions) — 그 리포 관례를 따른다.

## 검증 대상 — seam 지도 (S1~S5)

| Seam | 분리 대상 | 검증 질문 |
|---|---|---|
| S1 Term 연산자 내용 | 수치 J ↔ named-op | ED/DMRG/NQS는 named-op를 요구? 변환 경계? |
| S2 LocalSpace | S·dim·연산자 basis | 각 솔버가 LocalSpace로 뭘 요구(dim·연산자·charge)? |
| S3 Geometry base vs 파생 | base(격자·이웃·BC) ↔ BZ/MPS순서/graph | TN의 2D→1D ordering, NQS graph가 어떻게? |
| S4 자기질서/고전최적화 = LSWT단계 | system 정의에서 분리 | ED/TN/NQS는 정말 자기질서 불필요? |
| S5 관측량 스키마 vs 계산 | 공통 결과형 ↔ 방법별 계산 | 결과/관측량 형태가 얼마나 다른가? |

(맥락: LSWT `GOVERNMENT/Working-Pad/roadmap.md` Phase 0)

## 모델 (스테이지 1D → 2D)

spin-½, XXZ + 종자장, PBC:
H = Σ_<ij> [ Jxy(SˣSˣ+SʸSʸ) + Jz SᶻSᶻ ] − h Σ_i Sᶻ.

- **차원 스테이지:** ① 1D 사슬 N=8(스모크·toolchain) → ② 2D 정사각 4×4 PBC(seam 본 분석).
  geometry는 **인자**로(같은 코드, 격자만 교체). **1D→2D 코드 diff가 Geometry seam(S3)
  관찰 데이터**: ED=edge 리스트, TeNPy=2D Lattice+MPS snake, NetKet=graph.
- 파라미터: `(Jxy=1, Jz=1, h=0)` Heisenberg / `(1, 0.5, 0)` XXZ / `(1, 1, 0.5)` +field.
- probe: 횡자장 `h_x` → TeNPy Sz charge-conservation on/off seam.

## 솔버 / 교차검증

- Cluster_Ising의 `run_exact`/`run_solver`(ED/DMRG/NQS)를 **재사용** — 모델을 XXZ로 교체.
  ED는 hand-built(투명성) 또는 하네스 기존 ED 재사용 중 PLAN.md에서 결정.
- 비교: **ED = 유한-N 정확 기준**, DMRG ≈ ED(~1e-6), VMC ≈ ED(오차 내 ~1%).

## 검증 기준 — exact 1D XXZ (Bethe ansatz)

세 솔버 상호일치만으로는 *공유 버그*를 못 잡으므로 **해석적 정답과도 대조**.
convention: Δ=Jz/Jxy.

| 점 | 열역학극한 E/N | 근거 |
|---|---|---|
| Δ=1 (Heisenberg AFM) | 1/4 − ln2 ≈ −0.443147 | Hulthén 1938 |
| Δ=0 (XX) | −1/π ≈ −0.318310 | 자유페르미온(JW) |

**주의:** 위 값은 N→∞. 유한 N=8은 1/N²(c=1 CFT) 보정. →
- **XX점:** 자유페르미온 **유한-N 닫힌형**(PBC parity 주의) → ED와 **머신정밀도(~1e-12)** 일치.
- **Heisenberg점:** 1/4−ln2를 **외삽 앵커**(N=8,12,16 수렴).

## Bethe ansatz 정확해 정리 (`bethe-ansatz-xxz.md`, durable → LSWT research-space/theory/)

검증값의 근거가 되는 정확해를 **버리지 않는 이론 노트**로 정리. `exact_xxz` 코드가
이를 구현. 담을 내용: 모델·convention / XX점(JW 자유페르미온, 유한-N + −1/π) /
Heisenberg점(1/4−ln2) / 일반 −1<Δ<1(Yang-Yang II 적분 — **출처 확인 후 정확히
전사, 메모리로 쓰지 말 것**) / 유한크기 c=1 CFT 보정 / 레퍼런스(Bethe 1931; Hulthén
1938; Yang-Yang Phys. Rev. 150, 321 & 327 (1966); Karbach-Müller arXiv:cond-mat/9809162).

## 산출물 / 위치

- **코드 → `~/GitHub/Tensor-Network-Study`** 에 새 프로젝트(예: `Projects/XXZ_Multisolver/`,
  이름·구조는 PLAN.md에서 TN-study 관례 따라 확정). Cluster_Ising 하네스 재사용 +
  `geometry`(1D/2D), `viewer`, `exact_xxz`, seam 추출 추가. **publishable 품질**(아래).
- **LSWT로 승격:** `findings.md`(seam S1~S5 결론) + `bethe-ansatz-xxz.md` →
  LSWT `research-space/`(예: `research-space/theory/exact-benchmarks/`, 위치 성민 확정).
- 환경: miniconda env(`/Users/david/miniconda3/bin/python`)에 `physics-tenpy`,
  `netket` 설치(Cluster_Ising `requirements.txt` 참고). 버전을 `findings.md`에 기록.

## 시스템/Geometry 뷰어 (`viewer`)

Geometry를 *눈으로* 검증: (a) 세 솔버가 같은 계를 푸는지, (b) Geometry seam 가시화.
- 공통 geometry의 **sites + bonds** 플롯(1D/2D, PBC 결합, 인덱스·부분격자 라벨).
- **파생 view overlay:** TeNPy 2D→1D **MPS snake 경로**, NetKet **graph edges**.
- PNG 저장 + `findings.md`/글에서 참조. (옵션: 스핀/자기장 방향)

## 코드 품질 / 포스팅

TN-study는 이미 publishable 학습 리포. 그 톤으로 작성:
- 명료성(docstring·좋은 이름), 자기완결·재현성(seed 고정, 버전 명시, 재현 절차),
  서사 일관성(ED/DMRG/NQS *동일 흐름* geometry→model→solve→check — "비교"가 핵심),
  정확성 축 = exact 1D XXZ(Bethe). 단 **과도한 일반화 금지**(작은 스파이크).

## 구현 전 게이트 — `PLAN.md` (성민 리뷰 필수)

코딩 전 codex는 TN-study 새 프로젝트 안에 **`PLAN.md`를 먼저 작성**한다. 성민이
*파일을 직접 읽고 이해할 만큼 상세히*:
- Cluster_Ising에서 **무엇을 재사용/리팩터**하고 무엇이 새로운지,
- 각 파일 책임·공개 함수·입출력 / geometry·model의 데이터 흐름 / 1D↔2D 파라미터화,
- `viewer`가 무엇을 그리는지 / `exact_xxz`·crosscheck 판정 기준,
- LSWT로 승격할 `findings.md`·`bethe-ansatz-xxz.md` 형식.

**성민 리뷰·승인 후에야 솔버 코드 구현 시작.** (제안-우선)

## 첫 실행 순서

1. **`PLAN.md` 작성 → 성민 리뷰·승인** (위 게이트). 승인 전 코드 작성 금지.
2. `Projects/Cluster_Ising` 하네스 구조 파악 + env(`physics-tenpy`,`netket`) 설치·스모크.
3. XXZ 프로젝트 골격: `geometry`/`model`/`viewer` + 하네스 재사용 →
   **1D·2D geometry를 뷰어로 시각 확인**.
4. **1D N=8**: ED+DMRG+VMC GS 교차검증 + **XX점 머신정밀도**, **Heisenberg점 1/4−ln2 외삽**
   (`exact_xxz` 구현 + `bethe-ansatz-xxz.md` 정리).
5. **2D 4×4 PBC**: 동일 3방법 + 교차검증 (+ 뷰어로 2D·MPS snake 확인).
6. TeNPy/NetKet(+Cluster_Ising 하네스) 구조 + 모델정의 API 요구 입력 정리.
7. `findings.md`(seam S1~S5) 작성 → **LSWT `research-space/`로 `bethe-ansatz-xxz.md`와 함께 승격.**

## 알려진 리스크

- NetKet/jax 설치 무게. 2D DMRG bond-dim(4×4 가능). VMC는 변분 → 오차 내 일치로 판정.
- 크로스레포: 코드=TN-study, 결론=LSWT. 두 repo 혼동 주의.
- 비결정성(VMC seed) 고정.

## 범위 밖 (하지 않음)

- 프로덕션 ED/TN/NQS 솔버 / LSWT `lswt` 패키지 리팩터 — 안 함. 과도한 일반화 금지.
- seam 결론은 Phase 0 재구성 plan 입력. (코드는 TN-study 자산으로 보존)

## 2026-10-01 저장소 안 실측 — ED와 DMRG(TeNPy)가 같은 토러스 전개를 읽는다

사용자 위임(2026-10-01)으로, 별도 저장소 대신 이 저장소의 공통 자료형 위에서 seam을 실측했다. 모델을
`expand_on_torus`(D23, 모든 방법이 공유하는 규칙)로 한 번 전개하고, 같은 사이트·결합(3 x 3 교환 행렬)·사이트 장을
toolkit ED와 TeNPy 1.1.1 유한 DMRG 어댑터(약 40줄, `examples/solver_seam_tenpy_check.py`)에 넘겼다. TeNPy는 이 검사의
선택 의존성이고 패키지는 쓰지 않는다. 기록: `data-space/verification/261001-solver-seam-tenpy/report.json`.

| 경우 | 사이트 | DMRG - ED | 정확한 기준 |
|---|---|---|---|
| XX 고리 L=12 | 12 | 1.6e-14 | Jordan-Wigner 유한 L 닫힌형: ED - 기준 1.2e-14 |
| Heisenberg 고리 L=12 | 12 | 2.5e-14 | (E0 = -5.38739092) |
| 정사각 4x4 Heisenberg | 16 | 3.4e-14 | E0/N = -0.70178020(문헌): 차이 8e-9(기준값 자릿수) |
| NBCP 3x3, 1 T ∥ c | 9 | 7e-16 | — |
| NBCP 3x3, 1 T ∥ b*(U(1) 없음) | 9 | 2.9e-15 | — |
| NBCP 3x3 + J_PD 0.01, 0.5 T ∥ c(복소 S^y 곱) | 9 | 3.1e-15 | — |

XX 고리의 정확해는 새 회귀 테스트(`test_ed.py::test_xx_ring_matches_jordan_wigner_free_fermions`, L=8)로 고정했다.
DMRG 두 경우에서 TeNPy가 마지막 정준형 노름 경고(2e-5, 8e-2)를 냈으나 에너지는 ED와 1e-14 안에서 같다.

**Seam 결론(S1–S5).**

- **S1 항의 연산자 내용:** 수치 3 x 3 교환 행렬이면 충분하다. 어댑터는 각 결합을 `sum_ab J_ab S^a S^b`로 이름 붙은 연산자
  Sx, Sy, Sz에 그대로 옮긴다. 이름 붙은 항(XXZ, PD 등)은 필요 없다. 다만 U(1) 보존을 쓰려면 S+S-·SzSz 형태로 바꿀 축 판정이
  필요하다. ED의 `EDSector(axis=...)` 판정 규칙(`symmetry_tolerance`)을 그대로 재사용하면 된다.
- **S2 LocalSpace:** 필요한 것은 사이트마다 S뿐이다(`SpinSite(S)`; 이 어댑터는 S가 한 종류인 토러스만 받는다). 보존 charge는
  모델이 아니라 솔버 설정이다(장 ∥ b*에서는 `conserve='None'`이 필수).
- **S3 기하:** 솔버가 읽는 기하는 토러스 사이트 순서와 결합 목록뿐이다. 2D 토러스는 MPS 순서(여기서는 토러스 순서 그대로)에서
  최대 8–12 사이트 떨어진 결합이 된다. 4x4에서 DMRG 시간(129 s, chi 256)의 대부분이 이 장거리 결합 때문이다. 순서 최적화(snake 등)는
  어댑터 쪽 선택이며 `TorusCluster`는 바꿀 필요가 없다.
- **S4 자기 질서:** ED와 DMRG 모두 `SpinState`, 고전 상태, 국소 좌표계를 쓰지 않는다. 자기 질서는 LSWT 단계의 입력이라는 현재
  분리(D30)가 맞다.
- **S5 결과:** 공통으로 비교 가능한 것은 토러스 총에너지(E0)와 `ResultHeader`(모델 지문, 기하, 조건)다. 관측량 계산은 방법별로
  다르다(ED: 고유벡터, DMRG: MPS). 공통 결과형은 지금의 헤더 + 방법별 본문 구조로 충분하다.

**남은 것.** NQS(NetKet)는 설치·검증하지 않았다(jax 무게). TN 필드 규약(D30의 "TN 필드 규약")은 이 결과로 정할 수 있다: 장은
모델에 넣지 않고 `ExternalConditions`에서 `TorusCluster.fields(conditions)`로 사이트 장을 만들어 `-h_i . S_i`로 넣는다(ED와 같은
부호, 위 표에서 확인). 이 결정은 사용자 확인 대상이다. 1D Bethe 노트(`bethe-ansatz-xxz.md`)와 원래 계획의 TN-Study 하네스는 하지 않았다.
