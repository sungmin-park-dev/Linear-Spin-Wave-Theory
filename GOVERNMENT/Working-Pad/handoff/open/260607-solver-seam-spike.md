---
frontmatter-version: 1
title: Solver Seam Spike - ED / TN / NQS empirical seam verification
section: handoff/open
status: draft
execution-status: pending
last-edited-by: claude
created: 2026-06-07
updated: 2026-06-07
branch: main
---

# Solver Seam Spike — ED / TN / NQS 실측 seam 검증

## 목적 (Goal)

LSWT 코드를 "시스템 정의 ↔ 솔버 엔진"으로 재구성하기 전에, **각 솔버가
'시스템'에서 무엇을 읽는지를 실측으로 검증**한다. 현재 seam 지도(아래 S1~S5)는
TeNPy/NetKet *문서*와 일반 지식 기반의 armchair 분석이라, ED·TN·NQS로 작은
모델을 실제로 풀어 검증·수정한다.

- **버리는(throwaway) 스파이크.** 프로덕션 솔버 아님, 우리 패키지 리팩터 아님.
- 산출물(`findings.md`)은 **Phase 0 재구성 plan의 입력**으로만 쓴다.
- 부산물: TeNPy/NetKet *자체 패키지 구조* = 우리 seam 배치의 실전 레퍼런스.

## 검증 대상 — seam 지도 (S1~S5)

| Seam | 무엇을 분리 | 검증 질문 |
|---|---|---|
| S1 Term 연산자 내용 | term의 연산자 내용(수치 J ↔ named-op) | ED/TN/NQS는 정말 named-op를 요구하나? 변환 경계는? |
| S2 LocalSpace | 국소 자유도(S·dim·연산자 basis)를 Site 위치/순서와 분리 | 각 솔버가 LocalSpace로 뭘 요구하나(dim·연산자·charge)? |
| S3 Geometry base vs 파생 view | base(격자·이웃·BC) ↔ 파생(BZ / MPS 1D순서 / graph) | TN의 2D→1D ordering, NQS graph가 어떻게 들어오나? |
| S4 자기질서/고전최적화 = LSWT단계 | system 정의에서 분리(전처리) | ED/TN/NQS가 정말 자기질서 불필요한가(확인)? |
| S5 관측량 스키마 vs 계산 | 공통 결과형 ↔ 방법별 계산 | 각 솔버 결과/관측량 형태가 얼마나 다른가? |

(전체 맥락: `GOVERNMENT/Working-Pad/roadmap.md` Phase 0)

## 모델 (하나로 고정, 차원은 스테이지)

spin-½, **XXZ + 종방향 자기장**, PBC:

H = Σ_<ij> [ Jxy (Sx_i Sx_j + Sy_i Sy_j) + Jz Sz_i Sz_j ] − h Σ_i Sz_i

- **차원 = 스테이지 1D → 2D:**
  1. **1D 사슬 N=8** — toolchain/설치/API/교차검증을 싸게 확정(스모크).
  2. **2D 정사각 4×4 PBC** — geometry / MPS-1D-ordering seam 본 분석.
- 파라미터 세트: `(Jxy=1, Jz=1, h=0)` Heisenberg / `(1, 0.5, 0)` XXZ / `(1, 1, 0.5)` +field.
- **probe 옵션:** 횡자장 `h_x` 추가 → TeNPy의 Sz charge-conservation on/off seam 확인.
- **구현 방식:** 1D·2D는 *같은 솔버 파일*(`ed.py`/`tenpy_dmrg.py`/`netket_vmc.py`)에서
  geometry를 **인자로** 받아 처리(스테이지 = 실행 순서일 뿐, 별도 코드 아님).
  **각 프레임워크에서 1D→2D 시 바뀌는 코드 자체가 Geometry seam(S3) 관찰 데이터**:
  ED는 edge 리스트만, TeNPy는 2D Lattice + 2D→1D MPS ordering(snake), NetKet은 graph
  교체. 이 1D↔2D diff를 프레임워크별로 `findings.md`에 기록.

## 무엇을 설치 vs 빌드

| 방법 | 결정 | 비고 |
|---|---|---|
| ED | **직접 빌드** (numpy + `scipy.sparse.linalg.eigsh`) | 정확 기준 + "system에서 뭘 읽나"를 투명하게 작성 |
| TN | **TeNPy 설치** (`physics-tenpy`), DMRG GS | 모듈 구조(Site/Lattice/Model/Algorithm) 공부 대상 |
| NQS | **NetKet 설치** (`netket`→jax), VMC GS | 모듈 구조(Hilbert/Graph/Operator/Driver) 공부 대상 |
| (QuSpin) | 선택: 구조 참고만 | ED 레퍼런스 곁눈질 |

## 교차검증

세 방법 바닥상태 에너지/site 비교. **ED = 유한-N 정확 기준.** DMRG ≈ ED (~1e-6),
VMC ≈ ED (VMC 오차 내, ~1%). 불일치 시 모델 정의 차이 → 그 자체가 seam 정보.

## 검증 기준 — exact 1D XXZ (Bethe ansatz)

1D 스테이지는 **해석적 정답과도 대조**한다(세 솔버 상호일치만으로는 *공유 버그*를
못 잡으므로 독립 기준 필요). convention: H = Σ[SˣSˣ+SʸSʸ + Δ SᶻSᶻ], Δ=Jz/Jxy.

| 점 | 열역학극한 E/N | 근거 |
|---|---|---|
| Δ=1 (Heisenberg AFM) | 1/4 − ln2 ≈ −0.443147 | Hulthén 1938 |
| Δ=0 (XX) | −1/π ≈ −0.318310 | 자유페르미온(JW) |

**주의:** 위 값은 N→∞. 유한 N=8 PBC는 1/N²(c=1 CFT) 보정이 있어 *정확히* 그 값이
아님. 그래서 두 갈래로:
- **XX점(Δ=0):** 자유페르미온 유한-N 닫힌형 `E₀ = Σ_{k: cos k<0} cos k`,
  `k=2πm/N`, PBC fermion-parity 주의 → **ED와 머신정밀도(~1e-12) 일치** 요구.
  연산자 구성·JW까지 잡는 가장 강한 독립 검증.
- **Heisenberg점(Δ=1):** 열역학극한 1/4−ln2를 **외삽 앵커**로. N=8,12,16…에서
  ED/DMRG가 1/N²로 수렴하는지 확인. (더 엄밀히는 Yang-Yang I의 유한-N Bethe
  방정식을 풀어 정확 유한-N 값과 대조 — 선택)

레퍼런스: Bethe 1931; Hulthén 1938; **Yang & Yang, Phys. Rev. 150, 321 (1966)
[I, 유한계] & 327 (1966) [II, 무한계]**. 일반 Δ(예: 0.5) 적분 닫힌형은 Yang-Yang II
— 필요 시 출처 확인해 추가.
- 무료·교육용 (실제 유도): **Karbach & Müller, "Introduction to the Bethe Ansatz I",
  arXiv:cond-mat/9809162** (Heisenberg 바닥에너지 ¼−ln2 단계별 유도);
  XXZ 일반 Δ 공식은 integrability.org XXZ 페이지 / Takahashi 교과서.

## Bethe ansatz 정확해 정리 (`bethe-ansatz-xxz.md`, durable → research-space/theory/)

검증값의 *근거가 되는 정확해*를 별도 이론 노트로 정리한다. **버리지 않는다** —
스파이크 종료 시 `research-space/theory/`(예: `exact-benchmarks/`, 위치는 codex/성민
확정)로 승격. `exact_xxz.py`는 이 노트의 식을 구현한 것.

담을 내용:
- 모델·convention 명시: H = Σ[SˣSˣ+SʸSʸ + Δ SᶻSᶻ], Δ=Jz/Jxy.
- **XX점(Δ=0):** Jordan-Wigner → 자유페르미온. ε(k)=cos k, 바닥=음에너지 모드 채움.
  **유한-N 닫힌형**(PBC fermion-parity 주의) + 열역학극한 −1/π. (머신정밀도 검증 근거)
- **Heisenberg점(Δ=1):** Bethe ansatz 바닥에너지 1/4−ln2 (Hulthén; Karbach-Müller 유도).
- **일반 −1<Δ<1:** Yang-Yang II 적분 닫힌형(Δ=cos γ 매개). **출처 확인 후 정확히
  전사**(메모리로 적지 말 것) — Takahashi / integrability.org 대조.
- **유한크기 보정:** c=1 CFT, E(N)/N = e∞ + a/N² + … (유한 N ≠ 열역학극한 이유).
- 레퍼런스: Bethe 1931; Hulthén 1938; Yang-Yang 1966 (Phys. Rev. 150, 321 & 327);
  Karbach-Müller arXiv:cond-mat/9809162.

## 산출물 / 환경

**위치: 최상위 `sandbox/solver-seam-xxz/`** (canonical `*-space` 밖의 버리는 탐색
코드. `README.md`에 "findings 기록 후 폴더째 삭제 가능" 명시).

제안 파일 구조 (상세·확정은 `PLAN.md`에서 codex가 작성):

```text
sandbox/solver-seam-xxz/
├── PLAN.md         # ★ 구현 전 상세 설계 — 성민 리뷰 게이트 (아래 절)
├── README.md       # throwaway 명시 + 실행법
├── geometry.py     # 공통 geometry: 1D chain N / 2D square LxL (PBC) → sites + bonds
├── model.py        # 공통 XXZ 파라미터 (Jxy, Jz, h[, hx]) — geometry 독립
├── viewer.py       # ★ 시스템/geometry 뷰어 (아래 절)
├── ed.py           # hand-built ED → GS
├── tenpy_dmrg.py   # TeNPy DMRG → GS
├── netket_vmc.py   # NetKet VMC → GS
├── exact_xxz.py    # exact 1D XXZ 체크값 (XX 유한-N, Heisenberg 1/4-ln2)
├── bethe-ansatz-xxz.md  # ★ durable 이론 정리 (→ research-space/theory/)
├── crosscheck.py   # 세 솔버 + exact 비교표
└── findings.md     # 결론 (→ research-space 승격)
```

- **결론 승격:** 종료 시 `findings.md`(교훈)는 `research-space/`로 승격, 코드는 삭제 가능.
- 환경: miniconda env(`/Users/david/miniconda3/bin/python`)에
  `pip install physics-tenpy netket`. 설치 버전을 `findings.md`에 기록.

## 시스템/Geometry 뷰어 (`viewer.py`)

Geometry를 *눈으로* 검증하기 위한 뷰어. 목적: (a) 세 솔버가 **같은 계**를 푸는지
시각 확인, (b) Geometry seam을 눈에 보이게.

- 공통 `geometry.py`의 **sites + bonds**를 플롯: 1D 사슬, 2D 정사각(PBC 결합 포함),
  사이트 인덱스·부분격자 라벨.
- **프레임워크 파생 view overlay** (seam 가시화):
  - TeNPy: 2D→1D **MPS ordering(snake) 경로**를 2D 격자 위에 그림.
  - NetKet: **graph edges**.
- 출력 PNG를 폴더에 저장 + `findings.md`에서 참조. (옵션: 스핀/자기장 방향 표시)

## findings.md — 핵심 산출물 (각 솔버: ED/TN/NQS, + LSWT는 기존 코드 대조)

1. **모델 정의에 반드시 입력하는 것** → 5요소(LocalSpace/SiteSet/Geometry/Term/
   Parameters)에 매핑.
2. **솔버가 파생하는 것**(=system 아님): Hilbert basis / MPS 1D순서 /
   graph+symmetry / ansatz / sampler.
3. **라이브러리 자체 모듈 구조** → 우리 seam 배치 레퍼런스로 요약.
4. **seam S1~S5 검증/수정** + symmetry·charge-conservation seam 특별 확인.

## 알려진 리스크

- NetKet 설치가 jax 의존으로 무겁/느릴 수 있음. TeNPy 2D DMRG는 bond-dim 필요
  (4×4는 가능). VMC는 변분이라 ED와 정확히 안 맞음(오차 내 일치로 판정).
- 비결정성(VMC seed) — seed 고정해 재현.

## 구현 전 게이트 — 파일구조 계획 (`PLAN.md`, 성민 리뷰 필수)

코딩 전에 codex는 **`PLAN.md`를 먼저 작성**한다. 성민이 *파일을 직접 읽고 이해할 수
있을 만큼 상세히*:
- 각 파일의 책임 / 공개 함수·클래스 / 입출력,
- `geometry.py`·`model.py`가 세 솔버로 어떻게 흐르는지(데이터 흐름),
- geometry가 1D↔2D로 어떻게 파라미터화되는지,
- `viewer.py`가 무엇을 어떻게 그리는지,
- `exact_xxz.py`·`crosscheck.py`의 판정 기준.

**성민 리뷰·승인 후에야 실제 솔버 코드 구현 시작.** (CLAUDE.md 제안-우선 원칙)

## 첫 실행 순서

1. **`PLAN.md` 작성 → 성민 리뷰·승인** (위 게이트). 승인 전 솔버 코드 작성 금지.
2. miniconda env에 `physics-tenpy`, `netket` 설치 + import 스모크.
3. `geometry.py`/`model.py`/`viewer.py` 구현 → **1D·2D geometry를 뷰어로 시각 확인**.
4. **1D N=8**: ED(빌드)+DMRG+VMC GS → `crosscheck.py` 일치 + **XX점 머신정밀도**,
   **Heisenberg점 1/4−ln2 외삽** 대조. (`exact_xxz.py` 구현 + `bethe-ansatz-xxz.md` 정리)
5. **2D 4×4 PBC**: 동일 3방법 + 교차검증 (+ 뷰어로 2D geometry·MPS snake 확인).
6. TeNPy/NetKet 패키지 구조 + 모델 정의 API 요구 입력 정리.
7. `findings.md` 작성 + seam S1~S5 검증/수정 결론. **durable 노트(`findings.md`,
   `bethe-ansatz-xxz.md`)를 `research-space/`로 승격.**

## 범위 밖 (하지 않음)

- 프로덕션 ED/TN/NQS 솔버 구현 / 우리 `lswt` 패키지 리팩터 — 안 함.
- 결과는 `findings.md`로 Phase 0 재구성 plan에 입력만.
