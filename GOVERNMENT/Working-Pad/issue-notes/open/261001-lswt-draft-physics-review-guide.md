---
frontmatter-version: 1
title: LSWT draft 물리·수학 검토 가이드 (2026-10-01)
section: issue-notes/open
issue-type: review
status: in-review
last-edited-by: claude
created: 2026-10-01
updated: 2026-10-01
related:
  - GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md
  - docs/lswt/README.md
---

# LSWT draft 물리·수학 검토 가이드

## 배경

2026-09-30에 작성한 7개 draft(대각화, magnon observables, 열역학, topology, spin correlations, structure factor, LT)가 사용자 Human Physics and Mathematics Review를 기다리고 있다. 2026-10-01에 남은 skeleton 3개(worked example, paraunitarity proofs, thermodynamic derivations)도 본문 draft로 작성했다. 이 문서는 사용자가 10개 draft를 효율적으로 검토할 수 있도록 문서별로 꼭 봐야 할 주장과 유도 단계, 독립 검산에서 찾은 오류와 그 처리, 남은 의심 항목을 모은다.

독립 검산은 문서마다 별도 검산자가 모든 식을 다시 유도하고, 저렴한 항등식은 무작위 BdG 행렬·2-mode Fock 공간 ED·수치 적분으로 확인했다. 이 검산은 사용자 검토를 대체하지 않는다. 10개 문서 모두 `status: draft`이며 accepted가 아니다.

## 권장 검토 순서

1. [Paraunitary Diagonalization](../../../../docs/lswt/01-derivation/paraunitary-diagonalization.md) → [Appendix: Paraunitarity Proofs](../../../../docs/lswt/04-appendices/paraunitarity-proofs.md) → [Worked Example](../../../../docs/lswt/03-examples/worked-example.md). 뒤의 모든 observable이 여기 정의(T, Σ3, 입자–홀 구조, ΔE_zp)를 쓴다. Worked example은 일반 식을 한 band 닫힌 식으로 확인하는 용도다.
2. [Magnon Observables](../../../../docs/lswt/02-observables/magnon-observables.md), [Thermodynamics](../../../../docs/lswt/02-observables/thermodynamics.md), [Appendix: Thermodynamic Derivations](../../../../docs/lswt/04-appendices/thermodynamic-derivations.md).
3. [Topological Magnon Quantities](../../../../docs/lswt/02-observables/topological-magnon-quantities.md). 오늘 고친 gauge 문단은 코드 문서와도 연결된다.
4. [Spin Correlations](../../../../docs/lswt/02-observables/spin-correlations.md), [Structure Factor and Spectral Function](../../../../docs/lswt/02-observables/structure-factor-and-spectral-function.md).
5. [Appendix: Luttinger–Tisza Method](../../../../docs/lswt/04-appendices/luttinger-tisza-method.md).

## 2026-10-01에 고친 확인된 오류

모두 독립 유도와 수치로 확인한 뒤 고쳤다. 판단이 필요한 항목은 고치지 않고 아래 "남은 의심 항목"에 두었다.

| 문서 | 오류 | 수정 | 확인 방법 |
|---|---|---|---|
| topology, gauge 문단 | "Chern 수와 thermal Hall 전도도는 Fourier gauge에 무관"이라 썼다. Chern 수만 불변이다. κ는 곡률에 c2(ε)를 곱해 적분하므로 부분적분 뒤 ∇c2 × (주기 벡터장) 항이 남는다. | κ는 gauge에 의존하며 full-position gauge에서 계산한다고 고쳤다. full-position이 물리적 gauge라는 판단은 위치 연산자와의 대응에서 추론한 것이라고 본문에 명시했다. | C3을 깬 boson honeycomb(최근접 hopping 1, 0.6, 0.3), 120² mesh, T=2: full-position κ/T=0.02662, cell gauge 0.03536, Chern 수는 두 gauge에서 같다. C3을 보존하면 두 값이 1e-11까지 같다. |
| topology, −π²/3 문단 | c2 가중치가 "고온에서 κ/T→0을 직접 준다"고 썼다. 반대다. c2→π²/3이라 κ/T→−(π/6)ΣC_n이고, 0이 되는 것은 Chern 합 규칙 덕분이다. 직접 0으로 가는 것은 원문의 c2−π²/3 가중치다. | 문장을 고쳤다. | c2 적분 정의와 닫힌 식의 수치 비교, 극한 확인 |
| topology, 곡률 발산 조건 | 입자–홀 분모를 2ε_nk로 썼다. 비상반 스펙트럼에서는 ε_nk+ε_{m,−k}다. 띠 접촉에서 "발산"도 과한 표현이다. | "정의되지 않으며 일반적으로 발산"으로 바꾸고 분모를 일반형으로 고쳤다. | 유도 |
| topology, skyrmion 수 | 예외 배치를 "분자가 0"으로만 썼다. 삼중곱이 0이고 실수부가 양수가 아니면(한 대원 위에서 반원에 담기지 않는 세 방향) χ=±2π로 가지가 모호해 Q_sk가 1만큼 뛴다. | 예외 조건을 정확히 적고, 그 배치에서 Q_sk는 정의되지 않는다고 썼다. | 120° 동평면 근방에서 χ→2π, Van Oosterom–Strackee 입체각과 기계 정밀도 일치 |
| spin correlations, 켤레 관계 | [C^{αβ}(q,t)]* = C^{βα}(−q,−t)라 썼다. 맞는 식은 C^{βα}(q,−t)다. 원래 식은 교환자의 둘째 항이다. | 식과 유도 한 줄을 고쳤다. 고친 식이 SF 문서의 (α,β) Hermitian 성질을 함의한다. | 비상반 2-mode ED(β=2): C*=0.12595−0.10287i = C^{βα}(q,−t), C^{βα}(−q,−t)=0.0267−0.2873i |
| spin correlations, q+G 주기성 | "모든 site에서 G·r_I∈2πℤ"라 썼다. 원점에 의존하는 조건이다. C는 r_I−r_J에만 의존하므로 모든 쌍에서 G·(r_I−r_J)∈2πℤ가 맞다. | 고쳤다. | 유도 |
| structure factor, retarded 함수 | 교환자를 2i Im C^{αβ}(q,t)로 썼다. 일반형은 C^{αβ}(q,t)−[C^{αβ}(−q,t)]*이고, C(q)=C(−q)일 때만 2i Im C가 된다. DM 등 비상반 magnon에서 틀린다. | 일반형과 적용 조건을 적었다. 코드에는 아직 이 함수가 없다. | ED: 교환자 0.0992+0.3901i, 2i Im C=0.2057i |
| magnon observables, 양자 감소 예시 | "B_k=0, 예: collinear ferromagnet"이라 썼다. 국소 좌표계에서 J^xx≠J^yy, Kitaev·비대각 교환이 있으면 collinear FM도 B≠0이다. | "모멘트 방향 총 스핀 성분을 보존하는 collinear FM"으로 좁히고 반례를 적었다. | 국소 좌표계의 S⁺S⁺ 계수 |
| magnon observables, ⟨n⟩≥S | "≥S면 모멘트가 뒤집힌다"고 썼다. 같을 때는 0이 된다. | "사라지거나 뒤집힌다" | — |
| diagonalization, trace 상수 | trace 차감을 "HP Hamiltonian을 normal ordering해서 생긴 상수"라 했다. 실제로는 normal-ordered H₂를 대칭 Nambu 형태로 쓸 때 생기는 c-number를 되돌리는 항이다(종결 이슈 260910과 같다). | 설명을 고치고, 아직 provisional인 부분(on-site 항처럼 normal-ordered가 아닌 HP 항의 상수)을 따로 적었다. | 유도, 종결 이슈 대조 |
| diagonalization, 홀 열 순서 | "양수 먼저" 순서만으로는 홀 열 n이 ε_{n,−k}를 갖지 않는다. | "음수 고유값은 −k의 band 순서로" 추가 | 코드 순서 수치 확인(코드는 홀을 반대 순서로 둔다) |
| diagonalization, 부정치 H | "안정한 magnon을 기술하지 않는다"는 과했다. 부정치여도 실수 고유값과 음에너지 mode(자기장 반대로 편극된 FM)가 있을 수 있다. | 복소 고유값(동역학적 불안정)과 음에너지 mode 두 경우로 나눠 썼다. | 2×2 예: H 고유값 −0.51, 3.01인데 Σ3H 고유값 2.98, 0.52 |
| LT, 나선 바닥상태 | "바닥상태는 나선"을 "나선이 바닥상태 중 하나"로 낮췄다. 여러 Q가 최소인 경우(J1–J2, J2=J1/2) 다중 Q 바닥상태가 있다. | 고쳤다. | Lyons–Kaplan 주장 범위 |
| LT, 퇴화 고유공간 | 등방 교환의 3중 퇴화(전역 회전)와 하한 미도달 시의 퇴화를 구별하지 않았다. | 조건을 붙였다. | — |

## 문서별 검토 포인트

각 문서에서 결과를 좌우하거나 원문·표준 문헌과 다르게 쓴 단계다. 줄 번호는 2026-10-01 판 기준이다.

### Paraunitary Diagonalization

- 입자–홀 대칭 Σ1H*_{−k}Σ1=H_k로 홀 block이 ε_{n,−k}임을 보이는 논증(원문의 "E가 실수라서 Ẽ=E" 대체). 비상반 스펙트럼 ε_k≠ε_{−k}의 근거라서 DM 계에 직접 쓰인다.
- ΔE_zp 유도의 −k→k 재표기와 trace 차감. 운동량 집합의 반전 닫힘이 필요하다. 코드는 기본 mesh에서 이 조건을 만족하지만 사용자 지정 `k_points`에서는 확인하지 않는다.
- "양정치 ⇔ 양의 대각형으로 가는 paraunitary T 존재"(증명은 새 appendix).
- 영모드 범위(Open Physical Decision 7): AFM Goldstone은 Jordan block, FM Goldstone은 영에너지 boson이라는 분류, δ^{−1/4} 발산.

### Magnon Observables

- N_k(0)=diag(1+n_k, n_{−k})에서 홀 항목이 자기 band의 −k 점유를 쓰는지.
- P·Q 형태의 보손 수와 "양자 감소는 B에서만 생긴다"(오늘 FM 예시를 좁혔다).
- m_μ=(S_μ−⟨n_μ⟩)n_μ: 횡성분 평균 0은 정지(stationary) 기준 배치에 기대고, O(S⁰)까지만 맞다.
- 2D 영모드 power counting과 Mermin–Wagner 연결.

### Thermodynamics

- 적용 범위 문단: 기준 배치와 ε를 온도에 고정한 저온 근사. 2D Goldstone 계에서는 열적 보손 수가 T>0에서 로그 발산하므로 이 문서의 기준(moment 감소가 S에 견줄 만하면 실패)으로는 모든 T>0에서 통제되지 않는다. F, U, S, C가 유한하다는 진술과 모순은 아니지만, 유한함이 유효함을 뜻하지 않는다는 문장을 본문에 넣을지 결정이 필요하다(오늘 appendix에는 넣었다).
- 명칭: 원문 "specific heat" C=∂U/∂T(extensive)를 heat capacity로 부르고 사이트당 값을 따로 둔 선택. 코드는 사이트당 값이다.
- 영모드 표와 "유한 mesh의 영점 한 점 처리는 이산화 선택"이라는 진술. A=B 점에서는 T가 없으므로 그 점의 처리를 사용자가 확인할 필요가 있다.

### Topological Magnon Quantities

- BdG Kubo 곡률의 부호와 Σ3 가중(무작위 BdG에서 작은 plaquette paraunitary Berry 위상과 1e-7 상대 오차로 일치). A=iη⟨t|Σ3∇t⟩를 물리적 Bloch 상태(ED)로 고정한 D29와 같다.
- **오늘 고친 gauge 문단.** κ가 gauge에 의존하므로 어느 gauge가 물리적인지는 사용자 결정 항목이다. 문서는 full-position gauge(D13, D29와 같음)를 채택하고 이 판단이 추론임을 밝혔다.
- −π²/3 문단: 변형 H_λ=(1−λ)H+λ1로 Chern 합 0을 보이는 논증. Shindou et al. 2013 식 (29) 인용은 검산자가 원문을 열어 보지 못했다(이전 audit는 대조 완료로 기록). Goldstone(양반정치) 경우는 open.
- FHS·Kubo와 D31의 정합성: 문서는 "띠 접촉에서 Chern 수는 정의되지 않음, 정수 값만으로 고립을 증명하지 못함"으로 D31과 일치한다. 접촉을 둘러싼 plaquette의 위상이 π라서 FHS 정수가 반올림으로 정해진다는 D31 admissibility 이유를 본문에 넣을지는 선택 사항이다(현재는 코드 계약으로만 둠).
- κ 부호와 정규화 1/(N_k A_uc)→∫d²k/(2π)², 단위 k_B²T/ħ. notation 문서는 thermal Hall 단위를 아직 "not yet fixed"로 두지만 D29가 층당 k_B²/ħ로 정했다. notation 갱신이 필요하다.

### Spin Correlations

- 오늘 고친 켤레 관계(뒤 문서의 Hermitian 성질과 FDT가 여기에 기댄다).
- S 전개 차수: ⟨n⟩⟨n⟩(O(S⁰))는 두고 연결 ⟨nn⟩(O(S⁰))와 O(S⁰) HP 보정은 뺐다. 문서에 밝혀 두었지만 "S 차수까지"는 깨끗한 절단이 아니다.
- MBZ로 접지 않은 q에서 H_q, T_q를 쓰는 선택과 T_{q+G}=Λ_G T_q.
- 부격자 분해를 diag(V)로 쓰고 원문의 위상 인자를 넣지 않은 선택(원문은 full-position gauge에서 이중 계산).
- 실공간 동시각 상관의 T>0 식. 코드는 T=0만 검증했다.

### Structure Factor and Spectral Function

- 스펙트럼 함수를 교환자의 Fourier 변환으로 다시 정의한 것(원문은 −Im G_R/π). 비대각 성분에서 두 정의가 다르다. 정의 변경 승인이 필요하다.
- g tensor 문장: 횡 vertex만 바꾸면 안 되고 elastic moment도 g_μ m_μ가 된다. 비등방 g에서는 합 규칙 식이 그대로 성립하지 않는다(아직 고치지 않음).
- 합 규칙은 A의 0차 moment(동시각 교환자)다. audit는 "1차 moment"라 적었는데 문서 본문 표현은 맞다.
- broadening: e^{−η|t|}가 꼬리에서 detailed balance를 정확히 지키지 않는다는 점을 현상론 문장에 넣을지.
- 온도 무관성은 온도에 무관한 기준 배치에 기댄다.

### Luttinger–Tisza Method

- 4Q∈G, 2Q∉G 경우(1/4 파수)는 문헌 인용이 아니라 문서 안의 유도이고, x·y 각도에 대한 연속 1-매개변수 퇴화족을 함의한다. 물리적으로 확인이 필요하다.
- 2Q∈G 조건은 "cell-periodic gauge에서 실수 고유벡터"로 쓰면 더 명확하다.
- 유한 mesh에서 −q가 BZ 대표 집합 밖일 때 실수 조건에 gauge 인자 n_a(q+G)=e^{−iG·δ_a}n_a(q)가 필요하다는 점이 빠져 있다.
- 부격자별 Lagrange 하한에 등호 조건이 없다. 모든 부격자가 대칭으로 동등하면 최적 λ_a를 같게 둘 수 있어 이 일반화가 λ_LT보다 나아지지 않는다는 점을 적을지.
- LSWT 연결: 비정합 Q는 회전 좌표계나 정합 근사가 있어야 H_k가 정의되고, 무자기장이어야 한다.

### Worked Example (2026-10-01 신규)

- 원문은 A_k=A_{−k}를 가정했다. 이 draft는 A_k^±로 비상반 경우까지 다뤄 ε_{±k}=ω_k±A_k^−를 유도했다. 원문 범위를 넘는 추가다.
- 안정 조건을 원문의 |A|≥|B| 대신 A_k>0, A_{−k}>0, A_kA_{−k}>|B|²로 고쳤다. |A|≥|B|는 A<−|B|(음정치)와 등호(영모드)를 허용해서 충분조건이 아니다.
- 원문의 μ 전개 오류: ΔE_zp의 μ² 계수는 +B²/(2ω³)이 아니라 −B²/(4ω³), ⟨n⟩의 μ² 계수는 −3AB²/(4ω⁵)가 아니라 +3AB²/(4ω⁵)다. 원문의 ⟨a†a†aa⟩/ω μ² 항은 유도 근거가 없어 옮기지 않았다. ω→0에서 ⟨n⟩≈A/ω도 A/(2ω)로 고쳤다. 원문 Nambu 식의 등호에는 −½ΣA 상수가 빠져 있다.
- 확인: 2-mode Fock ED(비상반 A_k=1.7, A_{−k}=1.1, 복소 B)에서 바닥 에너지, 들뜸 에너지, T=0과 β=1.3의 ⟨n⟩, ⟨a_k a_{−k}⟩가 닫힌 식과 1e-14 이내로 일치했다. 코드의 `Diagonalizer.Colpa`도 같은 ε를 준다. μ 계수는 유한 차분으로 확인했다.

### Appendix: Paraunitarity Proofs (2026-10-01 신규)

- 원문 appendix의 T Σ3 T† 검산과 주석 처리된 T†Σ3T 검산을 보존하고, 고유벡터의 Σ3 직교성, 양정치에서 "고유값 부호 = Σ3 노름 부호", 대각화 가능성, 고유벡터로 T를 만드는 존재 증명, gauge 자유도의 증명, T_k=Σ1T*_{−k}Σ1 구조를 추가했다.
- 검토 포인트: sign rule(양정치가 아니면 깨진다는 문장)과 k≡−k 운동량에서 gauge가 일부 고정된다는 문장.

### Appendix: Thermodynamic Derivations (2026-10-01 신규)

- 원문 appendix의 엔트로피 유도와 상관행렬 유도를 옮기고, Z 인수분해, C=T∂S/∂T의 한 줄 증명, 영모드 표의 작은 x 전개(x⁴ 오차까지 수치 확인), 상관행렬에서 P·Q 형태 보손 수까지의 중간 단계를 추가했다.
- 원문 "adjusting units appropriately" 문구와 β(보손) 표기는 옮기지 않았다.

## 문서 범위 밖에서 발견한 항목 (코드·notation)

이 thread는 `docs/lswt`만 고쳤다. 아래는 다른 소유자에게 넘길 항목이다.

- **코드 docstring과 시험(topology):** `code-space/spintoolkit` 의 `berry.py` 모듈 docstring이 κ의 gauge 불변을 주장한다. D29는 "Chern 수와 kappa의 게이지 불변을 시험"한다고 적었지만, 실제 시험(`test_cell_gauge_gives_the_same_chern_numbers`)은 C3 대칭 Kitaev 모델에서 FHS Chern 수만 본다. 코드는 full-position gauge로 계산하므로 반환값은 문서가 채택한 정의와 같다. 고칠 것은 docstring과 D29 문구, 그리고 C3를 깬 모델에서 두 gauge의 κ가 다르다는 회귀 시험이다.
- **LT on-site 항:** 문서의 L_q는 S_a²D_a를 포함하지만 `lt_matrix`는 bilinear 항만 더한다(모델이 BILINEAR와 ZEEMAN만 받음, D30). audit의 "stage 6b `lt_matrix`와 같은 정의"는 on-site 항이 없을 때만 맞다.
- **영에너지 mesh 점:** 열역학 루틴 세 개가 영점을 서로 다르게 처리한다(−inf, 항 제거, k_B). audit에 이미 기록됨.
- **운동량 집합 반전 닫힘:** 사용자 지정 `k_points`에서 확인하지 않고 ΔE_zp와 ⟨n⟩ 식을 적용한다.
- **notation:** 같은 식에서 β가 Cartesian 첨자와 역온도로 함께 쓰인다(notation은 β_T를 요구). V(Colpa 유니터리 vs vertex V^α), Λ(Colpa 고유값 vs Λ_G) 기호 충돌. thermal Hall 단위가 notation에서 "not yet fixed"로 남아 있다.

## 검증

- 검산 스크립트: 세션 scratchpad(`check_worked_example.py`, `review/` 아래 `ed.py`, `gauge_kappa2.py`, `bdg_kubo.py`, `c2bl.py`, `chk.py`, `lt_check.py`). 저장소에는 넣지 않았다.
- 이 검산은 Human Physics and Mathematics Review를 대체하지 않는다.
