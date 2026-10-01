---
frontmatter-version: 1
title: 단일 Q 나선의 회전틀 LSWT (IncommensurateStructure, D34)
section: idea-proposals
status: in-review
author: claude
last-edited-by: claude
created: 2026-10-01
updated: 2026-10-01
source_refs:
  - conversation: 2026-10-01 사용자 위임("최고의 spin model 범용 솔빙 패키지를 만들어봐")
  - docs/lswt/sources/02-reference-papers/1402.6069v4.pdf
  - GOVERNMENT/Working-Pad/issue-notes/closed/260930-lt-step-and-commensurate-structure-decision.md
related:
  - GOVERNMENT/Working-Pad/idea-proposals/2026-09-23-spin-model-transfer-contract.md
  - docs/development/verification/spiral-rotating-frame-2026-10-01.json
---

# 단일 Q 나선의 회전틀 LSWT (D34)

> 상태: 구현·검증 완료, 사용자 물리·수학 검토 대기. 사용자 위임(2026-10-01)에 따라 추천안으로
> 진행했으며 아래 선택은 모두 되돌릴 수 있다. 근거 문헌은 S. Toth and B. Lake,
> J. Phys.: Condens. Matter **27**, 166002 (2015) (`docs/lswt/sources/02-reference-papers/1402.6069v4.pdf`).

## 1. 문제

LT 진단(D30, D32)은 $\mathbf q^*$가 비정합이면 유한 초격자가 없어 `SpinState`를 만들지 못하고,
`IncommensurateStructure`는 `NotImplementedError`만 내는 자리표시였다. 비정합 나선의 LSWT는
큰 근사 셀(유리수 근사)로 풀 수 있지만 셀이 커지고 $\mathbf q$가 정확하지 않다.

## 2. 상태 표현

$$\mathbf n_a(\mathbf R) = R_{\mathbf n}(2\pi\,\mathbf q\cdot(n_1,n_2))\,\mathbf d_a,\qquad
\mathbf R = n_1\mathbf a_1 + n_2\mathbf a_2 .$$

- $\mathbf q$: 원시 역격자 기저의 분율 좌표($\mathbf Q = q_1\mathbf b_1 + q_2\mathbf b_2$). 유리수·무리수 모두 허용.
- $\mathbf n$: 모든 사이트에 공통인 회전축(단위 벡터), $R_{\mathbf n}$은 오른손 회전.
- $\mathbf d_a$: 원점 셀의 사이트 $a$ 방향. **위상은 셀 번호만 쓴다**(사이트 오프셋 $\boldsymbol\tau_a$는 넣지 않음).
  그래서 $\mathbf d_a$가 곧 원점 셀의 실제 스핀이고, $\mathbf n\cdot\mathbf d_a$가 사이트별 원뿔각이다.
  Toth–Lake도 셀 위치 $\mathbf r_m$으로 위상을 준다(식 35). $(\mathbf q,\mathbf n)$과 $(-\mathbf q,-\mathbf n)$은 같은 상태다.
- `SpinState`와 같이 모델 fingerprint로 모델을 가리키고, 모델에 속하지 않는다.
- 정합 $\mathbf q$이면 `to_spin_state()`가 같은 상태를 초격자 `SpinState`로 만든다(교차 검증과 기존 경로 사용).

## 3. 회전틀 변환과 적용 조건

$\mathbf S_i = R_{\mathbf n}(\phi_i)\mathbf S'_i$로 쓰면 결합 항은
$\mathbf S_i^{T}J\mathbf S_j = \mathbf S_i'^{T} R_{\mathbf n}(\phi_i)^{T} J R_{\mathbf n}(\phi_j)\mathbf S'_j$이다.
$J$가 $\mathbf n$ 둘레의 모든 회전과 교환하면

$$J' (\Delta\mathbf n) = J\,R_{\mathbf n}(2\pi\,\mathbf q\cdot\Delta\mathbf n)$$

로 셀 $\mathbf R$에 의존하지 않는다. 교환 조건은 $J = \alpha(1-\mathbf n\mathbf n^T) + \beta\,\mathbf n\mathbf n^T + \gamma[\mathbf n]_\times$,
즉 축 $\mathbf n$의 XXZ 교환과 $\mathbf n$ 방향 DM 벡터다. Zeeman은 $\mathbf h_a = g_a^T\mathbf b \parallel \mathbf n$이면 그대로다.
그러면 회전틀 모델은 원시 셀 위의 평범한 병진 불변 모델이고 그 고전 상태는 원시 셀의 $\mathbf d_a$이므로,
**검증된 `solve_lswt`가 그대로 나선의 정확한 LSWT를 준다**(새 $H(\mathbf k)$ 구성기를 만들지 않음).
밴드 $\omega(\mathbf k)$의 $\mathbf k$는 회전틀(마그논) 운동량이다.

**결정(물리 우선): U(1) 대칭이 없는 모델은 거부한다.** 대칭이 깨지면 단일 Q 나선은 일반적으로 고전 정상
상태가 아니고(고차 조화가 생김), 마그논 $\mathbf k$가 $\mathbf k\pm2\mathbf Q$와 섞인다(Toth–Lake §V, 식 28).
단일 Q로 자른 LSWT는 통제된 전개가 아니므로 근사값을 내지 않고 `SpiralSymmetryError`로 위반 항을 모두 보고한다.
판정: $\|[J, R_{\mathbf n}(1\,\mathrm{rad})]\| \le 10^{-10}\max(1,\|J\|)$ (일반 각 하나와 교환하면 모든 회전과 교환),
가로 장 $\le 10^{-10}\max(1,|\mathbf h|)$. 정합 $\mathbf q$이면 초격자 경로가 대안이다.

## 4. 정상성

- 토크(국소장 $\times$ 스핀): 회전틀 모델의 기존 진단을 그대로 쓴다.
- 피치: 토크가 0이어도 $\mathbf q$가 고전 에너지의 극값이 아닐 수 있다(하이젠베르크 사슬에서 모든 $\mathbf q$의 토크가 0).
  해석적 $\partial E/\partial\mathbf q$를 결과에 기록하고 0이 아니면 경고한다. 이 경우 LSWT는 음의 모드로
  불안정을 드러내며(시험: $\mathbf q = q^* - 0.05$에서 `LSWTError`), 임의의 값을 내지 않는다.

## 5. 실험실 틀 구조인자

$R_{\mathbf n}(\phi) = R_0 + e^{i\phi}R_+ + e^{-i\phi}R_-$, $R_0 = \mathbf n\mathbf n^T$,
$R_\pm = (1-\mathbf n\mathbf n^T \mp i[\mathbf n]_\times)/2$로 두고 D13 전체 위치 Fourier 규약을 쓰면

$$\mathbf S(\mathbf k) = \sum_{m=0,\pm1} R_m \sum_i e^{-im\mathbf Q\cdot\boldsymbol\tau_i}\,\delta\mathbf S'_i(\mathbf k - m\mathbf Q).$$

회전틀 상관이 병진 불변이므로 $(m-m')\mathbf Q\notin G$이면 교차항이 사라지고
$S^{ab}(\mathbf k,\omega) = \sum_m [R_m S'(\mathbf k-m\mathbf Q,\omega) R_m^\dagger]^{ab}$ (Toth–Lake 식 40과 같은 내용;
행렬곱 형태라 식 40 앞의 회전 대칭화 적분이 필요 없다). 셀 번호 위상 규약 때문에 사이트 위상
$e^{-im\mathbf Q\cdot\boldsymbol\tau_i}$가 붙는다. 탄성 부분은 $\mathbf k = G + m\mathbf Q$에서
$F = R_m\sum_i e^{-i\mathbf k\cdot\boldsymbol\tau_i}\mathbf m_i$. $\mathbf Q$ 또는 $2\mathbf Q$가 역격자 벡터이면
가지 사이 umklapp가 남으므로(Toth–Lake의 "2Q = τ" 예외) 구현하지 않고 초격자 경로를 안내한다.

## 6. 결과 형식과 안전장치

- `solve_spiral_lswt(model, structure, conditions, geometry, settings) -> SpiralLSWTResult`.
  `SpiralLSWTResult.rotating`은 회전틀 모델의 `LSWTResult`(에너지·밴드·보손 수·열역학은 틀과 무관해 그대로 사용).
  머리부 method는 `"lswt-spiral"`, `state_ref`는 나선 fingerprint, 진단에 `wave_vector_gradient`.
- `rotating.extra["frame"] = "rotating"`: 실험실 스핀을 가정하는 `structure_factor`, `spin_correlation`,
  `bond_correlations`는 이 결과를 거부하고 `spiral_structure_factor`를 안내한다.
- Berry 곡률·Chern·thermal Hall: 회전틀 운동량은 좋은 양자수라 정의될 가능성이 높지만 **검증하지 않았으므로
  계산하지 않고 `TopologyError`를 낸다**. 검증(정합 나선의 초격자 대조)은 후속.
- 유한 토러스: 비정합 나선은 맞지 않으므로 열역학 극한만. 정합이면 `to_spin_state` + `solve_lswt`.

## 7. 기존 흐름과의 연결

```
luttinger_tisza(model) ─ 최소 q*, 강한 제약 만족(나선 진폭 u_a)
   ├─ 정합(분모 ≤ 12): LTWaveVector.state → solve_lswt (기존)
   └─ 비정합: IncommensurateStructure.from_lt(model, minimum)
          → refine_spiral(model, s, conditions)   # q, 원뿔각·위상 최소화(축 고정, 해석적 기울기)
          → solve_spiral_lswt → spiral_structure_factor / 열역학
classical_search(model, supercell) ─ 정합 근사 셀의 전역 탐색(기존); 나선 에너지(spiral_energy)와 비교 가능
```

- `LTWaveVector.amplitude`(새 필드, 기본값 None)에 강한 제약 적합 진폭 $u_a$를 보관한다.
  $u_a\cdot u_a = 0$, $|u_a|^2 = 2$일 때 $u_a = \mathbf e_1 + i\mathbf e_2$이고 회전축은 $\mathbf e_2\times\mathbf e_1$.
  사이트마다 축이 다르면 단일 축 나선이 아니므로 거부한다. $2\mathbf q$ 또는 $4\mathbf q$가 역격자 벡터인
  공선·quarter 경우도 거부하고 `LTWaveVector.state`를 안내한다.
- LT는 Zeeman을 무시하므로 축 방향 장의 원뿔 상태는 `refine_spiral(..., conditions)`이 찾는다.
- D17/D28 양자 상태 선택은 나선에 적용하지 않았다(후속).
- 기존 `LSWTSolver`(D30 사용 중단)는 쓰지 않는다. `AbstractMagneticStructure`를 상속하지 않고 `SpinState`와 같은
  frozen dataclass로 바꿨다(기존 스텁 인터페이스는 사용처가 없었다).

## 8. 검증 (`docs/development/verification/spiral-rotating-frame-2026-10-01.json`, 테스트 21개)

| 경우 | 대조 | 최대 차이 |
|---|---|---|
| 삼각 하이젠베르크 120°를 나선($\mathbf Q = K$)으로 | 해석식 $3JS\sqrt{(1-\gamma)(1+2\gamma)}$; 같은 운동량의 √3×√3 초격자 LSWT(에너지, 밴드, 보손 수, $S^{ab}(\mathbf q,\omega)$ 전체 텐서, $t = 0, 0.3$, Bragg) | 9e-15; 2e-16, 5e-15, 1e-12 |
| 정사각 J1–J2 사슬 + 강자성 Jy (비정합) | LT→refine 피치 $\cos 2\pi q = -J_1/4J_2$; $\omega = S\sqrt{(J_k-J_Q)((J_{k+Q}+J_{k-Q})/2-J_Q)}$ | 반올림 수준(BFGS 뒤 Newton, 최대 기울기 2e-16); 6e-15 |
| 같은 나선을 2-사이트 셀로 | 같은 데카르트 운동량에서 에너지·$S(\mathbf q,\omega)$ (셀 번호 위상 규약 확인) | 1e-13, 1e-9 이하 |
| 강자성 + 축 방향 DM | 피치 $\tan 2\pi q = D/|J|$; $q = 1/6$에서 6×1 초격자 | 1e-15; 에너지 0, 밴드 4e-15 |
| 축 방향 장의 원뿔 나선 | $\cos\theta = h/(S(J_0-J_Q))$; $q=1/3$ 초격자 | 3e-11; 에너지 6e-17, 밴드 1e-15 |
| 거부 | 비 U(1) 교환, 축에 수직인 장, 극값이 아닌 피치(음의 모드), 회전틀 결과의 실험실 물리량, 공선 LT 최소 | 오류로 보고 |

## 9. 남은 것

- 사용자 물리·수학 검토: 셀 번호 위상 규약, U(1) 거부 정책, 구조인자 식.
- 후속 후보: 나선 마그논 Berry 곡률·thermal Hall 검증, 2-마그논(세로) 연속체(D26 범위 밖), 나선에 D17/D28 상태 선택,
  비 U(1) 모델의 다중 조화 처리(초격자 근사나 다중 Q).
