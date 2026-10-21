---
title: 패키지 이론적 신뢰성 확인
topic: toolkit-development
priority: 1
created: '2026-10-21'
state: active
task: BCAO 계산(연구 주제 A) 전에, 패키지가 문헌값과 닫힌 식, 독립 구성을 실제 계산으로 재현하는지 확인한다. 2026-10-20 감사(코드 점검)의 다음 단계다.
end-condition: 아래 기준 문제가 모두 통과 또는 원인이 적힌 불일치이고, 열린 질문에 사용자 판정이 적혀 있다.
references:
- workbench/tasks/2026-10-15-remaining-development.md
- data-space/verification/261021-theory-reliability/tables.txt
- code-space/tests/test_methods/test_reliability_benchmarks.py
---

## 기준 문제와 결과 (2026-10-21)

이미 있던 검증(정사각 ⟨n⟩ = 0.1966, 삼각 1/S² 에너지, Hamer χ⊥, 붕괴 문턱, MAGSWT 이동량, kagome 열 홀, magnopy 비교)은 다시 하지 않고 빈 곳만 골랐다. 테스트는 `code-space/tests/test_methods/test_reliability_benchmarks.py`, 표는 `data-space/verification/261021-theory-reliability/tables.txt`.

| | 대상 | 기준 | 결과 |
|---|---|---|---|
| B1 | BCAO 결합 행렬 | 입방축 K–Γ–Γ′ 모형을 결정축으로 돌린 것, Maksimov et al. arXiv:2204.09695 식 (1)–(3) | 1e-16 일치. x축 방향 하나를 고르면 맞고, 반대 방향은 Jz± → −Jz±(z축 C2, 에너지 같음) |
| B2 | BCAO 고전 바닥상태 | 패키지와 무관한 최소화 코드 | 에너지 같음. 위상 관례(결합에 수직·평행·역회전)를 바꿔도 줄무늬 상태 에너지는 같다 |
| B3 | 벌집 J1–J3 LT | FM → 나선 J3 = 0.25, 나선 → 지그재그 J3 = 0.4 (2204.09695가 인용한 Rastelli 1979) | 일치, LT 에너지는 독립 계산과 1e-4 |
| B4 | 면내 장 편극상 LSWT (Jz±, Δ 포함) | 손으로 만든 Holstein–Primakoff 블로흐 행렬 | 1e-15 일치 |
| B5 | 정사각 1/S 마그논 | 균일 재규격화 Z_c = 1 + 0.1579/(2S) (Oguchi 1960) | 상대 2e-5 |
| B6 | 삼각 XY 극한 에너지 | Chernyshev–Zhitomirsky 2009 식 (92): −(3/2)S²[1 + 0.064515/2S + 0.013326/(2S)²] | LSWT 1e-7, S⁰ 차수 3e-4 |
| B6 | 삼각 Goldstone 속도 1/S | 같은 논문 식 (56): 1 − 0.11488/2S, 1 + 0.08285/2S | 메시 격자 위 k에서 2 % 안. 격자 밖 k에서는 크게 틀림 → D69 |
| B7 | 벌집 Néel S = 1/2 | Zheng–Oitmaa–Hamer 1991: E/(NS²) = −3/2 − 0.31476/S, M/S = 1 − 0.2582/S (arXiv:1811.04229에 인용된 값) | α = −0.31476, 2차 합 −0.5489, ΔS = 0.258 |
| B8 | 삼각 장 속 1/3 평탄부 경계 (UUD, 1/S) | Chubukov–Golosov 1991: h_c1 = 3JS − 0.5JS/(2S), h_c2 = 3JS + 1.3JS/(2S) (Starykh 리뷰 arXiv:1412.8482 인용, 두 자리) | 0.501, 1.289. S = 1/2, 1에서 같음. 공선이라 3차 꼭짓점 없이 HF 이동만으로 정해진다 |
| B9 | 낮은 장의 영점 에너지 상 선택 (Y 대 원뿔) | 고전 축퇴, 1/S에서 Y 선택. 감수율 χ = (1 + c/2S)/9J의 합 공식(Keselman et al. arXiv:2512.02150 End Matter 식 (3), (4))을 독립 계산: c_Y = −0.2844, c_cone = −0.4485 | Y가 모든 장에서 낮다. 메시 240, h ≤ 0.0075에서 −0.2835, −0.4476. h를 키우면 맞춘 계수가 움직인다(h = 0.02–0.08에서 −0.278): h² 밖의 비해석 항으로 보이며, 장 미분은 작은 h에서 구해야 한다 |
| B10 | 장 속 기울어진 BCAO 상태 LSWT (지그재그, 이중 지그재그, 면내·면외·사선 장, Jz±·Δ 포함) | 국소 좌표계와 초격자로 손수 만든 HP 행렬(항과 스핀 방향만 씀) | 1e-15 일치 |
| B10 | 편극상이 무른 장 = 고전 포화 장 | J3 > 0.4에서 M점: h_c = S(\|J1 − 3J3\| + 3(J1Δ1 + J3Δ3)) (Jz±와 무관), Jz± = 0에서 h_ab = S(\|J1 − 3J3\| − \|3J1 + 3J3\|) | 닫힌 식과 0.1 % 안. 고전 탐색의 자화도 h/h_s로 커져 h_s에서 1이 된다(연속 전이). Jz± = −0.8이면 면내 편극상은 그 장에서 이미 불안정하다(Jz±가 두 가로 성분을 잇는다) |
| B11 | 비등방 one-loop (Δ, Jz±; HF·3차·tadpole 모두 0 아님) | 큰 S 정확 대각화, 3×1 토러스 6자리 편극 BCAO, S = 1–3 (토러스에서 1/S 전개는 정확한 섭동론) | (E_ED − E_1L)·S가 −1.25e-3 → −1.17e-3으로 수렴(S⁰ 차수까지 정확). 들뜸 에너지 오차도 1/S² 상대 |
| B12 | 자기무모순 one-loop (SCOL, D72) | 정사각 Néel(고정점 = one-loop), XYZ 큰 S 정확 대각화, 가우스 에너지 정류성, UUD n↓ = 2n↑, 큰 S에서 Chubukov–Golosov | 모두 통과. 삼각 UUD S = 1/2: h_c1 = 1.2665, h_c2 = 2.0517, \|⟨Sz⟩\| 위 0.4168·아래 0.3336. Keselman et al. 2512.02150의 SCOL·DMRG 숫자는 그림뿐이라 대조 못 함(미확인). 표 `data-space/verification/261021-self-consistent/tables.txt` |

## 고친 것

- **D72** (사용자 "self-consistent를 일반 패키지로 더하자"): `solve_scswt` 추가. 3차 꼭짓점이 없는 상태만(있으면 Σ₃·tadpole이 같은 차수라 오류와 선택지). `frozen=`(다른 장의 HF 고정; S^z_tot 보존이면 정확), `initial=`. 간극 닫힘 너머 고정 모드는 에너지 NaN. 잠정 승인, 비교 대상 Keselman SCOL.
- **D69**: 비공선 상태에서 영역 중심으로 접힌 Goldstone 모드 근처의 1/S 마그논 에너지가 두 큰 항(HF+tadpole, Re Σ₃, 각각 1/|k|)의 상쇄로 나오는데, 격자 밖 k에서는 메시 오차가 커서 경고 없이 0.4–3.3배 틀렸다(기대 1.08). 상쇄 비 ρ > N/4이면 NaN과 경고. `magnon_energies`, `spectral_function`. 잠정 승인, 비교 대상은 CZ09 식 (56).
- **D71** (사용자 "η=0 주값 처리 … 진행하자", 추천안 NaN): 띠가 메시의 두 마그논 붕괴 연속체 안에 있으면(꼭짓점이 0인 통로 제외) `magnon_energies(eta=0)`의 에너지는 NaN과 경고. 주값이 필요하면 `spectral_function`의 `energy`.
- **D70** (사용자 승인 "그렇게하자"): `classical_search`의 기본 시드가 국소적으로 안정한 국소 최소에 갇혔다(BCAO J3 = 0.45, Jz± = 0.5, 셀 [[1,0],[0,6]]: 기본 시드 −0.30361(UUD), 다른 시드 −0.30419). 이제 기본값 `restarts="adaptive"`가 시드를 바꿔 다시 돌린다. 초격자 LT 하한에 닿으면 바로 멈추고, 아니면 연속 2번 낮아지지 않을 때(최대 8번) 멈춘다. 결과에 시드별 에너지와 하한을 적는다.

## 열린 것 (사용자 판정 필요)

1. **Maksimov J3 = 0.45 경계.** 논문(2308.10672)은 UUD → 이중 지그재그 경계를 Jz± ≈ 0.9로 적었는데, 같은 해밀토니안에서 두 코드 모두 0.55 근처다. Jz± = 0.7에서 이중 지그재그(−0.34152)가 UUD(−0.33337)보다 낮은 변분 상태가 있으므로 논문 값을 같은 모형으로 재현할 수 없다. J3 = 0.2의 삼중 지그재그 상한 0.6과 J3 = 0.34의 이중 지그재그 시작(≈0.2)은 맞는다. 원인은 논문 쪽 계산(정규화 또는 최소화)으로 보이며 단정하지 않는다. BCAO 연구에서 이 차이를 기록하고 넘어간다.
