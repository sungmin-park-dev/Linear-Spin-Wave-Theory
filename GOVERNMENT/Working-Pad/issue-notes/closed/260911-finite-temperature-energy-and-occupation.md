---
frontmatter-version: 1
title: Finite-temperature energy and occupation — propagation, band sum, and free-energy fixes
section: issue-notes/closed
issue-type: problem
status: closed
resolution: resolved
outcome: code-space/tests/test_solvers/test_finite_temperature.py
last-edited-by: codex
created: 2026-09-11
updated: 2026-09-11
closed: 2026-09-11
related:
  - GOVERNMENT/Working-Pad/issue-notes/closed/260910-zero-point-energy-normalization.md
  - GOVERNMENT/Working-Pad/issue-notes/open/260810-lswt-implementation-backlog.md
  - GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md
must-read: GOVERNMENT/Agents-Bylaws/templates/issue-notes-template.md
---

# Finite-temperature energy and occupation — propagation, band sum, and free-energy fixes

## 배경

T=0 에너지 반환식 수정 이후 사용자가 다음 검증을 승인했다. 앞선 정적 검토에서
확인한 온도 전달 누락과 열에너지 합산 문제를 해석 모델로 재현했고, 자유에너지
함수에서도 추가 결함을 발견했다. 이 이슈의 종결 범위는 아래 세 구현 결함과
동일 경로의 입력·수치 처리다. 전체 유한온도 LSWT의 물리적 검증을 뜻하지 않는다.

## 문제 정의

1. `solve(temperature=T)`는 온도 인자를 받지만 `diagnosing_lswt()`가 이를
   점유수 계산에 전달하지 않고, `lswt_correction()`이 항상 T=0을 사용했다.
2. `compute_quantum_energy(T>0)`는 `Epk * n_B(Epk)`를 band별 배열로 더했다.
   Scalar인 영점 항도 배열에 broadcast되어, pairing이 있으면 나중에 단순히
   배열을 합하는 우회 처리로는 영점 항을 band 수만큼 중복하게 된다.
3. `log_1_m_exp()`의 `result[mask1][mask2] = ...`는 임시 배열만 수정했다.
   유한한 양의 mode들의 열적 자유에너지는 실제 출력에서 전부 0으로 남았다.

## 본론

### 물리량 정의와 적용 범위

고정된 기준 상태에서 양정치이고 gap이 있는 quadratic boson Hamiltonian을
사용한다. 에너지는 meV, 온도는 K이며 beta=1/(k_B T),
k_B=0.08617333262 meV/K다. `temperature`를 에너지 단위라고 적었던
일부 docstring은 실제 구현과 맞지 않아 Kelvin으로 바로잡았다.

이전 이슈에서 검증한 T=0 상수 Delta E_0를 기준으로

$$
n_B(\omega)=\frac{1}{e^{\omega/(k_BT)}-1},\qquad
U_{\mathrm q}(T)=\Delta E_0+\sum_{k,n}\omega_{kn}n_B(\omega_{kn}),
$$

$$
F_{\mathrm q}(T)=\Delta E_0+k_BT\sum_{k,n}
\ln\!\left(1-e^{-\omega_{kn}/(k_BT)}\right).
$$

이는 classical energy를 제외한 값이다. 동일 가중치의 N_k 표본과 cell당
N_s spin을 사용하면 스핀당 값은 각각 N_k N_s로 나눈다. 총 U, F에는
같은 정규화의 classical energy를 더한다. 온도에 무관한 Hamiltonian에서
U=F-T partial F/partial T를 독립 검증 조건으로 사용했다.

`SolverResult.ground_state_energy`는 계속 T=0 에너지다. 유한온도 U나 F로
이 필드의 의미를 바꾸지 않았다. `result.data['boson_numbers']`와
`average_boson_number`는 요청한 온도에서의 HP boson 점유수다. Pairing이
있으면 이것은 단순한 magnon n_B와 다르며 Bogoliubov 변환을 포함해야 한다.

### 노트와 legacy 대조

- Primary PDF 13쪽 식 (63), (65)–(67), 14쪽 식 (69)의 분배함수, U, F를
  직접 대조했다. 같은 section의 reviewed TeX도 대조했다.
- 13쪽 식 (61)의 첫 등식에는 이전 이슈와 같은 trace 계수·k 합 문제가
  남는다. 이 식의 표기를 그대로 구현 근거로 채택하지 않고, 식 (67), (69)의
  최종식과 독립 oscillator 분배함수로 검증했다. 원본이나 이론 본문을 이번에
  수정·승인하지 않았으며, source 보완은 documentation audit에서 추적한다.
- Legacy `lswt_Hamiltonian.py`에도 thermal band 배열과 masked-copy 문제가
  그대로 있다. Pairing이 없는 동일 fixture로 legacy를 실제 실행해 같은 잘못된
  U 배열과 F=0을 확인했다. 이 경우 legacy와의 차이는 의도된 결함 수정이다.
- Legacy `linear_spin_wave_theory.py`의 점유수 계산도 T=0으로 고정돼 있었다.
  Legacy 원본은 보존했다.

### 수정 내용과 영향 경로

- `solver.py`: 기존 `self.T`를 diagnosis의 요청 온도로 갱신하고 kernel에 전달한다.
  동일 solver에서 T>0 → T=0 → 다른 T로 다시 계산해도 온도가 남지 않는지 확인했다.
- `hamiltonian.py`: thermal energy를 `sum(Epk * n_B)`로 더한다. 자유에너지 로그는
  x=omega/(k_B T)가 작을 때 `log(-expm1(-x))`, 클 때 `log1p(-exp(-x))`로
  계산하고 원래 배열에 직접 저장한다. 이전 저에너지 급수의 계수에 의존하지 않는다.
- `energy.py`: `quantum_free_energy_density_func()`의 Kelvin 단위 설명을 수정했다.
  실제 반환은 수정된 저수준 free-energy 경로를 사용한다.
- 함수 signature나 결과 필드는 추가하지 않았다. 영향 경로는 공개 solver의
  점유수, 저수준 U/F, `EnergyFunction`의 자유에너지 및 이를 사용하는 최적화다.
  T>0의 과거 결과는 다시 계산해야 한다. 전체 최적화의 상태 선택·수렴은 이번에
  검증하지 않았다.
- 요청 온도가 음수·NaN·무한대이면 이 경로들에서 `ValueError`로 거절한다.
  Free-energy helper는 유한한 비음수 mode만 받는다.

### Zero mode의 처리 경계

기존 helper는 1e-15 meV보다 작은 양의 에너지도 zero로 간주하고, 합산 시
무한 항을 제외했다. 수정 후 작은 양의 gap은 그대로 계산하며, T>0에서 정확히
0인 unconstrained oscillator의 발산은 -inf로 보존한다. 유한한 mode를 함께
넣어도 이 발산을 합에서 숨기지 않는다.

이는 물리적 Goldstone mode를 해결한 것이 아니다. 유한계의 collective mode,
연속 BZ 적분의 수렴, infrared 처방과 MAGSWT shift 해석은 별도다. 특히
MAGSWT가 인공 gap을 만든 결과는 원래 gapless 모델의 열역학 검증이 아니다.
T=0 helper의 thermal 항은 0으로 정의하며, zero mode의 유한온도 분배함수가
정상화된다고 주장하지 않는다.

### 재현 수치

두 독립 spin-1/2, Zeeman 계수 h=(0.2, 0.45) meV, T=1.2 K를 사용했다.
저수준 U_q/F_q 값은 N_k=2의 합이며, classical energy는 포함하지 않는다.

| 물리량 | 수정 전 | 수정 후 / 독립 기준 |
|---|---:|---:|
| 첫 번째 site 점유수 | 0 | 0.168983977274515 |
| 두 번째 site 점유수 | 0 | 0.013053152528116 |
| U_q (meV) | 배열 [0.011747837275304, 0.067593590909806] | scalar 0.079341428185110 |
| F_q (meV) | 0 | -0.034973344396933 |

위 배열의 순서는 magnon band 정렬 순서다. 스핀당 값으로 환산할 때는
N_k N_s=4로 나눈다. 공개 T=0 ground_state_energy는 -0.1625 meV/spin으로
유지된다. 이 수치는 quadratic 모델 fixture이며 물질의 측정값이 아니다.

### 검증 결과

신규 `test_finite_temperature.py` 17건은 수정 전 15건 실패, 2건 통과였다.
수정 후에는 17건 모두 통과한다.

- 독립 oscillator: T=0, 0.4, 1.2 K에서 U/F, 단위와 scalar 합을 대조한다.
  독립 cell 내용을 복제해 N_s를 늘려도 스핀당 정규화가 유지되는지 확인한다.
- 로그 helper: 작은 x부터 큰 x까지 80자리 Decimal 분배함수 기준과 비교한다.
  Scalar와 array 입력 모두 상대 허용오차 2e-14를 만족한다.
- 실수·복소 pairing dimer: h1=0.7, h2=1.1 meV, abs(g)=0.18 meV,
  phase=0 또는 0.61, T=2 K에서 원래 quadratic boson 행렬의 Gibbs 상태를
  직접 계산한다. Mode당 Fock cutoff 10→14에서 U/F/점유수의 최대 변화는
  9.83e-14 이하, cutoff 14와 현재 구현의 최대 차이는 6.81e-16 이하다.
  이는 full-spin ED가 아니라 quadratic boson 모델의 독립 검증이다.
- U=F-T partial F/partial T 관계는 0.8, 1.2 K에서 중앙차분으로 확인했고,
  절대 허용오차 1e-9 meV를 만족한다.
- 온도 재지정, T=0 복귀, 정확한 zero와 작은 양의 gap 구분, 잘못된 온도
  거절을 확인했다. 기존 B/B† 8건 및 T=0 energy 11건도 계속 통과한다.
- 전체: Python 3.13.5에서 70 passed, 기존 CommensurateStructure shape 실패
  5건, 보존된 legacy docstring의 SyntaxWarning 2건. 새 실패는 없다.

```sh
PYTHONPATH=code-space python -m pytest code-space/tests/test_solvers/test_finite_temperature.py -q -p no:cacheprovider
PYTHONPATH=code-space python -m pytest code-space/tests -q -p no:cacheprovider --tb=short
```

## 결론 / 미결 사항

2026-09-11 후속 갱신: 아래에 기록한 개별 S/C의 정규화와 통합 boson 점유수의
추가 N_s 나눗셈은 [별도 정규화 이슈](260911-thermodynamic-observable-normalization.md)에서
수정·검증했다. 이 문서의 수치와 테스트 수는 그 수정 전 검증 시점의 기록이다.

안정한 quadratic 모델의 온도 전달, U의 band 합, F의 열적 로그 계산 결함을
수정했다. 이 구현 범위를 종결하며, 다음은 implementation backlog에 남긴다.

- 별도 `Thermodynamics` 메서드의 정규화: 같은 두-spin fixture에서
  `compute_entropy_density()`는 0.047631155242518 meV/K를 반환하지만,
  스핀당 분배함수 기준은 0.023815577621259 meV/K다. 현재 결과는 cell당 값과
  일치한다. U의 per-spin 반환과 S/C의 cell당 반환, 문서의 density 의미를
  일관되게 정해야 한다. `compute_specific_heat()`도 코드상 N_s로 나누지 않는다.
- `compute_thermodynamic_quantities_at_T()`에는 이미 평균한 total boson 수를
  N_s로 다시 나누고 sublattice별 점유수도 N_s로 나누는 경로가 보인다.
  이 통합 함수와 개별 함수의 일치 여부는 후속 수치 검증 대상이다.
- 공통 Bose 함수의 극저에너지 근사, invalid-mode 제외, 유한온도 자기질서와
  1/S 근사의 유효성, Goldstone 처리 및 논문 재현은 별도 검증한다.

## 참조

- `code-space/lswt/solvers/solver.py` — 공개 온도 전달과 점유수
- `code-space/lswt/solvers/hamiltonian.py` — U/F 및 안정적인 logarithm
- `code-space/lswt/solvers/energy.py` — 스핀당 자유에너지
- `code-space/tests/test_solvers/test_finite_temperature.py` — 실행 가능한 검증
- `research-space/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf` — 13–14쪽
- `research-space/sources/01-editable-notes/note_lswt_reviewed.tex` — thermodynamics 전사
- `docs/02-observables/thermodynamics.md` — 이론 owner, 이번 변경 없음
- [이전 T=0 에너지 이슈](260910-zero-point-energy-normalization.md)
- [남은 구현 및 정규화 검토](../open/260810-lswt-implementation-backlog.md)
- [이론 검토 상태](../open/260809-lswt-documentation-audit.md)
