---
frontmatter-version: 1
title: Thermodynamic observables — per-spin densities and sublattice occupations
section: issue-notes/closed
issue-type: problem
status: closed
resolution: resolved
outcome: code-space/lswt/observables/thermodynamics.py
last-edited-by: codex
created: 2026-09-11
updated: 2026-09-11
closed: 2026-09-11
related:
  - code-space/tests/test_solvers/test_thermodynamic_normalization.py
  - GOVERNMENT/Working-Pad/issue-notes/closed/260911-finite-temperature-energy-and-occupation.md
  - GOVERNMENT/Working-Pad/issue-notes/open/260810-lswt-implementation-backlog.md
  - GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md
must-read: GOVERNMENT/Agents-Bylaws/templates/issue-notes-template.md
---

# Thermodynamic observables — per-spin densities and sublattice occupations

## 배경

유한온도 U/F 및 점유수 전달 결함을 수정한 뒤, 개별 entropy 함수와 통합
thermodynamics 함수의 정규화 차이가 남았다. 사용자가 다음 검증을 승인해
스핀당·단위격자당 값, sublattice별 점유수 및 온도 스캔의 일관성을 확인했다.

## 문제 정의

`compute_entropy_density()`와 `compute_specific_heat()`는 k 표본 수로만
나누어 cell당 값을 반환했다. 반면 U와 통합 함수의 S/C는 이미 스핀당 값이었다.
통합 함수는 각 sublattice의 점유수와 이미 sublattice 평균된 boson 수를
N_s로 다시 나누어 실제보다 작게 반환했다. N_s=2이면 각각 2배·1/2배 차이가
나고, 같은 모델의 cell을 복제해 N_s=4로 바꾸면 오류 배율도 달라진다.

또한 부모 solver 없이 `Thermodynamics()`를 사용하면, 통합 함수가 local
`num_sl`을 추론하고도 실제 배열·metric 생성에 `self.Ns=None`을 사용해
TypeError를 냈다. 온도 스캔의 결과 배열에도 같은 문제가 있었다.

## 본론

### 반환값의 물리적 의미

현재 U 반환 경로와 기존 통합 S/C의 기준에 맞춰 density를 스핀당 값으로
통일했다. 여기서 spin은 물리 site 하나를 뜻하며, 분모는 spin magnitude의
합이 아니다. 서로 다른 S를 가진 site도 각각 하나로 센다. 온도는 K,
에너지는 meV이며, 고정된 온도 독립 quadratic Hamiltonian을 가정한다.

| 반환값 | 정의와 정규화 | 단위 |
|---|---|---|
| Internal Energy Density / `compute_internal_energy` | zero-point+thermal 항을 N_k N_s로 나눔; classical energy 제외 | meV/spin |
| Entropy Density / `compute_entropy_density` | band 합을 N_k N_s로 나눔 | meV/(K spin) |
| Specific Heat Density / `compute_specific_heat` | band 합을 N_k N_s로 나눔 | meV/(K spin) |
| Sublattice Boson Numbers | 각 mu의 HP 점유수에 대해 k 평균만 수행 | site당 occupation |
| Total Boson Number | sublattice별 점유수의 산술평균 | spin당 occupation |

`Total Boson Number`라는 기존 key는 유지했지만 extensive total이 아니라
평균이라는 점을 docstring에 명시했다. 물리적으로 필요한 평균은

$$
\bar n_\mu=\frac{1}{N_k}\sum_k n^{\mathrm{HP}}_{k\mu},\qquad
\bar n=\frac{1}{N_s}\sum_\mu\bar n_\mu.
$$

`compute_bosonic_number_at_k()`의 두 번째 반환값은 이미 각 k에서 mu 평균을
낸 값이므로 통합 함수에서는 N_k로만 나눈다. Pairing이 있으면 HP 점유수는
Bogoliubov 변환을 포함하며 magnon n_B와 같다고 놓지 않는다.

Primary PDF 15쪽 식 (76), (77)–(79)의 S/C 전체 합과 식 (80)–(81)의
sublattice 평균, 16쪽 식 (84)–(85)의 correlation matrix를 직접 대조했다.
Reviewed TeX와 기존 source 검토 기록도 참조했다. 원본 전체의 notation이나
중간식이 모두 정확하다고 판정한 것은 아니며, 이번 작업에서 PDF·TeX 또는
`docs/` 이론 본문을 수정하거나 사용자 이론 acceptance를 대체하지 않았다.

### 수정 범위

실행 코드 변경은 `code-space/lswt/observables/thermodynamics.py`에 한정한다.

1. 개별 S/C의 분모를 `valid_count * num_sl`로 수정했다.
2. 통합 sublattice별 점유수와 이미 평균된 boson 수는 `valid_count`로 나눈다.
   U/S/C는 계속 `valid_count * num_sl`로 나눈다.
3. 통합 함수의 metric·Berry 함수 인자와 온도 스캔 배열에 추론된 `num_sl`을
   사용한다. 부모 solver 유무와 관계없이 같은 데이터로 계산할 수 있다.
4. 반환값의 의미와 단위를 docstring에 명시했다. 함수 signature와 결과 key는
   유지했다. 새 모듈 의존성은 없다.

개별 S/C의 과거 값을 이미 외부에서 N_s로 나누던 코드는 추가 나눗셈을
제거해야 한다. 통합 점유수에 N_s를 곱하던 보정도 제거해야 한다. 같은 저장소의
현재 code-space/examples 검색에서는 그러한 보정 호출을 찾지 못했지만,
외부 notebook이나 과거 분석은 확인하지 않았다. 기존 값을 사용한 결과는
반환 의미를 확인한 뒤 다시 계산해야 한다.

### Legacy와 수정 전후 수치

h=(0.2, 0.45) meV, S=(1/2,1), T=1.2 K의 독립 site를 사용했다.
Legacy `LSWT_THER`와 현재 함수를 같은 k_data로 실행했다. 아래 legacy 값은
이번 수정 전 현재 코드와 같아, 구현 차이가 기존 정규화 오류의 수정임을 확인했다.
이는 legacy Hamiltonian 조립 전체를 다시 검증한 실험이 아니다.

| 반환값 | 기존/legacy, N_s=2 | 기존/legacy, 복제 후 N_s=4 | 수정 후, 두 cell 표현에서 동일 |
|---|---:|---:|---:|
| 개별 entropy | 0.047631155243 | 0.095262310485 | 0.023815577621 |
| 개별 specific heat | 0.085255725449 | 0.170511450898 | 0.042627862725 |
| 통합 첫 번째 sublattice 점유수 | 0.084491988637 | 0.042245994319 | 0.168983977275 |
| 통합 평균 boson 점유수 | 0.045509282451 | 0.022754641225 | 0.091018564901 |
| quantum internal energy/spin | 0.019835357046 | 0.019835357046 | 0.019835357046 |

Entropy와 specific heat의 수정 후 단위는 meV/(K spin)이다. 기존 개별 S/C는
cell당 값이었으므로 해당 숫자를 그대로 per-spin이라고 해석하면 안 된다.

### 독립 검증

신규 `test_thermodynamic_normalization.py`는 21건이다. 수정 전에는 17건
실패·4건 통과였고, 수정 후 전부 통과한다.

- 독립 site 및 복소 pairing dimer의 해석적 분배함수로 U/S/C와 HP 점유수를
  검증한다. Pairing은 h1=0.2, h2=0.45 meV, g=0.035 exp(0.43i) meV이며
  모든 quadratic mode가 양수다. 두 mode의 해석적 에너지는
  `[sqrt((h1+h2)^2-4 abs(g)^2) ± (h1-h2)]/2`다.
- N_s=2/4, T=0/1.2 K에서 개별·통합 반환값이 같은 스핀당 해석해와
  절대 허용오차 1e-13 이내로 일치한다. 복제한 cell은 독립 dimer 두 개를
  포함한다. 임의의 folding/BZ 적분 scheme까지 검증한 것은 아니다.
- N_s=4에서 S=-partial F/partial T와 C=partial U/partial T를 별도
  중앙차분으로 확인한다. 1.2 K, step=1e-4 K, 절대 허용오차 1e-9를 사용한다.
- 부모 solver가 있는 경우와 없는 경우 모두, 온도 스캔 0/0.6/1.2 K에서
  scalar 배열 및 sublattice별 행의 정규화를 확인한다.
- 일부 sample에 synthetic invalid flag를 붙여 valid-count 분모 처리를
  검사하고, T>0에서 모두 제외된 통합 결과는 NaN임을 확인한다. 실제
  불안정 mode 제외가 물리적으로 타당하다는 검증은 아니다.
- 전체 테스트: Python 3.13.5, 91 passed, 기존 CommensurateStructure shape
  실패 5건, 보존된 legacy docstring SyntaxWarning 2건. 신규 실패는 없다.

```sh
PYTHONPATH=code-space python -m pytest code-space/tests/test_solvers/test_thermodynamic_normalization.py -q -p no:cacheprovider
PYTHONPATH=code-space python -m pytest code-space/tests -q -p no:cacheprovider --tb=short
```

## 결론 / 미결 사항

에너지·entropy·specific heat의 스핀당 정규화와 sublattice occupation 평균을
일치시켰고, 통합 함수 및 온도 스캔의 Ns 추론 결함을 수정했다. 이 구현 범위를
종결한다.

- Thermal Hall의 volume·온도·단위 계수는 이번 수정 범위에 포함하지 않는다.
  기존 독립 이슈가 계속 열려 있다. 통합 함수가 함께 반환하는 Thermal Hall을
  이번 테스트로 검증했다고 해석해서는 안 된다.
- `exclude_gamma`는 현재 specific heat와 boson 관련 함수에서 실제로 적용되지
  않는다. 이번에는 해당 specific-heat docstring만 사실에 맞게 정정했다.
  Gamma 제외 또는 Goldstone의 물리적 처방을 임의로 추가하지 않았다.
- Invalid sample을 제외한 평균은 전체 BZ의 물리적 열역학 결과와 일반적으로
  같지 않다. T=0의 invalid-state 처리, gapless 한계, 실제 BZ 적분의 가중치·수렴,
  온도에 따라 변하는 기준 상태, 1/S 근사의 유효성 및 논문 재현은 별도다.

## 참조

- `code-space/lswt/observables/thermodynamics.py` — 이번 구현 변경
- `code-space/tests/test_solvers/test_thermodynamic_normalization.py` — 독립 해석 기준과 회귀 테스트
- `legacy/modules/LinearSpinWaveTheory/lswt_thermodynamics.py` — 보존된 legacy 비교 대상
- `research-space/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf` — 15–16쪽
- `research-space/sources/01-editable-notes/note_lswt_reviewed.tex` — thermodynamics 전사
- `docs/02-observables/thermodynamics.md` — theory owner
- [직전 유한온도 구현 이슈](260911-finite-temperature-energy-and-occupation.md)
- [Thermal Hall의 별도 미결 사항](../open/260802-topology-thermal-hall-real-space-volume-bug.md)
- [Goldstone 관련 미결 사항](../open/260810-pseudo-goldstone-gap.md)
- [구현 backlog](../open/260810-lswt-implementation-backlog.md)
