---
frontmatter-version: 1
title: LSWT zero-point energy — trace factor verified and solver result corrected
section: issue-notes/closed
issue-type: problem
status: closed
resolution: resolved
outcome: code-space/lswt/solvers/solver.py
last-edited-by: codex
created: 2026-09-10
updated: 2026-09-11
closed: 2026-09-10
source: research-space/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf
related:
  - code-space/tests/test_solvers/test_zero_point_energy.py
  - code-space/lswt/solvers/hamiltonian.py
  - code-space/lswt/solvers/energy.py
  - GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md
must-read: GOVERNMENT/Agents-Bylaws/templates/issue-notes-template.md
---

# LSWT zero-point energy — trace factor verified and solver result corrected

## 배경

B/B† 구현 이슈를 종결한 뒤, 사용자가 LSWT 결과의 물리적 정확성을 물었다.
다음 검증 단위로 영점에너지의 1/2 계수, trace 상수항, bond 계수 및 스핀당
정규화를 선택했다. 원본 노트 A2·A5·A10, 관련 A11·A12와 실제 에너지 계산
경로를 대조했다. 이 기록은 구현 수정과 source 검토 근거를 소유하며,
사용자 승인된 이론 정본을 대신하지 않는다.

## 문제 정의

`LSWTHamiltonian.compute_quantum_energy(T=0)`와
`EnergyFunction.quantum_energy_density_func()`는 올바른 trace subtraction을
사용하고 있었다. 그러나 공개 `LSWTSolver.solve()`는 `ground_state_energy`에
`classical_energy + mean(magnon_eigenvalues)`를 넣었다. Magnon 에너지의 평균은
영점에너지 보정이 아니며, 필요한 1/2와 trace 상수항이 빠져 있었다.

독립 스핀 S=1/2, H=-h Sz, h=0.8에서 정확한 스핀당 에너지는 -0.4다.
수정 전 공개 솔버는 +0.4, 저수준 에너지 경로는 -0.4를 반환했다. 단위는
h와 같은 에너지 단위이며 h는 자기장 B 자체가 아니라 Zeeman 계수다.

## 본론

### 검증의 가정과 에너지 정의

- 실수 Cartesian inter-site bilinear exchange와 -h·S를 사용한다.
- Coupling 목록은 magnetic unit cell당 각 물리 bond를 한 번 포함한다.
  같은 sublattice끼리의 bond도 서로 다른 물리 site를 잇는다.
- 고정된 기준 스핀 상태 주위의 quadratic HP Hamiltonian을 검증한다.
  별도 on-site anisotropy, higher-order 1/S 항과 바닥상태 최적화는 포함하지 않는다.
- N_s는 magnetic unit cell의 spin 수, N_k는 동일 가중치의 운동량 표본 수다.
  e_cl은 이미 스핀당 정규화한 고전 에너지다.
- 양정치인 quadratic Hamiltonian과 완전한 MBZ 합 또는 그 대칭을 보존하는
  적분을 가정한다. 일반 band-path 표본을 같은 에너지 적분으로 취급하지 않는다.

### Normal ordering에서 trace 상수항까지

Normal-ordered quadratic Hamiltonian을

$$
\hat H_2=\sum_k a_k^\dagger A_k a_k
+\frac12\sum_k\left(a_k^\dagger B_k a_{-k}^\dagger+\mathrm{h.c.}\right)
$$

로 정의하고 Psi_k = (a_k, a†_-k)를 사용한다. Nambu 행렬의 hole block을
normal-ordering하면

$$
\frac12\sum_k\Psi_k^\dagger H_k\Psi_k
=\hat H_2+\frac12\sum_k\operatorname{Tr}A_{-k}.
$$

따라서 MBZ 전체에서 k↔-k로 합을 바꿀 수 있을 때

$$
\hat H_2=\frac12\sum_k\left(\Psi_k^\dagger H_k\Psi_k-
\operatorname{Tr}A_k\right).
$$

Bogoliubov 변환 뒤의 vacuum contribution을 포함하면

$$
\Delta E_0=\frac12\sum_k\left(\sum_{n=1}^{N_s}\omega_{kn}
-\operatorname{Tr}A_k\right).
$$

H_k의 전체 trace는 Tr A_k + Tr A_-k다. A_k가 Hermitian이라는 이유만으로
각 k에서 Tr H_k = 2 Tr A_k라고 할 수는 없다. 비상호적 분산에서는 실제로
A_k와 A_-k가 다르다. 그러나 완전한 합에서는

$$
\sum_k\operatorname{Tr}H_k=2\sum_k\operatorname{Tr}A_k,
$$

이므로 구현에 필요한 스핀당 에너지는

$$
e_0=e_{\mathrm{cl}}+
\frac{1}{N_kN_s}\sum_k\left[
\frac12\sum_n\omega_{kn}-\frac14\operatorname{Tr}H_k\right].
$$

**저수준 코드의 -Tr(H)/4는 이 convention에서 맞다.** 이를 -Tr(H)/2로
바꾸면 독립 스핀이나 number-conserving 강자성체에서도 잘못된 vacuum
에너지가 생긴다. B=0이면 sum_n omega_kn = Tr A_k이므로 완전한 합의
영점 보정은 0이어야 한다.

### 원본 노트와 review annotation 판정

| 항목 | 직접 대조한 결과와 보완 방향 |
|---|---|
| A5; 원본 8쪽 식 (41) 뒤 | -1/4를 -1/2로 바꾸라는 review annotation은 위 유도와 맞지 않는다. -1/4는 유지하고 RHS에도 k 합을 명시해야 한다. Tr H=2 Tr A는 일반적으로 합 전체에서 성립한다는 조건을 써야 한다. |
| A2; 원본 9쪽 식 (47) | 대표 bond를 한 번 세는 convention에서 같은 sublattice의 field coefficient는 h, exchange diagonal 기여는 bond의 두 endpoint 때문에 -2 S Jzz다. 따라서 괄호 전체의 2를 일괄 삭제하거나 field까지 두 배로 해서는 안 된다. 원본 link 집합의 의미를 명시한 뒤 식을 정리해야 한다. |
| A10; 원본 8쪽 식 (41) | 다른 cell의 같은 sublattice hopping이 trace 합에서 사라지는 근거는 완전한 Fourier 합이다. 임의의 k subset에서는 보장되지 않는다. DM이 hopping을 복소수로 만드는 것과 Hermitian conjugate 관계가 깨지는 것은 다르다. |
| A11; 원본 8쪽 식 (38), (42), (43) | mu_i=h_i^0-sum_j S_j Jzz와 radial derivative의 관계는 mu_i=-(partial E_cl/partial S_i)·n_i다. 원본에는 이 부호 및 중간식의 불일치가 있다. 같은 S이고 inter-site bond만 있을 때 -1/2 sum_i mu_i=(E_exc+E_zm/2)/S를 얻지만, 이것은 trace 상수 부분이며 전체 영점 보정이 아니다. S(S+1) 표기는 E_cl의 S 의존 정의까지 구분한 후 검토해야 한다. |
| A12와 원본 10쪽 식 (53)–(56) | 식 (55), (56)의 최종 에너지와 density는 위 결과와 일치한다. 중간 식 (53)에는 trace 항의 k 합이 생략돼 있고, 식 (54)는 trace를 sigma 합 안에 넣어 문자 그대로 읽으면 band 수만큼 반복한다. Trace는 k마다 한 번만 빼야 한다. E_cl, Delta E_0, E_0와 스핀당 e_0를 구분해야 한다. |

A10의 유한격자 조건은

$$
\frac{1}{N_k}\sum_k e^{ik\cdot R}
=\delta_{R,0\;\mathrm{mod}\;\mathrm{periodic\;supercell}}
$$

이다. R이 그 유한계에서 0으로 접히지 않는 inter-cell translation일 때만
해당 hopping 평균이 0이다. 일반 수치 BZ 적분의 수렴을 이 항등식으로
자동 인증하지 않는다. 이번 DM chain 테스트는 완전한 8점 periodic grid와
그 일부만 취한 표본을 비교한다.

이 표는 source-grounded 검토 결과와 수정 제안이다. Primary PDF·TeX 및
`docs/` 본문은 이번 작업에서 변경하지 않았으며, annotation 자체를 권위로
삼아 수식을 자동 수정하지 않았다. 해당 이론 검토 항목은 audit에서 계속
`open`으로 관리한다.

### 구현 수정과 영향 범위

`code-space/lswt/solvers/solver.py`의 `solve()`에서 각 k의 magnon 에너지와
동일한 `k_data` entry의 H_k를 함께 읽어, sum(omega)/2-Tr(H_k)/4를 계산한 뒤
N_k N_s로 정규화한다. 이미 계산한 결과를 재사용하므로 추가 대각화는 없다.

- 기존 `SolverResult.ground_state_energy` 필드와 함수 signature를 유지한다.
- Spectrum, Hamiltonian 조립, B/B† 배치, 미분 및 legacy 원본은 변경하지 않는다.
- MAGSWT가 적용되면 spectrum과 동일하게 이동된 H_k의 trace를 쓴다.
  이는 regularized quadratic 모델의 에너지이며, 원래 불안정한 모델의
  정확한 바닥에너지라고 해석할 수 없다.
- 기존 `EnergyFunction` 기반 최적화와 이전 V상 임시 검증은 이미 올바른
  저수준 trace 식을 사용했으므로 이번 공개 반환값 수정의 대상이 아니다.
- `LSWTSolver.solve().ground_state_energy`를 저장하거나 비교한 과거 결과는
  다시 계산해야 한다. 어떤 논문 결과가 이 API를 사용했는지는 확인하지 않았다.

### 해석해와 독립적인 Fock-space 검증

`code-space/tests/test_solvers/test_zero_point_energy.py`에 11개 테스트를 추가했다.
수정 전에는 공개 반환값 검증 9건이 실패하고, trace/Fourier 조건 검증 2건이
통과했다. 수정 후 11건이 모두 통과한다.

| 모델 | 독립 기준 | 수정 전 공개 에너지 | 수정 후 공개 에너지 |
|---|---|---:|---:|
| 독립 spin-1/2, h=0.8 | -hS | +0.4 | -0.4 |
| 독립 S=(1/2,1), h=(0.7,1.3) | -mean(h_i S_i) | +0.175 | -0.825 |
| 1-site cell 강자성체, S=1/2, Jx=-0.2, Jy=-0.11, h=0.4 | (Jx+Jy) S²-hS; Delta e_0=0 | +0.4325 | -0.2775 |
| 쌍생성 dimer, h1=0.7, h2=1.1, abs(g)=0.18 | quadratic boson 해석해와 Fock-space 대각화 | +0.1568163074 | -0.7340918463 |

표는 테스트 fixture의 스핀당 에너지이며, 물질 측정값이 아니다. Dimer의
quadratic Hamiltonian은

$$
H_2=h_1 n_1+h_2 n_2+g a_1^\dagger a_2^\dagger+g^*a_2a_1,
\qquad
\Delta E_0=\frac{\sqrt{(h_1+h_2)^2-4|g|^2}-h_1-h_2}{2}.
$$

실수와 복소 g를 모두 검사했다. Bosonic Fock cutoff를 mode당 7에서 11로
늘릴 때 에너지 변화는 1.41e-13 이하이고, cutoff 11과 해석해의 차이는
3.47e-17 이하다. Public solver의 스핀당 에너지도 절대 허용오차 1e-13을
만족한다. **이 대각화는 quadratic boson 모델의 검증이며, 유한 S의 원래
상호작용 spin 모델에 대한 LSWT 정확도를 보장하는 full-spin ED가 아니다.**

추가로 unit cell에 독립 dimer를 두 번 넣어 N_s와 band 수가 늘어도 스핀당
에너지가 변하지 않는지 확인했다. Number-conserving DM chain에서는 Tr H_k와
2 Tr A_k가 개별 k에서 다르지만 k,-k 합에서 일치하고 영점 보정이 0인지
검증했다. 같은 sublattice 강자성체의 dispersion은 field를 한 번, exchange
endpoint를 두 번 세는 식과 일치한다.

### 실행과 확인 결과

프로젝트 의존성이 설치된 Python에서 저장소 루트를 기준으로 실행한다.

```sh
PYTHONPATH=code-space python -m pytest code-space/tests/test_solvers -q -p no:cacheprovider
PYTHONPATH=code-space python -m pytest code-space/tests -q -p no:cacheprovider --tb=short
```

- 환경: Python 3.13.5, 로컬 Miniconda.
- Solver 회귀 테스트: 기존 B/B† 8건과 신규 에너지 11건, 총 19건 통과.
- 전체 테스트: 53건 통과, 기존 CommensurateStructure 실패 5건.
- 기존 legacy docstring escape SyntaxWarning 2건은 보존된 원본에서 발생한다.

## 결론 / 미결 사항

2026-09-11 후속 갱신: 아래에 기록한 유한온도 band 합 및 온도 전달 결함과
추가로 발견한 free-energy 계산 결함은
[별도 구현 이슈](260911-finite-temperature-energy-and-occupation.md)에서
수정·검증했다. 이 문서의 수치와 테스트 수는 2026-09-10 검증 시점의 기록이다.

공개 솔버의 T=0 energy assembly 결함은 수정하고 회귀 테스트로 종결한다.
저수준 -Tr(H)/4를 변경할 근거는 없으며, 원본의 일부 review annotation과
중간식에 대해 더 정확한 보완 방향을 기록했다.

다음은 별도 검토 범위다.

- A2/A5/A10/A11/A12의 이론 본문 반영 및 사용자의 Human Physics and Mathematics Review.
- 임의 모델에서의 고전 상태 최적화, 선형항 소거, Goldstone mode와 LSWT 근사 오차.
- Nonzero MAGSWT shift로 얻은 에너지의 물리적 해석 및 안정성 판정.
- 유한온도 처리: 코드 열람상 `compute_quantum_energy(T>0)`는 thermal band
  contribution을 scalar로 합하지 않고, 공개 solver의 temperature 전달도
  실제 occupation 계산에 반영되지 않는다. 이번 T=0 수정으로 이 문제를
  해결했다고 간주하지 않는다. 후속 범위는 implementation backlog에 기록한다.
- 모든 BZ 생성 방식의 적분 가중치·경계·수렴, 논문 결과의 재현 및 기존
  CommensurateStructure 실패는 이번 검증에 포함하지 않는다.

## 참조

- `code-space/lswt/solvers/solver.py` — 수정한 공개 에너지 반환 경로
- `code-space/lswt/solvers/hamiltonian.py` — 유지한 저수준 trace 식
- `code-space/lswt/solvers/energy.py` — 고전 에너지 및 스핀당 정규화
- `code-space/tests/test_solvers/test_zero_point_energy.py` — 해석 모델과 Fock-space 회귀 기준
- `research-space/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf` — 원본 8–10쪽
- `research-space/sources/01-editable-notes/note_lswt_reviewed.tex` — 원문 순서의 전사 대조
- `docs/01-derivation/momentum-space-bdg-hamiltonian.md` — trace 유도의 이론 owner
- `docs/02-observables/magnon-observables.md` — zero-point correction과 energy density의 이론 owner
- `GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md` — 열린 이론 검토
- `GOVERNMENT/Working-Pad/issue-notes/open/260810-lswt-implementation-backlog.md` — 유한온도 후속 구현
