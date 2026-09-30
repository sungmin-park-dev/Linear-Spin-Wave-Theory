---
frontmatter-version: 1
title: Topology thermal Hall real_space_volume bug
section: issue-notes/open
issue-type: problem
status: in-review
last-edited-by: codex
created: 2026-08-02
updated: 2026-09-12
source: docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf
related: code-space/spintoolkit/observables/topology.py
must-read: GOVERNMENT/Agents-Bylaws/templates/issue-notes-template.md
---

# Topology thermal Hall `real_space_volume` bug

## 배경

2026-09-11 후속 검증: 아래 과거 기록의 단순 계수 교체는 일반해가 아니다.
노트·참고 논문과 독립 두-band 모델을 대조해 area, 출력 차원 및 온도 인자의
혼용을 확인했다. 이후 사용자가 층당 kappa [W/K] 기본 및 명시적인 층간격을
통한 3D 환산을 승인했다. SI 반환 계약의 신규 32개 검사에 이어 BZ 범위와
Chern 정규화의 신규 40개 검사도 통과했다. 그 단계의 전체 결과는 163 passed와
기존 CommensurateStructure 5 failed였다. Band-isolation과 2026-09-12의 조절 가능한
고정 cutoff까지 포함한 현재 결과는 **198 passed, 기존 5 failed**다. 아래 과거 기록과 수정 전 수치는
역사 기록이며, 현재 구현·판정은 후반의 BZ 검증, 재계산·퇴화 조사 절과 결론에 둔다.

과거 CLAUDE.md/AGENTS.md의 "알려진 버그" 2번을 추적 가능한 issue-note로 옮긴
것이다. 당시에는 코드를 수정하지 않았다.

## 문제 정의

`Topology` 클래스의 thermal Hall conductance 계산에서 `real_space_volume`
산출에 `self.Ns` 곱셈이 빠져 있고, meV·Å 단위를 W/(m·K)로 바꾸는 변환
계수도 `1e-12`가 아니라 `1e-22`여야 한다고 CLAUDE.md에 기록돼 있다.

## 본론 — 2026-08-02 당시 기록

### 증상

`code-space/lswt/observables/topology.py:250-252`:

```python
real_space_volume = valid_count * np.sqrt(3) / 2
coefficiten_thc = (K_BOLTZMANN_MEV ** 2) / (H_BAR_MEV * real_space_volume)
coefficiten_thc *= 1.602176634 * 1e-12  # from meV, Angstrom to W/(m*K)
```

CLAUDE.md 기록과 대조하면:

1. `real_space_volume`이 `valid_count * np.sqrt(3) / 2`로만 계산되고
   `self.Ns` 곱셈이 없다 — 기록된 지적과 일치.
2. 단위 변환 계수가 `1.602176634 * 1e-12`로 남아 있다 — 기록된 `1e-22`
   교체가 아직 반영 안 됨.

이 audit에서는 CLAUDE.md 기록과 현재 코드가 여전히 같은 상태임을
재확인했을 뿐, `self.Ns`를 어디에 곱해야 하는지·`1e-22`가 맞는 값인지는
직접 재유도하지 않았다.

### 재현 조건

현재 구현에서 thermal Hall conductance를 physical unit으로 환산하는 모든 계산이 이 volume과 변환 계수를 사용한다. 다만 기대값을 고정할 독립 benchmark와 회귀 테스트는 아직 이 issue에 연결되어 있지 않다.

### 근본 원인 분석

코드와 과거 버그 기록의 불일치는 확인되었다. 그러나 `self.Ns`를 곱해야 하는 물리적 normalization과 `1e-22` 계수의 독립적인 단위 재유도는 아직 완료되지 않았으므로, 기록된 수정값의 최종 타당성은 `Unknown`이다.

### 해결 방안

1. Thermal Hall conductance 공식을 원본 PDF 노트(§Skyrmions, Topological
   Magnons, and Hall Effects)와 다시 대조해 `self.Ns` 곱셈 위치와
   단위 변환 계수 `1e-22`의 근거를 확인한다.
2. 물리적 의도가 불명확하면 임의로 고치지 않고 성민 확인을 받는다
   (프로젝트 규칙 4).
3. `docs/lswt/02-observables/topological-magnon-quantities.md`의
   review ledger ID **A7**("Thermal-Hall band sum", 현재 `open`)이 이
   코드 버그와 관련된 이론 검증 항목이다 — 함께 다룬다.
4. 수정 후 단위와 크기 order를 독립적으로 sanity-check한다(예: 알려진
   물질의 실측 thermal Hall 값과 order-of-magnitude 비교).

## 2026-09-11 독립 검증 — 실행 코드 수정 전

### 원문과 참고 논문

Primary PDF 24쪽 식 (145)–(149)를 직접 확인했다. 식 (148)은 hbar를 명시하지
않고 c2-pi²/3를 사용한다. 참고 논문 Neumann et al., arXiv:2109.00278v1의
식 (8)은 -k_B²T/(hbar V), physical band 합, c2만을 사용하며 V가 volume 또는
area임을 명시한다. 이 논문은 층간 결합을 무시한 2D 응답을 7.278 Å의 층간격으로
나누어 MnPS3의 3D 값으로 환산한다. 이 수치는 NBCP의 층간격이 아니다.
[논문 식 (8) 및 바로 뒤의 환산 설명](https://arxiv.org/html/2109.00278v1).

Matsumoto–Murakami, arXiv:1106.1987v1의 식 (23)도 같은 온도·hbar 인자를 가지며,
본문에서 2D V는 면적이라고 정의한다. [원 논문](https://arxiv.org/html/1106.1987v1).

Physical band n의 curvature를 구할 때 intermediate index m은 2N_s Nambu
공간 전체를 돌지만, 열수송 합은 N_s physical band를 센다. 두 합의 역할이
달라 자동적인 2배 계수를 넣지 않는다. c2와 c2-pi²/3의 적분이 같으려면
전체 physical band의 적분된 curvature 합이 0이어야 한다. 이번 두-band 모델은
이를 만족하지만, 임의 band 절단이나 불완전한 BZ 합에도 같다고 가정하지 않는다.
이론 A6/A7/A16 acceptance와 원문 수정은 아직 수행하지 않았다.

### 단위와 면적의 재유도

Cartesian k에 대한 curvature를 사용하면 단일 2D 층의 응답은

$$
\kappa_{xy}^{2D}=-\frac{k_B^2T}{\hbar}\int_{\mathrm{MBZ}}
\frac{d^2k}{(2\pi)^2}\sum_{n=1}^{N_s}c_2[n_B(\omega_{nk})]\Omega_{nk}.
$$

같은 가중치로 완전한 magnetic BZ를 표본화하면 적분은
`sum(c2 * Omega)/(N_k * A_mag)`와 같다. A_mag는 실제 magnetic unit cell의
면적 abs(det(a1,a2))이다. 임의의 band path나 불완전한 적분 구간에 같은 평균을
적용해서는 안 된다. Primitive BZ를 여러 번 덮는 방식도 coverage를 확인해야 한다.

- Curvature의 길이²와 k 적분의 길이^-2는 상쇄된다. 층당 kappa의 단위는 W/K,
  kappa/T는 W/K²다.
- k_B를 meV/K, hbar를 meV s로 쓰면 남은 meV→J 변환은 1.602176634e-22다.
- 층간격 d_m을 명시하면 kappa_3D=kappa_2D/d_m이고 단위는 W/(m K)다.
  d를 Å 숫자로 입력하면 변환의 조합이 1.602176634e-12/d_A가 된다.
  따라서 1e-12가 언제나 틀리고 1e-22가 언제나 옳다는 과거 지적은 부정확하다.
- Ns는 일반적인 면적이 아니다. A_mag=Ns*(sqrt(3)/2)*a²인 특정 triangular
  cell에서는 Ns가 나타나지만, 임의 격자나 basis에 이를 적용할 수 없다.
  Thermal Hall은 앞선 에너지·엔트로피의 per-spin 정규화와 다른 물리량이다.

현재 `Topology.compute_thermal_Hall()`은 kappa/T라고 설명하면서 단위를
microW/(m K)라고 쓰고, 통합 thermodynamics 함수는 온도 인자 없는 값을
'Thermal Hall Conductance'로 반환한다. 두 함수의 숫자에는 추가로 10^6 배율도
차이가 난다. kappa/T를 반환하려는 의도라면 T를 안 곱하는 것 자체는 오류가
아니지만, 같은 출력에 kappa의 단위·이름을 붙일 수는 없다.

### 독립 모델과 재현 수치

실행 가능한 검증은 `examples/thermal_hall_reference_check.py`에 두었다.
Number-conserving, 양정치인 bosonic two-band 모델을 Nambu로 확장해 Colpa
대각화·현재 curvature를 실행한다. 이것은 물질 모사나 full-spin ED가 아니다.

A(q)=0.8 I+0.1 d(q)·sigma meV,
d=(sin qx, sin qy, 1+cos qx+cos qy), q=length*k를 사용한다.
해석적인 upper/lower curvature는 각각
-minus/plus d·(partial_kx d cross partial_ky d)/(2 abs(d)^3)다.
Thermal weight는 현재 c2 구현을 재사용하지 않고
integral_x^infinity z² exp(-z)/(1-exp(-z))² dz로 독립 계산한다.
SI k_B와 hbar를 사용하여 층당 kappa를 계산했다.

| Grid, T=2 K | Chern upper/lower | 독립 층당 kappa [W/K] |
|---|---|---:|
| 12×12 | ±0.999006017221 | 1.354504504401e-13 |
| 24×24 | ±0.999999759275 | 1.355106043074e-13 |
| 48×48 | ±0.999999999999986 | 1.355106147809e-13 |

- Curvature의 현재 코드와 해석식 차이는 length=1에서 3.67e-15 이하,
  length=3.7을 포함하면 2.94e-14 이하다.
- 24×24에서 현재 Topology 반환은 782.370838760, 통합 반환은
  0.000782370838760이다. 이 숫자는 혼재된 기존 단위의 raw output이다.
- length를 1→3.7로 바꾸면 독립 층당 값과 Chern 수는 유지되지만, 현재 두
  반환값은 모두 13.69배가 된다. 이는 area를 고정한 정규화 결함을 직접 드러낸다.
- dy 성분의 부호를 뒤집으면 Chern과 Hall의 부호가 함께 반전된다.
- c2 kernel 자체는 x=0.2,1,4,15에서 독립 quadrature와 비교했고, 최대 차이는
  3.05e-14였다. 이 결과는 작은 occupation 전체 범위의 정밀도를 인증하지 않는다.

```sh
PYTHONPATH=code-space python examples/thermal_hall_reference_check.py
```

### 제안 당시 반환 기준과 구현 범위

2026-09-11 사용자에게 다음 반환 기준을 제안했다. 이어서 사용자가
"기본값을 층당 kappa_xy [W/K]로 통일하고, 층간격을 입력하면 3D 값으로
환산하는 기준"을 명시적으로 승인했다. 아래 권장안은 그 승인에 따라 적용했다.

권장안은 두 경로 모두 층당 kappa_xy [W/K]를 기본으로 반환하고, 선택적인
`layer_spacing_m`이 주어졌을 때 3D kappa_xy [W/(m K)]로 환산하는 것이다.
Topology의 세 번째 반환값과 통합/스캔 결과의 'Thermal Hall Conductance' key를
일관되게 맞춘다. kappa/T는 T>0에서 명시적으로 나누며 micro 접두사는 표시
단계에서 적용한다. 기본을 kappa/T로 원하는 경우에는 W/K² 계약으로 별도 확정한다.

단위 선택 이후에는 실제 MBZ 적분 가중치 또는 magnetic cell 면적을 동일한
규칙으로 사용하고, geometry가 없는 detached Thermodynamics에서 Hall 값을
임의로 만들지 않는 경계도 명시해야 한다. 불완전한 k sample, invalid mode와
legacy Hex_BZ coverage는 별도 판정이 필요하다. 이 항목들을 단순한 Ns 곱셈으로
숨기지 않는다.

## 2026-09-11 2단계 — SI 반환 계약 구현 및 검증

### 적용한 반환 계약

`Topology.compute_thermal_Hall()`의 세 번째 반환값, 통합 함수
`Thermodynamics.compute_thermodynamic_quantities_at_T()`의
`'Thermal Hall Conductance'`, `get_thermodynamic_quantities()`의 같은 key에
동일한 계약을 적용했다. 기존 tuple 및 dictionary 형태는 보존했다.

| 호출 기준 | 반환 물리량 | 단위 |
|---|---|---|
| 기본 `layer_spacing_m=None` | 층당 kappa_xy | W/K |
| `layer_spacing_m=d_m > 0` | kappa_xy_2D / d_m | W/(m K) |

세 함수에 keyword-only `layer_spacing_m`을 추가했고, 스캔 함수도 각 온도에
전달한다. 거리 입력은 metre이며, 동등한 독립 층의 간격을 뜻한다. 물질별
층간격을 임의로 설정하지 않는다. kappa/T가 필요하면 T>0에서 사용자가
반환값을 T로 나누고, micro 단위가 필요하면 표시 단계에서 10^6을 곱한다.
기존 결과를 그대로 이어 붙이면 단위가 섞이므로 과거 Hall 데이터는 재계산한다.

예를 들어 이미 대각화한 `solver`와 full-MBZ `k_data`에 대해:

```python
from lswt.observables.topology import Topology
from lswt.observables.thermodynamics import Thermodynamics

berry, chern, kappa_layer = Topology(solver).compute_thermal_Hall(k_data, 2.0)
# kappa_layer: W/K

# 7e-10 m is an arbitrary example, not an NBCP material parameter.
bulk = Thermodynamics(solver).compute_thermodynamic_quantities_at_T(
    k_data, 2.0, layer_spacing_m=7e-10
)["Thermal Hall Conductance"]
# bulk: W/(m K)
```

### 실행 코드와 영향 범위

- `topology.py`에 magnetic cell 면적 판독, 온도·층간격 검증, SI Hall 적분을
  위한 내부 함수를 두었다. 기존 `thermodynamics.py` → `topology.py` 의존
  방향 안에서 공용으로 사용하며 solver/core에는 새 의존을 넣지 않았다.
- 면적은 `solver.system.lattice_vectors`의 abs(det)이다. Legacy 부모 객체는
  `lattice_bz_settings[0]`에서 원래 magnetic cell을 읽는다. `bz_data['area']`는
  reciprocal grid area element이므로 이를 real-space 면적으로 오인하지 않는다.
- equal-weight full-MBZ 합을 `N_k*A_mag`로 나누고 `k_B²*T/hbar`와
  meV→J의 `1.602176634e-22`를 곱한다. Ns를 별도로 나누거나 곱하지 않는다.
- `layer_spacing_m`이 없으면 2D 값, 있으면 그 값으로 나눈 3D 값을 반환한다.
- 음수·비유한 온도, 0 이하·비유한 층간격, 퇴화하거나 잘못된 격자 벡터는
  ValueError다. 격자 정보가 아예 없는 detached Thermodynamics의 Hall은 NaN이며
  U/S/C 및 점유수는 기존 계산을 유지한다.
- Colpa 실패로 표시된 sample 또는 미분행렬이 빠진 sample이 하나라도 있으면
  Hall은 NaN이다. 유효 표본만 남겨 평균을 재정규화하지 않는다.
  `invalid_exclude=False`도 Hall 적분의 실패 판정을 해제하지 않는다.
- Topology의 비어 있거나 실패한 전체 적분은 Chern도 NaN으로 표시한다.
  유효 데이터의 Berry curvature와 Chern 계산식·부호·기존 `bz_type` 분기는
  변경하지 않았다. `bz_type`은 기존 Chern 정규화에만 관여한다.

### 독립 결과 및 legacy 수치 대비

T=2 K, 24×24 equal-weight MBZ에서 두 경로는 모두
**1.355106043093e-13 W/K**를 반환한다. 독립 SI 상수·해석 curvature·quadrature
결과는 **1.355106043074e-13 W/K**다. 실행 예제 전체의 최대 상대차는
1.45e-11 이하로, SI/meV 상수의 표기 정밀도 차이 수준이다.

| 검증 | 결과 |
|---|---|
| 12×12 / 24×24 / 48×48 층당 kappa | 1.354504504421e-13 / 1.355106043093e-13 / 1.355106147829e-13 W/K |
| 24→48 grid의 상대 변화 | 7.73e-8 |
| 길이 1→3.7 | Hall 상대 변화 2.89e-15; 기존 13.69배 오류 해소 |
| dy 성분 부호 반전 | curvature, Chern, Hall 부호 반전 |
| d=7e-10 m의 3D 환산 (24×24) | 1.935865775847e-4 W/(m K) |
| legacy curvature / Chern 대비 (동일 데이터) | 차이 0 |

원본 `legacy/modules/LinearSpinWaveTheory/lswt_topology.py`와
`lswt_thermodynamics.py`를 직접 실행해 동일한 k_data를 비교했다.
length=1일 때 과거 Topology raw 값은 782.3708387603748,
통합 raw 값은 0.0007823708387603747이다. 새 값과의 비율은 각각

- Topology: `T * (sqrt(3)/2)/A_mag * 1e-16`
- 통합: `T * (sqrt(3)/2)/A_mag * 1e-10`

과 일치한다. 이는 고정 삼각격자 면적, 빠져 있던 T, meV→J와 micro 배율을
분리한 변화다. length=3.7에서도 같은 비율식을 확인했다. Legacy 원본은 보존했다.

### 회귀 검증과 실행 방법

`code-space/tests/test_solvers/test_thermal_hall_normalization.py` **32개 통과**:

- T=0/0.8/2/5 K에서 두 공개 경로와 독립 SI 적분 비교.
- 두 층간격 및 온도 스캔의 2D→3D 환산과 다른 관측량 불변.
- 균일 길이 rescale, skew cell, orientation 반전, metre 크기 좌표 표현.
- curvature가 0인 세 번째 band를 추가해도 Hall이 Ns로 희석되지 않음.
- geometry 없는 부모·detached 호출, 빈 데이터, 실패 표시 sample,
  미분행렬 누락, 잘못된 온도·층간격·격자 입력 처리.

전체 `code-space/tests`: **123 passed, 5 failed**, legacy SyntaxWarning 2건.
실패 5건은 기존 `CommensurateStructure`의 angle shape 불일치이며 추가 실패는
없다. 이번 검증을 전체 테스트 통과로 보고하지 않는다.

```sh
PYTHONPATH=code-space python -m pytest code-space/tests/test_solvers/test_thermal_hall_normalization.py -q
PYTHONPATH=code-space python examples/thermal_hall_reference_check.py
PYTHONPATH=code-space python -m pytest code-space/tests -q
```

## 2026-09-11 3단계 — BZ 표본 범위와 Chern 정규화

사용자가 다음 검증 진행을 승인해 실제 `BrillouinZone`과 `LSWTSolver` 경로를
대조했다. 2단계는 외부에서 만든 full-MBZ benchmark의 Hall 정규화까지 검증했고,
이번 단계는 라이브러리가 실제로 그 표본 조건을 충족하는지 확인한다.

### 원인과 수정 전 재현

1. `_get_bz_setting(..., bz_type="Hex_60")`은 전달된 magnetic lattice를
   길이 1의 고정 triangular lattice로 바꿨다. 네 부격자 예제의
   `a1=(1,sqrt(3)), a2=(1,-sqrt(3))`를 명시적으로 넘겨도 polygon 면적이
   실제 MBZ의 **4배**가 됐다. BZ 타입을 tuple로만 넘긴 경우에는 이 교체가
   일어나지 않아 같은 옵션을 표현하는 방식에 따라 결과가 달랐다.
2. `Hex_30`, `Tetra`, `wigner_seitz`는 tuple 설정으로 지원하지만 현행 override
   분기에는 없어, solver가 이 옵션을 전달하면 ValueError였다. Legacy 분기의
   `ValueError(...)`는 raise하지 않았기 때문에 이 거부는 포팅 후 동작 차이다.
3. Hex helper가 반환하는 reciprocal_vectors의 크기는 해당 polygon을 만드는
   reciprocal lattice의 절반이었다. N=24, 직접 magnetic cell 설정에서는
   `max|a_i.G_j/(2*pi)-delta_ij|=0.5`였다. 실공간 a도 2배 잘못 교체되면 두 오류가
   상쇄되어 이 reciprocity 검사만 통과할 수 있으므로 polygon 면적을 함께 봐야 한다.
4. Hex grid의 경계 포함은 magnetic translation으로 같은 sample을 중복 계산했다.
   N=24에서 `N_k*delta_k_area/A_MBZ=1.01408179`였다. Tetra grid는 polygon 필터가
   한 경계 행을 제거해 같은 비율이 **47/48=0.97916667**이었다.
5. Wigner-Seitz option은 polygon만 Wigner-Seitz로 반환하고 실제 grid는
   parallelogram에 남겼다. 이 grid 자체는 full reciprocal cell 적분에는 유효하지만
   'polygon 안의 표본'이라는 반환 설명과 달랐다. 큰 skew의 unreduced basis에서는
   기존 최근접 벡터 가정으로 polygon도 잘못될 수 있었다.
6. Topology는 같은 k_data에서도 `bz_type="Hex_60"` 등에 따라 Chern을 Ns로
   나눴다. BZ 반복 횟수는 lattice/sample 영역으로 결정되며 band 수와 같지 않다.
   2-band benchmark를 한 번의 full MBZ에 표본화하면 이 분기로 Chern이 절반이 됐다.

### 적용한 수정과 의존 범위

- **`core/brillouin_zone.py`:** override는 표현 방식만 바꾸고 입력 lattice를
  보존한다. `simple`은 reciprocal parallelogram, `wigner_seitz` 및 호환 이름
  `Hex_60`/`Hex_30`은 실제 lattice의 Wigner-Seitz cell을 사용한다. Hex 이름으로
  고정 길이·방향·육각 대칭을 강제하지 않는다. `Tetra`/`tetra`는 실제 orthogonal
  lattice의 rectangle을 쓰며, 비직교 입력은 ValueError다.
- 공개 `BrillouinZone` 경로는 `_AnyBZ`의 공통 grid를 사용한다. reciprocal basis는
  항상 실제 a_i의 dual이며, grid는 **(2N)^2개 equal-weight periodic sample**이다.
  경계에 같은 reciprocal representative를 추가하거나 행을 삭제하지 않는다.
  기존 private Hex/Tetra helper는 공개 wrapper의 실행 경로에서 제외했다.
- Wigner-Seitz에서는 reciprocal basis를 Gauss reduction한 뒤 가장 가까운
  reciprocal translation으로 각 sample을 접는다. 격자 자체와 Fourier 주기성은
  보존되며 polygon 밖의 점이 반환되지 않는다. Polygon도 reduced basis에서 만든다.
  `get_nearest_lattices()` 자체는 바꾸지 않아 correlations 모듈의 호출은 보존했다.
- **`observables/topology.py`:** 노트 식 (145)의
  `C_n=integral(Omega_n d^2k)/(2*pi)`를
  `C_n=2*pi*mean_k(Omega_n)/A_mag`로 계산한다. Hall과 같은 full-MBZ 측도를 쓰며
  `bz_type`에 따른 Ns 나눗셈을 제거했다. 결과를 정수로 반올림하지 않는다.
  Berry Kubo 식 및 Chern의 기존 +integral 부호는 유지한다.
- **`solvers/solver.py`:** diagnosis가 만든 full grid의 key 집합을 내부 속성
  `_integration_k_keys`에 기록한다. 반환 tuple/dictionary의 구조는 바꾸지 않았다.
- **`observables/topology.py`, `observables/thermodynamics.py`:** 부모 solver에
  기록된 key와 다른 부분/대체 grid는 Chern·Hall을 NaN으로 표시한다. 개별 k의
  curvature 및 기존의 다른 thermodynamic quantities는 계속 조회할 수 있다.
  온도 스캔에도 같은 판정을 적용한다. 사용자 정의 부모 객체에는 이 기록이
  없을 수 있으므로 full-BZ 조건을 여전히 호출자가 보장해야 한다.

입력 model이 physical band 2개인 cell을 정의했다면 Chern 적분에도 그 cell의
BZ를 한 번 센다. 과거 primitive BZ 반복의 특수한 Ns 보정은 일반적인 default가
될 수 없다. 현재 generator는 한 magnetic reciprocal cell을 표본화하므로 별도의
반복 횟수 인자를 요구하지 않는다.

### 수렴·면적·legacy 검증

실행 가능한 `examples/bz_chern_reference_check.py`는 5가지 BZ 설정과 N=6,12,24를
비교한다. 앞 단계의 positive two-band boson 모델을 각 generator의 실제 k에
평가하며, Nbcp의 1–4 MSL을 물리적으로 재현한 실험은 아니다. Cell orientation이
음수이면 같은 d(q) 모델의 Cartesian chirality도 반전되므로 아래 Chern/Hall은
음수다. 이는 출력 부호 규칙을 바꾼 결과가 아니다.

| Hex_60, T=2 K | k 표본 수 | Chern upper/lower | 층당 kappa [W/K] |
|---|---:|---|---:|
| N=6 | 144 | -1.000994471683 / +1.000994471683 | -1.355708003879e-13 |
| N=12 | 576 | -1.000000240725 / +1.000000240725 | -1.355106252564e-13 |
| N=24 | 2304 | -1.000000000000020 / +1.000000000000020 | -1.355106147829e-13 |

- 5가지 설정 모두 `N_k*delta_k_area=A_MBZ`를 부동소수점 오차 수준에서 만족한다.
  Fourier character `mean(exp(i*k.a2))`의 최대 절댓값은 2.27e-16이다.
- Reciprocal duality, polygon area, 경계 translation 중복 없음과 polygon 내부
  포함을 검사했다. 큰 skew, 회전한 rectangle 및 metre 크기 좌표도 포함한다.
- 모든 표본에서 production Hall과 독립 curvature·c2 quadrature·SI 상수로 계산한
  값의 상대차는 1.45e-11 이하다. N=24에서 Chern의 ±1 오차는 2.0e-14 이하다.
- 원본 legacy BZ를 실행해 수정 전 Hex_60 override grid와 완전히 같음을 확인했다.
  N=12에서 과거 표본은 1333개, 새 full-MBZ grid는 576개다. 따라서 같은 N끼리
  '동일한 point 수의 알고리즘 비교'라고 해석하지 않는다.
- **동일한 새 grid**를 legacy Topology와 현재 Topology에 공급하면 curvature
  차이는 정확히 0이다. Legacy Chern은 `[-0.500000120363,+0.500000120363]`,
  새 값은 `[-1.000000240725,+1.000000240725]`로 불필요한 Ns=2 나눗셈이 제거됐다.
  이전 단계의 SI Hall 계약은 그대로 유지된다.

`code-space/tests/test_solvers/test_brillouin_zone_integration.py` **신규 40개 통과**.
이전 Hall 정규화 32개와 합쳐 **72개 통과**, 전체 **163 passed, 5 failed**이며
기존 CommensurateStructure 실패 5개와 legacy SyntaxWarning 2개 외에 추가 실패는 없다.

```sh
PYTHONPATH=code-space python examples/bz_chern_reference_check.py
PYTHONPATH=code-space python -m pytest code-space/tests/test_solvers/test_brillouin_zone_integration.py code-space/tests/test_solvers/test_thermal_hall_normalization.py -q
PYTHONPATH=code-space python -m pytest code-space/tests -q
```

### 적용 시 주의할 변경

같은 `N`에서 Hex/Tetra의 표본 수와 좌표가 바뀌므로 **그 표본으로 계산한**
phase scan·그림·적분 데이터는 재계산해야 한다. 모든 최적화·그림이 대상인 것은
아니다. 특히 `EnergyFunction`은 이미 `simple` grid를 사용한다. 아래 후속 조사에서
실제 호출 경로와 기존 산출물을 구분했다. Hex 이름의 orientation은 이제 실제 lattice가 결정한다.
단위 격자를 무시한 primitive BZ나 중복 경계와 맞추기 위한 수치 보정은 유지하지
않았다. 일반 BZ grid/corners와 band-path high-symmetry 좌표를 사용하는 호출자는
동일한 lattice에서 생성한 새 `bz_data`를 함께 사용해야 한다.

## 2026-09-11 후속 조사 — 재계산 대상과 유한에너지 밴드 퇴화

사용자 요청은 재계산이 필요한 결과를 확인하고 밴드 퇴화 문제를 설명하는 것이다.
이 단계에서는 라이브러리의 계산식·API를 추가 변경하지 않았다. 기존 산출물 조사,
고정 상태 비교 및 퇴화의 최소 재현을 수행했다. 앞 단계의 모든 phase scan·그림을
재계산해야 한다는 서술은 호출 경로를 구분하지 않아 과도했으므로 위에서 정정했다.

### 재계산 판정 — 이번 BZ 수정과 앞선 수정을 구분

| 사용한 경로 또는 결과 | 판정 | 이유와 범위 |
|---|---|---|
| 수정 전 `Topology`의 Chern·thermal Hall, 통합·온도 스캔의 Hall | 재계산 | Chern 측도 및 SI/T/면적 계약이 변경됐다. 기존 외부 보정도 확인한다. 기본은 층당 W/K, 입력한 `layer_spacing_m`으로 나눈 값만 W/(m K)다. |
| 수정 전 Hex/Tetra 등의 solver grid에 의존하는 k_data·BZ 적분 | 새 grid에서 재계산 | 잘못된 lattice override, 경계 중복·누락과 표본 위치가 바뀌었다. 점유수·양자 에너지·BZ 평균 상관함수 등 실제로 이 grid를 쓴 결과가 대상이다. |
| 수정 전 `LSWTSolver.solve().ground_state_energy` | 재계산 | 별도 종결 이슈에서 공개 에너지의 trace subtraction·스핀당 정규화를 수정했다. |
| 수정 전 유한온도 U/F·공개 solver 점유수, F를 사용한 최적화 | 재계산 | 온도 전달, thermal band sum 및 자유에너지 저장 결함의 영향을 받는다. `EnergyFunction`이라도 T>0의 자유에너지 최적화는 예외가 아니다. |
| 개별 S/C 및 통합 sublattice 점유수의 과거 반환값 | 반환 의미 확인 후 재계산 | Ns 정규화가 변경됐다. 외부에서 이미 Ns로 보정했다면 중복 보정을 제거한다. |
| `EnergyFunction`의 classical/T=0 quantum energy 최적화 | 이번 BZ·공개 영점에너지 수정만으로 재실행 불필요 | classical 항은 k 적분이 없고 quantum 항은 기존에도 올바른 저수준 trace 식과 `simple` grid를 사용했다. 최적화 수렴·다른 Hamiltonian 버전 문제는 별도다. |
| 동일한 물리 k에서의 spectrum·국소 curvature | 조건부 재사용 | 동일한 Hamiltonian, 미분, regularization일 때만 가능하다. 표본에서 결정한 MAGSWT uniform shift가 달라지면 같은 k 결과도 다시 계산한다. 밴드 퇴화의 개별 curvature는 아래 제한을 따른다. |
| 보존된 legacy B/B† 배치로 생성한 결과·논문 그림 | 코드 버전·원자료 먼저 확인 | 현재 배치는 이미 2026-06-02 수정됐다. 어느 결과가 이전 배치를 사용했는지 확인 없이 논문 전체의 오류를 선언하지 않는다. |

`examples/nbcp_hamiltonian_check.py`의 BZ 표본을 사용하는 matrix 검사는 새 grid에서
다시 수행해야 하지만, 그 스크립트 앞부분의 무작위 MAGSWT 최적화까지 이번 BZ
수정으로 무효가 됐다는 뜻은 아니다. 재현 비교에는 동일한 고정 상태를 먼저 쓴다.

### 현재 checkout에서 확인한 산출물

조사 시작 시 `data-space/`에는 `.gitkeep`만 있었다. 현재 checkout의 코드·예제·
legacy 실행 자료에서 과학 계산용 `.npz/.npy/.csv/.tsv/.json/.pkl/.pickle/.h5/.hdf5/
.ipynb/.dat` 저장 결과를 찾지 못했다. 의존성 환경, Git 내부, 다른 agent worktree 및
보존된 원문·참고문헌은 이 결과 파일 조사에서 제외했다. 외부 notebook·다른 컴퓨터의
자료까지 조사한 것으로 해석하지 않는다. 아래 JSON은 **이번 조사에서 새로 저장한**
진단 결과다.

- `examples/nbcp_one_msl.png`, `examples/nbcp_three_msl.png`: 실공간 스핀 구성 그림이다.
  Hall·Chern·BZ 적분 그림이 아니므로 이번 BZ 수정만을 이유로 다시 그리지 않는다.
- `examples/nbcp_spin_config.png`: 제목은 “NBCP Four MSL Ground State”지만 이미지에
  parameter, 최적화 seed, 각도·에너지 원자료와 수렴 기록이 없다. 생성 스크립트도
  특정하지 못했다. **역사적 그림; ground-state 주장 재사용 전 provenance 확인 필요**다.
- 세 이미지 모두 `HEAD:doc-space/examples/`의 같은 파일과 byte 단위로 동일했다.
  기존 파일은 덮어쓰지 않았다. 이 동일성은 그림의 물리적 정확성을 인증하지 않는다.
- 이론 도식과 원문 검토용 PDF render는 수치 결과 재계산 대상에 넣지 않았다.

### `EnergyFunction` grid 비교

현재 NBCP_CONFIG와 각 phase builder를 사용하되, angles는
`np.linspace(0.4, 1.7, num_angles)` rad로 고정했다. N=4, 64개 k에서 수정 전
`BrillouinZone(..., bz_type="simple")` snapshot과 현재 grid를 비교했다. 동일한 현재
Hamiltonian·에너지 함수에 두 grid를 각각 넣은 비교이며, 과거 Hamiltonian 전체와의
비교나 최적화된 물질 상태가 아니다.

| 고정 상태 | 최대 k 좌표 차이 | 새 grid의 quantum energy [meV/spin] | 이전 grid와 에너지 차이 |
|---|---:|---:|---:|
| One MSL | 0 | -0.00377015361056647 | 0 |
| Two MSL | 0 | -0.00522555665171534 | 0 |
| Three MSL | 0 | -0.00837830412551244 | 0 |
| Four MSL | 0 | -0.00642296963838957 | 0 |

입력과 source hash는
`data-space/verification/260911-recalculation-audit/optimization-grid-comparison.json`,
그림 동일성은 같은 폴더의 `existing-image-provenance.json`에 저장했다. 비교에 사용한
이전 BZ snapshot의 임시 경로와 hash도 기록했다. 임시 snapshot의 영구 보존을
보장하는 기록은 아니며, 현재 함수가 `simple`을 지정한다는 호출 경로도 함께 근거로 둔다.

### 유한에너지 퇴화 재현과 정확한 결함

밴드 퇴화는 동일한 k에서 서로 다른 mode의 에너지가 같아지는 것이다. 0이 아닌
에너지에서도 발생하며, 그 자체가 불안정성이나 잘못된 물리 모델을 뜻하지 않는다.
퇴화 공간 안에서는 eigenvector를 서로 섞어도 같은 energy eigenstate이므로,
추가 symmetry label 등으로 분리하지 않은 개별 밴드의 Abelian curvature/Chern을
일반적으로 유일하게 지정할 수 없다. 비퇴화 가정 및 나머지 밴드와 분리된 band
group의 확장은 [Fukui–Hatsugai–Suzuki, 식 (2), (16)](https://arxiv.org/html/cond-mat/0503172v2)
에서 확인했다. Bosonic pairing 모델에 적용할 때는 paraunitary metric을 별도로
반영해야 하며, 이 인용만으로 해당 구현이 검증된 것으로 보지 않는다.

현재 `compute_berry_curvature()`는 signed dynamical eigenvalue 차의 제곱
`(J_eval[n]-J_eval[m])**2`가 정확히 0이면 해당 항을 건너뛴다. 그 뒤에 spacing을
갱신하므로, 퇴화 항이 curvature 합뿐 아니라 작은 간격의 진단에서도 사라진다.

최소 재현 모델은 dimensionless k에 대해
`A(k)=0.8 I+0.1 kx sigma_x+0.1 ky sigma_y+delta sigma_z` meV,
`H(k)=diag(A(k), A(-k)^T)`다. k=0 근방의 local model이며 full BZ나 NBCP fit이
아니다. Positive definite인 상태를 `regularization="No"`로 대각화했다.

| delta [meV] | 실제 두 양의 밴드 간격 [meV] | 반환한 upper/lower curvature | 반환한 spacing [meV] |
|---|---:|---|---|
| 0.01 | 0.02 | -50 / +50 | 0.02 / 0.02 |
| 0.0001 | 0.0002 | -500000 / +500000 | 0.0002 / 0.0002 |
| 0 | 0, 두 mode 모두 0.8 meV | 0 / 0 | 0.8 / 1.6 |

delta>0에서는 독립 두-level 식 `Omega_upper/lower=(-0.005/delta^2,+0.005/delta^2)`와 상대오차
1.4e-12 이내로 일치했다. 가까운 밴드의 큰 curvature 자체는 물리적으로 가능하다.
**정확한 퇴화에서 반환한 0을 물리적인 0으로 판정할 수 없고, spacing 역시 실제
퇴화를 놓쳤다**는 것이 이 재현의 결론이다. 이 local 모델의 BZ Chern 또는 total
thermal Hall은 계산하지 않았다. 퇴화가 있으면 total Hall이 반드시 정의되지 않는다는
결론도 내리지 않는다. 나머지 밴드와 분리된 group의 projector/합산 응답은 별도로
검토할 수 있지만, 현재 skip으로 얻은 개별 값을 단순히 더해 정당화할 수는 없다.

재현 스크립트: `examples/band_degeneracy_check.py`. 당시 runtime hash와 숫자는
`data-space/verification/260911-recalculation-audit/band-degeneracy.json`에 저장했다.

```sh
PYTHONPATH=code-space python examples/band_degeneracy_check.py
```

이것은 **결함 재현**이며 해결 회귀 테스트가 아니다. 라이브러리 변경이 없어 전체
pytest는 다시 실행하지 않았고, 위 163 passed/5 failed는 직전 BZ 단계의 결과다.

### 다음 구현의 경계

1. 먼저 실제 band separation으로 정확한 퇴화와 수치적으로 가까운 band를 검출하고,
   개별 curvature/Chern의 사용 가능 여부를 명시해야 한다. Near-degeneracy에는
   에너지 scale에 따른 tolerance 및 k-mesh 수렴 검토가 필요하다.
2. 외부 band와 분리된 group이라면 projector/non-Abelian 처리를 검토한다. Chern
   group의 정의와 thermal weight를 적용한 Hall 합은 각각 검증해야 한다.
3. **Goldstone은 E→0의 별도 문제**다. 이번 0.8 meV 재현은 Goldstone도,
   pseudo-Goldstone gap의 계산도 아니다. 임의 onsite shift가 물리적 fluctuation gap을
   구한 것으로 해석하지 않는다. 그 정의·대상 모델은
   [별도 이슈](260810-pseudo-goldstone-gap.md)에서 계속 Unknown으로 둔다.

## 2026-09-11 4단계 — 고립된 밴드 가정과 수치적 분리 기준

> 아래 `64*eps64` 방식은 당시 구현 기록이다. 2026-09-12 5단계에서 고정 cutoff로
> 교체했다. 64는 솔버의 오차 분석에서 유도하거나 보장한 계수가 아니었다.

사용자는 밴드별 Chern 수를 다른 밴드와 갭으로 분리된 경우에 정의하는 것으로
가정하고 진행하며, 작은 갭의 계산을 위해 FHS 또는 명시적인 퇴화 기준을 제안했다.
이번에는 **명시적 수치 판정 기준을 적용하는 방식**을 구현했다. 일반적인 퇴화
band group의 불변량이나 pseudo-Goldstone gap을 구현 범위에 넣지 않았다.

### 적용 가정과 FHS의 역할

개별 밴드는 전체 BZ에서 `min_{k,m!=n}|E_n(k)-E_m(k)|>0`인 직접 갭으로
분리되어 있다고 가정한다. 서로 다른 k의 에너지 범위가 겹치는 간접 갭 문제와는
구별한다. Bosonic Kubo 계산에서는 양의 physical band 외에 음의 signed BdG
partner와의 separation도 확인한다. 유한한 mesh에서 통과했다는 사실만으로
sample 사이에 band touching이 없음을 증명하지 않는다.

[Fukui–Hatsugai–Suzuki, 식 (7)–(14)](https://arxiv.org/html/cond-mat/0503172v2)는
고유벡터 overlap의 link variable로 gauge-invariant lattice Chern 수를 계산한다.
직접 `(En-Em)^-2`를 사용하지 않는 대안이지만, **퇴화 검출기이거나 작은 갭의
수렴을 자동으로 보장하는 절차는 아니다**. 유한 mesh에서 정수가 반환되어도
연속 BZ의 올바른 Chern 수를 얻었다는 충분조건이 아니며, nonzero overlap,
plaquette phase/admissibility 및 mesh 수렴을 확인해야 한다. Bosonic 적용 시에는
paraunitary metric 및 BZ 경계에서의 basis 연결도 일치시켜야 한다. 이번 단계에서
FHS 알고리즘이나 공개 method 선택 API는 추가하지 않았다.

### 적용한 기준과 반환 의미

`lambda=diag(J)*eval`을 signed BdG energy라고 할 때, physical band n과 다른
모든 mode m을 다음의 **대칭적인 쌍별 기준**으로 비교한다.

```text
abs(lambda_n - lambda_m) <= 64 * eps64 * max(abs(lambda_n), abs(lambda_m))
eps64 = 2.220446049250313e-16
```

- 이 조건은 float64 반올림 수준에서 **수치적으로 분리되지 않음**을 뜻한다.
  유한한 값에 대해 exact physical degeneracy가 증명됐다는 뜻이 아니다.
  0.8 meV 근방에서는 약 `1.14e-14 meV`가 기준이다.
- 절대 에너지 cutoff는 두지 않는다. 에너지 단위나 Hamiltonian 전체를 rescale해도
  같은 판정을 유지한다. 저정밀 입력이나 ill-conditioned bosonic eigenproblem의
  오차를 인증하는 기준은 아니며, eigensolver residual/conditioning은 별도다.
- 분리되지 않은 band의 curvature는 `NaN`이다. 같은 band에 그런 sample이 하나라도
  있으면 Chern도 `NaN`이다. 나머지 분리된 band의 Chern은 계속 계산한다.
- 현재 band-sum Hall 경로는 하나의 band라도 unavailable이면 `NaN`을 반환한다.
  통합 열역학·온도 스캔·3D 환산에도 동일하게 적용하며 T=0에서도 계산 경로의
  사용 가능성을 같은 기준으로 표시한다. 물리적 total Hall이 반드시 정의되지
  않는다는 주장이 아니며, 별도의 group 응답 또는 알려진 T=0 극한을 계산한 것도 아니다.
- `level_spacing`은 배열에서 인접한 index만 보지 않고 **모든 다른 signed mode와의
  최소 차이**를 반환한다. 퇴화의 0을 빠뜨리지 않으며 임의로 energy-to-zero를
  섞지 않는다. 예를 들어 단일 0.8 meV band의 particle-hole separation은 1.6 meV다.
- 분리 가능한 작은 유한 gap의 곡률은 clip하지 않는다. Kubo 분자의 두 matrix
  element를 각각 energy difference로 나눈 뒤 곱해, 전체 energy scale이 작거나
  클 때 분모를 제곱하면서 생기는 underflow/overflow를 피한다.
- 기존 `DEFAULT_LEVEL_SPACING=0.01 meV`는 `verbose=True`에서 mesh 수렴을 확인하라는
  **안내에만** 사용한다. 이 수치로 퇴화 또는 유한값의 유효성을 판정하지 않는다.
  `verbose`는 unavailable band도 경고한다. 함수 signature·tuple·dictionary key는 유지한다.

실행 변경은 `observables/topology.py`, 이를 호출하는
`observables/thermodynamics.py`의 T=0 포함 공통 처리에 한정한다. `config.py`는
기존 안내값의 주석만 명확히 했다. Hamiltonian·대각화·BZ generator·legacy 원본은
바꾸지 않았다. `examples/band_degeneracy_check.py`는 JSON에서 NaN을 null과
availability flag로 기록하도록 수정했으며 과거 진단 JSON은 보존했다.

### 수정 전후 검증

신규 `code-space/tests/test_solvers/test_band_isolation.py`는 수정 전 **19 failed,
6 passed**, 수정 후 **25 passed**다. 정확한·roundoff 수준 퇴화, index가 떨어진
퇴화 band, 양·음 partner, 작은 유한 gap, gauge phase, 전체 energy rescale
`1e-160`–`1e160`, 분리된 나머지 band의 Chern 및 두 Hall 경로를 포함한다.
비퇴화 curvature는 보존된 legacy와 상대오차 `1e-14` 이내로 일치했다.

| local two-band probe | 수정 전 | 수정 후 |
|---|---|---|
| delta=0, 실제 gap=0 | curvature 0/0, spacing 0.8/1.6 meV | curvature NaN/NaN, spacing 0/0 |
| delta=1e-15 meV | exact-zero 분기에 걸리지 않을 수 있음 | 약 2e-15 meV 간격을 그대로 보고하고 curvature는 unavailable |
| delta=1e-6 meV, gap 약 2e-6 meV | 작은 유한 gap | curvature 약 -5e9/+5e9, 독립 해석해 상대오차 5.36e-11 이내 |

기존 SI·BZ/Chern 72개와 합쳐 **97 passed**다. 전체는 **188 passed, 5 failed,
legacy SyntaxWarning 2개**이며 기존 `CommensurateStructure`의 5개 실패 외에는
추가 실패가 없다. `examples/bz_chern_reference_check.py`도 재실행해 5가지 BZ와
N=6,12,24에서 기존 비퇴화 모델의 Chern·Hall 수렴을 유지함을 확인했다.

새 보고서는 `data-space/verification/260911-band-isolation/`에 두며,
`band-degeneracy-after.json`과 `bz-chern-after.json`에 수치를 저장한다.
직전 `260911-recalculation-audit/`의 결함 재현 자료는 수정 전 역사 기록이다.

```sh
PYTHONPATH=code-space python -m pytest code-space/tests/test_solvers/test_band_isolation.py code-space/tests/test_solvers/test_thermal_hall_normalization.py code-space/tests/test_solvers/test_brillouin_zone_integration.py -q
PYTHONPATH=code-space python examples/band_degeneracy_check.py
PYTHONPATH=code-space python examples/bz_chern_reference_check.py
PYTHONPATH=code-space python -m pytest code-space/tests -q
```

## 2026-09-12 5단계 — 비용 확인 및 조절 가능한 고정 cutoff

사용자는 eigenvalue error bound 계산 비용이 부담이 된다면
`abs(E_n-E_m)>eta_n+eta_m` 대신 별도의 numerical stability cutoff를 두자고 제안했다.
값의 지정 방식은 **검증용 기본값을 두고 계산마다 조절 가능하게** 하기로 답했다.
구체적인 `1e-8 meV`는 이번에 사용한 검증용 시작값이며, 모든 물질·모델에서
정확도를 보장하는 값으로 확정한 것은 아니다.

### 비용 측정과 선택

`examples/band_gap_cutoff_check.py`는 production Colpa와 같은 결과를 내는 경로에
선택적인 진단을 추가해 비교한다. 고정 seed의 complex positive-definite 행렬을
사용하며, 이미 계산된 Cholesky factor로 시작한다. 반복 순서를 섞고 5회 묶음의
중앙값을 기록했다. 아래는 k 한 점당 시간이고 실제 NBCP의 전체 실행 시간은 아니다.

| 행렬 크기 | 기본 Colpa [us] | 고정 cutoff 포함 [us] | 잔차·직교성의 Frobenius 진단 포함 [us] | spectral norm 진단 포함 [us] |
|---|---:|---:|---:|---:|
| 4×4 | 15.81 | 22.23 | 27.97 | 55.02 |
| 6×6 | 19.56 | 23.15 | 29.86 | 57.58 |
| 8×8 | 35.19 | 54.30 | 44.67 | 83.33 |
| 64×64 | 763.49 | 790.00 | 817.62 | 1781.98 |

**잔차 진단이 항상 큰 비용이라는 결론은 아니다.** 작은 행렬에서는 Python/NumPy
호출 비용과 측정 변동이 커서 cutoff가 모든 경우에 더 빠른 결과도 아니었다.
전체 반복값과 2×2, 32×32 결과도 JSON에 보존했다. Spectral norm은 이 측정에서
추가 비용이 더 컸다. 이 진단은 matrix 구성·Cholesky·잔차 평가의 반올림 오차를
합산한 certified bound가 아니므로 그 전체 비용을 측정한 것으로 보고하지 않는다.

기본 관측량 경로에는 매 k마다 잔차·matrix norm·eigenvalue error estimate를 추가하지
않고, 사용자가 제안한 cutoff 정책을 적용했다. 엄밀한 error bound 기능은 구현하지
않았다. 선택 이유는 계산 정책을 단순하게 유지하기 위해서이며, 큰 성능 향상이
모든 크기의 모델에서 보장된다는 주장이 아니다.

### 현재 계산 계약

```text
lambda = diag(J) * eval
spacing_n = min_{m != n} abs(lambda_n - lambda_m)
compute band-n curvature only when spacing_n > band_gap_cutoff
default band_gap_cutoff = 1e-8 meV
```

- `band_gap_cutoff`는 한 번의 계산에 지정하는 **절대 에너지 기준**이다. pair별
  `max(abs(E_n),abs(E_m))` 및 임의의 `64*eps64` 계수는 제거했다.
- 대각화된 모든 다른 signed BdG mode와 비교하므로 양의 밴드의 위·아래 이웃과
  음의 partner를 모두 확인한다. 보고하는 `level_spacing`은 실제 간격 그대로다.
- cutoff 이하이면 해당 curvature·Chern을 `NaN`으로 표시한다. 다른 isolated band는
  유지하며, band-sum Hall은 어느 band라도 제외되면 NaN이다. T=0, 통합 열역학,
  온도 스캔 및 층간격을 통한 3D 환산에도 같은 규칙을 적용한다.
- 에너지를 이동시키거나 분모를 cutoff로 대체하지 않는다. 따라서 이 값은 물리적
  gap도, exact degeneracy 판정도, eigenvalue error bound도 아니다.
- 값은 유한한 비음수여야 한다. `0`은 exact coincidence만 제외하는 진단 옵션이다.
  양의 값을 낮춰 계산을 허용해도 정확도를 보장하지 않는다.
- 공개 Hall 경로의 에너지 단위는 meV다. 저수준 함수를 다른 에너지 단위로 평가하는
  실험에서는 cutoff도 같은 비율로 변환해야 한다. 기존 `0.01 meV`는 verbose mesh
  안내 기준으로 유지하며 이 계산 cutoff와 구분한다.

공통 상수는 `config.DEFAULT_BAND_GAP_CUTOFF`다. 다음 경로에 같은 keyword-only
인자 `band_gap_cutoff`를 추가했다. 기존 positional argument와 반환 형식은 유지한다.

- `compute_berry_curvature()`
- `Topology.compute_thermal_Hall()`
- `Thermodynamics.compute_thermodynamic_quantities_at_T()`
- `Thermodynamics.get_thermodynamic_quantities()`

```python
# Energies and band_gap_cutoff are in meV; kappa is W/K per layer by default.
curvature, chern, kappa = topology.compute_thermal_Hall(
    k_data, Temperature=2.0, band_gap_cutoff=1e-8
)
```

실행 의존 범위는 config → topology → thermodynamics와 예제·검증 코드다.
Hamiltonian, 대각화 알고리즘, BZ generator 및 legacy 원본은 수정하지 않았다.

### 시작값의 근거와 검증 범위

기존 0.8 meV 중심 local two-band model에서 cutoff=0으로 계산을 허용한 뒤
독립 해석 곡률과 비교했다. gap 약 `1e-6`, `1e-8`, `1e-9`, `1e-10 meV`에서
상대오차는 각각 `6.09e-10`, `1.01e-8`, `1.65e-7`, `4.28e-6`이었다.
`1e-8 meV`는 이 예제에서 확인한 진단용 시작값이다. 다른 Hamiltonian의 conditioning,
고유벡터 오차, 모델의 허용 오차 또는 BZ mesh 수렴을 검증한 기준은 아니다.
실제 NBCP 적용에서는 cutoff를 바꾼 민감도와 mesh 수렴을 함께 확인해야 한다.

경계에서는 nominal 입력 gap이 아닌 **계산된 gap**으로 비교한다. 예제의 nominal
`1e-8 meV`는 대각화 후 약 `1.000000005e-8 meV`이므로 그 cutoff보다 약간 크다.
별도 회귀 테스트는 정확히 표현 가능한 에너지 차이를 사용해 equality에서
제외되고 바로 작은 cutoff에서는 허용되는지 확인한다.

Band-isolation 기존 25개를 새 정책에 맞게 갱신하고 10개를 추가해 **35 passed**다.
cutoff 조절·경계·입력 검증·두 Hall 경로·온도 스캔·3D 환산과 다른 열역학량 불변을
확인했다. 기존 72개와 합쳐 **107 passed**, 전체 **198 passed, 기존 5 failed**다.
비퇴화 legacy 비교는 통과했으며, 이번에는 고정 cutoff 이하의 결과를 의도적으로
제외하는 동작만 달라진다.

보고서는 `data-space/verification/260912-band-gap-cutoff/cost-and-sensitivity.json`에
저장했다. seed, 실행 환경, 반복별 시간, 4가지 cutoff 민감도와 source hash를 포함한다.
2026-09-11의 자료는 이전 판정 방식의 역사 기록으로 보존한다.

## 결론 / 미결 사항

층당/3D SI 계약, 실제 자기 격자를 보존하는 full-BZ 표본화, Chern의 불필요한 Ns
나눗셈 제거와 부분 solver-grid의 적분 거부까지 구현·독립 검증을 마쳤다.
이 이슈는 아래 물리적 적용 조건과 사용자 검토를 추적하기 위해 in-review로 유지한다.

1. **고립된 밴드의 적용 범위:** 사용자가 승인한 갭 가정 아래 조절 가능한 numerical
   cutoff와 unavailable 전파를 구현했다. 이는 정확도의 보장이 아니다. 실제 NBCP의
   gap 가정·cutoff 민감도·mesh 수렴은 별도 확인해야 한다. Certified error bound,
   FHS·퇴화 band-group 처리, Goldstone 및 regularization의 물리적 타당성은 이번
   검증 대상이 아니다.
2. **사용자 정의 데이터:** provenance가 없는 외부 k_data, 임의 band path,
   nonuniform weights와 직접 제작한 반복 BZ는 여전히 caller의 조건이다.
   내부 key 비교는 matrix 내용·Hamiltonian periodicity·사용자가 임의로 바꾼
   가중치를 인증하지 않는다. 별도의 공개 weighted-grid API는 추가하지 않았다.
3. **물질 및 이론 acceptance:** NBCP의 실측 절대값 재현 및 A6/A7/A16의 사용자
   물리·수학 검토는 별도다. 노트와 Neumann 논문에 나타나는 Chern의 부호 표기
   차이를 이번 정규화 수정으로 임의 통일하지 않았다. PDF·TeX·docs 이론 본문은
   수정하지 않았으며, accepted theory claim으로 승격하지 않는다.

## 참조

- `docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf` — thermal Hall 식의 primary evidence
- `docs/lswt/02-observables/topological-magnon-quantities.md` — 이론식과 convention owner
- `code-space/spintoolkit/observables/topology.py` — 현재 구현
- `code-space/spintoolkit/observables/thermodynamics.py` — 공통 SI 정규화와 온도 스캔
- `code-space/tests/test_solvers/test_thermal_hall_normalization.py` — 독립 회귀 검증
- `examples/thermal_hall_reference_check.py` — 재현 가능한 해석 모델과 결과 출력
- `code-space/spintoolkit/system/brillouin_zone.py` — 실제 magnetic cell의 full-BZ 생성
- `code-space/tests/test_solvers/test_brillouin_zone_integration.py` — BZ/Chern 신규 40개 검사
- `examples/bz_chern_reference_check.py` — BZ 표본 범위 및 Chern/Hall 수렴 보고서
