---
frontmatter-version: 1
title: Notation and Conventions
section: theory/lswt/foundations
status: in-review
last-edited-by: codex
created: 2026-06-04
updated: 2026-08-01
source: /Users/david/Downloads/Linear_Spin_Wave_Theory___Note.pdf
source-section: Summary of Notation and Symbols
---

# Notation and Conventions

이 문서는 LSWT Markdown 전체에서 공유하는 notation의 정본 후보이자 symbol
registry다. 원본 PDF의 `Summary of Notation and Symbols`는 primary evidence지만,
그 표에서 서로 충돌하는 기호를 그대로 정본으로 옮기지는 않는다. 이 문서의
canonical contract와 원본 표기가 다르면 아래 `Legacy Source Mapping`에 차이를
기록한다.

이 문서는 전역 기호의 의미와 사용 범위를 소유한다. 개별 물리식과 유도는 해당
domain 문서가 소유하며, 전역 기호를 다른 의미로 재사용하지 않는다. 아직 검증이
끝나지 않은 기호와 convention은 `Deferred Decisions`에 남기고 정본 정의처럼
사용하지 않는다.

## Typographic Contract

| Object | Form | Examples |
|---|---|---|
| scalar | italic | $S_I$, $E_{\mathrm{cl}}$, $\varepsilon_{n\mathbf{k}}$ |
| vector | bold | $\mathbf{r}_I$, $\mathbf{k}$, $\hat{\mathbf{S}}_I$ |
| Latin-letter matrix | sans serif | $\mathsf{J}_{\ell}$, $\mathsf{R}_I$, $\mathsf{H}_{\mathbf{k}}$ |
| quantum operator | hat | $\hat H$, $\hat a_{i\mu}$, $\hat\Psi_{\mathbf{k}}$ |
| set | calligraphic | $\mathcal{L}$ |
| identity matrix | sans serif | $\mathsf{I}_N$ |

Quantum operator만 hat을 붙인다. Classical spin vector와 classical direction은
$\mathbf{S}_I$, $\mathbf{n}_I$처럼 hat 없이 쓴다. Hamiltonian operator
$\hat H$와 그 quadratic coefficient matrix $\mathsf{H}_{\mathbf{k}}$도 이
규칙으로 구분한다.

$\Sigma_3$처럼 관습적인 Greek 이름을 가진 matrix는 Latin-letter matrix의
$\mathsf{}$ 규칙에 대한 예외로 둔다. 이러한 예외도 처음 등장할 때 matrix
dimension과 역할을 명시해야 한다.

Transpose, complex conjugate, Hermitian conjugate는 각각
$\mathsf{T}$, $*$, $\dagger$를 superscript로 쓴다. 자연지수는 lowercase $e$와
energy-density 기호의 혼동을 피하기 위해 $\exp(\cdots)$로 쓴다. 허수 단위는
$\mathrm{i}$로 쓴다.

## Index Namespace

| Index | Reserved meaning | Domain |
|---|---|---|
| $i,j$ | magnetic unit-cell index | real space |
| $\mu,\nu$ | magnetic sublattice index | one magnetic unit cell |
| $I,J$ | physical spin-site index | $I=(i,\mu)$, $J=(j,\nu)$ |
| $\ell$ | representative physical link | $\ell=(I,J)\in\mathcal{L}$ |
| $\alpha,\beta$ | laboratory-frame Cartesian component | $x,y,z$ |
| $n,m$ | positive-energy magnon-band index | momentum space |

Lowercase $n,m$을 unit-cell index로 사용하지 않는다. 이를 magnon-band index로
예약하면 sublattice $\mu$와 band $n$을 명확히 구분할 수 있다. Local circular
basis는 $+,-,0$ label을 직접 쓰고, 그 위상과 정확한 정의는 local-frame
문서에서 명시한다. 이 label을 $\alpha,\beta$의 Cartesian 의미와 섞지 않는다.

시스템 크기는 다음 기호로만 나타낸다.

| Symbol | Meaning |
|---|---|
| $N_{\mathrm{uc}}$ | magnetic unit cell의 수 |
| $N_{\mathrm{sub}}$ | magnetic unit cell당 magnetic sublattice의 수 |
| $N_{\mathrm{site}}$ | 전체 physical spin site의 수 |
| $N_k$ | discrete momentum point의 수 |

$$
N_{\mathrm{site}}
=N_{\mathrm{uc}}N_{\mathrm{sub}},
\qquad
\dim\mathsf{H}_{\mathbf{k}}=2N_{\mathrm{sub}}.
$$

원본의 $L$, $N$, $m_s$는 서로 다른 구간에서 의미가 바뀌므로 canonical
notation으로 사용하지 않는다.

## Site and Geometry Convention

Magnetic unit cell 안의 site $\mu$가 갖는 basis offset을
$\boldsymbol{\delta}_{\mu}$로 쓴다. Physical site와 그 위치는

$$
I=(i,\mu),
\qquad
\mathbf{r}_{I}
=\mathbf{R}_{i}+\boldsymbol{\delta}_{\mu}
$$

로 정의한다. 여기서 $\mathbf{R}_i$는 magnetic Bravais-lattice vector이고,
$\mathbf{r}_I$는 physical spin position이다. 같은 classical spin angle을
갖는다는 사실만으로 두 site가 같은 magnetic sublattice가 되는 것은 아니다.
Magnetic translation과 local environment를 포함한 system 정의가 같아야 한다.

2D primitive lattice vector는 $\mathbf{a}_1,\mathbf{a}_2$, reciprocal lattice
vector는 $\mathbf{b}_1,\mathbf{b}_2$로 쓰며

$$
\mathbf{a}_r\cdot\mathbf{b}_s=2\pi\delta_{rs}
\qquad (r,s\in\{1,2\})
$$

를 만족한다. Geometry vector는 모두 같은 Cartesian real-space 좌표계로
표현하고, momentum은 그 reciprocal Cartesian 좌표로 표현한다.

현재 코드와의 대응은 다음과 같다.

| Code field | Canonical notation | Contract |
|---|---|---|
| `lattice_vectors` | $\mathbf{a}_1,\mathbf{a}_2$ | Cartesian real-space vectors |
| `Site.position` | $\boldsymbol{\delta}_{\mu}$ | magnetic cell 안의 Cartesian position |
| `Coupling.displacement` | $\boldsymbol{\Delta}_{\ell}$ | source site에서 target site까지의 full Cartesian vector |
| `num_sites`, `Ns` | $N_{\mathrm{sub}}$ | 현재 구현의 legacy naming |

`Coupling.displacement`를 fractional coordinate로 해석하거나 solver가 이를
lattice vector로 자동 변환한다고 가정하지 않는다.

## Link Convention

$\mathcal{L}$은 inter-site physical coupling의 집합이며, 한 physical bond를
한 번 센다. $\ell=(I,J)$의 endpoint 순서는 bond displacement와 exchange
matrix의 방향을 정하지만 reverse link $\bar\ell=(J,I)$를 $\mathcal{L}$에
별도로 넣지는 않는다.

$$
\boldsymbol{\Delta}_{\ell}
=\mathbf{r}_{J}-\mathbf{r}_{I}
=(\mathbf{R}_{j}-\mathbf{R}_{i})
 +(\boldsymbol{\delta}_{\nu}-\boldsymbol{\delta}_{\mu}),
\qquad
\ell=((i,\mu),(j,\nu)).
$$

Laboratory Cartesian basis의 exchange matrix와 그 component는 각각
$\mathsf{J}_{\ell}$, $J_{\ell}^{\alpha\beta}$로 쓴다. Reverse orientation은

$$
\mathsf{J}_{\bar\ell}=\mathsf{J}_{\ell}^{\mathsf{T}},
\qquad
J_{JI}^{\beta\alpha}=J_{IJ}^{\alpha\beta}
$$

로 정의한다. Complex local circular basis로 변환한 coupling에는 transpose가
아니라 Hermitian conjugate가 대응한다.

같은 inter-site Hamiltonian을 independent ordered-pair sum으로 쓰는 경우에는
exchange term 앞에 $1/2$를 둔다. On-site anisotropy는 $\mathcal{L}$에 넣지 않고
별도 site sum으로 쓴다. Link sum과 independent site-pair sum을 한 유도 안에서
설명 없이 바꾸지 않는다.

## Spin and Local Frame

| Symbol | Meaning |
|---|---|
| $S_I$ | site $I$의 spin length |
| $\mathbf{n}_I$ | classical spin의 unit direction |
| $\mathbf{S}_I=S_I\mathbf{n}_I$ | classical spin vector |
| $\hat{\mathbf{S}}_I$ | laboratory-frame quantum spin operator |
| $\hat{\widetilde{\mathbf{S}}}_I$ | local-frame quantum spin operator |
| $\mathsf{R}_I$ | local components를 laboratory vector로 보내는 rotation matrix |

Rotation convention은

$$
\hat{\mathbf{S}}_I
=\mathsf{R}_I\hat{\widetilde{\mathbf{S}}}_I,
\qquad
\mathsf{R}_I^{\mathsf{T}}\mathsf{R}_I=\mathsf{I}_3
$$

로 둔다. 즉 $\mathsf{R}_I$의 열은 laboratory coordinates로 표현한 local basis
vector다. Local exchange matrix는

$$
\widetilde{\mathsf{J}}_{IJ}
=\mathsf{R}_I^{\mathsf{T}}
 \mathsf{J}_{IJ}
 \mathsf{R}_J
$$

로 정의한다. Polar angle $\theta_I$는 $+z$축에서 측정하고, azimuthal angle
$\phi_I$는 $+x$축에서 측정한다.

## Boson and Magnon Operators

Fourier transform은 operator의 representation만 바꾸므로 기호를 바꾸지 않는다.
Bogoliubov transformation으로 sublattice basis에서 magnon-band basis로 바뀔 때
처음으로 $a$에서 $b$로 바꾼다.

$$
\hat a_{i\mu}
\;\xrightarrow{\text{Fourier}}\;
\hat a_{\mathbf{k}\mu}
\;\xrightarrow{\text{Bogoliubov}}\;
\hat b_{\mathbf{k}n}.
$$

| Symbol | Meaning |
|---|---|
| $\hat a_{i\mu}$ | real-space Holstein--Primakoff boson |
| $\hat a_{\mathbf{k}\mu}$ | 같은 boson의 momentum-space representation |
| $\hat b_{\mathbf{k}n}$ | positive-energy magnon annihilation operator |
| $\hat n_{i\mu}=\hat a_{i\mu}^{\dagger}\hat a_{i\mu}$ | real-space HP boson number operator |
| $n_{\mathrm B}(\varepsilon)$ | Bose--Einstein distribution function |

Pre-diagonalization과 diagonal Nambu spinor는 각각

$$
\hat\Psi_{\mathbf{k}}
=
\begin{pmatrix}
\hat{\mathbf a}_{\mathbf{k}}\\
\hat{\mathbf a}_{-\mathbf{k}}^{\dagger}
\end{pmatrix},
\qquad
\hat\Phi_{\mathbf{k}}
=
\begin{pmatrix}
\hat{\mathbf b}_{\mathbf{k}}\\
\hat{\mathbf b}_{-\mathbf{k}}^{\dagger}
\end{pmatrix}
$$

로 쓴다. $\hat{\mathbf a}_{\mathbf{k}}$는 sublattice order,
$\hat{\mathbf b}_{\mathbf{k}}$는 positive-energy band order의 column vector다.
Fourier phase와 basis-position gauge는 momentum-space Hamiltonian 문서에서
검증 후 정의한다.

## BdG Matrices and Nambu Metric

| Symbol | Meaning |
|---|---|
| $\mathsf{H}_{\mathbf{k}}$ | $2N_{\mathrm{sub}}\times2N_{\mathrm{sub}}$ bosonic BdG matrix |
| $\mathsf{A}_{\mathbf{k}}$, $\mathsf{B}_{\mathbf{k}}$ | normal and anomalous BdG blocks |
| $\mathsf{T}_{\mathbf{k}}$ | paraunitary Bogoliubov transformation matrix |
| $\Sigma_3$ | Nambu particle--hole signature metric |

Nambu metric은

$$
\Sigma_3
\equiv
\begin{pmatrix}
\mathsf{I}_{N_{\mathrm{sub}}} & 0\\
0 & -\mathsf{I}_{N_{\mathrm{sub}}}
\end{pmatrix}
=\sigma_3\otimes\mathsf{I}_{N_{\mathrm{sub}}}.
$$

아래첨자 $3$은 matrix dimension이나 거듭제곱이 아니다. $2\times2$ Pauli matrix
$\sigma_3$의 particle--hole grading을 $2N_{\mathrm{sub}}$ 차원 Nambu space로
확장했다는 label이다. 대문자 $\Sigma_3$를 쓰면 작은 Pauli matrix와 확장된
metric을 구분할 수 있고, bare $\Sigma$는 향후 magnon self-energy에 사용할 수
있다. 이 선택은 원본과 코드의 metric symbol $J$가 exchange matrix
$\mathsf{J}_{\ell}$와 충돌하는 것도 방지한다.

$\Sigma_3$는 Nambu spinor의 위쪽 particle block에 $+1$, 아래쪽 hole block에
$-1$을 부여하며

$$
\Sigma_3^{\dagger}=\Sigma_3,
\qquad
\Sigma_3^2=\mathsf{I}_{2N_{\mathrm{sub}}}
$$

를 만족한다. Paraunitary transformation은 이 metric을 보존해야 한다.

$$
\mathsf{T}_{\mathbf{k}}^{\dagger}
\Sigma_3
\mathsf{T}_{\mathbf{k}}
=\Sigma_3.
$$

$\Sigma_3$를 처음 실제 계산에 도입하는 `paraunitary-diagonalization.md`에서는
기호만 참조하지 않고 Nambu ordering, matrix dimension, $+/-$ block의 의미와
metric-preservation condition을 함께 설명해야 한다.

## Energy Convention

| Symbol | Meaning |
|---|---|
| $\varepsilon_{n\mathbf{k}}$ | band $n$의 positive magnon energy |
| $E_{\mathrm{cl}}$ | total classical energy |
| $\Delta E_{\mathrm{zp}}$ | zero-point quantum correction |
| $E_{\mathrm{GS}}$ | LSWT ground-state energy |
| $\mathcal{E}_X=E_X/N_{\mathrm{site}}$ | magnetic site당 energy |

$$
E_{\mathrm{GS}}
=E_{\mathrm{cl}}+\Delta E_{\mathrm{zp}},
\qquad
\mathcal{E}_{\mathrm{GS}}
=\mathcal{E}_{\mathrm{cl}}+\Delta\mathcal{E}_{\mathrm{zp}}.
$$

Lowercase $e_X$는 자연상수와 혼동될 수 있으므로 energy density에 사용하지
않는다. $E_0$도 원본에서 total ground-state energy와 zero-point contribution을
모두 가리키므로 canonical notation으로 사용하지 않는다. 원본의 각 $E_0$은
문맥과 식을 검증해 $E_{\mathrm{GS}}$ 또는 $\Delta E_{\mathrm{zp}}$로 분류한다.

## Momentum and Field Symbols

$\mathbf{k}$는 magnetic translation lattice의 내부 magnon momentum으로,
$\mathbf{q}$는 neutron scattering 같은 외부 probe의 momentum transfer로
예약한다. Crystallographic Brillouin zone과 magnetic Brillouin zone을 구분할
때는 각각 `CBZ`, `MBZ`라고 명시하고, 의미가 불분명한 `FBZ`는 사용하지 않는다.

$\mathbf{B}_I$는 tesla 단위의 applied magnetic field를, $\mathbf{h}_I$는
Hamiltonian에 직접 들어가는 effective Zeeman-energy vector를 나타낸다.
$g$-tensor는 matrix 규칙에 따라 $\mathsf{g}_I$로 쓴다. 코드의
`magnetic_field`는 현재 $\mathbf B_I$가 아니라 $\mathbf h_I$에 대응한다.

## Legacy Source Mapping

이 표는 원본 기호를 보존하기 위한 non-normative mapping이다. 왼쪽 기호를
canonical Markdown에서 병행 사용한다는 뜻이 아니다.

| Source notation | Canonical notation | Reason |
|---|---|---|
| unit-cell $i,j$와 site/link $i,j$의 혼용 | cell $i,j$; site $I,J$ | index scope 분리 |
| $\mathbf r_i$ (cell), $\mathbf R_I$ (site) | $\mathbf R_i$ (cell), $\mathbf r_I$ (site) | rotation matrix와 위치 충돌 제거 |
| $\mathbf R_j$ (rotation matrix) | $\mathsf R_I$ | matrix typography와 site scope 적용 |
| $\delta_{ij}$ (undefined bond vector) | $\boldsymbol\Delta_{\ell}$ | basis offset $\boldsymbol\delta_\mu$와 구분 |
| real $a$, momentum $b$, diagonal $\beta$ | real/momentum $a$, diagonal $b$ | Fourier에서는 operator 이름 유지 |
| $\mathsf J$ (Nambu metric) | $\Sigma_3$ | exchange matrix $\mathsf J_{\ell}$와 구분 |
| $E_{\mathbf{k},\mu}$, $\varepsilon_{n,\mathbf{k}}$ | $\varepsilon_{n\mathbf{k}}$ | sublattice와 band index 분리 |
| $E_0$ | $E_{\mathrm{GS}}$ 또는 $\Delta E_{\mathrm{zp}}$ | 문맥별 의미를 검증해 분리 |
| $L$, $N$, $m_s$ | $N_{\mathrm{uc}}$, $N_{\mathrm{site}}$, $N_{\mathrm{sub}}$ | 크기 기호의 의미 고정 |

## Deferred Decisions

다음 항목은 notation을 먼저 예약하더라도 물리식 검증 전에는 확정하지 않는다.

- Fourier transform의 부호와 full-position/periodic gauge
- CBZ와 MBZ 사이의 folding 및 normalization
- momentum에 따른 magnon band ordering과 band identity
- diagonal energy matrix의 기호와 차원: $N_{\mathrm{sub}}$ positive-band matrix와
  $2N_{\mathrm{sub}}$ Nambu diagonal matrix를 구분하는 방법
- spin-correlation response matrix $\mathsf{R}_{\mathbf{k}}^{\alpha}$,
  $\mathsf{U}^{\alpha}$, $\mathsf{S}_{\mathbf{k}}$의 정확한 정의와 dagger convention
- local circular basis의 phase convention과 $+,-,0$ response formula의 범위
- zero-point $\mathbf{k}$-sum, normal-ordering constant term의 정규화와 코드 대응
- length, real time, thermal Hall conductivity의 단위 contract

이 항목들은 source review A6, A7, A12--A16, C13, C14 및 관련 코드 검증과
함께 처리한다. 수식 ID, citation, renderer 문법은 notation과 별도의 contract로
결정한다.
