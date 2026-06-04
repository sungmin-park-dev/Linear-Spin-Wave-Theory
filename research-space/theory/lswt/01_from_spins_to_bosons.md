# From Spins to Bosons

> Source: `research-space/sources/lswt/note_lswt_restructured.tex` §From Spins to Bosons
> Status: First draft

이 절의 목적은 일반적인 격자 스핀 Hamiltonian에서 시작해 LSWT가 실제로
대각화하는 quadratic bosonic Hamiltonian까지 가는 경로를 고정하는 것이다.
여기서 정하는 site, link, displacement, magnetic sublattice, Brillouin
zone convention은 이후 `common/` 후보가 될 수 있지만, 지금은 LSWT 문서
안에서 먼저 안정화한다.

## 1. Spin Hamiltonian and Index Convention

LSWT의 출발점은 실험실 좌표계의 스핀 Hamiltonian이다.

$$
\hat{H}
= \sum_{\ell=(I,J)\in \mathcal{L}}
  \sum_{\alpha,\beta}
  \hat{S}^{\alpha}_{I} J_{\ell}^{\alpha\beta}
  \hat{S}^{\beta}_{J}
- \sum_{I,\alpha} h_{I}^{\alpha}\hat{S}^{\alpha}_{I}.
$$

여기서 $I,J$는 물리적 spin site를 가리키는 composite index다. 이
문서에서는

$$
I=(n,\mu),\qquad
\mathbf{r}_{I}=\mathbf{R}_{n}+\boldsymbol{\delta}_{\mu},
$$

를 사용한다. $n$은 magnetic unit cell index이고, $\mu$는 그 magnetic
unit cell 안의 magnetic sublattice index다. 이 $\mu$는 결정학적 basis
site와 항상 같은 개념이 아니다. 결정학적 unit cell의 basis site가 magnetic
supercell 안에서 여러 magnetic sublattice로 나뉠 수도 있고, 반대로 같은
spin angle을 공유하는 site들을 같은 magnetic sublattice로 묶을 수도 있다.

### Link Convention

$\mathcal{L}$은 physical coupling link의 집합이다. 이 문서의 기본
convention은 **한 physical bond를 한 번 센다**는 것이다. 즉
$\ell=(I,J)$를 포함했다면, reverse link $(J,I)$를 별도로 더하지 않는다.
Hamiltonian을 ordered-pair sum $\sum_{I,J}$로 쓰는 경우에는 같은 물리를
나타내기 위해 exchange term 앞에 $1/2$가 필요하다.

Hermiticity는 reverse link를 명시적으로 두는 대신 다음 관계로 표현한다.

$$
J_{JI}^{\beta\alpha} = J_{IJ}^{\alpha\beta}.
$$

이 convention은 source LaTeX의 review note A4가 지적한 핵심 문제다.
이 문서에서는 link sum과 independent index sum을 섞어 쓰지 않는다.

### Displacement Convention

Momentum-space phase에 들어가는 bond displacement는 real-space vector로 둔다.

$$
\boldsymbol{\Delta}_{\ell}
= \mathbf{r}_{J}-\mathbf{r}_{I}
= (\mathbf{R}_{m}-\mathbf{R}_{n})
 +(\boldsymbol{\delta}_{\nu}-\boldsymbol{\delta}_{\mu})
\quad\text{for}\quad
\ell=((n,\mu),(m,\nu)).
$$

코드의 `Coupling.displacement`가 fractional cell displacement를 저장하는
경우, LSWT solver가 phase factor를 만들기 전에 lattice vectors로 real-space
vector로 변환해야 한다. 이론 문서에서 $\boldsymbol{\Delta}_{\ell}$는
항상 real-space vector다.

## 2. Classical Order and Local Frame

LSWT는 classical ordered state 주위의 작은 quantum fluctuation을 계산한다.
각 site $I$에는 classical spin direction

$$
\hat{\mathbf{n}}_{I}
= (\sin\theta_I\cos\varphi_I,\,
   \sin\theta_I\sin\varphi_I,\,
   \cos\theta_I)
$$

이 주어진다. Local frame은 $\hat{\mathbf{z}}_I$가
$\hat{\mathbf{n}}_I$와 일치하도록 잡는다.

$$
\hat{\mathbf{S}}_I = \mathbf{R}_I\widetilde{\mathbf{S}}_I,
\qquad
\mathbf{R}_I=\mathbf{R}(\theta_I,\varphi_I).
$$

Source LaTeX의 rotation matrix는 다음 convention을 사용한다.

$$
\mathbf{R}(\theta,\varphi)=
\begin{pmatrix}
\cos\theta\cos\varphi & -\sin\varphi & \sin\theta\cos\varphi\\
\cos\theta\sin\varphi &  \cos\varphi & \sin\theta\sin\varphi\\
-\sin\theta           & 0            & \cos\theta
\end{pmatrix}.
$$

Rotated exchange tensor와 field는

$$
\widetilde{\mathbf{J}}_{\ell}
= \mathbf{R}_{I}^{\mathrm{T}}\mathbf{J}_{\ell}\mathbf{R}_{J},
\qquad
\widetilde{\mathbf{h}}_{I}
= \mathbf{R}_{I}^{\mathrm{T}}\mathbf{h}_{I}.
$$

로 정의한다. Source LaTeX에는 field를
$\mathbf{h}_{I}^{\mathrm{T}}\mathbf{R}_{I}$처럼 row-vector 형태로 쓰는
부분이 있지만, 이 문서에서는 column-vector convention을 기본으로 둔다.

Complex local basis는

$$
\hat{\mathbf{e}}_{I}^{\pm}
=\mathbf{R}_{I}
  \frac{\hat{\mathbf{x}}\pm i\hat{\mathbf{y}}}{\sqrt{2}},
\qquad
\hat{\mathbf{e}}_{I}^{0}
=\mathbf{R}_{I}\hat{\mathbf{z}}
=\hat{\mathbf{n}}_{I}.
$$

로 정의한다. 이후 $\hat{\mathbf{e}}_I$처럼 superscript가 없는 표기는 쓰지
않고, local quantization axis는 항상 $\hat{\mathbf{e}}_I^0$로 쓴다.
이것은 source review note A3의 convention 수정이다.

## 3. Holstein-Primakoff Expansion

Local frame에서 spin operator는 Holstein-Primakoff boson으로 표현된다.
Linear spin-wave theory에서는 다음 leading terms를 사용한다.

$$
\widetilde{S}_{I}^{+}
=\sqrt{2S_I}\,\hat{a}_{I}
+O(S_I^{-1/2}),
\qquad
\widetilde{S}_{I}^{-}
=\sqrt{2S_I}\,\hat{a}_{I}^{\dagger}
+O(S_I^{-1/2}),
$$
$$
\widetilde{S}_{I}^{0}
=S_I-\hat{n}_{I},
\qquad
\hat{n}_{I}=\hat{a}_{I}^{\dagger}\hat{a}_{I}.
$$

같은 내용을 fluctuation의 수직 성분과 평행 성분으로 나누어 쓰면

$$
\delta\hat{\mathbf{S}}_I
= \delta\hat{\mathbf{S}}_I^{\perp}
 +\delta\hat{\mathbf{S}}_I^{\parallel},
\qquad
\delta\hat{\mathbf{S}}_I^{\parallel}
=-\hat{n}_{I}\hat{\mathbf{e}}_I^0.
$$

classical spin configuration이 spin magnitude constraint 아래에서
classical energy의 extremum이면 boson linear term은 사라진다. 이때

$$
\frac{\partial E_{\mathrm{cl}}}{\partial\mathbf{S}_{I}}
=\lambda_I\mathbf{S}_{I},
\qquad
\frac{\partial E_{\mathrm{cl}}}{\partial\mathbf{S}_{I}}
\cdot\delta\hat{\mathbf{S}}_I^\perp=0.
$$

여기서 solver boundary가 갈린다. Generic `SpinSystem`은 spin direction,
coupling, field, lattice data를 저장할 수 있지만, HP expansion과 $H_2$
truncation은 LSWT solver 내부 로직이다.

## 4. Quadratic Bosonic Hamiltonian in Real Space

다음 quantity를 정의한다.

$$
t_{\ell}^{\alpha\beta}
=\sqrt{S_I S_J}\,\widetilde{J}_{\ell}^{\alpha\beta},
\qquad \ell=(I,J).
$$

quadratic bosonic term만 남기면

$$
\begin{aligned}
H_2
=&
\sum_{\ell=(I,J)\in\mathcal{L}}
\Big[
t_{\ell}^{--}\hat{a}_{I}\hat{a}_{J}
+t_{\ell}^{-+}\hat{a}_{I}\hat{a}_{J}^{\dagger}
+t_{\ell}^{+-}\hat{a}_{I}^{\dagger}\hat{a}_{J}
+t_{\ell}^{++}\hat{a}_{I}^{\dagger}\hat{a}_{J}^{\dagger}
\\
&\hspace{7em}
-\widetilde{J}_{\ell}^{00}
 \left(S_I\hat{n}_{J}+S_J\hat{n}_{I}\right)
\Big]
+\sum_{I}\widetilde{h}_{I}^{0}\hat{n}_{I}.
\end{aligned}
$$

따라서 site $I$의 effective onsite coefficient는 $I$에 닿는 모든 link를
합쳐서 얻는다. $\partial I$를 site $I$에 incident한 link 집합,
$I'_{\ell}$를 link $\ell$에서 $I$의 반대편 endpoint라고 쓰면

$$
\mu_I
=\widetilde{h}_{I}^{0}
-\sum_{\ell\in\partial I}
 S_{I'_{\ell}}\widetilde{J}_{\ell}^{00}.
$$

Source LaTeX review note A8은 이 항이 unrotated $h_I$가 아니라
$\widetilde{h}_I^0$를 써야 한다고 지적한다. 이 초안은 rotated local
field를 채택한다.

## 5. Momentum-Space Representation

commensurate ordered state에서는 LSWT momentum-space problem을 magnetic
unit cell의 Brillouin zone에서 정의한다. 이 영역을
$\mathrm{BZ}_{\mathrm{mag}}$라고 쓰고, Fourier convention은 다음과 같다.

$$
\hat{b}_{n\mu}
=\frac{1}{\sqrt{L}}
\sum_{\mathbf{k}\in\mathrm{BZ}_{\mathrm{mag}}}
e^{i\mathbf{k}\cdot\mathbf{r}_{n\mu}}
\hat{b}_{\mathbf{k}\mu},
$$

$L$은 magnetic unit cell의 개수다. Real-space bond vector
$\boldsymbol{\Delta}_{\ell}$를 사용하면 hopping term과 anomalous term은
$e^{\pm i\mathbf{k}\cdot\boldsymbol{\Delta}_{\ell}}$ phase를 갖는다.

Nambu spinor는

$$
\Psi_{\mathbf{k}}^\dagger
=\left(
\hat{b}_{\mathbf{k}1}^{\dagger},\ldots,
\hat{b}_{\mathbf{k}m_s}^{\dagger},
\hat{b}_{-\mathbf{k}1},\ldots,
\hat{b}_{-\mathbf{k}m_s}
\right),
$$

로 둔다. 여기서 $m_s$는 magnetic unit cell 안의 magnetic sublattice 수다.
quadratic Hamiltonian은 bosonic BdG form으로

$$
H_2
=\frac{1}{2}
\sum_{\mathbf{k}\in\mathrm{BZ}_{\mathrm{mag}}}
\left(
\Psi_{\mathbf{k}}^\dagger
\mathsf{H}_{\mathbf{k}}
\Psi_{\mathbf{k}}
-\mathrm{Tr}\,\mathsf{A}_{\mathbf{k}}
\right),
$$
$$
\mathsf{H}_{\mathbf{k}}
=
\begin{pmatrix}
\mathsf{A}_{\mathbf{k}} & \mathsf{B}_{\mathbf{k}}\\
\mathsf{B}_{\mathbf{k}}^{\dagger} & \mathsf{A}_{-\mathbf{k}}^{*}
\end{pmatrix}.
$$

matrix constraint는

$$
\mathsf{A}_{\mathbf{k}}^\dagger=\mathsf{A}_{\mathbf{k}},
\qquad
\mathsf{B}_{-\mathbf{k}}=\mathsf{B}_{\mathbf{k}}^{\mathrm{T}}.
$$

Source의 explicit two-sublattice formula에는 아직 known review issue가
남아 있으므로, 이 초안에서는 canonical equation으로 승격하지 않는다.
$B_\mathbf{k}$ off-diagonal entry와 same-sublattice chemical-potential
factor를 확인한 뒤 다시 넣어야 한다.

## 검증 메모

- A4는 이 초안에서 처리했다. 이 문서는 independent ordered-pair sum이 아니라
  one-link counting $\sum_{\ell\in\mathcal{L}}$을 사용한다.
- A9는 이 초안에서 처리했다. $\boldsymbol{\Delta}_{\ell}$는
  $\mathbf{r}_{J}-\mathbf{r}_{I}$인 real-space bond vector다.
- A3는 이 초안에서 처리했다. local quantization axis는 항상
  $\hat{\mathbf{e}}_I^0$로 쓴다.
- A8은 이 초안에서 처리했다. onsite field contribution은
  $\widetilde{h}_I^0$를 사용한다.
- A10은 아직 열어 둔다. Same-sublattice normal-order term이 BZ 합에서
  사라진다는 주장은 $t^{-+}\neq(t^{+-})^*$인 경우 조건이 필요하다.
- A5는 아직 열어 둔다. trace/constant-term convention은 구현된 Hamiltonian
  construction과 대조해야 한다.
- A1은 canonical equation 기준으로 아직 열어 둔다. Source의
  two-sublattice $B_\mathbf{k}$ 식에는 confirmed typo가 있다.
- A2는 아직 열어 둔다. identical-sublattice $\mathsf{A}_{\mathbf{k}}$
  식의 factor 2는 검증이 필요하다.

## 작업 메모

- 이 초안은 source LaTeX의 `From Spins to Bosons` 절을 그대로 옮긴 것이
  아니라, convention과 검증 경계를 먼저 고정한 정본 초안이다.
- 다음 세부 작업은 `hamiltonian.py` 구현과 이 문서의 $H_2$,
  $\mathsf{A}_{\mathbf{k}}$, $\mathsf{B}_{\mathbf{k}}$ convention을
  대조하는 것이다.
- Explicit two-sublattice formula는 B-block typo와 factor-2 issue가
  확인될 때까지 canonical 본문에 넣지 않는다.

## Common 후보 메모

- site index convention:
  $I=(n,\mu)$, magnetic unit cell index와 magnetic sublattice index의
  분리.
- link/coupling counting:
  physical bond를 한 번 세는 $\mathcal{L}$ convention.
- displacement convention:
  solver 내부 phase factor는 real-space bond vector
  $\boldsymbol{\Delta}_{\ell}$를 사용.
- magnetic sublattice convention:
  crystallographic basis site와 magnetic sublattice는 일반적으로 다를 수
  있음.
- Brillouin zone convention:
  LSWT momentum-space Hamiltonian은 magnetic BZ에서 정의.
- solver boundary:
  `SpinSystem`에는 lattice/site/coupling/field/classical angle을 두고,
  HP boson, local-frame expansion, BdG matrix construction은 LSWT solver
  내부에 둔다.
