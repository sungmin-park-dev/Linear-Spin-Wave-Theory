---
frontmatter-version: 1
title: Momentum-Space BdG Hamiltonian
section: theory/lswt/derivation
status: draft
last-edited-by: codex
created: 2026-06-04
updated: 2026-06-04
source: /Users/david/Downloads/Linear_Spin_Wave_Theory___Note.pdf
source-section: Momentum Space Representations
---

# Momentum-Space BdG Hamiltonian

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

- A9는 이 초안에서 처리했다. $\boldsymbol{\Delta}_{\ell}$는
  $\mathbf{r}_{J}-\mathbf{r}_{I}$인 real-space bond vector다.
- A10은 아직 열어 둔다. Same-sublattice normal-order term이 BZ 합에서
  사라진다는 주장은 $t^{-+}\neq(t^{+-})^*$인 경우 조건이 필요하다.
- A5는 아직 열어 둔다. trace/constant-term convention은 구현된 Hamiltonian
  construction과 대조해야 한다.
- A1은 canonical equation 기준으로 아직 열어 둔다. Source의
  two-sublattice $B_\mathbf{k}$ 식에는 confirmed typo가 있다.
- A2는 아직 열어 둔다. identical-sublattice $\mathsf{A}_{\mathbf{k}}$
  식의 factor 2는 검증이 필요하다.
