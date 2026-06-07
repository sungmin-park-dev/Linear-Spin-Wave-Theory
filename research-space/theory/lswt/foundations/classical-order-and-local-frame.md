---
frontmatter-version: 1
title: Classical Order and Local Frame
section: theory/lswt/foundations
status: draft
last-edited-by: codex
created: 2026-06-04
updated: 2026-06-04
source: /Users/david/Downloads/Linear_Spin_Wave_Theory___Note.pdf
source-section: Rotations for Spin Models
---

# Classical Order and Local Frame

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

Source note의 rotation matrix는 다음 convention을 사용한다.

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

로 정의한다. Source note에는 field를
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
