---
frontmatter-version: 1
title: Bilinear Spin Hamiltonian
section: theory/lswt/foundations
status: draft
last-edited-by: codex
created: 2026-06-04
updated: 2026-06-04
source: /Users/david/Downloads/Linear_Spin_Wave_Theory___Note.pdf
source-section: Introduction to Spin Wave Theory, Eq. (1)-Eq. (3)
---

# Bilinear Spin Hamiltonian

이 파일은 LSWT 유도에서 사용할 spin Hamiltonian의 범위와 summation
convention을 고정한다. 원본 PDF는 $\sum_{ij}$를 쓰면서 $ij$가 lattice
index pair가 아니라 link를 뜻한다고 설명한다. 이 문서에서는 그 convention을
명시적으로 드러내기 위해 physical link를 $\ell=(I,J)\in\mathcal{L}$로 쓴다.

## Hamiltonian

$$
\hat{H}
= \sum_{\ell=(I,J)\in \mathcal{L}}
  \sum_{\alpha,\beta}
  \hat{S}^{\alpha}_{I} J_{\ell}^{\alpha\beta}
  \hat{S}^{\beta}_{J}
- \sum_{I,\alpha} h_{I}^{\alpha}\hat{S}^{\alpha}_{I}.
\tag{1}
$$

이 식에서 $I$와 $J$는 physical spin site이고,
$\alpha,\beta\in\{x,y,z\}$는 spin-component index다.
$\hat{S}^{\alpha}_{I}$는 site $I$에 있는 spin operator의 $\alpha$ component를
뜻한다. 첫 번째 항은 exchange Hamiltonian이고, 두 번째 항은 Zeeman term이다.

Link set $\mathcal{L}$은 각 physical coupling을 한 번만 센다. 같은
Hamiltonian을 $I,J$에 대한 independent ordered-pair sum으로 쓰면 exchange
term 앞에 $1/2$가 필요하다:

$$
\sum_{\ell=(I,J)\in\mathcal{L}}
\sum_{\alpha,\beta}
\hat{S}^{\alpha}_{I}J_{\ell}^{\alpha\beta}\hat{S}^{\beta}_{J}
\quad\longleftrightarrow\quad
\frac{1}{2}
\sum_{I,J}
\sum_{\alpha,\beta}
\hat{S}^{\alpha}_{I}J_{IJ}^{\alpha\beta}\hat{S}^{\beta}_{J}.
\tag{2}
$$

Eq. (2)는 새로운 물리 가정이 아니라 convention statement다. 이 식은 원본
노트의 link sum이 full site-pair sum으로 오해되는 것을 막기 위한 것이다.

## Exchange Interactions

Bilinear spin model의 exchange part는 inter-site coupling과 on-site
anisotropy로 support를 나누어 쓸 수 있다:

$$
\hat{H}_{\mathrm{ex}}
=
\sum_{\ell=(I,J)\in\mathcal{L}_{\mathrm{inter}}}
\sum_{\alpha,\beta}
\hat{S}^{\alpha}_{I}J_{\ell}^{\alpha\beta}\hat{S}^{\beta}_{J}
+
\sum_I
\hat{\mathbf{S}}_{I}\cdot\mathbf{A}_{I}\cdot\hat{\mathbf{S}}_{I}.
\tag{3}
$$

Inter-site tensor $\mathbf{J}_{\ell}$는 isotropic exchange,
Dzyaloshinskii-Moriya interaction, Kitaev-type coupling 같은 anisotropic
exchange를 표현할 수 있다. On-site tensor $\mathbf{A}_{I}$는 single-ion
anisotropy를 나타낸다. 예를 들어 $-A\sum_I(\hat{S}^x_I)^2$는 $x$ direction의
easy-axis anisotropy를 기술한다.

이 문서에서는 Eq. (3)의 support class를 LSWT의 출발 Hamiltonian으로 둔다.
Scalar spin chirality나 ring exchange처럼 두 개보다 많은 spin operator에
support를 갖는 interaction은 이 bilinear starting point 밖에 있다.

## Zeeman Term and g-Tensor

Magnetic field는 material의 $g$-tensor를 통해 spin과 결합한다.

$$
\sum_I\mathbf{h}_{I}\cdot\hat{\mathbf{S}}_{I}
= \mu_B
\sum_I
\sum_{\alpha,\beta}
B_I^\alpha g_I^{\alpha\beta}\hat{S}_I^\beta.
\tag{4}
$$

Energy를 meV로, magnetic field를 tesla로 측정할 때 Bohr magneton은
$\mu_B=0.057883\,\mathrm{meV/T}$다. Tensor $g_I^{\alpha\beta}$는 external
magnetic field와 spin operator 사이의 coupling을 매개한다. Magnetic
materials에서는 spin-orbit coupling과 crystal-field effect 때문에 이 tensor가
anisotropic할 수 있다.

## Convention Links

- Site, link, and displacement convention은 `notation-and-conventions.md`에서
  관리한다.
- 이 파일은 LSWT가 대상으로 삼는 bilinear Hamiltonian의 물리적 형태를
  설명한다.
- Local-frame rotation 이후의 $\widetilde{\mathbf{J}}_{\ell}$와
  $\widetilde{\mathbf{h}}_I$는 `classical-order-and-local-frame.md`에서
  정의한다.
- HP expansion 이후 이 Hamiltonian이 만드는 quadratic bosonic Hamiltonian은
  `../derivation/real-space-boson-hamiltonian.md`에서 다룬다.

## 검증 메모

- Source review note A4는 이 파일에서 처리한다. 본문은 ambiguous
  $\sum_{ij}$ 대신 one-link counting $\sum_{\ell\in\mathcal{L}}$를 쓴다.
- Source review note C22는 이 파일에서 처리한다. 본문은 `Eq.` 표기를
  기준으로 둔다.
- Eq. (3)의 on-site anisotropy를 link set에 포함할지, 별도 on-site support로
  둘지는 code convention과 대조할 때 다시 확인한다.

## Common 후보 메모

- one-link counting과 ordered-pair counting의 대응은 LSWT뿐 아니라
  Monte Carlo, tensor network, exact diagonalization에서도 공유해야 할
  Hamiltonian convention이다.
- bilinear interaction 밖의 scalar spin chirality, ring exchange는 향후
  common spin-Hamiltonian 문서에서 별도 Hamiltonian class로 정리할 후보다.
