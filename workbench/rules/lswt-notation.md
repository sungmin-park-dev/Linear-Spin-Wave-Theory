---
frontmatter-version: 1
title: LSWT Notation Rules
section: policies
status: accepted
last-edited-by: claude
created: 2026-10-20
updated: 2026-10-20
---

# LSWT Notation Rules

LSWT 이론 연구노트(`workbench/topics/*/lswt-*`)를 쓸 때 따르는 기호 작성 규칙이다. 2026-10-20 사용자 코멘트 c-20261019-1006에 따라 연구노트 *LSWT spin Hamiltonian*의 Notation and Conventions 절에서 글꼴 규칙과 원문 기호 대응표를 이 파일로 옮겼다. 내용은 2026-10-20 사용자가 그 노트를 accepted로 정할 때의 문장 그대로다. 기호의 물리적 정의(첨자, 격자, 링크 방향, 국소 틀, Nambu 표기, 에너지, 단위, 브릴루앙 영역 합, 띠 번호)는 연구노트 *LSWT conventions*가 소유한다.

## Typography

| Object | Form | Examples |
|:---|:---|:---|
| Scalar | Italic | $S_I$, $E_{\mathrm{cl}}$, $\varepsilon_{n\mathbf{k}}$ |
| Vector | Bold | $\mathbf{r}_I$, $\mathbf{k}$, $\hat{\mathbf{S}}_I$ |
| Latin-letter matrix | Sans serif | $\mathsf{J}_{\ell}$, $\mathsf{D}_I$, $\mathsf{R}_I$, $\mathsf{H}_{\mathbf{k}}$ |
| Quantum operator | Hat | $\hat H$, $\hat a_{i\mu}$, $\hat\Psi_{\mathbf{k}}$ |
| Set | Calligraphic | $\mathcal{L}$ |
| Identity matrix | Sans serif | $\mathsf{I}_N$ |

Hats denote quantum operators. Classical spin vectors and unit directions are written without hats, as $\mathbf S_I$ and $\mathbf n_I$. The Hamiltonian operator $\hat H$ is therefore distinct from its quadratic coefficient matrix $\mathsf H_{\mathbf k}$.

Conventional Greek matrix symbols, such as $\Sigma_3$, are exceptions to the sans-serif rule for Latin letters. Their dimensions and roles are stated where they are introduced. Transposition, complex conjugation, and Hermitian conjugation are denoted by superscripts $\mathsf T$, $*$, and $\dagger$, respectively. Exponentials are written as $\exp(\cdots)$, and the imaginary unit is $\mathrm i$.

A base letter may be shared by different object classes when typography makes the distinction explicit. Examples include the temperature $T$ and transformation matrix $\mathsf T_{\mathbf k}$, or the Dzyaloshinskii–Moriya vector $\mathbf D_\ell$ and on-site anisotropy matrix $\mathsf D_I$. The same matrix symbol is not assigned two different meanings.

## Legacy Source Mapping

This table translates source notation into the notation of the LSWT research notes. The source symbols in the first column are not additional alternatives for use in the derivation.

| Source notation | Current notation | Reason |
|:---|:---|:---|
| $i,j$ used for both cells and sites or links | Cells $i,j$; sites $I,J$ | Separates the index domains |
| $\mathbf r_i$ for a cell and $\mathbf R_I$ for a site | $\mathbf R_i$ for a cell and $\mathbf r_I$ for a site | Distinguishes positions from rotation matrices |
| $J_{ij}^{\alpha\beta}$ with $ij$ denoting a link | $J_\ell^{\alpha\beta}$, $\ell=(I,J)$ | Separates link identity from cell indices |
| $\mathbf A_i$, $A_i^{\alpha\beta}$ for single-ion anisotropy | $\mathsf D_I$, $D_I^{\alpha\beta}$ | Distinguishes anisotropy from the normal BdG block $\mathsf A_{\mathbf k}$ |
| $\mathbf R_j$ described as laboratory-to-local in the prose but used as local-to-laboratory in source equations | $\mathsf R_I$ with $\hat{\mathbf S}_I=\mathsf R_I\hat{\widetilde{\mathbf S}}_I$ | Follows the local-to-laboratory direction of the source equations; the conflicting source prose is recorded in `workbench/tasks/2026-08-09-lswt-documentation-audit.md` |
| $\delta_{ij}$ for an undefined bond vector | $\boldsymbol\Delta_\ell$ | Distinguishes the bond displacement from $\boldsymbol\delta_\mu$ |
| Real-space $a$, momentum-space $b$, diagonal $\beta$ | Real- and momentum-space $a$, diagonal $b$ | Preserves the operator name under Fourier transformation |
| $\mathsf J$ for the Nambu metric | $\Sigma_3$ | Distinguishes the metric from exchange matrices |
| $\mathbf h_j$ called an external magnetic field | Applied field $\mathbf B_I$ and Zeeman-energy vector $\mathbf h_I$ | Separates field and energy units |
| $R_{\mathbf k}^\alpha$ for a spin-response matrix | Spin vertex $\mathsf V^\alpha$ | Distinguishes the vertex from rotations; in the full-position convention it does not depend on the momentum |
| $E_{\mathbf k,\mu}$, $\varepsilon_{n,\mathbf k}$ | $\varepsilon_{n\mathbf k}$ | Distinguishes band and sublattice indices |
| $E_0$ | $E_{\mathrm{GS}}$ or $\Delta E_{\mathrm{zp}}$ | Separates the two source meanings |
| $L$, $N$, $m_s$ | $N_{\mathrm{uc}}$, $N_{\mathrm{site}}$, $N_{\mathrm{sub}}$ | Fixes the meaning of each system size |
