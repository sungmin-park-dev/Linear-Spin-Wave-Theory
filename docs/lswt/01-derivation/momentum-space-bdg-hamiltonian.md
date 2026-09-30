---
frontmatter-version: 1
title: Momentum-Space BdG Hamiltonian
doc-path: docs/lswt/01-derivation
status: draft
last-edited-by: codex
created: 2026-06-04
updated: 2026-09-06
source: docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf
source-section: Momentum Space Representations
---

# Momentum-Space BdG Hamiltonian

For a commensurate ordered state, the momentum-space LSWT problem is defined in the magnetic Brillouin zone, denoted by $\mathrm{MBZ}$. The derivation uses $N_{\mathrm{uc}}$ magnetic unit cells and $N_{\mathrm{sub}}$ magnetic sublattices per cell.

## Fourier Representation

The full-position Fourier convention used in the expressions below is

$$
\hat a_{i\mu}
=\frac{1}{\sqrt{N_{\mathrm{uc}}}}
\sum_{\mathbf k\in\mathrm{MBZ}}
\exp(\mathrm i\mathbf k\cdot\mathbf r_{i\mu})\,
\hat a_{\mathbf k\mu}.
$$

Here, $\mathbf r_{i\mu}$ is the full position of the spin site. The real-space bond vector $\boldsymbol\Delta_\ell=\mathbf r_J-\mathbf r_I$ produces phases of the form $\exp(\pm\mathrm i\mathbf k\cdot\boldsymbol\Delta_\ell)$ in the transformed hopping and anomalous terms. The sign and basis-position convention used here remain subject to the unresolved gauge and normalization conditions listed in [Notation and Conventions](../00-foundations/notation-and-conventions.md).

## Nambu Ordering and BdG Blocks

We order the Nambu spinor so that its Hermitian conjugate is

$$
\hat\Psi_{\mathbf k}^\dagger
=\left(
\hat a_{\mathbf k1}^\dagger,\ldots,
\hat a_{\mathbf k N_{\mathrm{sub}}}^\dagger,
\hat a_{-\mathbf k1},\ldots,
\hat a_{-\mathbf k N_{\mathrm{sub}}}
\right).
$$

The quadratic Hamiltonian is written in bosonic Bogoliubov–de Gennes (BdG) form as

$$
\hat H_2
=\frac12\sum_{\mathbf k\in\mathrm{MBZ}}
\left(
\hat\Psi_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\hat\Psi_{\mathbf k}
-\operatorname{Tr}\mathsf A_{\mathbf k}
\right),
$$

with block matrix

$$
\mathsf H_{\mathbf k}
=\begin{pmatrix}
\mathsf A_{\mathbf k} & \mathsf B_{\mathbf k}\\
\mathsf B_{\mathbf k}^\dagger & \mathsf A_{-\mathbf k}^*
\end{pmatrix}.
$$

The normal block $\mathsf A_{\mathbf k}$ and anomalous block $\mathsf B_{\mathbf k}$ satisfy

$$
\mathsf A_{\mathbf k}^\dagger=\mathsf A_{\mathbf k},
\qquad
\mathsf B_{-\mathbf k}=\mathsf B_{\mathbf k}^{\mathsf T}.
$$

These relations specify the block structure, but do not fix the explicit matrix elements or the additive energy convention. The trace subtraction above remains provisional until its normalization is reconciled with the real-space construction. Explicit two-sublattice expressions are not included here because the anomalous-block entries and same-sublattice factors remain unresolved.

## References

### Internal Documents

- [Notation and Conventions](../00-foundations/notation-and-conventions.md): defines the Fourier, momentum, Nambu-spinor, and matrix notation.
- [Real-Space Boson Hamiltonian](real-space-boson-hamiltonian.md): supplies the quadratic real-space terms transformed here.
- [Paraunitary Diagonalization](paraunitary-diagonalization.md): diagonalizes the bosonic BdG matrix constructed here.

### External Sources

- None.
