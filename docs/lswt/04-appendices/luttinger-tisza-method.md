---
frontmatter-version: 1
title: "Appendix: Luttinger–Tisza Method"
doc-path: docs/lswt/04-appendices
status: draft
last-edited-by: claude
created: 2026-06-07
updated: 2026-09-30
source: docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf
source-section: "None in the primary note; the restructured TeX contains only an empty TODO section (app:luttinger-tisza). Content follows Luttinger and Tisza (1946) and Lyons and Kaplan (1960)."
---

# Appendix: Luttinger–Tisza Method

LSWT expands about a classical reference configuration, and the expansion is meaningful only when that configuration minimizes the classical energy. The Luttinger–Tisza (LT) method gives a lower bound on the classical energy of a bilinear spin model by relaxing the fixed length of each spin to a single global constraint. The relaxed problem reduces to the diagonalization of a matrix in momentum space. When a configuration built from the minimizing eigenvectors also has unit spin length at every site, it attains the bound and is a classical ground state. This appendix derives the bound, the conditions for attaining it with a single wave vector, the generalization to inequivalent sublattices, and the limits of the method.

## Classical Energy as a Quadratic Form

We consider the bilinear exchange and single-ion terms of [Bilinear Spin Hamiltonian](../00-foundations/bilinear-spin-hamiltonian.md) without the Zeeman term. Replacing each spin operator by the classical vector $S_I\mathbf n_I$, with unit vector $\mathbf n_I$, and writing the link sum as an ordered-pair sum gives the classical energy

$$
E_{\mathrm{cl}}[\{\mathbf n_I\}]
=\frac12\sum_{I\neq J}S_IS_J\,\mathbf n_I^{\mathsf T}\mathsf J_{IJ}\mathbf n_J
+\sum_IS_I^2\,\mathbf n_I^{\mathsf T}\mathsf D_I\mathbf n_I,
$$

where $\mathsf J_{JI}=\mathsf J_{IJ}^{\mathsf T}$. The energy is a quadratic form in the $3N_{\mathrm{site}}$ components of the unit vectors. The ground-state problem minimizes this form under the strong constraint

$$
|\mathbf n_I|=1\quad\text{for every site }I.
$$

The LT method replaces the $N_{\mathrm{site}}$ strong constraints by the single weak constraint

$$
\sum_I|\mathbf n_I|^2=N_{\mathrm{site}},
$$

which every configuration satisfying the strong constraint also satisfies.

## Momentum-Space Form

The LT problem uses the unit cell of the Hamiltonian, with sites $I=(i,a)$ at $\mathbf r_{ia}=\mathbf R_i+\boldsymbol\delta_a$ and $a=1,\ldots,N_s$; the magnetic unit cell of the ground state is not known in advance. With the full-position Fourier convention of [Momentum-Space BdG Hamiltonian](../01-derivation/momentum-space-bdg-hamiltonian.md),

$$
\mathbf n_{ia}=\frac{1}{\sqrt{N_{\mathrm{uc}}}}\sum_{\mathbf q}\exp(\mathrm i\mathbf q\cdot\mathbf r_{ia})\,\mathbf n_a(\mathbf q),
\qquad
\mathbf n_a(-\mathbf q)=\mathbf n_a(\mathbf q)^*,
$$

where the reality condition follows from real $\mathbf n_{ia}$ and the momenta run over the Brillouin zone of the Hamiltonian's unit cell. Translation invariance makes the energy diagonal in $\mathbf q$:

$$
E_{\mathrm{cl}}=\sum_{\mathbf q}\mathbf n(\mathbf q)^\dagger\,\mathsf L_{\mathbf q}\,\mathbf n(\mathbf q),
\qquad
\big[\mathsf L_{\mathbf q}\big]_{ab}
=\frac12\sum_{J\in b}S_aS_b\,\mathsf J_{IJ}\exp\!\big(\mathrm i\mathbf q\cdot(\mathbf r_J-\mathbf r_I)\big)
+\delta_{ab}\,S_a^2\,\mathsf D_a,
$$ {#eq-lswt-lt-matrix}

where $\mathbf n(\mathbf q)$ collects the $3N_s$ components $\mathbf n_a(\mathbf q)$, $I$ is a fixed site of sublattice $a$, and the sum runs over the sites $J\neq I$ of sublattice $b$ coupled to $I$. Each $[\mathsf L_{\mathbf q}]_{ab}$ is a $3\times3$ block. The relation $\mathsf J_{JI}=\mathsf J_{IJ}^{\mathsf T}$ makes the $3N_s\times3N_s$ LT matrix $\mathsf L_{\mathbf q}$ Hermitian, and real interaction matrices give $\mathsf L_{-\mathbf q}=\mathsf L_{\mathbf q}^*$. The spin lengths are absorbed into $\mathsf L_{\mathbf q}$, so the constraints refer to unit vectors for any $S_a$. The weak constraint becomes $\sum_{\mathbf q}\|\mathbf n(\mathbf q)\|^2=N_{\mathrm{site}}$.

## Luttinger–Tisza Lower Bound

Let $\lambda_{\min}(\mathbf q)$ be the lowest eigenvalue of $\mathsf L_{\mathbf q}$ and $\lambda_{\mathrm{LT}}=\min_{\mathbf q}\lambda_{\min}(\mathbf q)$. Each momentum component satisfies $\mathbf n(\mathbf q)^\dagger\mathsf L_{\mathbf q}\mathbf n(\mathbf q)\ge\lambda_{\mathrm{LT}}\|\mathbf n(\mathbf q)\|^2$, so every configuration obeying the weak constraint, and in particular every configuration of unit spins, satisfies

$$
\frac{E_{\mathrm{cl}}}{N_{\mathrm{site}}}\ge\lambda_{\mathrm{LT}}.
$$ {#eq-lswt-lt-bound}

Equality holds exactly when $\mathbf n(\mathbf q)$ vanishes except at the minimizing wave vectors $\mathbf Q$ and their negatives, and lies in the eigenspace of $\lambda_{\mathrm{LT}}$ there. A configuration of this form that also satisfies the strong constraint attains the bound and is therefore a classical ground state, with energy $N_{\mathrm{site}}\lambda_{\mathrm{LT}}$. If no such configuration exists, $\lambda_{\mathrm{LT}}$ is a strict lower bound and the method does not determine the ground state.

## Single-Wave-Vector Configurations

The simplest candidates use one minimizing wave vector $\mathbf Q$ with an eigenvector $\mathbf w=(\mathbf w_1,\ldots,\mathbf w_{N_s})$ of $\lambda_{\mathrm{LT}}$. The component at $-\mathbf Q$ is fixed by the reality condition, since $\mathbf w^*$ is an eigenvector of $\mathsf L_{-\mathbf Q}$. The configuration is

$$
\mathbf n_{ia}=\operatorname{Re}\!\big[\mathbf u_a\exp(\mathrm i\mathbf Q\cdot\mathbf R_i)\big],
\qquad
\mathbf u_a=\mathbf w_a\exp(\mathrm i\mathbf Q\cdot\boldsymbol\delta_a),
$$

with the cell amplitude $\mathbf u_a$ and an overall scale of $\mathbf w$ still free. Its squared length is

$$
|\mathbf n_{ia}|^2=\frac12|\mathbf u_a|^2+\frac12\operatorname{Re}\!\big[(\mathbf u_a\cdot\mathbf u_a)\exp(2\mathrm i\mathbf Q\cdot\mathbf R_i)\big],
$$ {#eq-lswt-lt-single-q-length}

where $\mathbf u_a\cdot\mathbf u_a=\sum_\gamma(u_a^\gamma)^2$ carries no complex conjugation. The phases $\exp(2\mathrm i\mathbf Q\cdot\mathbf R_i)$ over all cells form a subgroup of the unit circle, and the strong constraint takes one of three forms:

- If $2\mathbf Q$ is a reciprocal-lattice vector, as at the zone center or at half reciprocal vectors, then $\exp(\mathrm i\mathbf Q\cdot\mathbf R_i)=\pm1$, the configuration is collinear within each sublattice, and the condition is $|\operatorname{Re}\mathbf u_a|=1$ after a suitable choice of the overall phase of $\mathbf w$.
- If $4\mathbf Q$ is a reciprocal-lattice vector but $2\mathbf Q$ is not, the phases $\exp(2\mathrm i\mathbf Q\cdot\mathbf R_i)$ take only the values $\pm1$. The condition is $\operatorname{Re}(\mathbf u_a\cdot\mathbf u_a)=0$ and $|\mathbf u_a|^2=2$. Writing $\mathbf u_a=\mathbf x_a+\mathrm i\mathbf y_a$, this requires $|\mathbf x_a|=|\mathbf y_a|=1$ but leaves the angle between $\mathbf x_a$ and $\mathbf y_a$ free. The configuration cycles through $\mathbf x_a,-\mathbf y_a,-\mathbf x_a,\mathbf y_a$; for $\mathbf y_a=\pm\mathbf x_a$ it is the collinear up-up-down-down pattern.
- Otherwise the phases take at least three values that are not confined to $\pm1$, and the condition is $\mathbf u_a\cdot\mathbf u_a=0$ and $|\mathbf u_a|^2=2$. Then $\mathbf x_a$ and $\mathbf y_a$ are orthonormal, and each sublattice forms a planar spiral in the plane spanned by $\mathbf x_a$ and $\mathbf y_a$.

When the eigenspace of $\lambda_{\mathrm{LT}}$ at $\mathbf Q$ has more than one dimension, $\mathbf w$ may be any vector in it, and the conditions become equations for its coefficients. For a Bravais lattice ($N_s=1$) with isotropic Heisenberg exchange, $\mathsf L_{\mathbf q}=S^2J(\mathbf q)\,\mathsf I_3$ with $J(\mathbf q)=\frac12\sum_{\boldsymbol\Delta}J_{\boldsymbol\Delta}\exp(\mathrm i\mathbf q\cdot\boldsymbol\Delta)$, where $\boldsymbol\Delta$ runs over all neighbor vectors. Every complex vector is then an eigenvector, a planar spiral always satisfies the strong constraint, and the ground state is a spiral with wave vector $\mathbf Q$, as shown by Lyons and Kaplan. For the triangular-lattice antiferromagnet with nearest-neighbor exchange $J>0$, $J(\mathbf q)=J\sum_{m=1}^{3}\cos(\mathbf q\cdot\mathbf a_m)$ over the three bond directions has its minimum $-\frac32J$ at the zone corner, and the spiral at that wave vector is the $120^\circ$ state with energy $-\frac32JS^2$ per site.

## Inequivalent Sublattices and the Generalized Method

For $N_s>1$, the minimizing eigenvector generally distributes its weight unequally among the sublattices, $|\mathbf w_a|\neq|\mathbf w_b|$, while the strong constraint requires equal lengths. The single weak constraint then allows configurations that no unit-spin state can reach, and the bound is often not attained. A stronger bound follows from one Lagrange multiplier $\lambda_a$ per sublattice. For unit spins, $\sum_I\lambda_{a(I)}(|\mathbf n_I|^2-1)=0$, so

$$
E_{\mathrm{cl}}
=\sum_{\mathbf q}\mathbf n(\mathbf q)^\dagger\big(\mathsf L_{\mathbf q}-\Lambda\big)\mathbf n(\mathbf q)
+N_{\mathrm{uc}}\sum_{a=1}^{N_s}\lambda_a,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_{N_s})\otimes\mathsf I_3.
$$

If $\mathsf L_{\mathbf q}-\Lambda$ is positive semidefinite at every $\mathbf q$, the first term is nonnegative and $E_{\mathrm{cl}}\ge N_{\mathrm{uc}}\sum_a\lambda_a$. The best bound of this form maximizes $\sum_a\lambda_a$ under that condition. Equal multipliers $\lambda_a=\lambda_{\mathrm{LT}}$ recover @eq-lswt-lt-bound, so the generalized bound is never weaker. Lyons and Kaplan introduced this generalization through adjustable parameters in the weak constraint, which allows crystallographically inequivalent spins.

## Scope and Limitations

- **Zeeman field.** The Zeeman term is linear in the spin vectors, so the energy is no longer a homogeneous quadratic form, and the bound above does not include it.
- **Multiple wave vectors.** When no single-wave-vector configuration satisfies the strong constraint, a combination of several minimizing wave vectors may still do so. Failure of the single-wave-vector construction therefore does not show that the bound is unattainable.
- **Degenerate minima.** A minimum of $\lambda_{\min}(\mathbf q)$ on a line or an extended region, or a degenerate eigenspace, signals a degenerate manifold of classical ground states. The choice among them is not made by the classical energy.

## Relation to Linear Spin-Wave Theory

LSWT requires a reference configuration of unit spins that is stationary under the fixed-length constraint, as described in [Classical Order and Local Frame](../00-foundations/classical-order-and-local-frame.md). A configuration that attains the LT bound is a global minimum of the zero-field classical energy. Every harmonic fluctuation about it therefore raises or preserves the energy, and its BdG matrix $\mathsf H_{\mathbf k}$ is at least positive semidefinite, as required in [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md). When the bound is not attained, $\lambda_{\mathrm{LT}}$ still bounds the energy of any candidate configuration obtained by numerical minimization from below, and the minimizing wave vectors indicate the magnetic unit cells to consider.

## References

### Internal Documents

- [Bilinear Spin Hamiltonian](../00-foundations/bilinear-spin-hamiltonian.md): defines the exchange and single-ion matrices and the link counting used in the classical energy.
- [Classical Order and Local Frame](../00-foundations/classical-order-and-local-frame.md): defines the classical reference configuration.
- [Momentum-Space BdG Hamiltonian](../01-derivation/momentum-space-bdg-hamiltonian.md): defines the full-position Fourier convention.
- [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md): states the positive-definiteness requirement on the BdG matrix.

### External Sources

- J. M. Luttinger and L. Tisza, "Theory of Dipole Interaction in Crystals," *Physical Review* **70**, 954–964 (1946), [doi:10.1103/PhysRev.70.954](https://doi.org/10.1103/PhysRev.70.954): representation of spin arrays as vectors in a many-dimensional space and reduction of the minimum-energy problem to a matrix diagonalization.
- D. H. Lyons and T. A. Kaplan, "Method for Determining Ground-State Spin Configurations," *Physical Review* **120**, 1580 (1960), [doi:10.1103/PhysRev.120.1580](https://doi.org/10.1103/PhysRev.120.1580): generalization with adjustable parameters in the weak constraint for inequivalent spins; spiral ground state for lattices of equivalent spins.
