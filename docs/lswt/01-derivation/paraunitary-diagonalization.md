---
frontmatter-version: 1
title: Paraunitary Diagonalization
doc-path: docs/lswt/01-derivation
status: draft
last-edited-by: claude
created: 2026-06-03
updated: 2026-09-30
source: docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf
source-section: "Diagonalization of Quadratic Boson Hamiltonian; Paraunitary Diagonalization (source pp. 9–11, Eqs. (48)–(60))"
---

# Paraunitary Diagonalization

The quadratic Hamiltonian $\hat H_2$ of [Momentum-Space BdG Hamiltonian](momentum-space-bdg-hamiltonian.md) couples the boson $\hat a_{\mathbf k\mu}$ to $\hat a_{-\mathbf k\nu}^\dagger$ through the anomalous block $\mathsf B_{\mathbf k}$. A linear transformation that mixes these operators separates $\hat H_2$ into independent magnon modes. Because the new operators must again be bosons, the transformation preserves the Nambu metric $\Sigma_3$ rather than the identity, and is paraunitary rather than unitary. The sections below derive the paraunitary condition, the particle–hole structure of the magnon spectrum, the diagonal magnon Hamiltonian with its zero-point correction, and Colpa's construction of the transformation. The construction requires $\mathsf H_{\mathbf k}$ to be positive definite; the last section describes what changes when it is not.

## Bogoliubov Transformation and the Paraunitary Condition

We write the Nambu spinor of Holstein–Primakoff bosons, $\hat\Psi_{\mathbf k}$, as a linear combination of the magnon spinor $\hat\Phi_{\mathbf k}$ defined in [Notation and Conventions](../00-foundations/notation-and-conventions.md),

$$
\hat\Psi_{\mathbf k}=\mathsf T_{\mathbf k}\hat\Phi_{\mathbf k}:
\qquad
\begin{pmatrix}
\hat{\mathbf a}_{\mathbf k}\\
\hat{\mathbf a}_{-\mathbf k}^\dagger
\end{pmatrix}
=\begin{pmatrix}
\mathsf P_{\mathbf k} & \mathsf Q_{-\mathbf k}\\
\mathsf Q_{\mathbf k}^* & \mathsf P_{-\mathbf k}^*
\end{pmatrix}
\begin{pmatrix}
\hat{\mathbf b}_{\mathbf k}\\
\hat{\mathbf b}_{-\mathbf k}^\dagger
\end{pmatrix},
$$ {#eq-lswt-bogoliubov-transformation}

where $\mathsf P_{\mathbf k}$ and $\mathsf Q_{\mathbf k}$ are $N_{\mathrm{sub}}\times N_{\mathrm{sub}}$ matrices. The first row reads $\hat a_{\mathbf k\mu}=\sum_n[(\mathsf P_{\mathbf k})_{\mu n}\hat b_{\mathbf kn}+(\mathsf Q_{-\mathbf k})_{\mu n}\hat b_{-\mathbf kn}^\dagger]$. The second row is the Hermitian conjugate of the first row at $-\mathbf k$, so the block structure of $\mathsf T_{\mathbf k}$ follows from the requirement that the lower components of $\hat\Psi_{\mathbf k}$ are the adjoints of the upper components of $\hat\Psi_{-\mathbf k}$.

Both spinors obey the same bosonic commutation relations. For the components of $\hat\Psi_{\mathbf k}$ these relations read $[(\hat\Psi_{\mathbf k})_p,(\hat\Psi_{\mathbf k}^\dagger)_q]=(\Sigma_3)_{pq}$, since the lower components are creation operators. Requiring the same relations for $\hat\Phi_{\mathbf k}$ and inserting the transformation gives $[(\hat\Psi_{\mathbf k})_p,(\hat\Psi_{\mathbf k}^\dagger)_q]=(\mathsf T_{\mathbf k}\Sigma_3\mathsf T_{\mathbf k}^\dagger)_{pq}$, so the transformation must satisfy

$$
\mathsf T_{\mathbf k}\Sigma_3\mathsf T_{\mathbf k}^\dagger=\Sigma_3
\quad\Longleftrightarrow\quad
\mathsf T_{\mathbf k}^\dagger\Sigma_3\mathsf T_{\mathbf k}=\Sigma_3.
$$ {#eq-lswt-paraunitary-condition}

In blocks, the first form contains the four conditions $\mathsf P_{\mathbf k}\mathsf P_{\mathbf k}^\dagger-\mathsf Q_{-\mathbf k}\mathsf Q_{-\mathbf k}^\dagger=\mathsf I$, $\mathsf P_{\mathbf k}\mathsf Q_{\mathbf k}^{\mathsf T}-\mathsf Q_{-\mathbf k}\mathsf P_{-\mathbf k}^{\mathsf T}=0$, and their conjugate counterparts. The two forms are equivalent because either one gives the inverse $\mathsf T_{\mathbf k}^{-1}=\Sigma_3\mathsf T_{\mathbf k}^\dagger\Sigma_3$. A unitary matrix preserves $\mathsf I$ instead of $\Sigma_3$; a paraunitary $\mathsf T_{\mathbf k}$ is unitary only when $\mathsf Q_{\mathbf k}=0$, that is, when $\hat H_2$ conserves the boson number.

## Diagonal Form and the Particle–Hole Structure of the Spectrum

We seek a paraunitary $\mathsf T_{\mathbf k}$ for which $\mathsf D_{\mathbf k}=\mathsf T_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}$ is diagonal. With the inverse above, this condition is equivalent to an eigenvalue problem:

$$
\mathsf T_{\mathbf k}^{-1}\,\Sigma_3\mathsf H_{\mathbf k}\,\mathsf T_{\mathbf k}
=\Sigma_3\mathsf T_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}
=\Sigma_3\mathsf D_{\mathbf k}.
$$

The columns of $\mathsf T_{\mathbf k}$ are therefore eigenvectors of the non-Hermitian matrix $\Sigma_3\mathsf H_{\mathbf k}$, and the signed diagonal entries $\lambda_{p\mathbf k}=(\Sigma_3)_{pp}(\mathsf D_{\mathbf k})_{pp}$ are its eigenvalues.

The BdG matrix relates $\mathbf k$ and $-\mathbf k$. With $\Sigma_1=\sigma_1\otimes\mathsf I_{N_{\mathrm{sub}}}$, which exchanges the particle and hole blocks, the block relations $\mathsf A_{\mathbf k}^\dagger=\mathsf A_{\mathbf k}$ and $\mathsf B_{-\mathbf k}=\mathsf B_{\mathbf k}^{\mathsf T}$ give

$$
\Sigma_1\mathsf H_{-\mathbf k}^*\Sigma_1=\mathsf H_{\mathbf k}.
$$ {#eq-lswt-bdg-particle-hole-symmetry}

Since $\Sigma_3\Sigma_1=-\Sigma_1\Sigma_3$, an eigenvector $\mathbf v$ of $\Sigma_3\mathsf H_{-\mathbf k}$ with eigenvalue $\lambda$ yields the eigenvector $\Sigma_1\mathbf v^*$ of $\Sigma_3\mathsf H_{\mathbf k}$ with eigenvalue $-\lambda^*$. For a positive-definite $\mathsf H_{\mathbf k}$, the section on Colpa's construction shows that the eigenvalues are real, $N_{\mathrm{sub}}$ of them positive and $N_{\mathrm{sub}}$ negative. We denote the positive eigenvalues at $\mathbf k$ by the magnon energies $\varepsilon_{n\mathbf k}$; the negative eigenvalues at $\mathbf k$ are then $-\varepsilon_{n,-\mathbf k}$. Ordering the positive eigenvalues first, the diagonal form is

$$
\mathsf T_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}
=\operatorname{diag}\!\left(\varepsilon_{1\mathbf k},\ldots,\varepsilon_{N_{\mathrm{sub}}\mathbf k},\varepsilon_{1,-\mathbf k},\ldots,\varepsilon_{N_{\mathrm{sub}},-\mathbf k}\right).
$$ {#eq-lswt-bdg-diagonal-form}

The hole columns of $\mathsf T_{\mathbf k}$ can be chosen as $\Sigma_1$ times the complex conjugates of the particle columns of $\mathsf T_{-\mathbf k}$, which reproduces the block structure of @eq-lswt-bogoliubov-transformation. The energies $\varepsilon_{n\mathbf k}$ and $\varepsilon_{n,-\mathbf k}$ need not coincide; they coincide when a symmetry of the model and the ordered state maps $\mathbf k$ to $-\mathbf k$.

## Diagonal Magnon Hamiltonian and Zero-Point Correction

Inserting $\hat\Psi_{\mathbf k}=\mathsf T_{\mathbf k}\hat\Phi_{\mathbf k}$ into $\hat H_2=\frac12\sum_{\mathbf k}\big(\hat\Psi_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\hat\Psi_{\mathbf k}-\operatorname{Tr}\mathsf A_{\mathbf k}\big)$ and using @eq-lswt-bdg-diagonal-form gives

$$
\hat H_2
=\frac12\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}
\left(\varepsilon_{n\mathbf k}\hat b_{\mathbf kn}^\dagger\hat b_{\mathbf kn}
+\varepsilon_{n,-\mathbf k}\hat b_{-\mathbf kn}\hat b_{-\mathbf kn}^\dagger\right)
-\frac12\sum_{\mathbf k\in\mathrm{MBZ}}\operatorname{Tr}\mathsf A_{\mathbf k}.
$$

The commutator $\hat b_{-\mathbf kn}\hat b_{-\mathbf kn}^\dagger=\hat b_{-\mathbf kn}^\dagger\hat b_{-\mathbf kn}+1$ and the relabeling $-\mathbf k\to\mathbf k$, allowed because the $N_{\mathrm{uc}}$ momenta of the MBZ form a set closed under inversion modulo reciprocal-lattice vectors, turn this expression into

$$
\hat H_2
=\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}\varepsilon_{n\mathbf k}\,\hat b_{\mathbf kn}^\dagger\hat b_{\mathbf kn}
+\Delta E_{\mathrm{zp}},
\qquad
\Delta E_{\mathrm{zp}}
=\frac12\sum_{\mathbf k\in\mathrm{MBZ}}\left(\sum_{n=1}^{N_{\mathrm{sub}}}\varepsilon_{n\mathbf k}-\operatorname{Tr}\mathsf A_{\mathbf k}\right).
$$ {#eq-lswt-diagonal-magnon-hamiltonian}

Each magnon mode $(\mathbf k,n)$ is an independent harmonic oscillator of energy $\varepsilon_{n\mathbf k}$. The magnon vacuum, annihilated by every $\hat b_{\mathbf kn}$, is the ground state of $\hat H_2$, and the LSWT ground-state energy is

$$
E_{\mathrm{GS}}=E_{\mathrm{cl}}+\Delta E_{\mathrm{zp}},
\qquad
\mathcal E_{\mathrm{GS}}=\frac{E_{\mathrm{GS}}}{N_{\mathrm{site}}},
\qquad
N_{\mathrm{site}}=N_{\mathrm{uc}}N_{\mathrm{sub}}.
$$ {#eq-lswt-zero-point-correction}

The zero-point correction $\Delta E_{\mathrm{zp}}$ is the energy of the magnon vacuum relative to the classical energy $E_{\mathrm{cl}}$; it combines the zero-point energies $\frac12\varepsilon_{n\mathbf k}$ of the oscillators with the constant produced by normal ordering the Holstein–Primakoff Hamiltonian. That constant is the trace subtraction in $\hat H_2$, whose normalization is still provisional in [Momentum-Space BdG Hamiltonian](momentum-space-bdg-hamiltonian.md); $\Delta E_{\mathrm{zp}}$ inherits that condition. Because $\operatorname{Tr}\mathsf A_{\mathbf k}$ is real and the momentum set is closed under inversion, $\sum_{\mathbf k}\operatorname{Tr}\mathsf A_{\mathbf k}=\frac12\sum_{\mathbf k}\operatorname{Tr}\mathsf H_{\mathbf k}$, so the correction can equally be written as $\frac12\sum_{\mathbf k}\sum_n\varepsilon_{n\mathbf k}-\frac14\sum_{\mathbf k}\operatorname{Tr}\mathsf H_{\mathbf k}$. The two forms agree after the momentum sum; their summands need not agree at a single momentum.

## Colpa's Construction for a Positive-Definite BdG Matrix

A paraunitary $\mathsf T_{\mathbf k}$ that brings $\mathsf H_{\mathbf k}$ to a diagonal form with positive entries exists if and only if $\mathsf H_{\mathbf k}$ is positive definite. If such a $\mathsf T_{\mathbf k}$ exists, $\mathsf H_{\mathbf k}=(\mathsf T_{\mathbf k}^{-1})^\dagger\mathsf D_{\mathbf k}\mathsf T_{\mathbf k}^{-1}$ is congruent to a positive diagonal matrix and is therefore positive definite. Conversely, Colpa's construction produces $\mathsf T_{\mathbf k}$ from a positive-definite $\mathsf H_{\mathbf k}$. We suppress the momentum index in the rest of this section.

A positive-definite $\mathsf H$ has a Cholesky factorization $\mathsf H=\mathsf K\mathsf K^\dagger$ with an invertible lower-triangular $\mathsf K$. The Hermitian matrix $\mathsf K^\dagger\Sigma_3\mathsf K$ is congruent to $\Sigma_3$, so by Sylvester's law of inertia it has $N_{\mathrm{sub}}$ positive and $N_{\mathrm{sub}}$ negative eigenvalues. We diagonalize it with a unitary matrix $\mathsf V$,

$$
\mathsf V\Lambda\mathsf V^\dagger=\mathsf K^\dagger\Sigma_3\mathsf K,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_{2N_{\mathrm{sub}}}),
$$

ordering the eigenvalues so that the positive ones come first. The diagonal matrix $\Lambda\Sigma_3$ then has only positive entries, and the transformation is

$$
\mathsf T=(\mathsf K^\dagger)^{-1}\mathsf V(\Lambda\Sigma_3)^{1/2}.
$$ {#eq-lswt-colpa-transformation}

Both requirements follow from the definitions of $\mathsf K$ and $\mathsf V$, using that the diagonal matrices $\Lambda$, $\Sigma_3$, and $(\Lambda\Sigma_3)^{1/2}$ commute and that $(\Lambda\Sigma_3)^{1/2}$ is real:

$$
\begin{aligned}
\mathsf T\Sigma_3\mathsf T^\dagger
&=(\mathsf K^\dagger)^{-1}\mathsf V(\Lambda\Sigma_3)^{1/2}\Sigma_3(\Lambda\Sigma_3)^{1/2}\mathsf V^\dagger\mathsf K^{-1}
=(\mathsf K^\dagger)^{-1}\mathsf V\Lambda\mathsf V^\dagger\mathsf K^{-1}
=\Sigma_3,\\
\mathsf T^\dagger\mathsf H\mathsf T
&=(\Lambda\Sigma_3)^{1/2}\mathsf V^\dagger\mathsf K^{-1}\,\mathsf K\mathsf K^\dagger\,(\mathsf K^\dagger)^{-1}\mathsf V(\Lambda\Sigma_3)^{1/2}
=\Lambda\Sigma_3.
\end{aligned}
$$

The eigenvalues $\lambda_p$ are those of $\Sigma_3\mathsf H$, because $\Sigma_3\mathsf H=\Sigma_3\mathsf K\mathsf K^\dagger$ is similar to $\mathsf K^\dagger\Sigma_3\mathsf K$ through $\mathsf K^\dagger$. They are the signed energies of the previous section, $\varepsilon_{n\mathbf k}$ on the particle columns and $-\varepsilon_{n,-\mathbf k}$ on the hole columns, and $\Lambda\Sigma_3$ is the diagonal matrix of @eq-lswt-bdg-diagonal-form. This also shows that the eigenvalues of $\Sigma_3\mathsf H_{\mathbf k}$ are real when $\mathsf H_{\mathbf k}$ is positive definite.

The construction fixes $\mathsf T_{\mathbf k}$ only up to $\mathsf T_{\mathbf k}\to\mathsf T_{\mathbf k}\mathsf U$, where $\mathsf U$ is paraunitary and commutes with $\Sigma_3\mathsf D_{\mathbf k}$. Such a $\mathsf U$ is block diagonal in the eigenspaces of the signed energies, which do not mix particle and hole columns when all energies are positive, and is unitary within each block. It therefore multiplies each column by a phase and mixes columns only within degenerate bands. The energies and $\Delta E_{\mathrm{zp}}$ do not depend on this freedom; the Berry connection of [Topological Magnon Quantities](../02-observables/topological-magnon-quantities.md) does. Hole columns obtained independently at $\mathbf k$ agree with the particle–hole partners of the particle columns at $-\mathbf k$ up to the same freedom.

## Positive-Semidefinite and Indefinite BdG Matrices

The quadratic form $\hat\Psi_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\hat\Psi_{\mathbf k}$ is the harmonic energy of the spin fluctuations about the classical reference configuration of [Classical Order and Local Frame](../00-foundations/classical-order-and-local-frame.md). Positive definiteness of $\mathsf H_{\mathbf k}$ at every momentum means that every harmonic fluctuation raises the energy. If $\mathsf H_{\mathbf k}$ is indefinite at some momentum, a fluctuation lowers the energy, the reference configuration is not a local minimum, and LSWT about it does not describe stable magnons. The Cholesky factorization then fails, and $\Sigma_3\mathsf H_{\mathbf k}$ may have complex eigenvalues.

A positive-semidefinite $\mathsf H_{\mathbf k}$ has zero modes, for example Goldstone modes of a broken continuous symmetry. Its Cholesky factor is singular, and Colpa's construction does not apply. A single mode with real blocks $\mathsf A=A$ and $\mathsf B=B$, for which $\varepsilon=\sqrt{A^2-B^2}$, shows the two possible behaviors at $\varepsilon=0$:

- For $A=B>0$, $\mathsf H$ is positive semidefinite and $\Sigma_3\mathsf H=A\begin{pmatrix}1&1\\-1&-1\end{pmatrix}$ satisfies $(\Sigma_3\mathsf H)^2=0$. The matrix is not diagonalizable, and no paraunitary $\mathsf T$ exists. The Hamiltonian $\frac A2(\hat a+\hat a^\dagger)^2$ contains one canonical coordinate but not its conjugate momentum, so the mode is not a harmonic oscillator. A Goldstone mode of an antiferromagnet at $\mathbf k=0$ has this form.
- For $A=B=0$, $\Sigma_3\mathsf H=0$ is diagonalizable, $\mathsf T=\mathsf I$ is paraunitary, and the mode is a boson of zero energy. The Goldstone mode of a Heisenberg ferromagnet at $\mathbf k=0$ and zero field has this form.

Adding a small positive shift $\delta\,\mathsf I_{2N_{\mathrm{sub}}}$ makes $\mathsf H_{\mathbf k}$ positive definite. The energies then converge as $\delta\to0$, but in the first case the entries of $\mathsf T$ grow as $\delta^{-1/4}$, since $\varepsilon=\sqrt{2A\delta+\delta^2}$ while the diagonal entries of $\mathsf T$ scale as $(A/\varepsilon)^{1/2}$. Zero modes usually occur at isolated momenta, where $\varepsilon_{n\mathbf k}\to0$ continuously; they do not change $\Delta E_{\mathrm{zp}}$. Quantities weighted by $1/\varepsilon_{n\mathbf k}$ or by the Bose–Einstein distribution, such as magnon numbers and correlation functions, require separate treatment and are discussed in the observable documents. A complete diagonalization theory for positive-semidefinite quadratic boson Hamiltonians, which brings them to a standard form that includes such zero modes, is given by Colpa (1986).

## References

### Internal Documents

- [Notation and Conventions](../00-foundations/notation-and-conventions.md): defines the Nambu spinors $\hat\Psi_{\mathbf k}$ and $\hat\Phi_{\mathbf k}$, the metric $\Sigma_3$, the paraunitary matrix $\mathsf T_{\mathbf k}$, and the energy symbols $\varepsilon_{n\mathbf k}$, $E_{\mathrm{cl}}$, $\Delta E_{\mathrm{zp}}$, and $E_{\mathrm{GS}}$.
- [Momentum-Space BdG Hamiltonian](momentum-space-bdg-hamiltonian.md): supplies $\hat H_2$, the BdG blocks $\mathsf A_{\mathbf k}$ and $\mathsf B_{\mathbf k}$, their block relations, and the provisional trace subtraction.
- [Classical Order and Local Frame](../00-foundations/classical-order-and-local-frame.md): defines the classical reference configuration about which $\mathsf H_{\mathbf k}$ is expanded.
- [Paraunitarity Proofs](../04-appendices/paraunitarity-proofs.md): is reserved for extended metric-preservation proofs.
- [Magnon Observables](../02-observables/magnon-observables.md): uses the magnon energies and the transformation $\mathsf T_{\mathbf k}$.
- [Topological Magnon Quantities](../02-observables/topological-magnon-quantities.md): uses the columns of $\mathsf T_{\mathbf k}$ and their gauge freedom in the Berry curvature.

### External Sources

- J. H. P. Colpa, "Diagonalization of the Quadratic Boson Hamiltonian," *Physica A* **93**, 327–353 (1978), [doi:10.1016/0378-4371(78)90160-7](https://doi.org/10.1016/0378-4371(78)90160-7): paraunitary diagonalization of a positive-definite quadratic boson Hamiltonian through the Cholesky factorization; the construction of @eq-lswt-colpa-transformation.
- J. H. P. Colpa, "Diagonalization of the Quadratic Boson Hamiltonian with Zero Modes," *Physica A* **134**, 377–416 (1986), [doi:10.1016/0378-4371(86)90056-7](https://doi.org/10.1016/0378-4371(86)90056-7), and **134**, 417–442 (1986), [doi:10.1016/0378-4371(86)90057-9](https://doi.org/10.1016/0378-4371(86)90057-9): standard form of positive-semidefinite quadratic boson Hamiltonians with zero modes.
