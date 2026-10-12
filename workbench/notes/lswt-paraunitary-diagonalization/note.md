<!-- Converted from LaTeX on 2026-10-11. The LaTeX original is kept in docs/archive/2026-10-11-lswt-latex-notes/lswt-paraunitary-diagonalization/. Equation numbers are fixed in this form; the original \label of each numbered equation is in the comment before it. -->

<!-- Paraunitary Diagonalization Converted on 2026-10-08 from docs/archive/2026-10-08-lswt-markdown/01-derivation/paraunitary-diagonalization.md (status: draft, source-section: "Diagonalization of Quadratic Boson Hamiltonian; Paraunitary Diagonalization (source pp. 9–11, Eqs. (48)–(60))"). Review status: draft; awaiting the user's physics and mathematics review. -->

# Paraunitary Diagonalization

The quadratic Hamiltonian $\hat H_2$ of Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian* couples the boson $\hat a_{\mathbf k\mu}$ to $\hat a_{-\mathbf k\nu}^\dagger$ through the anomalous block $\mathsf B_{\mathbf k}$. A linear transformation that mixes these operators separates $\hat H_2$ into independent magnon modes. Because the new operators must again be bosons, the transformation preserves the Nambu metric $\Sigma_3$ rather than the identity, and is paraunitary rather than unitary. The sections below derive the paraunitary condition, the particle–hole structure of the magnon spectrum, the diagonal magnon Hamiltonian with its zero-point correction, and Colpa’s construction of the transformation. The construction requires $\mathsf H_{\mathbf k}$ to be positive definite; the last section describes what changes when it is not.

## Bogoliubov Transformation and the Paraunitary Condition

We write the Nambu spinor of Holstein–Primakoff bosons, $\hat\Psi_{\mathbf k}$, as a linear combination of the magnon spinor $\hat\Phi_{\mathbf k}$ defined in Notation and Conventions in research note *LSWT spin Hamiltonian*,

<!-- eq 1: eq-lswt-bogoliubov-transformation -->

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
\end{pmatrix}, \tag{1}
$$

where $\mathsf P_{\mathbf k}$ and $\mathsf Q_{\mathbf k}$ are $N_{\mathrm{sub}}\times N_{\mathrm{sub}}$ matrices. The first row reads $\hat a_{\mathbf k\mu}=\sum_n[(\mathsf P_{\mathbf k})_{\mu n}\hat b_{\mathbf kn}+(\mathsf Q_{-\mathbf k})_{\mu n}\hat b_{-\mathbf kn}^\dagger]$. The second row is the Hermitian conjugate of the first row at $-\mathbf k$, so the block structure of $\mathsf T_{\mathbf k}$ follows from the requirement that the lower components of $\hat\Psi_{\mathbf k}$ are the adjoints of the upper components of $\hat\Psi_{-\mathbf k}$.

Both spinors obey the same bosonic commutation relations. For the components of $\hat\Psi_{\mathbf k}$ these relations read $[(\hat\Psi_{\mathbf k})_p,(\hat\Psi_{\mathbf k}^\dagger)_q]=(\Sigma_3)_{pq}$, since the lower components are creation operators. Requiring the same relations for $\hat\Phi_{\mathbf k}$ and inserting the transformation gives $[(\hat\Psi_{\mathbf k})_p,(\hat\Psi_{\mathbf k}^\dagger)_q]=(\mathsf T_{\mathbf k}\Sigma_3\mathsf T_{\mathbf k}^\dagger)_{pq}$, so the transformation must satisfy

<!-- eq 2: eq-lswt-paraunitary-condition -->

$$
\mathsf T_{\mathbf k}\Sigma_3\mathsf T_{\mathbf k}^\dagger=\Sigma_3
\quad\Longleftrightarrow\quad
\mathsf T_{\mathbf k}^\dagger\Sigma_3\mathsf T_{\mathbf k}=\Sigma_3. \tag{2}
$$

In blocks, the first form contains the four conditions $\mathsf P_{\mathbf k}\mathsf P_{\mathbf k}^\dagger-\mathsf Q_{-\mathbf k}\mathsf Q_{-\mathbf k}^\dagger=\mathsf I$, $\mathsf P_{\mathbf k}\mathsf Q_{\mathbf k}^{\mathsf T}-\mathsf Q_{-\mathbf k}\mathsf P_{-\mathbf k}^{\mathsf T}=0$, and their conjugate counterparts. The two forms are equivalent because either one gives the inverse $\mathsf T_{\mathbf k}^{-1}=\Sigma_3\mathsf T_{\mathbf k}^\dagger\Sigma_3$. A unitary matrix preserves $\mathsf I$ instead of $\Sigma_3$; a paraunitary $\mathsf T_{\mathbf k}$ is unitary only when $\mathsf Q_{\mathbf k}=0$, that is, when $\hat H_2$ conserves the boson number.

## Diagonal Form and the Particle–Hole Structure of the Spectrum

We seek a paraunitary $\mathsf T_{\mathbf k}$ for which $\mathsf E_{\mathbf k}=\mathsf T_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}$ is diagonal. With the inverse above, this condition is equivalent to an eigenvalue problem:

$$
\mathsf T_{\mathbf k}^{-1}\,\Sigma_3\mathsf H_{\mathbf k}\,\mathsf T_{\mathbf k}
=\Sigma_3\mathsf T_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}
=\Sigma_3\mathsf E_{\mathbf k}.
$$

The columns of $\mathsf T_{\mathbf k}$ are therefore eigenvectors of the non-Hermitian matrix $\Sigma_3\mathsf H_{\mathbf k}$, and the signed diagonal entries $\lambda_{p\mathbf k}=(\Sigma_3)_{pp}(\mathsf E_{\mathbf k})_{pp}$ are its eigenvalues.

The BdG matrix relates $\mathbf k$ and $-\mathbf k$. With $\Sigma_1=\sigma_1\otimes\mathsf I_{N_{\mathrm{sub}}}$, which exchanges the particle and hole blocks, the block relations $\mathsf A_{\mathbf k}^\dagger=\mathsf A_{\mathbf k}$ and $\mathsf B_{-\mathbf k}=\mathsf B_{\mathbf k}^{\mathsf T}$ give

<!-- eq 3: eq-lswt-bdg-particle-hole-symmetry -->

$$
\Sigma_1\mathsf H_{-\mathbf k}^*\Sigma_1=\mathsf H_{\mathbf k}. \tag{3}
$$

Since $\Sigma_3\Sigma_1=-\Sigma_1\Sigma_3$, an eigenvector $\mathbf v$ of $\Sigma_3\mathsf H_{-\mathbf k}$ with eigenvalue $\lambda$ yields the eigenvector $\Sigma_1\mathbf v^*$ of $\Sigma_3\mathsf H_{\mathbf k}$ with eigenvalue $-\lambda^*$. For a positive-definite $\mathsf H_{\mathbf k}$, the section on Colpa’s construction shows that the eigenvalues are real, $N_{\mathrm{sub}}$ of them positive and $N_{\mathrm{sub}}$ negative. We denote the positive eigenvalues at $\mathbf k$ by the magnon energies $\varepsilon_{n\mathbf k}$; the negative eigenvalues at $\mathbf k$ are then $-\varepsilon_{n,-\mathbf k}$. Ordering the positive eigenvalues first, and the negative eigenvalues in the band order at $-\mathbf k$, the diagonal form is

<!-- eq 4: eq-lswt-bdg-diagonal-form -->

$$
\mathsf T_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}
=\operatorname{diag}\!\left(\varepsilon_{1\mathbf k},\ldots,\varepsilon_{N_{\mathrm{sub}}\mathbf k},\varepsilon_{1,-\mathbf k},\ldots,\varepsilon_{N_{\mathrm{sub}},-\mathbf k}\right). \tag{4}
$$

The hole columns of $\mathsf T_{\mathbf k}$ can be chosen as $\Sigma_1$ times the complex conjugates of the particle columns of $\mathsf T_{-\mathbf k}$, which reproduces the block structure of (1). The energies $\varepsilon_{n\mathbf k}$ and $\varepsilon_{n,-\mathbf k}$ need not coincide; they coincide when a symmetry of the model and the ordered state maps $\mathbf k$ to $-\mathbf k$.

## Diagonal Magnon Hamiltonian and Zero-Point Correction

Inserting $\hat\Psi_{\mathbf k}=\mathsf T_{\mathbf k}\hat\Phi_{\mathbf k}$ into $\hat H_2=\frac12\sum_{\mathbf k}\big(\hat\Psi_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\hat\Psi_{\mathbf k}-\operatorname{Tr}\mathsf A_{\mathbf k}\big)$ and using (4) gives

$$
\hat H_2
=\frac12\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}
\left(\varepsilon_{n\mathbf k}\hat b_{\mathbf kn}^\dagger\hat b_{\mathbf kn}
+\varepsilon_{n,-\mathbf k}\hat b_{-\mathbf kn}\hat b_{-\mathbf kn}^\dagger\right)
-\frac12\sum_{\mathbf k\in\mathrm{MBZ}}\operatorname{Tr}\mathsf A_{\mathbf k}.
$$

The commutator $\hat b_{-\mathbf kn}\hat b_{-\mathbf kn}^\dagger=\hat b_{-\mathbf kn}^\dagger\hat b_{-\mathbf kn}+1$ and the relabeling $-\mathbf k\to\mathbf k$, allowed because the $N_{\mathrm{uc}}$ momenta of the MBZ form a set closed under inversion modulo reciprocal-lattice vectors, turn this expression into

<!-- eq 5: eq-lswt-diagonal-magnon-hamiltonian -->

$$
\hat H_2
=\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}\varepsilon_{n\mathbf k}\,\hat b_{\mathbf kn}^\dagger\hat b_{\mathbf kn}
+\Delta E_{\mathrm{zp}},
\qquad
\Delta E_{\mathrm{zp}}
=\frac12\sum_{\mathbf k\in\mathrm{MBZ}}\left(\sum_{n=1}^{N_{\mathrm{sub}}}\varepsilon_{n\mathbf k}-\operatorname{Tr}\mathsf A_{\mathbf k}\right). \tag{5}
$$

Each magnon mode $(\mathbf k,n)$ is an independent harmonic oscillator of energy $\varepsilon_{n\mathbf k}$. The magnon vacuum, annihilated by every $\hat b_{\mathbf kn}$, is the ground state of $\hat H_2$, and the LSWT ground-state energy is

<!-- eq 6: eq-lswt-zero-point-correction -->

$$
E_{\mathrm{GS}}=E_{\mathrm{cl}}+\Delta E_{\mathrm{zp}},
\qquad
\mathcal E_{\mathrm{GS}}=\frac{E_{\mathrm{GS}}}{N_{\mathrm{site}}},
\qquad
N_{\mathrm{site}}=N_{\mathrm{uc}}N_{\mathrm{sub}}. \tag{6}
$$

The zero-point correction $\Delta E_{\mathrm{zp}}$ is the energy of the magnon vacuum relative to the classical energy $E_{\mathrm{cl}}$; it combines the zero-point energies $\frac12\varepsilon_{n\mathbf k}$ of the oscillators with the trace subtraction in $\hat H_2$. The trace subtraction removes the constant $\frac12\sum_{\mathbf k}\operatorname{Tr}\mathsf A_{-\mathbf k}$ that appears when the normal-ordered quadratic Hamiltonian is rewritten in the symmetric Nambu form, which commutes the hole components $\hat a_{-\mathbf k}\hat a_{-\mathbf k}^\dagger$; replacing $\mathsf A_{-\mathbf k}$ by $\mathsf A_{\mathbf k}$ in this constant uses the same inversion closure as above. The real-space quadratic Hamiltonian is normal ordered and has no constant of its own, so this subtraction is exact; Trace Subtraction in Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian* shows this, and $E_{\mathrm{cl}}$ there includes the coherent-state single-ion energy. Because $\operatorname{Tr}\mathsf A_{\mathbf k}$ is real and the momentum set is closed under inversion, $\sum_{\mathbf k}\operatorname{Tr}\mathsf A_{\mathbf k}=\frac12\sum_{\mathbf k}\operatorname{Tr}\mathsf H_{\mathbf k}$, so the correction can equally be written as $\frac12\sum_{\mathbf k}\sum_n\varepsilon_{n\mathbf k}-\frac14\sum_{\mathbf k}\operatorname{Tr}\mathsf H_{\mathbf k}$. The two forms agree after the momentum sum; their summands need not agree at a single momentum.

## Colpa’s Construction for a Positive-Definite BdG Matrix

A paraunitary $\mathsf T_{\mathbf k}$ that brings $\mathsf H_{\mathbf k}$ to a diagonal form with positive entries exists if and only if $\mathsf H_{\mathbf k}$ is positive definite. If such a $\mathsf T_{\mathbf k}$ exists, $\mathsf H_{\mathbf k}=(\mathsf T_{\mathbf k}^{-1})^\dagger\mathsf E_{\mathbf k}\mathsf T_{\mathbf k}^{-1}$ is congruent to a positive diagonal matrix and is therefore positive definite. Conversely, Colpa’s construction produces $\mathsf T_{\mathbf k}$ from a positive-definite $\mathsf H_{\mathbf k}$. We suppress the momentum index in the rest of this section.

A positive-definite $\mathsf H$ has a Cholesky factorization $\mathsf H=\mathsf K\mathsf K^\dagger$ with an invertible lower-triangular $\mathsf K$. The Hermitian matrix $\mathsf K^\dagger\Sigma_3\mathsf K$ is congruent to $\Sigma_3$, so by Sylvester’s law of inertia it has $N_{\mathrm{sub}}$ positive and $N_{\mathrm{sub}}$ negative eigenvalues. We diagonalize it with a unitary matrix $\mathsf V$,

$$
\mathsf V\Lambda\mathsf V^\dagger=\mathsf K^\dagger\Sigma_3\mathsf K,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_{2N_{\mathrm{sub}}}),
$$

ordering the eigenvalues so that the positive ones come first. The diagonal matrix $\Lambda\Sigma_3$ then has only positive entries, and the transformation is

<!-- eq 7: eq-lswt-colpa-transformation -->

$$
\mathsf T=(\mathsf K^\dagger)^{-1}\mathsf V(\Lambda\Sigma_3)^{1/2}. \tag{7}
$$

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

The eigenvalues $\lambda_p$ are those of $\Sigma_3\mathsf H$, because $\Sigma_3\mathsf H=\Sigma_3\mathsf K\mathsf K^\dagger$ is similar to $\mathsf K^\dagger\Sigma_3\mathsf K$ through $\mathsf K^\dagger$. They are the signed energies of the previous section, $\varepsilon_{n\mathbf k}$ on the particle columns and $-\varepsilon_{n,-\mathbf k}$ on the hole columns, and $\Lambda\Sigma_3$ is the diagonal matrix of (4). This also shows that the eigenvalues of $\Sigma_3\mathsf H_{\mathbf k}$ are real when $\mathsf H_{\mathbf k}$ is positive definite.

The construction fixes $\mathsf T_{\mathbf k}$ only up to $\mathsf T_{\mathbf k}\to\mathsf T_{\mathbf k}\mathsf U$, where $\mathsf U$ is paraunitary and commutes with $\Sigma_3\mathsf E_{\mathbf k}$. Such a $\mathsf U$ is block diagonal in the eigenspaces of the signed energies, which do not mix particle and hole columns when all energies are positive, and is unitary within each block. It therefore multiplies each column by a phase and mixes columns only within degenerate bands. The energies and $\Delta E_{\mathrm{zp}}$ do not depend on this freedom; the Berry connection of Topological Magnon Quantities in research note *LSWT magnon topology* does. Hole columns obtained independently at $\mathbf k$ agree with the particle–hole partners of the particle columns at $-\mathbf k$ up to the same freedom.

## Positive-Semidefinite and Indefinite BdG Matrices

The quadratic form $\hat\Psi_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\hat\Psi_{\mathbf k}$ is the harmonic energy of the spin fluctuations about the classical reference configuration of Classical Order and Local Frame in research note *LSWT classical order*. Positive definiteness of $\mathsf H_{\mathbf k}$ at every momentum means that every harmonic fluctuation raises the energy. If $\mathsf H_{\mathbf k}$ is indefinite at some momentum, a fluctuation lowers the energy, the reference configuration is not a local minimum, and the magnon vacuum is not the ground state of $\hat H_2$. The Cholesky factorization then fails. The matrix $\Sigma_3\mathsf H_{\mathbf k}$ may have complex eigenvalues, which signal a dynamical instability, or real eigenvalues with a mode of negative energy, as for a ferromagnet polarized against the field; in either case the harmonic expansion does not describe the equilibrium state.

A positive-semidefinite $\mathsf H_{\mathbf k}$ has zero modes, for example Goldstone modes of a broken continuous symmetry. Its Cholesky factor is singular, and Colpa’s construction does not apply. A single mode with real blocks $\mathsf A=A$ and $\mathsf B=B$, for which $\varepsilon=\sqrt{A^2-B^2}$, shows the two possible behaviors at $\varepsilon=0$:

- For $A=B>0$, $\mathsf H$ is positive semidefinite and $\Sigma_3\mathsf H=A\begin{pmatrix}1&1\\-1&-1\end{pmatrix}$ satisfies $(\Sigma_3\mathsf H)^2=0$. The matrix is not diagonalizable, and no paraunitary $\mathsf T$ exists. The Hamiltonian $\frac A2(\hat a+\hat a^\dagger)^2$ contains one canonical coordinate but not its conjugate momentum, so the mode is not a harmonic oscillator. A Goldstone mode of an antiferromagnet at $\mathbf k=0$ has this form.

- For $A=B=0$, $\Sigma_3\mathsf H=0$ is diagonalizable, $\mathsf T=\mathsf I$ is paraunitary, and the mode is a boson of zero energy. The Goldstone mode of a Heisenberg ferromagnet at $\mathbf k=0$ and zero field has this form.

Adding a small positive shift $\delta\,\mathsf I_{2N_{\mathrm{sub}}}$ makes $\mathsf H_{\mathbf k}$ positive definite. The energies then converge as $\delta\to0$, but in the first case the entries of $\mathsf T$ grow as $\delta^{-1/4}$, since $\varepsilon=\sqrt{2A\delta+\delta^2}$ while the diagonal entries of $\mathsf T$ scale as $(A/\varepsilon)^{1/2}$. Zero modes usually occur at isolated momenta, where $\varepsilon_{n\mathbf k}\to0$ continuously; they do not change $\Delta E_{\mathrm{zp}}$. Quantities weighted by $1/\varepsilon_{n\mathbf k}$ or by the Bose–Einstein distribution, such as magnon numbers and correlation functions, require separate treatment and are discussed in the observable documents. A complete diagonalization theory for positive-semidefinite quadratic boson Hamiltonians, which brings them to a standard form that includes such zero modes, is given by Colpa (1986).

## References

### Internal Documents

- Notation and Conventions in research note *LSWT spin Hamiltonian*: defines the Nambu spinors $\hat\Psi_{\mathbf k}$ and $\hat\Phi_{\mathbf k}$, the metric $\Sigma_3$, the paraunitary matrix $\mathsf T_{\mathbf k}$, and the energy symbols $\varepsilon_{n\mathbf k}$, $E_{\mathrm{cl}}$, $\Delta E_{\mathrm{zp}}$, and $E_{\mathrm{GS}}$.

- Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*: supplies $\hat H_2$, the BdG blocks $\mathsf A_{\mathbf k}$ and $\mathsf B_{\mathbf k}$, their block relations, and the trace subtraction, which it shows to be exact.

- Classical Order and Local Frame in research note *LSWT classical order*: defines the classical reference configuration about which $\mathsf H_{\mathbf k}$ is expanded.

- Paraunitarity Proofs (Section *Appendix: Paraunitarity Proofs*): is reserved for extended metric-preservation proofs.

- Magnon Observables in research note *LSWT magnon thermodynamics*: uses the magnon energies and the transformation $\mathsf T_{\mathbf k}$.

- Topological Magnon Quantities in research note *LSWT magnon topology*: uses the columns of $\mathsf T_{\mathbf k}$ and their gauge freedom in the Berry curvature.

### External Sources

- J. H. P. Colpa, “Diagonalization of the Quadratic Boson Hamiltonian,” *Physica A* **93**, 327–353 (1978), [doi:10.1016/0378-4371(78)90160-7](https://doi.org/10.1016/0378-4371(78)90160-7): paraunitary diagonalization of a positive-definite quadratic boson Hamiltonian through the Cholesky factorization; the construction of (7).

- J. H. P. Colpa, “Diagonalization of the Quadratic Boson Hamiltonian with Zero Modes,” *Physica A* **134**, 377–416 (1986), [doi:10.1016/0378-4371(86)90056-7](https://doi.org/10.1016/0378-4371(86)90056-7), and **134**, 417–442 (1986), [doi:10.1016/0378-4371(86)90057-9](https://doi.org/10.1016/0378-4371(86)90057-9): standard form of positive-semidefinite quadratic boson Hamiltonians with zero modes.

<!-- Worked Example Converted on 2026-10-08 from docs/archive/2026-10-08-lswt-markdown/03-examples/worked-example.md (status: draft, source-section: "Example: Solving spin system with linear spin wave theory (source pp. 25–27)"). Review status: draft; awaiting the user's physics and mathematics review. -->

# Worked Example

A single magnetic sublattice, $N_{\mathrm{sub}}=1$, gives the smallest quadratic boson Hamiltonian that contains every ingredient of the general LSWT problem: a normal coefficient, an anomalous pairing coefficient, a paraunitary Bogoliubov transformation, a zero-point correction, and a quantum reduction of the boson number. Every quantity can be written in closed form. The example follows the steps of Paraunitary Diagonalization (Section *Paraunitary Diagonalization*) and Magnon Observables in research note *LSWT magnon thermodynamics* for this case, so the general formulas can be checked against explicit expressions. We allow the normal coefficients at $\mathbf k$ and $-\mathbf k$ to differ, which shows how a nonreciprocal spectrum $\varepsilon_{\mathbf k}\ne\varepsilon_{-\mathbf k}$ arises.

## Single-Sublattice BdG Hamiltonian

For $N_{\mathrm{sub}}=1$ the blocks $\mathsf A_{\mathbf k}$ and $\mathsf B_{\mathbf k}$ of Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian* are numbers. The block relations reduce to a real normal coefficient $A_{\mathbf k}$ and an even anomalous coefficient, $B_{-\mathbf k}=B_{\mathbf k}$. We drop the sublattice index and write the quadratic Hamiltonian in normal-ordered form,

$$
\hat H_2
=\sum_{\mathbf k\in\mathrm{MBZ}}\left[A_{\mathbf k}\,\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}
+\frac12\left(B_{\mathbf k}\,\hat a_{\mathbf k}^\dagger\hat a_{-\mathbf k}^\dagger+B_{\mathbf k}^*\,\hat a_{-\mathbf k}\hat a_{\mathbf k}\right)\right].
$$

Symmetrizing the normal term over $\mathbf k$ and $-\mathbf k$ produces the commutator $\hat a_{-\mathbf k}\hat a_{-\mathbf k}^\dagger=\hat a_{-\mathbf k}^\dagger\hat a_{-\mathbf k}+1$, and the Nambu form therefore carries the constant of the general BdG expression:

$$
\hat H_2=\frac12\sum_{\mathbf k\in\mathrm{MBZ}}\left[
\begin{pmatrix}\hat a_{\mathbf k}^\dagger & \hat a_{-\mathbf k}\end{pmatrix}
\mathsf H_{\mathbf k}
\begin{pmatrix}\hat a_{\mathbf k}\\ \hat a_{-\mathbf k}^\dagger\end{pmatrix}
-A_{\mathbf k}\right],
\qquad
\mathsf H_{\mathbf k}=\begin{pmatrix}A_{\mathbf k} & B_{\mathbf k}\\ B_{\mathbf k}^* & A_{-\mathbf k}\end{pmatrix}.
$$

The relabeling of $-\mathbf k$ to $\mathbf k$ in this step uses that the MBZ momenta form a set closed under inversion. The momentum $\mathbf k$ and its partner $-\mathbf k$ enter only through the even and odd parts of the normal coefficient,

$$
A_{\mathbf k}^{+}=\tfrac12\left(A_{\mathbf k}+A_{-\mathbf k}\right),
\qquad
A_{\mathbf k}^{-}=\tfrac12\left(A_{\mathbf k}-A_{-\mathbf k}\right),
\qquad
\mathsf H_{\mathbf k}=A_{\mathbf k}^{-}\Sigma_3+\begin{pmatrix}A_{\mathbf k}^{+} & B_{\mathbf k}\\ B_{\mathbf k}^* & A_{\mathbf k}^{+}\end{pmatrix},
$$

where $\Sigma_3=\operatorname{diag}(1,-1)$ is the Nambu metric. The odd part $A_{\mathbf k}^{-}$ is nonzero only when no symmetry of the model and the reference configuration maps $\mathbf k$ to $-\mathbf k$, for example when a Dzyaloshinskii–Moriya interaction has a component along the ordered moment of a ferromagnet.

## Stability Condition and Magnon Energies

The BdG matrix is positive definite when both diagonal entries and the determinant are positive:

<!-- eq 8: eq-lswt-example-stability -->

$$
A_{\mathbf k}>0,
\qquad
A_{-\mathbf k}>0,
\qquad
A_{\mathbf k}A_{-\mathbf k}>|B_{\mathbf k}|^2. \tag{8}
$$

For $A_{\mathbf k}^{-}=0$ these conditions reduce to $A_{\mathbf k}>|B_{\mathbf k}|$. The weaker statement $|A_{\mathbf k}|\ge|B_{\mathbf k}|$ is not sufficient: for $A_{\mathbf k}<-|B_{\mathbf k}|$ the matrix is negative definite and the magnon vacuum is the highest, not the lowest, state of the harmonic Hamiltonian, and equality gives a zero mode, which is treated in the last section.

The magnon energies are the eigenvalues of $\Sigma_3\mathsf H_{\mathbf k}$. The term $A_{\mathbf k}^{-}\Sigma_3$ contributes $A_{\mathbf k}^{-}$ times the identity to $\Sigma_3\mathsf H_{\mathbf k}$, so it shifts both eigenvalues equally, and the remaining matrix has the eigenvalues $\pm\omega_{\mathbf k}$ with

<!-- eq 9: eq-lswt-example-energies -->

$$
\omega_{\mathbf k}=\sqrt{\big(A_{\mathbf k}^{+}\big)^2-|B_{\mathbf k}|^2},
\qquad
\varepsilon_{\mathbf k}=\omega_{\mathbf k}+A_{\mathbf k}^{-},
\qquad
\varepsilon_{-\mathbf k}=\omega_{\mathbf k}-A_{\mathbf k}^{-}. \tag{9}
$$

The eigenvalues of $\Sigma_3\mathsf H_{\mathbf k}$ are $\varepsilon_{\mathbf k}$ and $-\varepsilon_{-\mathbf k}$, which is the particle–hole structure of Paraunitary Diagonalization (Section *Paraunitary Diagonalization*) for one band. Since $\omega_{-\mathbf k}=\omega_{\mathbf k}$ and $A_{-\mathbf k}^{-}=-A_{\mathbf k}^{-}$, the expression for $\varepsilon_{-\mathbf k}$ is the same formula evaluated at $-\mathbf k$. The odd part of the normal coefficient thus makes the spectrum nonreciprocal without changing $\omega_{\mathbf k}$. Under (8) both energies are positive, because $\omega_{\mathbf k}^2-(A_{\mathbf k}^{-})^2=A_{\mathbf k}A_{-\mathbf k}-|B_{\mathbf k}|^2>0$.

The eigenvalues remain real when $A_{\mathbf k}^{+}>|B_{\mathbf k}|$ but $A_{\mathbf k}A_{-\mathbf k}<|B_{\mathbf k}|^2$. In that case one of $\varepsilon_{\pm\mathbf k}$ is negative, $\mathsf H_{\mathbf k}$ is indefinite, and creating that magnon lowers the energy, so the magnon vacuum is not the ground state of $\hat H_2$. Real eigenvalues of $\Sigma_3\mathsf H_{\mathbf k}$ therefore do not by themselves establish that the reference configuration is a local energy minimum; positive definiteness of $\mathsf H_{\mathbf k}$ does.

## Bogoliubov Transformation

We write the anomalous coefficient as $B_{\mathbf k}=|B_{\mathbf k}|\exp(\mathrm i\phi_{\mathbf k})$, with $\phi_{-\mathbf k}=\phi_{\mathbf k}$, and define the Bogoliubov angle $\theta_{\mathbf k}\ge0$ by

$$
\cosh2\theta_{\mathbf k}=\frac{A_{\mathbf k}^{+}}{\omega_{\mathbf k}},
\qquad
\sinh2\theta_{\mathbf k}=\frac{|B_{\mathbf k}|}{\omega_{\mathbf k}},
\qquad
\tanh2\theta_{\mathbf k}=\frac{|B_{\mathbf k}|}{A_{\mathbf k}^{+}}.
$$

The angle is even in $\mathbf k$ because $A_{\mathbf k}^{+}$ and $|B_{\mathbf k}|$ are even. The transformation $\hat\Psi_{\mathbf k}=\mathsf T_{\mathbf k}\hat\Phi_{\mathbf k}$ to the magnon operators $\hat b_{\mathbf k}$ is

<!-- eq 10: eq-lswt-example-bogoliubov -->

$$
\begin{pmatrix}\hat a_{\mathbf k}\\ \hat a_{-\mathbf k}^\dagger\end{pmatrix}
=\begin{pmatrix}
\cosh\theta_{\mathbf k} & -\exp(\mathrm i\phi_{\mathbf k})\sinh\theta_{\mathbf k}\\
-\exp(-\mathrm i\phi_{\mathbf k})\sinh\theta_{\mathbf k} & \cosh\theta_{\mathbf k}
\end{pmatrix}
\begin{pmatrix}\hat b_{\mathbf k}\\ \hat b_{-\mathbf k}^\dagger\end{pmatrix}. \tag{10}
$$

In the notation of the general transformation, $\mathsf P_{\mathbf k}=\cosh\theta_{\mathbf k}$ and $\mathsf Q_{\mathbf k}=-\exp(\mathrm i\phi_{\mathbf k})\sinh\theta_{\mathbf k}$; the lower row is the Hermitian conjugate of the upper row at $-\mathbf k$ because $\theta_{\mathbf k}$ and $\phi_{\mathbf k}$ are even. The paraunitary condition $\mathsf T_{\mathbf k}\Sigma_3\mathsf T_{\mathbf k}^\dagger=\Sigma_3$ reduces to $\cosh^2\theta_{\mathbf k}-\sinh^2\theta_{\mathbf k}=1$ on the diagonal, and its off-diagonal entries cancel. The commutator $[\hat a_{\mathbf k},\hat a_{\mathbf k}^\dagger]=1$ therefore holds when it holds for the magnon operators.

The phase $\phi_{\mathbf k}$ can be removed by redefining $\hat a_{\mathbf k}\to\exp(\mathrm i\phi_{\mathbf k}/2)\hat a_{\mathbf k}$, which makes $B_{\mathbf k}$ real and nonnegative. Keeping the phase explicit shows where it appears in the anomalous average below.

## Diagonal Hamiltonian

Only the second matrix of the decomposition of $\mathsf H_{\mathbf k}$ needs to be diagonalized, since $\mathsf T_{\mathbf k}^\dagger\Sigma_3\mathsf T_{\mathbf k}=\Sigma_3$ leaves the term $A_{\mathbf k}^{-}\Sigma_3$ unchanged. For that matrix the transformation gives

$$
\mathsf T_{\mathbf k}^\dagger
\begin{pmatrix}A_{\mathbf k}^{+} & B_{\mathbf k}\\ B_{\mathbf k}^* & A_{\mathbf k}^{+}\end{pmatrix}
\mathsf T_{\mathbf k}
=\begin{pmatrix}
A_{\mathbf k}^{+}\cosh2\theta_{\mathbf k}-|B_{\mathbf k}|\sinh2\theta_{\mathbf k} & \exp(\mathrm i\phi_{\mathbf k})\big(|B_{\mathbf k}|\cosh2\theta_{\mathbf k}-A_{\mathbf k}^{+}\sinh2\theta_{\mathbf k}\big)\\
\exp(-\mathrm i\phi_{\mathbf k})\big(|B_{\mathbf k}|\cosh2\theta_{\mathbf k}-A_{\mathbf k}^{+}\sinh2\theta_{\mathbf k}\big) & A_{\mathbf k}^{+}\cosh2\theta_{\mathbf k}-|B_{\mathbf k}|\sinh2\theta_{\mathbf k}
\end{pmatrix}.
$$

The off-diagonal entries vanish for the angle $\tanh2\theta_{\mathbf k}=|B_{\mathbf k}|/A_{\mathbf k}^{+}$, and the diagonal entries equal $\big[(A_{\mathbf k}^{+})^2-|B_{\mathbf k}|^2\big]/\omega_{\mathbf k}=\omega_{\mathbf k}$. Adding back $A_{\mathbf k}^{-}\Sigma_3$ gives $\mathsf T_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}=\operatorname{diag}(\varepsilon_{\mathbf k},\varepsilon_{-\mathbf k})$, in agreement with (9). The diagonal magnon Hamiltonian and its zero-point correction follow from Paraunitary Diagonalization (Section *Paraunitary Diagonalization*):

<!-- eq 11: eq-lswt-example-zero-point -->

$$
\hat H_2=\sum_{\mathbf k\in\mathrm{MBZ}}\varepsilon_{\mathbf k}\,\hat b_{\mathbf k}^\dagger\hat b_{\mathbf k}+\Delta E_{\mathrm{zp}},
\qquad
\Delta E_{\mathrm{zp}}=\frac12\sum_{\mathbf k\in\mathrm{MBZ}}\left(\varepsilon_{\mathbf k}-A_{\mathbf k}\right)
=\frac12\sum_{\mathbf k\in\mathrm{MBZ}}\left(\omega_{\mathbf k}-A_{\mathbf k}^{+}\right). \tag{11}
$$

The second form follows because $A_{\mathbf k}^{-}$ is odd and cancels in the sum over a set closed under inversion. Each summand $\omega_{\mathbf k}-A_{\mathbf k}^{+}$ is negative when $B_{\mathbf k}\ne0$, so the anomalous terms lower the energy below the classical value; the nonreciprocal part does not change the zero-point correction. For $B_{\mathbf k}=0$ the angle vanishes, $\mathsf T_{\mathbf k}$ is the identity, and $\Delta E_{\mathrm{zp}}=0$: the magnon vacuum coincides with the boson vacuum.

## Boson Number and Anomalous Average

Inserting (10) into the averages and using the Gibbs occupations $n_{\mathbf k}=n_{\mathrm B}(\varepsilon_{\mathbf k})$ of Thermodynamics in research note *LSWT magnon thermodynamics*, with $\langle\hat b_{\mathbf k}^\dagger\hat b_{\mathbf k}\rangle=n_{\mathbf k}$ and vanishing averages of $\hat b\hat b$ and of products at different momenta, gives

<!-- eq 12: eq-lswt-example-averages -->

$$
\begin{aligned}
\langle\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}\rangle
&=\cosh^2\theta_{\mathbf k}\,n_{\mathbf k}+\sinh^2\theta_{\mathbf k}\,\big(1+n_{-\mathbf k}\big),\\
\langle\hat a_{\mathbf k}\hat a_{-\mathbf k}\rangle
&=-\exp(\mathrm i\phi_{\mathbf k})\cosh\theta_{\mathbf k}\sinh\theta_{\mathbf k}\,\big(1+n_{\mathbf k}+n_{-\mathbf k}\big).
\end{aligned} \tag{12}
$$

The first line is the single-band form of the boson number in Magnon Observables in research note *LSWT magnon thermodynamics*. In the magnon vacuum the occupations vanish, and

$$
\langle\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}\rangle_0=\sinh^2\theta_{\mathbf k}=\frac12\left(\frac{A_{\mathbf k}^{+}}{\omega_{\mathbf k}}-1\right),
\qquad
\langle\hat a_{\mathbf k}\hat a_{-\mathbf k}\rangle_0=-\frac{B_{\mathbf k}}{2\omega_{\mathbf k}}.
$$

The zero-temperature boson number is the quantum reduction of the ordered moment. It is nonzero only when $B_{\mathbf k}\ne0$, that is, when the harmonic Hamiltonian does not conserve the number of Holstein–Primakoff bosons. The anomalous average carries the phase of $B_{\mathbf k}$ and is the off-diagonal block of the correlation matrix $\mathsf T_{\mathbf k}\mathsf N_{\mathbf k}(0)\mathsf T_{\mathbf k}^\dagger$.

## Response to a Uniform Shift of the Normal Coefficient

The zero-point correction and the boson number are related by the Hellmann–Feynman theorem. Adding $\mu\sum_{\mathbf k}\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}$ to $\hat H_2$ shifts $A_{\mathbf k}\to A_{\mathbf k}+\mu$ and leaves $B_{\mathbf k}$ unchanged. The derivative of the ground-state energy with respect to $\mu$ is the ground-state average of the added operator, and (11) gives

$$
\frac{\partial\Delta E_{\mathrm{zp}}}{\partial\mu}
=\frac12\sum_{\mathbf k\in\mathrm{MBZ}}\left(\frac{A_{\mathbf k}^{+}+\mu}{\omega_{\mathbf k}(\mu)}-1\right)
=\sum_{\mathbf k\in\mathrm{MBZ}}\langle\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}\rangle_0,
\qquad
\omega_{\mathbf k}(\mu)=\sqrt{\big(A_{\mathbf k}^{+}+\mu\big)^2-|B_{\mathbf k}|^2}.
$$

This identity provides an independent check of the zero-temperature boson number: the boson number can be obtained either from the transformation $\mathsf T_{\mathbf k}$ or from the energies alone. To second order in $\mu$,

$$
\Delta E_{\mathrm{zp}}(\mu)=\Delta E_{\mathrm{zp}}(0)+\mu\sum_{\mathbf k}\langle\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}\rangle_0-\frac{\mu^2}{4}\sum_{\mathbf k}\frac{|B_{\mathbf k}|^2}{\omega_{\mathbf k}^3}+O(\mu^3),
\qquad
\frac{\partial\langle\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}\rangle_0}{\partial\mu}=-\frac{|B_{\mathbf k}|^2}{2\omega_{\mathbf k}^3},
$$

with $\omega_{\mathbf k}=\omega_{\mathbf k}(0)$. The second-order coefficient is negative, as second-order perturbation theory requires for the ground-state energy, and the boson number decreases when the normal coefficient grows relative to the pairing.

## Approach to a Zero Mode

The stability condition becomes marginal when $A_{\mathbf k}A_{-\mathbf k}\to|B_{\mathbf k}|^2$. For $A_{\mathbf k}^{-}=0$ this is the limit $\omega_{\mathbf k}\to0^+$ at fixed $A_{\mathbf k}^{+}>0$, which is the single-mode Goldstone case $A=B>0$ of Paraunitary Diagonalization (Section *Paraunitary Diagonalization*). In this limit $\cosh\theta_{\mathbf k}$ and $\sinh\theta_{\mathbf k}$ grow as $(A_{\mathbf k}^{+}/2\omega_{\mathbf k})^{1/2}$, and

$$
\langle\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}\rangle_0\simeq\frac{A_{\mathbf k}^{+}}{2\omega_{\mathbf k}},
\qquad
\langle\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}\rangle\simeq\frac{A_{\mathbf k}^{+}}{\omega_{\mathbf k}}\,n_{\mathrm B}(\omega_{\mathbf k})\simeq\frac{A_{\mathbf k}^{+}k_{\mathrm B}T}{\omega_{\mathbf k}^2}
\quad(\omega_{\mathbf k}\ll k_{\mathrm B}T).
$$

The occupation per mode diverges at the zero mode, while the energy per mode, $\frac12(\omega_{\mathbf k}-A_{\mathbf k}^{+})$, stays finite. Whether the momentum sum of the boson number converges then depends on the dispersion near the zero and on the dimension, as discussed in Magnon Observables in research note *LSWT magnon thermodynamics*. At $\omega_{\mathbf k}=0$ itself the transformation does not exist, and the single-mode boson number is undefined rather than large.

## References

### Internal Documents

- Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*: defines the Nambu spinor, the BdG blocks, their block relations, and the trace subtraction specialized here.

- Paraunitary Diagonalization (Section *Paraunitary Diagonalization*): supplies the paraunitary condition, the particle–hole structure, the zero-point correction, and the single-mode zero-mode cases.

- Magnon Observables in research note *LSWT magnon thermodynamics*: defines the correlation matrix and the boson number whose single-band forms are evaluated here.

- Thermodynamics in research note *LSWT magnon thermodynamics*: defines the Bose–Einstein occupations.

- Notation and Conventions in research note *LSWT spin Hamiltonian*: defines $\Sigma_3$, $\mathsf T_{\mathbf k}$, $\hat\Phi_{\mathbf k}$, and the energy symbols.

### External Sources

- None.

<!-- Appendix: Paraunitarity Proofs Converted on 2026-10-08 from docs/archive/2026-10-08-lswt-markdown/04-appendices/paraunitarity-proofs.md (status: draft, source-section: "Paraunitary Diagonalization (source pp. 10–11); restructured TeX appendix 'Para-unitarity proofs'"). Review status: draft; awaiting the user's physics and mathematics review. -->

# Appendix: Paraunitarity Proofs

The diagonalization of a bosonic Bogoliubov–de Gennes (BdG) matrix rests on a few algebraic facts about the indefinite metric $\Sigma_3=\sigma_3\otimes\mathsf I_{N_{\mathrm{sub}}}$: paraunitary matrices form a group, the eigenvectors of $\Sigma_3\mathsf H$ are orthogonal with respect to $\Sigma_3$, and positive definiteness of $\mathsf H$ fixes both the reality of the spectrum and the sign of each eigenvector’s $\Sigma_3$-norm. This appendix proves these facts and the properties of the transformation $\mathsf T_{\mathbf k}$ used in Paraunitary Diagonalization (Section *Paraunitary Diagonalization*). Throughout, $\mathsf H$ is a Hermitian $2N_{\mathrm{sub}}\times2N_{\mathrm{sub}}$ matrix, and the momentum index is suppressed where only one momentum enters.

## Paraunitary Matrices Form a Group

A matrix $\mathsf T$ is paraunitary when $\mathsf T\Sigma_3\mathsf T^\dagger=\Sigma_3$. Taking determinants gives $|\det\mathsf T|^2=1$, so $\mathsf T$ is invertible. Multiplying the condition by $\Sigma_3$ on the right gives $\mathsf T(\Sigma_3\mathsf T^\dagger\Sigma_3)=\mathsf I$, so

$$
\mathsf T^{-1}=\Sigma_3\mathsf T^\dagger\Sigma_3.
$$

Inserting this inverse into $\mathsf T^{-1}\mathsf T=\mathsf I$ gives $\Sigma_3\mathsf T^\dagger\Sigma_3\mathsf T=\mathsf I$, which is the second form $\mathsf T^\dagger\Sigma_3\mathsf T=\Sigma_3$. The argument runs equally in the other direction, so the two forms of the paraunitary condition are equivalent. If $\mathsf T_1$ and $\mathsf T_2$ are paraunitary, then $\mathsf T_1\mathsf T_2\Sigma_3\mathsf T_2^\dagger\mathsf T_1^\dagger=\mathsf T_1\Sigma_3\mathsf T_1^\dagger=\Sigma_3$, and the inverse $\Sigma_3\mathsf T_1^\dagger\Sigma_3$ satisfies the second form of the condition. Paraunitary matrices therefore form a group, the pseudo-unitary group $\mathrm U(N_{\mathrm{sub}},N_{\mathrm{sub}})$. A paraunitary matrix that is also unitary commutes with $\Sigma_3$ and is block diagonal in the particle and hole components.

## Metric Orthogonality of Eigenvectors

Let $\Sigma_3\mathsf H\mathbf v=\lambda\mathbf v$ and $\Sigma_3\mathsf H\mathbf w=\mu\mathbf w$. Multiplying the first relation by $\mathbf w^\dagger\Sigma_3$ and the conjugate of the second by $\mathbf v$, and using $\Sigma_3^2=\mathsf I$ and $\mathsf H^\dagger=\mathsf H$, gives

<!-- eq 13: eq-lswt-metric-orthogonality -->

$$
\mathbf w^\dagger\mathsf H\mathbf v=\lambda\,\mathbf w^\dagger\Sigma_3\mathbf v=\mu^*\,\mathbf w^\dagger\Sigma_3\mathbf v,
\qquad
(\lambda-\mu^*)\,\mathbf w^\dagger\Sigma_3\mathbf v=0. \tag{13}
$$

Eigenvectors whose eigenvalues satisfy $\lambda\ne\mu^*$ are therefore orthogonal in the indefinite inner product $\mathbf w^\dagger\Sigma_3\mathbf v$. For $\mathbf w=\mathbf v$ the relation reads $\mathbf v^\dagger\mathsf H\mathbf v=\lambda\,\mathbf v^\dagger\Sigma_3\mathbf v$, and an eigenvector with a nonreal eigenvalue has $\mathbf v^\dagger\Sigma_3\mathbf v=0$.

## Consequences of Positive Definiteness

Assume that $\mathsf H$ is positive definite. Three properties of $\Sigma_3\mathsf H$ follow.

**Real, nonzero eigenvalues with a fixed metric sign.** For an eigenvector $\mathbf v$, the relation $\mathbf v^\dagger\mathsf H\mathbf v=\lambda\,\mathbf v^\dagger\Sigma_3\mathbf v$ has a positive left-hand side. If $\lambda$ were not real, the previous section would give $\mathbf v^\dagger\Sigma_3\mathbf v=0$ and hence $\mathbf v^\dagger\mathsf H\mathbf v=0$, a contradiction. Therefore $\lambda$ is real and nonzero, $\mathbf v^\dagger\Sigma_3\mathbf v\ne0$, and

<!-- eq 14: eq-lswt-metric-sign-rule -->

$$
\operatorname{sign}\lambda=\operatorname{sign}\big(\mathbf v^\dagger\Sigma_3\mathbf v\big). \tag{14}
$$

Eigenvectors with positive eigenvalues have a positive $\Sigma_3$-norm and become particle columns of $\mathsf T$; eigenvectors with negative eigenvalues have a negative $\Sigma_3$-norm and become hole columns. For an indefinite $\mathsf H$ this link fails: an eigenvector with a positive eigenvalue can have a negative $\Sigma_3$-norm, which is the negative-energy mode of the indefinite case in Paraunitary Diagonalization (Section *Paraunitary Diagonalization*).

**Diagonalizability.** The Cholesky factorization $\mathsf H=\mathsf K\mathsf K^\dagger$ makes $\Sigma_3\mathsf H=\Sigma_3\mathsf K\mathsf K^\dagger$ similar, through $\mathsf K^\dagger$, to the Hermitian matrix $\mathsf K^\dagger\Sigma_3\mathsf K$:

$$
\mathsf K^\dagger\big(\Sigma_3\mathsf H\big)(\mathsf K^\dagger)^{-1}=\mathsf K^\dagger\Sigma_3\mathsf K.
$$

A Hermitian matrix is diagonalizable with real eigenvalues, so $\Sigma_3\mathsf H$ is diagonalizable with the same eigenvalues. For a positive-semidefinite $\mathsf H$, Jordan blocks, such as the one of the antiferromagnetic Goldstone mode, can therefore occur only when $\mathsf H$ is singular. For an indefinite $\mathsf H$ the Cholesky argument does not apply, and $\Sigma_3\mathsf H$ can fail to be diagonalizable even when $\mathsf H$ is invertible. An example with two modes is $\mathsf A=\operatorname{diag}(1,-0.7)$ and an anomalous coupling $0.15$ between the two modes, for which $\det\mathsf H\approx0.522$ while $\Sigma_3\mathsf H$ has a $2\times2$ Jordan block at the eigenvalue $0.85$. Such a case lies outside the stable reference configurations treated by LSWT.

**Equal numbers of positive and negative eigenvalues.** The matrix $\mathsf K^\dagger\Sigma_3\mathsf K$ is congruent to $\Sigma_3$ through the invertible matrix $\mathsf K$. By Sylvester’s law of inertia, congruent Hermitian matrices have the same numbers of positive and negative eigenvalues, so $\Sigma_3\mathsf H$ has $N_{\mathrm{sub}}$ positive and $N_{\mathrm{sub}}$ negative eigenvalues.

## Existence of the Paraunitary Transformation

The three properties construct $\mathsf T$ directly from eigenvectors. Within an eigenspace of $\Sigma_3\mathsf H$ with eigenvalue $\lambda$, the form $\mathbf w^\dagger\Sigma_3\mathbf v$ is definite with the sign of $\lambda$, because (14) holds for every vector of the eigenspace. Gram–Schmidt orthogonalization with respect to this definite form gives a basis with $\mathbf v^\dagger\Sigma_3\mathbf v=\operatorname{sign}\lambda$. Eigenvectors of different eigenvalues are already $\Sigma_3$-orthogonal by (13), since all eigenvalues are real. Collecting the $N_{\mathrm{sub}}$ vectors of positive eigenvalues first and the $N_{\mathrm{sub}}$ vectors of negative eigenvalues second gives a matrix $\mathsf T$ with $\mathsf T^\dagger\Sigma_3\mathsf T=\Sigma_3$. The eigenvalue equation $\Sigma_3\mathsf H\mathsf T=\mathsf T\,\Sigma_3\mathsf E$, with $\mathsf E$ the diagonal matrix of the absolute values of the eigenvalues, then gives $\mathsf T^\dagger\mathsf H\mathsf T=\mathsf T^\dagger\Sigma_3\mathsf T\,\Sigma_3\mathsf E=\mathsf E$, a diagonal matrix with positive entries.

Conversely, if a paraunitary $\mathsf T$ with $\mathsf T^\dagger\mathsf H\mathsf T=\mathsf E$ positive and diagonal exists, then $\mathsf H=(\mathsf T^{-1})^\dagger\mathsf E\,\mathsf T^{-1}$ is congruent to a positive diagonal matrix and is positive definite. A paraunitary transformation to a diagonal form with positive entries therefore exists if and only if $\mathsf H$ is positive definite.

## Colpa’s Transformation Satisfies Both Conditions

Colpa’s construction, $\mathsf T=(\mathsf K^\dagger)^{-1}\mathsf V(\Lambda\Sigma_3)^{1/2}$ with $\mathsf V\Lambda\mathsf V^\dagger=\mathsf K^\dagger\Sigma_3\mathsf K$, is verified in Paraunitary Diagonalization (Section *Paraunitary Diagonalization*) through $\mathsf T\Sigma_3\mathsf T^\dagger=\Sigma_3$ and $\mathsf T^\dagger\mathsf H\mathsf T=\Lambda\Sigma_3$. The second form of the paraunitary condition can also be checked directly. The matrix $\mathsf K^\dagger\Sigma_3\mathsf K$ is invertible, so $\Lambda$ is invertible, and $\Sigma_3^{-1}=\Sigma_3$ gives $\mathsf K^{-1}\Sigma_3(\mathsf K^\dagger)^{-1}=(\mathsf K^\dagger\Sigma_3\mathsf K)^{-1}=\mathsf V\Lambda^{-1}\mathsf V^\dagger$. Hence

$$
\mathsf T^\dagger\Sigma_3\mathsf T
=(\Lambda\Sigma_3)^{1/2}\mathsf V^\dagger\,\mathsf K^{-1}\Sigma_3(\mathsf K^\dagger)^{-1}\,\mathsf V(\Lambda\Sigma_3)^{1/2}
=(\Lambda\Sigma_3)^{1/2}\Lambda^{-1}(\Lambda\Sigma_3)^{1/2}
=\Sigma_3,
$$

where the last step uses that the diagonal matrices commute and $\Lambda\Sigma_3\Lambda^{-1}=\Sigma_3$. This is the check left in commented form in the restructured TeX appendix; by the first section it is equivalent to $\mathsf T\Sigma_3\mathsf T^\dagger=\Sigma_3$.

## Freedom in the Transformation

Let $\mathsf T$ and $\mathsf T'$ both be paraunitary with $\mathsf T^\dagger\mathsf H\mathsf T=\mathsf T'^\dagger\mathsf H\mathsf T'=\mathsf E$, with the same ordering of the diagonal entries. The matrix $\mathsf U=\mathsf T^{-1}\mathsf T'$ is paraunitary by the group property. From $\Sigma_3\mathsf H\mathsf T=\mathsf T\Sigma_3\mathsf E$ and the same relation for $\mathsf T'$,

$$
\mathsf U\,\Sigma_3\mathsf E=\mathsf T^{-1}\Sigma_3\mathsf H\mathsf T'=\Sigma_3\mathsf E\,\mathsf U,
$$

so $\mathsf U$ commutes with the diagonal matrix $\Sigma_3\mathsf E$ of signed energies and is block diagonal in its eigenspaces. When $\mathsf H$ is positive definite, each eigenspace contains only particle columns or only hole columns, because their signed energies have opposite signs. On such a block $\Sigma_3$ is $+\mathsf I$ or $-\mathsf I$, and the paraunitary condition restricted to the block is unitarity. The transformation is therefore unique up to a phase for each nondegenerate column and a unitary mixing within each degenerate set of columns.

## Particle–Hole Structure of the Transformation

The Bogoliubov transformation has the block form $\mathsf T_{\mathbf k}=\begin{pmatrix}\mathsf P_{\mathbf k}&\mathsf Q_{-\mathbf k}\\ \mathsf Q_{\mathbf k}^*&\mathsf P_{-\mathbf k}^*\end{pmatrix}$, so that the lower components of $\hat\Psi_{\mathbf k}$ are the adjoints of the upper components of $\hat\Psi_{-\mathbf k}$. With $\Sigma_1=\sigma_1\otimes\mathsf I_{N_{\mathrm{sub}}}$, which exchanges the two block rows when applied on the left and the two block columns when applied on the right, the block form is equivalent to

<!-- eq 15: eq-lswt-transformation-particle-hole -->

$$
\mathsf T_{\mathbf k}=\Sigma_1\mathsf T_{-\mathbf k}^*\Sigma_1. \tag{15}
$$

A transformation of this form can always be chosen. Let $\mathbf t$ be a particle column of $\mathsf T_{-\mathbf k}$ with $\Sigma_3\mathsf H_{-\mathbf k}\mathbf t=\varepsilon\,\mathbf t$ and $\mathbf t^\dagger\Sigma_3\mathbf t=1$. The particle–hole relation $\Sigma_1\mathsf H_{-\mathbf k}^*\Sigma_1=\mathsf H_{\mathbf k}$ and $\Sigma_3\Sigma_1=-\Sigma_1\Sigma_3$ give $\Sigma_3\mathsf H_{\mathbf k}(\Sigma_1\mathbf t^*)=-\varepsilon\,\Sigma_1\mathbf t^*$, and

$$
(\Sigma_1\mathbf t^*)^\dagger\Sigma_3(\Sigma_1\mathbf t^*)=\mathbf t^{\mathsf T}\Sigma_1\Sigma_3\Sigma_1\mathbf t^*=-\big(\mathbf t^\dagger\Sigma_3\mathbf t\big)^*=-1.
$$

The vector $\Sigma_1\mathbf t^*$ is therefore a correctly normalized hole column of $\mathsf T_{\mathbf k}$ with signed energy $-\varepsilon$, which is $-\varepsilon_{n,-\mathbf k}$ for the band of $\mathbf t$. Building the hole columns of $\mathsf T_{\mathbf k}$ in this way from the particle columns of $\mathsf T_{-\mathbf k}$, and doing the same with $\mathbf k$ and $-\mathbf k$ exchanged, yields (15). The resulting columns are $\Sigma_3$-orthogonal to the particle columns of $\mathsf T_{\mathbf k}$ by (13), since their signed energies have the opposite sign. A transformation computed independently at $\mathbf k$, such as Colpa’s, has hole columns that agree with these up to the freedom of the previous section. At a momentum with $-\mathbf k\equiv\mathbf k$ modulo a reciprocal-lattice vector, the relation constrains $\mathsf T_{\mathbf k}$ itself and fixes part of that freedom.

## References

### Internal Documents

- Paraunitary Diagonalization (Section *Paraunitary Diagonalization*): states the paraunitary condition, the particle–hole structure of the spectrum, Colpa’s construction, and the gauge freedom proved here.

- Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*: defines the BdG blocks and the block relations behind the particle–hole relation.

- Notation and Conventions in research note *LSWT spin Hamiltonian*: defines the Nambu metric and the paraunitary-matrix notation.

- Worked Example (Section *Worked Example*): evaluates these statements for a single sublattice.

### External Sources

- J. H. P. Colpa, “Diagonalization of the Quadratic Boson Hamiltonian,” *Physica A* **93**, 327–353 (1978), [doi:10.1016/0378-4371(78)90160-7](https://doi.org/10.1016/0378-4371(78)90160-7): metric-preserving diagonalization of a positive-definite quadratic boson Hamiltonian through the Cholesky factorization.
