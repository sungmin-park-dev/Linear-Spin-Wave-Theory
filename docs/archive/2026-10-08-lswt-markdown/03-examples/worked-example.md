---
frontmatter-version: 1
title: Worked Example
doc-path: docs/lswt/03-examples
status: draft
last-edited-by: claude
created: 2026-06-07
updated: 2026-10-01
source: docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf
source-section: "Example: Solving spin system with linear spin wave theory (source pp. 25–27)"
---

# Worked Example

A single magnetic sublattice, $N_{\mathrm{sub}}=1$, gives the smallest quadratic boson Hamiltonian that contains every ingredient of the general LSWT problem: a normal coefficient, an anomalous pairing coefficient, a paraunitary Bogoliubov transformation, a zero-point correction, and a quantum reduction of the boson number. Every quantity can be written in closed form. The example follows the steps of [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md) and [Magnon Observables](../02-observables/magnon-observables.md) for this case, so the general formulas can be checked against explicit expressions. We allow the normal coefficients at $\mathbf k$ and $-\mathbf k$ to differ, which shows how a nonreciprocal spectrum $\varepsilon_{\mathbf k}\ne\varepsilon_{-\mathbf k}$ arises.

## Single-Sublattice BdG Hamiltonian

For $N_{\mathrm{sub}}=1$ the blocks $\mathsf A_{\mathbf k}$ and $\mathsf B_{\mathbf k}$ of [Momentum-Space BdG Hamiltonian](../01-derivation/momentum-space-bdg-hamiltonian.md) are numbers. The block relations reduce to a real normal coefficient $A_{\mathbf k}$ and an even anomalous coefficient, $B_{-\mathbf k}=B_{\mathbf k}$. We drop the sublattice index and write the quadratic Hamiltonian in normal-ordered form,

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

$$
A_{\mathbf k}>0,
\qquad
A_{-\mathbf k}>0,
\qquad
A_{\mathbf k}A_{-\mathbf k}>|B_{\mathbf k}|^2.
$$ {#eq-lswt-example-stability}

For $A_{\mathbf k}^{-}=0$ these conditions reduce to $A_{\mathbf k}>|B_{\mathbf k}|$. The weaker statement $|A_{\mathbf k}|\ge|B_{\mathbf k}|$ is not sufficient: for $A_{\mathbf k}<-|B_{\mathbf k}|$ the matrix is negative definite and the magnon vacuum is the highest, not the lowest, state of the harmonic Hamiltonian, and equality gives a zero mode, which is treated in the last section.

The magnon energies are the eigenvalues of $\Sigma_3\mathsf H_{\mathbf k}$. The term $A_{\mathbf k}^{-}\Sigma_3$ contributes $A_{\mathbf k}^{-}$ times the identity to $\Sigma_3\mathsf H_{\mathbf k}$, so it shifts both eigenvalues equally, and the remaining matrix has the eigenvalues $\pm\omega_{\mathbf k}$ with

$$
\omega_{\mathbf k}=\sqrt{\big(A_{\mathbf k}^{+}\big)^2-|B_{\mathbf k}|^2},
\qquad
\varepsilon_{\mathbf k}=\omega_{\mathbf k}+A_{\mathbf k}^{-},
\qquad
\varepsilon_{-\mathbf k}=\omega_{\mathbf k}-A_{\mathbf k}^{-}.
$$ {#eq-lswt-example-energies}

The eigenvalues of $\Sigma_3\mathsf H_{\mathbf k}$ are $\varepsilon_{\mathbf k}$ and $-\varepsilon_{-\mathbf k}$, which is the particle–hole structure of [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md) for one band. Since $\omega_{-\mathbf k}=\omega_{\mathbf k}$ and $A_{-\mathbf k}^{-}=-A_{\mathbf k}^{-}$, the expression for $\varepsilon_{-\mathbf k}$ is the same formula evaluated at $-\mathbf k$. The odd part of the normal coefficient thus makes the spectrum nonreciprocal without changing $\omega_{\mathbf k}$. Under @eq-lswt-example-stability both energies are positive, because $\omega_{\mathbf k}^2-(A_{\mathbf k}^{-})^2=A_{\mathbf k}A_{-\mathbf k}-|B_{\mathbf k}|^2>0$.

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

$$
\begin{pmatrix}\hat a_{\mathbf k}\\ \hat a_{-\mathbf k}^\dagger\end{pmatrix}
=\begin{pmatrix}
\cosh\theta_{\mathbf k} & -\exp(\mathrm i\phi_{\mathbf k})\sinh\theta_{\mathbf k}\\
-\exp(-\mathrm i\phi_{\mathbf k})\sinh\theta_{\mathbf k} & \cosh\theta_{\mathbf k}
\end{pmatrix}
\begin{pmatrix}\hat b_{\mathbf k}\\ \hat b_{-\mathbf k}^\dagger\end{pmatrix}.
$$ {#eq-lswt-example-bogoliubov}

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

The off-diagonal entries vanish for the angle $\tanh2\theta_{\mathbf k}=|B_{\mathbf k}|/A_{\mathbf k}^{+}$, and the diagonal entries equal $\big[(A_{\mathbf k}^{+})^2-|B_{\mathbf k}|^2\big]/\omega_{\mathbf k}=\omega_{\mathbf k}$. Adding back $A_{\mathbf k}^{-}\Sigma_3$ gives $\mathsf T_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}=\operatorname{diag}(\varepsilon_{\mathbf k},\varepsilon_{-\mathbf k})$, in agreement with @eq-lswt-example-energies. The diagonal magnon Hamiltonian and its zero-point correction follow from [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md):

$$
\hat H_2=\sum_{\mathbf k\in\mathrm{MBZ}}\varepsilon_{\mathbf k}\,\hat b_{\mathbf k}^\dagger\hat b_{\mathbf k}+\Delta E_{\mathrm{zp}},
\qquad
\Delta E_{\mathrm{zp}}=\frac12\sum_{\mathbf k\in\mathrm{MBZ}}\left(\varepsilon_{\mathbf k}-A_{\mathbf k}\right)
=\frac12\sum_{\mathbf k\in\mathrm{MBZ}}\left(\omega_{\mathbf k}-A_{\mathbf k}^{+}\right).
$$ {#eq-lswt-example-zero-point}

The second form follows because $A_{\mathbf k}^{-}$ is odd and cancels in the sum over a set closed under inversion. Each summand $\omega_{\mathbf k}-A_{\mathbf k}^{+}$ is negative when $B_{\mathbf k}\ne0$, so the anomalous terms lower the energy below the classical value; the nonreciprocal part does not change the zero-point correction. For $B_{\mathbf k}=0$ the angle vanishes, $\mathsf T_{\mathbf k}$ is the identity, and $\Delta E_{\mathrm{zp}}=0$: the magnon vacuum coincides with the boson vacuum.

## Boson Number and Anomalous Average

Inserting @eq-lswt-example-bogoliubov into the averages and using the Gibbs occupations $n_{\mathbf k}=n_{\mathrm B}(\varepsilon_{\mathbf k})$ of [Thermodynamics](../02-observables/thermodynamics.md), with $\langle\hat b_{\mathbf k}^\dagger\hat b_{\mathbf k}\rangle=n_{\mathbf k}$ and vanishing averages of $\hat b\hat b$ and of products at different momenta, gives

$$
\begin{aligned}
\langle\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}\rangle
&=\cosh^2\theta_{\mathbf k}\,n_{\mathbf k}+\sinh^2\theta_{\mathbf k}\,\big(1+n_{-\mathbf k}\big),\\
\langle\hat a_{\mathbf k}\hat a_{-\mathbf k}\rangle
&=-\exp(\mathrm i\phi_{\mathbf k})\cosh\theta_{\mathbf k}\sinh\theta_{\mathbf k}\,\big(1+n_{\mathbf k}+n_{-\mathbf k}\big).
\end{aligned}
$$ {#eq-lswt-example-averages}

The first line is the single-band form of the boson number in [Magnon Observables](../02-observables/magnon-observables.md). In the magnon vacuum the occupations vanish, and

$$
\langle\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}\rangle_0=\sinh^2\theta_{\mathbf k}=\frac12\left(\frac{A_{\mathbf k}^{+}}{\omega_{\mathbf k}}-1\right),
\qquad
\langle\hat a_{\mathbf k}\hat a_{-\mathbf k}\rangle_0=-\frac{B_{\mathbf k}}{2\omega_{\mathbf k}}.
$$

The zero-temperature boson number is the quantum reduction of the ordered moment. It is nonzero only when $B_{\mathbf k}\ne0$, that is, when the harmonic Hamiltonian does not conserve the number of Holstein–Primakoff bosons. The anomalous average carries the phase of $B_{\mathbf k}$ and is the off-diagonal block of the correlation matrix $\mathsf T_{\mathbf k}\mathsf N_{\mathbf k}(0)\mathsf T_{\mathbf k}^\dagger$.

## Response to a Uniform Shift of the Normal Coefficient

The zero-point correction and the boson number are related by the Hellmann–Feynman theorem. Adding $\mu\sum_{\mathbf k}\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}$ to $\hat H_2$ shifts $A_{\mathbf k}\to A_{\mathbf k}+\mu$ and leaves $B_{\mathbf k}$ unchanged. The derivative of the ground-state energy with respect to $\mu$ is the ground-state average of the added operator, and @eq-lswt-example-zero-point gives

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

The stability condition becomes marginal when $A_{\mathbf k}A_{-\mathbf k}\to|B_{\mathbf k}|^2$. For $A_{\mathbf k}^{-}=0$ this is the limit $\omega_{\mathbf k}\to0^+$ at fixed $A_{\mathbf k}^{+}>0$, which is the single-mode Goldstone case $A=B>0$ of [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md). In this limit $\cosh\theta_{\mathbf k}$ and $\sinh\theta_{\mathbf k}$ grow as $(A_{\mathbf k}^{+}/2\omega_{\mathbf k})^{1/2}$, and

$$
\langle\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}\rangle_0\simeq\frac{A_{\mathbf k}^{+}}{2\omega_{\mathbf k}},
\qquad
\langle\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}\rangle\simeq\frac{A_{\mathbf k}^{+}}{\omega_{\mathbf k}}\,n_{\mathrm B}(\omega_{\mathbf k})\simeq\frac{A_{\mathbf k}^{+}k_{\mathrm B}T}{\omega_{\mathbf k}^2}
\quad(\omega_{\mathbf k}\ll k_{\mathrm B}T).
$$

The occupation per mode diverges at the zero mode, while the energy per mode, $\frac12(\omega_{\mathbf k}-A_{\mathbf k}^{+})$, stays finite. Whether the momentum sum of the boson number converges then depends on the dispersion near the zero and on the dimension, as discussed in [Magnon Observables](../02-observables/magnon-observables.md). At $\omega_{\mathbf k}=0$ itself the transformation does not exist, and the single-mode boson number is undefined rather than large.

## References

### Internal Documents

- [Momentum-Space BdG Hamiltonian](../01-derivation/momentum-space-bdg-hamiltonian.md): defines the Nambu spinor, the BdG blocks, their block relations, and the trace subtraction specialized here.
- [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md): supplies the paraunitary condition, the particle–hole structure, the zero-point correction, and the single-mode zero-mode cases.
- [Magnon Observables](../02-observables/magnon-observables.md): defines the correlation matrix and the boson number whose single-band forms are evaluated here.
- [Thermodynamics](../02-observables/thermodynamics.md): defines the Bose–Einstein occupations.
- [Notation and Conventions](../00-foundations/notation-and-conventions.md): defines $\Sigma_3$, $\mathsf T_{\mathbf k}$, $\hat\Phi_{\mathbf k}$, and the energy symbols.

### External Sources

- None.
