---
frontmatter-version: 1
title: Real-Space Boson Hamiltonian
doc-path: docs/lswt/01-derivation
status: draft
last-edited-by: codex
created: 2026-06-04
updated: 2026-09-06
source: docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf
source-section: Bosonic representations for spin Hamiltonian
---

# Real-Space Boson Hamiltonian

The quadratic boson Hamiltonian collects the terms of second order in the Holstein–Primakoff operators. The expression below includes inter-site exchange and the Zeeman term in the local frame. It does not yet include an explicit contribution from the separately defined on-site anisotropy matrix $\mathsf D_I$.

The circular-component expressions below are provisional: the contraction convention relating $\widetilde J_\ell^{pq}$ to the Cartesian exchange matrix is not fully specified. The labels $p,q\in\{+,-,0\}$ denote local circular components, rather than laboratory Cartesian indices.

## Quadratic Exchange and Field Terms

For a bond $\ell=(I,J)$, define the spin-weighted exchange coefficient

$$
t_\ell^{pq}=\sqrt{S_I S_J}\,\widetilde J_\ell^{pq},
\qquad \ell=(I,J).
$$

Retaining the quadratic terms gives the provisional component form

$$
\begin{aligned}
\hat H_2
=&\sum_{\ell=(I,J)\in\mathcal L}\Big[
t_\ell^{--}\hat a_I\hat a_J
+t_\ell^{-+}\hat a_I\hat a_J^\dagger
+t_\ell^{+-}\hat a_I^\dagger\hat a_J
+t_\ell^{++}\hat a_I^\dagger\hat a_J^\dagger
\\
&\hspace{7em}
-\widetilde J_\ell^{00}\left(S_I\hat n_J+S_J\hat n_I\right)
\Big]
+\sum_I\widetilde h_I^0\hat n_I.
\end{aligned}
$$

The first four terms contain the normal and anomalous boson products associated with each link. The remaining terms multiply local number operators. The field contribution involves the longitudinal component of the rotated Zeeman-energy vector, $\widetilde h_I^0$, rather than an unrotated laboratory component.

## Local Number-Operator Coefficient

Each site receives exchange contributions from all incident links. Let $\partial I$ be the set of links incident on $I$, and let $I'_\ell$ be the opposite endpoint of link $\ell$. For the exchange and field terms displayed above, the local coefficient is

$$
\mu_I
=\widetilde h_I^0
-\sum_{\ell\in\partial I}S_{I'_\ell}\widetilde J_\ell^{00}.
$$

This coefficient combines the rotated field with the longitudinal exchange contributions from neighboring spins. Its use inherits the circular-component convention and interaction scope stated above.

## References

### Internal Documents

- [Bilinear Spin Hamiltonian](../00-foundations/bilinear-spin-hamiltonian.md): defines the spin Hamiltonian expanded here.
- [Classical Order and Local Frame](../00-foundations/classical-order-and-local-frame.md): defines the rotated exchange tensor and field.
- [Holstein–Primakoff Expansion](holstein-primakoff-expansion.md): supplies the boson expansion and harmonic truncation.
- [Momentum-Space BdG Hamiltonian](momentum-space-bdg-hamiltonian.md): transforms this real-space quadratic Hamiltonian to momentum space.

### External Sources

- None.
