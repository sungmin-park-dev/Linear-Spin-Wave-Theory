---
frontmatter-version: 1
title: Topological Magnon Quantities
doc-path: docs/lswt/02-observables
status: draft
last-edited-by: claude
created: 2026-06-03
updated: 2026-10-01
source: docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf
source-section: "Skyrmions, Topological Magnons, and Hall Effects (source p. 24, Eqs. (143)–(149))"
---

# Topological Magnon Quantities

Two different objects carry topological labels in a spin-wave calculation. A lattice skyrmion number counts how often a classical spin texture wraps the unit sphere, and it depends only on the reference configuration. A magnon Chern number and the magnon thermal Hall conductivity depend on the eigenvectors of the bosonic Bogoliubov–de Gennes (BdG) matrix $\mathsf H_{\mathbf k}$, so they inherit its paraunitary normalization, its particle–hole structure, and the Fourier convention used to define $\mathsf H_{\mathbf k}$. The band quantities below assume a stable reference configuration, for which $\mathsf H_{\mathbf k}$ is positive definite at the momenta considered.

## Lattice Skyrmion Number of a Spin Texture

The skyrmion number of a lattice texture is defined on a triangulation of the lattice into oriented elementary triangles. For a triangle with vertices $I,J,K$ taken counterclockwise and unit spin directions $\mathbf m_I=\mathbf S_I/S_I$, the signed solid angle $\chi_{IJK}\in(-2\pi,2\pi)$ spanned by the three directions is fixed by

$$
\exp\!\left(\frac{\mathrm i\chi_{IJK}}{2}\right)
=\frac{1+\mathbf m_I\cdot\mathbf m_J+\mathbf m_J\cdot\mathbf m_K+\mathbf m_K\cdot\mathbf m_I+\mathrm i\,\mathbf m_I\cdot(\mathbf m_J\times\mathbf m_K)}{\left|1+\mathbf m_I\cdot\mathbf m_J+\mathbf m_J\cdot\mathbf m_K+\mathbf m_K\cdot\mathbf m_I+\mathrm i\,\mathbf m_I\cdot(\mathbf m_J\times\mathbf m_K)\right|}.
$$

The sign of the triple product fixes the orientation of the solid angle, and the complex argument selects the branch, so that $\chi_{IJK}$ is not restricted to $(-\pi,\pi)$. The skyrmion number sums the solid angles over all triangles of one magnetic unit cell,

$$
Q_{\mathrm{sk}}=\frac{1}{4\pi}\sum_{\triangle}\chi_{\triangle}.
$$ {#eq-lswt-lattice-skyrmion-number}

For a texture with the periodicity of the magnetic unit cell, the triangles tile a closed torus and $Q_{\mathrm{sk}}$ is an integer, except for exceptional configurations in which, on some triangle, the triple product vanishes while the real part $1+\mathbf m_I\cdot\mathbf m_J+\mathbf m_J\cdot\mathbf m_K+\mathbf m_K\cdot\mathbf m_I$ is not positive. Such a triangle has three directions on one great circle that are not contained in a half circle, or two antiparallel directions; its solid angle is $\pm2\pi$ or undefined, and $Q_{\mathrm{sk}}$ is not defined for that configuration. On the triangular lattice the elementary triangles are the up- and down-pointing plaquettes, two per crystallographic unit cell. The skyrmion number is a property of the classical reference configuration and does not require the spin-wave expansion.

## Berry Curvature of Bosonic Bogoliubov Bands

The paraunitary transformation $\mathsf T_{\mathbf k}$ diagonalizes the $2N_{\mathrm{sub}}\times2N_{\mathrm{sub}}$ matrix $\mathsf H_{\mathbf k}$ while preserving the Nambu metric,

$$
\mathsf T_{\mathbf k}^\dagger\Sigma_3\mathsf T_{\mathbf k}=\Sigma_3,
\qquad
\mathsf T_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}
=\operatorname{diag}\!\left(\varepsilon_{1\mathbf k},\ldots,\varepsilon_{N_{\mathrm{sub}}\mathbf k},\varepsilon_{1,-\mathbf k},\ldots,\varepsilon_{N_{\mathrm{sub}},-\mathbf k}\right).
$$

The first $N_{\mathrm{sub}}$ columns of $\mathsf T_{\mathbf k}$ belong to the physical magnon bands $n=1,\ldots,N_{\mathrm{sub}}$, and the remaining columns are their particle–hole partners. We use the signed energies $\lambda_{p\mathbf k}=(\Sigma_3)_{pp}\,(\mathsf T_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\mathsf T_{\mathbf k})_{pp}$ for $p=1,\ldots,2N_{\mathrm{sub}}$, which equal $\varepsilon_{n\mathbf k}$ on the particle columns and $-\varepsilon_{n,-\mathbf k}$ on the hole columns. With the paraunitary Berry connection $\mathcal A_{n\mathbf k}=\mathrm i\,(\Sigma_3)_{nn}\,[\mathsf T_{\mathbf k}^\dagger\Sigma_3\nabla_{\mathbf k}\mathsf T_{\mathbf k}]_{nn}$, the Berry curvature $\Omega_{n\mathbf k}=\partial_{k_x}\mathcal A^y_{n\mathbf k}-\partial_{k_y}\mathcal A^x_{n\mathbf k}$ of a physical band takes the Kubo form

$$
\Omega_{n\mathbf k}
=-2\,\operatorname{Im}\sum_{\substack{p=1\\p\neq n}}^{2N_{\mathrm{sub}}}
(\Sigma_3)_{nn}(\Sigma_3)_{pp}\,
\frac{\big[\mathsf T_{\mathbf k}^\dagger(\partial_{k_x}\mathsf H_{\mathbf k})\mathsf T_{\mathbf k}\big]_{np}\,
\big[\mathsf T_{\mathbf k}^\dagger(\partial_{k_y}\mathsf H_{\mathbf k})\mathsf T_{\mathbf k}\big]_{pn}}
{\left(\lambda_{n\mathbf k}-\lambda_{p\mathbf k}\right)^2}.
$$ {#eq-lswt-bdg-berry-curvature}

The sum runs over all $2N_{\mathrm{sub}}$ columns. Terms with another particle band $p=m\le N_{\mathrm{sub}}$ have the denominator $(\varepsilon_{n\mathbf k}-\varepsilon_{m\mathbf k})^2$, whereas terms with a hole column have $(\varepsilon_{n\mathbf k}+\varepsilon_{m,-\mathbf k})^2$ and carry the anomalous (pairing) couplings. When $\mathsf H_{\mathbf k}$ conserves the boson number, the hole columns decouple and the expression reduces to the Berry curvature of a Hermitian band problem. The curvature of band $n$ is defined where the band is separated from every other signed energy. It is undefined, and generically divergent, where two physical bands touch and where a particle energy meets a hole energy, $\varepsilon_{n\mathbf k}+\varepsilon_{m,-\mathbf k}\to0$; for a reciprocal spectrum the latter reduces to $\varepsilon_{n\mathbf k}\to0$.

The pointwise curvature depends on the Fourier convention used for $\mathsf H_{\mathbf k}$. A change between the full-position and the periodic (cell) gauge multiplies the particle and hole components of sublattice $I$ by the same momentum-dependent phase $\exp(\pm\mathrm i\mathbf k\cdot\mathbf r_I)$. This adds to $\mathcal A_{n\mathbf k}$ a term proportional to $\sum_I w_{nI}(\mathbf k)\,\mathbf r_I$, where $w_{nI}$ is the $\Sigma_3$-weighted weight of band $n$ on sublattice $I$, and changes $\Omega_{n\mathbf k}$ by the curl of that term. Because $w_{nI}(\mathbf k)$ is periodic over the magnetic Brillouin zone (MBZ), the curl integrates to zero, so the Chern number does not depend on the choice, while plots of $\Omega_{n\mathbf k}$ do. A momentum integral that weights $\Omega_{n\mathbf k}$ by a nonconstant function of $\varepsilon_{n\mathbf k}$ is not protected by this argument: after integration by parts the added term becomes the integral of the gradient of the weight crossed with the periodic vector field, which does not vanish in general. The thermal Hall conductivity below is such an integral. We evaluate it in the full-position gauge, in which $\nabla_{\mathbf k}\mathsf H_{\mathbf k}$ contains the intracell site positions in the same way as the position operator; this identification of the full-position gauge as the physical one is inferred from that correspondence rather than derived in this note. The two gauges give the same thermal Hall conductivity when a lattice symmetry makes the added term integrate to zero.

## Chern Number of an Isolated Magnon Band

For a physical band $n$ separated from all other signed energies throughout the MBZ, the Chern number is the integral of the Berry curvature over the MBZ,

$$
C_n=\frac{1}{2\pi}\int_{\mathrm{MBZ}}\Omega_{n\mathbf k}\,d^2\mathbf k
\approx\frac{1}{2\pi}\,\frac{A_{\mathrm{MBZ}}}{N_{\mathbf k}}\sum_{\mathbf k}\Omega_{n\mathbf k},
$$ {#eq-lswt-magnon-chern-number}

where the discrete form uses a uniform mesh of $N_{\mathbf k}$ momenta and $A_{\mathrm{MBZ}}=(2\pi)^2/A_{\mathrm{uc}}$ is the area of the MBZ for a magnetic unit cell of area $A_{\mathrm{uc}}$. The integral is an integer for an isolated band, whereas the discrete sum approaches it as the mesh is refined and converges slowly where the curvature is concentrated near a small gap. The lattice link-variable construction of Fukui, Hatsugai, and Suzuki evaluates the same invariant from the eigenvectors on neighboring mesh points; for bosonic bands the link variables use the paraunitary product $\mathbf t_{n\mathbf k}^\dagger\Sigma_3\mathbf t_{n\mathbf k'}$ of the columns $\mathbf t_{n\mathbf k}$ of $\mathsf T_{\mathbf k}$. That construction returns an integer on every mesh, including meshes that do not resolve a gap closing, so an integer value by itself does not establish that the band is isolated. Neither construction defines a Chern number for bands that touch; a group of touching bands carries only a combined invariant, which is not treated here.

## Magnon Thermal Hall Conductivity

The transverse thermal conductivity of a magnetic layer follows from the Berry curvature of the thermally occupied magnon bands. For isolated bands, the conductivity per layer is

$$
\kappa^{\mathrm{2D}}_{xy}
=-\frac{k_{\mathrm B}^2T}{\hbar}\,\frac{1}{N_{\mathbf k}A_{\mathrm{uc}}}
\sum_{\mathbf k}\sum_{n=1}^{N_{\mathrm{sub}}}
c_2\!\left(n_{\mathrm B}(\varepsilon_{n\mathbf k})\right)\Omega_{n\mathbf k},
$$ {#eq-lswt-magnon-thermal-hall}

with the Bose–Einstein distribution $n_{\mathrm B}(\varepsilon)=[\exp(\varepsilon/k_{\mathrm B}T)-1]^{-1}$ and the weight function

$$
c_2(x)=\int_0^x\left(\ln\frac{1+t}{t}\right)^2dt
=(1+x)\left(\ln\frac{1+x}{x}\right)^2-(\ln x)^2-2\operatorname{Li}_2(-x),
$$ {#eq-lswt-thermal-hall-weight}

where $\operatorname{Li}_2(z)=-\int_0^z\ln(1-t)\,t^{-1}\,dt$ is the dilogarithm. The weight vanishes as $x\to0$ and approaches $\pi^2/3$ as $x\to\infty$, so thermally unoccupied bands do not contribute. The factor $1/(N_{\mathbf k}A_{\mathrm{uc}})$ turns the momentum sum into $\int_{\mathrm{MBZ}}d^2\mathbf k/(2\pi)^2$; the Berry curvature and the cell area carry the same squared length unit, which cancels, and $\kappa^{\mathrm{2D}}_{xy}$ has the unit of $k_{\mathrm B}^2T/\hbar$. For a stack of equivalent, independent layers with interlayer spacing $d$, the three-dimensional conductivity is $\kappa_{xy}=\kappa^{\mathrm{2D}}_{xy}/d$.

The band sum in the thermal Hall conductivity requires the individual curvatures only through the combination $\sum_n c_2(n_{\mathrm B}(\varepsilon_{n\mathbf k}))\Omega_{n\mathbf k}$ at each momentum. Inserting the Kubo form and pairing the terms of two particle bands $n$ and $m$ gives

$$
\sum_{n=1}^{N_{\mathrm{sub}}}c_2^{(n)}\Omega_{n\mathbf k}
=-2\sum_{n<m}\left(c_2^{(n)}-c_2^{(m)}\right)
\frac{\operatorname{Im}\big([\mathsf T_{\mathbf k}^\dagger\partial_{k_x}\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}]_{nm}[\mathsf T_{\mathbf k}^\dagger\partial_{k_y}\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}]_{mn}\big)}{(\varepsilon_{n\mathbf k}-\varepsilon_{m\mathbf k})^2}
+\sum_{n=1}^{N_{\mathrm{sub}}}c_2^{(n)}\,\Omega^{\mathrm{ph}}_{n\mathbf k},
$$ {#eq-lswt-thermal-hall-pair-form}

where $c_2^{(n)}=c_2(n_{\mathrm B}(\varepsilon_{n\mathbf k}))$ and $\Omega^{\mathrm{ph}}_{n\mathbf k}$ collects the hole-column terms of the Kubo form. The weight difference vanishes for exactly degenerate particle bands and is proportional to the energy difference near a crossing, so this combination remains finite where individual curvatures are undefined. Only a degeneracy between a particle band and a hole column, which requires $\varepsilon_{n\mathbf k}\to0$, leaves it undefined.

The primary note writes the weight as $c_2(n_{\mathrm B})-\pi^2/3$ and normalizes by a system volume $V$ without $\hbar$. The volume factor is replaced here by the per-layer area $N_{\mathbf k}A_{\mathrm{uc}}$ and the explicit $\hbar$. The two weights differ by a constant. With the discrete MBZ sum written as an integral, the conductivity obtained with the primary-note weight exceeds the value from the per-layer formula by

$$
\frac{\pi^2}{3}\,\frac{k_{\mathrm B}^2T}{\hbar}\int_{\mathrm{MBZ}}\frac{d^2\mathbf k}{(2\pi)^2}\sum_{n=1}^{N_{\mathrm{sub}}}\Omega_{n\mathbf k}
=\frac{\pi k_{\mathrm B}^2T}{6\hbar}\sum_{n=1}^{N_{\mathrm{sub}}}C_n,
$$

so the two forms coincide when the Chern numbers of the physical bands sum to zero. For a Hamiltonian matrix $\mathsf H_{\mathbf k}$ that is positive definite on the whole MBZ, the sum does vanish. The family $\mathsf H_{\mathbf k,\lambda}=(1-\lambda)\mathsf H_{\mathbf k}+\lambda\mathbb 1$, $0\le\lambda\le1$, stays positive definite, so its particle and hole bands never touch. The total Chern number of the particle bands can change only at such a touching, so it equals its value at $\lambda=1$, where $\mathsf T_{\mathbf k}=\mathbb 1$ and every curvature vanishes (Shindou et al., Eq. (29)). The two weights therefore give the same conductivity. This note uses the weight without the constant, as Matsumoto and Murakami and Neumann et al. do; it gives thermally unoccupied bands zero weight directly. At high temperature every weight approaches $\pi^2/3$, so this form gives $\kappa_{xy}/T\to-(\pi k_{\mathrm B}^2/6\hbar)\sum_nC_n$, which vanishes by the sum rule above; the primary-note weight approaches zero in that limit directly. When $\mathsf H_{\mathbf k}$ is only positive semidefinite, as with Goldstone modes, particle and hole columns meet at zero energy and the argument does not apply as stated; that case is left open.

## References

### Internal Documents

- [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md): defines $\mathsf T_{\mathbf k}$, the Nambu metric, and the stability condition assumed for the band quantities.
- [Momentum-Space BdG Hamiltonian](../01-derivation/momentum-space-bdg-hamiltonian.md): defines $\mathsf H_{\mathbf k}$ and its Fourier convention, which fixes the gauge of the pointwise Berry curvature.
- [Magnon Observables](magnon-observables.md): supplies the magnon bands and occupations used by the thermal Hall conductivity.
- [Classical Order and Local Frame](../00-foundations/classical-order-and-local-frame.md): defines the classical spin directions used by the skyrmion number.
- [Notation and Conventions](../00-foundations/notation-and-conventions.md): fixes $\Sigma_3$, $\varepsilon_{n\mathbf k}$, $N_{\mathrm{sub}}$, and the MBZ notation, and lists the thermal-Hall units as not yet fixed.

### External Sources

- B. Berg and M. Lüscher, "Definition and Statistical Distributions of a Topological Number in the Lattice O(3) Sigma-Model," *Nuclear Physics B* **190**, 412-424 (1981), [doi:10.1016/0550-3213(81)90568-X](https://doi.org/10.1016/0550-3213(81)90568-X): signed solid angle of a lattice triangle and the integer lattice topological number.
- R. Shindou, R. Matsumoto, S. Murakami, and J. Ohe, "Topological chiral magnonic edge mode in a magnonic crystal," *Physical Review B* **87**, 174427 (2013), [doi:10.1103/PhysRevB.87.174427](https://doi.org/10.1103/PhysRevB.87.174427): Berry curvature and Chern number of bosonic BdG bands with the paraunitary normalization; Eq. (29) proves that the Chern numbers of the particle bands sum to zero for a positive definite Hamiltonian matrix.
- R. Matsumoto and S. Murakami, "Rotational motion of magnons and the thermal Hall effect," *Physical Review B* **84**, 184406 (2011), [doi:10.1103/PhysRevB.84.184406](https://doi.org/10.1103/PhysRevB.84.184406): magnon thermal Hall conductivity in terms of the Berry curvature and the weight $c_2$ without a constant term (Eq. (23)).
- T. Fukui, Y. Hatsugai, and H. Suzuki, "Chern Numbers in Discretized Brillouin Zone: Efficient Method of Computing (Spin) Hall Conductances," *Journal of the Physical Society of Japan* **74**, 1674-1677 (2005), [doi:10.1143/JPSJ.74.1674](https://doi.org/10.1143/JPSJ.74.1674): lattice link-variable construction of Chern numbers.
- R. R. Neumann, A. Mook, J. Henk, and I. Mertig, "Thermal Hall Effect of Magnons in Collinear Antiferromagnetic Insulators: Signatures of Magnetic and Topological Phase Transitions," *Physical Review Letters* **128**, 117201 (2022), [doi:10.1103/PhysRevLett.128.117201](https://doi.org/10.1103/PhysRevLett.128.117201): thermal Hall response of BdG magnon bands; cited by the primary note for the thermal Hall formula.
