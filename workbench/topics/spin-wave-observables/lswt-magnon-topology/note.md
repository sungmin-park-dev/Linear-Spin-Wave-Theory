<!-- Converted from LaTeX on 2026-10-11. The LaTeX original is kept in archive/2026-10-11-lswt-latex-notes/lswt-magnon-topology/. Equation numbers are fixed in this form; the original \label of each numbered equation is in the comment before it. -->

<!-- Topological Magnon Quantities Converted on 2026-10-08 from archive/2026-10-08-lswt-markdown/02-observables/topological-magnon-quantities.md (status: draft, source-section: "Skyrmions, Topological Magnons, and Hall Effects (source p. 24, Eqs. (143)–(149))"). Review status: draft; awaiting the user's physics and mathematics review. -->

# Topological Magnon Quantities

Two different objects carry topological labels in a spin-wave calculation. A lattice skyrmion number counts how often a classical spin texture wraps the unit sphere, and it depends only on the reference configuration. A magnon Chern number and the magnon thermal Hall conductivity depend on the eigenvectors of the bosonic Bogoliubov–de Gennes (BdG) matrix $\mathsf H_{\mathbf k}$, so they inherit its paraunitary normalization, its particle–hole structure, and the Bloch-basis convention used to define $\mathsf H_{\mathbf k}$. The band quantities below assume a stable reference configuration, for which $\mathsf H_{\mathbf k}$ is positive definite at the momenta considered.

## Lattice Skyrmion Number of a Spin Texture

The skyrmion number of a lattice texture is defined on a triangulation of the lattice into oriented elementary triangles. For a triangle with vertices $I,J,K$ taken counterclockwise and unit spin directions $\mathbf m_I=\mathbf S_I/S_I$, the signed solid angle $\chi_{IJK}\in(-2\pi,2\pi)$ spanned by the three directions is fixed by

$$
\exp\!\left(\frac{\mathrm i\chi_{IJK}}{2}\right)
=\frac{1+\mathbf m_I\cdot\mathbf m_J+\mathbf m_J\cdot\mathbf m_K+\mathbf m_K\cdot\mathbf m_I+\mathrm i\,\mathbf m_I\cdot(\mathbf m_J\times\mathbf m_K)}{\left|1+\mathbf m_I\cdot\mathbf m_J+\mathbf m_J\cdot\mathbf m_K+\mathbf m_K\cdot\mathbf m_I+\mathrm i\,\mathbf m_I\cdot(\mathbf m_J\times\mathbf m_K)\right|}.
$$

The sign of the triple product fixes the orientation of the solid angle, and the complex argument selects the branch, so that $\chi_{IJK}$ is not restricted to $(-\pi,\pi)$. The skyrmion number sums the solid angles over all triangles of one magnetic unit cell,

<!-- eq 1: eq-lswt-lattice-skyrmion-number -->

$$
Q_{\mathrm{sk}}=\frac{1}{4\pi}\sum_{\triangle}\chi_{\triangle}. \tag{1}
$$

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

<!-- eq 2: eq-lswt-bdg-berry-curvature -->

$$
\Omega_{n\mathbf k}
=-2\,\operatorname{Im}\sum_{\substack{p=1\\p\neq n}}^{2N_{\mathrm{sub}}}
(\Sigma_3)_{nn}(\Sigma_3)_{pp}\,
\frac{\big[\mathsf T_{\mathbf k}^\dagger(\partial_{k_x}\mathsf H_{\mathbf k})\mathsf T_{\mathbf k}\big]_{np}\,
\big[\mathsf T_{\mathbf k}^\dagger(\partial_{k_y}\mathsf H_{\mathbf k})\mathsf T_{\mathbf k}\big]_{pn}}
{\left(\lambda_{n\mathbf k}-\lambda_{p\mathbf k}\right)^2}. \tag{2}
$$

The sum runs over all $2N_{\mathrm{sub}}$ columns. Terms with another particle band $p=m\le N_{\mathrm{sub}}$ have the denominator $(\varepsilon_{n\mathbf k}-\varepsilon_{m\mathbf k})^2$, whereas terms with a hole column have $(\varepsilon_{n\mathbf k}+\varepsilon_{m,-\mathbf k})^2$ and carry the anomalous (pairing) couplings. When $\mathsf H_{\mathbf k}$ conserves the boson number, the hole columns decouple and the expression reduces to the Berry curvature of a Hermitian band problem. The curvature of band $n$ is defined where the band is separated from every other signed energy. It is undefined, and generically divergent, where two physical bands touch and where a particle energy meets a hole energy, $\varepsilon_{n\mathbf k}+\varepsilon_{m,-\mathbf k}\to0$; for a reciprocal spectrum the latter reduces to $\varepsilon_{n\mathbf k}\to0$.

The pointwise curvature depends on the Bloch-basis convention used for $\mathsf H_{\mathbf k}$. The full-position convention attaches the phase $\exp(\mathrm i\mathbf k\cdot(\mathbf R_i+\boldsymbol\tau_a))$ to the boson on sublattice $a$ of cell $i$, whereas the cell convention attaches $\exp(\mathrm i\mathbf k\cdot\mathbf R_i)$. The two are related by the sublattice-dependent phase matrix $\operatorname{diag}\exp(\pm\mathrm i\mathbf k\cdot\boldsymbol\tau_a)$, applied equally to the particle and hole components. This relation is not a gauge transformation. A gauge transformation multiplies each band by its own phase, $\mathbf t_{n\mathbf k}\to\exp(\mathrm i\chi_n(\mathbf k))\mathbf t_{n\mathbf k}$, and leaves every observable unchanged; the change of convention instead replaces the position operator $\sum_i\mathbf r_i\hat n_i$ by $\sum_i\mathbf R_i\hat n_i$, which describes a system in which every site sits at its cell origin. It adds to $\mathcal A_{n\mathbf k}$ the term $\pm\sum_a w_{na}(\mathbf k)\,\boldsymbol\tau_a$, where $w_{na}$ is the $\Sigma_3$-weighted weight of band $n$ on sublattice $a$, and changes $\Omega_{n\mathbf k}$ by the curl of that term. Because $w_{na}(\mathbf k)$ is periodic over the magnetic Brillouin zone (MBZ), the curl integrates to zero, so the Chern number is the same in both conventions, while plots of $\Omega_{n\mathbf k}$ differ. A momentum integral that weights $\Omega_{n\mathbf k}$ by a nonconstant function of $\varepsilon_{n\mathbf k}$ is not protected by this argument. For the thermal Hall conductivity below, integration by parts leaves a difference proportional to $\sum_a\hat{\mathbf z}\cdot(\mathbf J_a\times\boldsymbol\tau_a)$, where $\mathbf J_a$ is the momentum average of $\sum_nc_2'\,w_{na}\nabla_{\mathbf k}\varepsilon_{n\mathbf k}$, the weighted group velocity carried by sublattice $a$; it vanishes in general only when a symmetry maps each sublattice onto itself, for example a threefold or twofold rotation about a site. The thermal Hall response is defined through the position operator of the actual site positions, which enters the energy magnetization, and the full-position convention is the one whose Bloch states correspond to that operator. The cell-convention value is not a property of the system: relabeling the sites of one sublattice into a neighboring cell, $\boldsymbol\tau_a\to\boldsymbol\tau_a+\mathbf R$, describes the same system but changes that value by a term proportional to $\hat{\mathbf z}\cdot(\mathbf J_a\times\mathbf R)$, whereas the full-position value is unchanged. We therefore evaluate all band quantities in the full-position convention. The remaining approximation is the tight-binding assumption that the magnon position operator is diagonal at the site positions, which is a property of the model rather than a choice of convention.

## Chern Number of an Isolated Magnon Band

For a physical band $n$ separated from all other signed energies throughout the MBZ, the Chern number is the integral of the Berry curvature over the MBZ,

<!-- eq 3: eq-lswt-magnon-chern-number -->

$$
C_n=\frac{1}{2\pi}\int_{\mathrm{MBZ}}\Omega_{n\mathbf k}\,d^2\mathbf k
\approx\frac{1}{2\pi}\,\frac{A_{\mathrm{MBZ}}}{N_{\mathbf k}}\sum_{\mathbf k}\Omega_{n\mathbf k}, \tag{3}
$$

where the discrete form uses a uniform mesh of $N_{\mathbf k}$ momenta and $A_{\mathrm{MBZ}}=(2\pi)^2/A_{\mathrm{uc}}$ is the area of the MBZ for a magnetic unit cell of area $A_{\mathrm{uc}}$. The integral is an integer for an isolated band, whereas the discrete sum approaches it as the mesh is refined and converges slowly where the curvature is concentrated near a small gap. The lattice link-variable construction of Fukui, Hatsugai, and Suzuki [@fukuiChern2005] evaluates the same invariant from the eigenvectors on neighboring mesh points; for bosonic bands the link variables use the paraunitary product $\mathbf t_{n\mathbf k}^\dagger\Sigma_3\mathbf t_{n\mathbf k'}$ of the columns $\mathbf t_{n\mathbf k}$ of $\mathsf T_{\mathbf k}$. That construction returns an integer on every mesh, including meshes that do not resolve a gap closing, so an integer value by itself does not establish that the band is isolated. Neither construction defines a Chern number for bands that touch; a group of touching bands carries only a combined invariant, which is not treated here.

## Magnon Thermal Hall Conductivity

The transverse thermal conductivity of a magnetic layer follows from the Berry curvature of the thermally occupied magnon bands. For isolated bands, the conductivity per layer is

<!-- eq 4: eq-lswt-magnon-thermal-hall -->

$$
\kappa^{\mathrm{2D}}_{xy}
=-\frac{k_{\mathrm B}^2T}{\hbar}\,\frac{1}{N_{\mathbf k}A_{\mathrm{uc}}}
\sum_{\mathbf k}\sum_{n=1}^{N_{\mathrm{sub}}}
c_2\!\left(n_{\mathrm B}(\varepsilon_{n\mathbf k})\right)\Omega_{n\mathbf k}, \tag{4}
$$

with the Bose–Einstein distribution $n_{\mathrm B}(\varepsilon)=[\exp(\varepsilon/k_{\mathrm B}T)-1]^{-1}$ and the weight function

<!-- eq 5: eq-lswt-thermal-hall-weight -->

$$
c_2(x)=\int_0^x\left(\ln\frac{1+t}{t}\right)^2dt
=(1+x)\left(\ln\frac{1+x}{x}\right)^2-(\ln x)^2-2\operatorname{Li}_2(-x), \tag{5}
$$

where $\operatorname{Li}_2(z)=-\int_0^z\ln(1-t)\,t^{-1}\,dt$ is the dilogarithm. The weight vanishes as $x\to0$ and approaches $\pi^2/3$ as $x\to\infty$, so thermally unoccupied bands do not contribute. The factor $1/(N_{\mathbf k}A_{\mathrm{uc}})$ turns the momentum sum into $\int_{\mathrm{MBZ}}d^2\mathbf k/(2\pi)^2$; the Berry curvature and the cell area carry the same squared length unit, which cancels, and $\kappa^{\mathrm{2D}}_{xy}$ has the unit of $k_{\mathrm B}^2T/\hbar$. For a stack of equivalent, independent layers with interlayer spacing $d$, the three-dimensional conductivity is $\kappa_{xy}=\kappa^{\mathrm{2D}}_{xy}/d$.

The band sum in the thermal Hall conductivity requires the individual curvatures only through the combination $\sum_n c_2(n_{\mathrm B}(\varepsilon_{n\mathbf k}))\Omega_{n\mathbf k}$ at each momentum. Inserting the Kubo form and pairing the terms of two particle bands $n$ and $m$ gives

<!-- eq 6: eq-lswt-thermal-hall-pair-form -->

$$
\sum_{n=1}^{N_{\mathrm{sub}}}c_2^{(n)}\Omega_{n\mathbf k}
=-2\sum_{n<m}\left(c_2^{(n)}-c_2^{(m)}\right)
\frac{\operatorname{Im}\big([\mathsf T_{\mathbf k}^\dagger\partial_{k_x}\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}]_{nm}[\mathsf T_{\mathbf k}^\dagger\partial_{k_y}\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}]_{mn}\big)}{(\varepsilon_{n\mathbf k}-\varepsilon_{m\mathbf k})^2}
+\sum_{n=1}^{N_{\mathrm{sub}}}c_2^{(n)}\,\Omega^{\mathrm{ph}}_{n\mathbf k}, \tag{6}
$$

where $c_2^{(n)}=c_2(n_{\mathrm B}(\varepsilon_{n\mathbf k}))$ and $\Omega^{\mathrm{ph}}_{n\mathbf k}$ collects the hole-column terms of the Kubo form. The weight difference vanishes for exactly degenerate particle bands and is proportional to the energy difference near a crossing, so this combination remains finite where individual curvatures are undefined. Only a degeneracy between a particle band and a hole column, which requires $\varepsilon_{n\mathbf k}\to0$, leaves it undefined.

The primary note writes the weight as $c_2(n_{\mathrm B})-\pi^2/3$ and normalizes by a system volume $V$ without $\hbar$. The volume factor is replaced here by the per-layer area $N_{\mathbf k}A_{\mathrm{uc}}$ and the explicit $\hbar$. The two weights differ by a constant. With the discrete MBZ sum written as an integral, the conductivity obtained with the primary-note weight exceeds the value from the per-layer formula by

$$
\frac{\pi^2}{3}\,\frac{k_{\mathrm B}^2T}{\hbar}\int_{\mathrm{MBZ}}\frac{d^2\mathbf k}{(2\pi)^2}\sum_{n=1}^{N_{\mathrm{sub}}}\Omega_{n\mathbf k}
=\frac{\pi k_{\mathrm B}^2T}{6\hbar}\sum_{n=1}^{N_{\mathrm{sub}}}C_n,
$$

so the two forms coincide when the Chern numbers of the physical bands sum to zero. For a Hamiltonian matrix $\mathsf H_{\mathbf k}$ that is positive definite on the whole MBZ, the sum does vanish. The family $\mathsf H_{\mathbf k,\lambda}=(1-\lambda)\mathsf H_{\mathbf k}+\lambda\mathbb 1$, $0\le\lambda\le1$, stays positive definite, so its particle and hole bands never touch. The total Chern number of the particle bands can change only at such a touching, so it equals its value at $\lambda=1$, where $\mathsf T_{\mathbf k}=\mathbb 1$ and every curvature vanishes [@shindouTopological2013, Eq. (29)]. The two weights therefore give the same conductivity. This note uses the weight without the constant, as Matsumoto and Murakami [@matsumotoRotational2011, Eq. (23)] and Neumann et al. [@neumannThermal2022] do; it gives thermally unoccupied bands zero weight directly. At high temperature every weight approaches $\pi^2/3$, so this form gives $\kappa_{xy}/T\to-(\pi k_{\mathrm B}^2/6\hbar)\sum_nC_n$, which vanishes by the sum rule above; the primary-note weight approaches zero in that limit directly. When $\mathsf H_{\mathbf k}$ is only positive semidefinite, as with Goldstone modes, particle and hole columns meet at zero energy at isolated momenta, and the deformation argument does not apply as stated.

### Goldstone Points

At a momentum $\mathbf k_0$ where $\mathsf H_{\mathbf k_0}$ has a zero eigenvalue, $\mathsf T_{\mathbf k_0}$ does not exist and $\Omega_{n\mathbf k_0}$ is not defined. Away from $\mathbf k_0$ the band is separated from every other signed energy, so $\Omega_{n\mathbf k}$ is defined on the zone with one point removed, and two questions remain. The thermal Hall conductivity weights $\Omega_{n\mathbf k}$ by a bounded function, $c_2(n_{\mathrm B}(\varepsilon))\to\pi^2/3$ as $\varepsilon\to0$, so it is defined whenever $\Omega_{n\mathbf k}$ is integrable near $\mathbf k_0$; boundedness is sufficient. The Chern number (3) is defined for a band separated from every other signed energy on the whole MBZ. For a band that reaches zero at a Goldstone point we take it as the limit of the Chern number after a small symmetry-breaking term opens a gap at $\mathbf k_0$, the regularization commonly used for gapless magnon bands. Evaluated directly, (3) is then a finite integral over the zone with $\mathbf k_0$ removed, but integrability does not make it an integer. Removing a disk $\mathcal D_r$ of radius $r$ around $\mathbf k_0$, Stokes' theorem gives $2\pi C_n=\int_{\mathrm{MBZ}\setminus\mathcal D_r}\Omega_{n\mathbf k}\,d^2\mathbf k=2\pi m-\gamma_n(r)$ with an integer $m$, where $\gamma_n(r)$ is the Berry phase of band $n$ around $\partial\mathcal D_r$. $C_n$ is an integer only if $\gamma_n(r)\to0$ modulo $2\pi$ as $r\to0$. A massless Dirac point is the counterexample: $\Omega$ vanishes away from the touching point, but the loop phase is $\pi$. The rest of this section derives conditions under which $\gamma_n(r)\to0$ and the direct integral equals the gapped limit, and gives a zero mode for which both fail.

We assume that $\mathsf H_{\mathbf k}$ is positive semidefinite near $\mathbf k_0$ and positive definite at $\mathbf k\neq\mathbf k_0$ in a neighbourhood, and that exactly two eigenvalues of $\Sigma_3\mathsf H_{\mathbf k}$ tend to zero as $\mathbf k\to\mathbf k_0$: the energy $\varepsilon_{n\mathbf k}$ of the Goldstone band and the hole eigenvalue $-\varepsilon_{n',-\mathbf k}$ of the band that vanishes at $-\mathbf k_0$, with $n'=n$ when $\mathbf k_0$ is time-reversal invariant, $\mathbf k_0\equiv-\mathbf k_0$ modulo a reciprocal lattice vector. Both are separated from the other signed energies. Several bands that vanish at the same momentum, as for the three Goldstone modes of the $120^\circ$ state in a three-sublattice cell, touch there and have no separate Chern numbers, so they are excluded. Let $\mathsf P_{\mathbf k}$ be the spectral projector of $\Sigma_3\mathsf H_{\mathbf k}$ onto the two small eigenvalues. It depends smoothly on $\mathbf k$, including at $\mathbf k_0$, and the Nambu form restricted to its range is nondegenerate with one positive and one negative direction. For a basis $\mathbf x_{\mathbf k},\mathbf y_{\mathbf k}$ of that range, $\Sigma_3\mathsf H_{\mathbf k}$ acts on the coefficients $\mathbf w$ of $\mathbf t=w_1\mathbf x_{\mathbf k}+w_2\mathbf y_{\mathbf k}$ as $\mathsf g_{\mathbf k}^{-1}\mathsf h_{\mathbf k}$, with $\mathsf h_{\mathbf k}=[\mathbf x\ \mathbf y]^\dagger\mathsf H_{\mathbf k}[\mathbf x\ \mathbf y]$ and $\mathsf g_{\mathbf k}=[\mathbf x\ \mathbf y]^\dagger\Sigma_3[\mathbf x\ \mathbf y]$. This reduction is exact because the range is invariant. Two kinds of zero mode differ in the kernel of $\mathsf H_{\mathbf k_0}$.

**A linear Goldstone mode** has a one-dimensional kernel. Its vector $\mathbf x_0$ has zero Nambu norm: the zero eigenvalue of $\Sigma_3\mathsf H_{\mathbf k_0}$ is then defective, with a partner $\mathbf y_0$ satisfying $\Sigma_3\mathsf H_{\mathbf k_0}\mathbf y_0\propto\mathbf x_0$, so $\mathbf x_0^\dagger\Sigma_3\mathbf x_0\propto\mathbf x_0^\dagger\mathsf H_{\mathbf k_0}\mathbf y_0=0$. We choose a smooth basis with

<!-- eq 7: eq-lswt-goldstone-symplectic-basis -->

$$
\mathbf x_{\mathbf k}^\dagger\Sigma_3\mathbf x_{\mathbf k}=\mathbf y_{\mathbf k}^\dagger\Sigma_3\mathbf y_{\mathbf k}=0,\qquad
\mathbf x_{\mathbf k}^\dagger\Sigma_3\mathbf y_{\mathbf k}=\mathrm i,\qquad
\mathsf H_{\mathbf k_0}\mathbf x_{\mathbf k_0}=0, \tag{7}
$$

the coordinates of the Goldstone angle and its conjugate density. Then $\mathsf h_{\mathbf k}$ has the entries $\rho_{\mathbf k}=\mathbf x^\dagger\mathsf H_{\mathbf k}\mathbf x$, $f_{\mathbf k}=\mathbf x^\dagger\mathsf H_{\mathbf k}\mathbf y$ and $\chi_{\mathbf k}^{-1}=\mathbf y^\dagger\mathsf H_{\mathbf k}\mathbf y>0$, with $\rho=O(|\mathbf q|^2)$ and $f=O(|\mathbf q|)$ for $\mathbf q=\mathbf k-\mathbf k_0$. The two small signed energies are $\operatorname{Im}f_{\mathbf k}\pm s_{\mathbf k}$ with $s_{\mathbf k}=[\rho_{\mathbf k}/\chi_{\mathbf k}-(\operatorname{Re}f_{\mathbf k})^2]^{1/2}$, and the paraunitary vector of the Goldstone band is

<!-- eq 8: eq-lswt-goldstone-vector -->

$$
\mathbf t_{n\mathbf k}=(2\chi_{\mathbf k}s_{\mathbf k})^{-1/2}\left[\mathbf x_{\mathbf k}-\chi_{\mathbf k}\left(\operatorname{Re}f_{\mathbf k}+\mathrm i\,s_{\mathbf k}\right)\mathbf y_{\mathbf k}\right]. \tag{8}
$$

The odd part $\operatorname{Im}f$, which makes the dispersion nonreciprocal, shifts both energies and does not enter the vector. We assume that the dispersion is linear in every direction, so that the leading part of $s^2_{\mathbf k}$ is a positive definite quadratic form in $\mathbf q$; the norm of $\mathbf t_{n\mathbf k}$ then grows like $|\mathbf q|^{-1/2}$. Inserting (8) into $\mathcal A_{n\mathbf k}=\mathrm i\,\mathbf t^\dagger\Sigma_3\nabla_{\mathbf k}\mathbf t$ and using (7), the two terms of order $|\mathbf q|^{-1}$ that contain $s$ combine into the gradient of $\chi s\cdot(2\chi s)^{-1}=1/2$ and cancel, and the remaining terms of that order collect into

$$
\mathcal A_{n\mathbf k}=\frac{\mathbf a_0}{2\chi_{\mathbf k}s_{\mathbf k}}+O(1),\qquad
\mathbf a_0=\left[\mathrm i\,\mathbf x^\dagger\Sigma_3\nabla_{\mathbf k}\mathbf x+\nabla_{\mathbf k}(\chi\operatorname{Re}f)\right]_{\mathbf k_0}.
$$

The real vector $\mathbf a_0$ is fixed up to a positive factor by (7): the remaining basis changes, $\mathbf x\to\mathrm e^{\mathrm i\alpha}\lambda(\mathbf x+a\mathbf y)$ and $\mathbf y\to\mathrm e^{\mathrm i\alpha}\lambda^{-1}(\mathbf y+b\mathbf x)$ with real $\lambda,a,b$ and $a(\mathbf k_0)=0$, multiply it by $\lambda^2$. Two consequences follow. First, on the circle $\partial\mathcal D_r$ the leading term contributes $\oint\mathbf a_0\cdot\hat{\boldsymbol\theta}\,d\theta/(2\chi\hat s(\theta))$, where $\hat s(\theta)=s/r$ to leading order satisfies $\hat s(\theta+\pi)=\hat s(\theta)$ while $\hat{\boldsymbol\theta}$ changes sign, so this contribution vanishes and $\gamma_n(r)=O(r)$. Second, at a time-reversal-invariant $\mathbf k_0$ the particle–hole relation $\Sigma_1\mathsf H^*_{-\mathbf k}\Sigma_1=\mathsf H_{\mathbf k}$ maps a basis satisfying (7) at $\mathbf k_0+\mathbf q$ to one at $\mathbf k_0-\mathbf q$, which reverses the sign of $\mathbf a_0$; since $\mathbf a_0$ is fixed up to a positive factor, $\mathbf a_0=0$. There the connection is bounded in this gauge and $\Omega_{n\mathbf k}=O(|\mathbf q|^{-1})$, which is integrable. At a $\mathbf k_0$ that is not time-reversal invariant, $\mathbf a_0$ need not vanish; the curvature can then grow like $|\mathbf q|^{-2}$ with zero angular average, the integral avoiding $\mathbf k_0$ exists only as the limit over annuli, and the thermal Hall integral has the same restriction. This case is derived but not checked numerically.

**A quadratic zero mode** has a two-dimensional kernel spanned by a particle-type vector $\mathbf w$ and its hole partner $\bar{\mathbf w}$, with $\mathbf w^\dagger\Sigma_3\mathbf w=-\bar{\mathbf w}^\dagger\Sigma_3\bar{\mathbf w}=1$ and $\mathbf w^\dagger\Sigma_3\bar{\mathbf w}=0$. In that basis all entries of $\mathsf h_{\mathbf k}=\begin{pmatrix}\alpha_{\mathbf k}&\beta_{\mathbf k}\\\beta^*_{\mathbf k}&\delta_{\mathbf k}\end{pmatrix}$ are $O(|\mathbf q|^2)$, and the Goldstone vector is $\cosh\vartheta\,\mathbf w-\mathrm e^{-\mathrm i\phi}\sinh\vartheta\,\bar{\mathbf w}$ with $\tanh2\vartheta=2|\beta|/(\alpha+\delta)$ and $\phi=\arg\beta$, up to the smooth change of $\mathbf w,\bar{\mathbf w}$. If $\beta_{\mathbf k}/(\alpha_{\mathbf k}+\delta_{\mathbf k})\to0$, in particular when $\mathsf B_{\mathbf k}=0$ near $\mathbf k_0$ as for a collinear ferromagnet with its moment along the field, the vector is continuous at $\mathbf k_0$ and $\gamma_n(r)\to0$. If instead $\beta$ is of the same order as $\alpha+\delta$, the angle $\vartheta$ and the phase $\phi$ depend on the direction of $\mathbf q$ and stay finite as $r\to0$. The connection is $\mathcal A=-\sinh^2\vartheta\,\nabla\phi$ plus a bounded term, so

<!-- eq 9: eq-lswt-goldstone-loop-limit -->

$$
\lim_{r\to0}\gamma_n(r)=-\oint\sinh^2\vartheta(\theta)\,d\phi(\theta), \tag{9}
$$

evaluated with the $O(|\mathbf q|^2)$ parts of $\alpha,\beta,\delta$. When $\phi$ winds around $\mathbf k_0$ this is generically neither zero nor a multiple of $2\pi$, although the curvature stays bounded. The vanishing of $\mathsf B_{\mathbf k_0}$ alone is therefore not sufficient; what matters is how $\mathsf B_{\mathbf k}$ grows compared with $\mathsf A_{\mathbf k}$ near $\mathbf k_0$.

Under these conditions the direct integral equals the gapped limit. Let $\mathsf H_{\mathbf k}(\delta)$ depend continuously on $\delta\ge0$, equal the gapless matrix at $\delta=0$, be positive definite for $\delta>0$, and keep the two small eigenvalues separated from the others. For a linear Goldstone mode, (8) still applies for $\delta>0$ and gives a vector smooth throughout $\mathcal D_r$, because $\chi^{-1}>0$ and $s>0$ there. Then $\int_{\mathcal D_r}\Omega_{n\mathbf k}(\delta)\,d^2\mathbf k=\oint_{\partial\mathcal D_r}\mathcal A_{n\mathbf k}(\delta)$, which tends to the gapless loop integral $O(r)$ as $\delta\to0$, while the integral outside $\mathcal D_r$ converges to its gapless value. Letting $r\to0$ afterwards,

$$
\lim_{\delta\to0}C_n(\delta)=\lim_{r\to0}\frac{1}{2\pi}\int_{\mathrm{MBZ}\setminus\mathcal D_r}\Omega_{n\mathbf k}\,d^2\mathbf k
$$

for every such regularization. The left side is a limit of integers, so the integral avoiding $\mathbf k_0$ is an integer and does not depend on which term opened the gap. For a quadratic zero mode with $\beta/(\alpha+\delta)\to0$ the same argument holds for regularizations that keep $\beta=0$, such as a field along the moment; the check below also finds the same value for one regularization that breaks this. In the opposite case of (9), the gapped Chern number differs from the direct integral by $\lim\gamma_n/2\pi$.

The verification script `lswt_goldstone_chern_check.py` and check 6 of `lswt_bdg_blocks_check.py` test these statements on the honeycomb lattice with second-neighbour DM coupling $D=0.2J$ along $\hat z$ and on one square-lattice model, all at $S=1/2$. For the honeycomb ferromagnet at zero field ($\mathsf B_{\mathbf k}=0$, $\varepsilon_{1\mathbf k}\propto|\mathbf k|^2$), $\Omega_{1\mathbf k}\propto|\mathbf k|^4$. For the canted antiferromagnet in a field $h=2JS$ along $\hat z$, whose U(1) Goldstone mode is linear, $\varepsilon_{1\mathbf k}\approx0.58\,J|\mathbf k|$, $\Omega_{1\mathbf k}\propto|\mathbf k|^3$ with a prefactor that varies with direction with period $60^\circ$, and $\Omega_{2\mathbf k}\propto|\mathbf k|^4$. In both, the Chern numbers on a $60\times60$ mesh that avoids $\Gamma$ are $(+1,-1)$ and the loop phase vanishes like the flux, as $r^6$ and $r^5$. A canted antiferromagnet without threefold symmetry, with the three nearest-neighbour bonds scaled by $(1,1.3,0.8)$ and DM couplings $0.2J$ and $0.05J$ on the two sublattices in a field $h=J$, has a one-dimensional kernel at $\Gamma$, $\Omega_{1\mathbf k}\propto|\mathbf k|$, loop phases $3\times10^{-13}$ at $r=10^{-3}$, and Chern numbers $(+1,-1)$; three different nearest-neighbour anisotropies, $-KS^xS^x$, $-KS^yS^y$ and $-K(S^xS^y+S^yS^x)/2$ with $K=0.05$ and $0.005$, all give $(+1,-1)$. The ferromagnet gapped by a field along $\hat z$ together with the U(1)-breaking anisotropy $K(-S^xS^x+S^yS^y)$ also gives $(+1,-1)$. The decay rates in these models are faster than the bounds derived above, which are all the derivation guarantees. The case of (9) is realized by a square-lattice ferromagnet along $\hat z$ with the pseudo-dipolar anisotropy $K(\mathbf S_i\cdot\hat{\mathbf e})(\mathbf S_j\cdot\hat{\mathbf e})$, $K=0.3J$, on the first-neighbour bonds ($\hat{\mathbf e}$ along the bond) and $K/2$ on the second-neighbour bonds, with $S^zS^z$ terms that make all uniform directions degenerate. Its zero mode is accidental rather than a symmetry Goldstone mode, $\mathsf A_\Gamma=\mathsf B_\Gamma=0$ and $\mathsf B_{\mathbf k}\propto(k_x+\mathrm ik_y)^2$, so $\phi$ winds twice. The curvature tends to the constant $0.0048$, the loop phase tends to $-0.1494$, matching (9), and the integral avoiding $\Gamma$ is $0.0238$, not an integer, whereas a field of $0.05$ or $0.01$ along $\hat z$ gives Chern number $0$. A zero mode linear in some directions and quadratic in others, and Goldstone points away from time-reversal-invariant momenta, are not tested.

## References

### Internal Documents

- Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*: defines $\mathsf T_{\mathbf k}$, the Nambu metric, and the stability condition assumed for the band quantities.

- Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*: defines $\mathsf H_{\mathbf k}$ and its full-position Fourier convention, which fixes the Bloch basis of the pointwise Berry curvature.

- Magnon Observables in research note *LSWT magnon thermodynamics*: supplies the magnon bands and occupations used by the thermal Hall conductivity.

- Classical Order and Local Frame in research note *LSWT classical order*: defines the classical spin directions used by the skyrmion number.

- `code-space/examples/verification/lswt_bdg_blocks_check.py`, check 6: Berry curvature near the Goldstone point, the Berry phase around it and the Chern numbers of the two honeycomb models in Goldstone Points.

- `code-space/examples/verification/lswt_goldstone_chern_check.py`: the canted antiferromagnet without threefold symmetry with three regularizations, the ferromagnet with a U(1)-breaking regularization, and the pseudo-dipolar ferromagnet that realizes (9).

- Notation and Conventions in research note *LSWT conventions*: fixes $\Sigma_3$, $\varepsilon_{n\mathbf k}$, $N_{\mathrm{sub}}$, and the MBZ notation, and states the units of the Berry curvature and the per-layer thermal Hall conductivity.

### External Sources

- [@bergDefinition1981]: signed solid angle of a lattice triangle and the integer lattice topological number.

- [@shindouTopological2013]: Berry curvature and Chern number of bosonic BdG bands with the paraunitary normalization; Eq. (29) proves that the Chern numbers of the particle bands sum to zero for a positive definite Hamiltonian matrix.

- [@matsumotoRotational2011]: magnon thermal Hall conductivity in terms of the Berry curvature and the weight $c_2$ without a constant term (Eq. (23)).

- [@fukuiChern2005]: lattice link-variable construction of Chern numbers.

- [@neumannThermal2022]: thermal Hall response of BdG magnon bands; cited by the primary note for the thermal Hall formula.
