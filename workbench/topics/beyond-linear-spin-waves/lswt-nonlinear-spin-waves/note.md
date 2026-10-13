<!-- Written on 2026-10-12 as a new topic note. Review status: draft; awaiting the user's physics and mathematics review. -->

# Interacting Spin Waves to Order $S^0$

Linear spin-wave theory keeps the Holstein–Primakoff Hamiltonian to quadratic order in the bosons. The next terms in the expansion contain three and four bosons, and they are smaller by powers of $S^{-1/2}$. Treated in perturbation theory about the LSWT vacuum, they give the ground-state energy through order $S^0$, the magnon energies through relative order $1/S$, the $1/S$ shift of the classical spin directions, and the magnon decay rate. This note derives the cubic and quartic boson terms of a bilinear spin Hamiltonian with exchange and Zeeman terms, states the perturbative formulas built from them, and lists where the expansion is not defined. Single-ion terms are excluded, for the reason given in Where the Expansion Is Not Defined.

## Holstein–Primakoff Expansion Beyond Quadratic Order

The exact Holstein–Primakoff representation in the local frame of site $I$ is

$$
\hat{\widetilde S}_I^+=\sqrt{2S_I-\hat n_I}\;\hat a_I,
\qquad
\hat{\widetilde S}_I^-=\hat a_I^\dagger\sqrt{2S_I-\hat n_I},
\qquad
\hat{\widetilde S}_I^0=S_I-\hat n_I .
$$

Expanding the square root, $\sqrt{2S_I-\hat n_I}=\sqrt{2S_I}\,\big(1-\hat n_I/4S_I\big)+O(S_I^{-3/2})$, and using $\hat n_I\hat a_I=\hat a_I^\dagger\hat a_I\hat a_I$, the laboratory spin operator of Spin Operator in the Local Circular Basis in research note *LSWT boson Hamiltonian* becomes

<!-- eq 1: eq-nlswt-spin-operator -->

$$
\hat{\mathbf S}_I
=\left(S_I-\hat n_I\right)\mathbf e_I^0
+\sqrt{S_I}\left(\mathbf e_I^-\hat a_I+\mathbf e_I^+\hat a_I^\dagger\right)
-\frac{1}{4\sqrt{S_I}}\left(\mathbf e_I^-\hat a_I^\dagger\hat a_I\hat a_I+\mathbf e_I^+\hat a_I^\dagger\hat a_I^\dagger\hat a_I\right)
+O(S_I^{-3/2}). \tag{1}
$$

Every term is normal ordered on its site, and operators on different sites commute, so the boson Hamiltonian obtained by inserting Eq. (1) into a bilinear spin Hamiltonian is normal ordered. To write it compactly, we use the transverse operators $\hat a_I^-\equiv\hat a_I$ and $\hat a_I^+\equiv\hat a_I^\dagger$, which accompany $\mathbf e_I^-$ and $\mathbf e_I^+$, and their cubic corrections $\hat A_I^-\equiv\hat a_I^\dagger\hat a_I\hat a_I$ and $\hat A_I^+\equiv\hat a_I^\dagger\hat a_I^\dagger\hat a_I$.

## Order Counting in $1/S$

Grouping the terms by the number $n$ of boson operators gives $\hat H=\sum_n\hat H_n$. We count the Zeeman coupling $\mathbf h_I$ as order $S$, so that the classical configuration does not depend on $S$; the physical spin is inserted only at the end. With this counting

$$
\hat H_n\propto S^{\,2-n/2}:
\qquad
\hat H_0\sim S^2,\quad \hat H_2\sim S,\quad \hat H_3\sim S^{1/2},\quad \hat H_4\sim S^0 .
$$

$\hat H_1$ vanishes at a stationary configuration (Stationarity of the Reference Configuration in research note *LSWT classical order*). The ground-state energy through order $S^0$ needs $\hat H_4$ at first order and $\hat H_3$ at second order, since $(S^{1/2})^2/S=S^0$; the magnon energies through order $S^0$, that is relative order $1/S$, need the same two pieces. Eq. (1) keeps exactly the leading power of $S$ in $\hat H_0$ through $\hat H_4$. The omitted $O(S^{-3/2})$ transverse terms first contribute to $\hat H_3$ at order $S^{-1/2}$ and to the energy at order $S^{-1}$.

## Cubic and Quartic Boson Terms

For a link $\ell=(I,J)$ with exchange matrix $\mathsf J_\ell$, the circular components $\widetilde J_\ell^{pq}=(\mathbf e_I^p)^{\mathsf T}\mathsf J_\ell\,\mathbf e_J^q$ are those of Circular Components of the Exchange Matrix in research note *LSWT boson Hamiltonian*. The cubic terms come from two sources. The product of the longitudinal fluctuation $-\hat n$ on one site with the leading transverse part on the other gives

<!-- eq 2: eq-nlswt-cubic-link -->

$$
\hat H_3^{(\ell)}
=-\sum_{q=\pm}\Big[\sqrt{S_J}\,\widetilde J_\ell^{\,0q}\,\hat n_I\,\hat a_J^{\,q}
+\sqrt{S_I}\,\widetilde J_\ell^{\,q0}\,\hat a_I^{\,q}\,\hat n_J\Big]. \tag{2}
$$

The cubic corrections of Eq. (1) multiply the classical parts of all other spins and the field. With $\mathbf H_I=-\partial E_{\mathrm{cl}}/\partial\mathbf S_I$ the local field of research note *LSWT classical order*, their sum at site $I$ is

$$
\frac{1}{4\sqrt{S_I}}\,\mathbf H_I\cdot\left(\mathbf e_I^-\hat A_I^-+\mathbf e_I^+\hat A_I^+\right)=0,
$$

because a stationary configuration has $\mathbf H_I\parallel\mathbf e_I^0$ and $\mathbf e_I^0\cdot\mathbf e_I^\pm=0$. The Zeeman term is linear in the spins, so it contributes to $\hat H_3$ and $\hat H_4$ only through this vanishing combination. $\hat H_3$ is therefore the sum of Eq. (2) over links. It contains only the mixed components $\widetilde J^{0\pm}$ and $\widetilde J^{\pm0}$; these vanish for a collinear state of a Hamiltonian invariant under rotations about the ordering axis, such as the Néel state of the Heisenberg antiferromagnet, so $\hat H_3=0$ there.

The quartic terms contain the longitudinal–longitudinal product and the cubic corrections multiplied by the leading transverse part of the other spin:

<!-- eq 3: eq-nlswt-quartic-link -->

$$
\hat H_4^{(\ell)}
=\widetilde J_\ell^{\,00}\,\hat n_I\hat n_J
-\frac14\sum_{p,q=\pm}\widetilde J_\ell^{\,pq}\left[\sqrt{\frac{S_J}{S_I}}\,\hat A_I^{\,p}\hat a_J^{\,q}
+\sqrt{\frac{S_I}{S_J}}\,\hat a_I^{\,p}\hat A_J^{\,q}\right]. \tag{3}
$$

The product of a cubic correction with $-\hat n$ on the other site has five operators and belongs to $\hat H_5$. Eqs. (2) and (3) hold for nonsymmetric exchange matrices, unequal spin lengths and links within one sublattice.

## Vertices in the Magnon Basis

The Bogoliubov transformation $\hat\Psi_{\mathbf k}=\mathsf T_{\mathbf k}\hat\Phi_{\mathbf k}$ of Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization* expresses every $\hat a$ through the magnon operators $\hat b_{\mathbf kn}$ and $\hat b_{-\mathbf kn}^\dagger$. The real-space terms are normal ordered in $\hat a$, not in $\hat b$. Rewriting them in normal order with respect to $\hat b$ separates each into the pieces that enter at order $S^0$. We write $N_{\mathrm{uc}}$ for the number of magnetic unit cells and momenta, and a mode label $1=(\mathbf q_1,n_1)$ with energy $\varepsilon_1=\varepsilon_{n_1\mathbf q_1}$.

The cubic term becomes

<!-- eq 4: eq-nlswt-cubic-vertices -->

$$
\hat H_3
=\frac{1}{\sqrt{N_{\mathrm{uc}}}}\sum_{\mathbf q_1+\mathbf q_2+\mathbf q_3=0}s_{123}\,\hat b_1^\dagger\hat b_2^\dagger\hat b_3^\dagger
+\frac{1}{\sqrt{N_{\mathrm{uc}}}}\sum_{\mathbf q_1+\mathbf q_2=\mathbf k}d_{12;\mathbf kn}\,\hat b_1^\dagger\hat b_2^\dagger\hat b_{\mathbf kn}
+\text{h.c.}
+\hat H_1', \tag{4}
$$

where the source vertex $s_{123}$ is symmetric under all permutations of its three mode labels, the decay vertex $d_{12;\mathbf kn}$ is symmetric under $1\leftrightarrow2$, and the sums run over all band indices. $\hat H_1'$ collects the terms in which two operators of Eq. (2) are contracted in the LSWT vacuum. Translation invariance leaves only uniform terms,

$$
\hat H_1'=\sum_i\sum_\mu\left(\eta_\mu\,\hat a_{i\mu}^\dagger+\eta_\mu^*\,\hat a_{i\mu}\right)
=\frac12\sum_i\left(\boldsymbol\zeta^\dagger\hat\psi_i+\hat\psi_i^\dagger\boldsymbol\zeta\right),
\qquad
\hat\psi_i=\begin{pmatrix}\hat{\mathbf a}_i\\ \hat{\mathbf a}_i^{\dagger\mathsf T}\end{pmatrix},
\quad
\boldsymbol\zeta=\begin{pmatrix}\boldsymbol\eta\\ \boldsymbol\eta^*\end{pmatrix},
$$

with $\eta_\mu\sim S^{1/2}$. The quartic term becomes a constant $\langle\hat H_4\rangle$, a quadratic part, and a normal-ordered quartic part. The quadratic part is the Hartree–Fock decoupling of Eq. (3) with the vacuum contractions $\langle\hat a^\dagger\hat a\rangle$, $\langle\hat a\hat a\rangle$ and $\langle\hat a^\dagger\hat a^\dagger\rangle$; we write $\frac12\sum_{\mathbf k}\hat\Psi_{\mathbf k}^\dagger\,\delta\mathsf H^{\mathrm{HF}}_{\mathbf k}\hat\Psi_{\mathbf k}$ for it. The normal-ordered quartic part does not contribute at order $S^0$.

## The Tadpole and the $1/S$ Shift of the Classical Directions

The linear term $\hat H_1'$ is removed by a uniform condensate. Shifting $\hat a_{i\mu}\to\alpha_\mu+\hat a_{i\mu}$ and writing $\psi=(\boldsymbol\alpha,\boldsymbol\alpha^*)$, the energy per magnetic unit cell gains $\frac12\psi^\dagger\mathsf H_{\mathbf 0}\psi+\frac12(\boldsymbol\zeta^\dagger\psi+\psi^\dagger\boldsymbol\zeta)$, which is stationary at

<!-- eq 5: eq-nlswt-tadpole -->

$$
\psi=-\mathsf H_{\mathbf 0}^{-1}\boldsymbol\zeta,
\qquad
E_{\mathrm{tad}}=-\frac12\,\boldsymbol\zeta^\dagger\mathsf H_{\mathbf 0}^{-1}\boldsymbol\zeta . \tag{5}
$$

Since $\boldsymbol\zeta\sim S^{1/2}$ and $\mathsf H_{\mathbf 0}\sim S$, the condensate is $\alpha_\mu\sim S^{-1/2}$ and $E_{\mathrm{tad}}\sim S^0$. A classical Holstein–Primakoff amplitude $\alpha_\mu$ rotates the spin by $\delta\mathbf n_I=(\mathbf e_I^-\alpha_\mu+\mathbf e_I^+\alpha_\mu^*)/\sqrt{S_I}$, which is of order $1/S$. Eq. (5) is therefore the $1/S$ correction of the classical spin directions.

The linear coefficient has a direct meaning. Let $E_{\mathrm{zp}}(\boldsymbol\alpha)$ be the LSWT zero-point energy per magnetic unit cell of the configuration rotated uniformly by the classical amplitudes $\boldsymbol\alpha$. Then

<!-- eq 6: eq-nlswt-tadpole-gradient -->

$$
\eta_\mu=\frac{\partial E_{\mathrm{zp}}}{\partial\alpha_\mu^*}\bigg|_{\boldsymbol\alpha=0}, \tag{6}
$$

so the zero-point energy exerts a force on the classical directions, and $\mathsf H_{\mathbf 0}$ is the classical restoring matrix of a uniform deformation (Local Stability and the BdG Matrix in research note *LSWT classical order*). Eq. (5) is then the relaxation of the configuration in the potential $E_{\mathrm{cl}}+E_{\mathrm{zp}}$ to leading order. The same angle shift, written with the classical Hessian and the gradient of the zero-point energy, gives the rotation of the ordered moment in Magnon Observables in research note *LSWT magnon thermodynamics*. Eq. (6) is suggested by the following: shifting the bosons by $\boldsymbol\alpha$ changes the quadratic part of $\hat H_2+\hat H_3$ at first order in $\boldsymbol\alpha$ by an operator whose vacuum expectation is $\boldsymbol\eta^\dagger\boldsymbol\alpha+\boldsymbol\alpha^\dagger\boldsymbol\eta$. The equivalence of this shift with a rotation of the classical directions, to the order kept, is checked numerically rather than derived here; the check is listed under Checks.

## Ground-State Energy Through Order $S^0$

The ground-state energy per magnetic unit cell is

<!-- eq 7: eq-nlswt-energy -->

$$
E=E_{\mathrm{cl}}+E_{\mathrm{zp}}+\frac{\langle\hat H_4\rangle}{N_{\mathrm{uc}}}+E_3+E_{\mathrm{tad}}+O(S^{-1}),
\qquad
E_3=-\frac{6}{N_{\mathrm{uc}}^2}\sum_{\mathbf q_1,\mathbf q_2}\sum_{n_1n_2n_3}\frac{|s_{123}|^2}{\varepsilon_1+\varepsilon_2+\varepsilon_3}, \tag{7}
$$

with $\mathbf q_3=-\mathbf q_1-\mathbf q_2$. $E_{\mathrm{cl}}\sim S^2$ and $E_{\mathrm{zp}}\sim S$ are the classical and LSWT terms; the other three are of order $S^0$. $E_3$ is second-order perturbation theory in the source vertex: $\hat H_3|0\rangle$ contains three-magnon states, and the factor $6$ counts the pairings of the two sets of three operators for a symmetric $s_{123}$. The decay vertex annihilates the vacuum and does not contribute. $\langle\hat H_3\rangle=0$ because $\hat H_3$ has an odd number of operators.

## Magnon Energies Through Relative Order $1/S$

The magnon energy of band $n$ at momentum $\mathbf k$ through order $S^0$ is

<!-- eq 8: eq-nlswt-magnon-energy -->

$$
\widetilde\varepsilon_{n\mathbf k}
=\varepsilon_{n\mathbf k}
+\Big[\mathsf T_{\mathbf k}^\dagger\big(\delta\mathsf H^{\mathrm{HF}}_{\mathbf k}+\delta\mathsf H^{\mathrm{tad}}_{\mathbf k}\big)\mathsf T_{\mathbf k}\Big]_{nn}
+\operatorname{Re}\Pi^{(3)}_{n}(\mathbf k,\varepsilon_{n\mathbf k}), \tag{8}
$$

where $\delta\mathsf H^{\mathrm{tad}}_{\mathbf k}$ is the Nambu matrix of the quadratic part of Eq. (2) with one operator replaced by the condensate of Eq. (5). The static terms are first-order perturbation theory, because $[\mathsf T_{\mathbf k}^\dagger\mathsf M\mathsf T_{\mathbf k}]_{nn}$ is the first-order change of $\varepsilon_{n\mathbf k}$ for a Nambu perturbation $\frac12\hat\Psi^\dagger\mathsf M\hat\Psi$. The cubic self-energy is second order in $\hat H_3$,

<!-- eq 9: eq-nlswt-cubic-self-energy -->

$$
\Pi^{(3)}_{n}(\mathbf k,\omega)
=\frac{2}{N_{\mathrm{uc}}}\sum_{\mathbf q}\sum_{n_1n_2}\frac{|d_{12;\mathbf kn}|^2}{\omega-\varepsilon_1-\varepsilon_2+\mathrm i0^+}
-\frac{18}{N_{\mathrm{uc}}}\sum_{\mathbf q}\sum_{n_1n_2}\frac{|s_{(\mathbf kn)12}|^2}{\omega+\varepsilon_1+\varepsilon_2}, \tag{9}
$$

with $\mathbf q_1=\mathbf q$ and $\mathbf q_2=\mathbf k-\mathbf q$ in the decay term and $\mathbf q_2=-\mathbf k-\mathbf q$ in the source term. The decay term is the two-magnon intermediate state of $\hat H_3|\mathbf kn\rangle$: the matrix element is $2d_{12;\mathbf kn}/\sqrt{N_{\mathrm{uc}}}$ because the two created magnons can be paired in two ways, and the sum over ordered pairs counts each state twice, which leaves the factor $2$. The source term is the four-magnon intermediate state in which one of the three created magnons carries the quantum numbers of the external one; after the vacuum energy is subtracted, the factor $18=3\times3\times2$ counts the choice of that magnon in each of the two vertices and the pairing of the remaining two. Where $\varepsilon_{n\mathbf k}$ lies inside the two-magnon continuum, $\Gamma_{n\mathbf k}=-\operatorname{Im}\Pi^{(3)}_n(\mathbf k,\varepsilon_{n\mathbf k})$ is the half width of the magnon at this order; the decay kinematics and singularities are reviewed by Zhitomirsky and Chernyshev [@zhitomirskyColloquium2013].

Eq. (8) evaluates the self-energy at the LSWT energy. This is the strict expansion to relative order $1/S$; solving the pole equation self-consistently mixes in higher orders.

## Where the Expansion Is Not Defined

The formulas above are perturbation theory about the LSWT vacuum, so they inherit every condition of LSWT and add some of their own.

- **Non-stationary configurations.** If $\hat H_1\neq0$ at order $S^{3/2}$, the vacuum is not an expansion point, and the cancellation of the field terms in Eq. (2) fails.

- **Zero modes on the momenta used.** Eqs. (7) and (9) divide by magnon energies, and the Hartree–Fock contractions need $\mathsf T_{\mathbf k}$, which does not exist at a zero mode. The momenta must avoid Goldstone points; whether the integrals converge in the thermodynamic limit is a property of the model, observed below for the triangular lattice.

- **A tadpole along a zero mode of $\mathsf H_{\mathbf 0}$.** If $\boldsymbol\zeta$ has a component along a null vector of $\mathsf H_{\mathbf 0}$, Eq. (5) has no solution and the energy decreases without bound along that direction at this order. By Eq. (6) that component is the derivative of $E_{\mathrm{zp}}$ along a classically flat direction. For a zero mode generated by a symmetry of the Hamiltonian, $E_{\mathrm{zp}}$ is invariant along the direction and the component vanishes. For an accidental classical degeneracy, it vanishes only at a stationary point of $E_{\mathrm{zp}}$ on the degenerate manifold: quantum fluctuations select the state (order by disorder), and the expansion is defined only about a configuration so selected.

- **Single-ion terms.** The coherent-state rule of Single-Ion Term in research note *LSWT boson Hamiltonian* fixes the quadratic order of a single-ion term only. Its cubic and quartic orders, which require the same-site products of Eq. (1) with the $(1-1/2S)$ renormalization, are not derived here, so the formulas of this note apply to exchange and Zeeman terms only.

- **Degeneracies in the spectrum.** Eq. (8) is nondegenerate perturbation theory. At a crossing of bands $n$ and $m$ it must be replaced by the diagonalization of the corresponding $2\times2$ block, and at a threshold where $\varepsilon_{n\mathbf k}$ meets the edge of the two-magnon continuum, $\Pi^{(3)}$ can be singular.

## Accidental Zero Modes and the Pseudo-Goldstone Gap

When the classical energy has a continuous degeneracy that is not a symmetry of the Hamiltonian, $\mathsf H_{\mathbf 0}$ has a zero mode: the coordinate $x$ along the flat direction costs no classical energy, while its conjugate $p$ has a finite stiffness $K$, so that the uniform part of $\hat H_2$ along this pair is $\frac12Kp^2$. At a configuration selected as in the previous section, the zero-point energy adds $\frac12E_{\mathrm{zp}}''x^2$, and the mode acquires the pseudo-Goldstone gap

<!-- eq 10: eq-nlswt-pseudo-goldstone -->

$$
\Delta^2=K\,E_{\mathrm{zp}}''=\frac{C_\phi}{\chi}+O(S^0), \tag{10}
$$

where, for a flat global rotation by the angle $\phi$ generated by a spin component $\hat L=\sum_I\hat{\mathbf S}_I\cdot\hat{\mathbf c}$, $C_\phi$ is the curvature of the zero-point energy per site with respect to $\phi$, and $\chi$ is the classical uniform susceptibility of $\hat L$ per site, so that the classical energy cost of a uniform $L$ is $L^2/2N_{\mathrm{site}}\chi$. The leading term is of order $S$, so $\Delta\sim S^{1/2}$ while the other magnons are of order $S$. Eq. (10) is the leading order of the pole equation $\det[\omega\Sigma_3-\mathsf H_{\mathbf 0}-\Pi(\omega)]=0$ at $\mathbf k=0$, with $\Sigma_3$ the Nambu metric and $\Pi=\Pi_1+\Pi_2+\dots$ the Nambu self-energy ordered by loops. The one-loop part $\Pi_1\sim S^0$ enters through its static $xx$ element, which equals $E_{\mathrm{zp}}''$ by Eq. (6) applied along the flat direction. Rau, McClarty and Moessner [@rauPseudoGoldstone2018] discuss the order-by-quantum-disorder origin of such gaps.

The next order of $\Delta^2$, of order $S^0$, needs the one-loop self-energy beyond its static $xx$ element and the static two-loop element $\Pi_2^{xx}(0)\sim S^{-1}$. When that correction is not small compared with the leading term at the physical spin, the $1/S$ series does not determine the gap or the selected direction. This happens for the NBCP model at $S=1/2$, where the ratio of the next-order to the leading term is about $-20$ to $-27$ (research note in the NBCP repository and decision record D47).

## Momentum Meshes

The three-magnon sums require $\mathbf q_3=-\mathbf q_1-\mathbf q_2$ to lie on the same momentum set, and the Goldstone points must be avoided. The mesh $\mathbf q=\sum_a(n_a+\tfrac13)\mathbf b_a/N_a$, with $\mathbf b_a$ the reciprocal vectors of the magnetic cell and $n_a=0,\dots,N_a-1$, has both properties: the sum of two offsets $\tfrac13+\tfrac13$ equals $-\tfrac13$ modulo $1$, and no point has an integer coordinate, so $\Gamma$ is excluded. The mesh is not closed under $\mathbf q\to-\mathbf q$, so quantities that should agree in the thermodynamic limit can differ on a coarse mesh. On a finite torus whose allowed momenta are all gapped, the same formulas are exact Rayleigh–Schrödinger perturbation theory in $1/S$ for that cluster.

## Checks

The script `code-space/examples/verification/lswt_nonlinear_vertices_check.py` checks the derivations of this note; the package tests check the full energies.

1. **Cubic and quartic terms.** For a two-sublattice model with random nonsymmetric exchange on inter- and same-sublattice links, spins $S=1$ and $3/2$, a random field and a stationary configuration, Eqs. (2) and (3) agree with the package expansion monomial by monomial to $3\times10^{-16}$, including the cancellation of the field terms.

2. **Tadpole as a force.** For the same model, $2\operatorname{Re}\eta_\mu$ and $2\operatorname{Im}\eta_\mu$ agree with the numerical derivatives of $E_{\mathrm{zp}}$ under uniform rotations to $5\times10^{-11}$ on a $12\times12$ mesh. On a $6\times6$ mesh they differ by $2\times10^{-5}$; the two sides are different quadratures of the same Brillouin-zone integral.

3. **Square-lattice Heisenberg antiferromagnet.** $\hat H_3=0$ for the Néel state, and $\langle\hat H_4\rangle/N_{\mathrm{site}}=-0.012474\,J$ reproduces the $S^0$ term of $E/N_{\mathrm{site}}=-2JS^2-0.315895\,JS-0.012474\,J$ [@hamerThirdorder1992].

4. **Triangular-lattice Heisenberg antiferromagnet.** For the $120^\circ$ state, the $S^0$ energy per site is $-0.0080112$, $-0.0080123$ and $-0.0080125\,J$ on $12\times12$, $18\times18$ and $24\times24$ meshes. The Hartree–Fock and cubic parts separately still change by about $10^{-3}$ between the last two meshes ($0.0917$ and $-0.0998\,J$ at $24\times24$), and only their sum is converged; the tadpole vanishes by symmetry. Chernyshev and Zhitomirsky [@chernyshevSpin2009, Eqs. (39)–(42)] write the $S^0$ term as $-\tfrac32J(I_4+2I_3)/4$ with the Hartree–Fock constant $I_4=-0.254293$ and the source-vertex integral $I_3=0.13785(1)$, and quote $I_4+2I_3=0.02141$, that is, $-0.0080288\,J$. Their formulas evaluated independently in the rotating frame give $I_4=-0.2542933$, in agreement, and $I_4+2I_3=0.0213668$, so $I_3=0.137830$ and the $S^0$ energy is $-0.00801256\,J$. On the mesh $(n+\tfrac13)/N$ of the primitive zone, the momenta $\mathbf k$, $\mathbf q$ and $-\mathbf k-\mathbf q$ lie on one mesh, the sum is the $S^0$ energy of one twisted cluster, and it converges as $N^{-3}$ (change $3\times10^{-9}$ between $N=96$ and $144$), while $I_4$ and $I_3$ separately converge only as $N^{-1}$. The package value approaches the same number ($-0.0080125394\,J$ on a $36\times36$ mesh of the magnetic cell). The difference from the published coefficient therefore lies in the published value of $I_3$, which is $2\times10^{-5}$ larger, twice its stated uncertainty; because $I_4+2I_3$ is a near cancellation of two terms of size $0.27$, this difference becomes $0.2\%$ of the $S^0$ energy. The check is the script `code-space/examples/verification/lswt_triangular_s0_energy_check.py`.

5. **Large-$S$ exact diagonalization.** For a two-spin cluster with random exchange in a random field, a noncollinear configuration with nonzero cubic and tadpole terms, the $S^0$ coefficient extracted from exact energies at $S=8,\dots,24$ agrees with Eq. (7) within $10^{-6}$, and the $1/S$ corrections of both magnon energies agree with Eq. (8) within $2\times10^{-4}$ (package test `test_finite_torus_is_exact_large_s_perturbation_theory`).

## References

### Internal Documents

- Holstein–Primakoff Expansion, Real-Space Boson Hamiltonian and Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*: the leading spin operator, the circular exchange components, the BdG matrix $\mathsf H_{\mathbf k}$ and the single-ion rule whose quadratic order only is fixed.

- Stationarity of the Reference Configuration and Local Stability and the BdG Matrix in research note *LSWT classical order*: the local field $\mathbf H_I$, the torque condition used in the cancellation of the field terms, and $\mathsf H_{\mathbf 0}$ as the classical Hessian of a uniform deformation.

- Magnon Observables in research note *LSWT magnon thermodynamics*: the $1/S$ rotation of the ordered moment, which is the condensate of Eq. (5) written in classical angles.

- Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*: $\mathsf T_{\mathbf k}$, the magnon operators and the zero-point energy.

- Notation and Conventions in research note *LSWT spin Hamiltonian*: indices, system sizes and the Nambu metric $\Sigma_3$.

- `code-space/examples/verification/lswt_nonlinear_vertices_check.py`: checks 1 and 2.

- Decision records D46 (nonlinear spin-wave expansion) and D47 (next order of the pseudo-Goldstone gap) in research note *Toolkit decisions and references*.

### External Sources

- [@holsteinField1940]: the exact boson representation expanded in Eq. (1).

- [@hamerThirdorder1992]: the $S^0$ term $-0.012474\,J$ of the square-lattice ground-state energy per site used in check 3.

- [@chernyshevSpin2009]: cubic source and decay vertices of a noncollinear state, their second-order self-energies, and the ground-state energy coefficient $0.02141$ of Eq. (42) used in check 4.

- [@zhitomirskyColloquium2013]: kinematics of two-magnon decays and the singularities of the cubic self-energy.

- [@rauPseudoGoldstone2018]: pseudo-Goldstone gaps produced by quantum order by disorder.
