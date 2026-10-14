<!-- Converted from LaTeX on 2026-10-11. The LaTeX original is kept in archive/2026-10-11-lswt-latex-notes/lswt-classical-order/. Equation numbers are fixed in this form; the original \label of each numbered equation is in the comment before it. -->

<!-- Classical Order and Local Frame Converted on 2026-10-08 from archive/2026-10-08-lswt-markdown/00-foundations/classical-order-and-local-frame.md (status: draft, source-section: Rotations for Spin Models). Review status: draft; awaiting the user's physics and mathematics review. -->

# Classical Order and Local Frame

Linear spin-wave theory describes small quantum fluctuations about a classical spin configuration. At each site $I$, the spin direction is the unit vector

$$
\mathbf n_I
=(\sin\theta_I\cos\phi_I,\,
  \sin\theta_I\sin\phi_I,\,
  \cos\theta_I).
$$

The local quantization axis is chosen along $\mathbf n_I$. The polar angle $\theta_I$ is measured from the positive laboratory $z$ axis, and the azimuthal angle $\phi_I$ from the positive laboratory $x$ axis.

## Rotation to the Local Frame

The rotation matrix converts local spin components into laboratory components:

$$
\hat{\mathbf S}_I=\mathsf R_I\hat{\widetilde{\mathbf S}}_I,
\qquad
\mathsf R_I=\mathsf R(\theta_I,\phi_I),
$$

with

$$
\mathsf R(\theta,\phi)=
\begin{pmatrix}
\cos\theta\cos\phi & -\sin\phi & \sin\theta\cos\phi\\
\cos\theta\sin\phi &  \cos\phi & \sin\theta\sin\phi\\
-\sin\theta & 0 & \cos\theta
\end{pmatrix}.
$$

Its third column is $\mathbf n_I$, so the local longitudinal component lies along the classical spin. For a bond $\ell=(I,J)$, the exchange matrix and Zeeman-energy vector in the local frames are

$$
\widetilde{\mathsf J}_\ell
=\mathsf R_I^{\mathsf T}\mathsf J_\ell\mathsf R_J,
\qquad
\widetilde{\mathbf h}_I=\mathsf R_I^{\mathsf T}\mathbf h_I.
$$

The exchange term involves rotations at both endpoints, whereas the field term involves only the rotation at its own site. The field is written as a column vector; the equivalent row-vector expression is $\mathbf h_I^{\mathsf T}\mathsf R_I$.

## Local Circular Basis

Let $\mathbf e_x,\mathbf e_y,\mathbf e_z$ denote Cartesian coordinate unit vectors. The local circular basis, expressed in laboratory coordinates, is

$$
\mathbf e_I^\pm
=\mathsf R_I\frac{\mathbf e_x\pm\mathrm i\mathbf e_y}{\sqrt2},
\qquad
\mathbf e_I^0=\mathsf R_I\mathbf e_z=\mathbf n_I.
$$

The superscript $0$ identifies the longitudinal direction, and $\pm$ label the transverse circular vectors. These are classical basis vectors, so they carry no operator hats. The phase convention is explicit in their definitions. Exchange matrices are contracted with these vectors bilinearly, without complex conjugation, as defined in Real-Space Boson Hamiltonian in research note *LSWT boson Hamiltonian*.

## Stationarity of the Reference Configuration

The classical energy $E_{\mathrm{cl}}$ is the energy of the bilinear model with classical spins $\mathbf S_I=S_I\mathbf n_I$ and with the coherent-state value $S_I(S_I-\tfrac12)\mathbf n_I^{\mathsf T}\mathsf D_I\mathbf n_I+\tfrac{S_I}{2}\operatorname{Tr}\mathsf D_I$ of the single-ion term. Its gradient with respect to the classical spin defines the local field

$$
\mathbf H_I
=-\frac{\partial E_{\mathrm{cl}}}{\partial\mathbf S_I}
=\mathbf h_I-\sum_{\ell\in\partial I}S_{I'_\ell}\mathsf J_{II'_\ell}\mathbf n_{I'_\ell}-2\Big(S_I-\frac12\Big)\mathsf D_I\mathbf n_I,
$$

where $\partial I$ is the set of links incident on $I$, $I'_\ell$ is the opposite endpoint, and $\mathsf J_{II'}$ is oriented as in Link Counting in research note *LSWT spin Hamiltonian*. Under the fixed-length constraint, the configuration is stationary when every spin is parallel or antiparallel to its local field,

$$
\mathbf n_I\times\mathbf H_I=0\qquad\text{for every site }I.
$$

This torque-free condition is the Lagrange-multiplier condition of Fluctuations and Classical Stationarity in research note *LSWT boson Hamiltonian*, and it removes the terms linear in the bosons. Stationarity does not imply that the configuration is a minimum.

## Local Stability and the BdG Matrix

The quadratic boson Hamiltonian contains the second variation of the classical energy. For complex amplitudes $\alpha_I$, the classical spins

$$
\mathbf S_I(\alpha)
=\big(S_I-|\alpha_I|^2\big)\mathbf e_I^0
+\sqrt{S_I}\big(\mathbf e_I^-\alpha_I+\mathbf e_I^+\alpha_I^*\big)
$$

are real vectors with $|\mathbf S_I(\alpha)|^2=S_I^2+|\alpha_I|^4$, so the $\alpha_I$ are coordinates on the sphere of radius $S_I$ to second order. They are the classical counterparts of the Holstein–Primakoff bosons in Real-Space Boson Hamiltonian in research note *LSWT boson Hamiltonian*. Inserting $\mathbf S_I(\alpha)$ into the classical energy gives

$$
E_{\mathrm{cl}}[\alpha]
=E_{\mathrm{cl}}+E^{(1)}[\alpha]+E^{(2)}[\alpha]+O(|\alpha|^3),
\qquad
E^{(2)}[\alpha]=\hat H_2\big|_{\hat a_I\to\alpha_I}
=\frac12\sum_{\mathbf k}\Psi_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\Psi_{\mathbf k},
$$

where $\Psi_{\mathbf k}$ collects the Fourier amplitudes $\alpha_{\mathbf k\mu}$ and $\alpha_{-\mathbf k\mu}^*$ in the Nambu order. The second form carries no trace term because the amplitudes commute. The identity holds for any configuration, stationary or not, and for the single-ion term it holds with the coherent-state energy, whose second variation carries the factor $1-1/(2S_I)$ of the boson rule. The BdG matrix $\mathsf H_{\mathbf k}$ is therefore the classical Hessian in Bloch form, and its entries are of order $S$.

At a stationary configuration, $E^{(1)}=0$, and three cases follow:

- If $\mathsf H_{\mathbf k}$ is positive definite at every momentum, the configuration is a strict local minimum against small deformations, and LSWT about it is defined.

- If $\mathsf H_{\mathbf k}$ has a negative direction at some momentum, the configuration is a saddle point. No paraunitary transformation brings $\mathsf H_{\mathbf k}$ to a positive diagonal form, as shown in Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*, and the quadratic Hamiltonian has no ground state. LSWT about this configuration is undefined; complex or wrong-signature eigenvalues of $\Sigma_3\mathsf H_{\mathbf k}$ signal the instability and are not magnon energies.

- If $\mathsf H_{\mathbf k}$ is positive semidefinite with zero directions, the harmonic order does not decide stability along them. Zero directions arise from continuous symmetries of the Hamiltonian broken by the configuration (Goldstone modes) or from accidental classical degeneracies; the latter are decided only by higher orders or by fluctuations.

Positive semidefiniteness is a condition at every momentum of the Brillouin zone, and a test on a finite set of momenta can miss a negative direction between them. A negative direction next to a zero mode is the case a finite set of momenta resolves worst. Let $\mathsf H_{\mathbf k_0}$ have the null space spanned by the orthonormal columns of $\mathsf V_0$, and let $\mathsf W$ and the diagonal matrix $\mathsf D$ hold the remaining eigenvectors and eigenvalues. For $\mathbf k=\mathbf k_0+q\hat{\mathbf u}$ with a unit vector $\hat{\mathbf u}$, degenerate perturbation theory gives the eigenvalues of $\mathsf H_{\mathbf k}$ that vanish at $q=0$ as the eigenvalues of

$$
q\,\mathsf L(\hat{\mathbf u})+q^2\,\mathsf M(\hat{\mathbf u})+O(q^3),
\qquad
\mathsf L(\hat{\mathbf u})=\mathsf V_0^\dagger(\hat{\mathbf u}\cdot\nabla_{\mathbf k}\mathsf H)\mathsf V_0,
$$

$$
\mathsf M(\hat{\mathbf u})
=\frac12\mathsf V_0^\dagger\big[(\hat{\mathbf u}\cdot\nabla_{\mathbf k})^2\mathsf H\big]\mathsf V_0
-\mathsf V_0^\dagger(\hat{\mathbf u}\cdot\nabla_{\mathbf k}\mathsf H)\mathsf W\mathsf D^{-1}\mathsf W^\dagger(\hat{\mathbf u}\cdot\nabla_{\mathbf k}\mathsf H)\mathsf V_0,
$$

with all derivatives at $\mathbf k_0$; the second form of $\mathsf M$ is the standard second-order correction when $\mathsf L$ vanishes. A nonzero $\mathsf L(\hat{\mathbf u})$ is Hermitian and changes sign with $\hat{\mathbf u}$, so it gives a negative eigenvalue on one side of $\mathbf k_0$. With $\mathsf L=0$, the lowest eigenvalue of $\mathsf H_{\mathbf k}$ near $\mathbf k_0$ is $q^2\rho(\hat{\mathbf u})$ to leading order, where $\rho(\hat{\mathbf u})$ is the lowest eigenvalue of $\mathsf M(\hat{\mathbf u})$ and plays the role of a stiffness of the zero mode. A negative $\rho(\hat{\mathbf u})$ is a long-wavelength instability: $\mathsf H_{\mathbf k}$ has a negative direction for every small $q$ along $\hat{\mathbf u}$, with an eigenvalue of size $|\rho|q^2$, which is below any fixed numerical tolerance near $\mathbf k_0$. A $\rho(\hat{\mathbf u})$ that vanishes leaves the direction to higher orders, as does a zero direction at $\mathbf k_0=0$ beyond those of the broken continuous symmetries. For example, the polarized state of the square-lattice ferromagnet with nearest-neighbour coupling $J_1<0$ and an antiferromagnetic coupling $J_3>0$ between spins two lattice constants apart along $x$ has, for a flat spiral of wave vector $q\hat{\mathbf x}$, the classical energy per spin $E(q)=S^2(J_1\cos q+J_3\cos2q)$ plus the $q$-independent energy of the bonds along $y$. The pairing blocks of this collinear state vanish, so $\mathsf H_{q\hat{\mathbf x}}$ is the identity times the magnon energy $2[E(q)-E(0)]/S$, and $\rho(\hat{\mathbf x})=S(|J_1|-4J_3)$; for $J_3$ slightly above $|J_1|/4$, the instability is confined to $|q|<q_c$ with $q_c\to0$, and the Goldstone mode at $\mathbf k_0=0$ conceals it from any finite momentum mesh that does not contain points with $|q|<q_c$.

Zero directions left open at harmonic order are decided in two different ways. A vanishing $\rho(\hat{\mathbf u})$ is still a harmonic statement: $\mathsf H_{\mathbf k}$ is exact in $\mathbf k$, so the lowest eigenvalue along $\hat{\mathbf u}$ at small $q$ is given by a higher power of $q$. In the example above at $J_3=|J_1|/4$, $E(q)-E(0)=S^2|J_1|q^4/8+O(q^6)$, so the magnon energy is $S|J_1|q^4/4$ to leading order and the polarized state is a local minimum. An accidental zero direction at $\mathbf k_0=0$ is a statement about the amplitude instead. Let $\hat{\mathbf u}$ be a unit vector in the accidental null space of the classical Hessian, orthogonal to the generators of the global rotations that leave the energy invariant, and let $y$ run over the directions with positive curvature. The relaxed energy

$$
E_{\mathrm{eff}}(a)=\min_y E_{\mathrm{cl}}[a\hat{\mathbf u}+y]-E_{\mathrm{cl}}
=c_3a^3+c_4a^4+O(a^5)
$$

contains in $c_4$ the quartic term of $E_{\mathrm{cl}}$ along $\hat{\mathbf u}$ and the negative correction from the cubic couplings of $\hat{\mathbf u}$ to the stiff directions. A nonzero $c_3$ lowers the energy on one side, so the configuration is a saddle point; with $c_3=0$, the configuration is a local minimum along $\hat{\mathbf u}$ when $c_4>0$ and a saddle point when $c_4<0$. If $E_{\mathrm{eff}}$ vanishes to all orders, the configuration lies on a classically degenerate manifold. It is then not a strict minimum, and quantum or thermal fluctuations select among the degenerate configurations (order by disorder). A zero direction at a commensurate $\mathbf k_0\neq0$ is at $\mathbf k=0$ of the supercell that folds $\mathbf k_0$ to the zone centre and is treated there in the same way.

For example, on the square lattice with exchange $J_\perp(S^xS^x+S^yS^y)+J_zS^zS^z$ between nearest neighbours and a field $h$ along $z$, a uniform tilt by $\theta$ of the polarized ferromagnet with $J_\perp=-1$ and $J_z=-1+\delta$ costs $2S^2\delta\cos^2\theta-hS\cos\theta$ per spin. At $h=4S\delta$, the quadratic term vanishes and $E_{\mathrm{eff}}=S^2\delta\theta^4/2+O(\theta^6)$, so the state is a local minimum at its critical field. The polarized Heisenberg antiferromagnet at the saturation field $h=8JS$ has its zero mode at $(\pi,\pi)$; canting the two sublattices by $\theta$ costs $2JS^2\cos2\theta-hS\cos\theta$ per spin, which rises as $JS^2\theta^4$. The Néel state with an easy-axis single-ion term instead has $c_4<0$ at the field where its zone-centre mode softens, in accordance with a first-order spin flop. The Néel state of the easy-axis XXZ antiferromagnet at $h=4S\sqrt{J_z^2-J_\perp^2}$, the triangular Heisenberg antiferromagnet in a field and the three-sublattice NBCP state with $J_{PD}$ have relaxed energies that are flat within numerical precision up to rotations of $0.2$ rad, as expected for classically degenerate manifolds.

## Single-Q Spirals in a Rotating Frame

A single-$Q$ spiral rotates the spins of successive unit cells about a common axis. With the unit axis vector $\mathbf c$, the right-handed rotation $\mathsf R_{\mathbf c}(\phi)$ by the angle $\phi$ about it, and the directions $\mathbf d_\mu$ of the sublattices in the cell $\mathbf R_i=0$, the configuration is

$$
\mathbf n_{i\mu}=\mathsf R_{\mathbf c}(\mathbf Q\cdot\mathbf R_i)\,\mathbf d_\mu .
$$

The phase uses the cell vector $\mathbf R_i$; the same configuration written with the full position, $\mathsf R_{\mathbf c}(\mathbf Q\cdot\mathbf r_{i\mu})\mathbf d'_\mu$, has $\mathbf d'_\mu=\mathsf R_{\mathbf c}(-\mathbf Q\cdot\boldsymbol\delta_\mu)\mathbf d_\mu$. The wave vector $\mathbf Q$ may be incommensurate with the lattice, in which case no finite magnetic unit cell exists.

The rotating frame removes the cell dependence. With $\hat{\mathbf S}_{i\mu}=\mathsf R_{\mathbf c}(\mathbf Q\cdot\mathbf R_i)\hat{\mathbf S}'_{i\mu}$, a link from $(i,\mu)$ to $(j,\nu)$ carries the exchange matrix $\mathsf R_{\mathbf c}(\mathbf Q\cdot\mathbf R_i)^{\mathsf T}\mathsf J_\ell\mathsf R_{\mathbf c}(\mathbf Q\cdot\mathbf R_j)$. Its dependence on $\mathbf R_i$ follows from the circular basis $\mathbf c^\pm$, $\mathbf c^0=\mathbf c$ about the spiral axis, which the rotation multiplies by phases, $\mathsf R_{\mathbf c}(\phi)\mathbf c^\pm=\exp(\mp\mathrm i\phi)\mathbf c^\pm$. The component of the rotated matrix between $\mathbf c^p$ and $\mathbf c^q$ is

$$
(\mathbf c^p)^{\mathsf T}\mathsf R_{\mathbf c}(\phi_i)^{\mathsf T}\mathsf J_\ell\mathsf R_{\mathbf c}(\phi_j)\mathbf c^q
=\exp\!\big(-\mathrm i(s_p+s_q)\phi_i\big)\exp\!\big(-\mathrm i s_q\mathbf Q\cdot\boldsymbol d_\ell\big)\,(\mathbf c^p)^{\mathsf T}\mathsf J_\ell\mathbf c^q,
$$

with $\phi_i=\mathbf Q\cdot\mathbf R_i$, the cell displacement $\boldsymbol d_\ell=\mathbf R_j-\mathbf R_i$, and $s_\pm=\pm1$, $s_0=0$. The cell dependence disappears for every component exactly when $\mathsf J_\ell$ has no components with $s_p+s_q\ne0$, that is, when $\mathsf J_\ell$ commutes with all rotations about $\mathbf c$. Such matrices have the form

$$
\mathsf J_\ell=a_\ell\big(\mathsf I_3-\mathbf c\mathbf c^{\mathsf T}\big)+b_\ell\,\mathbf c\mathbf c^{\mathsf T}+d_\ell\,[\mathbf c]_\times ,
$$

an exchange of XXZ type about $\mathbf c$ together with a Dzyaloshinskii–Moriya vector along $\mathbf c$, where $[\mathbf c]_\times$ is the antisymmetric matrix of the cross product with $\mathbf c$. A symmetric single-ion matrix must then have the first two terms only, and the Zeeman-energy vector must be parallel to $\mathbf c$. Under these conditions the rotating-frame exchange is

$$
\mathsf J'_\ell=\mathsf J_\ell\,\mathsf R_{\mathbf c}(\mathbf Q\cdot\boldsymbol d_\ell),
$$

a real matrix that depends only on the link class. The rotating-frame Hamiltonian is translation invariant on the crystallographic unit cell, with the reference configuration $\mathbf d_\mu$, and the stationarity and stability conditions above and the BdG blocks of Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian* apply to it with $\mathsf J'_\ell$ in place of $\mathsf J_\ell$. The magnon momentum $\mathbf k$ is then the momentum in the rotating frame. Laboratory spin correlations combine rotating-frame correlations at $\mathbf k$ and $\mathbf k\pm\mathbf Q$, because $\mathsf R_{\mathbf c}(\phi)$ contains the harmonics $\exp(\pm\mathrm i\phi)$ and $1$ [@tothLinear2015, Eq. (40)]; their derivation in the conventions of these notes is not included here.

When $\mathsf J_\ell$ has components with $s_p+s_q=\pm1$ or $\pm2$, the rotating-frame couplings carry the harmonics $\exp(\mp\mathrm i\mathbf Q\cdot\mathbf R_i)$ or $\exp(\mp2\mathrm i\mathbf Q\cdot\mathbf R_i)$. The rotating-frame Hamiltonian is then not translation invariant, it couples rotating-frame momenta that differ by $\mathbf Q$ or $2\mathbf Q$, and the single-$Q$ spiral is in general not stationary, since the field acquires the same harmonics. A single-$Q$ spin-wave spectrum is not defined in this case. A commensurate $\mathbf Q$ can still be treated on its finite magnetic unit cell with the general construction.

### When the Rotation Condition Fails

The phase argument applies to the one-site terms in the same way. The single-ion term $\hat{\mathbf S}_{i\mu}^{\mathsf T}\mathsf A_\mu\hat{\mathbf S}_{i\mu}$ becomes $\hat{\mathbf S}'^{\mathsf T}_{i\mu}\mathsf R_{\mathbf c}(\phi_i)^{\mathsf T}\mathsf A_\mu\mathsf R_{\mathbf c}(\phi_i)\hat{\mathbf S}'_{i\mu}$, whose component between $\mathbf c^p$ and $\mathbf c^q$ is $\exp(-\mathrm i(s_p+s_q)\phi_i)(\mathbf c^p)^{\mathsf T}\mathsf A_\mu\mathbf c^q$ by the formula above with $\boldsymbol d_\ell=0$. Components with $s_p+s_q=\pm2$, an anisotropy within the plane perpendicular to $\mathbf c$, carry second harmonics; components with $s_p+s_q=\pm1$, which mix $\mathbf c$ with that plane, carry first harmonics. The Zeeman energy $-\mathbf h_\mu\cdot\hat{\mathbf S}_{i\mu}$ becomes $-\big(\mathsf R_{\mathbf c}(\phi_i)^{\mathsf T}\mathbf h_\mu\big)\cdot\hat{\mathbf S}'_{i\mu}$; the component of $\mathbf h_\mu$ along $\mathbf c$ is unchanged, and the transverse component rotates by $-\phi_i$ and carries the first harmonics $\exp(\pm\mathrm i\mathbf Q\cdot\mathbf R_i)$.

The harmonics decide whether the single-$Q$ spiral can be stationary. In the rotating frame the reference direction $\mathbf d_\mu$ is the same in every cell, so the stationarity condition of Stationarity of the Reference Configuration requires the rotating-frame local field $\mathbf h'_{i\mu}=-\mathsf R_{\mathbf c}(\phi_i)^{\mathsf T}\partial E_{\mathrm{cl}}/\partial\mathbf S_{i\mu}$ to be parallel to $\mathbf d_\mu$ in every cell. Without harmonics $\mathbf h'_{i\mu}$ does not depend on the cell, and stationarity is one condition per sublattice, met for example by the cone angle of a spiral in a field along $\mathbf c$. With harmonics $\mathbf h'_{i\mu}$ varies from cell to cell while $\mathbf d_\mu$ is fixed, so the torque $\mathbf d_\mu\times\mathbf h'_{i\mu}$ varies as well, and no cell-independent choice of $\mathbf d_\mu$ removes it in every cell. The stationary state is then a distorted spiral, whose rotation angle is not linear in $\mathbf R_i$ and whose spin directions contain further Fourier components. Two examples on a planar spiral, $\mathbf d\perp\mathbf c$, of a Bravais lattice with isotropic exchange, whose exchange torque vanishes by the example below, show the two kinds of harmonics.

- **Field in the spiral plane.** For $\mathbf h\perp\mathbf c$ at the angle $\phi_h$ in the plane, the Zeeman torque on the spin $S\mathbf n_i$ has the magnitude $S|\mathbf h|\,|\sin(\phi_i-\phi_h)|$ along $\mathbf c$, and its sign changes with the cell. The spins turn towards $\mathbf h$, so the rotation angle acquires a correction $\delta\phi_i$ that is odd in $\phi_i-\phi_h$ and has period $2\pi$ in it. To first order in $\delta\phi_i$, $\mathbf n_i$ then contains a uniform component along $\mathbf h$ and components at $\pm2\mathbf Q$, in addition to $\pm\mathbf Q$.

- **Easy axis in the spiral plane.** For $\mathsf A=-K\mathbf e\mathbf e^{\mathsf T}$ with $K>0$ and $\mathbf e\perp\mathbf c$ at the angle $\phi_e$, the single-ion energy of D37 is $-KS^2(1-\tfrac1{2S})\cos^2(\phi_i-\phi_e)$ per site, up to a constant, and its torque is proportional to $\sin2(\phi_i-\phi_e)$. The spins gather about $\pm\mathbf e$, $\delta\phi_i$ has period $\pi$ in $\phi_i-\phi_e$, and to first order $\mathbf n_i$ contains components at $\pm3\mathbf Q$ in addition to $\pm\mathbf Q$.

When the rotation condition fails, the linear spin-wave spectrum still exists for the distorted stationary state but must be computed on a magnetic unit cell. For a commensurate $\mathbf Q$, the classical minimum on the magnetic cell, found without assuming the single-$Q$ form, and the general construction of Rotation to the Local Frame give the spectrum within LSWT. For an incommensurate $\mathbf Q$, a commensurate approximant $\mathbf Q'$ close to $\mathbf Q$ gives a finite magnetic cell; the result then belongs to the approximant, and its dependence on the cell size has to be checked. A treatment that keeps the incommensurate $\mathbf Q$ and includes the harmonic couplings, which mix rotating-frame momenta $\mathbf k$, $\mathbf k\pm\mathbf Q$, $\mathbf k\pm2\mathbf Q,\ldots$, first needs the distorted reference state and is not derived in these notes.

The rotation invariance is the stated requirement of the rotating-frame method [@tothLinear2015, Eq. (13)], not a choice of the toolkit. The toolkit checks it before a rotating-frame calculation and stops with `SpiralSymmetryError` when a term breaks it, instead of returning a single-$Q$ spectrum that the derivation above does not define (decision D34). How other implementations of the method act on such models has not been checked for these notes.

### Example: Spirals on a Bravais Lattice

For one spin per cell with isotropic exchange $J_{\boldsymbol\Delta}$ on the neighbor vectors $\boldsymbol\Delta$, a planar spiral with $\mathbf d\perp\mathbf c$ is torque free for every $\mathbf Q$: the local field $-S\sum_{\boldsymbol\Delta}J_{\boldsymbol\Delta}\mathsf R_{\mathbf c}(\mathbf Q\cdot\boldsymbol\Delta)\mathbf d$ has the component $-S\sum_{\boldsymbol\Delta}J_{\boldsymbol\Delta}\sin(\mathbf Q\cdot\boldsymbol\Delta)$ perpendicular to $\mathbf d$ in the spiral plane, which vanishes because $J_{-\boldsymbol\Delta}=J_{\boldsymbol\Delta}$. With $\mathcal J_{\mathbf k}=\sum_{\boldsymbol\Delta}J_{\boldsymbol\Delta}\exp(\mathrm i\mathbf k\cdot\boldsymbol\Delta)$ summed over all neighbor vectors, which is $2J(\mathbf k)$ of the Luttinger–Tisza appendix, the circular components of $\mathsf J'$ in the local frame of $\mathbf d$ are $\widetilde J'^{00}=J_{\boldsymbol\Delta}\cos\varphi$, $\widetilde J'^{+-}=\tfrac12J_{\boldsymbol\Delta}(1+\cos\varphi)$ and $\widetilde J'^{++}=\tfrac12J_{\boldsymbol\Delta}(1-\cos\varphi)$ with $\varphi=\mathbf Q\cdot\boldsymbol\Delta$. The blocks are then

$$
A_{\mathbf k}=S\Big[\frac{\mathcal J_{\mathbf k}}2+\frac{\mathcal J_{\mathbf k+\mathbf Q}+\mathcal J_{\mathbf k-\mathbf Q}}4-\mathcal J_{\mathbf Q}\Big],
\qquad
B_{\mathbf k}=S\Big[\frac{\mathcal J_{\mathbf k}}2-\frac{\mathcal J_{\mathbf k+\mathbf Q}+\mathcal J_{\mathbf k-\mathbf Q}}4\Big],
$$

and the single-sublattice result $\varepsilon_{\mathbf k}^2=A_{\mathbf k}^2-|B_{\mathbf k}|^2$ of Worked Example in research note *LSWT paraunitary diagonalization* gives

$$
\varepsilon_{\mathbf k}^2
=S^2\big(\mathcal J_{\mathbf k}-\mathcal J_{\mathbf Q}\big)\Big(\frac{\mathcal J_{\mathbf k+\mathbf Q}+\mathcal J_{\mathbf k-\mathbf Q}}2-\mathcal J_{\mathbf Q}\Big).
$$

The two factors are, up to $S$, the eigenvalues $A_{\mathbf k}\pm B_{\mathbf k}$ of the real $2\times2$ BdG matrix. Both are nonnegative for every $\mathbf k$ exactly when $\mathbf Q$ is a global minimum of $\mathcal J_{\mathbf k}$, which is the Luttinger–Tisza condition of the appendix. For any other $\mathbf Q$, the eigenvalue $S(\mathcal J_{\mathbf k}-\mathcal J_{\mathbf Q})$ is negative at a minimizing wave vector, so the torque-free spiral is a saddle point and LSWT about it is undefined; $\varepsilon_{\mathbf k}^2$ is negative wherever exactly one of the two factors is negative. At a minimizing $\mathbf Q$ that is not half a reciprocal-lattice vector, the energy vanishes at $\mathbf k=0$ and at $\mathbf k=\pm\mathbf Q$: these are the three Goldstone modes of a noncollinear order, the rotation within the spiral plane and the two tilts of the plane. For the triangular-lattice $120^\circ$ state, $\mathcal J_{\mathbf k}=6J\gamma_{\mathbf k}$ with $\gamma_{\mathbf k}=\frac13\sum_m\cos(\mathbf k\cdot\mathbf a_m)$ and $\mathcal J_{\mathbf k\pm\mathbf Q}$ averaging to $-3J\gamma_{\mathbf k}$, so $\varepsilon_{\mathbf k}=3JS\sqrt{(1-\gamma_{\mathbf k})(1+2\gamma_{\mathbf k})}$.

## References

### Internal Documents

- Notation and Conventions in research note *LSWT spin Hamiltonian*: defines the rotation-matrix direction and local-basis notation.

- Bilinear Spin Hamiltonian in research note *LSWT spin Hamiltonian*: defines the laboratory-frame exchange matrices and Zeeman field rotated here.

- Holstein–Primakoff Expansion in research note *LSWT boson Hamiltonian*: uses the local quantization axis to introduce bosonic fluctuations and states the Lagrange-multiplier form of stationarity.

- Real-Space Boson Hamiltonian and Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*: supply the quadratic Hamiltonian whose classical form is the second variation, and the blocks used for the rotating-frame exchange.

- Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*: shows that a positive paraunitary diagonal form exists only for a positive-definite BdG matrix, and gives the single-sublattice energies used in the spiral example.

- Appendix: Luttinger–Tisza Method (Section *Appendix: Luttinger–Tisza Method*): gives the minimizing wave vectors and the function $J(\mathbf k)$.

- Verification script `code-space/examples/verification/lswt_bdg_blocks_check.py`: compares the second variation of the classical energy with the BdG quadratic form on a periodic cluster for a random non-stationary configuration with single-ion terms (relative difference $2\times10^{-8}$, the finite-difference error), and the rotating-frame blocks of a spiral with the closed form above, including the imaginary modes at a non-minimizing pitch ($2\times10^{-14}$).

- Verification script `code-space/examples/verification/lswt_spiral_harmonics_check.py`: checks the phase formula of the rotated exchange for a random matrix and axis ($7\times10^{-16}$), that the matrices commuting with the rotations about $\mathbf c$ are those without components of nonzero $s_p+s_q$, and the rotating-frame local field over 60 cells of an incommensurate triangular-lattice spiral: the same in every cell for a cone in a field along $\mathbf c$ with an axial single-ion term ($3\times10^{-14}$), and varying by $0.50$ for a field $|\mathbf h|=0.5$ in the spiral plane and by $0.15$ for an easy axis $K=0.3$ in the plane ($J=S=1$). `symmetry_violations` of the toolkit reports exactly the last two models.

### External Sources

- [@tothLinear2015]: Eqs. (3) and (6) define the single-$Q$ structure with cell-position phases and the rotating frame; Eq. (13) states the rotation invariance of the exchange matrices required for it; Eq. (21) gives the rotating-frame exchange; Eq. (40) combines the rotating-frame correlations at $\mathbf k$ and $\mathbf k\pm\mathbf Q$.

- [@lyonsMethod1960]: spiral ground states of Bravais lattices with isotropic exchange at the minimum of the exchange Fourier transform.

<!-- Appendix: Luttinger–Tisza Method Converted on 2026-10-08 from archive/2026-10-08-lswt-markdown/04-appendices/luttinger-tisza-method.md (status: draft, source-section: "None in the primary note; the restructured TeX contains only an empty TODO section (app:luttinger-tisza). Content follows Luttinger and Tisza (1946) and Lyons and Kaplan (1960)."). Review status: draft; awaiting the user's physics and mathematics review. -->

# Appendix: Luttinger–Tisza Method

LSWT expands about a classical reference configuration, and the expansion is meaningful only when that configuration minimizes the classical energy. The Luttinger–Tisza (LT) method gives a lower bound on the classical energy of a bilinear spin model by relaxing the fixed length of each spin to a single global constraint. The relaxed problem reduces to the diagonalization of a matrix in momentum space. When a configuration built from the minimizing eigenvectors also has unit spin length at every site, it attains the bound and is a classical ground state. This appendix derives the bound, the conditions for attaining it with a single wave vector, the generalization to inequivalent sublattices, and the limits of the method.

## Classical Energy as a Quadratic Form

We consider the bilinear exchange and single-ion terms of Bilinear Spin Hamiltonian in research note *LSWT spin Hamiltonian* without the Zeeman term. The classical energy is the expectation value of these terms in the spin-coherent product state with unit vectors $\mathbf n_I$, as defined in LSWT Overview in research note *LSWT spin Hamiltonian*. For the exchange terms this replaces each spin operator by the classical vector $S_I\mathbf n_I$. For the single-ion term the coherent-state value is $\langle\hat S_I^\alpha\hat S_I^\beta+\hat S_I^\beta\hat S_I^\alpha\rangle/2=S_I(S_I-\tfrac12)n_I^\alpha n_I^\beta+\tfrac{S_I}2\delta^{\alpha\beta}$, which differs from $S_I^2n_I^\alpha n_I^\beta$ at finite spin and reduces to the constant $\operatorname{Tr}\mathsf D_I/4$ at $S_I=\tfrac12$. Writing the link sum as an ordered-pair sum gives

$$
E_{\mathrm{cl}}[\{\mathbf n_I\}]
=\frac12\sum_{I\neq J}S_IS_J\,\mathbf n_I^{\mathsf T}\mathsf J_{IJ}\mathbf n_J
+\sum_IS_I\big(S_I-\tfrac12\big)\,\mathbf n_I^{\mathsf T}\mathsf D_I\mathbf n_I
+E_D,
\qquad
E_D=\sum_I\frac{S_I}{2}\operatorname{Tr}\mathsf D_I,
$$

where $\mathsf J_{JI}=\mathsf J_{IJ}^{\mathsf T}$ and the constant $E_D$ does not depend on the spin directions. Apart from $E_D$, the energy is a quadratic form in the $3N_{\mathrm{site}}$ components of the unit vectors. The ground-state problem minimizes this form under the strong constraint

$$
|\mathbf n_I|=1\quad\text{for every site }I.
$$

The LT method replaces the $N_{\mathrm{site}}$ strong constraints by the single weak constraint

$$
\sum_I|\mathbf n_I|^2=N_{\mathrm{site}},
$$

which every configuration satisfying the strong constraint also satisfies.

## Momentum-Space Form

The LT problem uses the unit cell of the Hamiltonian, with sites $I=(i,a)$ at $\mathbf r_{ia}=\mathbf R_i+\boldsymbol\delta_a$ and $a=1,\ldots,N_s$; the magnetic unit cell of the ground state is not known in advance. With the full-position Fourier convention of Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*,

$$
\mathbf n_{ia}=\frac{1}{\sqrt{N_{\mathrm{uc}}}}\sum_{\mathbf q}\exp(\mathrm i\mathbf q\cdot\mathbf r_{ia})\,\mathbf n_a(\mathbf q),
\qquad
\mathbf n_a(-\mathbf q)=\mathbf n_a(\mathbf q)^*,
$$

where the reality condition follows from real $\mathbf n_{ia}$ and the momenta run over the Brillouin zone of the Hamiltonian’s unit cell. Translation invariance makes the energy diagonal in $\mathbf q$:

<!-- eq 1: eq-lswt-lt-matrix -->

$$
E_{\mathrm{cl}}=\sum_{\mathbf q}\mathbf n(\mathbf q)^\dagger\,\mathsf L_{\mathbf q}\,\mathbf n(\mathbf q)+E_D,
\qquad
\big[\mathsf L_{\mathbf q}\big]_{ab}
=\frac12\sum_{J\in b}S_aS_b\,\mathsf J_{IJ}\exp\!\big(\mathrm i\mathbf q\cdot(\mathbf r_J-\mathbf r_I)\big)
+\delta_{ab}\,S_a\big(S_a-\tfrac12\big)\,\mathsf D_a, \tag{1}
$$

where $\mathbf n(\mathbf q)$ collects the $3N_s$ components $\mathbf n_a(\mathbf q)$, $I$ is a fixed site of sublattice $a$, and the sum runs over the sites $J\neq I$ of sublattice $b$ coupled to $I$. Each $[\mathsf L_{\mathbf q}]_{ab}$ is a $3\times3$ block. The relation $\mathsf J_{JI}=\mathsf J_{IJ}^{\mathsf T}$ makes the $3N_s\times3N_s$ LT matrix $\mathsf L_{\mathbf q}$ Hermitian, and real interaction matrices give $\mathsf L_{-\mathbf q}=\mathsf L_{\mathbf q}^*$. The spin lengths are absorbed into $\mathsf L_{\mathbf q}$, so the constraints refer to unit vectors for any $S_a$. The weak constraint becomes $\sum_{\mathbf q}\|\mathbf n(\mathbf q)\|^2=N_{\mathrm{site}}$.

## Luttinger–Tisza Lower Bound

Let $\lambda_{\min}(\mathbf q)$ be the lowest eigenvalue of $\mathsf L_{\mathbf q}$ and $\lambda_{\mathrm{LT}}=\min_{\mathbf q}\lambda_{\min}(\mathbf q)$. Each momentum component satisfies $\mathbf n(\mathbf q)^\dagger\mathsf L_{\mathbf q}\mathbf n(\mathbf q)\ge\lambda_{\mathrm{LT}}\|\mathbf n(\mathbf q)\|^2$, so every configuration obeying the weak constraint, and in particular every configuration of unit spins, satisfies

<!-- eq 2: eq-lswt-lt-bound -->

$$
\frac{E_{\mathrm{cl}}-E_D}{N_{\mathrm{site}}}\ge\lambda_{\mathrm{LT}}. \tag{2}
$$

Equality holds exactly when $\mathbf n(\mathbf q)$ vanishes except at the minimizing wave vectors $\mathbf Q$ and their negatives, and lies in the eigenspace of $\lambda_{\mathrm{LT}}$ there. A configuration of this form that also satisfies the strong constraint attains the bound and is therefore a classical ground state, with energy $N_{\mathrm{site}}\lambda_{\mathrm{LT}}+E_D$. If no such configuration exists, $\lambda_{\mathrm{LT}}$ is a strict lower bound and the method does not determine the ground state.

## Single-Wave-Vector Configurations

The simplest candidates use one minimizing wave vector $\mathbf Q$ with an eigenvector $\mathbf w=(\mathbf w_1,\ldots,\mathbf w_{N_s})$ of $\lambda_{\mathrm{LT}}$. The component at $-\mathbf Q$ is fixed by the reality condition, since $\mathbf w^*$ is an eigenvector of $\mathsf L_{-\mathbf Q}$. The configuration is

$$
\mathbf n_{ia}=\operatorname{Re}\!\big[\mathbf u_a\exp(\mathrm i\mathbf Q\cdot\mathbf R_i)\big],
\qquad
\mathbf u_a=\mathbf w_a\exp(\mathrm i\mathbf Q\cdot\boldsymbol\delta_a),
$$

with the cell amplitude $\mathbf u_a$ and an overall scale of $\mathbf w$ still free. Its squared length is

<!-- eq 3: eq-lswt-lt-single-q-length -->

$$
|\mathbf n_{ia}|^2=\frac12|\mathbf u_a|^2+\frac12\operatorname{Re}\!\big[(\mathbf u_a\cdot\mathbf u_a)\exp(2\mathrm i\mathbf Q\cdot\mathbf R_i)\big], \tag{3}
$$

where $\mathbf u_a\cdot\mathbf u_a=\sum_\gamma(u_a^\gamma)^2$ carries no complex conjugation. The phases $\exp(2\mathrm i\mathbf Q\cdot\mathbf R_i)$ over all cells form a subgroup of the unit circle, and the strong constraint takes one of three forms:

- If $2\mathbf Q$ is a reciprocal-lattice vector, as at the zone center or at half reciprocal vectors, then $\exp(\mathrm i\mathbf Q\cdot\mathbf R_i)=\pm1$, the configuration is collinear within each sublattice, and the condition is $|\operatorname{Re}\mathbf u_a|=1$ after a suitable choice of the overall phase of $\mathbf w$.

- If $4\mathbf Q$ is a reciprocal-lattice vector but $2\mathbf Q$ is not, the phases $\exp(2\mathrm i\mathbf Q\cdot\mathbf R_i)$ take only the values $\pm1$. The condition is $\operatorname{Re}(\mathbf u_a\cdot\mathbf u_a)=0$ and $|\mathbf u_a|^2=2$. Writing $\mathbf u_a=\mathbf x_a+\mathrm i\mathbf y_a$, this requires $|\mathbf x_a|=|\mathbf y_a|=1$ but leaves the angle between $\mathbf x_a$ and $\mathbf y_a$ free. The configuration cycles through $\mathbf x_a,-\mathbf y_a,-\mathbf x_a,\mathbf y_a$; for $\mathbf y_a=\pm\mathbf x_a$ it is the collinear up-up-down-down pattern.

- Otherwise the phases take at least three values that are not confined to $\pm1$, and the condition is $\mathbf u_a\cdot\mathbf u_a=0$ and $|\mathbf u_a|^2=2$. Then $\mathbf x_a$ and $\mathbf y_a$ are orthonormal, and each sublattice forms a planar spiral in the plane spanned by $\mathbf x_a$ and $\mathbf y_a$.

When the eigenspace of $\lambda_{\mathrm{LT}}$ at $\mathbf Q$ has more than one dimension, $\mathbf w$ may be any vector in it, and the conditions become equations for its coefficients. For a Bravais lattice ($N_s=1$) with isotropic Heisenberg exchange, $\mathsf L_{\mathbf q}=S^2J(\mathbf q)\,\mathsf I_3$ with $J(\mathbf q)=\frac12\sum_{\boldsymbol\Delta}J_{\boldsymbol\Delta}\exp(\mathrm i\mathbf q\cdot\boldsymbol\Delta)$, where $\boldsymbol\Delta$ runs over all neighbor vectors. Every complex vector is then an eigenvector, a planar spiral always satisfies the strong constraint, and a spiral with wave vector $\mathbf Q$ is a ground state, as shown by Lyons and Kaplan [@lyonsMethod1960]. When several inequivalent wave vectors minimize $J(\mathbf q)$, other ground states, including multiple-wave-vector states, can have the same energy. For the triangular-lattice antiferromagnet with nearest-neighbor exchange $J>0$, $J(\mathbf q)=J\sum_{m=1}^{3}\cos(\mathbf q\cdot\mathbf a_m)$ over the three bond directions has its minimum $-\frac32J$ at the zone corner, and the spiral at that wave vector is the $120^\circ$ state with energy $-\frac32JS^2$ per site.

## Inequivalent Sublattices and the Generalized Method

For $N_s>1$, the minimizing eigenvector generally distributes its weight unequally among the sublattices, $|\mathbf w_a|\neq|\mathbf w_b|$, while the strong constraint requires equal lengths. The single weak constraint then allows configurations that no unit-spin state can reach, and the bound is often not attained. A stronger bound follows from one Lagrange multiplier $\lambda_a$ per sublattice. For unit spins, $\sum_I\lambda_{a(I)}(|\mathbf n_I|^2-1)=0$, so

$$
E_{\mathrm{cl}}
=\sum_{\mathbf q}\mathbf n(\mathbf q)^\dagger\big(\mathsf L_{\mathbf q}-\Lambda\big)\mathbf n(\mathbf q)
+N_{\mathrm{uc}}\sum_{a=1}^{N_s}\lambda_a,
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_{N_s})\otimes\mathsf I_3.
$$

If $\mathsf L_{\mathbf q}-\Lambda$ is positive semidefinite at every $\mathbf q$, the first term is nonnegative and $E_{\mathrm{cl}}\ge N_{\mathrm{uc}}\sum_a\lambda_a$. The best bound of this form maximizes $\sum_a\lambda_a$ under that condition. Equal multipliers $\lambda_a=\lambda_{\mathrm{LT}}$ recover (2), so the generalized bound is never weaker. Lyons and Kaplan [@lyonsMethod1960] introduced this generalization through adjustable parameters in the weak constraint, which allows crystallographically inequivalent spins.

## Scope and Limitations

- **Zeeman field.** The Zeeman term is linear in the spin vectors, so the energy is no longer a homogeneous quadratic form, and the bound above does not include it.

- **Single-ion anisotropy.** An anisotropic $\mathsf D_I$ makes the diagonal blocks of $\mathsf L_{\mathbf q}$ distinguish spin directions, so the minimizing eigenvectors generally do not combine into unit spins; for example, easy-plane anisotropy with a spiral whose plane contains the hard axis. For such models the LT result is a lower bound and a list of candidate wave vectors rather than a solution.

- **Multiple wave vectors.** When no single-wave-vector configuration satisfies the strong constraint, a combination of several minimizing wave vectors may still do so. Failure of the single-wave-vector construction therefore does not show that the bound is unattainable.

- **Degenerate minima.** When the bound is attained, a minimum of $\lambda_{\min}(\mathbf q)$ on a line or an extended region, or a degenerate eigenspace beyond that required by a global spin-rotation symmetry, signals a degenerate manifold of classical ground states, and the choice among them is not made by the classical energy. The threefold degeneracy of $\mathsf L_{\mathbf q}$ for isotropic exchange reflects only the global rotation of all spins. When the bound is not attained, the degeneracy of the LT minimum does not determine the degeneracy of the true ground states.

## Relation to Linear Spin-Wave Theory

LSWT requires a reference configuration of unit spins that is stationary under the fixed-length constraint, as described in Classical Order and Local Frame (Section *Classical Order and Local Frame*). A configuration that attains the LT bound is a global minimum of the zero-field classical energy. Every harmonic fluctuation about it therefore raises or preserves the energy, and its BdG matrix $\mathsf H_{\mathbf k}$ is at least positive semidefinite, provided the quadratic boson Hamiltonian uses the same coherent-state value of the single-ion term as the classical energy above (Real-Space Boson Hamiltonian in research note *LSWT boson Hamiltonian*), as required in Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*. When the bound is not attained, $\lambda_{\mathrm{LT}}$ still bounds the energy of any candidate configuration obtained by numerical minimization from below, and the minimizing wave vectors indicate the magnetic unit cells to consider.

## References

### Internal Documents

- Bilinear Spin Hamiltonian in research note *LSWT spin Hamiltonian*: defines the exchange and single-ion matrices and the link counting used in the classical energy.

- Classical Order and Local Frame (Section *Classical Order and Local Frame*): defines the classical reference configuration.

- Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*: defines the full-position Fourier convention.

- Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*: states the positive-definiteness requirement on the BdG matrix.

### External Sources

- [@luttingerTheory1946]: representation of spin arrays as vectors in a many-dimensional space and reduction of the minimum-energy problem to a matrix diagonalization.

- [@lyonsMethod1960]: generalization with adjustable parameters in the weak constraint for inequivalent spins; spiral ground state for lattices of equivalent spins.
