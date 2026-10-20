<!-- Converted from LaTeX on 2026-10-11. The LaTeX original is kept in archive/2026-10-11-lswt-latex-notes/lswt-spin-hamiltonian/. Equation numbers are fixed in this form; the original \label of each numbered equation is in the comment before it. -->

<!-- Spin-Wave Theory Overview Converted on 2026-10-08 from archive/2026-10-08-lswt-markdown/00-foundations/lswt-overview.md (status: draft, source-section: "Introduction to Spin Wave Theory: model and workflow"). Review status: accepted by the user on 2026-10-20 after comparison with the primary source, Toth–Lake and the package (workbench/tasks/2026-10-20-theory-note-review.md). -->

# Spin-Wave Theory Overview

Spin-wave theory (SWT) describes low-energy collective spin excitations and the associated quantum fluctuations about an ordered reference state. The nature of that reference state determines the appropriate formulation: local spin directions describe magnetically ordered or field-polarized states, whereas multipolar and bond-ordered phases require enlarged local or cluster reference states. Phases without a suitable semiclassical product reference require a different low-energy description.

For states described by local spin directions, SWT is most commonly formulated using SU(2) spin-coherent states and Holstein–Primakoff bosons. Linear spin-wave theory (LSWT) is the harmonic approximation obtained by retaining terms up to quadratic order in these bosons.

![Representative reference states and phases classified according to whether they admit a Holstein–Primakoff spin-coherent reference.](figures/spin-wave-theory-classification.png)

*The classification is schematic and non-exhaustive. A compatible stationary reference and a stable quadratic Hamiltonian are prerequisites for Holstein–Primakoff LSWT, whereas a small spin reduction is an a posteriori check of the harmonic approximation.*

## Holstein–Primakoff Spin-Wave Theory

For spin length $S_I$ and a local unit vector $\mathbf n_I$, the reference configuration is represented by the spin-coherent product state

$$
|\Phi_{\mathrm{cl}}[\{\mathbf n_I\}]\rangle
=
\bigotimes_I |S_I,\mathbf n_I\rangle,
\qquad
\langle\Phi_{\mathrm{cl}}|
\hat{\mathbf S}_I
|\Phi_{\mathrm{cl}}\rangle
=
S_I\mathbf n_I.
$$

The classical energy associated with this reference state is

$$
E_{\mathrm{cl}}[\{\mathbf n_I\}]
=
\langle\Phi_{\mathrm{cl}}[\{\mathbf n_I\}]|
\hat H
|\Phi_{\mathrm{cl}}[\{\mathbf n_I\}]\rangle.
$$

The product state is the semiclassical expansion point, not an assertion that the exact quantum state is unentangled or has an ordered moment of magnitude $S_I$. The Holstein–Primakoff representation maps deviations from the local directions $\mathbf n_I$ to bosons, and the quadratic Hamiltonian describes their collective normal modes as magnons. The local-frame construction and bosonic expansion are developed in Classical Order and Local Frame in research note *LSWT classical order* and Holstein–Primakoff Expansion in research note *LSWT boson Hamiltonian*.

## Validity of Holstein–Primakoff LSWT

The existence of a spin-coherent reference configuration permits a Holstein–Primakoff representation to be constructed, but it does not by itself establish that LSWT is an appropriate approximation. Its prerequisites and self-consistency check have different logical roles:

- **Reference compatibility:** the state of interest must admit a low-energy description in terms of fluctuations about a local spin-coherent configuration.

- **Stationarity:** the reference configuration must satisfy $\mathbf n_I\times\partial E_{\mathrm{cl}}/\partial\mathbf n_I=0$, so that no term linear in the fluctuation bosons remains.

- **Harmonic stability:** the quadratic expansion must define a stable bosonic problem. The precise energetic and dynamical stability conditions, including the treatment of symmetry-required zero modes, belong to Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*.

- **A posteriori self-consistency:** after diagonalization, the local spin reduction $\delta S_I=\langle\hat a_I^\dagger\hat a_I\rangle$ should satisfy $\delta S_I/S_I\ll1$ wherever the harmonic approximation is used.

Failure of a prerequisite invalidates Holstein–Primakoff LSWT about the selected reference, whereas a large spin reduction signals that neglected interaction terms may be important; their leading effects are treated in research note *LSWT nonlinear spin waves*. Neither conclusion rules out every possible spin-wave description. The momentum-space derivation developed here also assumes a commensurate reference configuration with a finite magnetic unit cell; this is a restriction of that derivation rather than a general limitation of SWT.

In two dimensions, the finite-temperature interpretation requires an additional qualification. A short-range isotropic Heisenberg model cannot sustain spontaneous ferromagnetic or antiferromagnetic long-range order at nonzero temperature [@merminAbsence1966]. The same argument applies to any continuous spin-rotation symmetry that the order breaks and the short-range Hamiltonian keeps. Consequently, a thermodynamic ordered-state interpretation of finite-temperature LSWT requires a field or anisotropy that removes every such symmetry, interlayer coupling, a finite-size or crossover scale, or another physical infrared cutoff. A uniform field removes it for a ferromagnet polarized along the field, but not for an antiferromagnet: the rotation about the field axis remains a symmetry, and the transverse order of a canted antiferromagnet in an isotropic model is again absent at nonzero temperature. Easy-plane anisotropy likewise leaves the rotation about the plane normal.

## Reference States Beyond Local Magnetic Order

States without a local magnetic direction require a different reference state. Spin-nematic states with quadrupolar order but no magnetic dipole moment provide a representative example. Flavor-wave theory enlarges the local state space and can describe quadrupolar or higher multipolar order [@munizGeneralized2014]. Bond-operator and cluster-wave theories instead expand about singlet dimers or larger local units [@sachdevBondoperator1990]. These approaches may retain a harmonic bosonic structure, but their fluctuation variables are not Holstein–Primakoff bosons about local spin directions. Multipolar moments can also coexist with magnetic dipole order, so the absence of a local spin direction is not a defining property of all multipolar phases.

A quantum spin liquid has no suitable semiclassical product reference for a local spin-wave expansion. Its low-energy description may instead involve fractionalized spinons and emergent gauge fields [@balentsSpin2010]. The use of bosonic variables alone therefore does not determine whether a formulation is a spin-wave theory; the reference state and fluctuation expansion must also be specified.

## Bilinear Spin Hamiltonians

The Hamiltonian considered in the subsequent derivation is at most bilinear in the spin operators and can be written as

$$
\hat H
=
\sum_{\ell=(I,J)\in\mathcal L}
\hat{\mathbf S}_I
\mathbin{\cdot}
\mathsf J_\ell
\mathbin{\cdot}
\hat{\mathbf S}_J
+
\sum_I
\hat{\mathbf S}_I
\mathbin{\cdot}
\mathsf D_I
\mathbin{\cdot}
\hat{\mathbf S}_I
-
\sum_I
\mathbf h_I
\mathbin{\cdot}
\hat{\mathbf S}_I.
$$

The inter-site matrix $\mathsf J_\ell$ includes isotropic Heisenberg exchange, antisymmetric Dzyaloshinskii–Moriya interaction, and symmetric anisotropic exchange, including bond-dependent Kitaev-type couplings. The on-site matrix $\mathsf D_I$ represents single-ion anisotropy, and $\mathbf h_I$ is the Zeeman-energy coefficient determined by the applied field and the material-dependent $g$-tensor. Link orientation, counting conventions, and the relation between $\mathbf h_I$, the applied field, and the $g$-tensor are defined in Bilinear Spin Hamiltonian (Section *Bilinear Spin Hamiltonian*).

Genuine higher-order interactions, such as inter-site biquadratic coupling and four-spin ring exchange, lie outside this bilinear Hamiltonian class. Spin-wave expansions can be constructed for such models [@majumdarEffects2012], but the resulting harmonic coefficients must be derived from those interactions rather than inferred from the bilinear formulas above.

## References

### Internal Documents

- Notation and Conventions in research note *LSWT conventions*: defines the symbols, indices, geometry, and matrix typography used here.

- Bilinear Spin Hamiltonian (Section *Bilinear Spin Hamiltonian*): defines the Hamiltonian support and link-counting convention.

- Classical Order and Local Frame in research note *LSWT classical order*: defines the reference configuration and local-frame convention.

- Holstein–Primakoff Expansion in research note *LSWT boson Hamiltonian*: defines the bosonic representation and harmonic truncation.

- Real-Space Boson Hamiltonian in research note *LSWT boson Hamiltonian*: derives the quadratic Hamiltonian in real space.

- Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*: defines the Fourier and Nambu conventions.

- Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*: states the stability conditions and obtains the magnon modes.

### External Sources

- [@holsteinField1940]: introduces the Holstein–Primakoff boson representation.

- [@tothLinear2015]: formulates LSWT using local spin frames and discusses its small-fluctuation regime.

- [@munizGeneralized2014]: distinguishes the SU(2) spin-coherent formulation from generalized spin-wave theories for multipolar states.

- [@balentsSpin2010]: reviews quantum spin liquids, fractionalized excitations, and emergent gauge fields.

- [@sachdevBondoperator1990]: introduces the singlet–triplet bond-operator expansion used for dimerized quantum magnets.

- [@majumdarEffects2012]: provides a spin-wave treatment of a Hamiltonian containing four-spin ring exchange.

- [@merminAbsence1966]: establishes the absence of spontaneous ferro- or antiferromagnetic long-range order at nonzero temperature in the short-range isotropic one- and two-dimensional Heisenberg model.

<!-- Bilinear Spin Hamiltonian Converted on 2026-10-08 from archive/2026-10-08-lswt-markdown/00-foundations/bilinear-spin-hamiltonian.md (status: draft, source-section: Introduction to Spin Wave Theory, Eq. (1)-Eq. (3)). Review status: accepted by the user on 2026-10-20 after comparison with the primary source, Toth–Lake and the package (workbench/tasks/2026-10-20-theory-note-review.md). -->

# Bilinear Spin Hamiltonian

Localized spins coupled by bilinear interactions and a static magnetic field provide the starting point for the linear spin-wave expansion considered here. The interaction matrices specify both exchange between distinct sites and anisotropy at a single site; the Zeeman term couples the field to the magnetic moment associated with each spin.

## Hamiltonian

We write the spin Hamiltonian as

<!-- eq 1: eq-lswt-bilinear-spin-hamiltonian -->

$$
\hat H
=\hat H_{\mathrm{ex}}+\hat H_{\mathrm Z}. \tag{1}
$$

Here, $I$ and $J$ label physical spin sites, and $\alpha,\beta\in\{x,y,z\}$ denote laboratory-frame Cartesian components. The spin operators $\hat{\mathbf S}_I$ are dimensionless, with angular momentum measured in units of $\hbar$; $S_I$ denotes the spin quantum number at site $I$. Accordingly, the exchange and anisotropy matrices and the Zeeman coefficient $\mathbf h_I$ have units of energy. Site and matrix notation follows Notation and Conventions in research note *LSWT conventions*.

## Exchange Interactions and Single-Ion Anisotropy

The bilinear part of the Hamiltonian separates interactions between distinct sites from on-site anisotropy:

<!-- eq 2: eq-lswt-bilinear-exchange-support -->

$$
\hat H_{\mathrm{ex}}
=
\sum_{\ell=(I,J)\in\mathcal L}
\sum_{\alpha,\beta}
\hat S_I^{\alpha}J_{\ell}^{\alpha\beta}\hat S_J^{\beta}
+
\sum_I
\sum_{\alpha,\beta}
\hat S_I^\alpha D_I^{\alpha\beta}\hat S_I^\beta. \tag{2}
$$

The link set $\mathcal L$ contains one representative orientation of each physical inter-site coupling, with $I\ne J$. The matrix $\mathsf J_\ell$ describes the coupling on link $\ell$, whereas $\mathsf D_I$ describes anisotropy at site $I$. On-site terms are counted by the separate site sum and are not included in $\mathcal L$.

We restrict the interaction matrices to

$$
\mathsf J_\ell\in\mathbb R^{3\times3},
\qquad
\mathsf D_I\in\mathbb R^{3\times3},
\qquad
\mathsf D_I^{\mathsf T}=\mathsf D_I.
$$

Spin components at different sites commute, so a real $\mathsf J_\ell$ gives a Hermitian bond interaction even when the matrix is not symmetric. At the same site, the operator order matters. Symmetry of $\mathsf D_I$ makes the on-site term equal to its explicitly symmetrized form,

$$
\sum_{\alpha,\beta}
\hat S_I^\alpha D_I^{\alpha\beta}\hat S_I^\beta
=\frac12\sum_{\alpha,\beta}D_I^{\alpha\beta}
\left(\hat S_I^\alpha\hat S_I^\beta+\hat S_I^\beta\hat S_I^\alpha\right),
$$

which is Hermitian for real $D_I^{\alpha\beta}$.

### Link Counting

An equivalent sum over independent site indices contains both orientations of every coupled pair. For a representative link $\ell=(I,J)$, define $\mathsf J_{IJ}=\mathsf J_\ell$ and $\mathsf J_{JI}=\mathsf J_{IJ}^{\mathsf T}$, and set $\mathsf J_{IJ}=0$ for uncoupled pairs. Then

<!-- eq 3: eq-lswt-link-sum-equivalence -->

$$
\sum_{\ell=(I,J)\in\mathcal{L}}
\sum_{\alpha,\beta}
\hat{S}^{\alpha}_{I}J_{\ell}^{\alpha\beta}\hat{S}^{\beta}_{J}
=
\frac{1}{2}
\sum_{I\ne J}
\sum_{\alpha,\beta}
\hat{S}^{\alpha}_{I}J_{IJ}^{\alpha\beta}\hat{S}^{\beta}_{J}. \tag{3}
$$

The reverse-pair contribution is the same operator as the forward-pair contribution: substituting $J_{JI}^{\alpha\beta}=J_{IJ}^{\beta\alpha}$, interchanging the component indices, and commuting operators at distinct sites recovers the forward term. Thus, the factor $1/2$ removes the duplication in the ordered-pair sum. It is absent from the sum over $\mathcal L$ and does not apply to the separately defined on-site term.

### Interaction Forms and Their Scope

An exchange matrix proportional to the identity describes isotropic exchange. Its antisymmetric part represents Dzyaloshinskii–Moriya exchange, while symmetric anisotropic components describe interactions such as Kitaev-type couplings. These possibilities concern the components of the same bond matrix and do not change the link-counting convention.

The on-site matrix represents quadratic single-ion anisotropy. For example, choosing $D_I^{xx}=-A$ and all other components zero gives $-A\sum_I(\hat S_I^x)^2$. For $A>0$, the local eigenvalue $-A m_x^2$ is lowest at the largest allowed $|m_x|$, where $m_x$ is an eigenvalue of $\hat S_I^x$. For $S_I\ge1$, this term distinguishes different $|m_x|$ sectors and favors the $x$ axis. For a quantum spin $S_I=1/2$, however, $(\hat S_I^x)^2=\hat{\mathbb 1}_I/4$, so this term only adds a constant and does not split the local doublet.

More generally, the spin-$1/2$ identity $\hat S_I^\alpha\hat S_I^\beta+\hat S_I^\beta\hat S_I^\alpha=\delta_{\alpha\beta}\hat{\mathbb 1}_I/2$ gives

$$
\sum_{\alpha,\beta}
\hat S_I^\alpha D_I^{\alpha\beta}\hat S_I^\beta
=\frac14\operatorname{Tr}(\mathsf D_I)\hat{\mathbb 1}_I,
\qquad S_I=\frac12,
$$

where $\hat{\mathbb 1}_I$ is the identity on the local spin space. Within this spin-$1/2$ model, exchange anisotropy and an anisotropic $g$-tensor can remain nontrivial even though real symmetric quadratic on-site anisotropy contributes only a constant.

Independent higher-order spin interactions, such as scalar spin chirality or ring exchange, are outside the bilinear starting Hamiltonian defined here. This restriction specifies the model used in the following derivation, rather than the scope of spin-wave theory as a whole.

## Zeeman Term and g-Tensor

The scalar $g$-factor and the Bohr magneton $\mu_B$ are taken to be positive. We use the convention common in spin-wave calculations in which the magnetic moment at site $I$ is written, with spin operators in units of $\hbar$, as

$$
\hat{\boldsymbol\mu}_I=\mu_B\mathsf g_I\hat{\mathbf S}_I,
$$

so that a field favors spins parallel to it. For electron spins the physical moment is $-\mu_B\mathsf g_I\hat{\mathbf S}_I$. The Hamiltonian below at field $\mathbf B$ therefore equals the electron Hamiltonian at field $-\mathbf B$. Time reversal maps the electron Hamiltonian at $-\mathbf B$ to the electron Hamiltonian at $\mathbf B$, reversing every classical spin direction, $\mathbf n_I\to-\mathbf n_I$, and every magnon momentum, $\mathbf k\to-\mathbf k$. The two conventions therefore give the same many-body spectrum, free energy and magnetization at a given field, and their magnon dispersions are related by $\varepsilon_{n\mathbf k}\leftrightarrow\varepsilon_{n,-\mathbf k}$, so a nonreciprocal part of the dispersion, odd in $\mathbf k$, changes sign. Quantities odd under time reversal that are not defined through the moment operator, such as the sign of the thermal Hall conductivity or of a magnon chirality, refer to the reversed field and change sign when expressed in terms of the physical field of an electron system.

Here, $\mathsf g_I$ is the dimensionless $g$-tensor. In the isotropic case, $\mathsf g_I=g_I\mathsf I_3$ with $g_I>0$, where $\mathsf I_3$ is the $3\times3$ identity matrix. For an anisotropic coupling, positive $g$ values refer to the principal-axis coupling magnitudes; positivity is not an entrywise condition on the tensor in an arbitrary Cartesian basis.

The Zeeman energy is $-\sum_I\mathbf B_I\mathbin{\cdot}\hat{\boldsymbol\mu}_I$. Retaining the definition of $\mathbf h_I$ as the coefficient in $-\mathbf h_I\mathbin{\cdot}\hat{\mathbf S}_I$ gives

<!-- eq 4: eq-lswt-zeeman-g-tensor -->

$$
\hat H_{\mathrm Z}
=-\sum_I\mathbf h_I\mathbin{\cdot}\hat{\mathbf S}_I
=-\mu_B
\sum_I
\sum_{\alpha,\beta}
B_I^\alpha g_I^{\alpha\beta}\hat{S}_I^\beta. \tag{4}
$$

The first index of $g_I^{\alpha\beta}$ labels the field component, and the second labels the spin component. Collecting the coefficient of each spin operator therefore yields

$$
h_I^\beta=\mu_B\sum_\alpha B_I^\alpha g_I^{\alpha\beta},
\qquad
\mathbf h_I=\mu_B\mathsf g_I^{\mathsf T}\mathbf B_I.
$$

The transpose follows from the column-vector convention. The vector $\mathbf h_I$ has units of energy and is real for a Hermitian Zeeman term; it is distinct from the applied field $\mathbf B_I$, measured in tesla. In meV and tesla, $\mu_B\simeq0.0578838\,\mathrm{meV/T}$. In particular, an isotropic positive $g_I$ gives $\mathbf h_I=g_I\mu_B\mathbf B_I$ with the stated definition of $\mathbf h_I$.

Spin–orbit coupling and crystal-field effects can make $\mathsf g_I$ anisotropic. For a free electron, the positive spin $g$-factor is approximately $2.0023$; the electron sign is absorbed into the convention for the moment stated above.

## References

### Internal Documents

- Notation and Conventions in research note *LSWT conventions*: defines the site, link, component, and matrix notation used here.

- Classical Order and Local Frame in research note *LSWT classical order*: defines the local-frame rotation of exchange matrices and fields.

- Real-Space Boson Hamiltonian in research note *LSWT boson Hamiltonian*: develops the quadratic boson Hamiltonian generated from this spin model.

### External Sources

- [@tothLinear2015]: Sec. II describes exchange tensors and single-ion anisotropy; Eq. (11) gives the transposed reverse-bond relation. Its Eq. (1) writes the Zeeman term as $+\mu_B\mathbf H^{\mathsf T}\sum_{mi}\mathsf g_i\mathbf S_{mi}$, which has the physical electron sign and contracts the field with the first index of the $g$-tensor; the Hamiltonian here at field $\mathbf B$ therefore equals Toth–Lake Eq. (1) at field $-\mathbf B$.

- [@nistAtomicSpectroscopyZeeman]: positive electron spin $g$-factor and the sign of the Zeeman energy shift.

- [@nistCODATA2022]: values of the Bohr magneton and the free-electron $g$ factor.
