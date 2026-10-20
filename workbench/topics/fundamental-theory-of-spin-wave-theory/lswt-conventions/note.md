<!-- Moved on 2026-10-20 from research note LSWT spin Hamiltonian (section Notation and Conventions; user comment c-20261019-1006); the typography and legacy-mapping parts went to workbench/rules/lswt-notation.md. Originally converted on 2026-10-08 from archive/2026-10-08-lswt-markdown/00-foundations/notation-and-conventions.md (status: in-review, source-section: Summary of Notation and Symbols). Review status: accepted by the user on 2026-10-20 after comparison with the primary source, Toth–Lake and the package (workbench/tasks/2026-10-20-theory-note-review.md). -->

# Notation and Conventions

The notation below distinguishes spin sites, magnetic unit cells, interaction matrices, and bosonic operators throughout the LSWT derivation. Each symbol has a fixed meaning within its stated domain. Physical definitions and derivations are developed in the linked notes; the typography of symbols and the correspondence with the source notation are writing rules kept in `workbench/rules/lswt-notation.md`.

## Indices and System Size

| Index | Meaning | Domain |
|:---|:---|:---|
| $i,j$ | Magnetic unit-cell index | Real space |
| $\mu,\nu$ | Magnetic sublattice index | One magnetic unit cell |
| $I,J$ | Physical spin-site index | $I=(i,\mu)$, $J=(j,\nu)$ |
| $\ell$ | Representative physical link | $\ell=(I,J)\in\mathcal L$ |
| $\alpha,\beta,\gamma$ | Laboratory-frame Cartesian component | $x,y,z$ |
| $n,m$ | Positive-energy magnon-band index | Momentum space |

The indices $n,m$ are reserved for magnon bands, so a sublattice index $\mu$ remains distinct from a band index $n$. Local circular components are labeled $+,-,0$; these labels do not replace the Cartesian meanings of $\alpha,\beta,\gamma$. The local basis is defined in Classical Order and Local Frame in research note *LSWT classical order*.

The inverse temperature is

$$
\beta\equiv(k_{\mathrm B}T)^{-1}.
$$

When a Cartesian index $\beta$ also appears in the same expression or immediately adjacent discussion, the inverse temperature is written as $\beta_T$. This exception distinguishes an index from a scalar without reassigning $i,j$ as Cartesian indices. The thermodynamic entropy is denoted by $\mathcal S$ to distinguish it from the spin length $S_I$; its definition belongs to Thermodynamics in research note *LSWT magnon thermodynamics*.

System sizes are denoted by

| Symbol              | Meaning                                               |
|:--------------------|:------------------------------------------------------|
| $N_{\mathrm{uc}}$   | Number of magnetic unit cells                         |
| $N_{\mathrm{sub}}$  | Number of magnetic sublattices per magnetic unit cell |
| $N_{\mathrm{site}}$ | Total number of physical spin sites                   |
| $N_k$               | Number of discrete momentum points                    |

so that

$$
N_{\mathrm{site}}
=N_{\mathrm{uc}}N_{\mathrm{sub}},
\qquad
\dim\mathsf H_{\mathbf k}=2N_{\mathrm{sub}}.
$$

The source symbols $L$, $N$, and $m_s$ are replaced by these quantities because their meanings vary across the source sections.

## Sites and Geometry

The basis position of sublattice $\mu$ within a magnetic unit cell is $\boldsymbol\delta_\mu$. A physical site and its position are written as

$$
I=(i,\mu),
\qquad
\mathbf r_I=\mathbf R_i+\boldsymbol\delta_\mu,
$$

where $\mathbf R_i$ is a magnetic Bravais-lattice vector. Equal classical spin angles alone do not identify two sites as the same magnetic sublattice; the magnetic translation pattern and local environment also enter the sublattice assignment.

For a two-dimensional lattice, the primitive vectors $\mathbf a_1,\mathbf a_2$ and reciprocal vectors $\mathbf b_1,\mathbf b_2$ satisfy

$$
\mathbf a_r\cdot\mathbf b_s=2\pi\delta_{rs},
\qquad r,s\in\{1,2\}.
$$

Real-space vectors share one Cartesian coordinate system, and momenta are expressed in the corresponding reciprocal Cartesian coordinates.

## Links and Bond Orientation

The set $\mathcal L$ contains each physical inter-site bond once. The representative orientation $\ell=(I,J)$ fixes the bond displacement and the ordering of the exchange matrix. The reverse orientation $\bar\ell=(J,I)$ is not separately included in $\mathcal L$.

The displacement from the first endpoint to the second is

$$
\boldsymbol\Delta_\ell
=\mathbf r_J-\mathbf r_I
=(\mathbf R_j-\mathbf R_i)
 +(\boldsymbol\delta_\nu-\boldsymbol\delta_\mu),
\qquad
\ell=((i,\mu),(j,\nu)).
$$

The laboratory-frame exchange matrix is $\mathsf J_\ell$, with Cartesian components $J_\ell^{\alpha\beta}$. Reversing the bond orientation gives

$$
\mathsf J_{\bar\ell}=\mathsf J_\ell^{\mathsf T},
\qquad
J_{JI}^{\beta\alpha}=J_{IJ}^{\alpha\beta}.
$$

This relation uses the real Cartesian exchange matrices of the bilinear model. In the local circular basis it becomes $\widetilde J_{\bar\ell}^{qp}=\widetilde J_\ell^{pq}$, with the components defined in Real-Space Boson Hamiltonian in research note *LSWT boson Hamiltonian*. The relation between the link sum and an ordered-pair sum, together with the separate treatment of on-site interactions, is given in Bilinear Spin Hamiltonian in research note *LSWT spin Hamiltonian*.

## Spins and Local Frames

| Symbol | Meaning |
|:---|:---|
| $S_I$ | Spin length at site $I$ |
| $\mathbf n_I$ | Classical spin unit direction |
| $\mathbf S_I=S_I\mathbf n_I$ | Classical spin vector |
| $\hat{\mathbf S}_I$ | Laboratory-frame quantum spin operator |
| $\hat{\widetilde{\mathbf S}}_I$ | Local-frame quantum spin operator |
| $\mathsf R_I$ | Rotation from local components to laboratory components |

We use the rotation convention

$$
\hat{\mathbf S}_I
=\mathsf R_I\hat{\widetilde{\mathbf S}}_I,
\qquad
\mathsf R_I^{\mathsf T}\mathsf R_I=\mathsf I_3,
\qquad
\det\mathsf R_I=+1.
$$

The columns of $\mathsf R_I$ are the local basis vectors expressed in laboratory coordinates; they form a right-handed frame. The rotation must be proper: with $\det\mathsf R_I=-1$ the local components would obey $[\hat{\widetilde S}{}^x_I,\hat{\widetilde S}{}^y_I]=-\mathrm i\hat{\widetilde S}{}^z_I$, which is not the spin algebra on which the Holstein–Primakoff representation is built. The exchange matrix in the local frames is

$$
\widetilde{\mathsf J}_{IJ}
=\mathsf R_I^{\mathsf T}\mathsf J_{IJ}\mathsf R_J.
$$

The polar angle $\theta_I$ is measured from the positive $z$ axis, and the azimuthal angle $\phi_I$ from the positive $x$ axis. For a general configuration, $\mathsf R_I=\mathsf R_{i\mu}$ depends on the site. For periodic order with magnetic translation symmetry,

$$
\mathsf R_{i\mu}=\mathsf R_\mu.
$$

Accordingly, real-space expressions use $\mathsf R_I$, while the momentum-space derivation uses the sublattice rotation $\mathsf R_\mu$.

## Boson and Magnon Operators

A Fourier transform changes the representation of a boson operator without changing its name. The symbol changes from $a$ to $b$ only when a Bogoliubov transformation replaces the sublattice basis by the magnon-band basis:

$$
\hat a_{i\mu}
\;\xrightarrow{\text{Fourier}}\;
\hat a_{\mathbf k\mu}
\;\xrightarrow{\text{Bogoliubov}}\;
\hat b_{\mathbf k n}.
$$

| Symbol | Meaning |
|:---|:---|
| $\hat a_{i\mu}$ | Real-space Holstein–Primakoff boson |
| $\hat a_{\mathbf k\mu}$ | Momentum-space representation of the same boson |
| $\hat b_{\mathbf k n}$ | Positive-energy magnon annihilation operator |
| $\hat n_{i\mu}=\hat a_{i\mu}^\dagger\hat a_{i\mu}$ | Local Holstein–Primakoff boson-number operator |
| $n_{\mathrm B}(\varepsilon)$ | Bose–Einstein distribution function |

The Nambu spinors before and after diagonalization are

$$
\hat\Psi_{\mathbf k}
=\begin{pmatrix}
\hat{\mathbf a}_{\mathbf k}\\
\hat{\mathbf a}_{-\mathbf k}^\dagger
\end{pmatrix},
\qquad
\hat\Phi_{\mathbf k}
=\begin{pmatrix}
\hat{\mathbf b}_{\mathbf k}\\
\hat{\mathbf b}_{-\mathbf k}^\dagger
\end{pmatrix}.
$$

Here, $\hat{\mathbf a}_{\mathbf k}$ is a column vector in sublattice order, and $\hat{\mathbf b}_{\mathbf k}$ is a column vector in positive-energy band order. The Fourier phase and the treatment of basis positions are specified in Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*, subject to the convention limits listed below.

## BdG Matrices and the Nambu Metric

| Symbol | Meaning |
|:---|:---|
| $\mathsf H_{\mathbf k}$ | $2N_{\mathrm{sub}}\times2N_{\mathrm{sub}}$ bosonic BdG matrix |
| $\mathsf A_{\mathbf k}$, $\mathsf B_{\mathbf k}$ | Normal and anomalous BdG blocks |
| $\mathsf T_{\mathbf k}$ | Paraunitary Bogoliubov transformation matrix |
| $\Sigma_3$ | Nambu particle–hole signature metric |
| $\mathsf E_{\mathbf k}$ | Diagonal Nambu energy matrix $\mathsf T_{\mathbf k}^\dagger\mathsf H_{\mathbf k}\mathsf T_{\mathbf k}$, of dimension $2N_{\mathrm{sub}}$ |
| $\lambda_{p\mathbf k}$ | Signed eigenvalue of $\Sigma_3\mathsf H_{\mathbf k}$, $p=1,\ldots,2N_{\mathrm{sub}}$ |
| $\Lambda_{\mathbf G}$ | Sublattice phase matrix relating Nambu quantities at $\mathbf k$ and $\mathbf k+\mathbf G$ |
| $\mathsf V^\alpha$ | Spin vertex: row vector with $2N_{\mathrm{sub}}$ entries, independent of momentum |

For the particle-then-hole ordering above, the Nambu metric is

$$
\Sigma_3
\equiv
\begin{pmatrix}
\mathsf I_{N_{\mathrm{sub}}} & 0\\
0 & -\mathsf I_{N_{\mathrm{sub}}}
\end{pmatrix}
=\sigma_3\otimes\mathsf I_{N_{\mathrm{sub}}}.
$$

The subscript $3$ refers to the Pauli matrix $\sigma_3$, not to a matrix dimension or power. The metric extends its particle–hole grading to the $2N_{\mathrm{sub}}$-dimensional Nambu space. The uppercase symbol distinguishes this metric from the two-dimensional Pauli matrix and avoids using the exchange-matrix symbol $\mathsf J$ for a second purpose. The unindexed symbol $\Sigma$ is reserved for a possible magnon self-energy.

The metric assigns $+1$ to the particle block and $-1$ to the hole block, and satisfies

$$
\Sigma_3^\dagger=\Sigma_3,
\qquad
\Sigma_3^2=\mathsf I_{2N_{\mathrm{sub}}}.
$$

The diagonal energy matrix contains the positive magnon energies in both blocks,

$$
\mathsf E_{\mathbf k}
=\operatorname{diag}\left(\varepsilon_{1\mathbf k},\ldots,\varepsilon_{N_{\mathrm{sub}}\mathbf k},\varepsilon_{1,-\mathbf k},\ldots,\varepsilon_{N_{\mathrm{sub}},-\mathbf k}\right),
\qquad
\lambda_{p\mathbf k}=(\Sigma_3\mathsf E_{\mathbf k})_{pp},
$$

as derived in Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*. No separate symbol is used for the $N_{\mathrm{sub}}\times N_{\mathrm{sub}}$ matrix of positive energies at one momentum. The spin vertex $\mathsf V^\alpha$ expresses the transverse part of the spin component $\alpha$ as $\hat S^\alpha_{\mathbf q,\perp}=\mathsf V^\alpha\hat\Psi_{\mathbf q}$; its entries are the spin-weighted laboratory components of the circular basis vectors, and it is defined in Spin Correlations in research note *LSWT spin correlations*. All momentum dependence of the transverse spin operator, including sublattice phases, resides in $\hat\Psi_{\mathbf q}$.

A paraunitary transformation preserves this metric:

$$
\mathsf T_{\mathbf k}^\dagger\Sigma_3\mathsf T_{\mathbf k}=\Sigma_3.
$$

## Energies

| Symbol                               | Meaning                            |
|:-------------------------------------|:-----------------------------------|
| $\varepsilon_{n\mathbf k}$           | Positive magnon energy in band $n$ |
| $E_{\mathrm{cl}}$                    | Total classical energy             |
| $\Delta E_{\mathrm{zp}}$             | Zero-point quantum correction      |
| $E_{\mathrm{GS}}$                    | LSWT ground-state energy           |
| $\mathcal E_X=E_X/N_{\mathrm{site}}$ | Energy per magnetic site           |

The total and per-site energies obey

$$
E_{\mathrm{GS}}=E_{\mathrm{cl}}+\Delta E_{\mathrm{zp}},
\qquad
\mathcal E_{\mathrm{GS}}=\mathcal E_{\mathrm{cl}}+\Delta\mathcal E_{\mathrm{zp}}.
$$

The symbol $e_X$ is not used for energy density, to avoid confusion with the exponential base. The source symbol $E_0$ may refer either to the total ground-state energy or to the zero-point correction; these meanings are distinguished as $E_{\mathrm{GS}}$ and $\Delta E_{\mathrm{zp}}$.

## Units

Energies share one unit: the exchange and single-ion matrices, the Zeeman-energy vector $\mathbf h_I$, the magnon energies $\varepsilon_{n\mathbf k}$ and the thermal energy $k_{\mathrm B}T$. Spin operators are dimensionless. In time-dependent expressions we set $\hbar=1$, so frequencies are measured in energy units and times in units of $\hbar$ divided by the energy unit. Positions, lattice vectors and bond displacements share one length unit, and momenta are measured in its inverse; the Berry curvature then carries the square of the length unit. The thermal Hall conductivity of a single layer has the unit $k_{\mathrm B}^2T/\hbar$ and contains no length; a three-dimensional conductivity requires an interlayer spacing, as stated in Magnon Thermal Hall Conductivity in research note *LSWT magnon topology*.

## Momenta and Fields

The momentum $\mathbf k$ labels internal magnon modes associated with the magnetic translation lattice. The momentum $\mathbf q$ denotes an external momentum transfer, as in neutron scattering. The crystallographic and magnetic Brillouin zones are abbreviated CBZ and MBZ, respectively; the ambiguous abbreviation FBZ is not used.

The applied field $\mathbf B_I$ is measured in tesla. The vector $\mathbf h_I$ is the Zeeman-energy coefficient multiplying the spin operator in the Hamiltonian. The $g$-tensor is written as $\mathsf g_I$. Their relation, including the sign convention for the moment with positive scalar $g$, is defined in Bilinear Spin Hamiltonian in research note *LSWT spin Hamiltonian*.

## Brillouin Zones and Momentum Sums

The MBZ is the Brillouin zone of the magnetic Bravais lattice. With periodic boundary conditions on $N_{\mathrm{uc}}$ magnetic unit cells, it contains $N_{\mathrm{uc}}$ allowed momenta, and sums written over $\mathbf k\in\mathrm{MBZ}$ run over these momenta. When the magnetic unit cell contains $M$ crystallographic unit cells, the MBZ has $1/M$ of the area of the CBZ, $A_{\mathrm{MBZ}}=A_{\mathrm{CBZ}}/M$, and a crystallographic momentum $\mathbf q$ corresponds to the MBZ momentum $\mathbf k=\mathbf q-\mathbf G$ for a magnetic reciprocal-lattice vector $\mathbf G$. In the full-position Fourier convention, Nambu quantities at $\mathbf q$ and at its folded partner $\mathbf k$ are related by the sublattice phase matrix $\Lambda_{\mathbf G}$ of Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*, while magnon energies are equal. Observables at an external momentum $\mathbf q$ outside the MBZ are therefore evaluated at the unfolded momentum, without folding, as in Spin Correlations in research note *LSWT spin correlations*.

Extensive quantities are sums over the $N_{\mathrm{uc}}$ momenta and the $N_{\mathrm{sub}}$ bands. Values per magnetic site divide by $N_{\mathrm{site}}=N_{\mathrm{uc}}N_{\mathrm{sub}}$, and in the thermodynamic limit

$$
\frac{1}{N_{\mathrm{uc}}}\sum_{\mathbf k\in\mathrm{MBZ}}f(\mathbf k)
\;\longrightarrow\;
\frac{1}{A_{\mathrm{MBZ}}}\int_{\mathrm{MBZ}}d^2\mathbf k\,f(\mathbf k)
=\frac{A_{\mathrm{uc}}}{(2\pi)^2}\int_{\mathrm{MBZ}}d^2\mathbf k\,f(\mathbf k),
$$

where $A_{\mathrm{uc}}=(2\pi)^2/A_{\mathrm{MBZ}}$ is the area of the magnetic unit cell. A finite mesh of $N_k$ momenta approximates the integral; the mesh need not coincide with the $N_{\mathrm{uc}}$ momenta of a particular finite system.

## Magnon Band Labels

At each momentum, the band index orders the positive magnon energies,

$$
\varepsilon_{1\mathbf k}\le\varepsilon_{2\mathbf k}\le\cdots\le\varepsilon_{N_{\mathrm{sub}}\mathbf k}.
$$

This label is defined at every momentum, but it does not follow a band through a crossing: at a degeneracy, two labels exchange the branches they follow. A band identity across degeneracies is therefore not defined by the label. Sums over all bands at fixed momentum, such as the zero-point energy and the thermodynamic quantities, do not depend on it. Quantities attached to one band, such as its Berry curvature and Chern number, are defined only for a band separated from the others by a gap on the whole MBZ, or for a group of bands separated from the rest, as in Topological Magnon Quantities in research note *LSWT magnon topology*. The hole entries of $\mathsf E_{\mathbf k}$ follow the labels at $-\mathbf k$.

## Conventions Not Yet Fixed

The following convention is used throughout but awaits acceptance. Reserving a symbol does not establish the associated physical definition.

- Acceptance of the full-position Fourier convention with the phase $\exp(\mathrm i\mathbf k\cdot\mathbf r_{i\mu})$, which Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian* adopts and the spin-correlation and topology notes use. The cell convention is not developed; its relation to the full-position convention is stated in Topological Magnon Quantities in research note *LSWT magnon topology*.

The other items formerly listed here are fixed in the sections above: the circular components of exchange matrices (Links and Bond Orientation, with the definition in Real-Space Boson Hamiltonian in research note *LSWT boson Hamiltonian*), the Brillouin-zone folding and momentum-sum normalization, the band labels, the diagonal energy matrix and the spin vertex, and the units. The normalization of the trace subtraction is fixed in Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*.

## References

### Internal Documents

- Bilinear Spin Hamiltonian in research note *LSWT spin Hamiltonian*: defines the exchange, anisotropy, Zeeman, and link-counting equations that use this notation.

- Classical Order and Local Frame in research note *LSWT classical order*: defines the rotation and local-frame conventions.

- Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*: defines the Fourier, Nambu-spinor, and BdG-block definitions.

- Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*: defines the operational use of the Nambu metric and paraunitary transformation.

- Thermodynamics in research note *LSWT magnon thermodynamics*: defines the thermodynamic definitions that use the reserved energy and entropy symbols.

### External Sources

- None.
