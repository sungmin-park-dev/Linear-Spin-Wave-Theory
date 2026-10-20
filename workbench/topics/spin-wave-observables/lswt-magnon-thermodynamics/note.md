<!-- Converted from LaTeX on 2026-10-11. The LaTeX original is kept in archive/2026-10-11-lswt-latex-notes/lswt-magnon-thermodynamics/. Equation numbers are fixed in this form; the original \label of each numbered equation is in the comment before it. -->

<!-- Magnon Observables Converted on 2026-10-08 from archive/2026-10-08-lswt-markdown/02-observables/magnon-observables.md (status: draft, source-section: "Physical Quantities in Linear Spin Wave Theory (source p. 12, Table I); Number and Spin moment from Correlation Matrix (source pp. 15–16)"). Review status: draft; awaiting the user's physics and mathematics review. -->

# Magnon Observables

After the paraunitary diagonalization of Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*, every LSWT observable is built from two ingredients at each momentum of the magnetic Brillouin zone (MBZ): the magnon energies $\varepsilon_{n\mathbf k}$ and the transformation $\mathsf T_{\mathbf k}$. This document collects the quantities that follow directly from these ingredients, namely the magnon spectrum, the equal-time correlation matrix of the Holstein–Primakoff bosons, the boson number, and the reduced ordered moment. It also serves as an index to the documents that own the remaining observables.

## Magnon Spectrum and Band Structure

The magnon energies $\varepsilon_{n\mathbf k}$, $n=1,\ldots,N_{\mathrm{sub}}$, are the positive eigenvalues of $\Sigma_3\mathsf H_{\mathbf k}$ and are periodic over the MBZ. Each band is a branch $\varepsilon_{n\mathbf k}$ that continues across momentum; at crossings the band label is a matter of convention, since the energies are ordered at each momentum. The negative eigenvalues $-\varepsilon_{n,-\mathbf k}$ are the particle–hole partners and carry no additional information.

The magnetic unit cell of an ordered state may contain several crystallographic unit cells. A band structure plotted along a path of the crystallographic Brillouin zone (CBZ) then shows all $N_{\mathrm{sub}}$ bands of the magnetic cell at every momentum, including branches folded back by magnetic reciprocal-lattice vectors. Which folded branch carries weight at a given external momentum is determined by the one-magnon weights of Structure Factor and Spectral Function in research note *LSWT spin correlations*, not by the energies alone. At a momentum where $\mathsf H_{\mathbf k}$ is only positive semidefinite, some $\varepsilon_{n\mathbf k}$ vanishes; the energies remain well defined as limits, although $\mathsf T_{\mathbf k}$ may not exist there.

## Ground-State Energy

The LSWT ground-state energy $E_{\mathrm{GS}}=E_{\mathrm{cl}}+\Delta E_{\mathrm{zp}}$ and the zero-point correction $\Delta E_{\mathrm{zp}}=\frac12\sum_{\mathbf k}(\sum_n\varepsilon_{n\mathbf k}-\operatorname{Tr}\mathsf A_{\mathbf k})$ are derived with the diagonal magnon Hamiltonian in Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*. The energy per magnetic site is $\mathcal E_{\mathrm{GS}}=E_{\mathrm{GS}}/N_{\mathrm{site}}$. The correction depends on the energies alone, not on $\mathsf T_{\mathbf k}$, and its trace subtraction is the exact normalization shown in Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*.

## Equal-Time Correlation Matrix

The equal-time averages of all products of two Holstein–Primakoff bosons at momentum $\mathbf k$ form the $2N_{\mathrm{sub}}\times2N_{\mathrm{sub}}$ correlation matrix

<!-- eq 1: eq-lswt-boson-correlation-matrix -->

$$
\Big[\big\langle\hat\Psi_{\mathbf k}\hat\Psi_{\mathbf k}^\dagger\big\rangle\Big]
=\begin{pmatrix}
\langle\hat a_{\mathbf k\mu}\hat a_{\mathbf k\nu}^\dagger\rangle & \langle\hat a_{\mathbf k\mu}\hat a_{-\mathbf k\nu}\rangle\\
\langle\hat a_{-\mathbf k\mu}^\dagger\hat a_{\mathbf k\nu}^\dagger\rangle & \langle\hat a_{-\mathbf k\mu}^\dagger\hat a_{-\mathbf k\nu}\rangle
\end{pmatrix}_{\mu\nu}
=\mathsf T_{\mathbf k}\,\mathsf N_{\mathbf k}(0)\,\mathsf T_{\mathbf k}^\dagger, \tag{1}
$$

where each block is an $N_{\mathrm{sub}}\times N_{\mathrm{sub}}$ matrix with the sublattice indices $\mu,\nu$. The second equality substitutes $\hat\Psi_{\mathbf k}=\mathsf T_{\mathbf k}\hat\Phi_{\mathbf k}$. In the Gibbs state the magnon correlation matrix $\mathsf N_{\mathbf k}(0)=\operatorname{diag}(1+n_{1\mathbf k},\ldots,n_{1,-\mathbf k},\ldots)$ of Spin Correlations in research note *LSWT spin correlations* is diagonal, with the occupations $n_{n\mathbf k}=n_{\mathrm B}(\varepsilon_{n\mathbf k})$ of Thermodynamics (Section *Thermodynamics*). In the ground state, the magnon vacuum, the occupations vanish and $\mathsf N_{\mathbf k}(0)=\operatorname{diag}(\mathsf I_{N_{\mathrm{sub}}},0)$. The anomalous averages $\langle\hat a\hat a\rangle$ are nonzero even at zero temperature when $\mathsf H_{\mathbf k}$ contains the anomalous block $\mathsf B_{\mathbf k}$.

## Boson Number and Reduced Moment

The average number of Holstein–Primakoff bosons on sublattice $\mu$ is $\langle\hat n_\mu\rangle=N_{\mathrm{uc}}^{-1}\sum_i\langle\hat a_{i\mu}^\dagger\hat a_{i\mu}\rangle=N_{\mathrm{uc}}^{-1}\sum_{\mathbf k}\langle\hat a_{\mathbf k\mu}^\dagger\hat a_{\mathbf k\mu}\rangle$. The lower-right block of (1) contains these averages at $-\mathbf k$, and the momentum sum covers all momenta, so

<!-- eq 2: eq-lswt-boson-number -->

$$
\langle\hat n_\mu\rangle
=\frac{1}{N_{\mathrm{uc}}}\sum_{\mathbf k\in\mathrm{MBZ}}\Big[\mathsf T_{\mathbf k}\mathsf N_{\mathbf k}(0)\mathsf T_{\mathbf k}^\dagger\Big]_{N_{\mathrm{sub}}+\mu,\,N_{\mathrm{sub}}+\mu}
=\frac{1}{N_{\mathrm{uc}}}\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}
\Big[\big|(\mathsf P_{\mathbf k})_{\mu n}\big|^2n_{n\mathbf k}+\big|(\mathsf Q_{\mathbf k})_{\mu n}\big|^2\big(1+n_{n\mathbf k}\big)\Big]. \tag{2}
$$

The second form uses the blocks of the Bogoliubov transformation $\hat a_{\mathbf k\mu}=\sum_n[(\mathsf P_{\mathbf k})_{\mu n}\hat b_{\mathbf kn}+(\mathsf Q_{-\mathbf k})_{\mu n}\hat b_{-\mathbf kn}^\dagger]$ of Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization* and relabels $-\mathbf k\to\mathbf k$ in the hole-column terms. The $\mathsf Q$ term survives at zero temperature: the quantum reduction $N_{\mathrm{uc}}^{-1}\sum_{\mathbf k,n}|(\mathsf Q_{\mathbf k})_{\mu n}|^2$ arises only from the mixing of creation and annihilation operators and vanishes when $\mathsf B_{\mathbf k}=0$, as for a collinear ferromagnet whose Hamiltonian conserves the total spin component along the ordered moment. Exchange anisotropy transverse to the moment, such as $J^{xx}\ne J^{yy}$ in the local frame or Kitaev and off-diagonal couplings, produces $\mathsf B_{\mathbf k}\ne0$ and a quantum reduction even in a collinear ferromagnet. The $\mathsf P$ and $\mathsf Q$ terms proportional to $n_{n\mathbf k}$ give the thermal reduction.

The boson number reduces the ordered moment. With $\hat{\widetilde S}_I^0=S_I-\hat n_I$ and the local longitudinal direction $\mathbf n_\mu$ of Classical Order and Local Frame in research note *LSWT classical order*, the average spin of sublattice $\mu$ is

<!-- eq 3: eq-lswt-reduced-moment -->

$$
\mathbf m_\mu=\big(S_\mu-\langle\hat n_\mu\rangle\big)\,\mathbf n_\mu, \tag{3}
$$

since the transverse components are linear in the bosons and average to zero in the Gibbs state of the quadratic Hamiltonian. Equation (3) gives the length of the moment to order $S^0$. It gives the full moment to that order only when symmetry fixes the classical spin directions, for example a collinear state along an axis that the Hamiltonian preserves. For a canted or otherwise noncollinear reference configuration whose angles $\boldsymbol\theta$ are not fixed by symmetry, the zero-point energy $\Delta E_{\mathrm{zp}}(\boldsymbol\theta)$ of order $S$ shifts the angles that minimize $E_{\mathrm{cl}}+\Delta E_{\mathrm{zp}}$ by 

$$
\delta\boldsymbol\theta=-\big(\partial_{\boldsymbol\theta}^2E_{\mathrm{cl}}\big)^{-1}\partial_{\boldsymbol\theta}\Delta E_{\mathrm{zp}},
$$

 which is of order $S^{-1}$ because $E_{\mathrm{cl}}$ is of order $S^2$. The resulting rotation of the moment, $S_\mu\,(\partial\mathbf n_\mu/\partial\boldsymbol\theta)\cdot\delta\boldsymbol\theta$, is of order $S^0$, the same order as $\langle\hat n_\mu\rangle$. In the boson language it is the transverse expectation value of order $S^0$ generated by the cubic terms of the Holstein–Primakoff expansion, which the quadratic Hamiltonian omits; it is the condensate of The Tadpole and the $1/S$ Shift of the Classical Directions in research note *LSWT nonlinear spin waves*. To order $S^0$ the moment of such a state is therefore 

$$
\mathbf m_\mu=\big(S_\mu-\langle\hat n_\mu\rangle\big)\,\mathbf n_\mu+S_\mu\,\frac{\partial\mathbf n_\mu}{\partial\boldsymbol\theta}\cdot\delta\boldsymbol\theta,
$$

 and the magnetization obtained from $-\mathrm d(E_{\mathrm{cl}}+\Delta E_{\mathrm{zp}})/\mathrm dh$, with $h$ the strength of a uniform Zeeman field, contains both terms. The magnetization per site is $N_{\mathrm{sub}}^{-1}\sum_\mu\mathbf m_\mu$. These moments enter the elastic Bragg scattering of Spin Correlations in research note *LSWT spin correlations*. The reduction $\langle\hat n_\mu\rangle$ is of order $S^0$, compared with $S_\mu$, so the expansion is controlled only while $\langle\hat n_\mu\rangle\ll S_\mu$. A value $\langle\hat n_\mu\rangle\ge S_\mu$ would remove or reverse the moment and signals that LSWT about the chosen reference configuration no longer applies.

### Zero Modes

At a zero mode the summand of (2) can diverge. Near a Goldstone mode of an antiferromagnet, $\varepsilon_{n\mathbf k}\propto|\mathbf k-\mathbf k_0|$ and $|\mathsf P_{\mathbf k}|^2,|\mathsf Q_{\mathbf k}|^2\propto1/\varepsilon_{n\mathbf k}$, as for the single mode of Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*. At zero temperature the summand then grows as $1/|\mathbf k-\mathbf k_0|$, which is integrable in two dimensions, and the quantum reduction is finite. At nonzero temperature $n_{\mathrm B}(\varepsilon)\simeq k_{\mathrm B}T/\varepsilon$ adds a factor $1/\varepsilon$, and the summand grows as $|\mathbf k-\mathbf k_0|^{-2}$; the same power follows for a ferromagnetic Goldstone mode with $\varepsilon\propto|\mathbf k-\mathbf k_0|^2$ and bounded $\mathsf P_{\mathbf k}$. In two dimensions this gives a logarithmically divergent thermal boson number. The divergence is the LSWT form of the absence of spontaneous continuous-symmetry breaking at nonzero temperature in two dimensions, discussed in LSWT Overview in research note *LSWT spin Hamiltonian*. The thermodynamic potentials of Thermodynamics (Section *Thermodynamics*) remain finite in the same situation.

## Index of Observables

| Observable | Ingredients | Owner document |
|:---|:---|:---|
| Magnon energies and bands | $\varepsilon_{n\mathbf k}$ | This document; Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization* |
| Ground-state energy, zero-point correction | $\varepsilon_{n\mathbf k}$, $\operatorname{Tr}\mathsf A_{\mathbf k}$ | Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization* |
| Boson number, reduced moment, magnetization | $\mathsf T_{\mathbf k}$, $n_{n\mathbf k}$ | This document |
| Partition function, $F$, $U$, $\mathcal S$, $C$ | $\varepsilon_{n\mathbf k}$ | Thermodynamics (Section *Thermodynamics*) |
| Spin correlation function, elastic correlation | $\mathsf T_{\mathbf q}$, $\mathsf N_{\mathbf q}(t)$, $\mathbf m_\mu$ | Spin Correlations in research note *LSWT spin correlations* |
| Static and dynamic structure factors, spectral function | One-magnon weights $W_n^{\alpha\beta}$ | Structure Factor and Spectral Function in research note *LSWT spin correlations* |
| Berry curvature, Chern number, thermal Hall conductivity | $\mathsf T_{\mathbf k}$, $\partial_{\mathbf k}\mathsf H_{\mathbf k}$ | Topological Magnon Quantities in research note *LSWT magnon topology* |
| Classical lower bound and candidate orders | Classical energy only | Appendix: Luttinger–Tisza Method in research note *LSWT classical order* |

## References

### Internal Documents

- Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*: supplies $\varepsilon_{n\mathbf k}$, $\mathsf T_{\mathbf k}$ with its blocks $\mathsf P_{\mathbf k}$ and $\mathsf Q_{\mathbf k}$, and the ground-state energy.

- Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*: defines the Nambu spinor and the BdG blocks.

- Classical Order and Local Frame in research note *LSWT classical order*: defines the ordered directions $\mathbf n_\mu$.

- Thermodynamics (Section *Thermodynamics*): defines the occupations and the thermodynamic potentials.

- Spin Correlations in research note *LSWT spin correlations*: defines $\mathsf N_{\mathbf q}(t)$ and uses the reduced moments.

- Structure Factor and Spectral Function in research note *LSWT spin correlations*: determines the spectral weight of folded bands.

- Topological Magnon Quantities in research note *LSWT magnon topology*: builds topological responses from the bands and eigenvectors.

- Appendix: Luttinger–Tisza Method in research note *LSWT classical order*: bounds the classical energy of candidate reference configurations.

- LSWT Overview in research note *LSWT spin Hamiltonian*: states the finite-temperature qualification in two dimensions.

### External Sources

- None.

<!-- Thermodynamics Converted on 2026-10-08 from archive/2026-10-08-lswt-markdown/02-observables/thermodynamics.md (status: draft, source-section: "Thermodynamics in Linear Spin Wave Theory: Partition Function, Internal Energy, Free Energy, Entropy Expression, Specific Heat (source pp. 13–15)"). Review status: draft; awaiting the user's physics and mathematics review. -->

# Thermodynamics

In thermal equilibrium the LSWT Hamiltonian describes an ideal gas of magnons on top of the classical reference configuration. The partition function therefore factorizes over the magnon modes $(\mathbf k,n)$, and the free energy, internal energy, entropy, and heat capacity follow from single-oscillator expressions summed over the magnetic Brillouin zone (MBZ). Energy, temperature, and entropy symbols follow Notation and Conventions in research note *LSWT conventions*; the inverse temperature is $\beta=(k_{\mathrm B}T)^{-1}$ and the entropy is $\mathcal S$.

## Validity of the Magnon-Gas Description

We use the diagonal Hamiltonian of Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*,

$$
\hat H_{\mathrm{LSWT}}
=E_{\mathrm{cl}}+\hat H_2
=E_{\mathrm{GS}}+\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}\varepsilon_{n\mathbf k}\,\hat n_{\mathbf kn},
\qquad
\hat n_{\mathbf kn}=\hat b_{\mathbf kn}^\dagger\hat b_{\mathbf kn},
$$

with $E_{\mathrm{GS}}=E_{\mathrm{cl}}+\Delta E_{\mathrm{zp}}$. The classical reference configuration and the magnon energies $\varepsilon_{n\mathbf k}$ are held fixed as the temperature changes, and the quartic and higher Holstein–Primakoff terms that couple magnons are omitted. The results below are therefore the leading low-temperature description. Its corrections grow with the thermal magnon density, and the description fails when the thermal reduction of a sublattice moment, discussed in Magnon Observables (Section *Magnon Observables*), becomes comparable to the spin length. In two dimensions, the interpretation of these results as properties of an ordered phase requires the qualifications stated in LSWT Overview in research note *LSWT spin Hamiltonian*.

All quantities below assume $\varepsilon_{n\mathbf k}>0$, which holds when $\mathsf H_{\mathbf k}$ is positive definite. The section on zero modes describes the limit $\varepsilon_{n\mathbf k}\to0$.

## Partition Function

The Gibbs state $\hat\rho=\exp(-\beta\hat H_{\mathrm{LSWT}})/Z$ has the partition function $Z=\operatorname{Tr}\exp(-\beta\hat H_{\mathrm{LSWT}})$. The number operators of different modes commute, so the trace factorizes into independent geometric series, one for each mode:

<!-- eq 4: eq-lswt-partition-function -->

$$
Z=\exp(-\beta E_{\mathrm{GS}})\prod_{\mathbf k\in\mathrm{MBZ}}\prod_{n=1}^{N_{\mathrm{sub}}}\frac{1}{1-\exp(-\beta\varepsilon_{n\mathbf k})},
\qquad
-\ln Z=\beta E_{\mathrm{GS}}+\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}\ln\!\left(1-\exp(-\beta\varepsilon_{n\mathbf k})\right). \tag{4}
$$

Each geometric series converges because $\varepsilon_{n\mathbf k}>0$. The constant $E_{\mathrm{GS}}$ contributes the factor $\exp(-\beta E_{\mathrm{GS}})$; it shifts the internal and free energies but drops out of the entropy and heat capacity, which depend only on the thermal part of $\ln Z$.

The thermal occupation of mode $(\mathbf k,n)$ is the Bose–Einstein distribution

<!-- eq 5: eq-lswt-bose-einstein-distribution -->

$$
\langle\hat n_{\mathbf kn}\rangle
=n_{\mathrm B}(\varepsilon_{n\mathbf k}),
\qquad
n_{\mathrm B}(\varepsilon)=\frac{1}{\exp(\beta\varepsilon)-1}. \tag{5}
$$

## Internal and Free Energies

The internal energy is the thermal average of the Hamiltonian, $U=\langle\hat H_{\mathrm{LSWT}}\rangle=-\partial\ln Z/\partial\beta$. Differentiating (4) gives

<!-- eq 6: eq-lswt-internal-energy -->

$$
U=E_{\mathrm{GS}}+\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}\varepsilon_{n\mathbf k}\,n_{\mathrm B}(\varepsilon_{n\mathbf k}). \tag{6}
$$

The internal energy thus consists of the classical energy, the zero-point correction, and the energy of the thermally excited magnons. The Helmholtz free energy, defined by $Z=\exp(-\beta F)$, is

<!-- eq 7: eq-lswt-free-energy -->

$$
F=E_{\mathrm{GS}}+k_{\mathrm B}T\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}\ln\!\left(1-\exp(-\beta\varepsilon_{n\mathbf k})\right). \tag{7}
$$

Each logarithm is negative, so the thermal magnons lower $F$ below $E_{\mathrm{GS}}$. At $T=0$ both $U$ and $F$ reduce to $E_{\mathrm{GS}}$.

## Entropy

The entropy of the Gibbs state is the von Neumann entropy $\mathcal S=-k_{\mathrm B}\operatorname{Tr}(\hat\rho\ln\hat\rho)$. Inserting $\ln\hat\rho=-\beta\hat H_{\mathrm{LSWT}}-\ln Z$ gives $\mathcal S=(U-F)/T$, which coincides with the thermodynamic relation $\mathcal S=-\partial F/\partial T$. With (6) and (7),

$$
\mathcal S=k_{\mathrm B}\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}
\left[\beta\varepsilon_{n\mathbf k}\,n_{\mathrm B}(\varepsilon_{n\mathbf k})-\ln\!\left(1-\exp(-\beta\varepsilon_{n\mathbf k})\right)\right].
$$

The relations $\beta\varepsilon=\ln[(1+n_{\mathrm B})/n_{\mathrm B}]$ and $\ln(1-\exp(-\beta\varepsilon))=-\ln(1+n_{\mathrm B})$, which follow from the definition of $n_{\mathrm B}$, express the entropy through the occupations alone:

<!-- eq 8: eq-lswt-magnon-entropy -->

$$
\mathcal S=k_{\mathrm B}\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}
\Big[\big(1+n_{\mathbf kn}\big)\ln\big(1+n_{\mathbf kn}\big)-n_{\mathbf kn}\ln n_{\mathbf kn}\Big],
\qquad
n_{\mathbf kn}=n_{\mathrm B}(\varepsilon_{n\mathbf k}). \tag{8}
$$

This is the entropy of an ideal Bose gas with the occupations $n_{\mathbf kn}$. The zero-point correction does not enter, since it is a temperature-independent constant.

## Heat Capacity

The heat capacity is the temperature derivative of the internal energy, $C=\partial U/\partial T$. Only the occupations depend on temperature, and $\partial n_{\mathrm B}(\varepsilon)/\partial T=(\varepsilon/k_{\mathrm B}T^2)\exp(\beta\varepsilon)/(\exp(\beta\varepsilon)-1)^2$, so

<!-- eq 9: eq-lswt-magnon-heat-capacity -->

$$
C=k_{\mathrm B}\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}
\frac{(\beta\varepsilon_{n\mathbf k})^2}{4\sinh^2(\beta\varepsilon_{n\mathbf k}/2)}. \tag{9}
$$

Each mode contributes the heat capacity of a harmonic oscillator. The contribution approaches $k_{\mathrm B}$ for $\varepsilon_{n\mathbf k}\ll k_{\mathrm B}T$ and is exponentially small for $\varepsilon_{n\mathbf k}\gg k_{\mathrm B}T$, so at low temperature the heat capacity is dominated by the lowest magnon energies. The relation $C=T\,\partial\mathcal S/\partial T$ gives the same expression.

## Normalization and Momentum Sums

The quantities above are extensive: the sums run over the $N_{\mathrm{uc}}$ momenta of the MBZ and the $N_{\mathrm{sub}}$ bands. Values per magnetic site follow by dividing by $N_{\mathrm{site}}=N_{\mathrm{uc}}N_{\mathrm{sub}}$; for example, $U/N_{\mathrm{site}}=\mathcal E_{\mathrm{GS}}+N_{\mathrm{site}}^{-1}\sum_{\mathbf k,n}\varepsilon_{n\mathbf k}n_{\mathrm B}(\varepsilon_{n\mathbf k})$. In the thermodynamic limit the momentum sum becomes an integral over the MBZ, $N_{\mathrm{uc}}^{-1}\sum_{\mathbf k}\to A_{\mathrm{MBZ}}^{-1}\int_{\mathrm{MBZ}}d^2\mathbf k$, where $A_{\mathrm{MBZ}}$ is the area of the MBZ. A finite momentum mesh approximates this integral.

## Zero Modes

When $\mathsf H_{\mathbf k}$ is only positive semidefinite, some magnon energy vanishes, for example at a Goldstone mode. Near an isolated zero at $\mathbf k_0$ we assume $\varepsilon_{n\mathbf k}\propto|\mathbf k-\mathbf k_0|^a$ with $a>0$; linear ($a=1$) and quadratic ($a=2$) dispersions are the common cases. The summands behave as follows as $\varepsilon\to0$:

| Quantity | Summand for $\varepsilon\ll k_{\mathrm B}T$ | Limit at $\varepsilon=0$ |
|:---|:---|:---|
| $U$ | $\varepsilon\,n_{\mathrm B}(\varepsilon)\to k_{\mathrm B}T$ | finite |
| $C$ | $k_{\mathrm B}$ | finite |
| $F$ | $k_{\mathrm B}T\ln(\beta\varepsilon)$ | logarithmically divergent |
| $\mathcal S$ | $k_{\mathrm B}[1+\ln(k_{\mathrm B}T/\varepsilon)]$ | logarithmically divergent |

The logarithmic divergences are integrable. In two dimensions $\int d^2\mathbf q\,\ln|\mathbf q|$ converges near $\mathbf q=0$ for any $a>0$, so $F$, $U$, $\mathcal S$, and $C$ remain finite in the thermodynamic limit. A zero mode at an isolated momentum has vanishing weight in the MBZ integral. On a finite mesh that contains $\mathbf k_0$, the treatment of that single point is a discretization choice, and the continuum values do not depend on it. Quantities weighted by the occupations without a compensating factor of $\varepsilon$, such as the boson number and the sublattice moment, can diverge instead; they are treated in Magnon Observables (Section *Magnon Observables*).

## Magnetization at Nonzero Temperature

A uniform Zeeman field of strength $h$ along the unit vector $\mathbf e$ enters the spin Hamiltonian as $-h\sum_I\mathbf e\cdot\mathbf S_I$; a $g$-tensor replaces $\mathbf e$ by $\mathsf g^{\mathsf T}\mathbf e$ and changes nothing below. We define the magnetization per site at temperature $T$ as the field derivative of the free energy (7), <!-- eq 10: eq-lswt-thermal-magnetization -->

$$
M(h,T)=-\frac{1}{N_{\mathrm{site}}}\,\frac{\mathrm dF}{\mathrm dh}, \tag{10}
$$

 in which the classical reference configuration at each $h$ is the minimum of $E_{\mathrm{cl}}$ at that field. At $T=0$, $F=E_{\mathrm{GS}}$ and (10) reduces to the zero-temperature magnetization $-\mathrm d(E_{\mathrm{cl}}+\Delta E_{\mathrm{zp}})/\mathrm dh$ of Magnon Observables (Section *Magnon Observables*).

### Order of the Reference Configuration

Holding the reference configuration at the minimum of $E_{\mathrm{cl}}$ rather than of $F$ is consistent to the order of LSWT. Write $F(\boldsymbol\theta)=E_{\mathrm{cl}}(\boldsymbol\theta)+F_{\mathrm{qm}}(\boldsymbol\theta)$, where $\boldsymbol\theta$ collects the classical angles and $F_{\mathrm{qm}}=F-E_{\mathrm{cl}}$ is the zero-point and thermal magnon free energy of the quadratic Hamiltonian built at $\boldsymbol\theta$. The classical energy is of order $S^2$ and $F_{\mathrm{qm}}$ of order $S$ at fixed $k_{\mathrm B}T/S$. Minimizing $F$ instead of $E_{\mathrm{cl}}$ shifts the angles by <!-- eq 11: eq-lswt-thermal-angle-shift -->

$$
\delta\boldsymbol\theta(T)=-\big(\partial_{\boldsymbol\theta}^2E_{\mathrm{cl}}\big)^{-1}\partial_{\boldsymbol\theta}F_{\mathrm{qm}}, \tag{11}
$$

 of order $S^{-1}$, and lowers $F$ by $\tfrac12\,\partial_{\boldsymbol\theta}F_{\mathrm{qm}}\cdot(\partial_{\boldsymbol\theta}^2E_{\mathrm{cl}})^{-1}\partial_{\boldsymbol\theta}F_{\mathrm{qm}}$, of order $S^0$. This is the order of the quartic Holstein–Primakoff terms that LSWT omits, so (10) is complete to order $S^0$ without reminimizing the angles. Away from the classical minimum the boson Hamiltonian also has linear terms; they contribute to $F$ only at second order in their coefficients, which vanish at the minimum, so they do not affect the first derivatives used below.

### Reduced Moment and Canting-Angle Shift

The total derivative in (10) separates into three parts: 

$$
\frac{\mathrm dF}{\mathrm dh}
=\frac{\partial E_{\mathrm{cl}}}{\partial h}\bigg|_{\boldsymbol\theta}
+\frac{\partial F_{\mathrm{qm}}}{\partial h}\bigg|_{\boldsymbol\theta}
+\partial_{\boldsymbol\theta}F_{\mathrm{qm}}\cdot\frac{\mathrm d\boldsymbol\theta_{\mathrm{cl}}}{\mathrm dh},
$$

 since $\partial_{\boldsymbol\theta}E_{\mathrm{cl}}=0$ at the classical minimum. The first term is $-S_\mu\,\mathbf e\cdot\mathbf n_\mu$ summed over sites. For the second, the Hellmann–Feynman theorem for the Gibbs state gives $\partial F_{\mathrm{qm}}/\partial h|_{\boldsymbol\theta}=\langle\partial\hat H_2/\partial h\rangle$. In the local frame $\mathbf e\cdot\mathbf S_I=(\mathbf e\cdot\mathbf n_\mu)(S_\mu-\hat n_I)$ plus transverse components linear in the bosons, so the quadratic part of $\partial\hat H/\partial h$ is $\sum_I(\mathbf e\cdot\mathbf n_\mu)\hat n_I$ and the second term is $\sum_I(\mathbf e\cdot\mathbf n_\mu)\langle\hat n_\mu\rangle_T$. For the third, differentiating $\partial_{\boldsymbol\theta}E_{\mathrm{cl}}=0$ with respect to $h$ gives $\mathrm d\boldsymbol\theta_{\mathrm{cl}}/\mathrm dh=-(\partial_{\boldsymbol\theta}^2E_{\mathrm{cl}})^{-1}\partial_h\partial_{\boldsymbol\theta}E_{\mathrm{cl}}$ with $\partial_h\partial_{\boldsymbol\theta}E_{\mathrm{cl}}=-\sum_IS_\mu\,\mathbf e\cdot\partial_{\boldsymbol\theta}\mathbf n_\mu$. With (11) and the symmetry of $\partial_{\boldsymbol\theta}^2E_{\mathrm{cl}}$, <!-- eq 12: eq-lswt-thermal-magnetization-decomposition -->

$$
M(h,T)=\frac{1}{N_{\mathrm{sub}}}\sum_\mu\mathbf e\cdot\Big[\big(S_\mu-\langle\hat n_\mu\rangle_T\big)\,\mathbf n_\mu+S_\mu\,\frac{\partial\mathbf n_\mu}{\partial\boldsymbol\theta}\cdot\delta\boldsymbol\theta(T)\Big]. \tag{12}
$$

 This is the zero-temperature moment of Magnon Observables (Section *Magnon Observables*) with the boson number and the angle shift replaced by their thermal values. For a collinear state along $\mathbf e$ whose angles are fixed by symmetry, $\partial_{\boldsymbol\theta}F_{\mathrm{qm}}=0$ and $M$ is the thermal reduced moment alone. For a canted state the second term is of the same order $S^0$ as the reduction $\langle\hat n_\mu\rangle_T$, and the reduced moment alone does not give the magnetization at any temperature.

### Goldstone Modes in Two Dimensions

When the field leaves a continuous symmetry about $\mathbf e$, as for an isotropic Heisenberg model, the reference configuration breaks it and a Goldstone mode stays at zero energy for every $h$. In two dimensions $\langle\hat n_\mu\rangle_T$ then diverges logarithmically at $T>0$ (Section *Magnon Observables*), and so does $\partial_{\boldsymbol\theta}F_{\mathrm{qm}}$: a rotation of the angles away from the minimum gaps or destabilizes the Goldstone mode, whose thermal free energy depends on that gap as $k_{\mathrm B}T\ln\varepsilon$. The two terms of (12) diverge with opposite signs, and $M$ is finite. The finiteness follows directly from (10): the thermal part of $\mathrm dF/\mathrm dh$ along the classical minimum is $\sum_{\mathbf k,n}n_{\mathrm B}(\varepsilon_{n\mathbf k})\,\mathrm d\varepsilon_{n\mathbf k}/\mathrm dh$, and for a Goldstone mode $\varepsilon_{n\mathbf k_0+\mathbf q}=c(h)|\mathbf q|$ the summand tends to the constant $k_{\mathrm B}T\,c'(h)/c(h)$. The magnetization along the field is therefore a finite low-temperature quantity of LSWT in two dimensions, whereas the moment of each sublattice is not. For the same reason the condition $\langle\hat n_\mu\rangle_T\ll S_\mu$ of Magnon Observables (Section *Magnon Observables*) does not bound the range of validity here, since the left side diverges at every $T>0$. With a Goldstone mode in two dimensions we therefore use (10) only for $k_{\mathrm B}T$ small compared with the spin-wave energy scale, of order $JS$; for a gapped spectrum the boson-number condition applies as stated there.

### Example: Canted Square-Lattice Antiferromagnet

For the nearest-neighbour Heisenberg antiferromagnet on the square lattice with $J=1$, lattice constant $1$, spin $S$, and $\mathbf e=\hat{\mathbf z}$, the two sublattices cant toward the field by the angle $\theta$ out of the plane perpendicular to $\mathbf e$. In the frame rotated with each sublattice, one boson per site and momenta over the full square Brillouin zone describe the quadratic Hamiltonian at any $\theta$: 

$$
A_{\mathbf k}(\theta,h)=-4S\big(2\sin^2\theta-1\big)+h\sin\theta+4S\sin^2\theta\,\gamma_{\mathbf k},
\qquad
B_{\mathbf k}(\theta)=-4S\cos^2\theta\,\gamma_{\mathbf k},
$$

 with $\gamma_{\mathbf k}=\tfrac12(\cos k_x+\cos k_y)$, $\varepsilon_{\mathbf k}=(A_{\mathbf k}^2-B_{\mathbf k}^2)^{1/2}$, and $E_{\mathrm{cl}}/N_{\mathrm{site}}=2S^2(2\sin^2\theta-1)-hS\sin\theta$. The classical minimum $\sin\theta=h/8S$ gives $A_{\mathbf k}=4S(1+\sin^2\theta\,\gamma_{\mathbf k})$ and the classical magnetization $h/8$ up to the saturation field $h=8S$. The field enters $A_{\mathbf k}$ only through $h\sin\theta$, which makes the Hellmann–Feynman step of (12) explicit: $\partial F_{\mathrm{qm}}/\partial h|_\theta=N_{\mathrm{site}}\sin\theta\,\langle\hat n\rangle_T$ and $\mathrm d\theta_{\mathrm{cl}}/\mathrm dh=(8S\cos\theta)^{-1}$.

The Goldstone mode sits at $(\pi,\pi)$. With $\gamma\simeq-1+q^2/4$, $A-B\simeq Sq^2$ and $A+B\simeq8S\cos^2\theta$, so $\varepsilon\simeq cq$ with 

$$
c^2=8S^2\cos^2\theta=8S^2-\frac{h^2}{8}.
$$

 The thermal summand of $\langle\hat n\rangle_T$ near $(\pi,\pi)$ is $(A/\varepsilon)\,n_{\mathrm B}\simeq k_{\mathrm B}T/(2Sq^2)$, and its momentum integral gives $\langle\hat n\rangle_T\simeq(k_{\mathrm B}T/4\pi S)\ln(1/q_{\min})$ for a smallest momentum $q_{\min}$. The linear Goldstone mode gives $F_{\mathrm{th}}/N_{\mathrm{site}}\simeq-\zeta(3)(k_{\mathrm B}T)^3/(2\pi c^2)$, and with $\partial_hc^2=-h/4$, <!-- eq 13: eq-lswt-canted-square-thermal-magnetization -->

$$
M(h,T)-M(h,0)\simeq\frac{\zeta(3)\,h\,(k_{\mathrm B}T)^3}{8\pi c^4} \tag{13}
$$

 at low temperature. The magnetization rises as $T^3$, while each sublattice moment decreases without bound as $q_{\min}\to0$. The magnon at $\mathbf k=0$, the uniform precession about the field, has $\varepsilon=8S\sin\theta=h$ and $\mathrm d\varepsilon/\mathrm dh=1$; it lowers $M$ by a term of order $\exp(-h/k_{\mathrm B}T)$, which (13) omits.

## References

### Internal Documents

- Notation and Conventions in research note *LSWT conventions*: defines $\beta$, $\mathcal S$, $\varepsilon_{n\mathbf k}$, $E_{\mathrm{cl}}$, $\Delta E_{\mathrm{zp}}$, $E_{\mathrm{GS}}$, and the system-size symbols.

- Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*: supplies the diagonal magnon Hamiltonian, the zero-point correction, and the positive-definiteness condition.

- LSWT Overview in research note *LSWT spin Hamiltonian*: states the finite-temperature qualification for two-dimensional ordered states.

- Magnon Observables (Section *Magnon Observables*): uses the Bose–Einstein occupations for boson numbers and sublattice moments, and supplies the zero-temperature moment with the canting-angle shift that Magnetization at Nonzero Temperature extends to $T>0$.

- Thermodynamic Derivations (Section *Appendix: Thermodynamic Derivations*): is reserved for longer entropy and correlation-matrix derivations.

### External Sources

- None.

<!-- Appendix: Thermodynamic Derivations Converted on 2026-10-08 from archive/2026-10-08-lswt-markdown/04-appendices/thermodynamic-derivations.md (status: draft, source-section: "Thermodynamics in Linear Spin Wave Theory; Number and Spin moment from Correlation Matrix (source pp. 13–16); restructured TeX appendix 'Thermodynamic derivations'"). Review status: draft; awaiting the user's physics and mathematics review. -->

# Appendix: Thermodynamic Derivations

The thermodynamic quantities of a harmonic magnon gas and the equal-time correlation matrix of the Holstein–Primakoff bosons both follow from the Gibbs state of the diagonal magnon Hamiltonian. Thermodynamics (Section *Thermodynamics*) and Magnon Observables (Section *Magnon Observables*) state the results with short derivations; this appendix gives the intermediate steps: the factorization of the partition function, the entropy and heat-capacity identities, the small-energy expansions behind the zero-mode table, and the correlation-matrix route to the boson number. We assume, as in those documents, that $\mathsf H_{\mathbf k}$ is positive definite at every momentum of the magnetic Brillouin zone (MBZ), so that every magnon energy $\varepsilon_{n\mathbf k}$ is positive, except where the zero-mode limit is discussed.

## Factorization of the Partition Function

The diagonal Hamiltonian $\hat H_{\mathrm{LSWT}}=E_{\mathrm{GS}}+\sum_{\mathbf k,n}\varepsilon_{n\mathbf k}\hat n_{\mathbf kn}$ of Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization* is diagonal in the Fock basis $|\{m_{\mathbf kn}\}\rangle$ of magnon occupation numbers $m_{\mathbf kn}=0,1,2,\ldots$, with eigenvalues $E_{\mathrm{GS}}+\sum_{\mathbf k,n}\varepsilon_{n\mathbf k}m_{\mathbf kn}$. The trace is a sum over all occupation configurations, and the exponential of a sum is a product, so

$$
Z=\exp(-\beta E_{\mathrm{GS}})\prod_{\mathbf k,n}\sum_{m=0}^{\infty}\exp(-\beta\varepsilon_{n\mathbf k}m)
=\exp(-\beta E_{\mathrm{GS}})\prod_{\mathbf k,n}\frac{1}{1-\exp(-\beta\varepsilon_{n\mathbf k})}.
$$

Each geometric series converges because $\varepsilon_{n\mathbf k}>0$. The Gibbs state is the product of the single-mode thermal states $\hat\rho_{\mathbf kn}=(1-\exp(-\beta\varepsilon_{n\mathbf k}))\exp(-\beta\varepsilon_{n\mathbf k}\hat n_{\mathbf kn})$. The occupation of one mode follows from the derivative of the corresponding factor, $\langle\hat n_{\mathbf kn}\rangle=-\beta^{-1}\partial\ln Z/\partial\varepsilon_{n\mathbf k}=[\exp(\beta\varepsilon_{n\mathbf k})-1]^{-1}=n_{\mathrm B}(\varepsilon_{n\mathbf k})$.

## Internal Energy and Free Energy

The internal energy is $U=\operatorname{Tr}(\hat\rho\hat H_{\mathrm{LSWT}})=-\partial\ln Z/\partial\beta$. Differentiating the logarithm of each factor gives

$$
-\frac{\partial}{\partial\beta}\ln\frac{1}{1-\exp(-\beta\varepsilon)}
=\frac{\varepsilon\exp(-\beta\varepsilon)}{1-\exp(-\beta\varepsilon)}
=\varepsilon\,n_{\mathrm B}(\varepsilon),
$$

and the prefactor contributes $E_{\mathrm{GS}}$, which yields $U=E_{\mathrm{GS}}+\sum_{\mathbf k,n}\varepsilon_{n\mathbf k}n_{\mathrm B}(\varepsilon_{n\mathbf k})$. The free energy $F=-\beta^{-1}\ln Z$ follows by taking the logarithm of the product, $F=E_{\mathrm{GS}}+\beta^{-1}\sum_{\mathbf k,n}\ln(1-\exp(-\beta\varepsilon_{n\mathbf k}))$.

## Entropy

The von Neumann entropy of the Gibbs state uses $\ln\hat\rho=-\beta\hat H_{\mathrm{LSWT}}-\ln Z$:

$$
\mathcal S=-k_{\mathrm B}\operatorname{Tr}(\hat\rho\ln\hat\rho)
=k_{\mathrm B}\beta\operatorname{Tr}(\hat\rho\hat H_{\mathrm{LSWT}})+k_{\mathrm B}\ln Z
=\frac{U-F}{T}.
$$

The thermodynamic entropy $-\partial F/\partial T$ gives the same result. With $F=-k_{\mathrm B}T\ln Z$ and $\partial\beta/\partial T=-1/(k_{\mathrm B}T^2)$,

$$
-\frac{\partial F}{\partial T}
=k_{\mathrm B}\ln Z+k_{\mathrm B}T\,\frac{\partial\ln Z}{\partial\beta}\frac{\partial\beta}{\partial T}
=k_{\mathrm B}\ln Z+\frac{U}{T}
=\frac{U-F}{T}.
$$

For the Gibbs state the von Neumann entropy therefore reduces to the thermodynamic entropy. Inserting the expressions for $U$ and $F$, the constant $E_{\mathrm{GS}}$ cancels in $U-F$, and

$$
\mathcal S=k_{\mathrm B}\sum_{\mathbf k,n}\Big[\beta\varepsilon_{n\mathbf k}n_{\mathbf kn}-\ln\big(1-\exp(-\beta\varepsilon_{n\mathbf k})\big)\Big],
\qquad
n_{\mathbf kn}=n_{\mathrm B}(\varepsilon_{n\mathbf k}).
$$

The definition $n=[\exp(\beta\varepsilon)-1]^{-1}$ gives $\exp(\beta\varepsilon)=(1+n)/n$ and $1-\exp(-\beta\varepsilon)=1/(1+n)$, so

$$
\beta\varepsilon=\ln\frac{1+n}{n},
\qquad
\ln\big(1-\exp(-\beta\varepsilon)\big)=-\ln(1+n).
$$

Substituting these relations, the summand becomes $n\ln(1+n)-n\ln n+\ln(1+n)=(1+n)\ln(1+n)-n\ln n$, which is the occupation form of the entropy in Thermodynamics (Section *Thermodynamics*).

## Heat Capacity

The temperature derivative of the occupation is

$$
\frac{\partial n_{\mathrm B}(\varepsilon)}{\partial T}
=\frac{\varepsilon}{k_{\mathrm B}T^2}\,\frac{\exp(\beta\varepsilon)}{\big(\exp(\beta\varepsilon)-1\big)^2}
=\frac{\varepsilon}{k_{\mathrm B}T^2}\,\frac{1}{4\sinh^2(\beta\varepsilon/2)},
$$

where the second form uses $\exp(x)/(\exp(x)-1)^2=1/(\exp(x/2)-\exp(-x/2))^2$. Since only the occupations depend on temperature, $C=\partial U/\partial T=\sum_{\mathbf k,n}\varepsilon_{n\mathbf k}\,\partial n_{\mathbf kn}/\partial T$, which gives the expression $k_{\mathrm B}\sum(\beta\varepsilon)^2/[4\sinh^2(\beta\varepsilon/2)]$.

The relation $C=T\,\partial\mathcal S/\partial T$ follows from the occupation form of the entropy. The derivative of the summand with respect to the occupation is $k_{\mathrm B}\ln[(1+n)/n]=k_{\mathrm B}\beta\varepsilon$, so

$$
T\frac{\partial\mathcal S}{\partial T}
=T\sum_{\mathbf k,n}k_{\mathrm B}\beta\varepsilon_{n\mathbf k}\,\frac{\partial n_{\mathbf kn}}{\partial T}
=\sum_{\mathbf k,n}\varepsilon_{n\mathbf k}\frac{\partial n_{\mathbf kn}}{\partial T}
=\frac{\partial U}{\partial T}.
$$

## Small-Energy Expansions Behind the Zero-Mode Limits

With $x=\beta\varepsilon\to0^+$, the summands of the thermodynamic quantities have the expansions

$$
\begin{aligned}
\varepsilon\,n_{\mathrm B}(\varepsilon)&=k_{\mathrm B}T\left(1-\frac x2+\frac{x^2}{12}+O(x^4)\right),\\
k_{\mathrm B}T\ln\big(1-\exp(-x)\big)&=k_{\mathrm B}T\left(\ln x-\frac x2+\frac{x^2}{24}+O(x^4)\right),\\
k_{\mathrm B}\Big[x\,n_{\mathrm B}-\ln\big(1-\exp(-x)\big)\Big]&=k_{\mathrm B}\left(1-\ln x+\frac{x^2}{24}+O(x^4)\right),\\
k_{\mathrm B}\frac{x^2}{4\sinh^2(x/2)}&=k_{\mathrm B}\left(1-\frac{x^2}{12}+O(x^4)\right).
\end{aligned}
$$

They follow from $x/(\exp(x)-1)=1-x/2+x^2/12+O(x^4)$ and $\ln(1-\exp(-x))=\ln x-x/2+x^2/24+O(x^4)$. The energy and heat-capacity summands stay finite, while the free-energy and entropy summands diverge as $\ln x=\ln(\varepsilon/k_{\mathrm B}T)$. Near an isolated zero $\varepsilon_{n\mathbf k}\propto|\mathbf k-\mathbf k_0|^a$ with $a>0$, the divergent part of each summand is proportional to $a\ln|\mathbf k-\mathbf k_0|$, and in two dimensions $\int_{|\mathbf q|<q_0}d^2\mathbf q\,\ln|\mathbf q|=\pi q_0^2(\ln q_0-\frac12)$ is finite. The momentum integrals of $F$ and $\mathcal S$ therefore remain finite in the thermodynamic limit, as stated in Thermodynamics (Section *Thermodynamics*). Finiteness of these integrals does not imply that the magnon-gas description is controlled; that condition involves the boson number, treated below.

## Correlation Matrix of the Holstein–Primakoff Bosons

The equal-time correlation matrix $\langle\hat\Psi_{\mathbf k}\hat\Psi_{\mathbf k}^\dagger\rangle$ contains every average of two Holstein–Primakoff operators at momentum $\mathbf k$. Its blocks are

$$
\big\langle\hat\Psi_{\mathbf k}\hat\Psi_{\mathbf k}^\dagger\big\rangle
=\begin{pmatrix}
\langle\hat{\mathbf a}_{\mathbf k}\hat{\mathbf a}_{\mathbf k}^\dagger\rangle & \langle\hat{\mathbf a}_{\mathbf k}\hat{\mathbf a}_{-\mathbf k}\rangle\\
\langle\hat{\mathbf a}_{-\mathbf k}^\dagger\hat{\mathbf a}_{\mathbf k}^\dagger\rangle & \langle\hat{\mathbf a}_{-\mathbf k}^\dagger\hat{\mathbf a}_{-\mathbf k}\rangle
\end{pmatrix},
$$

where each block is the $N_{\mathrm{sub}}\times N_{\mathrm{sub}}$ matrix of averages over the sublattice indices. The average is linear, and the Bogoliubov transformation $\hat\Psi_{\mathbf k}=\mathsf T_{\mathbf k}\hat\Phi_{\mathbf k}$ has c-number coefficients, so

$$
\big\langle\hat\Psi_{\mathbf k}\hat\Psi_{\mathbf k}^\dagger\big\rangle
=\mathsf T_{\mathbf k}\big\langle\hat\Phi_{\mathbf k}\hat\Phi_{\mathbf k}^\dagger\big\rangle\mathsf T_{\mathbf k}^\dagger.
$$

The Gibbs state is diagonal in the magnon occupation numbers, so every average of two magnon operators that changes some occupation vanishes. This removes $\langle\hat b\hat b\rangle$, $\langle\hat b^\dagger\hat b^\dagger\rangle$, and the products of different modes. The remaining averages are $\langle\hat b_{\mathbf kn}\hat b_{\mathbf kn}^\dagger\rangle=1+n_{n\mathbf k}$ and $\langle\hat b_{-\mathbf kn}^\dagger\hat b_{-\mathbf kn}\rangle=n_{n,-\mathbf k}$, where $n_{n\mathbf k}=n_{\mathrm B}(\varepsilon_{n\mathbf k})$. Hence

$$
\big\langle\hat\Phi_{\mathbf k}\hat\Phi_{\mathbf k}^\dagger\big\rangle
=\operatorname{diag}\big(1+n_{1\mathbf k},\ldots,1+n_{N_{\mathrm{sub}}\mathbf k},\;n_{1,-\mathbf k},\ldots,n_{N_{\mathrm{sub}},-\mathbf k}\big),
$$

which is the matrix $\mathsf N_{\mathbf k}(0)$ of Spin Correlations in research note *LSWT spin correlations*; each hole entry carries the occupation of its own band at $-\mathbf k$. In the magnon vacuum all occupations vanish and $\mathsf N_{\mathbf k}(0)=\operatorname{diag}(\mathsf I_{N_{\mathrm{sub}}},0)$.

## Boson Number and Sublattice Moment

The boson number of sublattice $\mu$ averaged over the $N_{\mathrm{uc}}$ magnetic unit cells is $\langle\hat n_\mu\rangle=N_{\mathrm{uc}}^{-1}\sum_i\langle\hat a_{i\mu}^\dagger\hat a_{i\mu}\rangle$. The Fourier phases $\exp(\pm\mathrm i\mathbf k\cdot\mathbf r_{i\mu})$ of the full-position convention cancel in the product of an operator and its adjoint on the same sublattice, and the sum over cells gives $\sum_i\hat a_{i\mu}^\dagger\hat a_{i\mu}=\sum_{\mathbf k}\hat a_{\mathbf k\mu}^\dagger\hat a_{\mathbf k\mu}$. The diagonal entry $N_{\mathrm{sub}}+\mu$ of the correlation matrix at $\mathbf k$ is $\langle\hat a_{-\mathbf k\mu}^\dagger\hat a_{-\mathbf k\mu}\rangle$. The lower block row of $\mathsf T_{\mathbf k}$ is $(\mathsf Q_{\mathbf k}^*,\mathsf P_{-\mathbf k}^*)$, so

$$
\Big[\mathsf T_{\mathbf k}\mathsf N_{\mathbf k}(0)\mathsf T_{\mathbf k}^\dagger\Big]_{N_{\mathrm{sub}}+\mu,\,N_{\mathrm{sub}}+\mu}
=\sum_{n=1}^{N_{\mathrm{sub}}}\Big[\big|(\mathsf Q_{\mathbf k})_{\mu n}\big|^2\big(1+n_{n\mathbf k}\big)+\big|(\mathsf P_{-\mathbf k})_{\mu n}\big|^2n_{n,-\mathbf k}\Big].
$$

Summing over a momentum set that is closed under inversion and relabeling $-\mathbf k\to\mathbf k$ in the $\mathsf P$ term gives the boson number of Magnon Observables (Section *Magnon Observables*),

$$
\langle\hat n_\mu\rangle=\frac{1}{N_{\mathrm{uc}}}\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}\Big[\big|(\mathsf P_{\mathbf k})_{\mu n}\big|^2n_{n\mathbf k}+\big|(\mathsf Q_{\mathbf k})_{\mu n}\big|^2\big(1+n_{n\mathbf k}\big)\Big].
$$

The same result follows directly from $\hat a_{\mathbf k\mu}=\sum_n[(\mathsf P_{\mathbf k})_{\mu n}\hat b_{\mathbf kn}+(\mathsf Q_{-\mathbf k})_{\mu n}\hat b_{-\mathbf kn}^\dagger]$ and the magnon averages above. The longitudinal spin component in the local frame is $\hat{\widetilde S}_{i\mu}^0=S_\mu-\hat a_{i\mu}^\dagger\hat a_{i\mu}$ to the order of linear spin-wave theory, so the cell-averaged longitudinal moment of sublattice $\mu$ is $S_\mu-\langle\hat n_\mu\rangle$.

## References

### Internal Documents

- Thermodynamics (Section *Thermodynamics*): states the thermodynamic quantities and the zero-mode limits derived here.

- Magnon Observables (Section *Magnon Observables*): states the correlation matrix, the boson number, and the reduced moment derived here.

- Spin Correlations in research note *LSWT spin correlations*: defines the magnon correlation matrix $\mathsf N_{\mathbf q}(t)$, whose equal-time value is derived here.

- Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*: supplies the diagonal magnon Hamiltonian and the blocks of $\mathsf T_{\mathbf k}$.

- Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*: defines the full-position Fourier convention and the Nambu spinor.

- Notation and Conventions in research note *LSWT conventions*: fixes the thermodynamic and system-size notation.

### External Sources

- None.
