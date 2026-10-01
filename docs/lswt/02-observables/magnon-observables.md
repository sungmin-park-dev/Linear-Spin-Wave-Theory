---
frontmatter-version: 1
title: Magnon Observables
doc-path: docs/lswt/02-observables
status: draft
last-edited-by: claude
created: 2026-06-07
updated: 2026-10-01
source: docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf
source-section: "Physical Quantities in Linear Spin Wave Theory (source p. 12, Table I); Number and Spin moment from Correlation Matrix (source pp. 15–16)"
---

# Magnon Observables

After the paraunitary diagonalization of [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md), every LSWT observable is built from two ingredients at each momentum of the magnetic Brillouin zone (MBZ): the magnon energies $\varepsilon_{n\mathbf k}$ and the transformation $\mathsf T_{\mathbf k}$. This document collects the quantities that follow directly from these ingredients, namely the magnon spectrum, the equal-time correlation matrix of the Holstein–Primakoff bosons, the boson number, and the reduced ordered moment. It also serves as an index to the documents that own the remaining observables.

## Magnon Spectrum and Band Structure

The magnon energies $\varepsilon_{n\mathbf k}$, $n=1,\ldots,N_{\mathrm{sub}}$, are the positive eigenvalues of $\Sigma_3\mathsf H_{\mathbf k}$ and are periodic over the MBZ. Each band is a branch $\varepsilon_{n\mathbf k}$ that continues across momentum; at crossings the band label is a matter of convention, since the energies are ordered at each momentum. The negative eigenvalues $-\varepsilon_{n,-\mathbf k}$ are the particle–hole partners and carry no additional information.

The magnetic unit cell of an ordered state may contain several crystallographic unit cells. A band structure plotted along a path of the crystallographic Brillouin zone (CBZ) then shows all $N_{\mathrm{sub}}$ bands of the magnetic cell at every momentum, including branches folded back by magnetic reciprocal-lattice vectors. Which folded branch carries weight at a given external momentum is determined by the one-magnon weights of [Structure Factor and Spectral Function](structure-factor-and-spectral-function.md), not by the energies alone. At a momentum where $\mathsf H_{\mathbf k}$ is only positive semidefinite, some $\varepsilon_{n\mathbf k}$ vanishes; the energies remain well defined as limits, although $\mathsf T_{\mathbf k}$ may not exist there.

## Ground-State Energy

The LSWT ground-state energy $E_{\mathrm{GS}}=E_{\mathrm{cl}}+\Delta E_{\mathrm{zp}}$ and the zero-point correction $\Delta E_{\mathrm{zp}}=\frac12\sum_{\mathbf k}(\sum_n\varepsilon_{n\mathbf k}-\operatorname{Tr}\mathsf A_{\mathbf k})$ are derived with the diagonal magnon Hamiltonian in [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md). The energy per magnetic site is $\mathcal E_{\mathrm{GS}}=E_{\mathrm{GS}}/N_{\mathrm{site}}$. The correction depends on the energies alone, not on $\mathsf T_{\mathbf k}$, and it inherits the provisional trace convention of [Momentum-Space BdG Hamiltonian](../01-derivation/momentum-space-bdg-hamiltonian.md).

## Equal-Time Correlation Matrix

The equal-time averages of all products of two Holstein–Primakoff bosons at momentum $\mathbf k$ form the $2N_{\mathrm{sub}}\times2N_{\mathrm{sub}}$ correlation matrix

$$
\Big[\big\langle\hat\Psi_{\mathbf k}\hat\Psi_{\mathbf k}^\dagger\big\rangle\Big]
=\begin{pmatrix}
\langle\hat a_{\mathbf k\mu}\hat a_{\mathbf k\nu}^\dagger\rangle & \langle\hat a_{\mathbf k\mu}\hat a_{-\mathbf k\nu}\rangle\\
\langle\hat a_{-\mathbf k\mu}^\dagger\hat a_{\mathbf k\nu}^\dagger\rangle & \langle\hat a_{-\mathbf k\mu}^\dagger\hat a_{-\mathbf k\nu}\rangle
\end{pmatrix}_{\mu\nu}
=\mathsf T_{\mathbf k}\,\mathsf N_{\mathbf k}(0)\,\mathsf T_{\mathbf k}^\dagger,
$$ {#eq-lswt-boson-correlation-matrix}

where each block is an $N_{\mathrm{sub}}\times N_{\mathrm{sub}}$ matrix with the sublattice indices $\mu,\nu$. The second equality substitutes $\hat\Psi_{\mathbf k}=\mathsf T_{\mathbf k}\hat\Phi_{\mathbf k}$. In the Gibbs state the magnon correlation matrix $\mathsf N_{\mathbf k}(0)=\operatorname{diag}(1+n_{1\mathbf k},\ldots,n_{1,-\mathbf k},\ldots)$ of [Spin Correlations](spin-correlations.md) is diagonal, with the occupations $n_{n\mathbf k}=n_{\mathrm B}(\varepsilon_{n\mathbf k})$ of [Thermodynamics](thermodynamics.md). In the ground state, the magnon vacuum, the occupations vanish and $\mathsf N_{\mathbf k}(0)=\operatorname{diag}(\mathsf I_{N_{\mathrm{sub}}},0)$. The anomalous averages $\langle\hat a\hat a\rangle$ are nonzero even at zero temperature when $\mathsf H_{\mathbf k}$ contains the anomalous block $\mathsf B_{\mathbf k}$.

## Boson Number and Reduced Moment

The average number of Holstein–Primakoff bosons on sublattice $\mu$ is $\langle\hat n_\mu\rangle=N_{\mathrm{uc}}^{-1}\sum_i\langle\hat a_{i\mu}^\dagger\hat a_{i\mu}\rangle=N_{\mathrm{uc}}^{-1}\sum_{\mathbf k}\langle\hat a_{\mathbf k\mu}^\dagger\hat a_{\mathbf k\mu}\rangle$. The lower-right block of @eq-lswt-boson-correlation-matrix contains these averages at $-\mathbf k$, and the momentum sum covers all momenta, so

$$
\langle\hat n_\mu\rangle
=\frac{1}{N_{\mathrm{uc}}}\sum_{\mathbf k\in\mathrm{MBZ}}\Big[\mathsf T_{\mathbf k}\mathsf N_{\mathbf k}(0)\mathsf T_{\mathbf k}^\dagger\Big]_{N_{\mathrm{sub}}+\mu,\,N_{\mathrm{sub}}+\mu}
=\frac{1}{N_{\mathrm{uc}}}\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}
\Big[\big|(\mathsf P_{\mathbf k})_{\mu n}\big|^2n_{n\mathbf k}+\big|(\mathsf Q_{\mathbf k})_{\mu n}\big|^2\big(1+n_{n\mathbf k}\big)\Big].
$$ {#eq-lswt-boson-number}

The second form uses the blocks of the Bogoliubov transformation $\hat a_{\mathbf k\mu}=\sum_n[(\mathsf P_{\mathbf k})_{\mu n}\hat b_{\mathbf kn}+(\mathsf Q_{-\mathbf k})_{\mu n}\hat b_{-\mathbf kn}^\dagger]$ of [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md) and relabels $-\mathbf k\to\mathbf k$ in the hole-column terms. The $\mathsf Q$ term survives at zero temperature: the quantum reduction $N_{\mathrm{uc}}^{-1}\sum_{\mathbf k,n}|(\mathsf Q_{\mathbf k})_{\mu n}|^2$ arises only from the mixing of creation and annihilation operators and vanishes when $\mathsf B_{\mathbf k}=0$, as for a collinear ferromagnet whose Hamiltonian conserves the total spin component along the ordered moment. Exchange anisotropy transverse to the moment, such as $J^{xx}\ne J^{yy}$ in the local frame or Kitaev and off-diagonal couplings, produces $\mathsf B_{\mathbf k}\ne0$ and a quantum reduction even in a collinear ferromagnet. The $\mathsf P$ and $\mathsf Q$ terms proportional to $n_{n\mathbf k}$ give the thermal reduction.

The boson number reduces the ordered moment. With $\hat{\widetilde S}_I^0=S_I-\hat n_I$ and the local longitudinal direction $\mathbf n_\mu$ of [Classical Order and Local Frame](../00-foundations/classical-order-and-local-frame.md), the average spin of sublattice $\mu$ is

$$
\mathbf m_\mu=\big(S_\mu-\langle\hat n_\mu\rangle\big)\,\mathbf n_\mu,
$$ {#eq-lswt-reduced-moment}

since the transverse components average to zero. The magnetization per site is $N_{\mathrm{sub}}^{-1}\sum_\mu\mathbf m_\mu$. These moments enter the elastic Bragg scattering of [Spin Correlations](spin-correlations.md). The reduction $\langle\hat n_\mu\rangle$ is of order $S^0$, compared with $S_\mu$, so the expansion is controlled only while $\langle\hat n_\mu\rangle\ll S_\mu$. A value $\langle\hat n_\mu\rangle\ge S_\mu$ would remove or reverse the moment and signals that LSWT about the chosen reference configuration no longer applies.

### Zero Modes

At a zero mode the summand of @eq-lswt-boson-number can diverge. Near a Goldstone mode of an antiferromagnet, $\varepsilon_{n\mathbf k}\propto|\mathbf k-\mathbf k_0|$ and $|\mathsf P_{\mathbf k}|^2,|\mathsf Q_{\mathbf k}|^2\propto1/\varepsilon_{n\mathbf k}$, as for the single mode of [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md). At zero temperature the summand then grows as $1/|\mathbf k-\mathbf k_0|$, which is integrable in two dimensions, and the quantum reduction is finite. At nonzero temperature $n_{\mathrm B}(\varepsilon)\simeq k_{\mathrm B}T/\varepsilon$ adds a factor $1/\varepsilon$, and the summand grows as $|\mathbf k-\mathbf k_0|^{-2}$; the same power follows for a ferromagnetic Goldstone mode with $\varepsilon\propto|\mathbf k-\mathbf k_0|^2$ and bounded $\mathsf P_{\mathbf k}$. In two dimensions this gives a logarithmically divergent thermal boson number. The divergence is the LSWT form of the absence of spontaneous continuous-symmetry breaking at nonzero temperature in two dimensions, discussed in [LSWT Overview](../00-foundations/lswt-overview.md). The thermodynamic potentials of [Thermodynamics](thermodynamics.md) remain finite in the same situation.

## Index of Observables

| Observable | Ingredients | Owner document |
|---|---|---|
| Magnon energies and bands | $\varepsilon_{n\mathbf k}$ | This document; [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md) |
| Ground-state energy, zero-point correction | $\varepsilon_{n\mathbf k}$, $\operatorname{Tr}\mathsf A_{\mathbf k}$ | [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md) |
| Boson number, reduced moment, magnetization | $\mathsf T_{\mathbf k}$, $n_{n\mathbf k}$ | This document |
| Partition function, $F$, $U$, $\mathcal S$, $C$ | $\varepsilon_{n\mathbf k}$ | [Thermodynamics](thermodynamics.md) |
| Spin correlation function, elastic correlation | $\mathsf T_{\mathbf q}$, $\mathsf N_{\mathbf q}(t)$, $\mathbf m_\mu$ | [Spin Correlations](spin-correlations.md) |
| Static and dynamic structure factors, spectral function | One-magnon weights $W_n^{\alpha\beta}$ | [Structure Factor and Spectral Function](structure-factor-and-spectral-function.md) |
| Berry curvature, Chern number, thermal Hall conductivity | $\mathsf T_{\mathbf k}$, $\partial_{\mathbf k}\mathsf H_{\mathbf k}$ | [Topological Magnon Quantities](topological-magnon-quantities.md) |
| Classical lower bound and candidate orders | Classical energy only | [Appendix: Luttinger–Tisza Method](../04-appendices/luttinger-tisza-method.md) |

## References

### Internal Documents

- [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md): supplies $\varepsilon_{n\mathbf k}$, $\mathsf T_{\mathbf k}$ with its blocks $\mathsf P_{\mathbf k}$ and $\mathsf Q_{\mathbf k}$, and the ground-state energy.
- [Momentum-Space BdG Hamiltonian](../01-derivation/momentum-space-bdg-hamiltonian.md): defines the Nambu spinor and the BdG blocks.
- [Classical Order and Local Frame](../00-foundations/classical-order-and-local-frame.md): defines the ordered directions $\mathbf n_\mu$.
- [Thermodynamics](thermodynamics.md): defines the occupations and the thermodynamic potentials.
- [Spin Correlations](spin-correlations.md): defines $\mathsf N_{\mathbf q}(t)$ and uses the reduced moments.
- [Structure Factor and Spectral Function](structure-factor-and-spectral-function.md): determines the spectral weight of folded bands.
- [Topological Magnon Quantities](topological-magnon-quantities.md): builds topological responses from the bands and eigenvectors.
- [Appendix: Luttinger–Tisza Method](../04-appendices/luttinger-tisza-method.md): bounds the classical energy of candidate reference configurations.
- [LSWT Overview](../00-foundations/lswt-overview.md): states the finite-temperature qualification in two dimensions.

### External Sources

- None.
