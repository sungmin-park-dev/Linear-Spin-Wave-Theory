---
frontmatter-version: 1
title: Thermodynamics
doc-path: docs/lswt/02-observables
status: draft
last-edited-by: claude
created: 2026-06-03
updated: 2026-09-30
source: docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf
source-section: "Thermodynamics in Linear Spin Wave Theory: Partition Function, Internal Energy, Free Energy, Entropy Expression, Specific Heat (source pp. 13–15)"
---

# Thermodynamics

In thermal equilibrium the LSWT Hamiltonian describes an ideal gas of magnons on top of the classical reference configuration. The partition function therefore factorizes over the magnon modes $(\mathbf k,n)$, and the free energy, internal energy, entropy, and heat capacity follow from single-oscillator expressions summed over the magnetic Brillouin zone (MBZ). Energy, temperature, and entropy symbols follow [Notation and Conventions](../00-foundations/notation-and-conventions.md); the inverse temperature is $\beta=(k_{\mathrm B}T)^{-1}$ and the entropy is $\mathcal S$.

## Validity of the Magnon-Gas Description

We use the diagonal Hamiltonian of [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md),

$$
\hat H_{\mathrm{LSWT}}
=E_{\mathrm{cl}}+\hat H_2
=E_{\mathrm{GS}}+\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}\varepsilon_{n\mathbf k}\,\hat n_{\mathbf kn},
\qquad
\hat n_{\mathbf kn}=\hat b_{\mathbf kn}^\dagger\hat b_{\mathbf kn},
$$

with $E_{\mathrm{GS}}=E_{\mathrm{cl}}+\Delta E_{\mathrm{zp}}$. The classical reference configuration and the magnon energies $\varepsilon_{n\mathbf k}$ are held fixed as the temperature changes, and the quartic and higher Holstein–Primakoff terms that couple magnons are omitted. The results below are therefore the leading low-temperature description. Its corrections grow with the thermal magnon density, and the description fails when the thermal reduction of a sublattice moment, discussed in [Magnon Observables](magnon-observables.md), becomes comparable to the spin length. In two dimensions, the interpretation of these results as properties of an ordered phase requires the qualifications stated in [LSWT Overview](../00-foundations/lswt-overview.md).

All quantities below assume $\varepsilon_{n\mathbf k}>0$, which holds when $\mathsf H_{\mathbf k}$ is positive definite. The section on zero modes describes the limit $\varepsilon_{n\mathbf k}\to0$.

## Partition Function

The Gibbs state $\hat\rho=\exp(-\beta\hat H_{\mathrm{LSWT}})/Z$ has the partition function $Z=\operatorname{Tr}\exp(-\beta\hat H_{\mathrm{LSWT}})$. The number operators of different modes commute, so the trace factorizes into independent geometric series, one for each mode:

$$
Z=\exp(-\beta E_{\mathrm{GS}})\prod_{\mathbf k\in\mathrm{MBZ}}\prod_{n=1}^{N_{\mathrm{sub}}}\frac{1}{1-\exp(-\beta\varepsilon_{n\mathbf k})},
\qquad
-\ln Z=\beta E_{\mathrm{GS}}+\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}\ln\!\left(1-\exp(-\beta\varepsilon_{n\mathbf k})\right).
$$ {#eq-lswt-partition-function}

Each geometric series converges because $\varepsilon_{n\mathbf k}>0$. The constant $E_{\mathrm{GS}}$ contributes the factor $\exp(-\beta E_{\mathrm{GS}})$; it shifts the internal and free energies but drops out of the entropy and heat capacity, which depend only on the thermal part of $\ln Z$.

The thermal occupation of mode $(\mathbf k,n)$ is the Bose–Einstein distribution

$$
\langle\hat n_{\mathbf kn}\rangle
=n_{\mathrm B}(\varepsilon_{n\mathbf k}),
\qquad
n_{\mathrm B}(\varepsilon)=\frac{1}{\exp(\beta\varepsilon)-1}.
$$ {#eq-lswt-bose-einstein-distribution}

## Internal and Free Energies

The internal energy is the thermal average of the Hamiltonian, $U=\langle\hat H_{\mathrm{LSWT}}\rangle=-\partial\ln Z/\partial\beta$. Differentiating @eq-lswt-partition-function gives

$$
U=E_{\mathrm{GS}}+\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}\varepsilon_{n\mathbf k}\,n_{\mathrm B}(\varepsilon_{n\mathbf k}).
$$ {#eq-lswt-internal-energy}

The internal energy thus consists of the classical energy, the zero-point correction, and the energy of the thermally excited magnons. The Helmholtz free energy, defined by $Z=\exp(-\beta F)$, is

$$
F=E_{\mathrm{GS}}+k_{\mathrm B}T\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}\ln\!\left(1-\exp(-\beta\varepsilon_{n\mathbf k})\right).
$$ {#eq-lswt-free-energy}

Each logarithm is negative, so the thermal magnons lower $F$ below $E_{\mathrm{GS}}$. At $T=0$ both $U$ and $F$ reduce to $E_{\mathrm{GS}}$.

## Entropy

The entropy of the Gibbs state is the von Neumann entropy $\mathcal S=-k_{\mathrm B}\operatorname{Tr}(\hat\rho\ln\hat\rho)$. Inserting $\ln\hat\rho=-\beta\hat H_{\mathrm{LSWT}}-\ln Z$ gives $\mathcal S=(U-F)/T$, which coincides with the thermodynamic relation $\mathcal S=-\partial F/\partial T$. With @eq-lswt-internal-energy and @eq-lswt-free-energy,

$$
\mathcal S=k_{\mathrm B}\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}
\left[\beta\varepsilon_{n\mathbf k}\,n_{\mathrm B}(\varepsilon_{n\mathbf k})-\ln\!\left(1-\exp(-\beta\varepsilon_{n\mathbf k})\right)\right].
$$

The relations $\beta\varepsilon=\ln[(1+n_{\mathrm B})/n_{\mathrm B}]$ and $\ln(1-\exp(-\beta\varepsilon))=-\ln(1+n_{\mathrm B})$, which follow from the definition of $n_{\mathrm B}$, express the entropy through the occupations alone:

$$
\mathcal S=k_{\mathrm B}\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}
\Big[\big(1+n_{\mathbf kn}\big)\ln\big(1+n_{\mathbf kn}\big)-n_{\mathbf kn}\ln n_{\mathbf kn}\Big],
\qquad
n_{\mathbf kn}=n_{\mathrm B}(\varepsilon_{n\mathbf k}).
$$ {#eq-lswt-magnon-entropy}

This is the entropy of an ideal Bose gas with the occupations $n_{\mathbf kn}$. The zero-point correction does not enter, since it is a temperature-independent constant.

## Heat Capacity

The heat capacity is the temperature derivative of the internal energy, $C=\partial U/\partial T$. Only the occupations depend on temperature, and $\partial n_{\mathrm B}(\varepsilon)/\partial T=(\varepsilon/k_{\mathrm B}T^2)\exp(\beta\varepsilon)/(\exp(\beta\varepsilon)-1)^2$, so

$$
C=k_{\mathrm B}\sum_{\mathbf k\in\mathrm{MBZ}}\sum_{n=1}^{N_{\mathrm{sub}}}
\frac{(\beta\varepsilon_{n\mathbf k})^2}{4\sinh^2(\beta\varepsilon_{n\mathbf k}/2)}.
$$ {#eq-lswt-magnon-heat-capacity}

Each mode contributes the heat capacity of a harmonic oscillator. The contribution approaches $k_{\mathrm B}$ for $\varepsilon_{n\mathbf k}\ll k_{\mathrm B}T$ and is exponentially small for $\varepsilon_{n\mathbf k}\gg k_{\mathrm B}T$, so at low temperature the heat capacity is dominated by the lowest magnon energies. The relation $C=T\,\partial\mathcal S/\partial T$ gives the same expression.

## Normalization and Momentum Sums

The quantities above are extensive: the sums run over the $N_{\mathrm{uc}}$ momenta of the MBZ and the $N_{\mathrm{sub}}$ bands. Values per magnetic site follow by dividing by $N_{\mathrm{site}}=N_{\mathrm{uc}}N_{\mathrm{sub}}$; for example, $U/N_{\mathrm{site}}=\mathcal E_{\mathrm{GS}}+N_{\mathrm{site}}^{-1}\sum_{\mathbf k,n}\varepsilon_{n\mathbf k}n_{\mathrm B}(\varepsilon_{n\mathbf k})$. In the thermodynamic limit the momentum sum becomes an integral over the MBZ, $N_{\mathrm{uc}}^{-1}\sum_{\mathbf k}\to A_{\mathrm{MBZ}}^{-1}\int_{\mathrm{MBZ}}d^2\mathbf k$, where $A_{\mathrm{MBZ}}$ is the area of the MBZ. A finite momentum mesh approximates this integral.

## Zero Modes

When $\mathsf H_{\mathbf k}$ is only positive semidefinite, some magnon energy vanishes, for example at a Goldstone mode. Near an isolated zero at $\mathbf k_0$ we assume $\varepsilon_{n\mathbf k}\propto|\mathbf k-\mathbf k_0|^a$ with $a>0$; linear ($a=1$) and quadratic ($a=2$) dispersions are the common cases. The summands behave as follows as $\varepsilon\to0$:

| Quantity | Summand for $\varepsilon\ll k_{\mathrm B}T$ | Limit at $\varepsilon=0$ |
|---|---|---|
| $U$ | $\varepsilon\,n_{\mathrm B}(\varepsilon)\to k_{\mathrm B}T$ | finite |
| $C$ | $k_{\mathrm B}$ | finite |
| $F$ | $k_{\mathrm B}T\ln(\beta\varepsilon)$ | logarithmically divergent |
| $\mathcal S$ | $k_{\mathrm B}[1+\ln(k_{\mathrm B}T/\varepsilon)]$ | logarithmically divergent |

The logarithmic divergences are integrable. In two dimensions $\int d^2\mathbf q\,\ln|\mathbf q|$ converges near $\mathbf q=0$ for any $a>0$, so $F$, $U$, $\mathcal S$, and $C$ remain finite in the thermodynamic limit. A zero mode at an isolated momentum has vanishing weight in the MBZ integral. On a finite mesh that contains $\mathbf k_0$, the treatment of that single point is a discretization choice, and the continuum values do not depend on it. Quantities weighted by the occupations without a compensating factor of $\varepsilon$, such as the boson number and the sublattice moment, can diverge instead; they are treated in [Magnon Observables](magnon-observables.md).

## References

### Internal Documents

- [Notation and Conventions](../00-foundations/notation-and-conventions.md): defines $\beta$, $\mathcal S$, $\varepsilon_{n\mathbf k}$, $E_{\mathrm{cl}}$, $\Delta E_{\mathrm{zp}}$, $E_{\mathrm{GS}}$, and the system-size symbols.
- [Paraunitary Diagonalization](../01-derivation/paraunitary-diagonalization.md): supplies the diagonal magnon Hamiltonian, the zero-point correction, and the positive-definiteness condition.
- [LSWT Overview](../00-foundations/lswt-overview.md): states the finite-temperature qualification for two-dimensional ordered states.
- [Magnon Observables](magnon-observables.md): uses the Bose–Einstein occupations for boson numbers and sublattice moments.
- [Thermodynamic Derivations](../04-appendices/thermodynamic-derivations.md): is reserved for longer entropy and correlation-matrix derivations.

### External Sources

- None.
