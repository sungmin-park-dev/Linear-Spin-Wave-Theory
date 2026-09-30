# Summary of Notation and Symbols

This document summarizes the symbols, notation, and LaTeX command conventions used in the Linear Spin Wave Theory documentation.

## Custom LaTeX Commands

The original LaTeX document uses custom commands for mathematical notation. In these Markdown documents, they are expanded as follows:

| LaTeX Command | Markdown Rendering | Description |
|--------------|-------------------|-------------|
| `\NBCP` | NBCP | Material abbreviation |
| `\NBCPatom` | Na₂BaCo(PO₄)₂ | Material formula |
| `\kvec` | **k** | Momentum vector |
| `\bmdelta` | **δ** | Bold delta |
| `\bfr` | **r** | Position vector |
| `\bth` | **θ** | Bold theta |
| `\ha` | $\hat{a}$ | Hat operator a |
| `\hc` | $\hat{c}$ | Hat operator c |
| `\hI` | $\hat{\mathbb{I}}$ | Identity operator |
| `\hP` | $\hat{P}$ | Projection operator |
| `\rD` | $\mathrm{D}$ | Upright D |
| `\rU` | $\mathrm{U}$ | Upright U |
| `\ri` | $\mathrm{i}$ | Upright i (imaginary unit) |
| `\cA` | $\mathcal{A}$ | Calligraphic A |
| `\cB` | $\mathcal{B}$ | Calligraphic B |
| `\cC` | $\mathcal{C}$ | Calligraphic C |
| `\cD` | $\mathcal{D}$ | Calligraphic D |
| `\cE` | $\mathcal{E}$ | Calligraphic E |
| `\cG` | $\mathcal{G}$ | Calligraphic G |
| `\cH` | $\mathcal{H}$ | Calligraphic H (Hamiltonian) |
| `\cJ` | $\mathcal{J}$ | Calligraphic J |
| `\cM` | $\mathcal{M}$ | Calligraphic M |
| `\cN` | $\mathcal{N}$ | Calligraphic N |
| `\cP` | $\mathcal{P}$ | Calligraphic P |
| `\cQ` | $\mathcal{Q}$ | Calligraphic Q |
| `\cR` | $\mathcal{R}$ | Calligraphic R |
| `\cU` | $\mathcal{U}$ | Calligraphic U |
| `\Id` | $\mathrm{Id}$ | Identity |
| `\Tr` | $\mathrm{Tr}$ | Trace |
| `\Pf` | $\mathrm{Pf}$ | Pfaffian |

## Physical Quantities and Symbols

### Spin Hamiltonian and Related Quantities

| Symbol | Description |
|--------|-------------|
| $\hat{H}$ | Spin Hamiltonian |
| $J_{ij}^{\alpha\beta}$ | Exchange coupling between sites $i$ and $j$ for spin components $\alpha$ and $\beta$ |
| $h_j^{\alpha}$ | External magnetic field at site $j$ along direction $\alpha$ |
| $E_{\rm cl}$ | Classical ground state energy |
| $\mathbf{S}_j$ | Classical spin vector at site $j$ |
| $\hat{\mathbf{S}}_j$ | Quantum spin operator at site $j$ |
| $\delta\hat{\mathbf{S}}_j$ | Quantum fluctuation operator ($\hat{\mathbf{S}}_j = \mathbf{S}_j + \delta\hat{\mathbf{S}}_j$) |
| $\widetilde{\mathbf{S}}_j$ | Spin operator in the local reference frame |

### Index Conventions and Coordinate Systems

| Symbol | Description |
|--------|-------------|
| $i, j$ | Unit cell indices |
| $\mu, \nu, \sigma, \rho$ | Sublattice indices (typically $a, b, c, \ldots$) |
| $I = (i, \mu), J = (j, \nu)$ | Composite site indices combining unit cell and sublattice |
| $\mathbf{r}_i$ | Position of unit cell $i$ |
| $\boldsymbol{\delta}_\mu$ | Position of sublattice $\mu$ within the unit cell |
| $\mathbf{R}_I = \mathbf{r}_i + \boldsymbol{\delta}_\mu$ | Position of spin at site $I = (i, \mu)$ |
| $\alpha, \beta$ | Spin component indices ($x, y, z$ or $+, -, 0$) |
| $\mathbf{R}_j$ | Rotation matrix from laboratory frame to local frame at site $j$ |
| $\hat{\mathbf{e}}_j^{\alpha}$ | Basis vectors of local frame at site $j$ ($\alpha = +, -, 0$) |

### Boson Operators and Transformations

| Symbol | Description |
|--------|-------------|
| $\hat{a}_j, \hat{a}_j^{\dagger}$ | Bosonic annihilation and creation operators |
| $\hat{n}_j = \hat{a}_j^{\dagger}\hat{a}_j$ | Boson number operator |
| $\hat{b}_{\mathbf{k}\mu}, \hat{b}_{\mathbf{k}\mu}^{\dagger}$ | Momentum-space bosonic operators for sublattice $\mu$ |
| $\hat{\beta}_{\mathbf{k}\mu}, \hat{\beta}_{\mathbf{k}\mu}^{\dagger}$ | Diagonalized magnon operators (after Bogoliubov transformation) |
| $\Psi_{\mathbf{k}} = (\hat{\mathbf{b}}_{\mathbf{k}}, \hat{\mathbf{b}}_{-\mathbf{k}}^{\dagger})^T$ | Nambu spinor in momentum space |
| $\widetilde{\Psi}_{\mathbf{k}} = (\hat{\boldsymbol{\beta}}_{\mathbf{k}}, \hat{\boldsymbol{\beta}}_{-\mathbf{k}}^{\dagger})^T$ | Diagonalized Nambu spinor |

### Matrices and Transformation Operators

| Symbol | Description |
|--------|-------------|
| $\mathsf{H}_{\mathbf{k}}$ | Hamiltonian matrix in BdG form |
| $\mathsf{A}_{\mathbf{k}}, \mathsf{B}_{\mathbf{k}}$ | Blocks of the Hamiltonian matrix $\mathsf{H}_{\mathbf{k}}$ |
| $\mathsf{T}_{\mathbf{k}}$ | Paraunitary transformation matrix for Bogoliubov diagonalization |
| $\mathsf{E}_{\mathbf{k}}$ | Diagonal matrix of magnon energies |
| $\mathsf{N}_{\mathbf{k}}(t)$ | Time-dependent correlation matrix in magnon basis |
| $\mathsf{U}^{\alpha}$ | Spin projection operator in laboratory frame |
| $\mathsf{S}_{\mathbf{k}}$ | Matrix encoding spin magnitudes and sublattice phase factors |
| $\mathsf{C}^{\alpha\beta}(\mathbf{k}, t)$ | Correlation matrix in laboratory frame |
| $\mathsf{J}$ | Symplectic metric tensor |

### Physical Observables

| Symbol | Description |
|--------|-------------|
| $E_{\mathbf{k},\mu}$ | Magnon energy for band $\mu$ at momentum $\mathbf{k}$ |
| $n_{\mathbf{k},\mu} = 1/(e^{\beta E_{\mathbf{k},\mu}}-1)$ | Bose-Einstein distribution function |
| $E_0$ | Zero-point energy |
| $\mathcal{C}^{\alpha\beta}_{\mu\nu}(\mathbf{k}, t)$ | Sublattice spin-spin correlation function |
| $\mathcal{S}^{\alpha\beta}(\mathbf{k}, \omega)$ | Dynamic structure factor |
| $\mathcal{A}^{\alpha\beta}(\mathbf{k}, \omega)$ | Spectral function |
| $\Omega_{n,\mathbf{k}}$ | Berry curvature of the $n$-th magnon band |
| $C_n$ | Chern number of the $n$-th magnon band |

## References

- Original LaTeX document: `research-space/sources/lswt/note_lswt_restructured.tex`
- For detailed theory, see the [theory sections](sections/)
