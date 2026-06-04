---
frontmatter-version: 1
title: Spin Hamiltonian
section: theory/common
status: draft
last-edited-by: codex
created: 2026-06-04
updated: 2026-06-04
related-notation: research-space/theory/common/notation.md
---

# Spin Hamiltonian

## Purpose

- This note fixes the common convention for writing a spin Hamiltonian.
- The Hamiltonian defined here is the object to which solver methods such as
  LSWT, tensor-network methods, neural quantum states, and later approaches are
  applied.
- This section specifies only the operator structure. Lattice geometry,
  magnetic ordering, momentum-space conventions, and solver-specific
  approximations are introduced in separate sections.


## 1. Local Hamiltonian

### Physical Site Set

A spin system is defined on a set of physical sites $\Lambda$. Each site
$i\in\Lambda$ has a local Hilbert space $\mathcal{H}_i$.
The full Hilbert space is the tensor product of all site Hilbert spaces.
$$
\mathcal{H}_{\Lambda}
= \bigotimes_{i\in\Lambda}\mathcal{H}_i.
$$
For a spin-$S_i$ degree of freedom, the local dimension of $\mathcal{H}_i$ is $2S_i+1$.


### Local-Term Decomposition

In this project, we consider a type of Hamiltonian which can be written as a sum of finite-support local terms.
$$
\hat{H}
=\sum_{X\subset\Lambda}\hat{h}_X .
$$

Here $X\subset\Lambda$ is a finite set of sites. The operator $\hat h_X$
acts nontrivially only on the sites in $X$ and acts as identity outside $X$.
The set $X$ is called the support of the term.

### Support of Operator

$$
\mathrm{supp}(\hat h_X)=X .
$$

The support size $\lvert X\rvert$ determines the body number of the interaction.

| Support size | Term type | Example term |
|---|---|---|
| $\lvert X\rvert=1$ | onsite term | Zeeman term $-\mathbf{h}_i\cdot\hat{\mathbf{S}}_i$; single-ion anisotropy $D_i(\hat S_i^z)^2$ |
| $\lvert X\rvert=2$ | two-site term | Heisenberg exchange $J_{ij}\hat{\mathbf{S}}_i\cdot\hat{\mathbf{S}}_j$; DMI $\mathbf{D}_{ij}\cdot(\hat{\mathbf{S}}_i\times\hat{\mathbf{S}}_j)$ |
| $\lvert X\rvert\ge 3$ | multi-site term | scalar spin chirality on a triangle; ring exchange on a plaquette |

This body number is a statement about site support, not about the polynomial
degree of spin operators inside $\hat h_X$. For example, a single-ion
anisotropy term is still an onsite term because its support contains one site,
even if it is quadratic in spin operators.

### $k$-Local Hamiltonian

A Hamiltonian is called $k$-local if every term has support size at most $k$.

$$
\lvert X\rvert\le k
\quad\text{for every local term } \hat h_X .
$$

## 2. Representative Spin Hamiltonians

This section lists standard spin Hamiltonians that commonly appear in
two-dimensional magnetic systems. The goal is not to classify every possible model,
but to show how familiar examples fit into the local-Hamiltonian convention
above.

In the examples below, $\langle i,j\rangle$ denotes a chosen set of interacting
site pairs. The precise link-counting convention is specified separately.

### Heisenberg Model

The Heisenberg model is the canonical isotropic two-site spin interaction:

$$
\hat H
= \sum_{\langle i,j\rangle}
J_{ij}\hat{\mathbf S}_i\cdot\hat{\mathbf S}_j
- \sum_i \mathbf h_i\cdot\hat{\mathbf S}_i .
$$

Here $J_{ij}$ is the exchange coupling and $\mathbf h_i$ is an external field.
If $J_{ij}>0$, the interaction favors antiferromagnetic alignment; if
$J_{ij}<0$, it favors ferromagnetic alignment.

### XXZ Model

The XXZ model separates the in-plane and out-of-plane exchange couplings:

$$
\hat H
= \sum_{\langle i,j\rangle}
\left[
J_{ij}^{xy}
\left(
\hat S_i^x\hat S_j^x+\hat S_i^y\hat S_j^y
\right)
+ J_{ij}^{z}\hat S_i^z\hat S_j^z
\right].
$$

This model is useful when spin-rotation symmetry is reduced from full
$\mathrm{SU}(2)$ symmetry to a residual $\mathrm{U}(1)$ symmetry about the
$z$ axis.

### Dzyaloshinskii-Moriya Interaction

The Dzyaloshinskii-Moriya interaction (DMI) is an antisymmetric two-site
exchange term:

$$
\hat H_{\mathrm{DMI}}
= \sum_{\langle i,j\rangle}
\mathbf D_{ij}\cdot
\left(
\hat{\mathbf S}_i\times\hat{\mathbf S}_j
\right).
$$

The vector $\mathbf D_{ij}$ determines the orientation and strength of the
antisymmetric exchange. DMI is allowed when the bond lacks inversion symmetry.

### General Bilinear Exchange

Many common two-site spin interactions can be written using a $3\times3$
exchange matrix:

$$
\hat H
= \sum_{\langle i,j\rangle}
\hat{\mathbf S}_i^{\mathsf T}
\mathbf J_{ij}
\hat{\mathbf S}_j
- \sum_i \mathbf h_i\cdot\hat{\mathbf S}_i .
$$

Equivalently,

$$
\hat{\mathbf S}_i^{\mathsf T}
\mathbf J_{ij}
\hat{\mathbf S}_j
=
\sum_{\alpha,\beta\in\{x,y,z\}}
\hat S_i^\alpha
J_{ij}^{\alpha\beta}
\hat S_j^\beta .
$$

By choosing $\mathbf J_{ij}$ appropriately, this form includes the Heisenberg,
XXZ, compass-like, Kitaev-like, symmetric anisotropic, and antisymmetric DMI
interactions. This is the most convenient common form for solver interfaces
that accept general bilinear spin Hamiltonians.

### Multi-Site Interactions

Some spin Hamiltonians include genuine multi-site terms. A standard three-site
example is the scalar spin chirality term on triangular plaquettes:

$$
\hat H_{\chi}
= K_{\chi}
\sum_{\triangle(i,j,k)}
\hat{\mathbf S}_i\cdot
\left(
\hat{\mathbf S}_j\times\hat{\mathbf S}_k
\right).
$$

Four-site ring exchange is another common plaquette interaction:

$$
\hat H_{\mathrm{ring}}
= K
\sum_{\square(i,j,k,l)}
\left(
\hat P_{ijkl}
+ \hat P_{ijkl}^{-1}
\right).
$$

Here $\hat P_{ijkl}$ cyclically permutes the spins around a plaquette. These
terms are not bilinear in spin operators, but they still fit the local-term
decomposition $\hat H=\sum_X\hat h_X$ by using supports with
$\lvert X\rvert\ge 3$.

### Summary

| Model or term | Support size | Main coupling object | Typical role |
|---|---|---|---|
| Heisenberg model | $\lvert X\rvert=2$ | scalar exchange $J_{ij}$ | isotropic two-site exchange |
| XXZ model | $\lvert X\rvert=2$ | anisotropic couplings $J_{ij}^{xy}, J_{ij}^{z}$ | easy-axis or easy-plane exchange |
| DMI | $\lvert X\rvert=2$ | DMI vector $\mathbf D_{ij}$ | antisymmetric exchange on non-centrosymmetric bonds |
| General bilinear exchange | $\lvert X\rvert=2$ | exchange matrix $\mathbf J_{ij}$ | unified form for bilinear two-site interactions |
| Scalar spin chirality | $\lvert X\rvert=3$ | chiral coupling $K_{\chi}$ | three-site chiral plaquette interaction |
| Ring exchange | $\lvert X\rvert=4$ | cyclic permutation $\hat P_{ijkl}$ | four-site plaquette interaction |

References:

- J. Kempe, A. Kitaev, and O. Regev, "The Complexity of the Local Hamiltonian Problem," SIAM Journal on Computing 35, 1070 (2006). [arXiv:quant-ph/0406180](https://arxiv.org/abs/quant-ph/0406180), [DOI](https://doi.org/10.1137/S0097539704445226).
- S. Bravyi, M. B. Hastings, and F. Verstraete, "Lieb-Robinson bounds and the generation of correlations and topological quantum order," Physical Review Letters 97, 050401 (2006). [arXiv:quant-ph/0603121](https://arxiv.org/abs/quant-ph/0603121), [DOI](https://doi.org/10.1103/PhysRevLett.97.050401).
