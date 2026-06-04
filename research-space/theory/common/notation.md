# Common Notation

## Purpose

이 문서는 spin-system theory 문서에서 공통으로 쓰는 기호 규칙을 정리한다.

## Site and Support

| Symbol | Meaning | Scope | Notes |
|---|---|---|---|
| $\Lambda$ | 전체 physical site 집합 | common | 모든 spin Hamiltonian term의 site universe |
| $i,j,k$ | physical site index | common | $i,j,k\in\Lambda$ |
| $X,Y$ | finite support | common | local term이 작용하는 site 집합 |
| $|X|$ | support size | common | body 수를 나타냄 |
| $\ell$ | two-site link | common | $\ell=(i,j)$, two-site term에만 사용 |

## Hilbert Space

| Symbol | Meaning | Scope | Notes |
|---|---|---|---|
| $\mathcal{H}_i$ | site $i$의 local Hilbert space | common | spin-$S_i$이면 dimension $2S_i+1$ |
| $\mathcal{H}_\Lambda$ | 전체 Hilbert space | common | $\bigotimes_{i\in\Lambda}\mathcal{H}_i$ |
| $\mathbb{1}_i$ | site $i$의 identity operator | common | 필요할 때만 명시 |
| $\mathbb{1}_{\Lambda\setminus X}$ | support 밖 identity | common | $\hat h_X$를 전체 Hilbert space에 embedding할 때 사용 |

## Hamiltonian Terms

| Symbol | Meaning | Scope | Notes |
|---|---|---|---|
| $\hat H$ | 전체 Hamiltonian | common | $\hat H=\sum_X\hat h_X$ |
| $\hat h_X$ | support $X$의 local term | common | $X$ 밖에서는 identity로 작용 |
| $\hat h_i$ | onsite term | common | $X=\{i\}$인 경우 |
| $\hat h_{ij}$ | two-site term | common | $X=\{i,j\}$인 경우 |
| $\hat h_{ijk}$ | three-site term | common | $X=\{i,j,k\}$인 경우 |
| $\hat h_P$ | plaquette/ring term | common | $P$는 plaquette 또는 ring support |

## Spin Operators

| Symbol | Meaning | Scope | Notes |
|---|---|---|---|
| $\hat{\mathbf{S}}_i$ | site $i$의 spin vector operator | common | $(\hat S_i^x,\hat S_i^y,\hat S_i^z)$ |
| $\hat S_i^\alpha$ | spin component | common | $\alpha\in\{x,y,z\}$ |
| $S_i$ | site $i$의 spin length | common | local Hilbert space dimension과 연결 |
| $\alpha,\beta,\gamma$ | spin component index | common | Cartesian component에 사용 |

## Two-Site Couplings

| Symbol | Meaning | Scope | Notes |
|---|---|---|---|
| $\ell=(i,j)$ | oriented two-site link | common | endpoint order가 필요한 경우 사용 |
| $\mathbf{J}_\ell$ | link $\ell$의 exchange matrix | common | $3\times3$ matrix |
| $J_\ell^{\alpha\beta}$ | exchange matrix component | common | $\hat S_i^\alpha J_\ell^{\alpha\beta}\hat S_j^\beta$ |
| $\mathbf{h}_i$ | site $i$의 external field | common | field term에 사용 |
| $\mathbf{A}_i$ | site $i$의 single-ion anisotropy matrix | common | onsite quadratic spin term에 사용 |

## Reserved Symbols

| Symbol | Reserved For | Scope | Notes |
|---|---|---|---|
| $\mu,\nu$ | magnetic sublattice index | LSWT / magnetic structure | common Hamiltonian에서 physical site index로 쓰지 않음 |
| $\mathbf{R}$ | crystallographic unit cell vector | lattice convention | site index와 구분해서 사용 |
| $a,b$ | crystallographic basis site | lattice convention | magnetic sublattice와 구분 |
| $\mathbf{k}$ | momentum vector | momentum-space documents | BZ convention 문서에서 세부 정의 |
| $\boldsymbol{\delta}$ | basis position 또는 displacement | context-dependent | 사용 문서에서 의미를 명시해야 함 |
