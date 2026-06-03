# Introduction

> Source: `research-space/sources/lswt/note_lswt_restructured.tex` §Introduction
> Status: First draft

This document is the canonical Linear Spin Wave Theory (LSWT) reference for
this project. Its immediate purpose is not only to explain the current LSWT
implementation, but also to make LSWT a benchmark against which future
simulation methods for two-dimensional spin systems can be checked.

The present LSWT scope is a commensurate ordered spin system with a finite
magnetic unit cell. The document therefore starts from a lattice spin
Hamiltonian, fixes the local classical spin frame, expands spin operators in
bosons, constructs the quadratic bosonic Hamiltonian, diagonalizes it, and then
derives the physical quantities computed from the resulting magnon modes.

The main quantities covered by the LSWT workflow are:

- classical and LSWT-corrected ground-state energy;
- magnon band energies and eigenvectors;
- thermodynamic quantities such as free energy, entropy, specific heat, and
  boson occupation;
- spin correlations, structure factors, and spectral functions;
- topological quantities when the required band and Berry-curvature data are
  well defined.

Because this project is being generalized beyond LSWT, several conventions in
this document are deliberately treated as solver-facing definitions rather than
as implementation details. In particular, the LSWT write-up must clarify:

- how a physical site is indexed relative to a crystallographic unit cell,
  basis site, and magnetic sublattice;
- whether a link/coupling is counted once or represented by symmetric pairs;
- whether displacement coordinates are fractional cell displacements or real
  vectors;
- how commensurate magnetic structure, magnetic supercell, and spin angles
  determine magnetic sublattices;
- when the Brillouin zone is crystallographic and when it is folded by the
  magnetic unit cell;
- which definitions belong to the generic spin-system model and which are
  specific to the LSWT solver.

These convention questions are first resolved inside the LSWT document. Only
after they are stable should they be extracted into a shared `common/` theory
layer for Monte Carlo, tensor-network, neural quantum state, or other solvers.

## 작업 메모

- LaTeX source의 Introduction은 TODO 상태였으므로, 위 본문은 source TODO와
  현재 migration note의 프로젝트 방향을 반영한 첫 초안이다.
- 다음 이식 대상은 `01_from_spins_to_bosons.md`의 Hamiltonian 정의와
  link-counting convention이다.
- `common/` 후보는 아직 별도 파일로 분리하지 않는다.
