---
frontmatter-version: 1
title: General SpinModel IR
section: idea-proposals
status: draft
author: codex
last-edited-by: codex
created: 2026-06-04
confidence: medium
source_refs:
  - conversation: 2026-06-04 common SpinModel representation discussion for SWT, MC, TN, and NQS
---

# General SpinModel IR

## Summary

SWT, Monte Carlo, tensor-network, and neural-quantum-state solvers should be able to consume the same model Hamiltonian without each solver redefining the physical model. This note proposes a solver-agnostic SpinModel intermediate representation (IR). The central point is that the lattice is not the most fundamental object. The more fundamental layer is local Hilbert space, physical site set, and finite-support Hamiltonian terms. Lattice data then becomes a geometry provider that generates, labels, and constrains those terms.

## Proposal

### Core Principle

A general SpinModel should be defined first as

```text
LocalSpace + SiteSet + HamiltonianTerms
```

and only then supplemented by lattice, boundary, symmetry, and solver-specific views.

### Proposed Layers

```text
SpinModel
├── LocalSpace
├── SiteSet
├── Geometry
├── Terms
├── Parameters
└── SolverViews
```

`LocalSpace` records the local degree of freedom, such as spin \(S\), local dimension \(2S+1\), and the operator basis.

`SiteSet` records physical site identifiers, optional labels, and optional cell/basis metadata.

`Geometry` records lattice vectors, site positions, boundary conditions, periodic images, and neighbor generation rules. It should not be required for every finite model.

`Terms` records the Hamiltonian as local operator terms with finite support.

`Parameters` records named couplings, fields, units, and tunable model parameters.

`SolverViews` records derived forms needed by specific algorithms, such as a classical-spin energy evaluator, a local-operator term list, or a momentum-space BdG representation.

### Term IR

Each Hamiltonian contribution should be represented as a term with explicit support and operator content.

```text
Term
├── support: site ids
├── operators: local spin operators or a named operator expression
├── coupling: scalar, vector, matrix, tensor, callable, or parameter reference
├── geometry: optional displacement, bond type, periodic image, or orbit id
└── tags: optional semantic labels
```

Typical tags include `NN`, `NNN`, `exchange`, `anisotropic`, `zeeman`, `single-ion`, `ring-exchange`, and `constraint`.

For the bilinear spin Hamiltonian, the natural two-site term is

```text
support = [i, j]
operators = [S_i^a, S_j^b]
coupling = J_ij^{ab}
```

This keeps the Hamiltonian definition independent of whether a later solver treats the spin classically, expands it in Holstein-Primakoff bosons, maps it to an MPO, or evaluates local energies in an NQS sampler.

### Solver Views

`SWTView` should derive bilinear spin terms, magnetic structure, local frames, Fourier conventions, and the bosonic BdG matrix. This view may require lattice periodicity and a magnetic unit cell.

`MCView` should derive classical spin variables, local energy functions, neighbor lists, and update neighborhoods.

`TNView` should derive local Hilbert spaces, site ordering or graph structure, local operator terms, and MPO/PEPS conversion inputs.

`NQSView` should derive graph metadata, local basis information, local-energy evaluation, symmetry labels, and ansatz-side constraints.

These views should be generated from the common SpinModel IR. They should not become separate sources of truth for the same Hamiltonian.

### Minimal First Scope

The first practical scope should be a bilinear spin model with optional one-site terms.

```text
H = sum_{(i,j)} S_i^T J_{ij} S_j + sum_i S_i^T A_i S_i - sum_i h_i^T S_i
```

This covers the current LSWT source document while still leaving room for finite clusters, periodic lattices, and non-LSWT solvers.

## Rationale

Starting from the lattice is useful for periodic magnetic materials, but it is too narrow as the fundamental representation. Tensor-network and NQS methods often need local Hilbert spaces, site ordering, graphs, and local operator terms. Monte Carlo needs fast local-energy updates. SWT needs magnetic structures, local frames, and momentum-space conventions. A term-based IR can serve all of these views while preserving a single Hamiltonian definition.

This direction also matches the existing project principle that `SpinSystem` should remain solver-agnostic. Solver-specific objects should derive views from the model rather than embedding LSWT-only assumptions into the model definition.

## Prior-Art Review Targets

The next review pass should compare this proposal against existing model abstractions in spin and many-body libraries. Useful review targets include NetKet, TeNPy, QuSpin, and ITensor-style operator-network constructions. The goal is not to copy one library, but to identify the common separation between local space, graph or lattice, operator terms, and solver conversion.

## Open Questions

- Should the first implementation support spin-only `LocalSpace`, or mixed site types from the start?
- How should physical basis sites, magnetic sublattices, and solver-internal site orderings be separated?
- Should multi-site terms beyond one-site and two-site terms be first-class immediately?
- How should parameter names, symbolic couplings, and units be represented?
- What is the minimal finite-cluster representation needed before periodic lattice support?
- Which parts belong in `SpinSystem`, and which should remain separate as derived solver views?
