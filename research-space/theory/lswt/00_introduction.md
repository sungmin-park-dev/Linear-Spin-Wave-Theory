---
frontmatter-version: 1
title: Introduction
section: theory/lswt
status: draft
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-04
source: research-space/sources/lswt/note_lswt_restructured.tex
source-section: Introduction
---

# Introduction

Linear Spin Wave Theory (LSWT) is a method for describing quantum fluctuations
around a classically ordered magnetic state. The central idea is to choose a
local spin frame aligned with the classical order, express spin fluctuations in
terms of Holstein-Primakoff bosons, and keep the quadratic part of the resulting
bosonic Hamiltonian. The magnon modes are then obtained by diagonalizing this
quadratic bosonic problem.

## Scope

### Spin Model and Ordered Magnetic State

LSWT starts from a spin Hamiltonian together with a classical ordered magnetic
state. The spin operators are expanded around this ordered state after choosing
local spin frames aligned with the classical spin directions.

The detailed Hamiltonian used for the derivation is introduced in
`01_from_spins_to_bosons.md`.

### Remark

The current codebase starts from commensurate magnetic structures with finite
magnetic unit cells. This is an implementation starting point rather than a
fundamental restriction of LSWT.

## Overview of the LSWT Framework

The LSWT construction in this document follows the structure of the source
LaTeX note:

1. Start from a spin Hamiltonian written in the laboratory frame.
2. Choose a classical magnetic structure and rotate each spin into its local
   frame.
3. Apply the Holstein-Primakoff transformation in the local frame.
4. Keep the quadratic bosonic Hamiltonian.
5. Transform the quadratic Hamiltonian to momentum space when the magnetic
   structure is commensurate.
6. Diagonalize the bosonic Bogoliubov-de Gennes Hamiltonian.
7. Use the resulting magnon modes to compute physical observables.

The following sections develop these steps in order, beginning with the mapping
from spin operators to bosonic degrees of freedom.

## Reference

- Physical quantities are organized in `03_physical_quantities.md` and the
  following observable-specific sections.
