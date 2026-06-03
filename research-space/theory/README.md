# Linear Spin Wave Theory Documentation

This directory contains refined theoretical documentation for Linear Spin Wave Theory (LSWT), converted from the original LaTeX document.

The active LSWT canonical workspace is now [`lswt/`](lswt/). The older
`sections/` files remain as converted Markdown source material until their
contents are migrated into the canonical LSWT structure.

## 📖 Table of Contents

### Core Documentation

1. **[Notation and Symbols](notation.md)**
   Summary of mathematical notation, custom LaTeX commands, and symbol definitions used throughout the documentation.

### Theory Sections

2. **[I. Introduction to Spin Wave Theory](sections/01_spin_wave_theory_intro.md)**
   Foundation of spin wave theory including:
   - Spin Hamiltonian formulation
   - Holstein-Primakoff transformation
   - Bogoliubov transformation
   - Momentum space formulation

3. **[II. Physical Quantities](sections/02_physical_quantities.md)**
   Overview of computable physical quantities in LSWT framework:
   - Thermodynamic quantities (partition function, free energy, entropy, specific heat)
   - Correlation functions (dynamic structure factor, spectral function)
   - Topological properties (Chern number, thermal Hall conductance)

4. **[III. Thermodynamics](sections/03_thermodynamics.md)**
   Detailed derivations of thermodynamic quantities:
   - Partition function
   - Internal energy
   - Free energy
   - Entropy
   - Specific heat
   - Boson number expectation values

5. **[IV. Correlations](sections/04_correlations.md)**
   Spin-spin correlation functions:
   - Real-time dynamical correlations
   - Dynamic structure factor
   - Spectral function
   - Equal-time correlations

6. **[V. Topology](sections/05_topology.md)**
   Topological aspects of magnon systems:
   - Skyrmion number
   - Chern number and Berry curvature
   - Thermal Hall conductance

7. **[VI. Worked Example](sections/06_worked_example.md)**
   Step-by-step example of solving a quadratic boson Hamiltonian using LSWT methods.

## 📁 Directory Structure

```
research-space/theory/
├── README.md                    # This file
├── notation.md                  # Notation reference
├── lswt/                        # Active LSWT canonical workspace
└── sections/                    # Converted LSWT sections pending migration
│   ├── 01_spin_wave_theory_intro.md
│   ├── 02_physical_quantities.md
│   ├── 03_thermodynamics.md
│   ├── 04_correlations.md
│   ├── 05_topology.md
│   └── 06_worked_example.md
```

## 🔄 Conversion Process

These documents were created through the following workflow:

1. **Original LaTeX**: `../sources/lswt/note_lswt_restructured.tex`
2. **Pandoc Conversion**: LaTeX → Markdown (with equation preservation)
3. **Refinement**: Python script processing for:
   - Simplifying equation references
   - Converting custom LaTeX commands (e.g., `\kvec` → `\mathbf{k}`)
   - Cleaning up formatting artifacts
   - Organizing into logical sections

## 🔗 Related Files

- **Original LaTeX**: [`../sources/lswt/note_lswt_restructured.tex`](../sources/lswt/note_lswt_restructured.tex)

## 📚 Usage for Developers

When working with the LSWT package:

1. **Reference Implementation**: These documents explain the theoretical foundation for the code in `code-space/lswt/solvers/` and `code-space/lswt/observables/`
2. **Physical Quantities**: Section II provides a quick reference for what can be computed
3. **Notation**: Check `notation.md` when encountering unfamiliar symbols
4. **Worked Example**: Section VI demonstrates the complete workflow

## 🔍 Key Concepts

### Spin Wave Theory Basics
- Represents quantum spin fluctuations as bosonic excitations (magnons)
- Valid in the low-temperature limit where quantum fluctuations are small
- Provides analytical and numerical access to thermodynamic and dynamical properties

### Main Theoretical Steps
1. **Classical ground state**: Find spin configuration minimizing classical energy
2. **Local frame transformation**: Align local $z$-axis with classical spin direction
3. **Holstein-Primakoff**: Map spin operators to bosonic ladder operators
4. **Fourier transform**: Convert to momentum space
5. **Bogoliubov transformation**: Diagonalize quadratic Hamiltonian
6. **Physical quantities**: Calculate observables from magnon spectrum

### Applicability
- Frustrated magnets (triangular, kagome, honeycomb lattices)
- Magnetic skyrmion systems
- Topological magnon bands
- Quantum spin liquids (at mean-field level)

## 📝 Citation

If you use these theoretical notes or the LSWT package, please cite:

```bibtex
@misc{lswt_theory,
  author = {Park, Sung-Min},
  title = {Linear Spin Wave Theory: Theoretical Documentation},
  year = {2025},
  howpublished = {\url{https://github.com/...}},
}
```

*(Update with actual publication details when available)*

## 🤝 Contributing

Found an error or typo? Please:
1. Check the original LaTeX source: `research-space/sources/lswt/note_lswt_restructured.tex`
2. Open an issue describing the problem
3. If you fix it, submit a PR with changes to both LaTeX and Markdown

## 📧 Contact

**Author**: Sung-Min Park
**Email**: sungmin.park.0226@gmail.com
**Status**: On military leave (Oct 2024 - Apr 2025)

---

**Last Updated**: 2025-11-28
**Document Version**: 1.0
