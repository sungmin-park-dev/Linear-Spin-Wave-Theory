"""
LSWT: Linear Spin Wave Theory Package

A Python package for calculating spin wave excitations, thermodynamic properties,
and topological characteristics of magnetic systems using linear spin wave theory.

Main Components
---------------
core : Core domain models (Lattice, MagneticStructure, SpinSystem)
solvers : LSWT solver, optimization, diagonalization
physics : Thermodynamics, topology, correlations, observables
analysis : Phase finding, phase diagrams, quantum corrections
visualization : Plotting and visualization tools
io : Configuration loading, validation, export
utils : Constants, mathematical utilities, logging

Quick Start
-----------
>>> from lswt.core import TriangularLattice, CommensurateStructure, SpinSystem
>>> import numpy as np
>>>
>>> # Create triangular lattice
>>> lattice = TriangularLattice(lattice_constant=1.0)
>>>
>>> # Define 120° magnetic structure
>>> angles = np.array([[np.pi/2, 0], [np.pi/2, 2*np.pi/3], [np.pi/2, 4*np.pi/3]])
>>> mag_struct = CommensurateStructure(
...     num_basis_sites=1,
...     magnetic_supercell=(1, 1),
...     angles=angles
... )
>>>
>>> # Define interactions
>>> interactions = {
...     'nearest_neighbor': {'Jxy': 1.0, 'Jz': 1.0}
... }
>>>
>>> # Create spin system
>>> system = SpinSystem(lattice, mag_struct, interactions)
>>> print(system)

Current Version: 0.1.0-dev (Phase 1A)
"""

__version__ = "0.1.0-dev"
__author__ = "Sung-Min Park"
__email__ = "sungmin.park.0226@gmail.com"

# High-level API exports
from .core import (
    # Lattice
    AbstractLattice,
    TriangularLattice,
    create_lattice,

    # Magnetic Structure
    AbstractMagneticStructure,
    CommensurateStructure,

    # Spin System
    SpinSystem,
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',

    # Core abstractions
    'AbstractLattice',
    'TriangularLattice',
    'create_lattice',
    'AbstractMagneticStructure',
    'CommensurateStructure',
    'SpinSystem',
]
