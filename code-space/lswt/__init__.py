"""
LSWT: Linear Spin Wave Theory for 2D Quantum Magnets

A Python library for computing magnon spectra, topological properties,
and thermodynamics of arbitrary 2D spin models using linear spin wave theory.

Main Components
---------------
system : SpinSystem, exchange matrices, and lattice geometry
states : Classical magnetic structures
methods : Shared solver interface, optimization, and spin-wave calculations
definitions : Physical constants, numerical defaults, and spin basis
observables : Thermodynamics, topology (Berry/Chern), correlations
visualization : Band structure, Berry curvature, spin configuration plots

Quick Start
-----------
>>> import numpy as np
>>> from lswt import SpinSystem, LSWTSolver
>>> from lswt.system import exchange
>>>
>>> # Define sites, couplings, and lattice
>>> sites = [SpinSystem.Site("A", [0, 0], spin=0.5,
...          angles=[np.pi/2, 0], magnetic_field=[0, 0, 0])]
>>> J = exchange.heisenberg(1.0)
>>> couplings = [SpinSystem.Coupling(0, 0, J, [1.0, 0.0])]
>>> system = SpinSystem(sites, couplings, lattice_vectors=[[1, 0], [0.5, 0.866]])
"""

__version__ = "0.2.0-dev"
__author__ = "Sung-Min Park"
__email__ = "sungmin.park.0226@gmail.com"

# Physical systems
from lswt.system.spin_system import SpinSystem
# Backward-compatible aliases (to be removed in future versions)
from lswt.system.spin_system import SpinSite, Coupling
from lswt.system.exchange import heisenberg, xxz, xxz_with_soc, dzyaloshinskii_moriya, kitaev
from lswt.system.brillouin_zone import BrillouinZone

# Calculation methods
from lswt.methods.base import AbstractSolver, SolverResult
from lswt.methods.spin_wave.solver import LSWTSolver
from lswt.methods.optimization import SpinOptimizer
from lswt.methods.spin_wave.energy import EnergyFunction

__all__ = [
    '__version__', '__author__', '__email__',
    # Core
    'SpinSystem', 'SpinSite', 'Coupling',
    'heisenberg', 'xxz', 'xxz_with_soc', 'dzyaloshinskii_moriya', 'kitaev',
    'BrillouinZone',
    # Solvers
    'AbstractSolver', 'SolverResult',
    'LSWTSolver', 'SpinOptimizer', 'EnergyFunction',
]

from lswt._compat import install_legacy_imports as _install_legacy_imports

_install_legacy_imports()
del _install_legacy_imports
