"""
spintoolkit: calculation toolkit for 2D spin systems

Defines 2D spin models and connects them to classical optimization, linear
spin-wave theory and observables. Further methods (ED, tensor networks) are
added as subpackages of ``methods``. The conventional alias is
``import spintoolkit as stk``.

Main Components
---------------
system : SpinSystem, exchange matrices, and lattice geometry
states : Classical magnetic structures
methods : Shared solver interface, optimization, and calculation methods
    (``methods.lswt`` for linear spin-wave theory)
definitions : Physical constants, numerical defaults, and spin basis
observables : Thermodynamics, topology (Berry/Chern), correlations
visualization : Band structure, Berry curvature, spin configuration plots

The previous package name ``lswt`` remains importable as a deprecated alias.

Quick Start
-----------
>>> import numpy as np
>>> from spintoolkit import SpinSystem, LSWTSolver
>>> from spintoolkit.system import exchange
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
from spintoolkit.system.spin_system import SpinSystem
# Backward-compatible aliases (to be removed in future versions)
from spintoolkit.system.spin_system import SpinSite, Coupling
from spintoolkit.system.exchange import heisenberg, xxz, xxz_with_soc, dzyaloshinskii_moriya, kitaev
from spintoolkit.system.brillouin_zone import BrillouinZone

# Calculation methods
from spintoolkit.methods.base import AbstractSolver, SolverResult
from spintoolkit.methods.lswt.solver import LSWTSolver
from spintoolkit.methods.optimization import SpinOptimizer
from spintoolkit.methods.lswt.energy import EnergyFunction

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
