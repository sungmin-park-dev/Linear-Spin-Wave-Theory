"""
spintoolkit: calculation toolkit for 2D spin systems

Defines 2D spin models and connects them to classical optimization, linear
spin-wave theory and observables. Further methods (ED, tensor networks) are
added as subpackages of ``methods``. The conventional alias is
``import spintoolkit as stk``.

Main Components
---------------
system : ``SpinModel`` (Hamiltonian), external conditions, lattice geometry and symmetry
states : ``SpinState`` (commensurate) and ``IncommensurateStructure`` (single-Q spirals)
methods : classical search, Luttinger-Tisza, Monte Carlo, Landau-Lifshitz dynamics,
    exact diagonalization, linear spin-wave theory (``solve_lswt``) and
    comparison of candidate states
observables : bands, thermodynamics, structure factor and neutron intensity,
    Berry curvature, Chern numbers, thermal Hall, skyrmion number
visualization : band structure and spin configuration plots
models : benchmark Hamiltonians and reference states

All energies are dimensionless, in the unit E0 of the coupling coefficients;
fields are ``mu_B B / E0`` and temperatures ``k_B T / E0``.

Deprecated API
--------------
``SpinSystem`` (with ``SpinSite``, ``Coupling``), ``LSWTSolver``,
``SpinOptimizer``, ``EnergyFunction``, ``Topology.compute_thermal_Hall`` and
the old package name ``lswt`` work in 0.2 with a DeprecationWarning and are
removed in 0.3 (D43).

Quick Start
-----------
>>> import spintoolkit as stk
>>> from spintoolkit.models import triangular_heisenberg, state_120
>>> model = triangular_heisenberg(J=1.0, S=0.5)
>>> result = stk.solve_lswt(model, state_120(model),
...                         settings=stk.LSWTSettings(mesh=(24, 24)))
>>> round(result.ground_state_energy, 4)   # E_cl + E_zp per spin, units of J
-0.5388
"""

__version__ = "0.2.0.dev0"   # PEP 440; the single source of the package version
__author__ = "Sung-Min Park"
__email__ = "sungmin.park.0226@gmail.com"

# Physical systems
from spintoolkit.system.spin_system import SpinSystem
# Backward-compatible aliases, removed in 0.3 with SpinSystem (D43)
from spintoolkit.system.spin_system import SpinSite, Coupling
from spintoolkit.system.exchange import heisenberg, xxz, xxz_with_soc, dzyaloshinskii_moriya, kitaev
from spintoolkit.system.brillouin_zone import BrillouinZone

# Common model definition (transfer contract)
from spintoolkit.system.model import (
    SpinModel, Site, Term, SpinModelError, validate_spin_model,
)
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.symmetry import LayerCrystal, CrystalSymmetry, find_symmetry
from spintoolkit.states.spin_state import SpinState, SpinStateError, validate_spin_state
from spintoolkit.states.incommensurate import IncommensurateStructure

# Calculation methods
from spintoolkit.methods.base import AbstractSolver, SolverResult
from spintoolkit.methods.lswt import LSWTResult, LSWTSettings, solve_lswt
from spintoolkit.methods.nlswt import NLSWTResult, NLSWTSettings, solve_nlswt
from spintoolkit.methods.lswt.solver import LSWTSolver
from spintoolkit.methods.optimization import SpinOptimizer
from spintoolkit.methods.lswt.energy import EnergyFunction
from spintoolkit import models

__all__ = [
    '__version__', '__author__', '__email__',
    # Exchange matrices and zone geometry
    'heisenberg', 'xxz', 'xxz_with_soc', 'dzyaloshinskii_moriya', 'kitaev',
    'BrillouinZone',
    # Deprecated, removed in 0.3 (D43)
    'SpinSystem', 'SpinSite', 'Coupling',
    # Common model definition
    'SpinModel', 'Site', 'Term', 'SpinModelError', 'validate_spin_model',
    'ExternalConditions', 'CalculationGeometry',
    'LayerCrystal', 'CrystalSymmetry', 'find_symmetry',
    'SpinState', 'SpinStateError', 'validate_spin_state', 'IncommensurateStructure',
    'models',
    # Solvers
    'AbstractSolver', 'SolverResult', 'solve_lswt', 'LSWTSettings', 'LSWTResult',
    'solve_nlswt', 'NLSWTSettings', 'NLSWTResult',
    # Deprecated, removed in 0.3 (D43)
    'LSWTSolver', 'SpinOptimizer', 'EnergyFunction',
]
