"""Resolve historical import paths to the same current module objects.

The old core/solvers/config names own no implementation. Keep the bridge here
so existing notebooks, imports, monkeypatches, and pickles use current classes
without retaining a duplicate package tree. New code uses functional paths.
"""

from importlib import import_module
from importlib.machinery import ModuleSpec
import sys
from types import ModuleType


def install_legacy_imports():
    """Register the pre-reorganization paths after public imports are ready."""
    root = sys.modules['lswt']
    for name, exports in {
        'core': ['SpinSystem', 'SpinSite', 'Coupling', 'BrillouinZone'],
        'solvers': ['AbstractSolver', 'SolverResult', 'LSWTSolver',
                    'SpinOptimizer', 'EnergyFunction'],
    }.items():
        module = ModuleType('lswt.' + name, 'Compatibility import namespace.')
        module.__path__ = []
        module.__package__ = module.__name__
        module.__spec__ = ModuleSpec(module.__name__, loader=None, is_package=True)
        module.__all__ = list(exports)
        for export in exports:
            setattr(module, export, getattr(root, export))
        sys.modules[module.__name__] = module
        setattr(root, name, module)

    aliases = {
        'core.spin_system': 'system.spin_system',
        'core.exchange': 'system.exchange',
        'core.brillouin_zone': 'system.brillouin_zone',
        'core.diagonalization': 'methods.spin_wave.diagonalization',
        'core.lattice': 'system.lattice',
        'core.lattice.base': 'system.lattice.base',
        'core.lattice.presets': 'system.lattice.presets',
        'core.magnetic_structure': 'states',
        'core.magnetic_structure.base': 'states.base',
        'core.magnetic_structure.commensurate': 'states.commensurate',
        'core.magnetic_structure.incommensurate': 'states.incommensurate',
        'solvers.base': 'methods.base',
        'solvers.optimizer': 'methods.optimization',
        'solvers.energy': 'methods.spin_wave.energy',
        'solvers.hamiltonian': 'methods.spin_wave.hamiltonian',
        'solvers.solver': 'methods.spin_wave.solver',
        'config': 'definitions',
    }
    for old, new in aliases.items():
        module = import_module('lswt.' + new)
        sys.modules['lswt.' + old] = module
        parent, child = ('lswt.' + old).rsplit('.', 1)
        setattr(sys.modules[parent], child, module)

    root.core.Diagonalizer = import_module('lswt.methods.spin_wave').Diagonalizer
    root.core.__all__ += ['exchange', 'Diagonalizer']
