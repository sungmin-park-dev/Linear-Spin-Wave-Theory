"""Deprecated alias for :mod:`spintoolkit`, the package's previous name.

Keeps old imports, notebooks, monkeypatches and pickles working after the
2026-09-29 rename. Every ``lswt.<path>`` resolves to the same module object as
the matching ``spintoolkit.<path>``; ``methods.spin_wave`` is the old name of
``methods.lswt``. The pre-2026-09-23 ``core``, ``solvers`` and ``config`` paths
are kept as well. New code imports :mod:`spintoolkit`.
"""

from importlib import import_module
from importlib.machinery import ModuleSpec
import pkgutil
import sys
from types import ModuleType
import warnings

import spintoolkit
from spintoolkit import *  # noqa: F401,F403
from spintoolkit import __all__, __author__, __email__, __version__  # noqa: F401

warnings.warn(
    "The 'lswt' package was renamed to 'spintoolkit'; import spintoolkit instead.",
    DeprecationWarning, stacklevel=2,
)

# Current module paths whose segment was renamed on 2026-09-29.
_RENAMED = {'methods.lswt': 'methods.spin_wave'}

# Pre-2026-09-23 paths, mapped to current spintoolkit paths.
_HISTORICAL = {
    'core.spin_system': 'system.spin_system',
    'core.exchange': 'system.exchange',
    'core.brillouin_zone': 'system.brillouin_zone',
    'core.diagonalization': 'methods.lswt.diagonalization',
    'core.lattice': 'system.lattice',
    'core.lattice.base': 'system.lattice.base',
    'core.lattice.presets': 'system.lattice.presets',
    'core.magnetic_structure': 'states',
    'core.magnetic_structure.base': 'states.base',
    'core.magnetic_structure.incommensurate': 'states.incommensurate',
    'solvers.base': 'methods.base',
    'solvers.optimizer': 'methods.optimization',
    'solvers.energy': 'methods.lswt.energy',
    'solvers.hamiltonian': 'methods.lswt.hamiltonian',
    'solvers.solver': 'methods.lswt.solver',
    'config': 'definitions',
}

_NAMESPACES = {
    'core': ['SpinSystem', 'SpinSite', 'Coupling', 'BrillouinZone'],
    'solvers': ['AbstractSolver', 'SolverResult', 'LSWTSolver',
                'SpinOptimizer', 'EnergyFunction'],
}


class _Proxy(ModuleType):
    """Old-name package whose children differ from the current package."""

    def __init__(self, name, target):
        super().__init__(name, target.__doc__)
        self.__path__ = []
        self.__package__ = name
        self.__spec__ = ModuleSpec(name, loader=None, is_package=True)
        self._target = target

    def __getattr__(self, attr):
        return getattr(self._target, attr)


def _legacy_name(current):
    for new, old in _RENAMED.items():
        if current == new or current.startswith(new + '.'):
            return old + current[len(new):]
    return current


def _register(name, module):
    """Expose ``module`` as ``lswt.<name>`` in sys.modules and on its parent."""
    full = __name__ + '.' + name
    sys.modules[full] = module
    parent, _, child = full.rpartition('.')
    setattr(sys.modules[parent], child, module)


def _install():
    root = sys.modules[__name__]
    for old in _RENAMED.values():
        parent = old.rpartition('.')[0]
        if parent and __name__ + '.' + parent not in sys.modules:
            _register(parent, _Proxy(__name__ + '.' + parent,
                                     import_module('spintoolkit.' + parent)))
    for info in pkgutil.walk_packages(spintoolkit.__path__, 'spintoolkit.'):
        current = info.name[len('spintoolkit.'):]
        if __name__ + '.' + _legacy_name(current) in sys.modules:
            continue
        _register(_legacy_name(current), import_module(info.name))

    for name, exports in _NAMESPACES.items():
        module = ModuleType(__name__ + '.' + name, 'Compatibility import namespace.')
        module.__path__ = []
        module.__package__ = module.__name__
        module.__spec__ = ModuleSpec(module.__name__, loader=None, is_package=True)
        module.__all__ = list(exports)
        for export in exports:
            setattr(module, export, getattr(root, export))
        _register(name, module)
    for old, new in _HISTORICAL.items():
        _register(old, import_module('spintoolkit.' + new))
    root.core.Diagonalizer = import_module('spintoolkit.methods.lswt').Diagonalizer
    root.core.__all__ += ['exchange', 'Diagonalizer']


_install()
del _install
