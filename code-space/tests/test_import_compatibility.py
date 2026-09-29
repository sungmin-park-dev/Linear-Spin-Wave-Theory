"""Keep notebook imports and serialized systems usable after relocation."""

from importlib import import_module
import pickle

import numpy as np

from lswt import SpinSystem


def test_historical_imports_resolve_to_current_modules():
    for old, new in [
        ('core.spin_system', 'system.spin_system'),
        ('core.exchange', 'system.exchange'),
        ('core.brillouin_zone', 'system.brillouin_zone'),
        ('core.diagonalization', 'methods.spin_wave.diagonalization'),
        ('core.lattice', 'system.lattice'),
        ('core.lattice.base', 'system.lattice.base'),
        ('core.lattice.presets', 'system.lattice.presets'),
        ('core.magnetic_structure', 'states'),
        ('core.magnetic_structure.base', 'states.base'),
        ('core.magnetic_structure.commensurate', 'states.commensurate'),
        ('core.magnetic_structure.incommensurate', 'states.incommensurate'),
        ('solvers.base', 'methods.base'),
        ('solvers.optimizer', 'methods.optimization'),
        ('solvers.energy', 'methods.spin_wave.energy'),
        ('solvers.hamiltonian', 'methods.spin_wave.hamiltonian'),
        ('solvers.solver', 'methods.spin_wave.solver'),
        ('config', 'definitions'),
    ]:
        assert import_module('lswt.' + old) is import_module('lswt.' + new)
    core = import_module('lswt.core')
    solvers = import_module('lswt.solvers')
    assert core.SpinSystem is SpinSystem
    assert core.Diagonalizer is import_module('lswt.methods.spin_wave').Diagonalizer
    assert solvers.LSWTSolver is import_module('lswt').LSWTSolver


def test_pickle_with_historical_system_path_loads_current_classes():
    system = SpinSystem(lattice_vectors=np.eye(2))
    system.add_site('A', [0, 0], spin=0.5, angles=[0.3, 0.2],
                    magnetic_field=[0, 0, 0.1])
    system.add_coupling('A', 'A', np.eye(3), displacement=[1, 0])
    # Protocol 0 spells out import paths and allows an old-path fixture
    # without storing a Python-version-specific binary in the repository.
    payload = pickle.dumps(system, protocol=0)
    old_payload = payload.replace(b'lswt.system.spin_system\n',
                                  b'lswt.core.spin_system\n')
    assert old_payload != payload
    restored = pickle.loads(old_payload)
    assert type(restored) is SpinSystem
    assert type(restored.site('A')) is SpinSystem.Site
    assert type(restored.couplings[0]) is SpinSystem.Coupling
    np.testing.assert_array_equal(restored.get_angles_flat(), system.get_angles_flat())
    np.testing.assert_array_equal(restored.couplings[0].exchange_matrix,
                                  system.couplings[0].exchange_matrix)
