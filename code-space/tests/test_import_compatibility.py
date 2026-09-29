"""Keep notebook imports and serialized systems usable after relocation and rename."""

from importlib import import_module
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

import spintoolkit
from spintoolkit import SpinSystem

with pytest.warns(DeprecationWarning):
    lswt = import_module('lswt')


def test_previous_package_name_resolves_to_current_modules():
    for old, new in [
        ('system.spin_system', 'system.spin_system'),
        ('states.commensurate', 'states.commensurate'),
        ('definitions', 'definitions'),
        ('observables.topology', 'observables.topology'),
        ('methods.optimization', 'methods.optimization'),
        ('methods.spin_wave', 'methods.lswt'),
        ('methods.spin_wave.energy', 'methods.lswt.energy'),
        ('methods.spin_wave.hamiltonian', 'methods.lswt.hamiltonian'),
        ('methods.spin_wave.diagonalization', 'methods.lswt.diagonalization'),
        ('methods.spin_wave.solver', 'methods.lswt.solver'),
    ]:
        assert import_module('lswt.' + old) is import_module('spintoolkit.' + new)
    assert lswt.methods.spin_wave.energy.EnergyFunction is spintoolkit.EnergyFunction
    assert lswt.SpinSystem is SpinSystem
    assert not hasattr(spintoolkit.methods, 'spin_wave')


def test_historical_imports_resolve_to_current_modules():
    for old, new in [
        ('core.spin_system', 'system.spin_system'),
        ('core.exchange', 'system.exchange'),
        ('core.brillouin_zone', 'system.brillouin_zone'),
        ('core.diagonalization', 'methods.lswt.diagonalization'),
        ('core.lattice', 'system.lattice'),
        ('core.lattice.base', 'system.lattice.base'),
        ('core.lattice.presets', 'system.lattice.presets'),
        ('core.magnetic_structure', 'states'),
        ('core.magnetic_structure.base', 'states.base'),
        ('core.magnetic_structure.commensurate', 'states.commensurate'),
        ('core.magnetic_structure.incommensurate', 'states.incommensurate'),
        ('solvers.base', 'methods.base'),
        ('solvers.optimizer', 'methods.optimization'),
        ('solvers.energy', 'methods.lswt.energy'),
        ('solvers.hamiltonian', 'methods.lswt.hamiltonian'),
        ('solvers.solver', 'methods.lswt.solver'),
        ('config', 'definitions'),
    ]:
        assert import_module('lswt.' + old) is import_module('spintoolkit.' + new)
    core = import_module('lswt.core')
    solvers = import_module('lswt.solvers')
    assert core.SpinSystem is SpinSystem
    assert core.Diagonalizer is import_module('spintoolkit.methods.lswt').Diagonalizer
    assert solvers.LSWTSolver is spintoolkit.LSWTSolver


@pytest.mark.parametrize('old_path', [b'lswt.system.spin_system',
                                      b'lswt.core.spin_system'])
def test_pickle_with_previous_system_path_loads_current_classes(old_path):
    system = SpinSystem(lattice_vectors=np.eye(2))
    system.add_site('A', [0, 0], spin=0.5, angles=[0.3, 0.2],
                    magnetic_field=[0, 0, 0.1])
    system.add_coupling('A', 'A', np.eye(3), displacement=[1, 0])
    # Protocol 0 spells out import paths and allows an old-path fixture
    # without storing a Python-version-specific binary in the repository.
    payload = pickle.dumps(system, protocol=0)
    old_payload = payload.replace(b'spintoolkit.system.spin_system\n',
                                  old_path + b'\n')
    assert old_payload != payload
    restored = pickle.loads(old_payload)
    assert type(restored) is SpinSystem
    assert type(restored.site('A')) is SpinSystem.Site
    assert type(restored.couplings[0]) is SpinSystem.Coupling
    np.testing.assert_array_equal(restored.get_angles_flat(), system.get_angles_flat())
    np.testing.assert_array_equal(restored.couplings[0].exchange_matrix,
                                  system.couplings[0].exchange_matrix)


def test_previous_package_name_warns_on_first_import():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run(
        [sys.executable, '-W', "error:The 'lswt' package:DeprecationWarning",
         '-c', 'import lswt'],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode != 0
    assert "renamed to 'spintoolkit'" in result.stderr
