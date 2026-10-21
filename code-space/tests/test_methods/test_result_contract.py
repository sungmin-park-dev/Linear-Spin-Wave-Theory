"""Result contract (D66): read-only results, standard JSON with a round trip, provenance.

1. Arrays of a result are read-only and its dictionaries refuse changes, so the
   safety flags read by observables (``extra["frame"]``, ``extra["augmentation"]``,
   ``header.diagnostics``) cannot be switched off after the run; the caller's
   own arrays stay writable; ``dataclasses.replace`` gives a modified copy.
2. ``==`` between results compares identity instead of raising on arrays.
3. JSON is standard (no bare NaN/Infinity tokens) and ``load_json`` restores
   NaN, the signed infinities of D64 and complex arrays.
4. Every method result with numbers has ``to_json_dict``; LSWT writes the
   thermal block, lattices, positions, local frames and flags.
5. The header records python, numpy and scipy versions; git is asked once.
6. Monte Carlo averages record the seed.
"""

import copy
from dataclasses import replace
import json
import pickle

import numpy as np
import pytest

from spintoolkit.methods.classical import classical_search
from spintoolkit.methods.dynamics import ClassicalTorus
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.methods.monte_carlo import MonteCarlo, thermal_averages
from spintoolkit.methods.nlswt import NLSWTSettings, solve_nlswt
from spintoolkit.methods import result as result_module
from spintoolkit.methods.result import FrozenDict, frozen, load_json, save_json, to_jsonable
from spintoolkit.models import neel_state, polarized_state, square_heisenberg
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry


@pytest.fixture(scope="module")
def neel():
    model = square_heisenberg(J=1.0)
    return solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(4, 4)))


def strict_loads(text):
    def refuse(token):
        raise ValueError(f"non-standard JSON token {token}")
    return json.loads(text, parse_constant=refuse)


def test_result_arrays_and_flags_are_read_only(neel):
    with pytest.raises(ValueError):
        neel.eigenvalues[:] = 0
    with pytest.raises(ValueError):
        neel.boson_numbers[0] = 0
    with pytest.raises(TypeError):
        neel.extra["frame"] = "lab"
    with pytest.raises(TypeError):
        neel.header.diagnostics["stationary"] = False
    with pytest.raises(TypeError):
        neel.header.settings.update(regularization="none")
    assert isinstance(neel.extra, dict) and isinstance(neel.header.diagnostics, FrozenDict)


def test_freezing_keeps_the_callers_arrays_writable_and_copies_work(neel):
    mine = np.zeros(3)
    view = frozen({"a": mine, "b": {"c": mine}})
    mine[0] = 1.0
    assert view["a"][0] == 1.0 and not view["b"]["c"].flags.writeable
    flags = FrozenDict({"x": 1})
    assert copy.deepcopy(flags) == flags and pickle.loads(pickle.dumps(flags)) == flags
    changed = replace(neel, extra={"frame": "rotating"})
    assert changed.extra["frame"] == "rotating" and "frame" not in neel.extra
    with pytest.raises(TypeError):
        changed.extra["frame"] = "lab"


def test_results_compare_by_identity(neel):
    assert neel == neel
    assert neel != replace(neel)


def test_json_is_standard_and_round_trips_nan_inf_and_complex(tmp_path, neel):
    values = {"nan": float("nan"), "inf": np.inf, "minus": -np.inf,
              "array": np.array([1.0, np.nan, -np.inf]), "z": np.array([1 + 2j, np.nan])}
    text = json.dumps(to_jsonable(values), allow_nan=False)
    back = result_module._restore(strict_loads(text))
    assert np.isnan(back["nan"]) and back["inf"] == np.inf and back["minus"] == -np.inf
    np.testing.assert_array_equal(back["array"], values["array"])
    np.testing.assert_array_equal(back["z"], values["z"])
    path = save_json(neel, tmp_path / "neel.json", include_arrays=True)
    strict_loads(path.read_text())
    data = load_json(path)
    np.testing.assert_allclose(data["lswt"]["eigenvectors"], neel.eigenvectors)
    np.testing.assert_allclose(data["lswt"]["local_frames"], neel.local_frames)
    np.testing.assert_allclose(data["lswt"]["magnetic_lattice"], neel.magnetic_lattice)
    np.testing.assert_allclose(data["lswt"]["positions"], neel.positions)
    assert data["lswt"]["thermal"] is None and data["lswt"]["extra"] == {}


def test_thermal_infinities_survive_the_round_trip(tmp_path):
    model = square_heisenberg(J=-1.0)
    result = solve_lswt(model, polarized_state(model, (0, 0, 1)),
                        ExternalConditions(temperature=0.5),
                        settings=LSWTSettings(mesh=(6, 6), gapless=True))
    assert np.isposinf(result.thermal.boson_numbers).all()
    with pytest.raises(ValueError):
        result.thermal.boson_numbers[:] = 0
    data = load_json(save_json(result, tmp_path / "fm.json"))
    thermal = data["lswt"]["thermal"]
    assert np.isposinf(np.array(thermal["boson_numbers"])).all()
    assert np.isnan(np.array(thermal["moments"])).all()
    assert thermal["gapless"] is True


def test_every_method_result_writes_json(tmp_path):
    model = square_heisenberg(J=1.0)
    nlswt = solve_nlswt(model, neel_state(model), settings=NLSWTSettings(mesh=(6, 6)))
    data = load_json(save_json(nlswt, tmp_path / "nlswt.json"))
    assert data["nlswt"]["total"] == pytest.approx(nlswt.ground_state_energy)
    assert data["header"]["method"] == nlswt.header.method
    search = classical_search(model, [[1, 1], [1, -1]])
    data = load_json(save_json(search, tmp_path / "search.json"))
    assert data["energy"] == pytest.approx(search.energy)
    assert data["settings"]["seed"] == search.settings["seed"]
    assert len(data["state"]["directions"]) == 2 and data["code_version"]["numpy"] == np.__version__


def test_code_version_records_libraries_and_asks_git_once(monkeypatch, neel):
    import platform
    import scipy

    version = neel.header.code_version
    assert version["numpy"] == np.__version__ and version["scipy"] == scipy.__version__
    assert version["python"] == platform.python_version()
    result_module._code_version_cached()               # cached by the fixture already
    def no_git(*args, **kwargs):
        raise AssertionError("git called again")
    monkeypatch.setattr(result_module.subprocess, "run", no_git)
    assert result_module._code_version() == dict(version)


def test_monte_carlo_records_the_seed():
    model = square_heisenberg(J=1.0)
    torus = ClassicalTorus(model, CalculationGeometry.finite_torus([[4, 0], [0, 4]]),
                           ExternalConditions())
    averages, _ = thermal_averages(MonteCarlo(torus, 0.5, seed=7), torus.random_spins(1), 5, 10,
                                   bins=2)
    assert averages.settings["seed"] == 7 and averages.settings["measurements"] == 10
    generator, _ = thermal_averages(MonteCarlo(torus, 0.5, seed=np.random.default_rng(1)),
                                    torus.random_spins(1), 1, 2, bins=2)
    assert generator.settings["seed"] == "generator"
    with pytest.raises(ValueError):
        averages.series["energy"][0] = 0.0
