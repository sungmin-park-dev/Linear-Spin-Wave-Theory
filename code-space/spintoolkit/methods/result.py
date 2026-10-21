"""Common result header and JSON serialization (D14).

Every result of the new model types carries a :class:`ResultHeader` that
records where it came from (model and state fingerprints, calculation
geometry, external conditions, method settings, code version), the energy
unit label and normalization, and the diagnostics of the run. The method body
(LSWT, ED, ...) holds the numbers.

JSON output contains the header and the method's summary. Large arrays kept
for reuse within a calculation (Hamiltonians, eigenvectors) are written only
on request. The JSON is standard: NaN and infinities are written as the strings
``"NaN"``, ``"Infinity"`` and ``"-Infinity"`` and complex numbers as
``{"real": ..., "imag": ...}``; :func:`load_json` restores both.

Results are read-only (D66): their arrays are read-only views and their
dictionaries :class:`FrozenDict`, so the safety flags an observable checks
(for example ``extra["frame"]``) cannot be changed after the run. Use
:func:`dataclasses.replace` for a modified copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Dict, Mapping, Optional

import numpy as np

from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import SpinModel

#: Version of the result header layout (2: standard JSON for NaN and infinities).
RESULT_SCHEMA_VERSION = 2

_NON_FINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


class FrozenDict(dict):
    """A ``dict`` that refuses changes after construction (D66).

    It is still a ``dict`` (``isinstance``, ``json``, ``copy`` and ``pickle``
    work); every mutating method raises ``TypeError``.
    """

    def _refuse(self, *args, **kwargs):
        raise TypeError("result dictionaries are read-only; use dataclasses.replace "
                        "for a modified result")

    __setitem__ = __delitem__ = __ior__ = _refuse
    clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self):
        return (self.__class__, (dict(self),))


def frozen(value: Any) -> Any:
    """Read-only version of ``value``: arrays become read-only views, dicts :class:`FrozenDict`.

    Nested dicts are frozen recursively; other objects (lists included) are
    returned unchanged. The caller's arrays stay writable (a view is frozen,
    not the array).
    """
    if isinstance(value, np.ndarray):
        if not value.flags.writeable:
            return value
        view = value.view()
        view.setflags(write=False)
        return view
    if isinstance(value, Mapping):
        return FrozenDict({k: frozen(v) for k, v in value.items()})
    return value


def freeze_fields(result) -> None:
    """Freeze every array and dict field of a frozen dataclass in place (call in ``__post_init__``)."""
    for f in fields(result):
        value = getattr(result, f.name)
        if isinstance(value, (np.ndarray, Mapping)):
            object.__setattr__(result, f.name, frozen(value))


def state_fingerprint(state: SpinState) -> str:
    """SHA-256 of a state's model reference, supercell and directions (rounded to 1e-12)."""
    items = sorted((site, tuple(cell), tuple(np.round(vector, 12).tolist()))
                   for (site, cell), vector in state.directions.items())
    payload = json.dumps({"model_ref": state.model_ref,
                          "supercell": state.supercell.tolist(), "directions": items},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=1)
def _code_version_cached() -> Dict[str, Any]:
    import numpy
    import scipy

    from spintoolkit import __version__

    version: Dict[str, Any] = {"spintoolkit": __version__, "python": platform.python_version(),
                               "numpy": numpy.__version__, "scipy": scipy.__version__}
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                capture_output=True, text=True, timeout=5)
        if commit.returncode == 0:
            dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here.parent,
                                   capture_output=True, text=True, timeout=5)
            version["git_commit"] = commit.stdout.strip()
            version["package_tree_modified"] = bool(dirty.stdout.strip())
            return version
    except (OSError, subprocess.SubprocessError):
        pass
    commit = _installed_commit()
    if commit is not None:
        version["git_commit"] = commit
    return version


def _installed_commit() -> Optional[str]:
    """Commit of a pip install from git (PEP 610 ``direct_url.json``), if recorded."""
    try:
        from importlib import metadata

        text = metadata.distribution("spin-toolkit").read_text("direct_url.json")
        if text:
            commit = json.loads(text).get("vcs_info", {}).get("commit_id")
            return None if commit is None else str(commit)[:7]
    except Exception:          # no distribution metadata (source checkout): nothing to add
        pass
    return None


def _code_version() -> Dict[str, Any]:
    """Package, Python, numpy and scipy versions and the git commit, looked up once per process.

    From a git checkout the commit and whether the package tree has local
    changes (at the first call); from a pip install from git, the commit
    recorded by pip.
    """
    return dict(_code_version_cached())


@dataclass(frozen=True, eq=False)
class ResultHeader:
    """Provenance, units and diagnostics common to all methods.

    Attributes
    ----------
    schema_version : int
    method : str
        "lswt", "ed", ...
    model_ref : str
        :meth:`SpinModel.fingerprint`.
    state_ref : str or None
        :func:`state_fingerprint` of the reference state (None for ED).
    geometry : dict
        ``{"kind": ..., "cluster": ...}``.
    conditions : dict
        Dimensionless ``field`` and ``temperature`` (D20).
    settings : dict
        Method settings as used.
    code_version : dict
    energy_unit : str
        Label of E0 from the model metadata (all numbers are in E0).
    normalization : str
    diagnostics : dict
    """

    schema_version: int
    method: str
    model_ref: str
    state_ref: Optional[str]
    geometry: Dict[str, Any]
    conditions: Dict[str, Any]
    settings: Dict[str, Any]
    code_version: Dict[str, Any]
    energy_unit: str
    normalization: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        freeze_fields(self)

    @classmethod
    def build(cls, method: str, model: SpinModel, state: Optional[SpinState],
              geometry: Optional[CalculationGeometry], conditions: ExternalConditions,
              settings: Dict[str, Any], normalization: str,
              diagnostics: Optional[Dict[str, Any]] = None) -> "ResultHeader":
        geometry = geometry or CalculationGeometry.thermodynamic_limit()
        return cls(RESULT_SCHEMA_VERSION, method, model.fingerprint(),
                   None if state is None else state_fingerprint(state),
                   {"kind": geometry.kind,
                    "cluster": None if geometry.cluster is None else geometry.cluster.tolist()},
                   {"field": conditions.field.tolist(), "temperature": conditions.temperature},
                   settings, _code_version(), str(model.metadata.get("energy_unit", "E0")),
                   normalization, dict(diagnostics or {}))


def _json_float(value: float) -> Any:
    if math.isfinite(value):
        return value
    return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")


def _json_list(values: list) -> list:
    return [_json_list(v) if isinstance(v, list) else _json_float(v) if isinstance(v, float) else v
            for v in values]


def _real_array(value: np.ndarray) -> Any:
    if value.dtype.kind == "f" and not np.isfinite(value).all():
        out = value.tolist()
        return _json_float(out) if isinstance(out, float) else _json_list(out)
    return value.tolist()


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays, tuples, mappings and dataclasses to standard JSON types.

    Complex arrays become ``{"real": ..., "imag": ...}``; NaN and infinities
    become ``"NaN"``, ``"Infinity"`` and ``"-Infinity"``. Callables are dropped
    (written as None).
    """
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": _real_array(value.real), "imag": _real_array(value.imag)}
        return _real_array(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _json_float(float(value))
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": _json_float(value.real), "imag": _json_float(value.imag)}
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable({k: getattr(value, k) for k in value.__dataclass_fields__})
    if callable(value):
        return None
    return value


def save_json(result, path, include_arrays: bool = False) -> Path:
    """Write ``result.to_json_dict(include_arrays)`` to ``path`` as standard JSON."""
    path = Path(path)
    try:
        data = result.to_json_dict(include_arrays=include_arrays)
    except TypeError:          # results without large arrays take no argument
        data = result.to_json_dict()
    path.write_text(json.dumps(data, indent=1, allow_nan=False))
    return path


def _restore(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"real", "imag"}:
            real, imag = _restore(value["real"]), _restore(value["imag"])
            return np.asarray(real, dtype=float) + 1j * np.asarray(imag, dtype=float)
        return {k: _restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore(v) for v in value]
    if isinstance(value, str) and value in _NON_FINITE:
        return _NON_FINITE[value]
    return value


def load_json(path) -> Dict[str, Any]:
    """Read a result written by :func:`save_json` as a plain dictionary.

    ``"NaN"``, ``"Infinity"`` and ``"-Infinity"`` are restored to floats and
    ``{"real", "imag"}`` pairs to complex numpy arrays (or complex scalars as
    0-d arrays); other arrays stay nested lists.
    """
    return _restore(json.loads(Path(path).read_text()))
