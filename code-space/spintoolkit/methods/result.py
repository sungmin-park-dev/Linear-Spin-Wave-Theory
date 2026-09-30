"""Common result header and JSON serialization (D14).

Every result of the new model types carries a :class:`ResultHeader` that
records where it came from (model and state fingerprints, calculation
geometry, external conditions, method settings, code version), the energy
unit label and normalization, and the diagnostics of the run. The method body
(LSWT, ED, ...) holds the numbers.

JSON output contains the header and the method's summary. Large arrays kept
for reuse within a calculation (Hamiltonians, eigenvectors) are written only
on request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional

import numpy as np

from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import SpinModel

#: Version of the result header layout.
RESULT_SCHEMA_VERSION = 1


def state_fingerprint(state: SpinState) -> str:
    """SHA-256 of a state's model reference, supercell and directions (rounded to 1e-12)."""
    items = sorted((site, tuple(cell), tuple(np.round(vector, 12).tolist()))
                   for (site, cell), vector in state.directions.items())
    payload = json.dumps({"model_ref": state.model_ref,
                          "supercell": state.supercell.tolist(), "directions": items},
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _code_version() -> Dict[str, Any]:
    from spintoolkit import __version__

    version: Dict[str, Any] = {"spintoolkit": __version__}
    try:
        here = Path(__file__).resolve().parent
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here.parent,
                               capture_output=True, text=True, timeout=5)
        if commit.returncode == 0:
            version["git_commit"] = commit.stdout.strip()
            version["package_tree_modified"] = bool(dirty.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return version


@dataclass(frozen=True)
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


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays, tuples and dataclass-like dicts to JSON types.

    Complex arrays become ``{"real": ..., "imag": ...}``.
    """
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if hasattr(value, "__dataclass_fields__"):
        return to_jsonable({k: getattr(value, k) for k in value.__dataclass_fields__})
    return value


def save_json(result, path, include_arrays: bool = False) -> Path:
    """Write ``result.to_json_dict(include_arrays)`` to ``path``."""
    path = Path(path)
    path.write_text(json.dumps(result.to_json_dict(include_arrays=include_arrays), indent=1))
    return path


def load_json(path) -> Dict[str, Any]:
    """Read a result written by :func:`save_json` as a plain dictionary."""
    return json.loads(Path(path).read_text())
