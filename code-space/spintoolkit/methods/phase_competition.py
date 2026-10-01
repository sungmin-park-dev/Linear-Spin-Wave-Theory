"""Energy comparison of candidate magnetic states at harmonic order (D42).

For each candidate on its own magnetic supercell, :func:`compare_states`
reports, with the same model, field and per-spin normalization:

- the classical energy per spin after optional refinement to a stationary
  point, and the largest remaining torque;
- LSWT about that state on a shifted mesh of the magnetic zone whose density
  is matched across cells (``k_density`` primitive momenta per direction,
  ``ceil(k_density / sqrt(|det M|))`` magnetic ones). A state whose ``H(k)``
  is not positive definite on that mesh is reported as ``unstable`` with no
  zero-point energy: an unstable quadratic background has no harmonic
  vacuum (NBCP note, section 3). Stability is tested on the mesh only;
- ``E_cl + E_zp`` for the stable candidates;
- magnetization per spin and skyrmion number (:mod:`~spintoolkit.observables.texture`).

The ranking orders the stable candidates by harmonic energy, then the others
by classical energy. It compares only the states supplied: the lowest of an
incomplete set need not be the ground state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from spintoolkit.methods.classical import classical_energy, refine_classical, torques
from spintoolkit.observables.texture import SkyrmionCharge, skyrmion_charge
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import SpinModel


@dataclass(frozen=True)
class CandidateReport:
    """Classical and harmonic energies of one candidate (per spin, energy unit E0).

    ``status`` is ``"stable"`` or ``"unstable"``; ``zero_point_energy``,
    ``harmonic_energy`` and ``lowest_magnon`` are NaN unless stable.
    """

    name: str
    state: SpinState
    classical_energy: float
    max_torque: float
    status: str
    message: str
    zero_point_energy: float
    harmonic_energy: float
    lowest_magnon: float
    magnetization: np.ndarray
    skyrmion: SkyrmionCharge
    mesh: tuple

    def to_dict(self) -> Dict[str, Any]:
        def num(x):
            return None if not np.isfinite(x) else float(x)
        return {"name": self.name, "num_sites": len(self.state.directions),
                "supercell": self.state.supercell.tolist(),
                "classical_energy": num(self.classical_energy), "max_torque": num(self.max_torque),
                "status": self.status, "message": self.message,
                "zero_point_energy": num(self.zero_point_energy),
                "harmonic_energy": num(self.harmonic_energy),
                "lowest_magnon": num(self.lowest_magnon),
                "magnetization": np.round(self.magnetization, 12).tolist(),
                "skyrmion": self.skyrmion.to_dict(), "mesh": list(self.mesh)}


def magnetic_mesh(state: SpinState, k_density: int) -> tuple:
    """Mesh of the magnetic zone with about ``k_density`` primitive momenta per direction."""
    n = max(2, math.ceil(k_density / math.sqrt(state.num_cells)))
    return (n, n)


def compare_states(model: SpinModel, candidates: Mapping[str, SpinState],
                   conditions: Optional[ExternalConditions] = None, k_density: int = 24,
                   refine: bool = True) -> List[CandidateReport]:
    """Rank candidate states by classical and harmonic (LSWT) energy.

    Parameters
    ----------
    model : SpinModel
    candidates : mapping name -> SpinState
    conditions : ExternalConditions, optional
    k_density : int
        Primitive momenta per reciprocal direction for the zero-point energy.
    refine : bool
        Refine every candidate to a stationary point first
        (:func:`~spintoolkit.methods.classical.refine_classical`).

    Returns
    -------
    list of CandidateReport
        Stable candidates by ``harmonic_energy``, then the rest by
        ``classical_energy``.
    """
    from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
    from spintoolkit.methods.lswt.run import LSWTError

    conditions = conditions or ExternalConditions()
    reports = []
    for name, state in candidates.items():
        if refine:
            state = refine_classical(model, state, conditions)
        e_cl = float(classical_energy(model, state, conditions))
        torque = float(np.max(np.linalg.norm(
            np.array(list(torques(model, state, conditions).values())), axis=-1), initial=0.0))
        mesh = magnetic_mesh(state, k_density)
        status, message, e_zp, lowest = "stable", "", float("nan"), float("nan")
        try:
            result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=mesh))
            e_zp = float(result.zero_point_energy)
            lowest = float(np.min(result.bands()))
        except LSWTError as error:
            status, message = "unstable", str(error).splitlines()[0][:200]
        spins = np.array([model.site(site).spin for site, _ in state.directions])
        moments = spins[:, None] * np.array(list(state.directions.values()))
        reports.append(CandidateReport(
            name, state, e_cl, torque, status, message, e_zp,
            e_cl + e_zp if status == "stable" else float("nan"), lowest,
            moments.mean(axis=0), skyrmion_charge(model, state), mesh))
    stable = sorted((r for r in reports if r.status == "stable"), key=lambda r: r.harmonic_energy)
    other = sorted((r for r in reports if r.status != "stable"), key=lambda r: r.classical_energy)
    return stable + other
