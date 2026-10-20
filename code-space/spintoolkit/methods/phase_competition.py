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
  vacuum (NBCP note, section 3);
- the continuous-k local stability test (D51,
  :func:`~spintoolkit.methods.stability.local_stability`, on by default,
  D53). A state that passes the mesh but is ``unstable`` between mesh points
  (e.g. a negative Goldstone stiffness) is reported as ``unstable`` without
  zero-point energy, for the same reason (including a fall beyond harmonic
  order, D54); ``marginal`` and ``not stationary`` verdicts are attached but
  do not change the status;
- ``E_cl + E_zp`` for the stable candidates;
- magnetization per spin and skyrmion number (:mod:`~spintoolkit.observables.texture`).

With ``augmentation="minimal"`` (MAGSWT, D62) a classically unstable but
stationary candidate is reported as ``augmented`` with the upper estimate
``E_cl + E_zp(mu)`` and its shift ``mu``; such candidates should be prepared
with ``refine="stationary"`` (:func:`~spintoolkit.methods.stationary.find_stationary_state`,
D63), since energy minimization slides off a saddle point.

The ranking orders the stable and augmented candidates by harmonic energy,
then the others by classical energy. It compares only the states supplied: the lowest of an
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

    ``status`` is ``"stable"``, ``"augmented"`` (MAGSWT, D62: classically
    unstable, energy an upper estimate with shift ``augmentation_mu``),
    ``"unstable"`` or ``"not stationary"``; ``zero_point_energy`` and
    ``harmonic_energy`` are NaN unless stable or augmented, ``lowest_magnon``
    unless stable (augmented magnons are not those of the model).
    ``stability`` is the continuous-k :class:`~spintoolkit.methods.stability.StabilityReport`
    (verdict ``stable``, ``unstable``, ``marginal`` or ``not stationary``),
    or None when the test was switched off.
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
    stability: Optional[Any] = None
    augmentation_mu: float = float("nan")

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
                "skyrmion": self.skyrmion.to_dict(), "mesh": list(self.mesh),
                "local_stability": None if self.stability is None else self.stability.to_dict(),
                "augmentation_mu": num(self.augmentation_mu)}


def magnetic_mesh(state: SpinState, k_density: int) -> tuple:
    """Mesh of the magnetic zone with about ``k_density`` primitive momenta per direction."""
    n = max(2, math.ceil(k_density / math.sqrt(state.num_cells)))
    return (n, n)


def compare_states(model: SpinModel, candidates: Mapping[str, SpinState],
                   conditions: Optional[ExternalConditions] = None, k_density: int = 24,
                   refine: Any = True, local_stability: bool = True,
                   augmentation: Optional[Any] = None) -> List[CandidateReport]:
    """Rank candidate states by classical and harmonic (LSWT) energy.

    Parameters
    ----------
    model : SpinModel
    candidates : mapping name -> SpinState
    conditions : ExternalConditions, optional
    k_density : int
        Primitive momenta per reciprocal direction for the zero-point energy.
    refine : bool or "stationary"
        ``True``: refine every candidate to a nearby energy minimum first
        (:func:`~spintoolkit.methods.classical.refine_classical`).
        ``"stationary"``: refine to the nearby stationary point, saddles
        included (:func:`~spintoolkit.methods.stationary.find_stationary_state`);
        a candidate without one is ``"not stationary"``. ``False``: as given.
    local_stability : bool
        Run the continuous-k local stability test on every candidate
        (default True). An ``unstable`` verdict marks a mesh-stable candidate
        unstable; False keeps the mesh-only test.
    augmentation : None, "minimal" or float
        MAGSWT for classically unstable candidates (D62, see
        :class:`~spintoolkit.methods.lswt.LSWTSettings`).

    Returns
    -------
    list of CandidateReport
        Stable and augmented candidates by ``harmonic_energy``, then the rest
        by ``classical_energy``.
    """
    from spintoolkit.methods.stationary import find_stationary_state
    from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
    from spintoolkit.methods.lswt.run import LSWTError
    from spintoolkit.methods.stability import UNSTABLE
    from spintoolkit.methods.stability import local_stability as stability_test

    conditions = conditions or ExternalConditions()
    reports = []
    for name, state in candidates.items():
        stationary = True
        if refine == "stationary":
            search = find_stationary_state(model, state, conditions)
            state, stationary = search.state, search.converged
        elif refine:
            state = refine_classical(model, state, conditions)
        e_cl = float(classical_energy(model, state, conditions))
        torque = float(np.max(np.linalg.norm(
            np.array(list(torques(model, state, conditions).values())), axis=-1), initial=0.0))
        mesh = magnetic_mesh(state, k_density)
        status, message, e_zp, lowest = "stable", "", float("nan"), float("nan")
        mu = float("nan")
        if not stationary:
            status = "not stationary"
            message = f"no stationary point found near the seed (max torque {torque:.3g} E0)"
        else:
            try:
                result = solve_lswt(model, state, conditions,
                                    settings=LSWTSettings(mesh=mesh, augmentation=augmentation))
                e_zp = float(result.zero_point_energy)
                if "augmentation" in result.extra:
                    mu = float(result.extra["augmentation"]["mu"])
                if mu > 0:
                    status, message = "augmented", result.extra["augmentation"]["label"]
                else:
                    lowest = float(np.min(result.bands()))
            except LSWTError as error:
                status, message = "unstable", str(error).splitlines()[0][:200]
        stability = (stability_test(model, state, conditions)
                     if local_stability and stationary else None)
        if status == "stable" and stability is not None and stability.verdict == UNSTABLE:
            # Without augmentation, or when the shift search found nothing to lift.
            status, e_zp, lowest = "unstable", float("nan"), float("nan")
            detail = "; ".join(stability.reasons) or stability.summary()
            message = f"unstable between mesh points (local_stability): {detail}"[:200]
        spins = np.array([model.site(site).spin for site, _ in state.directions])
        moments = spins[:, None] * np.array(list(state.directions.values()))
        ranked = status in ("stable", "augmented")
        reports.append(CandidateReport(
            name, state, e_cl, torque, status, message, e_zp,
            e_cl + e_zp if ranked else float("nan"), lowest,
            moments.mean(axis=0), skyrmion_charge(model, state), mesh, stability, mu))
    stable = sorted((r for r in reports if r.status in ("stable", "augmented")),
                    key=lambda r: r.harmonic_energy)
    other = sorted((r for r in reports if r.status not in ("stable", "augmented")),
                   key=lambda r: r.classical_energy)
    return stable + other
