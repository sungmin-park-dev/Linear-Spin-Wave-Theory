"""Magnetization curve M(h) at classical and harmonic order, and the moment reduction.

The magnetization per site along the field direction ``e`` is the field
derivative of the ground-state energy per site,

    M(h) = - dE(h) / dh,    field = h e  (h = mu_B B / E0, M in mu_B),

which with the Zeeman term ``- mu_B B . g . S`` includes the g-tensor.
Two orders are reported:

- classical: ``E_cl(h)`` at the classical minimum followed from the previous
  field (``refine_classical``); by the Hellmann-Feynman theorem this equals
  the moment ``e . g . <S>_cl`` per site;
- harmonic: ``E_cl(h) + E_zp(h)``, the first 1/S correction (Zhitomirsky and
  Nikuni, PRB 57, 5013 (1998)). Its derivative contains both the reduction
  of the moment by zero-point fluctuations and the 1/S shift of the canting
  angle; ``g . (S - <n>) n_i`` alone would miss the latter and is not used;
- thermal (optional, D49): ``M(h, t) = -dF(h, t)/dh`` with
  ``F = E_cl + E_zp + t <sum_n ln(1 - exp(-omega/t))> / N_s`` (the free
  energy of :func:`~spintoolkit.observables.thermal.thermal_quantities`). It
  equals the harmonic value at ``t = 0`` and keeps the canting-angle shift at
  ``t > 0``; the moment sum ``ThermalResult.magnetization`` does not. The
  classical state is the zero-temperature one at each field (no thermal
  self-consistency), so this is a low-temperature result. Near a Goldstone
  mode the integrand ``n_B d omega/dh`` stays finite in two dimensions, so
  M is finite where boson numbers diverge.

Validity of the thermal values (D49). For a gapped spectrum ``beyond_lswt``
marks the temperatures where some thermal ``<n_i> > S_i``, as in
``ThermalResult.beyond_lswt``. Where the zero-mode scan finds zero modes or
candidates (``gapless``), ``<n_i>`` diverges at every ``t > 0`` in two
dimensions and gives no criterion: only ``t = 0`` is checked, and ``M(h, t)``
applies for ``t`` small compared with the spin-wave energy scale (``J S``).

The derivative is a central difference with step ``step`` at fixed state
branch (the state is refined at ``h +- step`` from the state at ``h``).
Where the reference state is unstable in LSWT the harmonic value is NaN. At
a first-order transition the followed branch is metastable beyond the
crossing; compare the energies with :func:`~spintoolkit.methods.phase_competition.compare_states`.

The ordered moment of each site at harmonic order is ``S_i - <n_i>``
(``LSWTResult.ordered_moments``); ``<n_i>`` is the moment reduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List, Optional, Sequence
import warnings

import numpy as np

from spintoolkit.methods.classical import classical_energy, refine_classical
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import SpinModel


@dataclass(frozen=True)
class MagnetizationCurve:
    """Magnetization per site against field.

    Attributes
    ----------
    fields : (nh,) array
        ``h`` along ``direction``.
    direction : (3,) array
        Unit field direction.
    classical : (nh,) array
        ``-dE_cl/dh`` per site (mu_B).
    harmonic : (nh,) array
        ``-d(E_cl + E_zp)/dh`` per site; NaN where LSWT is unstable.
    spins : (Ns,) array
    moment_reduction : (nh, Ns) array
        ``<n_i>`` at zero temperature; NaN where unstable.
    states : list of SpinState
        Classical state at each field.
    temperatures : (nt,) array
        ``t = k_B T / E0`` of ``thermal``; empty if not requested.
    thermal : (nh, nt) array
        ``-dF/dh`` per site at each field and temperature; NaN where LSWT is
        unstable. Column ``t = 0`` equals ``harmonic``.
    beyond_lswt : (nh, nt) bool array
        True where some ``<n_i> > S_i`` at that field and temperature; at
        ``t > 0`` checked only where ``gapless`` is False.
    gapless : (nh,) bool array
        True where the zero-mode scan of the LSWT result at that field finds
        zero modes or candidates; empty if ``temperatures`` is not given.
    """

    fields: np.ndarray
    direction: np.ndarray
    classical: np.ndarray
    harmonic: np.ndarray
    spins: np.ndarray
    moment_reduction: np.ndarray
    states: List[SpinState]
    temperatures: np.ndarray = field(default_factory=lambda: np.zeros(0))
    thermal: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    beyond_lswt: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    gapless: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    @property
    def ordered_moments(self) -> np.ndarray:
        """``S_i - <n_i>``, shape (nh, Ns)."""
        return self.spins[None] - self.moment_reduction


def _mesh(state: SpinState, k_density: int) -> tuple:
    n = max(2, math.ceil(k_density / math.sqrt(state.num_cells)))
    return (n, n)


def magnetization_curve(model: SpinModel, state: SpinState, fields: Sequence[float],
                        direction: Sequence[float] = (0.0, 0.0, 1.0), k_density: int = 24,
                        step: float = 1e-4, temperatures: Optional[Sequence[float]] = None
                        ) -> MagnetizationCurve:
    """Classical and harmonic magnetization along a field sweep.

    Parameters
    ----------
    model : SpinModel
        Must contain Zeeman terms for the field to act.
    state : SpinState
        Starting state at ``fields[0]``; it is refined and followed in the
        order of ``fields``.
    fields : sequence of float
        Field strengths ``h`` (dimensionless).
    direction : (3,) array_like
        Field direction.
    k_density : int
        Primitive momenta per reciprocal direction for ``E_zp`` (as in
        :func:`~spintoolkit.methods.phase_competition.compare_states`).
    step : float
        Field step of the central difference.
    temperatures : sequence of float, optional
        Dimensionless ``t = k_B T / E0 >= 0``; if given, ``thermal`` holds
        ``-dF/dh`` at these temperatures.

    Returns
    -------
    MagnetizationCurve
    """
    from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
    from spintoolkit.methods.lswt.run import LSWTError
    from spintoolkit.observables.thermal import magnon_free_energy, thermal_quantities
    from spintoolkit.observables.zero_modes import scan_zero_modes

    e = np.asarray(direction, dtype=float)
    e = e / np.linalg.norm(e)
    fields = np.asarray(fields, dtype=float)
    t = np.zeros(0) if temperatures is None else np.asarray(temperatures, dtype=float).ravel()
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise ValueError("temperatures must be finite and non-negative")
    settings = LSWTSettings(mesh=_mesh(state, k_density))

    def conditions(h):
        return ExternalConditions(field=tuple(h * e))

    def validity(result):
        """(beyond_lswt row, gapless) at one field."""
        if result is None or len(t) == 0:
            return np.zeros(len(t), dtype=bool), False
        report = scan_zero_modes(result)
        gapless = bool(report.has_zero or report.has_candidates)
        if gapless:
            return (t == 0) & bool(np.any(result.boson_numbers > result.spins)), True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            thermal_result = thermal_quantities(result, t, zero_modes=report, gapless=False)
        return thermal_result.beyond_lswt, False

    def energies(h, start):
        st = refine_classical(model, start, conditions(h))
        e_cl = float(classical_energy(model, st, conditions(h)))
        try:
            result = solve_lswt(model, st, conditions(h), settings=settings)
        except LSWTError:
            return st, e_cl, float("nan"), np.full(len(t), np.nan), None
        harm = e_cl + float(result.zero_point_energy)
        return st, e_cl, harm, harm + magnon_free_energy(result, t), result

    spins = None
    classical, harmonic, thermal, reduction, states = [], [], [], [], []
    beyond, gapless = [], []
    current = state
    for h in fields:
        current, _, _, _, result = energies(h, current)
        _, cl_plus, harm_plus, free_plus, _ = energies(h + step, current)
        _, cl_minus, harm_minus, free_minus, _ = energies(h - step, current)
        classical.append(-(cl_plus - cl_minus) / (2 * step))
        harmonic.append(-(harm_plus - harm_minus) / (2 * step))
        thermal.append(-(free_plus - free_minus) / (2 * step))
        if result is not None and spins is None:
            spins = np.asarray(result.spins, dtype=float)
        reduction.append(result.boson_numbers if result is not None else None)
        states.append(current)
        row, flag = validity(result)
        beyond.append(row)
        gapless.append(flag)
    if spins is None:                       # unstable everywhere: site order of the state
        spins = np.array([model.site(site).spin for site, _ in states[0].directions])
    reduction = np.array([np.full(len(spins), np.nan) if r is None else r for r in reduction])
    beyond = np.array(beyond, dtype=bool).reshape(len(fields), len(t))
    if np.any(beyond):
        warnings.warn(f"boson numbers exceed S at {int(beyond.sum())} (field, temperature) points: "
                      "LSWT is not valid there (see beyond_lswt)", UserWarning, stacklevel=2)
    return MagnetizationCurve(fields, e, np.array(classical), np.array(harmonic), spins,
                              reduction, states, t,
                              np.array(thermal).reshape(len(fields), len(t)), beyond,
                              np.array(gapless if len(t) else [], dtype=bool))
