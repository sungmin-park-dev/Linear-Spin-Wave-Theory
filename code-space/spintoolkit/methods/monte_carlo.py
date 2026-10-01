"""Classical Monte Carlo and thermal observables on a finite torus (D40).

Samples ``exp(-E/T)`` for classical spins of fixed length ``S_i`` with the
classical energy of :class:`~spintoolkit.methods.dynamics.ClassicalTorus`
(D37 coherent-state onsite value, D23 torus expansion). ``T`` is in the energy
unit of the model (``k_B = 1``).

Updates. Sites are split into colour classes with no bond inside a class
(greedy colouring of the torus bond graph), so a whole class is updated at
once from the fields of the others.

- Metropolis: ``s' = S normalize(s/S + sigma g)`` with an isotropic Gaussian
  ``g``. The proposal density depends only on ``s . s'``, so it is symmetric;
  acceptance ``min(1, exp(-dE/T))`` with the exact local energy change,
  onsite term included.
- Overrelaxation (sites without an onsite term): reflection of the spin about
  its local field, ``s' = 2 (s . h) h / |h|^2 - s``. It conserves the energy
  and is its own inverse, so it keeps ``exp(-E/T)`` and only speeds up
  decorrelation. Sites with an onsite term are not reflected (their energy
  is not linear in the spin).

Observables per site, for ``N`` spins:

- energy ``e`` and specific heat ``c = N (<e^2> - <e>^2) / T^2``;
- order parameter ``m_q = (1/N) sum_i exp(-i q . r_i) s_i`` (full positions, D13);
- helicity modulus for a twist about the spin axis ``n`` along the spatial
  direction ``u``. A twist rotates the target spin of every bond relative
  to its source by ``delta (u . d_b)`` with the bond vector
  ``d_b = r_target - r_source``:
  ``E(delta) = sum_b s_i^T J_b R_n(delta u . d_b) s_j``. Then
  ``Upsilon = (1/N) [<E''> - (<E'^2> - <E'>^2) / T]`` at ``delta = 0``. Field
  and onsite terms are local and do not change under the relative twist. For
  exchange that commutes with rotations about ``n`` this is the energy of the
  twisted configuration, the usual helicity modulus. For exchange that does
  not (e.g. NBCP J_PD or J_Gamma about z), the relative twist is not the
  energy of any spin configuration: such a twist is rejected unless
  ``u1_projection=True``, which uses the U(1)-symmetric part
  ``J_bar = (1/2 pi) int R_n(a)^T J R_n(a) da`` of every bond. That is the
  stiffness of the symmetric part only; the remainder (clock potential and
  angle-dependent gradient terms) is left out and must be treated separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from spintoolkit.methods.dynamics import ClassicalTorus


def colour_classes(torus: ClassicalTorus) -> List[np.ndarray]:
    """Site classes with no bond inside a class (greedy colouring)."""
    n = torus.num_sites
    neighbours = [set() for _ in range(n)]
    for i, j in zip(torus.bond_source, torus.bond_target):
        neighbours[i].add(j)
        neighbours[j].add(i)
    colour = -np.ones(n, dtype=int)
    for i in sorted(range(n), key=lambda x: -len(neighbours[x])):
        used = {colour[j] for j in neighbours[i]}
        c = 0
        while c in used:
            c += 1
        colour[i] = c
    return [np.flatnonzero(colour == c) for c in range(colour.max() + 1)]


@dataclass
class MonteCarlo:
    """Metropolis and overrelaxation sweeps at temperature ``T``.

    Parameters
    ----------
    torus : ClassicalTorus
    temperature : float
        ``T > 0`` in the model energy unit.
    step : float
        Width ``sigma`` of the Gaussian proposal (radians, roughly); keep it
        fixed while measuring (see :meth:`tune`).
    overrelaxation : int
        Overrelaxation sweeps per Metropolis sweep.
    seed : int or numpy Generator, optional
    """

    torus: ClassicalTorus
    temperature: float
    step: float = 0.5
    overrelaxation: int = 1
    seed: Optional[object] = None
    acceptance: float = field(default=float("nan"), init=False)

    def __post_init__(self):
        if not self.temperature > 0:
            raise ValueError("temperature must be positive")
        self._rng = np.random.default_rng(self.seed)
        self._classes = colour_classes(self.torus)
        self._has_onsite = np.any(self.torus.onsite != 0, axis=(1, 2))

    def _external_fields(self, spins: np.ndarray, sites: np.ndarray) -> np.ndarray:
        """Field on ``sites`` from everything but their own onsite term."""
        h = self.torus.fields(spins)[sites]
        return h + 2 * np.einsum("iab,ib->ia", self.torus.onsite[sites], spins[sites])

    def _metropolis(self, spins: np.ndarray) -> float:
        accepted = 0
        S = self.torus.lengths
        for sites in self._classes:
            old = spins[sites]
            trial = old / S[sites, None] + self.step * self._rng.normal(size=old.shape)
            new = S[sites, None] * trial / np.linalg.norm(trial, axis=1, keepdims=True)
            h = self._external_fields(spins, sites)
            A = self.torus.onsite[sites]
            dE = (-np.einsum("ia,ia->i", h, new - old)
                  + np.einsum("ia,iab,ib->i", new, A, new) - np.einsum("ia,iab,ib->i", old, A, old))
            accept = self._rng.random(len(sites)) < np.exp(-np.clip(dE, 0, None) / self.temperature)
            spins[sites[accept]] = new[accept]
            accepted += int(accept.sum())
        return accepted / self.torus.num_sites

    def _overrelax(self, spins: np.ndarray) -> None:
        for sites in self._classes:
            sites = sites[~self._has_onsite[sites]]
            if len(sites) == 0:
                continue
            h = self.torus.fields(spins)[sites]
            norm2 = np.einsum("ia,ia->i", h, h)
            ok = norm2 > 0
            s = spins[sites[ok]]
            hh = h[ok]
            spins[sites[ok]] = (2 * np.einsum("ia,ia->i", s, hh) / norm2[ok])[:, None] * hh - s

    def sweep(self, spins: np.ndarray, num_sweeps: int = 1) -> np.ndarray:
        """Advance ``spins`` (a copy is returned) by ``num_sweeps`` sweeps."""
        spins = np.array(spins, dtype=float)
        rates = []
        for _ in range(num_sweeps):
            rates.append(self._metropolis(spins))
            for _ in range(self.overrelaxation):
                self._overrelax(spins)
        self.acceptance = float(np.mean(rates)) if rates else float("nan")
        return spins

    def tune(self, spins: np.ndarray, target: float = 0.5, rounds: int = 20,
             sweeps: int = 10) -> np.ndarray:
        """Adjust ``step`` towards the acceptance ``target`` (use before measuring only)."""
        for _ in range(rounds):
            spins = self.sweep(spins, sweeps)
            self.step = float(np.clip(self.step * np.exp(self.acceptance - target), 1e-3, 10.0))
        return spins


def order_parameter(torus: ClassicalTorus, spins: np.ndarray, q) -> np.ndarray:
    """``m_q = (1/N) sum_i exp(-i q . r_i) s_i``, complex (3,)."""
    phase = np.exp(-1j * torus.positions @ np.asarray(q, dtype=float))
    return phase @ spins / torus.num_sites


def _generator(axis) -> np.ndarray:
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    return np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])   # G v = n x v


def u1_part(exchange: np.ndarray, axis) -> np.ndarray:
    """``(1/2 pi) int R_n(a)^T J R_n(a) da`` of each (3, 3) matrix (exact 8-point average)."""
    G = _generator(axis)
    total = np.zeros_like(np.asarray(exchange, dtype=float))
    for a in 2 * np.pi * np.arange(8) / 8:
        R = np.eye(3) + np.sin(a) * G + (1 - np.cos(a)) * G @ G
        total = total + R.T @ exchange @ R
    return total / 8


def twist_derivatives(torus: ClassicalTorus, spins: np.ndarray, axis=(0.0, 0.0, 1.0),
                      direction=(1.0, 0.0), u1_projection: bool = False) -> Tuple[float, float]:
    """``(dE/d delta, d2E/d delta2)`` of the total energy under the relative twist (module docstring).

    Raises
    ------
    ValueError
        If an exchange matrix does not commute with rotations about ``axis``
        and ``u1_projection`` is False.
    """
    G = _generator(axis)
    J = torus.bond_exchange
    if u1_projection:
        J = u1_part(J, axis)
    else:
        commutator = np.max(np.abs(J @ G - G @ J), initial=0.0)
        if commutator > 1e-12 * max(1.0, float(np.max(np.abs(J), initial=0.0))):
            raise ValueError("exchange is not U(1) symmetric about the twist axis: the relative "
                             "twist is not a configuration energy (use u1_projection=True for "
                             "the stiffness of the symmetric part)")
    theta = torus.bond_vectors @ (np.asarray(direction, dtype=float) / np.linalg.norm(direction))
    s_i, s_j = spins[torus.bond_source], spins[torus.bond_target]
    g_sj = s_j @ G.T
    gg_sj = g_sj @ G.T
    first = np.sum(theta * np.einsum("ma,mab,mb->m", s_i, J, g_sj))
    second = np.sum(theta ** 2 * np.einsum("ma,mab,mb->m", s_i, J, gg_sj))
    return float(first), float(second)


@dataclass(frozen=True)
class ThermalAverages:
    """Monte Carlo averages per site with binned standard errors.

    Attributes
    ----------
    temperature : float
    energy, energy_error : float
    specific_heat, specific_heat_error : float
    order : dict
        ``label -> (mean |m_q|, error)`` for the requested momenta.
    helicity : dict
        ``(axis, direction) label -> (Upsilon, error)``.
    acceptance : float
    series : dict
        Raw measurement series (energy per site, ``m_q`` vectors, twist derivatives).
    """

    temperature: float
    energy: float
    energy_error: float
    specific_heat: float
    specific_heat_error: float
    order: Dict[str, Tuple[float, float]]
    helicity: Dict[str, Tuple[float, float]]
    acceptance: float
    series: Dict[str, np.ndarray]


def _binned(values: np.ndarray, estimator, bins: int) -> Tuple[float, float]:
    """Estimator on the whole series and its jackknife error over ``bins`` blocks."""
    full = estimator(values)
    blocks = np.array_split(np.arange(len(values)), bins)
    leave_out = np.array([estimator(np.delete(values, b, axis=0)) for b in blocks])
    error = np.sqrt((bins - 1) * np.mean((leave_out - leave_out.mean()) ** 2))
    return float(full), float(error)


def thermal_averages(sampler: MonteCarlo, spins: np.ndarray, thermalization_sweeps: int,
                     measurements: int, sweeps_between: int = 1,
                     momenta: Optional[Dict[str, object]] = None,
                     twists: Optional[Dict[str, Tuple[object, object]]] = None,
                     bins: int = 20, u1_projection: bool = False) -> Tuple[ThermalAverages, np.ndarray]:
    """Thermalize, then measure every ``sweeps_between`` sweeps.

    Parameters
    ----------
    sampler : MonteCarlo
    spins : (n, 3) array
        Initial configuration.
    momenta : dict, optional
        ``label -> q`` (Cartesian) for order parameters.
    twists : dict, optional
        ``label -> (axis, direction)`` for helicity moduli.
    u1_projection : bool
        Passed to :func:`twist_derivatives`.
    bins : int
        Jackknife blocks for the errors (blocks should exceed the
        autocorrelation time; check by changing ``sweeps_between``).

    Returns
    -------
    ThermalAverages, final spins
    """
    torus, T = sampler.torus, sampler.temperature
    momenta, twists = momenta or {}, twists or {}
    spins = sampler.sweep(spins, thermalization_sweeps)
    energy = np.empty(measurements)
    order = {label: np.empty((measurements, 3), complex) for label in momenta}
    derivatives = {label: np.empty((measurements, 2)) for label in twists}
    rates = []
    for m in range(measurements):
        spins = sampler.sweep(spins, sweeps_between)
        rates.append(sampler.acceptance)
        energy[m] = torus.energy(spins)
        for label, q in momenta.items():
            order[label][m] = order_parameter(torus, spins, q)
        for label, (axis, direction) in twists.items():
            derivatives[label][m] = twist_derivatives(torus, spins, axis, direction,
                                                       u1_projection)
    N = torus.num_sites
    e, e_err = _binned(energy, np.mean, bins)
    c, c_err = _binned(energy, lambda x: N * np.var(x) / T ** 2, bins)
    order_out = {label: _binned(np.linalg.norm(v, axis=1), np.mean, bins)
                 for label, v in order.items()}
    helicity = {label: _binned(d, lambda x: (np.mean(x[:, 1]) - np.var(x[:, 0]) / T) / N, bins)
                for label, d in derivatives.items()}
    series = {"energy": energy, **{f"m_{k}": v for k, v in order.items()},
              **{f"twist_{k}": v for k, v in derivatives.items()}}
    return ThermalAverages(T, e, e_err, c, c_err, order_out, helicity,
                           float(np.mean(rates)), series), spins
