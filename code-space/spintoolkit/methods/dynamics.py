"""Classical spin dynamics on a finite torus: Landau-Lifshitz and Langevin (D38).

Spins are classical vectors ``s_i`` of fixed length ``S_i`` with the classical
energy of :mod:`spintoolkit.methods.classical` (onsite terms with the
coherent-state factor ``1 - 1/(2 S_i)``, D37), expanded on a finite periodic
torus with the rules shared by every method (D23). With ``hbar = 1`` and the
energy unit E0 (D20), time is measured in ``hbar / E0``.

Equation of motion. The Heisenberg equation of ``H = -h . S`` is
``dS/dt = S x h``; for a general Hamiltonian ``h_i = -dE/ds_i`` is the local
field and

    ds_i/dt = s_i x h_i                                    (Landau-Lifshitz)

conserves every spin length and the energy. Because ``E`` is quadratic in the
spins, ``h_i`` is affine in them, and the implicit midpoint rule
``s' = s + dt sbar x h(sbar)``, ``sbar = (s + s')/2`` conserves both exactly
(``(s' - s) . sbar = 0`` and ``E(s') - E(s) = -(s' - s) . h(sbar) = 0``). For
small oscillations it maps a mode of frequency ``w`` to the frequency
``(2/dt) arctan(w dt/2)``; at low temperature the normal modes of the linear
dynamics are exactly the LSWT frequencies of the same classical energy.

Langevin dynamics. With the dimensionless Gilbert damping ``alpha`` and a
Gaussian white noise ``xi_i``,

    ds_i/dt = s_i x (h_i + xi_i) - (alpha / S_i) s_i x (s_i x h_i),
    <xi_ia(t) xi_jb(t')> = (2 alpha T / S_i) delta_ij delta_ab delta(t - t'),

in the Stratonovich sense (the noise rotates the spin, so the length is kept).
The damping moves ``s_i`` along ``alpha S_i h_perp`` (mobility ``alpha S_i``),
the noise diffuses it on the sphere of radius ``S_i`` with coefficient
``S_i^2 (alpha T / S_i)``; their ratio is ``T``, so the stationary
distribution is ``exp(-E/T)`` with the uniform measure on each sphere (the
precession conserves ``E`` and the measure). ``T`` is in E0 (``k_B = 1``).
The integrator is the stochastic Heun scheme (Stratonovich-consistent) followed
by a projection onto the spheres.

Structure factor. ``s_q(t) = sum_i exp(-i q . r_i) s_i(t)`` with full site
positions (D13) at the torus momenta; ``S^ab(q, w)`` is the windowed
periodogram of the deterministic trajectories started from the given samples,

    S^ab(q, w) = < conj(s~^a_q(w)) s~^b_q(w) > / (N dt sum_t w_t^2),
    s~_q(w) = dt sum_t w_t exp(+i w t) s_q(t),

so that ``sum_w S(q, w) dw / 2 pi`` is the (window-weighted) time average of
``|s_q(t)|^2 / N``. This is the classical correlation function: no quantum
(detailed-balance) factor and no form factor is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy import sparse

from spintoolkit.definitions.defaults import (
    DYNAMICS_MIDPOINT_MAX_ITERATIONS, DYNAMICS_MIDPOINT_TOL)
from spintoolkit.states.spin_state import SpinState, validate_spin_state
from spintoolkit.system.cluster import allowed_momenta, expand_on_torus
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import SpinModel, onsite_renormalization


class ClassicalTorus:
    """Classical energy and local fields of a model on a finite torus.

    Parameters
    ----------
    model : SpinModel
    geometry : CalculationGeometry
        A finite torus (bonds folding onto one site are rejected, D23).
    conditions : ExternalConditions, optional
        Dimensionless field (default zero).

    Attributes
    ----------
    keys : tuple of (site_id, cell)
    lengths : (n,) array
        Spin lengths ``S_i``.
    positions : (n, 2) array
        Cartesian site positions.
    momenta : (N_c, 2) array
        Cartesian momenta allowed by the torus.
    """

    def __init__(self, model: SpinModel, geometry: CalculationGeometry,
                 conditions: Optional[ExternalConditions] = None):
        cluster = expand_on_torus(model, geometry)
        self.model = model
        self.geometry = geometry
        self.keys = cluster.keys
        self.lengths = np.asarray(cluster.spins, dtype=float)
        self.positions = cluster.positions
        self.momenta = allowed_momenta(model, geometry)[1]
        n = len(self.keys)
        # E = 1/2 s^T K s - z . s + c with s the flattened (3n,) spin vector.
        rows, cols, values = [], [], []

        def block(i, j, M):
            r, c = np.meshgrid(np.arange(3) + 3 * i, np.arange(3) + 3 * j, indexing="ij")
            rows.append(r.ravel())
            cols.append(c.ravel())
            values.append(np.asarray(M, dtype=float).ravel())

        for i, j, J in zip(cluster.source, cluster.target, cluster.exchange):
            block(i, j, J)
            block(j, i, J.T)
        # Bond vectors r_target - r_source of the unfolded bonds (twists, D40).
        bilinear = model.terms_of_kind("bilinear")
        vectors = []
        for t in cluster.bond_terms:
            (a, n1), (b, n2) = bilinear[t].participants
            vectors.append(model.cartesian_position(b, n2) - model.cartesian_position(a, n1))
        self.bond_source = np.asarray(cluster.source, dtype=int)
        self.bond_target = np.asarray(cluster.target, dtype=int)
        self.bond_exchange = np.asarray(cluster.exchange, dtype=float).reshape(-1, 3, 3)
        self.bond_vectors = np.array(vectors, dtype=float).reshape(-1, 2)
        kappa = np.array([onsite_renormalization(S) for S in self.lengths])
        #: (n, 3, 3) renormalized onsite matrices kappa_i A_i of the classical energy.
        self.onsite = kappa[:, None, None] * cluster.onsite
        for i in range(n):
            if np.any(cluster.onsite[i]):
                block(i, i, 2 * kappa[i] * cluster.onsite[i])
        if rows:
            self._K = sparse.csr_matrix((np.concatenate(values),
                                         (np.concatenate(rows), np.concatenate(cols))),
                                        shape=(3 * n, 3 * n))
        else:
            self._K = sparse.csr_matrix((3 * n, 3 * n))
        self._zeeman = cluster.fields(conditions).ravel()
        self._constant = 0.5 * float(np.sum(self.lengths * np.trace(cluster.onsite, axis1=1, axis2=2)))

    @property
    def num_sites(self) -> int:
        return len(self.keys)

    def fields(self, spins: np.ndarray) -> np.ndarray:
        """Local fields ``h_i = -dE/ds_i``, shape (n, 3), in E0."""
        return (self._zeeman - self._K @ np.asarray(spins, dtype=float).ravel()).reshape(-1, 3)

    def energy(self, spins: np.ndarray) -> float:
        """Classical energy per site in E0."""
        s = np.asarray(spins, dtype=float).ravel()
        return float((0.5 * s @ (self._K @ s) - self._zeeman @ s + self._constant) / self.num_sites)

    def spins_from_state(self, state: SpinState) -> np.ndarray:
        """Spin vectors ``S_i n_i`` of a state that tiles the torus, shape (n, 3)."""
        validate_spin_state(state, self.model, self.geometry)
        return np.array([S * state.direction(site, state.reduce_cell(cell))
                         for S, (site, cell) in zip(self.lengths, self.keys)])

    def random_spins(self, seed=None) -> np.ndarray:
        """Spins with independent, uniformly distributed directions (infinite temperature)."""
        rng = np.random.default_rng(seed)
        v = rng.normal(size=(self.num_sites, 3))
        return self.lengths[:, None] * v / np.linalg.norm(v, axis=1, keepdims=True)

    def project(self, spins: np.ndarray) -> np.ndarray:
        """Rescale every spin to its length ``S_i``."""
        return self.lengths[:, None] * spins / np.linalg.norm(spins, axis=1, keepdims=True)


class DynamicsError(RuntimeError):
    """An integrator step did not converge."""


@dataclass(frozen=True)
class ImplicitMidpoint:
    """Energy- and length-conserving integrator of the Landau-Lifshitz equation.

    Parameters
    ----------
    dt : float
        Time step in ``hbar / E0``. The fixed-point iteration converges for
        ``dt max|h_i|`` below about 1; accuracy needs it well below that.
    tol : float
        Convergence threshold on the change of the midpoint (in units of S).
    max_iterations : int
    """

    dt: float
    tol: float = DYNAMICS_MIDPOINT_TOL
    max_iterations: int = DYNAMICS_MIDPOINT_MAX_ITERATIONS

    def step(self, torus: ClassicalTorus, spins: np.ndarray) -> np.ndarray:
        half = 0.5 * self.dt
        mid = spins
        for _ in range(self.max_iterations):
            new = spins + half * np.cross(mid, torus.fields(mid))
            change = np.max(np.abs(new - mid))
            mid = new
            if change < self.tol * max(1.0, float(np.max(torus.lengths))):
                return 2 * mid - spins
        raise DynamicsError(f"implicit midpoint did not converge in {self.max_iterations} "
                            f"iterations (last change {change:.2e}); reduce dt")


@dataclass
class Langevin:
    """Stochastic Landau-Lifshitz-Gilbert integrator sampling ``exp(-E/T)``.

    Parameters
    ----------
    dt : float
        Time step in ``hbar / E0``.
    damping : float
        Dimensionless Gilbert damping ``alpha > 0``.
    temperature : float
        ``T`` in E0 (``k_B = 1``), ``T >= 0``.
    seed : int or numpy Generator, optional
    """

    dt: float
    damping: float
    temperature: float
    seed: Optional[object] = None

    def __post_init__(self):
        if self.damping <= 0:
            raise ValueError("Langevin dynamics needs a positive damping")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        self._rng = np.random.default_rng(self.seed)

    def _rate(self, torus, spins, noise):
        h = torus.fields(spins)
        damped = np.cross(spins, np.cross(spins, h)) * (self.damping / torus.lengths)[:, None]
        return np.cross(spins, h + noise) - damped

    def step(self, torus: ClassicalTorus, spins: np.ndarray) -> np.ndarray:
        # White noise averaged over the step: variance 2 alpha T / (S_i dt) per component.
        sigma = np.sqrt(2 * self.damping * self.temperature / (torus.lengths * self.dt))
        noise = sigma[:, None] * self._rng.normal(size=spins.shape)
        first = self._rate(torus, spins, noise)
        predicted = spins + self.dt * first
        second = self._rate(torus, predicted, noise)
        return torus.project(spins + 0.5 * self.dt * (first + second))


def evolve(torus: ClassicalTorus, spins: np.ndarray, integrator, num_steps: int,
           record_every: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Advance ``spins`` by ``num_steps`` steps of ``integrator``.

    Returns
    -------
    spins : (n, 3) array
        Final spins.
    trajectory : (num_records, n, 3) array or None
        Spins at steps ``0, record_every, 2 record_every, ...`` (before each of
        those steps) when ``record_every > 0``.
    """
    spins = np.array(spins, dtype=float)
    records = [] if record_every > 0 else None
    for step in range(num_steps):
        if records is not None and step % record_every == 0:
            records.append(spins.copy())
        spins = integrator.step(torus, spins)
    return spins, (np.array(records) if records is not None else None)


def thermal_samples(torus: ClassicalTorus, langevin: Langevin, num_samples: int,
                    thermalization_steps: int, decorrelation_steps: int,
                    initial: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """Spin configurations drawn from ``exp(-E/T)`` by Langevin dynamics.

    Starts from ``initial`` (default: random spins from the integrator's
    generator), runs ``thermalization_steps``, then keeps one configuration
    every ``decorrelation_steps``. Whether these are long enough is the
    caller's responsibility (check the energy or the autocorrelation).
    """
    spins = torus.random_spins(langevin._rng) if initial is None else np.array(initial, float)
    spins, _ = evolve(torus, spins, langevin, thermalization_steps)
    samples = []
    for _ in range(num_samples):
        spins, _ = evolve(torus, spins, langevin, decorrelation_steps)
        samples.append(spins.copy())
    return samples


@dataclass(frozen=True)
class ClassicalStructureFactor:
    """Classical dynamical structure factor ``S^ab(q, w)`` on a finite torus.

    Attributes
    ----------
    momenta : (N_c, 2) array
        Cartesian momenta of the torus.
    frequencies : (N_w,) array
        Angular frequencies in E0 (``hbar = 1``), ascending, both signs.
    intensity : (N_c, N_w, 3, 3) complex array
        ``S^ab(q, w)`` per site, global Cartesian axes.
    num_samples : int
    dt : float
    """

    momenta: np.ndarray
    frequencies: np.ndarray
    intensity: np.ndarray
    num_samples: int
    dt: float

    def trace(self) -> np.ndarray:
        """``sum_a S^aa(q, w)``, real, shape (N_c, N_w)."""
        return np.real(np.einsum("qwaa->qw", self.intensity))

    def equal_time(self) -> np.ndarray:
        """``sum_w S^ab(q, w) dw / 2 pi`` (window-weighted ``<s_-q^a s_q^b>/N``), (N_c, 3, 3)."""
        dw = self.frequencies[1] - self.frequencies[0]
        return self.intensity.sum(axis=1) * dw / (2 * np.pi)


def classical_structure_factor(torus: ClassicalTorus, samples: Iterable[np.ndarray], dt: float,
                               num_steps: int, window: str = "hann",
                               integrator: Optional[ImplicitMidpoint] = None
                               ) -> ClassicalStructureFactor:
    """``S^ab(q, w)`` averaged over deterministic trajectories from ``samples``.

    Parameters
    ----------
    torus : ClassicalTorus
    samples : iterable of (n, 3) arrays
        Initial configurations, e.g. from :func:`thermal_samples`.
    dt : float
        Time step; the frequency window is ``|w| < pi / dt`` and the
        resolution ``2 pi / (num_steps dt)``.
    num_steps : int
        Number of recorded times per trajectory.
    window : {"hann", "none"}
        Time window applied before the Fourier transform (reduces leakage of
        sharp lines; the normalization keeps the sum rule of the module docstring).
    integrator : ImplicitMidpoint, optional
        Default ``ImplicitMidpoint(dt)``; its ``dt`` must equal ``dt``.
    """
    integrator = integrator or ImplicitMidpoint(dt)
    if not np.isclose(integrator.dt, dt):
        raise ValueError("integrator.dt must equal dt")
    if window == "hann":
        weights = np.hanning(num_steps)
    elif window == "none":
        weights = np.ones(num_steps)
    else:
        raise ValueError("window must be 'hann' or 'none'")
    phases = np.exp(-1j * torus.momenta @ torus.positions.T)            # (N_c, n)
    total = 0.0
    count = 0
    for sample in samples:
        _, trajectory = evolve(torus, sample, integrator, num_steps, record_every=1)
        s_q = np.einsum("qi,tia->tqa", phases, trajectory)              # (t, N_c, 3)
        # dt sum_t w_t exp(+i w t) s(t) = dt * num_steps * ifft
        s_w = dt * num_steps * np.fft.ifft(weights[:, None, None] * s_q, axis=0)
        total = total + np.einsum("wqa,wqb->qwab", s_w.conj(), s_w)
        count += 1
    if count == 0:
        raise ValueError("no samples given")
    norm = count * torus.num_sites * dt * np.sum(weights ** 2)
    frequencies = 2 * np.pi * np.fft.fftfreq(num_steps, d=dt)
    order = np.argsort(frequencies)
    return ClassicalStructureFactor(torus.momenta, frequencies[order],
                                    total[:, order] / norm, count, dt)
