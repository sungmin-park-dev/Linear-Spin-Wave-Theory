"""Longitudinal two-magnon continuum of the spin structure factor (O(S^0), D56, D60).

In the local frame of site ``i`` (axes ``x_i, y_i, n_i``) the Holstein-Primakoff
spin component along the ordered moment is exactly ``S_i - a_i^dagger a_i``,
so the lab-frame spin carries, beside the one-magnon term of
:mod:`~spintoolkit.observables.structure_factor`, the bilinear

    delta S^a_i = - n_i^a a_i^dagger a_i.

With the per-site Fourier convention of that module (full positions, D13),

    L^a(q) = -(N)^{-1/2} sum_i n_i^a sum_k a_i^dagger(k) a_i(k + q)

(``i`` over the magnetic cell, ``k`` over the ``N_c`` cells' momenta,
``N = N_c N_s``). In the harmonic ground state its correlation is carried
only by two-magnon states. With ``psi(k) = T(k) phi(k)``,
``phi(k) = (b(k), b^dagger(-k))``, the creation part of ``L^b(q)^dagger`` is

    sum_k sum_{n n'} C^b_{n n'}(k) phi_{Ns+n}(-k-q) phi_{Ns+n'}(k),
    C^b_{n n'}(k) = -(N)^{-1/2} sum_i n_i^b T_{Ns+i, Ns+n}(-k-q) T_{i, Ns+n'}(k),

where ``phi_{Ns+n}(k) = b_n^dagger(-k)`` creates the magnon of energy
``E_{Ns+n}(k)`` (the hole column of ``T(k)``). At zero temperature

    S_2^{ab}(q, w) = 1/2 sum_k sum_{n n'} A^a_{n n'}(k)^* A^b_{n n'}(k)
                     delta(w - E_{Ns+n}(-k-q) - E_{Ns+n'}(k)),
    A_{n n'}(k) = C_{n n'}(k) + C_{n' n}(-k - q).

The factor 1/2 and the symmetrized amplitude count each pair of magnons
once. The sum over ``k`` is a mesh average, ``(1/N) sum_k -> (1/N_s) <.>_k``.

Order and scope: the one-magnon weights are O(S); this continuum is O(S^0).
The same order also contains the 1/S renormalization of the one-magnon
energies and weights by the magnon interactions (and their decay into the
continuum in noncollinear states), which is *not* included: the result is
the non-interacting two-magnon continuum. A lab-frame (commensurate) result
is required.

Finite temperature (D60): with all ``2 N_s`` columns,
``L^b(q)^dagger = sum_k sum_{mu nu} C^b_{mu nu}(k) phi_mu(-k-q) phi_nu(k)``,
where ``phi_mu`` (``mu < N_s``) annihilates a magnon of energy ``E_mu`` and
``phi_{Ns+n}`` creates one. For free bosons in thermal equilibrium Wick's
theorem gives the connected correlation

    S_2^{ab}(q, w) = 1/2 sum_k sum_{mu nu} A^a_{mu nu}(k)^* A^b_{mu nu}(k)
                     f_mu f_nu delta(w - e_mu - e_nu),

with ``e = +E, f = 1 + n_B(E)`` for a creation operator and
``e = -E, f = n_B(E)`` for an annihilation operator: pair creation at
``w1 + w2`` with ``(1 + n1)(1 + n2)``, thermal scattering of a magnon
(``b^dagger b``, the Landau-type term near ``w = 0``) at ``+-(w1 - w2)``
with ``(1 + n1) n2``, and pair annihilation at ``-w1 - w2`` with
``n1 n2``. Detailed balance ``S(-q, -w) = exp(-w / t) S(q, w)`` holds term
by term. The classical state is fixed and magnons do not interact, so
this is the low-temperature limit at this order; the static (equal-time)
sum is the connected thermal ``<L L^dagger>``. The elastic disconnected
part ``<L>`` (the thermal reduction of the ordered moment at Bragg
vectors) is not included.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence
import warnings

import numpy as np

from spintoolkit.observables.structure_factor import _diagonalization_at


@dataclass(frozen=True)
class TwoMagnonContinuum:
    """Two-magnon states contributing to ``S^{ab}(q, w)`` at a set of momenta.

    Attributes
    ----------
    q : (nq, 2) array
        Cartesian momenta.
    energies : (nq, M) array
        Pair energies ``w_n(k + q) + w_{n'}(-k)`` over the mesh and the band
        pairs; at ``t > 0`` the energy transfer ``e_mu + e_nu`` of every
        operator pair (creation, scattering, annihilation; may be negative).
    weights : (nq, M, 3, 3) complex array
        ``1/2 A^a* A^b`` per pair (times the thermal factors at ``t > 0``),
        per site, so that ``weights.sum(axis=1)`` is the equal-time
        connected longitudinal correlation.
    mesh : (nk, 2) array
        Cartesian momenta ``k`` of the internal sum.
    excluded : (nq,) int array
        Mesh points left out because ``k`` or ``-k - q`` has a zero mode.
    """

    q: np.ndarray
    energies: np.ndarray
    weights: np.ndarray
    mesh: np.ndarray
    excluded: np.ndarray
    settings: Dict[str, Any] = field(default_factory=dict)

    def static(self) -> np.ndarray:
        """Equal-time two-magnon ``S_2^{ab}(q)``, shape (nq, 3, 3)."""
        return self.weights.sum(axis=1)

    def spectrum(self, omega: Sequence[float], eta: float, shape: str = "lorentzian",
                 component: Optional[str] = None) -> np.ndarray:
        """Broadened ``S_2(q, w)`` on a frequency grid, shape (nq, nw).

        Components as in :meth:`StructureFactor.spectrum`.
        """
        omega = np.asarray(omega, dtype=float)
        if component is None:
            weight = np.real(np.einsum("qmaa->qm", self.weights))
        elif component == "neutron":
            weight = np.full(self.energies.shape, np.nan)
            for i, q in enumerate(self.q):
                norm = np.linalg.norm(q)
                if norm == 0:
                    continue
                unit = np.array([q[0], q[1], 0.0]) / norm
                projector = np.eye(3) - np.outer(unit, unit)
                weight[i] = np.real(np.einsum("ab,mab->m", projector, self.weights[i]))
        else:
            a, b = ("xyz".index(component[0]), "xyz".index(component[1]))
            weight = np.real(self.weights[:, :, a, b])
        x = omega[None, None, :] - self.energies[:, :, None]
        if shape == "lorentzian":
            profile = eta / np.pi / (x ** 2 + eta ** 2)
        elif shape == "gaussian":
            profile = np.exp(-0.5 * (x / eta) ** 2) / (eta * np.sqrt(2 * np.pi))
        else:
            raise ValueError("shape must be 'lorentzian' or 'gaussian'")
        return np.einsum("qm,qmw->qw", weight, profile)


def _mesh(result, mesh) -> np.ndarray:
    """Cartesian momenta of a half-step shifted ``n1 x n2`` mesh of the magnetic zone."""
    n1, n2 = mesh
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(result.magnetic_lattice)).T
    p = np.array([((i + 0.5) / n1, (j + 0.5) / n2) for i in range(n1) for j in range(n2)])
    return p @ reciprocal


def _occupations(E: np.ndarray, temperature: float) -> np.ndarray:
    """Bose factor ``1 / (exp(E / t) - 1)``; zero at ``t = 0``."""
    if temperature == 0.0:
        return np.zeros_like(E)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return 1.0 / np.expm1(E / temperature)


def two_magnon_continuum(result, q_points, mesh: Sequence[int] = (24, 24),
                         temperature: float = 0.0, zero_modes=None,
                         gapless: Optional[bool] = None) -> TwoMagnonContinuum:
    """Non-interacting longitudinal two-magnon continuum at momenta ``q_points`` (D56, D60).

    Parameters
    ----------
    result : LSWTResult
        Lab-frame result (commensurate state).
    q_points : (nq, 2) array_like
        Cartesian momenta.
    mesh : (n1, n2)
        Half-step shifted mesh of the magnetic zone for the internal momentum
        ``k`` (the shift avoids the zone centre).
    temperature : float
        ``t = k_B T`` in the energy unit of the model (as in
        :func:`~spintoolkit.observables.thermal.thermal_quantities`).
    zero_modes : ZeroModeReport, optional
        Scan of :func:`~spintoolkit.observables.zero_modes.scan_zero_modes`;
        computed when needed and not given (``t > 0`` only).
    gapless : bool, optional
        Overrides the scan, as in ``thermal_quantities`` (D25).

    Returns
    -------
    TwoMagnonContinuum
        At ``t = 0`` the pair-creation states only (D56). At ``t > 0`` every
        pair of quasiparticle operators: creation of two magnons at
        ``w1 + w2`` with weight ``(1 + n1)(1 + n2)``, scattering at
        ``w1 - w2`` with ``(1 + n1) n2`` and annihilation at ``-w1 - w2``
        with ``n1 n2`` (module docstring). If the spectrum is gapless the
        weights at ``t > 0`` are NaN.

    Raises
    ------
    ZeroModeCandidateError
        At ``t > 0`` if the scan finds candidates and ``gapless`` is None.

    Notes
    -----
    See the module docstring for the formula and its scope: O(S^0), without
    the interaction corrections of the same order. Mesh points where ``k`` or
    ``-k - q`` has a zero mode (e.g. a Goldstone mode at a Bragg vector) are
    left out and counted in ``excluded``; their measure vanishes as the mesh
    is refined, and the weights stay integrable there at ``t = 0``. At
    ``t > 0`` a zero mode makes the thermal weights diverge logarithmically
    with the mesh in two dimensions (the scattering weight near a zero mode
    goes as ``n |A|^2``), the same Mermin-Wagner divergence as the boson
    numbers of :func:`~spintoolkit.observables.thermal.thermal_quantities`;
    the continuum is then not defined and is reported as NaN.
    """
    from spintoolkit.methods.lswt.run import require_lab_frame

    require_lab_frame(result, "two_magnon_continuum")
    t = float(temperature)
    if t < 0:
        raise ValueError("temperature must be non-negative")
    settings: Dict[str, Any] = {"mesh": [int(m) for m in mesh], "temperature": t}
    undefined = False
    if t > 0:
        from spintoolkit.observables.thermal import ZeroModeCandidateError
        from spintoolkit.observables.zero_modes import scan_zero_modes

        report = zero_modes if zero_modes is not None else scan_zero_modes(result)
        if gapless is None:
            if report.has_candidates:
                raise ZeroModeCandidateError(report)
            undefined, decision = report.has_zero, "scan"
        else:
            undefined, decision = bool(gapless), "user"
        settings.update(gapless=undefined, gapless_decision=decision)
        if undefined:
            warnings.warn("gapless spectrum: the two-magnon continuum at t > 0 diverges with "
                          "the mesh in two dimensions and is reported as NaN", UserWarning,
                          stacklevel=2)
    q = np.atleast_2d(np.asarray(q_points, dtype=float))
    ns = result.num_sites
    n = np.asarray(result.directions, dtype=float)                 # (Ns, 3) ordered axes
    k = _mesh(result, mesh)
    nk = len(k)
    # operators phi_mu(p): mu < Ns annihilates (energy change -E), mu >= Ns creates (+E)
    cols = slice(ns, 2 * ns) if t == 0 else slice(0, 2 * ns)
    sign = np.r_[-np.ones(ns), np.ones(ns)][cols]
    m = len(sign)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, Ek, Tk, zk = _diagonalization_at(result, k)
    energies = np.full((len(q), nk * m * m), np.nan)
    weights = np.zeros((len(q), nk * m * m, 3, 3), dtype=complex)
    excluded = np.zeros(len(q), dtype=int)
    norm = 1.0 / np.sqrt(nk * ns)

    def factors(E):
        """Energy change and <phi^+ phi> of each operator: (1 + n) to create, n to annihilate."""
        nb = _occupations(E[:, cols], t)
        return sign * E[:, cols], np.where(sign > 0, 1.0 + nb, nb)

    for iq, qi in enumerate(q):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, Em, Tm, zm = _diagonalization_at(result, -k - qi)     # at -k - q

        def amplitude(T_first, T_second):
            """C^a_{mu nu} = -norm sum_i n_i^a T_first[Ns+i, mu] T_second[i, nu]."""
            return -norm * np.einsum("ia,kin,kim->kanm", n,
                                     T_first[:, ns:, cols], T_second[:, :ns, cols])

        # L^b(q)^+ = sum_k C(k) phi(-k-q) phi(k); the pair (phi_mu(-k-q), phi_nu(k)) also
        # appears at -k-q with the operators swapped, hence the symmetrized amplitude.
        C = amplitude(Tm, Tk)                       # C(k)
        C_swap = amplitude(Tk, Tm)                  # C(-k-q)
        A = C + np.swapaxes(C_swap, 2, 3)           # (nk, 3, mu, nu)
        W = 0.5 * np.einsum("kanm,kbnm->knmab", A.conj(), A)
        em, fm = factors(Em)
        ek, fk = factors(Ek)
        W = W * (fm[:, :, None] * fk[:, None, :])[..., None, None]
        pair = em[:, :, None] + ek[:, None, :]
        bad = zk | zm | ~np.all(np.isfinite(W.reshape(nk, -1)), axis=1)
        W[bad] = 0.0
        pair = np.where(bad[:, None, None], np.nan, pair)
        excluded[iq] = int(np.sum(bad))
        energies[iq] = pair.reshape(-1)
        weights[iq] = W.reshape(nk * m * m, 3, 3)
    if np.any(excluded):
        warnings.warn(f"{int(excluded.sum())} mesh points with a zero mode at k or -k - q "
                      f"were left out of the two-magnon sum", UserWarning, stacklevel=2)
    weights = np.where(np.isnan(energies)[:, :, None, None], 0.0, weights)
    energies = np.where(np.isnan(energies), 0.0, energies)
    if undefined:
        weights = np.full(weights.shape, np.nan + 0j)
    return TwoMagnonContinuum(q, energies, weights, k, excluded, settings)


def two_magnon_threshold(result, k_points, grid: Sequence[int] = (48, 48)) -> np.ndarray:
    """Lower edge of the two-magnon continuum, ``min_{q, n1, n2} w_n1(q) + w_n2(k - q)`` (D57).

    A magnon ``(k, n)`` can decay into two magnons only where
    ``w_n(k)`` lies above this edge. The minimum is taken on an
    ``N1 x N2`` grid of the magnetic zone (half-step offset) and refined
    from the four lowest grid points by a Nelder-Mead search.

    The condition is kinematic and necessary only: momentum is conserved
    modulo the magnetic reciprocal lattice, so a channel that a hidden
    symmetry forbids (e.g. a translation combined with a spin rotation in a
    canted antiferromagnet, which conserves the unfolded momentum) still
    counts here. Whether a magnon actually decays is decided by the vertex:
    :meth:`~spintoolkit.methods.nlswt.NLSWTResult.decay_rates`.

    Parameters
    ----------
    result : LSWTResult
        Lab-frame result.
    k_points : (m, 2) array_like
        Cartesian momenta.
    grid : (n1, n2)

    Returns
    -------
    (m,) array
    """
    from scipy.optimize import minimize

    from spintoolkit.methods.lswt.run import require_lab_frame

    require_lab_frame(result, "two_magnon_threshold")
    ns = result.num_sites
    k = np.atleast_2d(np.asarray(k_points, dtype=float))
    reciprocal = 2 * np.pi * np.linalg.inv(np.asarray(result.magnetic_lattice)).T
    n1, n2 = (int(n) for n in grid)
    u = np.array([((i + 0.5) / n1, (j + 0.5) / n2) for i in range(n1) for j in range(n2)])
    # w_n(p) is the hole-column eigenvalue at -p (see the module docstring)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, E1, _, _ = _diagonalization_at(result, -u @ reciprocal)

    def lowest(Ea, Eb):
        return np.nanmin(Ea[..., ns:][..., :, None] + Eb[..., ns:][..., None, :], axis=(-2, -1))

    out = np.empty(len(k))
    for i, ki in enumerate(k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, E2, _, _ = _diagonalization_at(result, u @ reciprocal - ki)
            edge = lowest(E1, E2)

            def total(p):
                q = np.asarray(p) @ reciprocal
                _, Ea, _, _ = _diagonalization_at(result, -q)
                _, Eb, _, _ = _diagonalization_at(result, q - ki)
                value = lowest(Ea[0], Eb[0])
                return float(value) if np.isfinite(value) else np.inf

            best = float(np.nanmin(edge))
            for start in np.argsort(np.where(np.isfinite(edge), edge, np.inf))[:4]:
                fit = minimize(total, u[start], method="Nelder-Mead",
                               options={"xatol": 1e-9, "fatol": 1e-12})
                best = min(best, float(fit.fun))
        out[i] = best
    return out
