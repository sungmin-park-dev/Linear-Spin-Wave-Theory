"""Longitudinal two-magnon continuum of the spin structure factor (O(S^0), D56).

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
the non-interacting two-magnon continuum. Only zero temperature is
implemented (at t > 0 the thermal scattering terms ``b^dagger b`` also
contribute); a lab-frame (commensurate) result is required.
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
        Pair energies ``w_n(k + q) + w_{n'}(-k)`` over the mesh and the band pairs.
    weights : (nq, M, 3, 3) complex array
        ``1/2 A^a* A^b`` per pair, per site, so that ``weights.sum(axis=1)``
        is the equal-time connected longitudinal correlation.
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


def two_magnon_continuum(result, q_points, mesh: Sequence[int] = (24, 24),
                         temperature: float = 0.0) -> TwoMagnonContinuum:
    """Non-interacting longitudinal two-magnon continuum at momenta ``q_points`` (D56).

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
        Only 0 is implemented.

    Returns
    -------
    TwoMagnonContinuum

    Notes
    -----
    See the module docstring for the formula and its scope: O(S^0), without
    the interaction corrections of the same order. Mesh points where ``k`` or
    ``-k - q`` has a zero mode (e.g. a Goldstone mode at a Bragg vector) are
    left out and counted in ``excluded``; their measure vanishes as the mesh
    is refined, and the weights stay integrable there.
    """
    from spintoolkit.methods.lswt.run import require_lab_frame

    require_lab_frame(result, "two_magnon_continuum")
    if float(temperature) != 0.0:
        raise NotImplementedError("the two-magnon continuum is implemented at zero temperature "
                                  "only; at t > 0 thermal b^dagger b scattering terms contribute")
    q = np.atleast_2d(np.asarray(q_points, dtype=float))
    ns = result.num_sites
    n = np.asarray(result.directions, dtype=float)                 # (Ns, 3) ordered axes
    k = _mesh(result, mesh)
    nk = len(k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, Ek, Tk, zk = _diagonalization_at(result, k)
    energies = np.full((len(q), nk * ns * ns), np.nan)
    weights = np.zeros((len(q), nk * ns * ns, 3, 3), dtype=complex)
    excluded = np.zeros(len(q), dtype=int)
    norm = 1.0 / np.sqrt(nk * ns)
    for iq, qi in enumerate(q):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, Em, Tm, zm = _diagonalization_at(result, -k - qi)     # at -k - q

        def amplitude(T_first, T_second):
            """C^a_{n n'} = -norm sum_i n_i^a T_first[Ns+i, Ns+n] T_second[i, Ns+n']."""
            return -norm * np.einsum("ia,kin,kim->kanm", n,
                                     T_first[:, ns:, ns:], T_second[:, :ns, ns:])

        # phi_{Ns+n}(-k-q) = b_n^dagger(k+q) and phi_{Ns+n'}(k) = b_n'^dagger(-k); the
        # energy of each is the hole-column eigenvalue at that momentum.
        C = amplitude(Tm, Tk)                       # C(k)
        C_swap = amplitude(Tk, Tm)                  # C(-k-q)
        A = C + np.swapaxes(C_swap, 2, 3)           # (nk, 3, n, n')
        W = 0.5 * np.einsum("kanm,kbnm->knmab", A.conj(), A)
        pair = Em[:, ns:][:, :, None] + Ek[:, ns:][:, None, :]
        bad = zk | zm | ~np.all(np.isfinite(W.reshape(nk, -1)), axis=1)
        W[bad] = 0.0
        pair = np.where(bad[:, None, None], np.nan, pair)
        excluded[iq] = int(np.sum(bad))
        energies[iq] = pair.reshape(-1)
        weights[iq] = W.reshape(nk * ns * ns, 3, 3)
    if np.any(excluded):
        warnings.warn(f"{int(excluded.sum())} mesh points with a zero mode at k or -k - q "
                      f"were left out of the two-magnon sum", UserWarning, stacklevel=2)
    weights = np.where(np.isnan(energies)[:, :, None, None], 0.0, weights)
    energies = np.where(np.isnan(energies), 0.0, energies)
    return TwoMagnonContinuum(q, energies, weights, k, excluded,
                              {"mesh": [int(m) for m in mesh], "temperature": 0.0})
