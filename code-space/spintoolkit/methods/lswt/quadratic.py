"""Quadratic boson Hamiltonian built directly from a ``SpinModel`` (D36).

The bosonic ``H(k)`` of linear spin-wave theory about a classical state on its
magnetic supercell, without the conversion to the former ``SpinSystem``. The
conventions are those of :class:`~spintoolkit.methods.lswt.hamiltonian.LSWTHamiltonian`
(which this module reproduces to round-off for the bilinear and Zeeman kinds):

- Nambu order ``(a_1 .. a_Ns, a_1^dagger .. a_Ns^dagger)`` over the magnetic
  sites ``(site_id, cell)`` in model-site order, then supercell-cell order;
- local frame ``R_i`` of each spin from its polar and azimuthal angles, columns
  the local x, y, z axes, z along the spin, and
  ``delta S_i = sqrt(S_i/2) [(x_i - i y_i) a_i + (x_i + i y_i) a_i^dagger]``;
- Fourier convention D13 with full site positions: a bond from ``i`` to ``j``
  carries ``exp(-i k . (r_i - r_j))`` on the ``a_i^dagger a_j`` element.

``H(k) = H_0 + sum_b [P_b exp(-i k . d_b) + Q_b exp(+i k . d_b)]`` is stored as
the constant matrices, so the matrix and its analytic k-derivatives at any
momentum cost one contraction.

Onsite terms (D37) enter as a bond from a spin to itself with the coefficient
``(1 - 1/(2S)) A`` and no phase. With that factor the classical energy is the
spin-coherent-state value, the linear term equals the exact one,
``2 (S - 1/2) sqrt(S/2) A_xz`` in the local frame, and the single-ion gap of
``D (S^z)^2`` is the exact ``(2S - 1)|D|``. The normal-ordering constant of
the transverse product ``(s^x)^2 = (S/2)(2 a^dagger a + 1 + ...)`` is not added
to the energy: the constant ``(S/2) tr A`` of the coherent-state value already
contains it (adding it again makes the error of the ground-state energy grow
like ``S``; checked against exact single-spin spectra up to S = 20).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import warnings

import numpy as np

from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import BILINEAR, ONSITE, ZEEMAN, SpinModel, onsite_renormalization

#: ``C`` maps the (+, -, z) ladder basis to Cartesian components (local frame).
_LADDER = np.array([[1, 1, 0], [1j, -1j, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)


def local_frame(direction) -> np.ndarray:
    """Local frame ``R`` of a spin along ``direction``: columns are x, y, z (z along the spin).

    Same gauge as the existing Hamiltonian: ``R = R_z(phi) R_y(theta)`` with the
    polar angle ``theta`` and azimuth ``phi`` (``phi = 0`` on the poles).
    """
    n = np.asarray(direction, dtype=float)
    n = n / np.linalg.norm(n)
    theta = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
    phi = float(np.arctan2(n[1], n[0])) if np.hypot(n[0], n[1]) > 0 else 0.0
    ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
    return np.array([[ct * cp, -sp, st * cp],
                     [ct * sp, cp, st * sp],
                     [-st, 0.0, ct]])


class QuadraticBoseHamiltonian:
    """``H(k)`` of LSWT about ``state`` (all values in E0, D20).

    Parameters
    ----------
    model : SpinModel
    state : SpinState
        Reference classical state; must belong to ``model``.
    conditions : ExternalConditions, optional
        Dimensionless field (default zero).

    Attributes
    ----------
    site_keys : tuple of (site_id, cell)
        Magnetic sites in Nambu order.
    spins : (Ns,) array
    directions : (Ns, 3) array
    positions : (Ns, 2) array
        Cartesian positions (D13 full-position gauge).
    local_frames : (Ns, 3, 3) array
    linear_terms : (Ns,) complex array
        Coefficient of ``a_i`` in the linear boson term (up to ``sqrt(S_i/2)``
        factors, same normalization as the existing Hamiltonian); zero at a
        stationary state.
    """

    def __init__(self, model: SpinModel, state: SpinState,
                 conditions: Optional[ExternalConditions] = None):
        conditions = conditions or ExternalConditions()
        self.site_keys: Tuple = tuple((site.id, cell) for site in model.sites for cell in state.cells)
        index: Dict = {key: i for i, key in enumerate(self.site_keys)}
        ns = len(self.site_keys)
        self.num_sites = ns
        self.spins = np.array([model.site(s).spin for s, _ in self.site_keys])
        self.directions = np.array([state.direction(s, c) for s, c in self.site_keys])
        self.positions = np.array([model.cartesian_position(s, c) for s, c in self.site_keys])
        self.local_frames = np.array([local_frame(n) for n in self.directions])

        constant = np.zeros((2 * ns, 2 * ns), dtype=complex)
        linear = np.zeros(ns, dtype=complex)
        field = conditions.field
        if np.any(field != 0):
            uncoupled = sorted(set(model.site_ids)
                               - {t.participants[0][0] for t in model.terms_of_kind(ZEEMAN)})
            if uncoupled:
                warnings.warn(f"sites {uncoupled} have no zeeman term and do not couple "
                              "to the applied field", UserWarning, stacklevel=3)
        for term in model.terms_of_kind(ZEEMAN):
            site = term.participants[0][0]
            h = term.coefficient.T @ field                      # -b^T g S = -h . S
            for cell in state.cells:
                i = index[(site, cell)]
                hx, hy, hz = h @ self.local_frames[i]
                constant[i, i] += hz
                constant[ns + i, ns + i] += hz
                linear[i] -= hx + 1j * hy

        def bond(i, j, J):
            """Phase coefficients ``(p, q)`` of ``S_i^T J S_j``; updates the constant and linear parts."""
            Si, Sj = self.spins[i], self.spins[j]
            hop = _LADDER.conj().T @ (self.local_frames[i].T @ J @ self.local_frames[j]) @ _LADDER
            tpm = np.sqrt(Si * Sj) * hop[1, 1]
            tpp = np.sqrt(Si * Sj) * hop[1, 0]
            t00 = hop[2, 2].real
            for k in (i, ns + i):
                constant[k, k] -= t00 * Sj
            for k in (j, ns + j):
                constant[k, k] -= t00 * Si
            p = np.zeros((2 * ns, 2 * ns), dtype=complex)   # coefficient of exp(-i k.d)
            q = np.zeros((2 * ns, 2 * ns), dtype=complex)   # coefficient of exp(+i k.d)
            p[i, j] += tpm
            q[j, i] += np.conj(tpm)
            p[ns + i, ns + j] += np.conj(tpm)
            q[ns + j, ns + i] += tpm
            p[i, ns + j] += tpp
            q[j, ns + i] += tpp
            q[ns + j, i] += np.conj(tpp)
            p[ns + i, j] += np.conj(tpp)
            linear[i] += np.sqrt(2) * Sj * hop[1, 2]
            linear[j] += np.sqrt(2) * Si * hop[2, 0]
            return p, q

        P, Q, deltas = [], [], []
        for term in model.terms_of_kind(BILINEAR):
            (a, n1), (b, n2) = term.participants
            for cell in state.cells:
                source = (cell[0] + n1[0], cell[1] + n1[1])
                target = (cell[0] + n2[0], cell[1] + n2[1])
                p, q = bond(index[(a, state.reduce_cell(source))],
                            index[(b, state.reduce_cell(target))], term.coefficient)
                P.append(p)
                Q.append(q)
                deltas.append(model.cartesian_position(a, source) - model.cartesian_position(b, target))
        for term in model.terms_of_kind(ONSITE):
            site = term.participants[0][0]
            kappa = onsite_renormalization(model.site(site).spin)
            for cell in state.cells:
                i = index[(site, cell)]
                p, q = bond(i, i, kappa * term.coefficient)
                constant += p + q

        self._constant = constant
        self._P = np.array(P).reshape(-1, 2 * ns, 2 * ns)
        self._Q = np.array(Q).reshape(-1, 2 * ns, 2 * ns)
        self._deltas = np.array(deltas, dtype=float).reshape(-1, 2)
        self.linear_terms = linear

    def _phases(self, momenta) -> Tuple[np.ndarray, np.ndarray]:
        k = np.atleast_2d(np.asarray(momenta, dtype=float))
        minus = np.exp(-1j * k @ self._deltas.T)                # (nk, nb)
        return k, minus

    def at(self, momenta) -> np.ndarray:
        """``H(k)`` for momenta ``(m, 2)``; returns ``(m, 2Ns, 2Ns)``."""
        k, minus = self._phases(momenta)
        H = np.broadcast_to(self._constant, (len(k),) + self._constant.shape).copy()
        if len(self._deltas):
            H += np.einsum("kb,bij->kij", minus, self._P) + np.einsum("kb,bij->kij", minus.conj(), self._Q)
        return H

    def derivatives_at(self, momenta) -> Tuple[np.ndarray, np.ndarray]:
        """Analytic ``(dH/dk_x, dH/dk_y)`` at momenta ``(m, 2)``."""
        k, minus = self._phases(momenta)
        out = []
        for axis in (0, 1):
            factor = -1j * self._deltas[:, axis]
            d = (np.einsum("kb,bij->kij", minus * factor, self._P)
                 + np.einsum("kb,bij->kij", (minus * factor).conj(), self._Q))
            out.append(d if len(self._deltas) else np.zeros((len(k),) + self._constant.shape, complex))
        return out[0], out[1]
