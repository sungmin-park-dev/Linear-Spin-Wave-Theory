"""Dynamical structure factor through order S^0 at zero temperature (D59).

The spin operator at momentum ``q`` (per-site Fourier convention of
:mod:`spintoolkit.observables.structure_factor`) is expanded with the same
Holstein-Primakoff truncation as the Hamiltonian
(:mod:`spintoolkit.methods.nlswt.expansion`) about the condensate of D46::

    S^a(q) = <S^a> delta_{q,G} + sum_m (A + dA)_{am} psi_m(q) + L^a(q) + ...

- ``A psi``: the LSWT one-magnon operator, order ``S^{1/2}``;
- ``dA psi``: its order ``S^{-1/2}`` corrections, from the Hartree-Fock
  contraction of the cubic terms ``-a^+ a a / (4S)`` of ``s^+`` (on-site
  ``<a^+ a>`` and ``<a a>``) and from the condensate ``<a>`` in the
  longitudinal term ``-n a^+ a`` (the 1/S rotation of the ordered moment);
- ``L^a(q) = -N^{-1/2} sum_i n_i^a a_i^+ a_i e^{-i q r_i}``: the two-magnon
  operator, order ``S^0`` (D56).

With ``psi = T phi`` (Bogoliubov, Colpa order), ``a = A T``, ``da = dA T``,
``g0(z) = diag(1/(z - w_n), -1/(z + w_h))`` and the retarded functions
``<<O1; O2^+>>_z``, the correlation ``chi^{ab}(z) = <<S^a(q); S^b(q)^+>>_z``
through order ``S^0`` is (exact Rayleigh-Schroedinger perturbation theory on
a finite torus)::

    chi = a g0 a^+ + da g0 a^+ + a g0 da^+ + a g0 M(z) g0 a^+
          + a g0 k(z) + kt(z) g0 a^+ + chi_LL(z),

    M  = T^+ (dH + sigma_3 Pi sigma_3) T,  k = T^+ sigma_3 K,  kt = Kt sigma_3 T,

where ``dH`` is the Hartree-Fock + tadpole matrix of D46 and ``Pi, K, Kt,
chi_LL`` are the blocks of one two-magnon bubble of the operators
``O = (j, L)``, ``j = [psi, H_3]``::

    B_{ab}(z) = 1/2 <sum_{n1 n2} [c_a c_b^* / (z - E) - d_a d_b^* / (z + E)]>_{q1},

``c`` the amplitudes to annihilate the pair ``(q1 n1, q - q1 n2)`` and ``d``
to create ``(q1 n1, -q - q1 n2)`` (:meth:`NonlinearSpinWaves.bilinear_pair_amplitudes`).
The equation of motion ``<<psi; L^+>> = G0 sigma_3 <<j; L^+>>`` gives the
cross terms. ``M`` contains the cubic self-energy of
:meth:`NonlinearSpinWaves.cubic_self_energy_matrix`; ``-(1/pi) Im chi`` is
``S^{ab}(q, w)`` for ``w > 0``.

Final states (finite torus, :func:`structure_factor_states`):

- one magnon (band group ``g`` of energy ``E``): pole at ``E + d_j``
  (``d_j`` the eigenvalues of ``M_gg(E)``, projectors ``P_j``) with residue
  ``R_j = a_g P_j a_g^+ + 1/2 a_g {P_j, M'_gg} a_g^+ + da_g P_j a_g^+ + h.c.
  + a_g P_j [sum_{v not in g} M_gv g0_v(E) a_v^+ + k_g(E)] + h.c.``,
  i.e. the LSWT weight times the quasiparticle residue, the eigenvector
  rotation by the off-diagonal self-energy (interband and anomalous), the
  operator corrections and the interference with the two-magnon operator;
- two magnons ``P``: weight ``1/2 |c_L(P) + a g0(E_P) T^+ sigma_3 X(P)|^2``,
  the longitudinal amplitude of D56 plus the transverse operator acting
  through a virtual one-magnon state and the cubic vertex
  (one-/two-magnon interference and the off-shell magnon tail).

Thermodynamic limit (:func:`structure_factor_s0`): the bubble is the Cauchy
transform of matrix two-magnon densities on a triangle grid (D57, D58), and
the resonant one-magnon poles are separated from the rest: the strict
expression minus its pole parts at ``E`` (finite), plus each pole resummed,
``R_j / (z - E - d_j)``. Where the magnon lies in the continuum, ``d_j`` has
the imaginary part ``-Gamma`` of D57 and ``R_j`` becomes complex (Fano
asymmetry); everywhere else the result is the strict order-S^0 spectrum.

Decaying magnons: inside the continuum ``M'(E + i0)`` follows the van Hove
structure of the two-magnon density, which is sharper than the line, and the
strict residue does not converge with the grid. The reported weight ``W_j``
and the one-magnon line therefore take ``M'``, the off-diagonal ``M`` and
``k, kt`` at ``E + i Gamma`` (the self-energy averaged over the line width),
while ``M_gg(E + i0)`` still gives the position and width; the difference is
of order ``Gamma d^2 Pi``, beyond a 1/S calculation. The total stays the
strict expression and the continuum is the total minus the line, so within a
few ``Gamma`` of a decaying magnon the split is a convention.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spintoolkit.methods.lswt.run import _diagonalize
from spintoolkit.methods.nlswt.engine import cauchy_transform, triangle_density
from spintoolkit.methods.nlswt.expansion import local_spin_polynomials

_KEYS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))


def spin_operator(solver, q) -> Tuple[np.ndarray, np.ndarray, list]:
    """``A``, ``dA`` (3, 2Ns) and the terms of ``L^a(q)`` (see the module docstring).

    The terms are ``(a, f, y, z)`` for :meth:`NonlinearSpinWaves.bilinear_pair_amplitudes`.
    """
    expansion = solver.expansion
    ns = solver.ns
    q = np.asarray(q, dtype=float).reshape(2)
    shift = solver.tadpole()[0]
    contract = solver.contract
    norm = 1.0 / np.sqrt(ns)
    A = np.zeros((3, 2 * ns), dtype=complex)
    dA = np.zeros((3, 2 * ns), dtype=complex)
    terms = []
    for s in range(ns):
        frame = expansion.local_frames[s]
        polys = local_spin_polynomials(float(expansion.spins[s]))
        P = {key: frame @ np.array([p.get(key, 0.0) for p in polys]) for key in _KEYS}
        pos = (float(expansion.positions[s][0]), float(expansion.positions[s][1]))
        lower, raise_ = (s, False, pos), (s, True, pos)
        n = contract(raise_, lower)                  # <a^+ a>
        pair = contract(lower, lower)                # <a a>
        pair_c = contract(raise_, raise_)            # <a^+ a^+>
        A[:, s] = norm * P[(0, 1)]
        A[:, ns + s] = norm * P[(1, 0)]
        # a^+ a a -> 2 n a + <a a> a^+ ;  a^+ a^+ a -> 2 n a^+ + <a^+ a^+> a ;
        # a^+ a with a = <a> + a~ -> <a^+> a~ + <a> a~^+
        dA[:, s] = norm * (P[(1, 1)] * shift[ns + s] + 2 * n * P[(1, 2)] + pair_c * P[(2, 1)])
        dA[:, ns + s] = norm * (P[(1, 1)] * shift[s] + pair * P[(1, 2)] + 2 * n * P[(2, 1)])
        phase = np.exp(-1j * q @ np.asarray(pos))
        for a in range(3):
            if P[(1, 1)][a] != 0:
                terms.append((a, norm * phase * P[(1, 1)][a], raise_, lower))
    return A, dA, terms


def _diagonal(solver, q):
    """Full Colpa energies (2Ns,) and paraunitary T at one momentum."""
    k = np.asarray(q, dtype=float).reshape(1, 2)
    H = np.asarray(solver.hamiltonian_at(k), dtype=complex)
    _, E, T, _ = _diagonalize(H, "none", k, 1e-12)
    return E[0].real, T[0]


def pair_vectors(solver, q, q1: np.ndarray):
    """Amplitudes ``c`` and ``d`` of ``O = (j_1 .. j_2Ns, L^x, L^y, L^z)`` and pair energies.

    Returns
    -------
    c, Ec, d, Ed
        ``c`` (m, 2Ns + 3, Ns, Ns) annihilates ``(q1 n1, q - q1 n2)`` of
        energy ``Ec`` (m, Ns, Ns); ``d`` creates ``(q1 n1, -q - q1 n2)`` of
        energy ``Ed``.
    """
    ns = solver.ns
    q = np.asarray(q, dtype=float).reshape(1, 2)
    E1, T1 = solver._bogoliubov(q1)
    Ex, Tx = solver._bogoliubov(q - q1)
    Ey, Ty = solver._bogoliubov(-q - q1)
    X, Y = solver.bilinear_pair_amplitudes(solver._commutator_terms(q[0]), 2 * ns,
                                           q1, T1, q - q1, Tx, -q - q1, Ty)
    _, _, terms = spin_operator(solver, q[0])
    CL, DL = solver.bilinear_pair_amplitudes(terms, 3, q1, T1, q - q1, Tx, -q - q1, Ty)
    return (np.concatenate([X, CL], axis=1), E1[:, :, None] + Ex[:, None, :],
            np.concatenate([Y, DL], axis=1), E1[:, :, None] + Ey[:, None, :])


class _Bubble:
    """``B(z)`` and ``B'(E)`` from matrix densities on a triangle grid or from a discrete pair sum."""

    def __init__(self, c, Ec, d, Ed, grid: Optional[Tuple[int, int]] = None, levels: int = 400):
        self.discrete = grid is None
        m, D = c.shape[:2]
        self.D = D
        if self.discrete:
            self.c = c.transpose(0, 2, 3, 1).reshape(-1, D)
            self.Ec = Ec.reshape(-1)
            self.d = d.transpose(0, 2, 3, 1).reshape(-1, D)
            self.Ed = Ed.reshape(-1)
            self.m = m
            return
        n1, n2 = grid
        ns = c.shape[2]
        top = 1.02 * max(float(np.max(Ec)), float(np.max(Ed)))
        self.x = np.linspace(0.0, top, int(levels))
        self.rho_c = np.zeros((len(self.x), D * D), dtype=complex)
        self.rho_d = np.zeros((len(self.x), D * D), dtype=complex)
        for i in range(ns):
            for j in range(ns):
                for vec, energy, rho in ((c, Ec, self.rho_c), (d, Ed, self.rho_d)):
                    v = vec[:, :, i, j]
                    if not np.any(v):
                        continue
                    g = 0.5 * (v[:, :, None] * v[:, None, :].conj()).reshape(n1, n2, D * D)
                    rho += triangle_density(energy[:, i, j].reshape(n1, n2), g, self.x)

    def __call__(self, z) -> np.ndarray:
        """``B(z)``, (nz, D, D); real ``z`` means ``z + i0``."""
        z = np.atleast_1d(np.asarray(z))
        D = self.D
        if self.discrete:
            pc = np.einsum("pa,pb->pab", self.c, self.c.conj())
            pd = np.einsum("pa,pb->pab", self.d, self.d.conj())
            out = (np.einsum("zp,pab->zab", 1.0 / (z[:, None] - self.Ec[None, :]), pc)
                   - np.einsum("zp,pab->zab", 1.0 / (z[:, None] + self.Ed[None, :]), pd))
            return 0.5 * out / self.m
        out = cauchy_transform(self.x, self.rho_c, z) + cauchy_transform(self.x, self.rho_d, -z)
        return out.reshape(len(z), D, D)

    def derivative(self, z) -> np.ndarray:
        """``B'(z)`` at real ``z + i0`` (or complex z), (nz, D, D)."""
        z = np.atleast_1d(np.asarray(z))
        D = self.D
        if self.discrete:
            pc = np.einsum("pa,pb->pab", self.c, self.c.conj())
            pd = np.einsum("pa,pb->pab", self.d, self.d.conj())
            out = (-np.einsum("zp,pab->zab", 1.0 / (z[:, None] - self.Ec[None, :]) ** 2, pc)
                   + np.einsum("zp,pab->zab", 1.0 / (z[:, None] + self.Ed[None, :]) ** 2, pd))
            return 0.5 * out / self.m
        out = (cauchy_transform(self.x, self.rho_c, z, derivative=True)
               - cauchy_transform(self.x, self.rho_d, -z, derivative=True))
        return out.reshape(len(z), D, D)


def _groups(energies: np.ndarray, tolerance: float = 1e-8) -> List[List[int]]:
    order = np.argsort(energies)
    scale = max(float(np.max(np.abs(energies))), 1.0)
    groups, current = [], [int(order[0])]
    for a, b in zip(order[:-1], order[1:]):
        if energies[b] - energies[a] <= tolerance * scale:
            current.append(int(b))
        else:
            groups.append(current)
            current = [int(b)]
    groups.append(current)
    return groups


class OrderS0Correlation:
    """``chi^{ab}(q, z)`` through order S^0 at one momentum (see the module docstring)."""

    def __init__(self, solver, q, bubble: _Bubble):
        ns = solver.ns
        self.ns = ns
        q = np.asarray(q, dtype=float).reshape(2)
        E, T = _diagonal(solver, q)
        self.E, self.T = E, T
        A, dA, _ = spin_operator(solver, q)
        self.a = A @ T
        self.da = dA @ T
        self.dH = np.asarray(solver.static_hamiltonian(q.reshape(1, 2)))[0]
        self.bubble = bubble
        self.s3 = np.r_[np.ones(ns), -np.ones(ns)]

    # ---- blocks ------------------------------------------------------------------
    def _blocks(self, B, derivative: bool = False):
        """``M``, ``k``, ``kt`` (Bogoliubov basis) and ``chi_LL`` from bubble values (nz, D, D)."""
        n2 = 2 * self.ns
        s3, T = self.s3, self.T
        Pi = B[:, :n2, :n2]
        K = B[:, :n2, n2:]
        Kt = B[:, n2:, :n2]
        LL = B[:, n2:, n2:]
        sigma = s3[None, :, None] * Pi * s3[None, None, :]
        if not derivative:
            sigma = sigma + self.dH[None]
        M = np.einsum("ma,zab,bn->zmn", T.conj().T, sigma, T)
        k = np.einsum("ma,a,zab->zmb", T.conj().T, s3, K)
        kt = np.einsum("zab,b,bn->zan", Kt, s3, T)
        return M, k, kt, LL

    def g0(self, z) -> np.ndarray:
        ns = self.ns
        z = np.atleast_1d(np.asarray(z))[:, None]
        return np.concatenate([1.0 / (z - self.E[None, :ns]), -1.0 / (z + self.E[None, ns:])], axis=1)

    def strict(self, z) -> np.ndarray:
        """Strict order-S^0 ``chi(z)``, (nz, 3, 3)."""
        z = np.atleast_1d(np.asarray(z))
        a, da = self.a, self.da
        g = self.g0(z)
        M, k, kt, LL = self._blocks(self.bubble(z))
        out = np.einsum("zm,am,bm->zab", g, a, a.conj())
        out += np.einsum("zm,am,bm->zab", g, da, a.conj())
        out += np.einsum("zm,am,bm->zab", g, a, da.conj())
        out += np.einsum("zm,am,zmn,zn,bn->zab", g, a, M, g, a.conj())
        out += np.einsum("zm,am,zmb->zab", g, a, k)
        out += np.einsum("zam,zm,bm->zab", kt, g, a.conj())
        return out + LL

    def poles(self):
        """One-magnon poles: list of ``(group, E, d_j, R_j, M_gg, W_j)``.

        ``R_j`` (3, 3) are the strict order-S^0 residues with the self-energy
        derivative at ``E + i0``; they define the pole parts subtracted from
        the continuum. ``W_j`` are the reported weights: equal to ``R_j`` for a
        stable magnon, and for a decaying one (``Gamma = -Im d > 0``) the same
        expression with the off-diagonal and derivative blocks at
        ``E + i Gamma``, i.e. averaged over the line width (module docstring).
        """
        ns = self.ns
        out = []
        for group in _groups(self.E[:ns]):
            E0 = float(np.mean(self.E[group]))
            g = np.array(group)
            M, k, kt, Mp = self._residue_blocks(E0)
            d, V = np.linalg.eig(M[np.ix_(g, g)])
            R = self._residues(g, E0, d, V, M, k, kt, Mp)
            gamma = float(np.max(-d.imag))
            if gamma > 1e-9 * max(1.0, abs(E0)):
                Ms, ks, kts, Mps = self._residue_blocks(E0 + 1j * gamma)
                W = self._residues(g, E0, d, V, Ms, ks, kts, Mps)
            else:
                W = R
            out.append((group, E0, d, R, M[np.ix_(g, g)], W))
        return out

    def _residue_blocks(self, z):
        M, k, kt, _ = self._blocks(self.bubble(np.array([z])))
        Mp, _, _, _ = self._blocks(self.bubble.derivative(np.array([z])), derivative=True)
        return M[0], k[0], kt[0], Mp[0]

    def _residues(self, g, E0, d, V, M, k, kt, Mp):
        a, da = self.a, self.da
        others = np.setdiff1d(np.arange(2 * self.ns), g)
        with np.errstate(divide="ignore"):
            g0 = self.g0(np.array([E0]))[0][others]
        right = (M[np.ix_(g, others)] * g0[None, :]) @ a[:, others].conj().T     # (|g|, 3)
        left = (a[:, others] * g0[None, :]) @ M[np.ix_(others, g)]               # (3, |g|)
        Vinv = np.linalg.inv(V)
        ag, dag = a[:, g], da[:, g]
        Mpg = Mp[np.ix_(g, g)]
        residues = []
        for j in range(len(g)):
            P = np.outer(V[:, j], Vinv[j])
            R = ag @ P @ ag.conj().T
            R = R + 0.5 * ag @ (P @ Mpg + Mpg @ P) @ ag.conj().T
            R = R + dag @ P @ ag.conj().T + ag @ P @ dag.conj().T
            R = R + ag @ P @ right + left @ P @ ag.conj().T
            R = R + ag @ P @ k[g] + kt[:, g] @ P @ ag.conj().T
            residues.append(R)
        return np.array(residues)

    def polar(self, z, poles) -> np.ndarray:
        """Strict pole parts of ``chi`` at the one-magnon energies, (nz, 3, 3)."""
        z = np.atleast_1d(np.asarray(z))
        out = np.zeros((len(z), 3, 3), dtype=complex)
        for group, E0, d, R, Mgg, _ in poles:
            ag = self.a[:, group]
            dz = (z - E0)[:, None, None]
            out += R.sum(axis=0)[None] / dz + (ag @ Mgg @ ag.conj().T)[None] / dz ** 2
        return out

    def continuum_density(self, x, poles) -> np.ndarray:
        """Two-magnon part ``-(1/pi) Im [chi - poles](x + i0)`` at real ``x > 0``, (nx, 3, 3).

        The strict ``chi`` minus its one-magnon pole parts is finite at the
        magnon energies; at zero broadening and positive frequency only the
        two-magnon states contribute, so the negative-frequency (hole)
        structure of the retarded function does not enter.
        """
        x = np.asarray(x, dtype=float)
        return spectral(self.strict(x) - self.polar(x, poles))

    @staticmethod
    def pole_part(z, poles, weights: bool = False) -> np.ndarray:
        """Resummed poles ``sum_j R_j / (z - E - d_j)`` (``W_j`` if ``weights``), (nz, 3, 3)."""
        z = np.atleast_1d(np.asarray(z))
        out = np.zeros((len(z), 3, 3), dtype=complex)
        for group, E0, d, R, _, W in poles:
            R = W if weights else R
            for j in range(len(group)):
                out += R[j][None] / (z - E0 - d[j])[:, None, None]
        return out


def spectral(chi: np.ndarray) -> np.ndarray:
    """``S^{ab}(w) = -(1/pi) (chi - chi^+) / 2i`` (Hermitian in a, b)."""
    return -(chi - np.conj(np.swapaxes(chi, -1, -2))) / (2j * np.pi)


def project(S: np.ndarray, q: np.ndarray, component: Optional[str]) -> np.ndarray:
    """Component of ``S^{ab}`` (..., 3, 3) as in :meth:`StructureFactor.spectrum`."""
    if component is None:
        return np.real(np.einsum("...aa->...", S))
    if component == "neutron":
        norm = np.linalg.norm(q)
        if norm == 0:
            return np.full(S.shape[:-2], np.nan)
        u = np.array([q[0], q[1], 0.0]) / norm
        return np.real(np.einsum("ab,...ab->...", np.eye(3) - np.outer(u, u), S))
    a, b = ("xyz".index(component[0]), "xyz".index(component[1]))
    return np.real(S[..., a, b])


def grid_momenta(solver, grid: Sequence[int]) -> np.ndarray:
    n1, n2 = (int(n) for n in grid)
    u = np.array([((i + 1 / 3) / n1, (j + 1 / 3) / n2) for i in range(n1) for j in range(n2)])
    return u @ solver.mesh.reciprocal
