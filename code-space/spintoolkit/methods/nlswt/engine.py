"""Interacting spin waves to order S^0: ground-state energy and 1/S magnon energies (D46).

The Holstein-Primakoff Hamiltonian ``H = H_0 + H_2 + H_3 + H_4`` of
:mod:`spintoolkit.methods.nlswt.expansion` is treated in perturbation theory
about the LSWT (Bogoliubov) vacuum of ``H_2``.

Ground-state energy per magnetic cell through order ``S^0``::

    E = H_0 + E_zp + <H_4> + E_cubic + E_tadpole

- ``E_zp``: LSWT zero-point energy (order S);
- ``<H_4>``: Wick (Hartree-Fock) expectation of the quartic terms;
- ``E_cubic = -6 sum |S^s|^2 / (w_1 + w_2 + w_3)``: second order in the
  three-magnon creation part ``H_3 = sum S^s_{abc} b_a^+ b_b^+ b_c^+ + ...``;
- ``E_tadpole = -1/2 zeta^+ H(0)^+ zeta``: the linear (q = 0) terms that the
  Wick contraction of ``H_3`` leaves, ``1/2 (zeta^+ psi + psi^+ zeta)``,
  shift the condensate by ``<psi> = -H(0)^+ zeta``; this is the 1/S
  correction of the classical angles.

Magnon energies through order ``S^0`` (relative 1/S), first order in the
static terms and on-shell second order in ``H_3``::

    w_n(k) + [T^+ (dH_HF + dH_tad) T]_nn + Sigma_3^nn(k, w_n(k)),
    Sigma_3 = sum 2|D^s|^2 / (w - w_1 - w_2 + i eta) - 18 sum |S^s|^2 / (w + w_1 + w_2),

``dH_HF`` the Hartree-Fock decoupling of ``H_4``, ``dH_tad`` the quadratic
part of ``H_3`` with one operator replaced by the condensate shift, and
``H_3 = sum D^s_{ab;c} b_a^+ b_b^+ b_c + ...`` the decay vertex. The sums run
over the same momenta as the vacuum, so on a finite torus every formula is
exact Rayleigh-Schroedinger perturbation theory in 1/S.

Momenta: three-magnon sums need ``q_3 = -q_1 - q_2`` on the mesh, and the
mesh must avoid the zone centre (Goldstone modes). A mesh offset by one third
of a step, ``(n + 1/3)/N``, has both properties; it is the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from spintoolkit.methods.lswt.run import LSWTError, _diagonalize
from spintoolkit.methods.nlswt.expansion import BosonExpansion, Monomial

PERMUTATIONS = tuple(permutations(range(3)))


def nambu_matrices(quadratic: Sequence[Tuple[complex, Monomial]], momenta: np.ndarray,
                   num_sites: int) -> np.ndarray:
    """Nambu matrices ``M(k)`` with ``sum quadratic monomials = 1/2 sum_k psi_k^+ M(k) psi_k + const``.

    Parameters
    ----------
    quadratic : sequence of (coefficient, monomial)
        Two-operator monomials (per magnetic cell; the Hamiltonian is their
        sum over magnetic-lattice translations).
    momenta : (m, 2) array
    num_sites : int

    Returns
    -------
    (m, 2Ns, 2Ns) complex array
        Hermitian when the monomials form a Hermitian operator.
    """
    k = np.atleast_2d(np.asarray(momenta, dtype=float))
    ns = num_sites
    M = np.zeros((len(k), 2 * ns, 2 * ns), dtype=complex)
    for c, ((s1, d1, r1), (s2, d2, r2)) in quadratic:
        delta = np.asarray(r1) - np.asarray(r2)
        minus = c * np.exp(-1j * k @ delta)
        plus = c * np.exp(1j * k @ delta)
        if d1 and not d2:                                   # a1^+ a2
            M[:, s1, s2] += minus
            M[:, ns + s2, ns + s1] += plus
        elif d2 and not d1:                                 # a1 a2^+ = a2^+ a1 (+ const)
            M[:, s2, s1] += plus
            M[:, ns + s1, ns + s2] += minus
        elif not d1:                                        # a1 a2
            M[:, ns + s1, s2] += minus
            M[:, ns + s2, s1] += plus
        else:                                               # a1^+ a2^+
            M[:, s1, ns + s2] += minus
            M[:, s2, ns + s1] += plus
    return M


@dataclass
class BogoliubovMesh:
    """LSWT vacuum on a set of momenta closed under addition (for three-magnon sums).

    Attributes
    ----------
    k : (nk, 2) array
        Cartesian momenta.
    fractional : (nk, 2) array
        Momenta in the magnetic reciprocal basis, in ``[0, 1)``.
    energies : (nk, Ns) array
        Magnon energies ``w_n(k)`` (Colpa order).
    vectors : (nk, 2Ns, 2Ns) array
        Paraunitary ``T_k``: ``psi_k = T_k phi_k``, ``phi_k = (b_k, b_{-k}^+)``.
    """

    k: np.ndarray
    fractional: np.ndarray
    energies: np.ndarray
    vectors: np.ndarray
    reciprocal: np.ndarray
    _lookup: Dict[Tuple[int, int], int] = field(default_factory=dict, repr=False)
    _resolution: int = 0

    def index(self, fractional: np.ndarray) -> np.ndarray:
        """Indices of momenta given in fractional coordinates (mod 1); raises if absent."""
        f = np.mod(np.atleast_2d(fractional), 1.0)
        keys = np.mod(np.rint(f * self._resolution).astype(np.int64), self._resolution)
        try:
            return np.array([self._lookup[(int(a), int(b))] for a, b in keys])
        except KeyError as exc:
            raise LSWTError("the momentum set is not closed under addition; three-magnon "
                            "sums need q1 + q2 + q3 = 0 inside the set (use a mesh offset "
                            "by 1/3 of a step or a finite torus)") from exc


def bogoliubov_mesh(hamiltonian_at, k: np.ndarray, fractional: np.ndarray,
                    reciprocal: np.ndarray, zero_mode_tolerance: float,
                    resolution: int) -> BogoliubovMesh:
    """Diagonalize ``H(k)`` on the momenta (no regularization: zero modes raise)."""
    H = np.asarray(hamiltonian_at(k), dtype=complex)
    _, E, T, _ = _diagonalize(H, "none", k, zero_mode_tolerance)
    ns = H.shape[1] // 2
    f = np.mod(fractional, 1.0)
    keys = np.mod(np.rint(f * resolution).astype(np.int64), resolution)
    lookup = {(int(a), int(b)): i for i, (a, b) in enumerate(keys)}
    if len(lookup) != len(k):
        raise LSWTError("momentum set has duplicate points at the chosen resolution")
    return BogoliubovMesh(k, f, E[:, :ns].real, T, reciprocal, lookup, resolution)


class Contractions:
    """Two-point functions of the LSWT vacuum in real space, ``<O_1 O_2>``.

    ``O = (site, dagger, position)``; computed as momentum averages over the
    mesh (exact on a finite torus).
    """

    def __init__(self, mesh: BogoliubovMesh):
        self.mesh = mesh
        ns = mesh.vectors.shape[1] // 2
        self.ns = ns
        P = np.diag(np.r_[np.ones(ns), np.zeros(ns)])
        W = np.einsum("kab,bc,kdc->kad", mesh.vectors, P, mesh.vectors.conj())
        self._n = np.transpose(W[:, :ns, :ns], (0, 2, 1)) - np.eye(ns)   # n_st(k) = <a_sk^+ a_tk>
        self._m = W[:, :ns, ns:]                                           # m_st(k) = <a_sk a_t,-k>
        self._cache: Dict = {}

    def _avg(self, table, s, t, displacement, sign):
        key = (id(table), s, t, sign, round(float(displacement[0]), 10),
               round(float(displacement[1]), 10))
        if key not in self._cache:
            phase = np.exp(sign * 1j * self.mesh.k @ displacement)
            self._cache[key] = complex(np.mean(phase * table[:, s, t]))
        return self._cache[key]

    def _pair(self, s, t, displacement):
        """``<a_s(r) a_t(r + d)>``, symmetrized so that ``<a_s a_t> = <a_t a_s>`` exactly.

        The identity needs the momentum set to be closed under ``k -> -k``; a
        mesh offset by 1/3 of a step is not, and violates it at order 1/N^2
        (which would make the static self-energy non-Hermitian).
        """
        return 0.5 * (self._avg(self._m, s, t, displacement, -1)
                      + self._avg(self._m, t, s, -np.asarray(displacement), -1))

    def __call__(self, A, B) -> complex:
        (s, ds, r), (t, dt, rp) = A, B
        r, rp = np.asarray(r, dtype=float), np.asarray(rp, dtype=float)
        if ds and not dt:                                    # <a_s^+(r) a_t(r')>
            return self._avg(self._n, s, t, rp - r, +1)
        if not ds and not dt:                                # <a_s(r) a_t(r')>
            return self._pair(s, t, rp - r)
        if ds and dt:                                        # <a_s^+(r) a_t^+(r')>
            return np.conj(self._pair(t, s, r - rp))
        same = s == t and np.allclose(r, rp)                 # <a_s(r) a_t^+(r')>
        return (1.0 if same else 0.0) + self._avg(self._n, t, s, r - rp, +1)


def wick4(contract: Contractions, ops) -> complex:
    """``<O1 O2 O3 O4>`` in a Gaussian state with zero mean."""
    o1, o2, o3, o4 = ops
    return (contract(o1, o2) * contract(o3, o4) + contract(o1, o3) * contract(o2, o4)
            + contract(o1, o4) * contract(o2, o3))


def _vectors(mesh_vectors: np.ndarray, op, q: np.ndarray, create: bool, ns: int) -> np.ndarray:
    """Amplitudes of operator ``op`` to create (or annihilate) ``b_{n,q}``; shape (m, Ns).

    Creation: ``a^+`` uses row ``s`` of ``T_q``, ``a`` uses row ``Ns + s``,
    both conjugated, with phase ``exp(-i q.r)``. Annihilation: ``a`` uses
    row ``s``, ``a^+`` row ``Ns + s``, with phase ``exp(+i q.r)``.
    """
    s, dagger, r = op
    r = np.asarray(r, dtype=float)
    if create:
        row = s if dagger else ns + s
        return np.exp(-1j * q @ r)[:, None] * mesh_vectors[:, row, :ns].conj()
    row = ns + s if dagger else s
    return np.exp(1j * q @ r)[:, None] * mesh_vectors[:, row, :ns]


def _group_cubic(cubic):
    """Merge cubic monomials that differ only by a common translation is not needed; keep list."""
    return [(c, m) for c, m in cubic if abs(c) > 0]


@dataclass(frozen=True)
class NonlinearEnergies:
    """Ground-state energy pieces per magnetic site through order S^0 (E0 units, D20).

    Attributes
    ----------
    classical, zero_point, hartree_fock, cubic, tadpole : float
    total : float
        Their sum.
    condensate : (2Ns,) complex array
        ``<psi>`` at q = 0 (``<a_s>``, then ``<a_s^+>``), order ``S^{-1/2}``.
    """

    classical: float
    zero_point: float
    hartree_fock: float
    cubic: float
    tadpole: float
    condensate: np.ndarray

    @property
    def total(self) -> float:
        return self.classical + self.zero_point + self.hartree_fock + self.cubic + self.tadpole

    @property
    def order_s0(self) -> float:
        """The order-S^0 part ``hartree_fock + cubic + tadpole``."""
        return self.hartree_fock + self.cubic + self.tadpole


class NonlinearSpinWaves:
    """Perturbation theory in ``H_3, H_4`` about the LSWT vacuum on a closed momentum set.

    Parameters
    ----------
    expansion : BosonExpansion
    mesh : BogoliubovMesh
    hamiltonian_at : callable
        ``H(k)`` of LSWT (D36), used for momenta outside the mesh and at q = 0.
    kernel_tolerance : float
        Relative size below which an eigenvalue of ``H(0)`` is a zero mode.
    """

    def __init__(self, expansion: BosonExpansion, mesh: BogoliubovMesh, hamiltonian_at,
                 kernel_tolerance: float = 1e-9, linear_tolerance: float = 1e-8):
        self.expansion = expansion
        self.mesh = mesh
        self.hamiltonian_at = hamiltonian_at
        self.ns = expansion.num_sites
        self.contract = Contractions(mesh)
        self.kernel_tolerance = kernel_tolerance
        linear = expansion.orders[1]
        scale = max(1.0, max((abs(c) for c, _ in expansion.orders[2]), default=1.0))
        self.max_linear = max((abs(c) for c, _ in linear), default=0.0)
        if self.max_linear > linear_tolerance * scale:
            raise LSWTError(f"the reference state is not stationary (linear boson term "
                            f"{self.max_linear:.3g}); the 1/S expansion needs a classical "
                            "stationary state")
        self._tadpole = None

    # ---- static pieces -------------------------------------------------------------
    def hartree_fock_energy(self) -> float:
        """``<H_4>`` per magnetic cell."""
        return float(np.real(sum(c * wick4(self.contract, m) for c, m in self.expansion.orders[4])))

    def zero_point_energy(self) -> float:
        """LSWT zero-point energy per magnetic cell on the mesh."""
        H = np.asarray(self.hamiltonian_at(self.mesh.k))
        return float(np.mean(np.sum(self.mesh.energies, axis=1) / 2
                             - np.real(np.trace(H, axis1=1, axis2=2)) / 4))

    def linear_terms(self) -> np.ndarray:
        """``zeta`` with linear terms ``1/2 (zeta^+ psi + psi^+ zeta)`` per magnetic cell."""
        ns = self.ns
        lam = np.zeros(ns, dtype=complex)      # coefficient of a_s
        mu = np.zeros(ns, dtype=complex)       # coefficient of a_s^+
        for c, m in self.expansion.orders[3]:
            for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
                value = c * self.contract(m[i], m[j])
                s, dagger, _ = m[k]
                if dagger:
                    mu[s] += value
                else:
                    lam[s] += value
        return np.r_[mu, lam]

    def tadpole(self):
        """Condensate shift ``<psi> = -H(0)^+ zeta`` and energy ``-1/2 zeta^+ H(0)^+ zeta`` per cell.

        Raises
        ------
        LSWTError
            If ``zeta`` has a component along a zero mode of ``H(0)``: the
            classical state is then not stationary at order S (the zero-point
            energy pushes it along a classically flat direction), and the
            order-S^0 energy about it is not defined.
        """
        if self._tadpole is None:
            zeta = self.linear_terms()
            if np.linalg.norm(zeta) < 1e-13:
                self._tadpole = (np.zeros_like(zeta), 0.0, zeta)
                return self._tadpole
            H0 = np.asarray(self.hamiltonian_at(np.zeros((1, 2))))[0]
            w, V = np.linalg.eigh(H0)
            scale = max(np.max(np.abs(w)), 1.0)
            kernel = np.abs(w) <= self.kernel_tolerance * scale
            if np.any(w < -self.kernel_tolerance * scale):
                raise LSWTError("H(k=0) has a negative eigenvalue: the reference state is unstable")
            proj = V[:, kernel].conj().T @ zeta
            zeta_scale = max(np.linalg.norm(zeta), 1e-300)
            if np.linalg.norm(proj) > 1e-7 * max(zeta_scale, 1.0) and np.linalg.norm(zeta) > 1e-12:
                raise LSWTError(
                    "the linear (tadpole) term has a component along a zero mode of H(k=0) "
                    f"(|projection| = {np.linalg.norm(proj):.3g}): quantum fluctuations push "
                    "the state along a classically flat direction (order by disorder), so the "
                    "order-S^0 energy about this state is not defined; choose the state at a "
                    "stationary point of the zero-point energy")
            inv = (V[:, ~kernel] / w[~kernel]) @ V[:, ~kernel].conj().T
            shift = -inv @ zeta
            energy = float(np.real(-0.5 * zeta.conj() @ inv @ zeta))
            self._tadpole = (shift, energy, zeta)
        return self._tadpole

    def static_quadratic_terms(self, condensate: Optional[np.ndarray] = None
                               ) -> List[Tuple[complex, Monomial]]:
        """Order-S^0 quadratic monomials: Hartree-Fock of ``H_4`` and ``H_3`` on the condensate.

        ``condensate`` (``<psi>`` at q = 0) defaults to :meth:`tadpole`.
        """
        out: List[Tuple[complex, Monomial]] = []
        for c, m in self.expansion.orders[4]:
            for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
                rest = tuple(m[x] for x in range(4) if x not in (i, j))
                out.append((c * self.contract(m[i], m[j]), rest))
        shift = self.tadpole()[0] if condensate is None else np.asarray(condensate)
        ns = self.ns
        for c, m in self.expansion.orders[3]:
            for x in range(3):
                s, dagger, _ = m[x]
                value = shift[ns + s] if dagger else shift[s]
                if value != 0:
                    out.append((c * value, tuple(m[y] for y in range(3) if y != x)))
        return out

    def static_hamiltonian(self, momenta, condensate: Optional[np.ndarray] = None) -> np.ndarray:
        """``dH(k)`` (Nambu) of the order-S^0 static terms (Hartree-Fock + tadpole)."""
        return nambu_matrices(self.static_quadratic_terms(condensate), np.atleast_2d(momenta),
                              self.ns)

    # ---- cubic vertices --------------------------------------------------------------
    def _cubic_groups(self):
        """Cubic monomials as distinct operators and ``{(o0, o1): [(c, o2), ...]}``."""
        if not hasattr(self, "_groups"):
            ops, groups = {}, {}
            for c, mono in self.expansion.orders[3]:
                ids = [ops.setdefault(op, len(ops)) for op in mono]
                groups.setdefault((ids[0], ids[1]), []).append((c, ids[2]))
            self._groups = (list(ops), groups)
        return self._groups

    def _unsymmetrized_source(self, V) -> np.ndarray:
        """``sum_t c_t C_{o0}(q1) C_{o1}(q2) C_{o2}(q3)`` from creation vectors ``V[label][op]``."""
        ns = self.ns
        _, groups = self._cubic_groups()
        m = V[0].shape[1]
        Y = {}
        for (o0, o1), items in groups.items():
            x = sum(c * V[2][o2] for c, o2 in items)
            y = V[1][o1][:, :, None] * x[:, None, :]
            Y[o0] = Y[o0] + y if o0 in Y else y
        out = np.zeros((m, ns, ns, ns), dtype=complex)
        for o0, y in Y.items():
            out += V[0][o0][:, :, None, None] * y[:, None, :, :]
        return out

    def _source(self, qs: Sequence[np.ndarray], Ts: Sequence[np.ndarray]) -> np.ndarray:
        """Symmetrized three-magnon creation amplitude ``s^s`` (N^{1/2} stripped); (m, Ns, Ns, Ns)."""
        ops, _ = self._cubic_groups()
        if not ops:
            return np.zeros((len(qs[0]),) + (self.ns,) * 3, dtype=complex)
        V = [np.array([_vectors(Ts[a], op, qs[a], True, self.ns) for op in ops]) for a in range(3)]
        out = 0
        for p in PERMUTATIONS:              # label p[j] is created by operator j
            U = self._unsymmetrized_source([V[p[0]], V[p[1]], V[p[2]]])
            out = out + np.transpose(U, (0,) + tuple(1 + np.argsort(p)))
        return out / 6.0

    def _decay(self, q1, q2, k, T1, T2, Tk) -> np.ndarray:
        """Symmetrized decay amplitude ``d^s(q1 n1, q2 n2; k n)`` (N^{1/2} stripped); (m, Ns, Ns, Ns)."""
        ns = self.ns
        m = len(q1)
        out = np.zeros((m, ns, ns, ns), dtype=complex)
        for c, mono in self.expansion.orders[3]:
            for x in range(3):                      # operator x annihilates (k, n)
                rest = [mono[y] for y in range(3) if y != x]
                A = _vectors(Tk, mono[x], k, False, ns)
                c1 = _vectors(T1, rest[0], q1, True, ns)
                c2 = _vectors(T2, rest[1], q2, True, ns)
                d1 = _vectors(T2, rest[0], q2, True, ns)
                d2 = _vectors(T1, rest[1], q1, True, ns)
                out += 0.5 * c * (np.einsum("mi,mj,mk->mijk", c1, c2, A)
                                  + np.einsum("mi,mj,mk->mijk", d2, d1, A))
        return out

    def cubic_energy(self, chunk: int = 4096) -> float:
        """``-6 <sum_n |s^s|^2 / (w1 + w2 + w3)>`` per magnetic cell."""
        mesh = self.mesh
        nk = len(mesh.k)
        total = 0.0
        pairs = np.array(np.meshgrid(np.arange(nk), np.arange(nk), indexing="ij")).reshape(2, -1).T
        for start in range(0, len(pairs), chunk):
            i1, i2 = pairs[start:start + chunk].T
            i3 = mesh.index(-(mesh.fractional[i1] + mesh.fractional[i2]))
            qs = [mesh.k[i1], mesh.k[i2], mesh.k[i3]]
            Ts = [mesh.vectors[i1], mesh.vectors[i2], mesh.vectors[i3]]
            S = self._source(qs, Ts)
            denom = (mesh.energies[i1][:, :, None, None] + mesh.energies[i2][:, None, :, None]
                     + mesh.energies[i3][:, None, None, :])
            total += float(np.sum(np.abs(S) ** 2 / denom))
        return -6.0 * total / nk ** 2

    def energies(self) -> NonlinearEnergies:
        """Ground-state energy pieces per magnetic site."""
        ns = self.ns
        shift, e_tad, _ = self.tadpole()
        return NonlinearEnergies(self.expansion.classical_energy / ns,
                                 self.zero_point_energy() / ns,
                                 self.hartree_fock_energy() / ns,
                                 self.cubic_energy() / ns, e_tad / ns, shift)

    # ---- spectrum -----------------------------------------------------------------
    def _bogoliubov(self, momenta):
        k = np.atleast_2d(np.asarray(momenta, dtype=float))
        H = np.asarray(self.hamiltonian_at(k), dtype=complex)
        _, E, T, _ = _diagonalize(H, "none", k, 1e-12)
        return E[:, :self.ns].real, T

    def cubic_self_energy(self, k, n: int, omega: complex) -> complex:
        """On-shell-type cubic self-energy ``Sigma_3^nn(k, omega)`` (mesh sum over q1).

        ``omega`` may carry a positive imaginary part (broadening).
        """
        mesh = self.mesh
        k = np.asarray(k, dtype=float).reshape(1, 2)
        Ek, Tk = self._bogoliubov(k)
        q1 = mesh.k
        T1, E1 = mesh.vectors, mesh.energies
        nk = len(q1)
        Tkk = np.broadcast_to(Tk, (nk,) + Tk.shape[1:])
        kk = np.broadcast_to(k, (nk, 2))
        q2 = k - q1
        E2, T2 = self._bogoliubov(q2)
        D = self._decay(q1, q2, kk, T1, T2, Tkk)[:, :, :, n]
        decay = np.sum(2 * np.abs(D) ** 2 / (omega - E1[:, :, None] - E2[:, None, :]))
        q2s = -k - q1
        E2s, T2s = self._bogoliubov(q2s)
        S = self._source([kk, q1, q2s], [Tkk, T1, T2s])[:, n]
        source = np.sum(-18 * np.abs(S) ** 2 / (omega + E1[:, :, None] + E2s[:, None, :]))
        return complex((decay + source) / nk)

    def _pair_amplitudes(self, k):
        """Two-magnon amplitudes of ``j_a = [psi_a(k), H_3]`` in the original Nambu basis.

        ``psi = (a_s(k), a_s^+(-k))``. Returns ``X`` (pairs annihilated by
        ``j_a``, ``q1 + q2 = k``), ``Y`` (pairs created, ``q1 + q2 = -k``),
        each (nk, 2Ns, Ns, Ns), and their pair energies (nk, Ns, Ns); one
        pair member runs over the mesh, so the sums are exact on a finite
        torus. N^{1/2} factors are stripped as in :meth:`_decay`.
        """
        key = tuple(np.round(np.asarray(k, dtype=float).reshape(2), 12))
        cache = self.__dict__.setdefault("_pairs", {})
        if key in cache:
            return cache[key]
        ns, mesh = self.ns, self.mesh
        k = np.asarray(k, dtype=float).reshape(1, 2)
        q1, E1, T1 = mesh.k, mesh.energies, mesh.vectors
        qx, qy = k - q1, -k - q1
        Ex, Tx = self._bogoliubov(qx)
        Ey, Ty = self._bogoliubov(qy)
        m = len(q1)
        X = np.zeros((m, 2 * ns, ns, ns), dtype=complex)
        Y = np.zeros((m, 2 * ns, ns, ns), dtype=complex)
        memo: Dict = {}

        def amp(op, label, q, T, create):
            if (op, label) not in memo:
                memo[(op, label)] = _vectors(T, op, q, create, ns)
            return memo[(op, label)]

        for c, mono in self.expansion.orders[3]:
            for x in range(3):
                s, dagger, r = mono[x]
                y, z = [mono[i] for i in range(3) if i != x]
                # [a_s(k), a_s^+(r)] = e^{-ik.r};  [a_s^+(-k), a_s(r)] = -e^{-ik.r}  (N^{-1/2} stripped)
                a, sign = (s, 1.0) if dagger else (ns + s, -1.0)
                f = sign * c * np.exp(-1j * k[0] @ np.asarray(r, dtype=float))
                X[:, a] += f * (amp(y, 1, q1, T1, False)[:, :, None] * amp(z, 2, qx, Tx, False)[:, None, :]
                                + amp(z, 1, q1, T1, False)[:, :, None] * amp(y, 2, qx, Tx, False)[:, None, :])
                Y[:, a] += f * (amp(y, 3, q1, T1, True)[:, :, None] * amp(z, 4, qy, Ty, True)[:, None, :]
                                + amp(z, 3, q1, T1, True)[:, :, None] * amp(y, 4, qy, Ty, True)[:, None, :])
        out = (X, E1[:, :, None] + Ex[:, None, :], Y, E1[:, :, None] + Ey[:, None, :])
        cache[key] = out
        return out

    def cubic_self_energy_matrix(self, k, omega: complex, derivative: int = 0) -> np.ndarray:
        """Cubic one-loop self-energy ``Sigma_3(k, omega)`` (or an omega derivative), Nambu basis.

        ``G^{-1}(k, w) = w sigma_3 - H(k) - dH(k) - Sigma_3(k, w)`` in the
        original basis ``psi = (a_s(k), a_s^+(-k))``, with
        ``Sigma_3 = sigma_3 Pi sigma_3`` and the bubble
        ``Pi_ab = 1/2 sum [X_a X_b^* / (w - E) - Y_a Y_b^* / (w + E)]``
        of ``j = [psi, H_3]``. In the Bogoliubov basis,
        ``T^+ Sigma_3 T`` has :meth:`cubic_self_energy` on its diagonal. No
        Bogoliubov transformation at ``k`` is needed, so ``k`` may carry a
        zero mode (the pseudo-Goldstone mode at k = 0).
        """
        X, EX, Y, EY = self._pair_amplitudes(k)
        m, n2 = X.shape[:2]
        factor = float(np.prod(np.arange(1, derivative + 1))) * (-1) ** derivative
        wx = (factor / (omega - EX) ** (derivative + 1)).reshape(m, -1)
        wy = (factor / (omega + EY) ** (derivative + 1)).reshape(m, -1)
        Xf, Yf = X.reshape(m, n2, -1), Y.reshape(m, n2, -1)
        pi = 0.5 * (np.einsum("man,mn,mbn->ab", Xf, wx, Xf.conj())
                    - np.einsum("man,mn,mbn->ab", Yf, wy, Yf.conj())) / m
        s3 = np.r_[np.ones(n2 // 2), -np.ones(n2 // 2)]
        return s3[:, None] * pi * s3[None, :]

    def magnon_energies(self, momenta, broadening: float = 0.0):
        """Magnon energies through order S^0 at ``momenta`` (on-shell 1/S correction).

        Returns
        -------
        lswt : (m, Ns) array
        static : (m, Ns) array
            Hartree-Fock + tadpole correction.
        cubic : (m, Ns) complex array
            ``Sigma_3^nn(k, w_n(k) + i broadening)``; ``-Im`` is the decay rate.
        """
        k = np.atleast_2d(np.asarray(momenta, dtype=float))
        E, T = self._bogoliubov(k)
        dH = self.static_hamiltonian(k)
        static = np.einsum("kan,kab,kbn->kn", T.conj(), dH, T)[:, :self.ns].real
        cubic = np.array([[self.cubic_self_energy(k[i], n, E[i, n] + 1j * broadening)
                           for n in range(self.ns)] for i in range(len(k))])
        return E, static, cubic
