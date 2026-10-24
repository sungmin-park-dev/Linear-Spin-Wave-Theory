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
import warnings

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


def degenerate_groups(energies: np.ndarray, tolerance: float = 1e-8):
    """Index groups of equal energies (relative tolerance)."""
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
    _, E, T, _, _ = _diagonalize(H, "none", k, zero_mode_tolerance)
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


def hartree_fock_terms(quartic: Sequence[Tuple[complex, Monomial]], contract: Contractions
                       ) -> List[Tuple[complex, Monomial]]:
    """Quadratic monomials of the Hartree-Fock decoupling of normal-ordered quartic terms.

    ``O1 O2 O3 O4 -> sum over the six pairs <Oi Oj> (remaining two)``, so that
    the Nambu matrix of the result is the derivative of ``<H_4>`` with respect
    to the two-point functions of the Gaussian state ``contract``.
    """
    out: List[Tuple[complex, Monomial]] = []
    for c, m in quartic:
        for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            rest = tuple(m[x] for x in range(4) if x not in (i, j))
            out.append((c * contract(m[i], m[j]), rest))
    return out


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


def _triangle_vertices(f: np.ndarray):
    """Vertex values of ``f`` and flat grid indices of the two triangles of each square.

    Each triangle's vertices are sorted by ``f``; arrays of shape (2 N1 N2, 3).
    """
    n1, n2 = f.shape
    idx = np.arange(n1 * n2).reshape(n1, n2)
    corners = [(0, 0), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 1)]
    I = np.concatenate([np.stack([np.roll(idx, (-a, -b), (0, 1)).ravel() for a, b in tri], axis=1)
                        for tri in corners])
    F = f.ravel()[I]
    order = np.argsort(F, axis=1)
    return np.take_along_axis(F, order, 1), np.take_along_axis(I, order, 1)


def _sorted_triangles(f: np.ndarray, g: np.ndarray):
    """Vertex values of the two triangles of each grid square, sorted by ``f``; (2 N1 N2, 3) each."""
    F, I = _triangle_vertices(f)
    return F, g.ravel()[I]


def triangle_density(f: np.ndarray, g: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """``rho(x) = int du dv g(u, v) delta(x - f(u, v))`` over the unit torus at each level x.

    ``f`` is the (N1, N2) node values of a periodic grid and ``g`` the node
    values of one function, shape (N1, N2), or of several, (N1, N2, C), real
    or complex. Each grid square is cut into two triangles; on each, ``f``
    and ``g`` are linear, and the integral is exact for that interpolation:
    the weight ``int delta(x - f) dA`` of the level line times the mean of
    ``g`` on it (the two-dimensional analogue of the tetrahedron method).

    Where ``f`` is constant on a triangle carrying weight (a flat band of pair
    energies) the measure has a delta-function atom, which a density cannot
    represent; the result is then NaN at every level, with a warning.

    Returns
    -------
    (levels,) or (levels, C) array
    """
    n1, n2 = f.shape
    g = np.asarray(g)
    single = g.ndim == 2
    flat = g.reshape(n1 * n2, -1)
    F, I = _triangle_vertices(f)
    G = [flat[I[:, v]] for v in range(3)]
    if _flat_triangle_atoms(F, G) is not None:
        warnings.warn(
            "triangle integration: the pair energy is constant on triangles carrying weight (a "
            "flat band); its spectral measure has a delta-function atom, which a density cannot "
            "represent, so the density is NaN. Use a finite torus (exact discrete poles) for "
            "decoupled or flat-band systems.", UserWarning, stacklevel=2)
        out = np.full((len(levels), flat.shape[1]), np.nan,
                      dtype=np.result_type(flat.dtype, float))
    else:
        out = _triangle_levels(F, G, levels, n1 * n2)
    return out[:, 0] if single else out


def _triangle_levels(F: np.ndarray, G, levels: Sequence[float], cells: int) -> np.ndarray:
    """Level-line integrals of :func:`triangle_density`; flat triangles contribute nothing."""
    f1, f2, f3 = F.T
    out = np.zeros((len(levels), G[0].shape[1]), dtype=np.result_type(G[0].dtype, float))
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, x in enumerate(np.asarray(levels, dtype=float)):
            low = (f1 < x) & (x <= f2)
            high = (f2 < x) & (x <= f3)
            w = np.zeros((len(f1), 3))
            if np.any(low):
                a, b, c = f1[low], f2[low], f3[low]
                t12, t13 = (x - a) / (b - a), (x - a) / (c - a)
                coef = (x - a) / ((b - a) * (c - a))          # 2 (x - a)/((b - a)(c - a)) * 1/2
                w[low] = coef[:, None] * np.stack([2 - t12 - t13, t12, t13], axis=1)
            if np.any(high):
                a, b, c = f1[high], f2[high], f3[high]
                t13, t23 = (x - a) / (c - a), (x - b) / (c - b)
                coef = (c - x) / ((c - a) * (c - b))
                w[high] = coef[:, None] * np.stack([1 - t13, 1 - t23, t13 + t23], axis=1)
            out[i] = w[:, 0] @ G[0] + w[:, 1] @ G[1] + w[:, 2] @ G[2]
    return out * (0.5 / cells)


def _flat_triangle_atoms(F: np.ndarray, G, tolerance: float = 1e-10):
    """Mask of triangles on which ``f`` is constant and ``g`` is not zero; None if there are none.

    On such a triangle ``int g delta(x - f) dA`` is a delta function in ``x`` of
    mass ``(area) <g>``, not a density.
    """
    scale = max(1.0, float(np.max(np.abs(F), initial=0.0)))
    flat = (F[:, 2] - F[:, 0]) <= tolerance * scale
    if not np.any(flat):
        return None
    weight = sum(np.max(np.abs(Gv), axis=1) for Gv in G)
    size = max(float(np.max(weight, initial=0.0)), 1e-300)
    atoms = flat & (weight > 1e-12 * size)
    return atoms if np.any(atoms) else None


def triangle_delta_integral(f: np.ndarray, g: np.ndarray) -> float:
    """``int du dv g(u, v) delta(f(u, v))`` over the unit torus (see :func:`triangle_density`).

    NaN where ``f`` vanishes identically on a triangle carrying weight: the
    integral is then a delta function of the level, not a number. A flat
    region at a non-zero value does not contribute.
    """
    n1, n2 = f.shape
    F, I = _triangle_vertices(-np.asarray(f, dtype=float))
    G = [np.asarray(g).reshape(n1 * n2, 1)[I[:, v]] for v in range(3)]
    atoms = _flat_triangle_atoms(F, G)
    if atoms is not None:
        scale = max(1.0, float(np.max(np.abs(F))))
        if np.any(atoms & (np.abs(F[:, 0]) <= 1e-10 * scale)):
            return float("nan")
    return float(np.real_if_close(_triangle_levels(F, G, [0.0], n1 * n2)[0, 0]))


def hilbert_transform(x: np.ndarray, rho: np.ndarray, omega: Sequence[float]) -> np.ndarray:
    """``int dx' rho(x') / (w - x' + i0)`` for ``rho`` piecewise linear on the grid ``x``.

    The real part is the principal value, exact for the linear interpolant
    (each segment gives ``(rho_0 + s (w - x_0)) ln|(w - x_0)/(w - x_1)| - s (x_1 - x_0)``);
    the imaginary part is ``-pi rho(w)``. ``rho`` vanishes outside ``x``.
    """
    return cauchy_transform(x, rho, np.asarray(omega, dtype=float))


def cauchy_transform(x: np.ndarray, rho: np.ndarray, z, derivative: bool = False) -> np.ndarray:
    """``F(z) = int dx' rho(x') / (z - x')`` (or ``F'(z)``) for ``rho`` piecewise linear on ``x``.

    ``rho`` has shape (L,) or (L, C) (several densities on the same grid,
    real or complex) and vanishes outside ``x``. Each segment contributes
    ``(rho_0 + s (z - x_0)) [ln(z - x_0) - ln(z - x_1)] - s (x_1 - x_0)``,
    exact for the linear interpolant. Complex ``z`` off the real axis uses
    the complex logarithm; real ``z`` means ``z + i0`` (principal value, and
    ``-i pi rho(z)``, or ``-i pi rho'(z)`` for the derivative).

    Returns
    -------
    (nz,) or (nz, C) array
    """
    x = np.asarray(x, dtype=float)
    rho = np.asarray(rho)
    single = rho.ndim == 1
    r = rho.reshape(len(x), -1)
    z = np.atleast_1d(np.asarray(z))
    real = not np.iscomplexobj(z) or np.all(np.imag(z) == 0)
    x0, x1 = x[:-1], x[1:]
    r0, r1 = r[:-1], r[1:]
    s = (r1 - r0) / (x1 - x0)[:, None]
    if real:
        w = np.real(z).astype(float)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            log = np.log(np.abs(w - x0)) - np.log(np.abs(w - x1))
            log = np.where(np.isfinite(log), log, 0.0)
            inv = np.where(w == x0, 0.0, 1.0 / (w - x0)) - np.where(w == x1, 0.0, 1.0 / (w - x1))
    else:
        w = z.astype(complex)[:, None]
        log = np.log(w - x0) - np.log(w - x1)
        inv = 1.0 / (w - x0) - 1.0 / (w - x1)
    if derivative:
        out = log @ s + inv @ r0 + (inv * (w - x0)) @ s
    else:
        out = log @ r0 + (log * (w - x0)) @ s - ((x1 - x0) @ s)[None, :]
    if real:
        out = out.astype(complex)
        w0 = np.real(z).astype(float)
        if derivative:
            seg = np.clip(np.searchsorted(x, w0) - 1, 0, len(x) - 2)
            inside = (w0 > x[0]) & (w0 < x[-1])
            out = out - 1j * np.pi * np.where(inside[:, None], s[seg], 0.0)
        else:
            for c in range(r.shape[1]):
                out[:, c] += -1j * np.pi * (np.interp(w0, x, r[:, c].real, left=0.0, right=0.0)
                                             + 1j * np.interp(w0, x, np.imag(r[:, c]), left=0.0,
                                                              right=0.0))
    return out[:, 0] if single else out

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
        out = hartree_fock_terms(self.expansion.orders[4], self.contract)
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
        _, E, T, _, _ = _diagonalize(H, "none", k, 1e-12)
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

    def resonant_on_mesh(self, momenta, tolerance: float = 1e-10) -> np.ndarray:
        """Bands whose energy lies inside the two-magnon decay continuum of the mesh (D71).

        ``True`` for ``(k, n)`` when some mesh pair ``(q, k - q)`` with a
        non-zero decay vertex (``|D|^2`` above ``tolerance`` times the largest
        one) has ``w_n1(q) + w_n2(k - q) < w_n(k)``. There the mesh sum of
        :meth:`cubic_self_energy` at ``omega = w_n(k)`` crosses its poles and
        its real part is not the principal value. A channel closed by the
        vertex (a hidden symmetry) does not count.
        """
        k_all = np.atleast_2d(np.asarray(momenta, dtype=float))
        mesh = self.mesh
        q1, T1, E1 = mesh.k, mesh.vectors, mesh.energies
        nk = len(q1)
        flags = np.zeros((len(k_all), self.ns), dtype=bool)
        for i, k in enumerate(k_all):
            k = k.reshape(1, 2)
            Ek, Tk = self._bogoliubov(k)
            E2, T2 = self._bogoliubov(k - q1)
            D = self._decay(q1, k - q1, np.broadcast_to(k, (nk, 2)), T1, T2,
                            np.broadcast_to(Tk, (nk,) + Tk.shape[1:]))
            weight = np.abs(D) ** 2
            pair = E1[:, :, None] + E2[:, None, :]
            for n in range(self.ns):
                w = weight[..., n]
                open_ = w > tolerance * max(float(w.max()), 1e-300)
                flags[i, n] = bool(np.any(open_ & (pair < Ek[0, n])))
        return flags

    def decay_rate(self, k, grid: Sequence[int] = (64, 64)) -> Tuple[np.ndarray, np.ndarray]:
        """Golden-rule decay rate ``Gamma_n(k) = -Im Sigma_3^nn(k, w_n(k) + i0)`` (D57).

        ``Gamma_n(k) = pi <sum_{n1 n2} 2 |D_{n1 n2; n}(q, k - q)|^2
        delta(w_n(k) - w_n1(q) - w_n2(k - q))>_q``, the ``eta -> 0`` limit of
        the decay part of :meth:`cubic_self_energy`, with the average over the
        magnetic zone done by linear interpolation on triangles of an
        ``N1 x N2`` grid offset by one third of a step (no broadening).

        Returns
        -------
        energies, rates : (Ns,) arrays
            LSWT energies ``w_n(k)`` and ``Gamma_n(k)`` (order S^0).
        """
        n1, n2 = (int(n) for n in grid)
        k = np.asarray(k, dtype=float).reshape(1, 2)
        Ek, Tk = self._bogoliubov(k)
        u = np.array([((i + 1 / 3) / n1, (j + 1 / 3) / n2) for i in range(n1) for j in range(n2)])
        q1 = u @ self.mesh.reciprocal
        q2 = k - q1
        E1, T1 = self._bogoliubov(q1)
        E2, T2 = self._bogoliubov(q2)
        m = len(q1)
        D = self._decay(q1, q2, np.broadcast_to(k, (m, 2)), T1, T2,
                        np.broadcast_to(Tk, (m,) + Tk.shape[1:]))
        weight = 2 * np.abs(D) ** 2                                    # (m, n1, n2, n)
        rates = np.zeros(self.ns)
        for n in range(self.ns):
            for a in range(self.ns):
                for b in range(self.ns):
                    f = (E1[:, a] + E2[:, b] - Ek[0, n]).reshape(n1, n2)
                    rates[n] += triangle_delta_integral(f, weight[:, a, b, n].reshape(n1, n2))
        return Ek[0], np.pi * rates

    def cubic_spectral_densities(self, k, grid: Sequence[int] = (64, 64), levels: int = 400):
        """Two-magnon densities of the cubic self-energy at ``k`` (D58).

        ``rho_dec_n(x) = <sum 2 |D_{n1 n2; n}(q, k - q)|^2 delta(x - w_n1(q) - w_n2(k - q))>_q``
        and ``rho_src_n(x) = <sum 18 |S_{n n1 n2}(k, q, -k - q)|^2 delta(x - w_n1(q) - w_n2(-k - q))>_q``
        on ``levels`` equally spaced energies from 0 to the top of the
        continua, by triangle integration (:func:`triangle_density`), so that
        ``Sigma_3^nn(k, w) = int rho_dec(x) / (w - x + i0) dx - int rho_src(x) / (w + x) dx``.

        Returns
        -------
        energies : (Ns,) array
            LSWT ``w_n(k)``.
        x : (levels,) array
        rho_dec, rho_src : (Ns, levels) arrays
        """
        n1, n2 = (int(n) for n in grid)
        ns = self.ns
        k = np.asarray(k, dtype=float).reshape(1, 2)
        Ek, Tk = self._bogoliubov(k)
        u = np.array([((i + 1 / 3) / n1, (j + 1 / 3) / n2) for i in range(n1) for j in range(n2)])
        q1 = u @ self.mesh.reciprocal
        m = len(q1)
        kk = np.broadcast_to(k, (m, 2))
        Tkk = np.broadcast_to(Tk, (m,) + Tk.shape[1:])
        E1, T1 = self._bogoliubov(q1)
        E2, T2 = self._bogoliubov(k - q1)
        E3, T3 = self._bogoliubov(-k - q1)
        decay = 2 * np.abs(self._decay(q1, k - q1, kk, T1, T2, Tkk)) ** 2      # (m, n1, n2, n)
        source = 18 * np.abs(self._source([kk, q1, -k - q1], [Tkk, T1, T3])) ** 2  # (m, n, n1, n2)
        top = max(float(np.max(E1[:, :, None] + E2[:, None, :])),
                  float(np.max(E1[:, :, None] + E3[:, None, :])))
        x = np.linspace(0.0, 1.02 * top, int(levels))
        rho_dec = np.zeros((ns, len(x)))
        rho_src = np.zeros((ns, len(x)))
        for a in range(ns):
            for b in range(ns):
                f_dec = (E1[:, a] + E2[:, b]).reshape(n1, n2)
                f_src = (E1[:, a] + E3[:, b]).reshape(n1, n2)
                for n in range(ns):
                    rho_dec[n] += triangle_density(f_dec, decay[:, a, b, n].reshape(n1, n2), x)
                    if np.any(source[:, n, a, b]):
                        rho_src[n] += triangle_density(f_src, source[:, n, a, b].reshape(n1, n2), x)
        return Ek[0], x, rho_dec, rho_src

    def cubic_spectral_density_matrix(self, k, group: Sequence[int],
                                      grid: Sequence[int] = (64, 64), levels: int = 400):
        """Matrix two-magnon densities of a group of bands at ``k`` (degenerate bands).

        ``rho_dec_nm(x) = <sum 2 D^*_{n1 n2; n} D_{n1 n2; m} delta(x - ...)>_q`` and
        ``rho_src_nm`` likewise, for ``n, m`` in ``group``, on the energies of
        :meth:`cubic_spectral_densities` (whose densities are the diagonal), so that
        ``Sigma_3^nm(k, w) = int rho_dec_nm / (w - x + i0) - int rho_src_nm / (w + x)``
        in the Bogoliubov basis of :meth:`cubic_self_energy_matrix`.

        Returns
        -------
        x : (levels,) array
        rho_dec, rho_src : (G, G, levels) complex arrays
        """
        g = list(group)
        n1, n2 = (int(n) for n in grid)
        k = np.asarray(k, dtype=float).reshape(1, 2)
        Ek, Tk = self._bogoliubov(k)
        u = np.array([((i + 1 / 3) / n1, (j + 1 / 3) / n2) for i in range(n1) for j in range(n2)])
        q1 = u @ self.mesh.reciprocal
        m = len(q1)
        kk = np.broadcast_to(k, (m, 2))
        Tkk = np.broadcast_to(Tk, (m,) + Tk.shape[1:])
        E1, T1 = self._bogoliubov(q1)
        E2, T2 = self._bogoliubov(k - q1)
        E3, T3 = self._bogoliubov(-k - q1)
        D = self._decay(q1, k - q1, kk, T1, T2, Tkk)[..., g]                  # (m, n1, n2, G)
        S = np.moveaxis(self._source([kk, q1, -k - q1], [Tkk, T1, T3])[:, g], 1, -1)
        decay = 2 * D[..., :, None].conj() * D[..., None, :]                  # (m, n1, n2, G, G)
        source = 18 * S[..., :, None].conj() * S[..., None, :]
        top = max(float(np.max(E1[:, :, None] + E2[:, None, :])),
                  float(np.max(E1[:, :, None] + E3[:, None, :])))
        x = np.linspace(0.0, 1.02 * top, int(levels))
        size = len(g)
        rho_dec = np.zeros((len(x), size * size), dtype=complex)
        rho_src = np.zeros((len(x), size * size), dtype=complex)
        for a in range(self.ns):
            for b in range(self.ns):
                f_dec = (E1[:, a] + E2[:, b]).reshape(n1, n2)
                f_src = (E1[:, a] + E3[:, b]).reshape(n1, n2)
                rho_dec += triangle_density(f_dec, decay[:, a, b].reshape(n1, n2, -1), x)
                if np.any(source[:, a, b]):
                    rho_src += triangle_density(f_src, source[:, a, b].reshape(n1, n2, -1), x)
        shape = (size, size, len(x))
        return x, rho_dec.T.reshape(shape), rho_src.T.reshape(shape)

    def _commutator_terms(self, k: np.ndarray):
        """``j_a = [psi_a(k), H_3]`` as ``(a, f, y, z)``: ``j_a = sum f y z`` per cell (N^{1/2} stripped)."""
        ns = self.ns
        out = []
        for c, mono in self.expansion.orders[3]:
            for x in range(3):
                s, dagger, r = mono[x]
                y, z = [mono[i] for i in range(3) if i != x]
                # [a_s(k), a_s^+(r)] = e^{-ik.r};  [a_s^+(-k), a_s(r)] = -e^{-ik.r}  (N^{-1/2} stripped)
                a, sign = (s, 1.0) if dagger else (ns + s, -1.0)
                out.append((a, sign * c * np.exp(-1j * k @ np.asarray(r, dtype=float)), y, z))
        return out

    def bilinear_pair_amplitudes(self, terms, size: int, q1, T1, qx, Tx, qy, Ty):
        """Two-magnon amplitudes of operators ``O_a = sum_cells f y z`` at total momentum ``k``.

        ``terms`` lists ``(a, f, y, z)``: operator ``O_a`` contains the
        two-boson monomial ``f y z`` (``f`` carrying the ``exp(-i k . r)`` of
        the translation sum). Returns ``c`` (pairs ``(q1, n1), (qx, n2)``
        annihilated, ``q1 + qx = k``) and ``d`` (pairs ``(q1, n1), (qy, n2)``
        created, ``q1 + qy = -k``), each (m, size, Ns, Ns) and symmetric under
        exchange of the two magnons, so that
        ``O_a = 1/2 sum_{q1 n1 n2} [c_a b b + d_a b^+ b^+] + ...`` with the
        N^{1/2} of the momentum sum stripped as in :meth:`_decay`.
        """
        ns = self.ns
        m = len(q1)
        c = np.zeros((m, size, ns, ns), dtype=complex)
        d = np.zeros((m, size, ns, ns), dtype=complex)
        memo: Dict = {}

        def amp(op, label, q, T, create):
            if (op, label) not in memo:
                memo[(op, label)] = _vectors(T, op, q, create, ns)
            return memo[(op, label)]

        for a, f, y, z in terms:
            c[:, a] += f * (amp(y, 1, q1, T1, False)[:, :, None] * amp(z, 2, qx, Tx, False)[:, None, :]
                            + amp(z, 1, q1, T1, False)[:, :, None] * amp(y, 2, qx, Tx, False)[:, None, :])
            d[:, a] += f * (amp(y, 3, q1, T1, True)[:, :, None] * amp(z, 4, qy, Ty, True)[:, None, :]
                            + amp(z, 3, q1, T1, True)[:, :, None] * amp(y, 4, qy, Ty, True)[:, None, :])
        return c, d

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
        X, Y = self.bilinear_pair_amplitudes(self._commutator_terms(k[0]), 2 * ns,
                                             q1, T1, qx, Tx, qy, Ty)
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
        for i in range(len(k)):
            for group in degenerate_groups(E[i]):
                if len(group) > 1:
                    static[i, group], cubic[i, group] = self.degenerate_corrections(
                        k[i], T[i], dH[i], group, E[i, group[0]] + 1j * broadening)
        return E, static, cubic

    def correction_block(self, k, T, dH, group, omega: complex) -> Tuple[np.ndarray, np.ndarray]:
        """Static and cubic corrections ``P (dH + Sigma_3(k, omega)) P`` on a degenerate group.

        ``T`` is the Bogoliubov matrix at ``k``, ``dH`` the static Hamiltonian
        there, both Nambu; the blocks are in the particle basis of the bands
        in ``group``, whose diagonals are the band-diagonal corrections.
        """
        g = np.asarray(group)
        static = (T.conj().T @ dH @ T)[np.ix_(g, g)]
        sigma = T.conj().T @ self.cubic_self_energy_matrix(np.reshape(k, (1, 2)), omega) @ T
        return static, sigma[np.ix_(g, g)]

    def degenerate_corrections(self, k, T, dH, group, omega: complex):
        """First-order corrections of an exactly degenerate group of bands.

        The diagonal elements of ``dH + Sigma_3`` in a degenerate subspace
        depend on the arbitrary basis of that subspace; the corrections are
        the eigenvalues of the projected matrix ``P (dH + Sigma_3(w_0)) P``
        (degenerate perturbation theory). Returns the static and cubic parts
        in the eigenbasis of the sum, ascending in the real part of the
        total, so that ``static + cubic`` are the eigenvalues.
        """
        static, sigma = self.correction_block(k, T, dH, group, omega)
        total = static + sigma
        if np.allclose(total, total.conj().T, atol=1e-12 * max(1.0, np.max(np.abs(total)))):
            values, V = np.linalg.eigh((total + total.conj().T) / 2)
            Vinv = V.conj().T
        else:
            values, V = np.linalg.eig(total)
            order = np.argsort(values.real)
            values, V = values[order], V[:, order]
            Vinv = np.linalg.inv(V)
        static_part = np.real(np.diag(Vinv @ static @ V))
        return static_part, values - static_part
