"""Pseudo-Goldstone gap to next order in 1/S (D47).

A classically flat global rotation (an accidental degeneracy, not a
symmetry) has a zero mode at k = 0 in LSWT; quantum fluctuations pin it and
give the pseudo-Goldstone gap ``Delta^2 = C_phi / chi`` at leading order. This
module computes the next order of ``Delta^2`` in 1/S.

The gap is the lowest pole of the k = 0 boson propagator,

    det[w sigma_3 - H(0) - Sigma(w)] = 0,

with ``H(0) ~ S`` (one zero mode: the coordinate ``x`` along the rotation is
free, its conjugate ``p`` has stiffness ``K``), the one-loop self-energy
``Sigma_1(w) ~ S^0`` (Hartree-Fock + tadpole + cubic bubble, the full Nambu
matrix of :meth:`NonlinearSpinWaves.cubic_self_energy_matrix`) and the
two-loop self-energy ``Sigma_2 ~ S^{-1}``. Since ``w ~ S^{1/2}``,
``Delta^2 = A + B`` with ``A = K Sigma_1^xx(0)`` (order S) and ``B`` (order
S^0) collecting

- the one-loop matrix beyond ``xx``: ``Sigma_1^pp``, ``Sigma_1^xp`` and its
  frequency derivative, ``d^2 Sigma_1^xx / dw^2`` and the hard-mode Schur
  complement;
- the two-loop static ``Sigma_2^xx(0)``. Only this element of ``Sigma_2``
  enters at this order; every other one is suppressed by a further 1/S.

``Sigma_2^xx(0)`` is a static uniform vertex, i.e. the curvature of the
effective potential ``Gamma(xbar)`` (Legendre transform with respect to the
field ``x``): the order-S^0 ground-state energy of the rotated state
``R(phi)`` with ``<x> = xbar(phi)`` held fixed, as a function of ``xbar``.
The background-field expansion about ``R(phi)`` reuses the order-S^0 energy
of :class:`NonlinearSpinWaves` (classical, zero-point, Hartree-Fock, cubic)
with two changes: the tadpole is minimized under the constraint (the
linear-in-bosons part of ``x`` in the rotated frame is held at zero), and
``xbar(phi)`` is evaluated to relative order 1/S with the exact
Holstein-Primakoff inverse ``a = (S + s^z)^{-1/2} s^+``. Then

    Gamma''(0) = E''(0) / xbar'(0)^2 = U_1 + U_2,
    U_1 = E_zp'' / xbar_1'^2 = Sigma_1^xx(0)          (Ward identity, checked),
    U_2 = [E_2'' - 2 E_zp'' xbar_2' / xbar_1'] / xbar_1'^2 = Sigma_2^xx(0).

``Sigma_2`` enters the pole as ``U_2 w^+ w`` (``w`` the functional that
measures ``x``); the result does not depend on the choice of ``w``. The
strict expansion is read off by scaling ``H(0) -> H(0)/t``,
``Sigma_1(w) -> Sigma_1(t w)``, ``Sigma_2 -> t Sigma_2`` and fitting
``t w(t)^2 = A + B t`` for ``t -> 0`` (``t = 1`` is the model's spin).

Momenta. Loop sums use the mesh of :class:`NLSWTSettings` offset by 1/3 of a
step (no zone centre, closed under addition). Such a mesh breaks the point
group of the lattice, and the sixfold (or fourfold) dependence on the
rotation angle is a small difference of large numbers: on one mesh it is
masked by the discretization error. Every quantity is therefore averaged over
the images of the mesh under the lattice point group (each image is an
equally valid quadrature; the average restores the symmetry).

Regime of validity. The expansion is in 1/S about the LSWT vacuum with
gapless internal soft modes; individual one-loop elements (``Sigma_1^pp``,
``d Sigma_1^xp / dw``, ``d^2 Sigma_1^xx / dw^2``) grow linearly with the mesh
size and cancel in ``B``. ``B`` itself is mesh-converged, and the optional
``pinning`` (a local field along every spin, which gaps the soft branch)
measures how much the result depends on the infrared. When ``|B / A|`` is
not small the series does not determine the gap at this spin.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.methods.lswt.run import LSWTError
from spintoolkit.methods.nlswt.engine import NonlinearSpinWaves, bogoliubov_mesh
from spintoolkit.methods.nlswt.expansion import expand_model
from spintoolkit.methods.result import ResultHeader
from spintoolkit.states.spin_state import SpinState, validate_spin_state
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import SpinModel


@dataclass(frozen=True)
class PseudoGoldstoneSettings:
    """Settings of :func:`pseudo_goldstone_gap`.

    Parameters
    ----------
    mesh : (int, int)
        ``N1 x N2`` mesh of the magnetic reciprocal cell offset by 1/3 step.
    step : float
        Rotation step (rad) of the five-point second derivatives in ``phi``.
    symmetrize : bool
        Average over the point-group images of the mesh (see module notes).
    pinning : float
        Local field along every spin, ``-pinning * n_i . S_i / S`` per spin
        (adds ``pinning`` to every boson energy). Zero is the model; a
        positive value gaps the soft branch and is an infrared diagnostic of
        the curvatures (the gap is then not computed).
    t_values : tuple of float
        Scaling parameters of the strict 1/S fit.
    """

    mesh: Tuple[int, int] = (24, 24)
    step: float = 0.02
    symmetrize: bool = True
    pinning: float = 0.0
    t_values: Tuple[float, ...] = (5e-4, 1e-3, 2e-3, 4e-3)
    zero_mode_tolerance: float = 1e-9

    def as_dict(self) -> Dict[str, Any]:
        return {"mesh": list(self.mesh), "mesh_offset": "1/3 step", "step": self.step,
                "symmetrize": self.symmetrize, "pinning": self.pinning,
                "t_values": list(self.t_values)}


@dataclass(frozen=True)
class PseudoGoldstoneResult:
    """Pseudo-Goldstone gap through relative order 1/S.

    Attributes
    ----------
    header : ResultHeader
    gap_squared : (float, float)
        ``(A, B)``: ``Delta^2 = A + B + ...`` at the model's spin, ``A`` the
        leading order (order S) and ``B`` the next (order S^0).
    curvature : dict
        ``U1``, ``U2`` (static curvatures of ``Gamma`` per magnetic cell in
        the units of the soft coordinate), ``sigma_xx`` (one-loop
        ``Sigma^xx(0)``, equal to ``U1`` by the Ward identity),
        ``zero_point``, ``order_s0`` (second derivatives in ``phi`` per
        magnetic cell of ``E_zp`` and of the order-S^0 energy).
    components : dict
        ``B_one_loop`` (``B`` without the two-loop static term) and
        ``B_two_loop_static``.
    """

    header: ResultHeader
    gap_squared: Tuple[float, float]
    curvature: Dict[str, float]
    components: Dict[str, float]

    @property
    def leading_gap(self) -> float:
        """``Delta = sqrt(A)``, the leading-order (``C_phi / chi``) gap."""
        return float(np.sqrt(self.gap_squared[0]))

    @property
    def relative_correction(self) -> float:
        """``B / A``: next-order correction of ``Delta^2`` relative to the leading order."""
        return float(self.gap_squared[1] / self.gap_squared[0])


def rotate_state(model: SpinModel, state: SpinState, axis, angle: float) -> SpinState:
    """``state`` with every spin rotated by ``angle`` about the global ``axis``."""
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return SpinState.from_function(model, state.supercell,
                                   lambda s, c: R @ state.direction(s, c),
                                   dict(state.provenance, rotation=[n.tolist(), float(angle)]))


def _point_group(reciprocal: np.ndarray) -> List[np.ndarray]:
    """Cartesian rotations and reflections that map the reciprocal lattice to itself."""
    ops = []
    for n in (1, 2, 3, 4, 6):
        for j in range(n):
            a = 2 * np.pi * j / n
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            for P in (np.eye(2), np.diag([1.0, -1.0])):
                G = R @ P
                M = reciprocal @ G.T @ np.linalg.inv(reciprocal)
                if np.allclose(M, np.rint(M), atol=1e-9) and \
                        not any(np.allclose(G, H, atol=1e-12) for H in ops):
                    ops.append(G)
    return ops


def _mesh_images(n1: int, n2: int, reciprocal: np.ndarray, symmetrize: bool):
    grid = np.array([((i + 1 / 3) / n1, (j + 1 / 3) / n2) for i in range(n1) for j in range(n2)])
    k = grid @ reciprocal
    res = 3 * n1 * n2 // gcd(n1, n2)
    images, keys = [], set()
    for G in (_point_group(reciprocal) if symmetrize else [np.eye(2)]):
        kg = k @ G.T
        fg = np.mod(kg @ np.linalg.inv(reciprocal), 1.0)
        key = tuple(sorted(map(tuple, np.round(fg * res).astype(int) % res)))
        if key not in keys:          # M and -M give identical Nambu sums: keep one of them
            keys.add(key)
            keys.add(tuple(sorted(map(tuple, np.round(-fg * res).astype(int) % res))))
            images.append((kg, fg))
    return images


def _constrained_tadpole(H0: np.ndarray, zeta: np.ndarray, c1: np.ndarray) -> float:
    """``min_psi 1/2 psi^+ H0 psi + Re zeta^+ psi`` subject to ``c1 . psi = 0`` (KKT)."""
    n = len(zeta)
    A = np.zeros((n + 1, n + 1), dtype=complex)
    A[:n, :n] = H0
    A[:n, n] = c1.conj()
    A[n, :n] = c1
    psi = np.linalg.solve(A, np.r_[-zeta, 0.0])[:n]
    return float(np.real(0.5 * psi.conj() @ H0 @ psi + np.real(zeta.conj() @ psi)))


def _nambu_real(v: np.ndarray, ns: int) -> np.ndarray:
    j = int(np.argmax(np.abs(v[:ns])))
    c = np.sqrt(np.conj(v[j]) / v[ns + j])
    v = v * c / abs(c)
    if np.max(np.abs(v[ns:] - v[:ns].conj())) > 1e-8:
        raise LSWTError("the k = 0 zero mode is not a real (Nambu-symmetric) coordinate")
    return v


class _Frame:
    """Everything at one rotation angle: per-mesh solvers and the angle-local quantities."""

    def __init__(self, model, state, conditions, images, reciprocal, settings):
        self.quadratic = QuadraticBoseHamiltonian(model, state, conditions)
        self.expansion = expand_model(model, state, conditions)
        ns = self.expansion.num_sites
        pin = settings.pinning
        self.hamiltonian_at = lambda k: np.asarray(self.quadratic.at(k)) + pin * np.eye(2 * ns)
        n1, n2 = settings.mesh
        self.solvers = [NonlinearSpinWaves(
            self.expansion,
            bogoliubov_mesh(self.hamiltonian_at, kg, fg, reciprocal, settings.zero_mode_tolerance,
                            3 * n1 * n2 // gcd(n1, n2)),
            self.hamiltonian_at) for kg, fg in images]
        self.H0 = self.hamiltonian_at(np.zeros((1, 2)))[0]
        self.zeta = np.mean([s.linear_terms() for s in self.solvers], axis=0)

    def local_densities(self):
        """Mesh-averaged ``<a_s^+ a_s>`` and ``<a_s a_s>`` (rotated frame, LSWT vacuum)."""
        ns = self.expansion.num_sites
        out = np.zeros((ns, 2), dtype=complex)
        for solver in self.solvers:
            for s in range(ns):
                r = tuple(self.expansion.positions[s])
                out[s, 0] += solver.contract((s, True, r), (s, False, r))
                out[s, 1] += solver.contract((s, False, r), (s, False, r))
        return out / len(self.solvers)


def _field_terms(frame: _Frame, frames0: np.ndarray, w: np.ndarray):
    """``xbar`` (leading, 1/S correction) and the linear coefficient ``c1`` of the field ``w . psi_0``.

    ``psi_0`` are the bosons of the reference frame; at the rotated frame,
    ``a_0 = (2S)^{-1/2} [s_0^+ + n_0 s_0^+ / (4S)]`` (exact HP inverse to this
    order) with ``s_0^+ = p . s``, ``n_0 = S - z . s`` in rotated local
    components and the rotated LSWT vacuum (no condensate: ``c1 . <psi> = 0``
    is the constraint).
    """
    ex = frame.expansion
    ns = ex.num_sites
    dens = frame.local_densities()
    lead, corr = np.zeros(ns, dtype=complex), np.zeros(ns, dtype=complex)
    c1 = np.zeros(2 * ns, dtype=complex)
    for s in range(ns):
        S = ex.spins[s]
        e, f = ex.local_frames[s], frames0[s]
        p = (f[:, 0] + 1j * f[:, 1]) @ e
        z = f[:, 2] @ e
        n, m = dens[s, 0].real, dens[s, 1]
        C = np.zeros((2, 2), dtype=complex)                  # <s^mu s^nu>, mu, nu in {x, y}, order S
        C[0, 0] = S / 2 * (m + np.conj(m) + 2 * n + 1)
        C[1, 1] = S / 2 * (2 * n + 1 - m - np.conj(m))
        C[0, 1] = S / 2j * (m - np.conj(m) - 1)
        C[1, 0] = S / 2j * (m - np.conj(m) + 1)
        transverse = sum(z[a] * p[b] * C[a, b] for a in range(2) for b in range(2))
        lead[s] = (p[2] * S + S * p[2] * (1 - z[2]) / 4) / np.sqrt(2 * S)
        corr[s] = (-p[2] * n + (-S * p[2] * n + 2 * S * n * z[2] * p[2] - transverse) / (4 * S)) \
            / np.sqrt(2 * S)
        ws, wd = w[s], w[ns + s]
        c1[s] = 0.5 * (ws * (p[0] - 1j * p[1]) + wd * (np.conj(p[0]) - 1j * np.conj(p[1])))
        c1[ns + s] = 0.5 * (ws * (p[0] + 1j * p[1]) + wd * (np.conj(p[0]) + 1j * np.conj(p[1])))
    x_lead = float(np.real(w @ np.r_[lead, lead.conj()]))
    x_corr = float(np.real(w @ np.r_[corr, corr.conj()]))
    return x_lead, x_corr, c1


def _strict_fit(H0, M0, M1, M2, extra, t_values):
    """``t w(t)^2 = A + B t + ...`` for the soft pole of ``w s3 - H0/t - M(t w) - t extra``."""
    ns = len(H0) // 2
    s3 = np.diag(np.r_[np.ones(ns), -np.ones(ns)])
    f = []
    for t in t_values:
        ev = np.linalg.eigvals(s3 @ (H0 / t + M0 + t * extra))
        w = abs(ev[np.argmin(np.abs(ev))])
        for _ in range(200):
            M = H0 / t + M0 + t * w * M1 + 0.5 * (t * w) ** 2 * M2 + t * extra
            ev = np.linalg.eigvals(s3 @ M)
            new = ev[np.argmin(np.abs(ev - w))]
            if abs(new - w) <= 1e-13 * abs(w):
                break
            w = new
        if abs(np.imag(w)) > 1e-8 * abs(w) or np.real(w) <= 0:
            raise LSWTError("the soft pole is not real and positive in the strict-expansion fit "
                            "(the leading-order curvature is not positive)")
        f.append(t * np.real(w) ** 2)
    c = np.polyfit(np.asarray(t_values), np.asarray(f), 2)
    return float(c[2]), float(c[1])


def pseudo_goldstone_gap(model: SpinModel, state: SpinState, axis=(0.0, 0.0, 1.0),
                         conditions: Optional[ExternalConditions] = None,
                         settings: PseudoGoldstoneSettings = PseudoGoldstoneSettings()
                         ) -> PseudoGoldstoneResult:
    """Pseudo-Goldstone gap at k = 0 through relative order 1/S (two loops).

    Parameters
    ----------
    model : SpinModel
        Bilinear and Zeeman terms.
    state : SpinState
        Classical state at an extremum (by symmetry) of the zero-point energy
        along the flat rotation; the minimum gives a real gap.
    axis : (3,) array_like
        Global axis of the classically flat rotation ``R(phi)``.
    conditions : ExternalConditions, optional
    settings : PseudoGoldstoneSettings

    Returns
    -------
    PseudoGoldstoneResult

    Raises
    ------
    LSWTError
        If the rotation is not classically flat, ``H(0)`` does not have
        exactly one zero mode, or the state is not stationary at order S.
    """
    conditions = conditions or ExternalConditions()
    validate_spin_state(state, model)
    if conditions.temperature > 0:
        raise ValueError("pseudo_goldstone_gap is a zero-temperature method")
    magnetic = state.magnetic_lattice(model)
    reciprocal = 2 * np.pi * np.linalg.inv(magnetic).T
    images = _mesh_images(*settings.mesh, reciprocal, settings.symmetrize)
    d = settings.step
    frames = {j: _Frame(model, rotate_state(model, state, axis, j * d), conditions, images,
                        reciprocal, settings) for j in (-2, -1, 0, 1, 2)}
    ref = frames[0]
    ns = ref.expansion.num_sites
    s3 = np.diag(np.r_[np.ones(ns), -np.ones(ns)])
    classical = {j: fr.expansion.classical_energy for j, fr in frames.items()}
    scale = max(1.0, abs(classical[0]))
    if max(abs(classical[j] - classical[0]) for j in classical) > 1e-10 * scale:
        raise LSWTError("the rotation about the given axis is not a classically flat direction")

    # soft coordinate at k = 0 (pinning lifts it; use the model's H(0) for the basis)
    H0 = np.asarray(ref.quadratic.at(np.zeros((1, 2))))[0]
    w0, V0 = np.linalg.eigh(H0)
    zero = np.abs(w0) <= 1e-8 * np.max(np.abs(w0))
    if zero.sum() != 1 or np.any(w0 < -1e-8 * np.max(np.abs(w0))):
        raise LSWTError(f"H(k=0) must have exactly one zero mode and no negative one "
                        f"(eigenvalues {w0[:3]})")
    kx = _nambu_real(V0[:, 0], ns)
    kp = -1j * np.linalg.pinv(H0, rcond=1e-10) @ s3 @ kx
    kp = kp - kx * (kx.conj() @ kp)
    w = (kp.conj() @ s3) / (kp.conj() @ s3 @ kx)            # measures the kx coefficient
    soft_force = abs(kx.conj() @ ref.zeta)
    if soft_force > 1e-7 * max(1.0, np.linalg.norm(ref.zeta)) and settings.pinning == 0:
        raise LSWTError(f"the state is not stationary along the rotation at order S (soft "
                        f"tadpole {soft_force:.3g}); choose a symmetric orientation")

    frames0 = ref.expansion.local_frames
    energy, xl, xc = {}, {}, {}
    zero_point, order_s0 = {}, {}
    for j, fr in frames.items():
        xl[j], xc[j], c1 = _field_terms(fr, frames0, w)
        zp = np.mean([s.zero_point_energy() for s in fr.solvers])
        e2 = np.mean([s.hartree_fock_energy() + s.cubic_energy() for s in fr.solvers]) \
            + _constrained_tadpole(fr.H0, fr.zeta, c1)
        zero_point[j], order_s0[j] = zp, e2

    def second(f):
        return (-f[2] + 16 * f[1] - 30 * f[0] + 16 * f[-1] - f[-2]) / (12 * d * d)

    def first(f):
        return (-f[2] + 8 * f[1] - 8 * f[-1] + f[-2]) / (12 * d)

    zpp, e2pp = second(zero_point), second(order_s0)
    x1, x2 = first(xl), first(xc)
    U1 = zpp / x1 ** 2
    U2 = (e2pp - 2 * zpp * x2 / x1) / x1 ** 2

    # one-loop self-energy at k = 0 (mesh average; condensate from the averaged tadpole)
    shift = -np.linalg.pinv(ref.H0, rcond=1e-10) @ ref.zeta
    k0 = np.zeros(2)
    static = np.mean([s.static_hamiltonian(k0, shift)[0] for s in ref.solvers], axis=0)
    cubic = [np.mean([s.cubic_self_energy_matrix(k0, 0j, n) for s in ref.solvers], axis=0)
             for n in range(3)]
    M0 = static + cubic[0]
    sigma_xx = float(np.real(kx.conj() @ M0 @ kx))
    extra = U2 * np.outer(w.conj(), w)
    flat = 1e-9 * max(abs(zero_point[0]), abs(order_s0[0]), 1e-300)
    if abs(zpp) < flat and abs(e2pp) < flat:
        A = B1 = B = 0.0      # the rotation is an exact symmetry at both orders: Goldstone mode
    elif settings.pinning == 0:
        A, B1 = _strict_fit(ref.H0, M0, cubic[1], cubic[2], 0 * extra, settings.t_values)
        _, B = _strict_fit(ref.H0, M0, cubic[1], cubic[2], extra, settings.t_values)
    else:                 # the pinned soft mode is classically gapped: no pseudo-Goldstone pole
        A = B1 = B = float("nan")
    header = ResultHeader.build(
        "nlswt_pseudo_goldstone", model, state, None, conditions, settings.as_dict(),
        "Delta^2 in E0^2; curvatures per magnetic cell",
        {"axis": list(map(float, axis)), "mesh_images": len(images),
         "ward_identity_relative_error": abs(sigma_xx - U1) / max(abs(U1), 1e-300)})
    return PseudoGoldstoneResult(
        header, (A, B),
        {"U1": U1, "U2": U2, "sigma_xx": sigma_xx, "zero_point": zpp, "order_s0": e2pp},
        {"B_one_loop": B1, "B_two_loop_static": B - B1})
