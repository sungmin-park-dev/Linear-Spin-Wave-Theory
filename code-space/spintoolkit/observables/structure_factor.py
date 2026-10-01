"""Spin structure factor and real-space correlations from LSWT (stage 4c, D26, D27).

Conventions
-----------
Spin Fourier components use the full site positions (D13) and are normalized
per site, ``S^a(q) = N^{-1/2} sum_r exp(-i q . r) S^a_r``, so that

    S^{ab}(q, w) = (1/2 pi) int dt exp(i w t) <S^a(q, t) S^b(q)^dagger>

is per site and ``S(q + G) = S(q)`` only for primitive reciprocal vectors G.

To the order of LSWT the result has two parts:

- one-magnon (transverse, O(S)): ``sum_n W_n^{ab}(q) delta(w - w_n(q))`` with
  ``delta S_i = sqrt(S_i/2) [(x_i - i y_i) a_i + (x_i + i y_i) a_i^dagger]`` in
  the local frame of the Hamiltonian. Particle modes appear at ``w = +w_n``
  with ``1 + n_B``, hole components at ``w = -w_n`` with ``n_B`` (detailed
  balance; they vanish at t = 0);
- elastic (Bragg): at magnetic reciprocal vectors, the ordered moments
  ``m_i = (S_i - <n_i>) n_i`` give ``S_el(q) = N sum_G I(G) delta_{q,G}`` with
  ``I^{ab}(G) = F^a(G) F^b(G)^* / N_s^2``, ``F(G) = sum_i m_i exp(-i G . tau_i)``.

The longitudinal two-magnon continuum is one order higher (O(S^0)) and is not
included. Temperatures are ``t = k_B T / E0``; energies are in E0.

The stored diagonalization of the LSWT result is reused for momenta on its
mesh; other momenta are diagonalized with the same regularization policy.

Equal-time real-space correlations (:func:`spin_correlation`,
:func:`bond_correlations`) and the ladder basis (:func:`to_ladder`) replace the
former ``observables/correlations.py`` (D27), which used a per-cell
normalization and applied the sublattice phase twice. Real-time correlations
and the retarded spectral function wait for the review of the response
conventions in the LSWT theory notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import warnings

import numpy as np

from spintoolkit.observables.zero_modes import ZeroModeReport, scan_zero_modes


def _bose(energy: np.ndarray, t: float) -> np.ndarray:
    if t == 0:
        return np.zeros_like(energy)
    with np.errstate(over="ignore", divide="ignore"):
        return 1.0 / np.expm1(energy / t)


def _deviation_vectors(result) -> np.ndarray:
    """A (3, 2Ns): delta S^a(q) = sum_m A[a, m] psi_m(q), per-site normalization."""
    ns = result.num_sites
    x, y = result.local_frames[:, :, 0], result.local_frames[:, :, 1]
    amplitude = np.sqrt(result.spins / (2 * ns))[:, None]
    lowering = amplitude * (x - 1j * y)            # multiplies a_i
    raising = amplitude * (x + 1j * y)             # multiplies a_i^dagger
    return np.concatenate([lowering, raising], axis=0).T


def _diagonalization_at(result, q: np.ndarray):
    """(H, E, T, zero) at momenta q: stored values on the mesh, otherwise computed.

    Where ``H(q)`` has a zero or negative mode (e.g. a Goldstone mode at a Bragg
    vector) E and T are NaN and ``zero`` is True.
    """
    from spintoolkit.methods.lswt.diagonalization import Diagonalizer
    from spintoolkit.methods.lswt.run import LSWTError, _diagonalize

    q = np.atleast_2d(np.asarray(q, dtype=float))
    H_out, E_out, T_out, zero = [], [], [], []
    stored = {tuple(np.round(k, 12)): i for i, k in enumerate(result.k_points)}
    settings = result.header.settings
    for qi in q:
        i = stored.get(tuple(np.round(qi, 12)))
        if i is not None:
            H_out.append(result.hamiltonians[i])
            E_out.append(result.eigenvalues[i])
            T_out.append(result.eigenvectors[i])
            zero.append(False)
            continue
        H = result.hamiltonian_at(qi)
        regularization = settings["regularization"]
        if regularization == "MAGSWT":
            H = H + result.header.diagnostics["max_regularization_shift"] * np.eye(H.shape[1])
            regularization = "none"
        try:
            H, E, T, _ = _diagonalize(np.asarray(H, dtype=complex), regularization, qi[None, :],
                                      settings["zero_mode_tolerance"])
        except LSWTError:
            n2 = H.shape[-1]
            H_out.append(np.asarray(H).reshape(n2, n2))
            E_out.append(np.full(n2, np.nan))
            T_out.append(np.full((n2, n2), np.nan, dtype=complex))
            zero.append(True)
            continue
        H_out.append(H[0])
        E_out.append(E[0])
        T_out.append(T[0])
        zero.append(False)
    return np.array(H_out), np.array(E_out), np.array(T_out), np.array(zero, dtype=bool)


@dataclass(frozen=True)
class StructureFactor:
    """One-magnon and elastic structure factor at a set of momenta.

    Attributes
    ----------
    q : (nq, 2) array
        Cartesian momenta (extended zone).
    energies : (nq, 2Ns) array
        Signed mode energies: ``+w`` for particle modes, ``-w`` for hole
        components (thermal, anti-Stokes).
    weights : (nq, 2Ns, 3, 3) complex array
        ``W^{ab}`` of each mode (Hermitian in a, b), per site.
    elastic : (nq, 3, 3) complex array
        Bragg intensity ``I^{ab}(q)`` where q is a magnetic reciprocal vector,
        zero elsewhere (coefficient of ``N delta_{q,G}``).
    bragg : (nq,) bool array
    zero_mode : (nq,) bool array
        ``H(q)`` has a zero mode at this q (inelastic weights diverge; NaN).
    temperature : float
    gapless : bool or None
    """

    q: np.ndarray
    energies: np.ndarray
    weights: np.ndarray
    elastic: np.ndarray
    bragg: np.ndarray
    zero_mode: np.ndarray
    temperature: float
    gapless: Optional[bool] = None
    zero_modes: Dict[str, Any] = field(default_factory=dict)

    def static(self) -> np.ndarray:
        """Inelastic equal-time ``S^{ab}(q) = sum_n W_n^{ab}(q)``, shape (nq, 3, 3)."""
        return self.weights.sum(axis=1)

    def trace(self) -> np.ndarray:
        """``sum_a W_n^{aa}(q)``, shape (nq, 2Ns)."""
        return np.real(np.einsum("qnaa->qn", self.weights))

    def neutron(self) -> np.ndarray:
        """Mode weights with the polarization factor ``delta_ab - q_a q_b / q^2`` (q in plane).

        Returns (nq, 2Ns); NaN at q = 0 where the factor is undefined.
        """
        out = np.full(self.energies.shape, np.nan)
        for i, q in enumerate(self.q):
            norm = np.linalg.norm(q)
            if norm == 0:
                continue
            unit = np.array([q[0], q[1], 0.0]) / norm
            projector = np.eye(3) - np.outer(unit, unit)
            out[i] = np.real(np.einsum("ab,nab->n", projector, self.weights[i]))
        return out

    def spectrum(self, omega: Sequence[float], eta: float, shape: str = "lorentzian",
                 component: Optional[str] = None) -> np.ndarray:
        """Broadened ``S(q, w)`` on a frequency grid, shape (nq, nw).

        ``component`` None sums the diagonal (trace); "neutron" applies the
        polarization factor; "xx", "zz", ... pick one Cartesian component.
        """
        omega = np.asarray(omega, dtype=float)
        if component is None:
            weight = self.trace()
        elif component == "neutron":
            weight = self.neutron()
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
        return np.einsum("qn,qnw->qw", weight, profile)

    def to_json_dict(self) -> Dict[str, Any]:
        from spintoolkit.methods.result import to_jsonable
        return to_jsonable({k: getattr(self, k) for k in self.__dataclass_fields__})


def _zero_mode_decision(result, temperature, zero_modes, gapless):
    if temperature == 0:
        return None, {}
    from spintoolkit.observables.thermal import ZeroModeCandidateError
    report = zero_modes if zero_modes is not None else scan_zero_modes(result)
    if gapless is None:
        if report.has_candidates:
            raise ZeroModeCandidateError(report)
        gapless = report.has_zero
    if gapless:
        warnings.warn("gapless spectrum: thermal weights diverge as t / w near the zero modes",
                      UserWarning, stacklevel=3)
    return bool(gapless), report.to_dict()


def _apply_moment_tensors(A: np.ndarray, tensors: Optional[np.ndarray], nq: int) -> np.ndarray:
    """Deviation vectors per momentum, ``(nq, 3, 2Ns)``, with site tensors applied.

    ``tensors`` (nq, Ns, 3, 3) maps the spin of site i to the scattering
    moment at each momentum (the neutron factor ``(g_i / 2) F_i(|Q|)``); None
    keeps the spin itself.
    """
    if tensors is None:
        return np.broadcast_to(A, (nq,) + A.shape)
    tensors = np.concatenate([tensors, tensors], axis=1)        # site of each Nambu column
    return np.einsum("qmab,bm->qam", tensors, A)


def _moments(tensors: Optional[np.ndarray], i: int, moments: np.ndarray) -> np.ndarray:
    return moments if tensors is None else np.einsum("sab,sb->sa", tensors[i], moments)


def structure_factor(result, q_points, temperature: float = 0.0,
                     zero_modes: Optional[ZeroModeReport] = None,
                     gapless: Optional[bool] = None,
                     moment_tensors: Optional[np.ndarray] = None) -> StructureFactor:
    """Mode-resolved LSWT structure factor at momenta ``q_points``.

    Parameters
    ----------
    result : LSWTResult
    q_points : (nq, 2) array_like
        Cartesian momenta, anywhere in reciprocal space.
    temperature : float
        ``t = k_B T / E0``. At t > 0 the zero-mode decision of D25 applies.
    zero_modes, gapless : optional
        As in :func:`~spintoolkit.observables.thermal.thermal_quantities`.
    moment_tensors : (nq, Ns, 3, 3) array, optional
        Site- and momentum-dependent tensors ``M_i(q)`` replacing ``S_i`` by
        ``M_i(q) S_i`` in both the inelastic and the elastic part (used by
        :func:`~spintoolkit.observables.neutron.neutron_intensity`). None
        (default) gives the spin structure factor.
    """
    from spintoolkit.methods.lswt.run import require_lab_frame

    require_lab_frame(result, "structure_factor")
    q = np.atleast_2d(np.asarray(q_points, dtype=float))
    t = float(temperature)
    if not np.isfinite(t) or t < 0:
        raise ValueError("temperature must be finite and non-negative")
    gapless, report = _zero_mode_decision(result, t, zero_modes, gapless)
    ns = result.num_sites
    A = _deviation_vectors(result)
    _, E, T, zero = _diagonalization_at(result, q)
    if np.any(zero):
        warnings.warn(f"H(q) has a zero mode at {int(zero.sum())} of {len(q)} momenta "
                      "(e.g. Goldstone modes at Bragg vectors): inelastic weights there are NaN; "
                      "the elastic part is computed", UserWarning, stacklevel=2)
    # H(k) is in the full-position gauge (D13): psi(q) already carries exp(-i q . tau_i),
    # so no further sublattice phase is applied.
    A_q = _apply_moment_tensors(A, _check_tensors(moment_tensors, len(q), ns), len(q))
    amplitudes = np.einsum("qam,qmn->qan", A_q, T)              # (nq, 3, 2Ns)
    energies = np.concatenate([E[:, :ns], -E[:, ns:]], axis=1)
    factors = np.concatenate([1.0 + _bose(E[:, :ns], t), _bose(E[:, ns:], t)], axis=1)
    weights = (np.einsum("qan,qbn->qnab", amplitudes, amplitudes.conj())
               * factors[:, :, None, None])

    moments = (result.spins - result.boson_numbers)[:, None] * result.directions
    fractional = q @ np.asarray(result.magnetic_lattice).T / (2 * np.pi)
    bragg = np.all(np.abs(fractional - np.rint(fractional)) < 1e-9, axis=1)
    elastic = np.zeros((len(q), 3, 3), dtype=complex)
    for i in np.flatnonzero(bragg):
        F = np.exp(-1j * result.positions @ q[i]) @ _moments(moment_tensors, i, moments)
        elastic[i] = np.outer(F, F.conj()) / ns ** 2
    return StructureFactor(q, energies, weights, elastic, bragg, zero, t, gapless, report)


def _check_tensors(tensors, nq, ns):
    if tensors is None:
        return None
    tensors = np.asarray(tensors)
    if tensors.shape != (nq, ns, 3, 3):
        raise ValueError(f"moment_tensors must have shape {(nq, ns, 3, 3)}, got {tensors.shape}")
    return tensors


def spiral_structure_factor(result, q_points, temperature: float = 0.0,
                            zero_modes: Optional[ZeroModeReport] = None,
                            gapless: Optional[bool] = None,
                            moment_tensors: Optional[np.ndarray] = None) -> StructureFactor:
    """Lab-frame structure factor of a single-Q spiral (rotating-frame LSWT, D34).

    With ``S_i = R_n(phi_i) S'_i``, ``phi_i = Q . R_i`` (cell vector ``R_i``)
    and ``R_n(phi) = R_0 + e^{i phi} R_+ + e^{-i phi} R_-``,
    ``R_0 = n n^T``, ``R_+- = (1 - n n^T -+ i [n]_x) / 2``, the lab-frame
    Fourier component (full positions, per site, D13) is

        S(k) = sum_m R_m sum_i exp(-i m Q . tau_i) delta S'_i(k - m Q),   m = 0, +1, -1,

    with ``tau_i`` the site offset in the cell. The rotating-frame correlations
    are translation invariant, so the cross terms between different ``m``
    vanish when neither ``Q`` nor ``2Q`` is a reciprocal vector, and

        S^{ab}(k, w) = sum_m [R_m S'(k - m Q, w) R_m^dagger]^{ab}

    (Toth and Lake 2015): each lab momentum carries the magnons
    ``omega(k)``, ``omega(k - Q)`` and ``omega(k + Q)``. The elastic part sits
    at ``k = G + m Q`` with ``F = R_m sum_i exp(-i k . tau_i) m_i``.

    Parameters
    ----------
    result : SpiralLSWTResult
    q_points : (nq, 2) array_like
        Cartesian lab-frame momenta.
    temperature, zero_modes, gapless
        As in :func:`structure_factor`; the zero-mode scan is of the
        rotating-frame result.
    moment_tensors : (nq, Ns, 3, 3) array, optional
        As in :func:`structure_factor`, applied to the lab-frame spins
        ``R_n(phi_i) S'_i``. The branch cross terms still vanish because the
        tensors are the same in every cell.

    Returns
    -------
    StructureFactor
        Lab frame, with ``3 x 2 Ns`` modes per momentum ordered
        ``m = 0, +1, -1``. ``zero_mode`` marks momenta where any of the three
        branches has a zero mode (e.g. ``k = 0, +-Q`` of a Heisenberg spiral);
        only that branch is NaN.

    Raises
    ------
    NotImplementedError
        If ``Q`` or ``2Q`` is a reciprocal vector: umklapp terms between the
        branches then survive; use the commensurate supercell with
        :func:`structure_factor`.
    """
    rotating = result.rotating
    Q = np.asarray(result.wave_vector, dtype=float)
    lattice = np.asarray(rotating.lattice)
    for multiple in (1, 2):
        f = multiple * Q @ lattice.T / (2 * np.pi)
        if np.all(np.abs(f - np.rint(f)) < 1e-9):
            raise NotImplementedError(
                f"{multiple}Q is a reciprocal lattice vector: branches m and m' with "
                f"(m - m') Q in G interfere. Use the commensurate supercell "
                "(IncommensurateStructure.to_spin_state, solve_lswt, structure_factor).")
    q = np.atleast_2d(np.asarray(q_points, dtype=float))
    t = float(temperature)
    if not np.isfinite(t) or t < 0:
        raise ValueError("temperature must be finite and non-negative")
    gapless, report = _zero_mode_decision(rotating, t, zero_modes, gapless)
    ns = rotating.num_sites
    n = np.asarray(result.structure.rotation_axis, dtype=float)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    P = np.eye(3) - np.outer(n, n)
    rotations = {0: np.outer(n, n).astype(complex), 1: (P - 1j * K) / 2, -1: (P + 1j * K) / 2}
    A = _deviation_vectors(rotating)                              # (3, 2Ns), rotating frame
    tau = np.asarray(rotating.positions)
    moments = (rotating.spins - rotating.boson_numbers)[:, None] * rotating.directions

    tensors = _check_tensors(moment_tensors, len(q), ns)
    energies, weights, zero = [], [], np.zeros(len(q), dtype=bool)
    elastic = np.zeros((len(q), 3, 3), dtype=complex)
    bragg = np.zeros(len(q), dtype=bool)
    for m in (0, 1, -1):
        shifted = q - m * Q
        _, E, T, zero_m = _diagonalization_at(rotating, shifted)
        zero |= zero_m
        phase = np.exp(-1j * m * tau @ Q)
        A_m = _apply_moment_tensors(rotations[m] @ (A * np.r_[phase, phase][None, :]),
                                    tensors, len(q))
        amplitudes = np.einsum("qam,qmn->qan", A_m, T)
        energies.append(np.concatenate([E[:, :ns], -E[:, ns:]], axis=1))
        factors = np.concatenate([1.0 + _bose(E[:, :ns], t), _bose(E[:, ns:], t)], axis=1)
        weights.append(np.einsum("qan,qbn->qnab", amplitudes, amplitudes.conj())
                       * factors[:, :, None, None])
        fractional = shifted @ lattice.T / (2 * np.pi)
        on_lattice = np.all(np.abs(fractional - np.rint(fractional)) < 1e-9, axis=1)
        for i in np.flatnonzero(on_lattice):
            rotated = moments @ rotations[m].T
            F = np.exp(-1j * tau @ q[i]) @ _moments(tensors, i, rotated)
            elastic[i] += np.outer(F, F.conj()) / ns ** 2
        bragg |= on_lattice
    if np.any(zero):
        warnings.warn(f"H(k) has a zero mode in some branch at {int(zero.sum())} of {len(q)} "
                      "momenta (e.g. Goldstone modes at k = 0, +-Q): those weights are NaN",
                      UserWarning, stacklevel=2)
    return StructureFactor(q, np.concatenate(energies, axis=1), np.concatenate(weights, axis=1),
                           elastic, bragg, zero, t, gapless, report)


#: Ladder basis (S^+, S^-, S^z) from Cartesian components: S^mu = sum_a LADDER[mu, a] S^a.
LADDER = np.array([[1.0, 1j, 0.0], [1.0, -1j, 0.0], [0.0, 0.0, 1.0]])


def to_ladder(tensor: np.ndarray) -> np.ndarray:
    """Cartesian ``T^{ab} = <S^a ... (S^b)^dagger>`` in the ladder basis ``(+, -, z)``.

    ``T^{mu nu} = sum_ab L[mu, a] T^{ab} L[nu, b]^*`` for the last two axes, so
    ``T^{++}`` pairs ``S^+`` with ``(S^+)^dagger = S^-``.
    """
    return np.einsum("ma,...ab,nb->...mn", LADDER, np.asarray(tensor), LADDER.conj())


def _correlator_setup(result):
    ns = result.num_sites
    kernel = np.concatenate([np.ones(ns), np.zeros(ns)])
    G = np.einsum("kmn,n,kln->kml", result.eigenvectors, kernel, result.eigenvectors.conj())
    A = _deviation_vectors(result) * np.sqrt(ns)                # per-site operators
    supercell = np.rint(np.asarray(result.magnetic_lattice)
                        @ np.linalg.inv(result.lattice)).astype(int)
    index = {key: i for i, key in enumerate(result.site_keys)}
    return G, A, supercell, index


def _fluctuation(result, G, A, i, j, r_i, r_j) -> np.ndarray:
    """``<delta S_i^a delta S_j^b>`` at zero temperature on the stored mesh."""
    ns = result.num_sites
    phases = np.exp(1j * result.k_points @ (np.asarray(r_i) - np.asarray(r_j)))
    Gij = np.einsum("k,kmn->mn", result.weights * phases, G)
    Ai = np.zeros((3, 2 * ns), dtype=complex)
    Aj = np.zeros((3, 2 * ns), dtype=complex)
    Ai[:, [i, ns + i]] = A[:, [i, ns + i]]
    Aj[:, [j, ns + j]] = A[:, [j, ns + j]]
    return Ai @ Gij @ Aj.conj().T


def spin_correlation(result, model, first, second) -> Dict[str, np.ndarray]:
    """Equal-time ``<S_i^a S_j^b>`` between any two sites at zero temperature.

    Parameters
    ----------
    result : LSWTResult
    model : SpinModel
    first, second : (site_id, cell)
        Model site and primitive cell (any integers; reduced to the magnetic cell).

    Returns
    -------
    dict
        ``ordered`` ``m_i m_j^T``, ``fluctuation`` ``<delta S_i delta S_j>`` and
        ``total`` (their sum), each (3, 3). The fluctuation part is quadratic in
        the bosons (the order of LSWT); the mesh of ``result`` sets the
        resolution at large distances. For the same site the fluctuation part
        is the harmonic ``<delta S_i^a delta S_i^b>``, not the exact
        ``S(S+1)`` identity.
    """
    from spintoolkit.methods.lswt.run import require_lab_frame
    from spintoolkit.states.spin_state import reduce_cell

    require_lab_frame(result, "spin_correlation")
    G, A, supercell, index = _correlator_setup(result)
    (a, ca), (b, cb) = first, second
    i = index[(a, reduce_cell(ca, supercell))]
    j = index[(b, reduce_cell(cb, supercell))]
    r_i, r_j = model.cartesian_position(a, ca), model.cartesian_position(b, cb)
    fluctuation = _fluctuation(result, G, A, i, j, r_i, r_j)
    m = (result.spins - result.boson_numbers)[:, None] * result.directions
    ordered = np.outer(m[i], m[j])
    return {"ordered": ordered, "fluctuation": fluctuation, "total": ordered + fluctuation}


def bond_correlations(result, model) -> Dict[str, Any]:
    """Equal-time correlations on every bilinear bond and the energy they imply.

    Onsite terms (D37) enter the energy but are not listed as bonds.

    ``<S_i^a S_j^b> = m_i^a m_j^b + <delta S_i^a delta S_j^b>`` (see
    :func:`spin_correlation`). The energy is evaluated to the order of LSWT:
    the product of ordered moments is linearized in the boson numbers, so the
    result equals ``result.ground_state_energy`` on the same mesh.

    Returns
    -------
    dict
        ``bonds``: list of per-term dicts (``label``, ``site_i``, ``site_j``,
        ``offset``, ``correlation`` (3, 3)); ``energy``: per site, E0.
    """
    from spintoolkit.methods.lswt.run import require_lab_frame
    from spintoolkit.system.model import BILINEAR, ZEEMAN
    from spintoolkit.states.spin_state import reduce_cell

    require_lab_frame(result, "bond_correlations")
    ns = result.num_sites
    G, A_cell, supercell, index = _correlator_setup(result)
    spins, n_bos = result.spins, result.boson_numbers
    bonds, energy = [], 0.0
    for term in model.terms_of_kind(BILINEAR):
        (a, n1), (b, n2) = term.participants
        J = term.coefficient
        for cell in {c for _, c in result.site_keys}:
            ci = (cell[0] + n1[0], cell[1] + n1[1])
            cj = (cell[0] + n2[0], cell[1] + n2[1])
            i = index[(a, reduce_cell(ci, supercell))]
            j = index[(b, reduce_cell(cj, supercell))]
            fluctuation = np.real(_fluctuation(result, G, A_cell, i, j, model.cartesian_position(a, ci),
                                               model.cartesian_position(b, cj)))
            n_i, n_j = result.directions[i], result.directions[j]
            ordered = np.outer((spins[i] - n_bos[i]) * n_i, (spins[j] - n_bos[j]) * n_j)
            linear = (spins[i] * spins[j] - spins[i] * n_bos[j] - spins[j] * n_bos[i]) \
                * np.outer(n_i, n_j)
            bonds.append({"label": term.label, "site_i": result.site_keys[i],
                          "site_j": result.site_keys[j], "offset": (tuple(n1), tuple(n2)),
                          "correlation": ordered + fluctuation})
            energy += np.sum(J * (linear + fluctuation))
    from spintoolkit.system.model import ONSITE, onsite_renormalization
    for term in model.terms_of_kind(ONSITE):
        # Coherent-state onsite energy to the order of LSWT (D37): kappa A on the
        # on-site correlation, plus (S/2) tr A, minus the normal-ordering constant
        # kappa (S/2)(tr A - n^T A n) that <delta S delta S> contains and the
        # constant (S/2) tr A already accounts for.
        site, A = term.participants[0][0], term.coefficient
        for i, (s, cell) in enumerate(result.site_keys):
            if s != site:
                continue
            S, n = spins[i], result.directions[i]
            kappa = onsite_renormalization(S)
            r = model.cartesian_position(site, cell)
            fluctuation = np.real(_fluctuation(result, G, A_cell, i, i, r, r))
            linear = (S * S - 2 * S * n_bos[i]) * np.outer(n, n)
            energy += kappa * np.sum(A * (linear + fluctuation)) + 0.5 * S * np.trace(A) \
                - kappa * 0.5 * S * (np.trace(A) - n @ A @ n)
    field_vector = np.asarray(result.header.conditions["field"], dtype=float)
    for term in model.terms_of_kind(ZEEMAN):
        site = term.participants[0][0]
        for i, (s, _) in enumerate(result.site_keys):
            if s == site:
                energy -= (term.coefficient.T @ field_vector) @ ((spins[i] - n_bos[i]) * result.directions[i])
    return {"bonds": bonds, "energy": float(energy / ns)}
