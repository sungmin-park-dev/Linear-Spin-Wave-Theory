"""Dynamical structure factor with dressed magnons and the two-magnon continuum (D58).

``S(q, w) = sum_n W_n(q) A_n(q, w) + S_2(q, w)``:

- ``W_n(q)``: LSWT one-magnon weights (:func:`~spintoolkit.observables.structure_factor.structure_factor`, O(S));
- ``A_n(q, w)``: magnon spectral function with the Hartree-Fock + tadpole shift
  and the frequency-dependent cubic self-energy
  (:meth:`~spintoolkit.methods.nlswt.NLSWTResult.spectral_function`), which
  broadens decaying magnons and moves part of their weight into the
  two-magnon continuum;
- ``S_2(q, w)``: the non-interacting longitudinal two-magnon continuum
  (:func:`~spintoolkit.observables.two_magnon.two_magnon_continuum`, O(S^0)).

Scope: this is dressed LSWT plus the non-interacting continuum, not the
complete order-S^0 structure factor. The 1/S corrections of the one-magnon
weights and, in noncollinear states, the one-magnon/two-magnon interference
term mixed by the cubic vertex are of the same order and are not included.
Zero temperature, lab-frame (commensurate) states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple
import warnings

import numpy as np

from spintoolkit.observables.structure_factor import structure_factor
from spintoolkit.observables.two_magnon import two_magnon_continuum


def _weight(factor, component: Optional[str]) -> np.ndarray:
    if component is None:
        return factor.trace()
    if component == "neutron":
        return factor.neutron()
    a, b = ("xyz".index(component[0]), "xyz".index(component[1]))
    return np.real(factor.weights[:, :, a, b])


def dressed_structure_factor(lswt_result, nlswt_result, q_points, omega, eta: float,
                             component: Optional[str] = None, mode: str = "on_shell",
                             grid: Tuple[int, int] = (64, 64), levels: int = 400,
                             two_magnon_mesh: Tuple[int, int] = (24, 24)) -> Dict[str, np.ndarray]:
    """``S(q, w)`` from dressed one-magnon peaks plus the two-magnon continuum (D58).

    Parameters
    ----------
    lswt_result : LSWTResult
        Lab-frame LSWT result of the same model and state.
    nlswt_result : NLSWTResult
        Result of :func:`~spintoolkit.methods.nlswt.solve_nlswt` for that state.
    q_points : (nq, 2) array_like
        Cartesian momenta, away from Bragg vectors and other zero modes.
    omega : (nw,) array_like
    eta : float
        Instrumental Lorentzian half width (also used for the continuum).
    component : None, "neutron" or "xx", "zz", ...
        As in :meth:`StructureFactor.spectrum`.
    mode : "on_shell" or "dyson"
        Line shape of the dressed magnons, as in
        :meth:`~spintoolkit.methods.nlswt.NLSWTResult.spectral_function`.
    grid, levels :
        Integration settings of the self-energy.
    two_magnon_mesh :
        Mesh of the continuum.

    Returns
    -------
    dict
        ``one_magnon``, ``two_magnon`` and ``total``, each (nq, nw); ``lswt``
        is the undressed one-magnon spectrum broadened by ``eta`` for comparison.
        With ``mode="dyson"``, ``negative_frequency_weight`` (nq, Ns) of the
        spectral functions flags unphysical poles (positive values).

    Notes
    -----
    Each LSWT mode is matched to the band of the same energy in the
    nonlinear calculation; a mode without such a band raises ValueError.
    """
    from spintoolkit.methods.lswt.run import require_unaugmented

    require_unaugmented(lswt_result, "dressed_structure_factor")
    q = np.atleast_2d(np.asarray(q_points, dtype=float))
    w = np.asarray(omega, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        factor = structure_factor(lswt_result, q)
    if np.any(factor.zero_mode):
        raise ValueError("q_points must avoid zero modes (Bragg vectors): the dressed "
                         "spectrum is not defined there")
    weight = _weight(factor, component)
    dressed = nlswt_result.spectral_function(q, w, eta, mode, grid, levels)
    one = np.zeros((len(q), len(w)))
    for i in range(len(q)):
        bands = dressed["lswt"][i]
        scale = max(float(np.max(bands)), 1.0)
        for mode in np.flatnonzero(factor.energies[i] > 0):
            match = np.flatnonzero(np.abs(bands - factor.energies[i, mode]) < 1e-6 * scale)
            if len(match) == 0:
                raise ValueError(f"LSWT mode at energy {factor.energies[i, mode]:.6g} has no "
                                 f"band of the nonlinear calculation at q = {q[i]}")
            # degenerate bands carry the same (group-mean) spectral function
            one[i] += weight[i, mode] * dressed["spectral"][i, match[0]]
    continuum = two_magnon_continuum(lswt_result, q, mesh=two_magnon_mesh)
    two = continuum.spectrum(w, eta, component=component)
    bare = factor.spectrum(w, eta, component=component)
    out = {"one_magnon": one, "two_magnon": two, "total": one + two, "lswt": bare}
    if mode == "dyson":
        out["negative_frequency_weight"] = dressed["negative_frequency_weight"]
    return out


def structure_factor_s0(nlswt_result, q_points, omega, eta: float,
                        component: Optional[str] = None, grid: Tuple[int, int] = (48, 48),
                        levels: int = 400) -> Dict[str, np.ndarray]:
    """Zero-temperature ``S(q, w)`` through order S^0 (D59).

    Unlike :func:`dressed_structure_factor` (D58), every order-S^0 term is
    included: the 1/S corrections of the one-magnon weights (quasiparticle
    residue, eigenvector rotation by the Hartree-Fock, tadpole and cubic
    self-energies, and the Hartree-Fock and condensate corrections of the
    spin operator), the longitudinal two-magnon continuum, and the
    one-/two-magnon interference through the cubic vertex together with the
    off-shell magnon tail in the continuum. See
    :mod:`spintoolkit.methods.nlswt.structure` for the formulas.

    Parameters
    ----------
    nlswt_result : NLSWTResult
        Result of :func:`~spintoolkit.methods.nlswt.solve_nlswt` for a
        commensurate state (lab frame).
    q_points : (nq, 2) array_like
        Cartesian momenta away from zero modes (Bragg vectors of a Goldstone mode).
    omega : (nw,) array_like
    eta : float
        Instrumental Lorentzian half width, positive. It should exceed a few
        level spacings of the densities (``continuum top / levels``).
    component : None, "neutron" or "xx", "zz", ...
        As in :meth:`StructureFactor.spectrum`.
    grid : (N1, N2)
        Triangle grid of the internal momentum (offset by one third of a step).
    levels : int
        Energies of the two-magnon densities.

    Returns
    -------
    dict
        ``total``, ``one_magnon`` (resummed pole terms) and ``continuum``
        (the rest), each (nq, nw); ``lswt`` (LSWT one-magnon spectrum
        broadened by ``eta``); ``energy`` (nq, Ns) complex pole positions
        (``Re`` the 1/S energy, ``-Im`` the decay rate of D57); ``weight``
        (nq, Ns, 3, 3) complex residues (LSWT weight + order-S^0
        correction; Hermitian for a stable magnon; for a decaying magnon
        averaged over its line width, see the Notes); ``lswt_energy`` and
        ``lswt_weight``; ``negative_weight`` (nq,): the fraction of the
        total spectrum on ``omega`` that is negative, a diagnostic of the
        expansion (zero where it is controlled).

    Notes
    -----
    Bands are ordered as the LSWT energies at each q. A residue whose real
    part is negative, or a total spectrum below zero beyond the broadening,
    means that the 1/S corrections exceed the LSWT terms (typically S = 1/2
    near a decay threshold); the expansion is then not controlled there.
    The same happens within a self-energy width of a strongly decaying
    magnon: the strict term ``a g M(w) g a^+`` follows the van Hove
    singularities of the two-magnon density with the factor
    ``1 / (w - E)^2``, which gives large spikes of either sign that a
    resummation would smooth. ``negative_weight`` reports it.

    For a decaying magnon the strict residue at ``E + i0`` follows the van
    Hove structure of the two-magnon density and does not converge with
    ``grid``; ``weight`` and ``one_magnon`` use the self-energy at
    ``E + i Gamma`` instead. ``total`` is the strict expression and
    ``continuum`` is ``total - one_magnon``, so within a few ``Gamma`` of a
    decaying magnon the split between them is a convention, and the
    continuum there depends on the grid at the level of a few percent of the
    line weight.
    """
    from spintoolkit.methods.lswt.run import LSWTError
    from spintoolkit.methods.nlswt.engine import cauchy_transform
    from spintoolkit.methods.nlswt.structure import (OrderS0Correlation, _Bubble, grid_momenta,
                                                     pair_vectors, project, spectral)

    if eta <= 0:
        raise ValueError("eta must be positive")
    solver = nlswt_result.solver
    ns = solver.ns
    q = np.atleast_2d(np.asarray(q_points, dtype=float))
    w = np.asarray(omega, dtype=float)
    z = w + 1j * eta
    q1 = grid_momenta(solver, grid)
    out = {key: np.zeros((len(q), len(w))) for key in ("total", "one_magnon", "continuum", "lswt")}
    energy = np.zeros((len(q), ns), dtype=complex)
    weight = np.zeros((len(q), ns, 3, 3), dtype=complex)
    lswt_energy = np.zeros((len(q), ns))
    lswt_weight = np.zeros((len(q), ns, 3, 3), dtype=complex)
    for i, qi in enumerate(q):
        try:
            c, Ec, d, Ed = pair_vectors(solver, qi, q1)
            bubble = _Bubble(c, Ec, d, Ed, tuple(int(n) for n in grid), levels)
            corr = OrderS0Correlation(solver, qi, bubble)
        except LSWTError as exc:
            raise ValueError(f"q = {qi} or a pair momentum has a zero mode: the order-S^0 "
                             f"spectrum is not defined there ({exc})") from exc
        poles = corr.poles()
        S_pole = spectral(corr.pole_part(z, poles))
        # continuum: exact density at zero broadening on the level grid (midpoints avoid
        # its nodes), then the Lorentzian of width eta as the transform of that density
        x = 0.5 * (bubble.x[1:] + bubble.x[:-1])
        for E0 in corr.E[:ns]:
            close = np.abs(x - E0) < 1e-6 * (x[1] - x[0])
            x[close] += 1e-3 * (x[1] - x[0])
        density = corr.continuum_density(x, poles)
        S_reg = spectral(cauchy_transform(x, density.reshape(len(x), 9), z).reshape(len(z), 3, 3))
        # the total is the strict chi; for a decaying magnon the line is drawn with the
        # weight averaged over its width and the continuum is the remainder
        S_line = spectral(corr.pole_part(z, poles, weights=True))
        out["total"][i] = project(S_pole + S_reg, qi, component)
        out["one_magnon"][i] = project(S_line, qi, component)
        out["continuum"][i] = out["total"][i] - out["one_magnon"][i]
        E = corr.E[:ns]
        a = corr.a[:, :ns]
        lswt_energy[i] = E
        lswt_weight[i] = np.einsum("an,bn->nab", a, a.conj())
        bare = np.einsum("zn,nab->zab", 1.0 / (z[:, None] - E[None, :]), lswt_weight[i])
        out["lswt"][i] = project(spectral(bare), qi, component)
        for group, E0, shifts, _, _, W in poles:
            for j, n in enumerate(group):
                energy[i, n] = E0 + shifts[j]
                weight[i, n] = W[j]
    total = out["total"]
    with np.errstate(invalid="ignore", divide="ignore"):
        negative = np.sum(np.maximum(-total, 0.0), axis=1) / np.sum(np.abs(total), axis=1)
    out.update(energy=energy, weight=weight, lswt_energy=lswt_energy, lswt_weight=lswt_weight,
               negative_weight=negative)
    return out


def structure_factor_states(nlswt_result, q_points) -> Dict[str, list]:
    """Order-S^0 final states of ``S(q, w)`` on the momenta of the calculation (D59).

    On a finite torus (``solve_nlswt(..., geometry=finite_torus)``) this is
    exact Rayleigh-Schroedinger perturbation theory: one-magnon states with
    the 1/S energies and order-S^0 weights, and two-magnon states (total
    momentum ``q``) at their LSWT pair energies with their order-S^0 weights.
    It reproduces the large-S expansion of exact diagonalization of that
    cluster. On the thermodynamic mesh it is a finite sample of the continuum.

    Parameters
    ----------
    nlswt_result : NLSWTResult
    q_points : (nq, 2) array_like
        Momenta of the torus (``q - q1`` must be a mesh momentum for every
        mesh ``q1``, e.g. magnetic reciprocal vectors on any torus).

    Returns
    -------
    dict
        Lists over q: ``one_magnon_energy`` (Ns,), ``one_magnon_weight``
        (Ns, 3, 3), ``lswt_weight`` (Ns, 3, 3), ``two_magnon_energy`` (P,) and
        ``two_magnon_weight`` (P, 3, 3), one entry per distinct pair energy
        (weights of degenerate pairs summed).
    """
    from spintoolkit.methods.nlswt.structure import OrderS0Correlation, _Bubble, pair_vectors

    solver = nlswt_result.solver
    ns = solver.ns
    q = np.atleast_2d(np.asarray(q_points, dtype=float))
    q1 = solver.mesh.k
    keys = ("one_magnon_energy", "one_magnon_weight", "lswt_weight", "two_magnon_energy",
            "two_magnon_weight")
    out: Dict[str, list] = {key: [] for key in keys}
    for qi in q:
        c, Ec, d, Ed = pair_vectors(solver, qi, q1)
        corr = OrderS0Correlation(solver, qi, _Bubble(c, Ec, d, Ed))
        E1 = np.zeros(ns, dtype=complex)
        W1 = np.zeros((ns, 3, 3), dtype=complex)
        for group, E0, shifts, R, _, _ in corr.poles():
            for j, n in enumerate(group):
                E1[n], W1[n] = E0 + shifts[j], R[j]
        a = corr.a[:, :ns]
        X, CL = c[:, :2 * ns], c[:, 2 * ns:]
        g = corr.g0(Ec.reshape(-1)).reshape(Ec.shape + (2 * ns,))           # (m, n1, n2, 2Ns)
        rotated = np.einsum("na,a,kajl->kjln", corr.T.conj().T, corr.s3, X)  # T^+ s3 X
        u = CL.transpose(0, 2, 3, 1) + np.einsum("am,kjlm->kjla", corr.a, g * rotated)
        W = 0.5 * np.einsum("kjla,kjlb->kjlab", u, u.conj()) / len(q1)
        energies, weights = [], []
        for E, Wp in zip(Ec.reshape(-1), W.reshape(-1, 3, 3)):
            scale = max(1.0, abs(E))
            for idx, E_prev in enumerate(energies):
                if abs(E - E_prev) < 1e-9 * scale:
                    weights[idx] = weights[idx] + Wp
                    break
            else:
                energies.append(E)
                weights.append(Wp)
        out["one_magnon_energy"].append(E1)
        out["one_magnon_weight"].append(W1)
        out["lswt_weight"].append(np.einsum("an,bn->nab", a, a.conj()))
        out["two_magnon_energy"].append(np.array(energies))
        out["two_magnon_weight"].append(np.array(weights))
    return out


@dataclass(frozen=True)
class InteractingPath:
    """Order-S^0 ``S(q, w)`` along a momentum path (:func:`interacting_path`).

    Attributes
    ----------
    Q : (m, 2) array
        Cartesian momenta.
    distance : (m,) array
    labels : list of str
    label_distances : (n,) array
    omega : (nw,) array
    intensity : (m, nw) array
        Total spectrum; NaN at momenta where it is not defined (zero modes).
    one_magnon, continuum, lswt : (m, nw) arrays
        As in :func:`structure_factor_s0`.
    energy : (m, Ns) complex array
        Pole positions (``-Im`` the decay rate).
    negative_weight : (m,) array
        Diagnostic of :func:`structure_factor_s0`.
    """

    Q: np.ndarray
    distance: np.ndarray
    labels: list
    label_distances: np.ndarray
    omega: np.ndarray
    intensity: np.ndarray
    one_magnon: np.ndarray
    continuum: np.ndarray
    lswt: np.ndarray
    energy: np.ndarray
    negative_weight: np.ndarray


def interacting_path(nlswt_result, omega, eta: float, path: Sequence[Any] = ("Γ", "K", "M", "Γ"),
                     points: int = 60, component: Optional[str] = None,
                     grid: Tuple[int, int] = (48, 48), levels: int = 400) -> InteractingPath:
    """:func:`structure_factor_s0` along a piecewise-straight path.

    Parameters
    ----------
    nlswt_result : NLSWTResult
    omega, eta, component, grid, levels
        As in :func:`structure_factor_s0`.
    path : sequence of str or (2,) array_like
        Names of the primitive zone or Cartesian momenta, as in
        :func:`~spintoolkit.observables.neutron.neutron_path`.
    points : int
        Momenta along the path. Each costs one two-magnon bubble on ``grid``.

    Returns
    -------
    InteractingPath
        Momenta where the spectrum is not defined (a zero mode at ``q`` or
        in a pair) are NaN; they are typically the Bragg vectors of a
        Goldstone mode.
    """
    from spintoolkit.observables.bands import _path_momenta, _vertices
    from spintoolkit.system.high_symmetry import high_symmetry_points

    lattice = getattr(nlswt_result, "lattice", None)
    if lattice is None:
        raise ValueError("the result has no primitive lattice (produced by solve_nlswt?)")
    labels, vertices = _vertices(path, high_symmetry_points(np.asarray(lattice, dtype=float)))
    q, distance, label_distances = _path_momenta(vertices, points)
    w = np.asarray(omega, dtype=float)
    ns = nlswt_result.solver.ns
    keys = ("total", "one_magnon", "continuum", "lswt")
    out = {key: np.full((len(q), len(w)), np.nan) for key in keys}
    energy = np.full((len(q), ns), np.nan + 0j)
    negative = np.full(len(q), np.nan)
    for i, qi in enumerate(q):
        try:
            one = structure_factor_s0(nlswt_result, [qi], w, eta, component, grid, levels)
        except ValueError as exc:
            if "zero mode" not in str(exc):
                raise
            continue
        for key in keys:
            out[key][i] = one[key][0]
        energy[i] = one["energy"][0]
        negative[i] = one["negative_weight"][0]
    return InteractingPath(q, distance, labels, label_distances, w, out["total"],
                           out["one_magnon"], out["continuum"], out["lswt"], energy, negative)
