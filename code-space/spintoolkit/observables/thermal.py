"""Finite-temperature LSWT quantities in dimensionless units (stage 4b, D20, D25).

Input is an :class:`~spintoolkit.methods.lswt.run.LSWTResult`; its stored
eigenvalues and eigenvectors are reused, nothing is diagonalized again. The
temperature is ``t = k_B T / E0``; energies are per site in E0, entropy and
specific heat per site in units of k_B. The existing kelvin/meV routines of
:mod:`spintoolkit.observables.thermodynamics` do the sums: only ``E / (k_B T)``
enters, so reading E0 as meV and passing ``T = t / k_B`` (k_B in meV/K) is exact.

Quantities (per site, ``N_s`` sites in the magnetic cell, ``omega_nk`` the
magnon energies, ``n_B`` the Bose factor):

- ``F = E_cl + Delta E_zp + t <sum_n ln(1 - exp(-omega/t))> / N_s``
- ``U = E_cl + Delta E_zp + <sum_n omega n_B> / N_s``
- ``S = <sum_n (1 + n_B) ln(1 + n_B) - n_B ln n_B> / N_s``,  ``C = dU/dt``
- boson numbers ``<a_i^dagger a_i>(t)``, moments ``S_i - <n_i>`` and the spin
  magnetization ``sum_i (S_i - <n_i>) n_i / N_s``

The magnetization here is the sum of the reduced moments along the fixed
classical axes ``n_i``. In a canted state it misses the 1/S shift of the
canting angle and is not the field derivative of F; the magnetization
``-dF/dh`` at finite temperature is
:func:`~spintoolkit.methods.magnetization.magnetization_curve` with
``temperatures`` (D49). In a collinear state along the field the two agree.

The classical reference state is held fixed (no self-consistency) and magnon
interactions are absent, so these are low-temperature results. In two
dimensions a zero mode makes the boson numbers diverge logarithmically at
``t > 0`` (Mermin-Wagner), while F, U, S and C stay finite for linear or
quadratic dispersion. The zero-mode scan decides which case applies; if it
finds candidates it cannot classify, the user must state ``gapless`` (D25).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence
import warnings

import numpy as np

from spintoolkit.definitions.constants import K_BOLTZMANN_MEV
from spintoolkit.methods.lswt.hamiltonian import log_1_m_exp
from spintoolkit.observables.thermodynamics import Thermodynamics, compute_bosonic_number_at_k
from spintoolkit.observables.zero_modes import ZeroModeReport, scan_zero_modes


class ZeroModeCandidateError(ValueError):
    """Zero-mode candidates need the user's decision (``gapless=True`` or ``False``)."""

    def __init__(self, report: ZeroModeReport):
        self.report = report
        super().__init__(
            "zero-mode candidates found; inspect them and pass gapless=True (zero modes: "
            "boson numbers and moments at t > 0 are reported as NaN) or gapless=False "
            f"(small gaps: computed). Candidates: {report.summary()}")


@dataclass(frozen=True)
class ThermalResult:
    """Finite-temperature quantities on a temperature grid.

    Attributes
    ----------
    temperatures : (nt,) array
        ``t = k_B T / E0``.
    free_energy, internal_energy : (nt,) arrays
        Per site, E0, including the classical and zero-point energies.
    entropy, specific_heat : (nt,) arrays
        Per site, units of k_B.
    boson_numbers, moments : (nt, Ns) arrays
        NaN at ``t > 0`` when the spectrum is gapless.
    magnetization : (nt, 3) array
        ``sum_i (S_i - <n_i>) n_i / N_s`` (no g-tensor, no canting-angle
        shift); NaN at ``t > 0`` when gapless.
    gapless : bool
    decision : str
        "scan" when the scan decided, "user" when ``gapless`` was given.
    zero_modes : dict
        The scan report.
    beyond_lswt : (nt,) bool array
        True where some ``<n_i> > S_i``: the moment would reverse and the
        spin-wave expansion is no longer valid at that temperature.
    """

    temperatures: np.ndarray
    free_energy: np.ndarray
    internal_energy: np.ndarray
    entropy: np.ndarray
    specific_heat: np.ndarray
    boson_numbers: np.ndarray
    moments: np.ndarray
    magnetization: np.ndarray
    gapless: bool
    decision: str
    zero_modes: Dict[str, Any] = field(default_factory=dict)
    beyond_lswt: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def to_json_dict(self) -> Dict[str, Any]:
        from spintoolkit.methods.result import to_jsonable
        return to_jsonable({k: getattr(self, k) for k in self.__dataclass_fields__})


def magnon_free_energy(result, temperatures: Sequence[float]) -> np.ndarray:
    """Thermal magnon part of F per site, ``t <sum_n ln(1 - exp(-omega/t))> / N_s``.

    Parameters
    ----------
    result : LSWTResult
    temperatures : sequence of float
        Dimensionless ``t = k_B T / E0 >= 0``.

    Returns
    -------
    (nt,) array
        In E0; zero at ``t = 0``, ``-inf`` at ``t > 0`` if a mesh point is an
        exact zero mode.
    """
    energies = result.eigenvalues[:, :result.num_sites]
    return np.array([float(result.weights @ np.sum(log_1_m_exp(energies, tk / K_BOLTZMANN_MEV),
                                                   axis=1)) / result.num_sites
                     for tk in np.asarray(temperatures, dtype=float).ravel()])


def _k_data(result) -> Dict[Any, Any]:
    """The stored diagonalization in the layout of the existing observables."""
    return {i: [[H], [E, T], [True, None]]
            for i, (H, E, T) in enumerate(zip(result.hamiltonians, result.eigenvalues,
                                              result.eigenvectors))}


def thermal_quantities(result, temperatures: Sequence[float],
                       zero_modes: Optional[ZeroModeReport] = None,
                       gapless: Optional[bool] = None) -> ThermalResult:
    """Finite-temperature quantities of an LSWT result.

    Parameters
    ----------
    result : LSWTResult
    temperatures : sequence of float
        Dimensionless ``t = k_B T / E0 >= 0``.
    zero_modes : ZeroModeReport, optional
        A scan from :func:`~spintoolkit.observables.zero_modes.scan_zero_modes`;
        run here if omitted.
    gapless : bool, optional
        The user's decision. Required when the scan reports candidates.

    Raises
    ------
    ZeroModeCandidateError
        If the scan finds candidates and ``gapless`` is None.
    """
    t = np.asarray(temperatures, dtype=float).ravel()
    if np.any(~np.isfinite(t)) or np.any(t < 0):
        raise ValueError("temperatures must be finite and non-negative")
    report = zero_modes if zero_modes is not None else scan_zero_modes(result)
    if gapless is None:
        if report.has_candidates:
            raise ZeroModeCandidateError(report)
        gapless_flag, decision = report.has_zero, "scan"
    else:
        gapless_flag, decision = bool(gapless), "user"
    if gapless_flag and np.any(t > 0):
        warnings.warn("gapless spectrum: boson numbers, moments and magnetization at t > 0 "
                      "diverge in two dimensions and are reported as NaN; F, U, S, C are finite",
                      UserWarning, stacklevel=2)

    ns = result.num_sites
    k_data = _k_data(result)
    thermo = Thermodynamics()
    thermo.Ns = ns
    free = result.classical_energy + result.zero_point_energy + magnon_free_energy(result, t)
    kelvin = t / K_BOLTZMANN_MEV
    internal, entropy, heat = [], [], []
    bosons, moments, magnetization = [], [], []
    for tk, T in zip(t, kelvin):
        internal.append(result.classical_energy
                        + thermo.compute_internal_energy(k_data, Temperature=T))
        entropy.append(thermo.compute_entropy_density(k_data, T) / K_BOLTZMANN_MEV)
        heat.append(thermo.compute_specific_heat(k_data, T) / K_BOLTZMANN_MEV)
        if gapless_flag and tk > 0:
            bosons.append(np.full(ns, np.nan))
        else:
            per_k = np.array([compute_bosonic_number_at_k(E, V, Temperature=T, num_sl=ns)[0]
                              for E, V in zip(result.eigenvalues, result.eigenvectors)])
            bosons.append(result.weights @ per_k)
        m = result.spins - bosons[-1]
        moments.append(m)
        magnetization.append(m @ result.directions / ns)
    bosons = np.array(bosons)
    beyond = np.any(bosons > result.spins, axis=1)
    if np.any(beyond):
        warnings.warn(f"boson numbers exceed S at t = {t[beyond].tolist()}: LSWT is not valid "
                      "there (moments would reverse)", UserWarning, stacklevel=2)
    return ThermalResult(t, free, np.array(internal), np.array(entropy),
                         np.array(heat), bosons, np.array(moments),
                         np.array(magnetization), gapless_flag, decision, report.to_dict(), beyond)
