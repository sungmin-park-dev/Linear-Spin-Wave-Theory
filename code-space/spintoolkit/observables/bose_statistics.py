"""Bose-Einstein statistics and magnon kernel functions.

This module provides functions for computing Bose-Einstein distributions
and various magnon kernels used in spin wave theory calculations:
- Static magnon kernel (zero and finite temperature)
- Real-time kernel (time-dependent correlations)
- Lorentzian kernel (frequency-domain broadening)
- Spectral kernel (spectral function calculations)

Ported from modules/Tools/magnon_kernel.py.
"""

import numpy as np
from typing import Union, Optional
from spintoolkit.definitions import (
    K_BOLTZMANN_MEV, H_BAR_MEV,
    DEFAULT_TEMPERATURE, DEFAULT_TIME, DEFAULT_OMEGA, DEFAULT_ETA,
    BETA_E_THRESHOLD, BETA_E_SMALL,
)


# ---------------------------------------------------------------------------
# Dimensionless forms (D20): energies in E0, ``t = k_B T / E0``, ``k_B = 1``.
# The Kelvin-based functions below serve the pre-0.2 API.
# ---------------------------------------------------------------------------

def _check(energy, t: float) -> np.ndarray:
    if not np.isfinite(t) or t < 0:
        raise ValueError("t must be finite and non-negative")
    energy = np.asarray(energy, dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(energy < 0):
        raise ValueError("magnon energies must be finite and non-negative")
    return energy


def bose_occupation(energy, t: float) -> np.ndarray:
    """``n_B = 1 / (exp(e / t) - 1)``; zero at ``t = 0``, ``+inf`` for an exact zero mode at ``t > 0``."""
    energy = _check(energy, t)
    if t == 0:
        return np.zeros_like(energy)
    with np.errstate(divide="ignore", over="ignore"):
        return 1.0 / np.expm1(energy / t)


def bose_free_energy(energy, t: float) -> np.ndarray:
    """``t ln(1 - exp(-e / t))`` per mode; zero at ``t = 0``, ``-inf`` for an exact zero mode."""
    energy = _check(energy, t)
    if t == 0:
        return np.zeros_like(energy)
    x = energy / t
    out = np.empty_like(x)
    small = x <= np.log(2)
    # expm1 avoids cancellation near zero; log1p keeps exponentially small terms at large x.
    with np.errstate(divide="ignore"):
        out[small] = np.log(-np.expm1(-x[small]))
    out[~small] = np.log1p(-np.exp(-x[~small]))
    return t * out


def bose_entropy(energy, t: float) -> np.ndarray:
    """``(1 + n) ln(1 + n) - n ln n = x n - ln(1 - e^-x)`` per mode, ``x = e / t``; zero at ``t = 0``."""
    energy = _check(energy, t)
    if t == 0:
        return np.zeros_like(energy)
    x = energy / t
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = x / np.expm1(x) - np.log(-np.expm1(-x))
    out[x == 0] = np.inf
    return out


def bose_heat_capacity(energy, t: float) -> np.ndarray:
    """``x^2 / (4 sinh^2(x / 2))`` per mode, ``x = e / t`` (1 at ``x -> 0``); zero at ``t = 0``."""
    energy = _check(energy, t)
    if t == 0:
        return np.zeros_like(energy)
    x = energy / t
    with np.errstate(over="ignore", invalid="ignore"):
        out = (x / (2 * np.sinh(x / 2))) ** 2
    out[x == 0] = 1.0
    return np.where(np.isfinite(out), out, 0.0)


def compute_bose_einstein_distribution(E_list, Temperature=DEFAULT_TEMPERATURE):
    """Compute the Bose-Einstein distribution n_B(E) = 1/(exp(E/kT) - 1).

    For large beta*E (>= BETA_E_THRESHOLD), returns 0 (exponentially suppressed).
    For small beta*E (< BETA_E_SMALL), uses the approximation n_B ~ kT/E.

    Parameters
    ----------
    E_list : np.ndarray
        Array of energies in meV.
    Temperature : float, optional
        Temperature in Kelvin (default: 0).

    Returns
    -------
    BE_distribution : np.ndarray
        Bose-Einstein occupation numbers for each energy.
    """
    beta_E = E_list / (K_BOLTZMANN_MEV * Temperature)
    BE_distrbution = np.zeros_like(E_list, dtype=float)

    mask_small = beta_E < BETA_E_SMALL
    mask = (beta_E < BETA_E_THRESHOLD) & (~mask_small)

    if np.any(mask):
        exp_be = np.exp(beta_E[mask])
        BE_distrbution[mask] = 1.0 / (exp_be - 1.0)

    if np.any(mask_small):
        BE_distrbution[mask_small] = 1.0 / beta_E[mask_small]

    return BE_distrbution


def compute_static_magnon_kernel(E_list, Temperature=DEFAULT_TEMPERATURE, Ns=None):
    """Compute the static magnon kernel including normal ordering and thermal occupation.

    At T=0, returns [1, 1, ..., 0, 0, ...] (normal ordering for particle/hole sectors).
    At T>0, adds the Bose-Einstein distribution on top.

    Parameters
    ----------
    E_list : np.ndarray
        Array of magnon energies, length 2*Ns (particle + hole sectors).
    Temperature : float, optional
        Temperature in Kelvin (default: 0).
    Ns : int, optional
        Number of sublattices. If None, inferred as len(E_list)//2.

    Returns
    -------
    kernel : np.ndarray
        Static magnon kernel array of same length as E_list.

    Raises
    ------
    ValueError
        If Temperature is negative.
    """
    Ns = int(len(E_list) // 2) if Ns is None else Ns
    normal_ordering = np.hstack([np.ones(Ns), np.zeros(Ns)])

    if Temperature == 0:
        return normal_ordering
    elif Temperature > 0:
        return normal_ordering + compute_bose_einstein_distribution(E_list, Temperature)
    else:
        raise ValueError("Temperature must be non-negative")


def compute_real_time_kernel(E_list, time=DEFAULT_TIME, Temperature=DEFAULT_TEMPERATURE,
                             static_corr=None, Ns=None):
    """Compute the real-time magnon kernel for time-dependent correlations.

    Multiplies the static kernel by the time evolution factor exp(-i*E*t),
    with appropriate sign conventions for particle and hole sectors.

    Parameters
    ----------
    E_list : np.ndarray
        Array of magnon energies, length 2*Ns.
    time : float, optional
        Time in units consistent with hbar/meV (default: 0).
    Temperature : float, optional
        Temperature in Kelvin (default: 0).
    static_corr : np.ndarray, optional
        Pre-computed static kernel. If None, computed internally.
    Ns : int, optional
        Number of sublattices. If None, inferred as len(E_list)//2.

    Returns
    -------
    kernel : np.ndarray
        Real-time magnon kernel (complex-valued).

    Raises
    ------
    ValueError
        If time is not a scalar number.
    """
    Ns = int(len(E_list) // 2) if Ns is None else Ns

    if static_corr is None:
        static_corr = compute_static_magnon_kernel(E_list, Temperature, Ns)

    if time == 0:
        return static_corr
    elif isinstance(time, (float, int, np.number)):
        # Particle sector: +E_k, Hole sector: -E_k (sign convention)
        magnon_energy = np.hstack([E_list[:Ns], -E_list[Ns:]])
        exp_miEt = np.exp(-1j * time * magnon_energy)
        return static_corr * exp_miEt
    else:
        raise ValueError(f"Time should be a number, received {time}")


def g_propagator(E_array, omega, eta):
    """Compute the retarded Green's function propagator.

    G(E, omega, eta) = 1 / (eta + i*(E - omega))

    Parameters
    ----------
    E_array : np.ndarray
        Energy array.
    omega : float
        Probe frequency.
    eta : float
        Broadening parameter (Lorentzian half-width).

    Returns
    -------
    G : np.ndarray
        Complex Green's function values.
    """
    return 1 / (eta + 1j * (E_array - omega))


def compute_lorentzian_kernel(E_list, omega=DEFAULT_OMEGA, eta=DEFAULT_ETA,
                              Temperature=DEFAULT_TEMPERATURE, static_corr=None, Ns=None):
    """Compute the Lorentzian-broadened magnon kernel.

    Uses the real part of the Green's function to produce Lorentzian line shapes
    centered at the magnon energies.

    Parameters
    ----------
    E_list : np.ndarray
        Array of magnon energies, length 2*Ns.
    omega : float, optional
        Probe frequency in meV (default: 0).
    eta : float, optional
        Lorentzian broadening in meV (default: 1e-3).
    Temperature : float, optional
        Temperature in Kelvin (default: 0).
    static_corr : np.ndarray, optional
        Pre-computed static kernel. If None, computed internally.
    Ns : int, optional
        Number of sublattices. If None, inferred as len(E_list)//2.

    Returns
    -------
    kernel : np.ndarray
        Lorentzian-broadened magnon kernel (real-valued).
    """
    Ns = int(len(E_list) // 2) if Ns is None else Ns

    if static_corr is None:
        static_corr = compute_static_magnon_kernel(E_list, Temperature, Ns)

    Epk = E_list[:Ns]
    Emk = E_list[Ns:]

    Fpk = 2 * np.real(g_propagator(Epk, omega, eta))
    Fmk = 2 * np.real(g_propagator(Emk, -omega, eta))

    kernel_F = np.hstack([Fpk, Fmk])
    return static_corr * kernel_F


def compute_spectral_kernel(E_list, omega=DEFAULT_OMEGA, eta=DEFAULT_ETA,
                            Temperature=DEFAULT_TEMPERATURE, static_corr=None, Ns=None):
    """Compute the spectral magnon kernel for dynamical structure factor calculations.

    Uses the antisymmetric combination of Green's functions to produce the spectral
    weight, suitable for computing S(k, omega).

    Parameters
    ----------
    E_list : np.ndarray
        Array of magnon energies, length 2*Ns.
    omega : float, optional
        Probe frequency in meV (default: 0).
    eta : float, optional
        Lorentzian broadening in meV (default: 1e-3).
    Temperature : float, optional
        Temperature in Kelvin (default: 0).
    static_corr : np.ndarray, optional
        Pre-computed static kernel. If None, computed internally.
    Ns : int, optional
        Number of sublattices. If None, inferred as len(E_list)//2.

    Returns
    -------
    kernel : np.ndarray
        Spectral magnon kernel (complex-valued).
    """
    Ns = int(len(E_list) // 2) if Ns is None else Ns

    if static_corr is None:
        static_corr = compute_static_magnon_kernel(E_list, Temperature, Ns)

    Epk = E_list[:Ns]
    Emk = E_list[Ns:]

    Gpk = 1 / 2j * (g_propagator(Epk, omega, eta) + g_propagator(Epk, -omega, eta))
    Gmk = 1 / 2j * (g_propagator(Emk, omega, eta) + g_propagator(Emk, -omega, eta))

    kernel_G = np.hstack([Gpk, Gmk.conj()])
    return static_corr * kernel_G
