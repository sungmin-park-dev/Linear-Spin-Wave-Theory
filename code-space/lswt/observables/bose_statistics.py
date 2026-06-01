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
from lswt.config import (
    K_BOLTZMANN_MEV, H_BAR_MEV,
    DEFAULT_TEMPERATURE, DEFAULT_TIME, DEFAULT_OMEGA, DEFAULT_ETA,
    BETA_E_THRESHOLD, BETA_E_SMALL,
)


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
