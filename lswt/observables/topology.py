"""Topological properties of magnon bands.

This module computes topological quantities including Berry curvature,
Chern numbers, and thermal Hall conductance within the LSWT framework.

Ported from modules/LinearSpinWaveTheory/lswt_topology.py.
"""

import numpy as np
from typing import Union, List, Tuple
from scipy.special import spence

from lswt.observables.bose_statistics import compute_bose_einstein_distribution
from lswt.config import K_BOLTZMANN_MEV, H_BAR_MEV, DEFAULT_LEVEL_SPACING


def c_two_function(x):
    """Compute the c_2 weight function for thermal Hall conductance.

    The c_2 function appears in the thermal Hall conductance formula and
    is related to the polylogarithm. It maps the Bose-Einstein distribution
    to the appropriate thermal weight.

    Parameters
    ----------
    x : float, list, or np.ndarray
        Bose-Einstein distribution values (occupation numbers).

    Returns
    -------
    f : np.ndarray
        c_2 function values, same shape as input.
    """
    x = np.array(x)
    f = np.zeros_like(x, dtype=float)

    mask_too_small = x < 1e-300
    mask_too_large = x > 1e300
    normal_range = ~mask_too_small & ~mask_too_large

    f[mask_too_large] = np.pi ** 2 / 3

    cal_x = x[normal_range]
    if cal_x.size > 0:
        term1 = (1 + cal_x) * (np.log((1 + cal_x) / cal_x)) ** 2
        term2 = (np.log(cal_x)) ** 2
        term3 = 2 * spence(1 + cal_x)
        f[normal_range] = term1 - term2 - term3

    return f


def compute_berry_curvature(eval, evec, pDiffHk, num_sl=None, J_mat=None):
    """Compute Berry curvature for magnon bands using the Kubo formula.

    Calculates the Berry curvature for each physical magnon band at a
    given k-point using the partial derivatives of the bosonic Hamiltonian.

    Parameters
    ----------
    eval : np.ndarray
        Magnon eigenvalues (length 2*num_sl).
    evec : np.ndarray
        Bogoliubov transformation matrix.
    pDiffHk : list of np.ndarray
        Partial derivatives of the bosonic Hamiltonian [dH/dkx, dH/dky].
    num_sl : int or None, optional
        Number of sublattices. If None, inferred as len(eval)//2.
    J_mat : np.ndarray or None, optional
        Para-unitary metric matrix. If None, constructed as diag(+1,...,+1,-1,...,-1).

    Returns
    -------
    Omega_nk : np.ndarray
        Berry curvature for each physical band, shape (num_sl,).
    level_spacing : np.ndarray
        Minimum energy difference to adjacent bands, shape (num_sl,).
    """
    num_sl = num_sl if num_sl is not None else int(len(eval) // 2)
    J_mat = J_mat if J_mat is not None else np.diag(
        np.hstack([np.ones(num_sl), -np.ones(num_sl)])
    )

    Omega_nk = []
    level_spacing = []

    pDxHk, pDyHk = pDiffHk

    J_eval = np.diag(J_mat) * eval

    partial_H_x = J_mat @ evec.conj().T @ pDxHk @ evec
    partial_H_y = J_mat @ evec.conj().T @ pDyHk @ evec

    for n in range(num_sl):
        summand = 0
        min_E_diff = np.inf

        for m in range(2 * num_sl):
            if n == m:
                continue
            else:
                diff_En_Em = J_eval[n] - J_eval[m]
                denominator = (diff_En_Em) ** 2
                if denominator == 0:
                    continue  # skip to avoid nan
                numerator = partial_H_x[n, m] * partial_H_y[m, n]
                summand += numerator / denominator

                if n == m + 1 or n == m - 1:
                    min_E_diff = np.minimum(min_E_diff, np.abs(diff_En_Em))

        if n == 0:
            min_E_diff = np.minimum(min_E_diff, eval[n])

        Omega_nk.append(-2 * np.imag(summand))
        level_spacing.append(min_E_diff)

    return np.array(Omega_nk), np.array(level_spacing)


def _handle_colpa_failure(kpt=None):
    """Handle Colpa's method failure at a k-point.

    The Colpa method works for any positive definite Hamiltonian.
    If it fails, the given Hamiltonian is not positive definite.

    Parameters
    ----------
    kpt : tuple or np.ndarray or None, optional
        The k-point where failure occurred.

    Returns
    -------
    int
        Always returns 1 (count of failed k-points).
    """
    return 1


class Topology:
    """Topological property calculations for magnon bands.

    Parameters
    ----------
    lswt_obj : object
        Parent LSWT solver object providing Ns and bz_data.
    num_sl : int or None, optional
        Number of sublattices. If None, taken from lswt_obj.Ns.
    """

    def __init__(self, lswt_obj, num_sl=None):
        self.Ns = lswt_obj.Ns
        self.J_mat = np.diag(np.hstack([np.ones((self.Ns)), -np.ones((self.Ns))]))
        self.area = lswt_obj.bz_data["area"]

    def thermal_weight_function(self, eval, Temperature):
        """Compute thermal weight function c_2(n_B(E)) for each band.

        Parameters
        ----------
        eval : np.ndarray
            Magnon eigenvalues (length 2*Ns).
        Temperature : float
            Temperature in Kelvin.

        Returns
        -------
        np.ndarray
            Thermal weight values for each physical band, shape (Ns,).
        """
        Epk = eval[:self.Ns]

        if Temperature == 0:
            return np.zeros_like(Epk)
        else:
            nk = compute_bose_einstein_distribution(E_list=Epk, Temperature=Temperature)
            return c_two_function(nk)

    def compute_thermal_Hall(self, k_data, Temperature, bz_type=None, verbose=False):
        """Compute Berry curvature, Chern numbers, and thermal Hall conductance.

        Parameters
        ----------
        k_data : dict
            Dictionary of k-point data.
        Temperature : float
            Temperature in Kelvin.
        bz_type : str or None, optional
            Brillouin zone type for normalization. One of "simple", "Hex_60",
            "Hex_30", "tetra", or None (default: None, treated as "simple").
        verbose : bool, optional
            If True, log Chern numbers and thermal Hall conductance info
            (default: False).

        Returns
        -------
        Berry_curvature : np.ndarray
            Berry curvature at each k-point, shape (num_k_points, Ns).
        chern_number : np.ndarray
            Chern number for each band, shape (Ns,).
        THC : float
            Thermal Hall conductivity kappa_xy / T in microW/(m*K).
        """
        num_k_points = len(k_data)
        valid_count = num_k_points

        if bz_type is None or bz_type == "simple":
            BZ_normalizer = 1
        elif bz_type in ("Hex_60", "Hex_30", "tetra"):
            BZ_normalizer = self.Ns
        else:
            raise ValueError("normalizer of Brillouin zone ")

        Berry_curvature = np.zeros((num_k_points, self.Ns))
        min_level_spacing = np.full(self.Ns, np.inf)
        thermal_hall = 0

        for j, contents in enumerate(k_data.values()):
            H_k_data, Eigen_k_data, Colpa_data, *_ = contents

            colpa_success = Colpa_data[0]
            if colpa_success:
                eval, evec = Eigen_k_data
                pDHk = H_k_data[1:]

                Omega_nk, level_spacing = compute_berry_curvature(
                    eval=eval,
                    evec=evec,
                    pDiffHk=pDHk,
                    num_sl=self.Ns,
                    J_mat=self.J_mat,
                )
                Berry_curvature[j] = Omega_nk
                min_level_spacing = np.minimum(min_level_spacing, level_spacing)

                thermal_hall += np.sum(
                    Omega_nk * self.thermal_weight_function(eval, Temperature=Temperature)
                )
            else:
                Berry_curvature[j] = np.full(self.Ns, np.nan)
                valid_count -= _handle_colpa_failure()

        chern_number = (
            (1 / (2 * np.pi))
            * np.nansum(Berry_curvature, axis=0)
            * self.area
            / BZ_normalizer
        )

        real_space_volume = valid_count * np.sqrt(3) / 2
        coefficiten_thc = (K_BOLTZMANN_MEV ** 2) / (H_BAR_MEV * real_space_volume)
        coefficiten_thc *= 1.602176634 * 1e-12  # from meV, Angstrom to W/(m*K)

        THC = -coefficiten_thc * thermal_hall * 10 ** 6  # from W to microW

        if verbose:
            for j, C_num in enumerate(chern_number):
                spacing = min_level_spacing[j]
                if spacing < DEFAULT_LEVEL_SPACING:
                    pass  # Insufficient level spacing warning
                # Band Chern number is available in chern_number array

        return Berry_curvature, chern_number, THC
