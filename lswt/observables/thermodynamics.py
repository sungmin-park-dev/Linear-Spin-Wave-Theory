"""Thermodynamic quantities from linear spin wave theory.

This module computes thermodynamic observables including internal energy,
entropy, specific heat, boson occupation numbers, and thermal Hall
conductance within the LSWT framework.

Ported from modules/LinearSpinWaveTheory/lswt_thermodynamics.py.
"""

import numpy as np
from typing import Union, Tuple
from tqdm import tqdm

from lswt.observables.bose_statistics import (
    compute_bose_einstein_distribution,
    compute_static_magnon_kernel,
)
from lswt.observables.topology import compute_berry_curvature, c_two_function
from lswt.config import (
    K_BOLTZMANN_MEV, H_BAR_MEV,
    DEFAULT_INVALID_EXCLUDE, DEFAULT_TEMPERATURE,
)


def compute_bosonic_number_at_k(eval, evec, Temperature=0, num_sl=None):
    """Compute boson occupation numbers at a single k-point.

    Parameters
    ----------
    eval : np.ndarray
        Eigenvalues from Bogoliubov diagonalization, length 2*num_sl.
    evec : np.ndarray
        Eigenvector matrix from Bogoliubov diagonalization.
    Temperature : float, optional
        Temperature in Kelvin (default: 0).
    num_sl : int or None, optional
        Number of sublattices. If None, inferred as len(eval)//2.

    Returns
    -------
    sublattice_boson_numbers_at_k : np.ndarray
        Boson numbers per sublattice at this k-point.
    all_boson_number_at_k : float
        Average boson number across sublattices at this k-point.
    """
    num_sl = int(len(eval) // 2) if num_sl is None else num_sl

    magnon_corr = np.diag(compute_static_magnon_kernel(eval, Temperature=Temperature))
    boson_corr = evec @ magnon_corr @ evec.T.conj()
    diag_elements = np.diag(boson_corr)

    sublattice_boson_numbers_at_k = np.real(diag_elements[num_sl:])
    all_boson_number_at_k = np.sum(sublattice_boson_numbers_at_k) / num_sl

    return sublattice_boson_numbers_at_k, all_boson_number_at_k


class Thermodynamics:
    """Thermodynamic calculations for spin wave systems.

    Parameters
    ----------
    lswt_obj : object or None, optional
        Parent LSWT solver object providing Ns (number of sublattices).
    """

    def __init__(self, lswt_obj=None):
        self.lswt_obj = lswt_obj
        self.Ns = lswt_obj.Ns if lswt_obj is not None else None

    def _infer_Ns_from_k_data(self, k_data):
        """Infer the number of sublattices from k_data.

        Parameters
        ----------
        k_data : dict
            Dictionary of k-point data.

        Returns
        -------
        num_sl : int
            Number of sublattices.

        Raises
        ------
        ValueError
            If k_data is empty.
        """
        if not k_data:
            raise ValueError("k_data is empty; cannot infer Ns.")
        first_k_data = next(iter(k_data.values()))
        num_sl = len(first_k_data[1][0]) // 2
        return num_sl

    def compute_internal_energy(self, k_data,
                                Temperature=DEFAULT_TEMPERATURE,
                                invalid_exclude=DEFAULT_INVALID_EXCLUDE):
        """Compute the quantum contribution of internal energy per unit cell.

        Parameters
        ----------
        k_data : dict
            Dictionary of k-point data.
        Temperature : float, optional
            Temperature in Kelvin (default: 0).
        invalid_exclude : bool, optional
            Whether to exclude k-points where Colpa's method fails.

        Returns
        -------
        float
            Internal energy per site, or np.nan if no valid k-points.
        """
        num_sl = self.Ns if self.Ns is not None else self._infer_Ns_from_k_data(k_data)
        iter_count = len(k_data)
        E_quantum = 0

        for k_key, contents in k_data.items():
            Ham_k_data, Eigen_data, Colpa_data, *_ = contents
            colpa_success = Colpa_data[0]

            if colpa_success or not invalid_exclude:
                Hk = Ham_k_data[0]
                Epk = Eigen_data[0][:num_sl]

                E_quantum += self.zero_point_energy_formula(Epk, Hk)

                if Temperature > 0:
                    E_quantum += self.excitation_energy_formula(Epk, Temperature)
            else:
                iter_count -= self._handle_colpa_failure(kpt=k_key)
                continue

        if iter_count == 0:
            return np.nan
        else:
            denominator = iter_count * num_sl
            return E_quantum / denominator

    def bosonic_momentum_correlation(self, k_data, Temperature=0, exclude_gamma=False):
        """Compute boson numbers for each k-point.

        Parameters
        ----------
        k_data : dict
            Dictionary of k-point data.
        Temperature : float, optional
            Temperature in Kelvin (default: 0).
        exclude_gamma : bool, optional
            Whether to exclude the gamma point (default: False).

        Returns
        -------
        msl_boson_numbers : np.ndarray
            Sublattice boson numbers, shape (num_k_pts, num_sl).
        all_boson_numbers : np.ndarray
            Average boson numbers, shape (num_k_pts,).
        eigenvalues_array : np.ndarray
            Eigenvalues array, shape (num_k_pts, 2*num_sl).
        iter_count : int
            Number of valid k-points.
        """
        num_sl = self.Ns if self.Ns is not None else self._infer_Ns_from_k_data(k_data)

        num_k_pts = len(k_data)
        all_boson_numbers = np.zeros(num_k_pts)
        msl_boson_numbers = np.zeros((num_k_pts, num_sl))

        eigenvalues_array = np.zeros((num_k_pts, 2 * num_sl))
        iter_count = len(k_data)

        for j, k_key in enumerate(k_data.keys()):
            _, Eigen_data, Colpa_k_data, *_ = k_data[k_key]
            eval, evec = Eigen_data
            colpa_success, _ = Colpa_k_data
            eigenvalues_array[j] = eval

            if colpa_success:
                msl_boson_numbers[j], all_boson_numbers[j] = compute_bosonic_number_at_k(
                    eval=eval, evec=evec, Temperature=Temperature, num_sl=num_sl
                )
            else:
                iter_count -= self._handle_colpa_failure(k_key)

        return msl_boson_numbers, all_boson_numbers, eigenvalues_array, iter_count

    def compute_boson_numbers(self, k_data, Temperature=0, exclude_gamma=False):
        """Compute average boson numbers across the Brillouin zone.

        Parameters
        ----------
        k_data : dict
            Dictionary of k-point data.
        Temperature : float, optional
            Temperature in Kelvin (default: 0).
        exclude_gamma : bool, optional
            Whether to exclude the gamma point (default: False).

        Returns
        -------
        average_sublat_boson_numbers : np.ndarray
            Average sublattice boson numbers, or np.nan if invalid.
        average_total_boson_numbers : float
            Average total boson number, or np.nan if invalid.
        """
        sublat_num, total_num, _, valid_counts = self.bosonic_momentum_correlation(
            k_data=k_data, Temperature=Temperature, exclude_gamma=exclude_gamma
        )

        if valid_counts == 0:
            return np.nan, np.nan

        average_total_boson_numbers = np.sum(total_num) / valid_counts
        average_sublat_boson_numbers = np.sum(sublat_num, axis=0) / valid_counts

        return average_sublat_boson_numbers, average_total_boson_numbers

    def compute_entropy_density(self, k_data, T, invalid_exclude=True):
        """Compute entropy density.

        Parameters
        ----------
        k_data : dict
            Dictionary of k-point data.
        T : float
            Temperature in Kelvin.
        invalid_exclude : bool, optional
            Whether to exclude invalid k-points (default: True).

        Returns
        -------
        float
            Entropy density in meV/K.

        Raises
        ------
        ValueError
            If T is negative.
        """
        num_sl = self.Ns if self.Ns is not None else self._infer_Ns_from_k_data(k_data)

        entropy = 0
        valid_count = len(k_data)

        if T == 0:
            return 0
        elif T > 0:
            for k_key, contents in k_data.items():
                _, Eigen_data, Colpa_data, *_ = contents

                if invalid_exclude and not Colpa_data[0]:
                    valid_count -= self._handle_colpa_failure(k_key)
                    continue
                else:
                    eval, _ = Eigen_data
                    Epk = eval[:num_sl]
                    nk = compute_bose_einstein_distribution(Epk, T)
                    entropy += self.entropy_function_at_k(nk)

            if valid_count == 0:
                return np.nan
            return K_BOLTZMANN_MEV * entropy / valid_count
        else:
            raise ValueError("T is temperature. T should be positive number")

    def compute_specific_heat(self, k_data, T, invalid_exclude=True, exclude_gamma=True):
        """Compute specific heat.

        Parameters
        ----------
        k_data : dict
            Dictionary of k-point data.
        T : float
            Temperature in Kelvin.
        invalid_exclude : bool, optional
            Whether to exclude invalid k-points (default: True).
        exclude_gamma : bool, optional
            Whether to exclude the gamma point (default: True).

        Returns
        -------
        float
            Specific heat in meV/K.

        Raises
        ------
        ValueError
            If T is negative.
        """
        num_sl = self.Ns if self.Ns is not None else self._infer_Ns_from_k_data(k_data)

        Cv = 0
        valid_count = len(k_data)

        if T == 0:
            return Cv
        elif T > 0:
            beta = 1 / (K_BOLTZMANN_MEV * T)
            for k_key, contents in k_data.items():
                _, Eigen_data, Colpa_data, *_ = contents

                if invalid_exclude and not Colpa_data[0]:
                    valid_count -= self._handle_colpa_failure(k_key)
                    continue
                else:
                    eval, _ = Eigen_data
                    Epk = eval[:num_sl]
                    nk = compute_bose_einstein_distribution(Epk, T)
                    Cv += self.specific_heat_function_at_k(Epk=Epk, beta=beta)

            if valid_count == 0:
                return np.nan
            return K_BOLTZMANN_MEV * Cv / valid_count
        else:
            raise ValueError("T is temperature. T should be positive number")

    @staticmethod
    def _handle_colpa_failure(kpt=None):
        """Handle Colpa's method failure at a k-point.

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

    @staticmethod
    def zero_point_energy_formula(Epk, Hk):
        """Compute zero-point energy contribution.

        Parameters
        ----------
        Epk : np.ndarray
            Positive magnon eigenvalues.
        Hk : np.ndarray
            Bosonic Hamiltonian matrix at k.

        Returns
        -------
        float
            Zero-point energy contribution.
        """
        E_magnon = np.sum(Epk)
        Trace_Hk = np.real(np.trace(Hk))
        return (E_magnon - Trace_Hk / 2) / 2

    @staticmethod
    def excitation_energy_formula(Epk, T):
        """Compute thermal excitation energy.

        Parameters
        ----------
        Epk : np.ndarray
            Positive magnon eigenvalues.
        T : float
            Temperature in Kelvin.

        Returns
        -------
        float
            Thermal excitation energy.
        """
        numbers = compute_bose_einstein_distribution(Epk, T)
        E_excitation = Epk * numbers
        return np.sum(E_excitation)

    @staticmethod
    def entropy_function_at_k(nk):
        """Compute entropy contribution at a single k-point.

        Parameters
        ----------
        nk : np.ndarray
            Bose-Einstein occupation numbers.

        Returns
        -------
        float
            Entropy contribution (without Boltzmann constant factor).
        """
        x = np.array(nk)
        f = np.zeros_like(nk, dtype=float)

        mask_too_small = x < 1e-323
        mask_too_large = x > 1e300
        normal_range = ~mask_too_small & ~mask_too_large

        cal_x = x[normal_range]
        if cal_x.size > 0:
            I_p_cal_x = 1 + cal_x
            f[normal_range] = I_p_cal_x * np.log(I_p_cal_x) - cal_x * np.log(cal_x)
        return np.sum(f)

    @staticmethod
    def specific_heat_function_at_k(Epk, beta):
        """Compute specific heat contribution at a single k-point.

        Parameters
        ----------
        Epk : np.ndarray
            Positive magnon eigenvalues.
        beta : float
            Inverse temperature 1/(k_B * T).

        Returns
        -------
        float
            Specific heat contribution (without Boltzmann constant factor).
        """
        x = np.array(beta * Epk)
        f = np.zeros_like(x, dtype=float)

        mask_too_large = x > 680
        mask_too_small = x < 1e-150
        normal_range = ~mask_too_small & ~mask_too_large

        cal_x = x[normal_range]
        if cal_x.size > 0:
            denominator = 4 * (np.sinh(cal_x / 2)) ** 2
            numerator = cal_x ** 2
            f[normal_range] = numerator / denominator

        small_x = x[mask_too_small]
        if small_x.size > 0:
            f[mask_too_small] = 1.0

        return np.sum(f)

    def compute_thermodynamic_quantities_at_T(self,
                                              k_data,
                                              Temperature=DEFAULT_TEMPERATURE,
                                              invalid_exclude=DEFAULT_INVALID_EXCLUDE):
        """Compute all thermodynamic quantities at a given temperature in a single pass.

        Parameters
        ----------
        k_data : dict
            Dictionary containing k-point data with eigenvalues and eigenvectors.
        Temperature : float, optional
            Temperature in Kelvin (default: 0).
        invalid_exclude : bool, optional
            Whether to exclude k-points where Colpa's method fails (default: True).

        Returns
        -------
        dict
            Dictionary containing:
            - 'Sublattice Boson Numbers': Average boson numbers for each sublattice
            - 'Total Boson Number': Average total boson number
            - 'Internal Energy Density': Internal energy per unit cell
            - 'Entropy Density': Entropy density
            - 'Specific Heat Density': Specific heat
            - 'Thermal Hall Conductance': Thermal Hall conductance
        """
        num_sl = self.Ns if self.Ns is not None else self._infer_Ns_from_k_data(k_data)

        valid_count = len(k_data)

        total_boson_number = 0
        sublattice_boson_numbers = np.zeros(num_sl)
        internal_energy = 0
        entropy = 0
        specific_heat = 0
        thermal_hall = 0
        J_mat = np.diag(np.hstack([np.ones((self.Ns)), -np.ones((self.Ns))]))

        beta = 1 / (K_BOLTZMANN_MEV * Temperature) if Temperature > 0 else 0

        for k_key, contents in k_data.items():
            Ham_k_data, Eigen_data, Colpa_data, *_ = contents
            colpa_success = Colpa_data[0]

            if not colpa_success and invalid_exclude:
                valid_count -= self._handle_colpa_failure(kpt=k_key)
                continue
            else:
                Hk = Ham_k_data[0]
                pDHk = Ham_k_data[1:]

                eval, evec = Eigen_data
                Epk = eval[:num_sl]

                # 1. Zero-point energy contribution
                internal_energy += self.zero_point_energy_formula(Epk=Epk, Hk=Hk)

                if Temperature > 0:
                    nk = compute_bose_einstein_distribution(Epk, Temperature)

                    # 2. Thermal excitation energy
                    E_excitation = np.sum(Epk * nk)
                    internal_energy += E_excitation

                    # 3. Entropy
                    entropy += self.entropy_function_at_k(nk)

                    # 4. Specific heat
                    specific_heat += self.specific_heat_function_at_k(Epk=Epk, beta=beta)

                    # 5. Thermal Hall conductance
                    Omega_nk, _ = compute_berry_curvature(
                        eval=eval,
                        evec=evec,
                        pDiffHk=pDHk,
                        num_sl=self.Ns,
                        J_mat=J_mat,
                    )

                    thermal_hall += np.sum(Omega_nk * c_two_function(nk))

                # 6. Boson numbers
                sl_boson_nums, total_boson_num = compute_bosonic_number_at_k(
                    eval=eval, evec=evec, Temperature=Temperature, num_sl=num_sl
                )
                sublattice_boson_numbers += sl_boson_nums
                total_boson_number += total_boson_num

        if valid_count == 0:
            return {
                'Sublattice Boson Numbers': np.nan,
                'Total Boson Number': np.nan,
                'Internal Energy Density': np.nan,
                'Entropy Density': np.nan,
                'Specific Heat Density': np.nan,
                'Thermal Hall Conductance': np.nan,
            }
        else:
            number_of_lattices = valid_count * num_sl

            sublattice_boson_numbers /= number_of_lattices
            total_boson_number /= number_of_lattices
            internal_energy /= number_of_lattices
            entropy /= number_of_lattices
            specific_heat /= number_of_lattices

            real_space_volume = valid_count * np.sqrt(3) / 2
            coefficiten_thc = (K_BOLTZMANN_MEV ** 2) / (H_BAR_MEV * real_space_volume)
            coefficiten_thc *= 1.602176634 * 1e-12  # from meV, Angstrom to W/(m*K)

            thermal_hall_conductance = -coefficiten_thc * thermal_hall

            if Temperature > 0:
                entropy *= K_BOLTZMANN_MEV
                specific_heat *= K_BOLTZMANN_MEV

            return {
                'Sublattice Boson Numbers': sublattice_boson_numbers,
                'Total Boson Number': total_boson_number,
                'Internal Energy Density': internal_energy,
                'Entropy Density': entropy,
                'Specific Heat Density': specific_heat,
                'Thermal Hall Conductance': thermal_hall_conductance,
            }

    def get_thermodynamic_quantities(self, k_data,
                                     Temperature_range=(0, 1, 0.02),
                                     N=30):
        """Compute thermodynamic quantities over a temperature range.

        Parameters
        ----------
        k_data : dict
            Dictionary of k-point data.
        Temperature_range : tuple, optional
            (T_start, T_end, T_step) in Kelvin (default: (0, 1, 0.02)).
        N : int, optional
            Unused parameter kept for compatibility (default: 30).

        Returns
        -------
        Temperature_values : np.ndarray
            Array of temperature values.
        results : dict
            Dictionary of thermodynamic quantities vs temperature.
        """
        T_start, T_end, T_step = Temperature_range
        T_end += T_step / 2

        Temperature_values = np.arange(T_start, T_end, T_step)

        num_T = len(Temperature_values)

        sublattice_boson_numbers = np.empty((self.Ns, num_T))
        total_boson_number = np.empty(num_T)
        internal_energy = np.empty(num_T)
        entropy = np.empty(num_T)
        specific_heat = np.empty(num_T)
        thermal_hall_conductance = np.empty(num_T)

        for j, T in enumerate(tqdm(Temperature_values, desc="Calculating thermodynamics")):
            result_T = self.compute_thermodynamic_quantities_at_T(
                k_data=k_data, Temperature=T
            )

            sublattice_boson_numbers[:, j] = result_T['Sublattice Boson Numbers']
            total_boson_number[j] = result_T['Total Boson Number']
            internal_energy[j] = result_T['Internal Energy Density']
            entropy[j] = result_T['Entropy Density']
            specific_heat[j] = result_T['Specific Heat Density']
            thermal_hall_conductance[j] = result_T['Thermal Hall Conductance']

        results = {
            'Internal Energy Density': internal_energy,
            'Entropy Density': entropy,
            'Specific Heat Density': specific_heat,
            'Thermal Hall Conductance': thermal_hall_conductance,
            'Sublattice Boson Numbers': sublattice_boson_numbers,
            'Total Boson Number': total_boson_number,
        }

        return Temperature_values, results
