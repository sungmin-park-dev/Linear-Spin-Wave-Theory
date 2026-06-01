"""LSWT Hamiltonian construction and diagonalization.

This module provides the LSWTHamiltonian class for constructing the
quadratic bosonic Hamiltonian in k-space, computing partial derivatives
for Berry curvature calculations, and evaluating quantum energies and
free energies.

Ported from: modules/LinearSpinWaveTheory/lswt_Hamiltonian.py
"""

import numpy as np
from typing import Tuple, List, Dict

from lswt.observables.bose_statistics import compute_bose_einstein_distribution
from lswt.core.diagonalization import Diagonalizer
from lswt.config import (
    K_BOLTZMANN_MEV,
    HIGH_BE_THRESHOLD, LOW_BE_THRESHOLD, ZERO_ENERGY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Free energy helper functions
# ---------------------------------------------------------------------------

def log_1_m_exp(energy, temperature):
    """Compute ln(1 - e^(-beta*E)) / beta with numerical stability.

    Parameters
    ----------
    energy : array_like
        Magnon energies.
    temperature : float
        Temperature in energy units matching K_BOLTZMANN_MEV.

    Returns
    -------
    result : np.ndarray
        Element-wise values of ln(1 - e^(-beta*E)) / beta.
    """
    energy = np.asarray(energy)
    beta = 1.0 / (K_BOLTZMANN_MEV * temperature)
    beta_E = beta * energy
    result = np.zeros_like(beta_E, dtype=float)

    zero_mask = np.abs(energy) < ZERO_ENERGY_THRESHOLD
    result[zero_mask] = -np.inf

    valid_mask = ~zero_mask
    if not np.any(valid_mask):
        return result

    beta_E_valid = beta_E[valid_mask]
    energy_valid = energy[valid_mask] if energy.ndim > 0 else energy

    # Low beta*E regime: Taylor expansion
    low_be_mask = beta_E_valid < LOW_BE_THRESHOLD
    if np.any(low_be_mask):
        be_low = beta_E_valid[low_be_mask]
        e_low = energy_valid[low_be_mask] if energy_valid.ndim > 0 else energy_valid
        result[valid_mask][low_be_mask] = e_low * (
            np.log(be_low) / be_low - 0.5 - be_low / 24.0 - be_low**3 / 2880.0
        )

    # High beta*E regime: asymptotic expansion
    high_be_mask = beta_E_valid > HIGH_BE_THRESHOLD
    if np.any(high_be_mask):
        be_high = beta_E_valid[high_be_mask]
        e_high = energy_valid[high_be_mask] if energy_valid.ndim > 0 else energy_valid
        result[valid_mask][high_be_mask] = e_high * (-np.exp(-be_high) / be_high)

    # Medium beta*E regime: direct evaluation
    med_be_mask = ~(low_be_mask | high_be_mask)
    if np.any(med_be_mask):
        be_med = beta_E_valid[med_be_mask]
        e_med = energy_valid[med_be_mask] if energy_valid.ndim > 0 else energy_valid
        result[valid_mask][med_be_mask] = e_med * (np.log1p(-np.exp(-be_med)) / be_med)

    return result


def bosonic_free_energy(energies, temperature):
    """Compute bosonic free energy F = sum ln(1 - e^(-beta*E)) / beta.

    Parameters
    ----------
    energies : array_like
        Magnon energies.
    temperature : float
        Temperature.

    Returns
    -------
    free_energy : float
        Total bosonic free energy contribution.
    """
    energies = np.asarray(energies)
    terms = log_1_m_exp(energies, temperature)
    finite_mask = np.isfinite(terms)
    return np.sum(terms[finite_mask]) if np.any(finite_mask) else -np.inf


# ---------------------------------------------------------------------------
# Main Hamiltonian class
# ---------------------------------------------------------------------------

class LSWTHamiltonian:
    """Construct and solve the LSWT quadratic bosonic Hamiltonian.

    Parameters
    ----------
    spin_info : dict
        Sublattice spin information keyed by sublattice name.
        Each entry contains 'Spin', 'Angles', 'Magnetic Field', etc.
    couplings : list of dict
        List of coupling dictionaries, each containing 'SpinI', 'SpinJ',
        'Exchange Matrix', and 'Displacement'.
    """

    def __init__(self, spin_info, couplings):
        self.spin_info = spin_info
        self.couplings = couplings
        self.Ns = len(self.spin_info)

    @staticmethod
    def _classical_spin_rotation_matrix(pol_ang, azm_ang):
        """Build the rotation matrix from the lab frame to the local spin frame.

        Parameters
        ----------
        pol_ang : float
            Polar angle (theta).
        azm_ang : float
            Azimuthal angle (phi).

        Returns
        -------
        Rot_spin : np.ndarray, shape (3, 3)
            Rotation matrix.
        """
        Rot_spin = np.array([
            [np.cos(pol_ang) * np.cos(azm_ang), -np.sin(azm_ang),
             np.sin(pol_ang) * np.cos(azm_ang)],
            [np.cos(pol_ang) * np.sin(azm_ang),  np.cos(azm_ang),
             np.sin(pol_ang) * np.sin(azm_ang)],
            [-np.sin(pol_ang),                    0,
             np.cos(pol_ang)]
        ])
        return Rot_spin

    def get_rmat_dict(self, angles=None):
        """Compute rotation matrices for all sublattices.

        Parameters
        ----------
        angles : array_like or None, optional
            Flat array of [theta1, phi1, theta2, phi2, ...].
            If None, use angles from spin_info.

        Returns
        -------
        rmat_dict : dict
            Rotation matrices keyed by sublattice name.
        """
        rmat_dict = {}
        if angles is None:
            for name_sl, sl_dict in self.spin_info.items():
                theta, phi = sl_dict["Angles"]
                spin_rot_mat = self._classical_spin_rotation_matrix(theta, phi)
                rmat_dict[name_sl] = spin_rot_mat
        else:
            if len(angles) == 2 * len(self.spin_info) and isinstance(angles, (list, np.ndarray, tuple)):
                for j, name_sl in enumerate(self.spin_info.keys()):
                    theta = angles[2 * j]
                    phi = angles[2 * j + 1]
                    spin_rot_mat = self._classical_spin_rotation_matrix(theta, phi)
                    rmat_dict[name_sl] = spin_rot_mat
            else:
                raise ValueError(
                    f"Number of angle variables must be equal to {2 * self.Ns}"
                )
        return rmat_dict

    @staticmethod
    def get_couplings(coupling_dict, rmat_dict):
        """Rotate the exchange matrix into the local spin frame and
        transform to the Holstein-Primakoff boson basis.

        Parameters
        ----------
        coupling_dict : dict
            Single coupling entry with 'SpinI', 'SpinJ', 'Exchange Matrix'.
        rmat_dict : dict
            Rotation matrices from get_rmat_dict().

        Returns
        -------
        Hop_J : np.ndarray, shape (3, 3)
            Transformed coupling matrix in the (+, -, z) basis.
        """
        Ri = rmat_dict[coupling_dict["SpinI"]]
        Rj = rmat_dict[coupling_dict["SpinJ"]]
        RJ = Ri.T @ coupling_dict["Exchange Matrix"] @ Rj
        sqrt2 = np.sqrt(2)
        Cmat = np.array([
            [1 / sqrt2,  1 / sqrt2,  0],
            [1j / sqrt2, -1j / sqrt2, 0],
            [0,           0,          1]
        ])
        Hop_J = Cmat.T.conj() @ RJ @ Cmat
        return Hop_J

    def Quadratic_Bose_Hamiltonian(self, kpoints, angles=None):
        """Build the quadratic bosonic Hamiltonian H(k) for all k-points.

        Parameters
        ----------
        kpoints : np.ndarray, shape (m, d)
            Array of k-points.
        angles : array_like or None, optional
            Spin angles override.

        Returns
        -------
        K_Hamiltonian : np.ndarray, shape (m, 2*Ns, 2*Ns)
            Hamiltonian matrices for each k-point.
        linear_term_list : dict
            Linear terms keyed by sublattice name (vanish at equilibrium).
        """
        m = len(kpoints)
        self.K_Hamiltonian = np.zeros((m, 2 * self.Ns, 2 * self.Ns), dtype=complex)
        self.SL_idx = {}
        self.linear_term_list = {}
        self.Rmat_dict = self.get_rmat_dict(angles=angles)

        # Single-ion / magnetic field contributions
        for i, (sl_name, sl_dict) in enumerate(self.spin_info.items()):
            self.SL_idx[sl_name] = i
            Rhix, Rhiy, Rhiz = sl_dict["Magnetic Field"] @ self.Rmat_dict[sl_name]
            hfp = Rhix + 1j * Rhiy
            self.K_Hamiltonian[:, i, i] += Rhiz
            self.K_Hamiltonian[:, self.Ns + i, self.Ns + i] += Rhiz
            self.linear_term_list[sl_name] = -hfp

        # Exchange coupling contributions
        for coupling_dict in self.couplings:
            sli = coupling_dict["SpinI"]
            Spi = self.spin_info[sli]["Spin"]
            i = self.SL_idx[sli]

            slj = coupling_dict["SpinJ"]
            Spj = self.spin_info[slj]["Spin"]
            j = self.SL_idx[slj]

            delta = coupling_dict["Displacement"]
            exp_mDk = np.exp(-1j * np.dot(kpoints, delta))
            exp_pDk = exp_mDk.conj()

            hop_t = self.get_couplings(coupling_dict, self.Rmat_dict)
            tpm = np.sqrt(Spi * Spj) * hop_t[1, 1]
            tpp = np.sqrt(Spi * Spj) * hop_t[1, 0]
            t00 = hop_t[2, 2]
            t0p = hop_t[2, 0]
            tp0 = hop_t[0, 2]

            # Diagonal (on-site) terms
            self.K_Hamiltonian[:, i, i] -= t00 * Spj
            self.K_Hamiltonian[:, j, j] -= t00 * Spi
            self.K_Hamiltonian[:, self.Ns + i, self.Ns + i] -= t00 * Spj
            self.K_Hamiltonian[:, self.Ns + j, self.Ns + j] -= t00 * Spi

            # Normal hopping (particle-conserving)
            self.K_Hamiltonian[:, i, j] += tpm * exp_mDk
            self.K_Hamiltonian[:, j, i] += (tpm * exp_mDk).conj()
            self.K_Hamiltonian[:, self.Ns + i, self.Ns + j] += (tpm * exp_pDk).conj()
            self.K_Hamiltonian[:, self.Ns + j, self.Ns + i] += tpm * exp_pDk

            # Anomalous hopping (particle-nonconserving)
            # B block (upper-right: [0:Ns, Ns:2Ns])
            self.K_Hamiltonian[:, i, self.Ns + j] += tpp * exp_mDk
            self.K_Hamiltonian[:, j, self.Ns + i] += tpp * exp_pDk
            # B† block (lower-left: [Ns:2Ns, 0:Ns])
            self.K_Hamiltonian[:, self.Ns + j, i] += (tpp * exp_mDk).conj()
            self.K_Hamiltonian[:, self.Ns + i, j] += (tpp * exp_pDk).conj()

            # Linear terms
            # RJ[0,2] + i*RJ[1,2] = sqrt(2) * hop_t[1,2]
            # RJ[2,0] + i*RJ[2,1] = sqrt(2) * hop_t[2,0]
            self.linear_term_list[sli] += np.sqrt(2) * Spj * hop_t[1, 2]
            self.linear_term_list[slj] += np.sqrt(2) * Spi * hop_t[2, 0]

        return self.K_Hamiltonian, self.linear_term_list

    def partial_derivatives_of_Hk(self, kpoints):
        """Compute partial derivatives dH/dk_x and dH/dk_y for Berry curvature.

        Parameters
        ----------
        kpoints : np.ndarray, shape (m, d)
            Array of k-points.

        Returns
        -------
        DxHk : np.ndarray, shape (m, 2*Ns, 2*Ns)
            Derivative with respect to k_x.
        DyHk : np.ndarray, shape (m, 2*Ns, 2*Ns)
            Derivative with respect to k_y.
        """
        m = len(kpoints)
        self.DxHk = np.zeros((m, 2 * self.Ns, 2 * self.Ns), dtype=complex)
        self.DyHk = np.zeros((m, 2 * self.Ns, 2 * self.Ns), dtype=complex)

        for coupling_dict in self.couplings:
            sli = coupling_dict["SpinI"]
            Spi = self.spin_info[sli]["Spin"]
            i = self.SL_idx[sli]

            slj = coupling_dict["SpinJ"]
            Spj = self.spin_info[slj]["Spin"]
            j = self.SL_idx[slj]

            delta = coupling_dict["Displacement"]
            exp_mDk = np.exp(-1j * np.dot(kpoints, delta))
            exp_pDk = exp_mDk.conj()

            hop_t = self.get_couplings(coupling_dict, self.Rmat_dict)
            tpm = np.sqrt(Spi * Spj) * hop_t[1, 1]
            tpp = np.sqrt(Spi * Spj) * hop_t[1, 0]

            pdmx = -1j * delta[0]
            pdpx = +1j * delta[0]
            pdmy = -1j * delta[1]
            pdpy = +1j * delta[1]

            # dH/dk_x contributions
            self.DxHk[:, i, j] += tpm * exp_mDk * pdmx
            self.DxHk[:, j, i] += (tpm * exp_mDk * pdmx).conj()
            self.DxHk[:, self.Ns + i, self.Ns + j] += (tpm * exp_pDk * pdpx).conj()
            self.DxHk[:, self.Ns + j, self.Ns + i] += tpm * exp_pDk * pdpx

            # B block (upper-right)
            self.DxHk[:, i, self.Ns + j] += tpp * exp_mDk * pdmx
            self.DxHk[:, j, self.Ns + i] += tpp * exp_pDk * pdpx
            # B† block (lower-left)
            self.DxHk[:, self.Ns + j, i] += (tpp * exp_mDk * pdmx).conj()
            self.DxHk[:, self.Ns + i, j] += (tpp * exp_pDk * pdpx).conj()

            # dH/dk_y contributions
            self.DyHk[:, i, j] += tpm * exp_mDk * pdmy
            self.DyHk[:, j, i] += (tpm * exp_mDk * pdmy).conj()
            self.DyHk[:, self.Ns + i, self.Ns + j] += (tpm * exp_pDk * pdpy).conj()
            self.DyHk[:, self.Ns + j, self.Ns + i] += tpm * exp_pDk * pdpy

            # B block (upper-right)
            self.DyHk[:, i, self.Ns + j] += tpp * exp_mDk * pdmy
            self.DyHk[:, j, self.Ns + i] += tpp * exp_pDk * pdpy
            # B† block (lower-left)
            self.DyHk[:, self.Ns + j, i] += (tpp * exp_mDk * pdmy).conj()
            self.DyHk[:, self.Ns + i, j] += (tpp * exp_pDk * pdpy).conj()

        return self.DxHk, self.DyHk

    def solve_k_Hamiltonian(self, k_points, Berry_curvature=True,
                            regularization="MAGSWT", threshold=1e-8):
        """Build and diagonalize the Hamiltonian for all k-points.

        Parameters
        ----------
        k_points : np.ndarray
            Array of k-points.
        Berry_curvature : bool, optional
            Whether to compute partial derivatives for Berry curvature
            (default: True).
        regularization : str, optional
            Regularization scheme, e.g. 'MAGSWT' (default: 'MAGSWT').
        threshold : float, optional
            Energy threshold for regularization (default: 1e-8).

        Returns
        -------
        k_data : dict
            Diagonalization results keyed by k-point index.
        chem_pot_mag : float
            Chemical potential from MAGSWT regularization.
        """
        K_Ham_num, linear_term = self.Quadratic_Bose_Hamiltonian(k_points)

        if Berry_curvature:
            Partial_Diff_H = self.partial_derivatives_of_Hk(k_points)
        else:
            Partial_Diff_H = None

        k_data, chem_pot_mag = Diagonalizer.get_K_data(
            k_points, K_Ham_num,
            regularization=regularization,
            partial_derivative_Hk=Partial_Diff_H,
            threshold=threshold
        )
        return k_data, chem_pot_mag

    def compute_quantum_energy(self, k_points, angles=None, T=0,
                               reg_type="MAGSWT", compute_free_energy=False):
        """Compute the zero-point quantum energy correction.

        Parameters
        ----------
        k_points : np.ndarray
            Array of k-points.
        angles : array_like or None, optional
            Spin angles override.
        T : float, optional
            Temperature (default: 0).
        reg_type : str, optional
            Regularization type (default: 'MAGSWT').
        compute_free_energy : bool, optional
            Unused flag kept for interface compatibility (default: False).

        Returns
        -------
        E_zero : float
            Zero-point energy (plus thermal contribution if T > 0).
        mu_magswt : float
            Chemical potential from MAGSWT regularization.
        """
        K_Ham_num, _ = self.Quadratic_Bose_Hamiltonian(k_points, angles=angles)
        K_Ham_num, Bose_E, mu_magswt = Diagonalizer.get_eigenvalue(
            K_Ham_num, reg_type=reg_type
        )

        E_zero = 0
        if T == 0:
            for Hk, Ek in zip(K_Ham_num, Bose_E):
                trace_hk = np.real(np.trace(Hk))
                sum_Ek = np.sum(Ek[:self.Ns])
                E_zero += sum_Ek / 2 - trace_hk / 4
        elif T > 0:
            for Hk, Ek in zip(K_Ham_num, Bose_E):
                trace_hk = np.real(np.trace(Hk))
                Epk = Ek[:self.Ns]
                sum_Ek = np.sum(Epk)
                E_zero += sum_Ek / 2 - trace_hk / 4
                BE_dist = compute_bose_einstein_distribution(Epk, Temperature=T)
                E_zero += Epk * BE_dist

        return E_zero, mu_magswt

    def compute_quantum_free_energy(self, k_points, angles=None, T=0,
                                    reg_type="MAGSWT"):
        """Compute the quantum free energy including bosonic contributions.

        Parameters
        ----------
        k_points : np.ndarray
            Array of k-points.
        angles : array_like or None, optional
            Spin angles override.
        T : float, optional
            Temperature (default: 0).
        reg_type : str, optional
            Regularization type (default: 'MAGSWT').

        Returns
        -------
        E_zero : float
            Free energy (zero-point + thermal if T > 0).
        mu_magswt : float
            Chemical potential from MAGSWT regularization.
        """
        K_Ham_num, _ = self.Quadratic_Bose_Hamiltonian(k_points, angles=angles)
        K_Ham_num, Bose_E, mu_magswt = Diagonalizer.get_eigenvalue(
            K_Ham_num, reg_type=reg_type
        )

        E_zero = 0
        if T == 0:
            for Hk, Ek in zip(K_Ham_num, Bose_E):
                trace_hk = np.real(np.trace(Hk))
                sum_Ek = np.sum(Ek[:self.Ns])
                E_zero += sum_Ek / 2 - trace_hk / 4
        elif T > 0:
            for Hk, Ek in zip(K_Ham_num, Bose_E):
                trace_hk = np.real(np.trace(Hk))
                Epk = Ek[:self.Ns]
                sum_Ek = np.sum(Epk)
                E_zero += sum_Ek / 2 - trace_hk / 4
                free_E = bosonic_free_energy(Epk, T)
                E_zero += free_E

        return E_zero, mu_magswt
