"""Spin-spin correlation functions from linear spin wave theory.

This module computes momentum-space and real-space spin-spin correlation
functions, structure factors, and spectral functions within the LSWT framework.

Ported from modules/LinearSpinWaveTheory/lswt_correlation.py.
"""

import numpy as np
from typing import Union, Optional, Dict, Tuple, List
from tqdm import tqdm

from lswt.observables.bose_statistics import (
    compute_static_magnon_kernel,
    compute_real_time_kernel,
    compute_lorentzian_kernel,
    compute_spectral_kernel,
)
from lswt.core.brillouin_zone import get_nearest_lattices
from lswt.config import (
    DEFAULT_TOLERANCE, DEFAULT_TEMPERATURE, DEFAULT_TIME,
    DEFAULT_OMEGA, DEFAULT_ETA, DEFAULT_DELTA_PEAK, Mat_C,
)


class Correlations:
    """Spin-spin correlation function calculator.

    Computes momentum-space and real-space spin correlation functions
    using the Holstein-Primakoff boson representation.

    Parameters
    ----------
    lswt_obj : object
        Parent LSWT solver object providing spin_info, lattice_bz_settings, Ns.
    msl_boson_number : dict or None, optional
        Mean-field sublattice boson numbers. If None, taken from lswt_obj.
    """

    def __init__(self, lswt_obj, msl_boson_number=None):
        self.lswt = lswt_obj
        self.spin_info = self.lswt.spin_info
        self.lattice_vectors = self.lswt.lattice_bz_settings[0]
        self.Ns = lswt_obj.Ns

        if hasattr(self, "msl_average_boson_number") or hasattr(lswt_obj, "msl_average_boson_number"):
            self.msl_boson_number = lswt_obj.msl_average_boson_number
        else:
            self.msl_boson_number = msl_boson_number

        self._initiation_lswt_corr()

    def _initiation_lswt_corr(self, cond_low_boson_number=True):
        """Initialize rotation matrices, sublattice positions, and spin moments.

        Parameters
        ----------
        cond_low_boson_number : bool, optional
            Whether to check if boson number is below the spin value
            and use classical spin if not (default: True).
        """
        self.rmat_dict = {}
        self.sublattices_dict = {}
        self.sublattice_pos = []
        self.sqrt_spins = []
        self.msl_spin_moment = np.zeros(self.Ns)

        low_boson_number = True

        if cond_low_boson_number:
            for name_sl, boson_num in self.msl_boson_number.items():
                spin = self.spin_info[name_sl]["Spin"]
                low_boson_number = (boson_num < spin) and low_boson_number

        # Replace with classical spin if boson number exceeds spin value
        classical_spin = cond_low_boson_number and not low_boson_number

        for j, (name_sl, sl_dict) in enumerate(self.spin_info.items()):

            self.sublattices_dict[name_sl] = j
            theta, phi = sl_dict["Angles"]
            self.rmat_dict[name_sl] = self._classical_spin_rotation_matrix(theta, phi)

            pos = sl_dict["Position"]
            spin = sl_dict["Spin"]
            boson_num = self.msl_boson_number[name_sl]

            self.sublattice_pos.append(pos)
            self.sqrt_spins.append(spin)
            self.msl_spin_moment[j] = spin if classical_spin else spin - boson_num

        self.sublattice_pos = np.array(self.sublattice_pos).T
        self.sqrt_spins = np.sqrt(np.array(self.sqrt_spins))

        a1, a2 = self.lattice_vectors

        nearest_lattices = get_nearest_lattices(a1, a2)

        if len(nearest_lattices) == 4:
            self.nearest_lattices = nearest_lattices[:2]
        else:
            self.nearest_lattices = nearest_lattices[::2]

    @staticmethod
    def delta_func_weight(q, nearest_vectors, delta_peak=50, eps=1e-10):
        """Compute delta function weight for classical contribution.

        Approximates the lattice delta function using the Dirichlet kernel.

        Parameters
        ----------
        q : np.ndarray or tuple
            Momentum vector.
        nearest_vectors : list of np.ndarray
            Nearest reciprocal lattice vectors.
        delta_peak : int or float, optional
            Peak width parameter (default: 50).
        eps : float, optional
            Regularization to avoid division by zero (default: 1e-10).

        Returns
        -------
        float
            Delta function weight at q.
        """
        delta_k = 1

        for a in nearest_vectors:
            q_dot_a = np.dot(q, a)
            numer = np.sin(delta_peak * q_dot_a / 2)
            denom = np.sin(q_dot_a / 2)
            reg_term = 1 if abs(denom) < eps else numer / (delta_peak * denom)
            delta_k *= reg_term

        return delta_k

    @staticmethod
    def _classical_spin_rotation_matrix(pol_ang, azm_ang):
        """Compute rotation matrix for classical spin direction.

        Parameters
        ----------
        pol_ang : float
            Polar angle (theta).
        azm_ang : float
            Azimuthal angle (phi).

        Returns
        -------
        np.ndarray
            3x3 rotation matrix from local to global frame.
        """
        Rot_spin = np.array([
            [np.cos(pol_ang) * np.cos(azm_ang), -np.sin(azm_ang),
             np.sin(pol_ang) * np.cos(azm_ang)],
            [np.cos(pol_ang) * np.sin(azm_ang), np.cos(azm_ang),
             np.sin(pol_ang) * np.sin(azm_ang)],
            [-np.sin(pol_ang), 0, np.cos(pol_ang)],
        ])
        return Rot_spin

    @staticmethod
    def search_key_matrices_dict(matrice, key_to_find, keys):
        """Look up a matrix in a dictionary with tolerance-based fallback.

        Parameters
        ----------
        matrice : dict
            Dictionary mapping k-point tuples to matrices.
        key_to_find : tuple
            The k-point to look up.
        keys : np.ndarray
            Array of all k-point keys for nearest-neighbor search.

        Returns
        -------
        np.ndarray
            The matrix at the matching k-point.

        Raises
        ------
        KeyError
            If no matching k-point is found within tolerance.
        """
        try:
            return matrice[key_to_find]
        except KeyError:
            distances = np.linalg.norm(keys - np.array(key_to_find), axis=1)
            min_idx = np.argmin(distances)

            if distances[min_idx] < DEFAULT_TOLERANCE:
                nearest_key = tuple(keys[min_idx])
                return matrice[nearest_key]
            else:
                raise KeyError(f"k-point {key_to_find} not found")

    def _get_sublattice_phase_factor(self, k_vector):
        """Generate sublattice phase factor for a given k vector.

        Parameters
        ----------
        k_vector : np.ndarray or tuple
            Momentum vector.

        Returns
        -------
        np.ndarray
            Phase factors exp(-i k . d) for each sublattice.
        """
        k_vector = np.array(k_vector)
        mikd = -1j * (k_vector @ self.sublattice_pos)
        return np.exp(mikd)

    def _get_spin_sublattice_matrix(self, k_vector):
        """Generate sublattice matrix S_k for a given k vector.

        Parameters
        ----------
        k_vector : np.ndarray or tuple
            Momentum vector.

        Returns
        -------
        S_k : np.ndarray
            Sublattice spin matrix, shape (2*Ns, 2*Ns).
        exp_mikd : np.ndarray
            Phase factors for each sublattice.
        """
        exp_mikd = self._get_sublattice_phase_factor(k_vector)
        sk = exp_mikd * self.sqrt_spins
        S_k = np.diag(np.hstack([sk, sk]))
        return S_k, exp_mikd

    def _get_U_V_alpha_matrix(self, coordinate_type="cartesian"):
        """Build rotation matrices U and V for spin components.

        Parameters
        ----------
        coordinate_type : str, optional
            Coordinate system: "cartesian" (or "xyz") for Sx,Sy,Sz;
            "complex" (or "ladder", "pm0", "+-0") for S+,S-,S0
            (default: "cartesian").

        Returns
        -------
        U_alphas_dict : dict
            U matrices for each spin component.
        V_alphas_dict : dict
            V matrices (classical spin direction) for each component.
        alphas : list
            Component labels for rows.
        betas : list
            Component labels for columns.

        Raises
        ------
        ValueError
            If coordinate_type is not recognized.
        """
        if coordinate_type in ("cartesian", "Cartesian", "xyz"):
            alphas = ["x", "y", "z"]
            betas = alphas
            get_U_sublattice = lambda mat: mat @ Mat_C

        elif coordinate_type in ("complex", "ladder", "pm0", "+-0"):
            alphas = ["-", "+", "0"]
            betas = ["+", "-", "0"]
            get_U_sublattice = lambda mat: Mat_C.T.conj() @ mat @ Mat_C

        else:
            raise ValueError("choose coordinate type either cartesian or ladder")

        U_alphas_dict = {alpha: [] for alpha in alphas}
        V_alphas_dict = {alpha: [] for alpha in alphas}
        for key, rmat in self.rmat_dict.items():
            sublattice_u_matrix = get_U_sublattice(rmat)
            U_alphas_dict[alphas[0]].append(sublattice_u_matrix[0, 0])
            U_alphas_dict[alphas[1]].append(sublattice_u_matrix[1, 0])
            U_alphas_dict[alphas[2]].append(sublattice_u_matrix[2, 0])

            V_alphas_dict[alphas[0]].append(sublattice_u_matrix[0, 2])
            V_alphas_dict[alphas[1]].append(sublattice_u_matrix[1, 2])
            V_alphas_dict[alphas[2]].append(sublattice_u_matrix[2, 2])

        for alpha in alphas:
            u_alpha = np.array(U_alphas_dict[alpha])
            U_alphas_dict[alpha] = np.hstack([u_alpha.conj(), u_alpha])
            V_alphas_dict[alpha] = np.array(V_alphas_dict[alpha])

        return U_alphas_dict, V_alphas_dict, alphas, betas

    def compute_real_space_spin_corr_function_in_local_frame_at_rvec(
        self,
        k_data,
        rvec,
        Temperature=DEFAULT_TEMPERATURE,
        time=DEFAULT_TIME,
    ):
        """Compute real-space spin correlation function in the local frame.

        Parameters
        ----------
        k_data : dict
            Dictionary of k-point data.
        rvec : np.ndarray or tuple
            Real-space displacement vector.
        Temperature : float, optional
            Temperature in Kelvin (default: 0).
        time : float, optional
            Time for dynamical correlations (default: 0).

        Returns
        -------
        np.ndarray
            Spin-spin correlation matrix in local frame, shape (2*Ns, 2*Ns).
        """
        rvec = np.array(rvec)

        RS_spin_spin_corr_func = np.zeros((self.Ns * 2, self.Ns * 2), dtype=complex)

        for k_key, value in k_data.items():
            _, Eigen_data, _ = value

            Eval, Evec = Eigen_data

            magnon_kernel = compute_real_time_kernel(Eval, time, Temperature)

            bosonic_corr_mat = Evec @ np.diag(magnon_kernel) @ Evec.T.conj()
            S_k, _ = self._get_spin_sublattice_matrix(k_key)
            spin_corr_mat = S_k.conj() @ bosonic_corr_mat @ S_k

            # Fourier transform from momentum to real space
            k_key_array = np.array(k_key)
            mikr = -1j * k_key_array @ rvec
            RS_spin_spin_corr_func += spin_corr_mat * np.exp(mikr)

        return RS_spin_spin_corr_func / len(k_data)

    def compute_TNT_for_structure(self, k_data,
                                  Temperature=DEFAULT_TEMPERATURE,
                                  omega=None, eta=DEFAULT_ETA):
        """Compute bosonic correlation matrix (TNT) for structure factor.

        Parameters
        ----------
        k_data : dict
            Dictionary of k-point data.
        Temperature : float, optional
            Temperature in Kelvin (default: 0).
        omega : float or None, optional
            Frequency for dynamical structure factor. If None, computes
            static structure factor (default: None).
        eta : float, optional
            Lorentzian broadening parameter (default: 1e-3).

        Returns
        -------
        TNT : dict
            Bosonic correlation matrices keyed by k-point.
        """
        TNT = {}

        for k_key, value in k_data.items():
            _, Eigen_data, _ = value
            Eval, Evec = Eigen_data

            if omega is None:
                magnon_kernel = compute_static_magnon_kernel(
                    Eval, Temperature=Temperature, Ns=self.Ns
                )
            elif isinstance(omega, (float, int, np.number)):
                magnon_kernel = compute_lorentzian_kernel(
                    Eval, omega, eta=eta, Temperature=Temperature, Ns=self.Ns
                )

            TNT[k_key] = Evec @ np.diag(magnon_kernel) @ Evec.T.conj()

        return TNT

    def compute_TNT_for_spectral(self, k_data,
                                 Temperature=DEFAULT_TEMPERATURE,
                                 omega=None, eta=DEFAULT_ETA):
        """Compute bosonic correlation matrix (TNT) for spectral function.

        Parameters
        ----------
        k_data : dict
            Dictionary of k-point data.
        Temperature : float, optional
            Temperature in Kelvin (default: 0).
        omega : float or None, optional
            Frequency (default: None).
        eta : float, optional
            Lorentzian broadening parameter (default: 1e-3).

        Returns
        -------
        TNT : dict
            Bosonic correlation matrices keyed by k-point.
        """
        TNT = {}

        for k_key, value in k_data.items():
            _, Eigen_data, _ = value
            Eval, Evec = Eigen_data
            magnon_kernel = compute_spectral_kernel(
                Eval, omega, eta, Temperature, Ns=self.Ns
            )
            TNT[k_key] = Evec @ np.diag(magnon_kernel) @ Evec.T.conj()

        return TNT

    def calculate_spin_corr_mat_in_local(self, TNT, k_points,
                                         classical_contribution=True,
                                         delta_peak=DEFAULT_DELTA_PEAK):
        """Calculate spin-spin correlation function in local magnetization frame.

        Parameters
        ----------
        TNT : dict
            Bosonic correlation matrices keyed by k-point.
        k_points : np.ndarray
            Array of k-points, shape (num_k, 2).
        classical_contribution : bool, optional
            Whether to include the classical (ordered moment) part (default: True).
        delta_peak : int or float, optional
            Delta function peak width (default: DEFAULT_DELTA_PEAK).

        Returns
        -------
        quantum_spin_corr_mat : np.ndarray
            Quantum contribution, shape (num_k, 2*Ns, 2*Ns).
        delta_func : np.ndarray or None
            Delta function weights, shape (num_k,), or None.
        sublattice_phase : np.ndarray or None
            Sublattice phase factors, shape (Ns, num_k), or None.
        """
        num_k = len(k_points)

        keys_bosonic_corr_mat = np.array(list(TNT.keys()))
        quantum_spin_corr_mat = np.zeros((num_k, self.Ns * 2, self.Ns * 2), dtype=complex)

        delta_func = np.zeros(num_k) if classical_contribution else None
        sublattice_phase = (
            np.zeros((self.Ns, num_k), dtype=complex) if classical_contribution else None
        )

        for j, kpt in enumerate(k_points):
            S_k, sublattice_phase_factor_k = self._get_spin_sublattice_matrix(kpt)

            bosonic_corr_mat = self.search_key_matrices_dict(
                TNT, key_to_find=tuple(kpt), keys=keys_bosonic_corr_mat
            )

            quantum_spin_corr_mat[j] = S_k @ bosonic_corr_mat @ S_k.conj()

            if classical_contribution:
                delta_k = self.delta_func_weight(
                    kpt, self.nearest_lattices, delta_peak=delta_peak
                )
                delta_func[j] = delta_k
                sublattice_phase[:, j] = sublattice_phase_factor_k

        return quantum_spin_corr_mat, delta_func, sublattice_phase

    def calculate_spin_corr_mat(self, TNT, k_points=None,
                                coordinate_type="cartesian",
                                sublattice=None,
                                classical_contribution=True):
        """Compute spin correlation matrix for all k-points.

        Parameters
        ----------
        TNT : dict
            Bosonic correlation matrices keyed by k-point.
        k_points : np.ndarray or None, optional
            Array of k-points. If None, uses keys from TNT.
        coordinate_type : str, optional
            "cartesian" or "ladder" (default: "cartesian").
        sublattice : tuple or None, optional
            Sublattice pair (mu, nu) to compute. If None, sums over all.
        classical_contribution : bool, optional
            Whether to include classical contribution (default: True).

        Returns
        -------
        corr_mat : dict
            Component-wise correlation matrices (e.g., 'xx', 'xy', ...).
        total_corr : np.ndarray
            Total spin correlation, shape (num_k,).
        k_points : np.ndarray
            The k-points used in the calculation.

        Raises
        ------
        ValueError
            If total correlation contains invalid values.
        """
        k_points = k_points if k_points is not None else np.array(list(TNT.keys()))

        num_k = len(k_points)
        U_dict, V_dict, alphas, betas = self._get_U_V_alpha_matrix(
            coordinate_type=coordinate_type
        )
        local_spin_corr_mat, delta_func, sublattice_phase = (
            self.calculate_spin_corr_mat_in_local(
                TNT,
                k_points=k_points,
                classical_contribution=classical_contribution,
                delta_peak=np.minimum(num_k, 10),
            )
        )
        corr_mat = {}

        for alpha in alphas:
            for beta in betas:
                U_alpha = np.diag(U_dict[alpha])
                U_beta = np.diag(U_dict[beta])
                SaSb_k = U_alpha @ local_spin_corr_mat @ U_beta.T.conj()

                corr_mat[alpha + beta] = self.get_quantum_spin_corr_mat(SaSb_k, sublattice)

                if classical_contribution:
                    V_alpha = V_dict[alpha]
                    V_beta = V_dict[beta]
                    corr_mat[alpha + beta] += self.get_classical_spin_corr_mat(
                        delta_func, sublattice_phase, sublattice, V_alpha, V_beta
                    )

        total_corr = np.zeros(num_k, dtype=complex)
        if coordinate_type == 'cartesian':
            total_corr = corr_mat['xx'] + corr_mat['yy'] + corr_mat['zz']
        else:
            total_corr = (corr_mat['+-'] + corr_mat['-+']) / 2 + corr_mat['00']

        if total_corr is None or not np.all(np.isfinite(total_corr)):
            raise ValueError("spin_corr_total contains invalid values or is None")

        return corr_mat, total_corr, k_points

    def get_quantum_spin_corr_mat(self, SaSb_k, sublattice):
        """Extract quantum spin correlation for given sublattice pair.

        Parameters
        ----------
        SaSb_k : np.ndarray
            Spin correlation tensor, shape (num_k, 2*Ns, 2*Ns).
        sublattice : tuple or None
            Sublattice pair (mu, nu), or None for total.

        Returns
        -------
        np.ndarray
            Quantum spin correlation, shape (num_k,).
        """
        mu_idx, nu_idx = None, None
        if sublattice is not None:
            mu, nu = sublattice
            mu_idx = self.sublattices_dict[mu]
            nu_idx = self.sublattices_dict[nu]

            result = (
                SaSb_k[:, mu_idx, nu_idx]
                + SaSb_k[:, mu_idx + self.Ns, nu_idx]
                + SaSb_k[:, mu_idx, nu_idx + self.Ns]
                + SaSb_k[:, mu_idx + self.Ns, nu_idx]
            )
            return result
        else:
            result = np.sum(SaSb_k, axis=(1, 2))
            return result

    def get_classical_spin_corr_mat(self, delta_func, sublattice_phase,
                                    sublattice, V_alpha, V_beta):
        """Compute classical spin correlation matrix.

        Parameters
        ----------
        delta_func : np.ndarray
            Delta function weights, shape (num_k,).
        sublattice_phase : np.ndarray
            Sublattice phase factors, shape (Ns, num_k).
        sublattice : tuple or None
            Sublattice pair (mu, nu), or None for total.
        V_alpha : np.ndarray
            V-vector for alpha component.
        V_beta : np.ndarray
            V-vector for beta component.

        Returns
        -------
        np.ndarray
            Classical spin correlation, shape (num_k,).
        """
        mu_idx, nu_idx = None, None
        if sublattice is not None:
            mu, nu = sublattice
            mu_idx = self.sublattices_dict[mu]
            nu_idx = self.sublattices_dict[nu]

        classical_spin_corr_mu_nu = np.zeros_like(delta_func, dtype=complex)

        for i, v_a_mu in enumerate(V_alpha):
            if sublattice is not None and i != mu_idx:
                continue

            lswt_spin_mu = self.msl_spin_moment[i]
            phase_mu = sublattice_phase[i]

            for j, v_b_nu in enumerate(V_beta):
                if sublattice is not None and j != nu_idx:
                    continue

                lswt_spin_nu = self.msl_spin_moment[j]
                phase_nu = (sublattice_phase[j]).conj()

                exp_phase = phase_mu * phase_nu
                spin_ab_product = v_a_mu * v_b_nu * lswt_spin_mu * lswt_spin_nu

                classical_spin_corr_mu_nu += spin_ab_product * exp_phase

        return classical_spin_corr_mu_nu * delta_func

    @staticmethod
    def _neutron_scattering_factor(alpha, beta, kx_arr, ky_arr, coordinate_type):
        """Compute the neutron scattering polarization factor.

        Implements the factor (delta_ab - q_a * q_b / |q|^2) that appears
        in the neutron scattering cross section.

        Parameters
        ----------
        alpha : str
            First component label ('x', 'y', '+', '-').
        beta : str
            Second component label.
        kx_arr : np.ndarray
            x-components of momentum transfer.
        ky_arr : np.ndarray
            y-components of momentum transfer.
        coordinate_type : str
            Coordinate system ("cartesian" or "ladder").

        Returns
        -------
        np.ndarray
            Neutron scattering factor for each k-point.
        """
        def q_alpha(comp):
            if comp == "x":
                return kx_arr
            elif comp == "y":
                return ky_arr
            elif comp == "+":
                return (kx_arr + 1j * ky_arr) / np.sqrt(2)
            elif comp == "-":
                return (kx_arr - 1j * ky_arr) / np.sqrt(2)
            else:
                return 0

        delta_ab = 1 if alpha == beta else 0

        q_norm = np.sqrt(kx_arr ** 2 + ky_arr ** 2)
        q_norm[np.where(q_norm == 0)] = 1

        if coordinate_type in ('cartesian', "Cartesian"):
            q_a = q_alpha(alpha)
            q_b = q_alpha(beta)
        elif coordinate_type in ("ladder", "+-0"):
            q_a = q_alpha(alpha)
            q_b = q_alpha(beta)

        q_ab = (q_a * q_b) / q_norm

        return delta_ab - q_ab

    def compute_real_space_correlations(self, k_data, angle_direction,
                                        distance_range=(0, 1, 0.1),
                                        time=DEFAULT_TIME,
                                        Temperature=DEFAULT_TEMPERATURE):
        """Compute real-space spin-spin correlation values along a given direction.

        Parameters
        ----------
        k_data : dict
            Dictionary of k-point data.
        angle_direction : float
            Angle (radians) defining the real-space direction.
        distance_range : tuple, optional
            (r_start, r_end, r_step) for the distance values (default: (0, 1, 0.1)).
        time : float, optional
            Time for dynamical correlations (default: 0).
        Temperature : float, optional
            Temperature in Kelvin (default: 0).

        Returns
        -------
        r_values : np.ndarray
            Array of distance values.
        RS_corr_in_local : np.ndarray
            Correlation matrices at each distance, shape (num_r, 2*Ns, 2*Ns).
        """
        direction = np.array(
            (np.cos(angle_direction), np.sin(angle_direction)), dtype=float
        )

        r_start, r_end, r_step = distance_range
        r_values = np.linspace(r_start, r_end, int((r_end - r_start) / r_step) + 1)
        rvec_list = [direction * r for r in r_values]

        RS_corr_in_local = np.empty(
            (len(r_values), 2 * self.Ns, 2 * self.Ns), dtype=complex
        )

        for j, rvec in enumerate(tqdm(rvec_list, desc="Calculating correlations")):
            RS_corr_in_local[j] = (
                self.compute_real_space_spin_corr_function_in_local_frame_at_rvec(
                    k_data=k_data, rvec=rvec, Temperature=Temperature, time=time
                )
            )

        return r_values, RS_corr_in_local
