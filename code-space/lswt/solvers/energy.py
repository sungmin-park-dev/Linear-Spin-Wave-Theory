"""Energy function construction for LSWT optimization.

This module provides the EnergyFunction class which constructs classical,
quantum, and total energy density functions for use in spin configuration
optimization. It wraps the Brillouin zone sampling and LSWT Hamiltonian
to provide callable energy functions of spin angles.
"""

import numpy as np

from lswt.core.brillouin_zone import BrillouinZone
from lswt.solvers.hamiltonian import LSWTHamiltonian


class EnergyFunction:
    """Construct energy density functions for spin system optimization.

    Builds classical and quantum energy density functions as callable
    objects of spin angles, suitable for use with SpinOptimizer.

    Parameters
    ----------
    spin_sys_data : dict
        Spin system data dictionary containing 'Spin info',
        'Couplings', and 'Lattice/BZ setting'.
    N : int
        Number of k-points along each direction for BZ sampling.
    update_args : bool, optional
        If True, cache intermediate results (default: False).

    Attributes
    ----------
    num_SL : int
        Number of sublattices.
    k_points : np.ndarray
        Array of k-points in the Brillouin zone.
    num_k_points : int
        Total number of k-points.
    energy_func : callable
        Total (classical + quantum) energy density function.
    classical_energy_density : float or None
        Cached classical energy density.
    quantum_energy_density : float or None
        Cached quantum energy density.
    mu_magswt : float or None
        Chemical potential for MAGSWT regularization.
    angle_args : np.ndarray or None
        Cached angle arguments.
    """

    def __init__(self, spin_sys_data, N, update_args=False):
        self._info_cache = None
        self.angle_args = None
        self.classical_energy_density = None
        self.quantum_energy_density = None
        self.mu_magswt = None
        self.num_k_points = None
        self.N = N
        self.spin_sys_data = spin_sys_data
        self.num_SL = len(self.spin_sys_data["Spin info"])
        self.update_args = update_args

        B_zone = BrillouinZone(
            self.spin_sys_data["Lattice/BZ setting"], bz_type="simple"
        )
        _, self.k_points, _ = B_zone.get_full(N=N)
        self.num_k_points = len(self.k_points)
        self.lswt_Ham = LSWTHamiltonian(
            self.spin_sys_data["Spin info"], self.spin_sys_data["Couplings"]
        )

        self.energy_func = self.lswt_energy_density_func

    @staticmethod
    def _cartesian_spin_odering(S, theta, phi):
        """Convert spherical spin coordinates to Cartesian vector.

        Parameters
        ----------
        S : float
            Spin magnitude.
        theta : float
            Polar angle.
        phi : float
            Azimuthal angle.

        Returns
        -------
        spin_vec : np.ndarray
            Cartesian spin vector [Sx, Sy, Sz].
        """
        return np.array([
            S * np.cos(phi) * np.sin(theta),
            S * np.sin(phi) * np.sin(theta),
            S * np.cos(theta),
        ])

    def classical_energy_density_func(self, angles):
        """Compute classical energy density for given spin angles.

        Parameters
        ----------
        angles : array-like
            Spin angles [theta1, phi1, theta2, phi2, ...] of length 2*num_SL.

        Returns
        -------
        classical_energy_density : float
            Classical energy per sublattice site.
        """
        if len(angles) != 2 * self.num_SL:
            raise ValueError(
                f"Expected {2 * self.num_SL} angles, got {len(angles)}"
            )

        if self.update_args:
            self.angle_args = angles

        spins_orderings = {}
        for i, (spin_name, spin_dict) in enumerate(
            self.spin_sys_data["Spin info"].items()
        ):
            S = spin_dict["Spin"]
            theta, phi = angles[2 * i : 2 * i + 2]
            spins_orderings[spin_name] = self._cartesian_spin_odering(
                S, theta, phi
            )

        classical_energy = 0.0
        for coupling_dict in self.spin_sys_data["Couplings"]:
            spin_i = spins_orderings[coupling_dict["SpinI"]]
            spin_j = spins_orderings[coupling_dict["SpinJ"]]
            classical_energy += (
                spin_i @ coupling_dict["Exchange Matrix"] @ spin_j
            )

        for spin_name, info in self.spin_sys_data["Spin info"].items():
            classical_energy -= np.dot(
                info["Magnetic Field"], spins_orderings[spin_name]
            )

        classical_energy_density = classical_energy / self.num_SL

        if self.update_args:
            self.classical_energy_density = classical_energy_density

        return classical_energy_density

    def quantum_energy_density_func(self, angles, reg_type=0):
        """Compute quantum energy density correction for given spin angles.

        Parameters
        ----------
        angles : array-like
            Spin angles [theta1, phi1, theta2, phi2, ...] of length 2*num_SL.
        reg_type : int, optional
            Regularization type for the LSWT Hamiltonian (default: 0).

        Returns
        -------
        quantum_energy_density : float
            Quantum correction to energy per sublattice site per k-point.
        """
        if len(angles) != 2 * self.num_SL:
            raise ValueError(
                f"Expected {2 * self.num_SL} angles, got {len(angles)}"
            )

        if self.update_args:
            self.angle_args = angles

        quantum_energy, self.mu_magswt = self.lswt_Ham.compute_quantum_energy(
            self.k_points,
            angles=angles,
            T=0,
            reg_type=reg_type,
        )
        quantum_energy_density = quantum_energy / (
            self.num_k_points * self.num_SL
        )

        if self.update_args:
            self.quantum_energy_density = quantum_energy_density

        return quantum_energy_density

    def lswt_energy_density_func(self, angles):
        """Compute total LSWT energy density (classical + quantum).

        Parameters
        ----------
        angles : array-like
            Spin angles [theta1, phi1, theta2, phi2, ...] of length 2*num_SL.

        Returns
        -------
        total_energy_density : float
            Total energy density per sublattice site.
        """
        self._info_cache = None
        if len(angles) != 2 * self.num_SL:
            raise ValueError(
                f"Expected {2 * self.num_SL} angles, got {len(angles)}"
            )

        E_density_cl = self.classical_energy_density_func(angles)
        E_density_qm = self.quantum_energy_density_func(angles)

        return E_density_cl + E_density_qm

    def quantum_free_energy_density_func(self, angles, reg_type=0,
                                          Temperature=0):
        """Compute quantum free energy density at finite temperature.

        Parameters
        ----------
        angles : array-like
            Spin angles [theta1, phi1, theta2, phi2, ...] of length 2*num_SL.
        reg_type : int, optional
            Regularization type (default: 0).
        Temperature : float, optional
            Temperature in energy units (default: 0).

        Returns
        -------
        quantum_free_energy_density : float
            Quantum free energy density per sublattice site per k-point.
        """
        if len(angles) != 2 * self.num_SL:
            raise ValueError(
                f"Expected {2 * self.num_SL} angles, got {len(angles)}"
            )

        if self.update_args:
            self.angle_args = angles

        quantum_energy, self.mu_magswt = (
            self.lswt_Ham.compute_quantum_free_energy(
                self.k_points,
                angles=angles,
                T=Temperature,
                reg_type=reg_type,
            )
        )

        quantum_free_energy_density = quantum_energy / (
            self.num_k_points * self.num_SL
        )

        if self.update_args:
            self.quantum_energy_density = quantum_free_energy_density

        return quantum_free_energy_density

    def get_info(self, print_info=False):
        """Get cached energy information.

        Parameters
        ----------
        print_info : bool, optional
            If True, print the cached information (default: False).

        Returns
        -------
        data : dict
            Dictionary with keys: 'angles', 'E_cl', 'E_qm', 'MAGSWT'.
        """
        if self._info_cache is None:
            data = {
                "angles": self.angle_args,
                "E_cl": self.classical_energy_density,
                "E_qm": self.quantum_energy_density,
                "MAGSWT": self.mu_magswt,
            }
            self._info_cache = data
            if print_info:
                for key, content in data.items():
                    print(f"{key}: \n{content} ")
            return data
        return self._info_cache

    def set_update_args(self, update_args):
        """Enable or disable caching of intermediate results.

        Parameters
        ----------
        update_args : bool
            If True, cache angle_args and energy values during computation.
        """
        self.update_args = update_args

    @staticmethod
    def _d_Spin_diff_theta(S, theta, phi):
        """Derivative of Cartesian spin vector with respect to theta.

        Parameters
        ----------
        S : float
            Spin magnitude.
        theta : float
            Polar angle.
        phi : float
            Azimuthal angle.

        Returns
        -------
        d_spin : np.ndarray
            Partial derivative dS/d(theta).
        """
        return np.array([
            +S * np.cos(phi) * np.cos(theta),
            +S * np.sin(phi) * np.cos(theta),
            -S * np.sin(theta),
        ])

    @staticmethod
    def _d_Spin_diff_phi(S, theta, phi):
        """Derivative of Cartesian spin vector with respect to phi.

        Parameters
        ----------
        S : float
            Spin magnitude.
        theta : float
            Polar angle.
        phi : float
            Azimuthal angle.

        Returns
        -------
        d_spin : np.ndarray
            Partial derivative dS/d(phi).
        """
        return np.array([
            -S * np.sin(phi) * np.sin(theta),
            +S * np.cos(phi) * np.sin(theta),
            0,
        ])

    @classmethod
    def Diff_classical_energy_density_func(cls, spin_sys_data):
        """Compute analytical gradient of classical energy density.

        Computes partial derivatives of the classical energy density
        with respect to each spin's theta and phi angles.

        Parameters
        ----------
        spin_sys_data : dict
            Spin system data dictionary containing 'Spin info' and 'Couplings'.

        Returns
        -------
        diff_classical_energy : dict
            Dictionary keyed by spin name, each containing
            'd_E_d_theta' and 'd_E_d_phi' gradient components.
        """
        angles = []
        for spin_name, value in spin_sys_data["Spin info"].items():
            theta, phi = value["Angles"]
            angles.extend([theta, phi])

        num_SL = len(spin_sys_data["Spin info"])
        spins_orderings = {}
        spins_d_thetas = {}
        spins_d_phis = {}

        for i, (spin_name, spin_dict) in enumerate(
            spin_sys_data["Spin info"].items()
        ):
            S = spin_dict["Spin"]
            theta, phi = angles[2 * i : 2 * i + 2]
            spins_orderings[spin_name] = cls._cartesian_spin_odering(
                S, theta, phi
            )
            spins_d_thetas[spin_name] = cls._d_Spin_diff_theta(S, theta, phi)
            spins_d_phis[spin_name] = cls._d_Spin_diff_phi(S, theta, phi)

        diff_classical_energy = {
            spin_name: {"d_E_d_theta": 0, "d_E_d_phi": 0}
            for spin_name in spin_sys_data["Spin info"]
        }

        for coupling_dict in spin_sys_data["Couplings"]:
            name_i = coupling_dict["SpinI"]
            name_j = coupling_dict["SpinJ"]
            exchange_matrix = coupling_dict["Exchange Matrix"]

            spin_i_d_theta = spins_d_thetas[name_i]
            diff_classical_energy[name_i]["d_E_d_theta"] += (
                spin_i_d_theta @ exchange_matrix @ spins_orderings[name_j]
            )

            spin_j_d_theta = spins_d_thetas[name_j]
            diff_classical_energy[name_j]["d_E_d_theta"] += (
                spins_orderings[name_i] @ exchange_matrix @ spin_j_d_theta
            )

            spin_i_d_phi = spins_d_phis[name_i]
            diff_classical_energy[name_i]["d_E_d_phi"] += (
                spin_i_d_phi @ exchange_matrix @ spins_orderings[name_j]
            )

            spin_j_d_phi = spins_d_phis[name_j]
            diff_classical_energy[name_j]["d_E_d_phi"] += (
                spins_orderings[name_i] @ exchange_matrix @ spin_j_d_phi
            )

        for spin_name, info in spin_sys_data["Spin info"].items():
            magnetic_field = info["Magnetic Field"]
            diff_classical_energy[spin_name]["d_E_d_theta"] -= np.dot(
                magnetic_field, spins_d_thetas[spin_name]
            )
            diff_classical_energy[spin_name]["d_E_d_phi"] -= np.dot(
                magnetic_field, spins_d_phis[spin_name]
            )

        for spin_name in diff_classical_energy:
            diff_classical_energy[spin_name]["d_E_d_theta"] /= num_SL
            diff_classical_energy[spin_name]["d_E_d_phi"] /= num_SL

        return diff_classical_energy
