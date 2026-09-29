"""Spin system optimizer for finding ground state spin configurations.

This module provides optimization routines for finding the minimum energy
spin configuration using differential evolution (DE) and L-BFGS-B methods,
including support for MAGSWT (modified spin wave theory) optimization.
"""

import numpy as np
from typing import List, Union
from scipy.optimize import minimize, differential_evolution


# Optimization method name constants
CLASSICAL_METHOD_NAME = ["classical", "Classical", "CLASSICAL"]
MAGSWT_METHOD_NAME = ["MAGSWT", "magswt"]
QUANTUM_METHOD_NAME = ["classical+quantum", "quantum"]

OPT_METHOD_NAMES = CLASSICAL_METHOD_NAME + MAGSWT_METHOD_NAME + QUANTUM_METHOD_NAME


class SpinOptimizer:
    """Optimizer for finding ground state spin configurations.

    Supports classical optimization via differential evolution,
    MAGSWT grid search optimization, and quantum (classical+quantum)
    optimization via L-BFGS-B refinement.

    Attributes
    ----------
    num_trials : int
        Number of optimization trials performed.
    """

    def __init__(self):
        self.num_trials = 0

    def wrapping_by_angles(self, cef_obj, angles_setting, verbose=False):
        """Wrap energy functions with fixed/free angle constraints.

        Parameters
        ----------
        cef_obj : EnergyFunction
            Energy function object providing classical/quantum energy methods.
        angles_setting : list, tuple, or None
            Angle constraints. None means all angles are free. A list/tuple
            of length 2*num_SL where None entries are free variables and
            numeric entries are fixed.
        verbose : bool, optional
            If True, print optimization details (default: False).

        Returns
        -------
        bounds : list of tuple
            Bounds for each free variable, (-pi, pi).
        E_tot_func : callable
            Total energy function of free variables.
        E_cl_func : callable
            Classical energy function of free variables.
        E_qm_func : callable
            Quantum energy function of free variables.
        """
        # Setup by angle_setting
        if angles_setting is None:
            print(f"[Opt] angle_setting: All angles will be optimized")
            num_variables = 2 * cef_obj.num_SL
            opt_angles_vars = list(range(num_variables))

        elif isinstance(angles_setting, (tuple, list)):
            if len(angles_setting) != 2 * cef_obj.num_SL:
                raise ValueError(
                    f"angles_setting length must be {2 * cef_obj.num_SL}, "
                    f"got {len(angles_setting)}"
                )
            opt_angles_vars = [
                i for i, angle in enumerate(angles_setting) if angle is None
            ]
            num_variables = len(opt_angles_vars)

            if verbose:
                if len(opt_angles_vars) == 0:
                    print("All angles are fixed")
                else:
                    print(f"[Opt] angle_setting: {angles_setting}")

        else:
            raise ValueError(
                "Invalid angles_setting. Must be None, tuple, or list."
            )

        E_cl_func = self._create_fixed_variable_function(
            cef_obj.classical_energy_density_func,
            angles_setting,
            opt_angles_vars,
        )
        E_qm_func = self._create_fixed_variable_function(
            cef_obj.quantum_energy_density_func,
            angles_setting,
            opt_angles_vars,
        )
        E_tot_func = self._create_fixed_variable_function(
            cef_obj.energy_func,
            angles_setting,
            opt_angles_vars,
        )

        bounds = [(-np.pi, np.pi)] * num_variables
        return bounds, E_tot_func, E_cl_func, E_qm_func

    @staticmethod
    def _create_fixed_variable_function(E_func, angle_setting, opt_angles_vars):
        """Create a reduced energy function with fixed variables substituted.

        Parameters
        ----------
        E_func : callable
            Original energy function accepting all angles.
        angle_setting : list or None
            Full angle list with None for free variables.
        opt_angles_vars : list of int
            Indices of free variables in the full angle list.

        Returns
        -------
        fixed_variable_function : callable
            Energy function accepting only the free variables.
        """
        def fixed_variable_function(reduced_variables):
            if angle_setting is None:
                return E_func(reduced_variables)
            current_angles = list(angle_setting)
            for i, idx in enumerate(opt_angles_vars):
                current_angles[idx] = reduced_variables[i]
            return E_func(current_angles)

        return fixed_variable_function

    @staticmethod
    def find_optimum_w_DE(func, bounds):
        """Find optimum using differential evolution.

        Parameters
        ----------
        func : callable
            Objective function to minimize.
        bounds : list of tuple
            Bounds for each variable.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Optimization result.
        """
        result = differential_evolution(
            func,
            bounds,
            strategy='best1bin',
            popsize=18,
            tol=1e-9,
            mutation=(0.5, 0.9),
            recombination=0.8,
            maxiter=800,
            polish=True,
            updating='immediate',
            seed=42,
        )
        return result

    @staticmethod
    def find_optimum_w_BFGS(func, bounds, init_points, randomness=0.0):
        """Find optimum using L-BFGS-B starting from initial points.

        Parameters
        ----------
        func : callable
            Objective function to minimize.
        bounds : list of tuple
            Bounds for each variable.
        init_points : np.ndarray
            Initial guess for the variables.
        randomness : float, optional
            Scale of random perturbation to initial points (default: 0.0).

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Optimization result.
        """
        if randomness != 0.0:
            init_points += 2 * np.pi * np.random.rand(len(bounds)) * randomness
        result = minimize(
            func,
            init_points,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-11, 'gtol': 1e-11},
        )
        return result

    def find_optimum_w_BFGS_from_DE(self, func, bounds, init_points,
                                     randomness=0.01, repeat=2):
        """Refine DE result using multiple L-BFGS-B runs with perturbation.

        Parameters
        ----------
        func : callable
            Objective function to minimize.
        bounds : list of tuple
            Bounds for each variable.
        init_points : np.ndarray
            Initial guess (typically from DE result).
        randomness : float, optional
            Scale of random perturbation for repeated runs (default: 0.01).
        repeat : int, optional
            Number of L-BFGS-B runs with perturbation (default: 2).

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Best optimization result among all runs.
        """
        result = self.find_optimum_w_BFGS(func, bounds, init_points, randomness=0.0)

        for i in range(repeat):
            result_new = self.find_optimum_w_BFGS(
                func, bounds, init_points, randomness=randomness
            )
            if result_new.fun < result.fun:
                result = result_new

        return result

    def find_minimum(self, cef_obj, opt_method, angle_setting=None,
                     verbose=False, full_range_search=False, num_search=6):
        """Find the minimum energy spin configuration.

        Parameters
        ----------
        cef_obj : EnergyFunction
            Energy function object.
        opt_method : str
            Optimization method name. One of OPT_METHOD_NAMES.
        angle_setting : list or None, optional
            Angle constraints (default: None, all free).
        verbose : bool, optional
            If True, print optimization progress (default: False).
        full_range_search : bool, optional
            If True, search over full 2*pi range for MAGSWT (default: False).
        num_search : int, optional
            Number of grid search points for MAGSWT (default: 6).

        Returns
        -------
        opt_result : dict
            Optimization result with keys: 'energy', 'angles', 'method',
            'E_cl', 'E_qm', 'MAGSWT'.
        cl_result : dict
            Classical optimization result with keys: 'E_cl', 'angles'.
        """
        if verbose:
            print(f"[Optimizer] Starting optimization with {opt_method}")

        bounds, E_tot_func, E_cl_func, E_qm_func = self.wrapping_by_angles(
            cef_obj, angle_setting, verbose=verbose
        )

        cef_obj.set_update_args(True)

        # Classical optimization
        DE_result = self.find_optimum_w_DE(E_cl_func, bounds)
        best_energy = DE_result.fun  # classical minimum energy
        best_angles = DE_result.x

        full_angles = self.recover_angles(best_angles, angle_setting)

        if verbose:
            print("=" * 30)
            print(f"Classical angles: {best_angles}")
            print(f"Classical energy: {best_energy}")
            print("=" * 30)

        # Calculate energies with full angles
        E_qm = 0

        if opt_method in CLASSICAL_METHOD_NAME:
            best_angles = full_angles
            E_cl = best_energy
            E_qm = cef_obj.quantum_energy_density_func(best_angles)
            mu_magswt = cef_obj.mu_magswt
            best_energy = E_cl + E_qm
            best_method = 'DE'

        elif opt_method in MAGSWT_METHOD_NAME:
            MAGSWT_result = self.magswt_for_nbcp(
                cef_obj,
                opt_angles=full_angles,
                opt_energy=best_energy,
                tl_angle=None,
                verbose=verbose,
                full_range_search=full_range_search,
                num_search=num_search,
            )

            # Optimization results from MAGSWT
            best_angles = MAGSWT_result["angles"]
            best_energy = MAGSWT_result["energy"]
            best_method = MAGSWT_result["method"]

            E_cl = MAGSWT_result["E_cl"]
            E_qm = MAGSWT_result["E_qm"]
            mu_magswt = MAGSWT_result["mu_MAGSWT"]

        elif opt_method in QUANTUM_METHOD_NAME:
            LSWT_result = self.find_optimum_w_BFGS_from_DE(
                E_tot_func, bounds, best_angles
            )

            if LSWT_result.fun < best_energy:
                # Optimization results from classical + quantum energy function
                best_energy = LSWT_result.fun
                best_angles = LSWT_result.x
                best_method = 'DE+BFGS'
                E_cl = E_cl_func(best_angles)
                E_qm = E_qm_func(best_angles)
                mu_magswt = cef_obj.mu_magswt  # get chemical potential

                # Recover full angle list
                best_angles = self.recover_angles(best_angles, angle_setting)

        cef_obj.set_update_args(False)

        if verbose:
            print(
                f"[Optimizer] Completed: Energy={best_energy:.6f}, "
                f"Method={best_method}"
            )

        opt_result = {
            "energy": best_energy,
            "angles": best_angles,
            "method": best_method,
            "E_cl": E_cl,
            "E_qm": E_qm,
            "MAGSWT": mu_magswt,
        }

        cl_result = {
            "E_cl": DE_result.fun,
            "angles": self.recover_angles(DE_result.x, angle_setting),
        }

        return opt_result, cl_result

    @staticmethod
    def magswt_for_nbcp(cef_obj, opt_angles, opt_energy, tl_angle=None,
                        verbose=False, full_range_search=False, num_search=6):
        """MAGSWT optimization by discrete rotation grid search.

        Searches for the minimum total energy (classical + quantum) by
        rotating all spins through a discrete set of angles.

        Parameters
        ----------
        cef_obj : EnergyFunction
            Energy function object.
        opt_angles : np.ndarray
            Initial (classically optimized) spin angles.
        opt_energy : float
            Classical energy at opt_angles.
        tl_angle : list or None, optional
            Rotation angle step [theta, phi]. If None, uses default
            based on lattice symmetry (default: None).
        verbose : bool, optional
            If True, print search progress (default: False).
        full_range_search : bool, optional
            If True, search over full 2*pi range (default: False).
        num_search : int, optional
            Number of discrete rotation steps (default: 6).

        Returns
        -------
        result : dict
            Dictionary with keys: 'angles', 'energy', 'method',
            'E_cl', 'E_qm', 'mu_MAGSWT'.
        """
        def angle_wrap(angles):
            """Wrap angles back to standard spherical coordinates."""
            reshaped = angles.reshape(-1, 2)
            new_angles = []
            for theta, phi in reshaped:
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                new_theta = np.arccos(z)
                new_phi = np.arctan2(y, x)
                new_angles.append([new_theta, new_phi])
            return np.array(new_angles).flatten()

        if tl_angle is None:
            theta_default = 0.0
            if full_range_search:
                num_search *= 2
                phi_default = 2 * np.pi / num_search  # default: 2*pi / 6
            else:
                phi_default = np.pi / num_search  # default: pi / 6

            tl_angle = [theta_default, phi_default]  # default rotation step [theta, phi]
            if verbose:
                print(
                    f"[Opt] Grid searches angle difference: "
                    f"theta={theta_default}, phi={phi_default}"
                )
        else:
            if len(tl_angle) != 2 or not isinstance(tl_angle, (list, tuple)):
                raise ValueError(
                    "tl_angle should be a list or tuple of two angles. "
                    "(theta, phi)"
                )

        cef_obj.set_update_args(True)
        num_sl = cef_obj.num_SL

        # Initial data from classical optimization
        best_method = "DE(MAGSWT)"
        best_angles = opt_angles  # angles initialization: classical optimal angles
        best_E_cl = opt_energy  # classical optimal energy
        best_E_qm = cef_obj.quantum_energy_density_func(best_angles)  # quantum correction
        mu_magswt = cef_obj.mu_magswt  # chemical potential for positive energy
        best_energy = best_E_cl + best_E_qm  # total energy

        for i in range(1, 1 + num_search):
            # Angle rotation respecting the lattice symmetry
            discrete_rot = np.array(tl_angle * num_sl) * i
            rot_angles = angle_wrap(opt_angles + discrete_rot)

            E_cl = cef_obj.classical_energy_density_func(rot_angles)
            E_qm = cef_obj.quantum_energy_density_func(rot_angles)
            E_tot = E_cl + E_qm

            if verbose:
                print(
                    f"[{i}th cal] E_cl: {E_cl:.5f}, E_qm: {E_qm:.5f}, "
                    f"E_tot: {E_tot:.5f}"
                )

            if E_tot < best_energy:
                if verbose:
                    print("=" * 30)
                    print(
                        f"New minimum energy detected at "
                        f"{np.round(rot_angles, 3)}"
                    )
                    print(f"E_new = {E_tot}")
                    print(f"E_old = {best_energy}")
                    print(f"Diff = {E_tot - best_energy}")
                    print("=" * 30)
                best_angles = rot_angles
                best_energy = E_tot
                best_E_cl = E_cl
                best_E_qm = E_qm
                mu_magswt = cef_obj.mu_magswt
                best_method = "MAGSWT"

        # Reset update flag
        cef_obj.set_update_args(False)

        return {
            "angles": best_angles,
            "energy": best_energy,
            "method": best_method,
            "E_cl": best_E_cl,
            "E_qm": best_E_qm,
            "mu_MAGSWT": mu_magswt,
        }

    @staticmethod
    def recover_angles(angles, angle_setting):
        """Recover full angle array from reduced (free) variables.

        Parameters
        ----------
        angles : np.ndarray
            Reduced angle array (free variables only).
        angle_setting : list or None
            Full angle constraint list. None entries correspond to free
            variables; numeric entries are fixed.

        Returns
        -------
        full_angles : np.ndarray
            Full angle array with fixed values restored.
        """
        full_angles = []
        opt_idx = 0
        for angle in angle_setting:
            if angle is None:
                full_angles.append(angles[opt_idx])
                opt_idx += 1
            else:
                full_angles.append(angle)
        return np.array(full_angles)
