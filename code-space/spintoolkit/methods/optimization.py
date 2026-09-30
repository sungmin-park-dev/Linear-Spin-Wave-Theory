"""Spin system optimizer for finding ground state spin configurations.

This module provides the classical optimization routine for finding the
minimum energy spin configuration using differential evolution (DE); the
zero-point energy is evaluated at the classical optimum.

Zero-point selection among degenerate classical states is done on the
classical manifold by :func:`spintoolkit.methods.state_selection.select_on_manifold`
(D17). The former ``quantum`` method, an unconstrained minimization of
E_cl + E_qm, was removed (D18): its result is not a classical stationary
point, so the linear boson terms do not vanish and the energy goes beyond
LSWT order. The MAGSWT grid search (``opt_method='MAGSWT'``: E_cl + E_qm at
azimuth shifts ``k pi/6``) was removed (D27): the orbit search reproduces it on
its grid and finds lower energies between grid points, and
:func:`~spintoolkit.methods.state_selection.orbit_energy_landscape` draws the
energies along the classical orbit on any angle grid.
"""

import numpy as np
from typing import List, Union
from scipy.optimize import differential_evolution


# Optimization method name constants
CLASSICAL_METHOD_NAME = ["classical", "Classical", "CLASSICAL"]

OPT_METHOD_NAMES = list(CLASSICAL_METHOD_NAME)

#: Removed method names and the reason given when they are requested.
REMOVED_METHODS = {
    "quantum": (
        "opt_method 'quantum' was removed (D18): minimizing E_cl + E_qm without the "
        "classical-manifold constraint does not give a valid LSWT reference state. "
        "Use opt_method 'classical' and then "
        "spintoolkit.methods.state_selection.select_on_manifold (D17)."),
    "MAGSWT": (
        "opt_method 'MAGSWT' (grid search of E_cl + E_qm over azimuth shifts k pi/6) was "
        "removed (D27). Use opt_method 'classical', then "
        "spintoolkit.methods.state_selection.select_on_manifold for the zero-point selection "
        "on the classical manifold, or orbit_energy_landscape for E_cl and E_qm along the orbit."),
}
REMOVED_METHOD_NAMES = {"classical+quantum": "quantum", "quantum": "quantum",
                        "MAGSWT": "MAGSWT", "magswt": "MAGSWT"}


class SpinOptimizer:
    """Optimizer for finding ground state spin configurations.

    Classical optimization via differential evolution; the zero-point energy is
    evaluated at the classical optimum.

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

    def find_minimum(self, cef_obj, opt_method, angle_setting=None,
                     verbose=False):
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

        Returns
        -------
        opt_result : dict
            Optimization result with keys: 'energy', 'angles', 'method',
            'E_cl', 'E_qm', 'MAGSWT'. 'energy' is E_cl + E_qm and 'angles'
            is the full angle list; 'MAGSWT' is the regularization shift used
            for E_qm (not a search method).
        cl_result : dict
            Classical optimization result with keys: 'E_cl', 'angles'.

        Raises
        ------
        ValueError
            If ``opt_method`` is not one of OPT_METHOD_NAMES; the removed
            ``quantum`` names get a message pointing to the D17 selection.
        """
        if opt_method in REMOVED_METHOD_NAMES:
            raise ValueError(REMOVED_METHODS[REMOVED_METHOD_NAMES[opt_method]])
        if opt_method not in OPT_METHOD_NAMES:
            raise ValueError(f"unknown opt_method {opt_method!r}; use one of {OPT_METHOD_NAMES}")
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
    def recover_angles(angles, angle_setting):
        """Recover full angle array from reduced (free) variables.

        Parameters
        ----------
        angles : np.ndarray
            Reduced angle array (free variables only).
        angle_setting : list or None
            Full angle constraint list. None entries correspond to free
            variables; numeric entries are fixed. None means all angles
            are free.

        Returns
        -------
        full_angles : np.ndarray
            Full angle array with fixed values restored.
        """
        if angle_setting is None:
            return np.array(angles, dtype=float)
        full_angles = []
        opt_idx = 0
        for angle in angle_setting:
            if angle is None:
                full_angles.append(angles[opt_idx])
                opt_idx += 1
            else:
                full_angles.append(angle)
        return np.array(full_angles)
