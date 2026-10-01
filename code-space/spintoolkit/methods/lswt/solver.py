"""High-level LSWT solver.

This module provides the LSWTSolver class, which orchestrates the full
LSWT workflow: Brillouin zone construction, Hamiltonian diagonalization,
quantum corrections, and access to thermodynamic / topological / correlation
observables.

Ported from: modules/LinearSpinWaveTheory/linear_spin_wave_theory.py
"""


import numpy as np
from typing import Tuple, List, Dict, Optional, Union

from spintoolkit._deprecation import internal_use, warn_deprecated
from spintoolkit.system.spin_system import SpinSystem
from spintoolkit.system.brillouin_zone import BrillouinZone
from spintoolkit.observables.bose_statistics import compute_static_magnon_kernel
from spintoolkit.methods.base import AbstractSolver, SolverResult
from spintoolkit.methods.lswt.hamiltonian import LSWTHamiltonian

# Observables are not attached to this deprecated solver (D30). They act on the
# LSWTResult of solve_lswt: observables.thermal.thermal_quantities,
# observables.berry (berry_curvature, chern_numbers, thermal_hall),
# observables.structure_factor, observables.bands.band_structure and
# observables.zero_modes.scan_zero_modes.


class LSWTSolver(AbstractSolver):
    """High-level Linear Spin Wave Theory solver (deprecated, D30).

    Use :func:`spintoolkit.methods.lswt.run.solve_lswt` on the common model
    types (``SpinModel``, ``SpinState``, ``ExternalConditions``); it keeps the
    diagonalization for the observables and reproduces this solver (stage
    4a). This class stays until the public-release cleanup.

    Parameters
    ----------
    system : SpinSystem or dict
        Spin system to solve. Accepts either a SpinSystem object
        or a legacy dict with keys 'Spin info', 'Couplings',
        'Lattice/BZ setting'.

    Examples
    --------
    >>> from spintoolkit import SpinSystem, LSWTSolver
    >>> from spintoolkit.system import exchange
    >>> import numpy as np
    >>>
    >>> sites = [SpinSystem.Site("A", [0, 0], spin=0.5,
    ...          angles=[np.pi/2, 0], magnetic_field=[0, 0, 0])]
    >>> J = exchange.heisenberg(1.0)
    >>> couplings = [SpinSystem.Coupling(0, 0, J, [1.0, 0.0])]
    >>> system = SpinSystem(sites, couplings, [[1, 0], [0.5, 0.866]])
    >>>
    >>> solver = LSWTSolver(system)
    >>> result = solver.solve(N=10, bz_type="Hex_60")
    >>> print(result.ground_state_energy)
    """

    def __init__(self, system: Union[SpinSystem, dict],
                 bz_type: str = "Hex_60"):
        warn_deprecated("LSWTSolver", "use spintoolkit.solve_lswt with SpinModel/SpinState")
        # Accept both SpinSystem and legacy dict
        if isinstance(system, dict):
            with internal_use():
                self._system = SpinSystem.from_legacy_dict(system)
            self._legacy_data = system
            # Extract bz_type from legacy dict if present
            if "Lattice/BZ setting" in system:
                _, legacy_bz = system["Lattice/BZ setting"]
                bz_type = legacy_bz
        else:
            self._system = system
            self._legacy_data = system.to_legacy_dict(bz_type=bz_type)

        super().__init__(self._system)

        self._bz_type = bz_type
        self.spin_system_data = self._legacy_data
        self.spin_info = self.spin_system_data["Spin info"]
        self.couplings = self.spin_system_data["Couplings"]
        self.lattice_bz_settings = self.spin_system_data["Lattice/BZ setting"]
        self.Ns = len(self.spin_info)
        self.T = 0
        self.Ham = LSWTHamiltonian(self.spin_info, self.couplings)

    def solve(self, N: int = 10, bz_type: Optional[str] = None,
              regularization: str = "MAGSWT",
              temperature: float = 0) -> SolverResult:
        """Run the full LSWT calculation and return standardized results.

        Parameters
        ----------
        N : int, optional
            BZ mesh density (default: 10).
        bz_type : str or None, optional
            Brillouin zone type. If None, uses the system's bz_type.
        regularization : str, optional
            Regularization scheme (default: 'MAGSWT').
        temperature : float, optional
            Temperature in Kelvin for boson occupations (default: 0).
            ``ground_state_energy`` remains the zero-temperature energy.

        Returns
        -------
        result : SolverResult
            Standardized solver output. ``ground_state_energy`` is the
            zero-temperature energy per spin for the supplied reference
            configuration. If regularization shifts the quadratic Hamiltonian,
            its zero-point contribution uses that same shifted Hamiltonian.
        """
        if bz_type is None:
            bz_type = self._bz_type

        k_data, bz_data, full_k_points = self.diagnosing_lswt(
            bz_type=bz_type, regularization=regularization,
            N=N, temperature=temperature,
        )

        # Collect eigenvalues into array: (num_k, num_bands)
        eigenvalues = []
        zero_point_per_k = []
        for k_key in sorted(k_data.keys()):
            hamiltonian_data, eigen_data, *_ = k_data[k_key]
            evals, _ = eigen_data
            eigenvalues.append(evals[:self.Ns])
            # Full BdG trace includes both Nambu sectors. Subtract its vacuum
            # constant using the same (possibly regularized) H as these modes.
            trace_hk = np.trace(hamiltonian_data[0]).real
            zero_point_per_k.append(np.sum(evals[:self.Ns]) / 2 - trace_hk / 4)
        eigenvalues = np.array(eigenvalues)

        ground_state_energy = self._classical_energy() + np.mean(zero_point_per_k) / self.Ns

        return SolverResult(
            ground_state_energy=ground_state_energy,
            method="LSWT",
            eigenvalues=eigenvalues,
            spin_config=self.system.get_angles_flat(),
            data={
                "k_data": k_data,
                "bz_data": bz_data,
                "k_points": full_k_points,
                "boson_numbers": self.msl_average_boson_number,
                "average_boson_number": self.average_boson_number,
                "magswt_onsite": self.magswt_onsite,
                "regularization": regularization,
            },
        )

    def _classical_energy(self) -> float:
        """Compute classical energy per site from current spin configuration."""
        from spintoolkit.methods.lswt.energy import EnergyFunction
        ef = EnergyFunction(self.spin_system_data, N=1, update_args=False)
        return ef.classical_energy_density_func(self.system.get_angles_flat())

    def diagnosing_lswt(self, bz_type="Hex_60", regularization="MAGSWT",
                        N=10, temperature=0):
        """Run the full LSWT diagnosis: diagonalize over the BZ and compute corrections.

        Parameters
        ----------
        bz_type : str, optional
            Brillouin zone type (default: 'Hex_60').
        regularization : str, optional
            Regularization scheme (default: 'MAGSWT').
        N : int, optional
            BZ mesh density (default: 10).
        temperature : float, optional
            Temperature in Kelvin for boson occupations (default: 0).

        Returns
        -------
        k_data : dict
            Diagonalization results for all k-points.
        bz_data : dict
            Brillouin zone data.
        full_k_points : np.ndarray
            All k-points in the BZ mesh.
        """
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("Temperature must be finite and non-negative")
        self.T = temperature
        bz = BrillouinZone(self.lattice_bz_settings, bz_type=bz_type)
        bz_data, full_k_points, _ = bz.get_full(N)

        self.bz_type = bz_type
        self.bz_data = bz_data

        k_data, chem_pot_magswt = self.Ham.solve_k_Hamiltonian(
            full_k_points,
            Berry_curvature=True,
            regularization=regularization
        )
        # Private provenance for BZ integrals: a path or a sliced dictionary
        # must not be normalized as the full grid produced by this diagnosis.
        self._integration_k_keys = frozenset(k_data)

        self.msl_average_boson_number, self.average_boson_number = (
            self.lswt_correction(k_data=k_data)
        )
        self.regularization = regularization
        self.magswt_onsite = chem_pot_magswt

        return k_data, bz_data, full_k_points

    def lswt_correction(self, k_data):
        """Compute average boson numbers at the current diagnosis temperature.

        Uses ``self.T``, set by ``diagnosing_lswt`` (zero on initialization).

        Parameters
        ----------
        k_data : dict
            Diagonalization results keyed by k-point index.

        Returns
        -------
        msl_average_boson_number : dict
            Average boson number for each sublattice.
        average_boson_number : float
            Average boson number across all sublattices.
        """
        average_boson_numbers = 0
        sublattice_boson_numbers = np.zeros(self.Ns)

        for k_key in k_data.keys():
            _, Eigen_data, *_ = k_data[k_key]
            eval, evec = Eigen_data
            magnon_kernel = compute_static_magnon_kernel(
                eval, Temperature=self.T, Ns=self.Ns
            )
            two_point = evec @ np.diag(magnon_kernel) @ evec.T.conj()
            diag_elements = np.diag(two_point)[self.Ns:]
            sublattice_boson_numbers += np.real(diag_elements)

        sublattice_boson_numbers = sublattice_boson_numbers / len(k_data)

        msl_average_boson_number = {}
        average_boson_number = 0
        total_spin_moment = np.zeros(3, dtype=float)

        for j, (name_sl, value) in enumerate(self.spin_info.items()):
            spin = value["Spin"]
            msl_boson_num = sublattice_boson_numbers[j]
            average_boson_number += msl_boson_num
            msl_average_boson_number[name_sl] = msl_boson_num

            theta, phi = value["Angles"]
            new_spin = spin - msl_boson_num
            spin_moment = new_spin * np.array([
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ])
            total_spin_moment += spin_moment

        average_boson_number = average_boson_number / len(self.spin_info)

        return msl_average_boson_number, average_boson_number
