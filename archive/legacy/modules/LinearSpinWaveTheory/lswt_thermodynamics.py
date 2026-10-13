import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple
from tqdm import tqdm  # 진행 표시줄

from modules.Tools.magnon_kernel import compute_bose_einstein_distribution, compute_static_magnon_kernel
from .lswt_topology import compute_Berry_curvature, c_two_function

from modules.constants import K_BOLTZMANN_meV, H_BAR_meV

"""
Default constant
"""
DEFAULT_INVALID_EXCLUDE = True
DEFAULT_TEMPERATURE = 0.0


def compute_bosonic_number_at_k(eval: np.ndarray, evec: np.ndarray, Temperature: float = 0, num_sl: Union[int, None] = None):
    
    num_sl = int(len(eval)//2) if num_sl is None else num_sl
        
    magnon_corr = np.diag(compute_static_magnon_kernel(eval, Temperature = Temperature))
    boson_corr  = evec @ magnon_corr @ evec.T.conj()
    diag_elements = np.diag(boson_corr)
        
    sublattice_boson_numbers_at_k = np.real(diag_elements[num_sl:]) 
    all_boson_number_at_k         = np.sum( sublattice_boson_numbers_at_k ) / num_sl
    
    return sublattice_boson_numbers_at_k, all_boson_number_at_k




class LSWT_THER:
    def __init__(self, lswt_obj = None):
        self.lswt_obj = lswt_obj
        self.Ns = lswt_obj.Ns if lswt_obj is not None else None

    def _infer_Ns_from_k_data(self, k_data):
        """k_data에서 Ns를 추론합니다."""
        if not k_data:
            raise ValueError("k_data가 비어 있어 Ns를 추론할 수 없습니다.")
        first_k_data = next(iter(k_data.values()))
        num_sl = len(first_k_data[1][0]) // 2  # Eigen_data의 첫 요소 길이의 절반
        return num_sl

    def compute_internal_energy(self, k_data, 
                               Temperature = DEFAULT_TEMPERATURE, 
                               invalid_exclude = DEFAULT_INVALID_EXCLUDE):
        """Compute the quantum contribution of internal energy per unit cell."""
        
        # Ns가 없으면 k_data에서 추론
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
                iter_count -= self._warning_Colpa_fails(kpt = k_key)
                continue

        
        if iter_count == 0:
            print("Warning: No valid k-points for calculation")
            return np.nan
        else:
            denominator = iter_count * num_sl  # Product of the number of unitcell and magnetic sublattice
            return E_quantum / denominator

    def bosonic_momentum_correlation(self, k_data, Temperature=0, exclude_gamma=False):
        """Compute boson numbers for each k-point."""
        
        # Ns가 없으면 k_data에서 추론
        num_sl = self.Ns if self.Ns is not None else self._infer_Ns_from_k_data(k_data)
        
        num_k_pts = len(k_data)                 # number k points in FBZ
        all_boson_numbers = np.zeros(num_k_pts) # average boson number
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
                    eval = eval, evec = evec, Temperature=Temperature, num_sl = num_sl
                    )
            else:
                iter_count -= self._warning_Colpa_fails(k_key)
                
        
        print(f"Total k-points: {num_k_pts}, Valid k-points: {iter_count}")
        
        return msl_boson_numbers, all_boson_numbers, eigenvalues_array, iter_count


    def compute_boson_numbers(self, k_data, Temperature = 0, exclude_gamma=False):
        """Compute average boson numbers."""
        sublat_num, total_num, _, valid_counts = self.bosonic_momentum_correlation(k_data = k_data,
                                                                                   Temperature = Temperature, 
                                                                                   exclude_gamma = exclude_gamma)
        
        if valid_counts == 0:
            print("Warning: No valid k-points for boson number calculation")
            return np.nan, np.nan
        
        average_total_boson_numbers = np.sum(total_num) / valid_counts
        average_sublat_boson_numbers = np.sum(sublat_num, axis=0) / valid_counts
        
        return average_sublat_boson_numbers, average_total_boson_numbers

    def compute_entropy_density(self, k_data, T, invalid_exclude=True):
        """Compute entropy density."""
        
        # Ns가 없으면 k_data에서 추론
        num_sl = self.Ns if self.Ns is not None else self._infer_Ns_from_k_data(k_data)
        
        entropy = 0
        valid_count = len(k_data)
        
        if T == 0:
            return 0
        elif T > 0:
            for k_key, contents in k_data.items():
                _, Eigen_data, Colpa_data, *_ = contents
                
                if invalid_exclude and not Colpa_data[0]:
                    valid_count -= self._warning_Colpa_fails(k_key)
                    continue
                else:
                    eval, _ = Eigen_data
                    Epk = eval[:num_sl]
                    nk = compute_bose_einstein_distribution(Epk, T)
                    entropy += self.entropy_function_at_k(nk)
            
            if valid_count == 0:
                print("Warning: No valid k-points for entropy calculation")
                return np.nan
            return K_BOLTZMANN_meV * entropy / valid_count
        else:
            raise ValueError("T is temperature. T should be positive number")

    def compute_specific_heat(self, k_data, T, invalid_exclude=True, exclude_gamma=True):
        """Compute specific heat"""
        
        # Ns가 없으면 k_data에서 추론
        num_sl = self.Ns if self.Ns is not None else self._infer_Ns_from_k_data(k_data)
        
        Cv = 0
        valid_count = len(k_data)
        
        if T == 0:
            return Cv
        elif T > 0:
            beta = 1 / (K_BOLTZMANN_meV * T)
            for k_key, contents in k_data.items():
                _, Eigen_data, Colpa_data, *_ = contents
                
                if invalid_exclude and not Colpa_data[0]:
                    valid_count -= self._warning_Colpa_fails(k_key)
                    continue
                else:
                    eval, _ = Eigen_data
                    Epk = eval[:num_sl]
                    nk = compute_bose_einstein_distribution(Epk, T)
                    Cv += self.specific_heat_function_at_k(Epk = Epk, beta=beta)
            
            if valid_count == 0:
                print("Warning: No valid k-points for specific heat calculation")
                return np.nan
            return K_BOLTZMANN_meV * Cv / valid_count
        else:
            raise ValueError("T is temperature. T should be positive number")

    @staticmethod
    def _warning_Colpa_fails(kpt: Union[Tuple, np.ndarray, None] = None):
        """Warn when Colpa's method fails."""
        print(f"[Warning] Colpa's method fails at k-point {kpt}")
        print("The bosonic Hamiltonian is not positive definite and non-positive eigenvalues may appear.")
        return 1

    @staticmethod
    def zero_point_energy_formula(Epk, Hk):
        E_magnon = np.sum(Epk)
        Trace_Hk = np.real(np.trace(Hk))
        return (E_magnon - Trace_Hk/2)/2
        
    @staticmethod
    def excitation_energy_formula(Epk, T):
        numbers = compute_bose_einstein_distribution(Epk, T)
        E_excitation = Epk * numbers
        return np.sum(E_excitation)

    @staticmethod
    def entropy_function_at_k(nk: np.ndarray):
        x = np.array(nk) 
        f = np.zeros_like(nk, dtype=float)

        # Handle extreme cases
        mask_too_small = x < 1e-323   # such x should be replaced with 0
        mask_too_large = x > 1e300    # such x should be replaced with np.pi**2/3
        normal_range   = ~mask_too_small & ~mask_too_large

        # Calculate for normal range values
        cal_x = x[normal_range]
        if cal_x.size > 0:  # Check if there are values to calculate
            I_p_cal_x = 1 + cal_x
            f[normal_range] = I_p_cal_x * np.log(I_p_cal_x) - cal_x * np.log(cal_x)
        return np.sum(f)

    @staticmethod
    def specific_heat_function_at_k(Epk: np.ndarray, beta: float):
        x = np.array(beta * Epk) 
        f = np.zeros_like(x, dtype=float)

        # Handle extreme cases
        mask_too_large = x > 680      # such x should be replaced with np.pi**2/3
        mask_too_small = x < 1e-150   # such x should be replaced with 0
        normal_range   = ~mask_too_small & ~mask_too_large

        # Calculate for normal range values
        cal_x = x[normal_range]
        if cal_x.size > 0:  # Check if there are values to calculate
            denominator = 4 * (np.sinh(cal_x/2))**2
            numerator = cal_x**2
            f[normal_range] = numerator / denominator

        small_x = x[mask_too_small]
        if small_x.size > 0:  # Check if there are values to calculate
            f[mask_too_small] = 1.0 

        return np.sum(f)

    def compute_thermodynamic_quantities_at_T(self, 
                                              k_data, 
                                              Temperature = DEFAULT_TEMPERATURE, 
                                              invalid_exclude = DEFAULT_INVALID_EXCLUDE):
        """
        Compute all thermodynamic quantities (boson number, internal energy, entropy, specific heat)
        at given temperature in a single pass.
        
        Parameters:
        -----------
        k_data : dict
            Dictionary containing k-point data with eigenvalues and eigenvectors
        Temperature : float, optional
            Temperature in energy units (default: DEFAULT_TEMPERATURE)
        invalid_exclude : bool, optional
            Whether to exclude k-points where Colpa's method fails (default: DEFAULT_INVALID_EXCLUDE)
        exclude_gamma : bool, optional
            Whether to exclude gamma point (default: False)
            
        Returns:
        --------
        dict
            Dictionary containing all computed thermodynamic quantities:
            - 'Sublattice Boson Numbers': Average boson numbers for each sublattice
            - 'Total Boson Number': Average total boson number
            - 'Internal Energy Density': Internal energy per unit cell
            - 'Entropy Density': Entropy density
            - 'Specific Heat Density': Specific heat
        """
        # Infer number of sublattices from k_data if not set
        num_sl = self.Ns if self.Ns is not None else self._infer_Ns_from_k_data(k_data)
        
        # Initialize counters and accumulators
        valid_count = len(k_data)
        
        total_boson_number = 0
        sublattice_boson_numbers = np.zeros(num_sl)
        internal_energy = 0
        entropy = 0
        specific_heat = 0
        thermal_hall = 0
        J_mat = np.diag(np.hstack([np.ones((self.Ns)), - np.ones((self.Ns))]))        
        
        # Set up for specific heat calculation if temperature > 0
        beta = 1 / (K_BOLTZMANN_meV * Temperature) if Temperature > 0 else 0
        
        # Process each k-point
        for k_key, contents in k_data.items():
            Ham_k_data, Eigen_data, Colpa_data, *_ = contents
            colpa_success = Colpa_data[0]
            
            # Skip invalid k-points if specified
            if not colpa_success and invalid_exclude:
                valid_count -= self._warning_Colpa_fails(kpt=k_key)
                continue
            else: 
                # Extract data
                Hk = Ham_k_data[0]
                pDHk = Ham_k_data[1:]
                
                eval, evec = Eigen_data
                Epk = eval[:num_sl]  # Positive eigenvalues for magnons
                
                # 1. Calculate zero-point energy contribution to internal energy
                internal_energy += self.zero_point_energy_formula(Epk = Epk, Hk = Hk)
                
                if Temperature > 0:
                    # Calculate Bose-Einstein distribution
                    nk = compute_bose_einstein_distribution(Epk, Temperature)
                    
                    # 2. Calculate thermal excitation contribution to internal energy
                    E_excitation = np.sum(Epk * nk)
                    internal_energy += E_excitation
                    
                    # 3. Calculate entropy
                    entropy += self.entropy_function_at_k(nk)
                    
                    # 4. Calculate specific heat
                    specific_heat += self.specific_heat_function_at_k(Epk = Epk, 
                                                                      beta = beta)
                    # 5. thermal_hall_conductance
                    Omega_nk, _ = compute_Berry_curvature(eval = eval,
                                                          evec = evec, 
                                                          pDiffHk = pDHk,
                                                          num_sl = self.Ns, 
                                                          J_mat = J_mat)
                    
                    thermal_hall += np.sum(Omega_nk * c_two_function(nk))

                
                # 5. Calculate boson numbers
                sl_boson_nums, total_boson_num = compute_bosonic_number_at_k(
                    eval = eval, evec = evec, Temperature = Temperature, num_sl = num_sl
                )
                sublattice_boson_numbers += sl_boson_nums
                total_boson_number += total_boson_num
        
        
        # Check if we have valid k-points
        if valid_count == 0:
            print("Warning: No valid k-points for calculation")
            return {'Sublattice Boson Numbers': np.nan,
                    'Total Boson Number': np.nan,
                    'Internal Energy Density': np.nan,
                    'Entropy Density': np.nan,
                    'Specific Heat Density': np.nan, 
                    'Thermal Hall Conductance': np.nan}
        else:        
            # Normalize results
            number_of_lattices = (valid_count * num_sl)
            
            sublattice_boson_numbers /= number_of_lattices 
            total_boson_number /= number_of_lattices 
            internal_energy /= number_of_lattices   # Per unit cell
            entropy /= number_of_lattices 
            specific_heat /= number_of_lattices 
            
            real_space_volume = valid_count * np.sqrt(3)/2
            coefficiten_thc = (K_BOLTZMANN_meV ** 2) /  (H_BAR_meV * real_space_volume)  # * Temperature
            coefficiten_thc *= 1.602176634 * 1e-12 # from meV, Angstrom to Watt / (meter Kelvin)
            
            thermal_hall_conductance = - coefficiten_thc * thermal_hall 
            
            # Apply Boltzmann constant to entropy and specific heat
            if Temperature > 0:
                entropy *= K_BOLTZMANN_meV
                specific_heat *= K_BOLTZMANN_meV
        
            return {'Sublattice Boson Numbers': sublattice_boson_numbers,
                    'Total Boson Number': total_boson_number,
                    'Internal Energy Density': internal_energy,
                    'Entropy Density': entropy,
                    'Specific Heat Density': specific_heat, 
                    'Thermal Hall Conductance': thermal_hall_conductance}


    def get_thermodynamic_quantities(self, k_data, 
                                     Temperature_range = (0, 1, 0.02),
                                     N = 30):
        
        T_start, T_end, T_step = Temperature_range
        T_end += T_step/2
        
        Temperature_values = np.arange(T_start, T_end, T_step)
        
        num_T = len(Temperature_values)
        
        sublattice_boson_numbers = np.empty( (self.Ns, num_T) )
        total_boson_number = np.empty( num_T )
        internal_energy = np.empty( num_T )
        entropy = np.empty( num_T )
        specific_heat = np.empty( num_T )
        thermal_hall_conductance = np.empty( num_T )
        
        for j, T in enumerate(tqdm(Temperature_values, desc="Calculating thermodynamics")):
            result_T = self.compute_thermodynamic_quantities_at_T(k_data = k_data, 
                                                                       Temperature = T)
            
            sublattice_boson_numbers[:, j] = result_T['Sublattice Boson Numbers']
            total_boson_number[j]          = result_T['Total Boson Number']
            internal_energy[j]             = result_T['Internal Energy Density']
            entropy[j]                     = result_T['Entropy Density']
            specific_heat[j]               = result_T['Specific Heat Density']
            thermal_hall_conductance[j]    = result_T['Thermal Hall Conductance']

        results =  {'Internal Energy Density': internal_energy,
                    'Entropy Density': entropy,
                    'Specific Heat Density': specific_heat, 
                    'Thermal Hall Conductance': thermal_hall_conductance, 
                    'Sublattice Boson Numbers': sublattice_boson_numbers,
                    'Total Boson Number': total_boson_number,}

        return Temperature_values, results

    
    def plot_thermodynamic_quantities(self, temperatures, results_dict):
        T = temperatures

        quantities = [
            ('Internal Energy', results_dict['Internal Energy Density'], "E (meV)"),
            ('Entropy Density', results_dict['Entropy Density'], "S (meV/K)"),
            ('Specific Heat',   results_dict['Specific Heat Density'], "C (meV/K)"),
            ('Thermal Hall Conductance', results_dict['Thermal Hall Conductance'], "κ_xy / T (W/(m·K)")
        ]

        sublattice_boson_numbers = results_dict['Sublattice Boson Numbers']
        total_boson_number       = results_dict['Total Boson Number']
        num_sublattices = sublattice_boson_numbers.shape[0]

        linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
        alphas     = [0.6, 0.7, 0.8, 0.5, 0.65]
        linewidths = [1.0, 1.2, 1.5, 0.8, 1.3]  # 다양하게 설정

        fig, axs = plt.subplots(3, 2, figsize=(12, 10))
        axs = axs.flatten()

        for i, (title, data, ylabel) in enumerate(quantities):
            axs[i].plot(T, data)
            axs[i].set_title(title)
            axs[i].set_xlabel("Temperature (K)")
            axs[i].set_ylabel(ylabel)

        # Boson number → axs[4]
        axs[4].plot(T, total_boson_number, label='Total', color='black', linewidth=2)
        for i in range(num_sublattices):
            axs[4].plot(
                T,
                sublattice_boson_numbers[i],
                linestyle=linestyles[i % len(linestyles)],
                alpha=alphas[i % len(alphas)],
                linewidth=linewidths[i % len(linewidths)],
                label=f"Sublattice {chr(65 + i)}"
            )
        axs[4].set_title("Boson Number")
        axs[4].set_xlabel("Temperature (K)")
        axs[4].set_ylabel("⟨n⟩")
        axs[4].legend()

        # 마지막 빈 plot
        axs[5].axis("off")

        plt.tight_layout()
        plt.show()

        return

