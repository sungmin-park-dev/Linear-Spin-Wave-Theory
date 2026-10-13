import numpy as np
from typing import Union, List, Tuple
from scipy.special import spence
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


from modules.Tools.magnon_kernel import compute_bose_einstein_distribution
from modules.constants import K_BOLTZMANN_meV, H_BAR_meV

"""
Tools for computing magnon topology
"""

DEFAULT_LEVEL_SPACING = 1e-2 # (meV)


def c_two_function(x: Union[float, list, np.ndarray]):
    x = np.array(x)                     # Input (Domain), bose einstein distribution
    f = np.zeros_like(x, dtype = float) # Create output array with same shape as input
    
    # Handle edge cases
    mask_too_small = x < 1e-300   # such x should be replaced with 0
    mask_too_large = x > 1e300    # such x should be replaced with np.pi**2/3
    normal_range   = ~mask_too_small & ~mask_too_large

    f[mask_too_large] = np.pi**2/3 # Set values for edge cases, zeros are already set.
    
    # Calculate for normal range values
    cal_x = x[normal_range]    
    if cal_x.size > 0:  # Check if there are values to calculate
        term1 = (1 + cal_x) * (np.log((1 + cal_x) / cal_x))**2
        term2 = (np.log(cal_x))**2
        term3 = 2 * spence(1 + cal_x)
        f[normal_range] = term1 - term2 - term3
    
    return f # - np.pi**2/3

def compute_Berry_curvature(eval: np.ndarray,   # magnon dispersion
                            evec: np.ndarray,   # Bogolioubouv transformation
                            pDiffHk: List[np.ndarray],  # partial differentiantion of bosonic Hamiltonian
                            num_sl = None,      # sublattice number
                            J_mat = None ):
    
    num_sl = num_sl if num_sl is not None else int(len(eval)//2)
    J_mat  = J_mat if J_mat is not None else np.diag(np.hstack([np.ones(num_sl), - np.ones(num_sl)])) 
    
    Omega_nk = []
    level_spacing = []
    
    pDxHk, pDyHk = pDiffHk
    
    J_eval = np.diag(J_mat) * eval 

    partial_H_x = J_mat @ evec.conj().T @ pDxHk @ evec
    partial_H_y = J_mat @ evec.conj().T @ pDyHk @ evec
        
    for n in range(num_sl):
        summand = 0
        min_E_diff = np.inf
        
        for m in range(2*num_sl):
            if n == m:
                continue
            else: 
                diff_En_Em = J_eval[n] - J_eval[m]
                denominator = (diff_En_Em) ** 2
                if denominator == 0:
                    continue  # skip this term to avoid nan
                numerator = partial_H_x[n, m] * partial_H_y[m, n]
                summand += numerator / denominator
                
                if n == m + 1 or n == m - 1:
                    min_E_diff = np.minimum(min_E_diff, np.abs(diff_En_Em))
        
        if n == 0:
            min_E_diff = np.minimum(min_E_diff, eval[n])
            
            
        Omega_nk.append(- 2 * np.imag(summand) )
        level_spacing.append( min_E_diff )
    
    return np.array(Omega_nk), np.array( level_spacing )


def _warning_Colpa_fails(kpt: Union[Tuple, np.ndarray, None] = None):
    """
    Colpa method is correct way of solving quadratic bose system.
    The colpa method works for any positive definite Hamitonian. 
    If it fails, it means that a given Hamiltonian is unphysical.
    """
    print(f"[Warning] Colpa's method fails at k-point {kpt}")
    print("The bosonic Hamiltonian is not positive definite and non-positive eigenvalues may appear.") 
    return 1

"""
Magnon topology
- Berry curvature
- Chern number
- Thermal Hall coefficient
"""

class LSWT_TOPO:
    def __init__(self, lswt_obj, 
                 num_sl = None, ):
        self.Ns = lswt_obj.Ns
        self.J_mat = np.diag(np.hstack([np.ones((self.Ns)), - np.ones((self.Ns))]))        
        
        self.area = lswt_obj.bz_data["area"]  
        return

    def thermal_weight_function(self, eval: np.ndarray, Temperature: Union[float, int]):
        Epk = eval[:self.Ns]
        
        if Temperature == 0 :
            return np.zeros_like(Epk)
        
        else: 
            nk = compute_bose_einstein_distribution(E_list = Epk, Temperature = Temperature)
            return c_two_function(nk)
    
    def compute_thermal_Hall(self, k_data, Temperature, bz_type = None):
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
        thermal_hall    = 0
        
        for j, contents in enumerate(k_data.values()):
            H_k_data, Eigen_k_data, Colpa_data, *_ = contents
            
            colpa_success = Colpa_data[0]
            if colpa_success:
                eval, evec = Eigen_k_data
                pDHk = H_k_data[1:]
                
                Omega_nk, level_spacing = compute_Berry_curvature(eval = eval,
                                                                  evec = evec, 
                                                                  pDiffHk = pDHk,
                                                                  num_sl = self.Ns, 
                                                                  J_mat = self.J_mat)
                Berry_curvature[j] = Omega_nk
                min_level_spacing = np.minimum( min_level_spacing, level_spacing )
                
                thermal_hall += np.sum(Omega_nk * self.thermal_weight_function(eval, Temperature = Temperature))
            else:
                Berry_curvature[j] = np.full(self.Ns, np.nan)
                valid_count -= _warning_Colpa_fails()


        chern_number = (1/(2*np.pi)) * np.nansum(Berry_curvature, axis = 0) * self.area / BZ_normalizer
        
        print(f"self.area: {self.area}, BZ_normalizer: {BZ_normalizer}")
        # BZ_volume = valid_count * self.area
        # real_space_volume = (2 * np.pi) ** 2 / BZ_volume
        # coefficiten_thc = (K_BOLTZMAN_meV ** 2) / real_space_volume * Temperature
        
        real_space_volume = valid_count * np.sqrt(3)/2 
        # valid count = number of k-points in the BZ
        coefficiten_thc = (K_BOLTZMANN_meV ** 2) /  (H_BAR_meV * real_space_volume)  # * Temperature
        coefficiten_thc *= 1.602176634 * 1e-12 # from meV, Angstrom to Watt / (meter Kelvin)
            
        THC = - coefficiten_thc * thermal_hall * 10**6 # from W to μW
        
        print("Chern number — ordered from low to high energy bands:")
        for j, C_num in enumerate(chern_number):
            spacing = min_level_spacing[j]
            if spacing < DEFAULT_LEVEL_SPACING:
                print(f"{self.Ns - j}-th band Chern number: {C_num:.5f}  [Warning: insufficient level spacing ΔE = {spacing:.5e}]")
            else:
                print(f"{self.Ns - j}-th band Chern number: {C_num:.5f}")

        print(f"Thermal Hall Conductivity (κ_xy / T):\n{THC} (μW/(m·K)")

        
        return Berry_curvature, chern_number, THC
        
            
            
    def plot_berry_curvature(self, full_k_points, Berry_curv, band_index=0, title="Berry Curvature", grid_size=50):
        """
        Plot 2D Berry curvature over the Brillouin zone, handling both regular and irregular k-point grids.
        
        Parameters:
        - full_k_points: List of [kx, ky] coordinates, shape (num_k_points, 2)
        - Berry_curv: Berry curvature array, shape (num_k_points, Ns)
        - band_index: Index of the band to plot (default: 0)
        - title: Plot title
        - grid_size: Size of the interpolation grid (default: 50 for 50x50)
        """
        band_index = (self.Ns - band_index) % self.Ns  # Ensure band_index is within bounds
        
        # Convert full_k_points to numpy array
        k_points = np.array(full_k_points)
        if k_points.shape[0] != Berry_curv.shape[0]:
            raise ValueError(f"Mismatch: {k_points.shape[0]} k-points vs {Berry_curv.shape[0]} Berry curvature points")
        
        # Extract kx, ky coordinates
        kx = k_points[:, 0]
        ky = k_points[:, 1]
        berry = np.log(np.abs(Berry_curv[:, band_index]) + 1e-10)  # Avoid log(0)
        # berry = Berry_curv[:, band_index] 
        
        # Check unique kx, ky values
        kx_unique = np.unique(kx)
        ky_unique = np.unique(ky)
        print(f"Unique kx: {len(kx_unique)}, Unique ky: {len(ky_unique)}")
        print(f"Total k-points: {len(kx)}, Expected grid: {grid_size}x{grid_size} = {grid_size**2}")
        
        # Try to reshape for a regular grid
        if len(kx) == grid_size * grid_size and len(kx_unique) == grid_size and len(ky_unique) == grid_size:
            print("Assuming regular grid...")
            KX, KY = np.meshgrid(kx_unique, ky_unique)
            berry_grid = berry.reshape(grid_size, grid_size)
            
            # Contour plot
            plt.figure(figsize=(8, 6))
            # contour = plt.contourf(KX, KY, berry_grid, levels=50, cmap='virumbled')
            contour = plt.contourf(KX, KY, berry_grid, levels=50, cmap='virumbled')
            plt.colorbar(contour, label='Berry Curvature')
            plt.xlabel(r'$k_x$')
            plt.ylabel(r'$k_y$')
            plt.title(f'{title} (Band {band_index + 1})')
            plt.gca().set_aspect('equal')
            plt.show()
        else:
            print("Irregular grid detected, using interpolation...")
            # Create a regular grid for interpolation
            kx_min, kx_max = np.min(kx), np.max(kx)
            ky_min, ky_max = np.min(ky), np.max(ky)
            kx_grid = np.linspace(kx_min, kx_max, grid_size)
            ky_grid = np.linspace(ky_min, ky_max, grid_size)
            KX, KY = np.meshgrid(kx_grid, ky_grid)
            
            # Interpolate Berry curvature
            berry_grid = griddata(k_points, berry, (KX, KY), method='cubic', fill_value=np.nan)
            
            # Contour plot
            plt.figure(figsize=(8, 6))
            # contour = plt.contourf(KX, KY, berry_grid, levels=50, cmap='viridis')
            contour = plt.contourf(KX, KY, berry_grid, levels=50, cmap='seismic')
            
            band_index = (self.Ns - band_index) % self.Ns  # Ensure band_index is within bounds
            
            plt.colorbar(contour, label='Berry Curvature')
            plt.xlabel(r'$k_x$')
            plt.ylabel(r'$k_y$')
            plt.title(f'{title} (Band {band_index})')
            plt.gca().set_aspect('equal')
            
            plt.legend()
            # plt.savefig(f"berry_curvature_plot_{band_index}.png", dpi=600)
            
            plt.show()
            
        
            
        
        