import numpy as np
from typing import Tuple, List, Dict

from modules.Tools.magnon_kernel import compute_bose_einstein_distribution
from modules.Tools.diagonalization import DIAG
from modules.constants import K_BOLTZMANN_meV



# Criteria for beta E regimes
HIGH_BE_THRESHOLD = 35.0           # βE > 35: (지수 근사)
LOW_BE_THRESHOLD = 0.1             # βE < 0.1: (Taylor 급수)
ZERO_ENERGY_THRESHOLD = 1e-15      # E < 1e-15: zero energy

"""
보존 입자 (Boson) 통계역학 함수들

βE 구간별 계산 방법:
- Zero energy: E < ZERO_ENERGY_THRESHOLD → -∞ (보스-아인슈타인 응축)  
- Low βE: βE < LOW_BE_THRESHOLD → Taylor 급수 (높은 정확도)
- High βE: βE > HIGH_BE_THRESHOLD → 지수 근사 (수치 안정성)
- Medium βE: LOW_BE_THRESHOLD ≤ βE ≤ HIGH_BE_THRESHOLD → log1p (균형)

사용법:
- bosonic_term_unified(energy, temperature): 단일 항 계산
- bosonic_free_energy(energies, temperature): 전체 자유에너지 계산
"""

def log_1_m_exp(energy: np.ndarray, temperature: float) -> np.ndarray:
    """ln(1 - e^(-βE))/β 계산"""
    energy = np.asarray(energy)
    
    
    beta = 1.0 / (K_BOLTZMANN_meV * temperature)
    beta_E = beta * energy
    
    result = np.zeros_like(beta_E, dtype = float)
    
    # Zero energy: E ≈ 0
    zero_mask = np.abs(energy) < ZERO_ENERGY_THRESHOLD
    result[zero_mask] = -np.inf
    
    valid_mask = ~ zero_mask
    if not np.any(valid_mask):
        return result
    
    beta_E_valid = beta_E[valid_mask]
    energy_valid = energy[valid_mask] if energy.ndim > 0 else energy
    
    # Low βE: βE < LOW_BE_THRESHOLD
    low_be_mask = beta_E_valid < LOW_BE_THRESHOLD
    if np.any(low_be_mask):
        be_low = beta_E_valid[low_be_mask]
        e_low = energy_valid[low_be_mask] if energy_valid.ndim > 0 else energy_valid
        
        result[valid_mask][low_be_mask] = e_low * (
            np.log(be_low)/be_low - 0.5 - be_low/24.0 - be_low**3/2880.0
        )
    
    # High βE: βE > HIGH_BE_THRESHOLD
    high_be_mask = beta_E_valid > HIGH_BE_THRESHOLD
    if np.any(high_be_mask):
        be_high = beta_E_valid[high_be_mask]
        e_high = energy_valid[high_be_mask] if energy_valid.ndim > 0 else energy_valid
        result[valid_mask][high_be_mask] = e_high * (-np.exp(-be_high) / be_high)
    
    # Medium βE: LOW_BE_THRESHOLD ≤ βE ≤ HIGH_BE_THRESHOLD
    med_be_mask = ~(low_be_mask | high_be_mask)
    if np.any(med_be_mask):
        be_med = beta_E_valid[med_be_mask]
        e_med = energy_valid[med_be_mask] if energy_valid.ndim > 0 else energy_valid
        result[valid_mask][med_be_mask] = e_med * ( np.log1p( - np.exp(-be_med) ) / be_med )
    
    return result


def bosonic_free_energy(energies, temperature):
    """F = Σ ln(1 - e^(-βE))/β 계산"""
    energies = np.asarray(energies)
    
    terms = log_1_m_exp(energies, temperature)
    finite_mask = np.isfinite(terms)
    
    return np.sum(terms[finite_mask]) if np.any(finite_mask) else -np.inf




"""
LSWT_HAMILTONIAN class for Linear Spin Wave Theory calculations.
"""

class LSWT_HAMILTONIAN:
    def __init__(self, spin_info, couplings):
        """Initialize the Linear Spin Wave Theory calculator.

        Args:
            spin_system_data (Dict): 'Spin info', 'Couplings', 'BZ setting', 'Lattice vectors'
        """
        self.spin_info = spin_info
        self.couplings = couplings
        self.Ns = len(self.spin_info)
        return
    
    @staticmethod
    def _classical_spin_rotation_matrix(pol_ang: float, azm_ang: float) -> np.ndarray:
        """Calculate rotation matrix for classical spin direction."""
        Rot_spin = np.array([[  np.cos(pol_ang)*np.cos(azm_ang), - np.sin(azm_ang), np.sin(pol_ang)*np.cos(azm_ang)],
                             [  np.cos(pol_ang)*np.sin(azm_ang),   np.cos(azm_ang), np.sin(pol_ang)*np.sin(azm_ang)],
                             [ -np.sin(pol_ang),                          0       , np.cos(pol_ang)                ]])
        return Rot_spin
    
    
    def get_rmat_dict(self, angles = None):
        """
        스핀 각도 업데이트 및 관련 행렬들을 재계산
        
        Parameters
        ----------
        angles : array, optional
            모든 스핀의 각도를 포함하는 1차원 배열 (theta1, phi1, theta2, phi2, ...)
            None인 경우 현재 spin_info에 있는 각도 사용
        """
        
        rmat_dict = {}
        
        if angles is None:
            for name_sl, sl_dict in self.spin_info.items():
                theta, phi = sl_dict["Angles"]    
                spin_rot_mat    = self._classical_spin_rotation_matrix(theta, phi)
                rmat_dict[name_sl] = spin_rot_mat 
        else:
            if len(angles) == 2 * len(self.spin_info) and isinstance(angles, (list, np.ndarray, tuple)):
                # print(f"Updating angles: \n{angles}")
                for j, name_sl in enumerate(self.spin_info.keys()):
                    theta = angles[2*j]
                    phi   = angles[2*j + 1]
                    # Update the rotation matrix
                    spin_rot_mat = self._classical_spin_rotation_matrix(theta, phi)
                    rmat_dict[name_sl] = spin_rot_mat
            else:
                raise ValueError(f"Number of angle variables must be equal to {2 * self.Ns}")
            
        return rmat_dict
    

    @staticmethod
    def get_couplings(coupling_dict, rmat_dict) -> List[complex]:
        """Calculate rotated exchange couplings."""
        # Calculate rotated exchange couplings
        Ri = rmat_dict[coupling_dict["SpinI"]]
        Rj = rmat_dict[coupling_dict["SpinJ"]]
        RJ = Ri.T @ coupling_dict["Exchange Matrix"] @ Rj
        
        sqrt2 = np.sqrt(2)
        Cmat = np.array([[   1 /sqrt2,     1 /sqrt2,      0],
                         [   1j/sqrt2,    -1j/sqrt2,      0],
                         [        0,          0,          1]])
        
        Hop_J = Cmat.T.conj() @ RJ @ Cmat

        return Hop_J

    def Quadratic_Bose_Hamiltonian(self, kpoints: np.ndarray,
                                   angles = None) -> np.ndarray:
        """Construct quadratic bose Hamiltonian."""
        
        # Initialize matrices
        m = len(kpoints)
        
        self.K_Hamiltonian = np.zeros((m, 2*self.Ns, 2*self.Ns), dtype=complex)

        self.SL_idx = {}
        self.linear_term_list = {}

        self.Rmat_dict = self.get_rmat_dict(angles = angles)

        # Magnetic Field
        for i, (sl_name, sl_dict) in enumerate(self.spin_info.items()):
                        
            self.SL_idx[sl_name] = i

            # Add magnetic field terms
            Rhix, Rhiy, Rhiz = sl_dict["Magnetic Field"] @ self.Rmat_dict[sl_name]
            hfp = Rhix + 1j*Rhiy

            self.K_Hamiltonian[:, i,      i     ] += Rhiz
            self.K_Hamiltonian[:, self.Ns + i, self.Ns + i] += Rhiz
            self.linear_term_list[sl_name] = - hfp

        
        # Coupling terms
        for coupling_dict in self.couplings:
            
            sli = coupling_dict["SpinI"]
            Spi = self.spin_info[sli]["Spin"]
            i = self.SL_idx[sli] 
            
            slj = coupling_dict["SpinJ"]
            Spj = self.spin_info[slj]["Spin"]
            j = self.SL_idx[slj] 
            
            delta = coupling_dict["Displacement"]
            
            # print(f"Coupling between {sli} (Si={Spi}) and {slj} (Sj={Spj}) ")
            
            exp_mDk = np.exp( - 1j * np.dot(kpoints, delta) )
            exp_pDk = exp_mDk.conj()
            
            # Get coupling constants
            hop_t =  self.get_couplings(coupling_dict, self.Rmat_dict)
            tpm = np.sqrt(Spi*Spj) * hop_t[1, 1]
            tpp = np.sqrt(Spi*Spj) * hop_t[1, 0]
            t00 = hop_t[2, 2]
            t0p = hop_t[2, 0]
            tp0 = hop_t[0, 2]

            # Add matrix elements
            # Onsite terms
            self.K_Hamiltonian[:,      i,      i] -= t00 * Spj
            self.K_Hamiltonian[:,      j,      j] -= t00 * Spi
            self.K_Hamiltonian[:, self.Ns + i, self.Ns + i] -= t00 * Spj
            self.K_Hamiltonian[:, self.Ns + j, self.Ns + j] -= t00 * Spi

            # Block A_{k} terms
            self.K_Hamiltonian[:,      i,      j] +=   tpm*exp_mDk
            self.K_Hamiltonian[:,      j,      i] +=  (tpm*exp_mDk).conj()
                
            # Block A^{*}_{-k} terms
            self.K_Hamiltonian[:, self.Ns + i, self.Ns + j] +=  (tpm*exp_pDk).conj()
            self.K_Hamiltonian[:, self.Ns + j, self.Ns + i] +=   tpm*exp_pDk

            # Block B terms
            self.K_Hamiltonian[:, self.Ns + i,      j] +=  tpp*exp_mDk
            self.K_Hamiltonian[:,      j, self.Ns + i] += (tpp*exp_mDk).conj()
                
            self.K_Hamiltonian[:, self.Ns + j,      i] +=  tpp*exp_pDk
            self.K_Hamiltonian[:,      i, self.Ns + j] += (tpp*exp_pDk).conj()
            
            # Add linear terms
            self.linear_term_list[sli] += Spi*tp0
            self.linear_term_list[slj] += Spj*t0p

        return self.K_Hamiltonian, self.linear_term_list


    
    def partial_derivatives_of_Hk(self, kpoints: np.ndarray) -> np.ndarray:
        """Construct quadratic bose Hamiltonian."""
        
        # Initialize matrices
        m = len(kpoints)
        self.DxHk = np.zeros((m, 2*self.Ns, 2*self.Ns), dtype=complex)
        self.DyHk = np.zeros((m, 2*self.Ns, 2*self.Ns), dtype=complex)
        
        # Coupling terms
        for coupling_dict in self.couplings:
            
            sli = coupling_dict["SpinI"]
            Spi = self.spin_info[sli]["Spin"]
            i = self.SL_idx[sli] 
            
            slj = coupling_dict["SpinJ"]
            Spj = self.spin_info[slj]["Spin"]
            j = self.SL_idx[slj] 
            
            delta = coupling_dict["Displacement"]
            
            # print(f"Coupling between {sli} (Si={Spi}) and {slj} (Sj={Spj}) ")
            
            exp_mDk = np.exp( - 1j * np.dot(kpoints, delta) )
            exp_pDk = exp_mDk.conj()
            
            # Get coupling constants
            hop_t =  self.get_couplings(coupling_dict, self.Rmat_dict)
            tpm = np.sqrt(Spi*Spj) * hop_t[1, 1]
            tpp = np.sqrt(Spi*Spj) * hop_t[1, 0]
            
            delta   = coupling_dict["Displacement"]
            
            exp_mDk = np.exp( - 1j * np.dot(kpoints, delta) )
            exp_pDk = exp_mDk.conj()
            pdmx = -1j * delta[0]
            pdpx = +1j * delta[0]
            pdmy = -1j * delta[1]
            pdpy = +1j * delta[1]


            # Add matrix elements
            # Block A_{k} terms
            self.DxHk[:,      i,      j] +=   tpm*exp_mDk*pdmx
            self.DxHk[:,      j,      i] +=  (tpm*exp_mDk*pdmx).conj()
                
            # Block A^{*}_{-k} terms
            self.DxHk[:, self.Ns + i, self.Ns + j] +=  (tpm*exp_pDk*pdpx).conj()
            self.DxHk[:, self.Ns + j, self.Ns + i] +=   tpm*exp_pDk*pdpx

            # Block B terms
            self.DxHk[:, self.Ns + i,      j] +=  tpp*exp_mDk*pdmx
            self.DxHk[:,      j, self.Ns + i] += (tpp*exp_mDk*pdmx).conj()
                
            self.DxHk[:, self.Ns + j,      i] +=  tpp*exp_pDk*pdpx
            self.DxHk[:,      i, self.Ns + j] += (tpp*exp_pDk*pdpx).conj()

            # Block A_{k} terms
            self.DyHk[:,      i,      j] +=   tpm*exp_mDk*pdmy
            self.DyHk[:,      j,      i] +=  (tpm*exp_mDk*pdmy).conj()
                
            # Block A^{*}_{-k} terms
            self.DyHk[:, self.Ns + i, self.Ns + j] +=  (tpm*exp_pDk*pdpy).conj()
            self.DyHk[:, self.Ns + j, self.Ns + i] +=   tpm*exp_pDk*pdpy

            # Block B terms
            self.DyHk[:, self.Ns + i,      j] +=  tpp*exp_mDk*pdmy
            self.DyHk[:,      j, self.Ns + i] += (tpp*exp_mDk*pdmy).conj()

            self.DyHk[:, self.Ns + j,      i] +=  tpp*exp_pDk*pdpy
            self.DyHk[:,      i, self.Ns + j] += (tpp*exp_pDk*pdpy).conj()

        return self.DxHk, self.DyHk
    

    def solve_k_Hamiltonian(self, k_points, Berry_curvature = True, regularization = "MAGSWT", threshold = 1e-8):
        """Diagonalize the quadratic boson Hamiltonian.
        Args:
            k_points: k-points of interest
        Returns:
            k_data: Dict; data with keys for each k-point
        """
        K_Ham_num, linear_term = self.Quadratic_Bose_Hamiltonian(k_points)
        
        if Berry_curvature:
            Partial_Diff_H = self.partial_derivatives_of_Hk(k_points)
        else:
            Partial_Diff_H = None
        
        k_data, chem_pot_mag = DIAG.get_K_data(k_points, 
                                               K_Ham_num, 
                                               regularization = regularization, 
                                               partial_derivative_Hk = Partial_Diff_H, 
                                               threshold = threshold)
        
        if (regularization == True) or (regularization == "MAGSWT") or (regularization == "magswt"):
            print(f"MAGSWT regularization is performed: \n{chem_pot_mag}")
        
        return k_data, chem_pot_mag
    
    def compute_quantum_energy(self, k_points, angles = None,
                               T = 0,
                               reg_type = "MAGSWT",
                               compute_free_energy = False) -> Tuple[float, float]:
        """
        Method for computing magnon energy
        E_quantum = 1/2 \sum_{k} (\sum_{mu} E_{mu} (k) - 1/2 * Tr(Hk))
        """
        K_Ham_num, _ = self.Quadratic_Bose_Hamiltonian(k_points, angles = angles)
        K_Ham_num, Bose_E, mu_magswt = DIAG.get_eigenvalue(K_Ham_num, reg_type = reg_type)
        
        E_zero = 0
        Free_E = 0
        if T == 0:
            for Hk, Ek in zip(K_Ham_num, Bose_E):
                trace_hk = np.real(np.trace(Hk))
                sum_Ek = np.sum(Ek[:self.Ns])
                E_zero += sum_Ek/2 - trace_hk/4
                
        elif T > 0:
            for Hk, Ek in zip(K_Ham_num, Bose_E):
                trace_hk = np.real(np.trace(Hk))
                Epk = Ek[:self.Ns]
                sum_Ek = np.sum(Epk)
                E_zero += sum_Ek/2 - trace_hk/4
                
                BE_dist = compute_bose_einstein_distribution(Epk, Temperature = T)
                E_zero += Epk * BE_dist
        
        return  E_zero, mu_magswt
        
        
    def compute_quantum_free_energy(self, k_points, 
                                    angles = None,
                                    T = 0,
                                    reg_type = "MAGSWT") -> Tuple[float, float]:
        """
        Method for computing magnon energy
        E_quantum = 1/2 \sum_{k} (\sum_{mu} E_{mu} (k) - 1/2 * Tr(Hk))
        """
        K_Ham_num, _ = self.Quadratic_Bose_Hamiltonian(k_points, angles = angles)
        K_Ham_num, Bose_E, mu_magswt = DIAG.get_eigenvalue(K_Ham_num, reg_type = reg_type)
        
        E_zero = 0
        if T == 0:
            for Hk, Ek in zip(K_Ham_num, Bose_E):
                trace_hk = np.real(np.trace(Hk))
                sum_Ek = np.sum(Ek[:self.Ns])
                E_zero += sum_Ek/2 - trace_hk/4
                
        elif T > 0:
            for Hk, Ek in zip(K_Ham_num, Bose_E):
                trace_hk = np.real(np.trace(Hk))
                Epk = Ek[:self.Ns]
                sum_Ek = np.sum(Epk)
                E_zero += sum_Ek/2 - trace_hk/4
                free_E = bosonic_free_energy(Epk, T)
                E_zero += free_E
                
        
        return  E_zero, mu_magswt

