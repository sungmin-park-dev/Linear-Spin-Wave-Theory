import numpy as np
from typing import Tuple, List, Dict

from modules.Tools.brillouin_zone import Brillouin_Zone
from modules.Tools.magnon_kernel import compute_static_magnon_kernel

from modules.LinearSpinWaveTheory.lswt_Hamiltonian import LSWT_HAMILTONIAN
from modules.LinearSpinWaveTheory.lswt_thermodynamics import LSWT_THER
from modules.LinearSpinWaveTheory.lswt_correlation import LSWT_CORR
from modules.LinearSpinWaveTheory.lswt_topology import LSWT_TOPO


class LSWT:
    def __init__(self, spin_system_data):
        """Initialize the Linear Spin Wave Theory calculator.

        Args:
            spin_system_data (Dict): 'Spin info', 'Couplings', 'BZ setting', 'Lattice vectors'
        """
        # Create an instance of PhysicalProperties
        
        # Essential Datas
        self.spin_system_data = spin_system_data
        self.spin_info = self.spin_system_data["Spin info"]
        self.couplings = self.spin_system_data["Couplings"]
        self.lattice_bz_settings = self.spin_system_data["Lattice/BZ setting"]
        self.Ns = len(self.spin_info)
        self.T = 0 
        self.kB = 8.617333262e-5 # eV/T
        
        self.Ham = LSWT_HAMILTONIAN(self.spin_info, self.couplings)

    # Methods for physical properties now delegate to the PhysicalProperties class
    
    def diagnosing_lswt(self, bz_type = "Hex_60", regularization = "MAGSWT", N = 10, temperature = 0):
        """_Intializing diagonsis tool box_
        """
        
        bz = Brillouin_Zone(self.lattice_bz_settings, bz_type = bz_type)
        
        bz_data, full_k_points, _ = bz.get_full(N)
        
        self.bz_type = bz_type
        self.bz_data = bz_data
        
        print("="*30, "\nInitializing diagnosis tool box\n", "="*30)
        k_data, chem_pot_magswt = self.Ham.solve_k_Hamiltonian(full_k_points, Berry_curvature = True, regularization =regularization)

        self.msl_average_boson_number, self.average_boson_number = self.lswt_correction(k_data = k_data)
        
        # self.k_data = dict(k_data)            # copy dictionary
        self.regularization = regularization
        self.magswt_onsite = chem_pot_magswt 

        self.ther = LSWT_THER(self)
        self.corr = LSWT_CORR(self)
        self.topo = LSWT_TOPO(self)       
        
        return k_data, bz_data, full_k_points 
    

    def lswt_correction(self, k_data):
        """
        compute zero temperature magnon occupation number correction
        """
        
        average_boson_numbers = 0
        sublattice_boson_numbers = np.zeros(self.Ns)
        
        print("="*15, " Boson number ", "="*15)        
        for k_key in k_data.keys():
            _, Eigen_data, *_ = k_data[k_key]
            eval, evec = Eigen_data
            
            magnon_kernel = compute_static_magnon_kernel(eval, Temperature = 0, Ns = self.Ns)
            two_point = evec @ np.diag(magnon_kernel) @ evec.T.conj()
            diag_elements = np.diag(two_point)[self.Ns:]
            
            sublattice_boson_numbers += np.real(diag_elements)
            
        sublattice_boson_numbers = sublattice_boson_numbers/len(k_data)
        
        msl_average_boson_number = {}
        lswt_correction_msl_spin_moment = {}
        
        average_boson_number = 0
        total_spin_moment = np.zeros(3, dtype= float)

        for j, (name_sl, value) in enumerate(self.spin_info.items()):
            
            spin = value["Spin"]            
            msl_boson_num = sublattice_boson_numbers[j]
            
            average_boson_number += msl_boson_num
            msl_average_boson_number[name_sl] = msl_boson_num 
            new_spin = spin - msl_boson_num
            
            # check
            devt = 100*msl_boson_num/(2*spin)
            print(f"{name_sl}-magnetic sublattice")
            print(f"\t spin moment:  {new_spin:.5f}",
                  f"\n\t boson number: {msl_boson_num:.5f}"
                  f"\t (n/2S)-deviation = {devt:.2f}%")
            
            theta, phi = value["Angles"]
            spin_moment = new_spin * np.array([np.sin(theta) * np.cos(phi), 
                                               np.sin(theta) * np.sin(phi), 
                                               np.cos(theta)]) 
            
            total_spin_moment += spin_moment
            print(f"\t spin moment in real space:", 
                  f"\n\t Sx = {spin_moment[0]:.4f}, Sy = {spin_moment[1]:.4f}, Sz = {spin_moment[2]:.4f}")
            
            lswt_correction_msl_spin_moment[name_sl] = spin_moment
            
        average_boson_number = average_boson_number/len(self.spin_info)        

        print(f"Total boson number: {average_boson_number:.5f}")
        print(f"Total Sx: {total_spin_moment[0]:.5f} ")
        print(f"Total Sy: {total_spin_moment[1]:.5f} ")
        print(f"Total Sz: {total_spin_moment[2]:.5f} ")
        
        return msl_average_boson_number, average_boson_number
