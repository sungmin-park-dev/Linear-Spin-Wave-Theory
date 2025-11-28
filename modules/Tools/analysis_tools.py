import numpy as np

from .brillouin_zone import Brillouin_Zone      # same folder as this file
from modules.LinearSpinWaveTheory.lswt_Hamiltonian import LSWT_HAMILTONIAN


def calculate_skyrmion_number(spin_sys_data):
    def spherical_to_cartesian(theta, phi, r=1.0):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def spherical_triangle_area(v1, v2, v3):
        scalar_triple = np.abs(np.dot(v1, np.cross(v2, v3)))
        denom = (1 + np.dot(v1, v2) + np.dot(v2, v3) + np.dot(v3, v1))
        return 2 * np.arctan2(scalar_triple, denom)

    spin_info = spin_sys_data["Spin info"]
    
    if len(spin_info) >= 4:
        spins = []
        for i, value in enumerate(spin_info.values()):
            theta, phi = value["Angles"]
            x, y, z = spherical_to_cartesian(theta, phi)
            spins.append(np.array([x, y, z]))
        
        faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
        total_solid_angle = 0
        for face in faces:
            v1, v2, v3 = [spins[i] for i in face]
            total_solid_angle += spherical_triangle_area(v1, v2, v3)
        
        return total_solid_angle / (4 * np.pi)
    else:
        return 0









class Create_Energy_Function:
    def __init__(self, spin_sys_data, N, update_args = False):
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

        B_zone = Brillouin_Zone(self.spin_sys_data["Lattice/BZ setting"], bz_type="simple")
        _, self.k_points, _ = B_zone.get_full(N=N)
        self.num_k_points = len(self.k_points)
        self.lswt_Ham = LSWT_HAMILTONIAN(self.spin_sys_data["Spin info"], self.spin_sys_data["Couplings"])
        
        
        self.energy_func = self.lswt_energy_density_func


    @staticmethod
    def _cartesian_spin_odering(S, theta, phi):
        return np.array([S * np.cos(phi) * np.sin(theta), 
                         S * np.sin(phi) * np.sin(theta), 
                         S * np.cos(theta)])

    def classical_energy_density_func(self, angles):
        if len(angles) != 2 * self.num_SL:
            raise ValueError(f"Expected {2 * self.num_SL} angles, got {len(angles)}")
        
        if self.update_args:
            self.angle_args = angles
            
        spins_orderings = {}
        for i, (spin_name, spin_dict) in enumerate(self.spin_sys_data["Spin info"].items()):
            S = spin_dict["Spin"]
            theta, phi = angles[2*i:2*i + 2]
            spins_orderings[spin_name] = self._cartesian_spin_odering(S, theta, phi)

        classical_energy = 0.0
        for coupling_dict in self.spin_sys_data["Couplings"]:
            spin_i = spins_orderings[coupling_dict["SpinI"]]
            spin_j = spins_orderings[coupling_dict["SpinJ"]]
            classical_energy += spin_i @ coupling_dict["Exchange Matrix"] @ spin_j

        for spin_name, info in self.spin_sys_data["Spin info"].items():
            classical_energy -= np.dot(info["Magnetic Field"], spins_orderings[spin_name])
            
        classical_energy_density = classical_energy / self.num_SL
        
        if self.update_args:
            self.classical_energy_density = classical_energy_density

        return classical_energy_density


    def quantum_energy_density_func(self, angles, reg_type = 0):
        if len(angles) != 2 * self.num_SL:
            raise ValueError(f"Expected {2 * self.num_SL} angles, got {len(angles)}")

        if self.update_args:
            self.angle_args = angles
        
        quantum_energy, self.mu_magswt = self.lswt_Ham.compute_quantum_energy(self.k_points, 
                                                                              angles = angles,  
                                                                              T = 0,
                                                                              reg_type = reg_type)
        quantum_energy_density = quantum_energy / (self.num_k_points * self.num_SL)
        
        if self.update_args:
            self.quantum_energy_density = quantum_energy_density
        
        return quantum_energy_density


    def lswt_energy_density_func(self, angles):
        self._info_cache = None
        if len(angles) != 2 * self.num_SL:
            raise ValueError(f"Expected {2 * self.num_SL} angles, got {len(angles)}")
        
        E_density_cl = self.classical_energy_density_func(angles)
        E_density_qm = self.quantum_energy_density_func(angles)

        return E_density_cl + E_density_qm


    def quantum_free_energy_density_func(self, angles, reg_type = 0, Temperature = 0):
        if len(angles) != 2 * self.num_SL:
            raise ValueError(f"Expected {2 * self.num_SL} angles, got {len(angles)}")

        if self.update_args:
            self.angle_args = angles
        
        quantum_energy, self.mu_magswt = self.lswt_Ham.compute_quantum_free_energy(self.k_points, 
                                                                                   angles = angles,  
                                                                                   T = Temperature,
                                                                                   reg_type = reg_type)
        
        quantum_free_energy_density = quantum_energy / (self.num_k_points * self.num_SL)
        
        if self.update_args:
            self.quantum_energy_density = quantum_free_energy_density
        
        return quantum_free_energy_density


    def get_info(self, print_info=False):
        if self._info_cache is None:
            data = {"angles": self.angle_args,
                    "E_cl": self.classical_energy_density,
                    "E_qm": self.quantum_energy_density,
                    "MAGSWT": self.mu_magswt}
            self._info_cache = data
            if print_info:
                for key, content in data.items():
                    print(f"{key}: \n{content} ")
            return data
        return self._info_cache

    def set_update_args(self, update_args: bool):
        self.update_args = update_args

    @staticmethod
    def _d_Spin_diff_theta(S, theta, phi):
        return np.array([+ S * np.cos(phi) * np.cos(theta), 
                        + S * np.sin(phi) * np.cos(theta), 
                        - S * np.sin(theta)])

    @staticmethod
    def _d_Spin_diff_phi(S, theta, phi):
        return np.array([- S * np.sin(phi) * np.sin(theta), 
                        + S * np.cos(phi) * np.sin(theta), 
                        0])

    @classmethod
    def Diff_classical_energy_density_func(cls, spin_sys_data):
        angles = []
        for spin_name, value in spin_sys_data["Spin info"].items():
            theta, phi = value["Angles"]
            angles.extend([theta, phi])
        
        num_SL = len(spin_sys_data["Spin info"])
        spins_orderings = {}
        spins_d_thetas = {}
        spins_d_phis = {}
        
        for i, (spin_name, spin_dict) in enumerate(spin_sys_data["Spin info"].items()):
            S = spin_dict["Spin"]
            theta, phi = angles[2*i:2*i + 2]
            spins_orderings[spin_name] = cls._cartesian_spin_odering(S, theta, phi)
            spins_d_thetas[spin_name] = cls._d_Spin_diff_theta(S, theta, phi)
            spins_d_phis[spin_name] = cls._d_Spin_diff_phi(S, theta, phi)

        diff_classical_energy = {spin_name: {"d_E_d_theta": 0, "d_E_d_phi": 0} 
                                for spin_name in spin_sys_data["Spin info"]}
        
        for coupling_dict in spin_sys_data["Couplings"]:
            name_i = coupling_dict["SpinI"]
            name_j = coupling_dict["SpinJ"]
            exchange_matrix = coupling_dict["Exchange Matrix"]
            
            spin_i_d_theta = spins_d_thetas[name_i]
            diff_classical_energy[name_i]["d_E_d_theta"] += spin_i_d_theta @ exchange_matrix @ spins_orderings[name_j]
            
            spin_j_d_theta = spins_d_thetas[name_j]
            diff_classical_energy[name_j]["d_E_d_theta"] += spins_orderings[name_i] @ exchange_matrix @ spin_j_d_theta
            
            spin_i_d_phi = spins_d_phis[name_i]
            diff_classical_energy[name_i]["d_E_d_phi"] += spin_i_d_phi @ exchange_matrix @ spins_orderings[name_j]
            
            spin_j_d_phi = spins_d_phis[name_j]
            diff_classical_energy[name_j]["d_E_d_phi"] += spins_orderings[name_i] @ exchange_matrix @ spin_j_d_phi
        
        for spin_name, info in spin_sys_data["Spin info"].items():
            magnetic_field = info["Magnetic Field"]
            diff_classical_energy[spin_name]["d_E_d_theta"] -= np.dot(magnetic_field, spins_d_thetas[spin_name])
            diff_classical_energy[spin_name]["d_E_d_phi"] -= np.dot(magnetic_field, spins_d_phis[spin_name])
        
        for spin_name in diff_classical_energy:
            diff_classical_energy[spin_name]["d_E_d_theta"] /= num_SL
            diff_classical_energy[spin_name]["d_E_d_phi"] /= num_SL

        return diff_classical_energy