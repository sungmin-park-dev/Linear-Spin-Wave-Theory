import numpy as np
from typing import Tuple, List, Dict, Any, Union, Optional


from modules.Tools.spin_system_optimizer import SpinSystemOptimizer

DEFAULT_SPIN= 1/2


class NBCP_UNIT_CELL:
    def __init__(self, config: Optional[Dict[str, Any]] = None, lattice_spacing: float = 1) -> None:
        """
        Initialize NBCP Unit Cell.

        Args:
            config: Configuration dictionary with keys 'S', 'Jxy', 'Jz', 'JGamma', 'JPD', 'h', 'aniso', 'nne'.
            lattice_spacing: Lattice constant for scaling displacements.
        """
        
        self.lattice_constant = lattice_spacing
        self.update_config(config)  # 사용자 config로 초기화


        # 격자 상수를 반영한 변위 설정
        dist_nn  = self.lattice_constant 
        self.Disp_xy_nn  = ((   dist_nn,              0.0               ),
                            ( - dist_nn / 2,   dist_nn * np.sqrt(3) / 2 ),
                            ( - dist_nn / 2, - dist_nn * np.sqrt(3) / 2 ))
        
        dist_nnn = self.lattice_constant * np.sqrt(3)
        self.Disp_xy_nnn = ((               0.0,             1 * dist_nnn  ),
                            ( - np.sqrt(3)/2 * dist_nnn, - 1/2 * dist_nnn  ),
                            ( + np.sqrt(3)/2 * dist_nnn, - 1/2 * dist_nnn  ))
   
    @staticmethod 
    def _validate_exchange_Jxy_params(config) -> tuple:
        """Validate exchange parameters and return Jx, Jy."""
        check_Jx_Jy = ("Jx" in config or "Jy" in config)
        check_Jxy = "Jxy" in config 
        
        if check_Jxy and not check_Jx_Jy:
            return config["Jxy"], config["Jxy"]
        elif check_Jx_Jy and not check_Jxy:
            return config["Jx"], config["Jy"]
        raise KeyError("Config is inappropriate for making TLAF.")

    @staticmethod 
    def _validate_exchange_Kxy_params(config) -> tuple:
        """Validate K exchange parameters and return Kx, Ky."""
        check_Kx_Ky = ("Kx" in config or "Ky" in config)
        check_Kxy = "Kxy" in config 
        
        if check_Kxy and not check_Kx_Ky:
            return config["Kxy"], config["Kxy"]
        elif check_Kx_Ky and not check_Kxy:
            return config["Kx"], config["Ky"]
        raise KeyError("Config is inappropriate for K parameters.")

    @staticmethod
    def compute_J_exch_mat(phi: float, x: float, y: float, z: float, pd: float, gamma: float, Dx: float = 0, Dy: float = 0, Dz: float = 0) -> np.ndarray:
        Dx = Dx * np.cos(phi) - Dy * np.sin(phi)  # 예시 변환
        Dy = Dx * np.sin(phi) + Dy * np.cos(phi)  # 예시 변환
        Dz = Dz  # z 성분은 회전 불변일 수 있음
        
        return np.array([[   x  + 2 * pd * np.cos(phi),   Dz  - 2 * pd * np.sin(phi), - Dy - gamma * np.sin(phi)],
                         [ - Dz - 2 * pd * np.sin(phi),     y - 2 * pd * np.cos(phi),   Dx + gamma * np.cos(phi)],
                         [   Dy - gamma * np.sin(phi),  - Dx  + gamma * np.cos(phi),            z         ]])

    # @staticmethod
    # def compute_J_exch_mat(phi: float, x: float, y: float, z: float, pd: float, gamma: float) -> np.ndarray:
    #     return np.array([[   x  + 2 * pd * np.cos(phi),  - 2 * pd * np.sin(phi), - gamma * np.sin(phi)],
    #                      [ - 2 * pd * np.sin(phi),     y - 2 * pd * np.cos(phi), + gamma * np.cos(phi)],
    #                      [ - gamma * np.sin(phi),       + gamma * np.cos(phi),            z         ]])

    @staticmethod
    def compute_K_exch_mat(phi: float, x: float, y: float, z: float, pd: float, gamma: float) -> np.ndarray:
        return np.array([[x - 2 * pd * np.cos(phi),   - 2 * pd * np.sin(phi),  gamma * np.sin(phi)],
                         [  - 2 * pd * np.sin(phi), y + 2 * pd * np.cos(phi),  gamma * np.cos(phi)],
                         [     gamma * np.sin(phi),      gamma * np.cos(phi),           z         ]])

    def update_config(self, config):
        self.config = config

        self.single_spin_anisotropy = "ssa" in config
        self.nearest_exchange = False
        self.next_nearest_exchange = False

        if self.single_spin_anisotropy:
            self.ssa_I_matrix = np.diag(config["ssa"])
        else:
            if hasattr(self, 'ssa_I_matrix'):
                del self.ssa_I_matrix

        # Nearest-neighbor exchange
        has_J_params = any(key in config for key in ("Jx", "Jy", "Jz", "Jxy", "JGamma", "JPD"))
        if has_J_params:
            Jx, Jy = self._validate_exchange_Jxy_params(config)
            Jz = config.get("Jz", 0)
            JPD = config.get("JPD", 0)
            JGamma = config.get("JGamma", 0)
            Dx = config.get("Dx", 0)
            Dy = config.get("Dy", 0)
            Dz = config.get("Dz", 0)
            print(f"Jx, Jy, Jz, JPD, JGamma: {Jx}, {Jy}, {Jz}, {JPD}, {JGamma}")

            if not all(val == 0 for val in (Jx, Jy, Jz, JPD, JGamma)):
                self.nearest_exchange = True
                self.Exch_J = tuple(
                    # self.compute_J_exch_mat(phi, Jx, Jy, Jz, JPD, JGamma)
                    self.compute_J_exch_mat(phi, Jx, Jy, Jz, JPD, JGamma, Dx, Dy, Dz)
                                    for phi in (0, 2 * np.pi / 3, 4 * np.pi / 3)
                                    )
                for j in range(len(self.Exch_J)):
                    print(f"Exch_J: \n{self.Exch_J[j]}") 

        
        # Next-nearest-neighbor exchange
        has_K_params = any(key in config for key in ("Kx", "Ky", "Kxy", "Kz", "KPD", "KGamma"))
        if has_K_params:
            Kx, Ky = self._validate_exchange_Kxy_params(config)
            Kz = config.get("Kz", 0)
            KPD = config.get("KPD", 0)
            KGamma = config.get("KGamma", 0)
            KGamma = config.get("KGamma", 0)

            if not all(val == 0 for val in (Kx, Ky, Kz, KPD, KGamma)):
                self.next_nearest_exchange = True
                self.Exch_K = tuple(
                    self.compute_K_exch_mat(phi, Kx, Ky, Kz, KPD, KGamma)  
                    for phi in (np.pi / 2, 7 * np.pi / 6, 11 * np.pi / 6)
                )
            else:
                self.next_nearest_exchange = False
                if hasattr(self, 'Exch_K'):
                    del self.Exch_K


    def _check_angles(self, angles: Union[Tuple, List], num_angles: int = None):
        if isinstance(angles, (list, tuple, np.ndarray)):
            two_Ns = len(angles)
            # Check if the total count is even
            if two_Ns % 2 != 0:
                raise ValueError(f"The number of angles is inappropriate: {two_Ns}\n"
                                "The number of angles must be even.\n"
                                "correct format: (theta_a, phi_a, theta_b, phi_b, ...)")
            
            # Create a new list to store the processed angles
            processed_angles = []
            for angle in angles:
                if angle is None:
                    # Replace None with a random angle between -π and π
                    processed_angles.append(np.pi * (2 * np.random.rand() - 1))
                else:
                    # Keep the existing angle value
                    processed_angles.append(angle)
            
            angles = np.array(processed_angles)
        
        elif (angles == None):
            if num_angles is None:
                raise ValueError("When angles is None, num_angles must be specified and be a positive integer.")
            elif isinstance(num_angles, int) and num_angles > 0:
                two_Ns = num_angles
            else:
                raise ValueError(f"The number of angles is inappropriate: {num_angles} \nThe number of angles must be a positive integer.")
            # Generate all random angles if None is provided
            angles = np.pi * (2 * np.random.rand(two_Ns) - 1)
        else:
            raise ValueError(f"The type of angles is inappropriate: {angles} \nThe type of angles must be a list or tuple \nor None but assign the number of angles.")
        
        return angles
    
    
    @staticmethod
    def _single_spin_anisotropy(spin_info: dict, coup_mat: np.ndarray):
        ssa_coups = []
        for msl_name in spin_info.keys():
            temp = {"SpinI": msl_name, 
                    "SpinJ": msl_name, 
                    "Exchange Matrix": coup_mat, 
                    "Displacement": (0, 0)
                    }
            ssa_coups.append(temp)
        return ssa_coups
    
    
    def _get_nn_couplings(self, spin_i, spin_j_list):
        nn_coups = []
        for j, spin_j in enumerate(spin_j_list):
            temp = {"SpinI": spin_i, 
                    "SpinJ": spin_j, 
                    "Exchange Matrix": self.Exch_J[j], 
                    "Displacement": self.Disp_xy_nn[j]}
            nn_coups.append(temp)
        return nn_coups
    
    def _get_nnn_couplings(self, spin_i, spin_j_list):
        nne_coups = []
        for j, spin_j in enumerate(spin_j_list):
            temp = {"SpinI": spin_i, 
                    "SpinJ": spin_j, 
                    "Exchange Matrix": self.Exch_K[j], 
                    "Displacement": self.Disp_xy_nnn[j]}
            print(f"{j}---Exchange Matrix\n", self.Exch_K[j])
            nne_coups.append(temp)
        return nne_coups



    def spin_system_data_one_msl(self, angles = None):
        
        theta_a, phi_a = self._check_angles(angles, num_angles=2)            
        
        spin_info = {"A": {"Position": (0, 0),
                          "Spin": DEFAULT_SPIN,
                          "Angles": (theta_a, phi_a),
                          "Magnetic Field": self.config["h"]}}
        
        lattice_vectors = ( np.array([self.lattice_constant/2, + self.lattice_constant*np.sqrt(3)/2]),
                            np.array([self.lattice_constant/2, - self.lattice_constant*np.sqrt(3)/2]) )
        
        couplings = []
        
        if self.single_spin_anisotropy:  # 현재: single_spin_anisotrpy
            ssa_coups = self._single_spin_anisotropy(spin_info, self.ssa_coup_matrix)
            couplings.extend(ssa_coups)
        
        if self.nearest_exchange:
            ne_coups_a  = self._get_nn_couplings( "A", ["A"]*3)
            couplings.extend(ne_coups_a)
        
        if self.next_nearest_exchange:
            nne_coups_a = self._get_nnn_couplings("A", ["A"]*3)    # [{}]
            couplings.extend(nne_coups_a)        

        spin_system_data = {"Spin info": spin_info,
                           "Couplings": couplings,
                           "Lattice/BZ setting": (lattice_vectors, "Hex_60")}
        
        return spin_system_data
        
        

    def spin_system_data_two_msl(self, angles = None):
        
        theta_a, phi_a, theta_b, phi_b = self._check_angles(angles, num_angles=4)
        
        spin_info = { "A": {"Position":   (0, 0),                   # Subblattice A information
                            "Spin":       DEFAULT_SPIN,                     
                            "Angles":     [ theta_a, phi_a ],
                            "Magnetic Field": self.config["h"]},
                    "B": {"Position":   (1/2, np.sqrt(3)/2),      # xy coordination
                            "Spin":       DEFAULT_SPIN,                       # S = 1/2, 1, 3/2, ...
                            "Angles":     [ theta_b, phi_b ],
                            "Magnetic Field": self.config["h"]}}
        
                
        lattice_vectors = (np.array([1.0,        0.0]), 
                           np.array([0.0, np.sqrt(3)]))
        
        couplings = []
        
        if self.single_spin_anisotropy:  # 현재: single_spin_anisotrpy
            ssa_coups = self._single_spin_anisotropy(spin_info, self.ssa_I_matrix)
            couplings.extend(ssa_coups)
        
        if self.nearest_exchange:
            ne_coups_a = self._get_nn_couplings("A", ["A", "B", "B"])
            ne_coups_b = self._get_nn_couplings("B", ["B", "A", "A"])    # [{}]
            couplings.extend(ne_coups_a + ne_coups_b)
            
        if self.next_nearest_exchange:
            nne_coups_a = self._get_nnn_couplings("A", ["A", "B", "B"])    # [{}]
            nne_coups_b = self._get_nnn_couplings("B", ["B", "A", "A"])    # [{}]
            couplings.extend(nne_coups_a + nne_coups_b)

        spin_system_data = {"Spin info": spin_info,
                            "Couplings": couplings,
                            "Lattice/BZ setting": (lattice_vectors, "Tetra")}
        
        return spin_system_data

        
    def spin_system_data_three_msl(self, angles = None):
        
        theta_a, phi_a, theta_b, phi_b, theta_c, phi_c = self._check_angles(angles, num_angles=6)
        
        spin_info = { "A": {"Position":   ( 1/2, np.sqrt(3)/2),
                        "Spin":       DEFAULT_SPIN,                     
                        "Angles":     [ theta_a, phi_a ], 
                        "Magnetic Field": self.config["h"]},
                    "B": {"Position":   (-1/2, np.sqrt(3)/2),   
                        "Spin":       DEFAULT_SPIN,                     
                        "Angles":     [ theta_b,  phi_b ],
                        "Magnetic Field": self.config["h"]},
                    "C": {"Position":   (0, 0), 
                        "Spin":       DEFAULT_SPIN,                     
                        "Angles":     [ theta_c,  phi_c ],
                        "Magnetic Field": self.config["h"]}}

        couplings = []

        if self.single_spin_anisotropy:  # 현재: single_spin_anisotrpy
            ssa_coups = self._single_spin_anisotropy(spin_info, self.ssa_I_matrix)
            couplings.extend(ssa_coups)
        
        if self.nearest_exchange:
            ne_coups_a = self._get_nn_couplings("A", ["B"]*3)
            ne_coups_b = self._get_nn_couplings("B", ["C"]*3)
            ne_coups_c = self._get_nn_couplings("C", ["A"]*3)
            couplings.extend(ne_coups_a + ne_coups_b + ne_coups_c)        
        
        if self.next_nearest_exchange:
            nne_coups_a = self._get_nnn_couplings("A", ["A"]*3)
            nne_coups_b = self._get_nnn_couplings("B", ["B"]*3)
            nne_coups_c = self._get_nnn_couplings("C", ["C"]*3)
            couplings.extend(nne_coups_a + nne_coups_b + nne_coups_c)
        
        lattice_vectors = (np.array([ 3/2, + np.sqrt(3)/2 ]),
                           np.array([ 3/2, - np.sqrt(3)/2 ]))

        spin_system_data = {"Spin info": spin_info,
                            "Couplings": couplings,
                            "Lattice/BZ setting": (lattice_vectors, "Hex_30")}

        return spin_system_data

    def spin_system_data_four_msl(self, angles = None):
        
        theta_a, phi_a, theta_b, phi_b, theta_c, phi_c, theta_d, phi_d = self._check_angles(angles, num_angles=8)
        
        spin_info = { "A": {"Position":   ( 1,           0   ),
                            "Spin":       DEFAULT_SPIN,                     
                            "Angles":     [ theta_a, phi_a ],
                            "Magnetic Field": self.config["h"]},
                      "B": {"Position":   ( 1/2, np.sqrt(3)/2),   
                            "Spin":       DEFAULT_SPIN,                     
                            "Angles":     [ theta_b, phi_b ],
                            "Magnetic Field": self.config["h"]},
                      "C": {"Position":   (-1/2, np.sqrt(3)/2),
                            "Spin":       DEFAULT_SPIN,                     
                            "Angles":     [ theta_c, phi_c ],
                            "Magnetic Field": self.config["h"]},
                      "D": {"Position":   (  0,          0   ),
                            "Spin":       DEFAULT_SPIN,                     
                            "Angles":     [ theta_d, phi_d ],
                            "Magnetic Field": self.config["h"]}}
        
        lattice_vectors = (np.array([self.lattice_constant, + self.lattice_constant*np.sqrt(3)]),
                           np.array([self.lattice_constant, - self.lattice_constant*np.sqrt(3)]))
        
        couplings = []
        
        if self.single_spin_anisotropy:  # 현재: single_spin_anisotrpy
            ssa_coups = self._single_spin_anisotropy(spin_info, self.ssa_I_matrix)
            couplings.extend(ssa_coups)
            
        if self.nearest_exchange:
            ne_coups_a = self._get_nn_couplings("A", ["D", "B", "C"])
            ne_coups_b = self._get_nn_couplings("B", ["C", "A", "D"])
            ne_coups_c = self._get_nn_couplings("C", ["B", "D", "A"])
            ne_coups_d = self._get_nn_couplings("D", ["A", "C", "B"])
            couplings.extend(ne_coups_a + ne_coups_b + ne_coups_c + ne_coups_d)

        if self.next_nearest_exchange:
            nne_coups_a = self._get_nnn_couplings("A", ["D", "B", "C"])
            nne_coups_b = self._get_nnn_couplings("B", ["C", "A", "D"])
            nne_coups_c = self._get_nnn_couplings("C", ["B", "D", "A"])
            nne_coups_d = self._get_nnn_couplings("D", ["A", "C", "B"])
            couplings.extend(nne_coups_a + nne_coups_b + nne_coups_c + nne_coups_d)

        spin_system_data = {"Spin info": spin_info,
                           "Couplings": couplings,
                           "Lattice/BZ setting": (lattice_vectors, "Hex_60")}
        
        return spin_system_data
    
if __name__ == "__main__":
    from modules.Tools.brillouin_zone import Brillouin_Zone
    from matplotlib import pyplot as plt
    
    bz = Brillouin_Zone()
    
    # Example usage
    config = {"S": 1/2, 
              "Jxy": 0.067, 
              "Jz": 0.125, 
              "JGamma": 0.00, 
              "JPD": 0.00, 
              "h": (0.0, 0.0, 0.35)}
    nbcp = NBCP_UNIT_CELL(config = config)
    
    one_msl_data = nbcp.spin_system_data_one_msl()
    two_msl_data = nbcp.spin_system_data_two_msl()
    three_msl_data = nbcp.spin_system_data_three_msl()
    four_msl_data = nbcp.spin_system_data_four_msl()
    
    
    N = 10
    
    fig = plt.figure(figsize=(12, 10))
    
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    axes = [ax1, ax2, ax3, ax4]
    titles = ["One MSL", "Two MSL", "Three MSL", "Four MSL"]
    msl_datas = [one_msl_data, two_msl_data, three_msl_data, four_msl_data] 
    
    type_setting = "simple"  # or "complex"
    
    for j, (ax, title, msl_data) in enumerate(zip(axes, titles, msl_datas)):
        if type_setting == "simple":
            bz_type = "simple"
        else:
            bz_type = msl_data["BZ setting"][1]
        
        bz_data, grid_points, _ = bz.get_full(bz_type, 
                                              msl_data["Lattice/BZ setting"][0], N)
        _, ax = bz.plot_polygon_and_grid(bz_data["BZ_corners"],
                                        HSP = bz_data["high_symmetry_points"], 
                                        band_path = bz_data["band_paths"], 
                                        grid_points=grid_points, 
                                        title=title, ax = ax)
    plt.show()