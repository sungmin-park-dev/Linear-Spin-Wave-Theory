import time
import numpy as np
from typing import Tuple, Optional, Dict, List, Any, Union
from matplotlib import pyplot as plt


from modules.result_repository import ResultRepository
from modules.SpinSystem.nbcp_unitcells import NBCP_UNIT_CELL
from modules.Tools.spin_system_optimizer import SpinSystemOptimizer
from modules.Tools.analysis_tools import Create_Energy_Function, calculate_skyrmion_number





class FIND_TLAF_PHASES:
    def __init__(self, config, 
                 lattice_spacing = 1, 
                 repository_obj = None,
                 ):
        
        self.lattice_constant = lattice_spacing
        self.config = config
        self.result_repository = ResultRepository() if repository_obj is None else repository_obj
        
        self.nbcp_unit_cells = NBCP_UNIT_CELL(self.config, lattice_spacing=lattice_spacing)

    @staticmethod
    def _print_spin_info(spin_system_data, verbose = False):
        if not verbose:
            return
        
        print("\n" + "="*50)
        print("Spin System Configuration")
        print("-"*50)
        print("\nSpin Information:")
        for spin_name, info in spin_system_data["Spin info"].items():
            print(f"\nSpin {spin_name}:")
            print(f"  Position: ({info['Position'][0]:.3f}, {info['Position'][1]:.3f})")
            print(f"  Magnitude: S = {info['Spin']}")
            print(f"  Angles: θ = {info['Angles'][0]:.6f}, φ = {info['Angles'][1]:.6f}")
            print(f"  Magnetic Field: {info['Magnetic Field']}")
        print("\n" + "="*50)

    def get_opt_environment(self, angles_setting, config = None):
        if config is not None:
            print("="*30)
            print("configuration updated")
            print("="*30)
            self.config = config
            self.nbcp_unit_cells.update_config(self.config)
            for key, value in config.items():
                print(f"{key}: {value}")
            print("="*30)

        if angles_setting is not None:
            angles_one_msl = angles_setting["One MSL"]
            angles_two_msl = angles_setting["Two MSL"]
            angles_three_msl = angles_setting["Three MSL"]
            angles_four_msl = angles_setting["Four MSL"]
        else:
            angles_one_msl = (None, None)
            angles_two_msl = (None, None, None, None)
            angles_three_msl = (None, None, None, None, None, None)
            angles_four_msl = (None, None, None, None, None, None, None, None)

        opt_env = {
            "One MSL": {"Data function": self.nbcp_unit_cells.spin_system_data_one_msl, "Angle setting": angles_one_msl},
            "Two MSL": {"Data function": self.nbcp_unit_cells.spin_system_data_two_msl, "Angle setting": angles_two_msl},
            "Three MSL": {"Data function": self.nbcp_unit_cells.spin_system_data_three_msl, "Angle setting": angles_three_msl},
            "Four MSL": {"Data function": self.nbcp_unit_cells.spin_system_data_four_msl, "Angle setting": angles_four_msl}
            }
        return opt_env


    def find_tlaf_phase(self, 
                        opt_method = "classical", 
                        angles_setting = None, 
                        N = 10, 
                        config = None, 
                        verbose = False,
                        full_range_search: bool = False,
                        num_search: int = 6):
        
        self.optimizer = SpinSystemOptimizer()
        opt_env = self.get_opt_environment(angles_setting, config)
        self.result_repository.add_tlaf_config(self.config)
        
        opt_best_E = np.inf
        cls_best_E = np.inf
        
        opt_spin_sys_data = None
        opt_phase_name = None
        
        cls_spin_sys_data = None
        cls_phase_name = None
        
        for phase_name, value in opt_env.items():
            if verbose:
                print(f"\n[START] Phase: {phase_name}")
                start_time = time.time()
            
            spin_sys_data_func  = value["Data function"]
            angles_setting      = value["Angle setting"]
            
            spin_sys_data = spin_sys_data_func()
            
            opt_result, cls_result = self.get_optimized_magnetic_unit_cell(spin_sys_data,
                                                                           self.optimizer, 
                                                                           opt_method, 
                                                                           angles_setting, 
                                                                           N = N, 
                                                                           verbose = verbose, 
                                                                           full_range_search = full_range_search,
                                                                           num_search = num_search)
            
            self.result_repository.add_opt_result(phase_name, opt_result)
            
            if opt_result["energy"] < opt_best_E:
                opt_best_E = opt_result["energy"]
                opt_best_angles = tuple(opt_result["angles"])
                opt_spin_sys_data = spin_sys_data_func(angles = tuple(opt_result["angles"]))
                opt_mu_magswt = opt_result["MAGSWT"]
                opt_phase_name = phase_name
                
                self.spin_sys_data_func = spin_sys_data_func
                
                if verbose: 
                    print(f"New ground state NBCP by {opt_method}  is detectd: {phase_name}")
            
            if cls_result["E_cl"] < cls_best_E:
                cls_best_E = cls_result["E_cl"]
                cls_best_angles = tuple(cls_result["angles"])
                cls_spin_sys_data = spin_sys_data_func(angles = tuple(cls_result["angles"]))
                cls_phase_name = phase_name
                if verbose:
                    print(f"New ground state NBCP by classical method is detectd: {phase_name}")
                
                            
            # self._print_spin_info(opt_spin_sys_data, verbose)
            if verbose:
                end_time = time.time()
                print(f"[FINISH] Phase: {phase_name} - Execution time: {end_time - start_time:.2f} seconds")
        
        if verbose:
            print("="*40)
            print(f"classical ground state: {cls_phase_name}\nEnergy = {cls_best_E:6f}")
            print(f"{opt_method} ground state: {opt_phase_name}\nEnergy = {opt_best_E:6f}")
            print("="*40)
        
        opt_result = {"phase_name": opt_phase_name, # one_msl, two_msl, ..
                      "energy": opt_best_E,         # groud state energy = E_cl + E_qu
                      "angles": opt_best_angles,    # angles, 
                      "spin_sys_data": opt_spin_sys_data,   # exchange matrix, positions in unitcell
                      "MAGSWT": opt_mu_magswt}      # 0 - 1e-9, okay, mu > 1e-9  -> positive definite, critical points.
        
        cls_result = {"phase_name": cls_phase_name,
                      "energy": cls_best_E,
                      "angles": cls_best_angles,
                      "spin_sys_data": cls_spin_sys_data}
        
        return opt_result, cls_result


    def calculate_physical_quantities(self, spin_sys_data):
        return {"skyrmion_number": calculate_skyrmion_number(spin_sys_data)}

    
    def get_optimized_magnetic_unit_cell(self, spin_sys_data, 
                                         opt_obj, opt_method, 
                                         angle_setting, 
                                         N=10, 
                                         verbose=False, 
                                         full_range_search: bool = False,
                                         num_search: int = 6):
        
        
        opt_obj = opt_obj or SpinSystemOptimizer()
        
        cef = Create_Energy_Function(spin_sys_data, N = N, update_args = True)
        
        opt_result, cls_result = opt_obj.find_minimum(cef, opt_method, 
                                                    angle_setting, 
                                                    verbose = verbose, 
                                                    full_range_search = full_range_search,
                                                    num_search = num_search,
                                                    )
        """
        format of opt results
        opt_result[phase_name] = { "energy": best_energy, 
                                    "angles": full_angles, 
                                    "method": best_method, 
                                    "E_cl": E_cl, 
                                    "E_qm": E_qm,
                                    "MAGSWT": mu_magswt}
        """
        
        return opt_result, cls_result

    
    def summarize_results(self, threshold = 1e-5, verbose=False):
        summary = []
        seen_phases = set()
        
        for phase_name, results in self.result_repository.opt_results.items():
            
            opt_method = results["method"]
            E_cl = results["E_cl"]
            E_qm = results["E_qm"]
            
            E_tot = E_cl if opt_method == "classical" else E_cl + E_qm
            
            # finding degenerate phases list
            degenerate = self._find_degenerate_states(phase_name, E_tot, opt_method, threshold)
                
            if phase_name not in seen_phases:
                summary.append({"method": opt_method,
                                "phase": phase_name,
                                "energy": E_tot,
                                "angles": results["angles"],
                                "degenerate": degenerate})
                seen_phases.add(phase_name)
                
                for deg_phase in degenerate:
                    deg_result = self.result_repository.opt_results[deg_phase]
                    if deg_phase not in seen_phases:
                        deg_E_tot = deg_result["E_cl"] if opt_method == "classical" else deg_result["E_cl"] + deg_result["E_qm"]
                        if abs(deg_E_tot - E_tot) < threshold:
                            summary.append({
                                "method": opt_method,
                                "phase": deg_phase,
                                "energy": deg_E_tot,
                                "angles": deg_result["angles"],
                                "degenerate": degenerate
                            })
                            seen_phases.add(deg_phase)

        if verbose:
            print("\n")
            print("="*54, "Optimization Summary", "="*54)
            print(f"{'Method':<15} {'Phase':<15} {'Energy':<10} {'Angles':<68} {'Degenerate States':<20}")
            print("-" * 130)
            for item in sorted(summary, key=lambda x: (x["energy"], x["phase"])):
                angles = ", ".join(f"{a:.4f}" for a in item["angles"])
                degenerate = [p for p in item["degenerate"] if p != item["phase"]]
                print(f"{item['method']:<15} {item['phase']:<15} {item['energy']:<10.5f} {angles:<68} {', '.join(degenerate) or 'None':<20}")
            print("="*130)
        
        return summary

    def _find_degenerate_states(self, best_phase, best_energy, opt_method, threshold):
        
        degenerate = []
        
        for phase_name, results in self.result_repository.opt_results.items():
            if phase_name == best_phase:
                continue

            E_cl = float(results["E_cl"] or 0)
            E_qm = float(results["E_qm"] or 0)
            E_tot = E_cl if opt_method == "classical" else E_cl + E_qm
                
            if abs(E_tot - best_energy) < threshold:
                degenerate.append(phase_name)
        
        return degenerate


    def modify_spin_sys_data_by_angles(self, opt_angles: np.ndarray, theta: float, phi: float, 
                                    N: int = 20):
        """
        Modify spin system angles by uniformly adding (theta, phi) to each spin,
        and return the updated spin system data and energy function object.
        """

        if opt_angles.ndim != 1 or len(opt_angles) % 2 != 0:
            raise ValueError("opt_angles must be a 1D array of length 2n.")
        
        n = len(opt_angles) // 2
        angles = opt_angles.copy()

        for i in range(n):
            angles[2*i:2*i+2] += np.array([theta, phi])
        
        spin_sys_data = self.spin_sys_data_func(angles=tuple(angles))
        cef = Create_Energy_Function(spin_sys_data, N=N, update_args=False)

        return cef, spin_sys_data

        
        


        
        

if __name__ == "__main__":
    config = {
        "S": 1/2,
        "Jxy": 0.067,
        "Jz": 0.125,
        "JGamma": 0.0,
        "JPD": 0.01,
        "h": (0, 0, 0.05),
        "Lande_g": 4.645,
        "Bohr_magnet": 5.788E-2
    }
    
    nbcp = FIND_TLAF_PHASES(config=config)
    best_nbcp = nbcp.find_tlaf_phase(opt_method = "MAGSWT", verbose=True)
    
    nbcp.summarize_results(verbose=True)
    nbcp.result_repository.save_to_json("results.json")