import json
import numpy as np

class ResultRepository:
    def __init__(self):
        self.tlaf_config = {}
        self.opt_results = {}  # {phase_name: {opt_method: {...}}}
        
    def add_tlaf_config(self, config = None):
        self.tlaf_config = config if config is not None else {}
    
    def add_opt_result(self, phase_name, opt_result):
        
        integrated_data = {
            "energy": opt_result["energy"],
            "angles": opt_result["angles"],
            "method": opt_result["method"],
            "E_cl":   opt_result["E_cl"],
            "E_qm":   opt_result["E_qm"],
            "MAGSWT": opt_result["MAGSWT"]
            }
        
        self.opt_results[phase_name] = integrated_data
        # self.opt_results.setdefault(phase_name, {}) = integrated_data
        
        return


    def save_to_json(self, filename):
        def serialize_array(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: serialize_array(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_array(item) for item in obj]
            elif isinstance(obj, tuple):
                return [serialize_array(item) for item in obj]
            return obj

        serializable_results = {}
        for phase_name, results in self.opt_results.items():
            serializable_results[phase_name] = {}
            for opt_method, data in results.items():
                serializable_results[phase_name][opt_method] = serialize_array({
                    "energy": data["energy"],
                    "angles": data["angles"],
                    "method": data["method"],
                    "E_cl": data["E_cl"],
                    "E_qm": data["E_qm"],
                    "MAGSWT": data["MAGSWT"],
                    "k_pts": data["k_pts"],
                    "physical_quantities": data["physical_quantities"],
                    "nbcp_config": data["nbcp_config"]
                })
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=4)