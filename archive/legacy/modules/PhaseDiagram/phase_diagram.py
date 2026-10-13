import numpy as np
import matplotlib.pyplot as plt

from .phase_analyzer import Phase, TLAFPhaseClassifier, get_phase_name
from . import visualization as vis
from . import file_io


from modules.SpinSystem.nbcp_phases import FIND_TLAF_PHASES

from modules.Tools.brillouin_zone import Brillouin_Zone
from modules.Tools.spin_system_optimizer import OPT_METHOD_NAMES

from modules.LinearSpinWaveTheory.lswt_Hamiltonian import LSWT_HAMILTONIAN
from modules.LinearSpinWaveTheory.lswt_topology import compute_Berry_curvature
from modules.LinearSpinWaveTheory.lswt_thermodynamics import compute_bosonic_number_at_k


"""
Default constant
"""
DEFAULT_DEGENERACY_CRITERIA = 1e-6

 

def get_physical_quantities(spin_sys_data, N=30):
    physical_quantities = {}
        
    # Essential Datas
    spin_info = spin_sys_data["Spin info"]
    couplings = spin_sys_data["Couplings"]
    lattice_bz_settings = spin_sys_data["Lattice/BZ setting"]
        
    Ns = len(spin_info)
    bz = Brillouin_Zone(lattice_bz_settings, bz_type = "simple")
    bz_data, full_k_points, _ = bz.get_full(N)
    bz_area = bz_data["area"]
    
    lswt_Ham = LSWT_HAMILTONIAN(spin_info, couplings)
    k_data, chem_pot_magswt = lswt_Ham.solve_k_Hamiltonian(full_k_points, 
                                                           Berry_curvature=True, 
                                                           regularization="MAGSWT")


    # print(f"len(k_data): {len(k_data)}")

    valid_count = len(k_data)
    J_mat = np.diag(np.hstack([np.ones((Ns)), -np.ones((Ns))]))

    """
    Quantities to compute
    """
    sublattice_boson_numbers = np.zeros(Ns)
    total_boson_numbers = 0
    
    sum_of_Berry_curvature = np.zeros(Ns)
    min_level_spacing = np.full(Ns, np.inf)
    
    min_eval = np.inf

    for k_key, value in k_data.items():
        Ham_k_data, Eigen_k_data, Colpa_k_data = k_data[k_key]

        colpa_success = Colpa_k_data[0]

        if colpa_success:            
            eval, evec = Eigen_k_data
            
            ### boson number ###
            sl_n_k, tot_n_k = compute_bosonic_number_at_k(eval, evec, Temperature=0, num_sl = Ns)
            
            sublattice_boson_numbers += sl_n_k
            total_boson_numbers      += tot_n_k
        
            ### Chern number ###
            pDHk = Ham_k_data[1:]
            Omega_nk, level_spacing = compute_Berry_curvature(eval = eval,
                                                              evec = evec, 
                                                              pDiffHk = pDHk,
                                                              num_sl = Ns, 
                                                              J_mat=J_mat)
            sum_of_Berry_curvature += Omega_nk
            
            min_level_spacing = np.minimum( min_level_spacing, level_spacing )
            
            min_eval = np.minimum(min_eval, np.min(eval))
            
            
        else: 
            valid_count -= 1
            
    # print(f"valid_count: {valid_count}")
    
    ### boson number ###
    msl_boson_number = sublattice_boson_numbers / valid_count
    all_boson_number = total_boson_numbers / valid_count
    
    physical_quantities["boson number"] = {"msl number": msl_boson_number,
                                            "average": all_boson_number}
    
    ### Chern number ###
    chern_number = ((1/(2*np.pi)) * sum_of_Berry_curvature * bz_area)
    
    physical_quantities["Chern number"] = {"Chern number": chern_number,
                                           "energy level spacing": min_level_spacing}
    
    # Fix: Changed "magon gap" to "magnon gap" to match the key used in process_method
    physical_quantities["magnon gap"] = {"magswt": chem_pot_magswt,
                                        "magnon gap": min_eval}
    
    return physical_quantities


class PhaseDiagramCalculator:
    """Unified phase diagram calculator with multiple optimization methods support"""
    def __init__(self, 
                 tlaf_config: dict, 
                 scanning_phase: dict, 
                 opt_method: str = "MAGSWT",
                 real_time_plot: bool = True,
                 display_method: str = "MAGSWT",
                 N = 20):
        
        """Initialize calculator with base parameters"""
        ########################
        ### define variables ###
        ########################
        self.tlaf_config = tlaf_config  # triangular lattice spin system parameters
        phase_scan_types = tuple(scanning_phase.keys())
        self.output_dir = file_io.create_output_directory( phase_scan_types )   # Path(dir_name)
        
        ##################################################################
        ### Triangular Latticle Spin System exchange matrix parameters ###
        ##################################################################
        phase_config = {}       
        
        print("Triangular lattice spin system exchange matrix paramters")
        for key, value in tlaf_config.items():
            if not (key in phase_scan_types):
                print(f"{key}: {value}")
                phase_config[key] = value
        
        phase_scan_content = {'phase scan type': phase_scan_types ,
                              'TLAF config': phase_config }
                
        ######################
        ### scanning phase ###
        ###################### 
        
        self.x_type = phase_scan_types[0]
        self.x_range = scanning_phase[self.x_type]
        self.x_values = np.arange(*self.x_range)
        
        self.y_type = phase_scan_types[1]
        self.y_range = scanning_phase[self.y_type]
        self.y_values = np.arange(*self.y_range)
        
        self.opt_method = None
        self.display_method = None
        
        self.real_time_plot = real_time_plot
            
        ###########################
        ### optimization method ###
        ###########################
        if opt_method in OPT_METHOD_NAMES:
            self.opt_method = opt_method
        else:
            raise ValueError(f"Choose optimization method in {OPT_METHOD_NAMES}")
        
        ######################
        ### Display method ###
        ######################
        if display_method in OPT_METHOD_NAMES:
            self.display_method = display_method
        else:
            raise ValueError(f"Choose method to display in {OPT_METHOD_NAMES}")
        
        file_io.save_configuration(self.output_dir, phase_scan_content)
        
        self.num_BZ = N
        
        
        return



    def calculate_phase_diagram(self, angles_setting):
        """Calculate phase diagram with real-time visualization"""
        if self.x_values is None or self.y_values is None:
            raise ValueError("Call setup_calculation first")
        
        
        cls_phase_map = np.zeros((len(self.x_values), len(self.y_values)), dtype = int)
        cls_phys_quant_map = np.empty((len(self.x_values), len(self.y_values)), dtype=object)
        # cls_degenerate_bool_map = np.zeros((len(self.x_values), len(self.y_values)), dtype = bool)
        
        opt_phase_map = np.zeros((len(self.x_values), len(self.y_values)), dtype = int)
        opt_phys_quant_map = np.empty((len(self.x_values), len(self.y_values)), dtype=object)
        # opt_degenerate_bool_map = np.zeros((len(self.x_values), len(self.y_values)), dtype = bool)

        # 실시간 플롯 초기화
        if self.real_time_plot:
            fig, ax, scatters, background = vis.initialize_plot(self.x_type, self.x_range, self.y_type, self.y_range)
            
            ax.set_title(f'Phase Diagram ({self.x_type} vs {self.y_type}): optimization method {self.display_method}')
        
        total_points = len(self.x_values) * len(self.y_values)
        
        
        # 메인 계산 루프
        for i, x in enumerate(self.x_values):
            for j, y in enumerate(self.y_values):
                
                # point
                x_pt = (i, x)
                y_pt = (j, y)
                
                # print process
                progress = ((i * len(self.y_values) + j) / total_points * 100)
                print(f"Progress: {progress:.1f}%")
                print(f"current {self.x_type}-value:\t{x:.3f}")
                print(f"current {self.y_type}-value:\t{y:.3f}")
                
                # triangular lattice 
                tlaf_params = self.tlaf_config.copy()
                
                tlaf_params[self.x_type] = (0, 0, x) if self.x_type == "h" else x
                tlaf_params[self.y_type] = (0, 0, y) if self.y_type == "h" else y

                print(tlaf_params)

                nbcp = FIND_TLAF_PHASES(config = tlaf_params)
                opt_result, cls_result = nbcp.find_tlaf_phase(opt_method = self.opt_method, 
                                                              angles_setting = angles_setting, 
                                                              N = self.num_BZ, 
                                                              verbose = False)

                # MAGSWT 처리
                opt_obj, opt_phase, opt_phys_quant = self.process_method("MAGSWT", x_pt, y_pt, opt_result, N = 20)                

                # # Classical 처리
                cls_obj, cls_phase, cls_phys_quant = self.process_method("classical", x_pt, y_pt, cls_result, N = 20)

            
                # 디버깅: opt_result와 cls_result 비교
                opt_energy = float(opt_result["energy"])
                opt_angles = opt_result["angles"]
                cls_energy = float(cls_result["energy"])
                cls_angles = cls_result["angles"]
                print(f"opt_result spin_sys_data: \t energy = {opt_energy:.4e} \n", f"angles: {opt_angles}")
                print(f"cls_result spin_sys_data: \t energy = {cls_energy:.4e} \n", f"angles: {cls_angles}")
                
                print(f"MAGSWT phase: {opt_obj.name} ({opt_phase})")
                print(f"Classical phase: {cls_obj.name} ({cls_phase})")
                
                
                opt_phase_map[i,j] = opt_phase
                cls_phase_map[i,j] = cls_phase
                
                opt_phys_quant_map[i, j] = opt_phys_quant      
                cls_phys_quant_map[i, j] = cls_phys_quant      

                if self.real_time_plot:
                    if self.display_method == "MAGSWT":
                        phase_to_plot = opt_obj
                    else:
                        phase_to_plot = cls_obj
                    
                    vis.update_plot(fig, ax, 
                                    value_x=x,
                                    value_y=y,
                                    phase=phase_to_plot,
                                    scatters=scatters, 
                                    background=background)
                

        print("\nCalculation completed!")
        
        # 결과 저장용 딕셔너리
        results = { "classical": { "phase_map": cls_phase_map,
                                  "physical_quantities_map": cls_phys_quant_map},
                   "MAGSWT":     { "phase_map": opt_phase_map,
                                  "physical_quantities_map": opt_phys_quant_map} }
        
        # 모든 메서드의 결과 저장
        for method in ["classical", "MAGSWT"]:
            self._save_method_results(method,
                                      results[method]["phase_map"],
                                      results[method]["physical_quantities_map"]
            )
        
        # 실시간 플롯 저장 및 표시
        if self.real_time_plot:
            fig.savefig(
                self.output_dir / f'phase_diagram_{self.display_method.lower()}_final.png',
                bbox_inches='tight', dpi=600
            )
            plt.show(block=True)
        
        return self.output_dir


    def process_method(self, method, x_pt, y_pt, result_data, N=20):
        """공통 메서드 처리 로직"""
        idx_i, value_x = x_pt
        idx_j, value_y = y_pt
        
        filename = f'{method.lower()}_{self.x_type}_{idx_i:03d}_{self.y_type}_{idx_j:03d}.json'
        
        # Classify the TLAF phase
        spin_sys_data = result_data["spin_sys_data"]
        angles = result_data["angles"]
        energy = result_data["energy"]
        
        phase_classifier = TLAFPhaseClassifier(spin_sys_data)
        phase = phase_classifier.phase        

        try:
            classified_phase = phase.value
            physical_quantities = get_physical_quantities(spin_sys_data, N=N)

            # Ensure all expected keys are in the physical_quantities dictionary
            if "magnon gap" not in physical_quantities:
                print(f"Warning: 'magnon gap' key not found in physical_quantities for {method}")
                print(f"Available keys: {physical_quantities.keys()}")
                # Use default values if the key doesn't exist
                physical_quantities["magnon gap"] = {
                    "magswt": 0.0,
                    "magnon gap": 0.0
                }
            
            # Safely extract values with proper error handling
            try:
                magnon_gap_magswt = physical_quantities["magnon gap"]["magswt"]
                magnon_gap_value = physical_quantities["magnon gap"]["magnon gap"]
                
                # Convert to float with safety check
                magswt_float = float(magnon_gap_magswt) if magnon_gap_magswt is not None else 0.0
                magnon_gap_float = float(magnon_gap_value) if magnon_gap_value is not None else 0.0
            
            except (KeyError, TypeError) as e:
                print(f"Error extracting magnon gap values: {e}")
                magswt_float = 0.0
                magnon_gap_float = 0.0

            point_result = {
                self.x_type: value_x,
                self.y_type: value_y,
                'phase': str(phase.name),
                'energy': float(energy),
                'angles': [float(ang) for ang in angles],  # np.array 가능성 고려
                'physical_quantities': {
                    'chern_number': {
                        'Chern number': [float(c) for c in physical_quantities['Chern number']['Chern number']],  
                        'energy level spacing':[float(c) for c in physical_quantities['Chern number']['energy level spacing']] 
                        },
                    'boson_number': {
                        'msl_number': [float(n) for n in physical_quantities['boson number']['msl number']],  # np.array 가능성
                        'average': float(physical_quantities['boson number']['average'])
                        },
                    'magnon gap': {  # Fix: Changed from 'magon_gap' to 'magnon gap'
                        'magswt': magswt_float,
                        'magnon_gap': magnon_gap_float
                        }
                    }
                }
            
            file_io.save_point_data(self.output_dir, filename, point_result)
            
            return phase, classified_phase, point_result

        except Exception as e:
            print(f"Error in {method} processing physical quantities at {self.x_type}={value_x:.3f}, {self.y_type}={value_y:.3f}: {e}")
            return None, None, None


    def _save_method_results(self, method_name, phase_map, physical_quantities_map):
        """특정 최적화 방법의 결과를 JSON 형식으로 저장"""
        import json
        import numpy as np
        
        # NumPy 배열을 JSON 직렬화 가능한 형태로 변환하는 도우미 함수
        def convert_numpy_to_python(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.number):
                return obj.item()  # np.float64 등을 float으로 변환
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(item) for item in obj]
            else:
                return obj
        
        # Phase map JSON으로 저장
        phase_map_path = self.output_dir / f'phase_map_{method_name.lower()}.json'
        with open(phase_map_path, 'w') as f:
            json.dump(phase_map.tolist(), f, indent=2)
        
        # Physical quantities 맵 JSON으로 저장
        if physical_quantities_map is not None:
            phys_quant_path = self.output_dir / f'physical_quantities_{method_name.lower()}.json'
            
            # 결과를 저장할 리스트 초기화
            results_json = []
            
            # physical_quantities_map의 각 항목을 처리
            for i in range(physical_quantities_map.shape[0]):
                row = []
                for j in range(physical_quantities_map.shape[1]):
                    point_data = physical_quantities_map[i, j]
                    
                    # None 값 처리
                    if point_data is None:
                        row.append(None)
                        continue
                    
                    try:
                        # NumPy 배열 등을 Python 기본 타입으로 변환
                        converted_data = convert_numpy_to_python(point_data)
                        row.append(converted_data)
                    except Exception as e:
                        print(f"Error processing physical quantities at [{i},{j}]: {e}")
                        print(f"Data that caused error: {type(point_data)}")
                        row.append(None)
                
                results_json.append(row)
            
            # JSON 파일로 저장
            with open(phys_quant_path, 'w') as f:
                json.dump(results_json, f, indent=2)
        
        # 최종 플롯 생성 및 저장
        self._create_final_plot(method_name, phase_map)


    def _create_final_plot(self, method_name, phase_map):
        """특정 최적화 방법에 대한 최종 플롯 생성 및 저장"""
        # 설정 로드
        config_path = self.output_dir / 'config.json'
        if not config_path.exists():
            print(f"Warning: Config file not found at {config_path}")
            return
            
        # 데이터 플롯 작업은 visualization 모듈의 plot_final_results 함수를 활용
        # 약간 수정이 필요하므로 이 함수에서 직접 처리
        
        plt.figure(figsize=(15, 10))
        
        # Phase별 색상 매핑
        unique_phases = np.unique(phase_map)
        
        # 각 상(phase)별로 산점도 생성
        for phase_idx in unique_phases:
            mask = (phase_map == phase_idx)
            phase = Phase(phase_idx)
            x_coords = []
            y_coords = []
            
            for i in range(len(self.x_values)):
                for j in range(len(self.y_values)):
                    if mask[i, j]:
                        x_coords.append(self.x_values[i])
                        y_coords.append(self.y_values[j])
            
            # 외부 스타일 함수 사용
            style = vis.get_phase_style(phase)
            plt.scatter(x_coords, y_coords,
                       c=style['color'],
                       marker=style['marker'],
                       s=style['size'],
                       edgecolors=style['edgecolor'],
                       linewidth=style['linewidth'],
                       label=f"{vis.get_phase_name(phase)} ({phase.value})",
                       alpha=0.8)
        
        # 축 범위와 그리드 설정
        x_start, x_end, x_step = self.x_range
        y_start, y_end, y_step = self.y_range
        plt.xlim(x_start, x_end)
        plt.ylim(y_start, y_end)
        plt.grid(True, linestyle='-', alpha=0.3, which='major', color='gray')
        plt.grid(True, linestyle=':', alpha=0.2, which='minor', color='gray')
        plt.minorticks_on()
        plt.xticks(np.arange(*self.x_range))
        plt.yticks(np.arange(*self.y_range))
    
        plt.xlabel(f'{self.x_type} (meV)')
        plt.ylabel(f"{self.y_type} (meV)")
        plt.title(f"Phase Diagram ({self.x_type} vs {self.y_type}) - optimization method: {method_name}")
        
        # 범례 설정
        legend = plt.legend(bbox_to_anchor=(1.05, 1), 
                           loc='upper left',
                           fontsize=10,
                           frameon=True,
                           title='Phases',
                           title_fontsize=12)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        # 레이아웃 조정
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # 결과 저장
        output_path = self.output_dir / f'phase_diagram_{method_name.lower()}.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=600)
        plt.close()  # 메모리 해제