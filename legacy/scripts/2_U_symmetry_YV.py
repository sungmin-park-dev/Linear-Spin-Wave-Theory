import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from modules.SpinSystem.nbcp_phases import FIND_TLAF_PHASES
from modules.Tools.analysis_tools import Create_Energy_Function
from modules.Tools.spin_system_optimizer import SpinSystemOptimizer
from modules.Plotters.spin_system_visualizer import SpinVisualizer
from modules.LinearSpinWaveTheory.linear_spin_wave_theory import LSWT



class Find_Quantum_Correction:
    def __init__(self, config):
        
        self.config = config 
        self.nbcp = FIND_TLAF_PHASES(config = self.config)
        
        print(f"Initial config: {self.config}")
    

    @staticmethod
    def _get_angles(spin_sys_data, offset = (0, 0)):
        angles = []
        
        delta_theta, delta_phi = offset
            
        for j, (key, value) in enumerate(spin_sys_data["Spin info"].items()):
            theta, phi = value["Angles"]
            
            if j == 0:
                default_phi = phi
            
            angles.append( theta + delta_theta)
            angles.append( phi + delta_phi - default_phi)
        
        return np.array(angles)
    
    @staticmethod
    def classical_Y_phase(config):
        # 파라미터 가져오기
        S = config["S"]
        Jxy = config["Jxy"]
        Jz = config["Jz"]
        h = config["h"]
        
        # h가 튜플인 경우 z-성분(세 번째 요소)만 추출
        if isinstance(h, tuple):
            hz = h[2]  # z-축 성분
        else:
            hz = h
        
        if hz <= 3*S*Jxy:
            theta = np.acos((hz+3*S*Jz)/(3*S*(Jxy+Jz)))
            return True, theta
        else:
            return False, None
    
    
    def get_nbcp_phase(self, pert, phase_name = None, angles_setting = None, show_visualizer = False):
        
        for pert_type, pert_value in pert.items():
            if pert_type not in self.config:
                raise ValueError(f"Invalid perturbation type: {pert_type}.")
            print(f"perturbation: {pert_type}, value: {pert_value}")
            self.config[pert_type] = pert_value

        if phase_name is None:
            nbcp_phase = self.nbcp.find_tlaf_phase(opt_method = "classical",
                                                   angles_setting = angles_setting, 
                                                   config = self.config)
        else:
            self.nbcp.nbcp_unit_cells.update_config(self.config)

            if phase_name == "One MSL":
                data_func = self.nbcp.nbcp_unit_cells.spin_system_data_one_msl
                spin_sys_data = data_func()

            elif phase_name == "Two MSL":
                data_func = self.nbcp.nbcp_unit_cells.spin_system_data_two_msl
                spin_sys_data = data_func()

            elif phase_name == "Three MSL":
                data_func = self.nbcp.nbcp_unit_cells.spin_system_data_three_msl
                spin_sys_data = data_func()

            elif phase_name == "Four MSL":
                data_func = self.nbcp.nbcp_unit_cells.spin_system_data_four_msl
                spin_sys_data = data_func()
            else:
                raise ValueError(f"Invalid phase name: {phase_name}.")
            
            if isinstance(angles_setting, dict) and phase_name in angles_setting:
                angle_setting = angles_setting[phase_name]
                print(f"Using angles for phase: {phase_name}, angles={angle_setting}")
            
            elif angles_setting is None:
                angle_setting = [None, None] * len(spin_sys_data["Spin info"])
            else:
                raise ValueError(f"Invalid angles setting for phase: {phase_name}.")
            
            cef = Create_Energy_Function(spin_sys_data, N = 10, update_args = False)
            
            optimizer = SpinSystemOptimizer()
            
            opt_result_data, cls_result_data = optimizer.find_minimum(cef, 
                                                                      opt_method = "classical",
                                                                      angle_setting = angle_setting,)
            
            nbcp_phase = data_func(angles = opt_result_data["angles"])
        
        return nbcp_phase

    @staticmethod
    def show_spin_conf(spin_sys_data, parameters_details = None,):
        visualizer = SpinVisualizer()
        fig_vis, (ax_xy, ax_angle) = visualizer.plot_system(spin_sys_data, figsize=(15, 6))
        fig_vis.suptitle(f"NBCP spin Configuration ({parameters_details})", fontsize=16, y=1.05)
        plt.show()        
        return

    def calculate_energy_vs_phi(self, spin_sys_data, rot_phi, N = 20):
        """
        스핀 시스템 데이터와 회전 각도를 기반으로 에너지를 계산합니다.
        """
        
        
        cef = Create_Energy_Function(spin_sys_data, 
                                    N = N, 
                                    update_args = True)
        
        energy_classical = []
        energy_quantum = []
        energy_total = []
        mu_magswt = []
        
        all_angles = []
        
        for key, value in spin_sys_data["Spin info"].items():
                angle = value["Angles"]
                all_angles.append(angle)
                print(f"{key}: {angle} in calculate_energy_vs_phi")
                
        print(f"Initial angles: {np.hstack(all_angles)}")
        
        for i, delta in enumerate(rot_phi):
            
            angles = self._get_angles(spin_sys_data, offset = (0, delta))

            # 고전 에너지와 양자 에너지 분리 계산
            E_classical = cef.classical_energy_density_func(angles)
            E_quantum = cef.quantum_energy_density_func(angles, reg_type = 0)
            E_total = E_classical + E_quantum
            
            print(f"Calculating energy for phi: {angles[1]:.4f}\n",{float(E_total)})
            
            
            energy_classical.append(E_classical)
            energy_quantum.append(E_quantum)
            energy_total.append(E_total)
            
            if cef.mu_magswt == None:
                mu = 0
            else:
                mu = cef.mu_magswt
            
            mu_magswt.append(mu)
            
        # 최대/최소 에너지 출력
        print("\nEnergy ranges:")
        print(f"  Classical: {min(energy_classical):.8f} to {max(energy_classical):.8f}, Δ={max(energy_classical)-min(energy_classical):.8f}")
        print(f"  Quantum:   {min(energy_quantum):.8f} to {max(energy_quantum):.8f}, Δ={max(energy_quantum)-min(energy_quantum):.8f}")
        print(f"  Total:     {min(energy_total):.8f} to {max(energy_total):.8f}, Δ={max(energy_total)-min(energy_total):.8f}")
            
        return energy_total, energy_classical, energy_quantum, mu_magswt
    
    def plot_energy_vs_phi(self, pert_type, phase_name=None,
                            j_start = 0,
                            j_end = 0.01,
                            j_step = 0.002,
                            angles_setting = None,
                            show_visualizer = False,
                            N = 20, 
                            delta_phi = 100):            
        """
        다양한 j 값에 대해 phi에 따른 에너지를 계산하고 플롯합니다.
        2x2 그리드로 양자 에너지, 고전 에너지, 총 에너지, mu_magswt를 표시합니다.
        """
        
        # 변수 이름 변경: phi -> rot_phi_range로 명확히 구분
        rot_phi_range = np.linspace(0, 2*np.pi, delta_phi)
        
        j_values = np.arange(j_start, j_end + j_step/2, j_step)  # 끝값 포함
        
        print(f"Calculating energies for j values: {j_values}")
        
        E_jval = {}
        E_classical_jval = {}
        E_quantum_jval = {}
        Mu_magswt_jval = {}
        
        # 색상 준비 - j값이 증가함에 따라 색상이 더 진해지도록 설정
        # 낮은 j값은 연한 색, 높은 j값은 진한 색
        base_color = 'blue'  # 기본 색상
        color_intensities = np.linspace(0.2, 0.8, len(j_values))  # 진하기 범위: 0.3(연함) ~ 1.0(진함)
        colors = []
        
        # 색상 생성 (파란색 계열, 진하기만 다르게)
        for intensity in color_intensities:
            colors.append((0, 0, intensity, 0.8))  # RGBA 형식 (0,0,intensity,1) = 파란색
        
        # 2x2 서브플롯 그리드 생성
        fig, axs = plt.subplots(2, 2, figsize=(14, 12), sharex=True)
        
        # 각 j 값에 대해 반복
        for i, j_val in enumerate(j_values):
            print(f"Calculating for {pert_type} = {j_val:.4f}")
            
            nbcp_phase = self.get_nbcp_phase(pert = {pert_type: j_val}, 
                                            phase_name = phase_name,
                                            angles_setting = angles_setting)
            
            if show_visualizer:
                details = f"{pert_type}={j_val:.4f}, phase={phase_name}"
                self.show_spin_conf(nbcp_phase, parameters_details = details)
            
            # 에너지 계산 - phase_name 직접 전달
            result = self.calculate_energy_vs_phi(nbcp_phase, rot_phi_range, N = N)
            energies, classical_energies, quantum_energies, mu_magswt = result
            
            # 에너지를 딕셔너리에 저장
            E_jval[j_val] = energies
            E_classical_jval[j_val] = classical_energies
            E_quantum_jval[j_val] = quantum_energies
            
            # mu_magswt 값 처리 - 이것의 구조는 확인이 필요합니다
            # 만약 이것이 스칼라 값이거나 리스트가 아닌 경우, 그래프를 그리기 위한 리스트로 변환
            if isinstance(mu_magswt, (int, float)):
                mu_magswt_list = [mu_magswt] * len(rot_phi_range)
            elif hasattr(mu_magswt, '__iter__'):  # 이미 리스트나 배열인 경우
                mu_magswt_list = mu_magswt
            else:
                print(f"Warning: Unexpected mu_magswt type: {type(mu_magswt)}")
                mu_magswt_list = [0] * len(rot_phi_range)  # 안전하게 기본값 설정
            
            # 에너지 범위 확인
            min_energy = min(energies)
            max_energy = max(energies)
            min_classical = min(classical_energies)
            max_classical = max(classical_energies)
            min_quantum = min(quantum_energies)
            max_quantum = max(quantum_energies)
            
            
            print(f"Energy ranges for {pert_type}={j_val:.4f}:")
            print(f"  Total:     {min_energy:.6f} to {max_energy:.6f}, Δ={max_energy-min_energy:.8f}")
            print(f"  Classical: {min_classical:.6f} to {max_classical:.6f}, Δ={max_classical-min_classical:.8f}")
            print(f"  Quantum:   {min_quantum:.6f} to {max_quantum:.6f}, Δ={max_quantum-min_quantum:.8f}")
            
            # 양자 에너지 (1, 1)
            new_quantum_energies = (np.array(quantum_energies) - np.average(quantum_energies))/np.abs(min_quantum - np.average(quantum_energies))
            
            axs[0, 0].plot(rot_phi_range, new_quantum_energies, color=colors[i], 
                            linestyle='-', linewidth=2,
                            label=f'{pert_type} = {j_val:.4f}')
            
            # 고전 에너지 (1, 2)
            axs[0, 1].plot(rot_phi_range, classical_energies, color=colors[i], 
                        linestyle='-', linewidth=2,
                        label=f'{pert_type} = {j_val:.4f}')
            
            # 총 에너지 (2, 1)
            axs[1, 0].plot(rot_phi_range, energies, color=colors[i], 
                        linestyle='-', linewidth=2,
                        label=f'{pert_type} = {j_val:.4f}')
            
            # mu_magswt (2, 2)
            axs[1, 1].plot(rot_phi_range, mu_magswt_list, color=colors[i], 
                        linestyle='-', linewidth=2,
                        label=f'{pert_type} = {j_val:.4f}')

        # hz 값 가져오기
        hz = self.config["h"][2] if isinstance(self.config["h"], tuple) else self.config["h"]
        
        # mu_magswt 값 디버깅
        print(f"mu_magswt type: {type(mu_magswt)}")
        if hasattr(mu_magswt, '__iter__'):
            print(f"mu_magswt length: {len(mu_magswt)}, first few values: {mu_magswt[:3] if len(mu_magswt) > 3 else mu_magswt}")

        S = self.config["S"]
        J = self.config["Jxy"] if "Jxy" in self.config.keys() else self.config["Jx"]
        Jz = self.config["Jz"]
        
        # 최적화 설정 및 위상 정보 문자열 만들기
        phase_info = f"Phase: {phase_name}" if phase_name else "Phase: Auto-detected"
        opt_settings = f"S={S}, J={J}, Jz={Jz}, hz={hz}, N={N}"
        
        # 전체 그래프 제목 설정
        fig.suptitle(f'Energy vs Phi Analysis\n{phase_info}, {opt_settings}', 
                    fontsize=16, y=0.98)
        
        # 양자 에너지 그래프 설정
        axs[0, 0].set_title('Quantum Energy')
        axs[0, 0].set_ylabel('Energy')
        axs[0, 0].legend(fontsize=9, loc = 'upper right')
        axs[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 고전 에너지 그래프 설정
        axs[0, 1].set_title('Classical Energy')
        axs[0, 1].set_ylabel('Energy')
        axs[0, 1].legend(fontsize=9)
        axs[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 총 에너지 그래프 설정
        axs[1, 0].set_title('Total Energy (Classical + Quantum)')
        axs[1, 0].set_xlabel('Phi (radians)')
        axs[1, 0].set_ylabel('Energy')
        axs[1, 0].legend(fontsize=9)
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # mu_magswt 그래프 설정
        axs[1, 1].set_title('Mu_magswt')
        axs[1, 1].set_xlabel('Phi (radians)')
        axs[1, 1].set_ylabel('Magnitude')
        axs[1, 1].legend(fontsize=9)
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 모든 그래프에 x축 범위 설정 (0부터 2π까지)
        for ax_row in axs:
            for ax in ax_row:
                ax.set_xlim(0, 2*np.pi)
                # x축 눈금 설정 (0, π/2, π, 3π/2, 2π)
                ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
                ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # 전체 제목을 위한 공간 확보
        plt.show()
        
        return rot_phi_range, E_jval, E_classical_jval, E_quantum_jval



# 실행 부분
if __name__ == "__main__":
    
    nbcp_config = {"S": 1/2,
                "Jxy": 0.076,
                "Jz": 0.125,
                "JGamma": 0.00,
                "JPD": 0.0,
                "KPD": 0.001,
                "KGamma": 0.001,
                "Kxy": 0.0,
                "Kz": 0.0,
                "h": (0.0, 0.0, 0.0),
                "nne": (0.00, 0.00, 0.0)}
    
    # ==========================
    # h = 0.05377
    h = 0.376418
    nbcp_config["h"] = (0.0, 0.0, h)  # z-축 성분만 설정    
    phase_name =  "Three MSL"           # Fix magnetic sublattice
    
    perturbation_type = "JPD"
    j_start = 0.0
    j_end = 0.01
    j_step = 0.002
    
    show_visualizer = False
    
    phi = 0.0
    
    # 위상에 따른 각도 설정 (튜플 형태)
    angle_setting = {"One MSL": (None, phi),
                     "Two MSL": (None, phi, None, phi),
                     "Three MSL": (None, phi, None, phi, None, phi),
                     "Four MSL": (None, None, None, None, None, None, None, None),}
    
    # ==========================

    
    # FIND_TLAF_PHASES 클래스에는 각도 설정을 전달하지 않음
    fqc = Find_Quantum_Correction(config = nbcp_config)
    
    # JPD 값 범위 확장 - phi 의존성을 더 잘 관찰하기 위해
    rot_phi_range, E_jval, E_classical_jval, E_quantum_jval = fqc.plot_energy_vs_phi(perturbation_type,
                                                                                     phase_name = phase_name,
                                                                                     j_start = j_start,
                                                                                     j_end = j_end,  
                                                                                     j_step = j_step,  # 단계 확장
                                                                                     angles_setting = angle_setting,
                                                                                     show_visualizer = show_visualizer,
                                                                                     N = 20)
    output_path = Path("datas")
    output_path.mkdir(exist_ok=True)  # 없으면 폴더 생성


    
    # 양자 에너지 결과를 엑셀로 저장
    df_dict = {"phi": rot_phi_range}
    for j_val in sorted(E_quantum_jval.keys()):
        quantum_energies = E_quantum_jval[j_val]
        col_name = f"{perturbation_type} = {j_val:.4f}"
        df_dict[col_name] = quantum_energies

    df_quantum = pd.DataFrame(df_dict)
    
    h_str = f"{h:.3f}".replace('.', 'p')  # → '0p376'
    j_start_str = f"{j_start:.3f}".replace('.', 'p')  # → '0p000'
    j_end_str = f"{j_end:.3f}".replace('.', 'p')  # → '0p010'
    
    filename = f"E_qm_pert_{perturbation_type}_from_{j_start_str}_to_{j_end_str}_h_{h_str}.xlsx"
    excel_filename = output_path / filename
    
    df_quantum.to_excel(excel_filename, index=False)
    print(f"Quantum energy data saved to {excel_filename}")