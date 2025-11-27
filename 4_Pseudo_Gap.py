import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from modules.SpinSystem.nbcp_phases import FIND_TLAF_PHASES
from modules.Tools.analysis_tools import Create_Energy_Function
from modules.Tools.spin_system_optimizer import SpinSystemOptimizer
from modules.Plotters.spin_system_visualizer import SpinVisualizer
from modules.LinearSpinWaveTheory.linear_spin_wave_theory import LSWT


DEFAULT_D_THETA = 1e-3*np.pi    #
DEFAULT_D_PHI   = np.pi/20    # Cons

class Compute_Pseudo_Gap:
    def __init__(self, spin_sys_data, Temperature = 0, config=None):
        """Type I 시스템의 pseudo goldstone gap 계산"""
        self.spin_sys_data = spin_sys_data
        self.Temperature = Temperature
        self.cef_obj = None
        
        if config is not None:
            self.config = config 
            print(f"Initial config: {self.config}")

    @staticmethod
    def _get_angles(spin_sys_data, offset=(0, 0)):
        """스핀 각도 추출 및 offset 적용"""
        angles = []
        delta_theta, delta_phi = offset
            
        for j, (key, value) in enumerate(spin_sys_data["Spin info"].items()):
            theta, phi = value["Angles"]
            angles.append(theta + delta_theta)
            angles.append(phi + delta_phi)
        
        return np.array(angles)

    def _get_E_cl_and_qm(self, angles, offset=(0, 0)):
        """
        Type I에서 classical/quantum 에너지 계산
        - Classical: theta, phi 모두 적용 (phi에 대해 불변)
        - Quantum: phi만 적용 (theta는 soft mode 아님)
        """
        if self.cef_obj is None:
            raise ValueError("cef_obj가 초기화되지 않음. calculate_pseudo_gap 먼저 호출 필요")
            
        N = int(len(angles) / 2)
        theta, phi = offset
        
        cl_angles = angles.copy()
        qm_angles = angles.copy()
        
        # Global rotation 적용
        for i in range(N):
            cl_angles[2*i:2*i+2] = angles[2*i:2*i+2] + np.array([theta, phi])   
            qm_angles[2*i:2*i+2] = angles[2*i:2*i+2] + np.array([0, phi])       
            
        E_cl = self.cef_obj.classical_energy_density_func(cl_angles)
        E_qm = self.cef_obj.quantum_free_energy_density_func(qm_angles,
                                                             reg_type=0,    ## need to check
                                                             Temperature = self.Temperature)

        # MAGSWT potential 체크
        if self.cef_obj.mu_magswt is not None:
            mu = self.cef_obj.mu_magswt
            if mu > 1e-5:
                print(f"[Warning] MAGSWT potential (negative eigenvalue): {mu}")
            else:
                print(f"[Note] MAGSWT potential (numerical): {mu}")
                
        return E_cl, E_qm

    def calculate_pseudo_gap(self, spin_sys_data=None, N=20, dtheta=None, dphi=None, use_five_point=False):
        """
        Type I 시스템의 pseudo goldstone gap 계산
        E(θ, φ) = E_cl(θ, φ) + E_qm(φ), 여기서 E_cl은 φ에 불변
        
        Args:
            use_five_point: True이면 5점 공식(O(h⁴)), False이면 2점 공식(O(h²)) 사용
        """
        if spin_sys_data is None:
            spin_sys_data = self.spin_sys_data
            
        # 에너지 함수 객체 생성
        self.cef_obj = Create_Energy_Function(spin_sys_data, N=N, update_args=True)
        
        angles = self._get_angles(spin_sys_data, offset=(0, 0))
        
        # 수치미분 간격
        dtheta = dtheta or DEFAULT_D_THETA 
        dphi = dphi or DEFAULT_D_PHI
        
        print(f"수치미분 방법: {'5점 공식 (O(h⁴))' if use_five_point else '2점 공식 (O(h²))'}")
        print(f"수치미분 간격: dtheta={dtheta:.6e}, dphi={dphi:.6e}")

        try:
            # 기준점
            E_cl_00, E_qm_00 = self._get_E_cl_and_qm(angles, offset=(0, 0))
            
            # theta 방향 (classical energy용)
            E_cl_p10, E_qm_p10 = self._get_E_cl_and_qm(angles, offset=(+dtheta, 0))
            E_cl_m10, E_qm_m10 = self._get_E_cl_and_qm(angles, offset=(-dtheta, 0))

            # phi 방향 (quantum energy용)
            E_cl_0p1, E_qm_0p1 = self._get_E_cl_and_qm(angles, offset=(0, +dphi))
            E_cl_0m1, E_qm_0m1 = self._get_E_cl_and_qm(angles, offset=(0, -dphi))
            
            if use_five_point:
                # 5점 공식용 추가 계산점 (±2h)
                E_cl_p20, E_qm_p20 = self._get_E_cl_and_qm(angles, offset=(+2*dtheta, 0))
                E_cl_m20, E_qm_m20 = self._get_E_cl_and_qm(angles, offset=(-2*dtheta, 0))
                E_cl_0p2, E_qm_0p2 = self._get_E_cl_and_qm(angles, offset=(0, +2*dphi))
                E_cl_0m2, E_qm_0m2 = self._get_E_cl_and_qm(angles, offset=(0, -2*dphi))
            
        except Exception as e:
            print(f"Error in energy calculation: {e}")
            return 0.0
        
        # Type I 수치미분: E(θ, φ) = E_cl(θ) + E_qm(φ)
        if use_five_point:
            # 5점 공식 (O(h⁴) 정확도)
            D2E_DTHETA2 = (-E_cl_p20 + 16*E_cl_p10 - 30*E_cl_00 + 16*E_cl_m10 - E_cl_m20) / (12 * dtheta**2)
            D2E_DPHI2 = (-E_qm_0p2 + 16*E_qm_0p1 - 30*E_qm_00 + 16*E_qm_0m1 - E_qm_0m2) / (12 * dphi**2)
            print(f"D2E_DTHETA2 = (-{E_cl_p20:.6f} + 16*{E_cl_p10:.6f} - 30*{E_cl_00:.6f} + 16*{E_cl_m10:.6f} - {E_cl_m20:.6f}) / (12*{dtheta**2:.2e}) = {D2E_DTHETA2:.6e}")
            print(f"D2E_DPHI2 = (-{E_qm_0p2:.6f} + 16*{E_qm_0p1:.6f} - 30*{E_qm_00:.6f} + 16*{E_qm_0m1:.6f} - {E_qm_0m2:.6f}) / (12*{dphi**2:.2e}) = {D2E_DPHI2:.6e}")
        else:
            # 2점 공식 (O(h²) 정확도)
            D2E_DTHETA2 = (E_cl_p10 + E_cl_m10 - 2 * E_cl_00) / dtheta**2
            D2E_DPHI2 = (E_qm_0p1 + E_qm_0m1 - 2 * E_qm_00) / dphi**2
            print(f"D2E_DTHETA2 = ({E_cl_p10:.8f} + {E_cl_m10:.8f} - 2*{E_cl_00:.8f}) / {dtheta**2:.2e} = {D2E_DTHETA2:.6e}")
            print(f"D2E_DPHI2 = ({E_qm_0p1:.8f} + {E_qm_0m1:.8f} - 2*{E_qm_00:.8f}) / {dphi**2:.2e} = {D2E_DPHI2:.6e}")
        
        D2E_DPHI_DTHETA = 0  # Type I에서 mixed derivative는 0
        
        # 디버깅: classical energy의 phi 불변성 확인
        print(f"\n=== Type I 특성 검증 ===")
        print(f"E_cl_00: {E_cl_00:.8f}")
        print(f"E_cl_0p1: {E_cl_0p1:.8f}")
        print(f"E_cl_0m1: {E_cl_0m1:.8f}")
        print(f"Classical phi-invariance check: |ΔE| = {abs(E_cl_0p1 - E_cl_00):.2e}")
        
        # Hessian determinant
        hessian_det = D2E_DTHETA2 * D2E_DPHI2 - D2E_DPHI_DTHETA**2
        print(f"\n=== Hessian 계산 ===")
        print(f"Hessian determinant = {D2E_DTHETA2:.6e} * {D2E_DPHI2:.6e} - {D2E_DPHI_DTHETA}² = {hessian_det:.6e}")
        
        if hessian_det < 0: 
            print(f"[Need to check]: Hessian determinant is negative = {hessian_det}")
            pseudo_gap = 0
        elif hessian_det == 0:
            print("[Note]: Exact goldstone mode (det = 0)")
            pseudo_gap = 0
        else: 
            spin = 1/2
            pseudo_gap = np.sqrt(hessian_det) * spin
            print(f"Pseudo gap: {pseudo_gap:.6e}")
            
        return pseudo_gap


class Test:
    def __init__(self, config, 
                 phase_name=None, 
                 angles_setting=None, 
                 Temperature=0, 
                 show = False,
                 N=20):
        """Type I 시스템의 pseudo gap 테스트"""
        self.config = config 
        self.Temperature = Temperature
        self.N = N
        print(f"Initial config: {self.config}")
        
        self.nbcp = FIND_TLAF_PHASES(config=self.config)
        self.nbcp_spin_sys_data = self._get_nbcp_phase(phase_name=phase_name, 
                                                       angles_setting=angles_setting)
        
        if show:
            visualizer = SpinVisualizer()

            
            fig, (ax_xy, ax_angle) = visualizer.plot_system(self.nbcp_spin_sys_data)
            fig.suptitle(f"NBCP spin Configuration", fontsize=16, y=1.05)
            plt.show()
        
        # Compute_Pseudo_Gap 객체 생성
        self.cpg = Compute_Pseudo_Gap(self.nbcp_spin_sys_data, Temperature = self.Temperature)
        
    def calculate_pseudo_gap(self, use_five_point=False):
        """pseudo gap 계산 실행"""
        pseudo_gap = self.cpg.calculate_pseudo_gap(N=self.N, use_five_point=use_five_point)
        print(f"pseudo_gap = {pseudo_gap}")
        return pseudo_gap
        
    def _get_nbcp_phase(self, phase_name=None, angles_setting=None):
        """NBCP phase 데이터 생성"""
        if phase_name is None:
            nbcp_phase = self.nbcp.find_tlaf_phase(opt_method="classical",
                                                   angles_setting=angles_setting, 
                                                   config=self.config, 
                                                   full_range_search = False,
                                                   num_search = 6)
        else:
            self.nbcp.nbcp_unit_cells.update_config(self.config)

            # Phase별 데이터 함수 매핑
            phase_functions = {
                "One MSL": self.nbcp.nbcp_unit_cells.spin_system_data_one_msl,
                "Two MSL": self.nbcp.nbcp_unit_cells.spin_system_data_two_msl,
                "Three MSL": self.nbcp.nbcp_unit_cells.spin_system_data_three_msl,
                "Four MSL": self.nbcp.nbcp_unit_cells.spin_system_data_four_msl
            }
            
            if phase_name not in phase_functions:
                raise ValueError(f"Invalid phase name: {phase_name}. "
                               f"Available: {list(phase_functions.keys())}")
            
            data_func = phase_functions[phase_name]
            spin_sys_data = data_func()
            
            # 각도 설정 처리
            if isinstance(angles_setting, dict) and phase_name in angles_setting:
                angle_setting = angles_setting[phase_name]
                print(f"Using angles for phase: {phase_name}, angles={angle_setting}")
            elif angles_setting is None:
                angle_setting = [None, None] * len(spin_sys_data["Spin info"])
            else:
                raise ValueError(f"Invalid angles setting for phase: {phase_name}.")
            
            # 에너지 함수 및 최적화
            cef = Create_Energy_Function(spin_sys_data, N=10, update_args=False)
            optimizer = SpinSystemOptimizer()
            
            opt_result_data, cls_result_data = optimizer.find_minimum(
                cef, 
                opt_method="classical",
                angle_setting=angle_setting,
                full_range_search = False, # range search for MAGSWT
                num_search = 6    # number of search for MAGSWT
                )
            
            nbcp_phase = data_func(angles=opt_result_data["angles"])
        
        return nbcp_phase


# 실행 부분
if __name__ == "__main__":
    
    """
    J_PD, J_Gamma 모두 양수인 경우,
    - 대칭적인 np.pi * (1/6, 2/6, .. )과 같은 점에서 최소가 안생길 수 있음
    - 이 경우, num_search를 키워야 정확도가 늘어남. 
    - 오차는 대략 O(1/n)
    
    @@@ Quantum Energy 에 대한 phi 미분이 불안정한 경향이 있음. 
    - 이 때 예측되는 미분값은 대략 1e-9 ~ 1e-12 크기로 단순 수치계산적 문제일 수 있음
    
    이 경우 아래의 방법을 시도
    -> DEFAULT_D_PHI  변화
    -> N (Brillouin zone point) 변화
    -> 미분 공식 >> 5점 공식
    
    
    @@@ 디버깅을 하여도 음의 Hessian determinant가 해결이 안되는 경우 @@@
    - 함수가 최솟(또는 극소값)이므로 이계도함수는 양수여야함
    - 일반적으로 해당 경우 고전 에너지의 2계 미분값은 수치적으로도 안정적이게 구해짐. 
    - 하지만, 양자 에너지의 경우 boson Hamiltinonian의 positive definiteness 및 고윳값 정확도 등의 이유로 상대적으로 불안정함.
    - 따라서 만일 디버깅(d_phi 상수 변화)을 함에도 해결이 되지 않는 경우,
      2_U_symmetry_YV.py의 phi 방향 회전 코드를 바탕으로 최소에너지 점 근처에서 양자 에너지의 2계 미분을 함수 근사등을 통해서 높은 정확도로 구한 뒤 
      곱하는 방식도 고려해봄직함.
    """

    nbcp_config = {
        "S": 1/2,
        "Jxy": 0.076,
        "Jz": 0.125,
        "JGamma": 0.02,
        "JPD": 0.000,
        "KPD": 0.00,
        "KGamma": 0.00,
        "Kxy": 0.0,
        "Kz": 0.0,
        "h": (0.0, 0.0, 0.05),
    }
    
    # 설정
    phase_name = "Three MSL"  # Fix magnetic sublattice
    Temperature = 0
    
    num_pts = 40
    
    # 위상에 따른 각도 설정
    phi = 0.0
    angles_setting = {"One MSL": (None, phi),
                      "Two MSL": (None, phi, None, phi),
                      "Three MSL": (None, phi, None, phi, None, phi),
                      "Four MSL": (None, None, None, None, None, None, None, None),}
    
    # Test 객체 생성 및 계산 실행
    try:
        test = Test(config=nbcp_config,
                   phase_name=phase_name, 
                   angles_setting=angles_setting, 
                   Temperature = Temperature, 
                   show= False,
                   N= num_pts)
        
        # pseudo gap 계산 (2점 공식)
        result_2pt = test.calculate_pseudo_gap(use_five_point=False)
        print(f"2-point formula result: {result_2pt}")
        
        print("\n" + "="*50)
        print("Using 5-point formula for comparison:")
        
        # pseudo gap 계산 (5점 공식)
        result_5pt = test.calculate_pseudo_gap(use_five_point=True)
        print(f"5-point formula result: {result_5pt}")
        print(f"Difference: {abs(result_5pt - result_2pt):.2e}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        raise
    