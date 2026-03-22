import numpy as np
from typing import Union, Optional, Dict, Tuple, List, Any, Literal
from matplotlib import pyplot as plt
from tqdm import tqdm  # 진행 표시줄


from modules.Tools.magnon_kernel import compute_static_magnon_kernel, compute_real_time_kernel, compute_lorentzian_kernel, compute_spectral_kernel
from modules.Plotters.momentum_space_plotter import MomentumSpacePlotter
from modules.Tools.brillouin_zone_tools import BZ_tools

"""
LSWT_CORR: 선형 스핀파 이론 (LSWT)에 기반한 상관관계 계산 모듈

이 모듈은 k-공간 및 실제 공간에서의 스핀 상관관계 함수를 계산하는 기능을 제공합니다.

공통 매개변수:
--------------
k_data : Dict
    k 포인트 데이터 (키: k 벡터(튜플 또는 배열), 값: 고유값/고유벡터 정보)
Temperature : float, default=DEFAULT_TEMPERATURE
    온도 (Kelvin)
omega : float or None
    주파수 (None이면 정적 구조 인자를 계산)
eta : float, default=DEFAULT_ETA
    감쇠 계수
time : float, default=DEFAULT_TIME
    시간 (실시간 상관관계 계산용)
rvec : np.ndarray
    실제 공간 위치 벡터 (실제 공간 계산용)

참고 사항:
---------
- 각 k-point에 대한 계산은 독립적으로 수행됩니다.
- 매그논 커널 계산은 magnon_kernel 모듈의 함수를 사용합니다.
- 결과는 국소(로컬) 프레임에서 계산됩니다.
"""

DEFAULT_TOLERANCE = 1e-8

Mat_C = np.array([[ 1/np.sqrt(2),    1/np.sqrt(2), 0],      # (S+ + S-)/2   ->  x
                  [1j/np.sqrt(2), - 1j/np.sqrt(2), 0],      # (S+ - S-)/2j  ->  y
                  [     0       ,        0       , 1]])     # Sz            ->  z


# 기본 파라미터 상수 (magnon_kernel 모듈과 일치)
DEFAULT_TEMPERATURE = 0       # 기본 온도 (K)
DEFAULT_TIME = 0              # 기본 시간 (무차원)
DEFAULT_OMEGA = 0             # 기본 주파수 (eV)
DEFAULT_ETA = 1e-3            # 기본 감쇠 계수 (eV)

DEFAULT_DELTA_PEAK = 100        # To approximate delta function, we introduce the weight.


class LSWT_CORR:
    def __init__(self, lswt_obj, msl_boson_number: Optional[dict] = None):
        """
        LSWT 객체를 참조하여 초기화합니다.
        
        Args:
            lswt_obj: 선형 스핀파 이론 객체 참조
        """
        # Store reference to parent LSWT object
        self.lswt = lswt_obj
        self.spin_info = self.lswt.spin_info
        self.lattice_vectors = self.lswt.lattice_bz_settings[0]
        self.Ns = lswt_obj.Ns
        
        if hasattr(self, "msl_average_boson_number") or hasattr(lswt_obj, "msl_average_boson_number"):
            self.msl_boson_number = lswt_obj.msl_average_boson_number
        else: 
            self.msl_boson_number = msl_boson_number
            
        
        self._initiation_lswt_corr()
        

    def _initiation_lswt_corr(self, cond_low_boson_number = True):
        # define 
        self.rmat_dict = {}         # rmat
        self.sublattices_dict = {}  # "A" : 1, "B" : 2, ...
        self.sublattice_pos = []    # (pos1, pos2)
        self.sqrt_spins = []        # 
        self.msl_spin_moment = np.zeros(self.Ns)
        
        low_boson_number = True
        
        if cond_low_boson_number:
            for name_sl, boson_num in self.msl_boson_number.items():
                spin = self.spin_info[name_sl]["Spin"]
                low_boson_number = (boson_num < spin) and low_boson_number
        
        ## replace classical spin
        classical_spin = cond_low_boson_number and not low_boson_number
        
        print("="*30)
        for j, (name_sl, sl_dict) in enumerate(self.spin_info.items()):
            
            self.sublattices_dict[name_sl] = j
            theta, phi = sl_dict["Angles"]
            self.rmat_dict[name_sl] = self._classical_spin_rotation_matrix(theta, phi)
            # preparation Sk matrix
            pos  = sl_dict["Position"]
            spin = sl_dict["Spin"]
            boson_num = self.msl_boson_number[name_sl]
            
            print(f"{name_sl}-msl >>> pos = ({float(pos[0]):.2f}, {float(pos[1]):.2f})  \t spin: {spin}")
            
            self.sublattice_pos.append(pos)
            self.sqrt_spins.append(spin)
            self.msl_spin_moment[j] = spin if classical_spin else spin - boson_num 
            
        print("="*30)
    
        self.sublattice_pos = np.array(self.sublattice_pos).T
        self.sqrt_spins     = np.sqrt(np.array(self.sqrt_spins))
        
        ## 
        a1, a2 = self.lattice_vectors 

        nearest_lattices = BZ_tools.get_nearest_lattices(a1, a2)
        
        if len(nearest_lattices) == 4:
            self.nearest_lattices = nearest_lattices[:2]
        else:
            self.nearest_lattices = nearest_lattices[::2]
        

    @staticmethod
    def delta_func_weight(q: Union[np.ndarray, Tuple], nearest_vectors,  delta_peak = 50, eps = 1e-10):
        delta_k = 1

        for a in nearest_vectors:
            q_dot_a = np.dot(q, a)
            numer = np.sin(delta_peak * q_dot_a / 2)
            denom = np.sin(q_dot_a / 2)
            reg_term = 1 if abs(denom) < eps else numer / ( delta_peak * denom )
            delta_k *= reg_term

        return delta_k    
    
    
    @staticmethod
    def _classical_spin_rotation_matrix(pol_ang: float, azm_ang: float) -> np.ndarray:
        """고전적 스핀 방향에 대한 회전 행렬을 계산합니다."""
        Rot_spin = np.array([[   np.cos(pol_ang)*np.cos(azm_ang), - np.sin(azm_ang),    np.sin(pol_ang)*np.cos(azm_ang)],
                             [   np.cos(pol_ang)*np.sin(azm_ang),   np.cos(azm_ang),    np.sin(pol_ang)*np.sin(azm_ang)],
                             [ - np.sin(pol_ang),                         0,            np.cos(pol_ang)]])
        return Rot_spin
    
    @staticmethod
    def search_key_matrices_dict(matrice, key_to_find, keys):
        try:
            return matrice[key_to_find]
        
        except KeyError:
            
            distances = np.linalg.norm(keys - np.array(key_to_find), axis=1)
            min_idx = np.argmin(distances)
                
            if distances[min_idx] < DEFAULT_TOLERANCE:
                nearest_key = tuple(keys[min_idx])
                return matrice[nearest_key]
            else:
                raise KeyError(f"k-point {key_to_find} not found")
    
    def _get_sublattice_phase_factor(self, k_vector: Union[np.ndarray, Tuple]) -> np.ndarray:
        """
        Generate sublattice phase factor for a given k vector
        """
        k_vector = np.array(k_vector)   # make sure it np.ndarray 
        mikd     = -1j * (k_vector @ self.sublattice_pos)
        return np.exp(mikd) 
    
    def _get_spin_sublattice_matrix(self, k_vector: Union[np.ndarray, Tuple]) -> np.ndarray:
        """
        Generate sublattice matrix S_k for a given k vector
        """
        exp_mikd = self._get_sublattice_phase_factor(k_vector)
        sk = exp_mikd * self.sqrt_spins
        S_k = np.diag( np.hstack([ sk, sk]))
        return S_k, exp_mikd


    def _get_U_V_alpha_matrix(self, coordinate_type = "cartesian"):
        
        if coordinate_type in ("cartesian", "Cartesian", "xyz"):
            alphas = ["x", "y", "z"]
            betas  = alphas 
            get_U_sublattice = lambda mat:  mat @ Mat_C # sublattice 별
            
        
        elif coordinate_type in ("complex", "ladder", "pm0", "+-0"):
            alphas = ["-", "+", "0"]
            betas  = ["+", "-", "0"]
            get_U_sublattice = lambda mat: Mat_C.T.conj() @ mat @ Mat_C
        
        else: 
            raise ValueError("choose coordinate type either cartesian or ladder")
        
        U_alphas_dict = {alpha: [] for alpha in alphas}
        V_alphas_dict = {alpha: [] for alpha in alphas}
        for key, rmat in self.rmat_dict.items():
            sublattice_u_matrix = get_U_sublattice(rmat)
            U_alphas_dict[alphas[0]].append(sublattice_u_matrix[0, 0])
            U_alphas_dict[alphas[1]].append(sublattice_u_matrix[1, 0])
            U_alphas_dict[alphas[2]].append(sublattice_u_matrix[2, 0])
            
            V_alphas_dict[alphas[0]].append(sublattice_u_matrix[0, 2])
            V_alphas_dict[alphas[1]].append(sublattice_u_matrix[1, 2])
            V_alphas_dict[alphas[2]].append(sublattice_u_matrix[2, 2])

        for alpha in alphas:
            u_alpha = np.array(U_alphas_dict[alpha])
            U_alphas_dict[alpha] = np.hstack([u_alpha.conj(), u_alpha])
            V_alphas_dict[alpha] = np.array(V_alphas_dict[alpha])
        
        return U_alphas_dict, V_alphas_dict, alphas, betas


    def compute_real_space_spin_corr_function_in_local_frame_at_rvec(self, 
                                                            k_data: Dict, 
                                                            rvec: Union[np.ndarray, Tuple], 
                                                            Temperature: float = DEFAULT_TEMPERATURE, 
                                                            time: float = DEFAULT_TIME) -> np.ndarray:
        """실제 공간에서의 스핀 상관 함수를 계산합니다."""
        rvec = np.array(rvec)
        
        RS_spin_spin_corr_func = np.zeros((self.Ns * 2, self.Ns * 2), dtype=complex)
        
        for k_key, value in k_data.items():
            _, Eigen_data, _ = value
            
            Eval, Evec = Eigen_data
            
            magnon_kernel = compute_real_time_kernel(Eval, time, Temperature)
            
            bosonic_corr_mat = Evec @ np.diag(magnon_kernel) @ Evec.T.conj()    # HP boson correlatin function
            S_k, _ = self._get_spin_sublattice_matrix(k_key)                         # create sublattice phase matrix
            spin_corr_mat = S_k.conj() @ bosonic_corr_mat @ S_k                 # from boson to "real" spin
            
            # 운동량에서 실제 공간으로 변환
            k_key_array = np.array(k_key)
            mikr = -1j * k_key_array @ rvec
            RS_spin_spin_corr_func += spin_corr_mat * np.exp(mikr)
        
        return RS_spin_spin_corr_func/len(k_data)
    

    
    def compute_TNT_for_structure(self, 
                                  k_data: Dict, 
                                  Temperature: float = DEFAULT_TEMPERATURE, 
                                  omega: Optional[float] = None, 
                                  eta: float = DEFAULT_ETA) -> Dict:
        
        TNT = {}    # TNT = bosonic correlation matrix
        
        for k_key, value in k_data.items():
            _, Eigen_data, _ = value
            Eval, Evec = Eigen_data
            
            if omega is None:   # compute magnon kernel
                magnon_kernel = compute_static_magnon_kernel(Eval, Temperature=Temperature, Ns=self.Ns)
            elif isinstance(omega, (float, int, np.number)):
                magnon_kernel = compute_lorentzian_kernel(Eval, omega, eta=eta, Temperature=Temperature, Ns=self.Ns)

            TNT[k_key] = Evec @ np.diag(magnon_kernel) @ Evec.T.conj()    # HP boson correlatin function
        
        return TNT
    
    
    def compute_TNT_for_spectral(self, 
                                 k_data: Dict, 
                                 Temperature: float = DEFAULT_TEMPERATURE,
                                 omega: Optional[float] = None, 
                                 eta: float = DEFAULT_ETA) -> Dict:
        
        TNT = {}    # TNT = bosonic correlation matrix
        
        for k_key, value in k_data.items():
            _, Eigen_data, _ = value
            Eval, Evec = Eigen_data
            magnon_kernel = compute_spectral_kernel(Eval, omega, eta, Temperature, Ns=self.Ns)
            TNT[k_key] = Evec @ np.diag(magnon_kernel) @ Evec.T.conj()    # HP boson correlatin function
        
        return TNT


    def calculate_spin_corr_mat_in_local(self,
                                         TNT: Dict[Tuple[float, float], np.ndarray],
                                         k_points: Union[ np.ndarray, None ], 
                                         classical_contribution = True,
                                         delta_peak = DEFAULT_DELTA_PEAK
                                         ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        
        """Calculate spin-spin correlation function in local magnetization frame"""

        num_k = len(k_points)

        ## For quantum contribution
        keys_bosonic_corr_mat = np.array(list(TNT.keys()))
        quantum_spin_corr_mat = np.zeros((num_k, self.Ns*2, self.Ns*2), dtype=complex)

        ## For classical contribution
        delta_func = np.zeros(num_k) if classical_contribution else None
        sublattice_phase = np.zeros((self.Ns, num_k), dtype = complex) if classical_contribution else None

        for j, kpt in enumerate(k_points):
            # classical ordering k-vector --> delta_k = 1
            S_k, sublattice_phase_factor_k = self._get_spin_sublattice_matrix(kpt)

            bosonic_corr_mat = self.search_key_matrices_dict(TNT, key_to_find=tuple(kpt), keys = keys_bosonic_corr_mat)
            
            quantum_spin_corr_mat[j] = S_k @ bosonic_corr_mat @ S_k.conj()
            
            if classical_contribution:
                delta_k = self.delta_func_weight(kpt, self.nearest_lattices, 
                                                 delta_peak = delta_peak)
                delta_func[j] = delta_k
                sublattice_phase[:, j] = sublattice_phase_factor_k 

        return quantum_spin_corr_mat, delta_func, sublattice_phase
    
    
    
    def calculate_spin_corr_mat(self,
                                TNT: Dict[Tuple[float, float], np.ndarray],
                                k_points: Optional[Union[np.ndarray, Tuple, List]] = None,
                                coordinate_type: str = "cartesian",
                                sublattice: Optional[Tuple] = None,
                                classical_contribution = True, 
                                ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        
        """k-포인트별 상관 행렬 계산, tuple 매핑 후 거리 계산 대체."""
        
        k_points = k_points if k_points is not None else np.array(list(TNT.keys()))

        num_k = len(k_points)
        U_dict, V_dict, alphas, betas = self._get_U_V_alpha_matrix(coordinate_type=coordinate_type)
        local_spin_corr_mat, delta_func, sublattice_phase = self.calculate_spin_corr_mat_in_local(TNT, 
                                                                                                  k_points = k_points,
                                                                                                  classical_contribution = classical_contribution,
                                                                                                  delta_peak = np.minimum(num_k, 10))
        corr_mat = {}
        
        for alpha in alphas:
            for beta in betas:
                U_alpha = np.diag(U_dict[alpha])
                U_beta  = np.diag(U_dict[beta])
                SaSb_k = U_alpha @ local_spin_corr_mat @ U_beta.T.conj()
                
                corr_mat[alpha+beta] = self.get_quantum_spin_corr_mat(SaSb_k, sublattice)
                
                if classical_contribution:
                    V_alpha = V_dict[alpha]
                    V_beta  = V_dict[beta]
                    corr_mat[alpha+beta] +=  self.get_classical_spin_corr_mat(delta_func, sublattice_phase, sublattice, V_alpha, V_beta)
    
        
        total_corr = np.zeros(num_k, dtype = complex)
        if coordinate_type == 'cartesian':
            total_corr = corr_mat['xx'] + corr_mat['yy'] + corr_mat['zz']
        else:
            total_corr = (corr_mat['+-'] + corr_mat['-+']) / 2 + corr_mat['00']
            
        if total_corr is None or not np.all(np.isfinite(total_corr)):
            raise ValueError("spin_corr_total contains invalid values or is None")
        
        return corr_mat, total_corr,  k_points
    
    def get_quantum_spin_corr_mat(self, SaSb_k, sublattice):
        # 부격자 인덱스 찾기
        mu_idx, nu_idx = None, None
        if sublattice is not None:
            mu, nu = sublattice
            mu_idx = self.sublattices_dict[mu]
            nu_idx = self.sublattices_dict[nu]
            
            result = (SaSb_k[:, mu_idx, nu_idx] + SaSb_k[:, mu_idx + self.Ns, nu_idx] 
                    + SaSb_k[:, mu_idx, nu_idx + self.Ns] + SaSb_k[:, mu_idx + self.Ns, nu_idx])
            print(f"Quantum corr for sublattice {sublattice}: {result.shape}")
            return result
        else:
            result = np.sum(SaSb_k, axis=(1, 2))  # (num_k,) shape 유지
            print(f"Quantum corr (no sublattice): {result.shape}")
            return result
        

    def get_classical_spin_corr_mat(self, delta_func, sublattice_phase, sublattice, V_alpha, V_beta):
        """
        고전적 스핀 상관 행렬 계산 함수
        
        Parameters:
        -----------
        delta_func : numpy.ndarray
            델타 함수 가중치
        sublattice_phase : numpy.ndarray
            부격자 위상
        sublattice : Optional[Tuple[str, str]]
            부격자 쌍 (mu, nu) 또는 None
        V_alpha : List or numpy.ndarray
            알파 요소의 V 벡터
        V_beta : List or numpy.ndarray
            베타 요소의 V 벡터
            
        Returns:
        --------
        numpy.ndarray
            고전적 스핀 상관 행렬
        """
        
        # 부격자 인덱스 찾기
        mu_idx, nu_idx = None, None
        if sublattice is not None:
            mu, nu = sublattice
            mu_idx = self.sublattices_dict[mu]
            nu_idx = self.sublattices_dict[nu]
        
        classical_spin_corr_mu_nu = np.zeros_like(delta_func, dtype = complex )
        
        for i, v_a_mu in enumerate(V_alpha):
            # 특정 부격자 쌍만 계산하는 경우 조건 확인
            if sublattice is not None and i != mu_idx:
                continue
            
            lswt_spin_mu = self.msl_spin_moment[i]
            phase_mu = sublattice_phase[i]
            
            for j, v_b_nu in enumerate(V_beta):
                # 특정 부격자 쌍만 계산하는 경우 조건 확인
                if sublattice is not None and j != nu_idx:
                    continue
                
                lswt_spin_nu = self.msl_spin_moment[j]
                phase_nu = (sublattice_phase[j]).conj()
                
                exp_phase = phase_mu * phase_nu     # phase factor, np.ndarray len(exp_phase) = len(delta_func)
                spin_ab_product = v_a_mu * v_b_nu * lswt_spin_mu * lswt_spin_nu # float
                
                classical_spin_corr_mu_nu += spin_ab_product * exp_phase
                    
        return classical_spin_corr_mu_nu * delta_func

    @staticmethod
    def _neutron_scattering_factor(alpha, beta, kx_arr, ky_arr, coordinate_type):
        
        def q_alpha(comp):
            if comp == "x":
                return kx_arr
            elif comp == "y":
                return ky_arr
            elif comp == "+":
                return (kx_arr + 1j * ky_arr)/np.sqrt(2)
            elif comp == "-":
                return (kx_arr - 1j * ky_arr)/np.sqrt(2)
            else:
                return 0
        
        delta_ab = 1 if alpha == beta else 0
    
        q_norm = np.sqrt(kx_arr ** 2 + ky_arr ** 2)
        q_norm[np.where(q_norm == 0)] = 1
        
        if coordinate_type in ('cartesian', "Cartesian"):
            q_a = q_alpha(alpha)
            q_b = q_alpha(beta)                    
        elif coordinate_type in ("ladder", "+-0"):
            q_a = q_alpha(alpha)
            q_b = q_alpha(beta)                    

        q_ab = (q_a * q_b) / q_norm 
        
        return delta_ab - q_ab


    def compute_real_space_correlations(self, 
                                        k_data: dict,
                                        angle_direction: float, 
                                        distance_range: Tuple = (0, 1, 0.1), 
                                        time: float = DEFAULT_TIME, 
                                        Temperature: float = DEFAULT_TEMPERATURE):
        """
        Compute real-space spin-spin correlation values along a given direction.
        """
        direction = np.array((np.cos(angle_direction), np.sin(angle_direction)), dtype=float)

        r_start, r_end, r_step = distance_range
        r_values = np.linspace(r_start, r_end, int((r_end - r_start)/r_step) + 1)
        rvec_list = [direction * r for r in r_values]

        RS_corr_in_local = np.empty((len(r_values), 2*self.Ns, 2*self.Ns), dtype=complex)

        print("Computing real-space spin-spin correlation along direction:", direction)

        for j, rvec in enumerate(tqdm(rvec_list, desc="Calculating correlations")):
            RS_corr_in_local[j] = self.compute_real_space_spin_corr_function_in_local_frame_at_rvec(
                k_data=k_data, rvec=rvec, Temperature=Temperature, time=time)

        return r_values, RS_corr_in_local


    def plot_real_space_correlation(self, 
                                    r_values: np.ndarray, 
                                    RS_corr_in_local: np.ndarray,
                                    comb_to_plot: List[Tuple[int, int]],
                                    component: Literal['real', 'imag', 'abs'] = 'abs',
                                    log_x: bool = False,
                                    log_y: bool = False):
        """
        Plot real-space correlation based on precomputed data.

        Parameters:
            r_values: 1D array of distances
            RS_corr_in_local: correlation values (len(r), 2Ns, 2Ns)
            comb_to_plot: list of (i,j) index pairs to plot
            component: 'real', 'imag', or 'abs'
            log_x: whether to use log scale for x-axis (distance)
            log_y: whether to use log scale for y-axis (correlation)
        """
        styles = ['-', '--', '-.', ':']
        alphas = [0.9, 0.7, 0.5, 0.3]
        widths = [1, 2, 4, 7]

        for idx, (i, j) in enumerate(comb_to_plot):
            idx_i_pm = '-' if i // self.Ns == 0 else '+'
            idx_j_pm = '+' if j // self.Ns == 0 else '-'
            idx_i = f'{chr(65 + i % self.Ns)}{idx_i_pm}'
            idx_j = f'{chr(65 + j % self.Ns)}{idx_j_pm}'

            y_data = RS_corr_in_local[:, i, j]
            if component == 'real':
                y_plot = y_data.real
            elif component == 'imag':
                y_plot = y_data.imag
            else:
                y_plot = np.abs(y_data)

            plt.plot(r_values, y_plot,
                    label=f'Corr: ⟨S{idx_i}·S{idx_j}⟩',
                    linestyle = styles[idx % len(styles)],
                    linewidth = widths[idx % len(widths)],
                    alpha = alphas[idx % len(alphas)])

        plt.xlabel('|r| (distance along direction)')
        ylabel_map = {'real': 'Re ⟨Sᵢ·Sⱼ⟩', 'imag': 'Im ⟨Sᵢ·Sⱼ⟩', 'abs': '|⟨Sᵢ·Sⱼ⟩|'}
        plt.ylabel(ylabel_map[component])

        if log_x:
            plt.xscale('log')
        if log_y:
            plt.yscale('log')

        plt.title('Spin Correlation vs Distance')
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.tight_layout()
        plt.show()




    def plot_correlation_FBZ(self,
                            k_data: Dict,
                            k_points = None,
                            func_type: str = "structure factor",
                            coordinate_type: str = 'cartesian',
                            classical_contribution = True,
                            sublattice: Optional[Tuple] = None,
                            Temperature = 0,
                            omega = None,
                            plotter_obj: Optional[object] = None,
                            config: Optional[Dict[str, Any]] = None) -> plt.Figure:
        config = config or {}
        config = config.copy()

        # 제목 설정
        if func_type.lower() in ("structure factor", "strcutre"):
            if omega is None:
                title = "Static Structure Factor"
            else:
                title = f"Structure Factor (ω = {omega})"
        elif func_type.lower() in ("spectral function", "spectral"):
            title = f"Spectral Function (ω = {omega})"
        else:
            title = f"{func_type} (ω = {omega})"  # 기타 func_type 대비

        # sublattice가 있으면 추가
        if sublattice is not None:
            title += f", Sublattice: (mu, nu) = {sublattice}"

        config['title'] = title

        # bz_data 안전 처리
        bz_data = getattr(self.lswt, 'bz_data', {"BZ_corners": [], "high_symmetry_points": {}})
        plotter = plotter_obj if plotter_obj is not None else MomentumSpacePlotter(self.lswt, bz_data)
        
        if func_type.lower() in ("structure factor", "strcutre"):
            TNT = self.compute_TNT_for_structure(k_data, Temperature = Temperature, omega = omega)
        elif func_type.lower() in ("spectral function", "spectral"):
            TNT = self.compute_TNT_for_spectral(k_data,  Temperature = Temperature, omega = omega)
        
        # 데이터 준비
        spin_corr_comp, spin_corr_total, k_points = self.calculate_spin_corr_mat(TNT, 
                                                                                 k_points = k_points,
                                                                                 coordinate_type = coordinate_type,
                                                                                 sublattice = sublattice,
                                                                                 classical_contribution = classical_contribution, )
        
        for key, value in list(spin_corr_comp.items()):
            if func_type.lower() in ("structure factor", "strcutre"):
                spin_corr_comp[key] = np.real(value)
            elif func_type.lower() in ("spectral function", "spectral"):
                spin_corr_comp[key] = -np.imag(value)

        
        if func_type.lower() in ("structure factor", "strcutre"):
            spin_corr_total =   np.real(spin_corr_total)
        elif func_type.lower() in ("spectral function", "spectral"):
            spin_corr_total = - np.imag(spin_corr_total)
        
        kx, ky = k_points.T
        
        # MomentumSpacePlotter의 플롯 메서드 호출
        return plotter.plot_all_spin_corr_func_in_real_space(kx, ky, spin_corr_comp, spin_corr_total, func_type, config)
    



