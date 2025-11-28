import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from typing import Dict, Tuple, List


from modules.Tools.brillouin_zone import Brillouin_Zone
from modules.Tools.magnon_kernel import compute_static_magnon_kernel

from modules.LinearSpinWaveTheory.linear_spin_wave_theory import LSWT
from modules.LinearSpinWaveTheory.lswt_thermodynamics import LSWT_THER
from modules.LinearSpinWaveTheory.lswt_correlation import LSWT_CORR
from modules.LinearSpinWaveTheory.lswt_Hamiltonian import LSWT_HAMILTONIAN
from modules.LinearSpinWaveTheory.lswt_topology import compute_Berry_curvature


def sig_digit(x):
    x = np.asarray(x)
    return np.where(x==0, 0, 10**np.floor(np.log10(np.abs(x))))

class Band_Plotter:
    def __init__(self, spin_system_data: Dict, bz_type = "Hex_60") -> None:
        """Initialize BandPlotter with spin system data."""
        self.spin_system_data = spin_system_data
        self.Ns = len(spin_system_data["Spin info"])
        self.J_mat = np.diag(np.hstack([np.ones((self.Ns)), - np.ones((self.Ns))]))
        
        # Initialize LSWT and gadjets for physical quantities
        self.lswt = LSWT(spin_system_data)
        self.ham = LSWT_HAMILTONIAN(self.lswt.spin_info, self.lswt.couplings)
        self.ther = LSWT_THER(self.lswt)
        
        self.BZ = Brillouin_Zone(self.spin_system_data["Lattice/BZ setting"], bz_type = bz_type)
        
        bz_data ,_ = self.BZ.get_bz_data()
            
        self.high_symmetry_points = bz_data["high_symmetry_points"]
        self.bz_boundary = bz_data["BZ_corners"]
        self.band_paths = bz_data["band_paths"]


    def prepare_HSP_path(self, 
                        reg_type=1, 
                        threshold=1e-8,
                        compute_berry_curvature = True, 
                        mag = 100) -> None:
        self.multiple_paths = list(self.band_paths.values())
        self.k_points = {}
        self.path_segments = {}
        self.numerical_bands = {}
        self.path_boson_numbers = {}
        self.total_boson_numbers = {}
        
        self.compute_berry_curvature = compute_berry_curvature
        if compute_berry_curvature:
            self.compute_berry_curvature = True
            self.berry_curvatures = {}

        for path_name, band_path in self.band_paths.items():
            print(f"=== {path_name} band path ===")
            k_points, path_interval = self.BZ.generate_kpath(band_path, 
                                                             HSP=self.high_symmetry_points, 
                                                             mag = mag)
            K_data, _ = self.ham.solve_k_Hamiltonian(k_points,
                                                    Berry_curvature = self.compute_berry_curvature,
                                                    regularization  = reg_type,
                                                    threshold = threshold)

            num_k = len(k_points)
            bands = np.zeros((num_k, self.Ns))
            sublattice_numbers = np.zeros((num_k, self.Ns))
            total_numbers = np.zeros(num_k)
            berry_curvatures = np.zeros((num_k, self.Ns)) if compute_berry_curvature else None

            for j, (kpt, (H_k_data, Eigen_k_data, Colpa_data)) in enumerate(K_data.items()):
                colpa_success = Colpa_data[0]

                if not colpa_success:
                    print(f"Colpa's method failed at k-point {kpt}.")
                    continue

                evals, evecs = Eigen_k_data
                bands[j] = evals[:self.Ns]

                kernel = compute_static_magnon_kernel(E_list = evals, 
                                                      Temperature = 0, 
                                                      Ns=self.Ns)
                msl_number = np.real(np.diag(evecs @ np.diag(kernel) @ evecs.T.conj())[:self.Ns]) - 1    # commutation
                sublattice_numbers[j] = msl_number
                total_numbers[j] = np.mean(msl_number)

                if compute_berry_curvature:
                    pDHk = H_k_data[1:]
                    Omega_nk, _ = compute_Berry_curvature(eval=evals,
                                                          evec=evecs,
                                                          pDiffHk=pDHk,
                                                          num_sl=self.Ns,
                                                          J_mat=self.J_mat)
                    berry_curvatures[j] = Omega_nk

            self.k_points[path_name] = k_points
            self.path_segments[path_name] = path_interval
            self.numerical_bands[path_name] = bands
            self.path_boson_numbers[path_name] = sublattice_numbers
            self.total_boson_numbers[path_name] = total_numbers
            
            if compute_berry_curvature:
                self.berry_curvatures[path_name] = berry_curvatures

    
        # Define line styles for different paths
    def _get_line_style(self, path_name, alpha = None):
        path_names = list(self.path_segments.keys())
        
        if path_name == path_names[0]:  # First path (usually 'Standard')
            alpha = 1.0 if alpha is None else alpha
            return dict(linestyle='-', linewidth=1.0, alpha = 1.0)
        elif path_name == path_names[1] if len(path_names) > 1 else "":  # Second path (usually 'Rotated')
            alpha = 1.0 if alpha is None else alpha
            return dict(linestyle='--', linewidth=1.0, alpha = 1.0)
        else:  # Other paths
            alpha = 1.0 if alpha is None else alpha
            return dict(linestyle='-', linewidth=2.5, alpha = 0.3)

    
    def plot_band_w_number(self, boson_numbers: list = None, y_min: float = 0.0, y_max: float = None, set_path_name = None) -> None:
        if not self.path_segments:
            raise ValueError("Band data has not been prepared. Run prepare_band first.")

        fig = plt.figure(figsize=(20, 6))  
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 0.8])  

        ax_bz     = fig.add_subplot(gs[0])
        ax_band   = fig.add_subplot(gs[1])
        ax_number = fig.add_subplot(gs[2])

        self._plot_brillouin_zone(ax_bz)
        if self.compute_berry_curvature:
            result = self._plot_band_structure_w_berry_curvature(ax_band, y_min, y_max, set_path_name = set_path_name)
        else:
            result = self._plot_band_structure(ax_band, y_min, y_max)
        
        self._plot_boson_numbers(ax_number, boson_numbers)

        plt.tight_layout(pad=1.5, w_pad=2.5)

        # fig.savefig("band.png", dpi=600)

        plt.show()
        
        
        
        return result


    def _plot_brillouin_zone(self, ax):
        self.BZ.plot_polygon_and_grid( vertices = self.bz_boundary,
                                      HSP = self.high_symmetry_points,
                                      band_path = self.band_paths,
                                      title=f"Brillouin Zone ({self.BZ.lattice_bz_setting[1]})",
                                      ax = ax)
    

    def _plot_band_structure(self, ax, y_min, y_max):
        """Plot the band structure on a given axis."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        path_names = list(self.path_segments.keys())
        first_path = path_names[0]

        # 기준 경로에서 고대칭점 경로와 위치 계산
        path_points = self.multiple_paths[0]
        path_intervals = self.path_segments[first_path]
        cumulative_points = np.cumsum([0] + path_intervals)

        # 고대칭점 레이블과 x축 위치
        labels = [r"$\Gamma$" if p == "Gamma" else p for p in path_points]
        x_ticks = cumulative_points[:len(labels)]

        # 경로마다 밴드 플롯
        for path_name in path_names:
            evals = self.numerical_bands[path_name]
            num_pts = evals.shape[0]
            total_length = sum(self.path_segments[path_name])

            x = np.linspace(0, total_length - 1, num_pts-1)  # k-point 위치

            # 각 밴드마다 플롯
            for i in range(self.Ns):
                ax.plot(x, evals[:-1, i],
                        color=colors[i % len(colors)],
                        label=f"{path_name}" if i == 0 else None,
                        **self._get_line_style(path_name))

        # x축 설정
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(x_ticks[0], x_ticks[-1])

        # y축 설정
        ax.set_ylim(bottom=y_min)
        if y_max is not None:
            ax.set_ylim(top=y_max)

        # 고대칭점 위치에 수직선 추가
        for x in x_ticks:
            ax.axvline(x, color='gray', linestyle='--', alpha=0.5)

        # 라벨 및 제목
        ax.set_title("Band Structure")
        ax.set_xlabel("Wave Vector")
        ax.set_ylabel("Energy (meV)")
        ax.legend(loc='upper right')
        
        return self.numerical_bands

    def _plot_band_structure_w_berry_curvature(self, ax, y_min, y_max, 
                                               set_path_name = None,
                                               intense = True):
        """Plot the band structure on a given axis. If Berry curvature is enabled,
        color each band according to normalized Berry curvature magnitude."""
        
        # cmap = cm.get_cmap('coolwarm')
        cmap = cm.get_cmap('seismic')
        
        norm = Normalize(vmin=-1, vmax=1)


        path_names = list(self.path_segments.keys())
        first_path = path_names[0]
        
        path_points = self.multiple_paths[0]
        path_intervals = self.path_segments[first_path]
        cumulative_points = np.cumsum([0] + path_intervals)

        # 고대칭점 레이블과 x축 위치
        labels = [r"$\Gamma$" if p == "Gamma" else p for p in path_points]
        x_ticks = cumulative_points[:len(labels)]
        
        
        # 모든 경로에 대한 x축 눈금과 라벨 계산
        for i, path_name in enumerate(path_names):
            x_offset = 0
            path_points = self.multiple_paths[i]
            path_intervals = self.path_segments[path_name]
            segment_length = sum(path_intervals)
            
            # 경로 끝점의 x 위치 추가
            x_offset += segment_length
        
        # 각 경로에 대해 밴드 플롯
        x_offset = 0
        for i, path_name in enumerate(path_names):
            if set_path_name is None or path_name == set_path_name:
                print("=" * 20)
                print(f"Plotting {path_name} band structure")
                print("=" * 20)
                evals = self.numerical_bands[path_name]
                num_pts = evals.shape[0]
                segment_length = sum(self.path_segments[path_name])
                x = np.linspace(x_offset, x_offset + segment_length, num_pts)

                berry = self.berry_curvatures[path_name]  # shape: (num_k, Ns)

                for band_idx in range(self.Ns):
                    y_vals = evals[:, band_idx]
                    
                    if np.isnan(y_vals).any():
                        continue

                    color_vals = berry[:, band_idx]
                    if intense:
                        max_val = np.max(np.abs(color_vals))
                        if max_val < 1e-8:
                            color_vals = np.zeros_like(color_vals)
                        elif max_val > 1.5*1e-1:
                            color_vals = color_vals / max_val
                        else:
                            color_vals = color_vals / max_val * sig_digit(max_val)
                    
                    color_array = cmap(norm(color_vals))

                    # 기본 라인 스타일 가져오기
                    base_style = self._get_line_style(path_name, alpha=1).copy()
                    
                    # 밴드 전체를 먼저 엷은 색으로 그려서 기본 골격이 보이도록 함
                    # 기본 스타일에서 alpha와 color만 변경
                    background_style = base_style.copy()
                    background_style['alpha'] = 0.7
                    background_style['color'] = 'gray'
                    ax.plot(x, y_vals, **background_style)
                    
                    # 이제 Berry 곡률로 색칠된 선분 그리기
                    for j in range(len(x) - 1):
                        # 기본 스타일을 유지하되 색상만 변경
                        segment_style = base_style.copy()
                        segment_style['color'] = color_array[j]
                        ax.plot(x[j:j+2], y_vals[j:j+2], **segment_style)


        # 고대칭점 수직선 및 축 설정
        for xtick in x_ticks:
            ax.axvline(xtick, color='gray', linestyle='--', alpha=0.5)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(x_ticks[0], x_ticks[-1])
        ax.set_ylim(bottom=y_min, top=y_max if y_max is not None else None)
        ax.set_title("Band Structure")
        ax.set_xlabel("Wave Vector")
        ax.set_ylabel("Energy (meV)")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Normalized Berry Curvature")

        ax.legend(loc='lower right')
        

        

    def _plot_boson_numbers(self, ax, boson_numbers=None):
        """Plot boson number along the k-path on the given axis."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        path_names = list(self.path_segments.keys())
        default_path = path_names[1] if len(path_names) > 1 else path_names[0]

        # x축용 고대칭점 위치 계산
        path_points = self.multiple_paths[0]
        path_intervals = self.path_segments[default_path]
        cumulative_points = np.cumsum([0] + path_intervals)

        labels = [r"$\Gamma$" if p == "Gamma" else p for p in path_points]
        x_ticks = cumulative_points[:len(labels)]

        # 경로를 따라 실제 boson number 데이터가 있는 경우
        if self.path_boson_numbers and default_path in self.path_boson_numbers:
            sublattice_bosons = self.path_boson_numbers[default_path]
            total_bosons = self.total_boson_numbers[default_path]

            num_pts = sublattice_bosons.shape[0]
            total_length = sum(path_intervals)
            x = np.linspace(0, total_length - 1, num_pts)

            # 총 boson number
            ax.plot(x, total_bosons, 'k-', linewidth=2, label='Total')

            # 각 sublattice별 boson number
            spins = list(self.spin_system_data["Spin info"].keys())
            for i in range(self.Ns):
                ax.plot(x, sublattice_bosons[:, i],
                        color=colors[i % len(colors)],
                        linewidth=2,
                        label=spins[i])

        # 외부에서 boson_numbers 배열이 주어진 경우
        elif boson_numbers is not None:
            spins = list(self.spin_system_data["Spin info"].keys())
            for i in range(self.Ns):
                val = boson_numbers[self.Ns + i] if len(boson_numbers) > self.Ns else boson_numbers[i]
                ax.axhline(y=val, color=colors[i % len(colors)], linewidth=2, label=spins[i])
            if len(boson_numbers) > self.Ns:
                total = np.mean(boson_numbers[self.Ns:self.Ns * 2])
                ax.axhline(y=total, color='k', linewidth=2, label='Total')

        # 시각 설정
        ax.set_title(f"Boson Number along k-{default_path}")
        ax.set_xlabel("Wave Vector")
        ax.set_ylabel(r"Boson Number $\langle n_k \rangle$")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(x_ticks[0], x_ticks[-1])
        ax.legend()

        # 고대칭점 수직선
        for x in x_ticks:
            ax.axvline(x=x, color='gray', linestyle='--', alpha=0.5)





    def prepare_dynamic_function(self, 
                                 path_name: str, 
                                 func_type: str,
                                 msl_boson_number: dict,
                                 omega_max: float = 0.5, 
                                 Temperature = 0, 
                                 num_omega_points = 100, 
                                 decay_factor = 0.1, 
                                 mag = 100,
                                 classical_contribution = False) -> None:
        """
        Prepare dynamic spectral function data along a specific k-path.
        
        Args:
            path_name: Name of the path in the Brillouin zone
            omega_max: Maximum frequency for calculation
            T: Temperature for the calculation
            num_omega_points: Number of points in the omega grid
            decay_factor: Decay factor (broadening parameter)
            
        Returns:
            None (stores results in instance variables)
        """        
        
        # Check if the specified path exists
        if path_name not in self.band_paths:    # self.band_paths is alread defined when initialized
            raise ValueError(f"Path '{path_name}' not found in band_paths")
        else: 
            band_path = self.band_paths[path_name]
            k_points, path_interval = self.BZ.generate_kpath(band_path, 
                                                             self.high_symmetry_points,
                                                             mag= mag)  # 인수 순서 수정
            
            print(f"band_path {path_name}: {band_path}")  
            print(f"Generated k_points length: {len(k_points)}")

        
        self.corr = LSWT_CORR(self.lswt, msl_boson_number = msl_boson_number)

        K_data, _ = self.ham.solve_k_Hamiltonian(k_points, Berry_curvature = False )
        num_k_points = len(k_points)
        
        # Create omega grid
        omegas = np.linspace(0, omega_max, num_omega_points)

        # Arrays to store different components of the spectral function
        # Shape: [omega, k-point]
        Dyn_func_all = np.full((num_omega_points, num_k_points), np.nan, dtype=complex)
        Dyn_func_zz  = np.full((num_omega_points, num_k_points), np.nan, dtype=complex)
        Dyn_func_pm  = np.full((num_omega_points, num_k_points), np.nan, dtype=complex)

        # Compute spectral functions for each omega
        for omega_idx, omega in enumerate(omegas):
            print(f"Computing spectral function for omega = {omega:.4f}")

            if func_type.lower() in ("structure factor", "strcutre"):
                TNT = self.corr.compute_TNT_for_structure(K_data, 
                                                          Temperature = Temperature, 
                                                          omega = omega,
                                                          eta = decay_factor)
            elif func_type.lower() in ("spectral function", "spectral"):
                TNT = self.corr.compute_TNT_for_spectral(K_data,  
                                                         Temperature = Temperature, 
                                                         omega = omega,
                                                         eta = decay_factor)
            
            # 데이터 준비
            corr_comp, corr_total, k_points = self.corr.calculate_spin_corr_mat(TNT, 
                                                                                k_points = k_points,
                                                                                coordinate_type = "cartesian",
                                                                                sublattice = None,
                                                                                classical_contribution = classical_contribution)
        
            for key, value in list(corr_comp.items()):
                if func_type.lower() in ("structure factor", "strcutre"):
                    corr_comp[key] = np.real(value)
                elif func_type.lower() in ("spectral function", "spectral"):
                    corr_comp[key] = -np.imag(value)

        
            if func_type.lower() in ("structure factor", "strcutre"):
                corr_total =   np.real(corr_total)
            elif func_type.lower() in ("spectral function", "spectral"):
                corr_total = - np.imag(corr_total)
        
            Dyn_func_all[omega_idx, :]   = corr_total
            Dyn_func_zz[ omega_idx, :]   = corr_comp["zz"] 
            Dyn_func_pm[ omega_idx, :]   = corr_comp["xx"] +  corr_comp["yy"]
        
        # Store the results in instance variables
        self.dynamic_data = {
            'path_name': path_name,
            'k_points': k_points,
            'omegas': omegas,
            'path_interval': path_interval,
            'path': band_path,
            'Q_all': Dyn_func_all,
            'Q_zz': Dyn_func_zz,
            'Q_pm': Dyn_func_pm,
            'decay_factor': decay_factor,
            'temperature': Temperature
        }
        
        print(f"Calculation complete for path {path_name}")
        return
                

    
    def plot_spectral_functions_1x3(self, cmap='coolwarm_r', vmin=None, vmax=None, log_scale=True, share_colorbar=True, contrast_enhance=0.99):
        """
        세 가지 스펙트럴 함수(A_all, A_zz, A_pm)만 1x3 레이아웃으로 시각화합니다.
        로그 스케일(10^(-x))로 표시하며, 작은 값은 파란색, 큰 값은 빨간색으로 표현합니다.
        
        Args:
            cmap: 컬러맵 (기본값: 'coolwarm_r' - 작은 값은 파란색, 큰 값은 빨간색)
            vmin, vmax: 색상 범위 최소/최대값 (로그 변환 후의 값)
            log_scale: 로그 스케일 사용 여부 (기본값: True)
            share_colorbar: 공유 컬러바 사용 여부
            contrast_enhance: 대비 강화 인자 (0~1 사이의 값, 1에 가까울수록 더 많은 데이터가 clip됨)
            
        Returns:
            Figure 객체
        """
        # 필요한 모듈 import
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        
        # 스펙트럴 함수 데이터 확인
        if not hasattr(self, 'dynamic_data'):
            raise ValueError("Spectral function data not available. Run prepare_dynamic_spectral_function first.")
        
        # 스펙트럴 데이터 추출
        path_name = self.dynamic_data['path_name']
        k_points = self.dynamic_data['k_points']
        omegas = self.dynamic_data['omegas']
        path_interval = self.dynamic_data['path_interval']
        path = self.dynamic_data['path']
        
        # 모든 스펙트럴 함수 데이터 가져오기
        spectral_types = ['Q_all', 'Q_zz', 'Q_pm']
        dynamic_data = {stype: self.dynamic_data[stype] for stype in spectral_types}
        
        # 플롯을 위해 절대값으로 변환
        intensities = {stype: np.abs(data) for stype, data in dynamic_data.items()}
        
        # 최대값 및 데이터 범위 계산
        all_intensities = np.concatenate([intensity.flatten() for intensity in intensities.values()])
        max_intensity = np.max(all_intensities)
        
        # 대비 강화를 위한 임계값 설정 (매우 작은 값 제외)
        # 0이 아닌 값들 중에서 적절한 하한 설정
        non_zero_intensities = all_intensities[all_intensities > 0]
        min_threshold = np.percentile(non_zero_intensities, 2) if len(non_zero_intensities) > 0 else 1e-16
        
        # 대비 강화를 위한 상한 설정 (매우 큰 값 제외)
        max_threshold = np.percentile(all_intensities, contrast_enhance * 100)
        
        # 임계값 처리
        for stype in spectral_types:
            # 매우 작은 값은 최소 임계값으로 설정
            intensities[stype] = np.clip(intensities[stype], min_threshold, max_threshold)
        
        # 로그 스케일 적용 (10^(-x) 형태로 변환)
        if log_scale:
            # log10(intensity) 계산 - 이제 작은 값은 작은 로그 값을 가짐 (파란색)
            intensities = {stype: np.log10(intensity) for stype, intensity in intensities.items()}
        
        # 공유 색상 스케일을 위한 전역 최소/최대값 찾기
        if vmin is None:
            vmin = min(np.min(intensity) for intensity in intensities.values())
        if vmax is None:
            vmax = max(np.max(intensity) for intensity in intensities.values())
        
        # 그림 생성
        fig = plt.figure(figsize=(18, 6))
        
        if share_colorbar:
            # 공유 컬러바가 있는 경우: 3개 플롯 + 컬러바 영역
            gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])
            axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        else:
            # 개별 컬러바가 있는 경우: 3개 플롯만 (컬러바는 make_axes_locatable로 추가)
            gs = fig.add_gridspec(1, 3)
            axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        
        # 고대칭점을 위한 x축 포인트 생성
        cumulative_points = np.cumsum([0] + path_interval)
        x_points = []
        x_labels = []
        
        for i, point in enumerate(path):
            x_points.append(cumulative_points[i])
            label = point
            x_labels.append("$\Gamma$" if label == "Gamma" else f"${label}$")
        
        # pcolormesh를 위한 메시그리드 생성
        num_k = len(k_points)
        X = np.linspace(0, cumulative_points[-1], num_k)
        Y = omegas
        X_mesh, Y_mesh = np.meshgrid(X, Y)
        
        # 각 스펙트럴 함수 플롯
        im_objects = []
        for i, (ax, stype) in enumerate(zip(axes, spectral_types)):
            im = ax.pcolormesh(X_mesh, Y_mesh, intensities[stype], cmap=cmap, 
                            shading='gouraud', vmin=vmin, vmax=vmax)
            im_objects.append(im)
            
            # 고대칭점에 수직선 추가
            for x_pos in x_points:
                ax.axvline(x=x_pos, color='black', linestyle='--', alpha=0.7)
            
            # 축 구성
            ax.set_xticks(x_points)
            ax.set_xticklabels(x_labels)
            ax.set_xlim(0, X[-1])
            ax.set_ylim(Y[0], Y[-1])
            
            # 제목 및 레이블 설정
            spectral_type_label = {
                'Q_all': 'Total (A)',
                'Q_zz': 'z-component (Azz)',
                'Q_pm': 'xx+yy component (Axx+Ayy)'
            }[stype]
            
            ax.set_title(f"{spectral_type_label}")
            ax.set_xlabel("Wave Vector")
            if i == 0:  # 첫 번째 플롯에만 y축 레이블 추가
                ax.set_ylabel("Energy (meV)")
        
        # 공유 컬러바 추가 (요청된 경우)
        if share_colorbar:
            cax = fig.add_subplot(gs[0, 3])
            cbar = fig.colorbar(im_objects[0], cax=cax)
            if log_scale:
                cbar.set_label('$\log_{10}(Intensity)$')
                # 컬러바 눈금에 10^x 형태로 표시
                tick_locations = np.linspace(vmin, vmax, 5)
                cbar.set_ticks(tick_locations)
                cbar.set_ticklabels([f'$10^{{{x:.1f}}}$' for x in tick_locations])
            else:
                cbar.set_label('Intensity')
        else:
            # 개별 컬러바 추가 (make_axes_locatable 사용)
            for i, (ax, im) in enumerate(zip(axes, im_objects)):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                if log_scale:
                    cbar.set_label('$\log_{10}(Intensity)$')
                    # 컬러바 눈금에 10^x 형태로 표시
                    tick_locations = np.linspace(vmin, vmax, 5)
                    cbar.set_ticks(tick_locations)
                    cbar.set_ticklabels([f'$10^{{{x:.1f}}}$' for x in tick_locations])
                else:
                    cbar.set_label('Intensity')
        
        # 대비 조절에 대한 정보 추가
        if contrast_enhance < 1.0:
            percent_clip = round((1 - contrast_enhance) * 100)
            max_intensity_value = 10**(vmax) if log_scale else max_intensity
            fig.suptitle(f"Dynamic Spectral Functions - Intensity clipped at {percent_clip}% ({max_threshold:.2e})", fontsize=16)
        else:
            fig.suptitle(f"Dynamic Spectral Functions along {path_name} path", fontsize=16)
        
        plt.tight_layout()
        return fig