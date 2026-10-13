import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from matplotlib.patches import ConnectionPatch
from itertools import product
from matplotlib.lines import Line2D


from modules.SpinSystem.nbcp_unitcells import NBCP_UNIT_CELL

exchange_colors = {
    0: '#2E86AB', 1: '#A23B72', 2: '#3CAB70', 3: '#F5B700', 4: '#0F8B8D',
    5: '#8963BA', 6: '#EC9A29', 7: '#2C5784', 8: '#9B4F0F', 9: '#1B998B'
}
linestyles = [
    (0, (1, 1)), (0, (3, 1)), (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)),
    (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 2)), (0, (3, 2)), (0, (3, 2, 1, 2)), '-'
]

class SpinVisualizer:
    def __init__(self):
        pass
    
    @staticmethod
    def _rotation_matrix(theta: float, phi: float) -> np.ndarray:
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi  # 정규화: [-π, π]
        if abs(theta - np.pi) < 0.01:
            theta = np.pi
        elif abs(theta + np.pi) < 0.01:
            theta = -np.pi
        R = np.array([
            [np.cos(theta)*np.cos(phi), -np.sin(phi), np.sin(theta)*np.cos(phi)],
            [np.cos(theta)*np.sin(phi),  np.cos(phi), np.sin(theta)*np.sin(phi)],
            [-np.sin(theta),             0,           np.cos(theta)]
        ])
        return R
    
    @staticmethod
    def calculate_rotations(spin_info):
        rmat_dict = {}
        for key, info in spin_info.items():
            theta, phi = info["Angles"]
            rmat_dict[key] = SpinVisualizer._rotation_matrix(theta, phi)
        return rmat_dict
    
    @staticmethod
    def generate_lattice_points(lattice_vectors):
        lattice_points = []
        n_range = [-2, -1, 0, 1, 2]  
        a1, a2 = lattice_vectors

        for n1, n2 in product(n_range, n_range):
            point = n1 * a1 + n2 * a2
            lattice_points.append(point)
        return np.array(lattice_points)
    
    @staticmethod
    def get_unique_matrices(couplings):
        unique_matrices = {}
        for coupling in couplings:
            matrix = np.array(coupling["Exchange Matrix"])
            matrix_tuple = tuple(tuple(np.round(row, 8).tolist()) for row in matrix)  # ndarray -> list -> tuple
            if matrix_tuple not in unique_matrices:
                unique_matrices[matrix_tuple] = len(unique_matrices)
        return unique_matrices
    
    @staticmethod
    def get_exchange_index(matrix, unique_matrices):
        matrix = np.array(matrix)
        matrix_tuple = tuple(tuple(np.round(row, 8).tolist()) for row in matrix)  # ndarray -> list -> tuple
        return unique_matrices.get(matrix_tuple, 0)
    
    @staticmethod
    def update_ax_xy(ax_xy, spin_system_data, 
                     show_couplings=True, 
                     title="Spin Configuration with Lattice", 
                     show_unit_cells = True):
        spin_info = spin_system_data["Spin info"]
        couplings = spin_system_data["Couplings"]
        
        rotation_matrices = SpinVisualizer.calculate_rotations(spin_info)
        lattice_points = SpinVisualizer.generate_lattice_points(spin_system_data["Lattice/BZ setting"][0])
        unique_matrices = SpinVisualizer.get_unique_matrices(couplings)
        
        ax_xy.set_title(title)
        ax_xy.set_xlabel("X")
        ax_xy.set_ylabel("Y")
        ax_xy.grid(True, linestyle=':', alpha=0.3)
        ax_xy.set_aspect('equal')
        ax_xy.scatter(lattice_points[:, 0], lattice_points[:, 1], 
                      color='lightgray', alpha=0.5, s=50, zorder=1)

        if show_couplings:
            plotted_links = set()
            
            for base_point in lattice_points:
                for coupling in couplings:
                    spin_i = coupling["SpinI"]
                    spin_j = coupling["SpinJ"]
                    
                    if spin_i not in spin_info or spin_j not in spin_info:
                        print(f"Warning: Invalid SpinI={spin_i} or SpinJ={spin_j} in coupling")
                        continue
                    
                    displacement = np.array(coupling["Displacement"])
                    
                    pos_i = np.array(spin_info[spin_i]["Position"])
                    start_point = base_point + pos_i
                    end_point   = base_point + pos_i + displacement
                    
                    link_id = tuple(sorted([tuple(np.round(start_point, 8)), tuple(np.round(end_point, 8))]))
                    
                    if link_id in plotted_links:
                        continue
                    
                    exchange_idx = SpinVisualizer.get_exchange_index(coupling["Exchange Matrix"], unique_matrices)
                    color = exchange_colors[exchange_idx]
                    linestyle = linestyles[exchange_idx % len(linestyles)]
                    
                    ax_xy.plot([start_point[0], end_point[0]], 
                               [start_point[1], end_point[1]],
                               linestyle=linestyle, color=color, 
                               alpha=0.7, linewidth=1.2, zorder=2)
                    
                    plotted_links.add(link_id)

        plotted_positions = set()
        max_range = 0
        for key, info in spin_info.items():
            x, y = info["Position"]
            max_range = max(max_range, abs(x), abs(y))
            spin = rotation_matrices[key] @ np.array([0, 0, 1]) * 0.6
            spin_color = plt.cm.coolwarm((spin[2] + 1) / 2)
            
            for lattice_point in lattice_points:
                pos_x = lattice_point[0] + x
                pos_y = lattice_point[1] + y
                pos_key = (round(pos_x, 8), round(pos_y, 8))
                if pos_key in plotted_positions:
                    print(f"Warning: Duplicate position {pos_key} for {key}")
                    continue
                plotted_positions.add(pos_key)
                
                ax_xy.quiver(pos_x, pos_y, spin[0], spin[1], 
                             angles='xy', scale_units='xy', scale=0.8,
                             color='k', zorder=4)
                ax_xy.scatter(pos_x, pos_y, color=spin_color, s=75,
                              label=f"{key}" if np.all(lattice_point == [0,0]) else "",
                              zorder=3)
        
        ax_xy.set(xlim=[-3, 3], ylim=[-3, 3], aspect='equal')
        legend_elements = []
        for i in range(len(unique_matrices)):
            color = exchange_colors[i]
            linestyle = linestyles[i % len(linestyles)]
            legend_elements.append(
                Line2D([0], [0], color=color, linestyle=linestyle, 
                       linewidth=1.2, alpha=0.7, label=f'J{i}')
            )
        ax_xy.legend(handles=legend_elements, title="Exchange Interactions",
                     loc='upper right', ncol=2)
        return ax_xy
    
    @staticmethod
    def update_ax_angle(ax_angle, spin_system_data, title="Angular Distribution"):
        spin_info = spin_system_data["Spin info"]
        rotation_matrices = SpinVisualizer.calculate_rotations(spin_info)
        
        ax_angle.set_title(title)
        ax_angle.plot(np.linspace(-np.pi, np.pi, 100), np.ones(100),
                      '--', color='gray', alpha=0.5)
        ax_angle.set_rticks([])
        ax_angle.set_rlim(0, 1.2)
        ax_angle.set_theta_offset(np.pi/2)

        for key, info in spin_info.items():
            theta, _ = info["Angles"]
            theta = np.mod(theta, 2 * np.pi)
            spin = rotation_matrices[key] @ np.array([0, 0, 1])
            spin_color = plt.cm.coolwarm((spin[2] + 1) / 2)
            ax_angle.annotate('', xy=(theta, 1), xytext=(theta, 0),
                              arrowprops=dict(arrowstyle='->', color=spin_color,
                                              lw=2, mutation_scale=15))
            ax_angle.text(theta, 1.1, key, ha='center', va='center', fontsize=8)
        return ax_angle
    
    @staticmethod
    def plot_system(spin_system_data, figsize=(15, 6), show_couplings=True, 
                   xy_title="Spin Configuration with Lattice", 
                   angular_title="Angular Distribution",
                   main_title=None):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(1, 2)
        
        ax_xy = fig.add_subplot(gs[0])
        ax_xy = SpinVisualizer.update_ax_xy(ax_xy, spin_system_data, show_couplings, xy_title)
        
        ax_angle = fig.add_subplot(gs[1], projection='polar')
        ax_angle = SpinVisualizer.update_ax_angle(ax_angle, spin_system_data, angular_title)
        
        if main_title:
            fig.suptitle(main_title, fontsize=16, y=1.05)
        
        plt.tight_layout()
        return fig, (ax_xy, ax_angle)
        opt_env = tlaf_obj.get_opt_environment(None)
        seen_phases = set()

        for phase_name, phase_results in results.items():
            if phase_name in seen_phases:
                continue

            for opt_method, data in phase_results.items():
                # 데이터 및 에너지 계산
                angles = data["angles"]
                spin_sys_data = opt_env[phase_name]["Data function"](angles=tuple(angles))
                E_tot = data["E_cl"] if opt_method == "classical" else data["E_cl"] + data["E_qm"]

                # Degenerate 상태 찾기
                degenerate = [phase_name]
                for deg_phase_name, deg_results in results.items():
                    if deg_phase_name != phase_name and opt_method in deg_results:
                        deg_data = deg_results[opt_method]
                        deg_E_tot = deg_data["E_cl"] if opt_method == "classical" else deg_data["E_cl"] + deg_data["E_qm"]
                        if abs(deg_E_tot - E_tot) < threshold:
                            degenerate.append(deg_phase_name)

                # 플롯 생성
                fig, axes = plt.subplots(len(degenerate), 2, figsize=(15, 5 * len(degenerate)), squeeze=False)
                for i, deg_phase in enumerate(degenerate):
                    deg_spin_sys_data = opt_env[deg_phase]["Data function"](angles=tuple(results[deg_phase][opt_method]["angles"]))

                    # XY 플롯
                    SpinVisualizer.update_ax_xy(
                        axes[i, 0], deg_spin_sys_data, show_couplings=(i == 0),
                        title=f"{opt_method} Phase: {deg_phase}"
                    )
                    axes[i, 0].text(0.05, 0.95, f"E_tot: {results[deg_phase][opt_method]['E_cl']:.6f}", 
                                    transform=axes[i, 0].transAxes, fontsize=10)

                    # Angle 플롯 (Polar)
                    ax_angle = fig.add_subplot(len(degenerate), 2, i * 2 + 2, projection='polar')
                    SpinVisualizer.update_ax_angle(ax_angle, deg_spin_sys_data, title=f"Angles: {deg_phase}")

                fig.suptitle(f"Phase: {phase_name}", fontsize=16, y=1.02)
                plt.tight_layout()
                plt.show()
                seen_phases.add(phase_name)