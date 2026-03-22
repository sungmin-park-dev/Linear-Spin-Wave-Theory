import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from typing import List, Tuple, Dict, Union


from modules.Tools.find_first_bz import get_BZ_hex, get_BZ_any, get_BZ_tetra


"""Brillouin Zone class for calculating BZ information"""

class Brillouin_Zone:    
    def __init__(self, lattice_bz_setting, bz_type = None):
        self.lattice_bz_setting = self.get_bz_setting(lattice_bz_setting, bz_type)
        self.bz_data, self.get_bz_grid = self.get_bz_data(self.lattice_bz_setting)
        return
        
    def get_bz_setting(self, bz_setting, bz_type = None, ):
        if bz_type is None:
            bz_setting = bz_setting
        
        elif bz_type == "simple":
            bz_setting = (bz_setting[0], "simple")        
        
        elif bz_type == "Hex_60":
            bz_setting = ( ( np.array([1/2, np.sqrt(3)/2]), 
                             np.array([1/2, - np.sqrt(3)/2]) ), 
                          "Hex_60")
        else: 
            ValueError(f"Unknown BZ type: {bz_type}")
        
        return bz_setting


    def get_bz_data(self, lattice_bz_setting = None):
        """브릴루앙 존 정보 획득
        Args:
            bz_type (str): "Hex_60", "Hex_30" 또는 "Tetra"
            params (float or tuple): 격자 상수 정보
        """
        if  lattice_bz_setting is None:
            lattice_vectors, bz_type = self.lattice_bz_setting
        elif isinstance(lattice_bz_setting, tuple):
            lattice_vectors, bz_type = lattice_bz_setting
        else:
            raise ValueError("Lattice vectors must be a tuple")
        
        rvec1, rvec2 = lattice_vectors
        
        # Calculate the length of the lattice vectors
        a = np.sum(rvec1**2)**0.5
        b = np.sum(rvec2**2)**0.5
        
        if bz_type == "simple":
            # Get the first Brillouin zone
            bz = get_BZ_any(rvec1, rvec2)
            bz_data = bz.get_BZ_parallelogram()

            return bz_data, bz.get_fbz_grid
        
        elif bz_type == "Hex_60":
            # Get the first Brillouin zone
            bz_hex = get_BZ_hex(a, phi = 0)
            bz_data = bz_hex.get_BZ_hex()
            
            return bz_data, bz_hex.get_fbz_grid
        
        elif bz_type == "Hex_30":
            # Get the first Brillouin zone
            bz_hex = get_BZ_hex(a, phi = np.pi/6)
            bz_data = bz_hex.get_BZ_hex()
            
            return bz_data, bz_hex.get_fbz_grid

        elif bz_type == "Tetra":
            # Get the first Brillouin zone
            bz_tetra = get_BZ_tetra(a, b)
            bz_data = bz_tetra.get_BZ_tetra()
            
            return bz_data, bz_tetra.get_fbz_grid

        elif bz_type == "wigner_seitz":
            # Get the first Brillouin zone
            bz_any = get_BZ_any(rvec1, rvec2)
            bz_data = bz_any.get_BZ_wigner_seitz_cell()
            
            return bz_data, bz_any.get_fbz_grid

        else:
            raise ValueError(f"Unknown BZ type: {bz_type}")



    def generate_kpath(self, band_path, HSP, mag=100):
        path_kpoints = []
        path_intervals = []
        
        for i in range(len(band_path) - 1):
            start_point = np.array(HSP[band_path[i]])
            end_point = np.array(HSP[band_path[i + 1]])
            norm = np.linalg.norm(end_point - start_point)
            num = max(2, int(np.round(norm * mag)))
            
            print(f"Segment {band_path[i]} -> {band_path[i+1]}: {num} points")
            
            segment = np.linspace(start_point, end_point, num, endpoint=False)
            path_kpoints.extend(segment)
            path_intervals.append(num)
        
        # 마지막 점 추가
        last_point = np.array(HSP[band_path[-1]])
        path_kpoints.append(last_point)
        path_intervals[-1] += 1
        
        path_kpoints = np.array(path_kpoints)
        unique_ks, counts = np.unique(path_kpoints, axis=0, return_counts=True)
        duplicates = unique_ks[counts > 1]
        if len(duplicates) > 0:
            for dup in duplicates:
                indices = [i for i, k in enumerate(path_kpoints) if np.allclose(k, dup)]
                print(f"Duplicate k-point {dup} at indices: {indices}")
        print(f"Total k_points: {len(path_kpoints)}")
        return path_kpoints, path_intervals

    
    def plot_polygon_and_grid(self, vertices, grid_points=None, HSP=None, band_path=None, title=None, ax=None):
        """
        Plot a polygon, grid points, HSP points, and band paths.
        
        Args:
            vertices (list of tuple): List of (x, y) coordinates defining the polygon vertices
            grid_points (numpy.ndarray, optional): Array of points inside the polygon
            HSP (dict, optional): Dictionary of high symmetry points
            band_path (dict, optional): Dictionary containing different band paths
            title (str, optional): Plot title
            ax (matplotlib.axes.Axes, optional): Axes object to plot on. If None, creates new figure.
        
        Returns:
            tuple: (matplotlib.figure.Figure, matplotlib.axes.Axes) - Figure and axes objects
        """
        # Create new figure and axes if not provided
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        
        polygon_x, polygon_y = zip(*vertices)
        polygon_x = list(polygon_x) + [polygon_x[0]]
        polygon_y = list(polygon_y) + [polygon_y[0]]
        
        # Plot grid points if provided (최하단 레이어)
        if grid_points is not None:
            ax.scatter(grid_points[:, 0], grid_points[:, 1], 
                    c='gray', s=10, alpha=0.3, label='Grid Points', zorder=1)
        
        # Plot the polygon boundary
        ax.plot(polygon_x, polygon_y, 'k-', linewidth=1, label='BZ boundary', zorder=2)
        
        # Plot band paths if provided
        if HSP is not None and band_path is not None:
            # Plot standard path (solid line)
            if "standard path" in band_path:
                path = band_path["standard path"]
                path_coords = np.array([HSP[k] for k in path])
                ax.plot(path_coords[:, 0], path_coords[:, 1], 
                        'r-', linewidth=1.5, label='Standard path', zorder=3)
            
            # Plot rotated path (dashed line)
            if "rotated path" in band_path:
                path = band_path["rotated path"]
                path_coords = np.array([HSP[k] for k in path])
                ax.plot(path_coords[:, 0], path_coords[:, 1], 
                        'g--', linewidth=1.5, label='Rotated path', zorder=4)
            
            # Plot inverse path first (thicker line)
            if "inverse path" in band_path:
                path = band_path["inverse path"]
                path_coords = np.array([HSP[k] for k in path])
                ax.plot(path_coords[:, 0], path_coords[:, 1], 
                        'b-', linewidth=3, alpha=0.3, label='Inverse path', zorder=5)
        
        # Plot HSP if provided (최상단 레이어)
        if HSP is not None:
            for point_name, coords in HSP.items():
                ax.scatter(coords[0], coords[1], c='purple', s=100, marker='*', zorder=6)
                ax.annotate(point_name, (coords[0], coords[1]), 
                            xytext=(5, 5), textcoords='offset points', zorder=7)
        
        ax.set_xlabel('$k_x$')
        ax.set_ylabel('$k_y$')
        if title:
            ax.set_title(title)
        ax.legend()
        ax.axis('equal')
        ax.grid(True)
        
        return fig, ax


    def get_full(self, N, center = (0,0), buffer = 0.1, print_idx = False):
        """브릴루앙 존 정보와 그리드 점들을 획득
        Args:
            bz_type (str): "Hex_60", "Hex_30" 또는 "Tetra"
            lattice_vectors (list): 격자 벡터
            N (int): 그리드 포인트 수
            center (tuple): 그리드 중심
            buffer (float): 경계 버퍼
            print_idx (bool): 인덱스 출력 여부
        Returns:
            tuple: (bz_data, grid_points, grid_indices)
        """
        
        bz_data, get_bz_grid = self.get_bz_data()
        
        grid_points, area, grid_indices = get_bz_grid(N, center = center, print_idx = print_idx)
        
        bz_data["area"] = area
        
        return bz_data, grid_points, grid_indices
    

# Example usage for subplots
if __name__ == "__main__":
    # Create a figure with 2x2 subplots
    fig = plt.figure(figsize=(12, 10))
    
    # Define various lattice vectors
    phi = np.pi * 0.4
    hex60_lv = (np.array([0.5, 0.866]), np.array([0.5, -0.866]))  # For Hex_60
    hex30_lv = (np.array([0.866, 0.5]), np.array([-0.866, 0.5]))  # For Hex_30
    random_lv = (np.array([0.7, 0.3]), np.array([0.2, 0.8]))  # Asymmetric lattice vectors
    angled_lv = (np.array([np.cos(phi), np.sin(phi)]), np.array([np.cos(phi), -np.sin(phi)]))
    
    # Hex_60 Brillouin Zone
    ax1 = fig.add_subplot(221)
    bz_hex60 = Brillouin_Zone((hex60_lv, "Hex_60"))
    bz_data, grid_points, _ = bz_hex60.get_full(10)
    BZ = bz_data["BZ_corners"]
    HSP = bz_data["high_symmetry_points"]
    band_path = bz_data["band_paths"]
    _, ax1 = bz_hex60.plot_polygon_and_grid(BZ, grid_points=grid_points, HSP=HSP, band_path=band_path, 
                                        title="Hexagonal Lattice (60°)", ax=ax1)
    
    # Hex_30 Brillouin Zone
    ax2 = fig.add_subplot(222)
    bz_hex30 = Brillouin_Zone((hex30_lv, "Hex_30"))
    bz_data, grid_points, _ = bz_hex30.get_full(10)
    BZ = bz_data["BZ_corners"]
    HSP = bz_data["high_symmetry_points"]
    band_path = bz_data["band_paths"]
    _, ax2 = bz_hex30.plot_polygon_and_grid(BZ, grid_points=grid_points, HSP=HSP, band_path=band_path, 
                                        title="Hexagonal Lattice (30°)", ax=ax2)
    
    # Wigner-Seitz Cell for arbitrary lattice
    ax3 = fig.add_subplot(223)
    bz_any = Brillouin_Zone((random_lv, "wigner_seitz"))
    bz_data, grid_points, _ = bz_any.get_full(15)
    BZ = bz_data["BZ_corners"]
    HSP = bz_data["high_symmetry_points"]
    band_path = bz_data["band_paths"]
    _, ax3 = bz_any.plot_polygon_and_grid(BZ, grid_points=grid_points, HSP=HSP, band_path=band_path, 
                                      title="Wigner-Seitz Cell (Arbitrary Lattice)", ax=ax3)
    
    # Simple Brillouin Zone with angled lattice vectors
    ax4 = fig.add_subplot(224)
    bz_simple = Brillouin_Zone((angled_lv, "simple"))
    bz_data, grid_points, _ = bz_simple.get_full(12)
    BZ = bz_data["BZ_corners"]
    HSP = bz_data["high_symmetry_points"]
    band_path = bz_data["band_paths"]
    _, ax4 = bz_simple.plot_polygon_and_grid(BZ, grid_points=grid_points, HSP=HSP, band_path=band_path, 
                                        title="Simple Brillouin Zone (Angled Lattice)", ax=ax4)
    
    plt.tight_layout()
    plt.show()