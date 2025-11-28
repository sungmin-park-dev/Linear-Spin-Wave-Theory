import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Dict, Any, Optional, List, Tuple

# Default plot configuration
DEFAULT_PLOT_CONFIG = {
    'figsize': (8, 6),
    'title': 'Plot',
    'title_fontsize': 14,
    'xlabel': '$k_x$',
    'xlabel_fontsize': 12,
    'ylabel': '$k_y$',
    'ylabel_fontsize': 12,
    'cmap': 'viridis',
    'colorbar_label': 'Value',
    'colorbar_fontsize': 12,
    'bz_linewidth': 2,
    'hsp_marker': '*',
    'hsp_color': 'red',
    'hsp_size': 50,
    'hsp_text_offset': (5, 5),
    'dpi': 100,
    'vmin': None,
    'vmax': None
}

class MomentumSpacePlotter:
    """Visualizes momentum space data with Brillouin Zone and high-symmetry points."""
    def __init__(self,
                 lswt_obj,  # lswt_obj: LSWT instance with spin information.
                 bz_data: Dict[str, Any]):  # bz_data: Dictionary with BZ corners and high-symmetry points.
        """Initializes momentum space plotter."""
        self.lswt = lswt_obj
        self.spin_info = lswt_obj.spin_info
        self.num_sublattices = len(self.spin_info)
        self.BZ = bz_data["BZ_corners"]
        self.HSP = bz_data["high_symmetry_points"]

    def _plot_single_data(self, ax, triang, data, config = None):
        cfg = DEFAULT_PLOT_CONFIG.copy()
        
        if config:
            # print("Debug: config in _plot_single_data =", config)
            cfg.update(config)

        # self._plot_bz_and_hsp(ax, cfg)
        tcf = ax.tripcolor(triang, data, cmap=cfg['cmap'], shading='gouraud',
                        vmin=cfg['vmin'], vmax=cfg['vmax'])
        self._add_colorbar(ax, tcf, cfg['colorbar_label'], cfg['colorbar_fontsize'])
        ax.set_title(cfg.get('title', 'Plot'), fontsize=cfg['title_fontsize'])
        ax.set_xlabel(cfg['xlabel'], fontsize=cfg['xlabel_fontsize'])
        ax.set_ylabel(cfg['ylabel'], fontsize=cfg['ylabel_fontsize'])
        ax.set_aspect('equal')

    def _plot_bz_and_hsp(self,
                         ax: plt.Axes,  # ax: Matplotlib axes to plot on.
                         config: Dict[str, Any]) -> None:  # config: Plot settings for BZ and HSP.
        """Plots Brillouin Zone boundaries and high-symmetry points."""
        bz_x, bz_y = zip(*self.BZ)
        bz_x = list(bz_x) + [bz_x[0]]
        bz_y = list(bz_y) + [bz_y[0]]
        ax.plot(bz_x, bz_y, 'k-', linewidth=config['bz_linewidth'])
        for name, coords in self.HSP.items():
            ax.scatter(coords[0], coords[1], c=config['hsp_color'], s=config['hsp_size'],
                       marker=config['hsp_marker'], zorder=6)
            ax.annotate(name, coords, xytext=config['hsp_text_offset'],
                        textcoords='offset points', zorder=7)

    def _add_colorbar(self,
                      ax: plt.Axes,  # ax: Matplotlib axes to attach colorbar.
                      mappable: Any,  # mappable: Colormap object.
                      label: str,  # label: Colorbar label.
                      fontsize: int) -> None:  # fontsize: Font size for colorbar label.
        """Adds a colorbar to the plot."""
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(mappable, cax=cax)
        cbar.set_label(label, fontsize=fontsize)

    def _set_path_axes(self,
                       ax: plt.Axes,  # ax: Matplotlib axes to configure.
                       x_points: List[float],  # x_points: X-axis tick positions.
                       x_labels: List[str]) -> None:  # x_labels: X-axis tick labels.
        """Configures axes for path plots."""
        ax.set_xticks(x_points)
        ax.set_xticklabels(x_labels)
        ax.set_xlim(x_points[0], x_points[-1])
        for x_pos in x_points:
            ax.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.5)


    """
    Correlation function plot
    """

    def plot_number_data(self,
                         k_points: np.ndarray,  # k_points: Array of k-points.
                         total_number: np.ndarray,  # total_number: Total number data.
                         component_numbers: np.ndarray,  # component_numbers: Component-wise number data.
                         component_names: Optional[List[str]] = None,  # component_names: Names of components.
                         config: Optional[Dict[str, Any]] = None) -> plt.Figure:  # config: Plot settings.
        """Visualizes number data with total sum on the left and components on the right."""
        config = config or {}
        component_names = component_names or [f'Component {i+1}' for i in range(component_numbers.shape[1])]
        triang = Triangulation(-k_points[:, 0], -k_points[:, 1])    # - sign is because of BdG formalism

        fig = plt.figure(figsize=(15, 8))
        gs = fig.add_gridspec(2, 3)

        ax_total = fig.add_subplot(gs[:, 0])
        self._plot_single_data(ax_total, triang, total_number,
                               {'title': 'Total Number', 'colorbar_label': 'Number', **config})

        num_components = min(component_numbers.shape[1], 4)
        for i in range(num_components):
            row, col = i // 2, (i % 2) + 1
            ax_comp = fig.add_subplot(gs[row, col])
            self._plot_single_data(ax_comp, triang, component_numbers[:, i],
                                   {'title': component_names[i], 'colorbar_label': 'Number', **config})

        plt.tight_layout()
        return fig

    def plot_all_spin_corr_func_in_real_space(self, kx, ky, spin_corr_comp, spin_corr_total, func_type, config):
        cfg = DEFAULT_PLOT_CONFIG.copy()
        if not isinstance(config, dict):
            print(f"Warning: config is not a dictionary, got {type(config)}: {config}. Using empty dict.")
            config = {}
        cfg.update(config)

        triang = Triangulation(kx, ky)

        fig = plt.figure(figsize=(20, 8), dpi=cfg['dpi'])
        gs = fig.add_gridspec(3, 4)

        # 전체 플롯의 상단 제목 (suptitle)
        fig.suptitle(cfg.get('title', 'Correlation Plot'), fontsize=cfg.get('title_fontsize', 14) + 2)

        if func_type in ("structure factor", "Strcutre"):
            spin_corr_total = np.real(spin_corr_total)
        elif func_type in ("spectral function", "Spectral"):
            spin_corr_total = -np.imag(spin_corr_total)/np.pi

        ax_total = fig.add_subplot(gs[:, 0])
        self._plot_single_data(ax_total, triang, spin_corr_total,
                            {**cfg, 'title': cfg.get('title', 'Total Correlation'), 'colorbar_label': 'Correlation'})

        # spin_corr_comp의 키를 직접 사용
        keys = sorted(spin_corr_comp.keys())  # 예: ['++', '+-', '+0', '-+', '--', '-0', '0+', '0-', '00']
        print("Debug: spin_corr_comp keys =", keys)  # 디버깅 로그
        for i in range(3):
            for j in range(3):
                row, col = i, j + 1
                ax_comp = fig.add_subplot(gs[row, col])
                key = keys[i * 3 + j]  # 3x3 그리드에 맞게 키 선택
                print(f"Debug: Setting title for subplot ({i}, {j}) to '{key}'")  # 디버깅 로그
                if func_type in ("structure factor", "Strcutre"):
                    spin_corr_ab = np.real(spin_corr_comp[key])
                elif func_type in ("spectral function", "Spectral"):
                    spin_corr_ab = -np.imag(spin_corr_comp[key])/np.pi
                self._plot_single_data(ax_comp, triang, spin_corr_ab,
                                    {**cfg, 'title': key, 'colorbar_label': 'Correlation'})

        plt.tight_layout()
        return fig