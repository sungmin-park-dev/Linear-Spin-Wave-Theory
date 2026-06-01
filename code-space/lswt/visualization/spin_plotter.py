"""Spin configuration plotter for SpinSystem objects.

Visualizes spin arrangements on a 2D lattice:
- Gray circle at each site (size = spin magnitude S)
- Spin marker fill color = Sz component (coolwarm colormap)
- Black quiver arrow = Sxy projection (in-plane spin direction)
- Pure z-spin (no xy component) = colored dot (red=up, blue=down)
- Optional polar subplot showing angular distribution of spins
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from itertools import product


# Bond colors / linestyles for distinct exchange matrices
BOND_COLORS = [
    '#888888', '#CC4444', '#4477AA', '#44AA77',
    '#AAAA44', '#AA44AA', '#44AAAA', '#AA7744',
]
BOND_LINESTYLES = [
    '-', (0, (4, 2)), (0, (1, 1)), (0, (4, 2, 1, 2)),
    (0, (6, 2)), (0, (2, 1)), (0, (6, 2, 2, 2)), (0, (1, 2)),
]

# Sublattice colors for polar plot labels
SUBLATTICE_COLORS = [
    '#2E86AB', '#A23B72', '#3CAB70', '#F5B700',
    '#0F8B8D', '#8963BA', '#EC9A29', '#2C5784',
]


def _spin_direction(theta, phi):
    """Convert spherical angles to Cartesian spin direction vector.

    Parameters
    ----------
    theta : float
        Polar angle from z-axis.
    phi : float
        Azimuthal angle in xy-plane.

    Returns
    -------
    np.ndarray
        (Sx, Sy, Sz) unit vector.
    """
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def _unique_exchange_matrices(couplings):
    """Identify unique exchange matrices among couplings."""
    unique = {}
    for c in couplings:
        key = tuple(tuple(np.round(row, 8)) for row in c.exchange_matrix)
        if key not in unique:
            unique[key] = len(unique)
    return unique


def _exchange_index(matrix, unique_map):
    """Get the index of an exchange matrix in the unique map."""
    key = tuple(tuple(np.round(row, 8)) for row in matrix)
    return unique_map.get(key, 0)


def _compute_min_site_distance(system):
    """Compute minimum distance between any two site positions."""
    a1 = system.lattice_vectors[0]
    a2 = system.lattice_vectors[1]

    min_dist = np.inf
    for i, si in enumerate(system.sites):
        for n1, n2 in product(range(-1, 2), range(-1, 2)):
            R = n1 * a1 + n2 * a2
            for j, sj in enumerate(system.sites):
                if i == j and n1 == 0 and n2 == 0:
                    continue
                d = np.linalg.norm(si.position - sj.position - R)
                if d > 1e-8 and d < min_dist:
                    min_dist = d

    if min_dist == np.inf:
        min_dist = np.linalg.norm(a1)
    return min_dist


def _compute_view_bounds(system, n_repeat, padding=0.3):
    """Compute tight view bounds from site positions in the repeated lattice."""
    a1 = system.lattice_vectors[0]
    a2 = system.lattice_vectors[1]
    n_range = range(-n_repeat, n_repeat + 1)

    all_x, all_y = [], []
    for site in system.sites:
        for n1, n2 in product(n_range, n_range):
            pos = site.position + n1 * a1 + n2 * a2
            all_x.append(pos[0])
            all_y.append(pos[1])

    return (min(all_x) - padding, max(all_x) + padding,
            min(all_y) - padding, max(all_y) + padding)


def _point_in_bounds(x, y, bounds, margin=0.3):
    """Check if a point is within the view bounds (with margin)."""
    xmin, xmax, ymin, ymax = bounds
    return (xmin - margin <= x <= xmax + margin and
            ymin - margin <= y <= ymax + margin)


def _line_in_bounds(x0, y0, x1, y1, bounds, margin=0.5):
    """Check if at least one endpoint of a line segment is within bounds."""
    return (_point_in_bounds(x0, y0, bounds, margin) or
            _point_in_bounds(x1, y1, bounds, margin))


# ======================================================================
# Main plot function
# ======================================================================

def plot_spin_configuration(system, n_repeat=1, figsize=(8, 8),
                            show_couplings=True, show_unit_cell=True,
                            show_polar=True, arrow_scale=None,
                            title=None, ax=None):
    """Plot spin configuration from a SpinSystem object.

    Visual encoding:
    - Gray circle: lattice site (size proportional to spin magnitude S)
    - Marker fill color: Sz component (coolwarm: blue=down, red=up)
    - Black arrow (quiver): Sxy projection (in-plane spin direction)
    - Pure z-spin: colored dot (red=up, blue=down), no arrow

    If show_polar=True and ax is None, creates a two-panel figure with
    the spin configuration on the left and a polar angle distribution
    on the right.

    Parameters
    ----------
    system : SpinSystem
        The spin system to visualize.
    n_repeat : int, optional
        Number of unit cell repetitions in each direction (default: 1).
    figsize : tuple, optional
        Figure size (default: (8, 8)). When show_polar=True, width is
        extended automatically.
    show_couplings : bool, optional
        Draw coupling bonds between sites (default: True).
    show_unit_cell : bool, optional
        Draw unit cell boundary (default: True).
    show_polar : bool, optional
        Show polar angle distribution subplot (default: True).
    arrow_scale : float or None, optional
        Quiver scale factor. If None, auto-computed from inter-site distance.
    title : str, optional
        Plot title (default: auto-generated).
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If provided, show_polar is ignored.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : matplotlib.axes.Axes or tuple of Axes
        Single axes if show_polar=False or ax provided,
        (ax_xy, ax_polar) tuple if show_polar=True.
    """
    # Create figure layout
    if ax is not None:
        fig = ax.get_figure()
        ax_xy = ax
        ax_polar = None
    elif show_polar:
        fig = plt.figure(figsize=(figsize[0] * 1.6, figsize[1]))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        ax_xy = fig.add_subplot(gs[0])
        ax_polar = fig.add_subplot(gs[1], projection='polar')
    else:
        fig, ax_xy = plt.subplots(1, 1, figsize=figsize)
        ax_polar = None

    _draw_spin_lattice(ax_xy, system, n_repeat, show_couplings,
                       show_unit_cell, arrow_scale, title)

    if ax_polar is not None:
        _draw_polar_angles(ax_polar, system)

    fig.tight_layout()

    if ax_polar is not None:
        return fig, (ax_xy, ax_polar)
    return fig, ax_xy


# ======================================================================
# Spin lattice plot (left panel)
# ======================================================================

def _draw_spin_lattice(ax, system, n_repeat, show_couplings,
                       show_unit_cell, arrow_scale, title):
    """Draw the 2D spin configuration on the given axes."""
    a1 = system.lattice_vectors[0]
    a2 = system.lattice_vectors[1]

    # View bounds and draw range
    bounds = _compute_view_bounds(system, n_repeat)
    xmin, xmax, ymin, ymax = bounds

    n_draw = n_repeat + 1
    n_range_draw = range(-n_draw, n_draw + 1)
    n_range_cell = range(-n_repeat, n_repeat + 1)

    # Sz colormap
    sz_cmap = plt.cm.coolwarm
    sz_norm = Normalize(vmin=-1, vmax=1)

    # Auto-compute arrow scale
    min_dist = _compute_min_site_distance(system)
    if arrow_scale is None:
        arrow_scale = 0.4 * min_dist

    # Spin magnitude for marker sizing
    spins = [site.spin for site in system.sites]
    s_max = max(spins) if spins else 1.0

    # ------------------------------------------------------------------
    # Layer 1: Coupling bonds
    # ------------------------------------------------------------------
    legend_handles = []
    if show_couplings and system.couplings:
        unique_map = _unique_exchange_matrices(system.couplings)
        plotted_bonds = set()

        for n1, n2 in product(n_range_draw, n_range_draw):
            R = n1 * a1 + n2 * a2
            for c in system.couplings:
                pos_i = system.sites[c.site_i].position + R
                pos_j = pos_i + c.displacement

                if not _line_in_bounds(pos_i[0], pos_i[1],
                                       pos_j[0], pos_j[1], bounds):
                    continue

                bond_key = (tuple(np.round(pos_i, 6)),
                            tuple(np.round(pos_j, 6)))
                bond_key_rev = (bond_key[1], bond_key[0])
                if bond_key in plotted_bonds or bond_key_rev in plotted_bonds:
                    continue
                plotted_bonds.add(bond_key)

                idx = _exchange_index(c.exchange_matrix, unique_map)
                color = BOND_COLORS[idx % len(BOND_COLORS)]
                ls = BOND_LINESTYLES[idx % len(BOND_LINESTYLES)]

                ax.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]],
                        color=color, linestyle=ls, alpha=0.5,
                        linewidth=1.0, zorder=1, solid_capstyle='round')

        for idx in range(len(unique_map)):
            color = BOND_COLORS[idx % len(BOND_COLORS)]
            ls = BOND_LINESTYLES[idx % len(BOND_LINESTYLES)]
            legend_handles.append(
                Line2D([0], [0], color=color, linestyle=ls,
                       linewidth=1.5, alpha=0.7, label=f'J{idx}'))

    # ------------------------------------------------------------------
    # Layer 2: Unit cell boundaries
    # ------------------------------------------------------------------
    if show_unit_cell:
        for n1, n2 in product(n_range_cell, n_range_cell):
            origin = n1 * a1 + n2 * a2
            corners = np.array([origin, origin + a1,
                                origin + a1 + a2, origin + a2])
            ax.add_patch(Polygon(
                corners, closed=True, fill=False,
                edgecolor='#BBBBBB', linewidth=0.6,
                linestyle='--', alpha=0.3, zorder=0))

    # ------------------------------------------------------------------
    # Layer 3: Gray lattice circles (size = spin magnitude)
    # Layer 4: Sz-colored spin markers
    # Layer 5: Black quiver arrows (Sxy direction)
    # ------------------------------------------------------------------
    # Base sizes (in scatter units)
    gray_base = 200      # gray background circle
    spin_base = 100      # Sz-colored marker on top

    for i, site in enumerate(system.sites):
        theta, phi_angle = site.angles
        spin_vec = _spin_direction(theta, phi_angle)
        sx, sy, sz = spin_vec
        sxy = np.sqrt(sx**2 + sy**2)
        sz_color = sz_cmap(sz_norm(sz))

        gray_size = gray_base * (site.spin / s_max)
        spin_size = spin_base * (site.spin / s_max)

        for n1, n2 in product(n_range_draw, n_range_draw):
            R = n1 * a1 + n2 * a2
            pos = site.position + R

            if not _point_in_bounds(pos[0], pos[1], bounds, margin=0.5):
                continue

            in_main = (-n_repeat <= n1 <= n_repeat and
                       -n_repeat <= n2 <= n_repeat)
            alpha = 1.0 if in_main else 0.3

            # Gray lattice circle (background)
            ax.scatter(pos[0], pos[1], c=['lightgray'], s=gray_size,
                       edgecolors='#AAAAAA', linewidths=0.5,
                       alpha=alpha * 0.6, zorder=3)

            # Sz-colored spin marker (on top of gray circle)
            ax.scatter(pos[0], pos[1], c=[sz_color], s=spin_size,
                       edgecolors='none',
                       alpha=alpha, zorder=4)

            if sxy > 1e-3:
                # Black quiver arrow for Sxy direction
                ax.quiver(pos[0], pos[1], sx, sy,
                          angles='xy', scale_units='xy',
                          scale=1.0 / arrow_scale,
                          color='black', alpha=alpha,
                          width=0.008, headwidth=4, headlength=3,
                          zorder=5)

                # Label opposite to arrow direction
                label_offset = min_dist * 0.25
                lx = -sx / sxy * label_offset
                ly = -sy / sxy * label_offset
                ax.text(pos[0] + lx, pos[1] + ly, site.label,
                        ha='center', va='center', fontsize=9,
                        fontweight='bold', alpha=alpha,
                        color='black', zorder=6)
            else:
                # Pure z-spin: colored dot
                dot_color = '#CC3333' if sz > 0 else '#3333CC'
                ax.scatter(pos[0], pos[1], c=[dot_color], s=25,
                           edgecolors='none', alpha=alpha, zorder=5)

                # Label offset downward for z-spin
                ax.text(pos[0], pos[1] - min_dist * 0.25, site.label,
                        ha='center', va='center', fontsize=9,
                        fontweight='bold', alpha=alpha,
                        color='black', zorder=6)

    # ------------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------------
    sm = plt.cm.ScalarMappable(cmap=sz_cmap, norm=sz_norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=20)
    cbar.set_label(r'$S_z$', fontsize=11)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.2, color='#CCCCCC')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)

    if title is None:
        label_str = f"'{system.label}' " if system.label else ""
        title = f"Spin Configuration {label_str}({system.num_sites} sublattices)"
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)

    # Legend: bond types only (sublattice info is in the polar plot)
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper left', fontsize=8,
                  framealpha=0.9, edgecolor='#CCCCCC')


# ======================================================================
# Polar angle distribution (right panel)
# ======================================================================

def _draw_polar_angles(ax, system):
    """Draw polar angle distribution of spins.

    Each sublattice is shown as a colored arrow from origin to the unit
    circle at angle theta. The arrow color uses coolwarm for Sz.
    """
    sz_cmap = plt.cm.coolwarm
    sz_norm = Normalize(vmin=-1, vmax=1)

    ax.set_title("Spin Angles", pad=15, fontsize=11, fontweight='bold')
    ax.plot(np.linspace(-np.pi, np.pi, 200), np.ones(200),
            '--', color='gray', alpha=0.4, linewidth=0.8)
    ax.set_rticks([])
    ax.set_rlim(0, 1.3)
    ax.set_theta_offset(np.pi / 2)  # 0 at top

    for i, site in enumerate(system.sites):
        theta, phi_angle = site.angles
        spin_vec = _spin_direction(theta, phi_angle)
        sz = spin_vec[2]
        sz_color = sz_cmap(sz_norm(sz))
        sub_color = SUBLATTICE_COLORS[i % len(SUBLATTICE_COLORS)]

        # Normalize theta to [0, 2*pi)
        theta_plot = np.mod(theta, 2 * np.pi)

        # Arrow from center to unit circle
        ax.annotate('', xy=(theta_plot, 1.0), xytext=(theta_plot, 0),
                    arrowprops=dict(arrowstyle='->', color=sz_color,
                                    lw=2.5, mutation_scale=15))

        # Sublattice label near arrow tip
        ax.text(theta_plot, 1.15, site.label, ha='center', va='center',
                fontsize=9, fontweight='bold', color=sub_color)
