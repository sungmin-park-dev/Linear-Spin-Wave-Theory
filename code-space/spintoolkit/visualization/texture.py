"""Spin texture figure: in-plane arrows, out-of-plane colour and solid-angle density (D42).

Top view of a ``SpinState`` over a patch of primitive cells. Each spin is an
arrow of its in-plane component on a dot coloured by ``n_z``. Optionally
every elementary triangle of the periodic triangulation
(:func:`~spintoolkit.observables.texture.lattice_triangles`) is filled with
its signed Berg-Luescher solid angle ``Omega``, the lattice skyrmion density
whose sum over a magnetic cell is ``4 pi Q``. Triangles where ``Omega`` is
undefined (three spins on a great circle outside one hemisphere, e.g. the
coplanar 120 degree state) are hatched, not coloured as zero. The title gives
``Q`` per magnetic cell, or "undefined".
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm

from spintoolkit.observables.texture import lattice_triangles, skyrmion_charge, solid_angle


def plot_spin_texture(model, state, ax: Optional[plt.Axes] = None, *,
                      cells: Optional[Sequence[int]] = None, solid_angles: bool = True,
                      triangles=None, arrow_scale: float = 0.4,
                      spin_cmap="coolwarm", density_cmap="PuOr_r",
                      colorbar: bool = True) -> plt.Axes:
    """Draw a spin texture with its solid-angle (skyrmion) density.

    Parameters
    ----------
    model : SpinModel
    state : SpinState
    ax : matplotlib.axes.Axes, optional
    cells : (2,) ints, optional
        Number of primitive cells drawn along each lattice vector (default:
        about three magnetic cells in each direction, at least six).
    solid_angles : bool
        Fill the triangles with ``Omega``.
    triangles : optional
        Elementary triangles, as for :func:`~spintoolkit.observables.texture.skyrmion_charge`.
    arrow_scale : float
        Arrow length of a fully in-plane spin, in units of the shortest
        lattice vector.
    spin_cmap, density_cmap : str or Colormap
        Colours of ``n_z`` (dots) and of ``Omega`` (triangles).
    colorbar : bool

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.2))
    if cells is None:
        n = max(6, 3 * math.ceil(math.sqrt(state.num_cells)))
        cells = (n, n)
    n1, n2 = int(cells[0]), int(cells[1])
    a = float(np.min(np.linalg.norm(model.lattice, axis=1)))

    if solid_angles:
        triangles = lattice_triangles(model) if triangles is None else [tuple(t) for t in triangles]
        polygons, omegas = [], []
        for c1 in range(n1):
            for c2 in range(n2):
                for triangle in triangles:
                    vertices = [(site, (c1 + c[0], c2 + c[1])) for site, c in triangle]
                    polygons.append([model.cartesian_position(s, c) for s, c in vertices])
                    omegas.append(solid_angle(*[state.direction(s, c) for s, c in vertices]))
        omegas = np.array(omegas)
        defined = np.isfinite(omegas)
        limit = float(np.max(np.abs(omegas[defined]))) if np.any(defined) else 1.0
        limit = limit if limit > 1e-12 else 1.0
        density = PolyCollection([p for p, d in zip(polygons, defined) if d],
                                 array=omegas[defined], cmap=density_cmap,
                                 norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
                                 edgecolors="face", linewidths=0.2, zorder=0)
        ax.add_collection(density)
        if not np.all(defined):
            ax.add_collection(PolyCollection([p for p, d in zip(polygons, defined) if not d],
                                             facecolors="none", edgecolors="0.5", hatch="//",
                                             linewidths=0.3, zorder=0))
        if colorbar and np.any(defined):
            ax.figure.colorbar(density, ax=ax, label=r"solid angle $\Omega$", shrink=0.8)

    positions, directions = [], []
    for c1 in range(n1):
        for c2 in range(n2):
            for site in model.sites:
                positions.append(model.cartesian_position(site.id, (c1, c2)))
                directions.append(state.direction(site.id, (c1, c2)))
    positions, directions = np.array(positions), np.array(directions)
    dots = ax.scatter(positions[:, 0], positions[:, 1], c=directions[:, 2], cmap=spin_cmap,
                      norm=Normalize(-1.0, 1.0), s=28, edgecolors="k", linewidths=0.4, zorder=2)
    ax.quiver(positions[:, 0], positions[:, 1], directions[:, 0], directions[:, 1],
              angles="xy", scale_units="xy", scale=1.0 / (arrow_scale * a), pivot="middle",
              width=0.004, color="k", zorder=3)
    if colorbar:
        ax.figure.colorbar(dots, ax=ax, label=r"$n_z$", shrink=0.8)

    charge = skyrmion_charge(model, state, triangles=triangles if solid_angles else None)
    if charge.integer is not None:
        text = f"Q = {charge.integer} per magnetic cell"
    elif np.isfinite(charge.charge):
        text = f"Q = {charge.charge:.4f} per magnetic cell (not an integer)"
    else:
        text = f"Q undefined ({charge.undefined} undefined triangles per magnetic cell)"
    ax.set_title(text, fontsize=9)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.03)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
