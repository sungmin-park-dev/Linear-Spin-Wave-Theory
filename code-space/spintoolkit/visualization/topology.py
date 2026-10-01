"""Berry curvature and thermal Hall figures.

Draws :class:`~spintoolkit.observables.berry.BerryCurvature` and
:class:`~spintoolkit.observables.berry.ThermalHall`. Nothing is recomputed
except the periodic images of the stored momenta used to fill the zone.

- The curvature is periodic in the reciprocal lattice of the magnetic cell,
  so the stored mesh is repeated by reciprocal vectors and drawn inside the
  first Brillouin zone (Wigner-Seitz cell) with a symmetric colour scale.
- Where a band is not separated by more than ``band_gap_cutoff`` its
  curvature is NaN (undefined) and is drawn in ``undefined_color``; the
  Chern number is then NaN as well and is printed as "undefined".
- Thermal Hall curves are ``kappa_xy / T`` in ``k_B^2 / hbar`` per layer
  against ``t = k_B T / E0`` (D29). Undefined temperatures (gapless spectra)
  stay gaps in the line.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.tri import Triangulation

from spintoolkit.system.high_symmetry import reciprocal_lattice, zone_boundary

UNDEFINED_COLOR = "0.75"


def _in_zone(k: np.ndarray, reciprocal: np.ndarray, margin: float) -> np.ndarray:
    """Mask of momenta inside the Wigner-Seitz cell enlarged by ``margin`` (relative)."""
    inside = np.ones(len(k), dtype=bool)
    for i in range(-2, 3):
        for j in range(-2, 3):
            if (i, j) == (0, 0):
                continue
            G = i * reciprocal[0] + j * reciprocal[1]
            inside &= k @ G <= (1 + margin) * (G @ G) / 2
    return inside


def _chern_text(curvature) -> str:
    if not curvature.full_zone:
        return ""
    values = curvature.chern_numbers()
    return ", ".join("undefined" if not np.isfinite(c) else f"{c:+.3f}" for c in values)


def plot_berry_curvature(curvature, lattice, band: int = 0, ax: Optional[plt.Axes] = None, *,
                         cmap="RdBu_r", vmax: Optional[float] = None,
                         undefined_color=UNDEFINED_COLOR, colorbar: bool = True) -> plt.Axes:
    """Draw the Berry curvature of one band over the magnetic Brillouin zone.

    Parameters
    ----------
    curvature : BerryCurvature
        Output of :func:`~spintoolkit.observables.berry.berry_curvature`.
    lattice : (2, 2) array_like
        Magnetic lattice of the result (``result.magnetic_lattice``).
    band : int
        Band index in ascending energy order.
    ax : matplotlib.axes.Axes, optional
    cmap : str or Colormap
        Diverging map; zero curvature is the centre colour.
    vmax : float, optional
        Colour range ``[-vmax, vmax]`` (default: largest finite ``|Omega|``).
    undefined_color : colour
        Colour of momenta where the band is not separated (NaN).
    colorbar : bool

    Returns
    -------
    matplotlib.axes.Axes
        The title lists the Chern numbers of all bands when the result holds
        a complete mesh.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    lattice = np.asarray(lattice, dtype=float)
    reciprocal = reciprocal_lattice(lattice)
    k = np.asarray(curvature.k_points, dtype=float)
    omega = np.asarray(curvature.curvature, dtype=float)[:, band]
    images = [(k + i * reciprocal[0] + j * reciprocal[1], omega)
              for i in range(-2, 3) for j in range(-2, 3)]
    points = np.vstack([p for p, _ in images])
    values = np.concatenate([v for _, v in images])
    keep = _in_zone(points, reciprocal, margin=0.15)
    points, values = points[keep], values[keep]

    finite = values[np.isfinite(values)]
    if vmax is None:
        vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    colormap = plt.get_cmap(cmap).with_extremes(bad=undefined_color)
    triangulation = Triangulation(points[:, 0], points[:, 1])
    corner_values = values[triangulation.triangles]
    triangulation.set_mask(~np.all(np.isfinite(corner_values), axis=1))
    face = corner_values.mean(axis=1)               # NaN faces are masked above
    zone = zone_boundary(lattice)
    closed = np.vstack([zone, zone[:1]])
    ax.fill(closed[:, 0], closed[:, 1], color=undefined_color, zorder=0, lw=0)
    mesh = ax.tripcolor(triangulation, facecolors=np.nan_to_num(face), cmap=colormap,
                        norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax),
                        rasterized=True, zorder=1)
    clip = plt.Polygon(closed, transform=ax.transData)
    mesh.set_clip_path(clip)
    ax.plot(closed[:, 0], closed[:, 1], color="k", lw=0.8, zorder=2)
    reach = 1.05 * float(np.max(np.abs(zone)))
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    chern = _chern_text(curvature)
    title = f"band {band}" + (f"; Chern numbers {chern}" if chern else "")
    ax.set_title(title, fontsize=9)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=r"$\Omega_n(\mathbf{k})$")
    return ax


def plot_thermal_hall(hall, ax: Optional[plt.Axes] = None, *, band_sum: bool = False,
                      temperature_scale: float = 1.0,
                      temperature_label: str = r"$k_B T / E_0$", **line) -> plt.Axes:
    """Draw ``kappa_xy / T`` (units of ``k_B^2 / hbar`` per layer) against temperature.

    Parameters
    ----------
    hall : ThermalHall
        Output of :func:`~spintoolkit.observables.berry.thermal_hall`.
    ax : matplotlib.axes.Axes, optional
    band_sum : bool
        Also draw the per-band sum (dashed); it is NaN where a band is not
        separated and agrees with the pair form otherwise.
    temperature_scale, temperature_label
        Display conversion of ``t``; a scale other than one needs its label.
    **line
        Passed to ``Axes.plot`` for the pair form.
    """
    scale = float(temperature_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("temperature_scale must be finite and positive")
    if scale != 1.0 and temperature_label == r"$k_B T / E_0$":
        raise ValueError("temperature_scale converts away from k_B T / E0; give the matching "
                         "temperature_label")
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 3.8))
    t = np.asarray(hall.temperatures, dtype=float) * scale
    ax.plot(t, hall.kappa_over_t, marker="o", ms=3, **line)
    if band_sum:
        ax.plot(t, hall.kappa_over_t_band_sum, ls="--", color="0.3", label="band sum")
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.set_xlabel(temperature_label)
    ax.set_ylabel(r"$\kappa_{xy}/T\ \ (k_B^2/\hbar)$")
    return ax
