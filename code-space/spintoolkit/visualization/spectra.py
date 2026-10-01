"""Neutron intensity figures: path maps, constant-energy slices and powder maps.

Draws the outputs of :func:`~spintoolkit.observables.neutron.neutron_path`,
:func:`~spintoolkit.observables.neutron.neutron_slice` and
:func:`~spintoolkit.observables.neutron.powder_average`. As in
:mod:`~spintoolkit.visualization.bands`, nothing is recomputed here.

- Units (D20): energies in E0, momenta in inverse model length units, the
  intensity per site without the instrument prefactor. A display unit for
  the energy needs an explicit ``energy_scale`` with its ``energy_label``.
- Undefined intensity (NaN: zero modes at Bragg vectors, ``Q = 0``) is drawn
  in ``undefined_color``, never interpolated or set to zero.
- The colour range defaults to the 99.5th percentile of the finite values,
  so that a few sharp peaks do not hide the rest of the spectrum; pass
  ``vmax`` to fix it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from spintoolkit.visualization.bands import DEFAULT_ENERGY_LABEL

DEFAULT_CMAP = "viridis"
UNDEFINED_COLOR = "0.75"


def _check_scale(scale: float, label: str) -> float:
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("energy_scale must be finite and positive")
    if scale != 1.0 and label == DEFAULT_ENERGY_LABEL:
        raise ValueError("energy_scale converts away from E0; give the matching energy_label")
    return scale


def _norm(data: np.ndarray, vmax: Optional[float], log: bool):
    finite = data[np.isfinite(data)]
    if vmax is None:
        vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    if vmax <= 0:
        vmax = 1.0
    if log:
        positive = finite[finite > 0]
        vmin = max(float(positive.min()) if positive.size else vmax * 1e-4, vmax * 1e-4)
        return LogNorm(vmin=vmin, vmax=vmax)
    return Normalize(vmin=0.0, vmax=vmax)


def _cmap(cmap, undefined_color):
    return plt.get_cmap(cmap).with_extremes(bad=undefined_color)


def _edges(centres: np.ndarray) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    if len(centres) == 1:
        return np.array([centres[0] - 0.5, centres[0] + 0.5])
    middle = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate([[2 * centres[0] - middle[0]], middle, [2 * centres[-1] - middle[-1]]])


def plot_intensity_path(spectrum, ax: Optional[plt.Axes] = None, *, energy_scale: float = 1.0,
                        energy_label: str = DEFAULT_ENERGY_LABEL, cmap=DEFAULT_CMAP,
                        vmax: Optional[float] = None, log: bool = False,
                        undefined_color=UNDEFINED_COLOR, colorbar: bool = True) -> plt.Axes:
    """Draw ``I(Q, w)`` along a momentum path.

    Parameters
    ----------
    spectrum : NeutronPath
        Output of :func:`~spintoolkit.observables.neutron.neutron_path`.
    ax : matplotlib.axes.Axes, optional
    energy_scale, energy_label
        As in :func:`~spintoolkit.visualization.bands.plot_bands`.
    cmap : str or Colormap
    vmax : float, optional
        Upper end of the colour range (default: 99.5th percentile).
    log : bool
        Logarithmic colour scale over four decades below ``vmax``.
    undefined_color : colour
        Colour of NaN cells.
    colorbar : bool

    Returns
    -------
    matplotlib.axes.Axes
    """
    scale = _check_scale(energy_scale, energy_label)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2))
    data = np.asarray(spectrum.intensity, dtype=float)
    mesh = ax.pcolormesh(_edges(spectrum.distance), _edges(np.asarray(spectrum.omega) * scale),
                         np.ma.masked_invalid(data.T), cmap=_cmap(cmap, undefined_color),
                         norm=_norm(data, vmax, log), shading="flat", rasterized=True)
    for d in spectrum.label_distances:
        ax.axvline(d, color="w", lw=0.6, ls="--", alpha=0.6)
    ax.set_xticks(np.asarray(spectrum.label_distances))
    ax.set_xticklabels(list(spectrum.labels))
    ax.set_xlim(spectrum.distance[0], spectrum.distance[-1])
    ax.set_ylabel(energy_label)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=r"$I(\mathbf{Q},\omega)$ (per site)")
    return ax


def plot_intensity_slice(spectrum, ax: Optional[plt.Axes] = None, *, cmap=DEFAULT_CMAP,
                         vmax: Optional[float] = None, log: bool = False,
                         undefined_color=UNDEFINED_COLOR, zones: bool = True,
                         colorbar: bool = True) -> plt.Axes:
    """Draw a constant-energy slice ``I(Q_x, Q_y, w)``.

    Parameters
    ----------
    spectrum : NeutronSlice
        Output of :func:`~spintoolkit.observables.neutron.neutron_slice`.
    zones : bool
        Outline the primitive Brillouin zones inside the window.
    Other parameters as in :func:`plot_intensity_path`.
    """
    from spintoolkit.system.high_symmetry import reciprocal_lattice, zone_boundary

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    data = np.asarray(spectrum.intensity, dtype=float)
    mesh = ax.pcolormesh(_edges(spectrum.q_x), _edges(spectrum.q_y), np.ma.masked_invalid(data),
                         cmap=_cmap(cmap, undefined_color), norm=_norm(data, vmax, log),
                         shading="flat", rasterized=True)
    if zones:
        corners = zone_boundary(spectrum.lattice)
        closed = np.vstack([corners, corners[:1]])
        b = reciprocal_lattice(spectrum.lattice)
        reach = max(abs(spectrum.q_x).max(), abs(spectrum.q_y).max())
        n = int(np.ceil(reach / min(np.linalg.norm(b, axis=1)))) + 1
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                shifted = closed + i * b[0] + j * b[1]
                ax.plot(shifted[:, 0], shifted[:, 1], color="w", lw=0.6, alpha=0.6)
    ax.set_xlim(spectrum.q_x[0], spectrum.q_x[-1])
    ax.set_ylim(spectrum.q_y[0], spectrum.q_y[-1])
    ax.set_aspect("equal")
    ax.set_xlabel(r"$Q_x$")
    ax.set_ylabel(r"$Q_y$")
    ax.set_title(rf"$\omega = {spectrum.energy:.4g}\,E_0$", fontsize=10)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=r"$I(\mathbf{Q},\omega)$ (per site)")
    return ax


def plot_powder(Q_magnitudes, omega, intensity, ax: Optional[plt.Axes] = None, *,
                energy_scale: float = 1.0, energy_label: str = DEFAULT_ENERGY_LABEL,
                cmap=DEFAULT_CMAP, vmax: Optional[float] = None, log: bool = False,
                undefined_color=UNDEFINED_COLOR, colorbar: bool = True) -> plt.Axes:
    """Draw a powder-averaged ``I(|Q|, w)`` map.

    Parameters
    ----------
    Q_magnitudes : (nQ,) array_like
    omega : (nw,) array_like
    intensity : (nQ, nw) array_like
        Output of :func:`~spintoolkit.observables.neutron.powder_average`.
    Other parameters as in :func:`plot_intensity_path`.
    """
    scale = _check_scale(energy_scale, energy_label)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.2))
    data = np.asarray(intensity, dtype=float)
    mesh = ax.pcolormesh(_edges(Q_magnitudes), _edges(np.asarray(omega) * scale),
                         np.ma.masked_invalid(data.T), cmap=_cmap(cmap, undefined_color),
                         norm=_norm(data, vmax, log), shading="flat", rasterized=True)
    ax.set_xlabel(r"$|\mathbf{Q}|$")
    ax.set_ylabel(energy_label)
    if colorbar:
        ax.figure.colorbar(mesh, ax=ax, label=r"$I(|\mathbf{Q}|,\omega)$ (per site)")
    return ax
