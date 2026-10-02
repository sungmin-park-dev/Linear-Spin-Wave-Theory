"""Phase diagram and thermodynamic curve figures.

- :func:`plot_phase_diagram` colours a rectangular grid of parameter points
  by the name of the winning state, for example the first entry of
  :func:`~spintoolkit.methods.phase_competition.compare_states` at each point.
  A point without a defined winner (``None``, e.g. no candidate is LSWT
  stable) is drawn in ``undefined_color`` and listed in the legend as
  "undefined"; it is never assigned to a neighbouring phase.
- :func:`plot_thermodynamics` draws the quantities of a
  :class:`~spintoolkit.observables.thermal.ThermalResult` against
  ``t = k_B T / E0``. NaN values (divergent boson numbers of gapless 2D
  spectra) stay gaps, and temperatures where some ``<n_i> > S_i``
  (``beyond_lswt``) are shaded, because the expansion has broken down there.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from spintoolkit.visualization.bands import _unstable_intervals

UNDEFINED_COLOR = "0.85"

_THERMAL_LABELS = {
    "free_energy": r"$F$ per site $(E_0)$",
    "internal_energy": r"$U$ per site $(E_0)$",
    "entropy": r"$S$ per site $(k_B)$",
    "specific_heat": r"$C$ per site $(k_B)$",
    "magnetization": r"$m_\alpha$ per site",
}


def _edges(centres) -> np.ndarray:
    centres = np.asarray(centres, dtype=float)
    if len(centres) == 1:
        return np.array([centres[0] - 0.5, centres[0] + 0.5])
    middle = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate([[2 * centres[0] - middle[0]], middle, [2 * centres[-1] - middle[-1]]])


def plot_phase_diagram(x, y, phases, ax: Optional[plt.Axes] = None, *,
                       colors: Optional[Mapping[str, object]] = None,
                       xlabel: str = "", ylabel: str = "",
                       undefined_color=UNDEFINED_COLOR, legend: bool = True,
                       boundaries: bool = True) -> plt.Axes:
    """Draw a phase diagram on a rectangular grid.

    Parameters
    ----------
    x : (nx,) array_like
        Values of the horizontal parameter (grid-point centres, ascending).
    y : (ny,) array_like
        Values of the vertical parameter.
    phases : (ny, nx) array_like of str or None
        Name of the winning state at each point; ``None`` where undefined.
    ax : matplotlib.axes.Axes, optional
    colors : mapping name -> colour, optional
        Default: matplotlib's ``tab10`` cycle in order of first appearance.
    xlabel, ylabel : str
    undefined_color : colour
    legend : bool
    boundaries : bool
        Draw black lines between cells of different phases.

    Returns
    -------
    matplotlib.axes.Axes
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    phases = np.asarray(phases, dtype=object)
    if phases.shape != (len(y), len(x)):
        raise ValueError(f"phases must have shape (len(y), len(x)) = {(len(y), len(x))}")
    names = []
    for name in phases.ravel():
        if name is not None and name not in names:
            names.append(name)
    palette = plt.get_cmap("tab10")
    colors = dict(colors or {})
    for i, name in enumerate(names):
        colors.setdefault(name, palette(i % 10))
    index = {name: i + 1 for i, name in enumerate(names)}
    codes = np.vectorize(lambda p: 0 if p is None else index[p], otypes=[int])(phases)
    cmap = ListedColormap([undefined_color] + [colors[n] for n in names])
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.2))
    xe, ye = _edges(x), _edges(y)
    ax.pcolormesh(xe, ye, codes, cmap=cmap, vmin=-0.5, vmax=len(names) + 0.5, shading="flat")
    if boundaries:
        for j in range(len(y)):
            for i in range(len(x)):
                if i + 1 < len(x) and codes[j, i] != codes[j, i + 1]:
                    ax.plot([xe[i + 1]] * 2, [ye[j], ye[j + 1]], color="k", lw=0.9)
                if j + 1 < len(y) and codes[j, i] != codes[j + 1, i]:
                    ax.plot([xe[i], xe[i + 1]], [ye[j + 1]] * 2, color="k", lw=0.9)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        handles = [Patch(facecolor=colors[n], edgecolor="k", label=str(n)) for n in names]
        if np.any(codes == 0):
            handles.append(Patch(facecolor=undefined_color, edgecolor="k", label="undefined"))
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
                  frameon=False)
    return ax


def plot_thermodynamics(thermal, quantities: Sequence[str] = ("specific_heat", "entropy"),
                        axes=None, *, temperature_scale: float = 1.0,
                        temperature_label: str = r"$k_B T / E_0$", **line):
    """Draw thermodynamic quantities of an LSWT result against temperature.

    Parameters
    ----------
    thermal : ThermalResult
        Output of :func:`~spintoolkit.observables.thermal.thermal_quantities`.
    quantities : sequence of str
        Any of ``free_energy``, ``internal_energy``, ``entropy``,
        ``specific_heat`` and ``magnetization`` (three components), one panel each.
    axes : sequence of matplotlib.axes.Axes, optional
    temperature_scale, temperature_label
        Display conversion of ``t``; a scale other than one needs its label.
    **line
        Passed to ``Axes.plot``.

    Returns
    -------
    list of matplotlib.axes.Axes
    """
    unknown = [q for q in quantities if q not in _THERMAL_LABELS]
    if unknown:
        raise ValueError(f"unknown quantities {unknown}; known: {sorted(_THERMAL_LABELS)}")
    scale = float(temperature_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("temperature_scale must be finite and positive")
    if scale != 1.0 and temperature_label == r"$k_B T / E_0$":
        raise ValueError("temperature_scale converts away from k_B T / E0; give the matching "
                         "temperature_label")
    if axes is None:
        _, axes = plt.subplots(1, len(quantities), figsize=(4.2 * len(quantities), 3.4),
                               squeeze=False)
        axes = list(axes[0])
    t = np.asarray(thermal.temperatures, dtype=float) * scale
    beyond = np.asarray(getattr(thermal, "beyond_lswt", np.zeros(len(t), bool)), dtype=bool)
    for ax, quantity in zip(axes, quantities):
        values = np.asarray(getattr(thermal, quantity), dtype=float)
        if quantity == "magnetization":
            for component, name in enumerate("xyz"):
                ax.plot(t, values[:, component], label=rf"$m_{name}$", **line)
            ax.legend(fontsize=8, frameon=False)
        else:
            ax.plot(t, values, **line)
        for start, stop in _unstable_intervals(t, beyond):
            ax.axvspan(start, stop, color="0.9", zorder=0, lw=0)
            ax.text(start, 1.0, r" $\langle n\rangle > S$", transform=ax.get_xaxis_transform(),
                    va="top", fontsize=7, color="0.4")
        ax.set_xlabel(temperature_label)
        ax.set_ylabel(_THERMAL_LABELS[quantity])
    return axes


def plot_magnetization_curve(curve, axes=None, *, field_scale: float = 1.0,
                             field_label: str = r"$h = \mu_B B / E_0$"):
    """Draw M(h) at classical and harmonic order, and the ordered moment per site.

    Parameters
    ----------
    curve : MagnetizationCurve
        Output of :func:`~spintoolkit.methods.magnetization.magnetization_curve`.
    axes : two matplotlib.axes.Axes, optional
        Panels for M(h) and for ``S_i - <n_i>``; a single Axes draws only M(h).
    field_scale, field_label
        Display conversion of ``h``; a scale other than one needs its label.

    Returns
    -------
    list of matplotlib.axes.Axes
    """
    scale = float(field_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("field_scale must be finite and positive")
    if scale != 1.0 and field_label == r"$h = \mu_B B / E_0$":
        raise ValueError("field_scale converts away from mu_B B / E0; give the matching "
                         "field_label")
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes = list(np.atleast_1d(axes))
    h = np.asarray(curve.fields) * scale
    axes[0].plot(h, curve.classical, color="0.4", ls="--", label="classical")
    axes[0].plot(h, curve.harmonic, color="C0", marker="o", ms=3, label="harmonic (1/S)")
    axes[0].set_xlabel(field_label)
    axes[0].set_ylabel(r"$M$ per site $(\mu_B)$")
    axes[0].legend(fontsize=8, frameon=False)
    if len(axes) > 1:
        moments = curve.ordered_moments
        for i in range(moments.shape[1]):
            axes[1].plot(h, moments[:, i], marker="o", ms=3, label=f"site {i}")
        axes[1].axhline(float(np.max(curve.spins)), color="0.6", lw=0.6, ls=":")
        axes[1].set_xlabel(field_label)
        axes[1].set_ylabel(r"ordered moment $S_i - \langle n_i\rangle$")
        if moments.shape[1] <= 6:
            axes[1].legend(fontsize=7, frameon=False)
    return axes
