"""Magnon band figure (stage 7).

Draws a :class:`~spintoolkit.observables.bands.BandStructure` as returned by
:func:`~spintoolkit.observables.bands.band_structure`. Nothing is recomputed
here: the plotted lines are the stored energies times the display scale, so a
figure cannot disagree with the numbers of the result.

- Units (D20): energies are in E0. A display unit is applied only through an
  explicit ``energy_scale`` together with its ``energy_label``.
- Unstable momenta (NaN energies, indefinite ``H(k)``) stay gaps in the lines
  and are shaded; they are never interpolated over.
- Zero modes (``H(k)`` only positive semidefinite) are marked at ``E = 0``.
  Whether a zero mode is a Goldstone mode is not decided by the figure.
- Bands of a magnetic cell are drawn folded, as ``band_structure`` returns them.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_ENERGY_LABEL = r"$E\ /\ E_0$"


def _unstable_intervals(distance: np.ndarray, unstable: np.ndarray) -> List[Tuple[float, float]]:
    """Path-length intervals covering consecutive unstable points (half a step on each side)."""
    if not np.any(unstable):
        return []
    edges = np.concatenate([[distance[0]], 0.5 * (distance[1:] + distance[:-1]), [distance[-1]]])
    intervals, start = [], None
    for i, flag in enumerate(unstable):
        if flag and start is None:
            start = edges[i]
        if not flag and start is not None:
            intervals.append((float(start), float(edges[i])))
            start = None
    if start is not None:
        intervals.append((float(start), float(edges[-1])))
    return intervals


def plot_bands(bands, ax: Optional[plt.Axes] = None, *, energy_scale: float = 1.0,
               energy_label: str = DEFAULT_ENERGY_LABEL, mark_zero_modes: bool = True,
               color: Optional[str] = "C0", linewidth: float = 1.4) -> plt.Axes:
    """Draw magnon bands along a Brillouin-zone path.

    Parameters
    ----------
    bands : BandStructure
        Output of :func:`~spintoolkit.observables.bands.band_structure`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (default: a new figure).
    energy_scale : float, optional
        Display factor applied to the energies (E0 to the display unit);
        must be finite and positive. A value other than one requires a
        matching ``energy_label``.
    energy_label : str, optional
        Label of the energy axis (default ``E / E0``).
    mark_zero_modes : bool, optional
        Mark points whose ``H(k)`` is only positive semidefinite at ``E = 0``.
    color : str or None, optional
        Line colour of all bands (None: matplotlib's cycle per band).
    linewidth : float, optional

    Returns
    -------
    matplotlib.axes.Axes

    Raises
    ------
    ValueError
        Invalid ``energy_scale``, or a scale other than one with the default label.
    """
    scale = float(energy_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("energy_scale must be finite and positive")
    if scale != 1.0 and energy_label == DEFAULT_ENERGY_LABEL:
        raise ValueError("energy_scale converts away from E0; give the matching energy_label")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.2))

    distance = np.asarray(bands.distance)
    energies = np.asarray(bands.energies) * scale
    unstable = np.any(np.isnan(energies), axis=1)

    for start, stop in _unstable_intervals(distance, unstable):
        ax.axvspan(start, stop, color="0.85", zorder=0, lw=0)
    for n in range(energies.shape[1]):
        ax.plot(distance, energies[:, n], color=color, lw=linewidth, zorder=2)
    zero = np.asarray(bands.zero_modes, dtype=bool)
    if mark_zero_modes and np.any(zero):
        ax.plot(distance[zero], np.zeros(int(zero.sum())), ls="none", marker="o", ms=5,
                mfc="none", mec="k", mew=1.0, zorder=3, label="zero mode")

    for d in bands.label_distances:
        ax.axvline(d, color="0.6", lw=0.6, ls="--", zorder=1)
    ax.set_xticks(np.asarray(bands.label_distances))
    ax.set_xticklabels(list(bands.labels))
    ax.set_xlim(distance[0], distance[-1])
    if np.any(np.isfinite(energies)):
        ax.set_ylim(min(0.0, float(np.nanmin(energies))), 1.05 * float(np.nanmax(energies)))
    ax.set_ylabel(energy_label)
    return ax
