# Stage 7 visualization: design proposal (draft for review)

Status: proposal, nothing implemented. Approval needed before code (CLAUDE.md rule 1, API change rule 2).

## Principle

Figures only draw what an observable already computed. No physics is recomputed in the plotting layer,
so a figure can never disagree with the numbers in a result or a verification record.

## 1. Magnon band figure

`spintoolkit.visualization.plot_bands(bands, ax=None, *, energy_scale=1.0, energy_label="E / E0", mark_zero_modes=True)`

- Input: the `BandStructure` returned by `observables.bands.band_structure` (D33). Returns the matplotlib `Axes`;
  the library never calls `plt.show()`.
- Units (D20): energies stay in E0. A display unit (for NBCP, meV) is applied only by an explicit
  `energy_scale` with its `energy_label`; the scale is printed in the figure caption of the example, never hidden.
- Unstable points (NaN energies, indefinite H(k)): drawn as gaps, not interpolated, and the gap range is
  reported. An unstable reference state must be visible in the figure, not smoothed over.
- Zero modes (`zero_modes` mask, positive semidefinite H(k)): open markers at E = 0 on that k.
  Goldstone vs accidental is not decided by the figure; the label comes from `scan_zero_modes` if supplied.
- Folding: bands of the magnetic cell are drawn folded on the primitive-zone path, as `band_structure` returns them.
  No unfolding: unfolding needs spectral weights (the structure factor), which is a separate observable.
  Coloring by S(q, omega) weight can come later as an option.
- Vertex labels and dashed vertical lines from `labels` / `label_distances`.

## 2. Spin configuration figure

Port `visualization/spin_plotter.py` to take `(SpinModel, SpinState)` instead of the deprecated `SpinSystem`.
Drawing rules stay as they are (Sz color, in-plane arrow, bonds by exchange matrix). The old signature keeps
working with a DeprecationWarning until the public-release cleanup (same treatment as D30).

## 3. Not in this stage

- Berry-curvature coloring of bands (legacy `_plot_band_structure_w_berry_curvature`): only meaningful where the
  D31 admissibility test passes; proposed as a later option that greys out undefined points rather than coloring them.
- Thermal and structure-factor plots: after the band figure is reviewed.

## 4. Review and verification

- Beamer review slides in `docs/development/` (interface, what each figure shows, style), as stage 4d and 6 did.
- Tests: the plotted line data equals `BandStructure.energies` exactly; NaN stays NaN; zero-mode markers sit at the
  masked k; square Neel and triangular 120 deg benchmark figures in the verification record.

## 5. Then: NBCP magnon bands (item 3)

`examples/nbcp_magnon_bands.py`: band figure for the NBCP phases at the field values used in arXiv:2505.06398, in meV
via the model's E0, compared side by side with the paper's figure. Differences are reported, not tuned away.
