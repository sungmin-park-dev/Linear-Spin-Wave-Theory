"""Figures drawn from computed results.

Magnon bands, neutron spectra (path maps, constant-energy slices, powder
maps), Berry curvature and thermal Hall, spin textures with their skyrmion
density, phase diagrams, thermodynamic curves and spin configurations. Every
figure draws stored numbers and shows undefined values as undefined.
"""

from spintoolkit.visualization.bands import plot_bands
from spintoolkit.visualization.phases import plot_phase_diagram, plot_thermodynamics
from spintoolkit.visualization.spectra import plot_intensity_path, plot_intensity_slice, plot_powder
from spintoolkit.visualization.spin_plotter import plot_spin_configuration
from spintoolkit.visualization.texture import plot_spin_texture
from spintoolkit.visualization.topology import plot_berry_curvature, plot_thermal_hall

__all__ = ["plot_bands", "plot_intensity_path", "plot_intensity_slice", "plot_powder",
           "plot_berry_curvature", "plot_thermal_hall", "plot_spin_texture",
           "plot_phase_diagram", "plot_thermodynamics", "plot_spin_configuration"]
