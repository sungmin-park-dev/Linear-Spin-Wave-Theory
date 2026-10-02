"""Figures drawn from computed results.

Magnon bands and density of states; neutron spectra (path maps,
constant-energy slices, powder maps, energy-integrated maps with Bragg
peaks, single spin components); magnetization curves; Berry curvature and
thermal Hall; spin textures with their skyrmion density; phase diagrams;
thermodynamic curves; spin configurations. Every figure draws stored
numbers and shows undefined values as undefined.
"""

from spintoolkit.visualization.bands import plot_bands
from spintoolkit.visualization.phases import (plot_magnetization_curve, plot_phase_diagram,
                                              plot_thermodynamics)
from spintoolkit.visualization.spectra import (plot_density_of_states, plot_intensity_path,
                                               plot_intensity_slice, plot_powder,
                                               plot_static_structure_factor)
from spintoolkit.visualization.spin_plotter import plot_spin_configuration
from spintoolkit.visualization.texture import plot_spin_texture
from spintoolkit.visualization.topology import plot_berry_curvature, plot_thermal_hall

__all__ = ["plot_bands", "plot_density_of_states", "plot_static_structure_factor",
           "plot_magnetization_curve", "plot_intensity_path", "plot_intensity_slice", "plot_powder",
           "plot_berry_curvature", "plot_thermal_hall", "plot_spin_texture",
           "plot_phase_diagram", "plot_thermodynamics", "plot_spin_configuration"]
