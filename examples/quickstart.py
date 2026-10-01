"""Quick start: the spin-1/2 triangular-lattice Heisenberg antiferromagnet.

The same code is shown in the README. Energies are in units of J.

1. Model and classical state: ``SpinModel`` holds the Hamiltonian, the
   120-degree state lives on a three-site magnetic cell.
2. LSWT: ``solve_lswt`` gives E_cl + E_zp = -0.5388 J per spin
   (Chubukov, Sachdev and Senthil, J. Phys.: Condens. Matter 6, 8891 (1994)).
3. Observables: magnon bands along Gamma-K-M-Gamma and the neutron intensity
   at the M point; finite-temperature thermodynamics of a gapped ferromagnet
   in a field.

Usage
-----
    python examples/quickstart.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code-space'))

import numpy as np

import spintoolkit as stk
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import polarized_state, state_120, triangular_heisenberg
from spintoolkit.observables.bands import band_structure, high_symmetry_points
from spintoolkit.observables.neutron import neutron_intensity
from spintoolkit.observables.thermal import thermal_quantities

# 1. Model (J = 1, S = 1/2) and the 120-degree state.
model = triangular_heisenberg(J=1.0, S=0.5)
state = state_120(model)

# 2. Linear spin-wave theory on a 24 x 24 mesh of the magnetic zone.
result = solve_lswt(model, state, settings=LSWTSettings(mesh=(24, 24)))
print(f"E_cl + E_zp = {result.ground_state_energy:.4f} J per spin")

# 3a. Magnon bands along Gamma-K-M-Gamma of the primitive zone.
bands = band_structure(result, ("Γ", "K", "M", "Γ"))
print(f"band maximum = {np.nanmax(bands.energies):.4f} J")

# 3b. Unpolarized neutron intensity at M (in-plane Q, g = 2, no form factor).
#     K is a Bragg vector of the 120-degree order: its Goldstone weight is NaN.
M = high_symmetry_points(model.lattice)["M"]
spectrum = neutron_intensity(result, [[M[0], M[1], 0.0]], g=2.0)
omega = np.linspace(0.0, 3.0, 301)
line = spectrum.broaden(omega, fwhm=0.05)
print(f"intensity peak at M: omega = {omega[np.nanargmax(line[0])]:.2f} J")

# 3c. Thermodynamics of the ferromagnet (J = -1) in a field h = 0.5 along z.
ferro = triangular_heisenberg(J=-1.0, S=0.5)
field = stk.ExternalConditions(field=(0.0, 0.0, 0.5))
gapped = solve_lswt(ferro, polarized_state(ferro), field)
thermo = thermal_quantities(gapped, [0.1, 0.5])
print("specific heat per spin at t = 0.1, 0.5:", np.round(thermo.specific_heat, 4))
