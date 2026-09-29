"""Physical constants in the meV convention.

The common model (``spintoolkit.system.model``) computes dimensionlessly in the
energy unit E0 of its coefficients. Physical units are tesla (field), kelvin
(temperature) and meV (energy); these constants convert between them, e.g.
``field = MU_B_MEV_PER_T * B[T] / E0[meV]`` and
``temperature = K_BOLTZMANN_MEV * T[K] / E0[meV]``. The existing LSWT modules
still take temperatures in kelvin and energies in meV.
"""

# Boltzmann constant in meV/K
K_BOLTZMANN_MEV = 8.617333262e-2

# Reduced Planck constant in meV·s
H_BAR_MEV = 6.582119569e-13

# Bohr magneton in meV/T (CODATA 2018)
MU_B_MEV_PER_T = 5.7883818060e-2
