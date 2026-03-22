"""Constants and default parameters for LSWT calculations.

All physical constants, numerical thresholds, and default parameter values
used across the LSWT package are defined here for consistency.
"""

import numpy as np

# =============================================================================
# Physical constants
# =============================================================================

# Boltzmann constant in meV/K
K_BOLTZMANN_MEV = 8.617333262e-2

# Reduced Planck constant in meV·s
H_BAR_MEV = 6.582119569e-13

# =============================================================================
# Default solver / observable parameters
# =============================================================================

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIME = 0.0
DEFAULT_OMEGA = 0.0
DEFAULT_ETA = 1e-3
DEFAULT_TOLERANCE = 1e-8
DEFAULT_DELTA_PEAK = 100
DEFAULT_LEVEL_SPACING = 1e-2       # meV, minimum band spacing for topology
DEFAULT_INVALID_EXCLUDE = True

# =============================================================================
# Bose-Einstein numerical thresholds
# =============================================================================

BETA_E_THRESHOLD = 700             # overflow guard for exp(beta*E)
BETA_E_SMALL = 1e-10               # Taylor-expansion cutoff

# =============================================================================
# Hamiltonian numerical thresholds
# =============================================================================

HIGH_BE_THRESHOLD = 35.0
LOW_BE_THRESHOLD = 0.1
ZERO_ENERGY_THRESHOLD = 1e-15

# =============================================================================
# Diagonalizer numerical thresholds
# =============================================================================

TOLERANCE_DEFAULT = 1e-10
EPSILON_DEFAULT = 1e-6
THRESHOLD_DEFAULT = 1e-8

# =============================================================================
# Transformation matrices
# =============================================================================

# Ladder operators to Cartesian: (S+, S-, Sz) -> (Sx, Sy, Sz)
Mat_C = np.array([
    [1 / np.sqrt(2), 1 / np.sqrt(2), 0],       # (S+ + S-)/2   -> x
    [1j / np.sqrt(2), -1j / np.sqrt(2), 0],     # (S+ - S-)/2j  -> y
    [0, 0, 1],                                    # Sz            -> z
])
