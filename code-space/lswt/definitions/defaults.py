"""Default calculation parameters and numerical thresholds."""

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIME = 0.0
DEFAULT_OMEGA = 0.0
DEFAULT_ETA = 1e-3
DEFAULT_TOLERANCE = 1e-8
DEFAULT_DELTA_PEAK = 100
DEFAULT_LEVEL_SPACING = 1e-2       # meV, verbose mesh advisory; not a degeneracy cutoff
DEFAULT_BAND_GAP_CUTOFF = 1e-8     # meV, configurable calculation policy; not an error bound
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

