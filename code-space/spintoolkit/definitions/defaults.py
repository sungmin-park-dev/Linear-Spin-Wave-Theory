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


# =============================================================================
# Classical refinement and zero-point state selection (D17, D19)
# =============================================================================

CLASSICAL_REFINE_GTOL = 1e-14      # E0 per radian per site, L-BFGS-B gradient tolerance
CLASSICAL_REFINE_NEWTON_STEPS = 5  # Newton steps after L-BFGS-B

SELECTION_MODE = "physics"         # "physics" or "fixed"
SELECTION_GAP_RATIO = 1e-3         # fixed mode: w_min / w_next below this is a null mode
SELECTION_RANK_TOLERANCE = 1e-4    # fixed mode: relative generator singular value treated as zero
SELECTION_RESIDUAL_FACTOR = 10.0   # harmonic amplitude must exceed this times the fit residual
SELECTION_ROUNDOFF_FACTOR = 1e3    # multiples of machine epsilon times |E| treated as round-off
SELECTION_ACCURACY_FACTOR = 10.0   # physics mode: multiples of the estimated state accuracy
SELECTION_COMPETITION_BAND = (0.1, 10.0)  # physics mode: C_cl / C_qm range called competition
SELECTION_ORBIT_POINTS = 36        # equally spaced orbit samples of E_qm
SELECTION_MAX_HARMONIC = 12        # highest Fourier harmonic fitted along the orbit

# =============================================================================
# Exact diagonalization (stage 3)
# =============================================================================

ED_DENSE_LIMIT = 2000              # largest block diagonalized densely
ED_LANCZOS_MIN_DIMENSION = 256     # smaller blocks are always diagonalized densely
ED_SYMMETRY_TOLERANCE = 1e-12      # relative size of sector-changing coefficients treated as round-off

# =============================================================================
# LSWT on the common model types (stage 4)
# =============================================================================

LSWT_DEFAULT_MESH = (24, 24)       # thermodynamic-limit mesh of the magnetic reciprocal cell
LSWT_STATIONARITY_TOLERANCE = 1e-8 # E0, largest torque accepted without a warning
LSWT_ZERO_MODE_TOLERANCE = 1e-10   # min eig H(k) / max |eig H(k)| at or below this is a zero mode

# =============================================================================
# Zero-mode scan and finite temperature (stage 4b)
# =============================================================================

ZERO_MODE_ZERO_TOLERANCE = 1e-12       # lambda_min(H) / scale at or below: numerically zero
ZERO_MODE_CANDIDATE_TOLERANCE = 1e-6   # up to this: candidate, the user decides
ZERO_MODE_MESH_SEEDS = 8               # lowest mesh points used as scan starting points
ZERO_MODE_SEARCH_THRESHOLD = 1e-2      # minimize lambda_min only from points at or below this
ZERO_MODE_LINE_DIRECTIONS = 12         # directions probed for lines of zero modes
