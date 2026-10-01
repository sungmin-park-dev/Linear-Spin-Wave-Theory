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
CLASSICAL_REFINE_ROUNDS = 20       # L-BFGS-B rounds, each re-centred on the previous result
CLASSICAL_REFINE_RECENTRE = 1e-3   # radians; a larger move in a round triggers another round
# Global classical search (D32): the differential-evolution settings of the former SpinOptimizer
CLASSICAL_SEARCH_POPSIZE = 18
CLASSICAL_SEARCH_TOL = 1e-9
CLASSICAL_SEARCH_MAXITER = 800
CLASSICAL_SEARCH_MUTATION = (0.5, 0.9)
CLASSICAL_SEARCH_RECOMBINATION = 0.8
CLASSICAL_SEARCH_SEED = 42

SELECTION_MODE = "physics"         # "physics" or "fixed"
SELECTION_GAP_RATIO = 1e-3         # fixed mode: w_min / w_next below this is a null mode
SELECTION_RANK_TOLERANCE = 1e-4    # fixed mode: relative generator singular value treated as zero
SELECTION_RESIDUAL_FACTOR = 10.0   # harmonic amplitude must exceed this times the fit residual
SELECTION_ROUNDOFF_FACTOR = 1e3    # multiples of machine epsilon times |E| treated as round-off
SELECTION_ACCURACY_FACTOR = 10.0   # physics mode: multiples of the estimated state accuracy
SELECTION_ADIABATIC_WARNING = 0.1   # warn when Gamma curvature / hard stiffness exceeds this (D28)
SELECTION_PATH_TOLERANCE = 1e-8     # relative gradient tolerance of the soft-path relaxation (D28)
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
# Topology (D29, confirmed as D31 on 2026-09-30)
TOPOLOGY_BAND_GAP_CUTOFF = 1e-8    # E0; signed BdG separation at or below this leaves a band's curvature undefined
TOPOLOGY_MIN_LINK_OVERLAP = 1e-3   # FHS: |u^+ eta u'| at or below this marks a band crossing or an unresolved mesh
TOPOLOGY_PLAQUETTE_PHASE_MARGIN = 1e-6  # FHS: a plaquette phase within this of +-pi (rad) is not admissible
TOPOLOGY_CHERN_AGREEMENT = 0.1     # Kubo integral within this of the FHS integer accepts the Chern number
TOPOLOGY_ADAPTIVE_RELATIVE = 1e-3   # adaptive k integration: relative tolerance of kappa_xy / T
TOPOLOGY_ADAPTIVE_ABSOLUTE = 1e-7   # adaptive k integration: absolute tolerance (units of k_B^2 / hbar)
TOPOLOGY_ADAPTIVE_MAX_POINTS = 200_000  # adaptive k integration: evaluation budget
TOPOLOGY_ADAPTIVE_MAX_DEPTH = 12    # adaptive k integration: halvings of an initial cell

# =============================================================================
# Zero-mode scan and finite temperature (stage 4b)
# =============================================================================

ZERO_MODE_ZERO_TOLERANCE = 1e-12       # lambda_min(H) / scale at or below: numerically zero
ZERO_MODE_CANDIDATE_TOLERANCE = 1e-6   # up to this: candidate, the user decides
ZERO_MODE_MESH_SEEDS = 8               # lowest mesh points used as scan starting points
ZERO_MODE_SEARCH_THRESHOLD = 1e-2      # minimize lambda_min only from points at or below this
ZERO_MODE_LINE_DIRECTIONS = 12         # directions probed for lines of zero modes
