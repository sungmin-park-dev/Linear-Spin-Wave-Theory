import numpy as np
from enum import Enum, auto
from typing import Union, Dict, List, Tuple, Optional
from functools import lru_cache

# Constants
EPS = 1e-2
THRESHOLD = 1e-3
XY_TOL = 1e-4

class Phase(Enum):
    UNIFORM_FM = auto()
    MODULATED_FM = auto()
    INPLANE_STRIPE_X = auto()
    INPLANE_STRIPE_Y = auto()
    INPLANE_STRIPE_XY = auto()
    ASYM_INPLANE_STRIPE = auto()
    COLLINEAR_AFM = auto()
    CANTED_AFM = auto()
    SPIRAL_AFM = auto()
    Y_PHASE = auto()
    UUD_PHASE = auto()
    V_PHASE = auto()
    FP_PHASE = auto()
    ISO_SKYRMION = auto()
    ANI_SKYRMION = auto()
    NON_SKYRMION = auto()
    DIST_STRIPE = auto()
    STRIPE_Y = auto()
    STRIPE_YZ = auto()
    ROT_SYMMETRIC_NON_COPLANAR = auto()
    UNKNOWN_PHASE = auto()

# Visualization styles for phases
PHASE_STYLES = {
    Phase.UNIFORM_FM: {'color': '#FF4500', 'marker': 'o', 'size': 80, 'edgecolor': 'darkred', 'linewidth': 1},
    Phase.MODULATED_FM: {'color': '#FF6347', 'marker': 's', 'size': 80, 'edgecolor': 'darkred', 'linewidth': 1},
    Phase.INPLANE_STRIPE_X: {'color': '#FF8C69', 'marker': '<', 'size': 85, 'edgecolor': 'orangered', 'linewidth': 1},
    Phase.INPLANE_STRIPE_Y: {'color': '#FF8C69', 'marker': '^', 'size': 90, 'edgecolor': 'orangered', 'linewidth': 1},
    Phase.INPLANE_STRIPE_XY: {'color': '#FF8C69', 'marker': '>', 'size': 85, 'edgecolor': 'orangered', 'linewidth': 1},
    Phase.ASYM_INPLANE_STRIPE: {'color': '#FF8C69', 'marker': 'v', 'size': 85, 'edgecolor': 'orangered', 'linewidth': 1},
    Phase.COLLINEAR_AFM: {'color': '#800080', 'marker': 'D', 'size': 80, 'edgecolor': 'purple', 'linewidth': 1},
    Phase.CANTED_AFM: {'color': '#FF00FF', 'marker': 'd', 'size': 85, 'edgecolor': 'purple', 'linewidth': 1},
    Phase.SPIRAL_AFM: {'color': '#FF6B6B', 'marker': 's', 'size': 70, 'edgecolor': 'darkred', 'linewidth': 1},
    Phase.Y_PHASE: {'color': '#4169E1', 'marker': 'D', 'size': 80, 'edgecolor': 'navy', 'linewidth': 1},
    Phase.UUD_PHASE: {'color': '#1E90FF', 'marker': 'p', 'size': 90, 'edgecolor': 'navy', 'linewidth': 1},
    Phase.V_PHASE: {'color': '#87CEEB', 'marker': 'h', 'size': 85, 'edgecolor': 'navy', 'linewidth': 1},
    Phase.FP_PHASE: {'color': '#B0E0E6', 'marker': '*', 'size': 100, 'edgecolor': 'navy', 'linewidth': 1},
    Phase.ISO_SKYRMION: {'color': '#32CD32', 'marker': 'X', 'size': 100, 'edgecolor': 'darkgreen', 'linewidth': 1.5},
    Phase.ANI_SKYRMION: {'color': '#98FB98', 'marker': 's', 'size': 80, 'edgecolor': 'darkgreen', 'linewidth': 1},
    Phase.NON_SKYRMION: {'color': '#90EE90', 'marker': 'P', 'size': 90, 'edgecolor': 'darkgreen', 'linewidth': 1},
    Phase.DIST_STRIPE: {'color': '#ADFF2F', 'marker': 'v', 'size': 85, 'edgecolor': 'darkgreen', 'linewidth': 1},
    Phase.STRIPE_Y: {'color': '#ADFF2F', 'marker': 'v', 'size': 85, 'edgecolor': 'darkgreen', 'linewidth': 1},
    Phase.STRIPE_YZ: {'color': '#ADFF2F', 'marker': 'v', 'size': 85, 'edgecolor': 'darkgreen', 'linewidth': 1},
    Phase.ROT_SYMMETRIC_NON_COPLANAR: {'color': '#FFD700', 'marker': 'H', 'size': 90, 'edgecolor': 'gold', 'linewidth': 1},
    Phase.UNKNOWN_PHASE: {'color': '#808080', 'marker': 'x', 'size': 80, 'edgecolor': 'black', 'linewidth': 1},
}

def get_phase_name(phase: Phase) -> str:
    """Get human-readable phase name."""
    return phase.name.replace('_', ' ').title()

def get_phase_style(phase: Phase) -> dict:
    """Get complete style for phase visualization."""
    return PHASE_STYLES[phase]

def is_close(a: float, b: float, tol: float = EPS) -> bool:
    """Check if two values are close within tolerance."""
    return abs(a - b) < tol

def spherical_to_cartesian(theta: float, phi: float, r: float = 1.0) -> np.ndarray:
    """Convert spherical coordinates to Cartesian coordinates."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def cartesian_to_spherical(spin: np.ndarray) -> Tuple[float, float]:
    """Convert Cartesian spin vector to spherical coordinates (theta, phi)."""
    x, y, z = spin
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-10:
        return 0, 0
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return theta, phi

def angle_between_spins(spin1: np.ndarray, spin2: np.ndarray) -> float:
    """Calculate angle between two spin vectors in degrees."""
    spin1 = spin1 / np.linalg.norm(spin1)
    spin2 = spin2 / np.linalg.norm(spin2)
    cos_angle = np.clip(np.dot(spin1, spin2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def spherical_triangle_area(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """Calculate the solid angle of a spherical triangle."""
    scalar_triple = np.abs(np.dot(v1, np.cross(v2, v3)))
    denom = (1 + np.dot(v1, v2) + np.dot(v2, v3) + np.dot(v3, v1))
    return 2 * np.arctan2(scalar_triple, denom)

def get_spin_vectors_list(spin_ordering: Union[Dict, list, tuple]) -> List[np.ndarray]:
    """Convert spin ordering to list of vectors."""
    if isinstance(spin_ordering, dict):
        return list(spin_ordering.values())
    elif isinstance(spin_ordering, (tuple, list)):
        return list(spin_ordering)
    else:
        raise ValueError("Invalid spin ordering type")

def calculate_skyrmion_number(spins_ordering: Union[Dict, list, tuple]) -> float:
    """Calculate skyrmion number from spin vectors."""
    spins_list = get_spin_vectors_list(spins_ordering)
    if len(spins_list) != 4:
        raise ValueError("Skyrmion number calculation requires 4 spins")
    
    faces = [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)]
    total_solid_angle = 0
    for face in faces:
        v1, v2, v3 = [spins_list[i] for i in face]
        total_solid_angle += spherical_triangle_area(v1, v2, v3)
    
    return total_solid_angle / (4 * np.pi)

def check_pure_z_component(spins_ordering: Union[Dict, list, tuple], z_tol: float = THRESHOLD) -> Tuple[bool, Optional[float]]:
    """Check if exactly one spin has pure z-component (±1)."""
    try:
        spins = np.array(get_spin_vectors_list(spins_ordering))
        xy_close_to_zero = np.all(np.abs(spins[:, :2]) < z_tol, axis=1)
        z_close_to_one = np.abs(np.abs(spins[:, 2]) - 1.0) < z_tol
        pure_z_indices = np.where(xy_close_to_zero & z_close_to_one)[0]
        
        if len(pure_z_indices) != 1:
            return False, None
        
        pure_z_idx = pure_z_indices[0]
        sign = np.sign(spins[pure_z_idx, 2])
        return True, sign
    except Exception:
        return False, None

def total_spin_xy_moment(spins_ordering: Union[Dict, list, tuple], xy_tol: float = XY_TOL) -> Tuple[bool, float]:
    """Check if xy-components sum to near zero."""
    try:
        spins = np.array(get_spin_vectors_list(spins_ordering))
        xy_sum = np.sum(spins[:, :2], axis=0)
        abs_xy_sum = np.sqrt(np.sum(xy_sum ** 2))
        return abs_xy_sum < xy_tol, abs_xy_sum
    except Exception:
        return False, 0.0

def is_coplanar(spins: Union[Dict, list, tuple], tol: float = 1e-2) -> bool:
    """Check if spin vectors are coplanar."""
    try:
        spin_vectors = np.array(get_spin_vectors_list(spins))
        rank = np.linalg.matrix_rank(spin_vectors, tol=tol)
        return rank <= 2
    except Exception:
        return False

@lru_cache(maxsize=128)
def get_unique_spins(spins_tuple: tuple, threshold: float = THRESHOLD) -> Dict[str, np.ndarray]:
    """Remove similar vectors and return unique spins."""
    spins = list(spins_tuple)
    unique_spins = {}
    
    for i, vector in enumerate(spins):
        vector = np.array(vector)
        is_unique = True
        
        for ukey, uvector in unique_spins.items():
            angle_threshold = 1 - threshold
            if np.dot(vector, uvector) > angle_threshold :
                is_unique = False
                break
            
            # if np.linalg.norm(vector - uvector) < threshold:
                # is_unique = False
                # break
        
        if is_unique:
            unique_spins[f"sl{i}"] = vector
    
    return unique_spins

def check_rotational_symmetry(spins_ordering: Union[Dict, list, tuple], tol: float = np.pi/6) -> bool:
    """Check if spins exhibit rotational symmetry around z-axis for ROT_SYMMETRIC_NON_COPLANAR."""
    try:
        spins = np.array(get_spin_vectors_list(spins_ordering))
        coords = np.array([cartesian_to_spherical(spin) for spin in spins])
        theta, phi = coords[:, 0], coords[:, 1] % (2 * np.pi)

        # Check z-components: all positive (theta < pi/2)
        if not np.all(theta < np.pi/2 + EPS):
            return False

        # Check for spin with z≈1 (theta ≈ 0)
        z_one_idx = np.where(np.abs(theta) < 1e-3)[0]
        if len(z_one_idx) != 1:
            return False
        z_one_idx = z_one_idx[0]

        # Get phi values for remaining three spins
        other_indices = np.setdiff1d(np.arange(4), z_one_idx)
        phi_subset = phi[other_indices]

        # Compute pairwise phi differences
        phi_diffs = np.abs(np.subtract.outer(phi_subset, phi_subset)) % (2 * np.pi)
        phi_diffs = np.minimum(phi_diffs, 2 * np.pi - phi_diffs)
        phi_diffs = phi_diffs[np.triu_indices(len(phi_subset), k=1)]

        # Check for approximate 60° or 120° differences
        target_angles = np.array([np.pi/3, 2*np.pi/3])  # 60°, 120°
        c3_like = np.any(np.isclose(phi_diffs[:, None], target_angles[None, :], atol=tol))

        # Verify non-coplanar
        return c3_like and not is_coplanar(spins_ordering, tol=1e-2)
    except Exception:
        return False



class TLAFPhaseClassifier:
    """Triangular Lattice Antiferromagnet Phase Classifier"""
    
    def __init__(self, spin_sys_data: dict):
        """Initialize phase classifier with spin system data."""
        if not isinstance(spin_sys_data, dict) or "Spin info" not in spin_sys_data:
            raise ValueError("spin_sys_data must be a dictionary with 'Spin info' key")
            
        self.spin_info = spin_sys_data["Spin info"]
        if not all("Angles" in v for v in self.spin_info.values()):
            raise ValueError("All spin info entries must contain 'Angles'")
            
        self.num_sl = len(self.spin_info)
        self.sky_criteria = 0.1
        self.threshold = THRESHOLD
        self.eps = EPS
        self.spins_ordering = self._get_spins_direction()   # Dict[msl] = np.ndarray
        self.phase = self._analyze_phase()

    def _get_spins_direction(self) -> Dict[str, np.ndarray]:
        """Convert spin angles to Cartesian vectors."""
        classical_spins = {}
        for key, value in self.spin_info.items():
            theta, phi = value["Angles"]
            
            # Adjust theta near π or -π
            if is_close(abs(theta), np.pi, tol=1e-2):
                theta = np.sign(theta) * np.pi
                
            if not (-np.pi <= theta <= np.pi and -2*np.pi <= phi <= 2*np.pi):
                raise ValueError(f"Invalid angles for {key}: theta={theta}, phi={phi}")
                
            spin = spherical_to_cartesian(theta, phi, r=1)
            if not is_close(np.linalg.norm(spin), 1.0, tol=self.eps):
                raise ValueError(f"Spin vector for {key} is not normalized")
                
            classical_spins[key] = spin
        return classical_spins

    def _analyze_phase(self) -> Phase:
        """Analyze phase based on number of sublattices."""
        try:
            if self.num_sl == 1:
                print("Number of msl: 1")
                return self._analyze_one_msl_phase()
            elif self.num_sl == 2:
                print("Number of msl: 2")
                return self._analyze_two_msl_phase()
            elif self.num_sl == 3:
                print("Number of msl: 3")
                return self._analyze_three_msl_phase()
            elif self.num_sl == 4:
                print("Number of msl: 4")
                return self._analyze_four_msl_phase()
            else:
                return Phase.UNKNOWN_PHASE
        except Exception:
            return Phase.UNKNOWN_PHASE


    def _check_skyrmion_phase(self, skyrmion_number: float) -> Optional[Phase]:
        """Determine if the phase is a skyrmion based on skyrmion number."""
        if abs(skyrmion_number) <= 1 - self.sky_criteria:
            return None
            
        found_pure_z_comp, _ = check_pure_z_component(self.spins_ordering, z_tol=self.threshold)
        spin_xy_moment_near_zero, _ = total_spin_xy_moment(self.spins_ordering, xy_tol=XY_TOL)
        
        if found_pure_z_comp and spin_xy_moment_near_zero:
            return Phase.ISO_SKYRMION
        
        return Phase.ANI_SKYRMION

    def _analyze_one_msl_phase(self) -> Phase:
        return Phase.FP_PHASE

    def _analyze_two_msl_phase(self, unique_spins: Optional[Dict[str, np.ndarray]] = None) -> Phase:
        """Analyze stripe-like phase for two-sublattice system or four-sublattice with two unique spins."""
        # Use unique spins if provided, else self.spins_ordering
        spins = list(unique_spins.values()) if unique_spins else get_spin_vectors_list(self.spins_ordering)

        # Validate input
        if len(spins) != 2:
            print(f"Invalid spin count: {len(spins)}. Expected 2.")
            return Phase.UNKNOWN_PHASE
        
        for i, spin in enumerate(spins):
            if not is_close(np.linalg.norm(spin), 1.0, tol = self.eps/10):
                print(f"Spin {i} not normalized: norm={np.linalg.norm(spin):.6f}")
                return Phase.UNKNOWN_PHASE

        # Thresholds
        z_eps = 0.05
        z_sym_tol = 0.02
        xy_sym_tol = 0.02
        angle_tol = np.radians(5)  # ±5°

        # Extract components
        z_components = [spin[2] for spin in spins]
        xy_components = [spin[:2] for spin in spins]
        xy_norms = [np.linalg.norm(xy) for xy in xy_components]

        # XY direction analysis
        xy_dominant = []
        angles = []
        for i, xy in enumerate(xy_components):

            if xy_norms[i] < self.eps:
                xy_dominant.append("none")
                angles.append(0.0)
            else:
                phi = np.arctan2(xy[1], xy[0]) % (2 * np.pi)
                angles.append(phi)
                phi_mod = phi % (np.pi / 3)  # 60° symmetry
                # print(f"\nx = {xy[0]:.6f}, y = {xy[1]:.6f}, phi (deg) = {np.degrees(phi):.6f}, phi_mod (deg) = {np.degrees(phi_mod):.6f}")

                if is_close(phi_mod, 0.0, tol=angle_tol) or is_close(phi_mod, np.pi / 3, tol=angle_tol):
                    xy_dominant.append("X")
                elif is_close(phi_mod, np.pi / 6, tol=angle_tol):
                    xy_dominant.append("Y")
                else:
                    xy_dominant.append("XY")


        # XY symmetry
        xy_symmetric = abs(xy_norms[0] - xy_norms[1]) < xy_sym_tol if xy_norms[0] > self.eps else True

        # Unique spin analysis
        unique_spins_dict = unique_spins or get_unique_spins(tuple(map(tuple, spins)), threshold=self.threshold)
        num_unique_spins = len(unique_spins_dict)
        unique_z_components = [spin[2] for spin in unique_spins_dict.values()]

        # Spin angle
        spin_angle = angle_between_spins(spins[0], spins[1])

        # Debugging
        print(f"Two MSL: z={z_components}, xy_norms={xy_norms}, angles={np.degrees(angles)}, "
            f"xy_dominant={xy_dominant}, xy_symmetric={xy_symmetric}, "
            f"num_unique_spins={num_unique_spins}, spin_angle={spin_angle:.2f}°")

        # Single unique spin
        if num_unique_spins == 1:
            print("Single unique spin.")
            return Phase.FP_PHASE

        # Phase determination
        if num_unique_spins == 2:
            z1, z2 = unique_z_components
            z_diff = abs(abs(z1) - abs(z2))

            # Case 1: Inplane
            if abs(z1) < z_eps and abs(z2) < z_eps:
                print("CASE 1: Inplane stripe")
                if xy_symmetric:
                    if xy_dominant[0] == xy_dominant[1] == "X":
                        return Phase.INPLANE_STRIPE_X
                    elif xy_dominant[0] == xy_dominant[1] == "Y":
                        return Phase.INPLANE_STRIPE_Y
                    else:
                        return Phase.INPLANE_STRIPE_XY
                return Phase.ASYM_INPLANE_STRIPE

            # Case 2: FM
            if z1 > 0 and z2 > 0:
                print("CASE 2: Ferromagnetic")
                if abs(z1) > 1 - self.eps and abs(z2) > 1 - self.eps:
                    return Phase.FP_PHASE
                if z_diff < z_sym_tol:
                    return Phase.UNIFORM_FM
                return Phase.MODULATED_FM

            # Case 3: AFM
            if z1 * z2 < 0:
                print("CASE 3: Antiferromagnetic")
                if z_diff < z_sym_tol and is_close(z1, -z2, tol=z_sym_tol):
                    if abs(spin_angle - 90) < 15:
                        return Phase.SPIRAL_AFM
                    return Phase.COLLINEAR_AFM
                return Phase.CANTED_AFM

        print("No matching phase.")
        return Phase.UNKNOWN_PHASE

    def _analyze_three_msl_phase(self) -> Phase:
        """Analyze phase for three sublattice system (Y, UUD, V, FP)."""
        spins = get_spin_vectors_list(self.spins_ordering)
        if len(spins) != 3:
            return Phase.UNKNOWN_PHASE
        
        spins = [spin / np.linalg.norm(spin) if np.linalg.norm(spin) > self.eps else spin for spin in spins]
        
        coords = [cartesian_to_spherical(spin) for spin in spins]
        theta = [c[0] for c in coords]
        phi = [c[1] for c in coords]
        z_components = [np.cos(t) for t in theta]
        xy_components = [np.sin(t) for t in theta]
        
        # FP phase check
        if all(is_close(t, 1, tol=self.eps) for t in z_components):
            return Phase.FP_PHASE
        
        # Find down spin (z ≈ -1)
        down_spin_idx = None
        for i, z in enumerate(z_components):
            if is_close(z, -1, tol=self.eps):
                down_spin_idx = i
                break
        
        # UUD and Y phase checks
        if down_spin_idx is not None:
            other_indices = [i for i in range(3) if i != down_spin_idx]
            
            # UUD phase check
            if all(is_close(xy_components[i], 0, tol = self.eps ) for i in other_indices):
                return Phase.UUD_PHASE
                
            # Y phase check
            if all(z_components[i] > 0 for i in other_indices):
                phi_diff = abs(phi[other_indices[0]] - phi[other_indices[1]])
                if is_close(phi_diff, np.pi, tol=self.eps) or is_close(phi_diff, 3*np.pi, tol=self.eps):
                    return Phase.Y_PHASE
        
        # V phase check
        for i in range(3):
            for j in range(i+1, 3):
                if angle_between_spins(spins[i], spins[j]) < 10:
                    k = 3 - i - j
                    phi_diff = abs(phi[i] - phi[k])
                    
                    if is_close(phi_diff, np.pi, tol=self.eps) or is_close(phi_diff, 3*np.pi, tol=self.eps):
                        return Phase.V_PHASE
        
        # Default for 3-sublattice system
        return Phase.UNKNOWN_PHASE

    def _analyze_coplanar_phase(self) -> Phase:
        """Analyze phase for coplanar four-sublattice system."""
        spins = np.array(get_spin_vectors_list(self.spins_ordering))
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        spin_xy_moment_near_zero, _ = total_spin_xy_moment(self.spins_ordering, xy_tol=XY_TOL)

        for i, j in pairs:
            dot_prod = np.dot(spins[i], spins[j])
            
            if is_close(abs(dot_prod), 1, tol=self.eps):
                
                remaining = [k for k in range(4) if k != i and k != j]
                
                dot_prod_rem = np.dot(spins[remaining[0]], spins[remaining[1]])
                
                if is_close(abs(dot_prod_rem), 1, tol=self.eps):
                    return Phase.STRIPE_YZ
                
                if abs(dot_prod_rem) < 0.9 and spin_xy_moment_near_zero:
                    return Phase.DIST_STRIPE
                
                return Phase.UNKNOWN_PHASE

        return Phase.STRIPE_Y

    def _analyze_four_msl_phase(self) -> Phase:
        """Analyze four sublattice phase based on skyrmion number and spin properties."""
        # Calculate skyrmion number and check skyrmion phase
        skyrmion_number = calculate_skyrmion_number(self.spins_ordering)
        skyrmion_phase = self._check_skyrmion_phase(skyrmion_number)
        
        # Get unique spins
        unique_spins = get_unique_spins(tuple(map(tuple, get_spin_vectors_list(self.spins_ordering))), threshold=self.threshold)
        num_unique_spins = len(unique_spins)
        
        print(f"unique spins: {unique_spins}")
        
        # Special cases based on unique spin count
        if skyrmion_phase is not None:
            if num_unique_spins == 2:
                return self._analyze_two_msl_phase(unique_spins=unique_spins)
            else:
                return skyrmion_phase

        if num_unique_spins == 1:
            return Phase.FP_PHASE
            
        if num_unique_spins == 2:
            return self._analyze_two_msl_phase(unique_spins=unique_spins)

        # Check if spins are coplanar
        if is_coplanar(self.spins_ordering, tol = self.threshold):
            return self._analyze_coplanar_phase()

        # Check for rotational symmetry (C3-like)
        if check_rotational_symmetry(self.spins_ordering, tol = np.pi/6):
            return Phase.ROT_SYMMETRIC_NON_COPLANAR

        # Default case
        return Phase.NON_SKYRMION

if __name__ == "__main__":
    import sys
    sys.path.append('/Users/sungminpark/Desktop/LSWT/Packages/LSWT_v5')
    from nbcp_phases import FIND_TLAF_PHASES
    from spin_system_visualizer import SpinVisualizer
    import matplotlib.pyplot as plt

    JGamma = 0.15#0.105
    JPD = 0.0
    h = 0.4900001

    # NBCP configuration
    nbcp_config = { "Jxy": 0.067,
                    "Jz": 0.125,
                    "JGamma": 0.,
                    "JPD": -0.05,
                    "Kxy": 0.0,
                    "Kz": 0.0,
                    "KPD": 0.00,
                    "KGamma": 0.0,
                    "h": (0.0, 0.0, 0.00001),
                    "aniso": ( 0.0, 0.0, 0.0),
                    "nne": (0.00, 0.00, 0.00)}
    
    # optimization setting
    opt_method = "MAGSWT" # Choose either "classical" or "MAGSWT" or "classical+quantum"
    N = 20                    # number of k points in the BZ
    phi = 0      
    angles_setting = {"One MSL": (None, phi),
                    "Two MSL": (None, None, None, None),
                    "Three MSL": (None, phi, None, phi, None , phi),
                    "Four MSL": (None, None, None, None, None, None, None, None),}



    # FIND_TLAF_PHASES로 데이터 생성
    nbcp = FIND_TLAF_PHASES(config = nbcp_config)

    nbcp_opt_result, nbcp_cls_result = nbcp.find_tlaf_phase(opt_method = opt_method,
                                                            angles_setting = angles_setting,
                                                            verbose = True,
                                                            N = N)
    nbcp.summarize_results(verbose = True)
    
    visualizer = SpinVisualizer()
    
    best_opt = nbcp_opt_result["spin_sys_data"]
    best_cls = nbcp_cls_result["spin_sys_data"]
    
    fig, (ax_xy, ax_angle) = visualizer.plot_system(best_opt)
    fig.suptitle(f"NBCP spin Configuration", fontsize=16, y=1.05)
    plt.show()
    
    fig, (ax_xy, ax_angle) = visualizer.plot_system(best_cls)
    fig.suptitle(f"NBCP spin Configuration", fontsize=16, y=1.05)
    plt.show()

    # Phase 분류 및 출력
    classifier = TLAFPhaseClassifier(best_opt)
    print(f"Spin-Wave Theory phase: {get_phase_name(classifier.phase)}")
    classifier = TLAFPhaseClassifier(best_cls)
    print(f"Classical Theory phase: {get_phase_name(classifier.phase)}")
    
    # physical_quantities = get_physical_quantities(nbcp_phase_spin_system_data)
    
    # for key1, value1 in physical_quantities.items():
    #     print(key1)
    #     for key2, value2 in value1.items():
    #         print(f"{key2}: \t {value2}")
        

