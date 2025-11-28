# Refined Architecture: Accounting for Incommensurate Structures

## Critical Realization

The commensurate vs incommensurate distinction is **architectural**, not a detail:

| Aspect | Commensurate | Incommensurate |
|--------|-------------|----------------|
| **Supercell** | Finite (q = rational) | Infinite (q = irrational) |
| **Degrees of freedom** | 2N angles (θᵢ, φᵢ) | 2 (q-vector) + axis |
| **Optimization** | Minimize over angles | Minimize over q |
| **Visualization** | Discrete arrows | Continuous spiral |
| **BZ** | Folded into magnetic BZ | Extended zone scheme |

This requires **separating crystallographic lattice from magnetic structure**.

---

## Revised Core Architecture

### Before (CLAUDE.md - Too Simple)

```
core/
├── lattice/           # Mixed: geometry + spin config
├── spin_system.py     # Contains everything
```

**Problem**: Assumes finite supercell everywhere

### After (Refined - Separation of Concerns)

```
core/
├── lattice/                    # Pure crystallographic geometry
│   ├── base.py                # AbstractLattice (no spin info)
│   ├── lattice2d.py           # Generic 2D
│   └── presets.py             # Triangular, Square, etc.
│
├── magnetic_structure/         # ✨ NEW: Spin configurations
│   ├── base.py                # AbstractMagneticStructure
│   ├── commensurate.py        # Finite supercell
│   └── incommensurate.py      # Spiral/helix (Phase 5+)
│
├── spin_system.py             # Combines Lattice + MagneticStructure + Interactions
├── interactions.py            # Exchange, DM, anisotropy
└── hamiltonian_builder.py     # Builds H from SpinSystem
```

---

## AbstractLattice (Revised - Geometry Only)

### What It Should NOT Have

❌ `get_spin_configuration()`
❌ `set_angles()`
❌ `num_magnetic_sublattices`
❌ Anything about spins/magnetism

### What It SHOULD Have

```python
class AbstractLattice(ABC):
    """Pure crystallographic lattice - no magnetic information"""

    @abstractmethod
    def get_primitive_vectors(self) -> np.ndarray:
        """Lattice vectors [a1, a2] in real space"""

    @abstractmethod
    def get_basis_positions(self) -> List[np.ndarray]:
        """Positions of atoms within unit cell (crystallographic)"""

    @abstractmethod
    def get_neighbors(self, order: int = 1) -> List[Neighbor]:
        """
        Returns neighbor information for crystallographic unit cell.

        Returns
        -------
        List[Neighbor]
            Each Neighbor contains:
            - source_site: int (basis index)
            - target_site: int (basis index)
            - displacement: np.ndarray (lattice vector)
            - distance: float
        """

    @abstractmethod
    def get_reciprocal_vectors(self) -> np.ndarray:
        """Reciprocal lattice vectors [b1, b2]"""

    def get_high_symmetry_points(self) -> Dict[str, np.ndarray]:
        """High-symmetry k-points (Γ, K, M, etc.)"""
```

**Key Point**: No magnetic information. This is pure geometry.

---

## AbstractMagneticStructure (New Abstraction)

```python
class AbstractMagneticStructure(ABC):
    """
    Represents the magnetic ordering pattern.

    Can be commensurate (finite supercell) or incommensurate (infinite).
    """

    @abstractmethod
    def get_spin_at_position(self, position: np.ndarray) -> np.ndarray:
        """
        Return spin direction at given position.

        Parameters
        ----------
        position : np.ndarray
            Real-space position

        Returns
        -------
        spin : np.ndarray
            Unit vector (Sx, Sy, Sz)
        """

    @abstractmethod
    def get_magnetic_unit_cell_size(self) -> Union[Tuple[int, int], None]:
        """
        Returns (n1, n2) supercell size if commensurate, None if incommensurate
        """

    @abstractmethod
    def get_num_magnetic_sublattices(self) -> Union[int, None]:
        """
        Number of magnetic sublattices if commensurate, None if incommensurate
        """

    @abstractmethod
    def to_dict(self) -> Dict:
        """Serialize for config/results"""
```

### Commensurate Implementation

```python
class CommensurateStructure(AbstractMagneticStructure):
    """
    Magnetic structure with finite supercell.

    Example: 3-sublattice 120° structure on triangular lattice
    """

    def __init__(self,
                 crystallographic_basis_size: int,
                 supercell: Tuple[int, int],  # (n1, n2)
                 angles: np.ndarray):  # shape: (n1*n2*basis, 2) for (θ, φ)
        """
        Parameters
        ----------
        crystallographic_basis_size : int
            Number of atoms in crystallographic unit cell
        supercell : Tuple[int, int]
            Magnetic supercell size (n1, n2) in units of lattice vectors
        angles : np.ndarray
            Spin angles (theta, phi) for each magnetic sublattice
        """
        self.basis_size = crystallographic_basis_size
        self.supercell = supercell
        self.angles = angles  # Shape: (Nmag, 2)
        self.Nmag = supercell[0] * supercell[1] * crystallographic_basis_size

    def get_spin_at_position(self, position: np.ndarray) -> np.ndarray:
        """Look up which magnetic sublattice this position belongs to"""
        # Map position to magnetic sublattice index
        mag_idx = self._position_to_sublattice(position)
        theta, phi = self.angles[mag_idx]

        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

    def get_magnetic_unit_cell_size(self):
        return self.supercell

    def get_num_magnetic_sublattices(self):
        return self.Nmag

    def to_dict(self):
        return {
            'type': 'commensurate',
            'supercell': self.supercell,
            'angles': self.angles.tolist()
        }
```

### Incommensurate Implementation (Future)

```python
class IncommensurateStructure(AbstractMagneticStructure):
    """
    Magnetic structure with incommensurate q-vector.

    Example: Spiral with q = (0.4, 0.4) on triangular lattice

    Based on: Toth & Lake, J. Phys. Condens. Matter 27, 166002 (2015)
    """

    def __init__(self,
                 q_vector: np.ndarray,  # (qx, qy) in reciprocal space
                 rotation_axis: np.ndarray,  # Axis perpendicular to spiral plane
                 reference_spin: np.ndarray):  # Spin at origin
        """
        Represents spin configuration as:
        S(r) = rotate(reference_spin, angle = q · r, axis = rotation_axis)
        """
        self.q_vector = q_vector
        self.rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        self.reference_spin = reference_spin / np.linalg.norm(reference_spin)

    def get_spin_at_position(self, position: np.ndarray) -> np.ndarray:
        """Continuous spiral"""
        angle = np.dot(self.q_vector, position)
        return self._rotate_vector(self.reference_spin,
                                   self.rotation_axis,
                                   angle)

    def get_magnetic_unit_cell_size(self):
        return None  # Incommensurate

    def get_num_magnetic_sublattices(self):
        return None  # Infinite

    def to_dict(self):
        return {
            'type': 'incommensurate',
            'q_vector': self.q_vector.tolist(),
            'rotation_axis': self.rotation_axis.tolist(),
            'reference_spin': self.reference_spin.tolist()
        }
```

---

## SpinSystem (Revised)

```python
class SpinSystem:
    """
    Complete physical system: lattice + magnetic structure + interactions.
    """

    def __init__(self,
                 lattice: AbstractLattice,
                 magnetic_structure: AbstractMagneticStructure,
                 interactions: List[Interaction],
                 spin_magnitude: float = 0.5):
        """
        Parameters
        ----------
        lattice : AbstractLattice
            Crystallographic lattice
        magnetic_structure : AbstractMagneticStructure
            Commensurate or incommensurate spin configuration
        interactions : List[Interaction]
            Exchange, DM, anisotropy terms
        spin_magnitude : float
            Spin quantum number S
        """
        self.lattice = lattice
        self.magnetic_structure = magnetic_structure
        self.interactions = interactions
        self.spin_magnitude = spin_magnitude

    @property
    def is_commensurate(self) -> bool:
        return self.magnetic_structure.get_magnetic_unit_cell_size() is not None

    @classmethod
    def from_config(cls, config_path: str):
        """Load from YAML - handles both commensurate and incommensurate"""
        config = load_yaml(config_path)

        # Load lattice
        lattice = build_lattice(config['lattice'])

        # Load magnetic structure (branch on type)
        if config['magnetic_structure']['type'] == 'commensurate':
            mag_struct = CommensurateStructure(...)
        elif config['magnetic_structure']['type'] == 'incommensurate':
            mag_struct = IncommensurateStructure(...)

        # Load interactions
        interactions = build_interactions(config['interactions'])

        return cls(lattice, mag_struct, interactions)
```

---

## YAML Configuration (Revised Schema)

### Commensurate Example

```yaml
project:
  name: "NBCP 3-Sublattice"

lattice:
  type: triangular
  lattice_constant: 1.0

magnetic_structure:
  type: commensurate
  supercell: [1, 1]  # q = 0 structure
  num_sublattices: 3
  angles:
    - [1.5708, 0.0]       # θ=π/2, φ=0
    - [1.5708, 2.0944]    # θ=π/2, φ=2π/3
    - [1.5708, 4.1888]    # θ=π/2, φ=4π/3

interactions:
  nearest_neighbor:
    Jxy: 0.076
    Jz: 0.125
    JGamma: 0.1

calculation:
  spin_magnitude: 0.5
  method: MAGSWT
```

### Incommensurate Example (Future)

```yaml
magnetic_structure:
  type: incommensurate
  q_vector: [0.4, 0.4]  # In units of reciprocal lattice
  rotation_axis: [0, 0, 1]  # Spiral in xy-plane
  reference_spin: [1, 0, 0]  # Spin at origin
```

---

## Optimization Changes

### Current (Commensurate Only)

```python
# Optimize over 2*Nmag angles
optimizer.minimize(energy_function,
                  x0=initial_angles,  # Shape: (2*Nmag,)
                  bounds=[(-π, π)] * (2*Nmag))
```

### Future (Incommensurate)

```python
# Optimize over q-vector (2D) + orientation
optimizer.minimize(energy_function,
                  x0=[qx, qy, theta_ref, phi_ref],  # Shape: (4,)
                  bounds=[(0, 2π), (0, 2π), (0, π), (0, 2π)])
```

**Key Point**: Very different optimization problems!

---

## Brillouin Zone Handling

### Commensurate

- **Magnetic BZ** = folded crystallographic BZ
- Example: 3-sublattice → BZ is 1/3 the size
- Use reduced zone scheme

### Incommensurate

- **Extended zone scheme** (Toth & Lake 2014)
- Need full crystallographic BZ + extended zones
- More k-points required

**Architecture Impact**:
```python
class BrillouinZone:
    def __init__(self, lattice: AbstractLattice,
                 magnetic_structure: AbstractMagneticStructure):
        self.lattice = lattice
        self.mag_struct = magnetic_structure

    def get_sampling_grid(self, N: int):
        if self.mag_struct.is_commensurate():
            return self._get_magnetic_bz_grid(N)
        else:
            return self._get_extended_zone_grid(N)
```

---

## Visualization Changes

### Current Approach (Commensurate Only)

```python
# Plot discrete arrows
for i, angle in enumerate(angles):
    position = sublattice_positions[i]
    arrow = spin_from_angles(angle)
    ax.arrow(position[0], position[1], arrow[0], arrow[1])
```

### Generalized Approach

```python
class SpinVisualizer:
    def plot_system(self, spin_system: SpinSystem):
        if spin_system.is_commensurate:
            self._plot_discrete(spin_system)
        else:
            self._plot_continuous(spin_system)

    def _plot_continuous(self, system):
        """Plot spiral/helix with color-coded phase"""
        # Create fine grid
        x = np.linspace(...)
        y = np.linspace(...)

        # Evaluate spin at each point
        for xi, yi in zip(x, y):
            spin = system.magnetic_structure.get_spin_at_position([xi, yi])
            # Color by phase angle
            color = phase_to_color(spin)
            ax.scatter(xi, yi, c=color)
```

---

## Migration Impact

### Phase 1 Changes (Immediate)

**Originally Planned**:
1. Create `AbstractLattice`
2. Create `SpinSystem`
3. Extract `TriangularLattice`

**Revised Plan**:
1. Create `AbstractLattice` (geometry only - no spin info)
2. Create `AbstractMagneticStructure` + `CommensurateStructure`
3. Create `SpinSystem` (combines both)
4. Extract `TriangularLattice` from `NBCP_UNIT_CELL`

**New Work**:
- Define `AbstractMagneticStructure` interface
- Implement `CommensurateStructure`
- Refactor current code to use `CommensurateStructure`

**Later (Phase 5+)**:
- Implement `IncommensurateStructure`
- Add extended zone BZ sampling
- Update visualization

---

## Current Code Mapping

### NBCP_UNIT_CELL (Current)

```python
class NBCP_UNIT_CELL:
    # MIXES: Lattice geometry + spin configuration + interactions

    def __init__(self, config):
        # Lattice geometry
        self.Disp_xy_nn = ...  # Triangular lattice specific

        # Spin configuration (commensurate)
        self.spin_system_data_three_msl(angles)  # 3-sublattice

        # Interactions
        self.J_ij_nn = ...  # Exchange matrices
```

### After Refactoring

**Lattice** (geometry):
```python
class TriangularLattice(AbstractLattice):
    def get_neighbors(self, order=1):
        # Return neighbor info (no spin config)
        return neighbors
```

**Magnetic Structure**:
```python
# Created by optimization or loaded from config
mag_struct = CommensurateStructure(
    crystallographic_basis_size=1,
    supercell=(1, 1),
    angles=np.array([[θ1, φ1], [θ2, φ2], [θ3, φ3]])
)
```

**SpinSystem**:
```python
system = SpinSystem(
    lattice=TriangularLattice(),
    magnetic_structure=mag_struct,
    interactions=build_nbcp_interactions(config)
)
```

---

## Benefits of This Architecture

### 1. **Clean Separation of Concerns**
- Lattice = pure geometry (reusable)
- MagneticStructure = spin configuration (optimization target)
- SpinSystem = combines everything

### 2. **Future-Proof for Incommensurate**
- Interface already supports it
- Implementation is isolated in `IncommensurateStructure`
- No changes to `Lattice` classes needed

### 3. **Clear Optimization Contracts**
```python
# Commensurate optimization
def optimize_commensurate(system: SpinSystem) -> np.ndarray:
    """Returns optimized angles"""

# Incommensurate optimization (future)
def optimize_incommensurate(system: SpinSystem) -> np.ndarray:
    """Returns optimized q-vector"""
```

### 4. **Type Safety**
- `SpinSystem.is_commensurate` → bool
- No more guessing from dictionary keys

### 5. **YAML Schema Validation**
```python
# Validator knows to check different fields
if config['magnetic_structure']['type'] == 'commensurate':
    assert 'angles' in config['magnetic_structure']
elif config['magnetic_structure']['type'] == 'incommensurate':
    assert 'q_vector' in config['magnetic_structure']
```

---

## Implementation Order (Revised)

### Phase 1A: Core Abstractions (Week 1)
- [ ] Define `AbstractLattice` (geometry only)
- [ ] Define `AbstractMagneticStructure`
- [ ] Implement `CommensurateStructure`
- [ ] Implement `SpinSystem` (combines both)

### Phase 1B: Triangular Lattice (Week 2)
- [ ] Extract `TriangularLattice` from `NBCP_UNIT_CELL`
- [ ] Migrate current code to use new structure
- [ ] Write tests: geometry separate from spins

### Phase 2-4: (Same as before)
- Solvers, Physics, Analysis, Visualization

### Phase 5+: Incommensurate Support
- [ ] Implement `IncommensurateStructure`
- [ ] Extended zone BZ sampling
- [ ] Continuous visualization
- [ ] Q-vector optimization

---

## Questions Before We Start

1. **Immediate Support**: Do you need incommensurate structures now, or can we implement the interface but delay the implementation?

2. **Current Code**: Are all your current calculations commensurate? (finite supercell)

3. **Priority**: Should we focus on getting commensurate working perfectly first?

4. **Testing**: Do you have reference results for both commensurate and incommensurate (if needed)?

---

## Recommendation

**Start with commensurate-only implementation, but design interfaces for both**:

1. ✅ Define `AbstractMagneticStructure` (anticipates incommensurate)
2. ✅ Implement `CommensurateStructure` fully
3. ⏸️ Stub `IncommensurateStructure` (interface only, raises NotImplementedError)
4. ✅ All other code uses `AbstractMagneticStructure` interface

This way:
- Your current work (commensurate) proceeds quickly
- Architecture is future-proof
- Adding incommensurate later is isolated to one module
- No refactoring needed when adding incommensurate support

**Does this revised architecture address your concerns?**
