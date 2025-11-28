# Architecture Review and Refactoring Plan

## Executive Summary

Your current codebase has a solid foundation with good separation of physics calculations, but is tightly coupled to NBCP (triangular lattice) specifics. The proposed architecture in CLAUDE.md provides excellent direction for making the code more modular, reusable, and maintainable.

**Key Recommendation**: Proceed with the refactoring - the architectural improvements will significantly enhance code quality and research productivity.

---

## Current Code Analysis

### Strengths ✅

1. **Good Physics Separation**
   - Thermodynamics, topology, and correlation calculations are well-separated
   - Each physics module has clear responsibilities
   - Clean delegation pattern in `LSWT` class

2. **Robust Optimization**
   - Well-implemented optimizer with multiple methods (DE, BFGS)
   - Flexible angle constraint system
   - Good convergence handling

3. **Comprehensive Functionality**
   - Full LSWT workflow implemented
   - MAGSWT regularization support
   - Berry curvature and topological calculations
   - Phase diagram capabilities

4. **Visualization**
   - Dedicated plotting modules
   - Separation of visualization from computation

### Issues ❌

1. **Tight Coupling to NBCP Lattice**
   - `NBCP_UNIT_CELL` hardcodes triangular lattice geometry
   - Displacement vectors (`Disp_xy_nn`, `Disp_xy_nnn`) are specific to triangular lattice
   - Cannot easily support other lattice types (square, honeycomb, kagome)

2. **No Lattice Abstraction**
   - Missing `AbstractLattice` interface
   - Geometry, neighbors, and BZ data mixed together
   - BZ handling is separate from lattice definition

3. **Dictionary-Based Data Passing**
   - `spin_system_data` is a nested dictionary
   - Keys like `"Spin info"`, `"Couplings"`, `"Lattice/BZ setting"` are strings
   - No type safety, easy to make errors
   - Hard to discover available data

4. **Configuration Issues**
   - No YAML-based configuration system
   - Config is just a Python dictionary
   - No validation or schema checking
   - Hard to share/reproduce experiments

5. **Module Organization**
   - `Tools/` is a catch-all directory
   - Not clear which modules are core vs utilities
   - Some cross-cutting concerns (BZ) split across files

6. **Naming Inconsistencies**
   - Mix of `lswt_correlation` vs `correlation.py`
   - `nbcp_` prefix pollutes core abstractions
   - Capital naming (`LSWT`, `NBCP_UNIT_CELL`) vs lowercase files

---

## Mapping: Current → Proposed Architecture

### Current Structure
```
modules/
├── LinearSpinWaveTheory/
│   ├── linear_spin_wave_theory.py      (LSWT main class)
│   ├── lswt_Hamiltonian.py             (Hamiltonian builder)
│   ├── lswt_thermodynamics.py          (Physics: thermo)
│   ├── lswt_topology.py                (Physics: topology)
│   └── lswt_correlation.py             (Physics: correlations)
├── SpinSystem/
│   ├── nbcp_unitcells.py               (Lattice-specific code)
│   └── nbcp_phases.py                  (Phase finding)
├── Tools/
│   ├── spin_system_optimizer.py        (Optimizer)
│   ├── brillouin_zone.py               (BZ handling)
│   ├── diagonalization.py              (Eigensolve)
│   ├── magnon_kernel.py                (Boson occupation)
│   └── analysis_tools.py               (Energy functions, etc.)
├── Plotters/                           (Visualization)
└── PhaseDiagram/                       (Analysis)
```

### Proposed Structure (from CLAUDE.md)
```
lswt/
├── core/
│   ├── lattice/
│   │   ├── base.py                     ← Abstract interface
│   │   ├── lattice2d.py                ← Generic 2D implementation
│   │   ├── presets.py                  ← Triangular, Square, etc.
│   │   └── builder.py                  ← Lattice construction helpers
│   ├── spin_system.py                  ← Core domain model
│   ├── hamiltonian.py                  ← Hamiltonian builder
│   └── interactions.py                 ← Interaction definitions
├── solvers/
│   ├── lswt.py                         ← LSWT solver
│   ├── optimizer.py                    ← Spin optimization
│   ├── eigensolve.py                   ← Eigenvalue problems
│   └── brillouin_zone.py               ← BZ sampling
├── physics/
│   ├── thermodynamics.py               ← From lswt_thermodynamics.py
│   ├── topology.py                     ← From lswt_topology.py
│   ├── correlations.py                 ← From lswt_correlation.py
│   └── observables.py                  ← New: general observables
├── analysis/
│   ├── phase_finder.py                 ← From nbcp_phases.py
│   ├── phase_diagram.py                ← From PhaseDiagram/
│   └── quantum_corrections.py          ← New
├── visualization/
│   ├── spin_viz.py                     ← From spin_system_visualizer.py
│   ├── band_plotter.py                 ← From updated_band_plotter.py
│   └── momentum_plotter.py             ← From momentum_space_plotter.py
├── io/
│   ├── config_loader.py                ← New: YAML loading
│   ├── validators.py                   ← New: validation
│   └── export_*.py                     ← From file_io.py + new
└── utils/
    ├── constants.py                    ← Already exists
    ├── math_utils.py                   ← New: common math
    └── logging.py                      ← New
```

---

## Detailed Migration Plan

### Phase 1: Core Abstractions (Week 1-2)

#### 1.1 Create Lattice Abstraction

**Goal**: Decouple lattice geometry from physics

**Tasks**:
- [ ] Create `lswt/core/lattice/base.py` with `AbstractLattice`
- [ ] Extract triangular lattice logic from `NBCP_UNIT_CELL` → `TriangularLattice`
- [ ] Implement `get_primitive_vectors()`, `get_basis_positions()`, `get_neighbors()`
- [ ] Add `get_brillouin_zone()` to return BZ data
- [ ] Create `presets.py` with `LATTICE_REGISTRY`

**Migration Strategy**:
```python
# Current: Hardcoded in NBCP_UNIT_CELL
self.Disp_xy_nn = ((dist_nn, 0.0), ...)

# After: In TriangularLattice class
class TriangularLattice(AbstractLattice):
    def get_neighbors(self, order=1):
        if order == 1:
            return self._nearest_neighbors()
        elif order == 2:
            return self._next_nearest_neighbors()
```

#### 1.2 Create SpinSystem Domain Model

**Goal**: Replace dictionary-based `spin_system_data` with typed class

**Current Problem**:
```python
# Accessing data is error-prone
spin_info = spin_system_data["Spin info"]
couplings = spin_system_data["Couplings"]
```

**Proposed**:
```python
class SpinSystem:
    def __init__(self, lattice: AbstractLattice, interactions: Dict):
        self.lattice = lattice
        self.interactions = interactions
        self.spin_config = None

    @classmethod
    def from_config(cls, config_path: str):
        """Load from YAML"""

    @property
    def num_sublattices(self) -> int:
        return len(self.lattice.get_basis_positions())
```

**Migration Path**:
1. Keep dictionary format initially for backward compatibility
2. Add `SpinSystem.to_dict()` method
3. Gradually migrate code to use `SpinSystem` objects
4. Remove dictionary format once migration complete

#### 1.3 Refactor Interactions System

**Current**:
- Interactions are defined as methods in `NBCP_UNIT_CELL`
- Exchange matrices computed on-the-fly
- Tightly coupled to specific model (Jxy, Jz, JGamma, JPD)

**Proposed**:
```python
# lswt/core/interactions.py
class Interaction:
    def __init__(self, type: str, sites: Tuple[int, int],
                 matrix: np.ndarray, displacement: np.ndarray):
        self.type = type  # 'exchange', 'anisotropy', 'DM', etc.
        self.sites = sites
        self.matrix = matrix
        self.displacement = displacement

class InteractionBuilder:
    """Helper to construct common interactions"""
    @staticmethod
    def heisenberg(J: float, sites: Tuple[int, int]):
        return Interaction('exchange', sites, J * np.eye(3), ...)

    @staticmethod
    def kitaev_gamma(J: float, Gamma: float, phi: float):
        # Your current compute_J_exch_mat logic
        ...
```

---

### Phase 2: Solvers (Week 3-4)

#### 2.1 Refactor LSWT Solver

**Current**: `linear_spin_wave_theory.py` (LSWT class)
- Good structure, but mixes concerns
- `diagnosing_lswt` is awkward name

**Proposed**: `lswt/solvers/lswt.py`
```python
class LSWTSolver:
    def __init__(self, system: SpinSystem):
        self.system = system
        self.optimizer = None
        self.results = None

    def optimize_spins(self, method='MAGSWT', **kwargs):
        """Find ground state spin configuration"""

    def build_boson_hamiltonian(self):
        """Construct magnon Hamiltonian"""

    def diagonalize(self, k_points: np.ndarray, **kwargs):
        """Solve for eigenvalues/eigenvectors"""

    def compute_thermodynamics(self, temperature: float):
        """Delegate to physics/thermodynamics.py"""
```

#### 2.2 Extract Optimizer

**Current**: `Tools/spin_system_optimizer.py`
- Already well-separated!
- Just needs minor cleanup

**Migration**:
- Move to `lswt/solvers/optimizer.py`
- Rename `SpinSystemOptimizer` → `SpinOptimizer`
- Keep the wrapping and constraint logic (it's good!)

#### 2.3 Brillouin Zone Integration

**Current**: `Tools/brillouin_zone.py` + `find_first_bz.py`
- BZ logic is separate from lattice
- Duplicated BZ type specifications

**Proposed**:
- Move core BZ logic to `lswt/solvers/brillouin_zone.py`
- Make `Lattice` responsible for providing BZ parameters
- Keep specialized BZ calculators in utils or separate module

```python
# In AbstractLattice
def get_brillouin_zone(self) -> BrillouinZone:
    return BrillouinZone(
        reciprocal_vectors=self.get_reciprocal_vectors(),
        high_symmetry_points=self.get_high_symmetry_points()
    )
```

---

### Phase 3: Physics Modules (Week 5-6)

**Good News**: These are already well-separated! Just need reorganization.

#### Migrations:
- `lswt_thermodynamics.py` → `physics/thermodynamics.py`
- `lswt_topology.py` → `physics/topology.py`
- `lswt_correlation.py` → `physics/correlations.py`

**Cleanup**:
- Remove `LSWT_THER`, `LSWT_CORR`, `LSWT_TOPO` class wrappers
- Make standalone functions that take k-space data
- Let `LSWTSolver` coordinate

**Example**:
```python
# Current (in lswt_topology.py)
class LSWT_TOPO:
    def __init__(self, lswt_obj):
        self.lswt = lswt_obj

    def calculate_berry_curvature(self):
        # Uses self.lswt.k_data, self.lswt.Ns, etc.

# Proposed (physics/topology.py)
def calculate_berry_curvature(eigenvalues, eigenvectors, k_points):
    """Calculate Berry curvature at k-points"""
    # Standalone function, no LSWT dependency
```

---

### Phase 4: Analysis & Visualization (Week 7-8)

#### 4.1 Phase Finding/Diagrams

**Current**:
- `nbcp_phases.py` (FIND_TLAF_PHASES)
- `PhaseDiagram/` module

**Issues**:
- Too NBCP-specific
- Should work for any spin system

**Proposed**:
```python
# lswt/analysis/phase_finder.py
class PhaseFinder:
    def __init__(self, system: SpinSystem):
        self.system = system

    def find_ground_state(self, angle_constraints=None):
        """Find lowest energy configuration"""

    def identify_phase(self, config):
        """Classify the phase (q-vector, symmetry)"""

# lswt/analysis/phase_diagram.py
class PhaseDiagram:
    def __init__(self, system_template: SpinSystem):
        self.template = system_template

    def scan_parameter(self, param_name, param_range):
        """Scan a parameter and find phases"""
```

#### 4.2 Visualization

**Current**: Already well-separated in `Plotters/`

**Migration**:
- Move to `lswt/visualization/`
- Minimal changes needed
- Add `visualizer.py` module for common plotting helpers

---

### Phase 5: Configuration System (Week 9)

#### 5.1 YAML Configuration

**Create**: `lswt/io/config_loader.py`

**Example config.yaml**:
```yaml
project:
  name: "NBCP Ground State"
  description: "Triangular lattice with Kitaev-Gamma interactions"

lattice:
  type: triangular
  lattice_constant: 1.0
  basis_sites: 1  # or 2, 3, 4 for q=0 phases

interactions:
  nearest_neighbor:
    Jxy: 0.076
    Jz: 0.125
    JGamma: 0.1
    JPD: 0.00

  anisotropy:
    Kxy: 0.0
    Kz: 0.0

magnetic_field:
  h: [0.0, 0.0, 0.376418]

calculation:
  spin: 0.5
  method: MAGSWT
  k_points: 20
  temperature: 0

optimization:
  angle_constraints:
    sublattice_0: [null, 0]  # theta free, phi=0
```

#### 5.2 Validation

**Create**: `lswt/io/validators.py`

```python
class ConfigValidator:
    def validate(self, config: dict):
        self.validate_schema(config)
        self.validate_types(config)
        self.validate_physics(config)

    def validate_physics(self, config):
        # S > 0
        # Lattice constant > 0
        # Check interaction matrix hermiticity
```

---

## Key Design Decisions to Review

### 1. ✅ Keep: UI Separation

**CLAUDE.md Proposal**: Separate `lswt_ui` package

**Recommendation**: Absolutely keep this design
- Current code has no UI (scripts are not UI)
- CLI can come later
- Focus on computation engine first

### 2. ⚠️ Reconsider: Hamiltonian Module Structure

**CLAUDE.md**: `core/hamiltonian.py` as separate module

**Current**: `lswt_Hamiltonian.py` inside LinearSpinWaveTheory

**Issue**: Hamiltonian construction is specific to LSWT (Holstein-Primakoff, Bogoliubov)

**Recommendation**:
```
lswt/
├── core/
│   ├── interactions.py          # Physical interactions (exchange, DM, etc.)
│   └── spin_system.py           # SpinSystem class
├── solvers/
│   └── lswt/
│       ├── hamiltonian.py       # LSWT-specific H construction
│       └── solver.py            # LSWTSolver
```

This makes it clear that Hamiltonian is LSWT-specific, not a general "core" concept.

### 3. ✅ Keep: Single Config File

**CLAUDE.md Proposal**: All configuration in one YAML file

**Recommendation**: Excellent choice
- Research workflow benefits from seeing everything
- Easy git diff
- Can split later if needed with `include:` directive

### 4. 🆕 Add: Result Serialization

**Missing from CLAUDE.md**: How to save/load results

**Recommendation**: Add to architecture
```python
# lswt/io/results.py
class LSWTResults:
    def __init__(self, system, solver_output, metadata):
        self.system = system
        self.eigenvalues = solver_output['eigenvalues']
        self.eigenvectors = solver_output['eigenvectors']
        # ...

    def save(self, path: str):
        """Save to HDF5 or npz"""

    @classmethod
    def load(cls, path: str):
        """Load from file"""
```

### 5. 🔄 Refine: BrillouinZone Design

**CLAUDE.md**: `solvers/brillouin_zone.py`

**Issue**: BZ is both:
- A property of the lattice (geometry)
- A sampling strategy for calculations (solver concern)

**Recommendation**: Split responsibilities
```
core/lattice/brillouin_zone.py  # Geometry: reciprocal vectors, HSPs
solvers/k_sampling.py            # Sampling: generate k-points, paths
```

---

## Migration Strategy

### Option A: Incremental (Recommended)

1. **Create new structure alongside old**
   ```
   modules/          # Keep existing
   lswt/             # New structure
   ```

2. **Migrate module-by-module**
   - Start with `core/lattice`
   - Add abstractions
   - Keep old code working

3. **Dual imports during transition**
   ```python
   # In lswt/__init__.py, support both:
   from lswt.core import SpinSystem  # New
   from lswt.compat import NBCP_UNIT_CELL  # Old, deprecated
   ```

4. **Update scripts gradually**
   - `1_do_it_swt.py` keeps working
   - Create `examples/01_basic.py` with new API
   - Eventually remove old scripts

5. **Delete old code after full migration**

### Option B: Big Bang (Not Recommended)

- Rewrite everything at once
- High risk of breaking functionality
- Hard to test incrementally

---

## Immediate Next Steps

### Week 1 Tasks

1. **Create directory structure**
   ```bash
   mkdir -p lswt/{core/{lattice,},solvers,physics,analysis,visualization,io,utils}
   ```

2. **Define AbstractLattice interface**
   - File: `lswt/core/lattice/base.py`
   - Key methods documented
   - Docstrings with physics context

3. **Extract TriangularLattice**
   - File: `lswt/core/lattice/presets.py`
   - Migrate geometry from `NBCP_UNIT_CELL`
   - Add unit tests

4. **Create SpinSystem scaffold**
   - File: `lswt/core/spin_system.py`
   - Define data model
   - Add `to_dict()` for compatibility

5. **Write first example**
   - File: `examples/00_migration_test.py`
   - Use new `TriangularLattice` + old LSWT
   - Verify compatibility

---

## Testing Strategy

### Unit Tests (High Priority)

Create `tests/test_core/`:
- `test_lattice.py`: Verify lattice geometry
  ```python
  def test_triangular_lattice_vectors():
      lattice = TriangularLattice()
      a1, a2 = lattice.get_primitive_vectors()
      assert np.allclose(a1, [1.0, 0.0])
      assert np.allclose(a2, [0.5, np.sqrt(3)/2])
  ```

- `test_spin_system.py`: Check SpinSystem construction
- `test_interactions.py`: Validate interaction matrices

### Integration Tests

Create `tests/test_integration/`:
- `test_nbcp_compatibility.py`: Ensure old results reproduce
  ```python
  def test_nbcp_ground_state_matches():
      # Old way
      old_result = ... # Using NBCP_UNIT_CELL

      # New way
      lattice = TriangularLattice()
      system = SpinSystem(lattice, interactions)
      new_result = ...

      assert np.allclose(old_result, new_result)
  ```

### Regression Tests

- Save current results as reference
- Compare after refactoring
- Use `tests/data/reference/nbcp_ground_state.npz`

---

## Risks and Mitigation

### Risk 1: Breaking existing research code

**Mitigation**:
- Keep old code working during transition
- Extensive integration tests
- Reproduce published results

### Risk 2: Over-abstraction

**Symptom**: Code becomes harder to understand
**Mitigation**:
- Follow "Clarity First" principle
- Don't abstract until pattern is clear
- Document physics meaning in all abstractions

### Risk 3: Incomplete migration

**Symptom**: Stuck with half-old, half-new code
**Mitigation**:
- Define clear migration milestones
- Complete each phase fully before moving on
- Track deprecations

### Risk 4: Performance regression

**Mitigation**:
- Benchmark critical paths (diagonalization, BZ sampling)
- Profile before and after
- Ensure NumPy/SciPy usage is optimal

---

## Recommendations Summary

### ✅ Proceed with Refactoring

The architecture in CLAUDE.md is excellent. The benefits outweigh the costs:

**Benefits**:
- Generic lattice support (square, honeycomb, kagome, custom)
- Type-safe SpinSystem (vs dictionaries)
- YAML-based reproducibility
- Better organized code
- Easier collaboration
- Publication-ready package

**Costs**:
- 2-3 months refactoring time
- Risk of breaking existing code (mitigable)
- Learning curve for collaborators

### 🎯 Refinements to CLAUDE.md

1. **Add**: Result serialization (HDF5/npz)
2. **Split**: BrillouinZone into geometry + sampling
3. **Clarify**: Hamiltonian belongs in solvers/lswt/, not core/
4. **Add**: Compatibility layer for migration period
5. **Add**: Integration test strategy

### 📋 Suggested Order

1. **Phase 1**: Core abstractions (lattice, spin_system) - **CRITICAL**
2. **Phase 2**: Solvers (optimizer, lswt) - **HIGH**
3. **Phase 3**: Configuration (YAML, validation) - **HIGH**
4. **Phase 4**: Physics modules - **MEDIUM** (already good)
5. **Phase 5**: Analysis/Visualization - **MEDIUM**
6. **Phase 6**: Documentation - **LOW** (ongoing)

---

## Questions for You

1. **Timeline**: Do you have 2-3 months for this refactoring?
2. **Collaborators**: Will others need to use the code during migration?
3. **Priority lattices**: Which lattices do you need first? (triangular, square, honeycomb, kagome)
4. **Backward compatibility**: How important is keeping old scripts working?
5. **Testing**: Do you have reference results for validation?

---

## Conclusion

Your current code is solid, but NBCP-specific. The CLAUDE.md architecture will transform it into a general, reusable LSWT package. The refactoring is well-justified and the design is sound.

**My recommendation**: Proceed with incremental migration, starting with Phase 1 (lattice abstractions).

Let me know which aspects you'd like to discuss further or if you'd like me to start implementing specific parts!
