"""
Phase 1A Demo: New Architecture

This example demonstrates the new core abstractions:
- TriangularLattice (geometry only)
- CommensurateStructure (magnetic configuration)
- SpinSystem (combines everything)

This replaces the old dictionary-based approach with typed objects.
"""

import numpy as np
import sys
from pathlib import Path

# Add lswt to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lswt.core import (
    TriangularLattice,
    CommensurateStructure,
    SpinSystem
)


def example_ferromagnet():
    """Example 1: Simple ferromagnet."""
    print("="*60)
    print("Example 1: Ferromagnetic structure on triangular lattice")
    print("="*60)

    # Create triangular lattice
    lattice = TriangularLattice(lattice_constant=1.0)
    print(f"\nLattice: {lattice}")

    # Primitive vectors
    a1, a2 = lattice.get_primitive_vectors()
    print(f"Primitive vectors:")
    print(f"  a1 = {a1}")
    print(f"  a2 = {a2}")

    # Nearest neighbors
    nn = lattice.get_neighbors(order=1)
    print(f"\nNearest neighbors: {len(nn)} bonds")
    for i, neighbor in enumerate(nn):
        print(f"  Bond {i+1}: displacement = {neighbor.displacement}, "
              f"distance = {neighbor.distance:.3f}")

    # Ferromagnetic structure (all spins up)
    mag_struct = CommensurateStructure(
        num_basis_sites=1,
        magnetic_supercell=(1, 1),
        angles=np.array([[0.0, 0.0]])  # θ=0 → z-direction
    )
    print(f"\nMagnetic structure: {mag_struct}")

    # Define interactions (ferromagnetic)
    interactions = {
        'nearest_neighbor': {
            'Jxy': -1.0,  # Negative → FM
            'Jz': -1.0
        }
    }

    # Create spin system
    system = SpinSystem(
        lattice=lattice,
        magnetic_structure=mag_struct,
        interactions=interactions,
        spin_magnitude=0.5
    )

    print(f"\n{system}")


def example_120_degree():
    """Example 2: 120° structure (NBCP-type)."""
    print("\n" + "="*60)
    print("Example 2: 120° structure on triangular lattice (NBCP)")
    print("="*60)

    # Lattice
    lattice = TriangularLattice(lattice_constant=1.0)

    # 120° magnetic structure (3-sublattice, spins in xy-plane)
    angles_120 = np.array([
        [np.pi/2, 0.0],           # θ=π/2, φ=0
        [np.pi/2, 2*np.pi/3],     # θ=π/2, φ=2π/3
        [np.pi/2, 4*np.pi/3]      # θ=π/2, φ=4π/3
    ])

    mag_struct = CommensurateStructure(
        num_basis_sites=1,
        magnetic_supercell=(1, 1),
        angles=angles_120
    )

    print(f"\nMagnetic structure: {mag_struct}")

    # Get spin directions
    spins = mag_struct.get_all_spin_directions()
    print("\nSpin directions:")
    for i, spin in enumerate(spins):
        print(f"  Sublattice {i}: S = [{spin[0]:+.3f}, {spin[1]:+.3f}, {spin[2]:+.3f}]")

    # Check total magnetization (should be ~0 for AFM)
    M_total = mag_struct.get_total_magnetization()
    print(f"\nTotal magnetization: M = [{M_total[0]:.6f}, {M_total[1]:.6f}, {M_total[2]:.6f}]")
    print(f"|M| = {np.linalg.norm(M_total):.6f} (should be ~0 for AFM)")

    # NBCP interactions
    interactions = {
        'nearest_neighbor': {
            'Jxy': 0.076,
            'Jz': 0.125,
            'JGamma': 0.1,
            'JPD': 0.0
        },
        'next_nearest_neighbor': {
            'Kxy': 0.0,
            'Kz': 0.0,
            'KGamma': 0.0,
            'KPD': 0.0
        },
        'magnetic_field': {
            'h': [0.0, 0.0, 0.376418]
        }
    }

    # Create system
    system = SpinSystem(
        lattice=lattice,
        magnetic_structure=mag_struct,
        interactions=interactions,
        spin_magnitude=0.5,
        metadata={'name': 'NBCP 120° phase', 'temperature': 0.0}
    )

    print(f"\n{system}")


def example_serialization():
    """Example 3: Serialization and comparison to legacy format."""
    print("\n" + "="*60)
    print("Example 3: Serialization to dict (for YAML/JSON)")
    print("="*60)

    # Create a simple system
    lattice = TriangularLattice(lattice_constant=1.0)
    mag_struct = CommensurateStructure(
        num_basis_sites=1,
        magnetic_supercell=(1, 1),
        angles=np.array([[np.pi/4, 0.0]])
    )
    interactions = {'nearest_neighbor': {'J': 1.0}}
    system = SpinSystem(lattice, mag_struct, interactions)

    # Serialize to dict (new format)
    data = system.to_dict()

    print("\nNew format (SpinSystem.to_dict()):")
    for key, value in data.items():
        print(f"  {key}: {value}")

    print("\n" + "-"*60)

    # Legacy format (for backward compatibility)
    print("\nLegacy format (for compatibility with current modules/):")
    try:
        legacy_data = system.to_legacy_dict()
        print("Keys:", list(legacy_data.keys()))
        print("\nSpin info sample:")
        for key, value in list(legacy_data['Spin info'].items())[:1]:
            print(f"  Sublattice {key}: {value}")
    except Exception as e:
        print(f"(Legacy format conversion: {e})")


def example_architecture_benefits():
    """Example 4: Show architectural benefits."""
    print("\n" + "="*60)
    print("Example 4: Architectural Benefits")
    print("="*60)

    # Same lattice, different magnetic structures
    lattice = TriangularLattice(lattice_constant=1.0)

    print("\n1. SAME LATTICE with different magnetic structures:")
    print("-" * 60)

    # FM
    fm_struct = CommensurateStructure(1, (1, 1), np.array([[0, 0]]))
    fm_system = SpinSystem(lattice, fm_struct, {'J': -1})
    print(f"FM:  {fm_system}")

    # 120°
    angles_120 = np.array([[np.pi/2, 0], [np.pi/2, 2*np.pi/3], [np.pi/2, 4*np.pi/3]])
    afm_struct = CommensurateStructure(1, (1, 1), angles_120)
    afm_system = SpinSystem(lattice, afm_struct, {'J': +1})
    print(f"120°: {afm_system}")

    print("\n2. CLEAN SEPARATION of concerns:")
    print("-" * 60)
    print("✓ Lattice = pure geometry (reusable)")
    print("✓ MagneticStructure = spin pattern (mutable for optimization)")
    print("✓ SpinSystem = combines all (passed to solvers)")

    print("\n3. FUTURE-PROOF for incommensurate:")
    print("-" * 60)
    print("✓ Interface already supports both commensurate & incommensurate")
    print("✓ is_commensurate property to branch")
    print("✓ IncommensurateStructure stub in place for Phase 5+")


if __name__ == '__main__':
    example_ferromagnet()
    example_120_degree()
    example_serialization()
    example_architecture_benefits()

    print("\n" + "="*60)
    print("Phase 1A Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  - Run unit tests: pytest tests/")
    print("  - Phase 1B: Implement LSWT solver using new architecture")
    print("  - Phase 2: Migrate existing modules/ code")
