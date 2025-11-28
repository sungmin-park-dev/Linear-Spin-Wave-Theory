"""
SpinSystem: Complete representation of a magnetic system.

This module defines the SpinSystem class which combines:
- Crystallographic lattice (geometry)
- Magnetic structure (spin configuration)
- Interactions (exchange, anisotropy, DM, etc.)

This is the main object that gets passed to solvers and analysis tools.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from .lattice import AbstractLattice
from .magnetic_structure import AbstractMagneticStructure


class SpinSystem:
    """
    Complete physical representation of a magnetic spin system.

    This class combines three independent components:
    1. Lattice: crystallographic geometry (AbstractLattice)
    2. Magnetic structure: spin configuration pattern (AbstractMagneticStructure)
    3. Interactions: physical couplings (exchange, anisotropy, etc.)

    Design Philosophy
    -----------------
    Clean separation of concerns:
    - Lattice determines GEOMETRY (neighbors, distances, BZ)
    - MagneticStructure determines SPIN PATTERN (commensurate/incommensurate)
    - Interactions determine PHYSICS (energy scales, anisotropies)
    - SpinSystem COMBINES all three

    This enables:
    - Same lattice with different magnetic structures
    - Same magnetic structure with different interactions
    - Easy optimization (magnetic structure is mutable)
    - Clear serialization (each component serializes independently)

    Parameters
    ----------
    lattice : AbstractLattice
        Crystallographic lattice defining geometry
    magnetic_structure : AbstractMagneticStructure
        Spin configuration (commensurate or incommensurate)
    interactions : Dict
        Interaction parameters. Structure depends on model type.
        For NBCP-style models:
            {
                'nearest_neighbor': {
                    'Jxy': float, 'Jz': float, 'JGamma': float, 'JPD': float
                },
                'next_nearest_neighbor': {
                    'Kxy': float, 'Kz': float, 'KGamma': float, 'KPD': float
                },
                'magnetic_field': {'h': [hx, hy, hz]},
                'single_ion_anisotropy': {'D': [Dx, Dy, Dz]}  # optional
            }
    spin_magnitude : float, optional
        Spin quantum number S (default: 0.5)
    metadata : Dict, optional
        Additional information (project name, description, etc.)

    Attributes
    ----------
    lattice : AbstractLattice
        The crystallographic lattice
    magnetic_structure : AbstractMagneticStructure
        The magnetic ordering pattern
    interactions : Dict
        Interaction parameters
    spin_magnitude : float
        Spin quantum number
    metadata : Dict
        Additional metadata

    Examples
    --------
    Create a simple ferromagnetic system on triangular lattice:

    >>> from lswt.core.lattice import TriangularLattice
    >>> from lswt.core.magnetic_structure import CommensurateStructure
    >>>
    >>> # Geometry
    >>> lattice = TriangularLattice(lattice_constant=1.0)
    >>>
    >>> # All spins up in z-direction
    >>> mag_struct = CommensurateStructure(
    ...     num_basis_sites=1,
    ...     magnetic_supercell=(1, 1),
    ...     angles=np.array([[0.0, 0.0]])  # θ=0 → ẑ
    ... )
    >>>
    >>> # Interactions
    >>> interactions = {
    ...     'nearest_neighbor': {'Jxy': -1.0, 'Jz': -1.0}  # Negative = FM
    ... }
    >>>
    >>> # Combine
    >>> system = SpinSystem(lattice, mag_struct, interactions)
    >>> print(system)

    Create NBCP-type system (3-sublattice on triangular lattice):

    >>> angles_120 = np.array([
    ...     [np.pi/2, 0.0],
    ...     [np.pi/2, 2*np.pi/3],
    ...     [np.pi/2, 4*np.pi/3]
    ... ])
    >>> mag_struct = CommensurateStructure(
    ...     num_basis_sites=1,
    ...     magnetic_supercell=(1, 1),
    ...     angles=angles_120
    ... )
    >>> interactions = {
    ...     'nearest_neighbor': {
    ...         'Jxy': 0.076, 'Jz': 0.125,
    ...         'JGamma': 0.1, 'JPD': 0.0
    ...     },
    ...     'magnetic_field': {'h': [0, 0, 0.376]}
    ... }
    >>> system = SpinSystem(lattice, mag_struct, interactions, spin_magnitude=0.5)

    Load from YAML configuration:

    >>> system = SpinSystem.from_config('config.yaml')
    """

    def __init__(self,
                 lattice: AbstractLattice,
                 magnetic_structure: AbstractMagneticStructure,
                 interactions: Dict,
                 spin_magnitude: float = 0.5,
                 metadata: Optional[Dict] = None):
        """
        Initialize SpinSystem.

        Parameters
        ----------
        lattice : AbstractLattice
            Crystallographic lattice
        magnetic_structure : AbstractMagneticStructure
            Magnetic ordering pattern
        interactions : Dict
            Interaction parameters
        spin_magnitude : float
            Spin quantum number S
        metadata : Dict, optional
            Additional metadata
        """
        # Validation
        if not isinstance(lattice, AbstractLattice):
            raise TypeError("lattice must be an AbstractLattice instance")

        if not isinstance(magnetic_structure, AbstractMagneticStructure):
            raise TypeError("magnetic_structure must be an AbstractMagneticStructure instance")

        if spin_magnitude <= 0:
            raise ValueError("spin_magnitude must be positive")

        # Store components
        self.lattice = lattice
        self.magnetic_structure = magnetic_structure
        self.interactions = interactions
        self.spin_magnitude = spin_magnitude
        self.metadata = metadata or {}

    @property
    def is_commensurate(self) -> bool:
        """Check if magnetic structure is commensurate."""
        return self.magnetic_structure.is_commensurate()

    @property
    def num_basis_sites(self) -> int:
        """Number of sites in crystallographic unit cell."""
        return len(self.lattice.get_basis_positions())

    @property
    def num_magnetic_sublattices(self) -> Optional[int]:
        """
        Number of magnetic sublattices.

        Returns None for incommensurate structures.
        """
        return self.magnetic_structure.get_num_magnetic_sublattices()

    @property
    def lattice_constant(self) -> float:
        """Characteristic lattice constant."""
        return self.lattice.get_lattice_constant()

    def to_dict(self) -> Dict:
        """
        Serialize to dictionary.

        Returns
        -------
        data : Dict
            Complete representation suitable for saving to YAML/JSON

        Notes
        -----
        This creates a complete snapshot that can be used to:
        - Save to configuration file
        - Save with results for reproducibility
        - Share with collaborators
        """
        return {
            'lattice': {
                'type': self.lattice.__class__.__name__,
                'lattice_constant': self.lattice_constant,
                # Additional lattice-specific parameters could go here
            },
            'magnetic_structure': self.magnetic_structure.to_dict(),
            'interactions': self.interactions,
            'spin_magnitude': self.spin_magnitude,
            'metadata': self.metadata
        }

    @classmethod
    def from_config(cls, config_path: str) -> 'SpinSystem':
        """
        Load system from YAML configuration file.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file

        Returns
        -------
        system : SpinSystem
            Loaded spin system

        Notes
        -----
        Configuration file format:
            lattice:
              type: triangular
              lattice_constant: 1.0

            magnetic_structure:
              type: commensurate
              num_basis_sites: 1
              magnetic_supercell: [1, 1]
              angles:
                - [1.5708, 0.0]
                - [1.5708, 2.0944]
                - [1.5708, 4.1888]

            interactions:
              nearest_neighbor:
                Jxy: 0.076
                Jz: 0.125
                JGamma: 0.1
                JPD: 0.0
              magnetic_field:
                h: [0.0, 0.0, 0.376]

            calculation:
              spin_magnitude: 0.5

            project:
              name: "NBCP Ground State"
              description: "..."

        Raises
        ------
        NotImplementedError
            This method requires the io.config_loader module (Phase 5)
        """
        raise NotImplementedError(
            "SpinSystem.from_config() requires the config loader module.\n"
            "This will be implemented in Phase 5 (Configuration System).\n"
            "\n"
            "For now, construct SpinSystem manually:\n"
            "  lattice = TriangularLattice(...)\n"
            "  mag_struct = CommensurateStructure(...)\n"
            "  system = SpinSystem(lattice, mag_struct, interactions)\n"
        )

    def to_legacy_dict(self) -> Dict:
        """
        Convert to legacy spin_system_data format for backward compatibility.

        Returns
        -------
        spin_system_data : Dict
            Dictionary in the format expected by current modules/ code:
            {
                'Spin info': {...},
                'Couplings': {...},
                'Lattice/BZ setting': {...}
            }

        Notes
        -----
        This is a temporary compatibility layer to allow incremental migration.
        Will be removed once all code uses the new SpinSystem interface.

        Only works for commensurate structures on single-atom basis lattices.
        """
        if not self.is_commensurate:
            raise ValueError("Legacy format only supports commensurate structures")

        if self.num_basis_sites != 1:
            raise ValueError("Legacy format only supports single-atom basis")

        # Build spin info
        spin_info = {}
        for i in range(self.num_magnetic_sublattices):
            spin_dir = self.magnetic_structure.get_spin_direction(
                site_index=0,
                unit_cell=(i, 0)  # Simplified - assumes 1D magnetic ordering
            )
            theta = np.arccos(spin_dir[2])
            phi = np.arctan2(spin_dir[1], spin_dir[0])

            # Position (simplified - would need proper mapping for general case)
            position = [0.0, 0.0]  # Placeholder

            spin_info[i] = {
                'Spin': self.spin_magnitude,
                'Angles': [theta, phi],
                'Position': position,
                'Magnetic Field': self.interactions.get('magnetic_field', {}).get('h', [0, 0, 0])
            }

        # Build couplings (simplified - assumes NBCP format)
        couplings = {
            'NN': self.interactions.get('nearest_neighbor', {}),
            'NNN': self.interactions.get('next_nearest_neighbor', {})
        }

        # Lattice/BZ setting
        lattice_vectors = self.lattice.get_primitive_vectors()
        lattice_bz_setting = (
            (lattice_vectors[0], lattice_vectors[1]),
            'Hex_60'  # Hardcoded for now - would need to be more sophisticated
        )

        return {
            'Spin info': spin_info,
            'Couplings': couplings,
            'Lattice/BZ setting': lattice_bz_setting
        }

    def __repr__(self) -> str:
        """String representation."""
        lattice_name = self.lattice.__class__.__name__
        mag_type = "commensurate" if self.is_commensurate else "incommensurate"
        num_sl = self.num_magnetic_sublattices

        return (f"SpinSystem(lattice={lattice_name}, "
                f"structure={mag_type}, "
                f"sublattices={num_sl}, "
                f"S={self.spin_magnitude})")

    def __str__(self) -> str:
        """Detailed string representation."""
        lines = [
            "="*50,
            "Spin System",
            "="*50,
            f"Lattice: {self.lattice}",
            f"Magnetic Structure: {self.magnetic_structure}",
            f"Spin Magnitude: S = {self.spin_magnitude}",
            f"Commensurate: {self.is_commensurate}",
            f"Number of Magnetic Sublattices: {self.num_magnetic_sublattices}",
            "",
            "Interactions:",
        ]

        for key, value in self.interactions.items():
            lines.append(f"  {key}: {value}")

        if self.metadata:
            lines.append("")
            lines.append("Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")

        lines.append("="*50)

        return "\n".join(lines)
