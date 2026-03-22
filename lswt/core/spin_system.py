"""SpinSystem: the central data structure combining lattice, spins, and interactions.

A SpinSystem holds all information needed for spin system solvers:
- Spin sites with positions, angles, spin magnitude, and magnetic field
- Couplings with exchange matrices and displacement vectors
- Lattice vectors

SpinSystem is solver-agnostic: it does not contain BZ type or any
solver-specific logic. Solver parameters (e.g., bz_type, regularization)
are passed to the solver directly.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Union


class SpinSystem:
    """Solver-agnostic spin system definition.

    Combines spin sites, couplings, and lattice information into a single
    object that can be passed to any solver (LSWT, ED, BdG, ...).

    Parameters
    ----------
    sites : list of SpinSystem.Site
        Magnetic sites in the magnetic unit cell.
    couplings : list of SpinSystem.Coupling
        Exchange couplings between sites.
    lattice_vectors : array_like
        Primitive lattice vectors, shape (2, 2): [a1, a2].

    Examples
    --------
    >>> import numpy as np
    >>> from lswt import SpinSystem
    >>> from lswt.core import exchange
    >>>
    >>> a1 = np.array([1.0, 0.0])
    >>> a2 = np.array([0.5, np.sqrt(3)/2])
    >>>
    >>> sites = [
    ...     SpinSystem.Site("A", [0, 0], spin=0.5, angles=[np.pi/2, 0],
    ...                     magnetic_field=[0, 0, 0.3]),
    ...     SpinSystem.Site("B", [0.5, 0.866], spin=0.5,
    ...                     angles=[np.pi/2, 2*np.pi/3],
    ...                     magnetic_field=[0, 0, 0.3]),
    ... ]
    >>> J = exchange.heisenberg(1.0)
    >>> couplings = [SpinSystem.Coupling(0, 1, J, [1.0, 0.0])]
    >>> system = SpinSystem(sites, couplings, [a1, a2])
    >>> system.site("A").spin
    0.5
    """

    # ------------------------------------------------------------------
    # Nested data classes
    # ------------------------------------------------------------------

    @dataclass
    class Site:
        """A single magnetic site in the unit cell.

        Parameters
        ----------
        label : str
            Site label (e.g., 'A', 'B', 'C').
        position : array_like
            2D position within the unit cell.
        spin : float
            Spin magnitude S (e.g., 0.5 for spin-1/2).
        angles : array_like
            Spherical angles (theta, phi) for spin direction.
        magnetic_field : array_like
            External magnetic field vector (3D).
        """
        label: str
        position: np.ndarray
        spin: float
        angles: np.ndarray
        magnetic_field: np.ndarray

        def __post_init__(self):
            self.position = np.asarray(self.position, dtype=float)
            self.angles = np.asarray(self.angles, dtype=float)
            self.magnetic_field = np.asarray(self.magnetic_field, dtype=float)
            if self.position.shape != (2,):
                raise ValueError(
                    f"position must be 2D, got shape {self.position.shape}")
            if self.angles.shape != (2,):
                raise ValueError(
                    f"angles must be (theta, phi), got shape {self.angles.shape}")
            if self.magnetic_field.shape != (3,):
                raise ValueError(
                    f"magnetic_field must be 3D, got shape {self.magnetic_field.shape}")

        def __repr__(self):
            return (f"Site('{self.label}', pos={self.position}, "
                    f"S={self.spin}, angles={self.angles})")

    @dataclass
    class Coupling:
        """A bilinear spin-spin coupling: H = S_i^T J S_j.

        Parameters
        ----------
        site_i : int
            Index of site i.
        site_j : int
            Index of site j.
        exchange_matrix : array_like
            3x3 exchange matrix J_ij.
        displacement : array_like
            Real-space displacement vector from site i to site j.
        """
        site_i: int
        site_j: int
        exchange_matrix: np.ndarray
        displacement: np.ndarray

        def __post_init__(self):
            self.exchange_matrix = np.asarray(self.exchange_matrix, dtype=float)
            self.displacement = np.asarray(self.displacement, dtype=float)
            if self.exchange_matrix.shape != (3, 3):
                raise ValueError(
                    f"exchange_matrix must be 3x3, got {self.exchange_matrix.shape}")
            if self.displacement.shape != (2,):
                raise ValueError(
                    f"displacement must be 2D, got {self.displacement.shape}")

    # ------------------------------------------------------------------
    # SpinSystem constructor
    # ------------------------------------------------------------------

    def __init__(self, sites, couplings, lattice_vectors):
        if not sites:
            raise ValueError("Must provide at least one Site")
        self.sites = tuple(sites)
        self.couplings = tuple(couplings)
        self.lattice_vectors = np.asarray(lattice_vectors, dtype=float)

        # Build label -> index map
        self._label_map: Dict[str, int] = {}
        for i, s in enumerate(self.sites):
            if s.label in self._label_map:
                raise ValueError(f"Duplicate site label: '{s.label}'")
            self._label_map[s.label] = i

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_sites(self) -> int:
        """Number of magnetic sites."""
        return len(self.sites)

    @property
    def num_sublattices(self) -> int:
        """Number of magnetic sublattices (alias for num_sites)."""
        return self.num_sites

    @property
    def num_couplings(self) -> int:
        """Number of couplings."""
        return len(self.couplings)

    # ------------------------------------------------------------------
    # Access methods
    # ------------------------------------------------------------------

    def site(self, key: Union[str, int]) -> "SpinSystem.Site":
        """Access a site by label or index.

        Parameters
        ----------
        key : str or int
            Site label (e.g., 'A') or site index (e.g., 0).

        Returns
        -------
        Site
            The requested site.

        Examples
        --------
        >>> system.site("A").position
        array([0., 0.])
        >>> system.site(0).spin
        0.5
        """
        if isinstance(key, str):
            if key not in self._label_map:
                raise KeyError(f"No site with label '{key}'")
            return self.sites[self._label_map[key]]
        elif isinstance(key, (int, np.integer)):
            return self.sites[key]
        else:
            raise TypeError(f"key must be str or int, got {type(key)}")

    def get_couplings(self, site_i: Union[str, int],
                      site_j: Union[str, int, None] = None
                      ) -> List["SpinSystem.Coupling"]:
        """Filter couplings by site label or index.

        Parameters
        ----------
        site_i : str or int
            First site label or index.
        site_j : str, int, or None, optional
            Second site label or index. If None, returns all couplings
            involving site_i.

        Returns
        -------
        list of Coupling
            Matching couplings.

        Examples
        --------
        >>> system.get_couplings("A", "B")  # A-B couplings only
        >>> system.get_couplings("A")        # all couplings involving A
        """
        idx_i = self._resolve_index(site_i)

        if site_j is None:
            return [c for c in self.couplings
                    if c.site_i == idx_i or c.site_j == idx_i]
        else:
            idx_j = self._resolve_index(site_j)
            return [c for c in self.couplings
                    if (c.site_i == idx_i and c.site_j == idx_j)
                    or (c.site_i == idx_j and c.site_j == idx_i)]

    def _resolve_index(self, key: Union[str, int]) -> int:
        """Convert label or index to integer index."""
        if isinstance(key, str):
            if key not in self._label_map:
                raise KeyError(f"No site with label '{key}'")
            return self._label_map[key]
        elif isinstance(key, (int, np.integer)):
            return int(key)
        else:
            raise TypeError(f"key must be str or int, got {type(key)}")

    # ------------------------------------------------------------------
    # Angle manipulation
    # ------------------------------------------------------------------

    def update_angles(self, angles):
        """Update spin angles for all sites.

        Parameters
        ----------
        angles : array_like
            Flat array [theta_0, phi_0, theta_1, phi_1, ...] of length
            2*num_sites, or array of shape (num_sites, 2).
        """
        angles = np.asarray(angles, dtype=float)
        if angles.ndim == 1:
            if len(angles) != 2 * self.num_sites:
                raise ValueError(
                    f"Expected {2 * self.num_sites} angles, got {len(angles)}")
            for i, site in enumerate(self.sites):
                site.angles = angles[2 * i:2 * i + 2].copy()
        elif angles.ndim == 2:
            if angles.shape != (self.num_sites, 2):
                raise ValueError(
                    f"Expected shape ({self.num_sites}, 2), got {angles.shape}")
            for i, site in enumerate(self.sites):
                site.angles = angles[i].copy()
        else:
            raise ValueError("angles must be 1D or 2D array")

    def get_angles_flat(self) -> np.ndarray:
        """Return all spin angles as flat array.

        Returns
        -------
        np.ndarray
            [theta_0, phi_0, theta_1, phi_1, ...].
        """
        return np.concatenate([site.angles for site in self.sites])

    # ------------------------------------------------------------------
    # Legacy compatibility (to be gradually removed)
    # ------------------------------------------------------------------

    def to_legacy_dict(self, bz_type: str = "Hex_60") -> dict:
        """Convert to legacy spin_system_data dict format.

        Parameters
        ----------
        bz_type : str, optional
            Brillouin zone type for the legacy format (default: 'Hex_60').

        Returns
        -------
        dict
            Legacy format: {"Spin info": ..., "Couplings": ...,
            "Lattice/BZ setting": ...}.
        """
        spin_info = {}
        for site in self.sites:
            spin_info[site.label] = {
                "Position": tuple(site.position),
                "Spin": site.spin,
                "Angles": tuple(site.angles),
                "Magnetic Field": site.magnetic_field.copy(),
            }

        couplings_list = []
        for c in self.couplings:
            couplings_list.append({
                "SpinI": self.sites[c.site_i].label,
                "SpinJ": self.sites[c.site_j].label,
                "Exchange Matrix": c.exchange_matrix.copy(),
                "Displacement": tuple(c.displacement),
            })

        a1 = self.lattice_vectors[0]
        a2 = self.lattice_vectors[1]
        lattice_bz_setting = ((a1, a2), bz_type)

        return {
            "Spin info": spin_info,
            "Couplings": couplings_list,
            "Lattice/BZ setting": lattice_bz_setting,
        }

    @classmethod
    def from_legacy_dict(cls, data) -> "SpinSystem":
        """Create SpinSystem from legacy spin_system_data dict.

        Parameters
        ----------
        data : dict
            Legacy format with keys "Spin info", "Couplings",
            "Lattice/BZ setting".

        Returns
        -------
        SpinSystem
            New SpinSystem instance.
        """
        spin_info = data["Spin info"]
        label_to_idx = {}
        sites = []
        for i, (label, info) in enumerate(spin_info.items()):
            label_to_idx[label] = i
            sites.append(SpinSystem.Site(
                label=label,
                position=np.array(info["Position"]),
                spin=info["Spin"],
                angles=np.array(info["Angles"]),
                magnetic_field=np.array(info["Magnetic Field"]),
            ))

        couplings = []
        for c in data["Couplings"]:
            couplings.append(SpinSystem.Coupling(
                site_i=label_to_idx[c["SpinI"]],
                site_j=label_to_idx[c["SpinJ"]],
                exchange_matrix=np.array(c["Exchange Matrix"]),
                displacement=np.array(c["Displacement"]),
            ))

        lattice_vecs, _bz_type = data["Lattice/BZ setting"]
        a1, a2 = lattice_vecs
        lattice_vectors = np.array([a1, a2])

        return cls(sites, couplings, lattice_vectors)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self):
        return (f"SpinSystem(num_sites={self.num_sites}, "
                f"num_couplings={self.num_couplings})")


# ======================================================================
# Backward-compatible aliases (to be removed in future versions)
# ======================================================================

SpinSite = SpinSystem.Site
Coupling = SpinSystem.Coupling
