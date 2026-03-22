"""Brillouin zone construction and grid generation.

This module provides tools for constructing first Brillouin zones (BZ) for
2D lattice systems, including hexagonal, tetragonal, and arbitrary lattices.
It supports both parallelogram and Wigner-Seitz cell constructions, high-symmetry
point identification, band path generation, and uniform grid sampling within
the BZ.

Merged from legacy modules:
- brillouin_zone_tools.py (BZ geometry utilities)
- find_first_bz.py (BZ construction for specific lattice types)
- brillouin_zone.py (main BrillouinZone wrapper class)
"""

import warnings
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from matplotlib.path import Path


# ---------------------------------------------------------------------------
# BZ utility functions (formerly BZ_tools class)
# ---------------------------------------------------------------------------

def reciprocal_vector(a1, a2):
    """Compute 2D reciprocal lattice vectors from real-space lattice vectors.

    Parameters
    ----------
    a1 : array_like
        First primitive lattice vector (2D).
    a2 : array_like
        Second primitive lattice vector (2D).

    Returns
    -------
    G1 : np.ndarray
        First reciprocal lattice vector (2D).
    G2 : np.ndarray
        Second reciprocal lattice vector (2D).
    """
    a1x, a1y = a1
    a2x, a2y = a2

    a1_3d = np.array([a1x, a1y, 0.0])
    a2_3d = np.array([a2x, a2y, 0.0])
    a3_3d = np.array([0.0, 0.0, 1.0])

    G1 = np.cross(a2_3d, a3_3d)
    G2 = np.cross(a3_3d, a1_3d)

    vol = np.dot(np.cross(a1_3d, a2_3d), a3_3d)
    G1 = G1 * (2 * np.pi / vol)
    G2 = G2 * (2 * np.pi / vol)

    return G1[:2], G2[:2]


def sorting_vectors_counter_clockwise(vectors, angle_origin=0):
    """Sort 2D vectors in counter-clockwise order by polar angle.

    Parameters
    ----------
    vectors : array_like
        List of 2D vectors to sort.
    angle_origin : float, optional
        Reference angle offset in radians (default: 0).

    Returns
    -------
    sorted_vectors : np.ndarray
        Vectors sorted by counter-clockwise angle.
    """
    angles = []
    for vector in vectors:
        vx, vy = vector
        angle = (np.arctan2(vy, vx) + 2 * np.pi - angle_origin) % (2 * np.pi)
        angles.append(angle)

    idx = np.argsort(angles)
    vectors = np.array(vectors)[idx]
    return vectors


def get_nearest_lattices(v1, v2):
    """Find nearest reciprocal lattice vectors for Wigner-Seitz construction.

    Parameters
    ----------
    v1 : np.ndarray
        First reciprocal lattice vector (2D).
    v2 : np.ndarray
        Second reciprocal lattice vector (2D).

    Returns
    -------
    sorted_lattices : np.ndarray
        Nearest lattice vectors sorted counter-clockwise.
    """
    dot_product = np.dot(v1, v2)
    nearest_lattices = [v1, v2, -v1, -v2]

    if np.isclose(dot_product, 0):
        return sorting_vectors_counter_clockwise(nearest_lattices)
    else:
        if np.sign(dot_product) == 1:
            nearest_lattices.append(v1 - v2)
            nearest_lattices.append(v2 - v1)
        elif np.sign(dot_product) == -1:
            nearest_lattices.append(v1 + v2)
            nearest_lattices.append(-(v1 + v2))

        return sorting_vectors_counter_clockwise(nearest_lattices)


def find_intersection_of_perp_plane(v1, v2):
    """Find intersection of perpendicular bisector planes.

    Solves the system: dot(G_i, k) = G_i^2 / 2 for i = 1, 2.

    Parameters
    ----------
    v1 : np.ndarray
        First vector (2D).
    v2 : np.ndarray
        Second vector (2D).

    Returns
    -------
    intersection : np.ndarray
        Intersection point (2D).
    """
    g1_sq = np.sum(v1**2)
    g2_sq = np.sum(v2**2)
    gmat = np.array([v1, v2])
    return np.array([g1_sq / 2, g2_sq / 2]) @ np.linalg.inv(gmat).T


def get_wigner_seitz_corners(G1, G2):
    """Calculate corners of the Wigner-Seitz Brillouin zone.

    Parameters
    ----------
    G1 : np.ndarray
        First reciprocal lattice vector (2D).
    G2 : np.ndarray
        Second reciprocal lattice vector (2D).

    Returns
    -------
    corners : np.ndarray
        BZ corner vertices sorted counter-clockwise.
    """
    sorted_bz_corners = get_nearest_lattices(G1, G2)
    corners = []

    for j in range(len(sorted_bz_corners)):
        g1 = sorted_bz_corners[j - 1]
        g2 = sorted_bz_corners[j]
        corner = tuple(find_intersection_of_perp_plane(g1, g2))
        corners.append(corner)

    corners = sorting_vectors_counter_clockwise(corners)
    return corners


# ---------------------------------------------------------------------------
# Private BZ helper classes (formerly get_BZ_hex, get_BZ_tetra, get_BZ_any)
# ---------------------------------------------------------------------------

class _HexBZ:
    """Hexagonal Brillouin zone construction.

    Parameters
    ----------
    a : float, optional
        Lattice constant (default: 1).
    phi : float, optional
        Rotation angle in radians (default: 0).
    """

    def __init__(self, a=None, phi=None):
        if a is None:
            a = 1

        self.a = a
        self.side_bz = 2 * np.pi / (3 * a) * 2
        length_G_vec = 2 * np.pi / (3 * a) * np.sqrt(3)

        if isinstance(phi, (float, int)) or phi is None:
            if phi is None:
                self.phi = 0
            else:
                self.phi = phi
        else:
            raise ValueError("phi must be a float")

        self.R1 = np.array([np.cos(np.pi / 3 - self.phi), -np.sin(np.pi / 3 - self.phi)])
        self.R2 = np.array([np.cos(np.pi / 3 + self.phi), +np.sin(np.pi / 3 + self.phi)])

        self.G1 = length_G_vec * np.array([np.cos(+np.pi / 6 - self.phi), +np.sin(+np.pi / 6 - self.phi)])
        self.G2 = length_G_vec * np.array([np.cos(-np.pi / 6 - self.phi), +np.sin(-np.pi / 6 - self.phi)])

    def _apply_rotation(self, point):
        """Apply rotation by phi to a 2D point.

        Parameters
        ----------
        point : tuple or np.ndarray
            Point (x, y) to rotate.

        Returns
        -------
        rotated : np.ndarray
            Rotated point.
        """
        x_temp, y_temp = point
        x = np.cos(self.phi) * x_temp - np.sin(self.phi) * y_temp
        y = np.sin(self.phi) * x_temp + np.cos(self.phi) * y_temp
        return np.array([x, y])

    def get_bz_data(self):
        """Compute hexagonal Brillouin zone data.

        Returns
        -------
        bz_data : dict
            Dictionary with keys: ``reciprocal_vectors``, ``BZ_corners``,
            ``high_symmetry_points``, ``band_paths``.
        """
        self.bz_vertices = [
            np.array([
                self.side_bz * np.cos(i * np.pi / 3 - self.phi),
                self.side_bz * np.sin(i * np.pi / 3 - self.phi),
            ])
            for i in range(6)
        ]

        Gamma = np.array([0.0, 0.0])
        K0 = self.bz_vertices[0]
        K1 = self.bz_vertices[1]
        K2 = self.bz_vertices[2]

        HSP = {
            'Γ': Gamma,
            'K': K0,
            'K\'': K1,
            '-K': -K0,
            'M': (K0 + K1) / 2,
            'M\'': (K1 + K2) / 2,
            '-M': -(K0 + K1) / 2,
        }

        band_path = {
            "standard path": ['K', 'Γ', 'M', 'K'],
            "rotated path": ['K\'', 'Γ', 'M\'', 'K\''],
            "inverse path": ['-K', 'Γ', '-M', '-K'],
        }

        return {
            "reciprocal_vectors": [self.G1, self.G2],
            "BZ_corners": self.bz_vertices,
            "high_symmetry_points": HSP,
            "band_paths": band_path,
        }

    def get_fbz_grid(self, N, center=(0, 0), print_idx=True):
        """Generate grid points inside the hexagonal first Brillouin zone.

        Parameters
        ----------
        N : int
            Number of grid points along each dimension.
        center : tuple, optional
            Center of the grid (default: (0, 0)).
        print_idx : bool, optional
            Whether to return grid indices (default: True).

        Returns
        -------
        grid_points : np.ndarray
            Points inside the BZ.
        area : float
            Area element per grid point.
        grid_indices : np.ndarray or None
            Grid indices if *print_idx* is True, else None.
        """
        # Ensure BZ vertices are computed
        if not hasattr(self, 'bz_vertices'):
            self.get_bz_data()

        x0, y0 = center

        dx = 4 * np.pi / (3 * self.a * N)
        dy = 4 * np.pi / (3 * self.a * N) / (2 * np.sqrt(3))

        length_dk = np.minimum(dx, dy)
        area = dx * dy

        Nx, Ny = N, int(np.ceil(N * (2 * np.sqrt(3)))) + 1

        points = []

        if print_idx:
            indices = []

        for i in range(-Ny - 1, Ny + 1):
            # Offset every other row by half a step
            offset = (i % 2) * dx / 2

            for j in range(-Nx, Nx + 1):
                x_temp = x0 + j * dx + offset
                y_temp = y0 + i * dy

                # Apply rotation
                point = self._apply_rotation((x_temp, y_temp))
                points.append(point)

                if print_idx:
                    indices.append((i, j))

        grid_points = np.array(points)

        if print_idx:
            grid_indices = np.array(indices)

        polygon_path = Path(self.bz_vertices)
        inside_mask = polygon_path.contains_points(grid_points, radius=dx * 0.01)

        if print_idx:
            return grid_points[inside_mask], area, grid_indices[inside_mask]
        else:
            return grid_points[inside_mask], area, None


class _TetraBZ:
    """Tetragonal Brillouin zone construction.

    Parameters
    ----------
    a : float, optional
        Lattice constant along x (default: 1).
    b : float, optional
        Lattice constant along y (default: sqrt(3)).
    """

    def __init__(self, a=None, b=None):
        if a is None or b is None:
            a, b = 1, np.sqrt(3)

        self.a = a
        self.b = b

        self.R1 = np.array([a, 0])
        self.R2 = np.array([0, b])

        self.G1 = np.array([2 * np.pi / a, 0])
        self.G2 = np.array([0, 2 * np.pi / b])

        self.Gamma = np.array([0.0, 0.0])
        self.phi = 0

    def _apply_rotation(self, point):
        """Apply rotation by phi to a 2D point.

        Parameters
        ----------
        point : tuple or np.ndarray
            Point (x, y) to rotate.

        Returns
        -------
        rotated : np.ndarray
            Rotated point.
        """
        x_temp, y_temp = point
        x = np.cos(self.phi) * x_temp - np.sin(self.phi) * y_temp
        y = np.sin(self.phi) * x_temp + np.cos(self.phi) * y_temp
        return np.array([x, y])

    def get_bz_data(self):
        """Compute tetragonal Brillouin zone data.

        Returns
        -------
        bz_data : dict
            Dictionary with keys: ``reciprocal_vectors``, ``BZ_corners``,
            ``high_symmetry_points``, ``band_paths``.
        """
        BZ = [
            (+self.G1[0] / 2, +self.G2[1] / 2),
            (-self.G1[0] / 2, +self.G2[1] / 2),
            (-self.G1[0] / 2, -self.G2[1] / 2),
            (+self.G1[0] / 2, -self.G2[1] / 2),
        ]

        HSP = {
            'Γ': self.Gamma,
            'X': self.G1 / 2,
            '-X': -self.G1 / 2,
            'Y': self.G2 / 2,
            'M': (self.G1 + self.G2) / 2,
            'M\'': (self.G2 - self.G1) / 2,
            '-M': -(self.G1 + self.G2) / 2,
        }

        band_path = {
            "standard path": ['X', 'Γ', 'M', 'X'],
            "rotated path": ['Y', 'Γ', 'M\'', 'Y'],
            "inverse path": ['-X', 'Γ', '-M', '-X'],
        }

        return {
            "reciprocal_vectors": [self.G1, self.G2],
            "BZ_corners": BZ,
            "high_symmetry_points": HSP,
            "band_paths": band_path,
        }

    def get_fbz_grid(self, N, center=(0, 0), print_idx=True):
        """Generate grid points inside the tetragonal first Brillouin zone.

        Parameters
        ----------
        N : int
            Number of grid points along each dimension.
        center : tuple or str, optional
            Center of the grid. Use ``"shift"`` for half-step offset
            (default: (0, 0)).
        print_idx : bool, optional
            Whether to return grid indices (default: True).

        Returns
        -------
        grid_points : np.ndarray
            Points inside the BZ.
        area : float
            Area element per grid point.
        grid_indices : np.ndarray or None
            Grid indices if *print_idx* is True, else None.
        """
        dG1, dG2 = self.G1 / (2 * N), self.G2 / (2 * N)

        if center == "shift":
            x0, y0 = -(dG1 + dG2) / 2
        elif center == (0, 0):
            x0, y0 = 0, 0
        else:
            x0, y0 = center

        area = np.abs(dG1[0] * dG2[1] - dG1[1] * dG2[0])

        points = []

        if print_idx:
            indices = []

        for i in range(-N, N):
            for j in range(-N, N):
                x = i * dG1[0] + j * dG2[0] + x0
                y = i * dG1[1] + j * dG2[1] + y0
                point = np.array([x, y])
                points.append(point)

                if print_idx:
                    indices.append((i, j))

        grid_points = np.array(points)

        if print_idx:
            grid_indices = np.array(indices)

        # Create polygon path for BZ boundary and filter points
        polygon_vertices = [
            (+self.G1[0] / 2, +self.G2[1] / 2),
            (-self.G1[0] / 2, +self.G2[1] / 2),
            (-self.G1[0] / 2, -self.G2[1] / 2),
            (+self.G1[0] / 2, -self.G2[1] / 2),
        ]
        polygon_path = Path(polygon_vertices)
        inside_mask = polygon_path.contains_points(grid_points)

        if print_idx:
            return grid_points[inside_mask], area, grid_indices[inside_mask]
        else:
            return grid_points[inside_mask], area, None


class _AnyBZ:
    """Arbitrary lattice Brillouin zone construction.

    Supports both parallelogram and Wigner-Seitz cell BZ representations.

    Parameters
    ----------
    r1 : array_like
        First real-space lattice vector (2D).
    r2 : array_like
        Second real-space lattice vector (2D).
    """

    def __init__(self, r1, r2):
        self.Gamma = np.array([0.0, 0.0])

        if isinstance(r1, (list, tuple)):
            self.R1 = np.array(r1)
        else:
            self.R1 = r1

        if isinstance(r2, (list, tuple)):
            self.R2 = np.array(r2)
        else:
            self.R2 = r2

        self.G1, self.G2 = reciprocal_vector(self.R1, self.R2)

    def get_bz_parallelogram(self, G_vector=None):
        """Generate the parallelogram first Brillouin zone.

        Parameters
        ----------
        G_vector : tuple of np.ndarray, optional
            Custom reciprocal vectors (G1, G2). Uses instance vectors if None.

        Returns
        -------
        bz_data : dict
            Dictionary with keys: ``reciprocal_vectors``, ``BZ_corners``,
            ``high_symmetry_points``, ``band_paths``.
        """
        if G_vector is None:
            G1, G2 = self.G1, self.G2
        else:
            G1, G2 = G_vector

        X = (G1 + G2) / 2
        Y = (G1 - G2) / 2

        BZ = [X, Y, -X, -Y]

        Gamma = np.array([0.0, 0.0])
        M = +G1 / 2
        N = -G2 / 2

        HSP = {
            'Γ': Gamma,
            'X': X, '-X': -X,
            'Y': Y, '-Y': -Y,
            'M': M, '-M': -M,
            'N': N, '-N': -N,
        }

        band_path = {
            "standard path": ['X', 'Γ', 'M', 'X'],
            "rotated path": ['Y', 'Γ', 'N', 'Y'],
            "inverse path": ['-X', 'Γ', '-M', '-X'],
        }

        return {
            "reciprocal_vectors": [G1, G2],
            "BZ_corners": BZ,
            "high_symmetry_points": HSP,
            "band_paths": band_path,
        }

    def get_bz_wigner_seitz_cell(self, G_vector=None):
        """Generate Wigner-Seitz cell for the first Brillouin zone.

        Parameters
        ----------
        G_vector : tuple of np.ndarray, optional
            Custom reciprocal vectors (G1, G2). Uses instance vectors if None.

        Returns
        -------
        bz_data : dict
            Dictionary with keys: ``reciprocal_vectors``, ``BZ_corners``,
            ``high_symmetry_points``, ``band_paths``.
        """
        if G_vector is None:
            G1, G2 = self.G1, self.G2
        else:
            G1, G2 = G_vector

        fbz_corners = get_wigner_seitz_corners(G1, G2)

        Gamma = np.array([0.0, 0.0])

        K_0 = np.array(fbz_corners[0])
        M_0 = (np.array(fbz_corners[0]) + np.array(fbz_corners[1])) / 2

        K_r = np.array(fbz_corners[1])
        M_r = (np.array(fbz_corners[1]) + np.array(fbz_corners[2])) / 2

        HSP = {
            'Γ': Gamma,
            'K': K_0,
            'K\'': K_r,
            '-K': -K_0,
            'M': M_0,
            'M\'': M_r,
            '-M': -M_0,
        }

        band_path = {
            "standard path": ['K', 'Γ', 'M', 'K'],
            "rotated path": ['K\'', 'Γ', 'M\'', 'K\''],
            "inverse path": ['-K', 'Γ', '-M', '-K'],
        }

        return {
            "reciprocal_vectors": [G1, G2],
            "BZ_corners": fbz_corners,
            "high_symmetry_points": HSP,
            "band_paths": band_path,
        }

    def get_fbz_grid(self, N, center=(0, 0), print_idx=True):
        """Generate grid points inside the first Brillouin zone.

        Parameters
        ----------
        N : int
            Number of grid points along each dimension.
        center : tuple or str, optional
            Center of the grid. Use ``"shift"`` for half-step offset
            (default: (0, 0)).
        print_idx : bool, optional
            Whether to return grid indices (default: True).

        Returns
        -------
        grid_points : np.ndarray
            Points inside the BZ.
        area : float
            Area element per grid point.
        grid_indices : np.ndarray or None
            Grid indices if *print_idx* is True, else None.
        """
        dG1, dG2 = self.G1 / (2 * N), self.G2 / (2 * N)

        if center == "shift":
            x0, y0 = -(dG1 + dG2) / 2
        elif center == (0, 0):
            x0, y0 = 0, 0
        else:
            x0, y0 = center

        area = np.abs(dG1[0] * dG2[1] - dG1[1] * dG2[0])

        points = []

        if print_idx:
            indices = []

        for i in range(-N, N):
            for j in range(-N, N):
                x = i * dG1[0] + j * dG2[0] + x0
                y = i * dG1[1] + j * dG2[1] + y0
                point = np.array([x, y])
                points.append(point)

                if print_idx:
                    indices.append((i, j))

        grid_points = np.array(points)

        if print_idx:
            grid_indices = np.array(indices)

        if print_idx:
            return grid_points, area, grid_indices
        else:
            return grid_points, area, None


# ---------------------------------------------------------------------------
# Main BrillouinZone class (formerly Brillouin_Zone)
# ---------------------------------------------------------------------------

class BrillouinZone:
    """Brillouin zone wrapper for various lattice types.

    This class dispatches to the appropriate BZ construction helper based on
    the specified BZ type and provides a unified interface for obtaining BZ
    data, k-point paths, and grid sampling.

    Parameters
    ----------
    lattice_bz_setting : tuple
        Tuple of ``(lattice_vectors, bz_type)`` where *lattice_vectors* is a
        pair of 2D arrays and *bz_type* is one of ``"simple"``, ``"Hex_60"``,
        ``"Hex_30"``, ``"Tetra"``, ``"wigner_seitz"``.
    bz_type : str, optional
        Override BZ type (default: None).

    Attributes
    ----------
    bz_data : dict
        Dictionary containing reciprocal vectors, BZ corners, high-symmetry
        points, and band paths.
    get_bz_grid : callable
        Grid generation function bound to the underlying BZ helper.
    """

    def __init__(self, lattice_bz_setting, bz_type=None):
        self.lattice_bz_setting = self._get_bz_setting(lattice_bz_setting, bz_type)
        self.bz_data, self.get_bz_grid = self._get_bz_data(self.lattice_bz_setting)

    def _get_bz_setting(self, bz_setting, bz_type=None):
        """Resolve BZ setting from inputs.

        Parameters
        ----------
        bz_setting : tuple
            Raw BZ setting from user.
        bz_type : str, optional
            Override BZ type.

        Returns
        -------
        resolved_setting : tuple
            ``(lattice_vectors, bz_type)`` tuple.
        """
        if bz_type is None:
            bz_setting = bz_setting

        elif bz_type == "simple":
            bz_setting = (bz_setting[0], "simple")

        elif bz_type == "Hex_60":
            bz_setting = (
                (np.array([1 / 2, np.sqrt(3) / 2]),
                 np.array([1 / 2, -np.sqrt(3) / 2])),
                "Hex_60",
            )
        else:
            raise ValueError(f"Unknown BZ type: {bz_type}")

        return bz_setting

    def _get_bz_data(self, lattice_bz_setting=None):
        """Dispatch to the appropriate BZ helper and retrieve BZ data.

        Parameters
        ----------
        lattice_bz_setting : tuple, optional
            ``(lattice_vectors, bz_type)`` tuple.

        Returns
        -------
        bz_data : dict
            BZ information dictionary.
        get_bz_grid : callable
            Grid generation function.
        """
        if lattice_bz_setting is None:
            lattice_vectors, bz_type = self.lattice_bz_setting
        elif isinstance(lattice_bz_setting, tuple):
            lattice_vectors, bz_type = lattice_bz_setting
        else:
            raise ValueError("Lattice vectors must be a tuple")

        rvec1, rvec2 = lattice_vectors

        # Calculate the length of the lattice vectors
        a = np.sum(rvec1**2) ** 0.5
        b = np.sum(rvec2**2) ** 0.5

        if bz_type == "simple":
            bz = _AnyBZ(rvec1, rvec2)
            bz_data = bz.get_bz_parallelogram()
            return bz_data, bz.get_fbz_grid

        elif bz_type == "Hex_60":
            bz_hex = _HexBZ(a, phi=0)
            bz_data = bz_hex.get_bz_data()
            return bz_data, bz_hex.get_fbz_grid

        elif bz_type == "Hex_30":
            bz_hex = _HexBZ(a, phi=np.pi / 6)
            bz_data = bz_hex.get_bz_data()
            return bz_data, bz_hex.get_fbz_grid

        elif bz_type == "Tetra":
            bz_tetra = _TetraBZ(a, b)
            bz_data = bz_tetra.get_bz_data()
            return bz_data, bz_tetra.get_fbz_grid

        elif bz_type == "wigner_seitz":
            bz_any = _AnyBZ(rvec1, rvec2)
            bz_data = bz_any.get_bz_wigner_seitz_cell()
            return bz_data, bz_any.get_fbz_grid

        else:
            raise ValueError(f"Unknown BZ type: {bz_type}")

    def generate_kpath(self, band_path, HSP, mag=100):
        """Generate a k-point path along high-symmetry points.

        Parameters
        ----------
        band_path : list of str
            Ordered list of high-symmetry point labels defining the path.
        HSP : dict
            Dictionary mapping labels to 2D coordinates.
        mag : int, optional
            Magnification factor controlling the number of points per segment
            (default: 100).

        Returns
        -------
        path_kpoints : np.ndarray
            Array of k-points along the path.
        path_intervals : list of int
            Number of k-points in each segment.
        """
        path_kpoints = []
        path_intervals = []

        for i in range(len(band_path) - 1):
            start_point = np.array(HSP[band_path[i]])
            end_point = np.array(HSP[band_path[i + 1]])
            norm = np.linalg.norm(end_point - start_point)
            num = max(2, int(np.round(norm * mag)))

            segment = np.linspace(start_point, end_point, num, endpoint=False)
            path_kpoints.extend(segment)
            path_intervals.append(num)

        # Append the last point
        last_point = np.array(HSP[band_path[-1]])
        path_kpoints.append(last_point)
        path_intervals[-1] += 1

        path_kpoints = np.array(path_kpoints)

        return path_kpoints, path_intervals

    def get_full(self, N, center=(0, 0), buffer=0.1, print_idx=False):
        """Obtain BZ data together with a grid of k-points.

        Parameters
        ----------
        N : int
            Number of grid points along each dimension.
        center : tuple, optional
            Grid center (default: (0, 0)).
        buffer : float, optional
            Boundary buffer (default: 0.1).
        print_idx : bool, optional
            Whether to return grid indices (default: False).

        Returns
        -------
        bz_data : dict
            BZ information dictionary (includes ``"area"`` key).
        grid_points : np.ndarray
            k-points inside the BZ.
        grid_indices : np.ndarray or None
            Grid indices if *print_idx* is True, else None.
        """
        bz_data, get_bz_grid = self._get_bz_data()

        grid_points, area, grid_indices = get_bz_grid(N, center=center, print_idx=print_idx)

        bz_data["area"] = area

        return bz_data, grid_points, grid_indices
