"""
Unit tests for TriangularLattice.

Tests geometric properties:
- Primitive vectors
- Reciprocal vectors
- Neighbor distances and displacements
- High-symmetry points
"""

import numpy as np
import pytest
from lswt.core.lattice import TriangularLattice


class TestTriangularLatticeGeometry:
    """Test basic geometric properties."""

    def test_primitive_vectors(self):
        """Test primitive lattice vectors."""
        lattice = TriangularLattice(lattice_constant=1.0)
        a1, a2 = lattice.get_primitive_vectors()

        # Check a1 = [1, 0]
        assert np.allclose(a1, [1.0, 0.0])

        # Check a2 = [0.5, √3/2]
        expected_a2 = [0.5, np.sqrt(3)/2]
        assert np.allclose(a2, expected_a2)

    def test_lattice_constant_scaling(self):
        """Test that vectors scale with lattice constant."""
        lattice = TriangularLattice(lattice_constant=2.0)
        a1, a2 = lattice.get_primitive_vectors()

        assert np.allclose(a1, [2.0, 0.0])
        assert np.allclose(a2, [1.0, np.sqrt(3)])

    def test_basis_positions(self):
        """Test basis positions (single atom at origin)."""
        lattice = TriangularLattice()
        basis = lattice.get_basis_positions()

        assert len(basis) == 1
        assert np.allclose(basis[0], [0.0, 0.0])

    def test_unit_cell_area(self):
        """Test unit cell area calculation."""
        lattice = TriangularLattice(lattice_constant=1.0)
        area = lattice.get_unit_cell_area()

        # Area = |a1 × a2| = 1 * (√3/2) = √3/2
        expected_area = np.sqrt(3) / 2
        assert np.isclose(area, expected_area)


class TestTriangularLatticeNeighbors:
    """Test neighbor calculations."""

    def test_nearest_neighbors_count(self):
        """Test that we get 3 nearest neighbor bonds."""
        lattice = TriangularLattice(lattice_constant=1.0)
        nn = lattice.get_neighbors(order=1)

        # 3 bonds in forward direction (total 6 including reverse)
        assert len(nn) == 3

    def test_nearest_neighbors_distance(self):
        """Test NN distances are all equal to a."""
        lattice = TriangularLattice(lattice_constant=1.5)
        nn = lattice.get_neighbors(order=1)

        for neighbor in nn:
            assert np.isclose(neighbor.distance, 1.5)

    def test_nearest_neighbors_displacements(self):
        """Test NN displacement vectors."""
        lattice = TriangularLattice(lattice_constant=1.0)
        nn = lattice.get_neighbors(order=1)

        # Extract displacements
        displacements = [n.displacement for n in nn]

        # Expected displacements
        expected = [
            np.array([1.0, 0.0]),
            np.array([-0.5, np.sqrt(3)/2]),
            np.array([-0.5, -np.sqrt(3)/2])
        ]

        # Check each displacement is present (order may vary)
        for exp_disp in expected:
            found = False
            for disp in displacements:
                if np.allclose(disp, exp_disp):
                    found = True
                    break
            assert found, f"Displacement {exp_disp} not found"

    def test_next_nearest_neighbors_count(self):
        """Test that we get 6 NNN bonds."""
        lattice = TriangularLattice(lattice_constant=1.0)
        nnn = lattice.get_neighbors(order=2)

        assert len(nnn) == 6

    def test_next_nearest_neighbors_distance(self):
        """Test NNN distances are all equal to a√3."""
        lattice = TriangularLattice(lattice_constant=1.0)
        nnn = lattice.get_neighbors(order=2)

        expected_distance = np.sqrt(3)
        for neighbor in nnn:
            assert np.isclose(neighbor.distance, expected_distance)

    def test_third_neighbors_count(self):
        """Test that we get 3 third-neighbor bonds."""
        lattice = TriangularLattice(lattice_constant=1.0)
        third = lattice.get_neighbors(order=3)

        assert len(third) == 3

    def test_third_neighbors_distance(self):
        """Test third-neighbor distances are 2a."""
        lattice = TriangularLattice(lattice_constant=1.0)
        third = lattice.get_neighbors(order=3)

        for neighbor in third:
            assert np.isclose(neighbor.distance, 2.0)


class TestTriangularLatticeReciprocal:
    """Test reciprocal lattice properties."""

    def test_reciprocal_vectors(self):
        """Test reciprocal lattice vectors satisfy a·b = 2π δ."""
        lattice = TriangularLattice(lattice_constant=1.0)
        a1, a2 = lattice.get_primitive_vectors()
        b1, b2 = lattice.get_reciprocal_vectors()

        # Check orthogonality relations
        assert np.isclose(np.dot(a1, b1), 2 * np.pi)
        assert np.isclose(np.dot(a2, b2), 2 * np.pi)
        assert np.isclose(np.dot(a1, b2), 0.0)
        assert np.isclose(np.dot(a2, b1), 0.0)

    def test_high_symmetry_points(self):
        """Test high-symmetry points in BZ."""
        lattice = TriangularLattice(lattice_constant=1.0)
        hsp = lattice.get_high_symmetry_points()

        # Check we have Γ, K, M
        assert 'Γ' in hsp
        assert 'K' in hsp
        assert 'M' in hsp

        # Γ is at origin
        assert np.allclose(hsp['Γ'], [0, 0])

        # K and M should be non-zero
        assert np.linalg.norm(hsp['K']) > 0
        assert np.linalg.norm(hsp['M']) > 0


class TestTriangularLatticeCoordinateConversion:
    """Test real-fractional coordinate conversions."""

    def test_fractional_to_real(self):
        """Test conversion from fractional to real coordinates."""
        lattice = TriangularLattice(lattice_constant=1.0)

        # Test (1, 0) → a1
        real = lattice.fractional_to_real(np.array([1, 0]))
        assert np.allclose(real, [1.0, 0.0])

        # Test (0, 1) → a2
        real = lattice.fractional_to_real(np.array([0, 1]))
        assert np.allclose(real, [0.5, np.sqrt(3)/2])

        # Test (1, 1) → a1 + a2
        real = lattice.fractional_to_real(np.array([1, 1]))
        assert np.allclose(real, [1.5, np.sqrt(3)/2])

    def test_real_to_fractional(self):
        """Test conversion from real to fractional coordinates."""
        lattice = TriangularLattice(lattice_constant=1.0)

        # Test a1 → (1, 0)
        frac = lattice.real_to_fractional(np.array([1.0, 0.0]))
        assert np.allclose(frac, [1, 0])

        # Test a2 → (0, 1)
        frac = lattice.real_to_fractional(np.array([0.5, np.sqrt(3)/2]))
        assert np.allclose(frac, [0, 1])

    def test_roundtrip_conversion(self):
        """Test that real → fractional → real is identity."""
        lattice = TriangularLattice(lattice_constant=1.0)

        test_positions = [
            [1.0, 0.5],
            [2.3, 1.7],
            [-1.5, 2.0]
        ]

        for pos in test_positions:
            pos = np.array(pos)
            frac = lattice.real_to_fractional(pos)
            pos_recovered = lattice.fractional_to_real(frac)
            assert np.allclose(pos, pos_recovered)


class TestTriangularLatticeValidation:
    """Test input validation."""

    def test_negative_lattice_constant_raises(self):
        """Test that negative lattice constant raises error."""
        with pytest.raises(ValueError, match="Lattice constant must be positive"):
            TriangularLattice(lattice_constant=-1.0)

    def test_zero_lattice_constant_raises(self):
        """Test that zero lattice constant raises error."""
        with pytest.raises(ValueError, match="Lattice constant must be positive"):
            TriangularLattice(lattice_constant=0.0)

    def test_invalid_neighbor_order_raises(self):
        """Test that invalid neighbor order raises error."""
        lattice = TriangularLattice()

        with pytest.raises(ValueError, match="not implemented"):
            lattice.get_neighbors(order=10)


class TestTriangularLatticeRepr:
    """Test string representations."""

    def test_repr(self):
        """Test __repr__ output."""
        lattice = TriangularLattice(lattice_constant=1.5)
        repr_str = repr(lattice)

        assert "TriangularLattice" in repr_str
        assert "1.500" in repr_str  # Lattice constant
        assert "basis_atoms=1" in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
