"""
Unit tests for CommensurateStructure.

Tests magnetic structure properties:
- Spin directions
- Sublattice indexing
- Optimization parameters
- Serialization
"""

import numpy as np
import pytest
from spintoolkit.states import CommensurateStructure

# Three coplanar directions 120 degrees apart (theta, phi).
ANGLES_120 = np.array([
    [np.pi/2, 0.0],
    [np.pi/2, 2*np.pi/3],
    [np.pi/2, 4*np.pi/3]
])

# Nearest-neighbour cell offsets of the triangular lattice with
# a1 = (1, 0), a2 = (1/2, sqrt(3)/2).
TRIANGULAR_NN_OFFSETS = [(1, 0), (0, 1), (-1, 1)]


def structure_120_basis():
    """120-degree order with the sqrt3 x sqrt3 cell taken as the crystal cell.

    Three basis sites, one magnetic sublattice each; the class does not see
    the geometry, so the three sites carry the three directions directly.
    """
    return CommensurateStructure(
        num_basis_sites=3,
        magnetic_supercell=(1, 1),
        angles=ANGLES_120
    )


def structure_120_supercell():
    """120-degree order on a one-site triangular lattice in a 3x3 diagonal cell.

    The diagonal (n1, n2) supercell cannot hold the sqrt3 x sqrt3 cell, so the
    3x3 cell repeats it three times: nine sublattices, phi = 2 pi (n1 - n2) / 3.
    Sublattice order follows the class: index = n1 + 3 * n2.
    """
    angles = np.array([ANGLES_120[(n1 - n2) % 3]
                       for n2 in range(3) for n1 in range(3)])
    return CommensurateStructure(
        num_basis_sites=1,
        magnetic_supercell=(3, 3),
        angles=angles
    )


STRUCTURES_120 = pytest.mark.parametrize(
    "make_structure", [structure_120_basis, structure_120_supercell],
    ids=["3-site-basis", "3x3-supercell"])


def triangular_nn_dot_products(structure):
    """Set of s_i . s_j over nearest-neighbour bonds of a one-site triangular lattice."""
    dots = set()
    for n1 in range(6):
        for n2 in range(6):
            s = structure.get_spin_direction(0, (n1, n2))
            for d1, d2 in TRIANGULAR_NN_OFFSETS:
                t = structure.get_spin_direction(0, (n1 + d1, n2 + d2))
                dots.add(round(float(s @ t), 12))
    return dots


class TestCommensurateStructureCreation:
    """Test creation and initialization."""

    def test_ferromagnetic_structure(self):
        """Test simple ferromagnetic structure (all spins up)."""
        structure = CommensurateStructure(
            num_basis_sites=1,
            magnetic_supercell=(1, 1),
            angles=np.array([[0.0, 0.0]])  # θ=0 → z-direction
        )

        assert structure.num_magnetic_sublattices == 1
        assert structure.is_commensurate()

    def test_120_degree_structure_basis(self):
        """Test 120° structure with a 3-site basis (sqrt3 x sqrt3 crystal cell)."""
        structure = structure_120_basis()

        assert structure.num_magnetic_sublattices == 3

    def test_120_degree_structure_supercell(self):
        """Test 120° structure on a one-site lattice in a 3x3 supercell."""
        structure = structure_120_supercell()

        assert structure.num_magnetic_sublattices == 9

    def test_two_sublattice_afm(self):
        """Test 2-sublattice antiferromagnet."""
        angles = np.array([
            [0.0, 0.0],      # Spin up
            [np.pi, 0.0]     # Spin down
        ])

        structure = CommensurateStructure(
            num_basis_sites=1,
            magnetic_supercell=(2, 1),  # 2x1 supercell
            angles=angles
        )

        assert structure.num_magnetic_sublattices == 2


class TestCommensurateStructureSpinDirections:
    """Test spin direction calculations."""

    def test_ferromagnetic_spin_directions(self):
        """Test that FM structure gives same spin everywhere."""
        structure = CommensurateStructure(
            num_basis_sites=1,
            magnetic_supercell=(1, 1),
            angles=np.array([[0.0, 0.0]])  # θ=0 → ẑ
        )

        # All sites should have same spin
        spin1 = structure.get_spin_direction(0, (0, 0))
        spin2 = structure.get_spin_direction(0, (1, 0))
        spin3 = structure.get_spin_direction(0, (5, 7))

        expected = [0, 0, 1]  # z-direction
        assert np.allclose(spin1, expected)
        assert np.allclose(spin2, expected)
        assert np.allclose(spin3, expected)

    @STRUCTURES_120
    def test_120_degree_spins_in_xy_plane(self, make_structure):
        """Test 120° structure spins are in xy-plane."""
        structure = make_structure()

        spins = structure.get_all_spin_directions()

        # All should have z-component = 0 (in xy-plane)
        assert np.allclose(spins[:, 2], 0.0)

        # All should be unit vectors
        for spin in spins:
            assert np.isclose(np.linalg.norm(spin), 1.0)

    @STRUCTURES_120
    def test_120_degree_total_magnetization_zero(self, make_structure):
        """Test that 120° structure has zero total magnetization."""
        structure = make_structure()

        M = structure.get_total_magnetization()
        assert np.allclose(M, [0, 0, 0], atol=1e-10)

    def test_120_degree_supercell_nearest_neighbours(self):
        """Test every nearest-neighbour pair of the 3x3 cell is at 120°."""
        dots = triangular_nn_dot_products(structure_120_supercell())

        assert dots == {-0.5}

    def test_3x1_supercell_is_not_120_degree(self):
        """Test a 3x1 cell cannot hold 120° order on the triangular lattice.

        The cell index wraps n2 mod 1, so spins along a2 are parallel.
        """
        structure = CommensurateStructure(
            num_basis_sites=1,
            magnetic_supercell=(3, 1),
            angles=ANGLES_120
        )

        dots = triangular_nn_dot_products(structure)

        assert 1.0 in dots

    def test_spin_directions_are_unit_vectors(self):
        """Test that all spin directions are normalized."""
        # Random angles
        np.random.seed(42)
        angles = np.random.rand(5, 2)
        angles[:, 0] *= np.pi  # theta: 0 to π
        angles[:, 1] *= 2 * np.pi  # phi: 0 to 2π

        structure = CommensurateStructure(
            num_basis_sites=1,
            magnetic_supercell=(5, 1),
            angles=angles
        )

        spins = structure.get_all_spin_directions()
        for spin in spins:
            assert np.isclose(np.linalg.norm(spin), 1.0)


class TestCommensurateStructureOptimization:
    """Test optimization parameter interface."""

    def test_get_optimization_parameters(self):
        """Test getting angles as optimization parameters."""
        angles = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ])

        structure = CommensurateStructure(
            num_basis_sites=1,
            magnetic_supercell=(3, 1),
            angles=angles
        )

        params = structure.get_optimization_parameters()

        # Should be flattened
        expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        assert np.allclose(params, expected)

    def test_set_optimization_parameters(self):
        """Test setting angles from optimization."""
        structure = CommensurateStructure(
            num_basis_sites=1,
            magnetic_supercell=(2, 1),
            angles=np.array([[0.0, 0.0], [0.0, 0.0]])
        )

        # Set new angles
        new_params = np.array([np.pi/2, 0.0, np.pi/2, np.pi])
        structure.set_optimization_parameters(new_params)

        # Check they were set correctly
        assert np.allclose(structure.angles[0], [np.pi/2, 0.0])
        assert np.allclose(structure.angles[1], [np.pi/2, np.pi])

    def test_optimization_roundtrip(self):
        """Test get → set → get returns same values."""
        angles = np.array([[1.0, 2.0], [3.0, 4.0]])

        structure = CommensurateStructure(
            num_basis_sites=1,
            magnetic_supercell=(2, 1),
            angles=angles
        )

        # Get params
        params1 = structure.get_optimization_parameters()

        # Modify structure
        structure.set_optimization_parameters(params1 * 1.5)

        # Set back to original
        structure.set_optimization_parameters(params1)

        # Get again
        params2 = structure.get_optimization_parameters()

        assert np.allclose(params1, params2)


class TestCommensurateStructureSerialization:
    """Test serialization to/from dict."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        angles = np.array([[0.0, 0.0], [np.pi, 0.0]])

        structure = CommensurateStructure(
            num_basis_sites=1,
            magnetic_supercell=(2, 1),
            angles=angles,
            spin_magnitude=0.5
        )

        data = structure.to_dict()

        # Check required fields
        assert data['type'] == 'commensurate'
        assert data['num_basis_sites'] == 1
        assert data['magnetic_supercell'] == [2, 1]
        assert data['spin_magnitude'] == 0.5
        assert np.allclose(data['angles'], angles.tolist())

    def test_from_dict(self):
        """Test reconstruction from dictionary."""
        data = {
            'type': 'commensurate',
            'num_basis_sites': 1,
            'magnetic_supercell': [3, 1],
            'angles': [[0.0, 0.0], [np.pi/2, 0.0], [np.pi, 0.0]],
            'spin_magnitude': 0.5
        }

        structure = CommensurateStructure.from_dict(data)

        assert structure.num_basis_sites == 1
        assert structure.magnetic_supercell == (3, 1)
        assert structure.num_magnetic_sublattices == 3
        assert np.allclose(structure.angles, data['angles'])

    def test_roundtrip_serialization(self):
        """Test to_dict → from_dict → to_dict preserves data."""
        angles = np.array([[1.2, 3.4], [5.6, 7.8]])

        structure1 = CommensurateStructure(
            num_basis_sites=1,
            magnetic_supercell=(2, 1),
            angles=angles,
            spin_magnitude=1.0
        )

        data = structure1.to_dict()
        structure2 = CommensurateStructure.from_dict(data)
        data2 = structure2.to_dict()

        assert data == data2


class TestCommensurateStructureValidation:
    """Test input validation."""

    def test_negative_basis_sites_raises(self):
        """Test that negative num_basis_sites raises error."""
        with pytest.raises(ValueError, match="num_basis_sites"):
            CommensurateStructure(
                num_basis_sites=0,
                magnetic_supercell=(1, 1),
                angles=np.array([[0, 0]])
            )

    def test_negative_supercell_raises(self):
        """Test that negative supercell dimensions raise error."""
        with pytest.raises(ValueError, match="supercell"):
            CommensurateStructure(
                num_basis_sites=1,
                magnetic_supercell=(0, 1),
                angles=np.array([[0, 0]])
            )

    def test_angles_shape_mismatch_raises(self):
        """Test that wrong number of angles raises error."""
        with pytest.raises(ValueError, match="shape mismatch"):
            # Supercell (2, 1) with 1 basis → 2 sublattices
            # But only provide 1 angle
            CommensurateStructure(
                num_basis_sites=1,
                magnetic_supercell=(2, 1),
                angles=np.array([[0, 0]])  # Should be 2 angles!
            )

    def test_angles_wrong_columns_raises(self):
        """Test that angles with wrong number of columns raises error."""
        with pytest.raises(ValueError, match="shape.*2"):
            CommensurateStructure(
                num_basis_sites=1,
                magnetic_supercell=(1, 1),
                angles=np.array([[0, 0, 0]])  # Should be (N, 2) not (N, 3)
            )

    def test_negative_spin_magnitude_raises(self):
        """Test that negative spin magnitude raises error."""
        with pytest.raises(ValueError, match="Spin magnitude"):
            CommensurateStructure(
                num_basis_sites=1,
                magnetic_supercell=(1, 1),
                angles=np.array([[0, 0]]),
                spin_magnitude=-0.5
            )


class TestCommensurateStructureRepr:
    """Test string representations."""

    def test_repr(self):
        """Test __repr__ output."""
        structure = CommensurateStructure(
            num_basis_sites=1,
            magnetic_supercell=(3, 1),
            angles=np.zeros((3, 2)),
            spin_magnitude=0.5
        )

        repr_str = repr(structure)

        assert "CommensurateStructure" in repr_str
        assert "supercell=(3, 1)" in repr_str
        assert "sublattices=3" in repr_str
        assert "S=0.5" in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
