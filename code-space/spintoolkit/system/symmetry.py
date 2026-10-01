"""Crystal symmetry of a layer: symmetry-allowed couplings and symmetric terms.

The allowed form of an exchange matrix or a g-tensor is fixed by the symmetry
of the *crystal*, including the non-magnetic ions (ligands) that mediate the
exchange, not by the magnetic sites alone. This module finds or closes the
symmetry group of a layer crystal and uses it to

- give the basis of exchange matrices allowed on a bond (Moriya rules, the
  four parameters ``J, Delta, J_pm_pm, J_z_pm`` of a triangular layer with
  ``D3d`` sites, ...) and of g-tensors allowed on a site;
- generate the full translation orbit of a coupling from one representative,
  rejecting a coefficient that breaks the declared symmetry;
- check an existing :class:`~spintoolkit.system.model.SpinModel` against the
  group.

Conventions
-----------
A layer crystal has the in-plane lattice ``A`` of the model (rows are the
primitive vectors, Cartesian x and y in the layer) and atoms at fractional
in-plane coordinates ``f`` and Cartesian heights ``z`` along the layer normal.
A symmetry operation acts on a point as

    r -> R r + t,

with a Cartesian orthogonal ``R`` that maps the layer onto itself
(``R = diag(R2, s)``, ``s = +1`` or ``-1``), an in-plane fractional translation
and a height shift. Spin, field and the magnetic moment are axial vectors, so
they transform with ``R_s = det(R) R``. A bilinear term ``S_i^T J S_j`` and a
Zeeman term ``-b^T g S_i`` are invariant when ``J -> R_s J R_s^T`` and
``g -> R_s g R_s^T`` accompany the site map; the factor ``det(R)`` cancels in
these quadratic forms but is kept so that the spin rotation is the physical
one. Time reversal leaves every supported term invariant and is not part of the
group.

Using only the magnetic sites as the crystal gives an upper bound of the true
symmetry: for example a flat lattice of magnetic ions has a mirror in the layer
plane that forbids in-plane DM vectors which ligands above and below the layer
allow. Pass the ligands whenever the allowed form of a coupling matters.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spintoolkit.system.model import BILINEAR, ZEEMAN, Site, SpinModel, Term

#: Tolerance of fractional positions and of orthogonality tests.
SYMMETRY_POSITION_TOLERANCE = 1e-6
#: Relative tolerance below which a coefficient counts as invariant.
SYMMETRY_COEFFICIENT_TOLERANCE = 1e-8
#: Largest group order accepted when closing user generators (layer groups have at most 48).
MAX_GROUP_ORDER = 48

Participant = Tuple[str, Tuple[int, int]]


class SymmetryError(ValueError):
    """The crystal, the operations or a coefficient are inconsistent with the symmetry."""


@dataclass(frozen=True)
class LayerCrystal:
    """Atoms of a layer crystal, magnetic and non-magnetic.

    Parameters
    ----------
    lattice : array_like, shape (2, 2)
        In-plane lattice ``A``; rows are the primitive vectors.
    positions : array_like, shape (n, 2)
        Fractional in-plane coordinates of the atoms.
    species : sequence of str
        Chemical species (or any label) of each atom; only equal species are
        mapped onto each other.
    heights : array_like, shape (n,), optional
        Cartesian heights along the layer normal, in the length unit of
        ``lattice`` (default zero).
    """

    lattice: np.ndarray
    positions: np.ndarray
    species: Tuple[str, ...]
    heights: Optional[np.ndarray] = None

    def __post_init__(self):
        lattice = np.array(self.lattice, dtype=float)
        positions = np.array(self.positions, dtype=float).reshape(-1, 2)
        species = tuple(str(s) for s in self.species)
        heights = (np.zeros(len(positions)) if self.heights is None
                   else np.array(self.heights, dtype=float).reshape(-1))
        if lattice.shape != (2, 2) or abs(np.linalg.det(lattice)) < 1e-12:
            raise SymmetryError("lattice must be two independent in-plane vectors")
        if not len(positions) == len(species) == len(heights) or not len(positions):
            raise SymmetryError("positions, species and heights must have the same "
                                "non-zero length")
        for name, value in (("lattice", lattice), ("positions", positions), ("heights", heights)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "species", species)

    @classmethod
    def from_model(cls, model: SpinModel, species: str = "M") -> "LayerCrystal":
        """Crystal of the magnetic sites only, all of one species at height zero.

        This gives an upper bound of the symmetry (see the module docstring).
        """
        return cls(model.lattice, [site.position for site in model.sites],
                   [species] * model.num_sites)


@dataclass(frozen=True)
class SymmetryOperation:
    """``r -> R r + t`` on a layer crystal.

    Attributes
    ----------
    rotation : (3, 3) array
        Cartesian orthogonal matrix ``R``; maps the layer onto itself.
    matrix : (2, 2) int array
        The same in-plane map on fractional row vectors, ``f -> f @ matrix``.
    translation : (2,) array
        Fractional in-plane translation in ``[0, 1)``.
    height_shift : float
        Cartesian translation along the normal.
    """

    rotation: np.ndarray
    matrix: np.ndarray
    translation: np.ndarray
    height_shift: float = 0.0

    @property
    def spin_rotation(self) -> np.ndarray:
        """``R_s = det(R) R``, the action on axial vectors (spin, field)."""
        return np.linalg.det(self.rotation) * self.rotation

    @property
    def normal_sign(self) -> int:
        """``+1`` if the operation keeps the layer normal, ``-1`` if it flips it."""
        return int(round(self.rotation[2, 2]))

    def apply(self, fractional: np.ndarray, height: float) -> Tuple[np.ndarray, float]:
        """Image ``(f @ matrix + t, s z + t_z)`` of a point, without reduction mod 1."""
        return (np.asarray(fractional, dtype=float) @ self.matrix + self.translation,
                self.normal_sign * height + self.height_shift)

    def _key(self) -> Tuple:
        t = np.round(np.mod(self.translation, 1.0), 6) % 1.0
        return (tuple(self.matrix.ravel()), self.normal_sign, tuple(t), round(self.height_shift, 6))


def _make_operation(lattice: np.ndarray, matrix: np.ndarray, sign: int,
                    translation, height_shift: float) -> SymmetryOperation:
    in_plane = (np.linalg.inv(lattice) @ matrix @ lattice).T
    rotation = np.zeros((3, 3))
    rotation[:2, :2] = in_plane
    rotation[2, 2] = sign
    t = np.mod(np.asarray(translation, dtype=float), 1.0)
    t[np.isclose(t, 1.0, rtol=0, atol=SYMMETRY_POSITION_TOLERANCE)] = 0.0
    for array in (rotation, matrix, t):
        array.setflags(write=False)
    return SymmetryOperation(rotation, matrix, t, float(height_shift))


def operation_from_rotation(lattice, rotation, translation=(0.0, 0.0),
                            height_shift: float = 0.0) -> SymmetryOperation:
    """Operation from a Cartesian rotation and a fractional translation.

    Parameters
    ----------
    lattice : array_like, shape (2, 2)
    rotation : array_like, shape (3, 3)
        Orthogonal; must map the layer and its lattice onto themselves.
    translation : array_like, shape (2,)
        Fractional in-plane translation.
    height_shift : float
        Translation along the normal.

    Raises
    ------
    SymmetryError
        If ``rotation`` is not orthogonal, mixes the normal with the plane, or
        does not map the lattice onto itself.
    """
    lattice = np.asarray(lattice, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3) or not np.allclose(rotation @ rotation.T, np.eye(3),
                                                   atol=SYMMETRY_POSITION_TOLERANCE):
        raise SymmetryError("rotation must be an orthogonal 3x3 matrix")
    if not (np.allclose(rotation[2, :2], 0, atol=SYMMETRY_POSITION_TOLERANCE)
            and np.allclose(rotation[:2, 2], 0, atol=SYMMETRY_POSITION_TOLERANCE)):
        raise SymmetryError("rotation must map the layer plane onto itself")
    matrix = lattice @ rotation[:2, :2].T @ np.linalg.inv(lattice)
    rounded = np.round(matrix)
    if not np.allclose(matrix, rounded, atol=SYMMETRY_POSITION_TOLERANCE):
        raise SymmetryError("rotation does not map the lattice onto itself")
    return _make_operation(lattice, rounded.astype(int), int(round(rotation[2, 2])),
                           translation, height_shift)


def _lattice_point_operations(lattice: np.ndarray) -> List[np.ndarray]:
    """Integer matrices that map the lattice onto itself by an orthogonal map."""
    found = []
    inverse = np.linalg.inv(lattice)
    for entries in itertools.product(range(-2, 3), repeat=4):
        matrix = np.array(entries).reshape(2, 2)
        if abs(round(np.linalg.det(matrix))) != 1:
            continue
        in_plane = inverse @ matrix @ lattice
        if np.allclose(in_plane @ in_plane.T, np.eye(2), atol=SYMMETRY_POSITION_TOLERANCE):
            found.append(matrix)
    return found


def _wrap(difference: np.ndarray) -> np.ndarray:
    return difference - np.round(difference)


def _maps_crystal(crystal: LayerCrystal, operation: SymmetryOperation, z_tolerance: float) -> bool:
    images, heights = operation.apply(crystal.positions, 0.0)
    heights = operation.normal_sign * crystal.heights + operation.height_shift
    used = np.zeros(len(crystal.positions), dtype=bool)
    for i, (image, height) in enumerate(zip(images, heights)):
        match = [j for j in range(len(crystal.positions))
                 if crystal.species[j] == crystal.species[i] and not used[j]
                 and np.all(np.abs(_wrap(image - crystal.positions[j])) < SYMMETRY_POSITION_TOLERANCE)
                 and abs(height - crystal.heights[j]) < z_tolerance]
        if not match:
            return False
        used[match[0]] = True
    return True


def find_symmetry(crystal: LayerCrystal) -> Tuple[SymmetryOperation, ...]:
    """All symmetry operations of a layer crystal, modulo lattice translations.

    Parameters
    ----------
    crystal : LayerCrystal
        Include the non-magnetic ions; the magnetic sites alone give an upper
        bound of the symmetry.

    Returns
    -------
    tuple of SymmetryOperation
        The identity first. The order equals the order of the layer point group
        times the number of centring translations of the given cell.
    """
    scale = float(np.max(np.linalg.norm(crystal.lattice, axis=1)))
    z_tolerance = SYMMETRY_POSITION_TOLERANCE * scale
    reference = 0
    found: Dict[Tuple, SymmetryOperation] = {}
    for matrix in _lattice_point_operations(crystal.lattice):
        for sign in (1, -1):
            for j in range(len(crystal.positions)):
                if crystal.species[j] != crystal.species[reference]:
                    continue
                t = crystal.positions[j] - crystal.positions[reference] @ matrix
                tz = crystal.heights[j] - sign * crystal.heights[reference]
                operation = _make_operation(crystal.lattice, matrix, sign, t, tz)
                if operation._key() not in found and _maps_crystal(crystal, operation, z_tolerance):
                    found[operation._key()] = operation
    return _identity_first(list(found.values()))


def _identity_first(operations: List[SymmetryOperation]) -> Tuple[SymmetryOperation, ...]:
    def is_identity(op):
        return (np.array_equal(op.matrix, np.eye(2, dtype=int)) and op.normal_sign == 1
                and np.allclose(op.translation, 0) and abs(op.height_shift) < 1e-12)
    return tuple(sorted(operations, key=lambda op: not is_identity(op)))


def compose(lattice, first: SymmetryOperation, second: SymmetryOperation) -> SymmetryOperation:
    """``second`` after ``first``: ``r -> second(first(r))``."""
    matrix = first.matrix @ second.matrix
    t = first.translation @ second.matrix + second.translation
    tz = second.normal_sign * first.height_shift + second.height_shift
    return _make_operation(np.asarray(lattice, dtype=float), matrix,
                           first.normal_sign * second.normal_sign, t, tz)


def close_group(lattice, generators: Sequence[SymmetryOperation]) -> Tuple[SymmetryOperation, ...]:
    """The group generated by ``generators`` (translations reduced mod 1).

    Raises
    ------
    SymmetryError
        If the closure exceeds :data:`MAX_GROUP_ORDER`, which happens when a
        generator carries a non-lattice translation of infinite order.
    """
    lattice = np.asarray(lattice, dtype=float)
    identity = _make_operation(lattice, np.eye(2, dtype=int), 1, (0.0, 0.0), 0.0)
    group = {identity._key(): identity}
    frontier = [identity]
    while frontier:
        new = []
        for op in frontier:
            for generator in generators:
                product = compose(lattice, op, generator)
                if product._key() not in group:
                    group[product._key()] = product
                    new.append(product)
        if len(group) > MAX_GROUP_ORDER:
            raise SymmetryError(f"the generators do not close into a layer group "
                                f"(more than {MAX_GROUP_ORDER} operations)")
        frontier = new
    return _identity_first(list(group.values()))


# ----------------------------------------------------------------------
# Action on a spin model
# ----------------------------------------------------------------------

def _orient(source: Participant, target: Participant) -> Tuple[Tuple, bool]:
    """Canonical bond key and whether (source, target) is reversed with respect to it.

    Same rule as the model fingerprint: the lexicographically smaller of
    ``(a, b, n2 - n1)`` and ``(b, a, n1 - n2)``.
    """
    (a, n1), (b, n2) = source, target
    forward = (a, b, tuple(int(y - x) for x, y in zip(n1, n2)))
    backward = (b, a, tuple(int(x - y) for x, y in zip(n1, n2)))
    return (backward, True) if backward < forward else (forward, False)


class CrystalSymmetry:
    """Symmetry group of a layer crystal acting on the sites of a spin model.

    Parameters
    ----------
    crystal : LayerCrystal
        The crystal with its non-magnetic ions. Every model site must coincide
        with one crystal atom (mod lattice translations).
    sites : sequence of Site
        Magnetic sites of the model (e.g. ``model.sites``). The model lattice
        must equal ``crystal.lattice``.
    operations : sequence of SymmetryOperation, optional
        Generators of the group; default :func:`find_symmetry` of ``crystal``.
        Generators are closed into a group and checked against the crystal.

    Raises
    ------
    SymmetryError
        If a site is not a crystal atom, two atoms share a site's in-plane
        position, an operation does not map the crystal onto itself, or an
        operation maps a magnetic site onto a position that is not a model site.
    """

    def __init__(self, crystal: LayerCrystal, sites: Sequence[Site],
                 operations: Optional[Sequence[SymmetryOperation]] = None):
        self.crystal = crystal
        self.lattice = crystal.lattice
        scale = float(np.max(np.linalg.norm(self.lattice, axis=1)))
        self._z_tolerance = SYMMETRY_POSITION_TOLERANCE * scale
        if operations is None:
            self.operations = find_symmetry(crystal)
        else:
            self.operations = close_group(self.lattice, operations)
            bad = [i for i, op in enumerate(self.operations)
                   if not _maps_crystal(crystal, op, self._z_tolerance)]
            if bad:
                raise SymmetryError(f"{len(bad)} operation(s) of the closed group do not map "
                                    "the crystal onto itself")
        self.sites = tuple(sites)
        self._positions = {s.id: np.asarray(s.position, dtype=float) for s in self.sites}
        self._heights = {s.id: self._height_of(s) for s in self.sites}
        self._site_maps = [self._site_map(op) for op in self.operations]

    def _height_of(self, site: Site) -> float:
        matches = [j for j in range(len(self.crystal.positions))
                   if np.all(np.abs(_wrap(site.position - self.crystal.positions[j]))
                             < SYMMETRY_POSITION_TOLERANCE)]
        if not matches:
            raise SymmetryError(f"site {site.id!r} at {site.position.tolist()} is not an atom "
                                "of the crystal")
        if len(matches) > 1:
            raise SymmetryError(f"site {site.id!r}: several crystal atoms share its in-plane "
                                "position; the height of the magnetic site is ambiguous")
        return float(self.crystal.heights[matches[0]])

    def _site_map(self, op: SymmetryOperation) -> Dict[str, Tuple[str, np.ndarray]]:
        out = {}
        for site in self.sites:
            image, height = op.apply(self._positions[site.id], self._heights[site.id])
            for other in self.sites:
                shift = image - self._positions[other.id]
                if (np.all(np.abs(_wrap(shift)) < SYMMETRY_POSITION_TOLERANCE)
                        and abs(height - self._heights[other.id]) < self._z_tolerance):
                    out[site.id] = (other.id, np.round(shift).astype(int))
                    break
            else:
                raise SymmetryError(f"an operation maps site {site.id!r} onto a position "
                                    "that is not a site of the model; include every "
                                    "symmetry-equivalent magnetic site")
        return out

    @property
    def order(self) -> int:
        """Number of operations (modulo lattice translations)."""
        return len(self.operations)

    # -- images ---------------------------------------------------------
    def image_of_participant(self, index: int, participant: Participant) -> Participant:
        """Image of ``(site_id, cell)`` under operation ``index``."""
        site, cell = participant
        target, shift = self._site_maps[index][site]
        new_cell = np.asarray(cell, dtype=int) @ self.operations[index].matrix + shift
        return target, tuple(int(x) for x in new_cell)

    def _bond_image(self, index: int, source: Participant, target: Participant, J: np.ndarray):
        """Canonical key and canonically oriented coefficient of the image bond."""
        new_source = self.image_of_participant(index, source)
        new_target = self.image_of_participant(index, target)
        R = self.operations[index].spin_rotation
        key, reversed_ = _orient(new_source, new_target)
        coefficient = R @ J @ R.T
        return key, (coefficient.T if reversed_ else coefficient)

    @staticmethod
    def _canonical(source: Participant, target: Participant, J: Any):
        key, reversed_ = _orient(source, target)
        J = np.asarray(J, dtype=float)
        return key, (J.T if reversed_ else J)

    @staticmethod
    def _null_space(blocks: List[np.ndarray]) -> np.ndarray:
        if not blocks:
            return np.eye(9).reshape(9, 3, 3)
        stacked = np.vstack(blocks)
        _, values, vh = np.linalg.svd(stacked)
        rank = int(np.sum(values > SYMMETRY_COEFFICIENT_TOLERANCE * max(1.0, values[0])))
        basis = vh[rank:].reshape(-1, 3, 3)
        return _tidy_basis(basis)

    # -- bonds ----------------------------------------------------------
    def bond_stabilizer(self, source: Participant, target: Participant) -> Tuple[int, ...]:
        """Indices of operations that map the bond onto itself (either orientation)."""
        key, _ = _orient(source, target)
        unit = np.eye(3)
        return tuple(i for i in range(self.order)
                     if self._bond_image(i, source, target, unit)[0] == key)

    def allowed_exchange(self, source: Participant, target: Participant) -> np.ndarray:
        """Basis of the exchange matrices ``J`` allowed on the bond ``source -> target``.

        ``J`` is in the orientation ``S_source^T J S_target``. An operation that
        keeps the bond requires ``J = R_s J R_s^T``; one that reverses it
        requires ``J^T = R_s J R_s^T``.

        Returns
        -------
        (m, 3, 3) array
            Orthonormal (Frobenius) basis; ``m`` is the number of independent
            parameters.
        """
        key, flipped = _orient(source, target)
        blocks = []
        for i in self.bond_stabilizer(source, target):
            columns = []
            for unit in np.eye(9):
                X = unit.reshape(3, 3)          # coefficient in canonical orientation
                _, image = self._bond_image(i, *_participants(key), X)
                columns.append((image - X).ravel())
            blocks.append(np.array(columns).T)
        basis = self._null_space(blocks)
        return np.transpose(basis, (0, 2, 1)) if flipped else basis

    def bilinear_terms(self, source: Participant, target: Participant, J: Any,
                       label: Optional[str] = None) -> Tuple[Term, ...]:
        """All symmetry-equivalent bilinear terms generated from one bond.

        Parameters
        ----------
        source, target : (site_id, (n1, n2))
            Representative bond.
        J : array_like, shape (3, 3)
            Exchange on the representative bond, ``S_source^T J S_target``.
        label : str, optional

        Returns
        -------
        tuple of Term
            One term per translation orbit of the bond star, the representative
            first, each in canonical orientation.

        Raises
        ------
        SymmetryError
            If ``J`` is not allowed by the bond's site symmetry; the message
            gives the symmetry-breaking part.
        """
        J = np.asarray(J, dtype=float)
        basis = self.allowed_exchange(source, target)
        projected = np.einsum("m,mab->ab", np.einsum("mab,ab->m", basis, J), basis)
        residual = J - projected
        if np.linalg.norm(residual) > SYMMETRY_COEFFICIENT_TOLERANCE * max(1.0, np.linalg.norm(J)):
            raise SymmetryError(
                f"J on bond {source}->{target} breaks the crystal symmetry; its forbidden "
                f"part is\n{np.array2string(residual, precision=6)}\n"
                "use allowed_exchange() for the allowed form")
        key, Jc = self._canonical(source, target, J)
        found: Dict[Tuple, np.ndarray] = {}
        for i in range(self.order):
            image_key, image = self._bond_image(i, *_participants(key), Jc)
            if image_key in found:
                if not np.allclose(found[image_key], image, atol=1e-9 * max(1.0, np.abs(J).max())):
                    raise SymmetryError("inconsistent images of one bond; the group is not closed")
                continue
            found[image_key] = image
        return tuple(Term.bilinear(*_participants(k), C, label=label) for k, C in found.items())

    # -- sites ----------------------------------------------------------
    def site_stabilizer(self, site: str) -> Tuple[int, ...]:
        """Indices of operations that map ``site`` onto itself (up to a lattice translation)."""
        return tuple(i for i, m in enumerate(self._site_maps) if m[site][0] == site)

    def allowed_g_tensor(self, site: str) -> np.ndarray:
        """Basis of g-tensors allowed on ``site``: ``g = R_s g R_s^T`` for its stabilizer.

        Returns
        -------
        (m, 3, 3) array
            Orthonormal (Frobenius) basis.
        """
        blocks = []
        for i in self.site_stabilizer(site):
            R = self.operations[i].spin_rotation
            blocks.append(np.array([(R @ u.reshape(3, 3) @ R.T - u.reshape(3, 3)).ravel()
                                    for u in np.eye(9)]).T)
        return self._null_space(blocks)

    def zeeman_terms(self, site: str, g: Any, label: Optional[str] = None) -> Tuple[Term, ...]:
        """Zeeman terms of the orbit of ``site`` generated from its g-tensor.

        Raises
        ------
        SymmetryError
            If ``g`` is not allowed by the site symmetry.
        """
        g = np.asarray(g, dtype=float)
        basis = self.allowed_g_tensor(site)
        projected = np.einsum("m,mab->ab", np.einsum("mab,ab->m", basis, g), basis)
        if np.linalg.norm(g - projected) > SYMMETRY_COEFFICIENT_TOLERANCE * max(1.0, np.linalg.norm(g)):
            raise SymmetryError(f"g-tensor of site {site!r} breaks the crystal symmetry; its "
                                f"forbidden part is\n{np.array2string(g - projected, precision=6)}")
        found: Dict[str, np.ndarray] = {}
        for i, site_map in enumerate(self._site_maps):
            image = site_map[site][0]
            R = self.operations[i].spin_rotation
            found.setdefault(image, R @ g @ R.T)
        return tuple(Term.zeeman(s, G, label=label) for s, G in found.items())

    # -- model check ----------------------------------------------------
    def check_model(self, model: SpinModel) -> List[str]:
        """Terms of ``model`` that are not invariant under the group.

        Returns
        -------
        list of str
            One entry per (term, operation) whose image is missing from the
            model or has a different coefficient; empty if the model is
            symmetric.
        """
        if not np.allclose(model.lattice, self.lattice):
            return ["model lattice differs from the crystal lattice"]
        if set(model.site_ids) != set(self._positions):
            return ["model sites differ from the sites of this symmetry object"]
        bonds = {}
        for term in model.terms_of_kind(BILINEAR):
            key, C = self._canonical(*term.participants, term.coefficient)
            bonds[key] = C
        g_tensors = {t.participants[0][0]: t.coefficient for t in model.terms_of_kind(ZEEMAN)}
        violations = []
        for key, C in bonds.items():
            scale = max(1.0, float(np.abs(C).max()))
            for i in range(self.order):
                image_key, image = self._bond_image(i, *_participants(key), C)
                other = bonds.get(image_key)
                if other is None:
                    violations.append(f"bond {key}: operation {i} maps it to {image_key}, "
                                      "which has no term")
                elif not np.allclose(other, image, atol=SYMMETRY_COEFFICIENT_TOLERANCE * scale):
                    violations.append(f"bond {key}: operation {i} maps it to {image_key} "
                                      "with a different exchange matrix")
        for site, g in g_tensors.items():
            for i, site_map in enumerate(self._site_maps):
                image = site_map[site][0]
                R = self.operations[i].spin_rotation
                other = g_tensors.get(image)
                if other is None or not np.allclose(other, R @ g @ R.T,
                                                    atol=SYMMETRY_COEFFICIENT_TOLERANCE * max(1.0, np.abs(g).max())):
                    violations.append(f"zeeman of {site!r}: operation {i} requires "
                                      f"g({image!r}) = R_s g R_s^T")
        return violations


def _participants(key: Tuple) -> Tuple[Participant, Participant]:
    a, b, delta = key
    return (a, (0, 0)), (b, tuple(delta))


def _tidy_basis(basis: np.ndarray) -> np.ndarray:
    """Reduced row-echelon form of the basis, re-orthonormalized: readable and deterministic."""
    if len(basis) == 0:
        return basis
    matrix = basis.reshape(len(basis), 9).copy()
    rows, cols = matrix.shape
    pivot_row = 0
    for col in range(cols):
        if pivot_row == rows:
            break
        pivot = pivot_row + int(np.argmax(np.abs(matrix[pivot_row:, col])))
        if abs(matrix[pivot, col]) < 1e-10:
            continue
        matrix[[pivot_row, pivot]] = matrix[[pivot, pivot_row]]
        matrix[pivot_row] /= matrix[pivot_row, col]
        for r in range(rows):
            if r != pivot_row:
                matrix[r] -= matrix[r, col] * matrix[pivot_row]
        pivot_row += 1
    matrix[np.abs(matrix) < 1e-12] = 0.0
    q, _ = np.linalg.qr(matrix.T)
    q = q.T
    # QR may flip signs; make the first non-zero entry of each vector positive.
    for row in q:
        nonzero = np.flatnonzero(np.abs(row) > 1e-12)
        if nonzero.size and row[nonzero[0]] < 0:
            row *= -1
    q[np.abs(q) < 1e-12] = 0.0
    return q.reshape(-1, 3, 3)
