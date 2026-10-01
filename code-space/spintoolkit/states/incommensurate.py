"""Single-Q spiral states, commensurate or incommensurate (D34).

An :class:`IncommensurateStructure` is the classical state

    n_a(R) = R_n(2 pi q . (n1, n2)) d_a,        R = n1 a1 + n2 a2,

where ``R_n(phi)`` is the right-handed rotation by ``phi`` about the common
unit axis ``n``, ``q`` is the ordering wave vector in the primitive reciprocal
basis (``Q = q1 b1 + q2 b2``, so ``Q . R = 2 pi q . (n1, n2)``) and ``d_a`` is
the direction of site ``a`` in the cell ``(0, 0)``. The phase uses the cell
index only, so ``d_a`` is literally the spin of site ``a`` in the origin cell;
the component ``n . d_a`` is the cone of site ``a`` (zero for a planar
spiral). ``(q, n)`` and ``(-q, -n)`` describe the same state.

The state is defined for any ``q``: rational ``q`` gives a commensurate state
that :meth:`IncommensurateStructure.to_spin_state` writes on a finite
supercell, irrational ``q`` has no finite magnetic cell. Linear spin-wave
theory about it is :func:`spintoolkit.methods.lswt.spiral.solve_spiral_lswt`
(rotating frame, Toth and Lake, J. Phys.: Condens. Matter 27, 166002 (2015)).

Like :class:`~spintoolkit.states.spin_state.SpinState`, the state refers to its
model by fingerprint and is not part of the model; stationarity and stability
are diagnosed by the calculation methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from spintoolkit.definitions.defaults import DEFAULT_TOLERANCE
from spintoolkit.states.spin_state import SpinState, SpinStateError
from spintoolkit.system.geometry import integer_matrix
from spintoolkit.system.model import SpinModel

SCHEMA_VERSION = 1

#: Largest |cross product| between the rotation axes of two sites of an LT
#: amplitude still treated as one common axis.
SPIRAL_AXIS_TOLERANCE = 1e-8


def cross_matrix(axis: Sequence[float]) -> np.ndarray:
    """``[n]_x`` with ``[n]_x v = n x v``."""
    x, y, z = axis
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_matrix(axis: Sequence[float], angle) -> np.ndarray:
    """Right-handed rotation about the unit ``axis`` by ``angle`` (Rodrigues).

    ``angle`` may be an array; the result then has shape ``angle.shape + (3, 3)``.
    """
    n = np.asarray(axis, dtype=float)
    angle = np.asarray(angle, dtype=float)
    K = cross_matrix(n)
    c, s = np.cos(angle)[..., None, None], np.sin(angle)[..., None, None]
    return c * np.eye(3) + s * K + (1 - c) * np.outer(n, n)


@dataclass(frozen=True)
class IncommensurateStructure:
    """Single-Q spiral (planar or conical) about one rotation axis.

    Parameters
    ----------
    model_ref : str
        Fingerprint of the model (:meth:`SpinModel.fingerprint`).
    wave_vector : array_like, shape (2,)
        Ordering wave vector ``q`` in the primitive reciprocal basis.
    rotation_axis : array_like, shape (3,)
        Unit axis ``n`` of the rotation, in the global spin frame.
    directions : mapping of site_id to array_like, shape (3,)
        Unit direction ``d_a`` of every model site in the cell ``(0, 0)``.
    provenance : mapping, optional
        Origin of the state, e.g. ``{"origin": "luttinger_tisza"}``.

    Raises
    ------
    SpinStateError
        If a vector is not a finite unit vector or the wave vector is not finite.
    """

    model_ref: str
    wave_vector: np.ndarray
    rotation_axis: np.ndarray
    directions: Mapping[str, np.ndarray]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        violations = []
        q = np.array(self.wave_vector, dtype=float, copy=True)
        if q.shape != (2,) or not np.all(np.isfinite(q)):
            violations.append(f"wave_vector must be finite with shape (2,), got {q.tolist()}")
        q.setflags(write=False)
        object.__setattr__(self, "wave_vector", q)
        axis = np.array(self.rotation_axis, dtype=float, copy=True)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) \
                or abs(np.linalg.norm(axis) - 1.0) > DEFAULT_TOLERANCE:
            violations.append(f"rotation_axis must be a unit vector, got {axis.tolist()}")
        axis.setflags(write=False)
        object.__setattr__(self, "rotation_axis", axis)
        directions = {}
        for site, vector in dict(self.directions).items():
            if np.iscomplexobj(vector):
                raise TypeError(f"direction of {site!r} must be real")
            array = np.array(vector, dtype=float, copy=True)
            if array.shape != (3,) or not np.all(np.isfinite(array)) \
                    or abs(np.linalg.norm(array) - 1.0) > DEFAULT_TOLERANCE:
                violations.append(f"{site!r}: direction must be a unit vector, got {array.tolist()}")
            array.setflags(write=False)
            directions[str(site)] = array
        if not directions:
            violations.append("state has no directions")
        object.__setattr__(self, "directions", MappingProxyType(directions))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))
        if violations:
            raise SpinStateError(violations)

    # -- construction -----------------------------------------------------
    @classmethod
    def planar(cls, model: SpinModel, wave_vector: Any, rotation_axis: Any,
               phases: Optional[Mapping[str, float]] = None,
               reference: Optional[Any] = None,
               provenance: Optional[Mapping[str, Any]] = None) -> "IncommensurateStructure":
        """Planar spiral ``d_a = R_n(phase_a) e`` with ``e`` perpendicular to ``n``.

        Parameters
        ----------
        model : SpinModel
        wave_vector : array_like, shape (2,)
        rotation_axis : array_like, shape (3,)
            Normalized here.
        phases : mapping of site_id to float, optional
            In-plane angle of each site in the cell ``(0, 0)`` (default zero).
        reference : array_like, shape (3,), optional
            Direction at phase zero; projected onto the plane (default: a
            fixed vector perpendicular to ``n``).
        """
        n = np.asarray(rotation_axis, dtype=float)
        n = n / np.linalg.norm(n)
        e = _perpendicular(n) if reference is None else np.asarray(reference, dtype=float)
        e = e - (e @ n) * n
        e = e / np.linalg.norm(e)
        phases = dict(phases or {})
        directions = {s: rotation_matrix(n, phases.get(s, 0.0)) @ e for s in model.site_ids}
        return cls(model.fingerprint(), wave_vector, n, directions,
                   provenance or {"origin": "planar"})

    @classmethod
    def from_lt(cls, model: SpinModel, wave_vector) -> "IncommensurateStructure":
        """The single-Q spiral of a Luttinger-Tisza minimum.

        Parameters
        ----------
        model : SpinModel
        wave_vector : LTWaveVector
            A minimum of :func:`~spintoolkit.methods.luttinger_tisza.luttinger_tisza`
            that satisfies the strong constraint with a spiral amplitude
            (``u_a . u_a = 0``, ``|u_a|^2 = 2``).

        Raises
        ------
        ValueError
            If the minimum has no single-Q state, is collinear (``2 q`` or
            ``4 q`` a reciprocal vector: use its ``state`` on the supercell),
            or rotates different sites about different axes (not one spiral).
        """
        if not wave_vector.strong_constraint or wave_vector.amplitude is None:
            raise ValueError("this LT minimum has no single-Q state with unit spins")
        u = np.asarray(wave_vector.amplitude)
        if np.max(np.abs(np.sum(u * u, axis=1))) > 1e-6:
            raise ValueError("the LT state at this minimum is not a spiral (2q or 4q is a "
                             "reciprocal vector); use LTWaveVector.state on its supercell")
        e1, e2 = np.real(u), np.imag(u)
        # n_a(R) = Re[u_a e^{i phi}] = e1 cos(phi) - e2 sin(phi): a rotation about e2 x e1.
        axes = np.cross(e2, e1)
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        if np.max(np.linalg.norm(np.cross(axes, axes[0]), axis=1)) > SPIRAL_AXIS_TOLERANCE \
                or np.min(axes @ axes[0]) < 0:
            raise ValueError("the LT amplitude rotates different sites about different axes; "
                             "it is not a single-axis spiral")
        directions = {s: e1[i] / np.linalg.norm(e1[i]) for i, s in enumerate(model.site_ids)}
        return cls(model.fingerprint(), wave_vector.fractional, axes[0], directions,
                   {"origin": "luttinger_tisza", "q": np.asarray(wave_vector.fractional).tolist()})

    # -- access -------------------------------------------------------------
    @property
    def site_ids(self) -> Tuple[str, ...]:
        """Site identifiers present in the state, sorted."""
        return tuple(sorted(self.directions))

    def cartesian_wave_vector(self, model: SpinModel) -> np.ndarray:
        """``Q = q @ B`` with the primitive reciprocal vectors ``B = 2 pi A^-T`` as rows."""
        return self.wave_vector @ (2 * np.pi * np.linalg.inv(model.lattice).T)

    def phase(self, cell: Sequence[int]) -> float:
        """Rotation angle ``2 pi q . cell`` of a primitive cell."""
        return float(2 * np.pi * self.wave_vector @ np.asarray(cell, dtype=float))

    def direction(self, site_id: str, cell: Sequence[int] = (0, 0)) -> np.ndarray:
        """Unit direction of a site in any cell of the infinite lattice."""
        return rotation_matrix(self.rotation_axis, self.phase(cell)) @ self.directions[site_id]

    def cone_angles(self) -> Dict[str, float]:
        """Angle between each ``d_a`` and the rotation axis (``pi/2`` for a planar spiral)."""
        return {s: float(np.arccos(np.clip(d @ self.rotation_axis, -1.0, 1.0)))
                for s, d in self.directions.items()}

    def commensurate_supercell(self, max_denominator: int = 64) -> Optional[np.ndarray]:
        """Smallest integer supercell on which the state is periodic, or None.

        ``q`` must be a fraction with denominators at most ``max_denominator``.
        """
        from fractions import Fraction
        from spintoolkit.methods.luttinger_tisza import _supercell

        fractions = [Fraction(float(x)).limit_denominator(max_denominator) for x in self.wave_vector]
        if any(abs(float(f) - x) > 1e-12 for f, x in zip(fractions, self.wave_vector)):
            return None
        fractions = [f - (f.numerator // f.denominator) for f in fractions]
        if all(f == 0 for f in fractions):
            return np.eye(2, dtype=int)
        return _supercell(fractions)

    def to_spin_state(self, model: SpinModel, supercell: Optional[Any] = None) -> SpinState:
        """The same state on a commensurate supercell, for the ordinary LSWT path.

        Parameters
        ----------
        model : SpinModel
        supercell : (2, 2) integer array, optional
            Default :meth:`commensurate_supercell`.

        Raises
        ------
        SpinStateError
            If ``q`` is not commensurate with the supercell.
        """
        matrix = self.commensurate_supercell() if supercell is None else \
            integer_matrix(supercell, "supercell")
        if matrix is None:
            raise SpinStateError([f"wave vector {self.wave_vector.tolist()} is not commensurate "
                                  "with a supercell of denominator <= 64"])
        winding = matrix @ self.wave_vector
        if np.max(np.abs(winding - np.rint(winding))) > 1e-9:
            raise SpinStateError([f"wave vector {self.wave_vector.tolist()} is not commensurate "
                                  f"with the supercell {np.asarray(matrix).tolist()}"])
        return SpinState.from_function(model, matrix, self.direction,
                                       {"origin": "spiral", **dict(self.provenance)})


def _perpendicular(n: np.ndarray) -> np.ndarray:
    trial = np.eye(3)[int(np.argmin(np.abs(n)))]
    v = trial - (trial @ n) * n
    return v / np.linalg.norm(v)


def validate_spiral(structure: IncommensurateStructure, model: SpinModel) -> List[str]:
    """Check a spiral against its model.

    Raises
    ------
    SpinStateError
        If the state belongs to another model or misses or adds sites.
    """
    violations = []
    if structure.model_ref != model.fingerprint():
        violations.append("model_ref does not match the model fingerprint "
                          "(the state was built for a different model)")
    expected, present = set(model.site_ids), set(structure.site_ids)
    if present - expected:
        violations.append(f"sites {sorted(present - expected)} are not in the model")
    if expected - present:
        violations.append(f"model sites {sorted(expected - present)} have no directions")
    if violations:
        raise SpinStateError(violations)
    return []
