"""Zero-mode scan of an LSWT result (stage 4b, D25).

A zero mode of the bosonic Hamiltonian makes finite-temperature boson numbers
diverge in two dimensions. It is detected from the lowest eigenvalue
``lambda_min`` of ``H(k)``, not from the magnon energy: at an exact zero mode
``eta H`` has a Jordan block, so round-off shifts ``omega`` by about the square
root of machine precision while ``lambda_min`` stays at round-off size. A small
physical gap ``Delta`` gives ``lambda_min ~ Delta^2 / bandwidth``, so a fixed
threshold cannot separate a tiny gap from a zero mode. Every examined point is
therefore put in one of three classes, with ``lambda = lambda_min / scale`` and
``scale`` the largest ``|eig H(k)|`` on the mesh:

- ``zero``:      ``lambda <= zero_tolerance`` (round-off size);
- ``candidate``: ``zero_tolerance < lambda <= candidate_tolerance``; the user
  decides whether it is a zero mode or a small gap;
- ``gapped``:    otherwise.

Points examined: the zone centre and the high-symmetry points of the magnetic
reciprocal cell, the high-symmetry points of the primitive cell folded into
the magnetic cell, and the lowest local minima of ``lambda_min`` on the mesh.
From each, ``lambda_min`` is minimized over continuous ``k`` so that zero modes
between mesh points are found (only from starting points whose value is
below ``search_threshold``; farther points cannot reach zero within a mesh
step), and a zero or candidate point is probed in twelve directions to flag
lines of zero modes.

At the zone centre the origin is classified when the model and state are
given: ``goldstone`` if a continuous spin-rotation symmetry of every term is
broken by the state, ``accidental`` if the classical Hessian of the magnetic
cell has more flat directions than the broken symmetries explain (e.g. the
NBCP Y and V states with J_PD or J_Gamma), otherwise ``unknown``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from spintoolkit.definitions.defaults import (
    ZERO_MODE_CANDIDATE_TOLERANCE, ZERO_MODE_LINE_DIRECTIONS, ZERO_MODE_MESH_SEEDS,
    ZERO_MODE_SEARCH_THRESHOLD, ZERO_MODE_ZERO_TOLERANCE)

ZERO = "zero"
CANDIDATE = "candidate"
GAPPED = "gapped"

_SPECIAL = (0.0, 0.5, 1 / 3, 2 / 3)


@dataclass(frozen=True)
class ZeroModePoint:
    """One examined momentum.

    Attributes
    ----------
    k : (2,) array
        Cartesian momentum.
    fractional : (2,) array
        Coordinates in the magnetic reciprocal basis, in ``[0, 1)``.
    lowest : float
        ``lambda_min / scale``.
    gap : float or None
        Lowest magnon energy (E0) when ``H(k)`` is positive definite.
    classification : {"zero", "candidate", "gapped"}
    source : str
    line_directions : tuple of float
        Directions (radians) along which ``lambda_min`` stays at or below the
        candidate tolerance.
    origin : str
    """

    k: np.ndarray
    fractional: np.ndarray
    lowest: float
    gap: Optional[float]
    classification: str
    source: str
    line_directions: Tuple[float, ...] = ()
    origin: str = "not examined"


@dataclass(frozen=True)
class ZeroModeReport:
    """Result of :func:`scan_zero_modes`.

    ``candidates`` holds the zero and candidate points (deduplicated), lowest
    first; ``global_minimum`` is the lowest examined point in any class.
    """

    candidates: Tuple[ZeroModePoint, ...]
    global_minimum: ZeroModePoint
    scale: float
    zero_tolerance: float
    candidate_tolerance: float
    zone_centre: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_zero(self) -> bool:
        return any(c.classification == ZERO for c in self.candidates)

    @property
    def has_candidates(self) -> bool:
        return any(c.classification == CANDIDATE for c in self.candidates)

    def summary(self) -> str:
        if not self.candidates:
            m = self.global_minimum
            return (f"gapped: lowest lambda_min/scale {m.lowest:.3g} at q={_show(m.fractional)}"
                    + (f", gap {m.gap:.6g}" if m.gap is not None else ""))
        lines = []
        for c in self.candidates:
            text = (f"{c.classification} at q={_show(c.fractional)} "
                    f"(lambda_min/scale {c.lowest:.3g}, {c.source}, origin {c.origin}")
            if c.gap is not None:
                text += f", gap {c.gap:.3g}"
            if c.line_directions:
                text += f", flat along {len(c.line_directions)} of {ZERO_MODE_LINE_DIRECTIONS} directions"
            lines.append(text + ")")
        return "; ".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        from spintoolkit.methods.result import to_jsonable
        return to_jsonable({"candidates": list(self.candidates),
                            "global_minimum": self.global_minimum, "scale": self.scale,
                            "zero_tolerance": self.zero_tolerance,
                            "candidate_tolerance": self.candidate_tolerance,
                            "zone_centre": self.zone_centre, "summary": self.summary()})


def _show(p) -> list:
    """Fractional momentum for display: rounded, then folded into [0, 1)."""
    return (np.mod(np.round(np.asarray(p, dtype=float), 6), 1.0) + 0.0).tolist()


def _classify(value: float, zero_tolerance: float, candidate_tolerance: float) -> str:
    if value <= zero_tolerance:
        return ZERO
    if value <= candidate_tolerance:
        return CANDIDATE
    return GAPPED


def _gap(H: np.ndarray) -> Optional[float]:
    from spintoolkit.methods.lswt.diagonalization import Diagonalizer

    n = H.shape[0] // 2
    try:
        K = np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return None
    energies, _ = Diagonalizer.Colpa(K, np.diag(np.r_[np.ones(n), -np.ones(n)]))
    return float(np.min(energies[:n]))


def _symmetry_origin(model, state, conditions) -> Dict[str, Any]:
    """Broken continuous symmetries and classical flat directions at k = 0."""
    from spintoolkit.methods.classical import tangent_expansion
    from spintoolkit.system.model import BILINEAR, ZEEMAN

    def cross(n):
        return np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

    field_vector = conditions.field
    columns = []
    for axis in np.eye(3):
        K = cross(axis)
        parts = [(K @ t.coefficient - t.coefficient @ K).ravel()
                 for t in model.terms if t.kind in (BILINEAR, "onsite")]
        parts += [np.cross(axis, t.coefficient.T @ field_vector)
                  for t in model.terms_of_kind(ZEEMAN)]
        columns.append(np.concatenate(parts) if parts else np.zeros(1))
    L = np.column_stack(columns)
    _, s, vt = np.linalg.svd(L)
    tolerance = 1e-10 * max(1.0, s.max(initial=0.0))
    symmetry_axes = vt[np.sum(s > tolerance):].T                   # (3, n_sym)
    expansion = tangent_expansion(model, state, conditions)
    generators = np.array([[np.cross(axis, n) @ e for axis in np.eye(3)]
                           for n, frame in zip(expansion.directions, expansion.frames)
                           for e in frame])
    moved = generators @ symmetry_axes if symmetry_axes.size else np.zeros((len(generators), 0))
    sv = np.linalg.svd(moved, compute_uv=False) if moved.size else np.zeros(0)
    broken = int(np.sum(sv > 1e-8 * max(1.0, sv.max(initial=0.0))))
    w = np.abs(np.linalg.eigvalsh(expansion.hessian))
    flat = int(np.sum(w <= 1e-10 * max(w.max(initial=0.0), 1e-300)))
    origin = ("goldstone" if broken and flat <= broken else
              "goldstone and accidental" if broken else
              "accidental" if flat else "unknown")
    return {"symmetry_axes": int(symmetry_axes.shape[1]) if symmetry_axes.size else 0,
            "broken_symmetries": broken, "classical_flat_directions": flat, "origin": origin}


def scan_zero_modes(result, model=None, state=None, conditions=None,
                    zero_tolerance: float = ZERO_MODE_ZERO_TOLERANCE,
                    candidate_tolerance: float = ZERO_MODE_CANDIDATE_TOLERANCE,
                    mesh_seeds: int = ZERO_MODE_MESH_SEEDS,
                    search_threshold: float = ZERO_MODE_SEARCH_THRESHOLD) -> ZeroModeReport:
    """Find zero modes and zero-mode candidates of an :class:`LSWTResult`.

    Parameters
    ----------
    result : LSWTResult
        Its stored ``H(k)`` (minus any regularization shift) give the mesh
        values; ``result.hamiltonian_at`` evaluates other momenta.
    model, state, conditions : optional
        Enable the origin classification at the zone centre.
    zero_tolerance, candidate_tolerance : float
        Class boundaries for ``lambda_min / scale``.
    mesh_seeds : int
        Number of lowest mesh points used as starting points.
    search_threshold : float
        Minimize ``lambda_min`` only from starting points at or below this.

    Returns
    -------
    ZeroModeReport
    """
    if result.hamiltonian_at is None or result.magnetic_lattice is None:
        raise ValueError("the result has no Hamiltonian evaluator (produced by solve_lswt?)")
    n2 = result.hamiltonians.shape[1]
    identity = np.eye(n2)
    mesh_H = result.hamiltonians - result.regularization_shift[:, None, None] * identity
    mesh_eigs = np.linalg.eigvalsh(mesh_H)
    scale = float(np.max(np.abs(mesh_eigs)))
    magnetic = np.asarray(result.magnetic_lattice)
    to_k = lambda p: 2 * np.pi * np.asarray(p, dtype=float) @ np.linalg.inv(magnetic).T

    def to_p(k):
        p = np.mod(np.asarray(k, dtype=float) @ magnetic.T / (2 * np.pi), 1.0)
        return np.where(np.isclose(p, 1.0, rtol=0, atol=1e-9), 0.0, p)

    def lowest(p):
        return float(np.linalg.eigvalsh(result.hamiltonian_at(to_k(p))[0])[0] / scale)

    seeds: List[Tuple[np.ndarray, str]] = []
    for a in _SPECIAL:
        for b in _SPECIAL:
            seeds.append((np.array([a, b]), "zone centre" if a == b == 0 else
                          "magnetic high-symmetry point"))
    primitive_inverse = np.linalg.inv(np.asarray(result.lattice)).T
    for a in _SPECIAL:
        for b in _SPECIAL:
            seeds.append((to_p(2 * np.pi * np.array([a, b]) @ primitive_inverse.T),
                          "primitive high-symmetry point"))
    mesh_lowest = mesh_eigs[:, 0] / scale
    fractional = (result.fractional if result.fractional is not None
                  else to_p(result.k_points))
    for i in np.argsort(mesh_lowest)[:mesh_seeds]:
        seeds.append((np.asarray(fractional[i]), "mesh minimum"))

    distinct: List[Tuple[np.ndarray, str]] = []
    for p0, source in seeds:
        p0 = to_p(to_k(p0))
        if all(np.linalg.norm(np.mod(p0 - q + 0.5, 1.0) - 0.5) > 1e-9 for q, _ in distinct):
            distinct.append((p0, source))
    step = 1.0 / max(4, int(np.sqrt(len(result.k_points))))
    examined: List[ZeroModePoint] = []
    for p0, source in distinct:
        start = lowest(p0)
        best_p, best = p0, start
        if zero_tolerance < start <= search_threshold:
            simplex = np.array([p0, p0 + [step, 0], p0 + [0, step]])
            opt = minimize(lambda p: lowest(p), p0, method="Nelder-Mead",
                           options={"initial_simplex": simplex, "xatol": 1e-12,
                                    "fatol": 1e-20, "maxiter": 300})
            if opt.fun < best:
                best_p, best = to_p(to_k(opt.x)), float(opt.fun)
        examined.append(ZeroModePoint(to_k(best_p), best_p, best, None,
                                      _classify(best, zero_tolerance, candidate_tolerance), source))

    unique: List[ZeroModePoint] = []
    for point in sorted(examined, key=lambda x: x.lowest):
        distance = [np.linalg.norm(np.mod(point.fractional - u.fractional + 0.5, 1.0) - 0.5)
                    for u in unique]
        if not distance or min(distance) > 1e-5:
            unique.append(point)

    reciprocal = 2 * np.pi * np.linalg.inv(magnetic).T
    delta = 0.02 * min(np.linalg.norm(reciprocal, axis=0))
    angles = np.arange(ZERO_MODE_LINE_DIRECTIONS) * np.pi / ZERO_MODE_LINE_DIRECTIONS
    zone_centre = {}
    if model is not None and state is not None:
        from spintoolkit.system.conditions import ExternalConditions
        zone_centre = _symmetry_origin(model, state, conditions or ExternalConditions())
    finished = []
    for point in unique:
        H0 = result.hamiltonian_at(point.k)[0]
        gap = _gap(H0) if point.classification != ZERO else None
        lines: Tuple[float, ...] = ()
        origin = "not examined"
        if point.classification != GAPPED:
            probes = np.array([point.k + sign * delta * np.array([np.cos(t), np.sin(t)])
                               for t in angles for sign in (1, -1)])
            values = np.linalg.eigvalsh(result.hamiltonian_at(probes))[:, 0] / scale
            flat = np.all(values.reshape(-1, 2) <= candidate_tolerance, axis=1)
            lines = tuple(float(t) for t, f in zip(angles, flat) if f)
            at_centre = np.linalg.norm(np.mod(point.fractional + 0.5, 1.0) - 0.5) < 1e-6
            origin = (zone_centre.get("origin", "not examined") if at_centre
                      else "not a uniform rotation (k != 0)")
        finished.append(ZeroModePoint(point.k, point.fractional, point.lowest, gap,
                                      point.classification, point.source, lines, origin))
    candidates = tuple(p for p in finished if p.classification != GAPPED)
    return ZeroModeReport(candidates, finished[0], scale, zero_tolerance, candidate_tolerance,
                          zone_centre)
