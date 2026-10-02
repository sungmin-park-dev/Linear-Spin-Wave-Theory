"""Magnon band structure along a path in the Brillouin zone (stage 4d, D33).

Input is an :class:`~spintoolkit.methods.lswt.run.LSWTResult`;
``H(k)`` is evaluated at the path momenta with ``result.hamiltonian_at``
(unregularized) and diagonalized by Colpa's method. The magnetic cell has
``Ns`` bands; on a path of the primitive (crystallographic) zone they appear
folded, e.g. three bands for a three-sublattice state.

Stability is read from ``H(k)`` itself. Positive definite (lowest eigenvalue
above ``tolerance`` times the largest): Colpa. Positive semidefinite within the
tolerance (a zero mode such as a Goldstone mode at a path vertex; round-off can
make it slightly positive, as in stage 4a): the energies are the moduli of the
eigenvalues of ``eta H``, which come in pairs ``+-E`` (``+-i delta`` of order
``sqrt(eps)`` at a Jordan block), and the point is marked in ``zero_modes``.
Indefinite (an unstable reference state): NaN and a warning.
Figures are left to the visualization stage (D33).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union
import warnings

import numpy as np

from spintoolkit.methods.lswt.diagonalization import Diagonalizer
from spintoolkit.system.high_symmetry import high_symmetry_points

PathPoint = Union[str, Sequence[float]]


@dataclass(frozen=True)
class BandStructure:
    """Magnon energies along a path.

    Attributes
    ----------
    k_points : (m, 2) array
        Cartesian momenta.
    distance : (m,) array
        Cumulative path length.
    energies : (m, Ns) array
        Magnon energies, ascending at each point (E0); NaN where unstable.
    labels : list of str
        Vertex names (explicit coordinates are shown as tuples).
    label_distances : (n,) array
        Path length at each vertex.
    zero_modes : (m,) bool array
        Points where ``H(k)`` is only positive semidefinite.
    lattice : str
        "primitive" or "magnetic".
    """

    k_points: np.ndarray
    distance: np.ndarray
    energies: np.ndarray
    labels: List[str]
    label_distances: np.ndarray
    zero_modes: np.ndarray
    lattice: str


def _vertices(path, named) -> Tuple[List[str], np.ndarray]:
    labels, points = [], []
    for vertex in path:
        if isinstance(vertex, str):
            if vertex not in named:
                raise ValueError(f"unknown point {vertex!r}; known: {sorted(named)}")
            labels.append(vertex)
            points.append(named[vertex])
        else:
            point = np.asarray(vertex, dtype=float)
            labels.append("(" + ", ".join(f"{x:.4g}" for x in point) + ")")
            points.append(point)
    return labels, np.array(points)


def _path_momenta(vertices: np.ndarray, points: int):
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
    counts = np.maximum(1, np.round(points * lengths / lengths.sum()).astype(int))
    k, distance = [vertices[0]], [0.0]
    for (start, end), length, count in zip(zip(vertices[:-1], vertices[1:]), lengths, counts):
        for step in range(1, count + 1):
            k.append(start + (end - start) * step / count)
            distance.append(distance[-1] + length / count)
    return np.array(k), np.array(distance), np.concatenate([[0.0], np.cumsum(lengths)])


def _energies(H: np.ndarray, tolerance: float) -> Tuple[np.ndarray, bool, bool]:
    """Ascending particle energies; flags (zero mode, unstable)."""
    ns = len(H) // 2
    eta = np.diag(np.r_[np.ones(ns), -np.ones(ns)])
    spectrum = np.linalg.eigvalsh(H)
    scale = max(float(np.max(np.abs(spectrum))), 1e-300)
    if spectrum[0] < -tolerance * scale:
        return np.full(ns, np.nan), False, True
    if spectrum[0] > tolerance * scale:
        try:
            E, _ = Diagonalizer.Colpa(np.linalg.cholesky(H), eta)
            return np.sort(E[:ns]), False, False
        except np.linalg.LinAlgError:
            pass
    moduli = np.sort(np.abs(np.linalg.eigvals(eta @ H)))
    return moduli[::2], True, False                    # each |E| appears twice (+-E)


def band_structure(result, path: Sequence[PathPoint] = ("Γ", "K", "M", "Γ"), points: int = 200,
                   lattice: str = "primitive", tolerance: float = 1e-9) -> BandStructure:
    """Magnon bands of an LSWT result along a piecewise-straight path.

    Parameters
    ----------
    result : LSWTResult
    path : sequence of str or (2,) array_like
        Vertex names of :func:`~spintoolkit.system.high_symmetry.high_symmetry_points`
        or explicit Cartesian momenta.
    points : int
        Approximate number of path points, distributed by segment length.
    lattice : {"primitive", "magnetic"}
        Zone whose high-symmetry points name the vertices.
    tolerance : float
        Relative tolerance for zero modes and instabilities.

    Returns
    -------
    BandStructure
    """
    if result.hamiltonian_at is None:
        raise ValueError("the result has no hamiltonian_at; use solve_lswt")
    if lattice not in ("primitive", "magnetic"):
        raise ValueError("lattice must be 'primitive' or 'magnetic'")
    real_space = result.lattice if lattice == "primitive" else result.magnetic_lattice
    labels, vertices = _vertices(path, high_symmetry_points(real_space))
    k, distance, label_distances = _path_momenta(vertices, points)
    H = result.hamiltonian_at(k)
    energies, zero, unstable = [], [], []
    for Hk in H:
        E, z, u = _energies(Hk, tolerance)
        energies.append(E)
        zero.append(z)
        unstable.append(u)
    if any(unstable):
        warnings.warn(f"H(k) is not positive semidefinite at {sum(unstable)} path points "
                      "(unstable reference state); their energies are NaN", UserWarning,
                      stacklevel=2)
    return BandStructure(k, distance, np.array(energies), labels, label_distances,
                         np.array(zero), lattice)


@dataclass(frozen=True)
class DensityOfStates:
    """Magnon density of states per magnetic site.

    Attributes
    ----------
    omega : (nw,) array
        Energies (E0).
    dos : (nw,) array
        ``g(w) = (1 / Ns) sum_n <delta(w - w_n(k))>_k``; integrates to one
        over all energies (one mode per site).
    num_k : int
        Momenta of the mesh.
    num_bands : int
    """

    omega: np.ndarray
    dos: np.ndarray
    num_k: int
    num_bands: int


def density_of_states(result, omega: Sequence[float], fwhm: float,
                      shape: str = "gaussian") -> DensityOfStates:
    """Broadened magnon density of states from the mesh of an LSWT result.

    Parameters
    ----------
    result : LSWTResult or SpiralLSWTResult
        Its momenta must sample the whole (magnetic) zone, i.e. a mesh from
        ``LSWTSettings.mesh``; explicit k-points give a weighted sum over
        those points only.
    omega : array_like
    fwhm : float
        Full width at half maximum of the broadening; it should exceed the
        level spacing of the mesh, about ``bandwidth / mesh``.
    shape : {"gaussian", "lorentzian"}

    Raises
    ------
    ValueError
        If some band energy is not finite (unstable state).
    """
    from spintoolkit.observables.neutron import broaden_modes

    lab = getattr(result, "rotating", result)
    bands = np.asarray(lab.bands(), dtype=float)                 # (nk, Ns)
    if not np.all(np.isfinite(bands)):
        raise ValueError("band energies are not all finite (unstable reference state)")
    weights = np.asarray(lab.weights, dtype=float)
    ns = bands.shape[1]
    g = broaden_modes(bands.reshape(1, -1), np.repeat(weights, ns)[None] / ns,
                      omega, fwhm, shape)[0]
    return DensityOfStates(np.asarray(omega, dtype=float), g, len(weights), ns)
