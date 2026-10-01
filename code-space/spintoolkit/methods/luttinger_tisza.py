"""Luttinger-Tisza (LT) diagnostic for classical ground states at zero field (D30).

The classical energy of the bilinear terms,

    E = sum_R sum_terms S_a S_b n_a(R + n1)^T J n_b(R + n2),

with unit vectors ``n`` and the Fourier convention of D13 (full positions),
``n_a(r) = N^-1/2 sum_q exp(i q . r_a) n_a(q)``, is ``E = sum_q n(q)^+ J(q) n(q)``
with the Hermitian ``3 Ns x 3 Ns`` matrix

    J(q)_ab = (1/2) [S_a S_b J e^{i q . (r_b - r_a)} + h.c.]   (summed over terms).

The weak constraint ``sum |n|^2 = N Ns`` gives the bound ``E / (N Ns) >= lambda_min``
(energy per site in E0, for any spin lengths). The bound is reached when a
state built from the lowest eigenvectors also satisfies the strong constraint
``|n_a(R)| = 1``. This diagnostic checks single-q states only:

- ``n_a(R) = Re[u_a exp(i q* . R)]`` with the cell amplitude
  ``u_a = w_a exp(i q* . r_a)`` (``w`` an eigenvector, ``r_a`` the site offset);
- ``|n_a(R)|^2 = |u_a|^2 / 2 + Re[(u_a . u_a) exp(2 i q* . R)] / 2``, and the
  phases ``exp(2 i q* . R)`` over all cells give three cases:
- ``2 q*`` a reciprocal vector (zone centre, half reciprocal vectors):
  ``exp(i q* . R) = +-1``, and ``|Re u_a| = 1`` is required;
- ``4 q*`` but not ``2 q*`` a reciprocal vector (quarter vectors): the phases
  are ``+-1``, so ``Re(u_a . u_a) = 0`` and ``|u_a|^2 = 2`` (the angle between
  ``Re u_a`` and ``Im u_a`` is free, e.g. up-up-down-down);
- otherwise: ``u_a . u_a = 0`` and ``|u_a|^2 = 2`` for every site (a spiral).

``w`` is sought in the eigenspace of ``lambda_min`` at ``q*``. Failure means no
single-q LT state; multi-q states and the generalized (Lyons-Kaplan) LT are not
treated, so it is not a proof that the bound is not reached. The field is not
included (Zeeman terms are ignored); the tool reports candidate wave vectors
and cells and never selects the state (the selection stays with the
classical search and D17/D28).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.model import BILINEAR, SpinModel

#: Relative tolerance on eigenvalues for minima, degeneracy and the eigenspace.
LT_TOLERANCE = 1e-9
#: Largest denominator of a commensurate q* in the primitive reciprocal basis.
LT_MAX_DENOMINATOR = 12
#: Residual of the strong-constraint fit below which a single-q state exists.
LT_STRONG_TOLERANCE = 1e-10


@dataclass(frozen=True)
class LTWaveVector:
    """One minimum ``q*`` of ``lambda_min(q)`` (up to ``q -> -q`` and G).

    Attributes
    ----------
    fractional : (2,) array
        ``q*`` in the primitive reciprocal basis, in ``[0, 1)``.
    cartesian : (2,) array
    multiplicity : int
        Dimension of the eigenspace of ``lambda_min`` at ``q*``.
    commensurate : bool
        ``q*`` is a fraction with denominator at most ``LT_MAX_DENOMINATOR``.
    fraction : tuple of str or None
        The fractions, e.g. ``("1/3", "1/3")``.
    supercell : (2, 2) int array or None
        Smallest magnetic supercell (rows in primitive units) with
        ``q* . R`` a multiple of ``2 pi`` for all its vectors.
    strong_constraint : bool
        A single-q state from the eigenspace has unit spins everywhere.
    strong_residual : float
        Residual of the best fit (zero when satisfied).
    state : SpinState or None
        That state on ``supercell``, when both hold.
    state_energy : float or None
        Its classical energy per site (bilinear terms, zero field); equals
        ``lambda_min`` up to round-off.
    amplitude : (Ns, 3) complex array or None
        Cell amplitude ``u_a`` of the best strong-constraint fit (spins
        ``Re[u_a exp(i q* . R)]``); for an incommensurate spiral it builds an
        :class:`~spintoolkit.states.incommensurate.IncommensurateStructure`.
    """

    fractional: np.ndarray
    cartesian: np.ndarray
    multiplicity: int
    commensurate: bool
    fraction: Optional[Tuple[str, str]]
    supercell: Optional[np.ndarray]
    strong_constraint: bool
    strong_residual: float
    state: Optional[SpinState] = field(default=None, repr=False)
    state_energy: Optional[float] = None
    amplitude: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass(frozen=True)
class LTReport:
    """Result of :func:`luttinger_tisza`.

    Attributes
    ----------
    lambda_min : float
        Lower bound on the classical energy per site (E0), zero field.
    minima : list of LTWaveVector
        Distinct minima (``q`` and ``-q`` identified).
    near_minimal_fraction : float
        Fraction of mesh points within the tolerance of ``lambda_min``; well
        above ``len(minima) / N^2`` means an extended degenerate set (lines,
        a flat band).
    extended_degeneracy : bool
    mesh : (int, int)
    diagnostics : dict
    """

    lambda_min: float
    minima: List[LTWaveVector]
    near_minimal_fraction: float
    extended_degeneracy: bool
    mesh: Tuple[int, int]
    diagnostics: Dict[str, Any]

    @property
    def strong_constraint(self) -> bool:
        """Some minimum admits a single-q state with unit spins (the LT bound is reached)."""
        return any(m.strong_constraint for m in self.minima)


def lt_matrix(model: SpinModel, q) -> np.ndarray:
    """``J(q)`` (``(m, 3Ns, 3Ns)``) for Cartesian momenta ``q`` ``(m, 2)``; see the module docstring."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    index = {site_id: i for i, site_id in enumerate(model.site_ids)}
    spins = {s.id: s.spin for s in model.sites}
    ns = model.num_sites
    M = np.zeros((len(q), 3 * ns, 3 * ns), dtype=complex)
    for term in model.terms_of_kind(BILINEAR):
        (a, n1), (b, n2) = term.participants
        d = model.cartesian_position(b, n2) - model.cartesian_position(a, n1)
        block = spins[a] * spins[b] * np.asarray(term.coefficient, dtype=float)
        phase = np.exp(1j * q @ d)
        i, j = 3 * index[a], 3 * index[b]
        M[:, i:i + 3, j:j + 3] += phase[:, None, None] * block[None]
    return 0.5 * (M + np.conj(np.transpose(M, (0, 2, 1))))


def _lowest(model, q):
    return np.linalg.eigvalsh(lt_matrix(model, q))[:, 0]


def _wrap_fraction(f):
    f = np.mod(np.asarray(f, dtype=float), 1.0)
    f[np.isclose(f, 1.0, atol=1e-9)] = 0.0
    return f


def _same_star(f1, f2, tol=1e-6):
    """q1 = +-q2 modulo reciprocal lattice vectors."""
    for sign in (1, -1):
        d = np.mod(f1 - sign * f2 + 0.5, 1.0) - 0.5
        if np.all(np.abs(d) < tol):
            return True
    return False


def _commensurate(f, max_denominator):
    fractions = [Fraction(float(x)).limit_denominator(max_denominator) for x in f]
    if not all(abs(float(fr) - x) < 1e-7 for fr, x in zip(fractions, f)):
        return None
    return fractions


def _supercell(fractions) -> np.ndarray:
    """Reduced basis of {R in Z^2 : f . R in Z}."""
    Q = int(np.lcm.reduce([fr.denominator for fr in fractions]))
    a = np.array([int(fr * Q) for fr in fractions])
    index = Q // int(np.gcd.reduce([a[0], a[1], Q]))
    box = range(-Q, Q + 1)
    vectors = [np.array([i, j]) for i in box for j in box
               if (i, j) != (0, 0) and (a[0] * i + a[1] * j) % Q == 0]
    vectors.sort(key=lambda v: (v @ v, -v[0], -v[1]))
    v1 = vectors[0]
    for v2 in vectors[1:]:
        det = int(round(v1[0] * v2[1] - v1[1] * v2[0]))
        if abs(det) == index:
            cell = np.array([v1, v2])
            return cell if det > 0 else np.array([v2, v1])
    raise RuntimeError("no supercell basis found")          # pragma: no cover


def _phase_case(fraction: np.ndarray) -> str:
    """Values of ``exp(2 i q* . R)``: "real" (2q* in G), "quarter" (4q* in G), "generic"."""
    def integer(x):
        return bool(np.all(np.abs(x - np.rint(x)) < 1e-9))
    if integer(2 * fraction):
        return "real"
    if integer(4 * fraction):
        return "quarter"
    return "generic"


def _strong_fit(space: np.ndarray, phases: np.ndarray, case: str,
                rng) -> Tuple[float, np.ndarray]:
    """Best cell amplitude ``u`` (Ns x 3) from the eigenspace ``space`` (3Ns x d).

    ``phases`` are ``exp(i q* . r_a)``; ``u_a = phases_a (space @ c)_a``;
    ``case`` from :func:`_phase_case` selects the strong-constraint residual.
    """
    ns = space.shape[0] // 3
    d = space.shape[1]

    def amplitude(x):
        c = x[:d] + 1j * x[d:]
        return phases[:, None] * (space @ c).reshape(ns, 3)

    def residual(x):
        u = amplitude(x)
        if case == "real":
            v = np.real(u)
            return float(np.sum((np.sum(v * v, axis=1) - 1.0) ** 2))
        dot = np.sum(u * u, axis=1)
        norm = np.sum(np.abs(u) ** 2, axis=1)
        mismatch = np.real(dot) if case == "quarter" else dot
        return float(np.sum(np.abs(mismatch) ** 2) + np.sum((norm - 2.0) ** 2))

    best = (np.inf, None)
    for _ in range(16):
        x0 = rng.standard_normal(2 * d) * np.sqrt(ns / d)
        res = minimize(residual, x0, method="BFGS", options={"gtol": 1e-14, "maxiter": 4000})
        if res.fun < best[0]:
            best = (float(res.fun), amplitude(res.x))
        if best[0] < LT_STRONG_TOLERANCE:
            break
    return best[0], best[1]


def luttinger_tisza(model: SpinModel, mesh: Tuple[int, int] = (48, 48),
                    tolerance: float = LT_TOLERANCE,
                    max_denominator: int = LT_MAX_DENOMINATOR, seed: int = 0) -> LTReport:
    """Luttinger-Tisza diagnostic of the bilinear terms at zero field.

    Parameters
    ----------
    model : SpinModel
    mesh : (int, int)
        Zone-centred mesh of the primitive reciprocal cell for the search;
        the minima are then refined continuously.
    tolerance : float
        Relative eigenvalue tolerance (minima, eigenspace, degeneracy).
    max_denominator : int
        Largest denominator accepted as commensurate.
    seed : int
        Random starts of the strong-constraint fit.

    Returns
    -------
    LTReport
    """
    rng = np.random.default_rng(seed)
    reciprocal = 2 * np.pi * np.linalg.inv(model.lattice).T
    n1, n2 = (int(n) for n in mesh)
    grid = np.array([(i / n1, j / n2) for i in range(n1) for j in range(n2)])
    values = _lowest(model, grid @ reciprocal)
    scale = max(float(np.max(np.abs(np.linalg.eigvalsh(lt_matrix(model, np.zeros(2)))))), 1e-300)
    floor = float(values.min())
    candidates = grid[np.argsort(values)[:16]]

    refined = []
    for f in candidates:
        res = minimize(lambda x: float(_lowest(model, (x @ reciprocal)[None])[0]), f,
                       method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-15, "maxiter": 4000})
        refined.append((float(res.fun), _wrap_fraction(res.x)))
    lambda_min = min(floor, min(v for v, _ in refined))
    threshold = lambda_min + tolerance * scale
    distinct: List[np.ndarray] = []
    for value, f in sorted(refined, key=lambda t: t[0]):
        if value > threshold:
            continue
        snapped = _commensurate(f, max_denominator)
        if snapped is not None:
            f = _wrap_fraction(np.array([float(x) for x in snapped]))
        if not any(_same_star(f, g) for g in distinct):
            distinct.append(f)
    # q and -q are one minimum; report the lexicographically smaller representative so the
    # choice does not depend on round-off in the ordering of degenerate eigenvalues.
    distinct = [min(f, _wrap_fraction(-f), key=tuple) for f in distinct]
    near_fraction = float(np.mean(values <= threshold))
    extended = near_fraction * n1 * n2 > 2 * max(1, len(distinct)) + 2

    minima = []
    for f in distinct:
        k = f @ reciprocal
        w_all, v_all = np.linalg.eigh(lt_matrix(model, k)[0])
        space = v_all[:, w_all <= w_all[0] + tolerance * scale]
        offsets = np.array([model.cartesian_position(site_id) for site_id in model.site_ids])
        residual, u = _strong_fit(space, np.exp(1j * offsets @ k), _phase_case(f), rng)
        strong = residual < LT_STRONG_TOLERANCE
        fractions = _commensurate(f, max_denominator)
        cell = _supercell(fractions) if fractions is not None else None
        state = energy = None
        if strong and cell is not None:
            state = _state_from_amplitude(model, cell, k, u)
            from spintoolkit.methods.classical import classical_energy
            energy = float(classical_energy(model, state, None))
        minima.append(LTWaveVector(
            f, k, int(space.shape[1]), fractions is not None,
            None if fractions is None else tuple(str(fr) for fr in fractions),
            cell, bool(strong), float(residual), state, energy, u))
    diagnostics = {"scale": scale, "tolerance": tolerance, "max_denominator": max_denominator,
                   "zeeman": "ignored (zero field)",
                   "single_q_only": "multi-q states and generalized LT are not treated"}
    return LTReport(float(lambda_min), minima, near_fraction, bool(extended), (n1, n2), diagnostics)


def _state_from_amplitude(model, cell, k, u) -> SpinState:
    """Unit spins ``Re[u_a exp(i k . R)]`` on ``cell`` (``R`` the cell's lattice vector)."""
    index = {site_id: i for i, site_id in enumerate(model.site_ids)}

    def direction(site, cell_offset):
        R = np.asarray(cell_offset, dtype=float) @ model.lattice
        v = np.real(u[index[site]] * np.exp(1j * k @ R))
        return v / np.linalg.norm(v)

    return SpinState.from_function(model, cell, direction,
                                   {"origin": "luttinger_tisza", "q": k.tolist()})
