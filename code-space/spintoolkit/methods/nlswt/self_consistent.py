"""Self-consistent one-loop spin waves: ``solve_scswt(model, state, ...)`` (D72).

The one-loop (Hartree-Fock) correction of :mod:`spintoolkit.methods.nlswt.engine`
evaluates the normal-ordered quartic terms in the LSWT vacuum. The
self-consistent one-loop scheme (SCOL; Keselman, Xu, Zhang, Batista and
Starykh, arXiv:2512.02150, End Matter and SM Sec. III) feeds the decoupling
back into the quadratic Hamiltonian and recomputes the two-point functions in
its own vacuum until they no longer change::

    H_SC(k) = H_2(k) + dH_HF[<a^+ a>, <a a>]_SC,
    <a^+ a>, <a a> from the Bogoliubov vacuum of H_SC.

The fixed point is a stationary point of the Gaussian energy
``<H_0 + H_2 + H_4>`` over Bogoliubov vacua (zero boson mean), because
``dH_HF`` is the derivative of ``<H_4>`` with respect to the two-point
functions. That energy is the ground-state energy reported here; it agrees
with the order-S^0 energy of :func:`~spintoolkit.methods.nlswt.solve_nlswt`
up to order ``1/S``. Magnon energies are the Bogoliubov energies of
``H_SC(k)``; they agree with the one-loop energies at relative order ``1/S``
and differ at relative order ``1/S^2``.

Scope: states without three-boson vertices (``H_3 = 0``), such as collinear
states along a symmetry axis (the up-up-down plateau, collinear
antiferromagnets, polarized states along a principal axis). With ``H_3 != 0``
the cubic self-energy and the tadpole enter magnon energies at the same order
as the Hartree-Fock term, and a Gaussian self-consistency that leaves them out
is not a consistent approximation; :func:`solve_scswt` refuses such states.

Cubic vertices in mean field (``SCSWTSettings(cubic="mean_field")``, D73;
Bader, Feng, Budaraju, Becca, Knolle and Pollmann, arXiv:2512.08768, SM
Sec. S3): the spin directions become variables. The Wick contraction of
``H_3`` in the Gaussian vacuum leaves linear terms; together with ``H_1`` of
the tilted directions they are the force on a boson condensate, and the
directions are rotated until that force vanishes::

    H_1(theta) + <H_3(theta)>_linear = 0,
    H_SC(k) = H_2(theta) + dH_HF[H_4(theta)],  both self-consistent.

This is the Hartree-Fock decoupling of ``H_3`` (its tadpole), with the
condensate absorbed into the directions so that ``<a> = 0`` in the rotated
frames. The energy ``<H_0 + H_2 + H_4>(theta)`` contains the tadpole energy
of :func:`solve_nlswt`. It leaves out the second-order process of the
three-magnon vertex (``E_cubic`` of :func:`solve_nlswt`, and ``Sigma_3`` in
the magnon energies), which is of the same order as the Hartree-Fock term; a
Gaussian state cannot hold three-magnon correlations. That energy, evaluated
in the self-consistent vacuum, is reported separately as
``energies.three_magnon`` and is not part of ``total``. Every order of
``H(theta)`` is expanded at the current directions (Bader et al. keep ``H_4``
at the classical directions; the difference is of higher order).

Field scans with frozen two-point functions (the SCOL plateau boundaries of
arXiv:2512.02150): pass ``frozen=`` a result at a reference field. The
quartic terms do not depend on the field, so ``dH_HF`` of the reference is
added to ``H_2`` at the new field without iterating.

Energies are per site in the energy unit E0 of the model (D20).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.methods.lswt.run import LSWTError, _diagonalize
from spintoolkit.methods.nlswt.engine import (Contractions, bogoliubov_mesh, hartree_fock_terms,
                                              nambu_matrices, wick4)
from spintoolkit.methods.nlswt.expansion import Monomial, expand_model
from spintoolkit.methods.nlswt.run import _momenta, _resolution
from spintoolkit.methods.result import ResultHeader, freeze_fields, state_fingerprint, to_jsonable
from spintoolkit.states.spin_state import SpinState, validate_spin_state
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import SpinModel


@dataclass(frozen=True)
class SCSWTSettings:
    """Method settings of :func:`solve_scswt`.

    Parameters
    ----------
    mesh : (int, int)
        Thermodynamic limit: ``N1 x N2`` mesh of the magnetic reciprocal cell
        offset by one third of a step (as :class:`NLSWTSettings`; it avoids
        the zone centre).
    tolerance : float
        Convergence: the largest change of a Hartree-Fock coefficient in one
        iteration, relative to the largest quadratic coefficient.
    max_iterations : int
    mixing : float
        Linear mixing ``c <- (1 - mixing) c + mixing c_new``, in ``(0, 1]``.
    zero_mode_tolerance : float
        ``min eig H(k) <= zero_mode_tolerance * max |eig H(k)|`` on the mesh
        raises (zero mode or instability).
    cubic_tolerance : float
        Three-boson coefficients up to this fraction of the largest quadratic
        coefficient count as zero.
    cubic : {"refuse", "mean_field"}
        States with three-boson vertices: ``"refuse"`` raises (D72);
        ``"mean_field"`` renormalizes the spin directions with the contracted
        cubic terms (D73, see the module docstring).
    max_tilt : float
        ``cubic="mean_field"``: largest rotation of a spin direction in one
        iteration (radians); larger Newton steps are scaled down.
    """

    mesh: Tuple[int, int] = (24, 24)
    tolerance: float = 1e-10
    max_iterations: int = 200
    mixing: float = 1.0
    zero_mode_tolerance: float = 1e-9
    cubic_tolerance: float = 1e-10
    cubic: str = "refuse"
    max_tilt: float = 0.2

    def __post_init__(self):
        mesh = tuple(int(n) for n in self.mesh)
        if len(mesh) != 2 or min(mesh) < 1:
            raise ValueError("mesh must be two positive integers")
        if not 0 < self.mixing <= 1:
            raise ValueError("mixing must be in (0, 1]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.cubic not in ("refuse", "mean_field"):
            raise ValueError('cubic must be "refuse" or "mean_field"')
        if not self.max_tilt > 0:
            raise ValueError("max_tilt must be positive")
        object.__setattr__(self, "mesh", mesh)

    def as_dict(self) -> Dict[str, Any]:
        return {"mesh": list(self.mesh), "mesh_offset": "1/3 step", "tolerance": self.tolerance,
                "max_iterations": self.max_iterations, "mixing": self.mixing,
                "zero_mode_tolerance": self.zero_mode_tolerance,
                "cubic_tolerance": self.cubic_tolerance, "cubic": self.cubic,
                "max_tilt": self.max_tilt}


@dataclass(frozen=True)
class SelfConsistentEnergies:
    """Gaussian ground-state energy per site (E0 units, D20).

    Attributes
    ----------
    classical : float
        ``H_0``.
    quadratic : float
        ``<H_2>`` in the self-consistent vacuum (the LSWT zero-point energy
        when evaluated in the LSWT vacuum).
    quartic : float
        ``<H_4>`` in the self-consistent vacuum.
    one_loop : float
        The energy through order S^0 of :func:`solve_nlswt` about the
        classical state (``H_0 + E_zp + <H_4>_LSWT`` without cubic vertices);
        NaN where the LSWT vacuum does not exist.
    three_magnon : float
        ``cubic="mean_field"``: the second-order energy of the three-magnon
        vertex (``E_cubic`` of :func:`solve_nlswt`) in the self-consistent
        vacuum, the part of order S^0 that the Gaussian scheme leaves out; not
        included in ``total``. Zero without cubic vertices.
    """

    classical: float
    quadratic: float
    quartic: float
    one_loop: float
    three_magnon: float = 0.0

    @property
    def total(self) -> float:
        return self.classical + self.quadratic + self.quartic


def _merge(terms: List[Tuple[complex, Monomial]]) -> Tuple[List[Monomial], np.ndarray]:
    """Sum coefficients of equal monomials; the monomial order is that of first appearance."""
    index: Dict[Monomial, int] = {}
    values: List[complex] = []
    for c, m in terms:
        if m in index:
            values[index[m]] += c
        else:
            index[m] = len(values)
            values.append(complex(c))
    return list(index), np.array(values, dtype=complex)


@dataclass(frozen=True, eq=False)
class SCSWTResult:
    """Self-consistent one-loop spin waves with the common header.

    Attributes
    ----------
    header : ResultHeader
    energies : SelfConsistentEnergies
        Per site; ``total`` is the Gaussian energy of the fixed point.
    k_points : (nk, 2) array
        Momenta of the two-point functions.
    spins : (Ns,) array
    boson_numbers : (Ns,) array
        ``<a_s^+ a_s>`` of every magnetic site in the self-consistent vacuum.
    lattice : (2, 2) array
    state : SpinState
        Directions of the self-consistent vacuum: the classical state, or the
        renormalized directions with ``cubic="mean_field"``.
    """

    header: ResultHeader
    energies: SelfConsistentEnergies
    k_points: np.ndarray
    spins: np.ndarray
    boson_numbers: np.ndarray
    lattice: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    state: Optional[SpinState] = field(default=None, repr=False, compare=False)
    _quadratic: Any = field(default=None, repr=False, compare=False)
    _monomials: Tuple = field(default=(), repr=False, compare=False)
    _coefficients: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _quartic: Tuple = field(default=(), repr=False, compare=False)
    _contract: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        freeze_fields(self)

    @property
    def ground_state_energy(self) -> float:
        """``<H_0 + H_2 + H_4>`` per site in the self-consistent vacuum."""
        return self.energies.total

    def hamiltonian(self, k_points) -> np.ndarray:
        """``H_SC(k) = H_2(k) + dH_HF(k)`` (Nambu), shape ``(m, 2Ns, 2Ns)``."""
        k = np.atleast_2d(np.asarray(k_points, dtype=float))
        terms = list(zip(self._coefficients, self._monomials))
        return np.asarray(self._quadratic.at(k), dtype=complex) + nambu_matrices(terms, k,
                                                                                 len(self.spins))

    def correction(self, k_points) -> np.ndarray:
        """``dH_HF(k)`` alone (Nambu)."""
        k = np.atleast_2d(np.asarray(k_points, dtype=float))
        return nambu_matrices(list(zip(self._coefficients, self._monomials)), k, len(self.spins))

    def magnon_energies(self, k_points) -> np.ndarray:
        """Bogoliubov energies of ``H_SC(k)``, ``(m, Ns)`` ascending.

        NaN (with a warning) at momenta where ``H_SC(k)`` is not positive
        definite: a zero mode or an instability of the state at this field,
        where magnon energies are not defined. NaN everywhere (with a warning)
        for a ``cubic="mean_field"`` result whose Goldstone (or accidental)
        modes the scheme gaps spuriously (``diagnostics["spurious_gap"]``,
        D73).
        """
        k = np.atleast_2d(np.asarray(k_points, dtype=float))
        if self.header.diagnostics.get("spurious_gap") is not None:
            warnings.warn("magnon_energies: the Gaussian scheme without Sigma_3 gaps the zero "
                          "modes of this state (spurious gap "
                          f"{self.header.diagnostics['spurious_gap']:.4g}); the magnon energies "
                          "are NaN (D73). Use solve_nlswt for the spectrum.",
                          UserWarning, stacklevel=2)
            return np.full((len(k), len(self.spins)), np.nan)
        H = self.hamiltonian(k)
        ns = len(self.spins)
        out = np.full((len(k), ns), np.nan)
        bad = 0
        for i in range(len(k)):
            try:
                _, E, _, _, _ = _diagonalize(H[i:i + 1], "none", k[i:i + 1],
                                             self.header.settings["zero_mode_tolerance"])
            except LSWTError:
                bad += 1
                continue
            out[i] = np.sort(E[0, :ns].real)
        if bad:
            warnings.warn(f"magnon_energies: H_SC(k) is not positive definite at {bad} of {len(k)} "
                          "momenta (zero mode or instability); the energies there are NaN",
                          UserWarning, stacklevel=2)
        return out

    def ordered_moments(self) -> np.ndarray:
        """``S_s - <n_s>`` along the classical direction; NaN where ``<n_s> > S_s``."""
        moments = self.spins - self.boson_numbers
        return np.where(moments < 0, np.nan, moments)

    def to_json_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        e = self.energies
        body = {"classical": e.classical, "quadratic": e.quadratic, "quartic": e.quartic,
                "total": e.total, "one_loop": e.one_loop, "three_magnon": e.three_magnon,
                "spins": self.spins,
                "boson_numbers": self.boson_numbers, "k_points": self.k_points,
                "lattice": self.lattice}
        return to_jsonable({"header": self.header, "scswt": body})


def _same_quartic(a, b, tolerance: float = 1e-12) -> bool:
    if len(a) != len(b):
        return False
    da = {m: c for c, m in a}
    return all(m in da and abs(da[m] - c) <= tolerance * max(1.0, abs(c)) for c, m in b)


def solve_scswt(model: SpinModel, state: SpinState,
                conditions: Optional[ExternalConditions] = None,
                geometry: Optional[CalculationGeometry] = None,
                settings: SCSWTSettings = SCSWTSettings(),
                frozen: Optional[SCSWTResult] = None,
                initial: Optional[SCSWTResult] = None) -> SCSWTResult:
    """Self-consistent one-loop spin waves about a classical stationary state (D72).

    Parameters
    ----------
    model : SpinModel
        Bilinear and Zeeman terms.
    state : SpinState
        Classical stationary state without three-boson vertices.
    conditions : ExternalConditions, optional
        Field; temperature must be zero.
    geometry : CalculationGeometry, optional
        Thermodynamic limit (mesh of ``settings``) or a finite torus.
    settings : SCSWTSettings
    frozen : SCSWTResult, optional
        Use the Hartree-Fock correction of this result (same model and state,
        another field) without iterating. Where ``H_SC(k)`` is not positive
        definite on the momenta (beyond a gap closing), the energies and boson
        numbers are NaN with a warning.
    initial : SCSWTResult, optional
        Start the iteration from the correction of this result (same model and
        state) instead of the LSWT vacuum; needed where the LSWT vacuum does
        not exist, such as a plateau state away from its classical field.
        With ``cubic="mean_field"``: a mean-field result of the same model and
        magnetic cell; the iteration starts from its directions and vacuum.

    Returns
    -------
    SCSWTResult

    Raises
    ------
    LSWTError
        Cubic vertices with ``cubic="refuse"`` (the scheme is not defined
        there), a non-stationary state (except with ``cubic="mean_field"``),
        a zero or negative mode on the momenta in any iteration, a cubic force
        along a zero mode, or no convergence.

    Warns
    -----
    UserWarning
        ``cubic="mean_field"`` on a state with zero modes at k = 0 and cubic
        vertices: the scheme gaps them, and ``magnon_energies`` is NaN (D73).
    """
    conditions = conditions or ExternalConditions()
    if frozen is not None and initial is not None:
        raise ValueError("give frozen= or initial=, not both")
    if conditions.temperature > 0:
        raise ValueError("solve_scswt is a zero-temperature method")
    geometry = geometry or CalculationGeometry.thermodynamic_limit()
    validate_spin_state(state, model, geometry)
    expansion = expand_model(model, state, conditions)
    scale = max(1.0, max((abs(c) for c, _ in expansion.orders[2]), default=1.0))
    linear = max((abs(c) for c, _ in expansion.orders[1]), default=0.0)
    cubic = max((abs(c) for c, _ in expansion.orders[3]), default=0.0)
    if settings.cubic == "mean_field":
        return _solve_mean_field(model, state, conditions, geometry, settings, initial, frozen,
                                 expansion, scale, cubic, linear)
    if linear > 1e-8 * scale:
        raise LSWTError(f"the reference state is not stationary (linear boson term {linear:.3g})")
    if cubic > settings.cubic_tolerance * scale:
        raise LSWTError(
            f"the state has three-boson vertices (largest {cubic:.3g}): the cubic self-energy "
            "and the tadpole then shift magnon energies at the same order as the Hartree-Fock "
            "term, and the self-consistent one-loop scheme, which keeps only the latter, is "
            "not defined (D72). Options: solve_nlswt (one loop with Sigma_3 and the tadpole), "
            'SCSWTSettings(cubic="mean_field") (directions renormalized by the contracted '
            "cubic terms, without Sigma_3; D73), or a state without cubic vertices.")
    k, grid, reciprocal = _momenta(model, state, geometry, settings)
    resolution = _resolution(np.mod(grid, 1.0))
    quadratic = QuadraticBoseHamiltonian(model, state, conditions)
    ns = expansion.num_sites
    quartic = expansion.orders[4]

    def per_site(contract, terms, expectation):
        return float(np.real(sum(c * expectation(contract, m) for c, m in terms))) / ns

    def at_sc(monomials, coefficients, kk):
        H = np.asarray(quadratic.at(kk), dtype=complex)
        if len(monomials):
            H = H + nambu_matrices(list(zip(coefficients, monomials)), kk, ns)
        return H

    def vacuum(monomials, coefficients):
        return Contractions(bogoliubov_mesh(lambda kk: at_sc(monomials, coefficients, kk), k,
                                            grid, reciprocal, settings.zero_mode_tolerance,
                                            resolution))

    pair = lambda contract, m: contract(m[0], m[1])  # noqa: E731
    classical = expansion.classical_energy / ns
    try:
        lswt = vacuum([], [])
    except LSWTError as exc:            # e.g. a plateau state away from its classical field
        lswt, missing = None, exc
    one_loop = float("nan")
    if lswt is not None:
        monomials, coefficients = _merge(hartree_fock_terms(quartic, lswt))
        one_loop = (classical + per_site(lswt, expansion.orders[2], pair)
                    + per_site(lswt, quartic, wick4))
    start = frozen if frozen is not None else initial
    if start is not None:
        if (start.header.model_ref != model.fingerprint()
                or start.header.state_ref != state_fingerprint(state)
                or not _same_quartic(start._quartic, quartic)):
            raise ValueError("frozen= and initial= need a result of the same model and state "
                             "(only the field may differ)")
        if lswt is not None and list(start._monomials) != monomials:
            raise ValueError("the reference result has a different Hartree-Fock structure")
        monomials, coefficients = list(start._monomials), np.array(start._coefficients)
    elif lswt is None:
        raise LSWTError(f"the LSWT vacuum does not exist ({missing}); start the iteration from "
                        "a self-consistent result of the same state at another field "
                        "(initial= or frozen=)")

    history: List[float] = []
    if frozen is not None:
        mode, iterations, converged = "frozen", 0, True
        try:
            contract = vacuum(monomials, coefficients)
        except LSWTError as exc:        # beyond a gap closing: no vacuum, energies undefined
            contract = None
            warnings.warn(f"solve_scswt(frozen=): H_SC(k) is not positive definite on the "
                          f"momenta ({str(exc).split('. A zero mode')[0]}); the state is "
                          "unstable at this field, and the energies and boson numbers are NaN. "
                          "magnon_energies stays available (NaN where H_SC(k) is not positive "
                          "definite).", UserWarning, stacklevel=2)
    else:
        mode, converged = "self_consistent", False
        iterations = 0
        for iterations in range(1, settings.max_iterations + 1):
            try:
                contract = vacuum(monomials, coefficients)
            except LSWTError as exc:
                raise LSWTError(f"self-consistent iteration {iterations}: {exc} The iteration "
                                "has no stable fixed point from this start (the state may be "
                                "unstable at this S and field); try mixing < 1") from exc
            new = _merge(hartree_fock_terms(quartic, contract))[1]
            change = float(np.max(np.abs(new - coefficients), initial=0.0))
            history.append(change)
            coefficients = (1 - settings.mixing) * coefficients + settings.mixing * new
            if change <= settings.tolerance * scale:
                converged = True
                break
        if not converged:
            raise LSWTError(f"self-consistency did not converge in {settings.max_iterations} "
                            f"iterations (last change {history[-1]:.3g}); try mixing < 1")
        contract = vacuum(monomials, coefficients)

    if contract is not None and geometry.kind != "finite_torus":
        # The mesh is offset by 1/3 step and misses k = 0 and the zone corners, where a
        # gap usually closes first; a vacuum needs H_SC(k) > 0 at every k (D75).
        found = _lowest_between_mesh(lambda kk: at_sc(monomials, coefficients, kk), k,
                                     reciprocal)
        if found is not None:
            value, where = found
            message = (f"H_SC(k) has a negative eigenvalue {value:.3g} at fractional "
                       f"k = {np.round(where, 6).tolist()} between the mesh points: the state "
                       "has no vacuum at this field")
            if frozen is None:
                raise LSWTError(message + " (D75)")
            warnings.warn(f"solve_scswt(frozen=): {message}; the energies and boson numbers are "
                          "NaN (D75)", UserWarning, stacklevel=2)
            contract = None
    if contract is None:
        energies = SelfConsistentEnergies(classical, float("nan"), float("nan"), one_loop)
        numbers = np.full(ns, np.nan)
    else:
        energies = SelfConsistentEnergies(classical, per_site(contract, expansion.orders[2], pair),
                                          per_site(contract, quartic, wick4), one_loop)
        numbers = np.array([contract((s, True, (0.0, 0.0)), (s, False, (0.0, 0.0))).real
                            for s in range(ns)])
    diagnostics = {"num_k": len(k), "mode": mode, "iterations": iterations,
                   "converged": converged, "changes": history[-5:],
                   "max_cubic_coefficient": cubic,
                   "momenta": ("finite torus" if geometry.kind == "finite_torus" else
                               f"{settings.mesh[0]}x{settings.mesh[1]} mesh offset by 1/3 step")}
    header = ResultHeader.build("scswt", model, state, geometry, conditions, settings.as_dict(),
                                "energies per site of the model (E0)", diagnostics)
    return SCSWTResult(header, energies, k, np.asarray(expansion.spins, dtype=float), numbers,
                       np.asarray(model.lattice, dtype=float), state, quadratic, tuple(monomials),
                       coefficients, tuple(quartic), contract)


def _lowest_between_mesh(at, k: np.ndarray, reciprocal: np.ndarray, seeds: int = 6,
                         tolerance: float = 1e-10) -> Optional[Tuple[float, np.ndarray]]:
    """A negative eigenvalue of ``H(k)`` off the mesh, or None.

    ``lambda_min`` is evaluated at the special points of the magnetic cell
    (fractions 0, 1/2, 1/3, 2/3) and minimized over continuous ``k`` from
    the lowest mesh points. Returns ``(lambda_min, fractional k)`` when it is
    below ``-tolerance`` times the largest ``|eig H|`` on the mesh.
    """
    from scipy.optimize import minimize

    eigs = np.linalg.eigvalsh(at(k))
    scale = float(np.max(np.abs(eigs)))
    lowest = lambda p: float(np.linalg.eigvalsh(at(np.atleast_2d(p) @ reciprocal))[0][0])  # noqa: E731
    special = (0.0, 0.5, 1 / 3, 2 / 3)
    starts = [np.array([a, b]) for a in special for b in special]
    fractions = k @ np.linalg.inv(reciprocal)
    starts += [fractions[i] for i in np.argsort(eigs[:, 0])[:seeds]]
    step = 1.0 / max(4, int(np.sqrt(len(k))))
    best, where = np.inf, None
    for p0 in starts:
        value = lowest(p0)
        if value > -tolerance * scale and value < 1e-2 * scale:
            simplex = np.array([p0, p0 + [step, 0], p0 + [0, step]])
            opt = minimize(lowest, p0, method="Nelder-Mead",
                           options={"initial_simplex": simplex, "xatol": 1e-10,
                                    "fatol": 1e-16 * scale, "maxiter": 300})
            if opt.fun < value:
                value, p0 = float(opt.fun), opt.x
        if value < best:
            best, where = value, np.mod(p0, 1.0)
    if best < -tolerance * scale:
        return best, where
    return None


class _TermsHamiltonian:
    """``H(k)`` (Nambu) of a list of quadratic monomials."""

    def __init__(self, terms, num_sites: int):
        self.terms = list(terms)
        self.num_sites = num_sites

    def at(self, k_points) -> np.ndarray:
        return nambu_matrices(self.terms, np.atleast_2d(k_points), self.num_sites)


def _linear_force(expansion, contract) -> np.ndarray:
    """``zeta = (mu, lambda)``: coefficients of ``a_s^+`` and ``a_s`` in ``H_1 + <H_3>_linear``.

    Per magnetic cell, in the convention ``1/2 (zeta^+ psi + psi^+ zeta)`` of
    :meth:`NonlinearSpinWaves.linear_terms`.
    """
    ns = expansion.num_sites
    mu = np.zeros(ns, dtype=complex)
    lam = np.zeros(ns, dtype=complex)
    for c, ((s, dagger, _),) in expansion.orders[1]:
        if dagger:
            mu[s] += c
        else:
            lam[s] += c
    for c, m in expansion.orders[3]:
        for i, j, x in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
            value = c * contract(m[i], m[j])
            s, dagger, _ = m[x]
            if dagger:
                mu[s] += value
            else:
                lam[s] += value
    return np.r_[mu, lam]


def _condensate(H0: np.ndarray, zeta: np.ndarray, scale: float) -> np.ndarray:
    """Newton step ``<psi> = -H(0)^+ zeta``; raises for a force along a zero mode."""
    w, V = np.linalg.eigh(H0)
    kernel = np.abs(w) <= 1e-9 * max(np.max(np.abs(w)), 1.0)
    projection = np.linalg.norm(V[:, kernel].conj().T @ zeta)
    if projection > 1e-8 * scale:
        raise LSWTError(
            "the contracted cubic force has a component along a zero mode of H_SC(k=0) "
            f"(|projection| = {projection:.3g}): quantum fluctuations push the state along a "
            "flat direction (order by disorder), and the directions have no stationary "
            "point near this state; start from the state the fluctuations select")
    inverse = (V[:, ~kernel] / w[~kernel]) @ V[:, ~kernel].conj().T
    return -inverse @ zeta


def _tilt(directions, frames, spins, shift, mixing, max_tilt):
    """Rotate every direction toward its condensate ``<s^+> = sqrt(2S) <a>``; largest angle."""
    ns = len(spins)
    phi = mixing * shift[:ns]
    local = np.stack([np.sqrt(2 * spins) * phi.real, np.sqrt(2 * spins) * phi.imag,
                      spins.astype(float)], axis=1)
    angles = np.arctan2(np.hypot(local[:, 0], local[:, 1]), local[:, 2])
    largest = float(np.max(angles, initial=0.0))
    if largest > max_tilt:
        local[:, :2] *= np.tan(max_tilt) / np.tan(largest)
    new = np.einsum("sab,sb->sa", frames, local)
    return new / np.linalg.norm(new, axis=1)[:, None], largest


def _solve_mean_field(model, state, conditions, geometry, settings, initial, frozen,
                      expansion, scale, cubic, linear):
    """``solve_scswt`` with ``cubic="mean_field"`` (D73).

    The start ``state`` need not be classically stationary (``H_1`` is part of
    the force); ``one_loop`` is then NaN.
    """
    if frozen is not None:
        raise ValueError('frozen= is not available with cubic="mean_field": the directions '
                         "depend on the field; use initial=")
    from spintoolkit.methods.nlswt.engine import NonlinearSpinWaves

    k, grid, reciprocal = _momenta(model, state, geometry, settings)
    resolution = _resolution(np.mod(grid, 1.0))
    ns = expansion.num_sites
    keys = tuple((site.id, cell) for site in model.sites for cell in state.cells)
    pair = lambda contract, m: contract(m[0], m[1])  # noqa: E731

    def per_site(contract, terms, expectation):
        return float(np.real(sum(c * expectation(contract, m) for c, m in terms))) / ns

    def mesh_of(terms):
        return bogoliubov_mesh(_TermsHamiltonian(terms, ns).at, k, grid, reciprocal,
                               settings.zero_mode_tolerance, resolution)

    def make_state(directions):
        return SpinState(model.fingerprint(), state.supercell,
                         {key: directions[i] for i, key in enumerate(keys)},
                         {"origin": "scswt_mean_field", "classical_state": state_fingerprint(state)})

    one_loop = float("nan")
    try:
        lswt_mesh = mesh_of(expansion.orders[2])
    except LSWTError as exc:
        lswt_mesh, missing = None, exc
    if lswt_mesh is not None:
        try:
            one_loop = NonlinearSpinWaves(expansion, lswt_mesh, _TermsHamiltonian(
                expansion.orders[2], ns).at).energies().total
        except LSWTError:             # tadpole along a zero mode: the one-loop energy is undefined
            one_loop = float("nan")
    # a zero mode of the classical H(k = 0) (Goldstone or accidental) with cubic vertices: at
    # one loop the Goldstone theorem holds through Hartree-Fock + tadpole + Sigma_3(0, 0) = 0,
    # and the Gaussian scheme, which has no Sigma_3, gaps the mode (Hohenberg-Martin)
    w0 = np.linalg.eigvalsh(_TermsHamiltonian(expansion.orders[2], ns).at(np.zeros((1, 2)))[0])
    zero_modes = int(np.sum(np.abs(w0) <= 1e-7 * max(np.max(np.abs(w0)), 1.0)))
    gapped = zero_modes > 0 and cubic > settings.cubic_tolerance * scale
    directions = np.array([state.direction(s, c) for s, c in keys], dtype=float)
    if initial is not None:
        if (initial.header.model_ref != model.fingerprint()
                or initial._contract is None or initial.state is None
                or not np.array_equal(initial.state.supercell, state.supercell)
                or initial.k_points.shape != k.shape or not np.allclose(initial.k_points, k)):
            raise ValueError("initial= needs a mean-field result of the same model, magnetic "
                             "cell and momenta (the field and the start state may differ)")
        contract = initial._contract
        directions = np.array([initial.state.direction(s, c) for s, c in keys], dtype=float)
    elif lswt_mesh is None:
        raise LSWTError(f"the LSWT vacuum does not exist ({missing}); start the iteration from "
                        "a self-consistent result of the same state at another field (initial=)")
    else:
        contract = Contractions(lswt_mesh)

    history: List[Tuple[float, float, float]] = []
    previous = None
    converged = False
    iterations = 0
    for iterations in range(1, settings.max_iterations + 1):
        current = expand_model(model, make_state(directions), conditions)
        hf = _merge(hartree_fock_terms(current.orders[4], contract))
        terms = list(current.orders[2]) + list(zip(hf[1], hf[0]))
        try:
            contract = Contractions(mesh_of(terms))
        except LSWTError as exc:
            raise LSWTError(f"self-consistent iteration {iterations}: {exc} The iteration has "
                            "no stable fixed point from this start (the state may be unstable "
                            "at this S and field); try mixing < 1 or a smaller max_tilt") from exc
        correction = nambu_matrices(list(zip(hf[1], hf[0])), k, ns)
        change = (float("inf") if previous is None
                  else float(np.max(np.abs(correction - previous), initial=0.0)))
        previous = correction
        zeta = _linear_force(current, contract)
        force = float(np.max(np.abs(zeta), initial=0.0))
        H0 = _TermsHamiltonian(terms, ns).at(np.zeros((1, 2)))[0]
        shift = _condensate(H0, zeta, scale)
        directions, tilt = _tilt(directions, current.local_frames, current.spins, shift,
                                 settings.mixing, settings.max_tilt)
        history.append((change, force, tilt))
        if change <= settings.tolerance * scale and force <= settings.tolerance * scale:
            converged = True
            break
    if not converged:
        raise LSWTError(f"self-consistency did not converge in {settings.max_iterations} "
                        f"iterations (last change {history[-1][0]:.3g}, force "
                        f"{history[-1][1]:.3g}); try mixing < 1")

    final_state = make_state(directions)
    current = expand_model(model, final_state, conditions)
    hf = _merge(hartree_fock_terms(current.orders[4], contract))
    terms = list(current.orders[2]) + list(zip(hf[1], hf[0]))
    mesh = mesh_of(terms)
    contract = Contractions(mesh)
    three_magnon = NonlinearSpinWaves(current, mesh, _TermsHamiltonian(terms, ns).at,
                                      linear_tolerance=np.inf).cubic_energy() / ns
    energies = SelfConsistentEnergies(current.classical_energy / ns,
                                      per_site(contract, current.orders[2], pair),
                                      per_site(contract, current.orders[4], wick4), one_loop,
                                      three_magnon)
    numbers = np.array([contract((s, True, (0.0, 0.0)), (s, False, (0.0, 0.0))).real
                        for s in range(ns)])
    classical = np.array([state.direction(s, c) for s, c in keys], dtype=float)
    shifts = np.arccos(np.clip(np.sum(classical * directions, axis=1), -1.0, 1.0))
    spurious_gap = None
    if gapped:
        H0 = _TermsHamiltonian(terms, ns).at(np.zeros((1, 2)))
        try:
            spurious_gap = float(np.min(_diagonalize(H0, "none", np.zeros((1, 2)),
                                                     settings.zero_mode_tolerance)[1][0, :ns].real))
        except LSWTError:
            spurious_gap = float("nan")
        warnings.warn(
            f"solve_scswt(cubic=\"mean_field\"): the classical state has {zero_modes} zero "
            "mode(s) at k = 0 (Goldstone or accidental) and three-boson vertices. Without "
            "Sigma_3 the Gaussian scheme gaps them (lowest energy at k = 0: "
            f"{spurious_gap:.4g}; the gap is of relative order S^-1/2), so magnon_energies "
            "returns NaN and the order-S^0 energy carries an error of relative order "
            "S^-1/2 instead of 1/S (D73). Options: solve_nlswt (Goldstone modes exact at "
            "order 1/S), or a state without zero modes.", UserWarning, stacklevel=3)
    diagnostics = {"num_k": len(k), "mode": "mean_field", "iterations": iterations,
                   "converged": converged, "changes": [h[0] for h in history[-5:]],
                   "forces": [h[1] for h in history[-5:]], "max_cubic_coefficient": cubic,
                   "classical_linear_term": linear,
                   "final_force": float(np.max(np.abs(_linear_force(current, contract)),
                                               initial=0.0)),
                   "angle_shifts": shifts.tolist(), "k0_zero_modes": zero_modes,
                   "spurious_gap": spurious_gap,
                   "momenta": ("finite torus" if geometry.kind == "finite_torus" else
                               f"{settings.mesh[0]}x{settings.mesh[1]} mesh offset by 1/3 step")}
    header = ResultHeader.build("scswt", model, state, geometry, conditions, settings.as_dict(),
                                "energies per site of the model (E0)", diagnostics)
    return SCSWTResult(header, energies, k, np.asarray(current.spins, dtype=float), numbers,
                       np.asarray(model.lattice, dtype=float), final_state,
                       _TermsHamiltonian(current.orders[2], ns), tuple(hf[0]), hf[1],
                       tuple(current.orders[4]), contract)
