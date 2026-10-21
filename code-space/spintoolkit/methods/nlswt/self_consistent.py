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
    """

    mesh: Tuple[int, int] = (24, 24)
    tolerance: float = 1e-10
    max_iterations: int = 200
    mixing: float = 1.0
    zero_mode_tolerance: float = 1e-9
    cubic_tolerance: float = 1e-10

    def __post_init__(self):
        mesh = tuple(int(n) for n in self.mesh)
        if len(mesh) != 2 or min(mesh) < 1:
            raise ValueError("mesh must be two positive integers")
        if not 0 < self.mixing <= 1:
            raise ValueError("mixing must be in (0, 1]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        object.__setattr__(self, "mesh", mesh)

    def as_dict(self) -> Dict[str, Any]:
        return {"mesh": list(self.mesh), "mesh_offset": "1/3 step", "tolerance": self.tolerance,
                "max_iterations": self.max_iterations, "mixing": self.mixing,
                "zero_mode_tolerance": self.zero_mode_tolerance,
                "cubic_tolerance": self.cubic_tolerance}


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
        ``H_0 + E_zp + <H_4>_LSWT``: the order-S^0 energy of
        :func:`solve_nlswt` for a state without cubic vertices; NaN where the
        LSWT vacuum does not exist.
    """

    classical: float
    quadratic: float
    quartic: float
    one_loop: float

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
    """

    header: ResultHeader
    energies: SelfConsistentEnergies
    k_points: np.ndarray
    spins: np.ndarray
    boson_numbers: np.ndarray
    lattice: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _quadratic: Any = field(default=None, repr=False, compare=False)
    _monomials: Tuple = field(default=(), repr=False, compare=False)
    _coefficients: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _quartic: Tuple = field(default=(), repr=False, compare=False)

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
        where magnon energies are not defined.
        """
        k = np.atleast_2d(np.asarray(k_points, dtype=float))
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
                "total": e.total, "one_loop": e.one_loop, "spins": self.spins,
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

    Returns
    -------
    SCSWTResult

    Raises
    ------
    LSWTError
        Cubic vertices (the scheme is not defined there), a non-stationary
        state, a zero or negative mode on the momenta in any iteration, or no
        convergence.
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
    if linear > 1e-8 * scale:
        raise LSWTError(f"the reference state is not stationary (linear boson term {linear:.3g})")
    cubic = max((abs(c) for c, _ in expansion.orders[3]), default=0.0)
    if cubic > settings.cubic_tolerance * scale:
        raise LSWTError(
            f"the state has three-boson vertices (largest {cubic:.3g}): the cubic self-energy "
            "and the tadpole then shift magnon energies at the same order as the Hartree-Fock "
            "term, and the self-consistent one-loop scheme, which keeps only the latter, is "
            "not defined (D72). Options: solve_nlswt (one loop with Sigma_3 and the tadpole), "
            "or a state without cubic vertices.")
    k, grid, reciprocal = _momenta(model, state, geometry, settings)
    resolution = _resolution(np.mod(grid, 1.0))
    quadratic = QuadraticBoseHamiltonian(model, state, conditions)
    ns = expansion.num_sites
    quartic = expansion.orders[4]

    def per_site(contract, terms, expectation):
        return float(np.real(sum(c * expectation(contract, m) for c, m in terms))) / ns

    def vacuum(monomials, coefficients):
        def at(kk):
            H = np.asarray(quadratic.at(kk), dtype=complex)
            if len(monomials):
                H = H + nambu_matrices(list(zip(coefficients, monomials)), kk, ns)
            return H
        return Contractions(bogoliubov_mesh(at, k, grid, reciprocal,
                                            settings.zero_mode_tolerance, resolution))

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
                       np.asarray(model.lattice, dtype=float), quadratic, tuple(monomials),
                       coefficients, tuple(quartic))
