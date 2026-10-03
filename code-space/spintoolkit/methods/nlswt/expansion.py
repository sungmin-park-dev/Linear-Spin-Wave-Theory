"""Holstein-Primakoff expansion of a ``SpinModel`` to fourth order in bosons (D46).

Every spin is written in its local frame ``R_i`` (columns: local x, y, z; z
along the classical spin), the same frame as the LSWT Hamiltonian (D36), with

    s^z = S - a^dagger a,
    s^+ = sqrt(2S) [a - a^dagger a a / (4S)] + O(S^{-3/2}),
    s^- = sqrt(2S) [a^dagger - a^dagger a^dagger a / (4S)] + O(S^{-3/2}).

These operators are normal ordered on each site, and operators on different
sites commute, so every product below is normal ordered. Grouped by the number
of boson operators ``n``, the Hamiltonian is ``H = sum_n H_n`` with
``H_n ~ S^{2 - n/2}`` (the field counts as order S, so that the classical
state does not depend on S). The truncation above keeps exactly the leading
power of S in ``H_0 .. H_4``: these give the ground-state energy through
order ``S^0`` and magnon energies through order ``S^0`` (relative 1/S).

A boson monomial is a tuple of operators ``(site, dagger, position)``,
``site`` the Nambu index of a magnetic-cell site, ``dagger`` a bool and
``position`` the Cartesian position of the physical site (the D13
full-position convention). The Hamiltonian per magnetic cell is the sum of
the monomials over all magnetic-lattice translations.

Onsite (single-ion) terms are not expanded here: the coherent-state rule of
D37 fixes their quadratic order only, and the cubic and quartic orders of the
``(1 - 1/2S)`` renormalization have not been derived.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from spintoolkit.methods.lswt.quadratic import local_frame
from spintoolkit.states.spin_state import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import BILINEAR, ONSITE, ZEEMAN, SpinModel

#: Single-site monomial ``(a^dagger)^p a^q`` keyed by ``(p, q)``.
SitePolynomial = Dict[Tuple[int, int], complex]
Operator = Tuple[int, bool, Tuple[float, float]]
Monomial = Tuple[Operator, ...]


class NonlinearExpansionError(ValueError):
    """The model contains a term the nonlinear expansion does not cover."""


def local_spin_polynomials(spin: float) -> Tuple[SitePolynomial, SitePolynomial, SitePolynomial]:
    """``(s^x, s^y, s^z)`` of one spin in its local frame as normal-ordered polynomials.

    Returns
    -------
    tuple of dict
        Each maps ``(p, q)`` to the coefficient of ``(a^dagger)^p a^q``.
    """
    root = np.sqrt(2.0 * spin)
    plus = {(0, 1): root, (1, 2): -root / (4.0 * spin)}
    minus = {(1, 0): root, (2, 1): -root / (4.0 * spin)}
    sx: SitePolynomial = defaultdict(complex)
    sy: SitePolynomial = defaultdict(complex)
    for key, value in plus.items():
        sx[key] += value / 2
        sy[key] += value / 2j
    for key, value in minus.items():
        sx[key] += value / 2
        sy[key] -= value / 2j
    sz = {(0, 0): complex(spin), (1, 1): -1.0 + 0j}
    return dict(sx), dict(sy), sz


def _site_ops(site: int, position, p: int, q: int) -> Tuple[Operator, ...]:
    pos = (float(position[0]), float(position[1]))
    return tuple([(site, True, pos)] * p + [(site, False, pos)] * q)


@dataclass(frozen=True)
class BosonExpansion:
    """Boson monomials of ``H`` per magnetic cell, grouped by operator number.

    Attributes
    ----------
    num_sites : int
        Sites of the magnetic cell (Nambu dimension ``2 * num_sites``).
    positions : (Ns, 2) array
        Home positions of the magnetic-cell sites (D13 gauge).
    spins : (Ns,) array
    local_frames : (Ns, 3, 3) array
    orders : dict
        ``orders[n]`` is a list of ``(coefficient, monomial)`` with ``n``
        operators, ``n = 0 .. 4``.
    """

    num_sites: int
    positions: np.ndarray
    spins: np.ndarray
    local_frames: np.ndarray
    orders: Dict[int, List[Tuple[complex, Monomial]]]

    @property
    def classical_energy(self) -> float:
        """``H_0`` per magnetic cell."""
        return float(np.real(sum(c for c, _ in self.orders[0])))


def expand_model(model: SpinModel, state: SpinState,
                 conditions: Optional[ExternalConditions] = None,
                 tolerance: float = 1e-15) -> BosonExpansion:
    """Holstein-Primakoff monomials of ``model`` about ``state`` up to four bosons.

    Parameters
    ----------
    model : SpinModel
        Bilinear and Zeeman terms only.
    state : SpinState
    conditions : ExternalConditions, optional
    tolerance : float
        Monomials with ``|coefficient|`` below this are dropped.

    Returns
    -------
    BosonExpansion

    Raises
    ------
    NonlinearExpansionError
        For onsite terms (see the module docstring).
    """
    if model.terms_of_kind(ONSITE):
        raise NonlinearExpansionError(
            "onsite terms are not supported by the nonlinear spin-wave expansion: the "
            "coherent-state (1 - 1/2S) rule of D37 is derived for the quadratic order only")
    conditions = conditions or ExternalConditions()
    keys = tuple((site.id, cell) for site in model.sites for cell in state.cells)
    index = {key: i for i, key in enumerate(keys)}
    spins = np.array([model.site(s).spin for s, _ in keys], dtype=float)
    positions = np.array([model.cartesian_position(s, c) for s, c in keys], dtype=float)
    frames = np.array([local_frame(state.direction(s, c)) for s, c in keys])
    polys = [local_spin_polynomials(S) for S in spins]
    orders: Dict[int, Dict[Monomial, complex]] = {n: defaultdict(complex) for n in range(5)}

    def add(n, monomial, value):
        if n > 4:
            return
        if monomial:                      # translate the first operator to its home position
            s0, _, r0 = monomial[0]
            shift = positions[s0] - np.asarray(r0)
            monomial = tuple((s, d, (round(r[0] + shift[0], 12) + 0.0, round(r[1] + shift[1], 12) + 0.0))
                             for s, d, r in monomial)
        orders[n][monomial] += value

    field = conditions.field
    for term in model.terms_of_kind(ZEEMAN):
        site = term.participants[0][0]
        h_global = term.coefficient.T @ field                    # energy -h . S
        for cell in state.cells:
            i = index[(site, cell)]
            h_local = h_global @ frames[i]
            for mu in range(3):
                if h_local[mu] == 0:
                    continue
                for (p, q), value in polys[i][mu].items():
                    add(p + q, _site_ops(i, positions[i], p, q), -h_local[mu] * value)

    for term in model.terms_of_kind(BILINEAR):
        (a, n1), (b, n2) = term.participants
        for cell in state.cells:
            source = (cell[0] + n1[0], cell[1] + n1[1])
            target = (cell[0] + n2[0], cell[1] + n2[1])
            i = index[(a, state.reduce_cell(source))]
            j = index[(b, state.reduce_cell(target))]
            delta = model.cartesian_position(a, source) - model.cartesian_position(b, target)
            ri = positions[i]
            rj = positions[i] - delta
            if np.allclose(delta, 0):
                raise NonlinearExpansionError("a bilinear term couples a spin to itself")
            Jt = frames[i].T @ term.coefficient @ frames[j]
            for mu in range(3):
                for nu in range(3):
                    if Jt[mu, nu] == 0:
                        continue
                    for (p, q), u in polys[i][mu].items():
                        for (r, s), v in polys[j][nu].items():
                            n = p + q + r + s
                            if n > 4:
                                continue
                            monomial = _site_ops(i, ri, p, q) + _site_ops(j, rj, r, s)
                            add(n, monomial, Jt[mu, nu] * u * v)

    cleaned = {n: [(c, m) for m, c in orders[n].items() if abs(c) > tolerance] for n in range(5)}
    if not cleaned[0]:
        cleaned[0] = [(0j, ())]
    return BosonExpansion(len(keys), positions, spins, frames, cleaned)
