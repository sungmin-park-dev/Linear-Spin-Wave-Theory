"""Classical energy, local fields and torques of a spin state.

Reads only the model's terms, so it works for any :class:`SpinModel` without
model-specific code. Spins are classical vectors ``S n`` of length ``S``:

    E = sum_bilinear S_i^T J S_j - sum_zeeman (mu_B B)^T g S_i

evaluated over one magnetic supercell and reported per site. The local field
``h_i = -dE/dS_i`` collects every term that contains spin ``i``; the torque
``S_i x h_i`` vanishes at a classical stationary point, the condition for the
linear boson terms of LSWT to vanish.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import warnings

import numpy as np

from spintoolkit.states.spin_state import SpinState, validate_spin_state
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import BILINEAR, ZEEMAN, SpinModel

#: Term kinds this consumer understands.
SUPPORTED_KINDS = (BILINEAR, ZEEMAN)

Key = Tuple[str, Tuple[int, int]]


def _prepare(model: SpinModel, state: SpinState,
             conditions: Optional[ExternalConditions]):
    unsupported = sorted({t.kind for t in model.terms} - set(SUPPORTED_KINDS))
    if unsupported:
        raise NotImplementedError(f"classical methods do not support term kinds {unsupported}")
    validate_spin_state(state, model)
    conditions = conditions or ExternalConditions()
    field = conditions.zeeman_field(model.units)
    if np.any(field != 0):
        coupled = {t.participants[0][0] for t in model.terms_of_kind(ZEEMAN)}
        uncoupled = sorted(set(model.site_ids) - coupled)
        if uncoupled:
            warnings.warn(f"sites {uncoupled} have no zeeman term and do not couple "
                          "to the applied field", UserWarning, stacklevel=3)
    spins = {(site, cell): model.site(site).spin * state.direction(site, cell)
             for site in model.site_ids for cell in state.cells}
    return field, spins


def _shift(cell, offset):
    return (cell[0] + offset[0], cell[1] + offset[1])


def classical_energy(model: SpinModel, state: SpinState,
                     conditions: Optional[ExternalConditions] = None) -> float:
    """Classical energy per site in the model's energy unit.

    Parameters
    ----------
    model : SpinModel
    state : SpinState
        Must belong to ``model``.
    conditions : ExternalConditions, optional
        Applied field (default zero). The temperature is not used.
    """
    field, spins = _prepare(model, state, conditions)
    energy = 0.0
    for term in model.terms_of_kind(BILINEAR):
        (a, n1), (b, n2) = term.participants
        for cell in state.cells:
            s_i = spins[(a, state.reduce_cell(_shift(cell, n1)))]
            s_j = spins[(b, state.reduce_cell(_shift(cell, n2)))]
            energy += s_i @ term.coefficient @ s_j
    for term in model.terms_of_kind(ZEEMAN):
        site = term.participants[0][0]
        for cell in state.cells:
            energy -= field @ term.coefficient @ spins[(site, cell)]
    return float(energy / (model.num_sites * state.num_cells))


def local_fields(model: SpinModel, state: SpinState,
                 conditions: Optional[ExternalConditions] = None) -> Dict[Key, np.ndarray]:
    """Local field ``h_i = -dE/dS_i`` on every spin of the supercell.

    Returns
    -------
    dict
        ``(site_id, cell) -> (3,)`` array in the model's energy unit.
    """
    field, spins = _prepare(model, state, conditions)
    fields = {key: np.zeros(3) for key in spins}
    for term in model.terms_of_kind(BILINEAR):
        (a, n1), (b, n2) = term.participants
        J = term.coefficient
        for cell in state.cells:
            i = (a, state.reduce_cell(_shift(cell, n1)))
            j = (b, state.reduce_cell(_shift(cell, n2)))
            fields[i] -= J @ spins[j]
            fields[j] -= J.T @ spins[i]
    for term in model.terms_of_kind(ZEEMAN):
        site = term.participants[0][0]
        for cell in state.cells:
            fields[(site, cell)] += term.coefficient.T @ field
    return fields


def torques(model: SpinModel, state: SpinState,
            conditions: Optional[ExternalConditions] = None) -> Dict[Key, np.ndarray]:
    """Torque ``S_i x h_i`` on every spin; all vanish at a stationary state.

    Returns
    -------
    dict
        ``(site_id, cell) -> (3,)`` array in the model's energy unit.
    """
    fields = local_fields(model, state, conditions)
    return {key: np.cross(model.site(key[0]).spin * state.direction(*key), value)
            for key, value in fields.items()}
