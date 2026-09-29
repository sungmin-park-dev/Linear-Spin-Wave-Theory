"""Bridge between the common model and the existing ``SpinSystem``.

The existing LSWT code (``methods/lswt``, ``observables``) takes a
:class:`SpinSystem` whose lattice is the magnetic cell. These functions let the
common :class:`SpinModel` / :class:`SpinState` pair drive that code unchanged,
and read existing systems into the common format.

Conventions
-----------
- Bond displacement (D13): a stored ``SpinSystem`` displacement is
  ``d = r_source - r_target``; the Fourier phase of the existing Hamiltonian is
  ``exp(-i k . d) = exp(+i k . (r_target - r_source))``.
- Positions in ``SpinSystem`` are Cartesian.
- Angles: a unit vector ``n`` maps to ``theta = arccos(n_z)`` and
  ``phi = atan2(n_y, n_x)`` (``phi = 0`` at the poles). Equivalent angle pairs
  such as ``(-theta, phi)`` and ``(theta, phi + pi)`` describe the same spin but
  a different local frame, so off-diagonal elements of ``H(k)`` differ by a
  gauge phase while spectra and energies agree.
- Field (D20): the ``SpinSystem`` field of site ``a`` is the Zeeman energy
  ``h_a = g_a^T b`` in the energy unit E0 of the coefficients. The existing
  LSWT thermodynamics interpret energies as meV and temperatures as kelvin.
"""

from __future__ import annotations

from typing import Optional, Tuple
import warnings

import numpy as np

from spintoolkit.states.spin_state import SpinState, validate_spin_state
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import BILINEAR, ZEEMAN, Site, SpinModel, Term
from spintoolkit.system.spin_system import SpinSystem

#: Term kinds that a SpinSystem can represent.
SUPPORTED_KINDS = (BILINEAR, ZEEMAN)

#: Absolute tolerance for recovering integer cell offsets from displacements.
OFFSET_TOLERANCE = 1e-9


def _angles(vector: np.ndarray) -> Tuple[float, float]:
    theta = float(np.arccos(np.clip(vector[2], -1.0, 1.0)))
    phi = float(np.arctan2(vector[1], vector[0])) if np.hypot(vector[0], vector[1]) > 0 else 0.0
    return theta, phi


def site_label(site_id: str, cell, num_cells: int) -> str:
    """Label of a magnetic site: the site id, plus the cell when the supercell is larger."""
    return site_id if num_cells == 1 else f"{site_id}@{cell[0]},{cell[1]}"


def to_spin_system(model: SpinModel, state: SpinState,
                   conditions: Optional[ExternalConditions] = None,
                   label: Optional[str] = None) -> SpinSystem:
    """Build the existing ``SpinSystem`` of a model and a state.

    The lattice of the result is the magnetic cell ``M @ A``; it has one site
    per model site and canonical cell of the supercell (model site order, then
    cell order) and one coupling per bilinear term and cell.

    Parameters
    ----------
    model : SpinModel
    state : SpinState
        Must belong to ``model``.
    conditions : ExternalConditions, optional
        Dimensionless field (default zero); the temperature is not stored.
    label : str, optional
        Label of the resulting system (default: the model id).

    Returns
    -------
    SpinSystem
    """
    unsupported = sorted({t.kind for t in model.terms} - set(SUPPORTED_KINDS))
    if unsupported:
        raise NotImplementedError(f"SpinSystem cannot represent term kinds {unsupported}")
    validate_spin_state(state, model)
    field = (conditions or ExternalConditions()).field
    g_of = {t.participants[0][0]: t.coefficient for t in model.terms_of_kind(ZEEMAN)}
    if np.any(field != 0):
        uncoupled = sorted(set(model.site_ids) - set(g_of))
        if uncoupled:
            warnings.warn(f"sites {uncoupled} have no zeeman term and do not couple "
                          "to the applied field", UserWarning, stacklevel=2)

    system = SpinSystem(lattice_vectors=state.magnetic_lattice(model),
                        label=label or model.metadata["model_id"])
    n = state.num_cells
    for site in model.sites:
        h = g_of[site.id].T @ field if site.id in g_of else np.zeros(3)
        for cell in state.cells:
            system.add_site(site_label(site.id, cell, n), model.cartesian_position(site.id, cell),
                            site.spin, _angles(state.direction(site.id, cell)), h)
    for term in model.terms_of_kind(BILINEAR):
        (a, n1), (b, n2) = term.participants
        for cell in state.cells:
            source = (cell[0] + n1[0], cell[1] + n1[1])
            target = (cell[0] + n2[0], cell[1] + n2[1])
            d = model.cartesian_position(a, source) - model.cartesian_position(b, target)
            system.add_coupling(site_label(a, state.reduce_cell(source), n),
                                site_label(b, state.reduce_cell(target), n),
                                term.coefficient, d)
    return system


def from_spin_system(system: SpinSystem,
                     model_id: str = "from_spin_system"
                     ) -> Tuple[SpinModel, SpinState, ExternalConditions]:
    """Read an existing ``SpinSystem`` into the common format.

    The magnetic cell of the system becomes the primitive cell of the model and
    the state has the supercell ``I``. Zeeman terms get ``g = I`` and the field
    is the common Zeeman energy of the sites.

    Raises
    ------
    ValueError
        If a displacement does not connect two lattice sites, or if the sites
        have different fields. A site-dependent field can come from different
        g-tensors under one applied field or from a local field that does not
        scale with the applied field; the stored values cannot tell these
        apart, so build such a model directly (site-dependent g) or request a
        ``local_field`` term kind.
    """
    lattice = np.asarray(system.lattice_vectors, dtype=float)
    inverse = np.linalg.inv(lattice)
    positions = [np.asarray(s.position, dtype=float) for s in system.sites]
    labels = [s.label for s in system.sites]
    fields = np.array([np.asarray(s.magnetic_field, dtype=float) for s in system.sites])
    if not np.allclose(fields, fields[0], rtol=0, atol=0):
        raise ValueError(
            "sites have different fields; a SpinSystem cannot tell site-dependent "
            "g-tensors under one applied field from local fields that do not scale "
            "with it. Build the model directly with site-dependent g, or request a "
            "local_field term kind.")
    sites = [Site(label, position @ inverse, s.spin)
             for label, position, s in zip(labels, positions, system.sites)]
    terms = []
    for coupling in system.couplings:
        i, j = coupling.site_i, coupling.site_j
        offset = (positions[i] - coupling.displacement - positions[j]) @ inverse
        rounded = np.rint(offset)
        if np.max(np.abs(offset - rounded)) > OFFSET_TOLERANCE:
            raise ValueError(f"coupling {labels[i]} -> {labels[j]} with displacement "
                             f"{coupling.displacement.tolist()} does not end on a lattice "
                             "site under r_target = r_source - d (D13)")
        terms.append(Term.bilinear((labels[i], (0, 0)), (labels[j], tuple(rounded.astype(int))),
                                   coupling.exchange_matrix))
    terms += [Term.zeeman(label, np.eye(3)) for label in labels]
    model = SpinModel(lattice, sites, terms,
                      {"model_id": model_id, "energy_unit": "meV",
                       "sources": [f"SpinSystem {system.label or ''}".strip()]})
    directions = {}
    for label, s in zip(labels, system.sites):
        theta, phi = np.asarray(s.angles, dtype=float)
        directions[(label, (0, 0))] = np.array([np.sin(theta) * np.cos(phi),
                                                np.sin(theta) * np.sin(phi), np.cos(theta)])
    state = SpinState(model.fingerprint(), np.eye(2, dtype=int), directions,
                      {"origin": "from_spin_system"})
    return model, state, ExternalConditions(field=fields[0])
