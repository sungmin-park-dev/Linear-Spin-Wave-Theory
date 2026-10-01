"""NBCP as a common SpinModel on the primitive triangular lattice.

The model has one Co site per primitive cell with the nearest- and
next-nearest-neighbour exchange matrices of :mod:`model.nbcp.exchange` and a
zeeman term. All values are dimensionless in the energy unit of the couplings
(meV for the parameter sets below); the field is ``b = mu_B B / meV`` when
the g-tensor is known, or the Zeeman energy ``h / meV`` with ``g = I``.

Bond records follow the transfer contract: the partner of a bond with the
existing displacement ``d`` sits at ``r - d`` (D13), so the stored cell offset
is ``-d @ inv(A)``.

Candidate magnetic structures (One to Four MSL) are :class:`SpinState` objects
on integer supercells of the same primitive lattice. The legacy angle order of
:mod:`model.nbcp.unit_cells` is kept, so existing angle arrays can be reused.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from spintoolkit._deprecation import internal_use
from spintoolkit.states.spin_state import SpinState, reduce_cell
from spintoolkit.system.model import Site, SpinModel, Term

from . import unit_cells
from .exchange import make_nn_exchange_matrices, make_nnn_exchange_matrices
from .unit_cells import DEFAULT_SPIN, DISP_NN, DISP_NNN

#: Primitive triangular lattice with nearest-neighbour distance one.
LATTICE = np.array([[0.5, np.sqrt(3) / 2], [0.5, -np.sqrt(3) / 2]])

#: Integer supercells of the candidate magnetic structures; ``M @ LATTICE``
#: equals the lattice vectors of the corresponding ``unit_cells`` builder.
SUPERCELLS: Dict[str, np.ndarray] = {
    "one_msl": np.array([[1, 0], [0, 1]]),
    "two_msl": np.array([[1, 1], [1, -1]]),
    "three_msl": np.array([[2, 1], [1, 2]]),
    "four_msl": np.array([[2, 0], [0, 2]]),
}

#: Published parameter sets. Exchange constants in meV; ``g`` is None when
#: only part of the g-tensor is known (use g = I and give the field as h).
PARAMETER_SETS: Dict[str, Dict[str, Any]] = {
    "woodland2025": {
        "parameters": {"Jxy": 0.0779, "Jz": 0.1225},
        "g": np.diag([4.200, 4.200, 4.716]),
        "sources": [
            "arXiv:2505.06398 (Woodland, Okuma, Stewart, Balz, Coldea), Table 1: "
            "Jz = 0.1225(10) meV, Jxy = 0.0779(7) meV, g_c = 4.716(7), g_ab = 4.200(7); "
            "nearest-neighbour XXZ, bonds counted once, S = 1/2 operators, "
            "Zeeman -mu_B (g_ab B_y S^y + g_c B_z S^z), z along c. The field offset "
            "dB = 0.041(4) T at 1.7 T || b* is a calibration, not a model parameter.",
        ],
    },
    "park2026_fig4": {
        "parameters": {"Jxy": 0.075, "Jz": 0.125},
        "g": None,
        "sources": [
            "docs/nbcp chapter 2 (Parameters and conventions), following Fig. 4 of "
            "Park et al.: J = 0.075 meV (J = 2 J_pm, i.e. Jxy), Jz = 0.125 meV, "
            "g_z = 4.645; the in-plane g is not given, so the field is supplied as the "
            "Zeeman energy h = g_z mu_B B for B || z.",
        ],
    },
}


def _offset(displacement: Sequence[float]) -> tuple:
    offset = -np.asarray(displacement, dtype=float) @ np.linalg.inv(LATTICE)
    rounded = np.rint(offset)
    if np.max(np.abs(offset - rounded)) > 1e-9:
        raise ValueError(f"displacement {displacement} is not a lattice vector")
    return tuple(int(x) for x in rounded)


def build_model(parameters: Mapping[str, float], g: Any = None,
                sources: Sequence[str] = (), spin: float = DEFAULT_SPIN,
                model_id: str = "nbcp") -> SpinModel:
    """NBCP SpinModel on the primitive triangular lattice.

    Parameters
    ----------
    parameters : mapping
        Exchange parameters in the keys of :mod:`model.nbcp.exchange`
        (``Jxy``, ``Jz``, ``JPD``, ``JGamma``, ``Dx``, ``Dy``, ``Dz``, ``Kxy``,
        ``Kz``, ``KPD``, ``KGamma``); missing keys are zero. The unit of these
        values is the energy unit E0 of the model.
    g : float or (3, 3) array or None
        Dimensionless g-tensor. None uses the identity, so the field must be
        given as the Zeeman energy ``h / E0``.
    sources : sequence of str
        Provenance of the parameters.
    spin : float
        Spin quantum number (default 1/2).
    model_id : str
        Model identifier.

    Returns
    -------
    SpinModel
    """
    parameters = dict(parameters)
    terms = []
    for label, matrices, displacements in [
        ("NN", make_nn_exchange_matrices(parameters), DISP_NN),
        ("NNN", make_nnn_exchange_matrices(parameters), DISP_NNN),
    ]:
        if matrices is None:
            continue
        for J, d in zip(matrices, displacements):
            terms.append(Term.bilinear(("Co", (0, 0)), ("Co", _offset(d)), J, label))
    g_tensor = np.eye(3) if g is None else np.asarray(g, dtype=float)
    if g_tensor.ndim == 0:
        g_tensor = g_tensor * np.eye(3)
    terms.append(Term.zeeman("Co", g_tensor))
    metadata = {"model_id": model_id, "energy_unit": "meV",
                "parameters": {**parameters,
                               "g": None if g is None else g_tensor.tolist()},
                "sources": list(sources)}
    return SpinModel(LATTICE, [Site("Co", (0.0, 0.0), spin)], terms, metadata)


def build_published_model(name: str, **overrides: float) -> SpinModel:
    """Build a model from :data:`PARAMETER_SETS`, optionally overriding parameters."""
    entry = PARAMETER_SETS[name]
    return build_model({**entry["parameters"], **overrides}, entry["g"],
                       entry["sources"], model_id=f"nbcp_{name}")


def legacy_cells(cell: str) -> Dict[str, tuple]:
    """Map each legacy sublattice label of ``cell`` to its canonical cell.

    Reads the site positions of the :mod:`model.nbcp.unit_cells` builder and
    reduces them modulo the supercell of :data:`SUPERCELLS`.
    """
    supercell = SUPERCELLS[cell]
    num_angles = 2 * abs(round(np.linalg.det(supercell)))
    with internal_use():
        legacy = getattr(unit_cells, cell)({"h": (0.0, 0.0, 0.0)}, angles=np.zeros(num_angles))
    if not np.allclose(legacy.lattice_vectors, supercell @ LATTICE):
        raise RuntimeError(f"{cell}: supercell does not match the legacy lattice vectors")
    inverse = np.linalg.inv(LATTICE)
    mapping = {}
    for site in legacy.sites:
        fractional = np.asarray(site.position, dtype=float) @ inverse
        rounded = np.rint(fractional)
        if np.max(np.abs(fractional - rounded)) > 1e-9:
            raise RuntimeError(f"{cell}: site {site.label} is not on the primitive lattice")
        mapping[site.label] = reduce_cell(rounded.astype(int), supercell)
    return mapping


def candidate_state(model: SpinModel, cell: str, angles: Sequence[float],
                    provenance: Optional[Mapping[str, Any]] = None) -> SpinState:
    """Candidate magnetic structure from legacy angles.

    Parameters
    ----------
    model : SpinModel
        An NBCP model from :func:`build_model`.
    cell : {"one_msl", "two_msl", "three_msl", "four_msl"}
    angles : sequence of float
        ``[theta_A, phi_A, theta_B, phi_B, ...]`` in the sublattice order of the
        legacy builder; a spin points along
        ``(sin theta cos phi, sin theta sin phi, cos theta)``.
    provenance : mapping, optional

    Returns
    -------
    SpinState
    """
    mapping = legacy_cells(cell)
    angles = np.asarray(angles, dtype=float).reshape(-1, 2)
    if len(angles) != len(mapping):
        raise ValueError(f"{cell} needs {2 * len(mapping)} angles, got {2 * len(angles)}")
    directions = {}
    for (label, canonical), (theta, phi) in zip(mapping.items(), angles):
        directions[("Co", canonical)] = np.array([np.sin(theta) * np.cos(phi),
                                                  np.sin(theta) * np.sin(phi), np.cos(theta)])
    return SpinState(model.fingerprint(), SUPERCELLS[cell], directions,
                     {"origin": "legacy_angles", "cell": cell,
                      "legacy_labels": {label: list(c) for label, c in mapping.items()},
                      **(provenance or {})})
