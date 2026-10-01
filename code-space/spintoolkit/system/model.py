"""Common spin-model definition shared by all calculation methods.

A :class:`SpinModel` is a Hamiltonian on an infinite two-dimensional periodic
lattice. It stores the primitive cell, the spin quantum number of each site,
and one record per translation orbit of each term:

    H = sum_R sum_bilinear S_(R+n1,a)^T J S_(R+n2,b)
        + sum_R sum_onsite S_(R,a)^T A_a S_(R,a)
        - sum_R sum_a b^T g_a S_(R,a)

Positions are fractional coordinates ``f`` of the lattice matrix ``A`` whose
rows are the primitive vectors, so a site sits at ``(R + f) @ A``. Spin
components use one global Cartesian frame and ``S`` operators (not Pauli
matrices). The field ``b`` is an external variable supplied with each
calculation; the model stores only the dimensionless g-tensor of each
``zeeman`` term.

All numbers are dimensionless: energies, the field ``b = mu_B B`` and the
temperature ``k_B T`` are expressed in the energy unit ``E0`` of the
coefficients (e.g. meV if the couplings are given in meV). Conversion to
physical units (tesla, kelvin, meV) is left to the user; the physical
conventions and the constants ``MU_B_MEV_PER_T`` and ``K_BOLTZMANN_MEV`` are
documented in :mod:`spintoolkit.definitions`. ``metadata["energy_unit"]`` may
record the name of ``E0``; it is a label only.

The term format is open to later kinds, but validation accepts only the kinds
in :data:`SUPPORTED_KINDS`. A model is validated when it is constructed, so a
``SpinModel`` object is always valid; all violations are reported together in
one :class:`SpinModelError`. Arrays are copied and made read-only.

See ``GOVERNMENT/Working-Pad/idea-proposals/2026-09-23-spin-model-transfer-contract.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA_VERSION = 1

BILINEAR = "bilinear"
ZEEMAN = "zeeman"
ONSITE = "onsite"
#: Number of participants required by each supported kind.
SUPPORTED_KINDS = {BILINEAR: 2, ZEEMAN: 1, ONSITE: 1}

#: Relative tolerance for lattice degeneracy and half-integer spin checks.
LATTICE_TOLERANCE = 1e-10

Participant = Tuple[str, Tuple[int, int]]


class SpinModelError(ValueError):
    """Raised when a model violates the transfer contract.

    Attributes
    ----------
    violations : list of str
        Every violation found, in the order checked.
    """

    def __init__(self, violations: Sequence[str]):
        self.violations = list(violations)
        lines = "\n".join(f"  - {v}" for v in self.violations)
        super().__init__(f"{len(self.violations)} spin-model violation(s):\n{lines}")


def _frozen_array(value: Any, name: str) -> np.ndarray:
    """Return a read-only float copy; reject complex input explicitly."""
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be real, got a complex array")
    array = np.array(value, dtype=float, copy=True)
    array.setflags(write=False)
    return array


def _offset(value: Any) -> Tuple:
    """Normalize a cell offset; keep non-integers so validation can report them."""
    entries = tuple(float(x) for x in value)
    if all(x.is_integer() for x in entries):
        return tuple(int(x) for x in entries)
    return entries


@dataclass(frozen=True)
class Site:
    """One site of the primitive cell.

    Parameters
    ----------
    id : str
        Stable identifier, unique within the model.
    position : array_like, shape (2,)
        Fractional coordinates with respect to the lattice matrix.
    spin : float
        Spin quantum number ``S`` (positive integer or half-integer).
    """

    id: str
    position: np.ndarray
    spin: float

    def __post_init__(self):
        object.__setattr__(self, "position", _frozen_array(self.position, "position"))
        object.__setattr__(self, "spin", float(self.spin))


@dataclass(frozen=True)
class Term:
    """One Hamiltonian term, recorded once per translation orbit.

    Use :meth:`bilinear` and :meth:`zeeman` to build the supported kinds; the
    general constructor accepts any kind so that later kinds need no schema
    change, but validation rejects kinds outside :data:`SUPPORTED_KINDS`.

    Parameters
    ----------
    kind : str
        Term kind.
    participants : sequence of (site_id, (n1, n2))
        Sites and integer cell offsets taking part in the term.
    coefficient : array_like
        Real coefficient array; its shape is fixed by the kind.
    label : str, optional
        Descriptive label (e.g. ``"NN"``). Consumers never rebuild
        coefficients from it, and it does not enter the fingerprint.
    """

    kind: str
    participants: Tuple[Participant, ...]
    coefficient: np.ndarray
    label: Optional[str] = None

    def __post_init__(self):
        participants = tuple((str(site), _offset(cell)) for site, cell in self.participants)
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "coefficient", _frozen_array(self.coefficient, "coefficient"))

    @classmethod
    def bilinear(cls, source: Participant, target: Participant, J: Any,
                 label: Optional[str] = None) -> "Term":
        """Two-site exchange ``S_source^T J S_target``.

        Parameters
        ----------
        source, target : (site_id, (n1, n2))
            The two participating sites with their cell offsets.
        J : array_like, shape (3, 3)
            Real exchange matrix in the global spin frame; need not be symmetric.
        label : str, optional
            Descriptive label.
        """
        return cls(BILINEAR, (source, target), J, label)

    @classmethod
    def zeeman(cls, site: str, g: Any, label: Optional[str] = None) -> "Term":
        """Field coupling ``-b^T g S`` of one site (``b = mu_B B`` in units of E0).

        Parameters
        ----------
        site : str
            Site identifier.
        g : array_like, shape (3, 3)
            Dimensionless g-tensor; ``g[alpha, beta]`` links spin component
            ``beta`` to field component ``alpha``. Use the identity when ``g`` is
            unknown and supply the Zeeman energy ``h`` as the field.
        label : str, optional
            Descriptive label.
        """
        return cls(ZEEMAN, ((site, (0, 0)),), g, label)

    @classmethod
    def onsite(cls, site: str, A: Any, label: Optional[str] = None) -> "Term":
        """Single-ion quadratic term ``S_site^T A S_site`` (D37).

        Parameters
        ----------
        site : str
            Site identifier.
        A : array_like, shape (3, 3)
            Real symmetric anisotropy matrix in the global spin frame, e.g.
            ``diag(0, 0, D)`` for ``D (S^z)^2``. The antisymmetric part of a
            same-site product is ``(i/2) eps_abc A_ab S^c``, which is not
            Hermitian, so it is rejected.
        label : str, optional

        Notes
        -----
        For ``S = 1/2`` the operator is the constant ``tr(A) / 4``. Classical and
        LSWT methods use the spin-coherent-state value
        ``S(S - 1/2) n^T A n + (S/2) tr A``, i.e. the coefficient ``A`` scaled
        by ``1 - 1/(2S)`` (see :func:`onsite_renormalization`).
        """
        return cls(ONSITE, ((site, (0, 0)),), A, label)


def _freeze_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    data = dict(metadata)
    data.setdefault("parameters", {})
    data.setdefault("sources", ())
    data["parameters"] = MappingProxyType(dict(data["parameters"]))
    data["sources"] = tuple(data["sources"])
    return MappingProxyType(data)


@dataclass(frozen=True)
class SpinModel:
    """Hamiltonian of a 2D periodic spin system.

    Parameters
    ----------
    lattice : array_like, shape (2, 2)
        Lattice matrix ``A``; each row is a primitive vector.
    sites : sequence of Site
        Sites of the primitive cell.
    terms : sequence of Term
        One record per translation orbit of each term.
    metadata : mapping
        Must contain ``model_id``; ``parameters`` and ``sources`` default to
        empty; ``energy_unit`` optionally names E0. Records only: solvers never
        recompute coefficients from it.
    schema_version : int, optional
        Transfer-contract version.

    Raises
    ------
    SpinModelError
        If the model violates the transfer contract.
    """

    lattice: np.ndarray
    sites: Tuple[Site, ...]
    terms: Tuple[Term, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(self, "lattice", _frozen_array(self.lattice, "lattice"))
        object.__setattr__(self, "sites", tuple(self.sites))
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        validate_spin_model(self)

    # -- access ---------------------------------------------------------
    @property
    def site_ids(self) -> Tuple[str, ...]:
        """Site identifiers in declaration order."""
        return tuple(site.id for site in self.sites)

    @property
    def num_sites(self) -> int:
        """Number of sites in the primitive cell."""
        return len(self.sites)

    def site(self, site_id: str) -> Site:
        """Return the site with identifier ``site_id``."""
        for site in self.sites:
            if site.id == site_id:
                return site
        raise KeyError(site_id)

    def terms_of_kind(self, kind: str) -> Tuple[Term, ...]:
        """Return the terms of one kind in declaration order."""
        return tuple(term for term in self.terms if term.kind == kind)

    def cartesian_position(self, site_id: str, cell: Sequence[int] = (0, 0)) -> np.ndarray:
        """Cartesian position ``(cell + f) @ A`` of a site."""
        return (np.asarray(cell, dtype=float) + self.site(site_id).position) @ self.lattice

    # -- identity -------------------------------------------------------
    def fingerprint(self) -> str:
        """SHA-256 of the physical content: lattice, sites and terms.

        Independent of term order, labels and metadata. Equivalent records of
        one bilinear term (translated or with reversed orientation and a
        transposed ``J``) give the same fingerprint.
        """
        digest = hashlib.sha256()
        digest.update(f"schema={self.schema_version};".encode())
        digest.update(_float_bytes(self.lattice))
        for site in sorted(self.sites, key=lambda s: s.id):
            digest.update(f"site={site.id};S={site.spin!r};".encode())
            digest.update(_float_bytes(site.position))
        for key, coefficient in sorted(_canonical_term(t) for t in self.terms):
            digest.update(repr(key).encode())
            digest.update(coefficient)
        return digest.hexdigest()


def _float_bytes(array: np.ndarray) -> bytes:
    # Adding 0.0 maps -0.0 to 0.0 so that equal values hash equally.
    return (np.asarray(array, dtype=np.float64) + 0.0).tobytes()


def _canonical_term(term: Term) -> Tuple[Tuple, bytes]:
    """Order-independent key of a term and the bytes of its coefficient."""
    if term.kind == BILINEAR:
        (a, n1), (b, n2) = term.participants
        forward = (a, b, tuple(y - x for x, y in zip(n1, n2)))
        backward = (b, a, tuple(x - y for x, y in zip(n1, n2)))
        if backward < forward:
            return (BILINEAR,) + backward, _float_bytes(term.coefficient.T)
        return (BILINEAR,) + forward, _float_bytes(term.coefficient)
    return (term.kind,) + term.participants, _float_bytes(term.coefficient)


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def validate_spin_model(model: SpinModel) -> None:
    """Check a model against the transfer contract.

    All violations are collected and reported together.

    Raises
    ------
    SpinModelError
        If any violation is found.
    """
    violations = []
    violations += _check_header(model)
    violations += _check_lattice(model.lattice)
    violations += _check_sites(model.sites)
    violations += _check_terms(model.terms, {site.id for site in model.sites})
    if violations:
        raise SpinModelError(violations)


def _check_header(model):
    out = []
    if model.schema_version != SCHEMA_VERSION:
        out.append(f"schema_version {model.schema_version} is not {SCHEMA_VERSION}")
    model_id = model.metadata.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        out.append("metadata.model_id must be a non-empty string")
    return out


def _check_lattice(lattice):
    if lattice.shape != (2, 2):
        return [f"lattice must have shape (2, 2), got {lattice.shape}"]
    if not np.all(np.isfinite(lattice)):
        return ["lattice has non-finite entries"]
    scale = np.prod(np.linalg.norm(lattice, axis=1))
    if scale == 0 or abs(np.linalg.det(lattice)) <= LATTICE_TOLERANCE * scale:
        return ["lattice vectors are not linearly independent"]
    return []


def _check_sites(sites):
    out = []
    if not sites:
        return ["model has no sites"]
    seen = set()
    for site in sites:
        if not isinstance(site, Site):
            out.append(f"site {site!r} is not a Site")
            continue
        if not isinstance(site.id, str) or not site.id:
            out.append(f"site id {site.id!r} must be a non-empty string")
        if site.id in seen:
            out.append(f"site id {site.id!r} is duplicated")
        seen.add(site.id)
        if site.position.shape != (2,) or not np.all(np.isfinite(site.position)):
            out.append(f"site {site.id!r}: position must be finite with shape (2,)")
        two_s = 2 * site.spin
        if not np.isfinite(two_s) or site.spin <= 0 or abs(two_s - round(two_s)) > LATTICE_TOLERANCE:
            out.append(f"site {site.id!r}: spin {site.spin} must be a positive integer or half-integer")
    return out


def _check_terms(terms, site_ids):
    out = []
    bilinear_keys = {}
    zeeman_sites = set()
    onsite_sites = set()
    for index, term in enumerate(terms):
        name = f"term {index}" + (f" ({term.label})" if term.label else "")
        if not isinstance(term, Term):
            out.append(f"{name} is not a Term")
            continue
        if term.kind not in SUPPORTED_KINDS:
            out.append(f"{name}: kind {term.kind!r} is not supported "
                       f"(supported: {', '.join(SUPPORTED_KINDS)})")
            continue
        if len(term.participants) != SUPPORTED_KINDS[term.kind]:
            out.append(f"{name}: {term.kind} needs {SUPPORTED_KINDS[term.kind]} "
                       f"participant(s), got {len(term.participants)}")
            continue
        if term.coefficient.shape != (3, 3):
            out.append(f"{name}: coefficient must have shape (3, 3), got {term.coefficient.shape}")
        elif not np.all(np.isfinite(term.coefficient)):
            out.append(f"{name}: coefficient has non-finite entries")
        valid = True
        for site, cell in term.participants:
            if site not in site_ids:
                out.append(f"{name}: endpoint {site!r} is not a site of the model")
                valid = False
            if len(cell) != 2 or not all(isinstance(x, int) for x in cell):
                out.append(f"{name}: cell offset {cell} must be two exact integers")
                valid = False
        if not valid:
            continue
        if term.kind == BILINEAR:
            (a, n1), (b, n2) = term.participants
            if a == b and n1 == n2:
                out.append(f"{name}: both participants are the same physical site "
                           "(use Term.onsite for a same-site term)")
                continue
            key, _ = _canonical_term(term)
            if key in bilinear_keys:
                out.append(f"{name}: duplicates term {bilinear_keys[key]} "
                           "(same bond up to translation or reversal)")
            else:
                bilinear_keys[key] = index
        elif term.kind in (ZEEMAN, ONSITE):
            site, cell = term.participants[0]
            if cell != (0, 0):
                out.append(f"{name}: {term.kind} cell offset must be (0, 0), got {cell}")
            seen = zeeman_sites if term.kind == ZEEMAN else onsite_sites
            if site in seen:
                out.append(f"{name}: site {site!r} has more than one {term.kind} term")
            seen.add(site)
            if term.kind == ONSITE and term.coefficient.shape == (3, 3):
                A = term.coefficient
                if np.max(np.abs(A - A.T)) > LATTICE_TOLERANCE * max(1.0, np.max(np.abs(A))):
                    out.append(f"{name}: onsite matrix must be symmetric (its antisymmetric "
                               "part gives a non-Hermitian operator)")
    return out


def onsite_renormalization(spin: float) -> float:
    """Factor ``1 - 1/(2S)`` applied to onsite coefficients by classical and LSWT methods (D37).

    In a spin coherent state ``<S^a S^b + S^b S^a>/2 = S(S - 1/2) n_a n_b +
    (S/2) delta_ab``, so the direction-dependent part of ``S^T A S`` is
    ``(1 - 1/(2S)) S^2 n^T A n`` and the rest is the constant ``(S/2) tr A``.
    The same factor makes the LSWT single-ion gap exact (``(2S - 1)|D|`` for
    ``D (S^z)^2``) and removes any onsite effect at ``S = 1/2``.
    """
    return 1.0 - 1.0 / (2.0 * spin)
