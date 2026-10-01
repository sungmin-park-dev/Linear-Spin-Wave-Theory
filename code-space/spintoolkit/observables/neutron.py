"""Unpolarized neutron intensity from the LSWT structure factor (D41).

Convention
----------
For magnetic moments ``mu_i = -mu_B g_i S_i`` the unpolarized cross section
per magnetic site is (Squires, ch. 7; Boothroyd, ch. 6)

    d^2 sigma / dOmega dE = (gamma r_0)^2 (k_f / k_i) exp(-2 W) I(Q, w),

    I(Q, w) = sum_ab (delta_ab - Q_a Q_b / Q^2) S_M^{ab}(Q, w),

where ``S_M`` is the structure factor of :mod:`structure_factor` with every
spin ``S_i`` replaced by the moment ``(g_i / 2) F_i(|Q|) S_i``. With ``g = 2``
and ``F = 1`` this is ``StructureFactor.neutron()``. The prefactor
``(gamma r_0)^2 k_f / k_i exp(-2 W)`` depends on the instrument and the
sample, not on the spin model, and is left out.

The models are two-dimensional (one magnetic layer, no interlayer coupling),
so ``S_M`` depends only on the in-plane momentum ``Q_par = (Q_x, Q_y)``. The
out-of-plane component ``Q_z`` enters only the polarization factor and
``|Q|`` in the form factor. Spin components are in the global spin frame,
which is the Cartesian frame of the lattice with ``z`` normal to the layer.

Momenta are Cartesian, in inverse model length units (those of
``model.lattice``); ``length_unit`` converts to inverse angstrom for the form
factor: ``|Q| [1/A] = |Q| / length_unit``.

The magnetic form factor is the dipole approximation
``F(Q) = <j0(s)> + c <j2(s)>`` with ``s = Q / 4 pi`` (Q in 1/A) and the
expansions ``<j0> = A e^{-a s^2} + B e^{-b s^2} + C e^{-c s^2} + D`` and
``<j2> = s^2 (A e^{-a s^2} + B e^{-b s^2} + C e^{-c s^2} + D)`` of Brown,
International Tables for Crystallography C, section 4.4.5. For a free ion
``c = (2 - g_J) / g_J``; the default ``c = 0`` is the spin-only value. For
effective spin-1/2 ions with unquenched orbital moment (Co2+ in octahedral
fields) neither value is exact, so ``c`` is left to the user.

Energy resolution is a Gaussian or Lorentzian whose full width at half
maximum may depend on the energy transfer. Momentum resolution is not
modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from spintoolkit.observables.structure_factor import (
    StructureFactor, spiral_structure_factor, structure_factor)

#: ``<j0>`` and ``<j2>`` coefficients (A, a, B, b, C, c, D) of 3d ions,
#: Brown, International Tables for Crystallography C, section 4.4.5. Only ions
#: on which the tables of Sunny.jl (``src/FormFactor.jl``) and periodictable
#: (``magnetic_ff.py``) agree to 2e-3 are included.
FORM_FACTOR_COEFFICIENTS: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "Ti1": ((0.5093, 36.7033, 0.5032, 10.3713, -0.0263, 0.3106, 0.0116),
            (6.1567, 27.2754, 2.6833, 8.9827, 0.4070, 3.0524, 0.0011)),
    "Ti2": ((0.5091, 24.9763, 0.5162, 8.7569, -0.0281, 0.9160, 0.0015),
            (4.3107, 18.3484, 2.0960, 6.7970, 0.2984, 2.5476, 0.0007)),
    "Ti3": ((0.3571, 22.8413, 0.6688, 8.9306, -0.0354, 0.4833, 0.0099),
            (3.3717, 14.4441, 1.8258, 5.7126, 0.2470, 2.2654, 0.0005)),
    "V1": ((0.4444, 32.6479, 0.5683, 9.0971, -0.2285, 0.0218, 0.2150),
           (4.7474, 23.3226, 2.3609, 7.8082, 0.4105, 2.7063, 0.0014)),
    "V2": ((0.4085, 23.8526, 0.6091, 8.2456, -0.1676, 0.0415, 0.1496),
           (3.4386, 16.5303, 1.9638, 6.1415, 0.2997, 2.2669, 0.0009)),
    "V3": ((0.3598, 19.3364, 0.6632, 7.6172, -0.3064, 0.0296, 0.2835),
           (2.3005, 14.6821, 2.0364, 6.1304, 0.4099, 2.3815, 0.0014)),
    "V4": ((0.3106, 16.8160, 0.7198, 7.0487, -0.0521, 0.3020, 0.0221),
           (1.8377, 12.2668, 1.8247, 5.4578, 0.3979, 2.2483, 0.0012)),
    "Cr1": ((-0.0977, 0.0470, 0.4544, 26.0054, 0.5579, 7.4892, 0.0831),
            (3.7768, 20.3456, 2.1028, 6.8926, 0.4010, 2.4114, 0.0017)),
    "Cr2": ((1.2024, -0.0055, 0.4158, 20.5475, 0.6032, 6.9560, -1.2218),
            (2.6422, 16.0598, 1.9198, 6.2531, 0.4446, 2.3715, 0.0020)),
    "Cr3": ((-0.3094, 0.0274, 0.3680, 17.0355, 0.6559, 6.5236, 0.2856),
            (1.6262, 15.0656, 2.0618, 6.2842, 0.5281, 2.3680, 0.0023)),
    "Cr4": ((-0.2320, 0.0433, 0.3101, 14.9518, 0.7182, 6.1726, 0.2042),
            (1.0293, 13.9498, 1.9933, 6.0593, 0.5974, 2.3457, 0.0027)),
    "Mn1": ((-0.0138, 0.4213, 0.4231, 24.6680, 0.5905, 6.6545, -0.0010),
            (3.2953, 18.6950, 1.8792, 6.2403, 0.3927, 2.2006, 0.0022)),
    "Mn2": ((0.4220, 17.6840, 0.5948, 6.0050, 0.0043, -0.6090, -0.0219),
            (2.0515, 15.5561, 1.8841, 6.0625, 0.4787, 2.2323, 0.0027)),
    "Mn3": ((0.4198, 14.2829, 0.6054, 5.4689, 0.9241, -0.0088, -0.9498),
            (1.2427, 14.9966, 1.9567, 6.1181, 0.5732, 2.2577, 0.0031)),
    "Mn4": ((0.3760, 12.5661, 0.6602, 5.1329, -0.0372, 0.5630, 0.0011),
            (0.7879, 13.8857, 1.8717, 5.7433, 0.5981, 2.1818, 0.0034)),
    "Fe1": ((0.1251, 34.9633, 0.3629, 15.5144, 0.5223, 5.5914, -0.0105),
            (2.6290, 18.6598, 1.8704, 6.3313, 0.4690, 2.1628, 0.0031)),
    "Fe2": ((0.0263, 34.9597, 0.3668, 15.9435, 0.6188, 5.5935, -0.0119),
            (1.6490, 16.5593, 1.9064, 6.1325, 0.5206, 2.1370, 0.0035)),
    "Fe3": ((0.3972, 13.2442, 0.6295, 4.9034, -0.0314, 0.3496, 0.0044),
            (1.3602, 11.9976, 1.5188, 5.0025, 0.4705, 1.9914, 0.0038)),
    "Fe4": ((0.3782, 11.3800, 0.6556, 4.5920, -0.0346, 0.4833, 0.0005),
            (1.5582, 8.2750, 1.1863, 3.2794, 0.1366, 1.1068, -0.0022)),
    "Co1": ((0.0990, 33.1252, 0.3645, 15.1768, 0.5470, 5.0081, -0.0109),
            (2.4097, 16.1608, 1.5780, 5.4604, 0.4095, 1.9141, 0.0031)),
    "Co2": ((0.4332, 14.3553, 0.5857, 4.6077, -0.0382, 0.1338, 0.0179),
            (1.9049, 11.6444, 1.3159, 4.3574, 0.3146, 1.6453, 0.0017)),
    "Co3": ((0.3902, 12.5078, 0.6324, 4.4574, -0.1500, 0.0343, 0.1272),
            (1.7058, 8.8595, 1.1409, 3.3086, 0.1474, 1.0899, -0.0025)),
    "Co4": ((0.3515, 10.7785, 0.6778, 4.2343, -0.0389, 0.2409, 0.0098),
            (1.3110, 8.0252, 1.1551, 3.1792, 0.1608, 1.1301, -0.0011)),
    "Ni1": ((0.0705, 35.8561, 0.3984, 13.8042, 0.5427, 4.3965, -0.0118),
            (2.1040, 14.8655, 1.4302, 5.0714, 0.4031, 1.7784, 0.0034)),
    "Ni2": ((0.0163, 35.8826, 0.3916, 13.2233, 0.6052, 4.3388, -0.0133),
            (1.7080, 11.0160, 1.2147, 4.1031, 0.3150, 1.5334, 0.0018)),
    "Ni4": ((-0.0090, 35.8614, 0.2776, 11.7904, 0.7474, 4.2011, -0.0163),
            (1.1612, 7.7000, 1.0027, 3.2628, 0.2719, 1.3780, 0.0025)),
    "Cu1": ((0.0749, 34.9656, 0.4147, 11.7642, 0.5238, 3.8497, -0.0127),
            (1.8814, 13.4333, 1.2809, 4.5446, 0.3646, 1.6022, 0.0033)),
    "Cu2": ((0.0232, 34.9686, 0.4023, 11.5640, 0.5882, 3.8428, -0.0137),
            (1.5189, 10.4779, 1.1512, 3.8132, 0.2918, 1.3979, 0.0017)),
    "Cu3": ((0.0031, 34.9074, 0.3582, 10.9138, 0.6531, 3.8279, -0.0147),
            (1.2797, 8.4502, 1.0315, 3.2796, 0.2401, 1.2498, 0.0015)),
    "Cu4": ((-0.0132, 30.6817, 0.2801, 11.1626, 0.7490, 3.8172, -0.0165),
            (0.9568, 7.4481, 0.9099, 3.3964, 0.3729, 1.4936, 0.0049)),
}


def _expansion(coefficients: Sequence[float], s2: np.ndarray) -> np.ndarray:
    A, a, B, b, C, c, D = coefficients
    return A * np.exp(-a * s2) + B * np.exp(-b * s2) + C * np.exp(-c * s2) + D


@dataclass(frozen=True)
class FormFactor:
    """Magnetic form factor ``F(Q) = <j0> + j2_weight <j2>`` (dipole approximation).

    Parameters
    ----------
    j0, j2 : tuple of 7 floats
        Coefficients ``(A, a, B, b, C, c, D)`` of the Brown expansions.
    j2_weight : float
        ``c`` in ``<j0> + c <j2>``; ``(2 - g_J) / g_J`` for a free ion,
        zero for a spin-only moment.
    label : str
    """

    j0: Tuple[float, ...]
    j2: Tuple[float, ...] = (0.0,) * 7
    j2_weight: float = 0.0
    label: str = ""

    @classmethod
    def from_ion(cls, label: str, j2_weight: float = 0.0) -> "FormFactor":
        """Tabulated ion, e.g. ``"Co2"`` for Co2+ (see ``FORM_FACTOR_COEFFICIENTS``)."""
        if label not in FORM_FACTOR_COEFFICIENTS:
            raise KeyError(f"no form factor for {label!r}; available: "
                           f"{', '.join(FORM_FACTOR_COEFFICIENTS)}")
        j0, j2 = FORM_FACTOR_COEFFICIENTS[label]
        return cls(j0, j2, float(j2_weight), label)

    @classmethod
    def point(cls) -> "FormFactor":
        """``F = 1`` (point moment)."""
        return cls((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), label="point")

    def __call__(self, Q: Any) -> np.ndarray:
        """``F(|Q|)`` for ``|Q|`` in inverse angstrom."""
        s2 = (np.asarray(Q, dtype=float) / (4 * np.pi)) ** 2
        value = _expansion(self.j0, s2)
        if self.j2_weight:
            value = value + self.j2_weight * s2 * _expansion(self.j2, s2)
        return value


SiteValue = Union[Any, Mapping[str, Any]]


def _per_site(value: SiteValue, site_ids: Sequence[str], default: Any, name: str) -> list:
    if value is None:
        return [default] * len(site_ids)
    if isinstance(value, Mapping):
        missing = sorted(set(site_ids) - set(value))
        if missing:
            raise KeyError(f"{name} has no entry for sites {missing}")
        return [value[s] for s in site_ids]
    return [value] * len(site_ids)


def _g_tensor(g: Any) -> np.ndarray:
    g = np.asarray(g, dtype=float)
    if g.ndim == 0:
        return g * np.eye(3)
    if g.shape != (3, 3):
        raise ValueError("g must be a scalar or a 3 x 3 tensor")
    return g


def _lab_result(result):
    return getattr(result, "rotating", result)


def _as_3d(Q: Any) -> np.ndarray:
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    if Q.shape[1] == 2:
        Q = np.column_stack([Q, np.zeros(len(Q))])
    if Q.shape[1] != 3:
        raise ValueError("Q must have shape (nq, 2) or (nq, 3)")
    return Q


@dataclass(frozen=True)
class NeutronIntensity:
    """Mode-resolved unpolarized neutron intensity ``I(Q, w)``.

    Attributes
    ----------
    Q : (nq, 3) array
        Cartesian momentum transfer (inverse model length units).
    energies : (nq, nmodes) array
        Signed mode energies as in :class:`StructureFactor`.
    intensities : (nq, nmodes) array
        ``sum_ab (delta_ab - Q_a Q_b / Q^2) W_M^{ab}`` of each mode, per site
        (NaN where ``H(Q_par)`` has a zero mode, and at ``Q = 0``).
    elastic : (nq,) array
        Bragg intensity with the same factors, the coefficient of
        ``N delta_{Q_par, G} delta(w)``.
    structure_factor : StructureFactor
        Moment structure factor ``S_M`` at ``Q_par``.
    """

    Q: np.ndarray
    energies: np.ndarray
    intensities: np.ndarray
    elastic: np.ndarray
    structure_factor: StructureFactor

    def broaden(self, omega: Sequence[float], fwhm: Union[float, Callable[[np.ndarray], Any]],
                shape: str = "gaussian") -> np.ndarray:
        """Inelastic ``I(Q, w)`` on a frequency grid, shape (nq, nw).

        Parameters
        ----------
        omega : array_like
            Energy transfers.
        fwhm : float or callable
            Full width at half maximum of the energy resolution, or a
            function of the mode energy (instrumental resolution ``FWHM(w)``,
            evaluated at ``|w_n|``).
        shape : {"gaussian", "lorentzian"}
        """
        omega = np.asarray(omega, dtype=float)
        width = fwhm(np.abs(self.energies)) if callable(fwhm) else fwhm
        width = np.broadcast_to(np.asarray(width, dtype=float), self.energies.shape)
        if np.any(~(width > 0)):
            raise ValueError("fwhm must be positive")
        x = omega[None, None, :] - self.energies[:, :, None]
        w = width[:, :, None]
        if shape == "gaussian":
            sigma = w / (2 * np.sqrt(2 * np.log(2)))
            profile = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        elif shape == "lorentzian":
            eta = w / 2
            profile = eta / np.pi / (x ** 2 + eta ** 2)
        else:
            raise ValueError("shape must be 'gaussian' or 'lorentzian'")
        return np.einsum("qn,qnw->qw", self.intensities, profile)


def neutron_intensity(result, Q, temperature: float = 0.0, *, g: SiteValue = None,
                      form_factor: SiteValue = None, length_unit: float = 1.0,
                      zero_modes=None, gapless: Optional[bool] = None) -> NeutronIntensity:
    """Unpolarized neutron intensity of an LSWT result at momentum transfers ``Q``.

    Parameters
    ----------
    result : LSWTResult or SpiralLSWTResult
    Q : (nq, 3) or (nq, 2) array_like
        Cartesian momentum transfer in inverse model length units; a missing
        ``Q_z`` is zero.
    temperature : float
        ``t = k_B T / E0`` (Bose factors, D25 zero-mode decision).
    g : scalar, (3, 3) array or mapping site id -> either, optional
        g-tensor in the global spin frame, moment ``mu = -mu_B g S``. Default 2
        (spin-only). Use the tensor of the model's Zeeman term when the field
        was given as ``mu_B B``.
    form_factor : FormFactor or mapping site id -> FormFactor, optional
        Default ``FormFactor.point()`` (F = 1).
    length_unit : float
        Length of one model unit in angstrom (the lattice constant when the
        lattice has unit nearest-neighbour distance).
    zero_modes, gapless : optional
        As in :func:`~spintoolkit.observables.structure_factor.structure_factor`.

    Returns
    -------
    NeutronIntensity
    """
    Q = _as_3d(Q)
    if not length_unit > 0:
        raise ValueError("length_unit must be positive")
    lab = _lab_result(result)
    site_ids = [key[0] for key in lab.site_keys]
    g_sites = [_g_tensor(x) for x in _per_site(g, site_ids, 2.0, "g")]
    f_sites = _per_site(form_factor, site_ids, FormFactor.point(), "form_factor")
    norm = np.linalg.norm(Q, axis=1)
    F = np.array([f(norm / length_unit) for f in f_sites]).T                # (nq, Ns)
    tensors = 0.5 * F[:, :, None, None] * np.array(g_sites)[None]
    solver = spiral_structure_factor if hasattr(result, "rotating") else structure_factor
    sf = solver(result, Q[:, :2], temperature, zero_modes, gapless, moment_tensors=tensors)
    projector = np.full((len(Q), 3, 3), np.nan)
    nonzero = norm > 0
    unit = Q[nonzero] / norm[nonzero, None]
    projector[nonzero] = np.eye(3)[None] - unit[:, :, None] * unit[:, None, :]
    intensities = np.real(np.einsum("qab,qnab->qn", projector, sf.weights))
    elastic = np.real(np.einsum("qab,qab->q", projector, sf.elastic))
    elastic[~sf.bragg] = 0.0
    return NeutronIntensity(Q, sf.energies, intensities, elastic, sf)


def _check_domain_rotation(R: np.ndarray, lattice: np.ndarray) -> None:
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-9):
        raise ValueError("domain operations must be orthogonal 3 x 3 matrices")
    if not (np.allclose(R[2, :2], 0, atol=1e-9) and np.allclose(R[:2, 2], 0, atol=1e-9)):
        raise ValueError("domain operations must map the layer onto itself")
    image = lattice @ R[:2, :2].T @ np.linalg.inv(lattice)
    if not np.allclose(image, np.rint(image), atol=1e-8):
        raise ValueError("domain operation does not map the lattice onto itself")


def domain_average(result, Q, omega, fwhm, rotations: Sequence[Any],
                   weights: Optional[Sequence[float]] = None, shape: str = "gaussian",
                   **kwargs) -> np.ndarray:
    """Broadened intensity averaged over symmetry-related domains, shape (nq, nw).

    A domain obtained from the computed state by a point operation ``R`` of
    the crystal (acting on positions and, as an axial vector, on spins) has
    ``I_R(Q, w) = I(R^{-1} Q, w)``: the polarization factor is invariant and
    the improper sign of the axial vector cancels in the quadratic form.
    This holds only when ``R`` is a symmetry of the Hamiltonian, the
    g-tensors and the field; that is not checked here, only that ``R``
    maps the layer and its lattice onto themselves.

    Parameters
    ----------
    result : LSWTResult or SpiralLSWTResult
    Q, omega, fwhm, shape
        As in :func:`neutron_intensity` and :meth:`NeutronIntensity.broaden`.
    rotations : sequence of (3, 3) arrays
        Point operations ``R`` in Cartesian coordinates; include the identity
        for the computed domain.
    weights : sequence of float, optional
        Domain populations (normalized here); equal by default.
    **kwargs
        Passed to :func:`neutron_intensity`.
    """
    Q = _as_3d(Q)
    lattice = np.asarray(_lab_result(result).lattice, dtype=float)
    rotations = [np.asarray(R, dtype=float) for R in rotations]
    for R in rotations:
        _check_domain_rotation(R, lattice)
    w = np.ones(len(rotations)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(rotations),) or np.any(w < 0) or w.sum() == 0:
        raise ValueError("weights must be non-negative, one per rotation")
    w = w / w.sum()
    total = 0.0
    for R, weight in zip(rotations, w):
        # Rows: (R^{-1} Q)^T = Q^T R for orthogonal R.
        total = total + weight * neutron_intensity(result, Q @ R, **kwargs).broaden(
            omega, fwhm, shape)
    return total


def sphere_directions(num: int) -> np.ndarray:
    """``num`` nearly uniform unit vectors (Fibonacci lattice), shape (num, 3)."""
    if num < 1:
        raise ValueError("num must be positive")
    i = np.arange(num) + 0.5
    z = 1 - 2 * i / num
    phi = np.pi * (1 + np.sqrt(5)) * i
    r = np.sqrt(1 - z ** 2)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def powder_average(result, Q_magnitudes, omega, fwhm, num_directions: int = 500,
                   shape: str = "gaussian", **kwargs) -> np.ndarray:
    """Spherically averaged inelastic intensity ``I(|Q|, w)``, shape (nQ, nw).

    The direction of ``Q`` is averaged over the unit sphere with a Fibonacci
    lattice of ``num_directions`` points (equal weights). For a layer the
    in-plane momentum is ``|Q| sin(theta)`` and ``Q_z = |Q| cos(theta)``.
    Elastic (Bragg) intensity is not included. Directions where the
    spectrum has a zero mode give NaN and are reported by a warning of
    :func:`neutron_intensity`; they are excluded from the average.

    Parameters
    ----------
    result : LSWTResult or SpiralLSWTResult
    Q_magnitudes : array_like
        ``|Q|`` in inverse model length units.
    omega, fwhm, shape
        As in :meth:`NeutronIntensity.broaden`.
    num_directions : int
    **kwargs
        Passed to :func:`neutron_intensity`.
    """
    Q_magnitudes = np.atleast_1d(np.asarray(Q_magnitudes, dtype=float))
    directions = sphere_directions(num_directions)
    Q = (Q_magnitudes[:, None, None] * directions[None]).reshape(-1, 3)
    spectra = neutron_intensity(result, Q, **kwargs).broaden(omega, fwhm, shape)
    spectra = spectra.reshape(len(Q_magnitudes), num_directions, -1)
    return np.nanmean(spectra, axis=1)
