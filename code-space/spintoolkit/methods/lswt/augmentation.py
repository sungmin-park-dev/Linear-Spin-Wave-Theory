"""Minimally augmented spin-wave theory (MAGSWT, D62).

LSWT about a classically unstable state has no harmonic vacuum: ``H(k)`` has a
negative direction and the zero-point energy is undefined (D61). MAGSWT
(Chernyshev, arXiv:2509.08877 section II, and its Refs. 34-36) instead
solves the augmented Hamiltonian

    H' = H + mu sum_i (S_i - S_i . n_i) = H + mu sum_i a_i^dagger a_i,

with ``n_i`` the classical direction of spin ``i``. The added term is exact in
Holstein-Primakoff bosons, leaves the classical energy unchanged and adds
``mu`` to the diagonal of every ``H(k)``. It is a positive operator for
``mu >= 0``, so ``E_gs(H) <= E_gs(H')``; the reported energy is the O(S)
estimate of ``E_gs(H')`` and is labelled an upper bound in that sense (not a
rigorous variational bound of the spin model).

Criterion. ``mu`` is the smallest shift that makes ``H(k) + mu`` positive
semidefinite at every continuous ``k``: ``mu_min = max(0, -min_k lambda_min
H(k))``. Only then is the boson vacuum the ground state of the augmented
quadratic Hamiltonian. The criterion of arXiv:2509.08877 (real magnon
energies, ``eps^2 > 0`` for ``g H``) agrees with it when the instability is
an imaginary mode but allows negative-energy (Landau-unstable) magnons; the
smallest ``mu`` with a real spectrum on the sample momenta is reported for
comparison as ``mu_real_spectrum``.

The minimum over ``k`` is searched as in :func:`~spintoolkit.methods.stability.local_stability`
(D51): the lowest points of an unshifted and a half-step shifted seed mesh,
high-symmetry points and the calculation momenta, refined by Nelder-Mead.
At ``mu = mu_min`` the augmented ``H(k)`` has a zero mode at the minimizing
``k``; the zero-point energy is continuous in ``mu`` with derivative equal
to the mean boson number, so the calculation uses ``mu_min`` plus a padding
of ``AUGMENTATION_PADDING`` times the spectral scale and records the
resulting energy offset bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy.optimize import minimize

from spintoolkit.definitions.defaults import (
    AUGMENTATION_BISECTION_STEPS, AUGMENTATION_PADDING, STABILITY_MESH, STABILITY_SEEDS,
    STABILITY_ZERO_TOLERANCE)


@dataclass(frozen=True)
class Augmentation:
    """Shift of the boson chemical potential used by MAGSWT (energy unit E0).

    Attributes
    ----------
    mu : float
        Shift added to the diagonal of every ``H(k)`` (``mu_min`` plus padding
        for ``"minimal"``; the requested value otherwise).
    mu_min : float
        ``max(0, -min_k lambda_min H(k))`` over continuous ``k``.
    k_min : (2,) array
        Momentum (magnetic reciprocal basis) of the lowest eigenvalue.
    lowest_eigenvalue : float
        ``min_k lambda_min H(k)`` before augmentation.
    scale : float
        Largest ``|eig H(k)|`` on the seed mesh.
    mu_real_spectrum : float
        Smallest shift for which every eigenvalue of ``g (H(k) + mu)`` is
        real on the sample momenta (the eps^2 criterion of
        arXiv:2509.08877), for comparison; at most ``mu_min``.
    requested : str or float
        ``"minimal"`` or the requested ``mu``.
    """

    mu: float
    mu_min: float
    k_min: np.ndarray
    lowest_eigenvalue: float
    scale: float
    mu_real_spectrum: float
    requested: Any

    @property
    def label(self) -> str:
        return f"upper bound (MAGSWT, mu={self.mu:.6g})"

    def as_dict(self) -> Dict[str, Any]:
        return {"mu": self.mu, "mu_min": self.mu_min, "k_min": np.asarray(self.k_min).tolist(),
                "lowest_eigenvalue": self.lowest_eigenvalue, "scale": self.scale,
                "mu_real_spectrum": self.mu_real_spectrum, "requested": self.requested,
                "label": self.label}


def _real_spectrum(H: np.ndarray, mu: float, tolerance: float) -> bool:
    n2 = H.shape[1]
    g = np.r_[np.ones(n2 // 2), -np.ones(n2 // 2)]
    w = np.linalg.eigvals(g[None, :, None] * (H + mu * np.eye(n2)))
    return bool(np.all(np.abs(w.imag) <= tolerance))


def find_augmentation(hamiltonian, magnetic_lattice: np.ndarray, momenta: np.ndarray,
                      requested, *, mesh: int = STABILITY_MESH,
                      seeds: int = STABILITY_SEEDS) -> Augmentation:
    """Minimal MAGSWT shift for the quadratic Hamiltonian ``hamiltonian``.

    Parameters
    ----------
    hamiltonian : QuadraticBoseHamiltonian
    magnetic_lattice : (2, 2) array
        Magnetic lattice vectors (rows).
    momenta : (m, 2) array
        Cartesian momenta of the calculation; they are among the samples.
    requested : "minimal" or float
        A float ``mu`` is checked against ``mu_min``.
    mesh, seeds : int
        Seed mesh per direction and number of Nelder-Mead starts.

    Returns
    -------
    Augmentation

    Raises
    ------
    ValueError
        If a requested ``mu`` is below ``mu_min`` (the augmented ``H(k)``
        still has a negative direction somewhere).
    """
    reciprocal = 2 * np.pi * np.linalg.inv(magnetic_lattice).T

    def to_k(p):
        return np.asarray(p, dtype=float) @ reciprocal

    grid = np.array([(i / mesh, j / mesh) for i in range(mesh) for j in range(mesh)])
    special = np.array([(a, b) for a in (0, 0.5, 1 / 3, 2 / 3) for b in (0, 0.5, 1 / 3, 2 / 3)])
    calc = np.asarray(momenta, dtype=float) @ magnetic_lattice.T / (2 * np.pi)
    samples = np.vstack([special, grid, grid + 0.5 / mesh, calc])
    H = hamiltonian.at(to_k(samples))
    eig = np.linalg.eigvalsh(H)
    lowest, scale = eig[:, 0], max(float(np.max(np.abs(eig))), 1e-300)

    def value(p) -> float:
        return float(np.linalg.eigvalsh(hamiltonian.at(to_k(np.atleast_2d(p))))[0, 0])

    best_value, best_p = float(lowest.min()), samples[int(np.argmin(lowest))]
    step = 1.0 / mesh
    starts = []
    for i in np.argsort(lowest):
        d = [np.linalg.norm(np.mod(samples[i] - s + 0.5, 1.0) - 0.5) for s in starts]
        if all(x > 1.5 * step for x in d):
            starts.append(samples[i])
        if len(starts) >= seeds:
            break
    for p0 in starts:
        simplex = np.array([p0, p0 + [step, 0], p0 + [0, step]])
        opt = minimize(value, p0, method="Nelder-Mead",
                       options={"initial_simplex": simplex, "xatol": 1e-10, "fatol": 1e-18,
                                "maxiter": 400})
        if opt.fun < best_value:
            best_value, best_p = float(opt.fun), np.mod(opt.x, 1.0)

    negative = best_value < -STABILITY_ZERO_TOLERANCE * scale
    mu_min = -best_value if negative else 0.0
    if requested == "minimal":
        mu = mu_min + AUGMENTATION_PADDING * scale if negative else 0.0
    else:
        mu = float(requested)
        if mu < mu_min:
            raise ValueError(f"augmentation mu={mu:.6g} is below the minimal shift "
                             f"mu_min={mu_min:.6g} (lowest eigenvalue of H(k) at "
                             f"q={np.round(best_p, 6).tolist()}): the augmented H(k) is "
                             "still not positive semidefinite")
    mu_real = 0.0
    if negative:
        H_all = np.concatenate([H, hamiltonian.at(to_k(np.atleast_2d(best_p)))])
        tolerance = 1e-8 * scale
        if not _real_spectrum(H_all, 0.0, tolerance):
            lo, hi = 0.0, mu_min + AUGMENTATION_PADDING * scale
            for _ in range(AUGMENTATION_BISECTION_STEPS):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if _real_spectrum(H_all, mid, tolerance) else (mid, hi)
            mu_real = hi
    return Augmentation(float(mu), float(mu_min), np.asarray(best_p, dtype=float),
                        float(best_value), scale, float(mu_real), requested)
