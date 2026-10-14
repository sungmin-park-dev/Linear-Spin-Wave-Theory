"""Dynamical structure factor with dressed magnons and the two-magnon continuum (D58).

``S(q, w) = sum_n W_n(q) A_n(q, w) + S_2(q, w)``:

- ``W_n(q)``: LSWT one-magnon weights (:func:`~spintoolkit.observables.structure_factor.structure_factor`, O(S));
- ``A_n(q, w)``: magnon spectral function with the Hartree-Fock + tadpole shift
  and the frequency-dependent cubic self-energy
  (:meth:`~spintoolkit.methods.nlswt.NLSWTResult.spectral_function`), which
  broadens decaying magnons and moves part of their weight into the
  two-magnon continuum;
- ``S_2(q, w)``: the non-interacting longitudinal two-magnon continuum
  (:func:`~spintoolkit.observables.two_magnon.two_magnon_continuum`, O(S^0)).

Scope: this is dressed LSWT plus the non-interacting continuum, not the
complete order-S^0 structure factor. The 1/S corrections of the one-magnon
weights and, in noncollinear states, the one-magnon/two-magnon interference
term mixed by the cubic vertex are of the same order and are not included.
Zero temperature, lab-frame (commensurate) states.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import warnings

import numpy as np

from spintoolkit.observables.structure_factor import structure_factor
from spintoolkit.observables.two_magnon import two_magnon_continuum


def _weight(factor, component: Optional[str]) -> np.ndarray:
    if component is None:
        return factor.trace()
    if component == "neutron":
        return factor.neutron()
    a, b = ("xyz".index(component[0]), "xyz".index(component[1]))
    return np.real(factor.weights[:, :, a, b])


def dressed_structure_factor(lswt_result, nlswt_result, q_points, omega, eta: float,
                             component: Optional[str] = None, mode: str = "on_shell",
                             grid: Tuple[int, int] = (64, 64), levels: int = 400,
                             two_magnon_mesh: Tuple[int, int] = (24, 24)) -> Dict[str, np.ndarray]:
    """``S(q, w)`` from dressed one-magnon peaks plus the two-magnon continuum (D58).

    Parameters
    ----------
    lswt_result : LSWTResult
        Lab-frame LSWT result of the same model and state.
    nlswt_result : NLSWTResult
        Result of :func:`~spintoolkit.methods.nlswt.solve_nlswt` for that state.
    q_points : (nq, 2) array_like
        Cartesian momenta, away from Bragg vectors and other zero modes.
    omega : (nw,) array_like
    eta : float
        Instrumental Lorentzian half width (also used for the continuum).
    component : None, "neutron" or "xx", "zz", ...
        As in :meth:`StructureFactor.spectrum`.
    mode : "on_shell" or "dyson"
        Line shape of the dressed magnons, as in
        :meth:`~spintoolkit.methods.nlswt.NLSWTResult.spectral_function`.
    grid, levels :
        Integration settings of the self-energy.
    two_magnon_mesh :
        Mesh of the continuum.

    Returns
    -------
    dict
        ``one_magnon``, ``two_magnon`` and ``total``, each (nq, nw); ``lswt``
        is the undressed one-magnon spectrum broadened by ``eta`` for comparison.
        With ``mode="dyson"``, ``negative_frequency_weight`` (nq, Ns) of the
        spectral functions flags unphysical poles (positive values).

    Notes
    -----
    Each LSWT mode is matched to the band of the same energy in the
    nonlinear calculation; a mode without such a band raises ValueError.
    """
    q = np.atleast_2d(np.asarray(q_points, dtype=float))
    w = np.asarray(omega, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        factor = structure_factor(lswt_result, q)
    if np.any(factor.zero_mode):
        raise ValueError("q_points must avoid zero modes (Bragg vectors): the dressed "
                         "spectrum is not defined there")
    weight = _weight(factor, component)
    dressed = nlswt_result.spectral_function(q, w, eta, mode, grid, levels)
    one = np.zeros((len(q), len(w)))
    for i in range(len(q)):
        bands = dressed["lswt"][i]
        scale = max(float(np.max(bands)), 1.0)
        for mode in np.flatnonzero(factor.energies[i] > 0):
            match = np.flatnonzero(np.abs(bands - factor.energies[i, mode]) < 1e-6 * scale)
            if len(match) == 0:
                raise ValueError(f"LSWT mode at energy {factor.energies[i, mode]:.6g} has no "
                                 f"band of the nonlinear calculation at q = {q[i]}")
            # degenerate bands carry the same (group-mean) spectral function
            one[i] += weight[i, mode] * dressed["spectral"][i, match[0]]
    continuum = two_magnon_continuum(lswt_result, q, mesh=two_magnon_mesh)
    two = continuum.spectrum(w, eta, component=component)
    bare = factor.spectrum(w, eta, component=component)
    out = {"one_magnon": one, "two_magnon": two, "total": one + two, "lswt": bare}
    if mode == "dyson":
        out["negative_frequency_weight"] = dressed["negative_frequency_weight"]
    return out
