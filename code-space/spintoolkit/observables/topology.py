"""Topological properties of magnon bands.

This module computes topological quantities including Berry curvature,
Chern numbers, and thermal Hall conductance within the LSWT framework.

Ported from modules/LinearSpinWaveTheory/lswt_topology.py.
"""

import warnings

import numpy as np
from scipy.special import spence

from spintoolkit.observables.bose_statistics import compute_bose_einstein_distribution
from spintoolkit.definitions import (
    K_BOLTZMANN_MEV, H_BAR_MEV, DEFAULT_LEVEL_SPACING, DEFAULT_BAND_GAP_CUTOFF,
)


def _validate_band_gap_cutoff(band_gap_cutoff):
    """Require a finite, non-negative numerical gap cutoff in meV."""
    if (not np.isscalar(band_gap_cutoff) or not np.isrealobj(band_gap_cutoff)
            or not np.isfinite(band_gap_cutoff) or band_gap_cutoff < 0):
        raise ValueError("band_gap_cutoff must be finite and non-negative (meV)")


def _band_separation(signed_energies, num_sl, band_gap_cutoff):
    """Return minimum signed BdG separations and bands excluded by the cutoff.

    Compare each physical band with every other signed mode, independent of
    array ordering. The caller supplies an absolute cutoff in the same energy
    unit as the spectrum (meV in the public Hall paths). No residual, matrix
    norm or eigenvalue error estimate is computed here.
    """
    differences = np.abs(signed_energies[:num_sl, None] - signed_energies[None, :])
    np.fill_diagonal(differences, np.inf)
    spacing = np.min(differences, axis=1)
    return spacing, spacing <= band_gap_cutoff


def _validate_thermal_hall_inputs(temperature, layer_spacing_m):
    """Validate temperature in K and optional layer spacing in metres."""
    if not np.isscalar(temperature) or not np.isfinite(temperature) or temperature < 0:
        raise ValueError("Temperature must be finite and non-negative")
    if layer_spacing_m is not None and (
        not np.isscalar(layer_spacing_m)
        or not np.isfinite(layer_spacing_m)
        or layer_spacing_m <= 0
    ):
        raise ValueError("layer_spacing_m must be finite and positive")


def _magnetic_cell_area(lswt_obj):
    """Return the magnetic cell area in the length units used by dH/dk.

    Missing geometry is unavailable (NaN), never a triangular-cell default.
    Legacy solver parents may supply the original Lattice/BZ setting.
    """
    system = getattr(lswt_obj, "system", None)
    vectors = getattr(system, "lattice_vectors", None)
    if vectors is None:
        settings = getattr(lswt_obj, "lattice_bz_settings", None)
        if settings is not None:
            vectors = settings[0]
    if vectors is None:
        return np.nan
    vectors = np.asarray(vectors, dtype=float)
    if vectors.shape != (2, 2) or not np.all(np.isfinite(vectors)):
        raise ValueError("Thermal Hall requires finite 2D magnetic lattice vectors")
    area = abs(np.linalg.det(vectors))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Thermal Hall requires a nonzero magnetic cell area")
    return area


def _thermal_hall_conductivity(weighted_curvature_sum, temperature,
                               num_k_points, cell_area, layer_spacing_m):
    """Integrate an equal-weight full-MBZ sum into SI kappa_xy.

    Integral d^2k/(2*pi)^2 = sum_k/(N_k*A_mag). Curvature and cell area
    carry the same squared length unit, which cancels. With meV constants,
    the remaining meV is converted to joules. No per-spin division applies.
    The caller must supply NaN for an incomplete or invalid sum.
    """
    if (num_k_points == 0 or not np.isfinite(cell_area)
            or not np.isfinite(weighted_curvature_sum)):
        return np.nan
    mev_to_joule = 1.602176634e-22
    prefactor = K_BOLTZMANN_MEV**2 * temperature / H_BAR_MEV * mev_to_joule
    kappa = -prefactor * weighted_curvature_sum / (num_k_points * cell_area)
    if layer_spacing_m is not None:
        kappa /= layer_spacing_m
    return float(kappa)


def _matches_integration_grid(lswt_obj, k_data):
    """Reject a partial/replaced grid when the parent recorded its full keys.

    Custom parents without provenance retain the documented full-BZ input
    precondition. Keys cannot verify the Hamiltonian or arbitrary sample weights.
    """
    keys = getattr(lswt_obj, "_integration_k_keys", None)
    return keys is None or frozenset(k_data) == keys


def c_two_function(x):
    """Compute the c_2 weight function for thermal Hall conductance.

    The c_2 function appears in the thermal Hall conductance formula and
    is related to the polylogarithm. It maps the Bose-Einstein distribution
    to the appropriate thermal weight.

    Parameters
    ----------
    x : float, list, or np.ndarray
        Bose-Einstein distribution values (occupation numbers).

    Returns
    -------
    f : np.ndarray
        c_2 function values, same shape as input.
    """
    x = np.array(x)
    f = np.zeros_like(x, dtype=float)

    mask_too_small = x < 1e-300
    mask_too_large = x > 1e300
    normal_range = ~mask_too_small & ~mask_too_large

    f[mask_too_large] = np.pi ** 2 / 3

    cal_x = x[normal_range]
    if cal_x.size > 0:
        term1 = (1 + cal_x) * (np.log((1 + cal_x) / cal_x)) ** 2
        term2 = (np.log(cal_x)) ** 2
        term3 = 2 * spence(1 + cal_x)
        f[normal_range] = term1 - term2 - term3

    return f


def compute_berry_curvature(eval, evec, pDiffHk, num_sl=None, J_mat=None,
                           *, band_gap_cutoff=DEFAULT_BAND_GAP_CUTOFF):
    """Compute Berry curvature for magnon bands using the Kubo formula.

    Calculates the Berry curvature for each physical magnon band at a
    given k-point using the partial derivatives of the bosonic Hamiltonian.

    Parameters
    ----------
    eval : np.ndarray
        Magnon eigenvalues (length 2*num_sl).
    evec : np.ndarray
        Bogoliubov transformation matrix.
    pDiffHk : list of np.ndarray
        Partial derivatives of the bosonic Hamiltonian [dH/dkx, dH/dky].
    num_sl : int or None, optional
        Number of sublattices. If None, inferred as len(eval)//2.
    J_mat : np.ndarray or None, optional
        Para-unitary metric matrix. If None, constructed as diag(+1,...,+1,-1,...,-1).
    band_gap_cutoff : float, optional
        Non-negative minimum allowed signed band separation in meV (default:
        1e-8). Separations <= this value make the affected band unavailable.
        Zero rejects only exact coincidences. This is a numerical calculation
        policy, not a physical degeneracy criterion or a certified error bound.

    Returns
    -------
    Omega_nk : np.ndarray
        Berry curvature for each physical band, shape (num_sl,).
        NaN if any other signed BdG mode is within band_gap_cutoff, including
        the cutoff boundary. No physical gap is inserted.
    level_spacing : np.ndarray
        Minimum absolute difference from every other signed BdG mode,
        shape (num_sl,). Includes zero differences and particle-hole partners;
        it is not a distance to zero or solely a positive-band gap.

    Notes
    -----
    Individual-band invariants assume isolation over the entire BZ. Passing
    this local cutoff does not establish that assumption, numerical accuracy,
    or mesh convergence. Gaps above the cutoff are retained without clipping.
    The default is a diagnostic starting value, not a universal stability
    guarantee. Check cutoff sensitivity and mesh convergence for each model.
    When rescaling the energy unit, rescale the cutoff by the same factor.
    """
    _validate_band_gap_cutoff(band_gap_cutoff)
    num_sl = num_sl if num_sl is not None else int(len(eval) // 2)
    J_mat = J_mat if J_mat is not None else np.diag(
        np.hstack([np.ones(num_sl), -np.ones(num_sl)])
    )

    pDxHk, pDyHk = pDiffHk

    J_eval = np.diag(J_mat) * eval
    level_spacing, excluded = _band_separation(J_eval, num_sl, band_gap_cutoff)
    Omega_nk = np.full(num_sl, np.nan)

    partial_H_x = J_mat @ evec.conj().T @ pDxHk @ evec
    partial_H_y = J_mat @ evec.conj().T @ pDyHk @ evec

    for n in range(num_sl):
        if excluded[n]:
            continue
        summand = 0

        for m in range(2 * num_sl):
            if n == m:
                continue
            difference = J_eval[n] - J_eval[m]
            # Divide before multiplying to avoid squaring very small/large
            # energy scales. This is the same Kubo expression for resolved bands.
            summand += ((partial_H_x[n, m] / difference)
                        * (partial_H_y[m, n] / difference))

        Omega_nk[n] = -2 * np.imag(summand)

    return Omega_nk, level_spacing


def _handle_colpa_failure(kpt=None):
    """Handle Colpa's method failure at a k-point.

    The Colpa method works for any positive definite Hamiltonian.
    If it fails, the given Hamiltonian is not positive definite.

    Parameters
    ----------
    kpt : tuple or np.ndarray or None, optional
        The k-point where failure occurred.

    Returns
    -------
    int
        Always returns 1 (count of failed k-points).
    """
    return 1


class Topology:
    """Topological property calculations for magnon bands.

    Parameters
    ----------
    lswt_obj : object
        Parent LSWT solver providing Ns, bz_data and system.lattice_vectors.
        The lattice vectors must describe the magnetic cell used by H(k).
    num_sl : int or None, optional
        Number of sublattices. If None, taken from lswt_obj.Ns.
    """

    def __init__(self, lswt_obj, num_sl=None):
        self.lswt_obj = lswt_obj
        self.Ns = lswt_obj.Ns
        self.J_mat = np.diag(np.hstack([np.ones((self.Ns)), -np.ones((self.Ns))]))
        self.area = lswt_obj.bz_data["area"]

    def thermal_weight_function(self, eval, Temperature):
        """Compute thermal weight function c_2(n_B(E)) for each band.

        Parameters
        ----------
        eval : np.ndarray
            Magnon eigenvalues (length 2*Ns).
        Temperature : float
            Temperature in Kelvin.

        Returns
        -------
        np.ndarray
            Thermal weight values for each physical band, shape (Ns,).
        """
        Epk = eval[:self.Ns]

        if Temperature == 0:
            return np.zeros_like(Epk)
        else:
            nk = compute_bose_einstein_distribution(E_list=Epk, Temperature=Temperature)
            return c_two_function(nk)

    def compute_thermal_Hall(self, k_data, Temperature, bz_type=None, verbose=False,
                             *, layer_spacing_m=None,
                             band_gap_cutoff=DEFAULT_BAND_GAP_CUTOFF):
        """Compute Berry curvature, Chern numbers, and thermal Hall conductance.

        Parameters
        ----------
        k_data : dict
            Equal-weight samples of the full magnetic Brillouin zone, with
            Cartesian dH/dkx and dH/dky in the reciprocal units of the real
            lattice vectors. Uniform repeated coverage is also allowed.
            Band paths and nonuniform/partial BZ samples are not supported.
            Solver parents reject keys that differ from their last full grid;
            for custom parents, coverage remains the caller's responsibility.
        Temperature : float
            Temperature in Kelvin.
        bz_type : str or None, optional
            Compatibility argument: "simple", "Hex_60", "Hex_30", "Tetra",
            "tetra", "wigner_seitz", or None. Both integrals use the actual
            magnetic cell area; no extra division by Ns is performed.
        verbose : bool, optional
            If True, warn about unavailable curvature and sampled spacings
            below DEFAULT_LEVEL_SPACING (meV). The latter is only a mesh
            convergence advisory, separate from band_gap_cutoff.
        layer_spacing_m : float or None, optional
            Positive distance between equivalent conducting layers in metres.
            If supplied, divide the 2D response by this distance to obtain
            the bulk 3D conductivity, assuming independent equivalent layers.
        band_gap_cutoff : float, optional
            Numerical minimum signed band separation in meV (default: 1e-8).
            Passed to compute_berry_curvature. Finite and non-negative;
            a calculation policy, not a certified eigenvalue error bound.

        Returns
        -------
        Berry_curvature : np.ndarray
            Berry curvature at each k-point, shape (num_k_points, Ns).
            Bands excluded by band_gap_cutoff have NaN entries.
        chern_number : np.ndarray
            C_n = integral(Omega_n d^2k)/(2*pi), shape (Ns,). Approximated by
            2*pi*mean(Omega_n)/A_mag with the same full-BZ measure as Hall.
            NaN for missing geometry, incomplete solver grids or failed samples.
            A band with unavailable curvature at any sample has NaN Chern;
            other isolated bands can remain available.
            The sign follows this module's Berry convention; values are not
            rounded to integers and require a mesh-convergence check.
        THC : float
            Kappa_xy per layer in W/K, or W/(m K) with layer_spacing_m.
            Returns NaN for missing geometry, empty data, failed samples or
            unavailable derivatives. This is kappa, not kappa/T, and has no
            micro prefix. Requires well-defined, nondegenerate band curvature.
            Any excluded band makes this band-sum implementation unavailable
            (NaN, including T=0); this is not a verdict on a grouped response.
        """
        _validate_thermal_hall_inputs(Temperature, layer_spacing_m)
        _validate_band_gap_cutoff(band_gap_cutoff)
        cell_area = _magnetic_cell_area(self.lswt_obj)
        num_k_points = len(k_data)
        valid_count = num_k_points

        if bz_type not in (None, "simple", "Hex_60", "Hex_30", "Tetra", "tetra", "wigner_seitz"):
            raise ValueError(f"Unknown BZ type: {bz_type}")

        Berry_curvature = np.zeros((num_k_points, self.Ns))
        min_level_spacing = np.full(self.Ns, np.inf)
        thermal_hall = 0

        for j, contents in enumerate(k_data.values()):
            H_k_data, Eigen_k_data, Colpa_data, *_ = contents

            colpa_success = Colpa_data[0]
            if colpa_success and len(H_k_data[1:]) == 2:
                eval, evec = Eigen_k_data
                pDHk = H_k_data[1:]

                Omega_nk, level_spacing = compute_berry_curvature(
                    eval=eval,
                    evec=evec,
                    pDiffHk=pDHk,
                    num_sl=self.Ns,
                    J_mat=self.J_mat,
                    band_gap_cutoff=band_gap_cutoff,
                )
                Berry_curvature[j] = Omega_nk
                min_level_spacing = np.minimum(min_level_spacing, level_spacing)

                thermal_hall += np.sum(
                    Omega_nk * self.thermal_weight_function(eval, Temperature=Temperature)
                )
            else:
                Berry_curvature[j] = np.full(self.Ns, np.nan)
                valid_count -= _handle_colpa_failure()

        chern_number = np.full(self.Ns, np.nan)

        # A sum over surviving points is not a full-BZ integral. Do not
        # silently reweight the survivors after a diagonalization failure.
        if (valid_count != num_k_points or num_k_points == 0
                or not _matches_integration_grid(self.lswt_obj, k_data)):
            thermal_hall = np.nan
        elif np.isfinite(cell_area):
            isolated = np.all(np.isfinite(Berry_curvature), axis=0)
            chern_number[isolated] = (
                2 * np.pi * np.mean(Berry_curvature[:, isolated], axis=0) / cell_area
            )
        THC = _thermal_hall_conductivity(
            thermal_hall, Temperature, num_k_points, cell_area, layer_spacing_m
        )

        if verbose:
            unavailable = np.flatnonzero(np.any(~np.isfinite(Berry_curvature), axis=0))
            if unavailable.size:
                warnings.warn(
                    f"Unavailable curvature for bands {unavailable.tolist()}; "
                    f"check band_gap_cutoff={band_gap_cutoff:g} meV and sample validity.",
                    RuntimeWarning, stacklevel=2,
                )
            small = np.flatnonzero(min_level_spacing < DEFAULT_LEVEL_SPACING)
            if small.size:
                warnings.warn(
                    f"Small sampled signed BdG spacings for bands {small.tolist()}: "
                    f"{min_level_spacing[small].tolist()} meV; check mesh convergence. "
                    "This advisory threshold does not define degeneracy.",
                    RuntimeWarning, stacklevel=2,
                )

        return Berry_curvature, chern_number, THC
