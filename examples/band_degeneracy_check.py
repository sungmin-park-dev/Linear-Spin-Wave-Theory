"""Audit individual-band Kubo validity near finite-energy degeneracy.

This local two-band model at k=0 is not a periodic BZ or an NBCP fit.
A(k) = 0.8 I + 0.1 kx sigma_x + 0.1 ky sigma_y + delta sigma_z (meV),
with dimensionless momenta and H(k) = diag(A(k), A(-k)^T).
The analytic individual curvature is defined here only for delta > 0.
This is a diagnostic report, not an assertion that degenerate output is valid.
Run with PYTHONPATH=code-space.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from lswt.methods.spin_wave.diagonalization import Diagonalizer
from lswt.definitions import DEFAULT_BAND_GAP_CUTOFF
from lswt.observables.topology import compute_berry_curvature


def probe(delta, *, band_gap_cutoff=DEFAULT_BAND_GAP_CUTOFF):
    """Compare a known nondegenerate limit and the exactly degenerate case."""
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.diag([1, -1])
    normal = 0.8 * np.eye(2) + delta * sigma_z
    matrix = np.zeros((1, 4, 4), dtype=complex)
    matrix[0, :2, :2] = normal
    matrix[0, 2:, 2:] = normal.T
    derivatives = []
    for sigma in (sigma_x, sigma_y):
        derivative = np.zeros_like(matrix)
        derivative[0, :2, :2] = 0.1 * sigma
        derivative[0, 2:, 2:] = -0.1 * sigma.T
        derivatives.append(derivative)

    data, _ = Diagonalizer.get_K_data(
        np.zeros((1, 2)), matrix, "No", partial_derivative_Hk=derivatives
    )
    h_data, (energies, vectors), _ = next(iter(data.values()))
    curvature, spacing = compute_berry_curvature(
        energies, vectors, h_data[1:], band_gap_cutoff=band_gap_cutoff
    )
    analytic = np.array([-0.005, 0.005]) / delta**2 if delta > 0 else None
    available = np.isfinite(curvature)
    return {
        "delta_meV": delta,
        "band_gap_cutoff_meV": band_gap_cutoff,
        "physical_energies_meV": energies[:2].tolist(),
        "physical_band_gap_meV": float(abs(energies[0] - energies[1])),
        "returned_curvature": [float(value) if valid else None
                               for value, valid in zip(curvature, available)],
        "band_curvature_available": available.tolist(),
        "reported_spacing_meV": spacing.tolist(),
        "individual_band_reference_defined": delta > 0,
        "analytic_curvature": None if analytic is None else analytic.tolist(),
        "relative_curvature_error": (
            None if analytic is None or not available.all()
            else float(np.max(abs(curvature / analytic - 1)))
        ),
    }


def main():
    """Write the observed output and source hashes without modifying the solver."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--band-gap-cutoff", type=float, default=DEFAULT_BAND_GAP_CUTOFF,
                        help="Numerical minimum signed gap in meV (default: 1e-8)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = [
        "examples/band_degeneracy_check.py",
        "code-space/lswt/methods/spin_wave/diagonalization.py",
        "code-space/lswt/observables/topology.py",
        "code-space/lswt/definitions/constants.py", "code-space/lswt/definitions/defaults.py", "code-space/lswt/definitions/spin_basis.py",
    ]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Local finite-energy degeneracy; no BZ integral or material verdict",
        "numpy_version": np.__version__,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sources
        },
        "records": [probe(delta, band_gap_cutoff=args.band_gap_cutoff)
                    for delta in (1e-2, 1e-4, 1e-6, 1e-15, 0.0)],
    }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(output, end="")
    if args.output is not None:
        args.output.write_text(output)


if __name__ == "__main__":
    main()
