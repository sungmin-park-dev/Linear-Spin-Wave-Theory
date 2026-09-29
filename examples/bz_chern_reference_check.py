"""Check generated magnetic-BZ coverage and Chern/Hall mesh convergence.

The two-band boson Hamiltonian is analytic and is not an NBCP material model.
Chern uses the local-note convention C = integral(Omega)/(2*pi). Run with
PYTHONPATH=code-space; --output optionally saves the numerical report as JSON.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from lswt.system.brillouin_zone import BrillouinZone
from lswt.observables.topology import Topology
from thermal_hall_reference_check import model, reference

CASES = {
    "simple": [[1, 0], [0.5, np.sqrt(3) / 2]],
    "Hex_60": [[1, np.sqrt(3)], [1, -np.sqrt(3)]],
    "Hex_30": [[1.5, np.sqrt(3) / 2], [1.5, -np.sqrt(3) / 2]],
    "Tetra": [[1, 0], [0, np.sqrt(3)]],
    "wigner_seitz": [[1, 0], [12.3, 0.7]],
}


def main():
    """Print coverage, independent curvature and response at three resolutions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = []
    for mode, cell in CASES.items():
        area = abs(np.linalg.det(cell))
        for n in (6, 12, 24):
            bz_data, points, _ = BrillouinZone((cell, mode), bz_type=mode).get_full(n)
            parent, data, omega, modes = model(lattice_vectors=cell, k_points=points)
            parent.bz_data = bz_data
            curvature, chern, hall = Topology(parent).compute_thermal_Hall(
                data, 2.0, bz_type=mode
            )
            q = points @ np.asarray(cell).T
            records.append({
                "bz_type": mode, "N": n, "num_k_points": len(points),
                "grid_area_over_MBZ": len(points) * bz_data["area"] * area / (2 * np.pi)**2,
                "first_fourier_moment": float(abs(np.mean(np.exp(1j * q[:, 1])))),
                "curvature_max_error": float(np.max(abs(curvature - omega))),
                "chern": chern.tolist(),
                "expected_chern": (np.sign(np.linalg.det(cell)) * np.array([1, -1])).tolist(),
                "kappa_W_per_K": hall,
                "independent_kappa_W_per_K": reference(omega, modes, area, 2.0),
            })
    output = json.dumps(records, indent=2) + "\n"
    print(output, end="")
    if args.output is not None:
        args.output.write_text(output)


if __name__ == "__main__":
    main()
