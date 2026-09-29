"""Audit Berry curvature and SI thermal Hall response with an analytic model.

This positive, number-conserving two-band boson model is not a fit to NBCP or
a microscopic spin-model reproduction. The reference uses two-level curvature
and independent quadrature for c_2. Run with PYTHONPATH=code-space.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.integrate import quad

from lswt.methods.spin_wave.diagonalization import Diagonalizer
from lswt.observables.thermodynamics import Thermodynamics
from lswt.observables.topology import Topology

KB_SI = 1.380649e-23
HBAR_SI = 1.0545718176461565e-34
MEV_J = 1.602176634e-22
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])


def model(n=24, length=1.0, orientation=1.0, lattice_vectors=None, k_points=None):
    """Build a full MBZ grid and analytic curvature in Cartesian coordinates.

    A(q) = 0.8 I + 0.1 d(q).sigma meV with q_i = k.a_i. Its eigenvalues
    lie between 0.5 and 1.1 meV. The Nambu matrix is diag(A(k), A(-k)^T).
    Supplied k_points replace the default grid for testing BZ generators.
    """
    cell = (np.eye(2) * length if lattice_vectors is None
            else np.asarray(lattice_vectors, dtype=float))
    q = (np.arange(n) + 0.5) * 2 * np.pi / n - np.pi
    q_points = np.array(np.meshgrid(q, q, indexing="ij")).reshape(2, -1).T
    momenta = q_points @ np.linalg.inv(cell.T)
    if k_points is not None:
        momenta = np.asarray(k_points, dtype=float)

    def blocks(k):
        x, y = (k @ cell.T).T
        d = np.array([np.sin(x), orientation * np.sin(y),
                      1 + np.cos(x) + np.cos(y)]).T
        dqx = np.array([np.cos(x), np.zeros_like(x), -np.sin(x)]).T
        dqy = np.array([np.zeros_like(x), orientation * np.cos(y), -np.sin(y)]).T
        dx = cell[0, 0] * dqx + cell[1, 0] * dqy
        dy = cell[0, 1] * dqx + cell[1, 1] * dqy
        normal = 0.8 * np.eye(2) + 0.1 * np.einsum("ki,imn->kmn", d, PAULI)
        derivatives = [0.1 * np.einsum("ki,imn->kmn", dk, PAULI) for dk in (dx, dy)]
        norm = np.linalg.norm(d, axis=1)
        curvature = np.sum(d * np.cross(dx, dy), axis=1) / (2 * norm**3)
        modes = np.column_stack([0.8 + 0.1 * norm, 0.8 - 0.1 * norm])
        return normal, derivatives, np.column_stack([-curvature, curvature]), modes

    a, da, omega, modes = blocks(momenta)
    am, dam, _, _ = blocks(-momenta)
    matrix = np.zeros((len(momenta), 4, 4), complex)
    matrix[:, :2, :2] = a
    matrix[:, 2:, 2:] = am.transpose(0, 2, 1)
    derivatives = []
    for direction in range(2):
        derivative = np.zeros_like(matrix)
        derivative[:, :2, :2] = da[direction]
        derivative[:, 2:, 2:] = -dam[direction].transpose(0, 2, 1)
        derivatives.append(derivative)
    data, _ = Diagonalizer.get_K_data(
        momenta, matrix, "No", partial_derivative_Hk=derivatives
    )
    area = abs(np.linalg.det(cell))
    parent = SimpleNamespace(
        Ns=2, bz_data={"area": (2 * np.pi)**2 / (len(momenta) * area)},
        system=SimpleNamespace(lattice_vectors=cell),
    )
    return parent, data, omega, modes


def c2_from_energy(x):
    """Evaluate c_2(n_B) independently as an integral from x=E/(k_B T)."""
    return quad(
        lambda z: z * z * np.exp(-z) / (-np.expm1(-z))**2,
        x, np.inf, epsabs=1e-12, epsrel=1e-12,
    )[0]


def reference(omega, modes, cell_area, temperature):
    """Return the full-MBZ analytic response in W/K per layer."""
    if temperature == 0:
        return 0.0
    x_values = modes * MEV_J / (KB_SI * temperature)
    weights = np.array([c2_from_energy(x) for x in x_values.flat]).reshape(modes.shape)
    integral = np.mean(np.sum(omega * weights, axis=1)) / cell_area
    return -(KB_SI**2 / HBAR_SI) * temperature * integral


def main():
    """Report mesh convergence, length invariance, sign and 3D conversion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()
    records = []
    layer_spacing_m = 7e-10  # Arbitrary test spacing, not a material parameter.
    for n, length, orientation in [(12, 1, 1), (24, 1, 1), (48, 1, 1),
                                    (24, 3.7, 1), (24, 1, -1)]:
        parent, data, omega, modes = model(n, length, orientation)
        topology = Topology(parent)
        numerical, chern, kappa_2d = topology.compute_thermal_Hall(data, 2.0)
        kappa_3d = topology.compute_thermal_Hall(
            data, 2.0, layer_spacing_m=layer_spacing_m
        )[2]
        combined = Thermodynamics(parent).compute_thermodynamic_quantities_at_T(data, 2.0)
        records.append({
            "n": n, "length": length, "orientation": orientation,
            "curvature_error": float(np.max(abs(numerical - omega))),
            "chern": chern.tolist(),
            "topology_kappa_W_per_K": kappa_2d,
            "combined_kappa_W_per_K": combined["Thermal Hall Conductance"],
            "reference_kappa_W_per_K": reference(omega, modes, length**2, 2.0),
            "layer_spacing_m": layer_spacing_m,
            "topology_kappa_W_per_m_K": kappa_3d,
        })
    output = json.dumps(records, indent=2) + "\n"
    print(output, end="")
    if args.output is not None:
        args.output.write_text(output)


if __name__ == "__main__":
    main()
