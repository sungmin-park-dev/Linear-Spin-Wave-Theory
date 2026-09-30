"""Snapshot NBCP model, Hamiltonian, energy and search results for refactor checks.

Run before and after a package move or rename, then compare the two snapshots.
The script resolves the package under either its current name (``spintoolkit``)
or its previous name (``lswt``), so the same file serves both sides of the
rename. It computes nothing new: every value comes from existing public calls.

Usage
-----
    python examples/package_regression_snapshot.py snapshot OUT.npz
    python examples/package_regression_snapshot.py compare A.npz B.npz
"""

from importlib import import_module
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code-space"), str(ROOT)]

from model import nbcp  # noqa: E402

SEED = 7301
MESH_N = 3
K_POINTS = np.array([[0.0, 0.0], [0.37, -0.29], [-0.81, 0.46], [1.13, 0.67]])
CELLS = [
    ("one_msl", 2, "Hex_60"),
    ("two_msl", 4, "Tetra"),
    ("three_msl", 6, "Hex_30"),
    ("four_msl", 8, "Hex_60"),
]
BASE = {"Jxy": 0.076, "Jz": 0.125, "h": (0.03, -0.04, 0.2)}
FAMILIES = {
    "xxz": {},
    "nn_soc": {"JPD": 0.013, "JGamma": -0.021},
    "nn_nnn_soc": {"JPD": 0.013, "JGamma": -0.021,
                   "Kxy": 0.004, "Kz": -0.005, "KPD": 0.002, "KGamma": -0.003},
    "dm": {"Dx": 0.002, "Dy": -0.001, "Dz": 0.003},
    "zero_exchange": {"Jxy": 0.0, "Jz": 0.0},
}
SEARCHES = [("xxz", "classical"), ("nn_soc", "classical")]
# Leaves of methods removed on purpose; a baseline may still contain them.
REMOVED_LEAVES = {"/MAGSWT/": "MAGSWT grid search removed (D27)"}


def load_package():
    """Return (energy module, hamiltonian module, optimization module, name)."""
    for name, lswt_path in (("spintoolkit", "methods.lswt"),
                            ("lswt", "methods.spin_wave")):
        try:
            energy = import_module(f"{name}.{lswt_path}.energy")
        except ImportError:
            continue
        hamiltonian = import_module(f"{name}.{lswt_path}.hamiltonian")
        optimization = import_module(f"{name}.methods.optimization")
        return energy, hamiltonian, optimization, name
    raise ImportError("Neither spintoolkit nor lswt is importable.")


def snapshot(path):
    energy, hamiltonian, optimization, package = load_package()
    rng = np.random.default_rng(SEED)
    leaves = {}
    for family, parameters in FAMILIES.items():
        config = {**BASE, **parameters}
        exch_j = nbcp.make_nn_exchange_matrices(config)
        exch_k = nbcp.make_nnn_exchange_matrices(config)
        for name, num_angles, bz_type in CELLS:
            for angle_case in ("fixed", "seeded_random"):
                if angle_case == "fixed":
                    angles = np.linspace(-1.1, 2.3, num_angles)
                else:
                    angles = rng.uniform(-np.pi, np.pi, num_angles)
                key = f"{family}/{name}/{angle_case}"
                system = getattr(nbcp, name)(config, angles, exch_j, exch_k)
                data = system.to_legacy_dict(bz_type)
                ham = hamiltonian.LSWTHamiltonian(data["Spin info"], data["Couplings"])
                k_ham, _ = ham.Quadratic_Bose_Hamiltonian(K_POINTS, angles=angles)
                leaves[key + "/H_k"] = np.asarray(k_ham)
                cef = energy.EnergyFunction(data, N=MESH_N)
                leaves[key + "/E_cl"] = np.asarray(
                    cef.classical_energy_density_func(angles))
    optimizer = optimization.SpinOptimizer()
    for family, method in SEARCHES:
        config = {**BASE, **FAMILIES[family]}
        exch_j = nbcp.make_nn_exchange_matrices(config)
        exch_k = nbcp.make_nnn_exchange_matrices(config)
        for name, num_angles, bz_type in CELLS:
            np.random.seed(SEED)
            start = np.random.uniform(-np.pi, np.pi, num_angles)
            system = getattr(nbcp, name)(config, start, exch_j, exch_k)
            cef = energy.EnergyFunction(system.to_legacy_dict(bz_type), N=MESH_N,
                                        update_args=True)
            # An explicit all-free list; find_minimum does not accept None yet.
            best, classical = optimizer.find_minimum(
                cef, method, [None] * num_angles, verbose=False)
            key = f"search/{family}/{method}/{name}"
            for label, result in (("best", best), ("classical", classical)):
                for field, value in result.items():
                    if value is None:
                        continue
                    if isinstance(value, str):
                        leaves[f"{key}/{label}/{field}"] = np.array(value)
                    else:
                        leaves[f"{key}/{label}/{field}"] = np.asarray(value, dtype=float)
    np.savez(path, **leaves)
    print(f"{package}: {len(leaves)} leaves -> {path}")


def compare(path_a, path_b):
    a, b = np.load(path_a), np.load(path_b)
    removed = sorted(k for k in set(a.files) - set(b.files)
                     if any(pattern in k for pattern in REMOVED_LEAVES))
    if sorted(set(a.files) - set(removed)) != sorted(b.files):
        missing = sorted((set(a.files) - set(removed)) ^ set(b.files))
        raise SystemExit(f"Leaf sets differ: {missing}")
    for pattern, reason in REMOVED_LEAVES.items():
        count = sum(pattern in k for k in removed)
        if count:
            print(f"{count} baseline leaves not compared: {reason}")
    worst = 0.0
    for key in b.files:
        if a[key].shape != b[key].shape:
            raise SystemExit(f"Shape differs: {key}")
        if a[key].dtype.kind in "US":
            if not np.array_equal(a[key], b[key]):
                raise SystemExit(f"Value differs: {key}")
            continue
        if a[key].size:
            diff = np.nanmax(np.abs(a[key] - b[key]))
            same_nan = np.array_equal(np.isnan(a[key]), np.isnan(b[key]))
            if not same_nan:
                raise SystemExit(f"NaN pattern differs: {key}")
            worst = max(worst, float(0.0 if np.isnan(diff) else diff))
    print(f"{len(b.files)} leaves compared; maximum difference {worst}")
    return worst


if __name__ == "__main__":
    command, *paths = sys.argv[1:]
    if command == "snapshot":
        snapshot(*paths)
    elif command == "compare":
        sys.exit(0 if compare(*paths) == 0.0 else 1)
    else:
        raise SystemExit(__doc__)
