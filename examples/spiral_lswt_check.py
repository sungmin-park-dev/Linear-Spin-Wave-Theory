"""Verify rotating-frame LSWT of single-Q spirals (D34).

Reports, for the benchmark spirals of ``tests/test_methods/test_spiral_lswt.py``:
the largest deviation of omega(k) from the analytic Heisenberg spiral
dispersion, the pitch found by LT + refinement against the analytic value, and
for commensurate pitches the largest differences from ordinary LSWT on the
supercell (energy, bands, lab-frame S^{ab}(q, w)) on the same momenta.

Usage
-----
    python examples/spiral_lswt_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from spintoolkit.methods.luttinger_tisza import luttinger_tisza
from spintoolkit.methods.lswt import (
    LSWTSettings, refine_spiral, solve_lswt, solve_spiral_lswt, spiral_energy_gradient)
from spintoolkit.models import triangular_heisenberg
from spintoolkit.observables.structure_factor import spiral_structure_factor, structure_factor
from spintoolkit.states.incommensurate import IncommensurateStructure
from spintoolkit.system.conditions import ExternalConditions
from tests.test_methods.test_spiral_lswt import (
    BONDS, J1, J2, JY, Q_EXACT, S, Z, chain_model, heisenberg_dispersion, unfolded)

OMEGA = np.linspace(-4, 4, 161)


def supercell_comparison(model, spiral, conditions, mesh):
    supercell = solve_lswt(model, spiral.to_spin_state(model), conditions,
                           settings=LSWTSettings(mesh=mesh))
    result = solve_spiral_lswt(model, spiral, conditions,
                               settings=LSWTSettings(k_points=unfolded(model, supercell)))
    q = np.random.default_rng(11).uniform(-6, 6, (24, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a, b = spiral_structure_factor(result, q), structure_factor(supercell, q)
    spectra = max(float(np.max(np.abs(a.spectrum(OMEGA, 0.05, "gaussian", c)
                                      - b.spectrum(OMEGA, 0.05, "gaussian", c))))
                  for c in ("xx", "yy", "zz", "neutron"))
    return {
        "supercell_sites": len(supercell.site_keys),
        "num_k_spiral": int(len(result.k_points)),
        "ground_state_energy": result.ground_state_energy,
        "energy_difference": abs(result.ground_state_energy - supercell.ground_state_energy),
        "band_difference": float(np.max(np.abs(np.sort(result.bands().ravel())
                                               - np.sort(supercell.bands().ravel())))),
        "static_structure_factor_difference": float(np.max(np.abs(a.static() - b.static()))),
        "spectrum_difference": spectra}


def main():
    began = time.time()
    cases = []

    model = triangular_heisenberg(J=1.0, S=S)
    spiral = IncommensurateStructure.planar(model, [1 / 3, 2 / 3], Z)
    entry = {"model": "triangular Heisenberg (120 deg as a spiral, Q = K)"}
    entry.update(supercell_comparison(model, spiral, None, (8, 8)))
    result = solve_spiral_lswt(model, spiral, settings=LSWTSettings(mesh=(24, 24)))
    a = model.lattice
    gamma = np.mean([np.cos(result.k_points @ d) for d in (a[0], a[1], a[1] - a[0])], axis=0)
    entry["analytic_dispersion_deviation"] = float(np.max(np.abs(
        result.bands()[:, 0] - 3 * S * np.sqrt((1 - gamma) * (1 + 2 * gamma)))))
    entry["ground_state_energy_24x24"] = result.ground_state_energy
    entry["reference_energy"] = -1.5 * S ** 2 * (1 + 0.436824 / (2 * S))
    cases.append(entry)

    model = chain_model(J1, J2, JY)
    report = luttinger_tisza(model, mesh=(24, 24))
    spiral = refine_spiral(model, IncommensurateStructure.from_lt(model, report.minima[0]))
    result = solve_spiral_lswt(model, spiral, settings=LSWTSettings(mesh=(24, 24)))
    Q = spiral.cartesian_wave_vector(model)
    q = spiral.wave_vector[0]
    cases.append({
        "model": "square J1-J2 chains (J1 = 1, J2 = 0.4) with Jy = -0.5: incommensurate",
        "lt_pitch": report.minima[0].fractional.tolist(),
        "refined_pitch": spiral.wave_vector.tolist(),
        "analytic_pitch": Q_EXACT,
        "pitch_error": abs(min(q, 1 - q) - Q_EXACT),
        "wave_vector_gradient": spiral_energy_gradient(model, spiral).tolist(),
        "analytic_dispersion_deviation": float(np.max(np.abs(
            result.bands()[:, 0] - heisenberg_dispersion(result.k_points, Q, BONDS)))),
        "ground_state_energy_24x24": result.ground_state_energy,
        "classical_energy": result.classical_energy})

    model = chain_model(J1=-1.0, J2=0.0, Jy=-1.0, D=np.sqrt(3.0))
    spiral = refine_spiral(model, IncommensurateStructure.planar(model, [0.1, 0.0], Z))
    q = spiral.wave_vector[0]
    entry = {"model": "ferromagnetic J = -1 with DM D = sqrt(3) along z (x bonds), Jy = -1",
             "refined_pitch": spiral.wave_vector.tolist(), "analytic_pitch": 1 / 6,
             "pitch_error": abs(min(q, 1 - q) - 1 / 6)}
    exact = IncommensurateStructure.planar(model, [round(q * 6) / 6, 0.0], Z)
    entry.update(supercell_comparison(model, exact, None, (4, 12)))
    cases.append(entry)

    model = chain_model(J1=1.0, J2=0.5, Jy=-0.5)
    h = 0.4 * S * (2 * (1.0 + 0.5 - 0.5) - 2 * (-0.5 - 0.25 - 0.5))
    conditions = ExternalConditions(field=(0, 0, h))
    start = IncommensurateStructure(model.fingerprint(), [0.3, 0.0], Z,
                                    {"A": [np.sin(1.2), 0, np.cos(1.2)]})
    spiral = refine_spiral(model, start, conditions)
    entry = {"model": "J1-J2 chains (J2 = 0.5, q = 1/3) in a field along the axis: cone",
             "field": h, "refined_pitch": spiral.wave_vector.tolist(),
             "cos_cone_angle": float(np.cos(spiral.cone_angles()["A"])),
             "analytic_cos_cone_angle": 0.4}
    exact = IncommensurateStructure(model.fingerprint(), [1 / 3, 0.0], Z,
                                    {"A": [np.sqrt(1 - 0.16), 0, 0.4]})
    entry.update(supercell_comparison(model, exact, conditions, (8, 8)))
    cases.append(entry)

    print(json.dumps({"date": time.strftime("%Y-%m-%d"), "cases": cases,
                      "seconds": round(time.time() - began, 1)}, indent=2))


if __name__ == "__main__":
    main()
