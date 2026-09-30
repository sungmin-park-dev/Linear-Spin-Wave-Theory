"""Verify stage 5b: magnon thermal Hall conductivity kappa_xy / T (D29).

1. Haldane magnons: implementation against an independent two-band
   calculation (curvature from finite differences of d(k), c2 by quadrature).
2. Pair form against the band sum; limits t -> 0 and t -> infinity; Kitaev
   [111] under field reversal.
3. Zero for coplanar Heisenberg states, including degenerate bands (Neel,
   120 degrees) where the band sum is undefined; continuity as the Haldane
   gap closes (D -> 0).
4. The existing SI routine (observables.topology.Topology) on the same k data.
5. NBCP Y (0.2 T) and V (1.4 T) with J_PD or J_Gamma = 0.01 meV: uniform-mesh
   convergence N = 12 ... 192 at t = 0.01 ... 0.1 meV.

Usage
-----
    python examples/lswt_stage5b_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from model import nbcp
from spintoolkit.definitions import H_BAR_MEV
from spintoolkit.definitions.constants import K_BOLTZMANN_MEV, MU_B_MEV_PER_T
from spintoolkit.methods.classical import refine_classical
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import neel_state, square_heisenberg, state_120, triangular_heisenberg
from spintoolkit.observables.berry import berry_curvature, thermal_hall
from spintoolkit.system.conditions import ExternalConditions
from tests.test_methods.test_berry import c2_quadrature, haldane, kitaev

SCAN = ROOT / 'data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json'


def quiet(function, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return function(*args, **kwargs)


def independent_haldane():
    result = haldane(0.2, (12, 12))[2]
    pauli = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]

    def d_vector(k):
        h = result.hamiltonian_at(k)[0][:2, :2]
        return np.real(np.trace(h)) / 2, np.array([np.real(np.trace(h @ p)) / 2 for p in pauli])

    def unit(q):
        d = d_vector(q)[1]
        return d / np.linalg.norm(d)

    step, ts = 1e-5, [0.1, 0.4, 2.0]
    reference = np.zeros(len(ts))
    for k in result.k_points:
        d0, d = d_vector(k)
        dx = (unit(k + [step, 0]) - unit(k - [step, 0])) / (2 * step)
        dy = (unit(k + [0, step]) - unit(k - [0, step])) / (2 * step)
        solid = unit(k) @ np.cross(dx, dy) / 2
        for i, t in enumerate(ts):
            reference[i] += (c2_quadrature(d0 - np.linalg.norm(d), t)
                             - c2_quadrature(d0 + np.linalg.norm(d), t)) * solid
    reference = -reference / len(result.k_points) / abs(np.linalg.det(result.magnetic_lattice))
    kappa = thermal_hall(result, ts).kappa_over_t
    return {'temperatures': ts, 'implementation': kappa.tolist(), 'independent': reference.tolist(),
            'max_relative_difference': float(np.max(np.abs(kappa / reference - 1)))}


def forms_and_limits():
    rows = []
    ts = [0.0, 0.02, 0.05, 0.2, 1.0, 10.0, 1e3]
    for name, result in (('haldane D=0.2 h=0.3', haldane(0.2, (24, 24))[2]),
                         ('kitaev K=-1 h=1 [111]', kitaev(mesh=(24, 24))),
                         ('kitaev K=-1 h=1 -[111]', kitaev(mesh=(24, 24), sign=-1))):
        hall = thermal_hall(result, ts)
        rows.append({'case': name, 'temperatures': ts, 'kappa_over_t': hall.kappa_over_t.tolist(),
                     'max_abs_pair_minus_band_sum': float(np.max(np.abs(
                         hall.kappa_over_t - hall.kappa_over_t_band_sum)))})
    return rows


def null_and_continuity():
    out = {}
    square, triangle = square_heisenberg(), triangular_heisenberg()
    ts = [0.1, 1.0]
    neel = quiet(thermal_hall, solve_lswt(square, neel_state(square), None,
                                          settings=LSWTSettings(mesh=(24, 24))), ts)
    out['square_neel'] = {'kappa_over_t': neel.kappa_over_t.tolist(),
                          'band_sum': [None if np.isnan(x) else x for x in neel.kappa_over_t_band_sum],
                          'gapless': neel.gapless}
    tri = quiet(thermal_hall, solve_lswt(triangle, state_120(triangle), None,
                                         settings=LSWTSettings(mesh=(24, 24))), ts)
    out['triangular_120'] = {'kappa_over_t': tri.kappa_over_t.tolist(),
                             'band_sum': [None if np.isnan(x) else x for x in tri.kappa_over_t_band_sum]}
    conditions = ExternalConditions(field=(0, 0, 1.0))
    state = refine_classical(triangle, state_120(triangle, ((1, 0, 0), (0, 0, 1))), conditions)
    canted = quiet(thermal_hall, solve_lswt(triangle, state, conditions,
                                            settings=LSWTSettings(mesh=(24, 24))), ts)
    out['triangular_canted_h1'] = {'kappa_over_t': canted.kappa_over_t.tolist()}
    out['haldane_gap_closing_t0.3'] = [
        {'D': D, 'kappa_over_t': float(thermal_hall(haldane(D, (24, 24))[2], [0.3]).kappa_over_t[0])}
        for D in (0.0, 1e-3, 1e-2, 0.05, 0.1, 0.2)]
    return out


def legacy_si():
    from types import SimpleNamespace

    from spintoolkit.observables.topology import Topology
    from tests.test_methods.test_thermal import nbcp_y

    model, state, conditions = nbcp_y({'JPD': 0.01})
    state = refine_classical(model, state, conditions)
    result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(12, 12)))
    dx, dy = result.hamiltonian_derivatives_at(result.k_points)
    k_data = {i: [[H, dx[i], dy[i]], [E, T], [True, None]]
              for i, (H, E, T) in enumerate(zip(result.hamiltonians, result.eigenvalues,
                                                result.eigenvectors))}
    parent = SimpleNamespace(Ns=result.num_sites, bz_data={'area': None},
                             system=SimpleNamespace(lattice_vectors=result.magnetic_lattice))
    rows = []
    for kelvin in (0.2, 0.5, 1.0):
        legacy = Topology(parent).compute_thermal_Hall(k_data, kelvin)[2]
        ours = quiet(thermal_hall, result, [K_BOLTZMANN_MEV * kelvin], gapless=True).kappa_over_t[0]
        si = ours * K_BOLTZMANN_MEV ** 2 * kelvin / H_BAR_MEV * 1.602176634e-22
        rows.append({'kelvin': kelvin, 'legacy_W_per_K': legacy, 'converted_W_per_K': si,
                     'relative_difference': abs(si / legacy - 1)})
    return {'case': 'NBCP Y 0.2 T, J_PD = 0.01 meV, 12 x 12 mesh, E0 = meV', 'rows': rows}


def nbcp_convergence():
    scan = json.loads(SCAN.read_text())
    rows = []
    ts = [0.01, 0.02, 0.05, 0.1]
    for phase, field, extra in (('Y', 0.2, {'JPD': 0.01}), ('Y', 0.2, {'JGamma': 0.01}),
                                ('V', 1.4, {'JPD': 0.01}), ('V', 1.4, {'JGamma': 0.01})):
        model = nbcp.build_model({'Jxy': 0.075, 'Jz': 0.125, **extra})
        theta = np.array(scan['states'][phase]['theta'])
        conditions = ExternalConditions(field=(0, 0, 4.645 * MU_B_MEV_PER_T * field))
        state = refine_classical(model, nbcp.candidate_state(
            model, 'three_msl', np.column_stack([theta, np.zeros(3)]).ravel()), conditions)
        for n in (12, 24, 48, 96, 192):
            start = time.time()
            result = solve_lswt(model, state, conditions, settings=LSWTSettings(mesh=(n, n)))
            curvature = berry_curvature(result)
            hall = quiet(thermal_hall, result, ts, gapless=True, curvature=curvature)
            rows.append({'phase': phase, 'field_T': field, 'couplings': extra, 'mesh': n,
                         'temperatures_meV': ts, 'kappa_over_t': hall.kappa_over_t.tolist(),
                         'max_pair_minus_band_sum': float(np.nanmax(np.abs(
                             hall.kappa_over_t - hall.kappa_over_t_band_sum))),
                         'max_abs_curvature': float(np.nanmax(np.abs(curvature.curvature))),
                         'min_particle_band_gap_meV': float(np.min(np.diff(curvature.energies, axis=1))),
                         'min_magnon_energy_meV': float(np.min(curvature.energies)),
                         'gapless': hall.gapless, 'seconds': round(time.time() - start, 1)})
    return rows


def main():
    report = {'independent_haldane': independent_haldane(), 'forms_and_limits': forms_and_limits(),
              'null_and_continuity': null_and_continuity(), 'legacy_si': legacy_si(),
              'nbcp_convergence': nbcp_convergence()}
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()
