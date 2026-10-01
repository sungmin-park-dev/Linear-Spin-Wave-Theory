"""NBCP unpolarized neutron intensity in a transverse field (D41).

Woodland et al. (arXiv:2505.06398) measured NBCP with B || b* above and below
the critical field. This script computes the LSWT neutron intensity for their
Table 1 parameters with the g-tensor of the same table and the Co2+ dipole
form factor (spin-only, j2 weight 0):

1. Polarized phase, 3.5 T. One magnon per momentum with Eq. (5) energy
   w = sqrt(a^2 - b^2), a = g_ab mu_B B - 3 Jxy + (Jz + Jxy) gamma / 2,
   b = (Jz - Jxy) gamma / 2, gamma = sum_d cos(k . d). The fluctuations
   along x (coefficient a - b, coupling Jxy) and along z (a + b, coupling Jz)
   are the two quadratures of one oscillator, so

       W_xx = (S/2) sqrt((a + b) / (a - b)),   W_zz = (S/2) sqrt((a - b) / (a + b)),
       I(Q, w) = F^2 [g_ab^2 (1 - Q_x^2/Q^2) W_xx + g_c^2 (1 - Q_z^2/Q^2) W_zz] / 4.

   The script checks the toolkit against this closed form on the path and
   for several Q_z, which tests the g-tensor, form factor and the polarization
   factor on the NBCP model itself.
2. Three-sublattice phase, 1.0 T (reference state (I) of
   ``nbcp_three_sublattice_bands.py``). The twofold rotation about b* is a
   symmetry of the XXZ model in a field along b*. It swaps the +c and -c
   canted sublattices together with their positions, so it maps state (I)
   onto itself up to a translation: the domain average over {E, C2} equals
   the single-domain intensity (checked at generic Q with Q_z != 0). The
   translation domains give the same intensity as well, so this phase needs
   no domain average for unpolarized neutrons. The map is shown at Q_z = 0
   and at Q_z = 1.5 / a, where the polarization factor suppresses the
   out-of-plane fluctuations.

The in-plane lattice constant enters only the form-factor envelope through
|Q| in 1/A; ``LATTICE_CONSTANT`` is an assumed value (the repository has no
structural data). The intensity is per Co site without the instrument
prefactor (gamma r0)^2 k_f/k_i exp(-2W). The paper's data are not in the
repository, so the maps are not compared with the measurement.

Usage
-----
    python examples/nbcp_neutron_spectrum.py > report.json
"""

import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from model import nbcp
from model.nbcp.model import LATTICE, PARAMETER_SETS, build_published_model
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods.classical import refine_classical
from spintoolkit.methods.lswt import LSWTSettings, solve_lswt
from spintoolkit.models import polarized_state
from spintoolkit.observables.neutron import FormFactor, domain_average, neutron_intensity
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.high_symmetry import high_symmetry_points

OUT = ROOT / 'data-space/verification/261001-nbcp-neutron-spectrum'
PARAMS = PARAMETER_SETS['woodland2025']['parameters']
G = PARAMETER_SETS['woodland2025']['g']
S = 0.5
LATTICE_CONSTANT = 5.3          # angstrom, assumed (nearest-neighbour Co-Co distance)
FORM_FACTOR = FormFactor.from_ion('Co2')
FWHM = 0.02                     # meV, energy resolution of the maps
C2_B = np.diag([-1.0, 1.0, -1.0])   # twofold rotation about b* (y)


def conditions_at(field_t):
    return ExternalConditions(field=(0.0, MU_B_MEV_PER_T * field_t, 0.0))


def path(model, points=241):
    K = high_symmetry_points(model.lattice)['K']
    corners = [-1.5 * K, -K, 0 * K, K, 1.5 * K]
    segments = [np.linspace(a, b, points // 4, endpoint=False) for a, b in zip(corners, corners[1:])]
    q = np.vstack(segments + [corners[-1][None]])
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]
    ticks = [distance[i * (points // 4)] for i in range(5)]
    return q, distance, ticks


def closed_form(q, qz, field_t):
    deltas = np.array([LATTICE[0], LATTICE[1], LATTICE[0] + LATTICE[1]])
    gamma = np.cos(q @ deltas.T).sum(axis=1)
    a = G[1, 1] * MU_B_MEV_PER_T * field_t - 3 * PARAMS['Jxy'] + (PARAMS['Jz'] + PARAMS['Jxy']) / 2 * gamma
    b = (PARAMS['Jz'] - PARAMS['Jxy']) / 2 * gamma
    W_xx = S / 2 * np.sqrt((a + b) / (a - b))
    W_zz = S / 2 * np.sqrt((a - b) / (a + b))
    Q = np.column_stack([q, np.full(len(q), qz)])
    u2 = Q ** 2 / np.sum(Q ** 2, axis=1)[:, None]
    F = FORM_FACTOR(np.linalg.norm(Q, axis=1) / LATTICE_CONSTANT)
    intensity = F ** 2 * (G[0, 0] ** 2 * (1 - u2[:, 0]) * W_xx + G[2, 2] ** 2 * (1 - u2[:, 2]) * W_zz) / 4
    return np.sqrt(a ** 2 - b ** 2), intensity, W_xx, W_zz


def three_sublattice_state(model, field_t, starts=30, seed=0):
    from spintoolkit.methods.classical import classical_energy
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(starts):
        angles = np.column_stack([np.arccos(rng.uniform(-1, 1, 3)), rng.uniform(0, 2 * np.pi, 3)])
        state = refine_classical(model, nbcp.candidate_state(model, 'three_msl', angles.ravel()),
                                 conditions_at(field_t))
        energy = classical_energy(model, state, conditions_at(field_t))
        if best is None or energy < best[0] - 1e-12:
            best = (energy, state)
    return best[1]


def main():
    model = build_published_model('woodland2025')
    kwargs = dict(g=G, form_factor=FORM_FACTOR, length_unit=LATTICE_CONSTANT)
    q, distance, ticks = path(model)
    omega = np.linspace(0, 1.0, 401)
    report = {'parameters': {**PARAMS, 'g': np.diag(G).tolist(), 'S': S},
              'lattice_constant_A_assumed': LATTICE_CONSTANT, 'form_factor': 'Co2 <j0>',
              'source': 'arXiv:2505.06398 Table 1; B || b*'}

    # 1. Polarized phase against the closed form.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        polarized = solve_lswt(model, polarized_state(model, (0, 1, 0)), conditions_at(3.5),
                               settings=LSWTSettings(mesh=(6, 6)))
    checks = []
    for qz in (0.0, 0.5, 1.5):
        Q = np.column_stack([q, np.full(len(q), qz)])
        keep = np.linalg.norm(Q, axis=1) > 0
        computed = neutron_intensity(polarized, Q[keep], **kwargs)
        energy, intensity, _, _ = closed_form(q[keep], qz, 3.5)
        particle = computed.energies > 0
        checks.append({'Q_z_per_model_length': qz,
                       'max_energy_deviation_meV': float(np.max(np.abs(
                           computed.energies[particle] - energy))),
                       'max_intensity_deviation': float(np.max(np.abs(
                           computed.intensities[particle] - intensity))),
                       'max_intensity': float(np.max(intensity))})
    gamma_point = closed_form(np.zeros((1, 2)), 1.0, 3.5)          # W does not depend on Q_z
    K = high_symmetry_points(model.lattice)['K']
    k_point = closed_form(K[None], 0.0, 3.5)
    report['polarized_3p5T'] = {
        'closed_form_checks': checks,
        'W_xx_over_W_zz': {'Gamma': float(gamma_point[2][0] / gamma_point[3][0]),
                           'K': float(k_point[2][0] / k_point[3][0])}}
    map_polarized = neutron_intensity(polarized, q, **kwargs).broaden(omega, FWHM)

    # 2. Three-sublattice phase, domain average over {E, C2 about b*}.
    state = three_sublattice_state(model, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        three = solve_lswt(model, state, conditions_at(1.0), settings=LSWTSettings(mesh=(12, 12)))
        single = neutron_intensity(three, q, **kwargs).broaden(omega, FWHM)
        tilted = neutron_intensity(three, np.column_stack([q, np.full(len(q), 1.5)]),
                                   **kwargs).broaden(omega, FWHM)
        rng = np.random.default_rng(1)
        generic = np.column_stack([rng.uniform(-4, 4, (20, 2)), rng.uniform(-2, 2, 20)])
        one = neutron_intensity(three, generic, **kwargs).broaden(omega, FWHM)
        averaged = domain_average(three, generic, omega, FWHM, [np.eye(3), C2_B], **kwargs)
    report['three_sublattice_1T'] = {
        'directions': {str(k): np.round(v, 6).tolist() for k, v in state.directions.items()},
        'C2_b_domain_average_minus_single_relative': float(
            np.nanmax(np.abs(one - averaged)) / np.nanmax(one)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    extent = [distance[0], distance[-1], omega[0], omega[-1]]
    panels = [(map_polarized, '3.5 T, polarized, Q_z = 0'),
              (single, '1.0 T, three-sublattice, Q_z = 0'),
              (tilted, '1.0 T, three-sublattice, Q_z = 1.5 / a')]
    for ax, (data, title) in zip(axes, panels):
        ax.imshow(np.nan_to_num(data).T, origin='lower', aspect='auto', extent=extent,
                  cmap='viridis', vmax=np.nanpercentile(data, 99.5))
        ax.set_xticks(ticks, ['M', "K'", 'Γ', 'K', 'M'])
        ax.set_title(title, fontsize=10)   # Q = 0 (Gamma at Q_z = 0) has no polarization factor: blank
    axes[0].set_ylabel('E (meV)')
    fig.suptitle('NBCP LSWT neutron intensity, B ∥ b*, Co²⁺ form factor, '
                 f'FWHM {FWHM} meV (arXiv:2505.06398 Table 1)', fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    figure = OUT / 'neutron-intensity.png'
    fig.savefig(figure, dpi=150)
    report['figure'] = str(figure.relative_to(ROOT))
    text = json.dumps(report, indent=1, ensure_ascii=False)
    (OUT / 'report.json').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
