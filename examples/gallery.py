"""Figure gallery: one example of every plot in ``spintoolkit.visualization``.

Energies are in units of the exchange J (E0), momenta in inverse lattice
constants. Every figure is computed from scratch.

1. Triangular S = 1/2 Heisenberg antiferromagnet, 120 degree state: bands,
   neutron I(Q, w) along Gamma-K-M-Gamma, the slice at w = 1.0 J (the flat
   line Q_x = pi, where gamma_k = -1/3 for every Q_y, has w = 1.0 J
   exactly) and the powder average. K is a Bragg vector: its Goldstone weight
   is undefined and drawn grey.
2. Honeycomb ferromagnet with next-nearest-neighbour DM (D = 0.1 J) in a field
   h = 0.1: Berry curvature of both bands (Chern numbers +-1) and kappa_xy / T.
3. Textures: a 12 x 12 skyrmion (Q = -1), the tetrahedral state (Q = -2 per
   four sites) and the coplanar 120 degree state (Q undefined).
4. Triangular ferromagnet in a field: specific heat and magnetization, with
   the range where <n> > S shaded.
5. Square-lattice J1-J2 model (S = 1/2) in a field: Neel against stripe,
   each refined to its canted stationary point. The colour is the lower
   E_cl + E_zp among the LSWT-stable candidates. Around J2 = J1 / 2 neither
   state is stable at harmonic order (the classical degeneracy gives soft
   lines), so the winner is "undefined" there rather than either state.

Usage
-----
    python examples/gallery.py          # writes data-space/gallery/*.png
"""

from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space')]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import spintoolkit as stk
from spintoolkit.methods.phase_competition import compare_states
from spintoolkit.models import (honeycomb_ferromagnet, polarized_state, state_120,
                                triangular_heisenberg)
from spintoolkit.observables.bands import band_structure
from spintoolkit.observables.berry import berry_curvature, thermal_hall
from spintoolkit.observables.neutron import neutron_path, neutron_slice, powder_average
from spintoolkit.observables.thermal import thermal_quantities
from spintoolkit.visualization import (plot_bands, plot_berry_curvature, plot_intensity_path,
                                       plot_intensity_slice, plot_phase_diagram, plot_powder,
                                       plot_spin_configuration, plot_spin_texture,
                                       plot_thermal_hall, plot_thermodynamics)

OUT = ROOT / 'data-space/gallery'


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f'{name}.png', dpi=130)
    plt.close(fig)
    print(OUT / f'{name}.png')


def skyrmion(model, L):
    """Core -z, background +z, vorticity +1 on an L x L cell (Q = -1)."""
    A = model.lattice
    period = np.diag([L, L]) @ A
    centre = (np.array([L / 2, L / 2]) + 0.25) @ A
    radius = L * np.linalg.norm(A[0]) / 4

    def direction(site, cell):
        r = np.asarray(cell, float) @ A - centre
        f = np.linalg.solve(period.T, r)
        r = (f - np.round(f)) @ period
        theta = np.pi * max(0.0, 1 - np.linalg.norm(r) / radius)
        phi = np.arctan2(r[1], r[0])
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    return stk.SpinState.from_function(model, np.diag([L, L]), direction)


def tetrahedral(model):
    t = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3)
    return stk.SpinState.from_function(model, np.diag([2, 2]),
                                       lambda site, c: t[2 * (c[0] % 2) + (c[1] % 2)])


def triangular_spectra():
    model = triangular_heisenberg(J=1.0, S=0.5)
    state = state_120(model)
    result = stk.solve_lswt(model, state, settings=stk.LSWTSettings(mesh=(24, 24)))
    fig = plot_spin_configuration(model, state, n_repeat=2)[0]
    save(fig, '01-spin-configuration')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    plot_bands(band_structure(result, ('Γ', 'K', 'M', 'Γ')), axes[0])
    axes[0].set_title('magnon bands (folded, magnetic cell)', fontsize=9)
    omega = np.linspace(0, 2, 241)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')               # Goldstone weight at K is NaN
        path = neutron_path(result, omega, 0.04, points=300, g=2.0)
    plot_intensity_path(path, axes[1])
    axes[1].set_title('neutron intensity, g = 2, FWHM 0.04 J', fontsize=9)
    save(fig, '02-bands-and-neutron-path')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cut = neutron_slice(result, 1.0, 0.1, points=161, g=2.0)
        Q = np.linspace(0.05, 8, 120)
        powder = powder_average(result, Q, omega, 0.04, num_directions=400, g=2.0)
    plot_intensity_slice(cut, axes[0])
    plot_powder(Q, omega, powder, axes[1])
    axes[1].set_title('powder average', fontsize=9)
    save(fig, '03-neutron-slice-and-powder')


def honeycomb_topology():
    model = honeycomb_ferromagnet(J=1.0, D=0.1)
    result = stk.solve_lswt(model, polarized_state(model), stk.ExternalConditions(field=(0, 0, 0.1)),
                            settings=stk.LSWTSettings(mesh=(48, 48)))
    curvature = berry_curvature(result)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    plot_berry_curvature(curvature, result.magnetic_lattice, 0, axes[0])
    plot_berry_curvature(curvature, result.magnetic_lattice, 1, axes[1])
    plot_thermal_hall(thermal_hall(result, np.linspace(0.05, 1.5, 30)), axes[2])
    axes[2].set_title('magnon thermal Hall', fontsize=9)
    save(fig, '04-berry-curvature-and-thermal-hall')


def textures():
    model = triangular_heisenberg()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    plot_spin_texture(model, skyrmion(model, 12), axes[0], cells=(12, 12))
    plot_spin_texture(model, tetrahedral(model), axes[1])
    plot_spin_texture(model, state_120(model), axes[2])
    save(fig, '05-spin-textures')


def thermodynamics():
    model = triangular_heisenberg(J=-1.0)
    result = stk.solve_lswt(model, polarized_state(model), stk.ExternalConditions(field=(0, 0, 0.2)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')               # <n> > S reported by the shading
        thermal = thermal_quantities(result, np.linspace(0.01, 3, 120))
    axes = plot_thermodynamics(thermal, ('specific_heat', 'entropy', 'magnetization'))
    save(axes[0].figure, '06-thermodynamics')


def square_j1j2(J2, S=0.5):
    """Square-lattice J1-J2 Heisenberg model (J1 = 1) with an isotropic Zeeman term."""
    terms = [stk.Term.bilinear(('A', (0, 0)), ('A', d), np.eye(3), label='J1')
             for d in [(1, 0), (0, 1)]]
    terms += [stk.Term.bilinear(('A', (0, 0)), ('A', d), J2 * np.eye(3), label='J2')
              for d in [(1, 1), (1, -1)]]
    terms.append(stk.Term.zeeman('A', np.eye(3)))
    return stk.SpinModel(np.eye(2), (stk.Site('A', (0.0, 0.0), S),), tuple(terms),
                         {'model_id': 'square_j1j2'})


def j1j2_phase_diagram():
    couplings = np.round(np.linspace(0.0, 1.0, 21), 3)
    fields = np.round(np.linspace(0.0, 2.0, 9), 3)
    phases = np.full((len(fields), len(couplings)), None, dtype=object)
    for i, h in enumerate(fields):
        for j, J2 in enumerate(couplings):
            model = square_j1j2(J2)
            candidates = {
                'Néel': stk.SpinState.from_function(
                    model, [[1, 1], [1, -1]], lambda s, c: np.array([1.0, 0, 0]) * (-1) ** (c[0] + c[1])),
                'stripe': stk.SpinState.from_function(
                    model, np.diag([2, 1]), lambda s, c: np.array([1.0, 0, 0]) * (-1) ** c[0])}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                best = compare_states(model, candidates, stk.ExternalConditions(field=(0, 0, h)))[0]
            phases[i, j] = best.name if best.status == 'stable' else None
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    plot_phase_diagram(couplings, fields, phases, ax, xlabel=r'$J_2 / J_1$',
                       ylabel=r'$h / J_1$', colors={'Néel': 'C0', 'stripe': 'C1'})
    ax.set_title(r'square $J_1$-$J_2$, $S = 1/2$: lowest $E_{cl}+E_{zp}$ (LSWT-stable states)',
                 fontsize=9)
    save(fig, '07-phase-diagram')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    triangular_spectra()
    honeycomb_topology()
    textures()
    thermodynamics()
    j1j2_phase_diagram()


if __name__ == '__main__':
    main()
