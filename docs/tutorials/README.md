# Tutorials

Five short tutorials that take you from a first calculation to topology and
neutron spectra. Each one has a companion script in
[`examples/tutorials/`](../../examples/tutorials/) that prints every number
quoted in the text, and `code-space/tests/test_tutorials.py` checks those
numbers on every commit. Each tutorial compares the package with a closed-form
result, the literature or exact diagonalization, so you also see how to
check a calculation of your own.

| Tutorial | You learn | Checked against |
|---|---|---|
| [1. First calculation](01-first-calculation.md) | model, ordered state, `solve_lswt`, energies, bands | Chubukov, Sachdev and Senthil (1994); closed-form magnon energy |
| [2. Your own model](02-own-model.md) | `SpinModel` from `Site` and `Term`, `classical_search`, `compare_states` | square-lattice spin-wave energy −0.6579 J |
| [3. Neutron scattering](03-neutron.md) | `neutron_intensity`, paths, powder average, Bragg peaks | closed-form Néel magnon energy and weight |
| [4. Magnetization curve](04-magnetization.md) | `magnetization_curve` at 1/S order, `solve_ed` | exact diagonalization of a 4 × 4 torus |
| [5. Topology and thermal Hall](05-topology.md) | Berry curvature, Chern numbers, `thermal_hall` | closed-form gap at K; D → −D symmetry |

Run every script from the repository root, for example

```bash
pip install .
python examples/tutorials/t01_first_calculation.py
```

The figures go to `data-space/tutorials/`.

## Units and conventions

All quantities are dimensionless. Energies are in the unit E0 of your
coupling constants (here J), fields are h = μ_B B / E0 with the g-tensor of
the model's Zeeman term, and temperatures are t = k_B T / E0. Momenta are in
inverse units of the lattice length you chose. Energies are per spin.

Spin-wave theory expands about an ordered classical state. When a quantity
is not defined at that order (a Chern number at a band touching, the
inelastic weight of a Goldstone mode at a Bragg vector), the package returns
NaN or raises an error instead of an arbitrary number. The tutorials show
where this happens.
