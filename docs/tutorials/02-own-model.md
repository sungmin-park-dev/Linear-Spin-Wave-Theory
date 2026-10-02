# 2. Your own model: the square-lattice J1–J2 Heisenberg model

Script: [`examples/tutorials/t02_own_model.py`](../../examples/tutorials/t02_own_model.py)

This tutorial builds a model from scratch, finds its classical ground state
without a guess, and ranks candidate orders. The model is

$$H = J_1 \sum_{\langle ij \rangle} \mathbf{S}_i\cdot\mathbf{S}_j
    + J_2 \sum_{\langle\langle ij \rangle\rangle} \mathbf{S}_i\cdot\mathbf{S}_j
    - \mathbf{h}\cdot\sum_i \mathbf{S}_i,$$

with J1 = 1 and S = 1/2.

## Building a SpinModel

A model is a lattice, a tuple of `Site`s and a tuple of `Term`s:

```python
import numpy as np
import spintoolkit as stk

def square_j1j2(J2, S=0.5):
    site = stk.Site("A", (0.0, 0.0), S)
    terms = [stk.Term.bilinear(("A", (0, 0)), ("A", d), np.eye(3), label="J1")
             for d in [(1, 0), (0, 1)]]
    terms += [stk.Term.bilinear(("A", (0, 0)), ("A", d), J2 * np.eye(3), label="J2")
              for d in [(1, 1), (1, -1)]]
    terms.append(stk.Term.zeeman("A", np.eye(3)))
    return stk.SpinModel(np.eye(2), (site,), tuple(terms),
                         {"model_id": "square_j1j2", "parameters": {"J2": J2, "S": S}})
```

- `Site(id, position, spin)` places a spin in the unit cell. The position is
  Cartesian, in the units of the lattice vectors.
- `Term.bilinear((site_i, cell_i), (site_j, cell_j), M)` adds
  S_i^a M_ab S_j^b for every translate of the bond. The cell offsets are in
  units of the lattice vectors. List each bond once. Here (1, 0) and (0, 1)
  cover all nearest-neighbour bonds of the square lattice, and (1, 1) and
  (1, −1) all second-neighbour bonds.
- The 3 × 3 matrix `M` can be any exchange: `stk.heisenberg`, `stk.xxz`,
  `stk.dzyaloshinskii_moriya` and `stk.kitaev` build common ones.
  An antisymmetric part is a DM interaction.
- `Term.zeeman(site, g)` couples the site to the field of
  `ExternalConditions(field=h)` as −h·g·S, where h = μ_B B in units of E0.
  Use `Term.onsite(site, A)` (S·A·S) for
  single-ion anisotropy when S ≥ 1.

## Finding the classical ground state

```python
from spintoolkit.methods.classical import classical_search

search = classical_search(model, np.diag([2, 2]))
```

`classical_search` minimizes the classical energy of every spin on the given
magnetic supercell (here 2 × 2) by differential evolution and then refines
it with an analytic gradient. At J2 = 0 it finds the Néel state:

```
classical energy -0.5000 J1 per spin
('A', (0, 0)) [-0.  0.  1.]
('A', (0, 1)) [ 0. -0. -1.]
('A', (1, 0)) [ 0. -0. -1.]
('A', (1, 1)) [-0.  0.  1.]
```

The overall direction (here z) is arbitrary because the model is
spin-rotation invariant. The search only explores the supercell you give.
An order that does not fit it (for example a spiral with a long period) is
missed, so try several supercells, or use Luttinger–Tisza
(`spintoolkit.methods.luttinger_tisza`) to find the ordering vector first.

The state goes straight into `solve_lswt`:

```
E_gs = -0.6579 J1 per spin, <S> = 0.3072
```

This is the spin-wave result E/N = −2J1 S(S + 0.158) for the square lattice
(Anderson 1952; Oguchi 1960). The ordered moment converges slowly with the
mesh to 0.303, for the reason explained in tutorial 1.

## Comparing candidate states

For J2 > J1/2 the classical ground state changes from Néel to the stripe
state. `compare_states` refines each candidate to a stationary point, solves
LSWT, and ranks the stable ones by E_cl + E_zp:

```python
from spintoolkit.methods.phase_competition import compare_states

neel = stk.SpinState.from_function(
    model, [[1, 1], [1, -1]], lambda site, cell: np.array([1.0, 0, 0]) * (-1) ** (cell[0] + cell[1]))
stripe = stk.SpinState.from_function(
    model, np.diag([2, 1]), lambda site, cell: np.array([1.0, 0, 0]) * (-1) ** cell[0])
reports = compare_states(model, {"Neel": neel, "stripe": stripe}, k_density=48)
```

`SpinState.from_function(model, supercell, f)` sets the direction of each
site from its cell index. The Néel state lives on the √2 × √2 cell with rows
(1, 1) and (1, −1).

```
J2 = 0.2: Neel   stable   E_cl = -0.4000  E_cl + E_zp = -0.5836
J2 = 0.2: stripe unstable E_cl = -0.1000  E_cl + E_zp = nan
J2 = 0.8: stripe stable   E_cl = -0.4000  E_cl + E_zp = -0.5712
J2 = 0.8: Neel   unstable E_cl = -0.1000  E_cl + E_zp = nan
```

A state that is a stationary point but not a minimum has negative magnon
energies. It is reported as `unstable`, and its harmonic energy is NaN, not a
number to compare. Near J2 = J1/2 neither state is stable at harmonic order,
and a phase diagram should leave that region undefined. See
`examples/gallery.py` (`j1j2_phase_diagram`) for the full scan drawn with
`plot_phase_diagram`.
