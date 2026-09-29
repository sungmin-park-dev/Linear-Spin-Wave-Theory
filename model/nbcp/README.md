# NBCP model workspace

This directory owns NBCP-specific model definitions. Reusable numerical methods
remain in `code-space/spintoolkit/`; the model builders do not select a solver.

| File | Responsibility |
| --- | --- |
| `exchange.py` | Assemble the three NN and NNN exchange matrices from NBCP parameters. |
| `unit_cells.py` | Build the one-, two-, three-, and four-sublattice `SpinSystem` candidates, including sites, spin directions, bonds, and lattice vectors. |
| `__init__.py` | Expose the model builders for repository calculations. |

## Using the model

Run from the repository root with `spintoolkit` installed, or set `PYTHONPATH=code-space:.`:

```python
import numpy as np

from model.nbcp import make_nn_exchange_matrices, three_msl

config = {"Jxy": 0.076, "Jz": 0.125, "h": (0.0, 0.0, 0.2)}
exchange = make_nn_exchange_matrices(config)
system = three_msl(config, angles=(0.3, 0.0, -0.3, 0.0, np.pi, 0.0),
                   Exch_J=exchange)
```

The existing builders use spin 1/2 and unit nearest-neighbor distance. Exchange
coefficients and `h` must share an energy unit; `h` is the Zeeman energy vector,
not a field in tesla. Builders only add the explicitly supplied `Exch_J` and
`Exch_K` bonds. Missing angles preserve the existing NumPy random initialization.

This is a repository workspace, not an additional installed Python package.
`examples/nbcp_ground_state.py` keeps its run configuration, candidate search
settings, optimization, and plotting. It re-exports the moved builders so existing
imports continue to work. Model-only examples import `model.nbcp` directly.
Research provenance lists include the new model files.

## Extraction verification (2026-09-23)

The [verification record](verification/model-extraction-2026-09-23.json) separates
the latest source checkout's inherited solver changes from this model extraction.
Seven solver-related source files and seven solver test files were copied as the
baseline; the original checkout was not edited.

All seven existing function bodies and signatures are unchanged. Forty model and
Hamiltonian comparisons and four full candidate searches at mesh `N=3` matched
the pre-extraction implementation exactly. The legacy regressions cover twelve
zero-DM cell/parameter combinations and thirty-six classical-energy evaluations.
Archived quantum-Hamiltonian and DM-rotation differences are excluded from those
legacy comparisons.

The new model tests passed (17). The full suite changed from 198 passed / 5 failed
to 215 passed / 5 failed, with the same five pre-existing magnetic-structure
angle-count failures. This is behavioral regression evidence, not acceptance of
a physical model or a converged phase diagram.

## Subsequent organization

NBCP-specific exploratory calculations and raw/intermediate reusable results
belong under this workspace. Only curated results belong in `data-space/`.
Existing outputs have not moved in this first extraction step.

Result reuse/cache machinery, shared physical-quantity and physical-constant
definitions, and further separation of search/calculation scripts are deferred.
No empty result directories or new numerical methods are introduced here.
