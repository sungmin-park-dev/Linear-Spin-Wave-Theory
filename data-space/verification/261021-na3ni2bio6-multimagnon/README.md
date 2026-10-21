# Na₃Ni₂BiO₆: few-magnon instabilities above saturation (2026-10-21)

Research code for the material-application flow (`workbench/flow/nnbo-*.yaml`).
It does not use spin-toolkit: the quantities are exact few-magnon problems and
product-state energies that the package does not provide.

Model (U(1) core, field along the easy axis z ∥ c*, S = 1, honeycomb):

H = Σ_{i<j} J_ij S_i·S_j − D Σ_i (S_i^z)² − h Σ_i S_i^z,  J1 (nearest, < 0), J3 (third neighbour, > 0).

Energies are measured from the polarized state without the Zeeman term, in units
of |J1| (J1 = −1). An n-magnon state has energy E_n + n h, so the field where an
n-magnon object becomes unstable is h_n = −E_n / n.

| File | What it does |
|---|---|
| `twomag.py` | one-magnon matrix, sparse two-magnon matrix on an L×L torus, brute-force ED for checks |
| `nmag.py` | general n-magnon sector (sites carry 0, 1 or 2 magnons) |
| `prodsearch.py` | all S^z product states (m = 1, 0, −1) in every supercell of ≤ n cells; largest field where one beats the polarized state |
| `su3mf.py` | same with general SU(3) product states (≤ 8 sites) |
| `envelope.py`, `collin.py` | product-state magnetization curve; energies relative to the zigzag–polarized line (shows the chain degeneracy) |
| `check.py`, `check_n.py` | two-magnon and n = 1–4 magnon spectra against full ED of the 8-site torus (agree to 1e-14) |
| `run1.py`, `scan.py`, `conv.py`, `dc.py`, `fitpt.py`, `run_n.py`, `run_n4.py` | runs; outputs collected in `results.txt` |

Main numbers (details in `results.txt`):

- Two-magnon bound pairs exist for both published parameter sets, but the polarized state collapses to zigzag (first order) at a higher field: Valentí-like (J3/|J1| = 0.37, D/|J1| = 0.21) h₂ = 0.061 vs h_col = 0.112; Shangguan (1.0, 3.0) h₂ = 1.081 vs 2.000.
- Pair binding does not depend on small D, so h₂ = h₂(0) − D; upper bound for pairs to beat the collapse: D_c/|J1| ≈ 0.115, 0.160, 0.230, 0.364, 0.508 at J3/|J1| = 0.34, 0.37, 0.40, 0.45, 0.50.
- Refit of the U(1) core to the saturation field (~7.8 T, g = 2.16) and the mean Curie–Weiss temperature (23.8 K): J3/|J1| ≈ 0.49, |J1| ≈ 2.0 meV; pair binding there ≤ 0.017 |J1|.
- Four-magnon clusters beat pairs at every point tried (32, 50, 72 sites): at (0.37, 0) h₄ − h₂ = 0.043, 0.028, 0.023 |J1|. Pair (quadrupolar) nematic rejected in this model; a four-magnon condensate is not excluded (h₅ < h₄ on 32 sites).
- Product states: at h_s = J1 + 3J3 every arrangement of zigzag chains with isolated down chains (M = 1/5, 1/3, 1/2, 3/5, 2/3, …) has the same energy, for any J1, J3, D.

Not checked: quantum corrections to the collapse field, U(1)-breaking terms (Kitaev, axis tilt), biquadratic exchange, n ≥ 5 on large lattices.
