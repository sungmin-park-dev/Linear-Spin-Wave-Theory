"""Solver-seam check: the same SpinModel solved by ED and by DMRG (TeNPy).

Handoff 260607 asks what each solver reads from the "system". This script
answers it on the toolkit's own types instead of a separate repository: a
model is expanded on a finite torus once (``expand_on_torus``, the D23 rules
shared by all methods), and that one list of sites, bonds (3 x 3 exchange
matrices) and site fields is handed to

- the toolkit ED (``solve_ed``), and
- a ~40-line TeNPy adapter (below): one MPS site per torus site in the
  torus order (S3: the MPS ordering is the only geometric input; torus bonds
  become couplings between arbitrary MPS sites), ``SpinSite(S,
  conserve='None')`` (S2), and every bond written as sum_ab J_ab S^a S^b with
  the named operators Sx, Sy, Sz (S1). No magnetic order or classical state
  is needed (S4).

Ground-state energies are compared with exact references:
- XX chain, L = 12, PBC: Jordan-Wigner free fermions with the parity-dependent
  boundary condition (exact for finite L);
- square 4 x 4 Heisenberg, PBC: E0/N = -0.70178020 (literature value, e.g.
  Schulz, Ziman and Poilblanc, J. Phys. I 6, 675 (1996));
- triangular 3 x 3 NBCP XXZ (Woodland 2025 parameters) in a field along c, in
  a transverse field (no U(1)), and with J_PD = 0.01 meV (complex S^y
  products): ED vs DMRG only.

TeNPy is an optional dependency of this check (pip install physics-tenpy);
it is not used by the package.

Usage
-----
    python examples/solver_seam_tenpy_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np

from model.nbcp.model import build_published_model
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.methods.ed import EDSector, solve_ed
from spintoolkit.models import square_heisenberg
from spintoolkit.system.cluster import expand_on_torus
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry
from spintoolkit.system.model import Site, SpinModel, Term

OPS = ('Sx', 'Sy', 'Sz')
OUT = ROOT / 'data-space/verification/261001-solver-seam-tenpy'


def dmrg_ground_energy(torus, conditions, chi_max=256, seed=7):
    """Lowest energy of the torus Hamiltonian by finite DMRG (TeNPy)."""
    import tenpy
    from tenpy.algorithms import dmrg
    from tenpy.models.lattice import Chain
    from tenpy.models.model import CouplingModel, MPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import SpinSite

    spins = set(np.round(torus.spins, 9))
    if len(spins) != 1:
        raise NotImplementedError('one spin length per torus in this adapter')
    site = SpinSite(S=float(spins.pop()), conserve='None')
    lattice = Chain(torus.num_sites, site, bc='open', bc_MPS='finite')

    class TorusModel(CouplingModel, MPOModel):
        def __init__(self):
            CouplingModel.__init__(self, lattice, explicit_plus_hc=False)
            for i, j, J in zip(torus.source, torus.target, torus.exchange):
                if i > j:                       # S_i . J . S_j = S_j . J^T . S_i
                    i, j, J = j, i, J.T
                for a in range(3):
                    for b in range(3):
                        if J[a, b] != 0:
                            self.add_coupling_term(float(J[a, b]), int(i), int(j), OPS[a], OPS[b])
            for i, h in enumerate(torus.fields(conditions)):
                for a in range(3):
                    if h[a] != 0:
                        self.add_onsite_term(-float(h[a]), i, OPS[a])
            MPOModel.__init__(self, lattice, self.calc_H_MPO())

    model = TorusModel()
    rng = np.random.default_rng(seed)
    psi = MPS.from_product_state(lattice.mps_sites(),
                                 [rng.normal(size=site.dim) for _ in range(torus.num_sites)],
                                 bc='finite')
    psi.canonical_form()
    info = dmrg.run(psi, model, {'mixer': True, 'max_E_err': 1e-12, 'max_sweeps': 60,
                                 'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-12}})
    return float(info['E']), tenpy.__version__, int(max(psi.chi))


def ed_ground_energy(model, geometry, conditions):
    return float(solve_ed(model, geometry, conditions, EDSector(), num_eigenvalues=1).energies()[0])


def chain_model(Jxy, Jz):
    """Spin-1/2 chain as a square-lattice model with bonds only along a1."""
    J = np.diag([Jxy, Jxy, Jz])
    return SpinModel(np.eye(2), [Site('A', (0.0, 0.0), 0.5)],
                     [Term.bilinear(('A', (0, 0)), ('A', (1, 0)), J, label='NN')],
                     {'model_id': f'chain_Jxy{Jxy}_Jz{Jz}'})


def xx_chain_exact(L, J=1.0):
    """Ground energy of J sum (SxSx + SySy) on a PBC ring (Jordan-Wigner)."""
    best = np.inf
    for n in range(L + 1):
        shift = 0.0 if n % 2 == 1 else np.pi / L       # periodic for odd n, antiperiodic for even
        eps = np.sort(J * np.cos(2 * np.pi * np.arange(L) / L + shift))
        best = min(best, eps[:n].sum())
    return float(best)


def case(name, model, cluster, conditions=None, reference=None):
    geometry = CalculationGeometry.finite_torus(np.array(cluster))
    torus = expand_on_torus(model, geometry)
    t0 = time.time()
    e_ed = ed_ground_energy(model, geometry, conditions)
    t1 = time.time()
    e_dmrg, version, chi = dmrg_ground_energy(torus, conditions)
    t2 = time.time()
    row = {'case': name, 'sites': torus.num_sites, 'bonds': len(torus.source),
           'max_mps_bond_range': int(np.max(np.abs(torus.source - torus.target))),
           'E_ed': e_ed, 'E_dmrg': e_dmrg, 'dmrg_minus_ed': e_dmrg - e_ed,
           'chi': chi, 'seconds_ed': round(t1 - t0, 2), 'seconds_dmrg': round(t2 - t1, 2),
           'tenpy': version}
    if reference is not None:
        row['E_reference'] = reference
        row['ed_minus_reference'] = e_ed - reference
    return row


def main():
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rows.append(case('XX chain L=12', chain_model(1.0, 0.0), [[12, 0], [0, 1]],
                         reference=xx_chain_exact(12)))
        rows.append(case('Heisenberg chain L=12', chain_model(1.0, 1.0), [[12, 0], [0, 1]]))
        rows.append(case('square 4x4 Heisenberg', square_heisenberg(), [[4, 0], [0, 4]],
                         reference=-0.70178020 * 16))
        nbcp = build_published_model('woodland2025')
        rows.append(case('NBCP 3x3, 1 T || c', nbcp, [[3, 0], [0, 3]],
                         ExternalConditions(field=(0, 0, MU_B_MEV_PER_T * 1.0))))
        rows.append(case('NBCP 3x3, 1 T || b*', nbcp, [[3, 0], [0, 3]],
                         ExternalConditions(field=(0, MU_B_MEV_PER_T * 1.0, 0))))
        rows.append(case('NBCP 3x3 + J_PD 0.01, 0.5 T || c',
                         build_published_model('woodland2025', JPD=0.01), [[3, 0], [0, 3]],
                         ExternalConditions(field=(0, 0, MU_B_MEV_PER_T * 0.5))))
    text = json.dumps({'cases': rows}, indent=1, ensure_ascii=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
