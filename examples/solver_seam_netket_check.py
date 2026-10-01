"""Solver-seam check, NQS: the same torus expansion read by NetKet.

Companion of ``solver_seam_tenpy_check.py`` (handoff 260607). The model is
expanded on a finite torus once (``expand_on_torus``, D23), and a ~20-line
NetKet adapter (below) writes every bond as sum_ab J_ab S^a_i S^b_j with
S = sigma / 2 and every site field as -h_i . S_i (the field convention
confirmed by the user on 2026-10-01, same sign as ED). As for DMRG, the
adapter needs only sites, bonds, exchange matrices and fields: no lattice
geometry, magnetic order or classical state (S1-S4).

Two separate questions are checked:

1. The seam: the NetKet operator's exact ground energy (``lanczos_ed``)
   equals the toolkit ED. This tests that the adapter reads the expansion
   correctly.
2. The method: an RBM (complex parameters, alpha = 2) optimized by VMC with
   stochastic reconfiguration, using exact sums over the Hilbert space
   (``FullSumState``, no sampling noise). Its relative energy error is a
   property of the ansatz and optimizer, not of the seam. The Hamiltonian is
   divided by Jz for the optimization only.

NetKet is an optional dependency of this check (pip install netket); it is
not used by the package.

Usage
-----
    python examples/solver_seam_netket_check.py > report.json
"""

import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT), str(ROOT / 'examples')]

import numpy as np

from model.nbcp.model import PARAMETER_SETS, build_published_model
from solver_seam_tenpy_check import ed_ground_energy
from spintoolkit.definitions.constants import MU_B_MEV_PER_T
from spintoolkit.models import square_heisenberg
from spintoolkit.system.cluster import expand_on_torus
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.geometry import CalculationGeometry

OUT = ROOT / 'data-space/verification/261001-solver-seam-netket'
VMC = {'alpha': 2, 'learning_rate': 0.01, 'diag_shift': 1e-2, 'iterations': 1500, 'seed': 3}


def netket_hamiltonian(torus, conditions):
    """The torus Hamiltonian as a NetKet LocalOperator (spin 1/2 only)."""
    import netket as nk

    if not np.allclose(torus.spins, 0.5):
        raise NotImplementedError('spin 1/2 only in this adapter')
    hilbert = nk.hilbert.Spin(s=0.5, N=torus.num_sites)
    pauli = (nk.operator.spin.sigmax, nk.operator.spin.sigmay, nk.operator.spin.sigmaz)
    H = nk.operator.LocalOperator(hilbert, dtype=complex)
    for i, j, J in zip(torus.source, torus.target, torus.exchange):
        for a in range(3):
            for b in range(3):
                if J[a, b] != 0:
                    H += 0.25 * float(J[a, b]) * pauli[a](hilbert, int(i)) @ pauli[b](hilbert, int(j))
    for i, h in enumerate(torus.fields(conditions)):
        for a in range(3):
            if h[a] != 0:
                H += -0.5 * float(h[a]) * pauli[a](hilbert, i)
    return hilbert, H


def vmc_energy(hilbert, H, scale):
    import netket as nk
    import optax

    state = nk.vqs.FullSumState(hilbert, nk.models.RBM(alpha=VMC['alpha'], param_dtype=complex),
                                seed=VMC['seed'])
    driver = nk.driver.VMC(H * (1 / scale), optax.sgd(VMC['learning_rate']), variational_state=state,
                           preconditioner=nk.optimizer.SR(diag_shift=VMC['diag_shift']))
    driver.run(n_iter=VMC['iterations'], show_progress=False)
    return float(state.expect(H).mean.real)


def case(name, model, cluster, conditions=None, scale=1.0, vmc=True, reference=None):
    import netket as nk

    geometry = CalculationGeometry.finite_torus(np.array(cluster))
    torus = expand_on_torus(model, geometry)
    e_ed = ed_ground_energy(model, geometry, conditions)
    hilbert, H = netket_hamiltonian(torus, conditions)
    e_exact = float(nk.exact.lanczos_ed(H, k=1)[0])
    row = {'case': name, 'sites': torus.num_sites, 'E_ed': e_ed,
           'netket_exact_minus_ed': e_exact - e_ed, 'netket': nk.__version__}
    if reference is not None:
        row['ed_minus_reference'] = e_ed - reference
    if vmc:
        t0 = time.time()
        e_vmc = vmc_energy(hilbert, H, scale)
        row.update({'E_rbm_vmc': e_vmc, 'rbm_relative_error': (e_vmc - e_ed) / abs(e_ed),
                    'seconds_vmc': round(time.time() - t0, 1)})
    return row


def main():
    jz = PARAMETER_SETS['woodland2025']['parameters']['Jz']
    nbcp = build_published_model('woodland2025')
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rows.append(case('NBCP 3x3, 1 T || c', nbcp, [[3, 0], [0, 3]],
                         ExternalConditions(field=(0, 0, MU_B_MEV_PER_T * 1.0)), scale=jz))
        rows.append(case('NBCP 3x3, 1 T || b*', nbcp, [[3, 0], [0, 3]],
                         ExternalConditions(field=(0, MU_B_MEV_PER_T * 1.0, 0)), scale=jz))
        rows.append(case('NBCP 3x3 + J_PD 0.01, 0.5 T || c',
                         build_published_model('woodland2025', JPD=0.01), [[3, 0], [0, 3]],
                         ExternalConditions(field=(0, 0, MU_B_MEV_PER_T * 0.5)), scale=jz))
        rows.append(case('square 4x4 Heisenberg', square_heisenberg(), [[4, 0], [0, 4]],
                         vmc=False, reference=-0.70178020 * 16))
    text = json.dumps({'vmc_settings': VMC, 'cases': rows}, indent=1, ensure_ascii=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'report.json').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
