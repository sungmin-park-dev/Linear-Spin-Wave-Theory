"""Large-S exact diagonalization check of the angle-dependent two-loop ingredients (D47).

The infinite-system pseudo-Goldstone gap cannot be compared with exact
diagonalization directly: on a finite cluster the k = 0 soft mode is a
quantum rotor whose large-S expansion has half-integer orders. What can be
compared exactly is the angle dependence that ``pseudo_goldstone_gap`` builds
on. A local field ``-lambda n_i(phi) . S_i`` along every spin of the rotated
state ``R(phi)`` keeps ``R(phi)`` stationary for every ``phi`` and gaps the
soft mode, so on a finite torus the 1/S expansion is regular and the
engine's perturbation theory is exact. Compared, per magnetic cell:

1. the order-S^0 energy ``E_2(phi)`` (Hartree-Fock, cubic, tadpole), whose
   curvature at ``phi = 0`` is the dominant part of ``U_2``;
2. the order-S^{-1/2} part of ``<a_0>``, the reference-frame boson of
   ``_field_terms`` (it sets ``xbar_2``), with the exact operator
   ``a = s^+ (S + 1 + s^z)^{-1/2}`` in ED.

Cluster: NBCP Y state on two magnetic cells (6 spins, momenta 0 and
b_1 / 2), J_PD = 0.010, h = 0.1075 and lambda = 0.05 meV at spin 1 (the
field and the pinning scale with S, so the classical state does not
change). ED at S = 2 ... 4.5; the order-S^0 coefficient is the intercept of
a polynomial fit in 1/S after subtracting the code's S^2 and S terms.

The reference boson uses the Holstein-Primakoff inverse expanded to first
order in ``n_0 / 2S``; classically ``n_0 / 2S = (1 - cos alpha) / 2`` for a
rotation by ``alpha`` away from the reference axis, so that expansion holds
for small rotations only (``pseudo_goldstone_gap`` uses ``|phi| <= 0.04``).
The large-angle rows show the expected deviation.

Run from the repository root (about one hour on four cores):
    python examples/nbcp_two_loop_ed_check.py
"""

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from model import nbcp
from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.methods.nlswt.engine import NonlinearSpinWaves, bogoliubov_mesh
from spintoolkit.methods.nlswt.expansion import expand_model
from spintoolkit.methods.nlswt.pseudo_goldstone import _Frame, _field_terms, rotate_state
from spintoolkit.system.conditions import ExternalConditions

OUT = ROOT / 'data-space/verification/261003-nbcp-two-loop-gap'
J, JZ, JPD, H1, LAM = 0.075, 0.125, 0.010, 0.1075, 0.05
CELLS = (2, 1)
MOMENTA = [(0.0, 0.0), (0.5, 0.0)]
PHIS = [0.0, 0.2, 0.4, np.pi / 6, np.pi / 3, 5 * np.pi / 6]
SPINS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


def y_model():
    model = nbcp.build_model({'Jxy': J, 'Jz': JZ, 'JPD': JPD}, spin=1.0)
    t = np.arccos((H1 + 3 * JZ) / (3 * (J + JZ)))
    state = nbcp.candidate_state(model, 'three_msl',
                                 np.column_stack([[t, -t, np.pi], np.zeros(3)]).ravel())
    return model, state


# ---- code side (spin 1: the coefficients of S^2, S, S^0, ...) ----------------------------

def code_side(phi):
    """Per magnetic cell: classical, zero-point and order-S^0 energies and ``<a_0>`` terms."""
    model, state = y_model()
    cond = ExternalConditions(field=[0, 0, H1])
    frames0 = expand_model(model, state, cond).local_frames
    rotated = rotate_state(model, state, (0, 0, 1), phi)
    quadratic = QuadraticBoseHamiltonian(model, rotated, cond)
    expansion = expand_model(model, rotated, cond)
    ns = expansion.num_sites
    hamiltonian = lambda k: np.asarray(quadratic.at(k)) + LAM * np.eye(2 * ns)
    reciprocal = 2 * np.pi * np.linalg.inv(rotated.magnetic_lattice(model)).T
    fg = np.array(MOMENTA)
    mesh = bogoliubov_mesh(hamiltonian, fg @ reciprocal, fg, reciprocal, 1e-9, 2)
    solver = NonlinearSpinWaves(expansion, mesh, hamiltonian)
    shift, tadpole, _ = solver.tadpole()
    frame = _Frame.__new__(_Frame)
    frame.expansion, frame.solvers = expansion, [solver]
    amplitude = []
    for s in range(ns):                           # <a_0,s> = sqrt(S) lead + (corr + c1.shift)/sqrt(S)
        terms = []
        for coefficient in (1.0, -1j):            # real and imaginary parts
            w = np.zeros(2 * ns, dtype=complex)
            w[s] = coefficient
            lead, corr, c1 = _field_terms(frame, frames0, w)
            terms.append((lead, corr + float(np.real(c1 @ shift))))
        amplitude.append([complex(terms[0][0], terms[1][0]), complex(terms[0][1], terms[1][1])])
    return {'classical': expansion.classical_energy - LAM * sum(expansion.spins),
            'zero_point': solver.zero_point_energy(),
            'order_s0': solver.hartree_fock_energy() + solver.cubic_energy() + tadpole,
            'amplitude': amplitude}


# ---- exact diagonalization ------------------------------------------------------------------

def spin_matrices(S):
    m = S - np.arange(int(round(2 * S + 1)))
    plus = np.diag(np.sqrt(S * (S + 1) - m[1:] * (m[1:] + 1)), 1)
    return [(plus + plus.T) / 2, (plus - plus.T) / 2j, np.diag(m).astype(complex)]


def cluster(model, state):
    """Sites, their sublattice, and the summed exchange of every ordered pair on the torus."""
    prim = np.asarray(model.lattice, dtype=float)
    magnetic = state.magnetic_lattice(model)
    Tinv = np.linalg.inv(np.diag(CELLS) @ magnetic)

    def key(r):
        f = np.mod(np.round(r @ Tinv, 9), 1.0)
        f[np.isclose(f, 1.0)] = 0.0
        return tuple(np.round(f, 6))

    sites, index, sublattice = [], {}, []
    for a in range(-6, 7):
        for b in range(-6, 7):
            r = a * prim[0] + b * prim[1]
            if key(r) in index:
                continue
            index[key(r)] = len(sites)
            sites.append(r)
            for s, cell in enumerate(state.cells):
                x = (r - np.asarray(cell) @ prim) @ np.linalg.inv(magnetic)
                if np.allclose(x, np.rint(x), atol=1e-8):
                    sublattice.append(s)
                    break
    pairs = {}
    for term in model.terms:
        if term.kind != 'bilinear':
            continue
        (_, o1), (_, o2) = term.participants
        d = (np.asarray(o2) - np.asarray(o1)) @ prim
        for i, r in enumerate(sites):
            j = index[key(r + d)]
            pairs[(i, j)] = pairs.get((i, j), 0) + np.asarray(term.coefficient, dtype=float)
    return np.array(sublattice), pairs


def ed_side(phi, S):
    """Ground-state energy per magnetic cell and ``<a_0,s>`` (exact operator) per sublattice."""
    model, state = y_model()
    sublattice, pairs = cluster(model, state)
    rotated = rotate_state(model, state, (0, 0, 1), phi)
    n = np.array([rotated.direction('Co', c) for c in state.cells])[sublattice]
    frames0 = expand_model(model, state, ExternalConditions(field=[0, 0, H1])).local_frames
    ops = spin_matrices(S)
    d, N = len(ops[0]), len(sublattice)
    shape = (d,) * N
    onsite = [-sum(S * (np.array([0, 0, H1]) + LAM * n[i])[a] * ops[a] for a in range(3))
              for i in range(N)]

    def apply(op, psi, i):
        return np.moveaxis(np.tensordot(op, psi, axes=([1], [i])), 0, i)

    def matvec(v):
        psi = v.reshape(shape)
        out = sum(apply(onsite[i], psi, i) for i in range(N))
        for (i, j), Jm in pairs.items():
            for b in range(3):
                pj = apply(ops[b], psi, j)
                for a in range(3):
                    if Jm[a, b] != 0:
                        out = out + Jm[a, b] * apply(ops[a], pj, i)
        return out.ravel()

    w, v = eigsh(LinearOperator((d ** N,) * 2, matvec=matvec, dtype=complex), k=1, which='SA',
                 tol=0, ncv=40)
    psi = v[:, 0].reshape(shape)
    amplitude = np.zeros(3, dtype=complex)
    for i in range(N):
        f = frames0[sublattice[i]]
        sz = sum(f[a, 2] * ops[a] for a in range(3))
        sp = sum((f[a, 0] + 1j * f[a, 1]) * ops[a] for a in range(3))
        ev, V = np.linalg.eigh(sz)
        a_op = sp @ (V @ np.diag(1 / np.sqrt(S + 1 + ev)) @ V.conj().T)
        amplitude[sublattice[i]] += np.vdot(psi.ravel(), apply(a_op, psi, i).ravel()) / CELLS[0]
    return float(w[0]) / CELLS[0], amplitude


def run(job):
    phi, S = job
    energy, amplitude = ed_side(phi, S)
    return {'phi': phi, 'S': S, 'energy_per_cell': energy,
            'amplitude': [[float(z.real), float(z.imag)] for z in amplitude]}


def analyse(code, ed):
    """Order-S^0 energy and order-S^{-1/2} amplitude from polynomial fits in 1/S."""
    S = np.array([r['S'] for r in ed])
    E = np.array([r['energy_per_cell'] for r in ed])
    residual = E - S ** 2 * code['classical'] - S * code['zero_point']
    energy = {deg: float(np.linalg.lstsq(np.vander(1 / S, deg + 1, increasing=True), residual,
                                         rcond=None)[0][0]) for deg in (2, 3, 4)}
    lead, corr = code['amplitude'][1]          # sublattice 1 (sublattice 0 is on the axis)
    a = np.array([complex(*r['amplitude'][1]) for r in ed])
    scaled = (a - np.sqrt(S) * lead) * np.sqrt(S)
    amp = [complex(np.linalg.lstsq(np.vander(1 / S[-k:], k - 1, increasing=True), scaled[-k:],
                                   rcond=None)[0][0]) for k in (3, 4, 5, 6)]
    lead_fit = complex(np.linalg.lstsq(np.vander(1 / S, 4, increasing=True), a / np.sqrt(S),
                                       rcond=None)[0][0])
    return energy, amp, lead_fit


def main():
    jobs = [(phi, S) for phi in PHIS for S in SPINS]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        ed_rows = list(pool.map(run, sorted(jobs, key=lambda j: -j[1])))
    rows, reference = [], None
    for phi in PHIS:
        code = code_side(phi)
        ed = sorted([r for r in ed_rows if r['phi'] == phi], key=lambda r: r['S'])
        energy, amp, lead_fit = analyse(code, ed)
        rows.append({'phi': phi, 'code': {k: code[k] for k in ('classical', 'zero_point', 'order_s0')},
                     'code_amplitude_s1': [[z.real, z.imag] for z in code['amplitude'][1]],
                     'ed': ed, 'ed_order_s0_by_degree': energy,
                     'ed_amplitude_s1_correction_by_points': [[z.real, z.imag] for z in amp],
                     'ed_amplitude_s1_leading': [lead_fit.real, lead_fit.imag]})
        reference = reference or rows[0]
        print(f"phi={phi:.4f} E2 code {code['order_s0']:.6f} ED {energy[4]:.6f} "
              f"dE2 code {code['order_s0'] - reference['code']['order_s0']:.6f} "
              f"ED {energy[4] - reference['ed_order_s0_by_degree'][4]:.6f} | "
              f"a0 correction code {code['amplitude'][1][1]:.4f} ED {amp[-2]:.4f}", flush=True)
    for r in rows[1:4]:                         # small angles: the regime the gap uses
        assert abs((r['code']['order_s0'] - rows[0]['code']['order_s0'])
                   - (r['ed_order_s0_by_degree'][4] - rows[0]['ed_order_s0_by_degree'][4])) < 3e-5
    record = {'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': __doc__.split('\n\n')[1].replace('\n', ' '),
              'parameters': {'J': J, 'Jz': JZ, 'JPD': JPD, 'h_spin1': H1, 'pinning_spin1': LAM,
                             'cells': CELLS, 'momenta': MOMENTA, 'spins': SPINS},
              'rows': rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'two-loop-ed-check.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
