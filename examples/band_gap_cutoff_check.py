"""Measure optional diagnostics and a fixed numerical band-gap cutoff.

Run with PYTHONPATH=code-space. Timings use warm, single-k complex positive
matrices, not NBCP ground states or an end-to-end solver benchmark. Residual
and orthogonality diagnostics are not certified error bounds: Cholesky,
matrix-construction and floating-point evaluation errors are not propagated.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from lswt.definitions import DEFAULT_BAND_GAP_CUTOFF
from lswt.methods.spin_wave.diagonalization import Diagonalizer
from lswt.observables.topology import _band_separation
from band_degeneracy_check import probe


def timed_colpa(k, metric, mode):
    """Match production Colpa, reusing intermediates for optional diagnostics."""
    adjoint = k.conj().T
    matrix = adjoint @ metric @ k
    values, vectors = np.linalg.eigh(matrix)
    values, vectors = values[::-1], vectors[:, ::-1]
    energies = values * np.diag(metric)
    transform = np.linalg.inv(adjoint) @ vectors @ np.diag(np.sqrt(energies))
    diagnostic = None
    if mode == 'fixed_cutoff':
        diagnostic = _band_separation(values, len(values)//2, DEFAULT_BAND_GAP_CUTOFF)
    elif mode in ('frobenius_diagnostics', 'spectral_diagnostics'):
        residual = matrix @ vectors - vectors * values
        orthogonality = vectors.conj().T @ vectors - np.eye(len(values))
        order = 'fro' if mode == 'frobenius_diagnostics' else 2
        diagnostic = (np.linalg.norm(residual, ord=order),
                      np.linalg.norm(orthogonality, ord=order))
    return energies, transform, diagnostic


def main():
    """Save timing distributions, cutoff sensitivity and the source identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(20260912)
    records = []
    modes = ['baseline', 'fixed_cutoff', 'frobenius_diagnostics', 'spectral_diagnostics']
    for size in [2, 4, 6, 8, 32, 64]:
        random = rng.normal(size=(size, size)) + 1j*rng.normal(size=(size, size))
        matrix = random @ random.conj().T / size + 0.3*np.eye(size)
        k = np.linalg.cholesky(matrix)
        metric = np.diag([1.]*(size//2) + [-1.]*(size//2))
        expected = Diagonalizer.Colpa(k, metric)
        actual = timed_colpa(k, metric, 'baseline')
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
        samples = {mode: [] for mode in modes}
        iterations = 250 if size <= 8 else 15
        for mode in modes:
            for _ in range(5):
                timed_colpa(k, metric, mode)
        for _ in range(5):
            for mode in rng.permutation(modes):
                start = time.perf_counter_ns()
                for _ in range(iterations):
                    timed_colpa(k, metric, mode)
                samples[mode].append((time.perf_counter_ns()-start)/iterations/1000)
        records.append({
            'matrix_size': size, 'physical_bands': size//2,
            'iterations_per_repeat': iterations,
            'median_us_per_k': {mode: float(np.median(v)) for mode, v in samples.items()},
            'samples_us_per_k': samples,
        })
        print(size, records[-1]['median_us_per_k'], flush=True)

    root = Path(__file__).resolve().parents[1]
    sources = ['examples/band_gap_cutoff_check.py', 'examples/band_degeneracy_check.py',
               'code-space/lswt/observables/topology.py', 'code-space/lswt/definitions/constants.py', 'code-space/lswt/definitions/defaults.py', 'code-space/lswt/definitions/spin_basis.py',
               'code-space/lswt/methods/spin_wave/diagonalization.py']
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': __doc__, 'platform': platform.platform(),
        'numpy_version': np.__version__, 'seed': 20260912,
        'timings': records,
        'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                          for name in sources},
        'sensitivity': [probe(gap/2, band_gap_cutoff=cutoff)
                        for gap in (1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 0)
                        for cutoff in (0, 1e-10, 1e-8, 1e-6)],
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
