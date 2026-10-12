"""Helpers shared by the worked-example scripts (closed-form momentum averages)."""

from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / 'code-space') not in sys.path:
    sys.path.insert(0, str(ROOT / 'code-space'))

import numpy as np


def zone_average(f, n):
    """Average of ``f(x1, x2)`` over the zone, ``x_i = k . a_i``, on an ``n x n`` mesh
    shifted by 1/3 of a step so that no point falls on Gamma or on K."""
    x = (np.arange(n) + 1 / 3) / n * 2 * np.pi
    X1, X2 = np.meshgrid(x, x, indexing='ij')
    return float(np.mean(f(X1, X2)))


def converged_average(f, n=1000):
    """Zone average extrapolated from meshes n and 2n, assuming an O(1/n) error.

    The error is O(1/n) for an integrand with a 1/|k| singularity (Goldstone
    modes in a boson number) and much smaller for a smooth one, for which the
    extrapolation changes nothing visible.
    """
    return 2 * zone_average(f, 2 * n) - zone_average(f, n)


def quiet():
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
