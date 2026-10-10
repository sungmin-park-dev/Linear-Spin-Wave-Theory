"""Independent check of the finite-temperature magnetization M(h, t) = -dF/dh (D49).

Canted square antiferromagnet, J = g = 1, field along z, spin S. In the frame
rotated with each sublattice (one boson per site, momentum over the full
square zone), the harmonic Hamiltonian at a canting angle theta out of the
plane perpendicular to the field is

    A_k(theta, h) = -4S (2 s2 - 1) + h sin(theta) + 4S s2 gamma_k,
    B_k(theta)    = -4S (1 - s2) gamma_k,         s2 = sin^2(theta),

with gamma_k = (cos kx + cos ky) / 2, omega_k = sqrt(A^2 - B^2) and
E_cl = 2S^2 (2 s2 - 1) - h S sin(theta) per site. The classical angle is
sin(theta) = h / 8S, where A_k = 4S (1 + s2 gamma_k). Nothing from
spintoolkit enters these formulas.

1. The decomposition. A depends on h only through h sin(theta), so
   dF_qm/dh at fixed theta is sin(theta) <n> (Hellmann-Feynman) and

       M = (S - <n>) sin(theta) - dF_qm/dtheta / (8S cos(theta)).

   The first term is the moment sum of ThermalResult.magnetization, the
   second the canting-angle shift. At t > 0 both diverge with the mesh near
   the Goldstone mode at (pi, pi); their sum does not.
2. The low-temperature law. Near (pi, pi), omega = c q with
   c^2 = 8 S^2 - h^2 / 8, so F_th = -zeta(3) t^3 / (2 pi c^2) and
   M(t) - M(0) = zeta(3) h t^3 / (8 pi c^4).
3. spintoolkit's magnetization_curve(..., temperatures=) against the closed
   form.

Usage
-----
    python examples/lswt_d49_thermal_magnetization_check.py > report.json
"""

import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'code-space'), str(ROOT)]

import numpy as np
from scipy.special import zeta

ZETA3 = float(zeta(3))


def gamma_mesh(n):
    k = (np.arange(n) + 0.5) / n * 2 * np.pi
    kx, ky = np.meshgrid(k, k)
    return 0.5 * (np.cos(kx) + np.cos(ky))


def coefficients(theta, h, S, gamma):
    s2 = np.sin(theta) ** 2
    A = -4 * S * (2 * s2 - 1) + h * np.sin(theta) + 4 * S * s2 * gamma
    B = -4 * S * (1 - s2) * gamma
    return A, B, np.sqrt(A * A - B * B)


def free_qm(theta, h, S, t, gamma):
    """Zero-point plus thermal magnon free energy per site."""
    A, _, w = coefficients(theta, h, S, gamma)
    thermal = t * np.mean(np.log(-np.expm1(-w / t))) if t > 0 else 0.0
    return 0.5 * np.mean(w - A) + thermal


def boson_number(theta, h, S, t, gamma):
    A, _, w = coefficients(theta, h, S, gamma)
    occupation = 1 / np.expm1(w / t) if t > 0 else 0.0
    return float(np.mean(A / w * (occupation + 0.5) - 0.5))


def theta_cl(h, S):
    return np.arcsin(h / (8 * S))


def magnetization_total(h, S, t, gamma, step=1e-4):
    """-d/dh [E_cl + F_qm] at the classical angle of each field."""
    def free(field):
        th = theta_cl(field, S)
        e_cl = 2 * S * S * (2 * np.sin(th) ** 2 - 1) - field * S * np.sin(th)
        return e_cl + free_qm(th, field, S, t, gamma)
    return -(free(h + step) - free(h - step)) / (2 * step)


def free_qm_theta_derivative(theta, h, S, t, gamma):
    """dF_qm/dtheta at fixed h, analytic: < (1/2 + n_B) d omega - dA / 2 >.

    A finite difference in theta is not used: for theta > theta_cl the mode
    at (pi, pi) moves towards instability, omega(theta) is not smooth on the
    scale of the smallest mesh momentum, and the difference error grows with n.
    """
    A, B, w = coefficients(theta, h, S, gamma)
    s, c = np.sin(theta), np.cos(theta)
    dA = c * (h - 16 * S * s + 8 * S * s * gamma)
    dB = 8 * S * s * c * gamma
    dw = (A * dA - B * dB) / w
    occupation = 1 / np.expm1(w / t) if t > 0 else 0.0
    return float(np.mean((0.5 + occupation) * dw - 0.5 * dA))


def decomposition(h, S, t, gamma):
    th = theta_cl(h, S)
    moment = (S - boson_number(th, h, S, t, gamma)) * np.sin(th)
    angle = -free_qm_theta_derivative(th, h, S, t, gamma) / (8 * S * np.cos(th))
    return moment, angle


def check_decomposition(S=0.5):
    rows = []
    for h in (1.0, 2.0):
        for t in (0.0, 0.3):
            for n in (100, 200, 400):
                gamma = gamma_mesh(n)
                total = magnetization_total(h, S, t, gamma)
                moment, angle = decomposition(h, S, t, gamma)
                rows.append({'h': h, 't': t, 'n': n, 'M': total, 'moment_sum': moment,
                             'angle_shift': angle, 'identity_error': abs(moment + angle - total)})
    return rows


def check_low_temperature(S=0.5, n=2000):
    gamma = gamma_mesh(n)
    rows = []
    for h in (1.0, 2.0, 3.0):
        c4 = (8 * S * S - h * h / 8) ** 2
        m0 = magnetization_total(h, S, 0.0, gamma)
        for t in (0.02, 0.04, 0.08, 0.16):
            dm = magnetization_total(h, S, t, gamma) - m0
            predicted = ZETA3 * h * t ** 3 / (8 * np.pi * c4)
            rows.append({'h': h, 't': t, 'dM': dm, 'dM_t3_law': predicted, 'ratio': dm / predicted})
    return rows


def check_package(S=0.5, n=600):
    from spintoolkit.methods.magnetization import magnetization_curve
    from spintoolkit.models import neel_state, square_heisenberg
    model = square_heisenberg(J=1.0, S=S)
    fields, temperatures = [1.0, 2.0, 3.0], [0.0, 0.1, 0.3, 0.6]
    gamma = gamma_mesh(n)
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        curve = magnetization_curve(model, neel_state(model, (1, 0, 0)), fields, k_density=96,
                                    temperatures=temperatures)
    for i, h in enumerate(fields):
        for j, t in enumerate(temperatures):
            closed = magnetization_total(h, S, t, gamma)
            rows.append({'h': h, 't': t, 'package': float(curve.thermal[i, j]),
                         'closed_form': closed, 'difference': abs(curve.thermal[i, j] - closed)})
    return rows


def main():
    report = {'decomposition': check_decomposition(),
              'low_temperature_t3_law': check_low_temperature(),
              'package_vs_closed_form': check_package()}
    print(json.dumps(report, indent=2, default=float))


if __name__ == '__main__':
    main()
