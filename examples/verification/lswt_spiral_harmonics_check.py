"""Check the rotation condition for single-Q spirals in the rotating frame.

Research note *LSWT classical order*, Single-Q Spirals in a Rotating Frame,
writes the component of the rotated exchange between the circular axis vectors
``c^p`` and ``c^q`` as a phase times the laboratory component, and states that
the rotating-frame Hamiltonian is translation invariant exactly when every
exchange and single-ion matrix commutes with the rotations about the spiral
axis and the Zeeman vector is parallel to it. This script checks:

1. The phase formula
   ``(c^p)^T R(phi_i)^T J R(phi_j) c^q
   = exp(-i (s_p + s_q) phi_i) exp(-i s_q (phi_j - phi_i)) (c^p)^T J c^q``
   for a random nonsymmetric ``J`` and a random axis.
2. That a matrix commutes with ``R_c(1 rad)`` exactly when its components with
   ``s_p + s_q != 0`` vanish, and that it then has the form
   ``a (1 - c c^T) + b c c^T + d [c]_x``.
3. The cell dependence of the rotating-frame local field on an incommensurate
   Heisenberg spiral of the triangular lattice, for the terms of the note's
   subsection When the Rotation Condition Fails. With a field along the axis
   and an axial single-ion term (a cone), the field is the same in every cell.
   A field in the spiral plane gives a first harmonic, an easy axis in the
   plane a second harmonic; a uniform reference cannot cancel either.
4. ``symmetry_violations`` of ``spintoolkit.methods.lswt.spiral`` reports the
   same terms and nothing for the axial ones.

Run with ``python examples/verification/lswt_spiral_harmonics_check.py``.
"""

import dataclasses

import numpy as np

from spintoolkit.methods.lswt.spiral import symmetry_violations
from spintoolkit.models import triangular_heisenberg
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.model import Term

RNG = np.random.default_rng(20261013)


def rotation(c, phi):
    """Right-handed rotation by phi about the unit vector c (Rodrigues)."""
    c = np.asarray(c, dtype=float)
    K = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
    return np.eye(3) + np.sin(phi) * K + (1 - np.cos(phi)) * K @ K


def axis_basis(c):
    """c^+, c^0, c^- with R_c(phi) c^{+-} = exp(-+ i phi) c^{+-}."""
    c = np.asarray(c, dtype=float)
    x = np.cross(c, [1.0, 0.0, 0.0] if abs(c[0]) < 0.9 else [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(c, x)
    return {+1: (x + 1j * y) / np.sqrt(2), 0: c.astype(complex), -1: (x - 1j * y) / np.sqrt(2)}


def check_phase_formula():
    c = RNG.normal(size=3)
    c /= np.linalg.norm(c)
    J = RNG.normal(size=(3, 3))
    basis = axis_basis(c)
    for v, s in ((basis[+1], 1), (basis[-1], -1)):
        assert np.allclose(rotation(c, 0.7) @ v, np.exp(-1j * s * 0.7) * v)
    worst = 0.0
    for _ in range(20):
        phi_i, phi_j = RNG.uniform(-np.pi, np.pi, size=2)
        rotated = rotation(c, phi_i).T @ J @ rotation(c, phi_j)
        for sp, cp in basis.items():
            for sq, cq in basis.items():
                lhs = cp @ rotated @ cq
                rhs = (np.exp(-1j * (sp + sq) * phi_i) * np.exp(-1j * sq * (phi_j - phi_i))
                       * (cp @ J @ cq))
                worst = max(worst, abs(lhs - rhs))
    return worst


def check_commutant():
    c = RNG.normal(size=3)
    c /= np.linalg.norm(c)
    basis = axis_basis(c)
    K = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
    a, b, d = RNG.normal(size=3)
    axial = a * (np.eye(3) - np.outer(c, c)) + b * np.outer(c, c) + d * K
    generic = RNG.normal(size=(3, 3))
    R = rotation(c, 1.0)

    def off_harmonic(M):
        return max(abs(basis[p] @ M @ basis[q]) for p in basis for q in basis if p + q != 0)

    return {
        "axial_commutator": float(np.max(np.abs(R @ axial - axial @ R))),
        "axial_harmonic_components": float(off_harmonic(axial)),
        "generic_commutator": float(np.max(np.abs(R @ generic - generic @ R))),
        "generic_harmonic_components": float(off_harmonic(generic)),
    }


def rotating_frame_fields(J, A, h, c, Q, d, S=1.0, cells=60):
    """R(phi_i)^T times the local field -dE/dS_i of a single-Q spiral.

    Triangular Bravais lattice, nearest-neighbour exchange matrix J on the six
    neighbour vectors, single-ion S^T A S with the coherent-state factor
    (1 - 1/2S) (D37), Zeeman energy -h . S, one spin per cell along
    R_c(Q . R_i) d. Cells R_i = (n, 0) along a1.
    """
    a1, a2 = np.array([1.0, 0.0]), np.array([0.5, np.sqrt(3) / 2])
    neighbours = [a1, a2, a2 - a1, -a1, -a2, a1 - a2]
    factor = 1 - 1 / (2 * S)
    fields = []
    for n in range(cells):
        R_i = n * a1
        phi_i = Q @ R_i
        spin_i = S * rotation(c, phi_i) @ d
        field = h.copy()
        for delta in neighbours:
            field -= J @ (S * rotation(c, Q @ (R_i + delta)) @ d)
        field -= 2 * factor * A @ spin_i
        fields.append(rotation(c, phi_i).T @ field)
    return np.array(fields)


def check_cell_dependence():
    c = np.array([0.0, 0.0, 1.0])
    Q = np.array([2 * np.pi * 0.3183, 0.0])  # incommensurate pitch along a1
    J = np.eye(3)
    zero = np.zeros((3, 3))
    theta = 1.1  # cone angle from the axis
    cone = np.array([np.sin(theta), 0.0, np.cos(theta)])
    plane = np.array([1.0, 0.0, 0.0])
    cases = {
        "cone, field along axis, axial easy-plane A": (
            J, np.diag([0.0, 0.0, 0.3]), np.array([0.0, 0.0, 0.5]), cone),
        "planar, field in spiral plane": (J, zero, np.array([0.5, 0.0, 0.0]), plane),
        "planar, easy axis in spiral plane": (J, -0.3 * np.outer(plane, plane), np.zeros(3), plane),
    }
    report = {}
    for name, (Jm, A, h, d) in cases.items():
        fields = rotating_frame_fields(Jm, A, h, c, Q, d)
        spread = np.max(np.linalg.norm(fields - fields.mean(axis=0), axis=1))
        # Torque d x h' on the uniform rotating-frame reference: its spread over
        # cells cannot be removed by any choice of the cell-independent d.
        torques = np.cross(d, fields)
        report[name] = {
            "field_spread": float(spread),
            "torque_spread": float(np.ptp(np.linalg.norm(torques, axis=1))),
        }
    return report


def check_package():
    base = triangular_heisenberg(J=1.0, S=1.0, g=1.0)
    site = base.site_ids[0]
    axis = (0, 0, 1)
    out = {}
    out["axial model, field along axis"] = symmetry_violations(
        dataclasses.replace(base, terms=base.terms + (Term.onsite(site, np.diag([0.0, 0.0, 0.3])),)),
        axis, ExternalConditions(field=(0, 0, 0.5)))
    out["field in spiral plane"] = symmetry_violations(base, axis, ExternalConditions(field=(0.5, 0, 0)))
    out["easy axis in spiral plane"] = symmetry_violations(
        dataclasses.replace(base, terms=base.terms + (Term.onsite(site, np.diag([-0.3, 0.0, 0.0])),)),
        axis, ExternalConditions())
    return out


def main():
    phase = check_phase_formula()
    print(f"1. phase formula, largest deviation: {phase:.2e}")
    assert phase < 1e-12
    commutant = check_commutant()
    print("2. commutant:", {k: f"{v:.2e}" for k, v in commutant.items()})
    assert commutant["axial_commutator"] < 1e-12 and commutant["axial_harmonic_components"] < 1e-12
    assert commutant["generic_commutator"] > 1e-2 and commutant["generic_harmonic_components"] > 1e-2
    cells = check_cell_dependence()
    print("3. rotating-frame local field over 60 cells:")
    for name, values in cells.items():
        print(f"   {name}: field spread {values['field_spread']:.3e}, "
              f"torque spread {values['torque_spread']:.3e}")
    assert cells["cone, field along axis, axial easy-plane A"]["field_spread"] < 1e-12
    assert cells["planar, field in spiral plane"]["torque_spread"] > 0.1
    assert cells["planar, easy axis in spiral plane"]["torque_spread"] > 0.1
    package = check_package()
    print("4. symmetry_violations:")
    for name, violations in package.items():
        print(f"   {name}: {violations if violations else 'none'}")
    assert not package["axial model, field along axis"]
    assert package["field in spiral plane"] and package["easy axis in spiral plane"]
    print("all checks passed")


if __name__ == "__main__":
    main()
