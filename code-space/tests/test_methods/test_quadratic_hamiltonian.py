"""Native LSWT H(k) from SpinModel (D36) against the former SpinSystem builder.

For the bilinear and Zeeman kinds the two must agree to round-off: H(k), its
analytic derivatives, the linear boson terms and the local frames.
"""

import warnings

import numpy as np
import pytest

from model import nbcp
from model.nbcp.model import legacy_cells
from spintoolkit.methods.lswt.hamiltonian import LSWTHamiltonian
from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
from spintoolkit.models import kitaev_honeycomb, neel_state, polarized_state, state_120, triangular_heisenberg
from spintoolkit.models import honeycomb_ferromagnet, square_heisenberg
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.system.conversion import to_spin_system

FIELD = ExternalConditions(field=(0.03, -0.04, 0.2))


def cases():
    yield "triangular 120", triangular_heisenberg(), state_120, FIELD
    yield "square neel", square_heisenberg(), lambda m: neel_state(m, (1, 1, 1)), ExternalConditions()
    yield "kitaev 111", kitaev_honeycomb(), lambda m: polarized_state(m, (1, 1, 1)), FIELD
    yield "honeycomb dm", honeycomb_ferromagnet(), lambda m: polarized_state(m, (0.2, 0.1, 1)), FIELD
    rng = np.random.default_rng(7)
    for cell in ("two_msl", "three_msl", "four_msl"):
        model = nbcp.build_model({"Jxy": 0.076, "Jz": 0.125, "JPD": 0.013, "JGamma": -0.021})
        angles = rng.uniform(0, 3, 2 * len(legacy_cells(cell)))
        yield f"nbcp {cell}", model, (lambda m, c=cell, a=angles: nbcp.candidate_state(m, c, a)), FIELD


@pytest.mark.parametrize("name, model, make_state, conditions", list(cases()), ids=lambda x: x if isinstance(x, str) else "")
def test_native_matches_spin_system_builder(name, model, make_state, conditions):
    state = make_state(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        system = to_spin_system(model, state, conditions)
    data = system.to_legacy_dict("simple")
    legacy = LSWTHamiltonian(data["Spin info"], data["Couplings"])
    angles = system.get_angles_flat()
    k = np.random.default_rng(1).uniform(-5, 5, size=(9, 2))
    H, linear = legacy.Quadratic_Bose_Hamiltonian(k, angles=angles)
    dx, dy = legacy.partial_derivatives_of_Hk(k)
    native = QuadraticBoseHamiltonian(model, state, conditions)
    scale = max(1.0, np.abs(H).max())
    np.testing.assert_allclose(native.at(k), H, rtol=0, atol=1e-14 * scale)
    ndx, ndy = native.derivatives_at(k)
    np.testing.assert_allclose(ndx, dx, rtol=0, atol=1e-13 * scale)
    np.testing.assert_allclose(ndy, dy, rtol=0, atol=1e-13 * scale)
    np.testing.assert_allclose(native.linear_terms, list(linear.values()), rtol=0, atol=1e-14 * scale)
    np.testing.assert_allclose(native.local_frames,
                               list(legacy.get_rmat_dict(angles=angles).values()), atol=1e-15)


def test_derivatives_match_finite_differences():
    model = nbcp.build_model({"Jxy": 0.076, "Jz": 0.125, "JPD": 0.013, "JGamma": -0.021})
    state = nbcp.candidate_state(model, "three_msl", np.linspace(0.2, 2.9, 6))
    native = QuadraticBoseHamiltonian(model, state, FIELD)
    k = np.array([[0.37, -1.21]])
    step = 1e-6
    for axis, analytic in enumerate(native.derivatives_at(k)):
        e = np.zeros(2)
        e[axis] = step
        numeric = (native.at(k + e) - native.at(k - e)) / (2 * step)
        np.testing.assert_allclose(analytic, numeric, atol=1e-8)
