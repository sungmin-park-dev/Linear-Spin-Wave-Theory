"""The new API runs without the pre-0.2 code that 0.3 removes (D43, audit ARCH-02..05, ARCH-14).

- Every consumer that loops over term kinds refuses a kind it does not implement
  instead of dropping it (a silently wrong H(k) otherwise).
- Onsite terms reach the spiral energy, the spin plot and the constrained
  zero-point energy provider.
- Bonds in the spin plot end on lattice sites.
- solve_lswt, thermal_quantities, the state-selection providers and the spin plot
  work with the SpinSystem-based modules blocked from import.
"""

import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

from spintoolkit.methods.classical import classical_energy
from spintoolkit.methods.lswt import solve_lswt, spiral_energy
from spintoolkit.models import honeycomb_ferromagnet, polarized_state, square_heisenberg
from spintoolkit.states.incommensurate import IncommensurateStructure
from spintoolkit.system import model as model_module
from spintoolkit.system.model import Site, SpinModel, Term

S = 0.5


def chain(A=None):
    J = np.eye(3)
    terms = [Term.bilinear(("A", (0, 0)), ("A", (1, 0)), J),
             Term.bilinear(("A", (0, 0)), ("A", (2, 0)), 0.5 * J),
             Term.bilinear(("A", (0, 0)), ("A", (0, 1)), -0.5 * J),
             Term.zeeman("A", np.eye(3))]
    if A is not None:
        terms.append(Term.onsite("A", A))
    return SpinModel(np.eye(2), [Site("A", (0, 0), S)], terms, {"model_id": "chain_onsite"})


@pytest.mark.parametrize("A", [np.diag([0.0, 0.0, 0.3]), np.diag([0.2, -0.1, 0.3])])
def test_spiral_energy_includes_onsite_terms(A):
    model = chain(A)
    cone = IncommensurateStructure(model.fingerprint(), [1 / 3, 0.0], [0, 0, 1],
                                   {"A": [np.sqrt(1 - 0.16), 0, 0.4]})
    assert spiral_energy(model, cone) == pytest.approx(
        classical_energy(model, cone.to_spin_state(model)), abs=1e-13)


@pytest.fixture
def biquadratic(monkeypatch):
    """A model with a term kind that no consumer implements."""
    monkeypatch.setitem(model_module.SUPPORTED_KINDS, "biquadratic", 2)
    base = square_heisenberg(J=1.0)
    term = Term("biquadratic", (("A", (0, 0)), ("A", (1, 0))), np.eye(3), None)
    return SpinModel(base.lattice, list(base.sites), list(base.terms) + [term],
                     {"model_id": "with_biquadratic"})


def test_unknown_term_kinds_are_refused(biquadratic):
    from spintoolkit.methods.luttinger_tisza import lt_matrix
    from spintoolkit.methods.lswt.quadratic import QuadraticBoseHamiltonian
    from spintoolkit.methods.nlswt.expansion import expand_model
    from spintoolkit.methods.state_selection import rotation_symmetry
    from spintoolkit.system.conditions import ExternalConditions

    state = polarized_state(biquadratic, (0, 0, 1))
    calls = [lambda: QuadraticBoseHamiltonian(biquadratic, state),
             lambda: solve_lswt(biquadratic, state),
             lambda: expand_model(biquadratic, state),
             lambda: lt_matrix(biquadratic, [[0.0, 0.0]]),
             lambda: rotation_symmetry(biquadratic, ExternalConditions(), [0, 0, 1], 1e-10)]
    for call in calls:
        with pytest.raises(NotImplementedError, match="biquadratic"):
            call()


def test_spin_plot_draws_onsite_models_and_bonds_end_on_sites():
    import matplotlib
    matplotlib.use("Agg")
    from spintoolkit.visualization.spin_plotter import _magnetic_cell, plot_spin_configuration

    model = chain(np.diag([0.0, 0.0, 0.3]))
    state = polarized_state(model, (1, 0, 0))
    plot_spin_configuration(model, state)
    honeycomb = honeycomb_ferromagnet()
    cell = _magnetic_cell(honeycomb, polarized_state(honeycomb, (0, 0, 1)))
    inverse = np.linalg.inv(cell.lattice_vectors.T)
    for bond in cell.couplings:
        end = cell.sites[bond.site_i].position + bond.displacement
        offsets = [inverse @ (end - site.position) for site in cell.sites]
        assert any(np.allclose(f, np.round(f), atol=1e-9) for f in offsets)


def test_constrained_zero_point_energy_accepts_onsite_terms():
    from spintoolkit.methods.state_selection import lswt_zero_point_energy

    base = square_heisenberg(J=-1.0)
    model = SpinModel(base.lattice, list(base.sites),
                      list(base.terms) + [Term.onsite("A", np.diag([0.0, 0.0, 0.3]))],
                      {"model_id": "square_ferro_easy_plane"})
    provider = lswt_zero_point_energy("simple", 6)
    value = provider(model, polarized_state(model, (1, 0, 0)), axis=[0, 0, 1])
    assert np.isfinite(value)


BLOCKED = textwrap.dedent('''
    import sys, types, warnings
    # Only the package's own deprecations (other libraries warn about theirs).
    warnings.filterwarnings("error", message=".*removed in spintoolkit", category=DeprecationWarning)
    BLOCK = ("spintoolkit.system.spin_system", "spintoolkit.methods.lswt.hamiltonian",
             "spintoolkit.methods.lswt.energy", "spintoolkit.methods.lswt.solver",
             "spintoolkit.observables.thermodynamics", "spintoolkit.methods.optimization")

    class Removed(type):
        """Stand-in for a name of a removed module: importing works, using it fails."""
        def __call__(cls, *args, **kwargs):
            raise RuntimeError(f"used removed {cls.__module__}.{cls.__name__}")
        def __getattr__(cls, attribute):
            raise RuntimeError(f"used removed {cls.__module__}.{cls.__name__}.{attribute}")

    def removed_module(name):
        module = types.ModuleType(name)
        module.__getattr__ = lambda attribute: Removed(attribute, (), {"__module__": name})
        return module

    for name in BLOCK:
        sys.modules[name] = removed_module(name)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    from spintoolkit.methods.lswt.run import LSWTSettings, solve_lswt
    from spintoolkit.methods.state_selection import lswt_zero_point_energy
    from spintoolkit.models import neel_state, square_heisenberg
    from spintoolkit.observables.thermal import thermal_quantities
    from spintoolkit.visualization.spin_plotter import plot_spin_configuration
    model = square_heisenberg(J=1.0)
    result = solve_lswt(model, neel_state(model), settings=LSWTSettings(mesh=(8, 8)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        thermal_quantities(result, [0.0, 0.2], gapless=True)
    lswt_zero_point_energy("simple", 4)(model, neel_state(model), axis=[0, 0, 1])
    plot_spin_configuration(model, neel_state(model))
    print("ok")
''')


def test_new_api_runs_without_the_pre_02_modules():
    """sys.modules stand-in for the 0.3 removal: the blocked modules may not be imported."""
    out = subprocess.run([sys.executable, "-c", BLOCKED], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.strip().endswith("ok")
