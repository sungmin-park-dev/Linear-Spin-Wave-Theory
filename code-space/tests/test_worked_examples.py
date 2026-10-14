"""The worked-example scripts agree with the closed forms of their research notes.

Each script in ``code-space/examples/worked_examples`` solves one model with the package
and with the closed forms of its note in ``workbench/topics/worked-examples/lswt-example-*``.
The tests check that the two agree and that the numbers quoted in the note
are the ones the script computes, so that a change of the package or of the
note that breaks the example fails here.
"""

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "code-space" / "examples" / "worked_examples"
NOTES = ROOT / "workbench" / "topics" / "worked-examples"


@pytest.fixture(autouse=True)
def example_path(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))


def run(name):
    return runpy.run_path(str(EXAMPLES / f"{name}.py"))["main"]()


def note(name):
    return (NOTES / f"lswt-example-{name}" / "note.md").read_text(encoding="utf-8")


def assert_quoted(text, *numbers):
    for number in numbers:
        assert number in text, f"{number!r} is not quoted in the note"


def test_square_ferromagnet():
    v = run("square_ferromagnet")
    assert v["classical_energy"] == pytest.approx(-2.1, abs=1e-12)
    assert abs(v["zero_point_energy"]) < 1e-12
    assert v["band_error"] < 1e-12
    np.testing.assert_allclose(v["moment_reduction"], v["moment_reduction_sum"], rtol=1e-9)
    np.testing.assert_allclose(v["moment_reduction"], v["moment_reduction_low_t"], rtol=0.02)
    text = note("square-ferromagnet")
    assert_quoted(text, "$E_{\\mathrm{cl}}/N=-2.1$", "\\varepsilon_M=8.1",
                  *(f"{a:.3e}".replace("e-0", "\\times10^{-") + "}"
                    for a in v["moment_reduction"] + v["moment_reduction_low_t"]))


def test_square_antiferromagnet():
    v = run("square_antiferromagnet")
    text = note("square-antiferromagnet")
    assert_quoted(text, f"{v['c']:.6f}", f"{v['moment_reduction']:.6f}",
                  f"{0.5 - v['moment_reduction']:.6f}")
    for S, s in v["spins"].items():
        assert s["ground_state_energy"] == pytest.approx(s["ground_state_closed"], abs=2e-6)
        assert s["band_error"] < 1e-6
        assert s["velocity"] == pytest.approx(s["velocity_closed"], rel=1e-5)
        assert s["moment_extrapolated"] == pytest.approx(s["moment_closed"], abs=2e-4)
        assert_quoted(text, f"{s['ground_state_closed']:.6f}", f"{s['velocity']:.4f}")
    half = v["spins"][0.5]
    assert_quoted(text, f"{half['moment']:.4f}", f"{half['moment_extrapolated']:.4f}")


def test_triangular_antiferromagnet():
    v = run("triangular_antiferromagnet")
    assert v["ground_state_energy"] == pytest.approx(v["ground_state_closed"], abs=2e-6)
    assert v["band_error"] < 1e-6
    assert v["velocity_gamma"] == pytest.approx(v["velocity_gamma_closed"], rel=1e-5)
    assert v["velocity_K"] == pytest.approx(v["velocity_K_closed"], rel=1e-5)
    assert v["moment_extrapolated"] == pytest.approx(v["moment_closed"], abs=2e-4)
    text = note("triangular-antiferromagnet")
    assert_quoted(text, f"{v['mean_root']:.6f}", f"{v['moment_reduction']:.6f}",
                  f"{v['ground_state_closed']:.6f}", f"{v['moment_closed']:.6f}",
                  f"{1 - v['mean_root']:.6f}", f"{v['moment']:.4f}",
                  f"{v['moment_extrapolated']:.4f}", f"{v['velocity_gamma']:.4f}",
                  f"{v['velocity_K']:.4f}")


def test_honeycomb_topological_ferromagnet():
    v = run("honeycomb_topological_ferromagnet")
    assert v["band_error"] < 1e-12
    assert v["gap_K"] == pytest.approx(v["gap_K_closed"], abs=1e-10)
    assert v["chern"][0.1] == [1.0, -1.0]
    assert v["chern"][-0.1] == [-1.0, 1.0]
    assert v["dirac_count"] == {0.1: 1.0, -0.1: -1.0}                # eq. (6): C_- = sgn D
    text = note("honeycomb-topological-ferromagnet")
    assert_quoted(text, *(f"{e:.4f}" for e in v["bands_K"]), f"{v['gap_K']:.4f}",
                  f"{v['top_gamma']:.4f}", "$(C_-,C_+)=(+1,-1)$")
