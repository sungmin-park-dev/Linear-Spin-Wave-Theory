"""The tutorial scripts run and reproduce the numbers quoted in docs/tutorials."""

import runpy
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = ROOT / "code-space" / "examples" / "tutorials"


def run(name, capsys):
    namespace = runpy.run_path(str(TUTORIALS / f"{name}.py"))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*removed in spintoolkit",
                                category=DeprecationWarning)
        value = namespace["main"]()
    return value, capsys.readouterr().out


def test_first_calculation(capsys):
    result, out = run("t01_first_calculation", capsys)
    assert round(result.ground_state_energy, 4) == -0.5388      # Chubukov et al. (1994)
    assert "magnon energies at M: [1.     1.5811 1.5811]" in out
    assert "mesh  96: E_gs = -0.538809  <S> = 0.2411" in out


def test_own_model(capsys):
    _, out = run("t02_own_model", capsys)
    assert "classical energy -0.5000 J1 per spin" in out
    assert "E_gs = -0.6579 J1 per spin" in out
    assert "J2 = 0.2: Neel   stable" in out
    assert "J2 = 0.8: stripe stable" in out


def test_neutron(capsys):
    spectrum, out = run("t03_neutron", capsys)
    for line in out.splitlines()[:4]:
        omega, closed = line.split("omega = ")[1].split(" (closed form ")[:2]
        assert float(omega) == pytest.approx(float(closed[:6]), abs=1e-4)
        weight, closed = line.split("I = ")[1].split(" (closed form ")
        assert float(weight) == pytest.approx(float(closed.rstrip(")")), abs=1e-4)
    assert "elastic 0.0987  <S>^2 = 0.0987" in out


def test_magnetization(capsys):
    (curve, (fields, m_ed)), out = run("t04_magnetization", capsys)
    assert np.allclose(curve.classical[fields_index(curve, 2.0)], 0.25)
    assert "1/S 0.0600" in out
    assert m_ed[-1] == 0.5
    for line in out.splitlines():
        if line.startswith("ED plateau"):
            value = float(line.split("M = ")[1].split()[0])
            classical = float(line.split("classical ")[1].split()[0])
            harmonic = float(line.split("1/S ")[1])
            assert abs(harmonic - value) < abs(classical - value)


def fields_index(curve, h):
    return int(np.argmin(abs(curve.fields - h)))


def test_topology(capsys):
    _, out = run("t05_topology", capsys)
    assert "bands at K: [1.3402 1.8598]  closed form: [1.3402 1.8598]" in out
    assert "Chern numbers (lower, upper): [ 1. -1.]" in out
    assert "D = 0: [nan nan]" in out
    assert "T = 0.5: kappa_xy / T = -0.1290" in out
    assert "D -> -D: [0.0009 0.0754 0.129  0.1131]" in out


def test_interacting(capsys):
    (_, _, path), out = run("t06_interacting", capsys)
    assert "E_(S^0) = -0.008012  total = -0.546820" in out
    assert "1/S       [   nan    nan 0.7313]" in out
    assert "1/S (PV)  [1.1778 1.1778 0.7355]" in out
    assert "decay     [0.1597 0.1597 0.    ]" in out
    assert "undefined path points (Goldstone modes): 3" in out
    assert "path points where the expansion is not controlled: [16]" in out
    assert "total 2.061  one magnon 1.530  continuum 0.531  (LSWT 2.052)" in out


def _undefined_names(code):
    """Names a code block reads but neither it, an import nor builtins define (a static NameError check)."""
    import ast
    import builtins

    tree = ast.parse(code)
    defined = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(used - defined)


@pytest.mark.parametrize("page", sorted((ROOT / "docs" / "tutorials").glob("0*.md")),
                         ids=lambda p: p.stem)
def test_tutorial_code_blocks_define_every_name(page):
    import re

    code = "\n".join(re.findall(r"```python\n(.*?)```", page.read_text(), re.S))
    assert code and _undefined_names(code) == []
