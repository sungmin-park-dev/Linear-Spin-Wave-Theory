"""Release metadata: one PEP 440 version, the license file, optional tqdm."""

import re
import subprocess
import sys
from pathlib import Path

import spintoolkit

ROOT = Path(__file__).resolve().parents[2]


def test_version_is_pep440_and_single_sourced():
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?",
                        spintoolkit.__version__)
    text = (ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in text
    assert 'version = {attr = "spintoolkit.__version__"}' in text


def test_license_file_exists():
    assert "MIT License" in (ROOT / "LICENSE").read_text()


def test_thermodynamics_imports_without_tqdm():
    code = ("import sys; sys.modules['tqdm'] = None; "
            "import spintoolkit.observables.thermodynamics as t; "
            "assert list(t.tqdm([1, 2], desc='x')) == [1, 2]")
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=ROOT / "code-space")


def test_quickstart_runs_without_deprecation_warnings():
    """The README quick start uses only the new API (D43)."""
    subprocess.run([sys.executable, "-W", "error::DeprecationWarning",
                    str(ROOT / "examples" / "quickstart.py")], check=True, capture_output=True)


def test_deprecated_api_names_the_removal_version():
    import warnings

    import numpy as np
    from spintoolkit._deprecation import REMOVAL_VERSION, internal_use

    assert REMOVAL_VERSION == "0.3"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spintoolkit.SpinSystem(lattice_vectors=np.eye(2))
        spintoolkit.SpinOptimizer()
    messages = [str(w.message) for w in caught if w.category is DeprecationWarning]
    assert len(messages) == 2
    assert all("removed in spintoolkit 0.3" in m for m in messages)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with internal_use():
            spintoolkit.SpinSystem(lattice_vectors=np.eye(2))
    assert not caught


def test_readme_quickstart_block_runs():
    """The first Python block of the README runs with the new API only."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
    assert "solve_lswt" in code
    prelude = f"import sys; sys.path.insert(0, {str(ROOT / 'code-space')!r})\n"
    subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c", prelude + code],
                   check=True, capture_output=True)
