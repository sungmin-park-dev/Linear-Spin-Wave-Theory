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
