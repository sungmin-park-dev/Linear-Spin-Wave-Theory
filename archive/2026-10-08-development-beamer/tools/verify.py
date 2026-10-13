"""Check the development PDF and its source snapshot using standard tools."""

import hashlib
import json
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
PDF = HERE / "output/pdf/development-log.pdf"
LOG = HERE / "build/development-log.log"
DIAGNOSTICS = (
    r"Overfull|Underfull|Missing character|Undefined control sequence|"
    r"LaTeX Error|Package .* Error|Emergency stop|Fatal error|"
    r"LaTeX Font Warning|There were undefined references|"
    r"Citation .* undefined|Reference .* undefined|KFB-BIB-PATCH-FAILED"
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*args):
    return subprocess.check_output(args, text=True)


def expand_source(path, source_hashes):
    """Read this document's literal input files in display order."""
    key = str(path.relative_to(HERE))
    assert key not in source_hashes, f"Repeated TeX input: {key}"
    source_hashes[key] = digest(path)
    source = path.read_text()

    def include(match):
        child = (HERE / match.group(1)).with_suffix(".tex")
        return expand_source(child, source_hashes)

    return re.sub(r"\\input\{([^}]+)\}", include, source)


def main():
    manifest = json.loads((HERE / "source-manifest.json").read_text())
    assert digest(HERE / "knowledge-factory-beamer.sty") == manifest["theme"]["sha256"]
    for path, expected in manifest["sources"].items():
        assert digest(ROOT / path) == expected, f"Source changed; review evidence: {path}"

    source_hashes = {}
    source = expand_source(HERE / "main.tex", source_hashes)
    body, appendix = re.split(r"\\appendix\b", source)
    section_pattern = r"\\section(?:\[[^]]*\])?\{([^}]+)\}"
    main_sections = re.findall(section_pattern, body)
    appendix_sections = re.findall(section_pattern, appendix)
    assert main_sections == [
        "개요", "시스템 구조와 계산 흐름", "공통 자료구조와 전달 규약",
        "모듈과 폴더 구성", "개발 계획",
    ], main_sections
    assert appendix_sections == ["구현·검증 기록", "결정·관리·참고자료"]
    for label in ["current-status", "implemented", "verification", "layout-verification"]:
        assert f"label={label}]" in appendix, f"Record outside appendix: {label}"
    assert r"\tableofcontents" in body
    frames = re.findall(r"\\begin\{frame\}(?:\[([^]]*)\])?", source)
    labels = [re.search(r"(?:^|,)label=([^,]+)", options).group(1) for options in frames]
    assert len(labels) == len(set(labels)), "Duplicate frame labels"
    diagnostics = re.findall(DIAGNOSTICS, LOG.read_text())
    assert not diagnostics, diagnostics

    info = run("pdfinfo", str(PDF))
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    assert pages == len(frames), (pages, len(frames))
    fonts = run("pdffonts", str(PDF))
    font_rows = fonts.splitlines()[2:]
    assert font_rows and all(row.split()[-5] == "yes" for row in font_rows), fonts
    text = run("pdftotext", "-layout", str(PDF), "-")
    assert "2D Spin-System Toolkit" in info
    for token in ["2D Spin-System Toolkit", "SpinModel", "217", "215", "198", "40", "17", "0.3333333"]:
        assert token in text, f"Missing expected content: {token}"

    report = {
        "pdf": "output/pdf/development-log.pdf",
        "pdf_sha256": digest(PDF),
        "main_tex_sha256": digest(HERE / "main.tex"),
        "tex_source_sha256": source_hashes,
        "main_sections": main_sections,
        "appendix_sections": appendix_sections,
        "pages": pages,
        "frame_labels": labels,
        "font_count": len(font_rows),
        "all_fonts_embedded": True,
        "log_diagnostics": diagnostics,
        "theme_hash_verified": True,
        "evidence_source_count": len(manifest["sources"]),
        "visual_review": "Required separately; not established by this script.",
    }
    (HERE / "output/qa.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {pages} pages, {len(font_rows)} embedded fonts, clean log and source hashes.")


if __name__ == "__main__":
    main()
