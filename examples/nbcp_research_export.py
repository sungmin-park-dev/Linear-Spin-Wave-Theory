"""Build the canonical NBCP LaTeX manuscript as a PDF preview.

Run from any directory. XeLaTeX and BibTeX are the document-build dependencies.
Scientific calculations and source files are never modified by this exporter. It
writes only generated files: docs/nbcp/output/ and the vector figures
docs/nbcp/figures/<name>.pdf compiled from docs/nbcp/figures/tikz/<name>.tex.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/nbcp/main.tex'
OUTPUT = ROOT / 'docs/nbcp/output'
FIGURES = ROOT / 'docs/nbcp/figures'
TIKZ = FIGURES / 'tikz'
MANIFEST = OUTPUT / 'research-export.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_tikz_figures(xelatex):
    """Compile each standalone TikZ source to a vector PDF next to its folder.

    A figure is rebuilt only when its source or the shared style changed, or the
    PDF no longer matches the recorded hash, so an unchanged figure keeps
    byte-identical output (PDF timestamps otherwise differ on every build).
    Returns the per-figure records for the manifest and the hashed source files.
    """
    style = TIKZ / 'style.tex'
    sources = sorted(p for p in TIKZ.glob('*.tex') if p != style)
    recorded = {}
    if MANIFEST.is_file():
        recorded = json.loads(MANIFEST.read_text()).get('tikz_figures', {})
    records = {}
    for source in sources:
        pdf = FIGURES / f'{source.stem}.pdf'
        record = {'source_sha256': digest(source), 'style_sha256': digest(style)}
        old = recorded.get(source.stem, {})
        current = (pdf.is_file() and old.get('pdf_sha256') == digest(pdf)
                   and all(old.get(key) == value for key, value in record.items()))
        if not current:
            with tempfile.TemporaryDirectory(prefix='nbcp-figure-') as directory:
                work = Path(directory)
                shutil.copy2(source, work)
                shutil.copy2(style, work)
                result = subprocess.run(
                    [xelatex, '-interaction=nonstopmode', '-halt-on-error',
                     '-no-shell-escape', source.name],
                    cwd=work, capture_output=True, text=True,
                )
                if result.returncode:
                    raise RuntimeError(f'{source.name}:\n' + result.stdout[-8000:]
                                       + result.stderr[-8000:])
                log = (work / f'{source.stem}.log').read_text(errors='replace')
                bad = [line for line in log.splitlines() if any(term in line for term in
                       ['Overfull', 'Missing character', 'undefined references'])]
                if bad:
                    raise RuntimeError(f'Figure review failed ({source.name}):\n'
                                       + '\n'.join(bad))
                shutil.copy2(work / f'{source.stem}.pdf', pdf)
        record['pdf_sha256'] = digest(pdf)
        records[source.stem] = record
    return records, ([style, *sources] if sources else [])


def manuscript_inputs():
    """Resolve the manuscript's literal TeX inputs and figure dependencies."""
    sources, figures, active = {}, {}, set()

    def visit(path):
        path = path.resolve()
        path.relative_to(ROOT)
        if path in active:
            raise ValueError(f'Cyclic TeX input: {path}')
        if path in sources:
            return
        active.add(path)
        text = path.read_text()
        sources[path] = text
        # Literal paths are relative to the main document's build directory.
        for target in re.findall(r'\\input\{([^}]+)\}', text):
            child = SOURCE.parent / target
            visit(child if child.suffix else child.with_suffix('.tex'))
        for names in re.findall(r'\\bibliography\{([^}]+)\}', text):
            for name in names.split(','):
                bib = (SOURCE.parent / f'{name.strip()}.bib').resolve()
                bib.relative_to(ROOT)
                sources[bib] = bib.read_text()
        for target in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
            figure = (SOURCE.parent / target).resolve()
            figure.relative_to(ROOT)
            if not figure.is_file():
                raise FileNotFoundError(figure)
            figures[figure] = digest(figure)
        active.remove(path)

    visit(SOURCE)
    return sources, figures


def main():
    xelatex = shutil.which('xelatex')
    if not xelatex:
        raise RuntimeError('XeLaTeX is required.')
    bibtex = shutil.which('bibtex')
    if not bibtex:
        raise RuntimeError('BibTeX is required.')
    # Figure PDFs must exist before the manuscript's \includegraphics targets resolve.
    tikz_records, tikz_inputs = build_tikz_figures(xelatex)
    sources, figures = manuscript_inputs()
    inputs = [*sources, *tikz_inputs, Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    figure_hashes = {str(p.relative_to(ROOT)): h for p, h in figures.items()}
    with tempfile.TemporaryDirectory(prefix='nbcp-research-') as directory:
        scratch = Path(directory)
        for path in [*sources, *figures]:
            target = scratch / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        build = scratch / SOURCE.parent.relative_to(ROOT)

        def run_xelatex():
            result = subprocess.run(
                [xelatex, '-interaction=nonstopmode', '-halt-on-error',
                 '-no-shell-escape', '-jobname=research-note', 'main.tex'],
                cwd=build, capture_output=True, text=True,
            )
            if result.returncode:
                raise RuntimeError(result.stdout[-8000:] + result.stderr[-8000:])

        # XeLaTeX writes the citations, BibTeX resolves them, and two more
        # XeLaTeX passes settle numbering, cross-references and the table of contents.
        run_xelatex()
        result = subprocess.run([bibtex, 'research-note'], cwd=build,
                                capture_output=True, text=True)
        blg = (build / 'research-note.blg').read_text(errors='replace')
        bib_bad = [line for line in blg.splitlines() if any(term in line for term in
                   ['Warning--', "I didn't find", 'error message', 'Illegal'])]
        if result.returncode > 1 or bib_bad:
            raise RuntimeError('BibTeX review failed:\n' + '\n'.join(bib_bad)
                               + result.stdout[-4000:] + result.stderr[-4000:])
        run_xelatex()
        run_xelatex()
        log = (build / 'research-note.log').read_text(errors='replace')
        bad = [line for line in log.splitlines() if any(term in line for term in
               ['Overfull', 'Missing character', 'undefined references',
                'multiply defined', 'There were undefined citations'])]
        if bad:
            raise RuntimeError('PDF review failed:\n' + '\n'.join(bad))
        for path in [*inputs, *figures]:
            expected = figure_hashes.get(str(path.relative_to(ROOT)),
                                         hashes.get(str(path.relative_to(ROOT))))
            if digest(path) != expected:
                raise RuntimeError(f'Input changed during build: {path}')
        OUTPUT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(build / 'research-note.pdf', OUTPUT / 'research-note.pdf')
        manifest = {
            'generated_utc': datetime.now(timezone.utc).isoformat(),
            'canonical_source': str(SOURCE.relative_to(ROOT)),
            'source_format': 'LaTeX',
            'review_status': 'in-review',
            'engine': subprocess.run([xelatex, '--version'], capture_output=True,
                                     text=True, check=True).stdout.splitlines()[0],
            'inputs_sha256': hashes,
            'figure_inputs_sha256': figure_hashes,
            'tikz_figures': tikz_records,
            'outputs_sha256': {'research-note.pdf': digest(OUTPUT / 'research-note.pdf')},
            'latex_review_messages': bad,
            'source_display_equations': sum(
                text.count(r'\begin{equation}') + text.count(r'\[')
                for text in sources.values()),
            'scope': 'PDF preview from canonical TeX; no physics scan rerun',
        }
        (OUTPUT / 'research-export.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
