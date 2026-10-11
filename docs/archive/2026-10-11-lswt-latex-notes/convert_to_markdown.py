"""Convert the LSWT research notes from LaTeX to Markdown (note.md), 2026-10-11.

Usage: python convert_to_markdown.py REPO_ROOT [NOTE_NAME ...]
Reads  <this folder>/<note>/main.tex and its \\input files,
writes REPO_ROOT/workbench/notes/<note>/note.md. Needs pandoc >= 3.
"""
import glob, os, re, subprocess, sys, tempfile

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(sys.argv[1])
only = sys.argv[2:]
macros = open(os.path.join(here, 'macros.tex'), encoding='utf8').read()
FMT = ('markdown-raw_tex-smart-header_attributes-grid_tables-multiline_tables'
       '-simple_tables+pipe_tables')


def build_tex(note_dir):
    main = open(os.path.join(note_dir, 'main.tex'), encoding='utf8').read()
    body = main.split('\\begin{document}')[1].split('\\end{document}')[0]
    body = re.sub(r'(?m)^\s*%.*$', '', body).replace('\\appendix', '')
    comments = []

    def include(m):
        name = m.group(1) if m.group(1).endswith('.tex') else m.group(1) + '.tex'
        text = open(os.path.join(note_dir, name), encoding='utf8').read()
        head = []
        lines = text.split('\n')
        while lines and lines[0].startswith('%'):
            head.append(lines.pop(0).lstrip('% ').rstrip())
        text = '\n'.join(l for l in lines if not l.lstrip().startswith('%'))
        comments.append(' '.join(h for h in head if h))
        return '\\rwcm{%d}\n%s\n' % (len(comments) - 1, text)

    body = re.sub(r'\\input\{([^}]+)\}', include, body)
    pre = '\\documentclass{article}\n' + macros + '\\newcommand{\\rwcm}[1]{RWCM#1RWCM}\n'
    return pre + '\\begin{document}\n' + body + '\n\\end{document}\n', comments


def post(md, comments):
    md = re.sub(r'\$\$(.+?)\$\$',
                lambda m: '\n\n$$\n' + m.group(1).strip() + '\n$$\n\n', md, flags=re.S)
    md = re.sub(r'(?m)^RWCM(\d+)RWCM$',
                lambda m: '<!-- ' + comments[int(m.group(1))].replace('--', '-') + ' -->', md)
    md = re.sub(r'`(<!--.*?-->)`\{=html\}', r'\1', md)
    return re.sub(r'\n{3,}', '\n\n', md).strip() + '\n'


for note_dir in sorted(glob.glob(os.path.join(here, 'lswt-*'))):
    name = os.path.basename(note_dir)
    if only and name not in only:
        continue
    tex, comments = build_tex(note_dir)
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'n.tex')
        open(src, 'w', encoding='utf8').write(tex)
        r = subprocess.run(['pandoc', '-f', 'latex', '-t', FMT, '--wrap=none',
                            '--lua-filter', os.path.join(here, 'obsidian.lua'), src],
                           capture_output=True, text=True, check=True)
    head = ('<!-- Converted from LaTeX on 2026-10-11. The LaTeX original is kept in '
            'docs/archive/2026-10-11-lswt-latex-notes/%s/. Equation numbers are fixed '
            'in this form; the original \\label of each numbered equation is in the '
            'comment before it. -->\n\n' % name)
    out = os.path.join(root, 'workbench', 'notes', name, 'note.md')
    open(out, 'w', encoding='utf8').write(head + post(r.stdout, comments))
    print(name, '->', os.path.relpath(out, root))
