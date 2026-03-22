#!/usr/bin/env python3
"""
Process converted LaTeX sections to refine Markdown format.

Transformations:
1. Fix equation references (remove verbose pandoc output)
2. Convert \kvec to \mathbf{k}
3. Fix figure paths to ../figures/
4. Clean up LaTeX remnants
"""

import re
from pathlib import Path

def clean_references(text):
    """Simplify verbose pandoc equation references."""
    # Pattern: [\[ref\]](#ref){reference-type="eqref" reference="ref"}
    # Replace with: (Eq. \ref{ref})
    pattern = r'\[\\\[([^\]]+)\\\]\]\(#([^\)]+)\)\{[^}]+\}'
    replacement = r'(Eq. \\ref{\1})'
    text = re.sub(pattern, replacement, text)

    # Also clean up simple references like "Eqn. [\[...\]](...)"
    pattern2 = r'Eqn?\.\s*\[\\\[([^\]]+)\\\]\]\(#[^\)]+\)\{[^}]+\}'
    replacement2 = r'Eq. \\ref{\1}'
    text = re.sub(pattern2, replacement2, text)

    return text

def convert_kvec(text):
    """Convert \kvec to \mathbf{k}."""
    text = text.replace(r'\kvec', r'\mathbf{k}')
    return text

def fix_figure_paths(text):
    """Update figure paths to use ../figures/ directory."""
    # This will depend on actual figure references in the documents
    # Placeholder for now
    return text

def clean_latex_remnants(text):
    """Remove or fix LaTeX-specific formatting that doesn't render well in Markdown."""
    # Remove double $$ around equation environments
    # Pattern: $$\begin{equation}
    text = re.sub(r'\$\$\\begin\{equation\}', r'$$\n\\begin{equation}', text)
    text = re.sub(r'\\end\{equation\}\$\$', r'\\end{equation}\n$$', text)

    # Similarly for align, align*
    text = re.sub(r'\$\$\\begin\{align', r'$$\n\\begin{align', text)
    text = re.sub(r'\\end\{align\*?\}\$\$', r'\\end{align}\n$$', text)

    return text

def process_section(input_path, output_path):
    """Process a single section file."""
    print(f"Processing {input_path.name}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply transformations
    content = clean_references(content)
    content = convert_kvec(content)
    content = fix_figure_paths(content)
    content = clean_latex_remnants(content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  → Saved to {output_path.name}")

def main():
    # Define paths
    latex_dir = Path(__file__).parent / "latex_sections"
    sections_dir = Path(__file__).parent / "sections"

    # Mapping of input to output filenames
    mappings = {
        "section_01_intro.md": "01_spin_wave_theory_intro.md",
        "section_02_physical.md": "02_physical_quantities.md",
        "section_03_thermo.md": "03_thermodynamics.md",
        "section_04_corr.md": "04_correlations.md",
        "section_05_topo.md": "05_topology.md",
        "section_06_example.md": "06_worked_example.md",
    }

    for input_name, output_name in mappings.items():
        input_path = latex_dir / input_name
        output_path = sections_dir / output_name

        if input_path.exists():
            process_section(input_path, output_path)
        else:
            print(f"Warning: {input_path} not found")

    print("\nDone! All sections processed.")

if __name__ == "__main__":
    main()
