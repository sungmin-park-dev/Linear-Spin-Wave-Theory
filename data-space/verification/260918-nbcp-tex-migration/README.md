# NBCP LaTeX migration verification

- `source-map.json`: previous Markdown hash, dated archive, ordered TeX files,
  and preserved semantic-label destinations.
- `document-check.json`: equation/table/link preservation, protected-file
  hashes checked against a pre-edit snapshot, render comparison and visual QA.
- Current build inputs and figure hashes are in
  [research-export.json](../../../docs/nbcp/output/research-export.json).
- Pre-migration source, converter and style are in
  [the dated archive](../../../docs/archive/nbcp/2026-09-18-markdown-source.zip).

The comparison used the former converter's complete LaTeX output, checked all
inline/display math and table blocks exactly, and compared every rendered page
at 72 dpi. Only authoring/build provenance prose was revised. This verifies
source migration and layout, not the physical validity of the research claims.
