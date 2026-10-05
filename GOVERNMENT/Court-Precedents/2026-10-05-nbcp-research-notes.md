---
frontmatter-version: 1
title: Decision - NBCP Research Notes as Content Source
section: decisions
decision-type: source-authority
status: accepted
last-edited-by: claude
created: 2026-10-05
updated: 2026-10-05
scope: docs/nbcp, workbench/notes
reviewed-by: user
reviewed-at: 2026-10-05
---

# NBCP research notes as content source

사용자는 2026-10-05에 NBCP 원고를 보존하고 그 내용을 research-workspace 앱의 연구노트로 복제하기로 했다("지금 원고는 보존하고, 연구노트를 중복해서 만들자. 추후 업데이트가 되면 원본은 연구노트로 하자"). 이 결정은 원본 위치에 관한 승인이다. 연구 내용의 물리·수학적 acceptance는 아니며 NBCP의 상태는 `in-review`로 유지한다.

1. `docs/nbcp/main.tex`와 장·부록·참고문헌 TeX는 2026-10-05 상태로 보존하고 더 고치지 않는다.
2. 같은 날 그 내용을 `workbench/notes/`의 주제별 LaTeX 연구노트 8개로 복제했다: `overview`(1·10장), `model-phase-diagram`(2–4장, 부록 E), `skyrmion-phase`(5장), `pseudo-goldstone-gap`(6장, 부록 A), `supersolidity-clock-rg`(7장, 부록 B), `y-phase`(8장, 부록 C), `v-phase`(9장), `numerical-verification`(부록 D). 본문은 글자 그대로이고, 다른 노트를 가리키는 참조(`\nbcpnoteref`)와 그림 경로(노트 폴더의 `figures/`)만 바꿨다.
3. 이후 NBCP 내용의 갱신은 연구노트에서 하며, 연구노트가 내용의 편집 원본이다. 원고를 연구노트에 맞춰 다시 만들지는 따로 정한다.
4. 연구노트는 본문만 둔다. 머리는 앱의 서식 "연구노트 (NBCP 스타일)"(`workbench/research.yaml`의 `latex-template: research-note`)이 붙이고, 원고 서식에만 있던 정의는 `workbench/macros.tex`에 둔다.
5. 2026-09-18 결정의 1항(원고가 유일한 편집 원본)은 이 결정으로 대체한다. 수식 label 보존, XeLaTeX, `docs/lswt/` Markdown 정본 원칙은 그대로다.

[이전 결정](2026-09-18-nbcp-latex-source-authority.md) · [연구노트](../../workbench/notes/)
