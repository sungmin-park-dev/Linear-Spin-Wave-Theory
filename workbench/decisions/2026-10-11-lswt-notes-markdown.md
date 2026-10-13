---
frontmatter-version: 1
title: Decision - LSWT Theory Research Notes as Markdown
section: decisions
decision-type: source-authority
status: accepted
last-edited-by: claude
created: 2026-10-11
updated: 2026-10-11
scope: workbench/notes/lswt-*, docs/archive/2026-10-11-lswt-latex-notes
reviewed-by: user
reviewed-at: 2026-10-11
---

# LSWT theory research notes as Markdown

사용자는 2026-10-11에 LSWT 이론 연구노트를 research-workspace 앱에서 직접 편집하겠다고 했다("앱에서 직접 편집하고 싶어"). 앱은 Markdown 노트를 기본 형식으로 다루고 기록 칸(`note-<폴더>`)도 Markdown 노트에만 있다. 정본 형식에 대한 계획은 사용자 승인 "추천대로 진행해"로 확정했다.

1. `workbench/notes/lswt-*` 7개의 본문 형식을 LaTeX에서 Markdown(`note.md`)으로 바꾼다. 이 결정은 [2026-10-08 결정](2026-10-08-lswt-development-research-notes.md)의 "LSWT 이론 연구노트는 LaTeX" 부분만 대체한다. 그 결정의 나머지(연구노트가 내용의 편집 원본, 2026-10-09 구조)는 그대로다.
2. 2026-10-11 상태의 LaTeX 노트(`main.tex`, 절 파일, `note.yaml`, 그림)와 변환에 쓴 `macros.tex`·`preamble.tex`는 `docs/archive/2026-10-11-lswt-latex-notes/`에 보존하고 더 고치지 않는다. 같은 폴더에 변환 도구(`convert_to_markdown.py`, `obsidian.lua`)를 둔다.
3. 변환은 충실 변환이다. 본문과 수식은 그대로이고, 번호를 붙인 수식은 `\tag{n}`으로 번호가 고정되며, `\eqref`·`\ref`는 번호 또는 절 이름 글자로 바뀐다. 원래 `\label` 이름은 각 번호 수식 앞의 HTML 주석에 남겼다. 노트 간 참조(`\lswtnoteref`)는 글자로 남는다.
4. 검증(2026-10-11): 원본 LaTeX와 `note.md`를 각각 pandoc 문서 트리로 읽어 수식 1,653개(인라인 포함)의 개수·순서·내용이 일치함을 확인했다(불일치 0). 수식 밖 텍스트의 차이는 참조 자리표시자가 번호·절 이름으로 바뀐 것과 인용부호 토큰 분할뿐이다. 앱이 7개를 Markdown 노트로 인식하고 `agent:check` 오류 0이다. 앱에서의 렌더링은 사용자가 확인한다.
5. 정본의 acceptance는 바뀌지 않는다. 7개 모두 draft이고, 사용자 Human Physics and Mathematics Review 전에는 accepted가 아니다. 형식 변경은 검토를 대신하지도 무효로 하지도 않는다.
6. 열린 항목:
   - 번호 수식 참조 정리(실제로 참조하는 식만 번호, 이름 태그)는 본문이 바뀌는 작업이라 따로 승인받아 진행한다.
   - Markdown 이론 노트의 공개용 PDF 경로(pandoc + XeLaTeX 등)는 공개 시점에 정한다.
   - 개발 노트 `dev-*`는 아직 LaTeX다. Markdown 전환 여부는 따로 정한다.

[2026-10-08 결정](2026-10-08-lswt-development-research-notes.md) · [연구노트](../../workbench/notes/) · [보존한 LaTeX 노트](../../docs/archive/2026-10-11-lswt-latex-notes/)
