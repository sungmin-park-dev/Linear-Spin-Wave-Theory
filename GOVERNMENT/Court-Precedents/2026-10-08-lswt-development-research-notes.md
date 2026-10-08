---
frontmatter-version: 1
title: Decision - LSWT Theory and Development Research Notes as Content Source
section: decisions
decision-type: source-authority
status: accepted
last-edited-by: claude
created: 2026-10-08
updated: 2026-10-08
scope: docs/lswt, docs/development, workbench/notes
reviewed-by: user
reviewed-at: 2026-10-08
---

# LSWT theory and development research notes as content source

사용자는 2026-10-08에 LSWT 이론 문서와 개발 노트도 research-workspace 앱에서 관리하기로 했다("우선 유도와 개발 노트 등도 지금 workspace에서 관리할 수 있도록 하고 싶어", 두 가지를 모두 옮기자는 추천에 "추천대로 하자"). 2026-10-05 NBCP 결정과 같은 방식이다.

1. `docs/lswt/`의 이론 Markdown 17개(`00-*`–`04-*`)와 `docs/development/`의 Beamer(`main.tex`, `sections/`, `appendices/`)는 2026-10-08 상태로 보존하고 더 고치지 않는다.
2. 같은 날 그 내용을 `workbench/notes/`의 LaTeX 연구노트 6개로 옮겼다.
   - LSWT 이론: `lswt-foundations`(00-foundations, 부록 Luttinger–Tisza), `lswt-derivation`(01-derivation, 03-examples, 부록 paraunitarity proofs), `lswt-observables`(02-observables, 부록 thermodynamic derivations). pandoc으로 변환했고 본문은 그대로다. semantic 식 label 59개는 `\label`/`\eqref`로, 문서 간 link는 절 참조 또는 `\lswtnoteref`로 바꿨다. 각 절 머리 주석에 원래 검토 상태(draft 16, in-review 1)를 남겼다.
   - 개발 노트: `dev-design`(본문 1–5장), `dev-verification`(부록 A), `dev-decisions`(부록 B, 결정 기록 D01–D47). 슬라이드 104장은 소절이 되었고 슬라이드 본문은 그대로 묶음(group) 안에 두었다. 표지·장 표지 슬라이드는 뺐다. Beamer 전용 환경의 논문형 정의는 `workbench/macros.tex`에 있다.
3. 이후 이 내용의 갱신은 연구노트에서 하며, 연구노트가 내용의 편집 원본이다. LSWT 이론 연구노트는 영어로 쓰고 `lswt-writing-style.md`를 계속 적용한다.
4. 2026-08-01 결정의 단일 정본 원칙은 위치만 바뀐다. LSWT 이론 claim을 소유하는 정본은 사용자 승인 Markdown 대신 사용자 승인 LSWT 연구노트다. Primary PDF가 원문 판정 근거인 점, 사용자 Human Physics and Mathematics Review 전에는 accepted가 아닌 점, 파생 PDF를 직접 고치지 않는 점은 그대로다. 2026-08-09 결정과 2026-09-16 통합 기록의 작성 경로 조항은 이 결정으로 대체한다.
5. 옮기기 전 점검(2026-10-08)에서 이론 문서의 확인된 오류 3개와 고칠 항목 2개를 찾았다. 옮길 때는 고치지 않았고, 연구노트에서 따로 고친다. 목록은 `/mnt/project-files/lswt-review/2026-10-08-lswt-workspace-check.md`(프로젝트 공유 폴더)에 있다.

[2026-08-01 결정](2026-08-01-lswt-markdown-source-authority.md) · [2026-10-05 NBCP 결정](2026-10-05-nbcp-research-notes.md) · [연구노트](../../workbench/notes/)
