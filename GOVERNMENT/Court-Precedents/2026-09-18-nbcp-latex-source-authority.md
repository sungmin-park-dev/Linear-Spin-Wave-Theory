---
frontmatter-version: 1
title: Decision - NBCP LaTeX Source Authority
section: decisions
decision-type: source-authority
status: accepted
last-edited-by: codex
created: 2026-09-18
updated: 2026-09-18
scope: docs/nbcp
reviewed-by: user
reviewed-at: 2026-09-18
---

# NBCP LaTeX source authority

사용자는 읽기 도구에 제약이 없음을 밝힌 뒤, LaTeX를 내용 원본으로 사용하는 제안의 범위를 NBCP 연구 문서로 한정하는 데 동의했다. 이 결정은 작성 형식과 원본 소유권에 관한 승인이다. 연구 내용의 물리·수학적 acceptance는 아니다.

1. `docs/nbcp/main.tex`와 포함된 장·부록·참고문헌 TeX 파일이 NBCP 내용의 유일한 편집 원본이다.
2. `preamble.tex`는 공통 서식, `metadata.tex`는 제목·날짜와 검토 상태를 관리한다. 수식의 semantic label과 근거 연결은 보존한다.
3. PDF는 XeLaTeX로 생성한다. `examples/nbcp_research_export.py`는 `docs/nbcp/output/`에 PDF와 생성 기록만 저장하며 중간 빌드 파일은 임시 디렉토리에서 처리한다.
4. Markdown은 navigation·검토 기록에 사용한다. 전환 직전 원본과 변환기는 `docs/archive/nbcp/2026-09-18-markdown-source.zip`에 보존하고 이후 편집하지 않는다.
5. `docs/lswt/`의 Markdown 단일 정본 원칙과 원자료 구분은 유지한다. 2026-09-16 NBCP 통합 기록의 Markdown 원본·변환 경로 조항만 이 결정으로 대체한다.
6. 장별 TeX로 나누는 작업은 새로운 물리 주장, 계산 결과 또는 acceptance를 추가하지 않는다. NBCP의 상태는 `in-review`다.

[편집 안내](../../docs/nbcp/README.md) · [이전 구조 기록](../Working-Pad/issue-notes/closed/260916-docs-topic-consolidation.md) · [전환 검증](../../data-space/verification/260918-nbcp-tex-migration/)
