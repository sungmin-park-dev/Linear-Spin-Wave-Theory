---
frontmatter-version: 1
title: Decision - LSWT Markdown Source Authority
section: decisions
decision-type: source-authority
status: accepted
last-edited-by: codex
created: 2026-08-01
updated: 2026-08-01
effective-date: 2026-08-01
scope: research-space/theory/lswt
reviewed-by: user
reviewed-at: 2026-08-01
supersedes:
  - "AGENTS.md historical editable-LaTeX-master record"
  - "GOVERNMENT/Agents-Bylaws/procedures/theory_code_verification_plan.md source-authority clauses"
---

# Decision - LSWT Markdown Source Authority

## Context

과거 운영 기록은 `note_lswt_restructured.tex`를 editable LaTeX master로
지정했다. 이후 원본 PDF, reviewed TeX, restructured TeX와 현재 Markdown
workspace를 대조한 결과, 각 자료는 구조와 재현 상태가 달라 동등한 master로
관리할 수 없음을 확인했다.

프로젝트의 목표 흐름은 이론 정립, 코드 발전, 웹 게시다. 사용자는 2026-08-01
Markdown을 유일한 지식 정본 작성면으로 두고 TeX·PDF·HTML을 파생 형식으로
관리하는 방향을 승인했다.

## Decision

1. `research-space/theory/lswt/`의 이론 내용 Markdown 중 사용자가
   `status: accepted`, `reviewed-by: user`, `reviewed-at`으로 승인한 문서만
   현재 LSWT 이론 정본이다.
2. README, map, audit, Working-Pad, `theory/common/`, draft와 in-review 문서는
   정본에 포함하지 않는다. 결정 시점의 accepted 이론 문서는 0개다.
3. 자료의 역할은 다음과 같이 분리한다.

   | Material | Role |
   |---|---|
   | 사용자 지정 원본 PDF | Primary evidence |
   | `legacy/research-notes/lswt/note_lswt_reviewed.tex` | Editable transcription |
   | `research-space/sources/lswt/note_lswt_restructured.tex` | Structural reference |
   | 기존 generated PDF | Historical snapshot |
   | Accepted LSWT Markdown | Unique theory canon |

4. 현재 workflow에는 수동 편집하는 LaTeX master를 두지 않는다. TeX, PDF,
   HTML은 동일한 accepted Markdown manifest에서 단방향 생성한다.
5. 생성된 산출물은 직접 수정하지 않는다. 의미·수식 변경은 Markdown으로,
   조판 변경은 template 또는 renderer 설정으로 환류한다.
6. theory accepted, code verified, web published는 서로 다른 상태다.
7. 과거 migration 기록과 legacy source는 삭제하거나 현재 결정에 맞게
   소급 수정하지 않는다.

## Evidence Identity

결정 당시 사용자 지정 원본 PDF의 식별값은 다음과 같다.

- Filename: `Linear_Spin_Wave_Theory___Note.pdf`
- SHA-256: `f00ac6c0bd33779702361a1b4237fe7962f000a2bd105c8c89ba37603ecd3503`
- Stable repository location: pending a separate preservation decision

이 식별값은 repo의 `note_lswt_restructured.pdf`와 다른 자료임을 구분하기 위한
것이며, 외부 경로의 파일을 저장소에 복사했다는 뜻이 아니다.

## Consequences

- `AGENTS.md`와
  `GOVERNMENT/Agents-Bylaws/procedures/theory_code_verification_plan.md`의
  editable LaTeX master 지정은 이 결정으로 대체된다.
- 첫 본문 작업은 `foundations/notation-and-conventions.md`에서 시작한다.
- `foundations/bilinear-spin-hamiltonian.md`와 함께 HTML·TeX·PDF 출력 pilot을
  거친 뒤 Markdown 문법과 수식·인용 계약을 확정한다.
- renderer와 공개 배포 방식은 pilot 검증 후 별도로 결정한다.

## Related Documents

- `GOVERNMENT/User-Constitution/single-knowledge-canon.md`
- `GOVERNMENT/Agents-Bylaws/procedures/lswt-canonical-document-lifecycle.md`
- `research-space/theory/lswt/README.md`
- `research-space/theory/lswt/current-sections-audit.md`

