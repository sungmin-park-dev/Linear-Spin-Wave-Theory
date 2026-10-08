---
frontmatter-version: 1
title: Documentation
doc-path: docs
status: in-review
last-edited-by: codex
created: 2026-09-16
updated: 2026-09-23
---

# 문서 안내

Toolkit 개발 설계, LSWT 일반 이론과 NBCP 연구 노트를 주제별로 관리한다. 현재 집필하는 본문, 그 근거인 참고자료, 과거 문서와 생성 출력물은 아래 위치로 구분한다.

| 찾는 내용 | 위치 | 용도 |
|---|---|---|
| Toolkit 개발 설계 | [development/](development/README.md) | 개발 목표·구조·공통 자료구조·폴더 역할을 설명하는 Beamer |
| 개발 설계 PDF | [development/output/pdf/](development/output/pdf/development-log.pdf) | 로컬 검토용 생성 PDF |
| 패키지 사용법(영어 튜토리얼) | [tutorials/](tutorials/README.md) | 첫 계산부터 중성자·M(h)·위상까지 5편. 각 수치는 해석해·문헌·ED와 대조하고 테스트로 확인한다. |
| LSWT 일반 이론 | [lswt/](lswt/README.md) | 정의·가정·일반 유도·관측량 설명. 번호 폴더는 읽는 순서다. |
| NBCP 연구 | [nbcp/](nbcp/README.md) | 모델별 질문·유도·수치 결과·기존 계산과의 비교 |
| LSWT 원자료 | [lswt/sources/](lswt/sources/README.md) | 원본 노트·TeX 전사본·외부 논문·검증 근거 |
| NBCP 원문 대조 | [Overleaf 대조 기록](nbcp/sources/overleaf-2026-09-16-review.md) | 원문 발췌·충돌·보류 기록 |
| 집필을 종료한 문서 | [archive/](archive/README.md) | 이전 문서·스냅샷. 기존 legacy 자료의 위치도 안내한다. |
| NBCP 출력물 | [nbcp/output/](nbcp/output/) | LaTeX에서 생성한 PDF와 생성 기록 |

## 현재 편집할 파일

[2026-10-05 결정](../GOVERNMENT/Court-Precedents/2026-10-05-nbcp-research-notes.md)과 [2026-10-08 결정](../GOVERNMENT/Court-Precedents/2026-10-08-lswt-development-research-notes.md)에 따라 내용의 편집 원본은 research-workspace 앱의 연구노트 [`workbench/notes/`](../workbench/notes/)다.

- 일반 이론: `lswt-foundations`, `lswt-derivation`, `lswt-observables`. [docs/lswt/](lswt/README.md)의 Markdown은 2026-10-08 상태로 보존하며 더 고치지 않는다.
- 개발 설계와 결정 기록: `dev-design`, `dev-verification`, `dev-decisions`. [development/](development/README.md)의 Beamer는 2026-10-08 상태로 보존한다.
- NBCP: 주제별 연구노트 8개. [nbcp/main.tex](nbcp/main.tex)는 2026-10-05 상태로 보존한다.
- 계산 코드와 결과 데이터는 각각 [examples/](../examples/)와 [data-space/](../data-space/)에서 관리한다. NBCP 노트에서 해당 실행·검증 기록을 연결한다.

폴더 위치는 검토 완료 여부를 뜻하지 않는다. LSWT 이론의 acceptance, NBCP 연구 검토, 코드 검증은 별도로 기록한다. 참고자료는 현재 주장을 확인하는 근거이며 본문과 동등한 편집 원본이 아니다.

현재 구조와 이전 경로의 대응은 [2026-09-16 문서 통합 기록](../GOVERNMENT/Working-Pad/issue-notes/closed/260916-docs-topic-consolidation.md)에 있다.
