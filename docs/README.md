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

Toolkit 개발 설계와 LSWT 일반 이론을 주제별로 관리한다. NBCP 연구는 2026-10-09 별도 저장소 `nbcp-spin-supersolid`로 옮겼다([결정](../GOVERNMENT/Court-Precedents/2026-10-09-nbcp-repository-split.md)). 현재 집필하는 본문, 그 근거인 참고자료, 과거 문서와 생성 출력물은 아래 위치로 구분한다.

| 찾는 내용 | 위치 | 용도 |
|---|---|---|
| Toolkit 개발 설계 | [development/](development/README.md) | 개발 목표·구조·공통 자료구조·폴더 역할을 설명하는 Beamer |
| 개발 설계 PDF | [development/output/pdf/](development/output/pdf/development-log.pdf) | 로컬 검토용 생성 PDF |
| 패키지 사용법(영어 튜토리얼) | [tutorials/](tutorials/README.md) | 첫 계산부터 중성자·M(h)·위상까지 5편. 각 수치는 해석해·문헌·ED와 대조하고 테스트로 확인한다. |
| LSWT 일반 이론 | [lswt/](lswt/README.md) | 정의·가정·일반 유도·관측량 설명. 번호 폴더는 읽는 순서다. |
| LSWT 원자료 | [lswt/sources/](lswt/sources/README.md) | 원본 노트·TeX 전사본·외부 논문·검증 근거 |
| 집필을 종료한 문서 | [archive/](archive/README.md) | 이전 문서·스냅샷. 기존 legacy 자료의 위치도 안내한다. |

## 현재 편집할 파일

[2026-10-08 결정](../GOVERNMENT/Court-Precedents/2026-10-08-lswt-development-research-notes.md)에 따라 내용의 편집 원본은 research-workspace 앱의 연구노트 [`workbench/notes/`](../workbench/notes/)다.

- 일반 이론: `lswt-foundations`, `lswt-derivation`, `lswt-observables`. [docs/lswt/](lswt/README.md)의 Markdown은 2026-10-08 상태로 보존하며 더 고치지 않는다.
- 개발 설계와 결정 기록: `dev-design`, `dev-verification`, `dev-decisions`. [development/](development/README.md)의 Beamer는 2026-10-08 상태로 보존한다.
- 계산 코드와 결과 데이터는 각각 [examples/](../examples/)와 [data-space/](../data-space/)에서 관리한다.

폴더 위치는 검토 완료 여부를 뜻하지 않는다. LSWT 이론의 acceptance와 코드 검증은 별도로 기록한다. 참고자료는 현재 주장을 확인하는 근거이며 본문과 동등한 편집 원본이 아니다.

현재 구조와 이전 경로의 대응은 [2026-09-16 문서 통합 기록](../GOVERNMENT/Working-Pad/issue-notes/closed/260916-docs-topic-consolidation.md)에 있다.
