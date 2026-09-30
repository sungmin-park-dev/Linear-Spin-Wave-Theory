---
frontmatter-version: 1
title: Linear Spin Wave Theory Documentation
doc-path: docs/lswt
status: in-review
last-edited-by: codex
created: 2026-08-09
updated: 2026-09-16
---

# Linear Spin Wave Theory Documentation

[전체 문서](../README.md) · [NBCP 연구](../nbcp/README.md)

## 개요

이 디렉토리는 spin-wave theory(SWT)의 개념과 LSWT 유도를 영어 Markdown으로 정리한다. 독자가 가정과 convention을 이해하고 수식을 따라 계산할 수 있도록 개념별 문서로 구성하며, 폴더 번호는 권장하는 읽는 순서를 나타낸다.

### 현재 상태

문서별 작성 범위와 검토 상태는 [Documentation Audit](../../GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md)에서 확인한다. 사용자가 Human Physics and Mathematics Review를 마치고 frontmatter에 `status: accepted`, `reviewed-by: user`, `reviewed-at`을 기록한 문서만 이론 정본이다.

## 작성·검토 기준

| 확인할 내용 | 기준 문서 |
|---|---|
| 예상 독자, 문서 종류별 설명 깊이, 영어 문체와 References | [LSWT Writing Style](../../GOVERNMENT/Agents-Bylaws/policies/lswt-writing-style.md) |
| 기호의 의미와 사용 범위, notation convention | [Notation and Conventions](00-foundations/notation-and-conventions.md) |
| 원본·전사본·참고자료의 위치와 원문 section 대응 | [Source Inventory](sources/README.md) |
| 원문 보존·재배치, 작업 단위와 완료 기준, 사용자 검토와 출력 | [Canonical Document Lifecycle](../../GOVERNMENT/Agents-Bylaws/procedures/lswt-canonical-document-lifecycle.md) |
| Markdown 정본 원칙과 이전 경로 결정 | [Source Authority](../../GOVERNMENT/Court-Precedents/2026-08-01-lswt-markdown-source-authority.md), [Docs Authoring Surface](../../GOVERNMENT/Court-Precedents/2026-08-09-lswt-docs-authoring-surface.md) |
| 현재 통합 경로 | [2026-09-16 통합 기록](../../GOVERNMENT/Working-Pad/issue-notes/closed/260916-docs-topic-consolidation.md) |
| 문서별 미해결 사항과 다음 작업 | [Documentation Audit](../../GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md), [Task Queue](../../GOVERNMENT/Working-Pad/TASK-QUEUE.md) |

각 문서의 역할을 먼저 확인하고 필요한 기준을 읽는다. Writing style과 lifecycle의 `in-review` 상태는 문서 전체가 accepted policy라는 뜻이 아니며, 명시적인 사용자 결정은 작업 기준으로 적용한다. Notation의 미결정 항목은 해당 문서와 audit에서 확인한다.

## 읽는 순서

### 00. Foundations

| 순서 | 문서 | 역할 |
|---|---|---|
| 1 | [LSWT Overview](00-foundations/lswt-overview.md) | 적용 범위와 전체 계산 흐름 |
| 2 | [Notation and Conventions](00-foundations/notation-and-conventions.md) | 기호, index, geometry와 matrix 계약 |
| 3 | [Bilinear Spin Hamiltonian](00-foundations/bilinear-spin-hamiltonian.md) | Exchange, on-site anisotropy와 Zeeman term |
| 4 | [Classical Order and Local Frame](00-foundations/classical-order-and-local-frame.md) | 고전 스핀 방향과 local-frame rotation |

### 01. Derivation

| 순서 | 문서 | 역할 |
|---|---|---|
| 1 | [Holstein--Primakoff Expansion](01-derivation/holstein-primakoff-expansion.md) | Spin operator의 boson expansion과 truncation |
| 2 | [Real-Space Boson Hamiltonian](01-derivation/real-space-boson-hamiltonian.md) | Real-space quadratic Hamiltonian |
| 3 | [Momentum-Space BdG Hamiltonian](01-derivation/momentum-space-bdg-hamiltonian.md) | Fourier transform, Nambu spinor와 BdG blocks |
| 4 | [Paraunitary Diagonalization](01-derivation/paraunitary-diagonalization.md) | Bogoliubov transformation, Colpa와 stability |

### 02. Observables

| 순서 | 문서 | 역할 |
|---|---|---|
| 1 | [Magnon Observables](02-observables/magnon-observables.md) | Spectrum, zero-point energy와 occupation |
| 2 | [Thermodynamics](02-observables/thermodynamics.md) | Partition function, energy, entropy와 specific heat |
| 3 | [Spin Correlations](02-observables/spin-correlations.md) | Real-time 및 sublattice correlations |
| 4 | [Structure Factor and Spectral Function](02-observables/structure-factor-and-spectral-function.md) | Static·dynamic structure factor와 spectral function |
| 5 | [Topological Magnon Quantities](02-observables/topological-magnon-quantities.md) | Berry curvature, Chern number와 thermal Hall response |

### 03. Examples

| 순서 | 문서 | 역할 |
|---|---|---|
| 1 | [Worked Example](03-examples/worked-example.md) | Single-mode quadratic-boson 풀이와 검증 연결 |

### 04. Appendices

| 문서 | 역할 |
|---|---|
| [Luttinger--Tisza Method](04-appendices/luttinger-tisza-method.md) | Classical-order 후보 탐색 보조법 |
| [Paraunitarity Proofs](04-appendices/paraunitarity-proofs.md) | Paraunitary relation의 증명 보충 |
| [Thermodynamic Derivations](04-appendices/thermodynamic-derivations.md) | 긴 thermodynamic 유도 |

## 자료의 역할과 정본 기준

같은 내용을 담은 자료가 여러 개일 때, 각 자료가 맡는 역할과 현재 이론 정본의
기준은 다음과 같다.

| Material | Role |
|---|---|
| 사용자 지정 원본 PDF | Primary evidence |
| `docs/lswt/sources/01-editable-notes/note_lswt_reviewed.tex` | Editable transcription |
| `docs/lswt/sources/01-editable-notes/note_lswt_restructured.tex` | Structural reference |
| `legacy/research-notes/lswt/converted-markdown/` | Source-only legacy Markdown |
| 사용자 승인 `docs/lswt/00-*`–`04-*`의 이론 Markdown | Unique current theory canon |

원본에 무엇이 기록되어 있었는지 확인할 때는 원본 PDF를 기준으로 삼는다.
프로젝트가 현재 어떤 이론 설명을 채택하는지는 사용자가 승인한 Markdown을
기준으로 삼는다.

Source가 충돌하거나 물리적 의도가 불명확하면 자동으로 병합하지 않는다.
운영 상태와 legacy coverage는
[`260809-lswt-documentation-audit.md`](../../GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md)에서
관리한다.

## Editing and Publication Boundary

- 의미와 수식은 대응하는 Markdown owner에서만 수정한다.
- Overview는 관심 모델과 계산 흐름을 보여주는 핵심 연결 수식을 요약할 수 있다. 상세 정의, convention, 유도와 semantic equation ID는 대응하는 owner 문서가 소유하며, 이 README는 물리식을 반복하지 않는다.
- Display equation은 번호 없이 쓰고, 다시 참조하는 핵심 식에만 semantic ID를
  붙인다.
- Theory acceptance, code verification과 web publication은 서로 다른 상태다.
- 공개 TeX, PDF와 HTML은 accepted Markdown에서 생성한다. 사용자 검토용 preview는 draft/in-review Markdown에서도 생성하며, 생성물을 직접 수정하지 않는다.
- Legacy Markdown은 누락 대조용 evidence이며 canonical claim으로 인용하지
  않는다.

## References

### Internal Documents

- [Single Knowledge Canon](../../GOVERNMENT/User-Constitution/single-knowledge-canon.md): defines the repository's single-canon principle.
- [LSWT Markdown Source Authority](../../GOVERNMENT/Court-Precedents/2026-08-01-lswt-markdown-source-authority.md): defines the authority relationship among Markdown, the source PDF, and TeX derivatives.
- [LSWT Docs Authoring Surface](../../GOVERNMENT/Court-Precedents/2026-08-09-lswt-docs-authoring-surface.md): records the earlier authoring location and source-path updates.
- [LSWT Canonical Document Lifecycle](../../GOVERNMENT/Agents-Bylaws/procedures/lswt-canonical-document-lifecycle.md): defines review, acceptance, and derived-output boundaries.

### External Sources

- No external paper is currently cited in this navigation and governance document.
