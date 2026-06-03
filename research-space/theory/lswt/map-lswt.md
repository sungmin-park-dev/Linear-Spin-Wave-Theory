---
frontmatter-version: 1
title: Map - lswt
section: theory/lswt
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-03
must-read: GOVERNMENT/Agents-Bylaws/templates/map-template.md
---

# Map - lswt

> 이 파일은 `research-space/theory/lswt/` 디렉토리의 인덱스입니다.
> 새 LSWT 정본 문서나 appendix를 추가하면 이 표도 갱신합니다.

LSWT 이론 문서를 정본화하기 위한 작업 공간.

## Contents

| Item | Role |
|---|---|
| `README.md` | 이 디렉토리의 목적, 기준 소스, 작업 원칙 |
| `map-lswt.md` | This navigation file |
| `current-sections-audit.md` | 기존 Markdown theory section의 내용 audit |
| `section-migration-plan.md` | 기존 Markdown/LaTeX source에서 새 LSWT 정본 파일로 가는 migration plan |
| `00_introduction.md` | LSWT 문서의 scope, benchmark 역할, common convention 질문 |
| `01_from_spins_to_bosons.md` | Hamiltonian, local frame, HP transformation, quadratic boson Hamiltonian, momentum representation |
| `02_diagonalization.md` | Colpa diagonalization, paraunitarity, validity and limitations |
| `03_physical_quantities.md` | Diagonalization 이후 계산되는 물리량 |
| `04_thermodynamics.md` | LSWT thermodynamic quantities |
| `05_correlations_and_structure_factors.md` | Correlations, structure factors, spectral functions |
| `06_worked_example.md` | Worked LSWT example |
| `appendices/` | Notation, Luttinger-Tisza, paraunitarity, derivations, topology appendices |

## Appendices

| Item | Role |
|---|---|
| `appendices/A_notation.md` | LSWT notation and symbol table |
| `appendices/B_luttinger_tisza.md` | Luttinger-Tisza appendix |
| `appendices/C_paraunitarity.md` | Paraunitarity proof appendix |
| `appendices/D_thermodynamic_derivations.md` | Thermodynamic derivations appendix |
| `appendices/E_topological_quantities.md` | Topological quantities appendix |

## Agent Instructions

- `README.md`에는 목적과 작업 원칙을 둔다. 파일 목록은 이 map에 둔다.
- `section-migration-plan.md`는 source-to-target 계획으로 유지하고, navigation 역할을 하지 않는다.
- `common/` 문서는 아직 만들지 않는다. LSWT 문서 안에서 convention이 안정된 뒤 분리한다.
- 본문 이식 중 code discrepancy나 common 후보가 보이면 해당 section의 작업 메모에 기록한다.
