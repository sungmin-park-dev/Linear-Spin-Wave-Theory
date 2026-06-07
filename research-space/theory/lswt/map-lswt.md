---
frontmatter-version: 1
title: Map - lswt
section: theory/lswt
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-06-04
must-read: GOVERNMENT/Agents-Bylaws/templates/map-template.md
---

# Map - lswt

> 이 파일은 `research-space/theory/lswt/` 디렉토리의 인덱스입니다.
> 새 LSWT 정본 문서나 appendix를 추가하면 이 표도 갱신합니다.

LSWT 이론 문서를 정본화하기 위한 작업 공간. 파일명에는 순번이나 분류코드를
넣지 않고, 1단계 폴더와 이 map으로 문서의 역할과 읽는 순서를 관리한다.

## Contents

| Item | Role |
|---|---|
| `README.md` | 이 디렉토리의 목적, 기준 소스, 작업 원칙 |
| `map-lswt.md` | This navigation file |
| `current-sections-audit.md` | 기존 Markdown theory section의 내용 audit |
| `foundations/` | LSWT 유도 전에 고정해야 하는 Hamiltonian, notation, local-frame convention |
| `derivation/` | spin operator에서 momentum-space BdG Hamiltonian과 diagonalization까지의 LSWT 본체 |
| `observables/` | diagonalization 이후 계산되는 물리량과 response function |
| `examples/` | worked example와 코드 검증으로 이어지는 사용 절차 |
| `appendices/` | 본문 흐름을 보조하는 증명과 유도 |

## Foundations

| Item | Role |
|---|---|
| `foundations/lswt-overview.md` | LSWT scope와 전체 계산 흐름 |
| `foundations/notation-and-conventions.md` | symbol table, site/link/displacement convention, magnetic sublattice convention |
| `foundations/bilinear-spin-hamiltonian.md` | bilinear exchange Hamiltonian, Zeeman term, g-tensor |
| `foundations/classical-order-and-local-frame.md` | classical ordered state, rotation matrix, local `+/-/0` basis |

## Derivation

| Item | Role |
|---|---|
| `derivation/holstein-primakoff-expansion.md` | HP transformation, transverse/longitudinal fluctuations, expansion order |
| `derivation/real-space-boson-hamiltonian.md` | real-space bosonic Hamiltonian, `H2`, `H4`, coefficient definitions |
| `derivation/momentum-space-bdg-hamiltonian.md` | Fourier convention, magnetic BZ, Nambu spinor, `A_k/B_k` block |
| `derivation/paraunitary-diagonalization.md` | Bogoliubov/Colpa diagonalization, paraunitarity, stability conditions |

## Observables

| Item | Role |
|---|---|
| `observables/magnon-observables.md` | spectrum, zero-point energy, occupation, post-diagonalization quantities |
| `observables/thermodynamics.md` | partition function, internal/free energy, entropy, specific heat |
| `observables/spin-correlations.md` | real-time spin-spin and sublattice correlations |
| `observables/structure-factor-and-spectral-function.md` | static/dynamic structure factor and spectral function |
| `observables/topological-magnon-quantities.md` | skyrmion number, Chern number, thermal Hall conductance |

## Examples

| Item | Role |
|---|---|
| `examples/worked-example.md` | worked LSWT example and validation route |

## Appendices

| Item | Role |
|---|---|
| `appendices/luttinger-tisza-method.md` | Luttinger-Tisza appendix |
| `appendices/paraunitarity-proofs.md` | Paraunitarity proof appendix |
| `appendices/thermodynamic-derivations.md` | Thermodynamic derivations appendix |

## Archived Records

| Item | Role |
|---|---|
| `GOVERNMENT/Working-Pad/issue-notes/closed/260604-lswt-section-migration-record.md` | 기존 Markdown/LaTeX source에서 새 LSWT 정본 파일로 가는 구현 완료 mapping 기록 |

## Agent Instructions

- `README.md`에는 목적과 작업 원칙을 둔다. 파일 목록은 이 map에 둔다.
- `common/` 문서는 아직 만들지 않는다. LSWT 문서 안에서 convention이 안정된 뒤 분리한다.
- 본문 이식 중 code discrepancy나 common 후보가 보이면 해당 section의 작업 메모에 기록한다.
- 새 파일을 추가할 때는 먼저 어느 폴더의 논리 단위인지 확인하고, 순번형 파일명은 쓰지 않는다.
