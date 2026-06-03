# LSWT Section Migration Plan

이 문서는 기존 Markdown 문서를 `research-space/theory/lswt/` 정본 구조로
옮기기 위한 계획과 source-to-target 대응표다. 아직 섹션 본문을 rewrite하지 않는다.

기준 소스:

- `research-space/sources/lswt/note_lswt_restructured.tex`

## 정본 구조 참조

현재 LSWT 정본 파일 목록과 각 파일의 역할은 `map-lswt.md`에서 관리한다.

## 기존 Markdown에서 새 구조로 가는 mapping

| 기존 파일 | 새 위치 후보 | 처리 방식 |
|---|---|---|
| `research-space/theory/README.md` | `research-space/theory/lswt/README.md` | 이미 새 workspace README로 별도 생성. 기존 README는 상위 gateway로 남긴다. |
| `research-space/theory/notation.md` | `research-space/theory/lswt/appendices/A_notation.md` | **선택 A 확정**. LSWT appendix로 이식한다. 나중에 common notation만 추출할 수 있음. |
| `research-space/theory/sections/01_spin_wave_theory_intro.md` | `00_introduction.md`, `01_from_spins_to_bosons.md`, `02_diagonalization.md` | 분할 필요. 현재 한 파일에 너무 많은 섹션이 들어 있음. |
| `research-space/theory/sections/02_physical_quantities.md` | `03_physical_quantities.md` | **선택 A 확정**. 독립 section으로 유지한다. |
| `research-space/theory/sections/03_thermodynamics.md` | `04_thermodynamics.md` | 새 파일에 점진적으로 이식한다. |
| `research-space/theory/sections/04_correlations.md` | `05_correlations_and_structure_factors.md` | 새 파일에 점진적으로 이식한다. convention과 notation 정리 필요. |
| `research-space/theory/sections/05_topology.md` | `appendices/E_topological_quantities.md` | **선택 B 확정**. appendix로 이식한다. Thermal Hall bug note와 함께 검증 필요. |
| `research-space/theory/sections/06_worked_example.md` | `06_worked_example.md` | 새 파일에 점진적으로 이식한다. source/bibliography frontmatter 정리 필요. |

## LaTeX section에서 새 Markdown으로 가는 mapping

| `note_lswt_restructured.tex` 섹션 | 새 Markdown 후보 | 비고 |
|---|---|---|
| `Introduction` | `00_introduction.md` | 기존 Markdown에는 명시적 00 문서 없음 |
| `From Spins to Bosons` | `01_from_spins_to_bosons.md` | classical ground state, HP, bosonic Hamiltonian, momentum representation 포함 |
| `Classical ground state and local frame` | `01_from_spins_to_bosons.md` | common 후보: site/local frame convention |
| `Holstein-Primakoff transformation` | `01_from_spins_to_bosons.md` | LSWT-specific |
| `Bosonic Hamiltonian construction` | `01_from_spins_to_bosons.md` | B/B† discrepancy 검증 필요 |
| `Momentum space representation` | `01_from_spins_to_bosons.md` | common 후보: Fourier/BZ/magnetic sublattice convention |
| `Diagonalization` | `02_diagonalization.md` | LSWT/BdG-specific |
| `Colpa algorithm` | `02_diagonalization.md` | positive-definite caveat 필요 |
| `Physical quantities after diagonalization` | `03_physical_quantities.md` | 독립 section으로 유지하되 diagonalization 결과와 연결 |
| `Validity and limitations` | `02_diagonalization.md` | LSWT benchmark 역할에 중요 |
| `Thermodynamics in LSWT` | `04_thermodynamics.md` | 직접 대응 |
| `Correlations and Structure Factors` | `05_correlations_and_structure_factors.md` | 직접 대응 |
| `Example` | `06_worked_example.md` | 직접 대응 |
| `Notation and Symbol Table` | `appendices/A_notation.md` | common notation 분리 가능성 있음 |
| `Luttinger-Tisza method` | `appendices/B_luttinger_tisza.md` | 새 파일 필요 |
| `Para-unitarity proofs` | `appendices/C_paraunitarity.md` | 새 파일 필요 |
| `Thermodynamic derivations` | `appendices/D_thermodynamic_derivations.md` | 새 파일 필요 |
| `Topological quantities` | `appendices/E_topological_quantities.md` | 기존 topology 문서 이동 후보 |

## 실행 방식

- **선택 B 확정**: 기존 파일을 바로 이동/분할하지 않고, 새 LSWT 정본 파일을
  먼저 만든 뒤 내용을 점진적으로 이식한다.
- 기존 `research-space/theory/sections/` 파일은 source material로 남긴다.

## 이번 단계에서 하지 않는 일

- 기존 theory section 본문을 아직 rewrite하지 않는다.
- 기존 `sections/` 파일을 아직 삭제하지 않는다.
- `research-space/theory/common/`을 아직 만들지 않는다.
- code implementation은 건드리지 않는다.

## 다음 승인 필요 항목

- 새 skeleton 파일에 어느 섹션부터 내용을 이식할지.
- 각 파일의 표준 frontmatter와 verification block 형식.
- common 후보 기록을 각 섹션 안에 둘지, 별도 `common-candidates.md`로 둘지.
