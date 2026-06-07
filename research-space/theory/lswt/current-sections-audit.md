# Current LSWT Theory Sections Audit

이 문서는 현재 `research-space/theory/` 아래에 있는 Markdown 문서가 무엇을
담고 있는지 파악하기 위한 audit이다. 아직 이론 내용 자체는 수정하지 않는다.

> 2026-06-04 update: 이 audit은 번호형 파일 배치를 벗어나기 전의 상태를
> 기록한 문서다. 현재 활성 구조는 `map-lswt.md`의 `foundations/`,
> `derivation/`, `observables/`, `examples/`, `appendices/` 구성을 따른다.

기준 소스:

- `research-space/sources/lswt/note_lswt_restructured.tex`

## 현재 문서 목록

| 현재 파일 | 분량 | 현재 역할 | 상태 |
|---|---:|---|---|
| `research-space/theory/README.md` | 144 lines | 기존 theory 문서 안내 | 새 `lswt/` 위치에 맞게 갱신 필요 |
| `research-space/theory/notation.md` | 112 lines | 기호와 notation 요약 | LSWT appendix로 이식하기로 결정 |
| `research-space/theory/sections/01_spin_wave_theory_intro.md` | 955 lines | SWT 도입, HP, 회전, Hamiltonian, momentum, diagonalization까지 포함 | 너무 많은 내용이 한 파일에 몰림 |
| `research-space/theory/sections/02_physical_quantities.md` | 45 lines | 물리량 요약 표 | 독립 section으로 유지하기로 결정 |
| `research-space/theory/sections/03_thermodynamics.md` | 325 lines | 열역학 물리량 | restructured 소스와 대체로 직접 대응 |
| `research-space/theory/sections/04_correlations.md` | 903 lines | correlation, structure factor, spectral function | restructured 소스와 직접 대응하지만 길고 convention 의존도가 큼 |
| `research-space/theory/sections/05_topology.md` | 85 lines | skyrmion, Chern, Thermal Hall | appendix로 이식하기로 결정 |
| `research-space/theory/sections/06_worked_example.md` | 240 lines | quadratic boson Hamiltonian 예제 | restructured의 worked example와 대응 |

## LaTeX 기준 구조

`note_lswt_restructured.tex`의 주요 구조:

| LaTeX 위치 | 내용 |
|---|---|
| Introduction | 전체 목적과 모델 배경 |
| From Spins to Bosons | classical ground state, local frame, HP, bosonic Hamiltonian, momentum representation |
| Diagonalization | Colpa algorithm, physical quantities after diagonalization, validity and limitations |
| Thermodynamics in LSWT | partition function, internal energy, free energy, entropy, specific heat, boson number |
| Correlations and Structure Factors | spin-spin correlation, sublattice correlation, structure factors, spectral function |
| Example | quadratic boson Hamiltonian worked example |
| Appendix: Notation | notation and symbol table |
| Appendix: Luttinger-Tisza | future/appendix material |
| Appendix: Para-unitarity | proof material |
| Appendix: Thermodynamic derivations | derivation details |
| Appendix: Topological quantities | skyrmion number, Chern number, thermal Hall |

## 주요 차이

- 기존 Markdown의 `01_spin_wave_theory_intro.md`는 restructured 기준의
  `From Spins to Bosons`와 `Diagonalization`을 한 파일에 섞어 담고 있다.
- 기존 `02_physical_quantities.md`는 독립 section으로 유지한다.
- 기존 `05_topology.md`는 appendix로 이식한다.
- `notation.md`는 LSWT appendix로 이식한다.

## Common 후보로 보이는 항목

LSWT 정리 중 다음 항목은 나중에 `common/`으로 추출될 가능성이 높다.

| 후보 | 이유 | 지금 처리 |
|---|---|---|
| site index convention | MC/TN/NQS도 physical site와 basis/site indexing을 공유해야 함 | LSWT 문서에서 먼저 명확히 쓴다 |
| link/coupling counting | Hamiltonian 정의와 solver별 energy 계산에 공통으로 필요 | LSWT Hamiltonian 정리 중 기록 |
| displacement convention | real-space vs fractional coordinate가 여러 solver에 영향 | lattice 문서 정리 중 기록 |
| magnetic sublattice convention | commensurate/incommensurate, BZ folding, solver input에 직접 영향 | LSWT momentum section에서 우선 정리 |
| Brillouin zone convention | LSWT band/BZ뿐 아니라 MC/TN finite-size convention에도 연결됨 | LSWT에서 crystallographic vs magnetic BZ를 먼저 명확히 함 |

## Audit 결론

현재 Markdown 문서는 내용 자산으로는 충분하지만 정본 구조로는 부족하다.
다음 작업은 기존 파일을 바로 수정하는 것이 아니라, LSWT 정본 파일 구조를
먼저 확정한 뒤 필요한 내용을 옮기는 것이다.
