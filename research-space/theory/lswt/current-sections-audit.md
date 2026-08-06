---
frontmatter-version: 1
title: Current LSWT Theory Documentation Audit
section: theory/lswt
status: in-review
last-edited-by: codex
created: 2026-06-03
updated: 2026-08-01
---

# Current LSWT Theory Documentation Audit

> Snapshot: 2026-08-01

이 문서는 `research-space/theory/lswt/`의 coverage, lifecycle, source review,
열린 물리 질문을 추적한다. 이 audit 갱신은 이론 식이나 코드가 검증되었다는
뜻이 아니다.

## Approved Source Authority Snapshot

- Primary evidence:
  `/Users/david/Downloads/Linear_Spin_Wave_Theory___Note.pdf`
- Editable transcription:
  `legacy/research-notes/lswt/note_lswt_reviewed.tex`
- Structural reference:
  `research-space/sources/lswt/note_lswt_restructured.tex`
- Unique canonical authoring surface: `research-space/theory/lswt/`의 이론
  내용 Markdown
- User-approved accepted canon: none
- Legacy converted sources: `research-space/theory/sections/`,
  `research-space/theory/notation.md`

이 source authority는 2026-08-01 사용자가 승인했다. 결정 원문은
`GOVERNMENT/Court-Precedents/2026-08-01-lswt-markdown-source-authority.md`에
있다. Routing 승인은 아래 draft 또는 skeleton의 이론 내용을 승인한 것이
아니다.

Navigation과 파일 역할 경계의 기준은 `map-lswt.md`다. 아래 inventory는
2026-07-31 시점의 coverage evidence이며 navigation 목록을 대체하지 않는다.

## Active Workspace Inventory

### Draft Content

| File | Coverage | Lifecycle | Next verification boundary |
|---|---|---|---|
| `foundations/lswt-overview.md` | LSWT scope와 계산 흐름 | `draft` | PDF introduction과 validity/limitations TODO의 범위 확인 |
| `foundations/notation-and-conventions.md` | site, link, displacement convention | `draft` | 전체 symbol table, index와 BZ convention 이식 |
| `foundations/bilinear-spin-hamiltonian.md` | bilinear Hamiltonian, exchange, Zeeman term | `draft` | on-site support와 code convention 대조 |
| `foundations/classical-order-and-local-frame.md` | classical direction, rotation, local basis | `draft` | complex-basis exchange tensor 정의 확인 |
| `derivation/holstein-primakoff-expansion.md` | Leading HP expansion과 linear-term condition | `draft` | exact HP, expansion hierarchy, Dyson-Maleev 범위 결정 |
| `derivation/real-space-boson-hamiltonian.md` | Quadratic \(H_2\) | `draft` | odd terms와 \(H_4\)의 문서 범위 결정 |
| `derivation/momentum-space-bdg-hamiltonian.md` | General Nambu/BdG form | `draft` | A1, A2, A5, A10 해결 전 explicit block 보류 |

### Draft Skeletons

| File | Intended coverage | Lifecycle |
|---|---|---|
| `derivation/paraunitary-diagonalization.md` | Bogoliubov transformation, Colpa, stability | `draft` skeleton |
| `observables/magnon-observables.md` | Spectrum, energy correction, occupation, correlation matrix | `draft` skeleton |
| `observables/thermodynamics.md` | Partition function, energy, entropy, specific heat | `draft` skeleton |
| `observables/spin-correlations.md` | Real-time and sublattice correlations | `draft` skeleton |
| `observables/structure-factor-and-spectral-function.md` | Static/dynamic structure factor and spectral function | `draft` skeleton |
| `observables/topological-magnon-quantities.md` | Skyrmion number, Chern number, thermal Hall | `draft` skeleton |
| `examples/worked-example.md` | Single-mode quadratic-boson example | `draft` skeleton |
| `appendices/luttinger-tisza-method.md` | Luttinger-Tisza method | `draft` skeleton; source TODO |
| `appendices/paraunitarity-proofs.md` | Paraunitarity proof material | `draft` skeleton |
| `appendices/thermodynamic-derivations.md` | Entropy and correlation-matrix derivations | `draft` skeleton |

현재 합계는 일부 본문이 작성된 draft 7개, skeleton 10개, accepted 0개다.

## Source and Legacy Retention

| Material | Current judgment |
|---|---|
| Primary PDF | 27 pages. Source review annotation 33개, unique review ID 31개를 포함 |
| `note_lswt_reviewed.tex` | Primary PDF의 section order와 review ID에 대응하는 전사 보조 자료 |
| `note_lswt_restructured.tex` | 구조 재편 참고 자료. 과거 editable-master 지정은 superseded됐고, Primary PDF와 section order가 다르며 현재 clean build가 완료되지 않음 |
| `note_lswt_restructured.pdf` | Primary PDF와 다른 generated snapshot |
| `theory/sections/*.md` | 상세 유도가 아직 남아 있으므로 source-only 상태로 보존 |
| `theory/notation.md` | Symbol 이식이 끝날 때까지 source-only 상태로 보존 |
| `theory/common/` | Cross-solver convention candidate workspace. Draft 2개, accepted 0개이며 scope decision은 open |

Legacy Markdown에는 오래된 표기, 변환 오류, 최신 source와 다른 정규화가
있다. 따라서 직접 정본으로 승격하지 않고 PDF와 TeX를 구간별로 대조한다.

## Review Ledger

상태 표기:

- `draft-routed`: 현재 draft가 issue를 명시적으로 반영하거나 보류했다.
- `open`: 물리 또는 notation 검증이 필요하다.
- `source-cleanup`: 원자료 문장·구조 정리 항목이며 이론 검증과 분리한다.

같은 ID가 반복된 A3와 B10 때문에 annotation은 33개이고 unique ID는 31개다.

| ID | Target | Status | Boundary |
|---|---|---|---|
| B8 | `foundations/lswt-overview.md` | `source-cleanup` | Abstract 문장 범위 |
| C22 | `foundations/bilinear-spin-hamiltonian.md` | `draft-routed` | `Eq.` 표기 |
| A4 | `foundations/notation-and-conventions.md` | `draft-routed` | Link sum과 ordered-pair sum |
| C23 | `foundations/classical-order-and-local-frame.md` | `source-cleanup` | Commented matrix 제거 |
| B17 | `derivation/holstein-primakoff-expansion.md` | `source-cleanup` | Quantum-fluctuation 표현 |
| A3 | Local-frame and real-space documents | `draft-routed` | \(\hat{\mathbf e}_I^0\) superscript |
| A8 | `derivation/real-space-boson-hamiltonian.md` | `draft-routed` | Rotated local field |
| A9 | `derivation/momentum-space-bdg-hamiltonian.md` | `draft-routed` | Bond displacement 정의 |
| A10 | `derivation/momentum-space-bdg-hamiltonian.md` | `open` | Same-sublattice term의 FBZ sum 조건 |
| A5 | `derivation/momentum-space-bdg-hamiltonian.md` | `open` | Constant와 trace factor |
| A11 | Momentum-space and magnon-observable documents | `open` | Constant term과 \(S(S+1)\) correction 근거 |
| A1 | `derivation/momentum-space-bdg-hamiltonian.md` | `open` | \(B_{\mathbf k}\) off-diagonal typo와 block 식 |
| B10 | Momentum-space source | `source-cleanup` | SJ/SP color annotation 제거 |
| A2 | `derivation/momentum-space-bdg-hamiltonian.md` | `open` | Identical-sublattice factor 2 |
| A12 | Diagonalization and magnon observables | `open` | Ground-state energy와 zero-point correction 명칭 |
| B18 | `derivation/paraunitary-diagonalization.md` | `open` | Positive-semidefinite Goldstone-mode caveat |
| B11 | `observables/magnon-observables.md` | `source-cleanup` | First-person convention |
| C14 | Magnon and response observables | `open` | \(S_k\), \(\bar S_k\), \(U^\beta\) notation |
| B20 | `observables/spin-correlations.md` | `source-cleanup` | Correlation introduction 문장 |
| C17 | Correlation and response documents | `source-cleanup` | Roadmap과 상세 정의 중복 |
| A13 | `observables/spin-correlations.md` | `open` | Discrete momentum delta |
| A14 | `observables/structure-factor-and-spectral-function.md` | `open` | Block별 time dependence |
| A15 | `observables/structure-factor-and-spectral-function.md` | `open` | Retarded function의 basis 범위 |
| C15 | `examples/worked-example.md` | `source-cleanup` | TOC subsection 처리 |
| B21 | `examples/worked-example.md` | `source-cleanup` | Number-expectation heading |
| C13 | `foundations/notation-and-conventions.md` | `open` | \(R_k^\alpha\), energy symbol 누락 |
| B19 | `observables/thermodynamics.md` | `source-cleanup` | Entropy 설명 |
| B9 | `observables/topological-magnon-quantities.md` | `source-cleanup` | Primary PDF에는 orphan text가 남고 restructured draft에서만 제거됨 |
| A6 | `observables/topological-magnon-quantities.md` | `open` | Physical bands와 \(2N\) BdG sum |
| A7 | `observables/topological-magnon-quantities.md` | `open` | Thermal-Hall band sum |
| A16 | `observables/topological-magnon-quantities.md` | `open` | Primary PDF는 \(\varepsilon_{n,k}\), restructured draft는 \(E_{\mathbf k,n}\)을 사용 |

## Source TODOs

| Source area | Current state |
|---|---|
| Introduction | Restructured TeX에 content TODO가 남아 있음 |
| Validity and limitations | Restructured TeX에 content TODO가 남아 있음 |
| Luttinger-Tisza method | Primary PDF와 reviewed TeX에는 section이 없고, restructured TeX에만 빈 TODO section이 있음 |

## Source Tooling Issues

- Primary PDF metadata의 title은 LSWT 문서 제목과 일치하지 않는다.
- Primary PDF와 repo의 `note_lswt_restructured.pdf`는 서로 다른 PDF이며
  대체 관계가 아니다.
- Restructured TeX는 `\much`가 이미 `\mu_{\rm ch}`를 포함한 상태에서
  sublattice 첨자를 다시 붙이는 부분 때문에 double-subscript 오류가 난다.
- Review 문구의 `e^{-\riEt}`는 `\riEt`라는 undefined control sequence로
  해석된다.
- \(\mu_{\rm ch}\)의 물리적 범위가 확정되지 않았으므로 이 audit에서는 TeX를
  수정하지 않는다.

## Open Physical Decisions

다음 항목은 현재 `Unknown`이며 사용자 확인 또는 별도 이론-코드 검증 전에는
정본 식으로 확정하지 않는다.

1. \(\mu_{\rm ch}\)가 global scalar인지 sublattice-dependent quantity인지
2. Local complex basis에서 \(\widetilde J^{\pm\pm}\)를 정의하는 정확한 변환
3. On-site anisotropy를 link convention과 분리된 support로 둘지
4. Site, unit-cell, sublattice, band index와 boson-operator 이름
5. Crystallographic BZ와 magnetic BZ의 관계
6. Ground-state energy, zero-point correction, constant/trace convention
7. Correlation과 structure factor에서 \(N\), \(L\), \(m_s\) normalization
8. Positive-semidefinite Goldstone mode를 Colpa 문서 범위에 포함할지
9. Physical \(N\) bands와 \(2N\) BdG space의 topology sum convention
10. \(H_4\), Dyson-Maleev, Luttinger-Tisza를 현재 정본화 범위에 포함할지

## Structural Gate Status

- [x] 5개 하위 폴더의 navigation map 추가
- [x] 상위 map의 audit 역할을 coverage·lifecycle snapshot으로 정리
- [x] 10개 skeleton의 `draft` frontmatter 통일
- [x] `theory/common/`을 provisional candidate workspace로 명시
- [x] Markdown 단일 정본과 PDF·TeX·HTML 역할에 대한 사용자 결정
- [ ] Notation contract에 필요한 사용자 결정

완료 표시는 source authority와 문서 구조를 확정했다는 뜻이다. 이론 식이나
코드가 검증되었다는 뜻이 아니다. 다음 작업은
`foundations/notation-and-conventions.md`의 notation contract이며, 이를
검토하기 전에는 관련 TeX, 이론 수식, 코드 또는 API를 수정하지 않는다.
