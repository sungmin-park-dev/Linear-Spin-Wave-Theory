---
frontmatter-version: 1
title: Theory Notes & Code Verification Plan
section: procedures
status: needs-review
last-edited-by: codex
created: 2026-05-31
updated: 2026-06-03
---

# Theory Notes & Code Verification Plan

> 작성일: 2026-05-31  
> 목적: 이론 노트 정리 + 코드 구현 일치 검증을 동시에 진행하기 위한 계획서
> 상태: 검토 필요. 현재 `GOVERNMENT/Agents-Bylaws/procedures/`에 있으나, 현 코드·문서 구조와 대조한 뒤 실행 기준으로 사용한다.

---

## 1. 배경 및 목표

### 현재 문제
- LaTeX 노트가 세 버전(`note.tex`, `note_lswt_reviewed.tex`, `note_lswt_restructured.tex`)으로 분산
- 마크다운 섹션(`research-space/theory/sections/`)은 구버전 구조 기반 — 최신 LaTeX와 불일치
- 코드(`code-space/lswt/`)가 이론과 실제로 일치하는지 체계적으로 검증된 적 없음

### 목표
1. `note_lswt_restructured.tex`를 단일 LaTeX 기준 소스로 확정
2. 마크다운 섹션을 restructured 구조에 맞게 재편
3. 각 섹션 정리 시 대응하는 코드 모듈을 함께 대조하여 불일치 기록

---

## 2. 기준 소스 확정

| 파일 | 상태 |
|---|---|
| `research-space/notes/note_lswt_restructured.tex` | ✅ **Master** (가장 최신, 최고 구조) |
| `research-space/notes/note_lswt_reviewed.tex` | 🗃️ archive로 이동 |
| `research-space/notes/note.tex` | 🗃️ archive로 이동 |
| `research-space/notes/hamiltonian_convention.tex` | 🗃️ archive로 이동 (내용 이미 흡수됨) |

**즉시 실행**: `research-space/notes/archive/` 디렉토리 생성 후 위 파일들 이동.

---

## 3. 섹션 구조 재편

### restructured.tex 기준 새 섹션 구조

```
research-space/theory/sections/
├── 00_introduction.md          ← 신규 (TODO: 내용 미작성)
├── 01_from_spins_to_bosons.md  ← 기존 01 재편 (Classical ground state 추가)
├── 02_diagonalization.md       ← 기존 01 후반부 분리 + 신규 내용 추가
├── 03_thermodynamics.md        ← 기존 03 그대로 (✅ 일치)
├── 04_correlations.md          ← 기존 04 그대로 (✅ 일치)
├── 05_worked_example.md        ← 기존 06 → 번호 변경
└── appendices/
    ├── A_notation.md           ← 기존 notation.md 이동
    ├── B_luttinger_tisza.md    ← 신규
    ├── C_paraunitary.md        ← 신규
    ├── D_thermo_derivations.md ← 신규
    └── E_topology.md           ← 기존 05_topology.md 이동 (본문→부록)
```

### 기존 파일 → 새 파일 매핑

| 기존 마크다운 | 처리 |
|---|---|
| `01_spin_wave_theory_intro.md` | → `01_from_spins_to_bosons.md` (재편) |
| `02_physical_quantities.md` | 🗑️ 삭제 (restructured에서 제거된 섹션) |
| `03_thermodynamics.md` | → `03_thermodynamics.md` (그대로) |
| `04_correlations.md` | → `04_correlations.md` (그대로) |
| `05_topology.md` | → `appendices/E_topology.md` (부록으로 이동) |
| `06_worked_example.md` | → `05_worked_example.md` (번호 변경) |
| `notation.md` | → `appendices/A_notation.md` |

---

## 4. 섹션별 작업 계획

각 섹션마다 다음 순서로 진행:
1. **LaTeX 읽기** — restructured.tex 해당 섹션 숙지
2. **마크다운 업데이트** — 누락 내용 추가, 구조 정렬
3. **코드 대조** — 대응 모듈과 수식/알고리즘 비교
4. **불일치 기록** — `## ⚠️ Code Discrepancies` 섹션에 기재

### 섹션별 우선순위 및 대응 코드

| 우선순위 | 섹션 | 대응 코드 모듈 | 특이사항 |
|---|---|---|---|
| 1 | §2.1 Classical ground state & local frame | `solvers/hamiltonian.py` → `_classical_spin_rotation_matrix()`, `get_rmat_dict()` | 마크다운에 아예 없는 섹션 |
| 2 | §2.2 Holstein-Primakoff | `solvers/hamiltonian.py` → `get_couplings()` | 기존 01에 있으나 구조 재편 필요 |
| 3 | §2.3 Bosonic Hamiltonian | `solvers/hamiltonian.py` → `Quadratic_Bose_Hamiltonian()` | **알려진 버그: B/B† swap** |
| 4 | §2.4 Momentum space | `solvers/hamiltonian.py` → `Quadratic_Bose_Hamiltonian()` | |
| 5 | §3 Diagonalization | `core/diagonalization.py` → `Diagonalizer` | Colpa 알고리즘 전체 |
| 6 | §3.2 Physical quantities after diag | `solvers/hamiltonian.py` → `compute_quantum_energy()` | 신규 섹션 |
| 7 | §4 Thermodynamics | `observables/thermodynamics.py` | 기존 마크다운과 거의 일치 예상 |
| 8 | §5 Correlations | `observables/correlations.py` | 기존 마크다운과 거의 일치 예상 |
| 9 | Appendix E: Topology | `observables/topology.py` | **알려진 버그: real_space_volume** |
| 10 | Appendices B, C, D | 수학적 보조 증명 | 코드 대조 불필요 |

---

## 5. 알려진 버그 처리 계획

섹션 정리 중 해당 부분에 도달하면 버그 수정을 함께 진행.

### Bug 1: B/B† block swap (`solvers/hamiltonian.py` L278-284)
- **발견 섹션**: §2.3 Bosonic Hamiltonian construction
- **증상**: B_k와 B†_k 블록이 뒤바뀜. Γ perturbation이 있는 비대칭 B_k에서 틀린 결과.
- **수정 방법**: SM v5 수정 코드로 교체
- **검증**: `doc-space/examples/nbcp_hamiltonian_check.py`의 bosonic constraint 테스트 통과 확인

### Bug 2: Thermal Hall `real_space_volume` (`observables/topology.py`)
- **발견 섹션**: Appendix E Topology
- **증상**: `self.Ns` 곱셈 누락 + 단위 변환 계수 `1e-12` → `1e-22` 수정 필요
- **수정 방법**: 수식 재도출 후 수정

---

## 6. 섹션 마크다운 파일 형식

각 섹션 파일의 표준 구조:

```markdown
# 섹션 제목

> **Source**: `research-space/notes/note_lswt_restructured.tex` §X.X  
> **Code**: `code-space/lswt/모듈명.py`  
> **Status**: 🔴 Draft / 🟡 Review / 🟢 Verified

---

## 이론 내용

(수식 및 설명)

---

## Code Correspondence

| 이론 | 코드 | 위치 |
|---|---|---|
| 수식/개념 | 함수명 | 파일:줄번호 |

---

## ⚠️ Discrepancies

(불일치 발견 시 기록)

---

## ✅ Verification

(검증 완료 시 결과 기록)
```

---

## 7. 완료 기준

- [ ] `research-space/notes/archive/` 정리 완료
- [ ] 마크다운 섹션 구조 재편 완료
- [ ] 우선순위 1-6 섹션 정리 + 코드 대조 완료
- [ ] Bug 1 (B/B† swap) 수정 + 검증 완료
- [ ] 우선순위 7-9 섹션 정리 + 코드 대조 완료
- [ ] Bug 2 (Thermal Hall) 수정 + 검증 완료
- [ ] 부록 B, C, D 마크다운 작성 완료
- [ ] `doc-space/examples/nbcp_hamiltonian_check.py` 전체 통과

---

## 8. 다음 단계 (이 계획 완료 후)

- 밴드 플롯 예제 구현
- `LSWTHamiltonian` → `SpinSystem` 직접 수용 (legacy dict 제거)
- pytest 테스트 작성
