---
frontmatter-version: 1
title: Task Queue
section: working-pad
status: in-review
last-edited-by: claude
created: 2026-06-03
updated: 2026-10-05
---

# Task Queue

> 이 파일은 LSWT `GOVERNMENT/Working-Pad/`의 활성 작업 인덱스다.
> 상세 내용은 링크된 파일에 둔다.

## 현재 작업

> **2026-10-05:** NBCP 원고를 그날 상태로 보존하고 내용을 `workbench/notes/` 연구노트 8개로 복제했다. 이후 NBCP 내용의 원본은 연구노트다([결정](../Court-Precedents/2026-10-05-nbcp-research-notes.md)). 아래 기록의 "원본은 `docs/nbcp/main.tex`"는 그 전의 상태다.
>
> **2026-10-03 현재:** 구현은 끝났고 대부분 항목이 2026-10-02 일괄 검토 안내(`/mnt/project-files/review/2026-10-02-batch-review-guide.md`)의 사용자 검토를 기다린다. 검토 없이 진행할 수 있는 연구 작업은 4번(wall–vortex 결합, 비선형 스핀파)이다.
>
> **2026-09-30 재개(사용자 결정):** 같은 날의 전체 멈춤을 풀었다. 하루 연구 시간 5시간을 Emergence EB 대표작 2시간, TN+NQS 1.5시간, LSWT 1.5시간으로 나눈다. NBCP spin supersolid의 열린 유도·계산은 research-workspace 앱에 등록해 `workbench/blocks/`의 블록으로 진행한다(Y vortex·wall → 유한온도 matching → 유한 크기 열적 검증, quantum gradient, V clock·pinning, 전역 경쟁). 블록은 과정 기록이고 NBCP 내용 원본은 `docs/nbcp/main.tex`다. 아래 순위는 그대로 둔다.

| 순위 | 유형 | 내용 | 상태 | 파일 |
|---|---|---|---|---|
| 1 | issue | `docs/lswt/` LSWT 이론 문서 — 구현·검증된 부분부터 한 문서씩 작성; 기존 다음 묶음: local circular component 및 real-space H2 | in-review; 2026-10-01 독립 검산 오류 14건 수정, 10개 draft 작성 완료 — 사용자 물리·수학 검토 대기(일괄 검토 C1, 가이드 `issue-notes/open/261001-lswt-draft-physics-review-guide.md`); LT 진단 1/4 파수 거짓 음성 수정은 사용자 확인 대기 | `issue-notes/open/260809-lswt-documentation-audit.md` |
| 2 | handoff | 솔버 seam 스파이크 (ED/TN/NQS로 XXZ 풀어 실측 검증) | in-review; ED·TeNPy DMRG·NetKet이 같은 토러스 전개에서 1e-14 일치, RBM VMC 상대 오차 1e-3–8e-3, TN 필드 규약 `-h_i . S_i` 사용자 확인(2026-10-01) — 일괄 검토 C2에서 closed 이동 여부 확인 대기 | `handoff/open/260607-solver-seam-spike.md` |
| 3 | issue | Thermal Hall 면적·단위·반환 기준 | in-review; 구현은 toolkit 5a–5d(D29)로 해결; full-position Bloch 규약만 물리적임을 닫힌 식으로 정리(PR #14, 일괄 검토 A3); NBCP 최근접 XXZ는 대칭으로 0, J_PD·J_Γ가 있어야 0이 아님; pseudo-Goldstone 갭은 Δ²=C_φ/χ_z로 정리(PR #20·#22, 1-loop로 0.1–0.4% 확인); 남은 것: J_PD·J_Γ 크기, 이론 문서 A6/A7/A16 검토(A7 Goldstone 경우만 open) | `issue-notes/open/260802-topology-thermal-hall-real-space-volume-bug.md` |
| 4 | issue | NBCP 연구노트와 arXiv 열적 주장 검토 | in-review; density wall·vortex core(PR #29)·vortex pair(PR #30) 고전 계산 완료, 유한온도 cutoff matching 정리(PR #29); vortex–벽 상호작용·조화 벽 자유에너지 고전 계산 완료(2026-10-05, `data-space/verification/261005-y-vortex-wall/`); V 강성·V matching(Λ ≈ 0.1)·Debye–Waller 인자 완료(2026-10-05, `data-space/verification/261005-v-matching-debye-waller/`); V 위상 모형 유한온도 순서: J_Γ/J_PD ≳ 0.01에서 대수 구간이 닫힘(2026-10-05, `data-space/verification/261005-v-clock-pinning-mc/`); 남은 것: 양자 S=1/2 유한온도·core(사용자 장비), 1/S에 기대지 않는 S=1/2 각도 퍼텐셜(1/S² 갭은 PR #37·#39에서 계산했고 S=1/2에서 통제되지 않음), Fig. 4(b) 출처 확인(사용자) | `issue-notes/open/260810-pseudo-goldstone-gap.md` |
| 5 | issue | NBCP band plot와 남은 LSWT 구현 backlog | draft; NBCP 횡자기장 편극상과 3부분격자 상(0.7·1.0·1.4 T) 밴드 계산·검증 완료(2026-10-01, `data-space/verification/261001-nbcp-*-bands/`); 관측량 묶음(D41 중성자, D45 DOS·S(Q)·M(h)) 구현(PR #23·#25·#26); 남은 것: 측정 데이터 대조(데이터 필요), 공개 배포는 12번 | `issue-notes/open/260810-lswt-implementation-backlog.md` |
| 6 | idea | 공통 SpinModel IR 표현법 | draft | `idea-proposals/2026-06-04-general-spin-model-ir.md` |
| 7 | idea | Project Knowledge Philosophy 사본 | draft | `idea-proposals/2026-05-30-project-knowledge-philosophy.md` |
| 8 | issue | NBCP 문서 구조와 코드 물리 구현 검토 | in-review; 9장·4부록 LaTeX 원본 전환 및 claim-to-code 대응 작성; 변위 방향은 D13으로 해결(2026-09-30), 원고 문장 수정 대기; 다음: 독립 bond-count 검토 | `issue-notes/open/260918-nbcp-physics-code-review.md` |
| 9 | idea | 모델 간 공통 전달 규약 — 필드·단위·검증·NBCP 변위 대응 | in-review; D01–D45 결정 기록, 사용자 검토는 일괄 검토 A·B | `idea-proposals/2026-09-23-spin-model-transfer-contract.md` |
| 10 | development | 2D Spin-System Toolkit 개발 설계와 기능별 폴더 구성 | 0–7단계 구현·검증(7단계 시각화: 밴드·스핀 배치·중성자·위상·열역학 그림, D44); 영어 튜토리얼 5편(PR #28); 개발 Beamer(D30–D45 포함) 사용자 검토 대기(일괄 검토 C3) | `../../docs/development/README.md` |
| 11 | idea | 단일 Q 나선의 회전틀 LSWT (`IncommensurateStructure`, D34) | in-review; 구현·검증(테스트 21개, 해석식·초격자 대조), 사용자 물리·수학 검토 대기(일괄 검토 B1) | `idea-proposals/2026-10-01-spiral-rotating-frame-lswt.md` |
| 12 | release | 공개 배포 준비 (0.2.0, TestPyPI → PyPI) | 일괄 검토 대기(`/mnt/project-files/review/2026-10-02-batch-review-guide.md`); 검토 반영 뒤 TestPyPI, PyPI는 사용자가 "공개"라고 쓸 때만 진행 | `../../docs/tutorials/README.md` |

## 동기화 규칙

- `handoff/open/`, `issue-notes/open/`, `vault-staging/`, `idea-proposals/`에 활성 항목을 추가하면 이 표에도 등록한다.
- 항목이 완료되거나 닫히면 이 표에서 제거하고 해당 파일을 `closed/`로 옮기거나 정본 위치로 승격한다.
- 이 표는 상세 내용을 반복하지 않는다. 상세 맥락은 대상 파일에 둔다.

## 읽기 순서

1. 이 파일
2. `handoff/open/`
3. `issue-notes/open/`
4. `idea-proposals/`
5. `vault-staging/`
