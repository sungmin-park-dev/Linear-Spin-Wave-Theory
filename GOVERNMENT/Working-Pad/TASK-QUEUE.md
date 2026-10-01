---
frontmatter-version: 1
title: Task Queue
section: working-pad
status: in-review
last-edited-by: claude
created: 2026-06-03
updated: 2026-10-01
---

# Task Queue

> 이 파일은 LSWT `GOVERNMENT/Working-Pad/`의 활성 작업 인덱스다.
> 상세 내용은 링크된 파일에 둔다.

## 현재 작업

> **2026-09-30 재개(사용자 결정):** 같은 날의 전체 멈춤을 풀었다. 하루 연구 시간 5시간을 Emergence EB 대표작 2시간, TN+NQS 1.5시간, LSWT 1.5시간으로 나눈다. NBCP spin supersolid의 열린 유도·계산은 research-workspace 앱에 등록해 `workbench/blocks/`의 블록으로 진행한다(Y vortex·wall → 유한온도 matching → 유한 크기 열적 검증, quantum gradient, V clock·pinning, 전역 경쟁). 블록은 과정 기록이고 NBCP 내용 원본은 `docs/nbcp/main.tex`다. 아래 순위는 그대로 둔다.

| 순위 | 유형 | 내용 | 상태 | 파일 |
|---|---|---|---|---|
| 1 | issue | `docs/lswt/` LSWT 이론 문서 — 진행(C): 구현·검증된 부분부터 한 문서씩 작성(topology → 대각화 → 열역학 → structure factor → LT); 기존 다음 묶음: local circular component 및 real-space H2 | in-review; 17개 문서 일괄 문체 교정 반영; 2026-09-30 topology·대각화·열역학·상관함수·structure factor·LT·magnon-observables draft 작성·사용자 물리·수학 검토 대기, 남은 skeleton 3개; LT 진단 1/4 파수 거짓 음성은 수정·검증 완료(Beamer 검토본 43), 사용자 확인 대기 | `issue-notes/open/260809-lswt-documentation-audit.md` |
| 2 | handoff | 솔버 seam 스파이크 (ED/TN/NQS로 XXZ 풀어 실측 검증) | pending — TN-Study에 findings draft v1(1D, Heisenberg점) 있음, 2D 4×4 단계 미완료라 LSWT 승격 전 | `handoff/open/260607-solver-seam-spike.md` |
| 3 | issue | Thermal Hall 면적·단위·반환 기준 | in-review; 구현 항목은 toolkit 5a–5d(D29)로 해결(`docs/development/verification/stage5*`); 남은 것: NBCP 적용 조건, 이론 문서 A6/A7/A16 검토(2026-09-30 topology draft에 반영; A7 −π²/3는 양정치에서 동치로 정리, Goldstone 경우만 open) | `issue-notes/open/260802-topology-thermal-hall-real-space-volume-bug.md` |
| 4 | issue | NBCP 연구노트와 arXiv 열적 주장 검토 | in-review; Y angular·smooth-wave와 density wall 164개 local minima 대조 완료; 벽 폭 3–4a, 장력 양수·크기 수렴 및 metastability 확인; 다음: vortex core·wall 결합, 이후 thermal 검증; quantum/thermal matching 미완료 | `issue-notes/open/260810-pseudo-goldstone-gap.md` |
| 5 | issue | NBCP band plot와 남은 LSWT 구현 backlog | draft; H·Colpa 검증 끝(toolkit 2–4a), band plot 작성·검토와 시각화 포팅·공개 배포 등 남음 | `issue-notes/open/260810-lswt-implementation-backlog.md` |
| 6 | idea | 공통 SpinModel IR 표현법 | draft | `idea-proposals/2026-06-04-general-spin-model-ir.md` |
| 7 | idea | Project Knowledge Philosophy 사본 | draft | `idea-proposals/2026-05-30-project-knowledge-philosophy.md` |
| 8 | issue | NBCP 문서 구조와 코드 물리 구현 검토 | in-review; 9장·4부록 LaTeX 원본 전환 및 claim-to-code 대응 작성; 변위 방향은 D13으로 해결(2026-09-30), 원고 문장 수정 대기; 다음: 독립 bond-count 검토 | `issue-notes/open/260918-nbcp-physics-code-review.md` |
| 9 | idea | 모델 간 공통 전달 규약 — 필드·단위·검증·NBCP 변위 대응 | in-review; D01–D29 결정과 0–5단계 구현 반영 | `idea-proposals/2026-09-23-spin-model-transfer-contract.md` |
| 10 | development | 2D Spin-System Toolkit 개발 설계와 기능별 폴더 구성 | 0–6단계(2b, 4d, 5a–5d, 6a–6c 포함) 구현·검증, 7단계(시각화) 밴드·스핀 배치 그림 구현(2026-10-01), NBCP 밴드 그림 다음; Beamer 검토본 42 사용자 검토 대기 | `../../docs/development/README.md` |

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
