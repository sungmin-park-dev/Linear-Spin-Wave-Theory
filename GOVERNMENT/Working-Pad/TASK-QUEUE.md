---
frontmatter-version: 1
title: Task Queue
section: working-pad
status: in-review
last-edited-by: claude
created: 2026-06-03
updated: 2026-10-10
---

# Task Queue

> 이 파일은 LSWT `GOVERNMENT/Working-Pad/`의 활성 작업 인덱스다.
> 상세 내용은 링크된 파일에 둔다.

## 현재 작업

> **2026-10-10 현재 (다른 세션이 이어받을 때 여기부터):**
> - 범위: `spin-toolkit`은 2D 스핀 모형의 반고전 툴킷, LSWT 중심이다([D48 결정](../Court-Precedents/2026-10-10-package-scope-semiclassical.md)). 양자 다체 계산(DMRG·QMC·TN)은 범위 밖, ED는 검증용.
> - 정본: LSWT 이론은 `workbench/notes/lswt-*` 연구노트 7개, 개발 설계·결정은 `dev-design`·`dev-verification`·`dev-decisions`(D01–D49)다([2026-10-08 결정](../Court-Precedents/2026-10-08-lswt-development-research-notes.md)). `docs/lswt/`·`docs/development/`는 보존본이라 고치지 않는다.
> - 개념노트: 공유 라이브러리 `research-library`에 이 프로젝트용 10개(`workbench/research.yaml`의 `concepts:`)가 있다. 모두 draft이고, 할 일은 각 `concepts/<id>.memo.md`에 있다.
> - 사용자 검토: 구현·문서 대부분이 일괄 검토(`project-review/2026-10-02-batch-review-guide.md`)를 기다린다. 사용자가 "잠시 보류"라고 했으니 검토를 재촉하지 않고, 검토 없이 할 수 있는 아래 일을 먼저 한다.
> - 합치기: PR은 CI가 초록이면 사용자가 "병합"이라고 쓸 때 합친다.
>
> **다음 작업 (검토 없이 할 수 있는 것, 추천 순서):**
> 1. ~~`observables/thermal.py` 자화 보정~~ — 2026-10-10 구현(D49, 사용자 "추천대로 진행해"): `magnetization_curve(..., temperatures=)`가 M(h,t) = −∂F/∂h를 내고 t = 0에서 `harmonic`과 같다. `ThermalResult.magnetization`은 기울기 각 보정 없는 모멘트 합으로 두고 문서에 밝혔다. 검증은 `dev-verification` D49 절(독립 닫힌 식 검산 포함), 유도는 `lswt-magnon-thermodynamics`의 Magnetization at Nonzero Temperature 절(draft); D49 열린 항목 중 기준 상태(E_cl 최소화)와 유효 범위 표시(`beyond_lswt`, `gapless`)는 사용자 결정으로 닫았고, 2D Goldstone에서 유한한 M은 설명을 붙여 판정 대기. PR #57.
> 2. ~~개념노트 메모 할 일 중 문헌으로 확인할 수 있는 것~~ — 2026-10-10 (research-library PR #17): bib 7개 중 6개 Crossref 일치, `landauTheory1935`는 DOI가 없어 재수록본으로 저자·제목만 확인. 정사각 Z_c = 1 + C/2S (C = 0.1579)를 적분과 `solve_nlswt`로 확인해 본문에 넣음. 남은 것(메모): Oguchi 원문의 식 위치, O(S⁰) 에너지 E/N = −2J(S + C/2)²의 출처, LL 1935 원 학술지 권·쪽.
> 3. 공개 배포(12번): 일괄 검토가 끝난 뒤 TestPyPI, PyPI는 사용자가 "공개"라고 쓸 때만.
>
> **2026-10-09:** NBCP 연구(그때 순위 4, 8과 NBCP 연구노트·보조 노트)는 별도 저장소 `nbcp-spin-supersolid`로 옮겼다([결정](../Court-Precedents/2026-10-09-nbcp-repository-split.md)). 그 작업은 그 저장소의 TASK-QUEUE에서 관리한다.
>
> **2026-10-03:** 구현은 끝났고 대부분 항목이 2026-10-02 일괄 검토 안내의 사용자 검토를 기다린다. (그때의 4번 연구 작업은 NBCP 저장소로 옮겼다.)
>
> **2026-09-30 재개(사용자 결정):** 같은 날의 전체 멈춤을 풀었다. 하루 연구 시간 5시간을 Emergence EB 대표작 2시간, TN+NQS 1.5시간, LSWT 1.5시간으로 나눈다.

| 순위 | 유형 | 내용 | 상태 | 파일 |
|---|---|---|---|---|
| 1 | issue | LSWT 이론 문서 — 2026-10-08부터 `workbench/notes/lswt-*` 연구노트 7개가 정본(`docs/lswt/`는 보존본); 기존 다음 묶음: local circular component 및 real-space H2 | in-review; 2026-10-01 독립 검산 오류 14건 수정, 10개 draft 작성 완료 — 사용자 물리·수학 검토 대기(일괄 검토 C1, 가이드 `issue-notes/open/261001-lswt-draft-physics-review-guide.md`); LT 진단 1/4 파수 거짓 음성 수정은 사용자 확인 대기 | `issue-notes/open/260809-lswt-documentation-audit.md` |
| 2 | handoff | 솔버 seam 스파이크 (ED/TN/NQS로 XXZ 풀어 실측 검증) | 범위 밖(D48, 2026-10-10): 더 진행하지 않고 결과만 보존; 이전 기록: ED·TeNPy DMRG·NetKet이 같은 토러스 전개에서 1e-14 일치, RBM VMC 상대 오차 1e-3–8e-3, TN 필드 규약 `-h_i . S_i` 사용자 확인(2026-10-01) — 일괄 검토 C2에서 closed 이동 여부 확인 대기 | `handoff/open/260607-solver-seam-spike.md` |
| 3 | issue | Thermal Hall 면적·단위·반환 기준 | in-review; 구현은 toolkit 5a–5d(D29)로 해결; full-position Bloch 규약만 물리적임을 닫힌 식으로 정리(PR #14, 일괄 검토 A3); NBCP 최근접 XXZ는 대칭으로 0, J_PD·J_Γ가 있어야 0이 아님; pseudo-Goldstone 갭은 Δ²=C_φ/χ_z로 정리(PR #20·#22, 1-loop로 0.1–0.4% 확인); 남은 것: J_PD·J_Γ 크기, 이론 문서 A6/A7/A16 검토(A7 Goldstone 경우만 open) | `issue-notes/open/260802-topology-thermal-hall-real-space-volume-bug.md` |
| 5 | issue | NBCP band plot와 남은 LSWT 구현 backlog | draft; NBCP 횡자기장 편극상과 3부분격자 상(0.7·1.0·1.4 T) 밴드 계산·검증 완료(2026-10-01, `data-space/verification/261001-nbcp-*-bands/`); 관측량 묶음(D41 중성자, D45 DOS·S(Q)·M(h)) 구현(PR #23·#25·#26); 남은 것: 측정 데이터 대조(데이터 필요), 공개 배포는 12번 | `issue-notes/open/260810-lswt-implementation-backlog.md` |
| 6 | idea | 공통 SpinModel IR 표현법 | draft | `idea-proposals/2026-06-04-general-spin-model-ir.md` |
| 7 | idea | Project Knowledge Philosophy 사본 | draft | `idea-proposals/2026-05-30-project-knowledge-philosophy.md` |
| 9 | idea | 모델 간 공통 전달 규약 — 필드·단위·검증·NBCP 변위 대응 | in-review; D01–D49 결정 기록(`dev-decisions` 연구노트), 사용자 검토는 일괄 검토 A·B | `idea-proposals/2026-09-23-spin-model-transfer-contract.md` |
| 10 | development | 2D Spin-System Toolkit 개발 설계와 기능별 폴더 구성 | 0–7단계 구현·검증(7단계 시각화: 밴드·스핀 배치·중성자·위상·열역학 그림, D44); 영어 튜토리얼 5편(PR #28); 개발 설계·결정은 `dev-*` 연구노트(D01–D49)가 정본, 사용자 검토 대기(일괄 검토 C3); 범위는 D48(반고전 툴킷) | `../../docs/development/README.md` |
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
