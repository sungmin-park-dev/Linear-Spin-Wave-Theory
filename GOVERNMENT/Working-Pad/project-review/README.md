# Project review files

Claude 프로젝트 공유 폴더(`/mnt/project-files/`)에만 있던 2026-10-01–02 검토·계획 문서를 저장소로 옮긴 사본이다. research-workspace 앱(연구 작업대)이 `workbench/research.yaml`의 `sources.materials`로 이 폴더를 자료로 보여 준다.

- 과정 기록이며 정본이 아니다. NBCP 내용 원본은 `docs/nbcp/main.tex`, LSWT 이론 정본은 `docs/lswt/` Markdown이다.
- 문서 안의 `/mnt/project-files/...` 경로는 원래 위치다. 옮긴 사본의 위치는 아래 표를 따른다.

| 파일 | 내용 | 원래 위치 |
|---|---|---|
| `2026-10-01-park2026-gap-analysis.md` | arXiv:2601.20963v1의 빈틈과 이론 문제 분석 (10/2 vortex 쌍 갱신 포함) | `nbcp-review/` |
| `2026-10-01-nbcp-physics-review.md` | NBCP 연구 문서 물리 점검 | `nbcp-review/` |
| `2026-10-01-nbcp-harmonic-phase-diagram.png` | 조화 차수 위상 그림 (데이터·스크립트는 `data-space/verification/261001-nbcp-harmonic-phase-diagram/`) | `nbcp-review/phase-diagram/` |
| `2026-10-01-nbcp-skx-competition.md` | 4-site skyrmion crystal 경쟁 결과 | `roadmap/` |
| `2026-10-01-general-spin-solver-roadmap.md` | 범용 스핀 솔버 로드맵 | `roadmap/` |
| `2026-10-01-goal-gap-assessment.md` | 두 프로젝트 목표 대비 현황 | `roadmap/` |
| `2026-10-01-lswt-drafts-review.pdf` | LSWT draft 검토용 PDF (파생물; 원본은 `docs/lswt/`, 검토 가이드는 `issue-notes/open/261001-lswt-draft-physics-review-guide.md`) | `lswt-review/` |
| `2026-10-02-batch-review-guide.md` | 사용자 일괄 검토 안내 (A1–A6, B1–B7, C1–C3) | `review/` |
| `2026-10-02-cutoff-resolution.md` | 유한온도 matching cutoff 해결 (NBCP 7장) | `nbcp-thermal/cutoff-split/` |
| `2026-10-02-vortex-core.md` | Y 고전 vortex core. 읽기 3번(큰 fugacity로 낮은 MC T_BKT 설명)은 vortex-pair 요약에서 철회됨 | `nbcp-thermal/vortex-core/` |
| `2026-10-02-vortex-pair-summary.md` | Y 고전 vortex 쌍 (PR #30, NBCP 7.4절) | `nbcp-thermal/vortex-pair/` |
| `2026-10-02-one-over-S-squared.md` | pseudo-Goldstone gap의 1/S² 가능성 검토와 1-loop χ 크기 | `nbcp-thermal/` |
| `stage7-visualization-proposal.md` | Stage 7 시각화 설계 제안 (draft) | `task-queue/` |

공유 폴더에서 이미 저장소에 있던 파일(각도 자유에너지·고전 열적 scan README, stage7 그림, spiral memo, 그림 묶음 `data-space/gallery/`, 10/2 계산 기록 `data-space/verification/261002-*`)은 옮기지 않았다. 자화 곡선 비교 데이터(`roadmap/magnetization/`, 스크립트·json 60여 개)는 공유 폴더에 둔다. 아직 PR이 아닌 코드 patch(`roadmap/patches/`, `task-queue/*.patch`)는 해당 스레드가 PR로 올린다.
