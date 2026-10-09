# Project review files

Claude 프로젝트 공유 폴더(`/mnt/project-files/`)에만 있던 2026-10-01–02 검토·계획 문서를 저장소로 옮긴 사본이다. research-workspace 앱(연구 작업대)이 `workbench/research.yaml`의 `sources.materials`로 이 폴더를 자료로 보여 준다.

- 과정 기록이며 정본이 아니다. NBCP 검토 기록은 2026-10-09 저장소 `nbcp-spin-supersolid`로 옮겼다. LSWT 이론 정본은 `docs/lswt/` Markdown이다.
- 문서 안의 `/mnt/project-files/...` 경로는 원래 위치다. 옮긴 사본의 위치는 아래 표를 따른다.

| 파일 | 내용 | 원래 위치 |
|---|---|---|
| `2026-10-01-general-spin-solver-roadmap.md` | 범용 스핀 솔버 로드맵 | `roadmap/` |
| `2026-10-01-goal-gap-assessment.md` | 두 프로젝트 목표 대비 현황 | `roadmap/` |
| `2026-10-01-lswt-drafts-review.pdf` | LSWT draft 검토용 PDF (파생물; 원본은 `docs/lswt/`, 검토 가이드는 `issue-notes/open/261001-lswt-draft-physics-review-guide.md`) | `lswt-review/` |
| `2026-10-02-batch-review-guide.md` | 사용자 일괄 검토 안내 (A1–A6, B1–B7, C1–C3) | `review/` |
| `stage7-visualization-proposal.md` | Stage 7 시각화 설계 제안 (draft) | `task-queue/` |

공유 폴더에서 이미 저장소에 있던 파일(각도 자유에너지·고전 열적 scan README, stage7 그림, spiral memo, 그림 묶음 `data-space/gallery/`, 10/2 계산 기록 `data-space/verification/261002-*`)은 옮기지 않았다. 자화 곡선 비교 데이터(`roadmap/magnetization/`, 스크립트·json 60여 개)는 공유 폴더에 둔다. 아직 PR이 아닌 코드 patch(`roadmap/patches/`, `task-queue/*.patch`)는 해당 스레드가 PR로 올린다.
