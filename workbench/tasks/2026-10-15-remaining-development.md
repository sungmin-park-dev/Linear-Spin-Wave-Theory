---
title: 남은 개발 정리 (2026-10-15)
topic: toolkit-development
priority: 2
created: '2026-10-15'
state: paused
task: 2026-10-15 기준으로 spin-toolkit에 남은 일을 공개 준비, 열린 검토, 고를 수 있는 추가 기능으로 나눠 적는다. 새 기능 개발은 맡긴 항목을 모두 마쳤다.
issues:
- text: 추가 기능의 우선순위는 정하지 않았다. 실제 계산에 필요해지면 고른다.
references:
- workbench/tasks/2026-10-15-release-0-2-0.md
- workbench/tasks/2026-08-02-thermal-hall-real-space-volume.md
- workbench/tasks/2026-08-09-lswt-documentation-audit.md
- workbench/tasks/2026-08-10-lswt-implementation-backlog.md
- docs/reviews/2026-10-15-spin-wave-package-comparison.md
---

> 멈춘 까닭: 남은 일은 사용자의 검토와 "공개" 결정, 또는 추가 기능 선택을 기다린다.

## 요약과 결론

맡긴 개발 항목은 모두 끝났고 관련 PR(#79, #81, #82, #71, #64, #83)은 병합됐다. 시각화는 그림 함수 14개(`spintoolkit.visualization`)로 공개 전 조건(2026-10-01 사용자 결정)을 채웠다. 남은 일은 다음 세 묶음이다.

### 1. 공개 전에 할 일

| 항목 | 상태 | 작업 파일 |
|---|---|---|
| 0.2.0 배포 정리 | GitHub에서 설치한다. PyPI·TestPyPI에는 올리지 않는다(2026-10-16 사용자 결정). 0.2.0, 태그 `v0.2.0`(2026-10-16) | `2026-10-15-release-0-2-0.md` |
| 이론 노트 12편, 개발 노트, 튜토리얼 | draft. 논문 수치와 대조한 검토를 거쳐야 accepted가 되고, 공개용 문서는 accepted 노트로만 만든다 (일괄 검토 24–26번) | `2026-10-13-batch-review.md` |

### 2. 열려 있는 검토

| 항목 | 남은 것 | 작업 파일 |
|---|---|---|
| 열 홀 | 이론 문서 A6/A7/A16 검토. A7의 Goldstone 띠 Chern 수는 PR #71로 유도했다 | `2026-08-02-thermal-hall-real-space-volume.md` |
| 이론 문서 점검 | 열린 물리 결정 9개, LT 진단의 1/4 파수 거짓 음성 수정 확인 | `2026-08-09-lswt-documentation-audit.md` |
| 잠정 승인 결정 D38–D40, D50–D60 | 계산에 쓰일 때 논문 수치와 대조 | `2026-10-13-batch-review.md` |

### 3. 고를 수 있는 추가 기능

다른 스핀파 패키지와의 비교(`docs/reviews/2026-10-15-spin-wave-package-comparison.md`)에서 나온 후보와 옛 목록을 모았다. 우선순위는 정하지 않았다.

| 후보 | 근거 |
|---|---|
| 모멘트 감소의 메시 수렴 개선 | Goldstone 근처 로그 수렴 때문에 192² 메시에서 정사각 AFM 0.1953 (SpinW 0.1966). 적응형 적분이나 해석적 처리 |
| 편극 중성자 | SpinW에 있다. 2D 범위 안에서 더할 수 있다 |
| 데이터 맞춤, 장비 분해능 | SpinW, Sunny에 있다. 지금은 가우스 넓힘만 있다 |
| CIF·공간군 입력, 장거리 쌍극자(Ewald), SU(N) | Sunny에 있다. 범위 결정(D48) 안에 둘지 먼저 정해야 한다 |
| real-space BdG 솔버 | 장기 과제 (`2026-08-10-lswt-implementation-backlog.md`) |

다중 Q 비정합 구조는 사용자가 "패스"했다. 3D는 범위 밖이다(2026-10-01).

추천 순서: 추가 기능은 실제로 쓸 계산이 생길 때 고른다.

2026-10-15 "추천대로 진행해"로 한 것: 튜토리얼 `docs/tutorials/02-own-model.md`에 단일 이온 관례 안내(magnopy처럼 보정 없는 HP와 맞추려면 A/(1 − 1/2S)를 넣는다. SpinW·Sunny의 관례는 확인하지 못해 "그 코드의 문서를 확인하라"로만 적음), 비교 보고서의 라이선스 표기(MIT), 일괄 검토 28번 기록, 배포 워크플로(2026-10-16에 업로드 없는 빌드 검사로 바꿈).

## 근거

- 작업 파일 `workbench/tasks/` 6개와 `main` 946d883(PR #83 병합)을 2026-10-15에 확인했다.
- 패키지 비교의 magnopy 교차검증은 `data-space/verification/261015-magnopy-comparison/`에서 다시 돌릴 수 있다.
