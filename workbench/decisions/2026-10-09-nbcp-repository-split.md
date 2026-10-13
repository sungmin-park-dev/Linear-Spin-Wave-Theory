---
title: NBCP 연구를 스핀파 패키지 저장소에서 분리
status: accepted
decided: 2026-10-09
reviewed-by: user
---

# NBCP 연구를 스핀파 패키지 저장소에서 분리 (2026-10-09)

## 결정

연구 작업대 앱은 저장소 하나를 프로젝트 하나로 읽는다. 스핀파 이론·패키지와 NBCP 연구를 다른 프로젝트로 보기 위해, NBCP 연구를 새 저장소 `sungmin-park-dev/nbcp-spin-supersolid`로 옮긴다. 이 저장소는 스핀파 이론과 패키지 `spin-toolkit` 프로젝트(앱 제목 "Spin Wave Theory")가 된다. 사용자가 결정 카드에서 "프로젝트 둘"과 "모두 옮기기"를 골랐다(2026-10-09).

## 옮긴 것 (새 저장소로)

- 연구노트 8개, 보조 노트 8개, 일지의 NBCP 항목, 보존 원고 `docs/nbcp/`
- NBCP 계산·검증 스크립트 `examples/nbcp_*`와 `examples/pseudo_goldstone_*`, 그 결과 `data-space/verification/`의 NBCP 폴더
- NBCP 연구 테스트 `code-space/tests/test_models/test_nbcp_*.py`
- NBCP 결정 2개, 검토·이슈·인계 기록
- 파일 경로는 그대로 두었다. 노트와 기록이 가리키는 경로가 이 저장소 안에서 그대로 맞는다. 옮긴 파일의 git 이력도 함께 가져왔다.

## 두 저장소에 함께 있는 것

- `model/nbcp/`: NBCP 모형 정의. 새 저장소의 사본이 연구용 원본이고, 이 저장소의 사본은 공개된 모형으로 패키지 회귀 테스트(`test_nbcp.py` 등)가 쓰는 고정본이다.
- `data-space/verification/260912-pseudo-goldstone/`: 패키지 회귀 테스트도 같은 수치를 읽는다.
- `legacy/modules/`, `legacy/scripts/`: 이전 코드. NBCP 예제가 비교 기준으로 불러온다.

## 다른 저장소를 가리키는 경로

이 저장소의 옛 기록(issue-notes, idea-proposals, 개발 노트 등)에 나오는 `examples/nbcp_*`, `docs/nbcp/`, NBCP `data-space/verification/` 폴더, NBCP 연구노트와 보조 노트는 새 저장소의 같은 경로를 뜻한다. 새 저장소의 기록에 나오는 `code-space/spintoolkit/`, `docs/lswt/`, `docs/development/`, `examples/lswt_*`, 개발 결정 D01–D47처럼 패키지에 속한 경로는 패키지 저장소 `sungmin-park-dev/Linear-Spin-Wave-Theory`를 뜻한다. 옛 기록의 경로는 고치지 않는다.

## 패키지

새 저장소는 `spin-toolkit`을 이 저장소의 고정 커밋에서 설치한다(그 저장소의 `pyproject.toml`). 패키지를 새 버전으로 바꿀 때는 이 줄의 커밋을 바꾸고 테스트를 돌린다.
