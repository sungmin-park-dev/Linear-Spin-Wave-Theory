---
title: 결정과 문서 일괄 검토
topic: toolkit-development
priority: 1
created: '2026-10-13'
state: working
task: 2026-10-12 일괄 검토 목록(28개 항목)에 대한 사용자 판정을 차례로 이 파일 표에 남기고, 판정이 나면 dev-decisions 연구노트의 그 결정 상태 줄도 함께 고친다.
end-condition: 목록 28개 모두에 사용자 판정이 적혀 있고, dev-decisions의 상태 줄이 판정과 같다.
references:
- workbench/topics/toolkit-development/dev-decisions
- docs/reviews/2026-10-02-batch-review-guide.md
- docs/reviews/2026-10-01-lswt-draft-physics-review-guide.md
---

> 2026-10-15 `GOVERNMENT/Working-Pad/issue-notes/open/261012-batch-review-ledger.md`에서 옮겼다. 옛 머리말: title 일괄 검토 판정 기록, status in-review, created 2026-10-13, updated 2026-10-14.

# 일괄 검토 판정 기록

2026-10-12 일괄 검토 목록(`/mnt/project-files/review/2026-10-12-batch-review-list.md`, 28개 항목)에 대한 사용자 판정을 순서대로 남긴다. 스레드는 "결정과 문서 일괄 검토"다. 판정이 나면 해당 결정 기록(`workbench/topics/toolkit-development/dev-decisions`)의 상태 줄도 함께 고친다.

| 번호 | 항목 | 판정 | 날짜 | 비고 |
|---|---|---|---|---|
| 1 | D34 단일 Q 나선의 회전틀 LSWT | 승인 ("ok") | 2026-10-13 | 사용자 질문(축 회전과 교환 조건의 뜻, 일반적 처리인지)에 따라 `lswt-classical-order`에 "When the Rotation Condition Fails" 절과 검산 스크립트 `code-space/examples/verification/lswt_spiral_harmonics_check.py`를 더함(PR #63) |
| 2 | D35 결정 대칭과 대칭 허용 결합 | 승인 ("승인하자.") | 2026-10-13 | 축 벡터 R_s = det(R)R, 시간 반전 제외, 대칭 깨는 계수 거부, 자기 이온만의 대칭은 상한 |
| 3 | D37 단일 이온 항의 결맞음 상태 규칙 (1−1/2S)A | 승인 ("그래") | 2026-10-13 | 사용자 질문(⟨S_aS_b⟩ 식의 성립 범위, 계수 뜻, large-S 규칙, 결맞음 상태와 고전 스핀의 대응, 길이 √(S(S+1))의 문제)에 답한 뒤 승인. 고전 길이는 S로 두고 같은 자리 항에만 결맞음 값을 쓴다 |
| 4 | D38 고전 스핀 동역학(랑주뱅, S(q,ω)) | 잠정 승인 ("일단은 okay") | 2026-10-14 | 사용자: 상 계산에는 동역학이 쓰이지 않으므로 "동역학은 추후 다시 보기로 하자". 들뜸·유한 온도 S(q,ω)를 쓸 때 다시 검토 |
| 5 | D39 유한 온도 각도 자유 에너지 f(φ,T), 부드러운 모드 차단 Λ | 잠정 승인 | 2026-10-14 | 사용자: "일단 동역학과 유한 온도는 추천대로 진행하고, 추후 실제 계산이 필요할 때 다시 확인하자." 6번(D40)은 Claude가 유한 온도 범위로 읽어 함께 기록 |
| 6 | D40 고전 MC와 열 관측량, U(1)이 아닌 교환의 강성 거부 | 잠정 승인 | 2026-10-14 | 사용자: "일단 동역학과 유한 온도는 추천대로 진행하고, 추후 실제 계산이 필요할 때 다시 확인하자." 6번(D40)은 Claude가 유한 온도 범위로 읽어 함께 기록 |
| 7 | D41 비편극 중성자 세기, 장비·Debye–Waller 계수 제외, 2D 층의 Q_z | 승인 ("실험쪽도 다 추천대로 진행해.") | 2026-10-14 | 세기는 상대값 |
| 3a | D36 SpinModel에서 직접 만드는 LSWT H(k) (목록에서 빠졌던 항목) | 승인 ("ok") | 2026-10-15 | 내부 구현, 규약 그대로, 기존 구성기와 1e-16 일치 |
| 8 | D42 스커미온 수(공면·반평행은 "정의 안 됨")와 후보 상태 비교 | 승인 ("ok") | 2026-10-15 | 추천한 안정성 검사 변경(연속 k 검사 D51)은 이미 D53으로 `compare_states` 기본값이 되어 있어 코드 변경 없음 |
| 9 | D44 공개 전 그림 묶음, 정의 안 되는 값은 회색·빗금 | 승인 ("승인대로 하자.") | 2026-10-15 | |
| 10 | D45 표준 관측량(DOS, S^{ab}, 적분 세기, M(h)=−dE/dh) | 승인 ("승인대로 하자.") | 2026-10-15 | |
