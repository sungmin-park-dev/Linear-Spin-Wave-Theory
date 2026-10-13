---
frontmatter-version: 1
title: 일괄 검토 판정 기록
section: working-pad
status: in-review
last-edited-by: claude
created: 2026-10-13
updated: 2026-10-13
---

# 일괄 검토 판정 기록

2026-10-12 일괄 검토 목록(`/mnt/project-files/review/2026-10-12-batch-review-list.md`, 28개 항목)에 대한 사용자 판정을 순서대로 남긴다. 스레드는 "결정과 문서 일괄 검토"다. 판정이 나면 해당 결정 기록(`workbench/notes/dev-decisions`)의 상태 줄도 함께 고친다.

| 번호 | 항목 | 판정 | 날짜 | 비고 |
|---|---|---|---|---|
| 1 | D34 단일 Q 나선의 회전틀 LSWT | 승인 ("ok") | 2026-10-13 | 사용자 질문(축 회전과 교환 조건의 뜻, 일반적 처리인지)에 따라 `lswt-classical-order`에 "When the Rotation Condition Fails" 절과 검산 스크립트 `examples/verification/lswt_spiral_harmonics_check.py`를 더함(PR #63) |
| 2 | D35 결정 대칭과 대칭 허용 결합 | 승인 ("승인하자.") | 2026-10-13 | 축 벡터 R_s = det(R)R, 시간 반전 제외, 대칭 깨는 계수 거부, 자기 이온만의 대칭은 상한 |
| 3 | D37 단일 이온 항의 결맞음 상태 규칙 (1−1/2S)A | 승인 ("그래") | 2026-10-13 | 사용자 질문(⟨S_aS_b⟩ 식의 성립 범위, 계수 뜻, large-S 규칙, 결맞음 상태와 고전 스핀의 대응, 길이 √(S(S+1))의 문제)에 답한 뒤 승인. 고전 길이는 S로 두고 같은 자리 항에만 결맞음 값을 쓴다 |
