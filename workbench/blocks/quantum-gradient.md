---
id: quantum-gradient
title: Quantum gradient 보정 · hard-coordinate 완화
status: blocked
created: 2026-09-30
next: S=1/2 각도 퍼텐셜을 1/S 전개 없이 구할 방법과 크기(ED·DMRG가 φ를 분해할 수 있는 조건)를 정한다
blocked-reason: 1/S 전개가 S=1/2에서 통제되지 않음 — 다음 차수 gap이 선도 차수의 −28배(Y)·−20배(V), 각도 곡률은 −27배·−19배(D46·D47, PR #34·#37·#39). 작은 클러스터 ED는 φ를 분해하지 못함
resume-condition: "1/S에 기대지 않는 방법으로 S=1/2 각도 퍼텐셜을 계산할 수 있을 때(사용자 장비·계산 서버)"
grounds: docs/nbcp/chapters/10-discussion.tex, issue-notes/open/260810-pseudo-goldstone-gap.md
grounds: docs/nbcp/appendices/a-pseudo-goldstone-gap.tex
grounds: workbench/notes/pseudo-goldstone-gap (Next-order gap, 2026-10-03)
---

<!-- 결론은 이 노트가 아니라 연구노트 workbench/notes/pseudo-goldstone-gap/에 옮겨 적는다 (2026-10-05부터 NBCP 원본은 연구노트) -->
