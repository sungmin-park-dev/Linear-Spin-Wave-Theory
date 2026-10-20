---
frontmatter-version: 1
title: Decision - Zeeman Sign Convention of the Theory Notes
section: decisions
decision-type: convention
status: accepted
last-edited-by: claude
created: 2026-10-20
updated: 2026-10-20
scope: workbench/topics/*/lswt-*, code-space/spintoolkit
reviewed-by: user
reviewed-at: 2026-10-20
---

# Zeeman sign convention of the theory notes

사용자는 2026-10-20 이론 노트 검토에서 "노트대로"를 골랐다.

1. 이론 노트, 개발 노트(D09)와 패키지는 같은 Zeeman 관례를 쓴다: $\hat H_Z=-\mu_B\sum_I\mathbf B_I^{\mathsf T}\mathsf g_I\hat{\mathbf S}_I$, 모멘트 $\hat{\boldsymbol\mu}_I=+\mu_B\mathsf g_I\hat{\mathbf S}_I$, $\mathbf h_I=\mu_B\mathsf g_I^{\mathsf T}\mathbf B_I$. $g$와 $\mu_B$는 양수다.
2. 2026-09-06 이론 문서의 전자 부호 결정($\hat{\boldsymbol\mu}=-\mu_B\mathsf g\hat{\mathbf S}$)은 이것으로 대체한다.
3. 전자계와의 관계: 이 관례의 장 $\mathbf B$는 전자 해밀토니안의 장 $-\mathbf B$와 같다. 에너지·스펙트럼·자화는 같고, 열 Hall 부호 같은 시간 반전 홀수 양은 장 방향이 뒤집힌다. SpinW(Toth–Lake Eq. 1)처럼 전자 부호를 쓰는 계산과 비교할 때는 장 부호를 뒤집는다.

근거: 세 곳이 이미 같은 부호라 코드 변경이 필요 없고, 전자 부호로 바꾸면 공개 API의 장 방향을 뒤집어야 한다. 검사: `code-space/examples/verification/lswt_spin_hamiltonian_check.py` 5·6번. 기록: `workbench/tasks/2026-10-20-theory-note-review.md`.
