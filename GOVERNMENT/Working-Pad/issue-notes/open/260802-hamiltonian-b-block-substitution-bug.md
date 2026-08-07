---
frontmatter-version: 1
title: LSWTHamiltonian B/B† block substitution bug
section: issue-notes/open
issue-type: problem
status: draft
last-edited-by: claude
created: 2026-08-02
updated: 2026-08-02
source: CLAUDE.md 알려진 버그 #1 (PDF 노트 Linear_Spin_Wave_Theory___Note 참조)
related: code-space/lswt/solvers/hamiltonian.py
---

# LSWTHamiltonian B/B† block substitution bug

CLAUDE.md/AGENTS.md의 "알려진 버그" 1번을 추적 가능한 issue-note로 옮긴
것이다. 코드는 수정하지 않았다 — 현재 상태를 기록하고 후속 결정을 위한
근거만 남긴다.

## 요약

`code-space/lswt/solvers/hamiltonian.py`의 `K_Hamiltonian` 조립에서
$B_{\mathbf k}$와 $B_{\mathbf k}^\dagger$ 블록에 값을 넣는 부분이 뒤바뀌어
있다고 기록돼 있다 (원본 PDF 노트 기준). $B_{\mathbf k}$가 실수 대칭인 경우
(stripe phase)에는 결과가 우연히 같지만, $\Gamma$ perturbation처럼
$B_{\mathbf k}$가 비대칭인 경우에는 틀린 결과를 낸다.

## 현재 코드 상태

`hamiltonian.py:279-284`:

```python
# Anomalous hopping (particle-nonconserving)
# B block (upper-right: [0:Ns, Ns:2Ns])
self.K_Hamiltonian[:, i, self.Ns + j] += tpp * exp_mDk
self.K_Hamiltonian[:, j, self.Ns + i] += tpp * exp_pDk
# B† block (lower-left: [Ns:2Ns, 0:Ns])
self.K_Hamiltonian[:, self.Ns + j, i] += (tpp * exp_mDk).conj()
self.K_Hamiltonian[:, self.Ns + i, j] += (tpp * exp_pDk).conj()
```

같은 패턴이 Berry-curvature용 미분 행렬에도 반복된다 (`partial_derivatives_of_Hk`,
`hamiltonian.py:341-346`, `354-359`).

주석은 upper-right를 $B_{\mathbf k}$, lower-left를 $B_{\mathbf k}^\dagger$로
표준 Nambu 배치대로 라벨링하고 있고, `H[N_s{+}j, i] = \text{conj}(H[i, N_s{+}j])`
관계 자체는 Hermiticity를 만족한다. 즉 이 코드가 CLAUDE.md가 지적하는 정확히
어느 지점에서 틀리는지(block 위치인지, 두 anomalous 항 사이의 index/phase
pairing인지)는 이 audit에서 재확인하지 않았다 — **원본 PDF의 v5 수정 코드와
직접 대조해야 확정할 수 있다.**

## Required Follow-Up

1. CLAUDE.md가 언급하는 "SM의 v5 수정 코드"를 원본 PDF 노트에서 다시 찾아
   현재 구현과 항별로 대조한다.
2. 정확히 어떤 재배치가 필요한지 확인한 뒤 — 물리적 의도가 불명확하면
   임의로 고치지 않고 성민 확인을 받는다 (프로젝트 규칙 4).
3. 수정 후 stripe phase(실수 대칭 $B_{\mathbf k}$)에서는 기존 legacy 수치와
   bit-level 일치를 유지하는지, $\Gamma$ perturbation처럼 비대칭 $B_{\mathbf k}$가
   나오는 케이스에서 달라지는지 둘 다 검증한다.
4. `research-space/theory/lswt/derivation/momentum-space-bdg-hamiltonian.md`의
   review ledger ID **A1**("$B_{\mathbf k}$ off-diagonal typo와 block 식",
   현재 `open`)이 이 코드 버그와 대응한다 — 이론 문서 검증과 코드 수정을
   같은 단위로 묶어 진행한다.

## 범위 밖

이 issue-note는 문제를 기록하고 재현 지점을 명확히 하는 것까지만 한다.
실제 수정, 물리적 판단, legacy 대비 검증은 여기서 하지 않는다.
