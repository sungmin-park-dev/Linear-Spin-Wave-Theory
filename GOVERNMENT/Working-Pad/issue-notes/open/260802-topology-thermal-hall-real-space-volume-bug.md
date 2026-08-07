---
frontmatter-version: 1
title: Topology thermal Hall real_space_volume bug
section: issue-notes/open
issue-type: problem
status: draft
last-edited-by: claude
created: 2026-08-02
updated: 2026-08-02
source: CLAUDE.md 알려진 버그 #2
related: code-space/lswt/observables/topology.py
---

# Topology thermal Hall `real_space_volume` bug

CLAUDE.md/AGENTS.md의 "알려진 버그" 2번을 추적 가능한 issue-note로 옮긴
것이다. 코드는 수정하지 않았다.

## 요약

`Topology` 클래스의 thermal Hall conductance 계산에서 `real_space_volume`
산출에 `self.Ns` 곱셈이 빠져 있고, meV·Å 단위를 W/(m·K)로 바꾸는 변환
계수도 `1e-12`가 아니라 `1e-22`여야 한다고 CLAUDE.md에 기록돼 있다.

## 현재 코드 상태

`code-space/lswt/observables/topology.py:250-252`:

```python
real_space_volume = valid_count * np.sqrt(3) / 2
coefficiten_thc = (K_BOLTZMANN_MEV ** 2) / (H_BAR_MEV * real_space_volume)
coefficiten_thc *= 1.602176634 * 1e-12  # from meV, Angstrom to W/(m*K)
```

CLAUDE.md 기록과 대조하면:

1. `real_space_volume`이 `valid_count * np.sqrt(3) / 2`로만 계산되고
   `self.Ns` 곱셈이 없다 — 기록된 지적과 일치.
2. 단위 변환 계수가 `1.602176634 * 1e-12`로 남아 있다 — 기록된 `1e-22`
   교체가 아직 반영 안 됨.

이 audit에서는 CLAUDE.md 기록과 현재 코드가 여전히 같은 상태임을
재확인했을 뿐, `self.Ns`를 어디에 곱해야 하는지·`1e-22`가 맞는 값인지는
직접 재유도하지 않았다.

## Required Follow-Up

1. Thermal Hall conductance 공식을 원본 PDF 노트(§Skyrmions, Topological
   Magnons, and Hall Effects)와 다시 대조해 `self.Ns` 곱셈 위치와
   단위 변환 계수 `1e-22`의 근거를 확인한다.
2. 물리적 의도가 불명확하면 임의로 고치지 않고 성민 확인을 받는다
   (프로젝트 규칙 4).
3. `research-space/theory/lswt/observables/topological-magnon-quantities.md`의
   review ledger ID **A7**("Thermal-Hall band sum", 현재 `open`)이 이
   코드 버그와 관련된 이론 검증 항목이다 — 함께 다룬다.
4. 수정 후 단위와 크기 order를 독립적으로 sanity-check한다(예: 알려진
   물질의 실측 thermal Hall 값과 order-of-magnitude 비교).

## 범위 밖

이 issue-note는 문제를 기록하고 재현 지점을 명확히 하는 것까지만 한다.
실제 수정, 물리적 판단, 단위 재유도는 여기서 하지 않는다.
