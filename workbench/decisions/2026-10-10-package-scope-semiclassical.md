---
title: spin-toolkit 범위를 2D 스핀 모형의 반고전 툴킷으로 정함
status: accepted
decided: 2026-10-10
reviewed-by: user
---

# spin-toolkit 범위: 2D 스핀 모형의 반고전 툴킷 (2026-10-10)

## 결정

`spin-toolkit`은 2D 스핀 모형의 반고전(semiclassical) 툴킷이다. 고전 질서를 찾고 그 위의 요동을 계산하며, 중심은 선형 스핀파 이론(LSWT)이다. 사용자가 세 안("LSWT만", "중간", "일반 2D") 가운데 추천한 중간 안을 골랐다("추천대로 하자", 2026-10-10).

## 범위 안

- 고전 상태: 에너지, 전역 탐색, Luttinger–Tisza, 고전 Monte Carlo, Landau–Lifshitz·Langevin 동역학, 조화 자유에너지
- 스핀파: `solve_lswt`(공액 상태, 회전 좌표 나선), `solve_nlswt`(1/S 보정), 유사 Goldstone gap
- 관측량과 그림: 위 결과에서 나오는 열역학, 구조 인자, 중성자 세기, 위상, 자화 곡선
- 작은 클러스터 정확 대각화(ED): 스핀파 결과를 검증하는 용도로만

## 범위 밖

- 양자 다체 계산(DMRG, 양자 Monte Carlo, 텐서망 등). 성숙한 패키지(TeNPy, ITensor, NetKet 등)가 이미 있고, 같은 수준으로 짓고 검증할 수 없다.
- 3D 격자 (기존 결정 그대로)

## 근거

- 지금 있는 방법들은 "고전 질서 → 그 위의 요동"이라는 한 흐름을 이룬다. LSWT는 고전 바닥 상태가 있어야 돌므로, 고전 방법을 빼면 쓰기 어려워진다.
- 물리적 근거가 분명한 계산을 문헌과 대조해 검증하는 것이 이 패키지의 강점이다. 비슷한 도구는 SpinW(MATLAB), Sunny(Julia)이고, 이 패키지는 Python과 2D, 문헌 대조된 위상·1/S 계산으로 구분된다.

## 바뀐 것

- `pyproject.toml` 설명: "Semiclassical toolkit for 2D spin models: classical order, linear and nonlinear spin-wave theory"
- `README.md` 첫머리에 범위 문단
- `CLAUDE.md`·`AGENTS.md`의 "ED, TN 등으로 확장"과 `EDSolver`·`BdGSolver` 예정 줄을 범위 문장으로 바꿈
- 개발 결정 기록 D48(`workbench/notes/dev-decisions`)
