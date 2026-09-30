---
frontmatter-version: 1
title: Handoff — NBCP Y stability to angular matching
section: handoff/closed
status: draft
from: codex
to: codex
execution-status: done
last-edited-by: codex
created: 2026-09-17
updated: 2026-09-18
---

# Handoff — NBCP Y stability to angular matching

## 2026-09-18 인수 및 실행

사용자의 재개 요청으로 A의 Y angular matching 계산과 검증을 수행했다. [결과와 한계](../../issue-notes/open/260810-pseudo-goldstone-gap.md#2026-09-18--y-angular-matching)는 기존 연구 이슈에, 본문은 [단일 연구노트의 angular matching 절](../../../../docs/nbcp/research-note.md#sec-nbcp-y-angular-matching)에 기록했다. 인계 실행은 완료했으며 물리·수학 검토 상태는 `in-review`로 유지한다. B의 비선형 결함과 thermal matching은 미실행이다. 아래 내용은 2026-09-17 인계 시점의 기록으로 보존한다.

사용자가 현재 대화를 정리하고 다음 계산부터 새 대화에서 진행하도록 요청했다. 이 문서는 재개 지점이며 새 이론 원본이 아니다. 이번 정리에서는 추가 물리 계산을 수행하지 않았다. 다음 대화의 첫 목표는 **Y상의 각도별 강성 rho(phi)와 angular potential E(phi)를 비교하여 상수 강성 및 6차 조화항 근사의 정확도를 평가하는 것**이다.

## 원본과 현재 작업 상태

- 저장소: `/Users/david/GitHub/Linear-Spin-Wave-Theory`.
- NBCP 내용의 단일 편집 원본: [research-note.md](../../../../docs/nbcp/research-note.md). 상태는 `in-review`이며 수치검사 통과를 사용자 물리·수학 acceptance로 간주하지 않는다.
- 생성 PDF: [research-note.pdf](../../../../docs/nbcp/output/research-note.pdf). 현재 36쪽, 가장 최근 Y stability 절은 15–17쪽이다. PDF/중간 TeX를 직접 편집하지 않는다.
- 계산 이력과 미결 사항: [pseudo-Goldstone 이슈](../../issue-notes/open/260810-pseudo-goldstone-gap.md).
- 저장소에는 이 대화 이전부터 다수의 modified/deleted/untracked 파일이 있다. 세션의 변경과 기존 변경을 구분해야 한다. 이번 Y stability 작업에서는 production package와 legacy를 수정하지 않았고 commit/push를 수행하지 않았다. Git 상태 전체를 일괄 정리하거나 되돌리지 않는다.

## 이번 세션 완료 사항

### 1. 이전 유도와 연구 범위

원고는 NBCP 소개 → phase 분석 → Y/V supersolidity → pseudo-Goldstone gap 부록 순서다. LSWT 일반 이론은 `docs/lswt/`로 분리했다. SOC-induced common global-z orbit와 zero-field internal XXZ orbit를 구분하고, 곡률과 켤레 응답을 함께 쓰는 gap 정규화를 정리했다. Legacy anomalous block 및 uniform-theta 응답의 차이는 기존 검증 자료와 부록에 남아 있다.

Y의 zero-SOC 고전 강성, SOC에서 hard-coordinate Schur complement, mixed frequency–momentum term, clock RG convention과 미시적 검증 절차를 정리했다. SOC가 있으면 단일 +k 에너지 기울기의 제곱을 정적 강성으로 바로 환산하지 않는다. 자세한 식은 원고의 `sec-nbcp-y-soc-conditions`, `sec-nbcp-clock-rg`, `sec-nbcp-clock-matching-tests`가 소유한다.

기존 네 E(phi) 패널은 Y–PD/Y–Gamma/V–PD/V–Gamma를 각각 분리한 것이다. Active SOC=0.010 meV, 다른 SOC=0; Y는 B=0.2 T, V는 B=1.4 T. 기존 N48/P72 데이터를 사용했으며 서로 다른 절대 y축을 명시했다. 그림 번호는 원고 렌더링에 따라 달라지므로 `fig-yv-orbit` ID를 사용한다.

### 2. 새 Y 전체 BZ 조화 안정성 검사

- 실행 스크립트: [nbcp_y_stability.py](../../../../examples/nbcp_y_stability.py).
- 계산 기록: [stability-check.json](../../../../data-space/verification/260917-y-stability/stability-check.json).
- 그림: [y-stability-screen.png](../../../../data-space/verification/260917-y-stability/y-stability-screen.png), 같은 폴더의 SVG.
- 문서 검증: [document-check.json](../../../../data-space/verification/260917-y-stability/document-check.json).

J=0.075, Jz=0.125 meV, S=1/2, B=0.2 T, gz=4.645의 nearest-neighbor Y를 계산했다. SOC 쌍은 (0,0), (0.005,0), (0,0.005), (0.005,0.005), (0.010,0), (0,0.010), (0.010,0.010) meV이다. 이는 검증용 SOC 값이며 실제 물질에서 측정된 값으로 간주하지 않는다.

각 쌍에 60도 영역의 5도 간격 12각도를 사용하고, 기존 개별 SOC축의 resolved quantum-selected angle 4개를 추가하여 88개 배경을 검사했다. Quantum hard-coordinate 재최적화는 수행하지 않았다. Gamma 포함 24²/48²/96² reciprocal-cell grid의 총 1,064,448점과 다중 시작점 연속 최소화에서 음의 정적 곡률 또는 원점 이외의 복소 진동수를 검출하지 않았다. 정적 원점 nullity는 1이다.

| 결과 | 값 / 의미 |
|---|---|
| 정적 두 번째 고유값의 정련 최솟값 | 0.02009477 meV/cell; 동적 gap이 아님 |
| 두 번째 순서 동적 에너지의 정련 최솟값 | SOC에 따라 0.10388870–0.10458305 meV |
| 작은 k에서 global-phase tangent overlap | 제곱 overlap >0.9971; ka<=0.1, 12방향 |
| 인접 static projector overlap | >0.9995; 전체 BZ band tracking은 아님 |
| 강성의 최소 고유값 | 무SOC 0.00382336, 두 SOC 모두 0.010일 때 최솟값 0.00168185 meV |
| N48→96 동적 최솟값 변화 | 최대 4.58e-5 meV |
| N96 이후 연속 정련 감소량 | 최대 1.22e-5 meV |
| M1 실측 | 계산 약 74초, Python process peak RSS 약 110 MiB; 기기 RAM 8GB |

직접 Cartesian Hessian→Nambu 전체 행렬 비교의 오차는 4.2e-17 meV 이하이다. Scalar/vectorized kernel, reciprocal periodicity, 60도 회전 spectral set, torque를 확인했다. JPD=0.08 meV control에서는 575/576점의 불안정성을 검출했다. 기존 anomalous pairing 회귀 8개가 통과했다. 첫 pytest collection은 PYTHONPATH 누락으로 실패했지만 경로를 지정한 재실행은 통과했다. Legacy의 기존 docstring escape 경고 2개는 남아 있다.

기본 static cutoff=1e-10 meV/cell, imaginary-frequency cutoff=1e-8 meV는 조절 가능한 수치 진단값이다. 원점 defective zero pair의 약 3e-9 meV imaginary roundoff는 정확한 정적 nullity와 phase tangent로 따로 식별했다. Onsite shift나 quantum pinning을 삽입하지 않았다. Non-Gamma 판정은 1e-12–1e-8 범위의 진단 cutoff에서도 유지된다.

원고의 기존 수식 79개와 기존 code/legacy/data 파일 136개를 보존했다. 수식 1개를 추가하여 총 80개, semantic ID는 94개이다. PDF 36쪽 전체와 새 절 상세 렌더링을 검토했으며 LaTeX overflow·미해결 참조가 없다. 정리 시점에 원고·PDF·계산 JSON의 hash가 최종 문서 검증 기록과 일치함을 다시 확인했다.

## 해석 한계와 미결 사항

1. 이 결과는 한 자기장에서 유한한 momentum/angle 표본과 국소 정련으로 수행한 harmonic screen이다. 전체 parameter space의 안정성 증명이나 경쟁상보다 낮은 에너지의 증명이 아니다.
2. Static Hessian eigenvalues는 동적 excitation energy가 아니다. 두 번째 순서 동적 에너지도 전체 BZ에서 eigenvector로 추적한 특정 hard branch라고 부르지 않는다. Hard-mode correlation length 및 finite-T matching scale은 아직 정하지 않았다.
3. `three_msl`의 stored d는 basis 좌표와 비교하면 ri-rj modulo magnetic translations에 해당하고 API 문구는 rj-ri이다. 현재 해석은 neighbor rj=ri-d 및 현재 코드의 k 좌표를 쓴다. 전체 k 반전은 이번 안정성 판정에 영향을 주지 않지만 odd-in-k 응답의 실험 방향 부호를 정할 때 수정·검토가 필요하다. Production convention을 조용히 변경하지 않는다.
4. SOC Y의 6차 potential은 symmetry-allowed이다. 이것만으로 실제 NBCP의 thermal RG basin 또는 두 BKT 전이를 확정하지 않는다. Generic V with Gamma의 3차 항과 pure-PD V의 6차 예외도 유지한다.
5. 고전 강성과 leading zero-point potential은 근사 차수가 다르다. Quantum gradient correction, quantum hard-coordinate relaxation, S=1/2 오차 평가는 미완료다. 현재 계산을 quantum-renormalized/thermal coefficient로 보고하지 않는다.

## 다음 세션 작업 목록

### A. Y angular matching — 다음 실행 단위

먼저 원고와 위 JSON을 읽고 현재 hash 및 작업 트리를 확인한다. 기존 안정성 계산은 변경·실패·미해결 검증 필요성이 없다면 다시 전부 돌리지 않는다. 현재 B=0.2 T의 안정한 SOC 점에서 시작하고 phase/자기장 범위를 임의로 넓히지 않는다.

1. `nbcp_y_soc_conditions.py::background/reduction`과 새 stability 자료를 사용해 rho(mu,nu;phi)의 각도 의존성을 분해한다. Angular resolution을 늘려 수렴을 확인하고, 두 tensor eigenvalues·determinant·회전하는 principal axes를 구분한다. 개별 tensor component에 scalar potential의 6차 선택 규칙을 그대로 강제하지 말고 tensor covariance를 확인한다.
2. 기존 E(phi)의 current-Hamiltonian 배열을 먼저 재사용한다. Fourier 계수는 symmetry를 강제하기 전에 추출하며 lambda6, lambda12 및 더 높은 항을 비교한다. Curvature에서는 n² 가중치가 있으므로 에너지 진폭 비만으로 고차항을 버리지 않는다.
3. Pure-Gamma Y의 angular energy는 약 1e-12 meV/spin 수준으로 매우 작다. 기존 noise/convergence 기록을 확인하고 유효 자릿수·불확실성을 명시한다. Mixed SOC는 기존 개별-axis energy scan에 없는 조합이므로 필요하면 새로 계산하며, 단순 합으로 대신하지 않는다.
4. 상수 강성+6차 potential의 오차와 각도 의존 강성+고차 potential의 차이를 정량화한다. 우선 T=0의 고전 강성/leading vacuum energy 조합임을 명시한다. 이 단계만으로 thermal RG 초기조건이 완성됐다고 보고하지 않는다.
5. 별도 진단 스크립트와 새 검증 데이터에 계산을 저장하고, 본문은 기존 단일 Markdown owner에 반영한다. 기존 문서 디자인 및 표준 LaTeX 인터페이스를 유지한다. 구현·독립 검증 및 사용자 검토의 경계를 지킨다.

### B. 후속 단계 — A의 결과에 따라 범위 확정

Domain wall/vortex의 비선형 고전 최적화 → 유효모형 RG/classical MC 및 크기 수렴 순서로 진행한다. 밀도 translation domain과 angular clock domain을 구분하고 fractional vortex를 사전에 가정하지 않는다. Finite-T matching에서 이미 적분한 phase mode를 다시 MC/RG에 이중 포함하지 않는다. 경쟁상 비교와 자기장 의존성은 별도의 검증 축이다.

M1에서 ED/TN을 먼저 시작할 필요는 없다. 작은 ED는 후속 양자 benchmark, 좁은 cylinder TN은 제한된 보완 계산으로 검토할 수 있다. 본래 frustrated SOC quantum model의 QMC는 sign problem을 먼저 확인해야 한다. 유효 classical model의 MC가 실제 S=1/2 물질의 양자 thermal phase diagram을 대신하지 않는다.

## 재현 환경과 명령

아래는 저장소 루트에서 실행한다. M1에 설치된 scientific Python을 사용하며 작은 행렬 계산은 BLAS thread 1개로 시작한다.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /Users/david/miniconda3/bin/python examples/nbcp_y_stability.py
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=code-space /Users/david/miniconda3/bin/python -m pytest code-space/tests/test_solvers/test_hamiltonian_pairing.py -q
PYTHONDONTWRITEBYTECODE=1 /Users/david/miniconda3/bin/python examples/nbcp_research_export.py
```

Exporter는 Quarto/XeLaTeX를 사용한다. 이 환경에서는 Quarto의 sysctl 접근 때문에 PDF 출력 실행에 sandbox escalation이 필요했다. 설치된 Mathematica/Wolfram 15는 `/Applications/Wolfram.app/Contents/MacOS/WolframKernel`이며 symbolic 자료는 기존 `.wl`을 참고한다. 위 세 명령은 재현 안내이며 새 세션마다 전부 실행할 의무는 없다.

## 참고 문서와 데이터

- [Task queue](../../TASK-QUEUE.md), [handoff map](../map-handoff.md).
- [Y SOC numerical checks](../../../../data-space/verification/260917-y-soc-conditions/soc-conditions-check.json), [SOC diagnostic](../../../../examples/nbcp_y_soc_conditions.py).
- [Saved N48/P72 angular scans](../../../../data-space/verification/260912-pseudo-goldstone/scan-N48-P72.json), [clock-anisotropy audit](../../../../examples/nbcp_clock_anisotropy_audit.py).
- [Clock RG checks](../../../../data-space/verification/260917-clock-rg/clock-rg-check.json), [four-panel provenance](../../../../data-space/verification/260917-clock-matching/four-case-figure.json).
- Reviewed paper: [arXiv:2601.20963v1](https://arxiv.org/abs/2601.20963v1). Gap-theory papers and detailed source locations are already recorded in the manuscript.
