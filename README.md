# Linear Spin Wave Theory

> **개발 상태:** Alpha · 연구용 · 공개 배포 전 검증 중

2차원 양자 스핀 모델의 선형 스핀파 이론
(Linear Spin Wave Theory, LSWT) 계산을 위한 Python 라이브러리입니다.

격자, 자기 사이트, 교환 상호작용을 정의하고 LSWT 해밀토니안을 구성·대각화하여
마그논 스펙트럼과 양자 보정을 계산하는 재사용 가능한 도구를 목표로 합니다.

- 현재는 2차원 LSWT를 중심으로 개발하고 있습니다.

## 프로젝트 목표
- 물리적 의미가 드러나는 스핀 시스템 정의
- 시스템 정의, solver, 관측량, 시각화의 역할 분리
- 이론 문서와 코드 구현의 단계별 대조 검증
- 코드를 통해 예제 적용
   - NBCP(Na₂BaCo(PO₄)₂) 관련 계산의 재현
   - 기타 예제 탐색 필요
- `AbstractSolver`와 `SolverResult`를 이용한 solver 확장

## 현재 지원 범위

| 구성요소 | 현재 상태 |
|---|---|
| `SpinSystem` | 사이트, 격자 벡터, 교환 결합 정의 구현 |
| Exchange matrix | Heisenberg, XXZ, SOC, DM, Kitaev 및 NBCP용 교환행렬 지원 |
| `LSWTSolver` | Brillouin zone 계산과 LSWT 대각화 흐름 구현 |
| `SpinOptimizer` | 고전 바닥상태 최적화 구현 |
| `SolverResult` | solver 공통 결과 인터페이스 구현 |
| Observables | 열역학·위상·상관함수 모듈은 있으나 고수준 solver 연결 미완료 |
| Visualization | 스핀 배열 시각화 구현 |
| Legacy compatibility | 기존 dictionary 입력 형식과 호환 유지 |

구현된 모듈이 모두 검증 완료되었거나 안정적인 공개 API라는 뜻은 아닙니다.
현재 저장소는 이론–코드 대응과 수치 검증을 진행 중입니다.

## 기본 구조

```text
SpinSystem
    └── LSWTSolver.solve()
            └── SolverResult
```

`SpinSystem`은 solver에 독립적인 시스템 정의를 담당합니다. LSWT 고유 계산은
solver 계층에 두고, 관측량과 시각화는 별도 모듈로 분리합니다.

## NBCP 검증 사례

이 프로젝트는 Na₂BaCo(PO₄)₂(NBCP)의 삼각격자 스핀 모델을 주요 검증 사례로
사용합니다.

관련 연구:

- Woodland et al., [“From continuum excitations to sharp magnons via transverse magnetic field in the spin-1/2 Ising-like triangular lattice antiferromagnet Na₂BaCo(PO₄)₂”](https://arxiv.org/abs/2505.06398), Phys. Rev. B 112, 104413 (2025)
- Gao et al., [“Spin supersolidity in nearly ideal easy-axis triangular quantum antiferromagnet Na₂BaCo(PO₄)₂”](https://doi.org/10.1038/s41535-022-00500-3), npj Quantum Materials 7, 89 (2022)

현재 저장소에는 다음 개발용 검증 스크립트가 있습니다.

- `examples/nbcp_ground_state.py`
- `examples/nbcp_hamiltonian_check.py`

NBCP의 LSWT 해밀토니안, Colpa 대각화, 밴드 구조, 열역학·위상 관측량을
포함한 end-to-end 재현은 아직 완료되지 않았습니다.

## 문서 안내

- [개발 설계와 Beamer](docs/development/README.md) — 2D Spin-System Toolkit의 목표·계산 흐름·공통 규약·폴더 구성
- [개발 설계 PDF](docs/development/output/pdf/development-log.pdf) — 로컬 검토용 생성 문서
- [패키지 폴더 안내](code-space/spintoolkit/README.md) — 기능별 모듈의 책임과 옛 이름 `lswt` 호환
- [NBCP 모델 작업 공간](model/nbcp/README.md) — 모델 고유 정의와 계산 구성

- [문서 안내](docs/README.md) — LSWT·NBCP와 본문·참고자료·아카이브·출력물 구분
- [LSWT 일반 이론](docs/lswt/README.md) — 개념별 이론 문서와 읽는 순서
- [NBCP 연구 노트](docs/nbcp/README.md) — 모델별 유도·계산·비교 기록
- `GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md` — 문서 coverage와 열린 검토 항목
- `examples/` — 실행 예제와 검증 스크립트
- `code-space/spintoolkit/` — Python 패키지 구현(옛 이름 `lswt`는 `code-space/lswt/` 호환 패키지)
- `legacy/` — 과거 코드와 연구 노트 보존 영역

## 테스트

```bash
python -m pytest code-space/tests -q
```

자동 테스트는 공통 모델·상태, 고전 에너지와 상태 선택, 정확 대각화, LSWT와
물리량(열역학·구조인자·위상량), 모델 구성과 옛 import 호환성을 포함합니다. 단계별 검증
기록은 `docs/development/verification/`에서 관리합니다.

## 알려진 제한사항

- 비정합(incommensurate) 자기 구조는 미구현 상태입니다.
- 작은 회피 교차 근처에 곡률이 몰린 모델(NBCP 등)의 thermal Hall은 균일 격자로 수렴이 느려
  적응형 적분(`AdaptiveIntegration`)과 수렴 확인이 필요합니다.
- 비균일 pseudo-Goldstone soft mode 처리는 지원하지 않습니다.
- LSWT는 질서화된 준고전적 상태를 중심으로 사용하는 근사입니다.

연구 결과에 사용하기 전에는 모델 정의, 부호와 단위 convention, 수렴성 및
수치 결과를 독립적으로 검증해야 합니다.

## 향후 작업

1. LSWT Hamiltonian과 Colpa 대각화 검증
2. NBCP 밴드 구조 예제 완성
3. 관측량 모듈과 `LSWTSolver` 연결
4. 자동화된 수치 회귀 테스트 확대
5. 밴드 및 Berry curvature 시각화
6. ED와 real-space BdG solver 확장

## 라이선스

패키지 메타데이터에는 MIT 라이선스가 지정되어 있지만, 저장소의 정식
`LICENSE` 파일은 아직 추가되지 않았습니다. 공개 배포 전에 라이선스 문서를
확정할 예정입니다.

## 저자

- Sung-Min Park
- Email: sungmin.park.0226@gmail.com
