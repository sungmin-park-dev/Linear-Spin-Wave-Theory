# CLAUDE.md

> LSWT 패키지 개발 가이드. Claude가 코드 작업 시 참고하는 문서.

---

## 프로젝트 개요

2D 스핀 모델의 Linear Spin Wave Theory 계산을 위한 Python 라이브러리.
논문 arXiv:2501.20963 결과를 재현할 수 있도록 공개 배포 목표.

**핵심 기능**: 격자 정의 → LSWT 대각화 → 물리량 계산 (열역학, 위상, 상관함수)

**설계 원칙**:
- 명확성 우선 — 물리적 의미가 코드에 드러나야 함
- 관심사 분리 — 시스템 정의 / 솔버 / 관측량(observables) / 시각화
- `AbstractSolver` 인터페이스로 다른 방법론(real-space BdG, ED, TN 등) 확장 가능
- `SpinSystem`은 solver-agnostic — LSWT 전용 로직을 넣지 않음
- Python config 사용 (YAML 아님) — exchange matrix(3×3)를 numpy로 직접 정의

---

## 현재 디렉토리 구조

```
lswt/
├── __init__.py                  # 공개 API export
├── core/
│   ├── spin_system.py           # SpinSystem (nested: Site, Coupling)
│   ├── exchange.py              # heisenberg(), xxz(), xxz_with_soc(), dm(), kitaev()
│   ├── brillouin_zone.py        # BrillouinZone (Hex/Tetra/WS)
│   ├── diagonalization.py       # Diagonalizer (Colpa + MAGSWT)
│   ├── lattice/                 # AbstractLattice + TriangularLattice (구현됨, SpinSystem 미연결)
│   └── magnetic_structure/      # CommensurateStructure (구현됨, SpinSystem 미연결)
├── solvers/
│   ├── base.py                  # AbstractSolver, SolverResult
│   ├── solver.py                # LSWTSolver(AbstractSolver) — bz_type은 solver에서 관리
│   ├── hamiltonian.py           # LSWTHamiltonian (보손 해밀토니안, Berry curvature)
│   ├── optimizer.py             # SpinOptimizer (DE + L-BFGS-B + MAGSWT)
│   └── energy.py                # EnergyFunction (고전 + 양자 에너지, legacy dict 기반)
├── observables/
│   ├── bose_statistics.py       # Bose-Einstein 커널
│   ├── thermodynamics.py        # 내부에너지, 엔트로피, 비열
│   ├── topology.py              # Berry curvature, Chern number, thermal Hall
│   └── correlations.py          # 상관함수, 구조인자, 스펙트럼 함수
├── visualization/               # ❌ 미포팅 (legacy/modules/Plotters/)
└── config.py                    # 물리 상수, 기본값, 수치 임계값

examples/                        # 예제 스크립트 (작성 중)
tests/                           # pytest 테스트 (미작성)

legacy/                          # 원본 legacy 코드 (아카이브, 수정 안 함)
├── scripts/                     # 원본 실행 스크립트 (1~4번, modified_do_it)
└── modules/                     # 원본 모듈
```

---

## 핵심 API

```python
from lswt import SpinSystem, LSWTSolver

# 시스템 정의 — nested dataclass
site_a = SpinSystem.Site("A", [0, 0], spin=0.5, angles=[θ, φ], magnetic_field=[0, 0, h])
coup = SpinSystem.Coupling(0, 1, J_matrix, displacement)
system = SpinSystem([site_a, ...], [coup, ...], lattice_vectors)

# 접근
system.site("A").position       # label 또는 index로 접근
system.get_couplings("A", "B")  # 필터링된 coupling 리스트

# 솔버 실행 — bz_type은 solver에서 지정
solver = LSWTSolver(system, bz_type="Hex_60")
result = solver.solve(N=10)

# 결과 (SolverResult)
result.ground_state_energy   # float
result.eigenvalues           # np.ndarray (num_k, num_bands)
result.method                # str
result.data                  # dict (솔버별 고유 데이터)
```

**Legacy 호환**: `LSWTSolver`는 legacy dict 형식도 받음. `SpinSite`, `Coupling` alias 유지 (점진적 제거 예정).

**향후 목표**:
```
SpinSystem ──┬── LSWTSolver(system).solve()  → SolverResult
             ├── EDSolver(system).solve()    → SolverResult  (미구현)
             └── BdGSolver(system).solve()   → SolverResult  (미구현)
```

---

## 코딩 컨벤션

- **Naming**: 모듈 `snake_case.py`, 클래스 `PascalCase`, 함수 `snake_case`, 상수 `UPPER_SNAKE_CASE`
- **Docstring**: NumPy 스타일
- **Import 순서**: 표준 라이브러리 → 서드파티 (`numpy`, `scipy`) → 로컬 (`lswt.*`)
- **Type hints**: 사용 권장
- **언어**: 코드·주석·docstring은 영어, 사용자 대화는 한국어

---

## 작업 원칙

> 이 섹션은 Claude가 코드 작업 시 반드시 따라야 하는 규칙이다.

1. **제안 우선**: 파일 생성·수정·삭제 전 반드시 변경 계획을 먼저 제시하고 승인 대기.
2. **인터페이스 변경은 토의 후 결정**: API(함수명, 데이터 구조, 클래스 인터페이스) 변경은 선 제안 → 토의 → 성민 확정 순서를 따름. 임의로 결정하지 않음.
3. **영향 범위 명시**: 모듈 간 의존성 변경이 생기면 영향받는 모듈을 명시할 것.
4. **물리적 의도 불명확 시 질문**: legacy 로직의 물리적 의미가 불분명하면 임의 해석하지 말고 반드시 질문할 것.
5. **단계별 검증 후 진행**: 각 단계는 구현 → legacy 수치 대비 검증 → 성민 확인 → 다음 단계 순서. 검증 전 다음 단계 착수 금지.

---

## 진행 상황

### 완료
- [x] 패키지 구조 및 `pyproject.toml`
- [x] Core: SpinSystem, exchange, BrillouinZone, Diagonalizer
- [x] Solvers: LSWTSolver, LSWTHamiltonian, SpinOptimizer, EnergyFunction
- [x] Observables: bose_statistics, thermodynamics, topology, correlations
- [x] AbstractSolver + SolverResult 공통 인터페이스
- [x] Legacy 코드와 검증 완료 (3661 k-points, 고유값 차이 0)
- [x] Legacy 아카이빙 (`modules/`, 스크립트 → `legacy/`)
- [x] 상수 통합 (`config.py`), `utils/` 제거, `physics/` → `observables/` 이름 변경
- [x] SpinSystem 리팩터링 — nested Site/Coupling, label 첫 인자, bz_type solver로 이동
- [x] `site()`, `get_couplings()` 접근 메서드 추가

### 미완료
- [ ] NBCP 예제 스크립트 (`examples/`) — 밴드 플롯 포함
- [ ] `exchange.py`: bond-angle dependent exchange 함수 (설계 미확정, NBCP 재현에 필수)
- [ ] `EnergyFunction` → `SpinSystem` 직접 수용 (현재 legacy dict 기반)
- [ ] Observables 모듈 ↔ LSWTSolver 연결 (TODO 주석 상태)
- [ ] `solver.hamiltonian_at(kx, ky)` convenience method (합의 완료, 미구현)
- [ ] k_data dict → dataclass 전환 (합의 완료, 급하지 않음)
- [ ] 시각화 포팅 (`legacy/modules/Plotters/` → `lswt/visualization/`)
- [ ] `core/lattice/`, `core/magnetic_structure/` → SpinSystem 연결 (구현됨, 미연결)
- [ ] `tests/` pytest 테스트
- [ ] 공개 정리 (`.gitignore`, README)
- [ ] Real-space BdG 솔버 (AbstractSolver 상속)
- [ ] IncommensurateStructure 구현 (장기 과제)
