# AGENTS.md

> LSWT 패키지 개발 가이드. Codex가 코드 작업 시 참고하는 문서.

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

> 루트 구조는 2026-06-01 `GOVERNMENT/` + `*-space/` 마이그레이션 기준이다.
> 설계 사본은 `GOVERNMENT/Working-Pad/idea-proposals/2026-05-30-project-knowledge-philosophy.md`에 둔다.

```
project-root/
├── AGENTS.md
├── CLAUDE.md
├── pyproject.toml
│
├── GOVERNMENT/                  # 운영 지식
│   ├── User-Constitution/
│   │   └── map-user-constitution.md
│   ├── Court-Precedents/
│   │   └── map-decisions.md
│   ├── Agents-Bylaws/
│   │   ├── map-agents-bylaws.md
│   │   ├── policies/
│   │   │   ├── map-policies.md
│   │   │   ├── directory-structure-guide.md
│   │   │   ├── frontmatter-policy.md
│   │   │   └── naming-convention.md
│   │   ├── procedures/
│   │   │   ├── map-procedures.md
│   │   │   └── theory_code_verification_plan.md
│   │   └── templates/
│   │       ├── map-templates.md
│   │       └── ...
│   └── Working-Pad/
│       ├── TASK-QUEUE.md
│       ├── map-working-pad.md
│       ├── handoff/
│       ├── idea-proposals/
│       ├── inbox/
│       ├── issue-notes/
│       └── vault-staging/
│
├── code-space/                  # 코드 구현
│   ├── lswt/                    # 메인 패키지
│   └── tests/                   # pytest 테스트
│
├── doc-space/                   # 사용 문서와 실행 예제
│   └── examples/
│       ├── nbcp_ground_state.py
│       └── nbcp_hamiltonian_check.py
│
├── research-space/              # 현재 이론·논문·수식 작업
│   ├── sources/                 # 정본 작업 중 실제 참조하는 원자료
│   │   └── lswt/
│   └── theory/
│       ├── sections/
│       └── notation.md
│
├── legacy/                      # 원본 legacy 코드와 과거 연구노트 아카이브
│   ├── modules/
│   ├── scripts/
│   └── research-notes/
└── data-space/                  # 계산 결과 데이터
```

---

## 핵심 API

```python
from lswt import SpinSystem, LSWTSolver

# 시스템 정의 — builder 패턴 (권장)
system = SpinSystem(lattice_vectors=[[1, 0], [0.5, np.sqrt(3)/2]])
system.add_site("A", [0, 0], spin=0.5, angles=[θ, φ], magnetic_field=[0, 0, h])
system.add_site("B", [0.5, 0.5], spin=0.5, angles=[θ, φ], magnetic_field=[0, 0, h])
system.add_coupling("A", "B", J_matrix, displacement=[1, 0])

# list 기반 생성도 지원 (하위 호환)
system = SpinSystem(sites=[...], couplings=[...], lattice_vectors=[[...], [...]])

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

> 이 섹션은 Codex가 코드 작업 시 반드시 따라야 하는 규칙이다.

1. **제안 우선**: 파일 생성·수정·삭제 전 반드시 변경 계획을 먼저 제시하고 승인 대기.
2. **인터페이스 변경은 토의 후 결정**: API(함수명, 데이터 구조, 클래스 인터페이스) 변경은 선 제안 → 토의 → 성민 확정 순서를 따름. 임의로 결정하지 않음.
3. **영향 범위 명시**: 모듈 간 의존성 변경이 생기면 영향받는 모듈을 명시할 것.
4. **물리적 의도 불명확 시 질문**: legacy 로직의 물리적 의미가 불분명하면 임의 해석하지 말고 반드시 질문할 것.
5. **단계별 검증 후 진행**: 각 단계는 구현 → legacy 수치 대비 검증 → 성민 확인 → 다음 단계 순서. 검증 전 다음 단계 착수 금지.

---

## 진행 상황

### 2026-06-01 구조 마이그레이션
- [x] 루트 폴더를 `GOVERNMENT/`, `code-space/`, `doc-space/`, `research-space/` 기준으로 재편
- [x] `pyproject.toml` 패키지 탐색 경로를 `code-space`로 갱신
- [x] AAD 철학 문서 사본을 `GOVERNMENT/Working-Pad/idea-proposals/`에 배치
- [x] 오래된 이론-코드 검증 계획서를 `GOVERNMENT/Agents-Bylaws/procedures/`로 이동하고 `needs-review`로 표시
- [x] `.DS_Store`, `__pycache__`, `*.egg-info` 생성물 정리 및 ignore 규칙 추가

### 2026-05-31 사전 정리
- [x] 코드, 지식, 운영 문서의 중간 분류 작업 수행
- [x] `research-space/theory/latex_sections/` 삭제 (pandoc 쓰레기 파일)
- [x] `research-space/sources/lswt/` 정리 (master = `note_lswt_restructured.tex` 확정)
- [x] 이론-코드 검증 계획서 초안 작성
- [x] AAD 마이그레이션 사전 작업 문서 사본 준비

### 완료 (이전)
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
- [x] SpinSystem builder 패턴 — `add_site()`, `add_coupling()` (label 기반), optional `label` 필드
- [x] `exchange.py`: `bond_angle_exchange()`, `nnn_exchange()` 추가 (NBCP용)
- [x] NBCP 고전 바닥상태 최적화 (`doc-space/examples/nbcp_ground_state.py`) — legacy와 bit-level 일치 검증 완료
- [x] Spin configuration visualizer (`code-space/lswt/visualization/spin_plotter.py`) — quiver + Sz colormap + polar angle plot

### 알려진 버그 (PDF 노트 참조: Linear_Spin_Wave_Theory___Note)
1. **B 블록 치환 오류** (`hamiltonian.py` 278-282줄): B_k와 B†_k 블록이 뒤바뀌어 있음. Bk가 실수 대칭인 경우(stripe phase)에는 결과 동일하지만, Γ perturbation 등 비대칭 Bk에서는 틀린 결과. SM의 v5 수정 코드로 교체 필요.
2. **Thermal Hall `real_space_volume`** (`topology.py`): `self.Ns` 곱셈 누락 + 단위 변환 계수 `1e-12` → `1e-22` 수정 필요.
3. **Pseudo-Goldstone gap**: 비균일 soft mode 처리 미구현 (장기 과제).

### NBCP 밴드 플롯 예제 진행 상황
1. [x] Classical ground state 최적화 — legacy 기반, SpinSystem builder 패턴으로 전환 완료
2. [x] Spin configuration visualizer — 평면 플롯 + polar angle 분포
3. [ ] LSWT 해밀토니안 검증
4. [ ] Colpa 대각화 검증
5. [ ] 밴드 플롯

### 미완료
- [ ] `EnergyFunction` → `SpinSystem` 직접 수용 (현재 legacy dict 경유: `system.to_legacy_dict()`)
- [ ] Observables 모듈 ↔ LSWTSolver 연결 (TODO 주석 상태)
- [ ] `solver.hamiltonian_at(kx, ky)` convenience method (합의 완료, 미구현)
- [ ] k_data dict → dataclass 전환 (합의 완료, 급하지 않음)
- [ ] 시각화: 밴드 플로터, interactive exchange viewer 등 추가 포팅
- [ ] `core/lattice/`, `core/magnetic_structure/` → SpinSystem 연결 (구현됨, 미연결) → triangular preset
- [ ] `code-space/tests/` pytest 테스트
- [ ] 공개 정리 (`.gitignore`, README)
- [ ] Real-space BdG 솔버 (AbstractSolver 상속)
- [ ] IncommensurateStructure 구현 (장기 과제)
