# CLAUDE.md

> **프로젝트 아키텍처 및 개발 가이드**  
> 이 문서는 LSWT 패키지의 설계 철학, 구조, 확장 방법을 설명합니다.

---

## 📖 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [설계 철학](#설계-철학)
3. [아키텍처](#아키텍처)
4. [디렉토리 구조](#디렉토리-구조)
5. [핵심 컴포넌트](#핵심-컴포넌트)
6. [설정 시스템](#설정-시스템)
7. [확장 가이드](#확장-가이드)
8. [코딩 컨벤션](#코딩-컨벤션)
9. [테스트 전략](#테스트-전략)
10. [문서화 가이드](#문서화-가이드)
11. [주요 설계 결정](#주요-설계-결정)

---

## 🎯 프로젝트 개요

### 목적
Linear Spin Wave Theory (LSWT) 계산을 위한 Python 패키지로, frustrated magnetic systems의 위상도, 열역학적 성질, 위상학적 특성을 연구합니다.

### 목표 사용자
- **Primary**: 연구실 내부 연구자
- **Secondary**: 외부 연구자 (공개 배포)

### 핵심 기능
- 2D 격자 시스템 (삼각, 정사각, 허니컴, 카고메, 커스텀)
- LSWT 계산 (고전, MAGSWT)
- 위상도 계산
- 양자 보정 계산
- 열역학/위상학적 물리량 계산
- 시각화

---

## 🏛️ 설계 철학

### 1. **명확성 우선 (Clarity First)**
- 복잡한 추상화보다 명확한 구현
- 물리적 의미가 코드에 드러나야 함
- "Explicit is better than implicit"

### 2. **문서화 중심 (Documentation Driven)**
- 코드보다 문서를 먼저 읽고 이해 가능
- 모든 모듈에 물리적 배경 설명
- 사용 예제 중심의 튜토리얼

### 3. **실용주의 (Pragmatism)**
- YAGNI: 필요할 때 추가
- 현재 필요: PBC k-space 방법
- 미래 확장성: 인터페이스로 확보

### 4. **관심사의 분리 (Separation of Concerns)**
- 계산 엔진 (`lswt`) ↔ UI (`lswt_ui`) 완전 분리
- 도메인 모델 ↔ 솔버 ↔ 물리량 계산 분리

### 5. **재현성 보장 (Reproducibility)**
- 모든 계산이 설정 파일로 재현 가능
- 설정 파일의 버전 관리 가능
- 결과와 설정을 함께 저장

---

## 📂 패키지 vs 프로젝트 분리

### 핵심 개념

**lswt 패키지 (도구)**:
- Python 코드만 포함
- `pip install lswt`로 시스템에 설치
- `/usr/local/lib/python3.x/site-packages/lswt/`에 위치
- 모든 프로젝트에서 공유

**사용자 프로젝트 (데이터)**:
- 설정 파일 + 입력 데이터 + 계산 결과
- 사용자가 원하는 위치에 생성 (`~/research/my_project/`)
- 각 프로젝트는 독립적
- Git으로 버전 관리 가능

### 구조 비교
```
설치된 패키지 (코드):
/usr/local/.../lswt/
└── (Python 코드만)

사용자 프로젝트 (데이터):
~/research/my_project/
├── config.yaml
├── data/
└── output/          # ← 결과 저장 위치
```

### VASP와의 비교
```bash
# VASP
cd ~/calc/TiO2/
vim INCAR POSCAR
vasp
ls  # OUTCAR 생성됨

# LSWT (동일한 패턴)
cd ~/research/triangular/
vim config.yaml
lswt run
ls output/  # 결과 생성됨
```

## 🏗️ 아키텍처

### 전체 구조

```
┌─────────────────────────────────────────────┐
│              User Interfaces                │
│        (CLI / Desktop GUI / Web)            │
│            [lswt_ui package]                │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           High-Level API                    │
│    SpinSystem.from_config()                 │
│    LSWTSolver.solve()                       │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┼──────────┐
        │          │          │
┌───────▼────┐ ┌──▼─────┐ ┌──▼────────┐
│  Analysis  │ │Solvers │ │ Physics   │
│            │ │        │ │           │
│ • Phase    │ │• LSWT  │ │• Thermo   │
│   Finder   │ │• Optim │ │• Topology │
│ • Phase    │ │        │ │• Corr     │
│   Diagram  │ │        │ │           │
└────────────┘ └────────┘ └───────────┘
        │          │          │
        └──────────┼──────────┘
                   │
┌──────────────────▼──────────────────────────┐
│          Core Domain Models                 │
│                                             │
│  • Lattice (geometry)                       │
│  • SpinSystem (physical system)             │
│  • Hamiltonian (interactions)               │
└─────────────────────────────────────────────┘
```

### 데이터 흐름

```
config.yaml
    │
    ▼
[ConfigLoader] → [Validator]
    │
    ▼
[SpinSystem]
    │
    ├─→ [Lattice] (geometry)
    ├─→ [Interactions] (physics)
    └─→ [Hamiltonian] (operator)
    │
    ▼
[LSWTSolver]
    │
    ├─→ [Optimizer] (find ground state)
    ├─→ [Eigensolve] (diagonalize)
    └─→ [BrillouinZone] (k-points)
    │
    ▼
[Results]
    │
    ├─→ [Physics] (compute observables)
    ├─→ [Analysis] (phase diagram, etc.)
    └─→ [Visualization] (plots)
    │
    ▼
output/
```

---

## 📁 디렉토리 구조

### lswt/ (계산 엔진)

```
lswt/
├── lswt/
│   ├── __init__.py              # High-level API export
│   ├── core/                    # 핵심 도메인
│   │   ├── lattice/            # 격자 구조
│   │   │   ├── base.py         # AbstractLattice
│   │   │   ├── lattice2d.py    # 2D 구현
│   │   │   ├── presets.py      # 내장 격자
│   │   │   └── builder.py      # 생성 헬퍼
│   │   ├── spin_system.py      # SpinSystem 클래스
│   │   ├── hamiltonian.py      # 해밀토니안 빌더
│   │   └── interactions.py     # 상호작용 정의
│   │
│   ├── solvers/                 # 계산 엔진
│   │   ├── lswt.py             # LSWT 솔버
│   │   ├── optimizer.py        # 스핀 최적화
│   │   ├── eigensolve.py       # 고유값 문제
│   │   └── brillouin_zone.py   # BZ 샘플링
│   │
│   ├── physics/                 # 물리량
│   │   ├── thermodynamics.py   # 열역학
│   │   ├── topology.py         # 위상학
│   │   ├── correlations.py     # 상관함수
│   │   └── observables.py      # 관측량
│   │
│   ├── analysis/                # 고수준 분석
│   │   ├── phase_finder.py
│   │   ├── phase_diagram.py
│   │   └── quantum_corrections.py
│   │
│   ├── visualization/           # 시각화
│   │   ├── spin_viz.py
│   │   ├── band_plotter.py
│   │   └── momentum_plotter.py
│   │
│   ├── io/                      # 입출력
│   │   ├── config_loader.py    # 설정 로드
│   │   ├── validators.py       # 검증
│   │   └── export_*.py         # 결과 저장
│   │
│   └── utils/                   # 유틸리티
│       ├── constants.py
│       ├── math_utils.py
│       └── logging.py
│
├── examples/
+├── templates/                    # ✨ 프로젝트 템플릿
+│   ├── basic/
+│   │   ├── config.yaml
+│   │   └── README.md
+│   ├── phase_diagram/
+│   └── custom_lattice/
├── tests/
└── docs/
```

### lswt_ui/ (인터페이스, 별도 패키지)

```
lswt_ui/
├── lswt_ui/
│   ├── cli/                     # CLI
│   │   ├── main.py
│   │   └── commands/
│   │
│   └── desktop/                 # GUI (추후)
│       ├── app.py
│       └── widgets/
│
└── pyproject.toml
```

### 사용자 프로젝트 구조
```
~/research/my_project/           # 사용자가 생성
├── config.yaml                  # 프로젝트 설정
├── data/                        # 입력 데이터 (optional)
├── output/                      # 결과 저장
│   ├── 2025-01-15_143022/
│   │   ├── results.xlsx
│   │   ├── plots/
│   │   └── config_used.yaml
│   └── latest -> 2025-01-15_143022/
├── analysis/                    # 추가 분석 (optional)
└── notes/                       # 연구 노트 (optional)
```

---

## 🧩 핵심 컴포넌트

### 1. Lattice System

**역할**: 격자 기하학적 구조 정의

**핵심 인터페이스**:
```python
class AbstractLattice(ABC):
    @abstractmethod
    def get_primitive_vectors(self) -> np.ndarray:
        """기본 격자 벡터 반환 [a1, a2]"""
        
    @abstractmethod
    def get_basis_positions(self) -> List[np.ndarray]:
        """단위셀 내 원자 위치"""
        
    @abstractmethod
    def get_neighbors(self, order: int = 1) -> Dict:
        """이웃 리스트 {site: [neighbor_sites]}"""
        
    @abstractmethod
    def get_brillouin_zone(self) -> BrillouinZone:
        """역격자 벡터, high-symmetry points"""
```

**설계 이유**:
- 물리와 기하학 분리
- Custom lattice 정의 용이
- 미래 확장성 (3D, OBC)

### 2. SpinSystem

**역할**: 물리적 시스템 표현 (격자 + 상호작용 + 스핀 배열)

**주요 메서드**:
```python
class SpinSystem:
    def __init__(self, lattice, interactions, spin_config=None):
        self.lattice = lattice
        self.interactions = interactions
        self.spin_config = spin_config
    
    @classmethod
    def from_config(cls, config_path: str):
        """YAML에서 시스템 생성"""
        
    def build_hamiltonian(self) -> Hamiltonian:
        """해밀토니안 생성"""
```

### 3. LSWTSolver

**역할**: LSWT 계산 실행

**워크플로**:
```python
solver = LSWTSolver(system)
solver.optimize_spins()      # 바닥 상태 찾기
solver.build_boson_hamiltonian()  # 보손 해밀토니안 구성
solver.diagonalize_kspace()  # k-공간 대각화
results = solver.get_results()
```

### 4. Physics Modules

**Thermodynamics**:
- 내부 에너지, 엔트로피, 비열
- 보손 개수 (온도 의존)

**Topology**:
- Berry curvature
- Chern number
- Thermal Hall conductance

**Correlations**:
- 실공간 상관함수
- 구조인자
- 스펙트럼 함수

---

## ⚙️ 설정 시스템

### 설정 파일 철학

**단일 파일 접근**:
- 모든 설정을 한 곳에 (`config.yaml`)
- 한눈에 전체 파악 가능
- Git으로 버전 관리 용이

**계층적 구조**:
```yaml
project:          # 메타데이터
lattice:          # 격자 구조
interactions:     # 물리 파라미터
calculation:      # 계산 설정
analysis:         # 분석 옵션
output:           # 출력 설정
```

### 검증 전략

**3단계 검증**:
1. **Schema 검증**: YAML 구조 확인
2. **Type 검증**: 데이터 타입 확인
3. **Physics 검증**: 물리적 제약 확인

```python
# validators.py
class ConfigValidator:
    def validate_schema(self, config: dict):
        """YAML 구조가 올바른지"""
        
    def validate_types(self, config: dict):
        """타입이 맞는지 (S > 0, Jxy ∈ ℝ, etc.)"""
        
    def validate_physics(self, config: dict):
        """물리적으로 의미있는지"""
```

### 우선순위 체계

```
기본값 → 설정 파일 → 환경변수 → 커맨드라인
```

---

## 🔧 확장 가이드

### Custom Lattice 추가

**Step 1**: `AbstractLattice` 상속

```python
# lswt/core/lattice/my_lattice.py
from lswt.core.lattice.base import AbstractLattice

class MyCustomLattice(AbstractLattice):
    def __init__(self, param1, param2):
        self.param1 = param1
        # ...
    
    def get_primitive_vectors(self):
        return np.array([[a1x, a1y], [a2x, a2y]])
    
    # 다른 메서드 구현...
```

**Step 2**: 등록

```python
# lswt/core/lattice/presets.py
from .my_lattice import MyCustomLattice

LATTICE_REGISTRY = {
    'triangular': TriangularLattice,
    'square': SquareLattice,
    'my_custom': MyCustomLattice,  # 추가
}
```

**Step 3**: 설정 파일에서 사용

```yaml
lattice:
  type: my_custom
  param1: value1
  param2: value2
```

### 새로운 물리량 추가

**Step 1**: 함수 작성

```python
# lswt/physics/my_observable.py
def compute_my_quantity(k_data, **kwargs):
    """
    새로운 물리량 계산
    
    Parameters
    ----------
    k_data : dict
        k-point 데이터 (eigenvalues, eigenvectors)
    
    Returns
    -------
    result : float or array
        계산 결과
    """
    # 구현
    return result
```

**Step 2**: `LSWTSolver`에 메서드 추가

```python
# lswt/solvers/lswt.py
class LSWTSolver:
    def compute_my_quantity(self, **kwargs):
        from lswt.physics.my_observable import compute_my_quantity
        return compute_my_quantity(self.k_data, **kwargs)
```

---

## 📝 코딩 컨벤션

### Naming

```python
# Modules: lowercase_with_underscores
spin_system.py
phase_diagram.py

# Classes: PascalCase
class SpinSystem:
class LSWTSolver:
class TriangularLattice:

# Functions/Methods: snake_case
def calculate_energy():
def find_ground_state():
def get_neighbors():

# Constants: UPPER_SNAKE_CASE
K_BOLTZMANN_MEV = 8.617333262e-2
DEFAULT_TOLERANCE = 1e-8

# Private: _leading_underscore
def _internal_helper():
class _PrivateClass:
```

### Imports

```python
# 표준 라이브러리
import os
from pathlib import Path

# 서드파티
import numpy as np
import matplotlib.pyplot as plt

# 로컬
from lswt.core import SpinSystem
from lswt.utils import constants
```

### Docstrings

**NumPy 스타일 사용**:

```python
def calculate_energy(system, angles, temperature=0):
    """Calculate energy of spin configuration.
    
    This function computes the total energy including both
    classical and quantum contributions.
    
    Parameters
    ----------
    system : SpinSystem
        The spin system to analyze
    angles : np.ndarray
        Spin angles [theta1, phi1, theta2, phi2, ...]
    temperature : float, optional
        Temperature in energy units (default: 0)
    
    Returns
    -------
    energy : float
        Total energy per site
    
    Notes
    -----
    The energy is calculated as:
    
    .. math:: E = E_{cl} + E_{qm}(T)
    
    where :math:`E_{cl}` is classical and :math:`E_{qm}`
    is quantum contribution.
    
    Examples
    --------
    >>> system = SpinSystem.from_config('config.yaml')
    >>> angles = np.array([0, 0, np.pi/3, 0])
    >>> E = calculate_energy(system, angles)
    >>> print(f"Energy: {E:.6f}")
    """
```

### Type Hints

```python
from typing import List, Dict, Tuple, Optional
import numpy as np

def process_data(
    data: np.ndarray,
    indices: List[int],
    options: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, Dict]:
    """함수 설명"""
    if options is None:
        options = {}
    # ...
    return result, metadata
```

---

## 🧪 테스트 전략

### 테스트 레벨

**1. Unit Tests** (단위 테스트)
```python
# tests/test_core/test_lattice.py
def test_triangular_lattice_vectors():
    lattice = TriangularLattice()
    vectors = lattice.get_primitive_vectors()
    
    assert vectors.shape == (2, 2)
    assert np.allclose(vectors[0], [1.0, 0.0])
    assert np.allclose(vectors[1], [0.5, np.sqrt(3)/2])
```

**2. Integration Tests** (통합 테스트)
```python
# tests/test_integration/test_workflow.py
def test_full_calculation_workflow():
    system = SpinSystem.from_config('test_config.yaml')
    solver = LSWTSolver(system)
    results = solver.solve()
    
    assert results.energy < 0  # 반강자성
    assert len(results.eigenvalues) > 0
```

**3. Regression Tests** (회귀 테스트)
```python
# tests/test_regression/test_known_results.py
def test_square_lattice_known_result():
    """Known result from literature"""
    system = create_square_lattice(J=1.0, h=0.0)
    solver = LSWTSolver(system)
    E = solver.ground_state_energy()
    
    # Compare with published value
    E_published = -2.0  # from Reference [1]
    assert abs(E - E_published) < 1e-6
```

### Fixtures

```python
# tests/conftest.py
import pytest

@pytest.fixture
def triangular_system():
    """기본 삼각격자 시스템"""
    return SpinSystem.from_config('test_configs/triangular.yaml')

@pytest.fixture
def solver(triangular_system):
    """LSWT solver"""
    return LSWTSolver(triangular_system)
```

---

## 📚 문서화 가이드

### 문서 계층

```
docs/
├── user_guide/           # 사용자 가이드
│   ├── installation.rst
│   ├── quickstart.rst
│   ├── basic_usage.rst
│   └── custom_lattices.rst
│
├── theory/               # 이론 배경
│   ├── lswt_formalism.rst
│   ├── quantum_corrections.rst
│   └── topology.rst
│
├── api/                  # API 레퍼런스
│   └── modules.rst       # auto-generated
│
└── tutorials/            # 튜토리얼 (Jupyter)
    ├── 01_introduction.ipynb
    └── 02_phase_diagrams.ipynb
```

### 예제 작성 원칙

**점진적 복잡도**:

```python
# examples/01_quick_start.py
"""최소한의 예제 - 5분 안에 실행"""

# examples/02_custom_lattice.py
"""커스텀 격자 - 중급"""

# examples/03_phase_diagram.py
"""위상도 계산 - 고급"""
```

**자기 완결적 (Self-contained)**:
- 각 예제는 독립적으로 실행 가능
- 필요한 설정 파일 포함
- 출력 결과 설명

---

## 🎯 주요 설계 결정

### 결정 1: UI 완전 분리

**이유**:
- 계산 엔진과 UI는 다른 속도로 발전
- 계산만 필요한 사용자는 UI 의존성 불필요
- 다양한 UI 실험 가능 (CLI, Desktop, Web)

**트레이드오프**:
- ✅ 모듈성 증가
- ✅ 유지보수 용이
- ❌ 두 패키지 동기화 필요

### 결정 2: 플러그인 시스템 제외

**이유**:
- 현재 사용 패턴에서 불필요
- 추가 복잡성 > 얻는 이익
- 필요시 상속으로 충분히 확장 가능

**대안**:
- `AbstractLattice`, `AbstractSolver` 인터페이스
- 사용자는 subclass로 확장

### 결정 3: PBC 중심, OBC는 추후

**이유**:
- 현재 연구 필요성: PBC
- OBC는 알고리즘이 완전히 다름 (희소 행렬, local DOS)
- 조기 최적화 방지

**확장성 확보**:
- 인터페이스는 PBC/OBC 독립적으로 설계
- 추후 `lswt.solvers.realspace` 추가 가능

### 결정 4: 단일 설정 파일

**이유**:
- 연구 워크플로: 한눈에 전체 파악 중요
- Git diff가 명확
- 설정 공유 용이

**유연성 확보**:
- 섹션별로 include 가능 (선택적)
- 환경변수 치환 지원

### 결정 5: NumPy 스타일 Docstring

**이유**:
- 물리/과학 커뮤니티 표준
- 수식 지원 우수
- Sphinx 호환성

---

## 🚀 구현 순서

### Phase 1A: 핵심 추상화 ✅ **COMPLETE** (2025-01-28)
- [x] 디렉토리 구조
- [x] `core/lattice` (AbstractLattice + TriangularLattice)
- [x] `core/magnetic_structure` (Commensurate + Incommensurate stub)
- [x] `core/spin_system` (SpinSystem combining all)
- [x] 단위 테스트 (57 tests, 450+ lines)
- [x] 예제 스크립트 (phase1a_demo.py)

**Status**: Ready for review and Phase 1B

### Phase 1B: 확장 및 호환성 (예정)
- [ ] Interactions 모듈 설계
- [ ] SquareLattice, HoneycombLattice 구현
- [ ] Legacy 호환성 검증 및 수정
- [ ] `io/config_loader` (YAML)

### Phase 2: 솔버 (3주)
- [ ] LSWT 솔버 리팩토링 (새 SpinSystem 사용)
- [ ] Optimizer 이식
- [ ] BZ 샘플링 통합

### Phase 3: 물리량 (2주)
- [ ] Thermodynamics
- [ ] Topology
- [ ] Correlations

### Phase 4: 분석 (2주)
- [ ] Phase finder
- [ ] Phase diagram
- [ ] Quantum corrections

### Phase 5: 문서화 및 고급 기능
- [ ] API 문서
- [ ] 튜토리얼
- [ ] Incommensurate structures 구현
- [ ] 이론 배경

---

## 📞 Contact & Contribution

**Author**: Sung-Min Park  
**Email**: sungmin.park.0226@gmail.com  
**Status**: On military leave (Oct 2024 - Apr 2025)

**기여 방법**:
1. 이슈 등록
2. 브랜치 생성 (`feature/new-feature`)
3. 커밋 (`git commit -m 'Add feature'`)
4. Pull Request

---

## 📄 License

MIT License (연구실 정책 확인 후 최종 결정)

---

**Last Updated**: 2025-01-XX  
**Version**: 0.1.0-dev