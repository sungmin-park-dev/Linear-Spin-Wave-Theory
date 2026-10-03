# spintoolkit 패키지의 기능별 구성

2D 스핀계 계산 도구의 공통 패키지다. 관례 별칭은 `import spintoolkit as stk`이다.
2026-09-29에 패키지 이름을 `lswt`에서 `spintoolkit`으로 바꿨다.

| 위치 | 담당 기능 |
| --- | --- |
| `system/` | `SpinSystem`, 교환행렬, 실공간 격자와 Brillouin zone 기하 |
| `system/model.py` | 공통 모델 `SpinModel`(`Site`, `Term`)과 전달 규약 검증, `fingerprint`. 모든 값은 계수의 에너지 단위 E0 기준 무차원 |
| `system/conditions.py` | 외부 조건 `ExternalConditions`: 무차원 장 `field = μ_B B/E0`와 온도 `temperature = k_B T/E0` |
| `system/geometry.py` | 계산격자 조건 `CalculationGeometry`: 열역학 극한 또는 유한 토러스 |
| `system/conversion.py` | 공통 모델·상태와 기존 `SpinSystem` 사이의 양방향 변환(결합 변위 `d = r_source − r_target`, D13) |
| `states/` | 정합·비정합 자기구조의 표현. 비정합 구현은 기존 stub 상태 |
| `states/spin_state.py` | 고전 스핀 배열 `SpinState`: 정수 초격자와 (사이트, 셀)별 단위 벡터, 상태 검증 |
| `methods/base.py` | `AbstractSolver`와 `SolverResult` 공통 인터페이스 |
| `methods/optimization.py` | 기존 `SpinSystem` 기반 고전 스핀상태 탐색(`SpinOptimizer`; 새 자료형용 전역 탐색으로 옮긴 뒤 사용 중단 예정, D30) |
| `methods/classical.py` | `SpinModel`의 항만 읽는 고전 에너지·국소장·토크 |
| `methods/lswt/` | 새 진입점 `solve_lswt`(D24), 보손 해밀토니안, Colpa 대각화, 에너지 평가(이전 `methods/spin_wave/`). 기존 `LSWTSolver`는 사용 중단(D30) |
| `methods/nlswt/` | 비선형 스핀파 `solve_nlswt`(D46): HP 4차 전개, O(S⁰) 바닥 에너지, 1/S 마그논 에너지; `pseudo_goldstone_gap`(D47): 유사 Goldstone gap의 다음 차수 |
| `definitions/` | 물리상수, 수치 기본값, 스핀 기저 변환 규약 |
| `models/` | 표준 벤치마크 해밀토니안(사각·삼각격자 하이젠버그)과 해석적 기준 스핀 배열 |
| `observables/` | 보스 통계, 열역학, 위상, 상관함수 |
| `visualization/` | 스핀 배열과 계산 결과 표시 |

`definitions/constants.py`에는 물리상수, `defaults.py`에는 계산 기본값과
허용오차, `spin_basis.py`에는 기존 기저 변환 행렬을 둔다. 수치와 정규화는
이번 이동에서 변경하지 않았다. 단위·출처를 포함하는 별도 데이터 파일은
후속 설계 범위다.

모델 고유 구성은 저장소의 `model/<name>/`에 둔다. 향후 ED·TN 구현은
`methods/`의 독립 하위 패키지로 추가한다. 빈 구현 폴더는 만들지 않는다.

## 현재 경계와 다음 단계

1단계(2026-09-29)에서 공통 모델 `SpinModel`, 상태 `SpinState`, 외부 조건,
계산격자 조건과 이를 읽는 고전 에너지 계산을 추가했다. LSWT는 아직 기존
`SpinSystem`을 입력으로 받으며, 두 형식의 연결은 2단계다.

2026-09-23 정리는 파일 소유 위치와 import 경로의 이행이다. 기존 `SpinSystem`에는
여전히 스핀 방향이 들어 있고, `EnergyFunction`은 고전 항과 스핀파 보정을
함께 평가한다. 이를 새로운 공통 `SpinModel` 규약으로 분리하는 작업은 아직
수행하지 않았다. 현재 관측량 구현도 LSWT 결과 형식에 의존한다.

새 코드는 기능별 경로를 사용한다.

```python
from spintoolkit.system.model import SpinModel, Site, Term
from spintoolkit.states import SpinState
from spintoolkit.system.conditions import ExternalConditions
from spintoolkit.methods.lswt import solve_lswt
```

기존 `SpinSystem`·`LSWTSolver` 경로는 호환을 위해 남아 있으며, `LSWTSolver`와
SI 단위 Hall API(`Topology.compute_thermal_Hall`)는 사용 중단 경고를 낸다(D30).

최상위 `from spintoolkit import SpinSystem, LSWTSolver` API를 제공한다.
옛 이름은 `code-space/lswt/` 호환 패키지가 소유한다. `import lswt`는
DeprecationWarning을 내고, `lswt.<경로>`를 같은 `spintoolkit` 모듈 객체로
연결한다(`methods.spin_wave`는 `methods.lswt`). 2026-09-23 이전의 `lswt.core`,
`lswt.solvers`, `lswt.config`도 같은 곳에서 연결한다. 호환 패키지는 계산 코드를
복제하지 않으며, 모듈의 실제 `__module__`과 `__file__`은 새 위치를 가리킨다.
옛 경로로 저장한 pickle도 이 연결로 복원된다.

개발 현황과 이행 검증은
[개발 기록](../../docs/development/README.md)에서 관리한다.
