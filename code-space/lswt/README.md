# LSWT 패키지의 기능별 구성

| 위치 | 담당 기능 |
| --- | --- |
| `system/` | `SpinSystem`, 교환행렬, 실공간 격자와 Brillouin zone 기하 |
| `states/` | 정합·비정합 자기구조의 표현. 비정합 구현은 기존 stub 상태 |
| `methods/base.py` | `AbstractSolver`와 `SolverResult` 공통 인터페이스 |
| `methods/optimization.py` | 고전·양자 에너지 함수를 사용하는 스핀상태 최적화 |
| `methods/spin_wave/` | LSWT 솔버, 보손 해밀토니안, Colpa 대각화, 에너지 평가 |
| `definitions/` | 물리상수, 수치 기본값, 스핀 기저 변환 규약 |
| `observables/` | 보스 통계, 열역학, 위상, 상관함수 |
| `visualization/` | 스핀 배열과 계산 결과 표시 |

`definitions/constants.py`에는 물리상수, `defaults.py`에는 계산 기본값과
허용오차, `spin_basis.py`에는 기존 기저 변환 행렬을 둔다. 수치와 정규화는
이번 이동에서 변경하지 않았다. 단위·출처를 포함하는 별도 데이터 파일은
후속 설계 범위다.

모델 고유 구성은 저장소의 `model/<name>/`에 둔다. 향후 ED·TN 구현은
`methods/`의 독립 하위 패키지로 추가한다. 빈 구현 폴더는 만들지 않는다.

## 현재 경계와 다음 단계

이번 정리는 파일 소유 위치와 import 경로의 이행이다. 기존 `SpinSystem`에는
여전히 스핀 방향이 들어 있고, `EnergyFunction`은 고전 항과 스핀파 보정을
함께 평가한다. 이를 새로운 공통 `SpinModel` 규약으로 분리하는 작업은 아직
수행하지 않았다. 현재 관측량 구현도 LSWT 결과 형식에 의존한다.

새 코드는 기능별 경로를 사용한다.

```python
from lswt.system import SpinSystem, exchange
from lswt.states import CommensurateStructure
from lswt.methods.spin_wave import LSWTSolver
```

최상위 `from lswt import SpinSystem, LSWTSolver` API는 유지한다.
옛 `lswt.core`, `lswt.solvers`, `lswt.config` import는 `_compat.py`에서
같은 구현 객체로 연결한다. 이 파일은 호환 경로만 소유하며 계산 코드를 복제하지
않는다. 모듈의 실제 `__module__`과 `__file__`은 새 위치를 가리킨다.

개발 현황과 이행 검증은
[개발 기록](../../docs/development/README.md)에서 관리한다.
