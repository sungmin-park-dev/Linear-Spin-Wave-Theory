---
frontmatter-version: 1
title: 2D Spin-System Toolkit 1차 개발 계획
section: idea-proposals
status: in-review
author: claude
last-edited-by: claude
created: 2026-09-29
updated: 2026-09-29
source_refs:
  - conversation: 2026-09-29 development Beamer review and 1st development scope
  - GOVERNMENT/Working-Pad/idea-proposals/2026-09-23-spin-model-transfer-contract.md
related:
  - GOVERNMENT/Working-Pad/idea-proposals/2026-09-23-spin-model-transfer-contract.md
---

# 2D Spin-System Toolkit 1차 개발 계획

## Summary

1차 개발의 범위, 패키지 배치, 벤치마크와 구현 단계를 소유한다. 모델·항·상태·계산
요청·결과의 규약은
[[GOVERNMENT/Working-Pad/idea-proposals/2026-09-23-spin-model-transfer-contract|SpinModel 전달 규약]]이
소유한다. 결정 ID(D03–D15)는 전달 규약의 결정 목록과 `docs/development` Beamer의
결정 기록을 따른다. 이 문서는 2026-09-29 전달 규약에서 계획 부분을 분리해 만들었다.

## 1. 1차 개발 범위(D07)

| 축 | 1차 범위 | 후속 |
|---|---|---|
| 모델 | NBCP, 사각격자 하이젠버그, 삼각격자 하이젠버그 | 키타에프(후속 벤치마크, §3) |
| 항 형식 | `terms[]` 일반 형식; 허용 kind는 `bilinear`, `zeeman` | 같은 사이트 이차항, 3·4-스핀 항 |
| 장 | 모델은 사이트별 g-텐서, 장 `B`는 외부 변수; `H_Z = -mu_B B^T g S` | — |
| 계산계 | 열역학 극한 주기계: 자기 단위격자와 k점 격자 | 유한 클러스터 솔버, 열린 경계 |
| 상태 | 정합 고전 상태; 상태 검증과 기준 상태 진단(정상성·안정성) | 비정합 상태 |
| 방법 | 고전 에너지·최적화, LSWT | ED·TN 솔버, real-space BdG |
| 물리량 | 스펙트럼, 바닥상태 에너지, 열역학량, 상관함수, 위상량(thermal Hall 포함) | — |
| 검증 | 최소 ED 검증 도구(1-마그논, 필요 시 2-마그논 섹터); TN은 문헌값 비교 | — |

위상량(Berry 곡률, Chern 수, thermal Hall)은 1차 범위에 포함하되 단계별 구현에서
마지막 단계로 둔다. 최소 ED 검증 도구는 규약 검증용이지만 향후 ED 솔버로 확장할
계획이므로 `methods/` 아래에 둔다(D10).

## 2. 패키지와 폴더(D03, D04, D10, D11, D15)

공통 패키지의 이름을 계산 방법 하나(LSWT)가 아닌 도구 전체로 바꾼다. import 이름은
`spintoolkit`, 관례 별칭은 `stk`(`import spintoolkit as stk`)이다. 배포 이름은
`spin-toolkit`이다. 계산 방법은 그 아래 `methods/`의 하위 패키지로 둔다.

```text
code-space/spintoolkit/
├── system/  states/  definitions/  observables/  visualization/
├── models/              # 표준 벤치마크 모델(사각·삼각 하이젠버그, 이후 키타에프)
└── methods/
    ├── lswt/            # 이전 methods/spin_wave/
    ├── ed/              # 최소 ED 검증 도구 → 향후 ED 솔버
    ├── tn/              # 후속
    └── optimization.py
```

`code-space/`는 Python 패키지가 아니므로 `code-space/methods/lswt/`처럼 두면
`methods`, `system` 등이 각각 최상위 import 이름이 된다. 이를 피하기 위해 하나의
우산 패키지 아래에 둔다. 이름 변경은 새 코드를 작성하기 전의 별도 단계(§4의 0단계)로
2026-09-29에 수행했다. 기존 저장 객체에는 `lswt.` 모듈 경로가 기록되어 있으므로
`code-space/lswt/` 호환 패키지가 옛 이름을 DeprecationWarning과 함께 새 패키지의
별칭으로 유지한다.

공통 자료형·검증은 `code-space/spintoolkit/system/model.py`에 둔다. 공개 사용자와
테스트가 함께 쓰는 작은 표준 벤치마크 모델은 `spintoolkit/models/`에, NBCP 같은 연구
모델은 `model/<name>/`에 둔다. 연구 모델은 `model/<name>/model.py`에서 모델을 만들고
`model/<name>/calculate.py`에서 계산을 실행한다. 파일명은 향후 구현 경로이며 이 문서
작성으로 생성되지 않는다. 동일 파일 안의 모델 생성 함수와 후보 상태 생성 함수도
다른 객체를 반환한다. 모델별 보조 파일과 하위 폴더는 구현이 독립적으로 커질 때
추가한다.

기존 물리상수·수치 기본값·기저 행렬은 `definitions/`로 옮겼다. 공통 물리량의
값·단위·출처를 담는 자료 파일은 후속 설계 범위다. 결과 저장·재사용은 추후
`model/<name>/` 안에서 도입하고, `data-space/`에는 정돈된 결과를 둔다. 저장 키는
모델뿐 아니라 계산계·상태·외부 변수·방법 설정·사용한 코드와 상수 버전을 포함해야
한다. 이번 범위에서 저장 엔진이나 빈 폴더를 만들지 않는다.

## 3. 모델별 적용과 벤치마크

| 모델 | 범위 | 모델별 입력 | 공통 출력에서 달라지는 부분 |
|---|---|---|---|
| NBCP | 1차 | Jxy, Jz, PD, Gamma, g 등 | 삼각격자 사이트·이웃 연결과 결합별 J, 비등방 g |
| 사각격자 하이젠버그 | 1차 벤치마크 | J, S, g 등 | 사각격자 연결과 등방적 J 행렬 |
| 삼각격자 하이젠버그 | 1차 벤치마크 | J, S, g 등 | 삼각격자 연결과 등방적 J 행렬 |
| 키타에프 | 후속 벤치마크(D11) | Kx, Ky, Kz, S 등 | honeycomb 두 사이트 basis와 결합별 스핀 성분에 대응하는 J 행렬 |

모델별 입력 파라미터의 이름은 물리적 의미를 유지한다. 공통 출력을 만들기 위해
여러 모델을 하나의 거대한 선택적 config 사전으로 합치지 않는다.

**벤치마크의 검증 기준.** LSWT는 근사이므로 정확한 일치를 기대할 비교와 근사
비교를 구분한다. 아래 장의 값 `h`는 등방 g에서 `h = g mu_B B`이다.

| 벤치마크 | 정확한 비교 | 근사 비교 | 주의 |
|---|---|---|---|
| 사각격자 하이젠버그 | 고전 Néel 에너지 `-2JS^2`/사이트; `h > h_sat = 8JS` 완전 편극상의 1-마그논 에너지 = 같은 토러스의 ED | `h = 0` 바닥상태 에너지·부격자 자화와 문헌 QMC 값 | Goldstone 모드 처리 |
| 삼각격자 하이젠버그 | 고전 120° 에너지 `-3JS^2/2`/사이트; `h > h_sat = 9JS` 완전 편극상의 1-마그논 에너지 = 같은 토러스의 ED | `h = 0` 바닥상태 에너지·부격자 자화와 문헌 DMRG·급수 전개 값 | `0 < h < h_sat`에서 고전 우연 축퇴와 가짜 영에너지 모드; `h = 0` 스펙트럼의 강한 재규격화는 물리적 불일치; 1차 벤치마크는 `h = 0`과 `h > h_sat`로 한정(D18) |
| NBCP | 기존 1–4 MSL 수치 보존 | — | `J_PD`, `Gamma`가 0이 아니면 U(1) 대칭이 없어 편극상 ED 정확 일치가 성립하지 않음 |

완전 편극상의 정확 일치는 장 방향의 U(1) 대칭 때문에 편극 상태와 1-마그논 상태가
정확한 고유상태라는 사실에 근거한다. 이 비교는 1-마그논 섹터의 차원이 사이트 수
정도이므로 계산 비용이 작다. 같은 유한 토러스에서 비교하므로 유한 전개의 항
중복도와 `cell_offset` 부호도 함께 검증한다. 벤치마크마다 사용할 장의 범위를
명시한다. TN은 1차 범위에서 문헌값 비교로 제한한다. 벤치마크의 완전 편극상은
근사 방법인 LSWT를 정확한 기준과 비교할 수 있는 드문 경우다.

위상량 단계에서 하이젠버그 벤치마크의 마그논 Berry 곡률은 0이므로 null test가 된다.
0이 아닌 thermal Hall·Chern 수의 기준에는 해석해가 알려진 별도 모델이 필요하다.

**키타에프(후속).** 정확히 풀리는 대상은 LSWT가 아니라 스핀 액체 바닥상태다.

- ED·TN 솔버 단계: `S = 1/2` honeycomb 모델의 Majorana 정확해를 기준으로
  바닥상태 에너지, flux gap, 최근접 이웃에서 끊기는 정적 스핀 상관을 검증한다.
  U(1) 대칭이 없으므로 최소 ED 도구로는 검증할 수 없고, flux 보존량을 쓰는 ED
  솔버가 필요하다.
- LSWT: 고전 키타에프 모델은 바닥상태가 거시적으로 축퇴되어 LSWT 기준 상태가 없다.
  강한 장에서 편극된 키타에프 자석의 마그논 밴드는 0이 아닌 Chern 수를 가지므로,
  위상량 단계에서 0이 아닌 thermal Hall의 기준 모델로 쓸 수 있다(선택 사항).
  이는 ED와의 정확 일치가 아니라 해석적 밴드 공식과 정수 Chern 수의 확인이다.
- 스키마 검증: 비-Bravais honeycomb 격자, 결합 의존 이방성 항, Pauli 규약에서
  `S` 연산자 규약으로의 환산을 함께 확인한다.

## 4. 구현 단계와 통과 조건

단계마다 구현, 수치 대조, 사용자 확인을 기록한 뒤 다음 단계로 진행한다.

| 단계 | 작업 | 통과 조건 |
|---|---|---|
| 0 | 패키지 이름 변경: `lswt` → `spintoolkit`, `methods/spin_wave/` → `methods/lswt/` | 이행 전후 테스트 결과 동일(현재 217 통과 / 5 실패), 40건·4회 수치 대조 차이 0, 옛 import와 저장 객체 복원 |
| 1 | 공통 자료형과 검증: `SpinModel`, `terms`, 상태 검증 | NBCP·사각·삼각 하이젠버그의 작은 fixture를 같은 소비자가 모델명 분기 없이 읽음; 공통 검증 사례 통과 |
| 2 | NBCP 연결 | 1–4 MSL의 기존 호출과 수치 결과 보존; `r_target = r_source - d` 변환 후 `H(k)` 원소 단위 일치; 고전 에너지, k와 -k의 대응 확인 |
| 2b | 영점 에너지 상태 선택(D17, D18) | 고전 manifold 위 궤도 탐색이 NBCP Y 상태에서 `examples/nbcp_y_orbit_axis_check.py`·`nbcp_y_orbit_criteria_check.py`의 축·선택 결과와 λ₆ 기준값을 재현; D19 판정 기준 두 모드; MAGSWT 재현 비교; 구현과 함께 `quantum` 삭제 |
| 3 | 벤치마크와 최소 ED 검증 도구(`methods/ed/`) | 고전 에너지 정확 일치; 완전 편극상 1-마그논 에너지가 같은 토러스의 LSWT와 일치; 문헌값과의 근사 비교는 허용오차와 출처를 명시 |
| 4 | LSWT 물리량 | 스펙트럼, 영점 보정을 포함한 바닥상태 에너지, 열역학량, 상관함수를 벤치마크와 NBCP에서 확인 |
| 5 | 위상량 | Berry 곡률·Chern 수·thermal Hall; 하이젠버그 null test; 0이 아닌 기준 모델 비교(키타에프 편극상은 선택); 기존 real-space volume 이슈 해소 |

**진행 상황.** 0단계는 2026-09-29 완료했다. 테스트는 이전과 같은 결과(호환 테스트 1개를
4개로 교체해 220 통과 / 같은 5 실패)였고, `examples/package_regression_snapshot.py`로
비교한 208개 값(40건의 `H(k)`·고전 에너지와 16개 탐색 결과)의 최대 차이는 0이었다.
기록은 `docs/development/verification/package-rename-2026-09-29.json`에 있다.

1단계는 2026-09-29 완료했다. `SpinModel`·`SpinState`·`ExternalConditions`·
`CalculationGeometry`와 검증, 고전 에너지·국소장·토크 계산(`methods/classical.py`),
벤치마크 해밀토니안과 해석적 기준 스핀 배열(`models/heisenberg.py`)을 추가했다. 같은
고전 에너지 함수가 모델명 분기 없이 사각 Néel `-2JS^2`, 삼각 120° `-3JS^2/2`, 편극상을
정확히 재현하고, NBCP 기본 셀 fixture는 기존 `EnergyFunction`과 1e-15 안에서 일치했다.
기존 테스트 225개의 결과는 그대로이고 새 테스트 53개가 통과했으며(273 통과 / 같은 5 실패),
208개 수치 스냅샷의 차이는 0이었다. 기록은
`docs/development/verification/stage1-common-types-2026-09-29.json`에 있다.

2단계는 2026-09-29 완료했다(D21). `model/nbcp/model.py`에 NBCP 모델(`build_model`), 파라미터
세트(문헌값 `woodland2025`: arXiv:2505.06398 Table 1; 원고값 `park2026_fig4`), 후보 상태
(`candidate_state`)를 두고, `system/conversion.py`로 기존 `SpinSystem`과 양방향 변환한다.
변환한 `H(k)`가 기존 네 셀 생성 함수와 원소 단위로 같아(최대 1.7e-16) D13이 확정되었고,
고전 에너지와 E_qm도 일치했다. 테스트는 348 통과 / 같은 5 실패(새 테스트 72개), 208개
스냅샷 차이 0이다. 기록은 `docs/development/verification/stage2-nbcp-connection-2026-09-29.json`.

2b단계를 2026-09-29 구현하고 검증했다(D22). `methods/classical.py`에 접평면 좌표의 해석적
Hessian(`tangent_expansion`)과 국소 정밀화(`refine_classical`)를, `methods/state_selection.py`에
`select_on_manifold`와 D19의 두 판정 모드를 추가하고, `quantum` 경로를 삭제했다. NBCP Y·V 상태의
J_PD·J_Gamma 16회(N = 6, 12, 두 모드)가 모두 `selected`이고, 조화 진폭과 곡률이 독립 스캔(N = 48)과
N = 6에서 1.1%, N = 12에서 0.2% 안에서 일치했다. Y의 J_Gamma 변동 1.64e-12는 해상 기준의 약
900배로 분해되었다. DE 출발 36회, 삼각격자 120°·정사각 편극 벤치마크, MAGSWT 재현도 확인했다.
테스트는 373 통과 / 같은 5 실패(새 테스트 29개, 삭제한 `quantum` 테스트 8개), 208개 스냅샷 차이
0이다. 기록은 `docs/development/verification/state-selection-stage2b-2026-09-29.json`. 물리 근거
모드의 계수, 장을 기울인 대조군의 `competition` 판정, MAGSWT 유지 여부는 사용자 검토 항목이다.

스펙트럼 일치만으로 올바른 변환이라고 판정하지 않는다. 벡터를 뒤집거나 행렬을
전치하는 규칙은 실제 식과 대응시킨다.

공통 검증에는 다음 사례를 포함한다: 잘못된 단위, 사이트당 중복된 Zeeman 항,
`B != 0`에서 Zeeman 항이 없는 사이트의 경고, 허용 목록 밖의 kind, 존재하지 않는
endpoint, 병진·역방향 중복, 같은 물리 사이트 반복, 분수 셀 이동, 비대칭 J, 같은
basis 사이의 셀 간 결합, 입력 배열 변조, 작은 주기계에서의 항 중복도, 계산계와
정합하지 않는 상태, 행렬식이 0인 자기 초격자, 최소가 아닌 주기의 상태. 수치
허용오차의 값은 구현 단계에서 명시하고 테스트에 고정한다. 물리적 정당성이나
ED/TN 실행 가능성은 별도 검증이다.

**삼각격자 벤치마크의 범위(D18).** `h = 0`의 120° 상태는 축퇴가 대칭에서만 오고 문헌의 LSWT
값과 근사 비교하는 기준이다. `h > h_sat`의 완전 편극 상태는 z축 U(1) 대칭 때문에 정확한 고유상태이고
1-마그논 에너지가 LSWT 분산과 정확히 같아, 같은 토러스의 ED와 기계 정밀도로 일치해야 하는 구현 검증
기준이다(마그논 갭 `h - h_sat`). `0 < h < h_sat`의 고전 바닥상태는 전역 회전이 아닌 우연 축퇴 방향을
가져 현재 D17 범위 밖이므로, 여러 차원 영공간 확장 뒤에 벤치마크로 쓴다.

## Open Questions

- 최소 ED 도구에 2-마그논 섹터를 포함할지 여부(3단계).
- 키타에프 편극상을 5단계의 0이 아닌 thermal Hall 기준 모델로 사용할지 여부.

## 변경 이력

- 2026-09-29 (claude): SpinModel 전달 규약에서 1차 범위·패키지·벤치마크·구현 단계를
  분리해 작성했다. 내용은 분리 전 규약 문서와 같다.
- 2026-09-29 (claude): 0단계(패키지 이름 변경) 완료와 검증 결과를 기록하고, 배포 이름
  `spin-toolkit`과 `lswt` 호환 패키지의 위치를 반영했다.
- 2026-09-29 (claude): 1단계 완료와 검증 결과를 기록했다.
- 2026-09-29 (claude): D17을 2단계 뒤 2b 단계로 배치하고 삼각격자 벤치마크 범위(D18)를 적었다.
- 2026-09-29 (claude): 2단계 완료와 검증 결과를 기록했다.
- 2026-09-29 (claude): 2b단계 구현과 검증 결과를 기록했다.

## 관련 기록

- [[GOVERNMENT/Working-Pad/idea-proposals/map-idea-proposals|제안 인덱스]]
- [[GOVERNMENT/Working-Pad/idea-proposals/2026-09-23-spin-model-transfer-contract|SpinModel 전달 규약]]
- [[GOVERNMENT/Working-Pad/TASK-QUEUE|활성 작업 인덱스]]
