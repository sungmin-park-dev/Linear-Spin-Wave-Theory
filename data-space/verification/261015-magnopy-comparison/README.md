# magnopy 교차검증 (2026-10-15)

spin-toolkit LSWT 결과를 독립 패키지 magnopy 0.6.1과 비교한 스크립트와 결과다. 해석은 `docs/reviews/2026-10-15-spin-wave-package-comparison.md`에 있다.

| 파일 | 내용 |
|---|---|
| `convert.py` | `SpinModel`·`SpinState`를 magnopy 해밀토니안으로 바꾸는 변환기 |
| `run_compare.py` | 정합 모형 8개의 띠(magnetic BZ 무작위 k 300개)와 영점 에너지 비교, `results.json`에 저장 |
| `spiral_compare.py` | 원뿔 나선: 회전 좌표계 `solve_spiral_lswt` 대 magnopy 5-사이트 초격자 (회전 방향 반대 상태는 대조군) |
| `results.json` | `run_compare.py` 결과 |

결과: 단일 이온 항이 없는 모형은 띠와 영점 에너지가 1e-14 안에서 같다. 단일 이온 D(S^z)²가 있으면 |D|만큼 다르다. spin-toolkit는 (1 − 1/2S) 보정(D37)으로 정확한 D(2S−1)을 내고, magnopy는 2DS를 낸다.

실행은 `code-space/`에서 `pip install magnopy==0.6.1` 뒤 `python ../data-space/verification/261015-magnopy-comparison/run_compare.py`.
