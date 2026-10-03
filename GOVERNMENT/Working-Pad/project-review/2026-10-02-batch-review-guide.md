# 일괄 검토 안내 (2026-10-02)

구현이 끝났으니, 그동안 "사용자 검토 대기"로 둔 항목을 한 번에 보실 수 있게 모았습니다. 순서는 PyPI 공개 전에 꼭 확인이 필요한 것부터입니다. 각 항목은 "무엇을 결정했나 → 물리적으로 봐 주실 점 → 위치" 순서입니다. 코드 수치 일치는 이미 확인했지만, 물리·수학 판단은 사용자 검토로만 확정됩니다.

검토 결과는 항목 번호와 함께 "1 OK", "3 질문: …"처럼 짧게 주시면 됩니다.

## A. 공개 전에 꼭 볼 것 (패키지 동작을 정하는 결정)

### A1. M(h)의 1/S 보정 (D45)
- 결정: M = −d(E_cl + E_zp)/dh, 중심 차분. g(S−⟨n⟩)cosθ 근사는 쓰지 않음.
- 볼 점: 기울기 각의 1/S 이동 항 −gS sinθ·δθ가 같은 차수라는 유도. 오늘 iDMRG, ED(무한·같은 클러스터), 사이트별 ED·DMRG 비교 결과: 1,132개 중 1,096개에서 더 가까움, 예외는 열린 클러스터 모서리 근처.
- 위치: 튜토리얼 4 `docs/tutorials/04-magnetization.md`, 데이터 `/mnt/project-files/roadmap/magnetization/`

### A2. 단일 이온 항의 (1 − 1/2S) 규칙 (D37)
- 결정: 고전·LSWT 모두 결맞음 상태 값 S(S−½)nᵀAn + (S/2)trA, 즉 계수 (1−1/2S)A. 정규 순서 상수는 더하지 않음.
- 볼 점: S=1/2에서 단일 이온 항이 상수가 되는 것, D(S^z)² 갭 (2S−1)|D|가 정확해지는 것. 문헌과 다른 코드마다 이 항의 관례가 달라, 같은 입력이라도 값이 다를 수 있음.
- 위치: `docs/development/appendices/b-decisions-references.tex` D36–D37 프레임

### A3. Thermal Hall: full-position Bloch 규약 (D29, 이슈 260802)
- 결정: κ_xy는 각 사이트의 실제 위치로 위상을 잡는 규약으로 계산. Chern 수는 두 규약에서 같음.
- 볼 점: 이 규약이 유일하게 물리적이라는 유도(부격자를 다른 셀로 옮겨 적으면 셀 규약 값만 바뀜).
- 위치: `GOVERNMENT/Working-Pad/issue-notes/open/260802-topology-thermal-hall-real-space-volume-bug.md`, 튜토리얼 5

### A4. 중성자 세기 정의 (D41)
- 결정: I = Σ(δ−Q̂Q̂)S_M^{ab}, S_i → ½gF(|Q|)S_i. (γr₀)²k_f/k_i e^{−2W}는 제외. Bragg 벡터의 Goldstone 비탄성 세기는 NaN.
- 볼 점: 2D 층에서 Q_z는 편극 인자와 형상인자에만 들어가는 것. 형상인자 표(3d 이온, 쌍극자 근사).
- 위치: D41 프레임, 튜토리얼 3

### A5. 공개 API와 옛 API 삭제 일정 (D43, 이미 결정하심)
- 확인만: 0.2에서 경고, 0.3에서 삭제.

### A6. 영어 튜토리얼 5편 (PR #28, 오늘 병합)
- 볼 점: 설명이 물리적으로 맞는지, 빠진 주제가 있는지.
- 위치: `docs/tutorials/README.md`

## B. 방법론 결정 (Claude가 위임받아 정한 것)

| 번호 | 결정 | 물리적으로 봐 주실 점 | 위치 |
|---|---|---|---|
| B1 | D34 단일 Q 나선 회전틀 LSWT | 축에 대해 U(1)이 아닌 모델은 근사 대신 거부; 위상은 셀 번호로만 | `GOVERNMENT/Working-Pad/idea-proposals/2026-10-01-spiral-rotating-frame-lswt.md` |
| B2 | D35 결정 대칭과 대칭 허용 결합 | 스핀은 축 벡터 R_s = det(R)R; 시간 반전은 군에 넣지 않음; 대칭을 깨는 계수는 대칭화하지 않고 거부 | D35 프레임 |
| B3 | D38 고전 동역학·Langevin | 잡음 세기 2αT/S, Stratonovich 해석, 정상 분포 e^{−E/T} | D38 프레임 |
| B4 | D39 유한 온도 조화 자유 에너지 f(φ,T) | 부드러운 모드 차단 Λ를 위상 이론으로 넘기는 방식 | D39 프레임 |
| B5 | D40 고전 MC와 비틀림 강성 | 축에 대해 U(1)이 아닌 교환은 거부하고 U(1) 부분 강성만 제공 | D40 프레임 |
| B6 | D42 스커미온 수와 후보 비교 | 공면 120°처럼 입체각이 모호하면 "정의 안 됨"; 후보 비교는 주어진 후보 안의 순위일 뿐 바닥상태 판정이 아님 | D42 프레임 |
| B7 | D44 그림 묶음 | 정의되지 않는 값은 회색·빗금으로 표시하고 0으로 채우지 않음 | D44 프레임, `examples/gallery.py` |

## C. 이론 문서와 기록

| 번호 | 내용 | 위치 |
|---|---|---|
| C1 | LSWT 이론 draft 10편 물리·수학 검토 (독립 검산으로 오류 14건 수정 완료, 남은 의심 항목 정리됨) | 검토 PDF `/mnt/project-files/lswt-review/2026-10-01-lswt-drafts-review.pdf`, 가이드 `GOVERNMENT/Working-Pad/issue-notes/open/261001-lswt-draft-physics-review-guide.md` |
| C2 | 솔버 seam 스파이크 (ED·TeNPy·NetKet이 같은 토러스에서 1e-14 일치) → closed로 옮겨도 되는지 | `GOVERNMENT/Working-Pad/handoff/open/260607-solver-seam-spike.md` |
| C3 | 개발 설계 Beamer (D30–D45 결정 기록 포함) | 원본 `docs/development/main.tex`; PDF는 저장소에 없어 원하시면 만들어 드립니다 |

NBCP 논문 점검 항목(Rau 공식 적용 범위, V의 Z₆, Fig. 4(b) 출처 등)은 "NBCP 연구 문서 점검·정교화" 스레드에서 따로 정리돼 있어 여기서는 뺐습니다.

## 검토 뒤 순서

검토에서 바뀌는 것을 반영한 뒤, 버전을 0.2.0으로 정하고 TestPyPI에 먼저 올려 설치를 확인한 다음 PyPI에 공개합니다. PyPI 공개는 되돌릴 수 없는 작업이라 사용자가 직접 "공개"라고 써 주실 때만 진행합니다.
