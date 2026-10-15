# spin-toolkit와 다른 스핀파 패키지 비교 (2026-10-15)

비교 대상은 SpinW(MATLAB, 새 Python판 pySpinW), Sunny.jl(Julia), SpinWaveGenie(C++), magnopy·pyMagCalc·topwave(Python)다. spin-toolkit는 `main` b4cdc46 기준이다.

## 1. 요약

- **수치 일치:** 클라우드에서 실제로 돌릴 수 있는 magnopy 0.6.1과 같은 모형 9개를 비교했다. 단일 이온 항이 없는 모형 7개는 magnon 띠와 영점 에너지가 기계 정밀도(1e-14)로 같았다.
- **다른 점:** 단일 이온 항 D(S^z)²에서만 값이 갈린다. spin-toolkit는 (1 − 1/2S) 보정(D37)으로 단일 스핀의 정확한 들뜸 에너지 D(2S−1)을 낸다. magnopy는 보정이 없는 HP를 써서 2DS를 낸다. SpinW도 기본적으로 보정하지 않는 것으로 안다(미확인). Sunny의 dipole 모드는 같은 종류의 재규격화를 쓴다.
- **문헌값:** SpinW가 공개한 1/S 모멘트 감소값과도 맞는다. 정사각 격자 0.1966, 삼각 격자 0.2612이고, 우리 값은 메시 192²에서 0.1953과 0.2601이며 메시를 키울수록 그 값으로 다가간다.
- **앞선 점:** 선형 너머 계산을 갖춘 공개 패키지는 찾지 못했다. 1/S 에너지와 magnon 보정, magnon 붕괴, 2-magnon 연속체, O(S⁰) S(q,ω)가 그렇다. Berry 곡률·Chern 수·열 홀을 한 흐름으로 묶은 공개 패키지도 찾지 못했다.
- **뒤처진 점:** 실험 쪽 기능이다. 측정 데이터 맞춤, 장비 분해능, 편극 중성자, CIF와 공간군 입력이 없다. 장거리 쌍극자(Ewald)와 SU(N)(다극자)도 없다.
- **다른 점(설계):** 2D만 다루고, 정의되지 않는 양은 NaN으로 낸다(D61). 다른 패키지는 3D 일반이고, 불안정한 상태에도 숫자를 내는 경우가 많다.

## 2. magnopy 수치 교차검증

spin-toolkit의 `SpinModel`과 `SpinState`를 magnopy 해밀토니안으로 자동 변환하는 변환기(`data-space/verification/261015-magnopy-comparison/convert.py`)를 썼다. magnetic BZ의 무작위 k 300개에서 띠를 비교했고, 같은 메시에서 영점 에너지(ZPE)를 비교했다. ZPE는 스핀당 값이다.

| 모형 | 사이트/셀 | max\|Δω\| | ZPE spin-toolkit | ZPE magnopy |
|---|---|---|---|---|
| 정사각 AFM S=1/2, Néel | 2 | 6e-15 | −0.157943 | −0.157943 |
| 정사각 AFM S=1/2, h = h_sat/2에서 기운 상태 | 2 | 1e-14 | −0.084962 | −0.084962 |
| 삼각 AFM S=1/2, 120° | 3 | 5e-15 | −0.163807 | −0.163807 |
| 벌집 FM + 2번째 이웃 DM (D=0.1) | 2 | 3e-15 | 0 | 0 |
| 정사각 FM, 비등방 J(비대각 포함) + 장, S=1 | 1 | 7e-15 | −0.009763 | −0.009763 |
| 같은 모형, S=5/2 | 1 | 2e-14 | −0.025802 | −0.025802 |
| 위 모형 + 단일 이온 −0.3(S^z)², S=1 | 1 | **0.32** | −0.008651 | −0.007809 |
| 위 모형 + 단일 이온, S=5/2 | 1 | **0.33** | −0.021162 | −0.020302 |
| J1–J2 원뿔형 나선 (q = 2π/5, 장 있음), S=1 | 1(회전 좌표계) / 5(magnopy 초격자) | 3e-14 | −0.076900 | (같은 초격자 값) |

**단일 이온 항 판정.** 교환 없이 단일 이온 항과 장만 둔 스핀으로 확인했다. 정확한 답은 h + |D|(2S−1)이다.

| S | 정확 | spin-toolkit | magnopy |
|---|---|---|---|
| 1 | 0.5 | 0.5 | 0.8 |
| 3/2 | 0.8 | 0.8 | 1.1 |
| 5/2 | 1.4 | 1.4 | 1.7 |

spin-toolkit가 정확한 값을 낸다. magnopy의 차이는 |D|로 일정하다. 비교 모형에서 Δω가 |D|보다 조금 큰 이유는 비대각 교환이 만드는 anomalous 항(ω = √(A² − |B|²)) 때문이다.

**나선 판정.** 회전 좌표계 결과(`solve_spiral_lswt`)의 ω(k + nQ)(n = 0…4)를 모은 값이 magnopy 5-사이트 초격자 띠와 같았다. 회전 방향을 반대로 둔 상태와 비교하면 0.77만큼 달라진다. 따라서 이 일치는 우연이 아니다.

**문헌값.** 스핀당 바닥 에너지는 정사각 −0.65795, 삼각 −0.53881이다. 문헌값 −0.658과 −0.539(Chernyshev–Zhitomirsky)와 맞는다. 모멘트 감소는 위 요약에 적었다. Goldstone 근처의 로그 수렴 때문에 메시 수렴이 느리다(96² → 192²에서 0.0013 증가). SpinW는 무작위 Q 10⁷개를 쓴다.

**돌리지 못한 것.** Sunny(Julia)와 SpinW(MATLAB)는 직접 돌리지 못했다. 클라우드 네트워크 정책이 Julia 다운로드(julialang.org)를 막고, MATLAB도 없다. 두 패키지와의 비교는 문서와 공개 수치 기준이다. Sunny를 직접 비교하려면 Julia가 있는 맥에서 하거나 환경 네트워크 허용 목록에 julialang.org를 추가해야 한다.

## 3. 기능 비교

Y는 있음, N은 없음, ?는 확인하지 못함이다. 다른 패키지의 항목은 2026-10-15에 웹 문서에서 조사했다.

| 기능 | spin-toolkit | SpinW | Sunny.jl | magnopy | pyMagCalc |
|---|---|---|---|---|---|
| 언어 / 라이선스 | Python / MIT | MATLAB GPL-3, pySpinW 알파(Rust 핵심) | Julia / MIT | Python / GPL-3 | Python / MIT |
| 차원 | 2D만 | 3D | 3D | 3D | 3D |
| LSWT 정합 구조, 초격자 | Y | Y | Y | Y | Y |
| 단일-Q 나선·원뿔(회전 좌표계) | Y (D34) | Y | Y | ? | Y |
| SU(N)·다극자, entangled units | N | N | Y | N | N |
| 단일 이온 (1−1/2S) 보정 | Y (D37) | 기본은 N로 앎(미확인) | Y (dipole 모드 재규격화) | N (검증함) | ? |
| 고전 바닥 상태 탐색 | Y (최적화, Luttinger–Tisza) | Y (`optmagk`, 담금질) | Y | Y | ? |
| 고전 MC | Y | Y | Y (Wang–Landau, 병렬 템퍼링 포함) | N | N |
| Landau–Lifshitz / Langevin, 고전 S(q,ω) | Y (D38) | N | Y (SU(N) 포함) | N | Y |
| 국소 안정성 판정, 상태 비교 | Y (D51, D53) | 부분 | 부분 | N | N |
| 1/S 바닥 에너지, magnon 보정 | Y (D46) | N (모멘트 감소만) | N (고전 극한 재규격화만) | N | 영점 에너지와 모멘트만 |
| magnon 붕괴, 2-magnon 연속체 | Y (D56, D57) | N | N | N | N |
| O(S⁰) 전체 S(q,ω), 스펙트럼 함수 | Y (D58, D59) | N | N | N | N |
| Berry 곡률, Chern 수, 열 홀 | Y | N | N | N | N |
| 유한 온도 magnon 열역학, M(h,T) | Y (D49) | MC만 | MC만 | N | N |
| 중성자 I(Q,ω), 형상 인자, 분말 평균 | Y (D41, 비편극) | Y | Y | N | Y |
| 편극 중성자 | N | Y | ? | N | ? |
| 장비 분해능, Horace 연동 | N (가우스 넓힘만) | Y | 넓힘만 | N | ? |
| 데이터 맞춤 | N | Y | Y (v0.9) | N | Y (분말) |
| CIF·공간군, 대칭 허용 결합 | 부분 (D35 층 대칭) | Y | Y (CIF, mCIF) | N (TB2J 입력) | ? |
| 장거리 쌍극자 (Ewald) | N | ? | Y | Y (쌍극자 항) | ? |
| 큰 셀용 KPM/Lanczos | N | N | Y | N | N |
| 정의되지 않는 양 → NaN | Y (D61) | N | N | N | N |
| 작은 ED 검증 | Y | N | N | N | N |

SpinWaveGenie(C++, ORNL)는 2015년 이후 사실상 멈춰 있다. 기능은 LSWT, 분해능 합성곱과 q 영역 적분 정도라 표에서 뺐다. topwave(Python, 알파)는 Wilson loop로 위상 불변량을 계산하지만 Berry 곡률이나 열 홀 기능은 문서에 없다.

## 4. 이 비교가 말하는 것

1. **선형 LSWT 핵심은 독립 구현과 정확히 맞는다.** 비대칭 DM, 비대각 교환, 기운 상태, 3-부격자, 원뿔 나선까지 맞는다. 이 부분은 공개해도 될 만큼 믿을 만하다.
2. **단일 이온 관례는 우리 쪽이 물리적으로 옳다.** 다만 SpinW나 magnopy와 결과를 비교하는 사용자는 이 차이에 놀랄 수 있다. 튜토리얼에 "SpinW·magnopy와 비교할 때는 A를 (1−1/2S)로 나눠 넣으면 같아진다"는 안내가 있으면 좋다.
3. **차별점은 선형 너머와 위상이다.** 1/S 보정, magnon 붕괴, 2-magnon 연속체, O(S⁰) S(q,ω), Chern·열 홀을 한 패키지에서 하는 곳은 찾지 못했다. 공개할 때 내세울 부분이다. 대신 이 부분은 다른 패키지와 직접 비교할 수 없으니 논문 수치와 대조하는 검토가 더 중요하다(잠정 승인 D46, D56–D60).
4. **실험 연구자 입장에서는 SpinW와 Sunny가 앞선다.** 맞춤, 분해능, 편극, CIF, 3D 지원이 그렇다. 2D 범위 결정(D48)을 지키면 3D는 일부러 하지 않는 것이다. 편극 중성자와 맞춤은 범위 안에서 더할 수 있는 후보다.
5. **수렴 속도.** Goldstone 모드가 있을 때 균일 메시로 모멘트 감소를 계산하면 수렴이 느리다(192²에서 아직 0.0013 모자람). 적응형 적분이나 Goldstone 근처 해석적 처리가 개선 후보다.

## 5. 다시 돌리기

스크립트와 결과(`results.json`)는 `data-space/verification/261015-magnopy-comparison/`에 있다. magnopy는 패키지 의존성이 아니라 따로 설치한다.

```bash
pip install magnopy==0.6.1
cd code-space
python ../data-space/verification/261015-magnopy-comparison/run_compare.py
python ../data-space/verification/261015-magnopy-comparison/spiral_compare.py
```

## 출처

- SpinW: https://spinw.org/spinwdoc/features · https://spinw.org/spinw_moment · https://github.com/spinw/pySpinW
- Sunny: https://github.com/SunnySuite/Sunny.jl · arXiv:2501.13095 · JOSS 10.21105/joss.08138
- magnopy: https://docs.magnopy.org · pyMagCalc: https://github.com/kit-matan/pyMagCalc · topwave: https://topwave.readthedocs.io · SpinWaveGenie: https://github.com/SpinWaveGenie/SpinWaveGenie
- 삼각 격자: Chernyshev & Zhitomirsky, PRB 79, 144416 (2009)
