---
frontmatter-version: 1
title: LSWT legacy B/B† block swap — existing fix verified
section: issue-notes/closed
issue-type: problem
status: closed
resolution: resolved
outcome: code-space/tests/test_solvers/test_hamiltonian_pairing.py
last-edited-by: codex
created: 2026-08-02
updated: 2026-09-10
closed: 2026-09-10
source: research-space/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf
related:
  - code-space/lswt/solvers/hamiltonian.py
  - legacy/modules/LinearSpinWaveTheory/lswt_Hamiltonian.py
  - GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md
must-read: GOVERNMENT/Agents-Bylaws/templates/issue-notes-template.md
---

# LSWT legacy B/B† block swap — existing fix verified

## 배경

2026-08-02 과거 CLAUDE.md/AGENTS.md의 "알려진 버그"를 이슈로 옮겼다.
당시 기록은 **현재 패키지**의 B/B† 배치가 틀렸다고 기술했지만, 2026-08-10
보완에서는 실제 원인이 block 배치인지 index/phase pairing인지 재확인하지
않았으므로 `Unknown`이라고 명시했다. "SM v5 수정 코드" 확인과 회귀 테스트도
미완료였다. 이 설명을 근거로 현재 구현을 다시 뒤집어서는 안 된다.

2026-09-10 arXiv:2601.20963의 V상 각도별 영점에너지를 확인하는 과정에서,
동일한 스핀 방향·결합·운동량 격자를 넣어도 legacy와 현재 패키지의 이방성
진폭이 달랐다. 사용자 요청에 따라 원본 노트, 내부 검증 노트, 실제 코드와
Git 이력을 대조하고 독립적인 스핀 행렬 기반 회귀 테스트를 추가했다.

## 문제 정의

보존된 legacy 구현은 쌍생성 계수를 B† 위치에 넣는다. 현재 패키지에는
2026-06-02 커밋에 이미 올바른 배치가 반영돼 있었다. 이번에 해결한 범위는
**현재 구현을 버그로 지목한 기록의 정정, 기존 수정의 독립 검증, 재발 방지**다.
이번 작업에서 production Hamiltonian과 legacy 원본의 수식은 변경하지 않았다.

## 본론

### 증상과 재현 조건

고정한 스핀 방향과 결합에서 두 구현의 정상 블록 A는 같지만, legacy의
오른쪽 위 블록은 현재 코드의 B†와 같다. B와 B†가 다르면 조립된 행렬이
달라진다. **B = B†는 이 교환의 영향을 없애는 충분조건**이다.
단순히 B가 복소수이거나 스핀이 비공선이라는 사실만으로 스펙트럼 차이를
단정할 수는 없다. 행렬이 달라도 특별한 대칭에 의해 같은 고유값을 가질 수 있다.

앞선 V상 확인에서는 PD-only, Gamma-only, 혼합 SOC에 대해 행렬과 이방성
진폭 차이를 관찰했다. 이번 영구 회귀 테스트는 재현이 작은 합성 모델을
사용하며, 특정 물질의 바닥상태나 유한온도 상을 검증하는 테스트가 아니다.

### 근본 원인: 계수의 의미와 Nambu 위치

Nambu column을 Psi_k = (a_k, a†_-k)로 정하면

$$
H_k=\begin{pmatrix}A_k&B_k\\B_k^\dagger&A_{-k}^*\end{pmatrix}
$$

의 오른쪽 위 B는 a†_k a†_-k, 즉 쌍생성 항이다. 양쪽 코드의
`hop_t = C† (R_i^T J_ij R_j) C`와 `tpp = sqrt(S_i S_j) * hop_t[1,0]`는 같다.
코드가 쓰는 C에서 q = C† S_local의 첫 성분은 S-/sqrt(2), 따라서 생성
연산자에 대응한다. 실제 bilinear의 계수 행렬은 C^T RJ C = P hop_t이며,
P는 첫 두 행을 교환한다. 그러므로 쌍생성 계수는 `hop_t[1,0]`이다.

한 bond의 같은 upper-right 원소를 비교하면 다음과 같다.

| 구현 | H[i, Ns+j]에 들어가는 항 |
|---|---|
| 현재 `code-space/lswt/solvers/hamiltonian.py` | tpp exp(-i k·delta) |
| 보존된 `legacy/modules/LinearSpinWaveTheory/lswt_Hamiltonian.py` | tpp* exp(-i k·delta) |

정상항 A는 그대로이므로 이는 전체 기저를 일관되게 변환한 결과가 아니다.
실수 Cartesian 결합에서도 local rotation과 원형 성분 변환 뒤의 tpp는
복소수가 될 수 있고, 이때 정상항과 쌍생성 항의 상대 위상이 바뀐다.

이번 코드 검증은 `a_r = sum_k exp(-i k·r) a_k`를 명시해서 사용한다.
`docs/01-derivation/momentum-space-bdg-hamiltonian.md`의 Fourier/gauge 검토는
별도로 남는다. 여기서 코드 convention을 고정한 사실은 이론 문서의 부호
선택이나 acceptance를 확정한 것이 아니다.

### 수정 이력과 노트의 증거 범위

| 근거 | 확인한 내용 |
|---|---|
| `301a3c5` (2026-03-22), 당시 `lswt/solvers/hamiltonian.py` | 이전 B/B† 배치가 존재한다. |
| `c735573648e9129607cbde7a493ad9270bb287e0` (2026-06-02) | 현재 배치로 변경됐다. dH/dkx, dH/dky도 같은 커밋에서 변경됐다. |
| `research-space/sources/03-verification-notes/hamiltonian_convention.tex`, sections 6 and 8 | 현재 코드의 pair creation 배치를 맞다고 설명하고 legacy의 교환을 지적한다. 내부 검증 자료이며 theory canon은 아니다. |
| 원본 PDF 8쪽, 식 (40), (44), (45) | Nambu 순서와 B/B†의 위치를 확인할 수 있다. |
| 원본 PDF 9쪽, 식 (46) 부근 SJ/SP 토의 | 켤레 표기에 대한 오탈자 토의와 "The code is implemented without this typo."라는 답변이 있다. 특정 commit이나 파일 버전은 제시하지 않는다. |

이력은 다음 비교로 재확인할 수 있다.

```sh
git diff 301a3c5:lswt/solvers/hamiltonian.py c735573:code-space/lswt/solvers/hamiltonian.py
```

"SM v5 수정 코드"는 저장소에서 확인한 별도 source artifact가 아니다.
그 명칭과 c735573의 동일성은 **Unknown**으로 남긴다. 현재 배치의 판정은
그 명칭을 추정하지 않고 실제 diff와 아래의 독립적인 연산자 대조에 근거한다.
같은 커밋의 linear-term 수정은 이번 B/B† 이슈의 검증 범위에 포함하지 않는다.

### 회귀 테스트와 독립적인 기준값

`code-space/tests/test_solvers/test_hamiltonian_pairing.py`에 8개 테스트를 추가했다.
작은 유한차원 spin matrix를 직접 만들고 다음 행렬 원소를 추출한다.

- vacuum에서 두 스핀을 각각 한 단계 낮춘 상태로 가는 원소: pair creation.
- 한 magnon이 다른 site로 이동하는 원소: normal hopping.
- one-magnon diagonal energy에서 vacuum energy를 뺀 값: quadratic local term.

이 기준값은 production의 rotation, C matrix, `get_couplings()`를 호출하지
않는다. 실수 쌍생성 대조군과 복소 쌍생성 사례에 대해 서로 다른 spin 크기
(1/2, 1, 3/2), 여러 bond 방향, 다른 cell의 같은 sublattice를 잇는 bond를
포함한다. 실제 숫자와 전체 fixture는 테스트 파일에 고정돼 있다.

| 검증 | 결과 |
|---|---|
| 현재 전체 H(k)와 독립적인 spin-matrix 기준 비교 | 2건 통과; 최대 원소 오차 4.45e-16 |
| dH/dkx, dH/dky와 독립 기준의 중심차분 비교 | 4건 통과; step 1e-5, 최대 원소 오차 3.67e-11 |
| 보존된 legacy 직접 실행 및 B/B† 교환의 영향 분리 | 2건 통과 |
| legacy의 두 비대각 블록만 메모리상 교환한 뒤 현재 H와 비교 | 두 fixture 모두 최대 원소 차이 0 |
| 복소 fixture의 legacy 대 기준 H 차이 | 최대 0.0613734 |
| 복소 fixture의 legacy 대 기준 magnon 고유값 차이 | 최대 0.000348959; 두 quadratic H 모두 양정치이며 인위적 shift 없음 |

Fixture의 exchange와 field coefficient는 같은 임의 에너지 단위를 사용한다.
이 표의 수치는 물질의 실측값이나 논문 재현 오차가 아니다.

**Hermiticity와 B(-k) = B(k)^T만으로는 이 오류를 잡지 못한다.** 실제
legacy 구현은 복소 fixture에서도 두 조건을 통과하지만, 독립적인 계수 기준과
비교하면 실패한다. 이것이 기존 구조 검사에 이번 테스트를 추가한 이유다.

오류 검출력을 확인하기 위해 production 파일을 수정하지 않고 별도 프로세스의
메모리에서 해당 method만 legacy method로 교체해 실행했다.

- `Quadratic_Bose_Hamiltonian` 교체: 복소 기준 테스트 1건 실패, 실수 대조군 1건 통과.
- `partial_derivatives_of_Hk` 교체: 복소 kx/ky 테스트 2건 실패, 실수 대조군 2건 통과.

이는 의도적인 negative control이며 정상 구현의 테스트 실패가 아니다.

### 실행과 전체 테스트 기준선

프로젝트 의존성이 설치된 Python에서 저장소 루트를 기준으로 실행한다.

```sh
PYTHONPATH=code-space python -m pytest code-space/tests/test_solvers/test_hamiltonian_pairing.py -q -p no:cacheprovider
PYTHONPATH=code-space python -m pytest code-space/tests -q -p no:cacheprovider --tb=short
```

검증 환경은 Python 3.13의 로컬 Miniconda runtime이다.
추가 전 전체 테스트는 **34 passed, 5 failed**, 추가 후는 **42 passed, 5 failed**다.
기존 5건은 모두 `CommensurateStructure`의 sublattice 수/angles shape 계약에
관련된 같은 실패다. 기존 owner는
`GOVERNMENT/Working-Pad/issue-notes/open/260602-commensurate-structure-test-failures.md`이며,
이번 작업에서 해당 API나 테스트를 변경하지 않았다. Legacy를 import할 때의
기존 docstring escape `SyntaxWarning` 2건도 보존된 원본에서 발생한다.

`examples/nbcp_hamiltonian_check.py`의 구조 진단은 B의 복소수 여부로 고유값
변화를 단정하지 않도록 B와 B†의 실제 차이를 표시하고, 계수 검증은 위 회귀
테스트로 안내한다. 이 예제 전체의 MAGSWT 최적화는 이번에 다시 실행하지 않았다.
변경한 진단 구간은 복소 Hermitian B, 실수 비대칭 B, 복소 non-Hermitian B의
세 입력으로 따로 실행해 메시지가 실제 B/B† 차이를 따르는지 확인했다.

## 결론 / 미결 사항

**이 구현 이슈는 종결한다.** 현재 코드에 이미 반영된 B/B† 수정과 두 운동량
미분의 배치를 검증했고, legacy로 되돌리면 실패하는 회귀 테스트를 추가했다.
현재 구현을 다시 뒤집을 필요가 없다. Legacy는 과거 결과의 출처를 추적할 수
있도록 수정하지 않았으며, 앞으로의 계산은 현재 패키지를 기준으로 수행한다.

다음 항목은 이 종결에 포함되지 않는다.

- **A1 및 theory acceptance:** 원본 식 (46)의 표기·명시적 block 식 검토는
  documentation audit에서 계속 `open`이다. 자동 테스트는 사용자의 Human
  Physics and Mathematics Review를 대체하지 않는다. 내부 검증 노트 전체의
  서술 정확성이나 모든 LSWT 수식을 이번 결과로 승인하지 않는다.
- **과거 결과 재계산:** 논문에 사용된 정확한 코드 버전·parameter·원자료를
  확인하지 않았으므로 논문 결과의 오류나 재현 완료를 선언하지 않는다.
  기존에 관찰한 각도 주기와 진폭 차이의 원인을 설명한 범위다.
- **다른 물리·구현 문제:** linear term, MAGSWT 최적화, 유한온도 BKT/초고체,
  topology observable 전체 및 별도 CommensurateStructure 실패는 미검증 또는
  각자의 기존 이슈 범위다.

## 참조

- `code-space/lswt/solvers/hamiltonian.py` — 현재 구현; `Quadratic_Bose_Hamiltonian`, `partial_derivatives_of_Hk`
- `code-space/tests/test_solvers/test_hamiltonian_pairing.py` — 재현 가능한 fixture와 회귀 테스트
- `legacy/modules/LinearSpinWaveTheory/lswt_Hamiltonian.py` — 수정하지 않은 이전 구현
- `research-space/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf` — primary evidence, 8–9쪽
- `research-space/sources/03-verification-notes/hamiltonian_convention.tex` — 독립적인 기존 convention 토의
- `docs/01-derivation/momentum-space-bdg-hamiltonian.md` — 이론 작성면, draft 유지
- `GOVERNMENT/Working-Pad/issue-notes/open/260809-lswt-documentation-audit.md` — A1 및 이론 검토 상태
- `GOVERNMENT/Working-Pad/issue-notes/map-issue-notes.md` — 종결 이슈 인덱스
