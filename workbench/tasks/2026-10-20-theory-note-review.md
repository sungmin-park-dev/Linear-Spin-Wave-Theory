---
title: 이론 노트 검토 (draft → accepted)
topic: fundamental-theory-of-spin-wave-theory
priority: 1
created: '2026-10-20'
state: active
task: 이론 노트 12편을 주제 순서대로 하나씩 원문 PDF, 인용 문헌, 패키지 계산과 대조하고, 고친 것과 남은 것을 적은 뒤 사용자가 accepted를 정한다.
end-condition: 12편 모두 사용자 판정(accepted 또는 고칠 것)이 이 표에 적혀 있다.
references:
- workbench/tasks/2026-10-13-batch-review.md
- workbench/tasks/2026-08-09-lswt-documentation-audit.md
- workbench/rules/lswt-canonical-document-lifecycle.md
---

## 진행 표

| 순서 | 노트 | 대조 | 사용자 판정 |
|---|---|---|---|
| 1 | `lswt-spin-hamiltonian` | 2026-10-20 끝 | accepted (2026-10-20) |
| 1a | `lswt-conventions` (1의 Notation 절을 옮긴 것) | 2026-10-20 끝 | accepted (2026-10-20, 옮기기 전 판정 그대로) |
| 2 | `lswt-classical-order` | | |
| 3 | `lswt-boson-hamiltonian` | | |
| 4 | `lswt-paraunitary-diagonalization` | | |
| 5 | `lswt-magnon-thermodynamics` | | |
| 6 | `lswt-spin-correlations` | | |
| 7 | `lswt-magnon-topology` | | |
| 8 | `lswt-nonlinear-spin-waves` | | |
| 9–12 | `lswt-example-*` 4편 | | |

검토 상태는 노트 안 각 절 머리 주석(`Review status: ...`)에 적혀 있다. accepted로 바꿀 때는 그 주석을 고친다.

## 1. lswt-spin-hamiltonian (2026-10-20)

세 절: Spin-Wave Theory Overview, Notation and Conventions, Bilinear Spin Hamiltonian. 검토 뒤 2026-10-20 사용자 코멘트 c-20261019-1006에 따라 Notation and Conventions 절을 둘로 나눴다: 물리 관례(첨자, 격자, 링크, 국소 틀, Nambu·BdG, 에너지, 단위, 브릴루앙 영역 합, 띠 번호, 열린 관례)는 새 노트 `lswt-conventions`, 글꼴 규칙과 원문 기호 대응표는 규칙 `workbench/rules/lswt-notation.md`. 글자는 바꾸지 않았고, 다른 노트 8편의 링크를 새 노트로 고쳤다.

### 대조한 것

- **원문 PDF p. 3, Source Eqs. (1)–(3):** 해밀토니안, 단일 이온 예 $-A\sum(S^x)^2$, $\mu_B=0.057883$ meV/T, $g^{\alpha\beta}$의 첫 첨자가 장 성분인 점이 노트와 같다. 노트가 원문과 다르게 쓴 것은 (가) 링크 하나씩 세는 합과 순서쌍 합의 관계(Eq. 3), (나) 고전 에너지를 결맞음 상태 기댓값으로 정의해 단일 이온 항이 $S(S-\tfrac12)\mathbf n^{\mathsf T}\mathsf D\mathbf n+\tfrac S2\operatorname{Tr}\mathsf D$가 되는 점(D37, 원문은 $S^2\mathbf n^{\mathsf T}\mathsf D\mathbf n$), (다) 스핀 1/2에서 단일 이온 항이 상수라는 설명이다. 셋 다 아래 수치 검사로 확인했다.
- **Toth–Lake (arXiv:1402.6069v4, alphaXiv 본문으로 확인):** Eq. (1)의 Zeeman 항은 $+\mu_B\mathbf H^{\mathsf T}\sum\mathsf g_i\mathbf S$(전자 부호, 장이 g의 첫 첨자와 축약), Eq. (11)은 $\mathsf J_{ij}(\mathbf d)=\mathsf J_{ji}^{\mathsf T}(-\mathbf d)$, Sec. II의 x축 easy-axis 예. 노트의 인용 문장과 같다.
- **인용 키:** 노트가 쓰는 `[@key]` 9개가 모두 research-library `references.bib`에 있고, 서지가 인용 목적과 맞는다.
- **상수:** 노트의 $\mu_B\simeq0.0578838$ meV/T와 자유 전자 $g\approx2.0023$은 CODATA 2018·2022 어느 값과도 표시 자릿수에서 같다. 패키지 `MU_B_MEV_PER_T = 5.7883818060e-2`는 CODATA 2018 값이다(2022 값 5.7883817982e-2와 상대 차 1e-9; 2022 값은 기억에 의존, NIST 표는 이 환경에서 못 열었다).
- **수치 검사:** `code-space/examples/verification/lswt_spin_hamiltonian_check.py`, 13개 모두 통과.
  1. 결맞음 상태 $\langle\{S^a,S^b\}\rangle/2=S(S-\tfrac12)n_an_b+\tfrac S2\delta_{ab}$ (S = 1/2…5/2)
  2. 스핀 1/2에서 $\sum D^{ab}S^aS^b=\tfrac14\operatorname{Tr}\mathsf D$, 스핀 1에서 $-A(S^x)^2$의 바닥이 $|m_x|=1$
  3. 비대칭 실수 $\mathsf J$의 결합 연산자가 에르미트, 링크 합 = 순서쌍 합의 절반
  4. 고유 회전($\det=+1$)만 스핀 대수를 지킨다
  5. 이 노트의 장 $\mathbf B$에서의 해밀토니안과 전자 해밀토니안이 같은 스펙트럼, 같은 유한 온도 모멘트를 준다(DM, 비등방 g를 넣은 3사이트)
  6. 패키지 `Term.zeeman`이 $\mathbf h=\mu_B\mathsf g^{\mathsf T}\mathbf B$, 장과 평행한 스핀이 낮은 에너지, `Term.onsite`의 고전 에너지가 1의 결맞음 값

### 고친 것

1. **국소 틀 회전에 $\det\mathsf R_I=+1$ 추가 (Notation).** $\mathsf R^{\mathsf T}\mathsf R=\mathsf I$만으로는 반전이 섞인 회전이 허용되고, 그때 국소 성분이 $[\tilde S^x,\tilde S^y]=-\mathrm i\tilde S^z$가 되어 HP 표현의 전제가 깨진다(검사 4). `lswt-classical-order`의 $\mathsf R(\theta,\phi)$는 이미 고유 회전이라 다른 노트 내용은 바뀌지 않는다.
2. **Mermin–Wagner 문장 보완 (Overview).** "symmetry-breaking field"가 반강자성체의 균일 장도 포함하는 것처럼 읽혔다. 균일 장은 장 축 둘레 회전을 남기므로 기운 반강자성체의 가로 질서는 2D 유한 온도에서 여전히 없다(D49의 Goldstone 조건과 같은 내용). easy-plane 이방성도 같다. 같은 논리가 모든 연속 대칭에 적용된다는 점은 Mermin–Wagner 논문의 Bogoliubov 부등식 논증에서 나온 일반화이고, 새 인용은 달지 않았다.
3. **"anisotropic magnetic coupling" → "anisotropic $g$-tensor" (Bilinear Hamiltonian).** 뜻을 분명히 했다.

### 사용자 판단이 필요한 것

- **Zeeman 부호 관례.** 2026-09-06 사용자는 이론 문서에 전자 부호($\hat{\boldsymbol\mu}=-\mu_B\mathsf g\hat{\mathbf S}$, $\mathbf h=-\mu_B\mathsf g^{\mathsf T}\mathbf B$)를 쓰기로 했다(문서 점검 파일 "Zeeman Convention Decision"). 2026-10-08 커밋 3901c51이 노트를 패키지·D09 관례($\hat H_Z=-\mu_B\mathbf B^{\mathsf T}\mathsf g\hat{\mathbf S}$, 모멘트 $+\mu_B\mathsf g\hat{\mathbf S}$)로 바꾸고, 전자계와는 $\mathbf B\to-\mathbf B$ 관계라고 적었다. 두 관례 모두 일관되고(검사 5), 에너지·스펙트럼·자화는 같으며 열 Hall 부호 같은 시간 반전 홀수 양만 장 방향이 뒤집힌다. 이 변경에 대한 사용자 승인 기록은 찾지 못했다. 추천: 지금 노트대로(패키지·D09와 같음) 두고 09-06 결정을 이것으로 대체한다고 기록. SpinW(Toth–Lake)는 전자 부호를 쓰므로 비교할 때 장 부호를 뒤집어야 한다는 점은 노트에 이미 있다.
- **노트 안의 열린 관례:** Conventions Not Yet Fixed 절의 full-position Fourier 관례 승인은 `lswt-boson-hamiltonian` 검토 때 정한다. 이 노트를 accepted로 해도 그 항목은 열린 채로 남고, 노트가 그렇게 적고 있다.

### 사용자 판정 (2026-10-20)

- Zeeman 부호: "노트대로". 패키지·D09 관례를 이론 노트 관례로 확정하고 2026-09-06 전자 부호 결정을 대체한다. 근거: 이론 노트·개발 노트·코드가 이미 같은 부호이고, 전자 부호로 바꾸면 패키지 API의 장 방향을 뒤집어야 한다.
- 이에 따라 노트의 세 절 머리 주석을 accepted로 바꿨다. full-position Fourier 관례는 노트에 적힌 대로 열린 채 `lswt-boson-hamiltonian` 검토로 넘긴다.

### 형식 메모

- 노트에 `\tag{1}`–`\tag{4}`가 있다. 옛 lifecycle 규칙 3.2는 Markdown에 `\tag`를 금하지만, 2026-10-11 Markdown 전환 때 번호를 고정한 형식이고 lifecycle 규칙 자체가 옛 경로(archive Markdown)를 기준으로 쓰여 있다. 물리 내용과는 무관해 고치지 않았다.
