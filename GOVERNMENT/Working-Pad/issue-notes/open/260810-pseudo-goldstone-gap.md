---
frontmatter-version: 1
title: Pseudo-Goldstone gap and nonuniform soft-mode handling
section: issue-notes/open
issue-type: problem
status: in-review
last-edited-by: claude
created: 2026-08-10
updated: 2026-09-30
source: GOVERNMENT/Working-Pad/issue-notes/closed/260810-agents-progress-history.md
related:
  - code-space/spintoolkit/methods/lswt/diagonalization.py
  - docs/lswt/01-derivation/paraunitary-diagonalization.md
  - legacy/scripts/4_Pseudo_Gap.py
  - examples/pseudo_goldstone_comparison.py
  - docs/nbcp/research-note.md
must-read: GOVERNMENT/Agents-Bylaws/templates/issue-notes-template.md
---

# Pseudo-Goldstone gap and nonuniform soft-mode handling

## 배경

과거 `AGENTS.md`에는 “Pseudo-Goldstone gap: 비균일 soft mode 처리 미구현 (장기 과제)”이라고 기록되어 있었다. 이 문구에는 대상 모델, 기대하는 gap의 정의, 재현 조건과 참고 유도가 포함되어 있지 않았으므로 해결되지 않은 물리·구현 문제로 분리한다.

현재 `Diagonalizer`는 positive definiteness를 복원하기 위한 k-independent MAGSWT shift, k-dependent onsite shift와 direct $JH$ fallback을 제공한다. 이 기능들이 과거 기록의 “비균일 soft mode 처리”와 같은 문제를 다루는지는 확인되지 않았다.

## 문제 정의

초기 2026-08-10 기록에서는 대상 모델과 재현 조건이 `Unknown`이었다. 2026-09-12 비교에서는 arXiv:2601.20963의 SOC-induced Y/V pseudo-Goldstone mode와 legacy gap 계산을 특정했다. 아래 초기 진단은 탐색 당시의 기록으로 보존하며, 현재 판정은 후속 비교를 따른다. Production API 구현과 사용자 물리·수학 검토는 아직 완료되지 않았다.

## 본론

### 초기 증상 (2026-08-10 탐색 범위)

- 코드와 테스트에서 `pseudo-Goldstone`을 명시적으로 다루는 API나 회귀 fixture가 발견되지 않는다.
- `docs/lswt/01-derivation/paraunitary-diagonalization.md`는 Goldstone-mode caveat만 작업 메모로 남기고 있다.
- Source TeX의 review item B18은 표준 Colpa construction이 positive-definite Hamiltonian을 요구하며 positive-semidefinite Goldstone mode에는 수정 절차가 필요하다고 지적한다.
- 현재 regularization이 물리적 gap을 계산하는 절차인지, 수치적 Cholesky 안정화를 위한 장치인지는 문서에서 구분되어 있지 않다.

### 초기 재현 조건

재현에 필요한 spin model, parameter point, ordering state, momentum, system-size scaling과 기대 gap 값이 기록되어 있지 않다. 정확한 재현 조건은 `Unknown`이다.

### 초기 근본 원인 분석

과거 기록이 구현 과제의 이름만 남기고 물리적 대상과 acceptance criterion을 남기지 않은 것이 현재 확인 가능한 직접 원인이다. Pseudo-Goldstone gap이 quadratic LSWT의 zero-mode 처리 문제인지, higher-order fluctuation으로 생성되는 gap인지, 특정 legacy 방법을 가리키는지는 아직 판정하지 않는다.

### 초기 해결 방안

1. 문제를 처음 제기한 source 또는 대상 모델을 찾아 pseudo-Goldstone gap의 정의와 예상 spectrum을 고정한다.
2. exact Goldstone zero mode, numerical soft mode, pseudo-Goldstone gap을 구분한다.
3. 요구되는 계산이 quadratic LSWT 범위인지 higher-order correction인지 판정한다.
4. 물리적 기대값이 정해진 뒤에만 regularization 또는 별도 solver 변경안을 제안한다.
5. 최소 재현 모델, momentum-resolved spectrum과 tolerance를 포함한 회귀 테스트를 작성한다.
6. 이론 설명과 코드 검증 결과를 `paraunitary-diagonalization.md` 및 관련 구현에 함께 반영한다.

### 2026-09-12: SOC Y/V 회전 경로와 gap 비교

#### 대상과 회전의 의미

대상은 nearest-neighbor XXZ + PD + Gamma 모델의 공통 global spin-z 회전이다. 세 bond orientation의 교환행렬 합이 $3\operatorname{diag}(J,J,J_z)$이고, 각 부격자 쌍이 세 orientation을 모두 포함하므로 PD/Gamma 기여가 고전적 three-sublattice 에너지에서 상쇄된다. 따라서

$$
\mathbf S_\alpha(\phi)=R_z(\phi)\mathbf S_\alpha(0)
$$

는 SOC가 있어도 같은 고전 에너지를 갖는다. SOC가 유한하면 전체 Hamiltonian의 정확한 연속 대칭은 아니므로 spin-wave zero-point energy가 이 degeneracy를 lift할 수 있다. SOC가 0이면 이 회전은 정확한 U(1) 대칭이며 **이 모드의** gap은 0이다.

이 모드는 arXiv:2505.07229에서 다룬 zero-field pure-XXZ의 내부 비균일 pseudo-Goldstone mode와 구분한다. 앞선 대화에서 후자의 회전을 중심으로 설명한 것은 현재 SOC 코드의 global-z 회전을 부정하는 근거가 될 수 없다. 또한 제한된 three-sublattice manifold의 등고전 에너지만으로 전역 ground state의 안정성을 증명할 수는 없다.

#### 재현 조건과 계산

- $S=1/2$, $J=0.075$ meV, $J_z=0.125$ meV, $T=0$; NNN 및 DM 항은 0.
- $h=g_z\mu_B B$, $g_z=4.645$; Y: $B=0.2$ T, V: $B=1.4$ T.
- $J_{PD}$ 또는 $J_\Gamma$를 0부터 0.020 meV까지 각각 변화시킨다. 두 구현에 같은 parameter와 spin state를 공급한다. 원래 gap script 기본값의 $J=0.076$, $h=0.05$ meV를 그대로 실행한 결과라는 뜻은 아니다.
- 각 Hamiltonian의 zero-point energy 최소각 $\phi_*$에서 곡률을 계산한다. 에너지는 physical spin당 값이며 positive physical bands와 normal-ordering trace를 포함한다.
- $C_\phi=\partial_\phi^2 e_{\mathrm{sw}}(\phi_*)$, $m_z=(S/3)\sum_\alpha\cos\vartheta_\alpha$, $\chi_z=\partial m_z/\partial h$로 두면 leading-order gap은 $\Delta_{\mathrm{PG}}=\sqrt{C_\phi/\chi_z}$이다.
- 자기 unit cell당 고전 polar Hessian $A$와 $w_\alpha=-S\sin\vartheta_\alpha$를 사용해 $d\boldsymbol\vartheta/dh=A^{-1}w$, $\chi_z=w^TA^{-1}w/3$를 계산한다. Y와 V의 $\chi_z$는 각각 1.1111111111, 1.5509777351 meV$^{-1}$/spin이다.
- Cholesky shift를 사용하지 않는다. 샘플된 angular orbit에서 quadratic Hamiltonian이 불안정하면 gap을 배정하지 않으며, 미소 곡률이 수치 분해능 이하인 경우도 물리적 gap 0과 구분한다.

#### 확인한 legacy 문제

1. `legacy/scripts/4_Pseudo_Gap.py`의 공통 azimuth 회전은 이 SOC 모드에 적절하다. 그러나 hard curvature 계산에서 모든 polar angle에 같은 증분을 주는 것은 올바른 켤레 모드를 구성하지 않는다. 이번 signed Y 좌표에서 그 변형과 global-z 회전의 Berry pairing은 0이며, V에서도 올바른 부격자별 응답과 다르다.
2. 같은 스핀을 $(\vartheta,\phi)\mapsto(-\vartheta,\phi+\pi)$로 재표현해도 현재 zero-point energy는 동일하지만, legacy uniform-theta curvature는 달라진다. Y에서는 0.0206658746에서 0.0766217798 meV/spin으로 바뀐다. 기존 결과의 좌표 의존성을 직접 확인했다.
3. `sqrt(hessian_det) * spin` 및 hardcoded $S=1/2$는 이 잘못된 좌표쌍의 normalization을 보정하지 못한다. 단순히 곱하기를 나누기로 바꾸는 것만으로 해결되지 않는다.
4. `modules.Tools.analysis_tools.Create_Energy_Function` import는 수정된 `code-space/spintoolkit`(당시 `code-space/lswt`)가 아니라 archived Hamiltonian을 사용한다. 따라서 기존 B/B-dagger 조립 문제의 영향이 gap 계산에도 남아 있다.
5. 음의 determinant와 에너지 계산 exception을 0으로 반환한다. Invalid 계산과 물리적인 gap closing을 구분해야 하며, gap class 내부에서 angular minimum도 찾지 않는다. Mixed curvature를 0으로 고정하는 것은 이번 leading type-I 근사에서는 정당화되며, 별도의 leading-order 오류로 판정하지 않는다(2026-09-15 후속 검토).
6. `legacy/scripts/2_U_symmetry_YV.py`의 angular energy range 정규화는 exact-U(1) 조건에서 roundoff를 증폭할 수 있다. 비교 그림은 절대 에너지 단위를 유지한다.

#### 수치 결과와 검증 범위

현재 Hamiltonian과 canonical response로 구한 $N=48$ angular-fit gap:

| Phase | 유한한 SOC 항 (다른 항은 0) | Gap (meV) |
|---|---|---:|
| Y | $J_{PD}=0.010$ meV | 0.00686217 |
| V | $J_{PD}=0.010$ meV | 0.01564682 |
| Y | $J_\Gamma=0.010$ meV | 0.0000072960 |
| V | $J_\Gamma=0.010$ meV | 0.00203898 |

- 네 대표점에서 독립적인 five-point 곡률로 확인했다. $N=48\to96$ 및 미분 간격 $0.04\to0.02$ rad 변화에 따른 gap 변화는 각각 0.01% 미만이다. 전체 parameter curve의 오차 상한이나 higher-order correction의 상한이라는 뜻은 아니다.
- 독립 Cartesian HP 조립과 현재 quadratic matrix 차이는 Y/V에서 $4.2\times10^{-17}$ meV 이하이다. 같은 진단점에서 legacy matrix 차이는 각각 약 0.01536, 0.02017 meV이다.
- Relaxed magnetization의 finite difference와 six-dimensional classical weak-pinning dynamics로 susceptibility 및 gap normalization을 별도로 확인했다.
- 기존 `test_hamiltonian_pairing.py` 회귀 검사 8개가 통과했다. 비교 자체는 원래 legacy class의 finite-difference 실행 로그도 보존한다.
- 샘플된 current Hamiltonian에서 Y의 $J_{PD}\ge0.015$ meV, V의 $J_{PD}\ge0.0125$ meV는 일부 회전각에서 불안정했다. 이는 조사한 이산점의 판정이며 정확한 phase boundary를 뜻하지 않는다. Y의 가장 작은 비영 $J_\Gamma=0.0025$ meV 곡률은 분해능 이하로 남겼다.

재현 코드: `examples/pseudo_goldstone_comparison.py`, `examples/pseudo_goldstone_validation.py`, `examples/pseudo_goldstone_plot.py`.

결과: [비교 보고서](../../../../docs/nbcp/research-note.md#sec-nbcp-gap-appendix), [gap 그림](../../../../data-space/verification/260912-pseudo-goldstone/yv-pseudo-gap-comparison.png), [회전·에너지 그림](../../../../data-space/verification/260912-pseudo-goldstone/yv-equal-energy-orbits.png). 같은 결과 디렉토리에 raw arrays, source hashes, convergence JSON 및 unchanged legacy class 실행 로그를 보존했다.

사용자의 TeX/Markdown 저장 요청에 따라 같은 비교 기록의 TeX와 PDF preview(현재는 [단일 연구노트 PDF](../../../../docs/nbcp/output/research-note.pdf)로 통합)를 추가했다. 당시에는 Markdown을 수정 원본으로 유지하며 `examples/pseudo_goldstone_export.py`로 파생물을 생성했다(현재 실행법은 아래 단일 원본 정리 항목 참조). 그림 재생성은 이미 저장된 Markdown을 덮어쓰지 않는다. 수식 정의, 켤레 응답의 유도, 코드 위치, 두 그림과 수치 표를 포함했고, TeX 컴파일 및 6쪽의 출력 상태를 확인했다. 이 저장·조판 확인은 물리 acceptance가 아니다.

### 2026-09-15: Y/V 켤레 모드의 명시적 유도와 독립 검증

기존 [비교 보고서](../../../../docs/nbcp/research-note.md#sec-nbcp-gap-appendix)의 `Detailed canonical-mode check`에 Berry 항, 고정 자화에서의 에너지 최소화, Y/V의 명시적 응답과 검증을 추가했다. Markdown으로 내용을 수정하고 TeX/PDF는 같은 원본에서 재생성한다. 2026-09-12의 quantum-energy scan 및 gap 표는 다시 계산하거나 변경하지 않았다.

- Per-spin Berry 항은 $\hbar\,\delta m_z\dot\phi$이며, $w_\alpha=-S\sin\vartheta_\alpha$로 두면 $\delta m_z=\mathbf w^T\delta\boldsymbol\vartheta/3$이다. 임의의 polar 방향 $\mathbf u$에 대한 pairing은 $b_{\mathbf u}=\mathbf w^T\mathbf u/3$이다.
- 고정 자화에서 고전 에너지를 최소화하면 $\delta\boldsymbol\vartheta=A^{-1}\mathbf w\,\delta m_z/\chi_z$를 얻는다. 단지 $b_{\mathbf u}\ne0$인 것만으로 올바른 저에너지 응답이 되는 것은 아니다. Positive $A$에 대한 Cauchy-Schwarz 부등식으로 이 relaxed 방향이 최소 hard stiffness를 준다는 것을 보였다.
- Y의 signed chart $(t,-t,\pi)$에서 uniform polar 방향 $(1,1,1)$은 $b=0$이다. 올바른 응답은 $d\boldsymbol\vartheta/dh=(-1,1,0)/[3S(J+J_z)\sin t]$, $\chi_z=2/[9(J+J_z)]$이다.
- V의 uniform polar 방향은 $b\ne0$이므로 restricted surface에서 켤레쌍으로 정규화할 수 있다. 그러나 부격자별 relaxed 응답과 다르다. 기존 표현은 이 둘을 구분하도록 정밀화했다. 대표 V점에서 Berry 계수만 고친 uniform-direction trial gap은 weak-pinning 극한에서 relaxed 결과의 약 15.1032배다. 이는 **기존 코드 출력의 비율이 아니라**, 정규화만 바꾸어서는 해결되지 않는다는 검증이다.
- Mixed curvature의 고전 부분은 common-azimuth 불변성 때문에 0이고, quantum mixed curvature의 제곱은 추가 soft mode가 없는 leading type-I 전개에서 subleading이다. 따라서 이 항을 0으로 둔 것 자체를 이번 leading discrepancy의 별도 원인으로 지목하지 않는다.
- 좌표를 뒤집은 뒤 다시 $(1,1,1)$을 쓰면 물리적으로 다른 변형을 선택한다. 그 hard curvature가 바뀌는 것은 에너지 함수의 좌표 불변성 오류를 뜻하지 않는다. 같은 물리적 relaxed 응답을 변환하면 susceptibility는 불변이다.

검증 스크립트는 `examples/pseudo_goldstone_conjugate_mode.py`, 결과는 `data-space/verification/260912-pseudo-goldstone/conjugate-mode-validation-260915.json`에 저장했다. Y/V 공식과 full Hessian 응답, 독립적인 field finite difference, nonlinear constrained relaxation, 각 phase의 8가지 coordinate chart, six-dimensional weak-pinning dynamics를 확인했다. Field tangent의 상대 오차는 $8.0\times10^{-9}$ 이하, chart별 susceptibility 차이는 $2.1\times10^{-8}$ 이하이며, 약한 pinning에서 gap 계수와 mode profile도 일치했다. 이 검증은 local classical reduction을 지지하며 full one-loop self-energy 또는 사용자 물리 acceptance를 대신하지 않는다.

보완한 Markdown으로 TeX/PDF를 재생성하고 PDF 9쪽을 확인했다. TeX log에 overfull box나 missing character는 없으며, 수식·그림·표의 출력과 원본/파생물 hash 일치를 확인했다. Package API와 legacy script의 변경은 이번 단계에 포함하지 않았다.

### 2026-09-16: 리서치 노트 편집과 출력

사용자 요청에 따라 비교 기록을 리서치 노트 형식으로 재구성했다. 첫 페이지에 핵심 gap 관계와 기호·단위 표를 두고, 본문은 모델과 회전, Berry 항과 constrained relaxation, Y/V 응답, 수치 비교 순서로 정리했다. Legacy 진단, 수렴성 표, 재현 명령과 작업 이력은 부록으로 이동했다. 이전 `Detailed canonical-mode check`의 내용은 현재 `Canonical response and the gap` 및 `Y and V states`에 있다.

기존 16개 display equation의 수식 내용과 5개 데이터 표의 수치가 보존되었음을 대조했다. 14개 핵심 수식에 semantic ID를 부여했고, Quarto transform에서 수식 번호와 참조를 생성한 뒤 XeLaTeX로 출력한다. 기존 데이터로 두 그림의 글자·범례·단위 표기와 해상도를 개선했다. 계산 코드를 실행해 물리량을 재계산한 작업은 아니다. 여러 canonical-response 검사가 같은 고전 에너지 또는 Hessian을 공유한다는 검증 범위도 명시했다.

최종 PDF 11쪽을 렌더링해 확인했으며, overfull box, missing character, 미해결 수식 참조는 없다. Markdown, TeX/PDF 및 그림의 hash는 `report-export.json`으로 연결한다. 이 편집·출력 확인은 추가 물리 검증이나 theory acceptance를 의미하지 않으며, 상태는 `in-review`로 유지한다.

### 2026-09-16: arXiv 열적 주장으로 복귀 — 순수 PD 축의 V 대칭

arXiv:2601.20963v1의 Section II, Figure 4 및 clock 유효모형을 다시 대조했다. 일반적인 $J_\Gamma\ne0$ V의 threefold pinning과 $J_\Gamma=0$ PD-only 축을 구분해야 한다. 순수 PD 모델은 global spin-$z$ $\pi$ 회전이 정확한 대칭이므로, V orbit의 $2\pi/3$ 주기와 결합하면 허용되는 angular harmonic은 6의 배수다. 따라서 generic V의 $p=3$ clock 논증을 PD-only 축까지 그대로 확장할 수 없다.

별도 [검토 노트](../../../../docs/nbcp/research-note.md#sec-nbcp-clock)에 대칭 유도, Fourier 정의, 기존 세 mesh의 재분석, Gaussian RG의 적용 조건과 다음 연구를 기록했다. $J_{\rm PD}=0.010$ meV인 V의 $A_6$는 $1.0239780\times10^{-5}$ meV/spin이고, $A_3$는 약 $4.3\times10^{-19}$ meV/spin의 수치 잡음 수준이다. $J_\Gamma=0.010$ meV인 V에서는 $A_3=7.1609759\times10^{-7}$ meV/spin이다. 이 계수들은 gap이 아니라 angular energy Fourier amplitude다.

`examples/nbcp_clock_anisotropy_audit.py`는 원래 scan을 덮어쓰지 않고 결과를 `data-space/verification/260916-clock-anisotropy/`에 저장한다. 128개 새 비대칭 momentum 표본의 Cartesian HP matrix 검사도 수행했고, 순수 PD의 $\pi$ 회전 및 $\Gamma$ 부호 반전 covariance 오차는 $2.9\times10^{-17}$ meV 미만이다. 이 검사는 기존 bond geometry와 state를 공유한다.

현재 결론은 순수 PD V의 중간 algebraic phase가 phase-only clock RG에서 대칭상 허용된다는 것이다. 유한온도 spin stiffness, vortex fugacity, density-order 안정성이나 실제 thermal phase boundary를 계산한 결과는 아니다. 다음 연구는 공간적 위상 강성과 pinning의 연속체 정규화를 구성하는 것이다. Production API 설계는 기존 미완료 항목으로 유지한다.

### 2026-09-16: 기존 Overleaf 연구노트 통합

사용자가 로그인해 둔 Overleaf 프로젝트에서 `main.tex`, `analysis.tex`, `Three_MSL.tex`, `Stripe_MSL.tex`, `Four_MSL.tex` 전체 텍스트를 읽었다. `paper.tex`는 제목·초록·목차와 중복 관계를 확인했고 `Three_MSL.tex` 원문 전체를 포함한다. 원격 문서를 수정하지 않고, [통합 연구노트](../../../../docs/nbcp/research-note.md)를 NBCP 소개 → phase 분석 → Y/V supersolidity → gap 부록 순서로 구성했다. 기존 gap 노트를 출력 시 전체 포함하여 상세 유도와 표의 편집 원본을 한 곳으로 유지한다.

[원문 대조 기록](../../../../docs/nbcp/sources/overleaf-2026-09-16-review.md)에 수식 차이, 원문 위치·발췌, 채택·보류 항목을 남겼다. 특히 YY 교환행렬 성분, ladder 계수, inversion에서 axial-spin 변환, UUD 고유값의 S 인자, Γ/V 스핀 배치 명명과 임계장 숫자의 불일치를 구분했다. NNN 및 stripe Hessian과 SkX의 signed charge 초안은 검증된 결론으로 채택하지 않았다. 자료는 선별 검토 기록이며 전체 프로젝트 ZIP 백업을 확보한 것은 아니다.

새 `nbcp_phase_source_check.py`는 bond sum, stripe bond count, V 미분, Γ→V 벡터 대응, UUD Cartesian Hessian과 임계장을 확인한다. 에너지 오차는 $5.6\times10^{-17}$ meV 이하, V gradient 유한차분 차이는 $2.6\times10^{-12}$ meV 이하, Hessian 차이는 $7.3\times10^{-10}$ meV 이하이다. 기존 gap scan과 iDMRG 계산을 재실행한 것은 아니며, 전역 phase 안정성의 검증도 아니다. 출력 서식은 별도 표준 LaTeX preamble과 Lua filter로 분리하고 기존 색상·서체를 유지했다.

통합 PDF는 19쪽이다. 전체 페이지 렌더링과 수식 참조·링크를 확인했다. gap 원본의 display 수식 16개와 모든 표 행이 보존되었다. 최종 출력 경고는 없으며 검증 결과는 `data-space/verification/260916-nbcp-integration/document-check.json`에 남겼다. 문서 출력 완료와 물리적 acceptance를 구분한다.

## 결론 / 미결 사항

SOC Y/V 모드의 재현 조건, 등고전 에너지 회전과 legacy 차이에 대한 비교를 완료했다. 현재 상태는 사용자 물리·수학 검토를 위한 `in-review`이며 이슈를 닫지 않는다. 이번 작업은 별도 비교 스크립트와 기록에 한정했고 production package와 legacy script는 수정하지 않았다.

SOC 없는 Y의 고전적 공간 강성과 Y의 SOC 확장 조건은 아래 2026-09-17 기록처럼 검증했다. 사용자 지정 순서에 따라 현재는 Y에 집중한다. 이어서 B=0.2 T에서 개별 SOC축의 저장된 양자 선택 각도를 포함한 88개 배경에 대해 전체 BZ 조화 안정성 검사를 수행했고, 2026-09-18에는 각도별 고전 gradient·leading vacuum potential matching을 완료했다. 다음 단계는 사용자 물리 검토 후 비선형 density wall·angular defect 계산의 범위를 정하는 것이다. 결합벡터 방향 convention, 경쟁상과의 에너지 비교 및 양자 hard-coordinate 완화는 여전히 미결이다. V 강성과 유효 clock 모형의 양자·열적 정규화는 후속 범위다. 구현 측면에서는 canonical response를 사용하는 production 계산 설계, 오류 상태 구분, 사용자 검토 후 해당 theory owner 반영이 남아 있다. Full one-loop self-energy와의 독립 비교, $S=1/2$에서 higher-order 보정, 전역 phase 안정성 및 유한온도 gap은 검증하지 않았다. 원래 AGENTS 문구의 모든 nonuniform soft-mode 문제가 이 특정 SOC 모드와 같은 문제인지도 확정하지 않는다. 논문 Figure 4에 사용된 정확한 코드 revision과 원자료는 확보하지 않았으므로 published curve를 완전히 재현했다고 주장하지 않는다.

## 참조

- [arXiv:2601.20963](https://arxiv.org/html/2601.20963v1) — SOC 모델, global-z orbit 및 Figure 4 조건
- [arXiv:1805.00947](https://arxiv.org/html/1805.00947v2) — leading-order curvature gap과 켤레 좌표
- [arXiv:2505.07229](https://arxiv.org/html/2505.07229v1) — nonlinear soft path; zero-field internal XXZ 모드와의 구분
- `docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf` — 원본 LSWT note
- `docs/lswt/sources/01-editable-notes/note_lswt_reviewed.tex` — review item B18
- `docs/lswt/00-foundations/lswt-overview.md` — positive-definite/semidefinite 적용 경계
- `docs/lswt/01-derivation/paraunitary-diagonalization.md` — 대각화 문서 owner
- `code-space/spintoolkit/methods/lswt/diagonalization.py` — 현재 Colpa 및 regularization 구현

## 2026-09-16 — 단일 Markdown 원본으로 정리

사용자의 중복 병합 요청에 따라 본문, clock 상세 분석과 gap 부록을 [research-note.md](../../../../docs/nbcp/research-note.md) 하나로 합쳤다. 기존 수식·표·적용 조건과 계산 데이터는 보존했다. PDF는 같은 주제의 `output/`에서 생성하며 중간 TeX·Markdown·그림 사본은 보관하지 않는다. 별도 gap exporter와 그림 생성기의 초기 Markdown 생성 기능을 폐기해 중복 문서가 재생성되지 않도록 했다. 위의 과거 export 경로·쪽수·실행법은 당시 이력이며 현재 출력법은 NBCP README를 따른다. 보존 검사는 [문서 통합 기록](../closed/260916-docs-topic-consolidation.md)에 추가한다. 남은 물리 연구 범위와 `in-review` 상태는 유지한다.


## 2026-09-17 — SOC 없는 Y 상태의 고전적 공간 강성

사용자가 Mathematica 또는 Python 중 적합한 도구로 진행하도록 승인했다. 이번 범위는 $J_{\rm PD}=J_\Gamma=0$, $T=0$, 양의 hard Hessian을 갖는 Y branch의 고전 강성 및 조화 LSWT 대조다. V·SOC·양자 강성 보정·유한온도 계산은 포함하지 않았다.

- [통합 연구노트의 유도](../../../../docs/nbcp/research-note.md#sec-nbcp-y-stiffness)에 physical site coordinate에서의 twist, 결합 수, spin당/면적 정규화, 내부 완화와 분산의 관계를 추가했다.
- [Mathematica 검증](../../../../examples/nbcp_y_stiffness.wl)은 실제 Wolfram 15.0 커널에서 실행했다. 8개 기호 검사가 통과했다. 특히 axial C spin의 두 tangent 방향을 포함한 mixed twist/internal Hessian이 정확히 0이다. Uniform global azimuth를 제외한 hard Hessian이 양수일 때 내부 완화는 $Q^4$부터 에너지에 기여한다.
- 결과는 $\rho_s^{\rm cl}=JS^2(1-c^2)/\sqrt3$, $c=(h+3SJ_z)/[3S(J+J_z)]$이며, $u_Y^2=a_{\rm spin}\rho_s^{\rm cl}/\chi_z$다. $B=0.2$ T에서 $\rho_s^{\rm cl}=0.00382336077954$ meV, $u_Y/a=0.0545895118739$ meV이다.
- [Python 검증](../../../../examples/nbcp_y_stiffness.py)은 5개 자기장, 4개 방향, 4개 간격에서 5개 내부 자유도를 완화한 에너지 차분과 production LSWT 저에너지 분산을 비교했다. 최종 간격 $qa=0.001$에서 최대 상대 차이는 각각 $2.79\times10^{-6}$과 $2.56\times10^{-6}$이다. 이는 수렴 진단이며 물리적 오차 상한이 아니다.
- 별도 Cartesian constrained Hessian으로 조립한 dynamics의 세 양의 에너지와 production LSWT의 최대 차이는 $1.10\times10^{-13}$ meV이다. 독립적으로 site/color를 열거한 36-site triangular torus의 twisted-boundary energy와 3-site expression 차이는 $2.09\times10^{-17}$ meV/spin 이하다. 동일 모델·매개변수를 쓰는 대수·수치 대조이며 독립 many-body 검증은 아니다.
- [검증 자료](../../../../data-space/verification/260917-y-stiffness/)에 symbolic/numerical JSON, source hashes와 그림을 저장했다. Legacy gap scan과 production package는 수정하지 않았다. Mathematica의 symbolic simplification과 수치 일치는 사용자 physics acceptance를 대신하지 않으며 `in-review`를 유지한다.

재현 명령(저장소 루트, 설치된 Wolfram 경로 기준):

```bash
WolframKernel=/Applications/Wolfram.app/Contents/MacOS/WolframKernel \
  wolframscript -file examples/nbcp_y_stiffness.wl
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  python examples/nbcp_y_stiffness.py
python examples/nbcp_research_export.py
```

다음 단계는 우선 SOC 없는 V에서 같은 정규화를 검증하는 것이다. SOC를 켠 경우에는 XXZ의 $L=0$과 uniform-twist 표현을 그대로 가정하지 않고 gradient·pinning·내부 모드 결합을 다시 유도해야 한다. 이번 고전 강성을 곧바로 renormalized thermal stiffness나 BKT 전이온도로 해석하지 않는다.

출력·보존 검사: PDF 24쪽을 렌더링해 확인했고, LaTeX의 overfull·누락 문자·미해결 참조 경고는 없다. 기존 display 수식 42개와 표 행 67개를 보존했으며, 새 수식은 7개다. 기존 코드·legacy·검증 데이터 123개 파일의 hash가 유지되었다. [문서 검증 기록](../../../../data-space/verification/260917-y-stiffness/document-check.json)에 결과를 저장했다.


## 2026-09-17 — Y 유도 중간 계산 보강 및 SOC 확장 조건

사용자가 수식 유도의 계산 과정을 보강하고 Y phase에서 SOC 확장 조건을 검토하도록 요청했다. [단일 연구노트](../../../../docs/nbcp/research-note.md#sec-nbcp-y-stiffness)에 결합별 에너지, canting stationarity와 Q² 변위, 6개 regular tangent 좌표, uniform Hessian, 5차원 hard projector, 제곱 완성 및 cell/spin susceptibility 변환을 추가했다. 원래 zero-SOC 결과는 유지한다.

[SOC 절](../../../../docs/nbcp/research-note.md#sec-nbcp-y-soc-conditions)에서는 고정 tensor를 사용하는 uniform spiral 치환의 실패를 설명하고, 작은 진폭의 Fourier 변동으로 정적·동적 Schur complement를 유도했다. Uniform bond sum에서 SOC가 상쇄되는 것과, 방향 벡터가 곱해진 first moment가 상쇄되는 것은 다르다. SOC에서 내부 좌표가 위상 기울기에 선형 결합하므로 정적 강성은 direct term에서 hard-mode 완화 항을 빼야 한다. Gamma가 있으면 mixed frequency–momentum 항도 나타나며, 양의 k 분산 기울기의 제곱 대신 ±k 양의 에너지의 곱으로 정적 강성을 추출해야 한다.

검증 자료는 [새 전용 폴더](../../../../data-space/verification/260917-y-soc-conditions/)에 저장했다. Wolfram 15.0의 13개 exact checks가 통과했다. Python 비교는 B=0.2 T, SOC 4쌍, 기준각 3개, 방향 3개, ka 4개를 사용한다. ka=0.001에서 최대 상대 차이는 static 1.28e-7, paired dispersion 1.18e-7, 개별 에너지 6.09e-8 미만이다. 별도 tangent dynamics와 production spectrum 차이는 1.46e-13 meV 미만이다. 576-site torus에서 exact-length spin 에너지로 Fourier curvature와 cell 정규화를 추가 대조했다. 이들은 같은 모델의 계산 대조이며 독립 실험·many-body 검증이 아니다.

### 미결: NBCP builder의 결합벡터와 API 정의가 반대

> **2026-09-30 해결(코드 규약):** toolkit 전달 규약 D13(사용자 결정 2026-09-29)이 저장 변위를 `d = r_source - r_target`
> (이웃은 `r_i - d`)로 정했고, `SpinSystem` 설명도 2단계에서 이 정의로 고쳤다. 2단계에서 `H(k)` 원소 단위 일치(1.7e-16),
> 3단계에서 ED 1-마그논 스펙트럼의 운동량 부호, 5a에서 `H(k)` 입자 블록 = ED Bloch 행렬(켤레 아님, 2.7e-14)로 확인했다.
> 따라서 아래의 `r_j = r_i - d` 해석으로 한 계산의 k는 물리적 Bloch 운동량이다. 실험실 방향을 붙이는 데 남은 것은
> 모델 `x`축과 결정 축의 방향 관계(물질 입력)뿐이다. 원고의 "unresolved displacement" 문장 수정은
> `260918-nbcp-physics-code-review.md`에 수정안과 함께 남긴다.

`examples/nbcp_ground_state.py`의 three_msl basis positions와 lattice vectors를 기준으로, 저장된 d에서 (r_i-r_j)를 빼면 magnetic lattice의 정수 병진이다. 반대로 API가 명시하는 (r_j-r_i)를 빼면 정수 병진에서 최대 1/3만큼 어긋난다. `code-space/lswt/core/spin_system.py`의 Coupling/add_coupling 설명과 LSWT notation 문서는 forward displacement를 사용하지만, 현재 Hamiltonian과 NBCP 입력을 real-space로 일관되게 해석하려면 neighbor r_j=r_i-d를 사용한다. 검사 결과는 numerical JSON의 `bond_orientation_audit`에 9개 결합별로 남겼다.

전역 방향 반전은 zero-SOC의 even-in-k 강성과 기존 균일 에너지에 영향을 주지 않으므로 이번 발견만으로 기존 gap scan이 틀렸다고 판정하지 않는다. 그러나 SOC odd-in-k 분산의 실험적 방향 부호를 붙이기 전에는 반드시 convention을 정리해야 한다. 이번 식과 수치는 현재 코드의 k 좌표 기준이다. Production API나 builder를 이번 작업에서 바꾸지 않았으며, 수정은 영향 범위를 명시한 별도 작업으로 남긴다.

현재 계산은 classical harmonic Y branch의 국소 확장이다. H>0과 정적 tensor 양수는 검사한 점에서 확인했지만 full-BZ 안정성, 경쟁상, 양자 선택 각도, 양자 gradient 보정, 유한온도 phase boundary는 계산하지 않았다. Quantum pinning curvature만 조화 분산에 삽입한 결과로 보고하지 않는다. 사용자 physics acceptance 전까지 `in-review`를 유지한다. 과거 기록의 V-first 순서는 이번 Y SOC 검토 요청으로 대체되며, V 계산은 수행하지 않았다.

출력·보존 확인: PDF 28쪽의 전체 렌더링 및 새 유도 페이지 상세 검토를 완료했다. 기존 수식 49개를 유지하고 14개를 추가했다. 수식·그림·절 ID 68개에 중복이나 미해결 참조가 없으며, 기존 code/legacy/data 127개 파일 hash는 유지되었다. 기존 서식 preamble, filter, exporter도 변경하지 않았다. [문서 검증 기록](../../../../data-space/verification/260917-y-soc-conditions/document-check.json)에 결과를 저장했다.


## 2026-09-17 — Clock RG 상세 유도와 SOC Y의 판정 범위

사용자 요청에 따라 연구노트의 [clock RG 유도](../../../../docs/nbcp/research-note.md#sec-nbcp-clock-rg)를 확장했다. 정적 partition function, physical area와 cutoff 정규화, constant anisotropic tensor의 면적 보존 좌표변환, Gaussian propagator와 vertex dimension, momentum-shell 평균, vortex winding·core fugacity, neutral dipole에 의한 stiffness flow를 연결했다. 비선형 계수는 cosine amplitude y_p와 charge fugacity y_p/2, vortex fugacity y_v의 convention을 명시한 뒤 유도했으며 weak-coupling truncation임을 구분했다.

[SOC Y 적용 범위](../../../../docs/nbcp/research-note.md#sec-nbcp-y-rg-status)의 결론은 여섯 겹 pinning 및 국소적인 양의 고전 강성이 clock 기술과 양립하지만, 유한온도 RG trajectory는 미검증이라는 것이다. 특정 phi0에서 계산한 tensor를 전체 compact phase circle의 상수 tensor로 자동 치환할 수 없다. 비상반 분산의 frequency–momentum 항은 static sector에서 사라지므로 그 자체는 clock RG의 반례가 아니다. Thermal matching, vortex core, density order 유지와 finite-size 지표가 남아 있다. 전이온도 수치는 추정하지 않았다.

[SymPy 검사](../../../../examples/nbcp_clock_rg_checks.py)로 shell 적분, dipole 계수, dual field 정규화, tensor 변환 및 p=6의 경계 지수 1/9·1/4를 확인했다. 12개 검사가 통과했고 [결과](../../../../data-space/verification/260917-clock-rg/clock-rg-check.json)에 정의·기존 Y 데이터 해시·미계산 항목을 저장했다. 이는 유도 정규화 검사이며 실제 NBCP thermal RG 검증이나 사용자 physics acceptance가 아니다.

출력 확인: PDF 32쪽 전체를 렌더링하고 새 유도·판정 부분 16–20쪽을 상세 검토했다. LaTeX overflow·미해결 참조 경고는 없다. 이전 clock 절 밖의 수식 58개 및 기존 code/legacy/data 130개 파일은 보존했다. [문서 검증 기록](../../../../data-space/verification/260917-clock-rg/document-check.json)을 저장했다.


## 2026-09-17 — Clock 환원의 근거와 검증 계산, 네 SOC 패널

사용자가 universality 근거의 충분성, 네 가지 환원 조건을 검증할 계산, Y/V와 PD/Gamma의 네 E(phi) 패널을 요청했다. [연구노트의 검증 절](../../../../docs/nbcp/research-note.md#sec-nbcp-clock-matching-tests)에 symmetry selection rule과 실제 RG basin의 구분을 명시하고, spectrum/mode separation → 전체 phi gradient·potential matching → density wall·복합 결함 → thermal RG 및 size scaling 순서를 구체화했다. 추가 soft mode, thermal occupation과 mode elimination의 차이, phase-mode double counting, density translation wall과 angular clock wall의 차이, fractional vortex의 허용·confinement 여부를 분리했다. 표준 clock 모형의 두 전이에 대한 계산 근거를 추가했으나 이를 NBCP 검증으로 보고하지 않는다.

[Figure 3](../../../../data-space/verification/260917-clock-matching/yv-angular-energy-four-cases.png)은 Y–PD, Y–Gamma, V–PD, V–Gamma 네 패널로 교체했다. Active coupling은 0.010 meV, inactive coupling은 0이며 Y의 B=0.2 T, V의 B=1.4 T이다. Current Hamiltonian의 기존 N=48, 72각도 데이터를 사용했다. Delta e는 zero-point energy에서 그 최솟값을 뺀 값이며, classical energy의 analytic constancy를 이용한 semiclassical angular energy 차이다. 각 패널의 절대 에너지 단위를 명시하고 크기로 정규화하지 않았다. Legacy 비교는 기존 gap 그림에 유지하며, 종전 orbit 그림도 원자료로 보존한다.

네 패널의 peak-to-peak 값은 각각 2.87355e-6, 3.28588e-12, 2.04814e-5, 1.43220e-6 meV/spin이다. 지배 harmonic은 6,6,6,3이다. [그림 생성 기록](../../../../data-space/verification/260917-clock-matching/four-case-figure.json)에 source hash, 패널 매개변수와 실제 그린 배열을 기록했다. 새 물리 scan, finite-temperature 계산이나 위 검증 계획의 실행은 하지 않았다.

문서 검증: PDF 34쪽 전체 렌더링과 새 검증 절 20–22쪽 및 Figure 3(26쪽)를 확인했다. 기존 display 수식 75개와 기존 code/legacy/data 파일 132개를 보존했다. 수식·그림 참조 및 로컬 링크가 유효하며 기존 서식과 exporter는 변경하지 않았다. [출력 검증 기록](../../../../data-space/verification/260917-clock-matching/document-check.json)에 저장했다.


## 2026-09-17 — M1에서 Y 전체 BZ 조화 안정성 검사

사용자가 제안한 M1 계산 순서에 동의하여 첫 단계인 spectrum/stability screen을 수행했다. [새 진단](../../../../examples/nbcp_y_stability.py)은 production 및 legacy를 수정하지 않고 B=0.2 T, J=0.075, Jz=0.125 meV, S=1/2의 고전적 Y 배경을 사용한다. SOC 7쌍 (0,0), (0.005,0), (0,0.005), (0.005,0.005), (0.010,0), (0,0.010), (0.010,0.010) meV는 검증용 값이며 물질의 추정 계수가 아니다. 각 쌍의 60도 영역을 5도 간격으로 검사하고, 기존 데이터에서 resolved된 개별 SOC축의 양자 선택 각도 4개를 추가해 총 88개 배경을 계산했다. Quantum hard-coordinate 재최적화는 하지 않았다.

[결과 JSON](../../../../data-space/verification/260917-y-stability/stability-check.json)에 24²/48²/96²의 Gamma 포함 reciprocal-cell grid와 다중 시작점 연속 최소화 결과를 저장했다. 기본 격자 평가 수는 1,064,448이다. 모든 배경에서 음의 정적 곡률 및 원점 이외의 복소 진동수를 검출하지 않았다. 원점 정적 nullity는 1이며, 고유벡터 projector를 ka<=0.1에서 추적한 위상 tangent overlap은 0.9971 이상, 인접 projector overlap은 0.9995 이상이다. 이는 원점 근처의 mode character 검사이고 전체 BZ의 band identity tracking은 아니다.

정적 lambda2의 정련 최솟값은 0.02009477 meV/cell이다. 두 번째 순서의 동적 에너지 정련 최솟값은 SOC에 따라 0.10388870–0.10458305 meV이며 정적 곡률과 구분한다. N48→96의 동적 최솟값 차이는 최대 4.58e-5 meV, N96에서 연속 정련으로 더 낮아진 크기는 최대 1.22e-5 meV이다. 정련의 모든 시작점은 수렴했다. 정적 양수성은 후보 배경의 local minimum 조건이며 경쟁상과의 에너지 비교나 전역 phase diagram 검증은 아니다.

검증은 Cartesian Hessian→Nambu 전체 행렬 변환(최대 차이 4.2e-17 meV), scalar/vectorized kernel 일치, reciprocal periodicity, 60도 회전 spectral set 비교, torque, 불안정 control(JPD=0.08 meV; 575/576 점의 음의 곡률과 복소 진동수)을 포함한다. Legacy anomalous-block 차이는 재확인하여 별도로 기록했다. 기존 pairing 회귀 8개가 통과했다. 첫 pytest 호출은 PYTHONPATH 누락으로 collection에 실패했으며 올바른 프로젝트 경로를 지정해 재실행했다. Legacy docstring의 기존 escape 경고 2개는 변경하지 않았다.

정적 cutoff=1e-10 meV/cell, 복소 진동수 cutoff=1e-8 meV는 조절 가능한 수치 검사값이며 고유값 오차 상한이나 물리적 퇴화 기준이 아니다. 원점의 defective zero pair가 만드는 약 3e-9 meV imaginary roundoff는 정확한 정적 nullity와 위상 tangent로 구분했다. 인위적인 diagonal shift나 pinning은 넣지 않았다. 전체 k 반전은 이번 최소값과 안정성 판정을 바꾸지 않으며 odd-in-k의 실험적 방향 convention은 미해결 상태로 유지했다.

최종 정련 포함 계산의 실측 시간은 약 74초, Python 계산 프로세스 최대 RSS는 약 110 MiB였다(Apple M1, 8GB; 다른 앱·OS·PDF 출력 메모리는 제외). 이는 ED/TN 또는 thermal MC 성능 측정이 아니다. [연구노트 새 절](../../../../docs/nbcp/research-note.md#sec-nbcp-y-full-zone-stability)에 방법, 수치, 세 패널 그림과 해석 한계를 기록했다. 다음은 rho(phi)와 E(phi)의 angular matching이며 경쟁상, density wall/vortex, thermal RG/MC, 양자 정규화는 미완료다. 사용자 물리·수학 acceptance 전까지 in-review를 유지한다.

출력·보존 검증: PDF 36쪽 전체 렌더링 및 새 절 15–17쪽 상세 검토를 완료했다. 기존 수식 79개를 보존하고 행렬 변환식 1개를 추가했다. 총 94개 semantic ID에 중복·미해결 참조가 없고 local link가 유효하다. 기존 code/legacy/data 136개 파일, 서식과 exporter의 hash가 유지되었다. [문서 검증 기록](../../../../data-space/verification/260917-y-stability/document-check.json)에 남겼다.

## 2026-09-17 — 새 대화로 인계

사용자가 현재 작업을 정리하고 다음 계산부터 새 대화에서 진행하도록 요청했다. [다음 대화 인계 문서](../../handoff/closed/260917-codex-to-codex-nbcp-angular-matching.md)에 완료 범위, 단일 원본, 수치 및 검증 근거, 미결 convention, 재현법과 Y angular matching의 첫 실행 순서를 기록했다. Task queue와 handoff map에 등록했다. 이 정리에서는 물리 계산·원고·PDF를 변경하지 않았고, 세 결과물의 hash가 최종 검증 기록과 일치함을 확인했다. 연구 이슈는 in-review, 다음 계산은 pending으로 유지한다.

## 2026-09-18 — Y angular matching

사용자의 인계 재개 요청에 따라 A의 범위만 수행했다. 기존 원고·PDF·stability JSON의 hash가 인계 기록과 일치함을 먼저 확인했다. B=0.2 T, 기존 SOC 7쌍을 유지하고, production·legacy·이전 데이터는 수정하지 않았다. [진단 스크립트](../../../../examples/nbcp_y_angular_matching.py), [계산 기록](../../../../data-space/verification/260918-y-angular-matching/angular-matching-check.json), [독립 대조 스크립트](../../../../examples/nbcp_y_angular_validation.py) 및 [독립 대조 결과](../../../../data-space/verification/260918-y-angular-matching/independent-check.json)를 추가했다.

- 고전 강성은 72/144/288/576개의 전체 원 각도에서 비교했다. Tensor component의 n=2,4 성분과 60도 회전 covariance를 확인했으며 두 eigenvalues, determinant, principal axes를 따로 기록했다. Pure Gamma에서는 eigenvalues와 determinant가 일정해도 principal axes가 회전한다. Angular mean tensor로 치환할 때 directional gradient energy의 최대 상대 오차는 pure PD=0.010에서 43.13%, 두 SOC=0.010에서 44.54%, pure Gamma=0.010에서 0.3356%이다.
- 기존 current-Hamiltonian N12/N24/N48 P72 개별 SOC축 배열을 재사용하고, N48 P144로 각도를 정련했다. 혼합 SOC는 N24/N48에서 별도 full-circle scan을 했으며, 6개 nonzero SOC점 모두 N96 P72로 momentum 수렴을 확인했다. Fourier 추출 전에 n=6 선택 규칙을 강제하지 않았다. 기존 full-BZ 안정성 계산은 재실행하지 않았다.
- Pure PD=0.010에서 lambda12/lambda6=0.002885, sixth-only energy error/lambda6=0.002886, 최소점 곡률 상대 오차=1.142%이다. 동일 강도의 mixed SOC도 곡률 오차가 1.143%이다. N48→96 lambda6 변화는 PD 포함 사례에서 상대 2.1e-5–3.7e-5이며 수렴 진단이지 오차 상한은 아니다.
- Mixed SOC의 sixth coefficient는 pure-axis coefficient의 합보다 크다. 단순 합은 mixed 결과 대비 0.005쌍에서 5.71%, 0.010쌍에서 13.96% 작다. 작은 pure-Gamma potential을 근거로 이 교차 효과를 무시할 수 없다.
- Pure Gamma의 lambda6는 0.005에서 약 2.5e-14, 0.010에서 약 1.643e-12 meV/spin이다. Empirical coefficient floor 약 1.8e-16 meV/spin 아래의 고차항을 0으로 해석하지 않았다. Unresolved sampled harmonics에 n²를 가중한 보수적 곡률 진단은 약 2.7e-12 meV/spin이다. Local five-point derivative에서는 작은 Gamma의 step 축소가 cancellation noise를 키우는 것을 재확인했다.
- 새 37개 비대칭 momentum의 동적 spectral covariance 오차는 9.44e-16 meV 이하, 49개 off-grid tensor 표현 오차는 3.91e-18 meV 이하이다. Cartesian→Nambu와 production 전체 행렬 차이는 5.57e-17 meV 이하이고 별도 direct-eigenvalue energy sum도 1.22e-17 meV/spin 내에서 일치했다. PD 포함 사례의 five-point curvature는 step=0.02에서 Fourier 값과 상대 2.70e-6 이내로 일치했다. 같은 모델의 계산 대조이며 독립 many-body 검증은 아니다.

계산 실측은 주 진단 약 284초, Python peak RSS 약 225 MiB이다. 독립 대조와 PDF 출력 시간은 제외한다. [단일 연구노트](../../../../docs/nbcp/research-note.md#sec-nbcp-y-angular-matching)에 근사 정의·오차·수치 분해능·그림을 반영했다. 고전 강성/leading zero-point potential의 혼합 차수, quantum gradient와 hard-coordinate 완화, bond-direction convention 및 thermal matching 미완료를 유지한다. 사용자 물리·수학 검토 전까지 `in-review`이며, 비선형 결함·RG/MC·경쟁상 계산을 시작하지 않았다. 인계는 done으로 닫고 활성 연구 이슈에서 후속 작업을 추적한다. Commit/push는 수행하지 않았다.

출력·보존 검증: PDF 40쪽 전체를 렌더링하고 contact sheet 및 새 절 17–21쪽을 상세 검토했다. 기존 수식 80개를 유지하고 4개를 추가했으며 semantic ID 100개에 중복·미해결 참조가 없다. 기존 code/legacy/examples/data 파일 166개의 hash와 exporter·서식을 보존했다. Quarto의 sandbox sysctl 제한은 허용된 escalation으로 해결했고 LaTeX overflow·누락 문자·미해결 참조는 없다. [문서 검증 기록](../../../../data-space/verification/260918-y-angular-matching/document-check.json)에 보존 파일 해시와 최종 출력 해시를 저장했다.


## 2026-09-18 — Nonlinear smooth-wave gradient test

사용자가 첫 비선형 시험을 승인하여 B=0.2 T, SOC (0.005,0), (0.010,0), (0.010,0.010) meV에서 수행했다. [진단 스크립트](../../../../examples/nbcp_y_nonlinear_gradient.py)와 [결과](../../../../data-space/verification/260918-y-nonlinear-gradient/nonlinear-gradient-check.json)를 추가했다. 각 magnetic cell의 arg(n_A^+−n_B^+)를 유한 진폭의 주기파로 고정하고 나머지 5개 좌표를 full classical energy로 완화했다. Regular Y chart 내 local minimum이며 density wall/vortex, quantum potential, finite T는 포함하지 않는다.

216조건, 최대 49,152 spins의 계산과 수치 대조가 Apple M1 8GB에서 40.6초, 최대 RSS 165 MiB로 완료됐다(figure/PDF 생성 제외). A=1 rad에서 full angular rho 에너지 오차 최대값은 L=64에서 0.90%, L=128에서 0.24%이며, rho(phi0) 고정은 각각 10.34%, 10.06%다. L 증가가 파장 증가임을 명시했으며, 이를 열적 finite-size scaling이나 모든 결함의 continuum 타당성으로 해석하지 않는다.

Cell phase와 site Fourier convention을 맞춘 finite-q harmonic 비교는 tiny amplitude에서 상대 오차 5.9e-7 이내다. Analytic gradient, explicit bond energy, norm/phase constraints, random all-cell starts와 fixed-wavelength torus doubling을 검증했다. [원고 절](../../../../docs/nbcp/research-note.md#sec-nbcp-y-nonlinear-gradient)에 정의·결과·그림·한계를 기록했다. 다음은 density wall 및 angular defect 범위 확정이며, quantum/thermal matching과 사용자 물리 acceptance는 미완료다.

검증 기록: 기존 code/legacy/examples/data 189개 파일의 hash와 기존 수식 84개를 보존했고 현재 수식은 86개다. Semantic ID 104개에 중복·미해결 참조·깨진 local link가 없다. PDF 42쪽 전체 contact sheet와 새 절 21–23쪽을 검토했고 LaTeX 경고는 없다. 216조건 중 1개 line-search abnormal flag는 residual gradient 3.52e-10 meV였으며 perturbed restart가 정상 종료하며 에너지를 1.23e-18 meV/spin 이내로 재현했다. [상세 검증](../../../../data-space/verification/260918-y-nonlinear-gradient/document-check.json)에 기록했다.


## 2026-09-18 — Classical density-domain walls

사용자의 후속 계산 승인에 따라 density wall부터 수행했다. [진단](../../../../examples/nbcp_y_density_wall.py), [정련](../../../../examples/nbcp_y_density_wall_validation.py), [그림 생성](../../../../examples/nbcp_y_density_wall_plot.py)을 추가했다. [주 계산](../../../../data-space/verification/260918-y-density-wall/density-wall-check.json)과 [정련 결과](../../../../data-space/verification/260918-y-density-wall/density-wall-validation.json)는 164개 해를 기록한다. B=0.2 T, zero SOC 및 (0.005,0), (0.010,0), (0.010,0.010) meV를 사용했다. Production/legacy와 기존 데이터는 수정하지 않았다.

Periodic magnetic-cell torus의 반대편에 translation-related Y domain 두 층씩을 고정하고 나머지 spins는 full sphere에서 완화했다. 열린 경계가 없으며 두 벽의 평균 초과 에너지를 측정한다. L=32,64,128에서 8개 pinned phase offset, 이후 L=256,512, W=4,8,16, 두 domain과 strip orientation, 여러 seed, broad phase twist 및 third-domain seed를 비교했다. Main 14.3초/95 MiB, refinement 58.6초; 최대 6,144 spins.

낮은 sampled branch의 L=512 평균 장력은 위 SOC 순서로 0.017505, 0.015691, 0.013461, 0.011646 meV/a이고 L=256 대비 변화는 최대 0.198%다. Density participation width는 3.09–3.80a이다. 이는 local variational minima의 수렴이며 모든 벽의 양의 장력에 대한 lower-bound proof가 아니다. Third-domain seed는 더 높은 multi-interface 해에 남았으므로 그 값을 elementary wall tension으로 해석하지 않는다. Pure PD의 반대 위상차에서 broad-twist seed가 직접 보간보다 낮은 해를 찾았다. 따라서 pinned-offset curve로 intrinsic phase jump나 fractional charge를 확정할 수 없다.

[원고 절](../../../../docs/nbcp/research-note.md#sec-nbcp-y-density-wall)에 정의와 한계를 반영했다. Density wall core는 microscopic scale이므로 smooth-wave matching만으로 처리할 수 없다. Vortex core, wall-vortex composite와 finite-T density stability는 아직 미계산이다. 사용자 물리 검토 전까지 in-review를 유지한다.

출력 검증: 기존 code/examples/legacy/data 195개 파일의 hash와 기존 수식 86개를 보존했다. 현재 수식 87개, semantic ID 107개에 중복·미해결 참조·깨진 local link가 없다. PDF 44쪽 전체와 새 절 23–25쪽을 확인했고 LaTeX 경고가 없다. 164개 해 중 한 line-search abnormal 종료는 projected force 6.39e-9 meV였고 perturbed restart가 정상 종료하며 장력을 8.25e-13 meV/a 이내로 재현했다. [검증 기록](../../../../data-space/verification/260918-y-density-wall/document-check.json)에 수치 및 source hash를 저장했다.


## 2026-09-18 — User-approved chapter reorganization

사용자가 Introduction → Model Hamiltonian → Phase Diagram → Skyrmion → Y/V common low-energy theory → Supersolidity criteria → Y → V → Discussion의 9장 구성을 승인했다. `docs/nbcp/research-note.md` 한 파일 안에서 기존 내용을 재배치하고, gap·RG·강성 상세 유도와 수치 근거를 부록 A–D로 분리했다. Skyrmion·V의 미계산 범위를 명시하고 기존 Gamma 명칭의 좌표 관계 및 Psi와의 구분을 유지했다.

[별도 코드 물리 검토 기록](260918-nbcp-physics-code-review.md)에 주장–구현–기존 근거–남은 검토 대응표와 보존된 legacy findings를 옮겼다. 현재 pass는 문서 재구성과 초기 implementation mapping이며 새로운 수치 물리 검증은 아니다. 기존 연구 계산의 다음 단계(vortex 및 thermal matching)는 유지하고, 문서·구현 검토를 별도 task queue 항목으로 등록했다.


## 2026-09-18: NBCP source format

The user-approved NBCP-only LaTeX migration is complete. The active manuscript
is [main.tex](../../../../docs/nbcp/main.tex) with separate chapters and
appendices. Historical Markdown paths below refer to the former source;
[the current editing guide](../../../../docs/nbcp/README.md) and
[the migration review](260918-nbcp-physics-code-review.md#2026-09-18-nbcp-only-latex-source-migration)
record the new roles and preservation checks. Numerical results, open physics
questions and the in-review status are unchanged.
