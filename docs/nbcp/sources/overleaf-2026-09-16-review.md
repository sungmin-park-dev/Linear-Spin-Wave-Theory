---
frontmatter-version: 1
title: Overleaf NBCP Source Review and Integration Record
doc-path: docs/nbcp/sources
status: in-review
last-edited-by: codex
created: 2026-09-16
updated: 2026-09-16
---

# Overleaf 원문 대조와 통합 기록

> 현재 편집 경로 (2026-09-18): [NBCP main.tex](../main.tex)와 장·부록별 TeX. 아래 대조·통합 기록의 Markdown 경로와 장 번호는 당시 상태이며, 현재 구조는 [편집 안내](../README.md)를 따른다.


확인한 프로젝트: [Linear Spin Wave Theory - Note](https://ko.overleaf.com/project/69bc9b43bd54009956d38ac5), 2026-09-16 로그인된 사용자 페이지. 파일 선택 후 Source Editor에서 전체 선택·복사로 원문을 읽었다. 아래 줄 번호와 문자 수는 그 시점의 편집기 텍스트 기준이다. 원격 파일은 수정하지 않았다.

이 기록은 **선별 발췌와 검토 기록**이며 전체 프로젝트 백업이 아니다. ZIP 다운로드 동작은 수행했으나 도구가 로컬 파일 경로를 제공하지 않아 저장·무결성을 확인하지 못했다. 그림 원본, bibliography, `Issue/`, `Meetings/`, LSWT 일반 노트는 이번 통합의 전수 검토 범위에 포함하지 않았다. 현재 표시된 `analysis.tex` PDF만으로 프로젝트 전체를 판정하지 않았다.

## 확인 범위와 본문 배치

| Overleaf 파일 | 확인한 텍스트 | 원래 역할 | 로컬 반영 |
|---|---|---|---|
| `main.tex` | 633줄, 36,746문자, 전체 | 물질 배경, NN 모델, 매개변수, 고전 임계장 | [연구노트](../research-note.md) 1–2장 |
| `analysis.tex` | 324줄, 18,923문자, 전체 | NN·NNN 교환행렬의 대칭 유도 | 1장 NN 유도; NNN 보류 |
| `analysis_tex_files/Three_MSL.tex` | 846줄, 36,737문자, 전체 | Y/UUD/Γ/V/P 분석 | 2장 상태 정의, 에너지와 임계장 |
| `analysis_tex_files/Stripe_MSL.tex` | 418줄, 16,282문자, 전체 | Stripe 고전 에너지와 안정성 초안 | 2장 경쟁 상태; 미검증 Hessian 결과 보류 |
| `analysis_tex_files/Four_MSL.tex` | 217줄, 9,263문자, 전체 | 4부격자 배치와 chirality 초안 | 2장 경쟁 상태와 위상 진단 구분 |
| `paper.tex` | 870줄, 38,046문자; 제목·초록·구조 확인 | 논문 형식의 별도 초안 | 본문에 `Three_MSL.tex` 전체가 정확히 포함됨을 문자열 비교로 확인; 중복 통합하지 않음 |

`main.tex`의 제목은 “Notes on Spin Supersolid Material”, 날짜는 October 28–30, 2024이다. 현재 열린 `analysis.tex`는 “Symmetry of the NBCP”라는 별도 문서다. `paper.tex`는 현재 arXiv 원고와 동일하다고 확인되지 않았으므로 출판본 provenance로 취급하지 않는다.

## 수식과 주장 대조

| 항목 | 원문 위치 및 관찰 | 통합본 처리와 근거 |
|---|---|---|
| XXZ ladder 계수 | `main.tex:135`는 $J_{xy}=J_\pm$, `:381`은 $J_{xy}=2J_\pm$ | Cartesian $J$를 기준으로 고정. $S^xS^x+S^yS^y=(S^+S^-+S^-S^+)/2$에서 $J=2J_\pm$ 유도 |
| 교환행렬 yy 성분 | `main.tex:361`은 `2J_{xy}-J_{\rm PD} \cos\varphi` | 같은 파일의 앞선 유도, `analysis.tex`, Gao2022 Eq. (2)와 대조해 $J-2J_{\rm PD}\cos\varphi$ 사용. 전자는 SOC=0에서도 xx=yy 조건을 위반 |
| 공간 반전에서 spin 변환 | `analysis.tex:199` 및 `main.tex` 대칭 유도에 $S^{-\alpha}$와 $(-1)^2$ | Spin은 axial vector. 성분 부호 반전 없이 site 교환으로 $J=J^{\mathsf T}$를 유도; 결론과 잘못된 중간 논리를 구분 |
| 3부격자에서 SOC 소거 범위 | `Three_MSL.tex:5`는 Hamiltonian의 의존성을 completely eliminate한다고 서술 | 균일 3부격자 **고전 에너지**에 한정. 유한 운동량 H2와 다른 unit cell에는 남음 |
| UUD Hessian 고유값 | `main.tex:587`의 첫 고유값은 $h/3-J_{xy}$ | 해당 행렬에서 직접 $h/3-JS$ 유도. 원문의 나머지 행렬과 최종 $h_{c1}=3SJ$에는 이미 S가 있어 중간식 불일치 |
| 임계장 값 | `Three_MSL.tex:373,529–538,750–753`의 0.1005, 0.3278, 0.4755 meV | 명시한 J=0.076에서 0.114000, 0.314316, 0.489000 meV 재계산. J=0.075 결과도 별도 표시. 과거 scan 결과를 재현했다고 하지 않음 |
| Γ와 V 명명 | `Three_MSL.tex:425–566` Γ와 `:566–781` V | Γ의 두 동일 canted spin을 가진 벡터 가족은 Rz(π), v=ψ−π로 현재 V에 대응. 원문 V의 A/B transverse 부호는 서로 반대이므로 동일한 가족이 아님 |
| Γ 미분식 | `Three_MSL.tex`, Classical Analysis의 $\partial_\psi E_\Gamma$ | 원문 $J_z\cos\psi\sin\theta$ 항은 에너지 미분과 불일치. 아래 독립 미분식을 기록하고 통합본문은 공통 signed-angle 에너지에서 V stationarity를 다시 유도 |
| Stripe 유한장 안정성 | `Stripe_MSL.tex:247` 이후 정확히 in-plane 상태에서 h 독립 Hessian을 논의 | h≠0이면 z 방향 선형 torque가 존재. 기울어진 정상상태를 먼저 구해야 함. 원문 Hessian과 임계 PD 수치를 검증 결과로 채택하지 않음 |
| NNN 축·index | `analysis.tex:264,313,315`: $K_{y\gamma}=0$ for γ=y,z, 그러나 제시 행렬은 Kyy 허용 | y 축 C2라면 x,z가 반전되어 off-diagonal yx,yz를 제한. 기준 bond 각과 회전량도 혼재. 독립 NNN 모델 복원은 보류 |
| SkX solid angle | `Four_MSL.tex:212`는 triple product에 절댓값 | signed chirality와 integer topological charge를 구분. 방향과 atan2 branch를 확인하기 전 Q를 계산 결과로 인용하지 않음 |
| 물질·실험 역사 | `main.tex:32`의 “Initially discovered in solid helium-4”, 표의 thermal transition을 QPT로 부르는 서술 | 검증되지 않은 역사·실험 판단을 재사용하지 않음. 물질 도입은 확인한 Gao2022 원문과 모델의 적용 조건 중심으로 재작성 |

Γ 초안의 에너지에서 직접 미분하면, 3-spin cell 기준으로

$$
\frac{\partial E_\Gamma^{\rm cell}}{\partial\psi}
=-6S^2J\sin\theta\cos\psi
+6S^2J_z\cos\theta\sin\psi-hS\sin\psi.
$$

원문에 있는 두 항의 동일한 `cos ψ sin θ` 조합과 다르다. 이는 원문 에너지의 미분 대조이며, 그 ansatz의 전역 안정성을 판정하는 결과가 아니다.

## 원문 선별 발췌

아래는 보존한 원문 조각이다. 오류를 수정한 버전이 아니며 위 대조표와 함께 읽는다.

```tex
% main.tex:135
To align with the convention, We choose four-parameters as $(J_{xy}, J_{z}, J_{\Gamma}, J_{\rm PD}) = (J_{\pm}, J_{z}, J_{z\pm}, J_{\pm\pm})$, and their values can be found in Tab.~\ref{tab: model parameter in Ref-Chi2024Dynamical}.
% main.tex:361
        -2J_{\rm PD} \sin\varphi        &   2J_{xy}-J_{\rm PD} \cos\varphi  & J_{\Gamma} \cos\varphi\\
% main.tex:587
    \lambda_1 = \frac{g_z \mu_0 B}{3} - J_{xy}, \quad \lambda_{2,3} = J_z S + \frac{J_{xy} S}{2} \pm \sqrt{\left( J_z S - \frac{g_z \mu_0 B}{3} - \frac{J_{xy} S}{2} \right)^2 + 2 (J_{xy} S)^2}.
% analysis.tex:199
        = J_{\boldsymbol\delta_{1}}^{\alpha\beta} \hat{S}^{-\alpha}_{j} \hat{S}^{-\beta}_{i}
% analysis_tex_files/Three_MSL.tex:5
This symmetry constraint eliminates the dependence of the Hamiltonian on $J_{\Gamma}$ and $J_{\rm PD}$ completely.
% analysis_tex_files/Three_MSL.tex:373
Here, for $(S, J_{xy}, J_{z}) = (1/2, 0.076 \, \mathrm{meV}, 0.125 \, \mathrm{meV})$, the critical points are $h_{c1} = 0.1005 \, \mathrm{meV}$ ($0.424 \, \mathrm{T}$) and $h_{c2} = 0.3278 \, \mathrm{meV}$ ($1.169 \, \mathrm{T}$).
% analysis_tex_files/Four_MSL.tex:212
    \tan \frac{\Omega_f}{2} = \frac{|\mathbf{n}_i \cdot (\mathbf{n}_j \times \mathbf{n}_k)|}{1 + \mathbf{n}_i \cdot \mathbf{n}_j + \mathbf{n}_j \cdot \mathbf{n}_k + \mathbf{n}_k \cdot \mathbf{n}_i}
```

## 저장·검증 범위

- 통합 본문: [research-note.md](../research-note.md). 1장 NBCP, 2장 phase, 3장 Y/V supersolidity.
- Gap 부록은 [같은 원본의 부록 A](../research-note.md#sec-nbcp-gap-appendix)에 병합했다. 본문과 함께 한 파일에서 편집한다.
- Clock 상세 유도와 수치 근거도 [같은 원본의 3장](../research-note.md#sec-nbcp-clock)에 병합했다.
- 이번 재계산: [phase-source-check.json](../../../data-space/verification/260916-nbcp-integration/phase-source-check.json). 임계장, V 미분, 상태 대응과 stripe bond count 검증.
- 최초 통합 당시의 문서 검증(후속 단일 파일 병합 이전): [document-check.json](../../../data-space/verification/260916-nbcp-integration/document-check.json). 기존 gap 노트의 display 수식 16개와 표의 46개 행(헤더·구분선 포함)이 모두 조립본에 보존되었다. 통합본 semantic ID 43개는 중복이 없고, 문서 링크와 수식 참조를 확인했다. PDF 19쪽 전체를 렌더링해 확인했고 최종 LaTeX 출력의 overfull box·누락 글자·미해결 참조 경고는 없다.
- 새 논문 gap curve 재계산, microscopic thermal simulation, iDMRG phase boundary 재현은 수행하지 않았다. 기존 gap·clock 데이터는 유지했다.

판정은 `in-review`이다. 원문의 교정 제안과 국소 수치 검사는 사용자의 물리·수학 검토 및 arXiv 주장 검증 완료를 뜻하지 않는다.
