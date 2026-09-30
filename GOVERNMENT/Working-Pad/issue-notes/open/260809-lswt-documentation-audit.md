---
frontmatter-version: 1
title: Current LSWT Documentation Audit
section: issue-notes/open
issue-type: review
status: in-review
last-edited-by: claude
created: 2026-06-03
updated: 2026-09-30
---

# Current LSWT Theory Documentation Audit

> Snapshot: 2026-09-06. 작업 재개 점검에 이어 Zeeman convention의 사용자 결정을 반영했다.
> Inventory baseline은 2026-07-31, concept ownership mapping과 `docs/` migration은
> 2026-08-09 기준이며, 이번 기준 문서 보완으로 이론 coverage를 올리지는 않았다.

이 문서는 `docs/lswt/` 이론 문서의 coverage, lifecycle, source review, legacy consolidation과
열린 물리 질문을 추적한다. 독자용 이론 본문이 아니므로 Working-Pad에서
관리한다. 이 audit 갱신은 이론 식이나 코드가 검증되었다는 뜻이 아니다.

## Approved Source Authority Snapshot

- Primary evidence:
  `docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf`
- Editable transcription:
  `docs/lswt/sources/01-editable-notes/note_lswt_reviewed.tex`
- Structural reference:
  `docs/lswt/sources/01-editable-notes/note_lswt_restructured.tex`
- Unique canonical authoring surface: `docs/`의 이론 내용 Markdown
- User-approved accepted canon: none
- Legacy converted sources:
  `legacy/research-notes/lswt/converted-markdown/`

Markdown 단일 정본 원칙은 2026-08-01 사용자가 승인했고, active authoring
surface를 `docs/`로 올리는 구조는 2026-08-09 승인했다. 기존 결정 원문은
`GOVERNMENT/Court-Precedents/2026-08-01-lswt-markdown-source-authority.md`에
있으며 새 경로를 반영하는
[후속 결정](../../../Court-Precedents/2026-08-09-lswt-docs-authoring-surface.md)은
2026-09-05 기준 문서 보완에서 Court-Precedents로 옮겨 기록했다.
Routing 승인은 아래 draft 또는 skeleton의 이론 내용을 승인한 것이 아니다.

Navigation과 읽는 순서의 기준은 `docs/README.md`다. 아래 inventory는 coverage,
lifecycle과 review boundary를 추적하며 독자용 navigation을 대체하지 않는다.

## Work Resumption — 2026-09-05

사용자는 이전 작업 상태와 지침 점검 결과를 확인하고, 예상 독자·문서별 설명 깊이·원문 보존·검토 완료 기준을 기존 기준 문서에 보완하도록 승인했다. 이번 승인 범위는 기준 문서 정비이며 이론 본문 수정은 다음 작업으로 남긴다.

| 항목 | 확인한 상태와 다음 행동 |
|---|---|
| 이전 본문 작업 | 2026-08-13 Overview 본문과 분류 그림 보완까지 반영돼 있다. 2026-08-18에는 notation의 필요한 부분 점검 후 Bilinear Spin Hamiltonian 검토를 다음 작업으로 제안했다. Overview의 사용자 물리·수학 acceptance는 대기다. |
| 기준 문서 | Writing style에 독자와 문서별 설명 깊이, lifecycle에 원문 이식과 작업 단위별 완료 기준을 반영했다. AGENTS와 CLAUDE의 문서·코드 검증 범위를 맞추고 `docs/README.md`를 기준 문서 진입점으로 정리했다. 기존 `in-review` 정책 전체를 accepted로 승격한 것은 아니다. |
| 현재 작성 수준 | 일부 본문 7개, skeleton 10개, accepted 0개다. Overview를 제외한 16개 이론 파일에 한국어가 남아 있다. 문체 교정과 미작성 내용 보완을 구분해야 한다. |
| 표기 적용 | Notation이 정한 대각화 전 $a$, cell index $i,j$, 크기 기호와 달리 momentum-space 초안은 $b$, $n$, $L$, $m_s$를 사용한다. Local-frame 초안의 classical vector hat과 matrix typography도 기준 대조가 필요하다. 이번에는 이론 파일을 수정하지 않았다. |
| 작성 범위 | Real-space 문서의 작업 메모는 $H_4$ 이식을 요구하지만 아래 Open Physical Decisions에서는 포함 범위가 미결정이다. 원문에 있다는 이유만으로 범위를 확장하지 않고, 해당 단위 검토 시 이 충돌을 해결한다. |
| 다음 본문 단위 | `bilinear-spin-hamiltonian.md`를 원본 PDF의 Source Eq. (1)–(3)과 대조한다. 필요한 notation만 먼저 점검하고, 영어 본문·interaction support·link counting·Zeeman convention과 작업 메모 분리를 다룬다. 본문 수정은 대기다. |

현재 완료한 것은 작업 상태 복원과 기준 문서 보완이다. 원문 수식의 재검증, 사용자 물리·수학 acceptance와 코드 검증은 수행하지 않았다. 문서별 검토 범위와 완료 근거는 [lifecycle](../../../Agents-Bylaws/procedures/lswt-canonical-document-lifecycle.md)의 작업 단위 기준에 따라 이 audit에 이어 기록한다.

기준 문서 검증: 변경한 문서의 내부 링크·frontmatter·공백과 AGENTS/CLAUDE의 공통 지침 일치를 확인했다. 작업 시작 시점의 파일 내용과 대조해 승인 범위 밖의 변경이 없고, 이론 본문·원자료·코드와 2026-08-01 결정 원문이 그대로임을 확인했다.

## Corpus Editorial Pass — 2026-09-06

사용자는 문서마다 순차적으로 승인을 반복하는 대신, 공통 말투와 톤을 정하고 한 번에 수정하도록 승인했다. 이 작업은 기존 writing policy를 기준으로 `docs/lswt/00-foundations`부터 `docs/lswt/04-appendices`까지 17개 문서에 적용했다. 이후 사용자 물리·수학 검토와 acceptance는 별도다. 이번 일괄 편집 승인은 기존 미결 물리 가정 전체의 승인을 뜻하지 않는다.

### 적용 범위

- Writing style에 차분한 영어 이론 강의노트의 서술 방식과 간단한 예시를 보완했다. 물리적 대상 → 정의·식 → 조건·의미를 연결하되 문서마다 같은 문단 구조를 강제하지 않는다. 정책 상태는 `in-review`를 유지한다.
- 본문이 있는 7개 문서는 영어 서술, 문단 연결, heading, 수식 주변 정의와 reference description을 정리했다. Overview는 기존 톤을 유지하고 h를 Zeeman-energy coefficient로 명확히 했다.
- 목차 수준인 10개 문서는 영어 scope와 topic outline으로 정리했다. source와 source-section을 가능한 경우 frontmatter로 옮겼다. 새 유도를 작성한 것으로 보지 않으며 coverage는 계속 skeleton이다. Luttinger–Tisza의 source-section과 검증된 참고문헌은 여전히 미정이다.
- 작업·검증·Common 후보 메모와 notation의 코드 대응을 아래 이관 기록에 보존했다. 새 Common 문서를 만들지 않았다. Reference description의 citation-key 작업 문구는 본문에서 제거했지만 source bibliography의 키와 원자료는 변경하지 않았다.

### 수식과 표기 변경

| 대상 | 반영한 변경 | 의미 또는 검토 경계 |
|---|---|---|
| Hamiltonian | 앞선 원문 대조로 준비한 영어 수정안 반영. reverse matrix transpose와 비결합 쌍의 zero matrix 조건, on-site Hermiticity 설명, A>0 조건과 spin-1/2 상수항 유도 보충 | primary PDF p. 3 Source Eqs. (1)–(3) 및 앞선 독립 대수 검토를 근거로 한다. 네 semantic equation ID는 유지한다. 새 spin-1/2 결과의 사용자 검토는 대기다. |
| Zeeman | 양의 scalar g, 명시적 electron minus 및 h = -mu_B g^T B 유지 | 같은 날의 사용자 결정을 변경하지 않았다. identity matrix typography만 기존 규칙에 맞췄다. |
| Local frame | classical n/e에서 hat 제거, quantum local spin에는 hat 추가, R/J를 sans serif로 정리, varphi를 phi로 통일, imaginary unit를 upright로 정리 | 회전행렬 성분, local-to-laboratory 방향과 circular basis의 ± 부호 및 sqrt(2) 계수는 유지한다. |
| HP | local spin operator hat과 classical basis typography 정리; lambda의 역할과 stationarity의 논리 설명 | 원래 leading expansion, truncation 차수와 stationarity 식은 유지한다. |
| Real space | Hamiltonian operator hat 추가; circular placeholder indices를 Cartesian alpha/beta와 구별한 p,q로 정리 | 실제 ++,+-,-+,--,00 성분과 계수는 유지한다. Circular contraction definition은 미확정이라고 명시한다. |
| Momentum space | diagonalization 전 b→a, cell n→i, L→N_uc, m_s→N_sub, BZ_mag→MBZ, Nambu/operator hats, exp와 transpose typography 정리 | Fourier 부호, normalization 계수, BdG block 배치, conjugation 및 trace subtraction은 변경하지 않았다. Trace/constant normalization은 provisional로 표시한다. |
| Notation | shared conventions를 영어로 정리하고 source mapping 보존 | circular basis의 reverse orientation이 일반적으로 dagger라는 기존 문장은 component definition이 없어 미결로 분리했다. 이를 transpose 또는 dagger로 새로 확정한 것은 아니다. |

### 내용 보존 및 보류

- Hamiltonian의 기존 source disposition은 primary Source Eqs. (1)–(3), link footnote 및 A4/C22를 유지한다. 원본의 Mott-insulator 일반 배경 문장은 이전 초안부터 빠져 있었으며, 이번 정의 중심 범위에서도 재삽입하지 않았다. 위치는 primary PDF p. 3의 Source Eq. (3) 앞이다.
- Single-ion example은 A>0에서 easy-axis라는 조건을 명시했다. Spin-1/2의 real symmetric quadratic on-site term이 Tr(D)/4 times identity라는 결과는 본문에 반교환관계로 유도했다. LSWT 수치 계산으로 검증한 결과가 아니다.
- Real-space H2 초안에는 독립적인 on-site D 항이 포함돼 있지 않다. 따라서 표시된 식의 범위를 inter-site exchange와 Zeeman으로 한정해 서술했다. D의 이식은 별도 내용 보완 대상으로 남긴다.
- H4와 odd terms를 반드시 이식하라는 이전 작업 문구는 과거 기록으로 아래에 보존한다. 현재 포함 범위는 기존 Open Physical Decisions에 따라 미정이며, 이번 편집으로 범위를 확장하지 않았다.
- Fourier/gauge, A1/A2/A5/A10, energy constant, response normalization, Goldstone와 topology 범위의 미결 상태는 유지한다. 본문에 보이는 provisional 조건과 audit의 미결 항목은 서로 대응한다.
- 기존 DOI `10.1103/PhysRevLett.128.117201`의 author/title이 `D. Go and H.-W. Lee, Thermal Hall Effect of Magnons`로 잘못 기록돼 있었다. [APS 출판사 페이지](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.117201)에 따라 R. R. Neumann, A. Mook, J. Henk, I. Mertig 및 실제 전체 제목으로 수정했다. Source bibliography에도 같은 잘못된 metadata가 있으나 원자료는 변경하지 않았다. 이 논문과 향후 thermal Hall 식의 세부 대응은 아직 검증하지 않았다.

### 다음 검토 묶음

문체 확인은 개별 문장 승인 대신 이 편집본 전체로 진행한다. 내용 보완은 (1) local circular component 정의와 real-space H2의 on-site D 포함, (2) Fourier 및 BdG normalization, (3) diagonalization과 observable skeleton 이식으로 묶어 진행한다. 특히 첫 묶음의 정의가 뒤 공식의 의미를 결정하므로 그 물리적 선택부터 확인한다. 목차 10개를 본문으로 완성하는 작업과 모든 문서의 Human Physics and Mathematics Review는 아직 남아 있다.

### 검증

17개 이론 문서의 영어 본문, source metadata, 내부 링크, 수식 delimiter와 brace를 확인했다. 기존 네 semantic equation ID는 모두 유지되며 중복이 없다. Overview, local-frame, HP와 real-space display 식은 명시한 typography 및 index 치환을 정규화한 뒤 이전 식과 일치했다. Momentum-space 식도 b→a, cell n→i, L→N_uc, m_s→N_sub 및 exponential typography 치환 후 일치하며, Fourier 부호·계수·conjugation·BdG 배치를 바꾸지 않았다.

현재 Markdown owner에서 임시 Quarto reader preview를 생성했다. 안내 페이지를 포함한 HTML 18개, 페이지별 단일 제목, 모든 local link·anchor·그림·resource와 MathML 출력을 확인했다. 출력에는 MathML error node가 없었다. 브라우저가 local-file URL을 보안정책으로 차단해 실제 화면의 시각 검토는 수행하지 못했다. 이 제한을 우회하지 않았으며 시각 검토는 대기 상태다. Preview는 임시 파생물이며 이론 source가 아니다.

변경 전 snapshot과 비교해 변경 범위가 이론 문서 17개, writing policy, 기존 audit와 TASK-QUEUE의 총 20개 파일임을 확인했다. 원본 PDF·TeX·bibliography, 코드·테스트·실행 예제 및 다른 governance 문서는 보존했다. 파일 상태는 기존 draft/in-review를 유지한다. 이 검사는 편집과 출력 구조의 검증이며, 원문 전체 재검증·사용자 물리·수학 acceptance·코드 검증 또는 10개 skeleton의 본문 완성을 뜻하지 않는다.

## Zeeman Convention Decision — 2026-09-06

사용자는 $g$를 양수로 두고 부호를 식에 명시하도록 결정했다. 부호를 $g$에 흡수해 원문의 Zeeman minus를 유지하려던 제안은 철회했다. 이 승인은 Zeeman convention에 한정하며, Hamiltonian 전체 영문 수정안이나 formal tensor reality 조건 전체의 승인을 뜻하지 않는다.

- `bilinear-spin-hamiltonian.md`의 Zeeman section에 $\hat{\boldsymbol\mu}_I=-\mu_B\mathsf g_I\hat{\mathbf S}_I$를 명시했다. $g_I>0$와 $\mu_B>0$를 사용한다. 스핀 연산자는 $\hbar$ 단위의 무차원 연산자다.
- $\hat H_Z=-\sum_I\mathbf B_I\cdot\hat{\boldsymbol\mu}_I$이므로 field contraction은 $+\mu_B\sum_I\mathbf B_I^{\mathsf T}\mathsf g_I\hat{\mathbf S}_I$가 된다. 기존 내부 정의 $\hat H_Z=-\sum_I\mathbf h_I\cdot\hat{\mathbf S}_I$를 유지해 $\mathbf h_I=-\mu_B\mathsf g_I^{\mathsf T}\mathbf B_I$로 맞췄다.
- 양수라는 결정은 scalar $g$ 및 principal-axis coupling magnitude에 적용한다. 임의 Cartesian basis의 모든 matrix entry가 양수라고 해석하지 않는다. 일반 tensor의 기저 선택과 formal reality 조건은 아래 Open Physical Decisions item 3에서 계속 추적한다.
- Source difference: primary PDF p. 3, Source Eqs. (1), (3)은 $\mathbf h=+\mu_B\mathsf g^{\mathsf T}\mathbf B$와 음의 field contraction을 사용한다. 원자료는 보존하고, 이번 사용자 convention 결정에 따른 차이를 여기 기록한다. Toth–Lake arXiv:1402.6069v4 p. 2 Eq. (1)은 양의 Zeeman contraction을 사용한다. NIST Atomic Spectroscopy의 Zeeman 설명도 양의 electron spin $g$와 양의 energy shift를 사용한다.
- 영향: 이 변경은 $\mathbf B\mapsto\mathbf h$ 변환의 부호를 바꾼다. $\mathbf h$로 쓴 뒤 문서의 식은 같은 정의를 유지한다. 현재 notation은 코드의 `magnetic_field`가 $\mathbf h$에 대응한다고 기록하므로, tesla field로부터 입력을 만드는 코드·예제의 부호 검증은 후속 작업이다. 코드는 이번에 수정하지 않았다.
- 승인 범위 밖의 link counting 설명 보완, spin-1/2 on-site 항의 유도, 나머지 영어 교정 및 작업 메모 이동은 전체 수정안 검토 대상으로 남긴다. 파일 상태는 `draft`를 유지하며 전체 Human Physics and Mathematics Review 완료로 기록하지 않는다.

검증 결과: 자기모멘트의 명시적 minus, 양의 field contraction과 $\mathbf h=-\mu_B\mathsf g^{\mathsf T}\mathbf B$가 서로 일치함을 대수적으로 확인했다. 기존 semantic equation ID, draft 상태 및 내부 링크를 유지했다. Zeeman section·외부 근거·수정일 외 본문은 변경 전과 동일하며, 다른 이론 문서·원자료·코드·예제도 변경 전과 동일함을 확인했다. Quarto로 MathML HTML preview를 생성하고 출력에 Zeeman 수식 anchor와 갱신한 부호 정의가 들어 있음을 확인했다. 출력과 resource는 임시 디렉토리에 두었으며, source 옆의 생성 resource는 정리했다. 브라우저에서의 시각 검토, 문서 전체 Human Physics and Mathematics Review 및 legacy 코드 검증은 별도 대기다.

## Topology Draft — 2026-09-30

사용자의 연구·검토 항목 C 지시에 따라 `docs/lswt/02-observables/topological-magnon-quantities.md`를 skeleton에서 본문 draft로 작성했다. 구현·검증된 toolkit 부분(stage 5a–5d, D29, D31)과 대응하는 문서부터 진행한다. 작성 전 writing style(`in-review`)과 lifecycle을 읽었다. Primary PDF p. 24 Source Eqs. (143)–(149)와 reviewed TeX를 대조했다. 이 draft는 `status: draft`이며 사용자 물리·수학 acceptance를 받지 않았다.

### 검토 묶음

| 항목 | Draft의 처리 | 사용자 확인 사항 |
|---|---|---|
| Skyrmion number (Source Eq. (143)) | 원문의 절댓값과 arctan branch 대신 부호 있는 Berg–Lüscher solid angle, 즉 \(e^{i\chi/2}\propto 1+\mathbf m_I\cdot\mathbf m_J+\dots+i\,\mathbf m_I\cdot(\mathbf m_J\times\mathbf m_K)\)를 쓴다. 원문의 plaquette 예시는 elementary triangle 분할로 바꿨다. | 사용자 결정(2026-09-30): LSWT 문서 scope에 포함. 부호 있는 정의는 물리·수학 검토에서 확인 |
| Berry curvature (Source Eqs. (144)–(146)) | \(T^\dagger\Sigma_3T=\Sigma_3\), 2N column 합과 Σ3 부호, particle–hole 분모를 명시했다(A6). Gauge 문단에서 \(e^{\pm i\mathbf k\cdot\mathbf r_I}\)를 두 Nambu block에 같게 곱하면 curvature의 추가 항이 주기 함수의 curl이라 적분이 0임을 적었다. | 합 범위와 gauge 문단 |
| Chern number | \(A_{\mathrm{MBZ}}/N_{\mathbf k}\) 합과 paraunitary link의 FHS를 적었다. 정수 값만으로 band 고립을 판정할 수 없다고 명시했다. | 수치 판정 기준(D31)은 코드 계약으로만 두고 본문에 넣지 않았다. |
| Thermal Hall (Source Eqs. (147)–(149)) | 층당 \(\kappa^{\mathrm{2D}}_{xy}=-(k_B^2T/\hbar)(N_{\mathbf k}A_{\mathrm{uc}})^{-1}\sum c_2\Omega\), 3D는 \(/d\), pair form을 적었다. 원문의 V와 ħ 누락을 교체했다. | 단위·정규화(A7 앞부분은 2026-09-11 사용자 승인과 같음) |
| −π²/3 상수항 (A7) | 원문 형태가 이 draft보다 \((\pi k_B^2T/6\hbar)\sum_nC_n\)만큼 크다는 관계를 적었다. 문헌 대조 후 양정치 H에서 \(\sum_nC_n=0\)(Shindou et al. 2013 식 (29), \(\mathsf H_\lambda=(1-\lambda)\mathsf H+\lambda\mathbb 1\) 변형)으로 두 형태가 같다고 교체했다. | 반영 완료(2026-09-30 사용자 승인). 양반정치(Goldstone) 경우는 open. 코드 변경 없음. |
| c2 명칭 | 원문의 "c2 is the Spence function" 문장은 옮기지 않고 \(\mathrm{Li}_2\)로 정의했다. | 사용자 동의(2026-09-30) |
| 생략 문장 | 원문의 관측 난이도 서술("hard to observe" 취지)과 B9 orphan text는 옮기지 않았다. | 사용자 동의(2026-09-30) |

표기 변경: J→Σ3, FBZ→MBZ, \(\varepsilon_{n,k}\)→\(\varepsilon_{n\mathbf k}\), V→\(N_{\mathbf k}A_{\mathrm{uc}}\)와 명시적 ħ. Semantic equation ID 6개(`eq-lswt-lattice-skyrmion-number`, `eq-lswt-bdg-berry-curvature`, `eq-lswt-magnon-chern-number`, `eq-lswt-magnon-thermal-hall`, `eq-lswt-thermal-hall-weight`, `eq-lswt-thermal-hall-pair-form`)를 새로 부여했다.

### 문헌 대조 — −π²/3 (2026-09-30)

| 문헌 | 가중치 | 확인 내용 |
|---|---|---|
| Matsumoto–Murakami, PRB 84, 184406 (2011) 식 (23) | c2 | 상수항 없음, anomalous term 없는 ferromagnet |
| Neumann et al., PRL 128, 117201 (2022) 식 (8) | c2 | 원본 노트가 인용한 식. physical band N개 합, Chern 합 0 언급 |
| Zhang–Gao–Chen, arXiv:2305.04830 식 (1)–(4) | c2 | 고온 극한 κ/T → −(π²k_B²/6ħ)ΣC_n = 0 |
| arXiv:2606.16704 식 (8) | c2 − π²/3 | "Matsumoto–Murakami convention"이라 인용하나 2011 원 논문에는 상수항이 없다 |
| Shindou–Matsumoto–Murakami–Ohe, PRB 87, 174427 (2013) 식 (29) | — | 양정치 bosonic BdG에서 particle band Chern 합 0 증명 |

Matsumoto–Shindou–Murakami, PRB 89, 054420 (2014)은 초록만 확인해 −π²/3의 출처인지 확인하지 못했다.

### 검증

Quarto HTML preview(embed-resources, MathJax)를 scratchpad에 생성했다. 경고가 없었고, equation ID 6개가 한 번씩 출력됐으며, display 식 9개의 delimiter와 brace가 균형을 이뤘다. 브라우저 pane이 local-file URL을 열지 못해 화면 시각 검토는 사용자 preview로 넘긴다. 이 검사는 출력 구조 검증이며 물리·수학 acceptance가 아니다. Notation 문서에서 Fourier gauge와 thermal-Hall 단위가 "not yet fixed"인 상태는 바꾸지 않았다.

## Active Workspace Inventory

### Draft Content

| File | Coverage | Lifecycle | Next verification boundary |
|---|---|---|---|
| `docs/lswt/00-foundations/lswt-overview.md` | SWT와 HP-LSWT의 범위, 적용 조건과 reference-state 분류 | `draft` | 보완된 prerequisite/self-consistency 구분과 2D finite-temperature caveat의 사용자 검토 |
| `docs/lswt/00-foundations/notation-and-conventions.md` | site, link, displacement convention | `in-review` | 첫 Human Physics and Mathematics Review와 deferred convention 확인 |
| `docs/lswt/00-foundations/bilinear-spin-hamiltonian.md` | bilinear Hamiltonian, exchange, Zeeman term | `draft` | Human review와 on-site term의 code representation 대조 |
| `docs/lswt/00-foundations/classical-order-and-local-frame.md` | classical direction, rotation, local basis | `draft` | complex-basis exchange tensor 정의 확인 |
| `docs/lswt/01-derivation/holstein-primakoff-expansion.md` | Leading HP expansion과 linear-term condition | `draft` | exact HP, expansion hierarchy, Dyson-Maleev 범위 결정 |
| `docs/lswt/01-derivation/real-space-boson-hamiltonian.md` | Quadratic \(H_2\) | `draft` | odd terms와 \(H_4\)의 문서 범위 결정 |
| `docs/lswt/01-derivation/momentum-space-bdg-hamiltonian.md` | General Nambu/BdG form | `draft` | A1, A2, A5, A10 해결 전 explicit block 보류 |
| `docs/lswt/02-observables/topological-magnon-quantities.md` | Lattice skyrmion number, BdG Berry curvature, Chern number (Kubo, FHS), per-layer magnon thermal Hall | `draft` (2026-09-30 본문 작성) | 아래 Topology Draft 검토 묶음의 사용자 물리·수학 검토 |

### Draft Skeletons

| File | Intended coverage | Lifecycle |
|---|---|---|
| `docs/lswt/01-derivation/paraunitary-diagonalization.md` | Bogoliubov transformation, Colpa, stability | `draft` skeleton |
| `docs/lswt/02-observables/magnon-observables.md` | Spectrum, energy correction, occupation, correlation matrix | `draft` skeleton |
| `docs/lswt/02-observables/thermodynamics.md` | Partition function, energy, entropy, specific heat | `draft` skeleton |
| `docs/lswt/02-observables/spin-correlations.md` | Real-time and sublattice correlations | `draft` skeleton |
| `docs/lswt/02-observables/structure-factor-and-spectral-function.md` | Static/dynamic structure factor and spectral function | `draft` skeleton |
| `docs/lswt/03-examples/worked-example.md` | Single-mode quadratic-boson example | `draft` skeleton |
| `docs/lswt/04-appendices/luttinger-tisza-method.md` | Luttinger-Tisza method | `draft` skeleton; source TODO |
| `docs/lswt/04-appendices/paraunitarity-proofs.md` | Paraunitarity proof material | `draft` skeleton |
| `docs/lswt/04-appendices/thermodynamic-derivations.md` | Entropy and correlation-matrix derivations | `draft` skeleton |

현재 합계는 일부 본문이 작성된 draft 8개, skeleton 9개, accepted 0개다.

## Source and Legacy Retention

| Material | Current judgment |
|---|---|
| Primary PDF | 27 pages. Source review annotation 33개, unique review ID 31개를 포함 |
| `note_lswt_reviewed.tex` | Primary PDF의 section order와 review ID에 대응하는 전사 보조 자료 |
| `note_lswt_restructured.tex` | 구조 재편 참고 자료. 과거 editable-master 지정은 superseded됐고, Primary PDF와 section order가 다르며 현재 clean build가 완료되지 않음 |
| `note_lswt_restructured.pdf` | Primary PDF와 다른 generated snapshot |
| `converted-markdown/sections/*.md` | 상세 유도가 아직 남아 있으므로 source-only 상태로 보존 |
| `converted-markdown/notation.md` | Symbol 이식이 끝날 때까지 source-only 상태로 보존 |
| `converted-markdown/common-candidates/` | 과거 cross-solver convention candidate. Accepted canon이 아니며 unique content만 concept owner로 검토 이식 |
| `converted-markdown/navigation-snapshots/` | `docs/README.md`로 병합하기 전 navigation의 historical snapshot |

Legacy Markdown에는 오래된 표기, 변환 오류, 최신 source와 다른 정규화가
있다. 따라서 직접 정본으로 승격하지 않고 PDF와 TeX를 구간별로 대조한다.

## Concept Ownership and Legacy Coverage

이 표는 과거 Markdown의 내용을 현재 concept file로 통합하기 위한 structural
mapping이다. `partial`은 일부 본문이 있으나 source coverage를 아직 구간별로
검증하지 않았다는 뜻이고, `skeleton`은 소유 파일만 있으며 상세 이식 전이라는
뜻이다. `source-reviewed`는 primary PDF, reviewed TeX와 legacy Markdown의 해당
구간을 대조하여 안전한 통합 내용을 반영했지만 human acceptance는 아직 받지
않았다는 뜻이다. 어느 상태도 물리식이나 수학 전개가 최종 검증됐다는 뜻이
아니다.

Concept ownership은 다음 원칙을 따른다.

- 한 정의, claim 또는 수식은 한 canonical content file만 소유한다.
- `notation-and-conventions.md`는 symbol namespace와 표기 계약만 소유하고,
  Hamiltonian이나 observable의 물리적 정의는 해당 concept file이 소유한다.
- Overview는 흐름에 필요한 핵심 관계식을 요약하고 소유 문서로 연결할 수 있다.
  상세 정의, convention, 유도와 semantic equation ID는 owner가 소유한다.
  Physical-quantity index는 각 observable owner로 연결한다.
- Main document는 정의, 가정과 최종 결과를 소유하고 appendix는 긴 증명과
  보조 유도만 소유한다.
- Worked example은 정해진 정의를 적용하며 일반 이론을 다시 정의하지 않는다.
- Legacy file은 아래 모든 concept의 coverage가 확인될 때까지 source-only로
  보존한다.

| Legacy source | Concept | Canonical owner | State | Duplicate and consolidation boundary |
|---|---|---|---|---|
| `notation.md` | Custom LaTeX commands | renderer/template layer | `source-only` | Theory claim이 아니므로 content file로 이식하지 않는다. 필요한 macro만 출력 계층에서 결정한다. |
| `notation.md` | Cell, site, sublattice, band와 geometry index | `docs/lswt/00-foundations/notation-and-conventions.md` | `source-reviewed` | 다른 문서는 기호를 사용하고 index를 재정의하지 않는다. Source와 다른 cell/site position 기호는 mapping에 기록했다. |
| `notation.md`, `01_spin_wave_theory_intro.md` | Link orientation, displacement와 exchange symbol | `docs/lswt/00-foundations/notation-and-conventions.md` | `partial` | Link vocabulary와 geometry만 소유한다. Hamiltonian counting 식은 아래 bilinear owner로 보냈다. Detailed displacement and basis-gauge coverage는 이후 source 구간 대조가 필요하다. |
| `notation.md`, `01_spin_wave_theory_intro.md` | One-link counting, ordered-pair factor와 interaction support | `docs/lswt/00-foundations/bilinear-spin-hamiltonian.md` | `source-reviewed` | Notation 문서의 중복 물리 설명을 owner link로 줄이고, one-link factor와 별도 on-site sum을 Hamiltonian owner에 통합했다. |
| `notation.md` | Boson, magnon, Nambu, matrix와 energy notation | `docs/lswt/00-foundations/notation-and-conventions.md` | `source-reviewed` | Source symbol을 canonical symbol로 대응했다. Response matrix와 energy correction의 세부 정의는 계속 open이다. |
| `notation.md`, `02_physical_quantities.md` | Thermodynamic symbol definitions | `docs/lswt/02-observables/thermodynamics.md` | `skeleton` | Notation 문서는 symbol reservation만 남기고 물리적 정의와 식은 thermodynamics가 소유한다. |
| `notation.md`, `02_physical_quantities.md` | Correlation symbols | `docs/lswt/02-observables/spin-correlations.md` | `skeleton` | Real-time/equal-time correlator symbol과 index scope만 이 owner에서 정의한다. |
| `notation.md`, `02_physical_quantities.md` | Structure-factor and spectral symbols | `docs/lswt/02-observables/structure-factor-and-spectral-function.md` | `skeleton` | Fourier response와 spectral-function symbol은 이 owner에서 정의한다. |
| `01_spin_wave_theory_intro.md` | LSWT scope, assumptions와 calculation flow | `docs/lswt/00-foundations/lswt-overview.md` | `partial` | SWT와 HP-LSWT의 범위, reference-state 분류와 validity condition을 overview 수준으로 반영했다. Prerequisite, harmonic stability와 a posteriori spin-reduction check를 분리하고 2D finite-temperature caveat를 추가했으며, 사용자 검토 전에는 `covered`로 올리지 않는다. |
| `01_spin_wave_theory_intro.md` | Bilinear Hamiltonian, exchange, Zeeman term과 g-tensor | `docs/lswt/00-foundations/bilinear-spin-hamiltonian.md` | `source-reviewed` | 원본 Eq. (1)--Eq. (3)의 link counting, on-site support와 Zeeman 부호를 한 owner에 통합했다. On-site matrix는 $\mathsf D_I$로 분리하고 $\mathsf J_\ell$는 real, $\mathsf D_I$는 real symmetric으로 정했다. Code data structure 검증은 별도다. |
| `01_spin_wave_theory_intro.md` | Classical stationarity, spin direction, rotation과 local complex basis | `docs/lswt/00-foundations/classical-order-and-local-frame.md` | `partial` | Notation 문서에는 rotation symbol과 방향 contract만 남긴다. Source rotation prose의 반대 방향은 Eq. (16), Eq. (22) 대조로 source error로 분류했다. |
| `01_spin_wave_theory_intro.md` | Exact HP mapping, Dyson-Maleev alternative, truncation과 odd-term condition | `docs/lswt/01-derivation/holstein-primakoff-expansion.md` | `partial` | Current LSWT scope에서 Dyson-Maleev와 higher-order term의 포함 범위는 open이다. |
| `01_spin_wave_theory_intro.md` | Even/odd expansion과 real-space constant, linear, quadratic, quartic terms | `docs/lswt/01-derivation/real-space-boson-hamiltonian.md` | `partial` | Main LSWT derivation은 quadratic truncation을 소유하고 quartic detail은 범위 결정 전 보류한다. |
| `01_spin_wave_theory_intro.md` | Fourier convention, MBZ, Nambu spinor와 BdG block | `docs/lswt/01-derivation/momentum-space-bdg-hamiltonian.md` | `partial` | Fourier sign, gauge, same-sublattice factor와 B block은 open review item이다. |
| `01_spin_wave_theory_intro.md` | Bogoliubov transform, paraunitary condition, spectrum과 canonical diagonal form | `docs/lswt/01-derivation/paraunitary-diagonalization.md` | `skeleton` | Energy correction과 observables를 이 문서에서 반복하지 않는다. |
| `01_spin_wave_theory_intro.md` | Colpa construction, positivity와 Goldstone-mode boundary | `docs/lswt/01-derivation/paraunitary-diagonalization.md` | `skeleton` | Algorithm과 적용 조건은 main document가 소유한다. |
| `01_spin_wave_theory_intro.md` | Paraunitarity and diagonalization proofs | `docs/lswt/04-appendices/paraunitarity-proofs.md` | `skeleton` | Main document의 결과를 다시 정의하지 않고 증명만 보충한다. |
| `01_spin_wave_theory_intro.md`, `02_physical_quantities.md` | Magnon bands, ground-state energy와 zero-point correction | `docs/lswt/02-observables/magnon-observables.md` | `skeleton` | Diagonalization 문서는 spectrum 생성까지만 다루고 energy observable은 여기서 정의한다. |
| `02_physical_quantities.md` | Post-diagonalization quantity index | `docs/lswt/02-observables/magnon-observables.md` | `skeleton` | 수식 복제 표가 아니라 각 observable owner로 가는 index로 다시 작성한다. |
| `03_thermodynamics.md` | Partition function, internal energy, free energy, entropy와 specific heat | `docs/lswt/02-observables/thermodynamics.md` | `skeleton` | 정의와 최종 LSWT 식은 main observable 문서가 소유한다. |
| `03_thermodynamics.md` | Long thermodynamic derivations | `docs/lswt/04-appendices/thermodynamic-derivations.md` | `skeleton` | Main document에 필요한 가정과 최종 결과를 남기고 중간 전개를 appendix로 보낸다. |
| `03_thermodynamics.md` | Boson occupation, sublattice moment와 correlation matrix | `docs/lswt/02-observables/magnon-observables.md` | `skeleton` | 온도 의존 분포는 thermodynamics를 참조하되 spin reduction 정의는 여기서 소유한다. |
| `04_correlations.md` | Real-time/equal-time correlator, symmetry와 local-to-lab response | `docs/lswt/02-observables/spin-correlations.md` | `skeleton` | Structure factor와 spectral transform은 다음 owner로 분리한다. |
| `04_correlations.md` | Static and dynamic structure factors | `docs/lswt/02-observables/structure-factor-and-spectral-function.md` | `skeleton` | Correlator 정의를 반복하지 않고 normalization contract를 참조한다. |
| `04_correlations.md` | Retarded Green function and spectral function | `docs/lswt/02-observables/structure-factor-and-spectral-function.md` | `skeleton` | Response basis와 dagger convention은 open review item이다. |
| `05_topology.md` | Lattice skyrmion number | `docs/lswt/02-observables/topological-magnon-quantities.md` | `source-reviewed` | 2026-09-30 원문 식을 부호 있는 Berg–Lüscher solid angle로 작성했다. Scope 포함 여부는 사용자 검토에서 확인한다. |
| `05_topology.md` | Berry curvature and Chern number | `docs/lswt/02-observables/topological-magnon-quantities.md` | `source-reviewed` | 2026-09-30 2N column 합(Σ3 부호)의 curvature와 physical band Chern을 명시했다(A6). 사용자 acceptance 대기. |
| `05_topology.md` | Magnon thermal Hall response | `docs/lswt/02-observables/topological-magnon-quantities.md` | `source-reviewed` | 2026-09-30 층당 κ, ħ, N_k A_uc 정규화와 pair form을 작성했다. 원본 −π²/3 상수항(A7)은 open. |
| `06_worked_example.md` | Single-mode quadratic-boson solution and expectation values | `docs/lswt/03-examples/worked-example.md` | `skeleton` | 일반 Bogoliubov 정의는 derivation 문서를 참조하고 예제 고유 계산만 소유한다. |

`docs/lswt/04-appendices/luttinger-tisza-method.md`는 legacy converted Markdown에 대응 본문이
없고 restructured TeX에 빈 TODO만 있다. 따라서 소유 파일은 유지하되 primary
source와 현재 LSWT scope가 확인될 때까지 `source-only`에 준하는 미결 상태로
둔다.

이 mapping이 확정돼도 legacy 내용이 이식됐다는 뜻은 아니다. 각 row는 source
구간 대조, canonical draft 반영, 사용자 검토를 거쳐야 `covered`로 바뀐다.
Legacy files는 active authoring surface의 중복을 제거하기 위해
`legacy/research-notes/lswt/converted-markdown/`으로 이동했지만, 이 이동 자체는
어느 row도 `covered`로 바꾸지 않는다. 모든 row가 `covered`가 될 때까지
source-only evidence로 보존한다.

## Review Ledger

상태 표기:

- `draft-routed`: 현재 draft가 issue를 명시적으로 반영하거나 보류했다.
- `open`: 물리 또는 notation 검증이 필요하다.
- `source-cleanup`: 원자료 문장·구조 정리 항목이며 이론 검증과 분리한다.

같은 ID가 반복된 A3와 B10 때문에 annotation은 33개이고 unique ID는 31개다.

| ID | Target | Status | Boundary |
|---|---|---|---|
| B8 | `docs/lswt/00-foundations/lswt-overview.md` | `source-cleanup` | Abstract 문장 범위 |
| C22 | `docs/lswt/00-foundations/bilinear-spin-hamiltonian.md` | `draft-routed` | `Eq.` 표기 |
| A4 | `docs/lswt/00-foundations/bilinear-spin-hamiltonian.md` | `draft-routed` | Link sum과 ordered-pair sum |
| C23 | `docs/lswt/00-foundations/classical-order-and-local-frame.md` | `source-cleanup` | Commented matrix 제거 |
| B17 | `docs/lswt/01-derivation/holstein-primakoff-expansion.md` | `source-cleanup` | Quantum-fluctuation 표현 |
| A3 | Local-frame and real-space documents | `draft-routed` | \(\hat{\mathbf e}_I^0\) superscript |
| A8 | `docs/lswt/01-derivation/real-space-boson-hamiltonian.md` | `draft-routed` | Rotated local field |
| A9 | `docs/lswt/01-derivation/momentum-space-bdg-hamiltonian.md` | `draft-routed` | Bond displacement 정의 |
| A10 | `docs/lswt/01-derivation/momentum-space-bdg-hamiltonian.md` | `open` | 2026-09-10 완전한 Fourier 합의 조건을 검증했다. DM의 복소 hopping과 Hermitian conjugate 관계를 구분해 본문에 반영하고 사용자 검토를 받아야 한다. |
| A5 | `docs/lswt/01-derivation/momentum-space-bdg-hamiltonian.md` | `open` | 2026-09-10 normal ordering과 해석 모델에서 -Tr(H)/4를 확인했다. -1/2 변경 annotation, 누락된 k 합과 pointwise trace 등식의 조건은 본문 반영·사용자 검토 대기다. |
| A11 | Momentum-space and magnon-observable documents | `open` | 2026-09-10 radial derivative 부호 및 trace 상수 관계를 대조했다. \(S(S+1)\) 표현은 E_cl의 S 의존 정의를 구분한 후 사용자 검토가 필요하다. |
| A1 | `docs/lswt/01-derivation/momentum-space-bdg-hamiltonian.md` | `open` | \(B_{\mathbf k}\) off-diagonal typo와 block 식. 2026-09-10 코드의 기존 수정 검증은 별도 종결했으며 이론 acceptance는 미완료다. |
| B10 | Momentum-space source | `source-cleanup` | SJ/SP color annotation 제거 |
| A2 | `docs/lswt/01-derivation/momentum-space-bdg-hamiltonian.md` | `open` | 2026-09-10 대표 bond당 exchange endpoint 2회, field 1회 기여를 검증했다. 원본 link 집합 정의와 식 (47)의 보완·사용자 검토는 남아 있다. |
| A12 | Diagonalization and magnon observables | `open` | 공개 솔버의 T=0 energy assembly는 수정했다. 원본 식 (53)–(54)의 합 범위, E_cl·Delta E_0·E_0·e_0 구분과 이론 acceptance는 별도다. |
| B18 | `docs/lswt/01-derivation/paraunitary-diagonalization.md` | `open` | Positive-semidefinite Goldstone-mode caveat |
| B11 | `docs/lswt/02-observables/magnon-observables.md` | `source-cleanup` | First-person convention |
| C14 | Magnon and response observables | `open` | \(S_k\), \(\bar S_k\), \(U^\beta\) notation |
| B20 | `docs/lswt/02-observables/spin-correlations.md` | `source-cleanup` | Correlation introduction 문장 |
| C17 | Correlation and response documents | `source-cleanup` | Roadmap과 상세 정의 중복 |
| A13 | `docs/lswt/02-observables/spin-correlations.md` | `open` | Discrete momentum delta |
| A14 | `docs/lswt/02-observables/structure-factor-and-spectral-function.md` | `open` | Block별 time dependence |
| A15 | `docs/lswt/02-observables/structure-factor-and-spectral-function.md` | `open` | Retarded function의 basis 범위 |
| C15 | `docs/lswt/03-examples/worked-example.md` | `source-cleanup` | TOC subsection 처리 |
| B21 | `docs/lswt/03-examples/worked-example.md` | `source-cleanup` | Number-expectation heading |
| C13 | `docs/lswt/00-foundations/notation-and-conventions.md` | `open` | Source \(R_k^\alpha\)를 canonical spin vertex \(\mathsf V_{\mathbf k}^{\alpha}\)로 분리했으며 정확한 definition, dagger convention과 energy symbol coverage는 open |
| B19 | `docs/lswt/02-observables/thermodynamics.md` | `source-cleanup` | Entropy 설명 |
| B9 | `docs/lswt/02-observables/topological-magnon-quantities.md` | `source-cleanup` | Primary PDF에는 orphan text가 남고 restructured draft에서만 제거됨. 2026-09-30 draft는 orphan text를 옮기지 않았다. |
| A6 | `docs/lswt/02-observables/topological-magnon-quantities.md` | `draft-routed` | Physical bands와 \(2N\) BdG sum. 2026-09-30 draft는 curvature의 중간 합을 2N column 전체(Σ3 부호)로, Chern·κ를 physical band로 명시했다. 코드(D29)와 같고 사용자 검토 대기다. |
| A7 | `docs/lswt/02-observables/topological-magnon-quantities.md` | `draft-routed` | 2026-09-11 참고 논문 식 (8)의 physical-band 합·hbar·온도·area/volume 정의를 대조했다. 사용자가 층당 κ 기본 및 층간격을 통한 3D 환산을 승인했고 코드 32개 회귀를 통과했다. 원본 c2 상수항과 이론 acceptance는 별도 검토다. 2026-09-30 draft는 상수항 없는 c2를 쓰고, 원본 형태와의 차이가 (πk_B²T/6ħ)ΣC_n임을 적었다. 코드도 상수항 없는 c2를 쓴다. 2026-09-30 사용자 요청으로 문헌을 대조해, H가 MBZ 전체에서 양정치이면 particle band Chern 합이 0(Shindou et al. 2013 식 (29))이라 두 형태가 같음을 본문에 반영했다. 양반정치(Goldstone) 경우만 open이다. |
| A16 | `docs/lswt/02-observables/topological-magnon-quantities.md` | `draft-routed` | Primary PDF는 \(\varepsilon_{n,k}\), restructured draft는 \(E_{\mathbf k,n}\)을 사용. 2026-09-30 draft는 notation 문서의 \(\varepsilon_{n\mathbf k}\)를 따른다. |

## Source TODOs

| Source area | Current state |
|---|---|
| Introduction | Restructured TeX에 content TODO가 남아 있음 |
| Validity and limitations | Restructured TeX에 content TODO가 남아 있음 |
| Luttinger-Tisza method | Primary PDF와 reviewed TeX에는 section이 없고, restructured TeX에만 빈 TODO section이 있음 |

## Source Tooling Issues

- Primary PDF metadata의 title은 LSWT 문서 제목과 일치하지 않는다.
- Primary PDF와 repo의 `note_lswt_restructured.pdf`는 서로 다른 PDF이며
  대체 관계가 아니다.
- Restructured TeX는 `\much`가 이미 `\mu_{\rm ch}`를 포함한 상태에서
  sublattice 첨자를 다시 붙이는 부분 때문에 double-subscript 오류가 난다.
- Review 문구의 `e^{-\riEt}`는 `\riEt`라는 undefined control sequence로
  해석된다.
- Rotation introduction의 laboratory-to-local 설명은 source Eq. (16), Eq. (22)와
  $\widetilde{\mathsf J}_{ij}=\mathsf R_i^{\mathsf T}\mathsf J_{ij}\mathsf R_j$가
  요구하는 local-to-laboratory 방향과 반대다. Canonical notation은 수식들과
  일치하는 후자를 따른다.
- 이전 canonical Markdown의 string-valued `section: theory/lswt/...`
  frontmatter는 Quarto 1.8.27의 숫자형 예약 field와 충돌했다. `docs/` migration은
  theory document에 `doc-path`를 사용하도록 바꿨으며 renderer 재검증이 필요하다.
- \(\mu_{\rm ch}\)의 물리적 범위가 확정되지 않았으므로 이 audit에서는 TeX를
  수정하지 않는다.

## Open Physical Decisions

다음 항목은 현재 `Unknown`이며 사용자 확인 또는 별도 이론-코드 검증 전에는
정본 식으로 확정하지 않는다.

1. \(\mu_{\rm ch}\)가 global scalar인지 sublattice-dependent quantity인지
2. Local complex basis에서 \(\widetilde J^{\pm\pm}\)를 정의하는 정확한 변환
3. Bilinear model의 \(\mathsf g_I\), \(\mathbf B_I\), \(\mathbf h_I\)에
   real-valued condition을 명시할지
4. Crystallographic BZ와 magnetic BZ의 관계
5. Ground-state energy, zero-point correction, constant/trace convention
6. Correlation과 structure factor에서 \(N\), \(L\), \(m_s\) normalization
7. Positive-semidefinite Goldstone mode를 Colpa 문서 범위에 포함할지
8. Physical \(N\) bands와 \(2N\) BdG space의 topology sum convention (2026-09-30 topology draft가 D29 convention을 제안; 사용자 결정 대기)
9. \(H_4\), Dyson-Maleev, Luttinger-Tisza를 현재 정본화 범위에 포함할지

## Structural Gate Status

- [x] Canonical theory document를 `docs/00-...`부터 `docs/04-...`로 이동
- [x] 상위·하위 navigation을 `docs/README.md` 하나로 병합
- [x] Operational audit를 Working-Pad로 분리
- [x] Legacy converted Markdown과 navigation snapshot을 source-only로 보존
- [x] 10개 skeleton의 `draft` frontmatter 통일
- [x] Markdown 단일 정본과 PDF·TeX·HTML 역할에 대한 사용자 결정
- [x] Notation contract에 필요한 사용자 결정
- [x] Quarto 예약 field `section` 대신 theory document용 `doc-path` 채택
- [x] `docs/` authoring surface를 반영하는 source-authority precedent 기록 정비 — 2026-09-05

완료 표시는 source authority와 문서 구조 및 notation contract를 확정했다는
뜻이다. 이론 식이나 코드가 검증되었거나 모든 draft에 notation이 적용됐다는
뜻은 아니다. 다음 본문 작업은 필요한 notation의 국소 대조 후
`docs/lswt/00-foundations/bilinear-spin-hamiltonian.md`를 검토하는 것이다.

## Relocated Work Notes — 2026-09-06

아래는 이번 일괄 문체 교정에서 본문 밖으로 옮긴 이전 작업 기록이다. 새로운 정책이나 현재 실행 지시가 아니며, 충돌하는 과거 문구는 위의 최신 검토 범위와 기존 Open Physical Decisions에 따른다.

### bilinear-spin-hamiltonian

> ## 검증 메모
>
> - Source review note A4는 이 파일에서 처리한다. 본문은 ambiguous
>   $\sum_{ij}$ 대신 one-link counting $\sum_{\ell\in\mathcal{L}}$를 쓴다.
> - Source review note C22는 이 파일에서 처리한다. Canonical equation은 semantic
>   ID로 참조하고, 원본의 번호가 필요할 때만 `Source Eq.` 표기를 사용한다.
> - Source Eq. (2)는 inter-site exchange와 on-site anisotropy를 별도 sum으로
>   분리한다. Canonical $\mathcal L$도 inter-site link만 포함한다. 현재 code가
>   on-site term을 어떤 data structure로 표현하는지는 별도의 code-verification
>   항목이며 theory definition과 구분한다.
> - Source는 material parameter의 Hermiticity 조건을 명시하지 않는다. 현재
>   canonical contract는 사용자 결정에 따라 $\mathsf J_\ell$를 real matrix로,
>   $\mathsf D_I$를 real symmetric matrix로 제한한다. $\mathsf g_I$와 field의
>   formal reality condition은 별도 검토한다.

> ## Common 후보 메모
>
> - one-link counting과 ordered-pair counting의 대응은 LSWT뿐 아니라
>   Monte Carlo, tensor network, exact diagonalization에서도 공유해야 할
>   Hamiltonian convention이다.
> - bilinear interaction 밖의 scalar spin chirality, ring exchange는 향후
>   common spin-Hamiltonian 문서에서 별도 Hamiltonian class로 정리할 후보다.


### momentum-space-bdg-hamiltonian

2026-09-10 후속 코드 검증: 현재 `LSWTHamiltonian`의 B/B† 배치와 두 운동량
미분은 독립적인 spin-matrix 기준 회귀 테스트 8건으로 확인했다. Legacy 원본과
현재 구현의 차이, c735573 수정 이력 및 stale issue 정정은
[종결 이슈](../closed/260802-hamiltonian-b-block-substitution-bug.md)에 기록했다.
이 구현 이슈의 종결은 A1, Fourier/gauge, A2/A5/A10 또는 사용자의 Human
Physics and Mathematics Review를 완료한 것이 아니다. 아래의 원문 수식
검토 항목은 계속 열린 상태로 둔다.

2026-09-10 영점에너지 후속 검증: normal ordering에서 도출한
\(\Delta e_0=(N_kN_s)^{-1}\sum_k[\sum_n\omega_{kn}/2-\operatorname{Tr}H_k/4]\)를
독립 스핀, 같은 sublattice 강자성체, 실수·복소 pairing dimer 및 bosonic
Fock-space 대각화로 확인했다. 저수준 trace 식은 유지하고, 잘못된 magnon
평균을 더하던 공개 `LSWTSolver.solve()` 반환값을 수정했다. 비상호적 DM
분산에서는 \(\operatorname{Tr}H_k=2\operatorname{Tr}A_k\)가 각 k에서
성립하지 않아도 k↔-k를 보존하는 합에서 성립한다. A2/A5/A10/A11/A12의
원본 수식 대조, 보충 방향과 11개 회귀 테스트는
[영점에너지 종결 이슈](../closed/260910-zero-point-energy-normalization.md)에
기록했다. 구현 오류만 종결했으며, 이론 본문 반영과 사용자 검토는 남아 있다.
아래 인용은 과거 초안의 검토 메모로 보존한다.

> ## 검증 메모
>
> - A9는 이 초안에서 처리했다. $\boldsymbol{\Delta}_{\ell}$는
>   $\mathbf{r}_{J}-\mathbf{r}_{I}$인 real-space bond vector다.
> - A10은 아직 열어 둔다. Same-sublattice normal-order term이 BZ 합에서
>   사라진다는 주장은 $t^{-+}\neq(t^{+-})^*$인 경우 조건이 필요하다.
> - A5는 아직 열어 둔다. trace/constant-term convention은 구현된 Hamiltonian
>   construction과 대조해야 한다.
> - A1은 canonical equation 기준으로 아직 열어 둔다. Source의
>   two-sublattice $B_\mathbf{k}$ 식에는 confirmed typo가 있다.
> - A2는 아직 열어 둔다. identical-sublattice $\mathsf{A}_{\mathbf{k}}$
>   식의 factor 2는 검증이 필요하다.

> 원본의 explicit two-sublattice formula는 B block off-diagonal typo와 same-sublattice chemical-potential factor를 확인하기 전 이식하지 않는다.


### paraunitary-diagonalization

> ## 작업 메모
>
> - Colpa algorithm
> - Bogoliubov transformation
> - Positive-definite condition and Goldstone-mode caveat
> - Validity and limitations

> ## 검증 메모
>
> - Colpa diagonalization 코드와 수식 대응을 확인한다.


### real-space-boson-hamiltonian

> ## 검증 메모
>
> - A8은 이 초안에서 처리했다. onsite field contribution은
>   $\widetilde{h}_I^0$를 사용한다.
> - $H_4$와 odd terms는 원본 PDF에 남아 있으므로 본문 확장 시 누락 없이
>   이식한다.


### magnon-observables

> ## 작업 메모
>
> - Diagonalization 결과와 thermodynamics/correlations/topology observables 사이의 gateway 역할을 맡긴다.
> - Magnon spectrum
> - Zero-point energy
> - Bose occupation
> - Correlation matrix after diagonalization


### spin-correlations

> ## 작업 메모
>
> - Real-time spin-spin correlation function
> - Sublattice spin-spin correlation function

> ## Common 후보 메모
>
> - sublattice phase factor convention
> - Fourier transform convention


### structure-factor-and-spectral-function

> ## 작업 메모
>
> - Static structure factor
> - Dynamic structure factor
> - Spectral function

> ## Common 후보 메모
>
> - scattering momentum convention
> - spectral broadening convention


### thermodynamics

2026-09-11 정규화 후속: Primary PDF 15–16쪽의 S/C 전체 합과 sublattice별
점유수 평균을 대조했다. 개별·통합 U/S/C를 스핀당 값으로 맞추고, 통합 점유수의
중복 N_s 나눗셈 및 Ns 추론 결함을 수정했다. N_s=2/4의 cell 복제,
S=-partial F/partial T, C=partial U/partial T와 온도 스캔을 21개 테스트로
확인했다. [정규화 종결 이슈](../closed/260911-thermodynamic-observable-normalization.md)에
근거와 영향 범위를 기록했다. 이론 본문 반영·사용자 검토와 Thermal Hall,
Goldstone 및 invalid-mode 처방은 별도다.

2026-09-11 구현 검증: Primary PDF 13–14쪽의 분배함수와 U/F 최종식을
독립 oscillator 및 복소 pairing boson 모델에 대조했다. 온도 전달, U의 band
합산, F의 masked-copy 결함을 수정했고, U=F-T partial F/partial T도 확인했다.
[유한온도 구현 종결 이슈](../closed/260911-finite-temperature-energy-and-occupation.md)에
17개 회귀 테스트와 근거를 기록했다. 원본 식 (61)의 trace 계수·누락된 k 합,
E_0의 명칭 및 이론 본문 반영은 사용자 검토가 남아 있다. 별도 observables의
per-spin/per-cell 정규화와 Goldstone 처리는 이 종결에 포함하지 않는다.

> ## 작업 메모
>
> - Partition function
> - Internal energy
> - Free energy
> - Entropy
> - Specific heat
> - Number and spin moment from correlation matrix


### topological-magnon-quantities

2026-09-11 Thermal Hall 검증: Primary PDF 24쪽 및 Neumann et al. 식 (8),
Matsumoto–Murakami 식 (23)을 대조했다. 현재 curvature는 독립 두-band 해석식과
일치하지만, 같은 2D 모델의 길이 표현을 바꾸면 Hall 반환값이 길이²만큼 달라져
area 정규화 오류를 재현했다. `1e-12`→`1e-22`, `Ns` 곱셈만으로 해결할 수 없고,
사용자가 층당 κ [W/K] 기본·명시적인 층간격에 따른 3D 환산을 승인해 두
실행 경로와 온도 스캔에 적용했다. 독립 회귀 32개 통과, 전체 123 passed와
기존 CommensurateStructure 5 failed를 확인했다.
[열린 이슈](260802-topology-thermal-hall-real-space-volume-bug.md)에 유도,
`examples/thermal_hall_reference_check.py`의 수치와 legacy 대비를 기록했다.
이어서 실제 BZ의 격자 교체·경계 누락/중복 및 Chern의 불필요한 Ns 나눗셈을
수정했다. 신규 BZ 검사 40개와 이전 Hall 검사 32개 통과, 전체 163 passed와
기존 5 failed다. 다음 확인은 퇴화·soft mode의 처방이며, A6/A7/A16의 이론
acceptance와 외부 사용자 정의 weighted grid의 계약은 별도다.

> ## 작업 메모
>
> - Skyrmion number
> - Chern number
> - Thermal Hall conductance

> ## 검증 메모
>
> - Thermal Hall `real_space_volume` 및 단위 변환 이슈와 함께 대조한다.


### worked-example

> ## 작업 메모
>
> - Quadratic boson Hamiltonian example
> - Bogoliubov transformation
> - Positive-definite condition
> - 코드 예제와 연결할 때는 `examples/`의 실행 예제와 수치 검증 결과를 함께 확인한다.


### luttinger-tisza-method

> ## 작업 메모
>
> - 아직 본문 이식 전이다.


### paraunitarity-proofs

> ## 작업 메모
>
> - 아직 본문 이식 전이다.


### thermodynamic-derivations

> ## 작업 메모
>
> - Entropy derivation
> - Correlation matrix derivation for number and spin moment


### notation-and-conventions

> | Code field | Canonical notation | Contract |
> |---|---|---|
> | `lattice_vectors` | $\mathbf{a}_1,\mathbf{a}_2$ | Cartesian real-space vectors |
> | `Site.position` | $\boldsymbol{\delta}_{\mu}$ | magnetic cell 안의 Cartesian position |
> | `Coupling.displacement` | $\boldsymbol{\Delta}_{\ell}$ | source site에서 target site까지의 full Cartesian vector |
> | `num_sites`, `Ns` | $N_{\mathrm{sub}}$ | 현재 구현의 legacy naming |
>
> `Coupling.displacement`를 fractional coordinate로 해석하거나 solver가 이를
> lattice vector로 자동 변환한다고 가정하지 않는다.

> 코드의 `magnetic_field`는 현재 applied field B가 아니라 Zeeman-energy coefficient h에 대응한다.

> Cartesian real exchange의 complex circular basis reverse orientation에는 Hermitian conjugate가 대응한다는 기존 문장이 있었다. Circular-component definition이 미확정이므로 이 문장을 일반적으로 확정된 규칙으로 사용하지 않는다.

> Paraunitary 문서는 metric의 Nambu ordering, dimension, +/- block 및 metric preservation을 실제 사용 지점에서 설명해야 한다.

> Deferred notation은 source review A6, A7, A12--A16, C13, C14 및 관련 코드 검증과 함께 처리한다.


### holstein-primakoff-expansion

> Generic SpinSystem은 spin direction, coupling, field, lattice data를 저장하고, HP expansion과 H2 truncation은 LSWT solver에 속한다.


### classical-order-and-local-frame

> Source A3: local quantization axis에는 superscript 0을 유지하고, superscript 없는 e_I 표기는 사용하지 않는다. 원문의 row-vector field h_I^T R_I와 본문의 column-vector R_I^T h_I는 같은 contraction을 나타낸다.
