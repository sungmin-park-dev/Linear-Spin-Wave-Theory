---
frontmatter-version: 1
title: NBCP Research Notes
doc-path: docs/nbcp
status: in-review
last-edited-by: claude
created: 2026-09-16
updated: 2026-09-21
---

# NBCP 연구 노트

**NBCP 내용은 LaTeX에서 직접 편집한다.** [main.tex](main.tex)가 본문 10장, 부록 4개와 참고문헌을 연결한다. 각 내용은 해당 TeX 파일 한 곳에서만 관리하며 PDF는 읽기용 출력이다.

[전체 문서](../README.md) · [LSWT 일반 이론](../lswt/README.md) · [연구노트 PDF](output/research-note.pdf)

## 목차

1. [Introduction](chapters/01-introduction.tex): 연구 질문, 방법과 근거의 범위.
2. [Model Hamiltonian](chapters/02-model-hamiltonian.tex): effective spin, 대칭, 교환행렬, 단위와 매개변수.
3. [Phase Diagram](chapters/03-phase-diagram.tex): 고전 상도 구성, 영점에너지 보정.
4. [Characterization of Magnetic Phases](chapters/04-phase-characterization.tex): Y·UUD·V·P 정의와 에너지, stripe·4-sublattice 경쟁 상.
5. [Skyrmion Phase](chapters/05-skyrmion-phase.tex): texture·위상전하의 검토 범위와 미검증 항목.
6. [Y and V: Symmetry and Low-Energy Theory](chapters/06-yv-low-energy.tex): 정확한 U(1), accidental degeneracy, pinning과 gap.
7. [Supersolidity: Definition and Conditions](chapters/07-supersolidity.tex): 정의, 유효모델·RG 조건과 미시적 검증 기준.
8. [Supersolidity in Y](chapters/08-y-phase.tex): 안정성, angular matching, smooth wave와 density wall, 남은 thermal 검증.
9. [Supersolidity in V](chapters/09-v-phase.tex): PD/Gamma 대칭 차이, 기존 angular/gap 결과와 미계산 항목.
10. [Discussion and Conclusions](chapters/10-discussion.tex): 근거가 확보된 주장과 적용 한계.

- [부록 A — pseudo-Goldstone gap](appendices/a-pseudo-goldstone-gap.tex): 공액 응답과 Y/V 상세 유도.
- [부록 B — clock RG](appendices/b-clock-rg.tex): 정규화, shell 계산과 coupled flow.
- [부록 C — Y stiffness](appendices/c-y-stiffness.tex): zero-SOC 및 SOC 미시적 유도.
- [부록 D — 수치 검증과 재현](appendices/d-numerical-verification.tex): 수렴, 계산 근거와 실행 경로.
- [코드 물리 구현 검토](../../GOVERNMENT/Working-Pad/issue-notes/open/260918-nbcp-physics-code-review.md): 주장과 코드·근거의 대응, convention 문제와 독립 검토 항목.

## 파일 역할

| 위치 | 역할 | 직접 편집 |
|---|---|---|
| [main.tex](main.tex) | 장 순서·부록 전환·전체 조립 | 구성 변경 시 |
| [chapters/](chapters/) · [appendices/](appendices/) | 연구 내용의 유일한 편집 원본 | 해당 장에서 직접 |
| [figures/tikz/](figures/tikz/) | TikZ 그림의 편집 원본(`<이름>.tex`)과 공통 서식(`style.tex`) | 해당 `.tex`에서 직접 |
| [figures/](figures/) | 위 원본에서 생성된 벡터 PDF(`<이름>.pdf`) | 원본 수정 후 재생성 |
| [references.bib](references.bib) · [references.tex](references.tex) | 문헌 항목과 근거의 역할(`.bib`), 참고문헌 절(`.tex`) | 출처를 보완할 때 |
| [preamble.tex](preamble.tex) · [metadata.tex](metadata.tex) | 공통 서식·제목·날짜·검토 상태 | 서식·메타데이터 변경 시 |
| [output/](output/) | 생성 PDF와 입력 파일 해시 기록 | 원본 수정 후 재생성 |
| [Overleaf 원문 대조 기록](sources/overleaf-2026-09-16-review.md) | 선별 발췌·충돌·보류 기록 | 원문 대조 기록을 보완할 때 |
| [계산·검증 데이터](../../data-space/verification/) | 수치 배열·그림·검증 JSON | 계산 스크립트로 생성 |
| [이전 Markdown 보관본](../archive/nbcp/2026-09-18-markdown-source.zip) | 전환 직전 원본·변환기·서식 | 역사 자료, 편집하지 않음 |

`research-note.md`는 옛 링크를 위한 안내 파일이다. 본문을 다시 작성하거나 TeX에서 Markdown 본문을 자동 생성하지 않는다. 기존 semantic ID는 TeX의 `\label`로 유지하며 표시 번호는 `\ref` 또는 `\eqref`로 참조한다. 기존 ID의 현재 파일은 [source map](../../data-space/verification/260918-nbcp-tex-migration/source-map.json)에 기록되어 있다.

## PDF 갱신

XeLaTeX와 BibTeX가 설치된 환경에서 저장소 루트 기준으로 실행한다. Quarto는 더 이상 필요하지 않다.

```bash
python examples/nbcp_research_export.py
```

계산을 다시 실행하지 않고 기존 그림을 읽어 `output/research-note.pdf`와 `output/research-export.json`을 갱신한다. `figures/tikz/`의 TikZ 그림은 원본이나 `style.tex`가 바뀐 것만 XeLaTeX로 다시 컴파일해 `figures/<이름>.pdf`로 저장한다. 빌드 중간 파일과 그림 사본은 임시 폴더에서 처리한다. 문서를 구성하는 TeX 파일·출력 프로그램·그림의 해시와 컴파일 진단을 생성 기록에 남긴다.

편집기의 직접 컴파일을 사용할 때에는 `docs/nbcp/`를 작업 디렉토리로 하고 XeLaTeX로 `main.tex`를 연다. 참고문헌 번호를 갱신하려면 XeLaTeX → BibTeX → XeLaTeX 두 번 순서로 컴파일한다(Overleaf는 자동). 그림은 저장소의 `data-space/verification/`에서 상대 경로로 읽는다. 이 폴더만 Overleaf에 올리면 그림이 포함되지 않으므로 별도 업로드용 묶음이 필요하다. 현재 저장소가 유일한 편집 원본이며 온라인 사본은 만들지 않았다.

## 근거와 검토 상태

문헌은 원본의 해당 주장 가까이에 연결한다. 각 문헌의 역할은 [references.bib](references.bib)의 `note` 필드에 기록하며, [Clock 분석과 gap 분석의 근거 목록](references.tex)이 이를 인용한다. 일반 LSWT 원자료는 [LSWT sources](../lswt/sources/README.md)를 참조하며 중복 복사하지 않는다. Overleaf 대조 기록은 선별 발췌와 검토 기록이며 전체 온라인 프로젝트 백업은 아니다.

노트는 `in-review`이다. 남은 물리 검토는 [열린 연구 이슈](../../GOVERNMENT/Working-Pad/issue-notes/open/260810-pseudo-goldstone-gap.md)에서 관리한다. 병합 전 파일의 복구용 압축본과 보존 검사는 [통합 검증 자료](../../data-space/verification/260916-nbcp-integration/)에 있다. 구조와 출력 검증은 물리적 acceptance를 뜻하지 않는다.


## 캡션과 문헌 인용

- 모든 그림에는 아래쪽에 번호와 캡션을 둔다.
- TikZ 그림은 `figures/tikz/<이름>.tex`(`standalone`)에서 만들고, 본문에는 `\includegraphics{figures/<이름>.pdf}`를 원래 크기로 넣는다. 본문에 `tikzpicture`를 직접 쓰지 않는다.
- 모든 표에는 아래쪽에 번호와 캡션을 둔다. 여러 페이지에 걸친 표는 마지막 부분 아래에 캡션을 두고 열 머리글을 반복한다.
- 문헌은 `\cite{key}`로 인용하여 `[번호]`로 표시한다. 동일 문헌은 `references.bib`의 항목 하나로 관리하며, BibTeX(`unsrturl`)가 본문 첫 인용 순서로 번호를 매긴다.
- 원 논문의 절·수식·그림 위치는 번호 인용 옆에 남긴다. 계산 데이터·코드·라이선스 링크는 문헌 인용과 구분하여 직접 연결한다.
