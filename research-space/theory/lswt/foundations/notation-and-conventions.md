---
frontmatter-version: 1
title: Notation and Conventions
section: theory/lswt/foundations
status: draft
last-edited-by: codex
created: 2026-06-04
updated: 2026-06-04
source: /Users/david/Downloads/Linear_Spin_Wave_Theory___Note.pdf
source-section: Summary of Notation and Symbols
---

# Notation and Conventions

이 파일은 LSWT 본문 전체에서 공유하는 notation과 convention을 고정하는
위치다. 원본 PDF의 `Summary of Notation and Symbols`가 기준이지만, link
counting, displacement, magnetic sublattice처럼 유도 전체에 영향을 주는
규칙도 여기에 모은다.

## Site Index Convention

물리적 spin site는 composite index로 쓴다.

$$
I=(n,\mu),\qquad
\mathbf{r}_{I}=\mathbf{R}_{n}+\boldsymbol{\delta}_{\mu}.
$$

여기서 $n$은 magnetic unit cell index이고, $\mu$는 그 magnetic unit cell
안의 magnetic sublattice index다. 이 $\mu$는 결정학적 basis site와 항상
같은 개념이 아니다. 결정학적 unit cell의 basis site가 magnetic supercell
안에서 여러 magnetic sublattice로 나뉠 수도 있고, 반대로 같은 spin angle을
공유하는 site들을 같은 magnetic sublattice로 묶을 수도 있다.

## Link Convention

$\mathcal{L}$은 physical coupling link의 집합이다. 이 문서의 기본 convention은
**한 physical bond를 한 번 센다**는 것이다. 즉 $\ell=(I,J)$를 포함했다면,
reverse link $(J,I)$를 별도로 더하지 않는다. Hamiltonian을 ordered-pair
sum $\sum_{I,J}$로 쓰는 경우에는 같은 물리를 나타내기 위해 exchange term
앞에 $1/2$가 필요하다.

Hermiticity는 reverse link를 명시적으로 두는 대신 다음 관계로 표현한다.

$$
J_{JI}^{\beta\alpha} = J_{IJ}^{\alpha\beta}.
$$

이 convention은 source review note A4가 지적한 핵심 문제다. 이 문서에서는
link sum과 independent index sum을 섞어 쓰지 않는다.

## Displacement Convention

Momentum-space phase에 들어가는 bond displacement는 real-space vector로 둔다.

$$
\boldsymbol{\Delta}_{\ell}
= \mathbf{r}_{J}-\mathbf{r}_{I}
= (\mathbf{R}_{m}-\mathbf{R}_{n})
 +(\boldsymbol{\delta}_{\nu}-\boldsymbol{\delta}_{\mu})
\quad\text{for}\quad
\ell=((n,\mu),(m,\nu)).
$$

코드의 `Coupling.displacement`가 fractional cell displacement를 저장하는
경우, LSWT solver가 phase factor를 만들기 전에 lattice vectors로 real-space
vector로 변환해야 한다. 이론 문서에서 $\boldsymbol{\Delta}_{\ell}$는 항상
real-space vector다.

## 작업 메모

- 원본 PDF의 symbol table을 이 파일로 이식한다.
- $R_k^\alpha$, $\epsilon_{n,\mathbf{k}}$처럼 review note에서 누락으로 지적된
  symbol을 같이 보완한다.
- 추후 common notation으로 분리할 항목이 보이면 기록한다.
