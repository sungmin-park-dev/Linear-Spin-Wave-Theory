---
frontmatter-version: 1
title: Real-Space Boson Hamiltonian
section: theory/lswt/derivation
status: draft
last-edited-by: codex
created: 2026-06-04
updated: 2026-06-04
source: /Users/david/Downloads/Linear_Spin_Wave_Theory___Note.pdf
source-section: Bosonic representations for spin Hamiltonian
---

# Real-Space Boson Hamiltonian

다음 quantity를 정의한다.

$$
t_{\ell}^{\alpha\beta}
=\sqrt{S_I S_J}\,\widetilde{J}_{\ell}^{\alpha\beta},
\qquad \ell=(I,J).
$$

quadratic bosonic term만 남기면

$$
\begin{aligned}
H_2
=&
\sum_{\ell=(I,J)\in\mathcal{L}}
\Big[
t_{\ell}^{--}\hat{a}_{I}\hat{a}_{J}
+t_{\ell}^{-+}\hat{a}_{I}\hat{a}_{J}^{\dagger}
+t_{\ell}^{+-}\hat{a}_{I}^{\dagger}\hat{a}_{J}
+t_{\ell}^{++}\hat{a}_{I}^{\dagger}\hat{a}_{J}^{\dagger}
\\
&\hspace{7em}
-\widetilde{J}_{\ell}^{00}
 \left(S_I\hat{n}_{J}+S_J\hat{n}_{I}\right)
\Big]
+\sum_{I}\widetilde{h}_{I}^{0}\hat{n}_{I}.
\end{aligned}
$$

따라서 site $I$의 effective onsite coefficient는 $I$에 닿는 모든 link를
합쳐서 얻는다. $\partial I$를 site $I$에 incident한 link 집합,
$I'_{\ell}$를 link $\ell$에서 $I$의 반대편 endpoint라고 쓰면

$$
\mu_I
=\widetilde{h}_{I}^{0}
-\sum_{\ell\in\partial I}
 S_{I'_{\ell}}\widetilde{J}_{\ell}^{00}.
$$

Source review note A8은 이 항이 unrotated $h_I$가 아니라
$\widetilde{h}_I^0$를 써야 한다고 지적한다. 이 초안은 rotated local
field를 채택한다.

## 검증 메모

- A8은 이 초안에서 처리했다. onsite field contribution은
  $\widetilde{h}_I^0$를 사용한다.
- $H_4$와 odd terms는 원본 PDF에 남아 있으므로 본문 확장 시 누락 없이
  이식한다.
