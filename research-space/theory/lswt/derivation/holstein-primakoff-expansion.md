---
frontmatter-version: 1
title: Holstein-Primakoff Expansion
section: theory/lswt/derivation
status: draft
last-edited-by: codex
created: 2026-06-04
updated: 2026-06-04
source: /Users/david/Downloads/Linear_Spin_Wave_Theory___Note.pdf
source-section: Spin to Boson Transformations
---

# Holstein-Primakoff Expansion

Local frame에서 spin operator는 Holstein-Primakoff boson으로 표현된다.
Linear spin-wave theory에서는 다음 leading terms를 사용한다.

$$
\widetilde{S}_{I}^{+}
=\sqrt{2S_I}\,\hat{a}_{I}
+O(S_I^{-1/2}),
\qquad
\widetilde{S}_{I}^{-}
=\sqrt{2S_I}\,\hat{a}_{I}^{\dagger}
+O(S_I^{-1/2}),
$$
$$
\widetilde{S}_{I}^{0}
=S_I-\hat{n}_{I},
\qquad
\hat{n}_{I}=\hat{a}_{I}^{\dagger}\hat{a}_{I}.
$$

같은 내용을 fluctuation의 수직 성분과 평행 성분으로 나누어 쓰면

$$
\delta\hat{\mathbf{S}}_I
= \delta\hat{\mathbf{S}}_I^{\perp}
 +\delta\hat{\mathbf{S}}_I^{\parallel},
\qquad
\delta\hat{\mathbf{S}}_I^{\parallel}
=-\hat{n}_{I}\hat{\mathbf{e}}_I^0.
$$

classical spin configuration이 spin magnitude constraint 아래에서
classical energy의 extremum이면 boson linear term은 사라진다. 이때

$$
\frac{\partial E_{\mathrm{cl}}}{\partial\mathbf{S}_{I}}
=\lambda_I\mathbf{S}_{I},
\qquad
\frac{\partial E_{\mathrm{cl}}}{\partial\mathbf{S}_{I}}
\cdot\delta\hat{\mathbf{S}}_I^\perp=0.
$$

여기서 solver boundary가 갈린다. Generic `SpinSystem`은 spin direction,
coupling, field, lattice data를 저장할 수 있지만, HP expansion과 $H_2$
truncation은 LSWT solver 내부 로직이다.
