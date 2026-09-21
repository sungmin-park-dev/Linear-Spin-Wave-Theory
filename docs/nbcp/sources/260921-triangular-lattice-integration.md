# Supplied triangular-lattice source: integration record

Source: `260921-triangular-lattice-supplied.tex`, provided by the user on
2026-09-21. It is a reference snapshot, not a canonical content owner.

- Lines 148-180: the repeated triangular grid and nearest-neighbor vectors
  inform the TikZ figure in Section 2. Colors retain the bond-phase convention
  already used by the canonical exchange matrix.
- Lines 184-256: Appendix E now makes the inversion, twofold rotation and
  threefold covariance steps explicit, including parameter definitions.
- The source inversion expression assigns component sign changes to spins.
  The canonical axial-vector transformation is retained: spatial inversion
  exchanges the sites without flipping spin components. The final symmetric
  exchange constraint agrees with the supplied source.
- The supplied rotation matrix has the opposite angle sign to an active
  Cartesian rotation. Its inverse-matrix covariance expression gives the
  same tensor. Appendix E consistently uses the existing active convention
  R_z(beta) J_0 R_z(beta)^T; the general-angle result was checked symbolically.
- The next-nearest-neighbor vectors and exchange terms are not incorporated
  into the current nearest-neighbor model. Their bond-axis symmetry argument
  remains pending review. In particular, the source line 264 vanishing-index
  statement includes K_yy, while its subsequent matrix retains K_yy.
- The supplied bibliography keys are not imported as unverified bibliography
  entries. Existing verified source citations are retained.

Source SHA256: b95658151564670eea58b553abe0a3122ce632ff09a6cea3352168631c207431
