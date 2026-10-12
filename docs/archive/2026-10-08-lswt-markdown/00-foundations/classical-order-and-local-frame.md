---
frontmatter-version: 1
title: Classical Order and Local Frame
doc-path: docs/lswt/00-foundations
status: draft
last-edited-by: codex
created: 2026-06-04
updated: 2026-09-06
source: docs/lswt/sources/00-primary-source/Linear_Spin_Wave_Theory___Note.pdf
source-section: Rotations for Spin Models
---

# Classical Order and Local Frame

Linear spin-wave theory describes small quantum fluctuations about a classical spin configuration. At each site $I$, the spin direction is the unit vector

$$
\mathbf n_I
=(\sin\theta_I\cos\phi_I,\,
  \sin\theta_I\sin\phi_I,\,
  \cos\theta_I).
$$

The local quantization axis is chosen along $\mathbf n_I$. The polar angle $\theta_I$ is measured from the positive laboratory $z$ axis, and the azimuthal angle $\phi_I$ from the positive laboratory $x$ axis.

## Rotation to the Local Frame

The rotation matrix converts local spin components into laboratory components:

$$
\hat{\mathbf S}_I=\mathsf R_I\hat{\widetilde{\mathbf S}}_I,
\qquad
\mathsf R_I=\mathsf R(\theta_I,\phi_I),
$$

with

$$
\mathsf R(\theta,\phi)=
\begin{pmatrix}
\cos\theta\cos\phi & -\sin\phi & \sin\theta\cos\phi\\
\cos\theta\sin\phi &  \cos\phi & \sin\theta\sin\phi\\
-\sin\theta & 0 & \cos\theta
\end{pmatrix}.
$$

Its third column is $\mathbf n_I$, so the local longitudinal component lies along the classical spin. For a bond $\ell=(I,J)$, the exchange matrix and Zeeman-energy vector in the local frames are

$$
\widetilde{\mathsf J}_\ell
=\mathsf R_I^{\mathsf T}\mathsf J_\ell\mathsf R_J,
\qquad
\widetilde{\mathbf h}_I=\mathsf R_I^{\mathsf T}\mathbf h_I.
$$

The exchange term involves rotations at both endpoints, whereas the field term involves only the rotation at its own site. The field is written as a column vector; the equivalent row-vector expression is $\mathbf h_I^{\mathsf T}\mathsf R_I$.

## Local Circular Basis

Let $\mathbf e_x,\mathbf e_y,\mathbf e_z$ denote Cartesian coordinate unit vectors. The local circular basis, expressed in laboratory coordinates, is

$$
\mathbf e_I^\pm
=\mathsf R_I\frac{\mathbf e_x\pm\mathrm i\mathbf e_y}{\sqrt2},
\qquad
\mathbf e_I^0=\mathsf R_I\mathbf e_z=\mathbf n_I.
$$

The superscript $0$ identifies the longitudinal direction, and $\pm$ label the transverse circular vectors. These are classical basis vectors, so they carry no operator hats. The phase convention is explicit in their definitions; contractions used to turn exchange matrices into circular components require a separate component convention.

## References

### Internal Documents

- [Notation and Conventions](notation-and-conventions.md): defines the rotation-matrix direction and local-basis notation.
- [Bilinear Spin Hamiltonian](bilinear-spin-hamiltonian.md): defines the laboratory-frame exchange matrices and Zeeman field rotated here.
- [Holstein–Primakoff Expansion](../01-derivation/holstein-primakoff-expansion.md): uses the local quantization axis to introduce bosonic fluctuations.

### External Sources

- None.
