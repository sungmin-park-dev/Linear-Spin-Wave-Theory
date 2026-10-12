<!-- Written on 2026-10-12 as a worked example. Review status: draft; awaiting the user's physics and mathematics review. -->

# Square-Lattice Antiferromagnet

The nearest-neighbor Heisenberg antiferromagnet on the square lattice is the standard example of a collinear antiferromagnet in linear spin-wave theory (LSWT). Unlike the ferromagnet, its harmonic Hamiltonian contains anomalous terms, so the ground state has zero-point fluctuations that lower the energy and reduce the ordered moment. We use unit lattice constant, one spin $S$ per site, and

<!-- eq 1: eq-ex-afm-hamiltonian -->

$$
\hat H=J\sum_{\langle ij\rangle}\hat{\mathbf S}_i\cdot\hat{\mathbf S}_j,
\qquad J>0, \tag{1}
$$

with $2N$ nearest-neighbor bonds for $N$ sites and no field. Energies are in units of $J$ in the numerical values below.

## Néel State and BdG Blocks

The Néel state, $\mathbf n_A=\hat z$ on one checkerboard sublattice and $\mathbf n_B=-\hat z$ on the other, is the classical ground state of (1), with $E_{\mathrm{cl}}/N=-2JS^2$. Its BdG blocks are derived in Example: Square-Lattice Néel Antiferromagnet in research note *LSWT boson Hamiltonian*: on the two-site magnetic cell,

<!-- eq 2: eq-ex-afm-blocks -->

$$
\mathsf A_{\mathbf k}=4JS\,\mathsf I_2,
\qquad
\mathsf B_{\mathbf k}=-4JS\gamma_{\mathbf k}\begin{pmatrix}0&1\\1&0\end{pmatrix},
\qquad
\gamma_{\mathbf k}=\tfrac12(\cos k_x+\cos k_y). \tag{2}
$$

The anomalous block couples a boson on $A$ to one on $B$. In the combinations $(\hat a_{\mathbf kA}\pm\hat a_{\mathbf kB})/\sqrt2$ the problem splits into two copies of the single-sublattice problem of Worked Example in research note *LSWT paraunitary diagonalization*, with $A_{\mathbf k}=4JS$ and $B_{\mathbf k}=\mp4JS\gamma_{\mathbf k}$.

## Magnon Dispersion

The single-sublattice result $\varepsilon_{\mathbf k}=\sqrt{A_{\mathbf k}^2-|B_{\mathbf k}|^2}$ gives one doubly degenerate band on the magnetic Brillouin zone (MBZ),

<!-- eq 3: eq-ex-afm-dispersion -->

$$
\varepsilon_{\mathbf k}=4JS\sqrt{1-\gamma_{\mathbf k}^2}. \tag{3}
$$

Because $\gamma_{\mathbf k+\mathbf Q}=-\gamma_{\mathbf k}$ for $\mathbf Q=(\pi,\pi)$, the energy (3) has the periodicity of the MBZ, and along a path of the crystallographic zone the two folded copies $\varepsilon_{\mathbf k}$ and $\varepsilon_{\mathbf k+\mathbf Q}$ coincide. The band vanishes at $\Gamma$ and at $M=\mathbf Q$, where $\gamma_{\mathbf k}^2=1$; these are the Goldstone modes of the broken spin-rotation symmetry, and both fold onto $\Gamma$ of the MBZ. Near $\Gamma$, $1-\gamma_{\mathbf k}^2=|\mathbf k|^2/2+O(|\mathbf k|^4)$, so the Goldstone mode is linear,

<!-- eq 4: eq-ex-afm-velocity -->

$$
\varepsilon_{\mathbf k}\simeq c\,|\mathbf k|,
\qquad
c=2\sqrt2\,JS. \tag{4}
$$

On the boundary of the MBZ, $|k_x|+|k_y|=\pi$, the sum $\cos k_x+\cos k_y$ vanishes, so $\gamma_{\mathbf k}=0$ and $\varepsilon_{\mathbf k}=4JS$ at every point of it, including $X=(\pi,0)$ and $(\pi/2,\pi/2)$. This degeneracy along the zone boundary is a property of the harmonic theory.

## Ground-State Energy

The zero-point correction per site is evaluated in the same example of research note *LSWT boson Hamiltonian*,

<!-- eq 5: eq-ex-afm-energy -->

$$
\frac{E_{\mathrm{GS}}}{N}=-2JS^2-2JS\,C=-2JS\,(S+C),
\qquad
C=1-\Big\langle\sqrt{1-\gamma_{\mathbf k}^2}\Big\rangle=0.157947, \tag{5}
$$

where $\langle\cdot\rangle$ is the average over the Brillouin zone. The integrand is smooth up to the cusps at the Goldstone points, so the average converges quickly with the mesh. Equation (5) is the LSWT energy of [@andersonApproximate1952] and the leading two terms of the $1/S$ series of [@zhengSpinwave1993]. At $S=1/2$ it gives $E_{\mathrm{GS}}/N=-0.657947\,J$, and at $S=1$ it gives $-2.315895\,J$.

## Reduced Ordered Moment

In the magnon vacuum, the boson number of the single-sublattice problem is $\langle\hat n\rangle=\frac12\langle A_{\mathbf k}/\varepsilon_{\mathbf k}-1\rangle$ (Boson Number and Anomalous Average in research note *LSWT paraunitary diagonalization*). The two combinations of Néel State and BdG Blocks contribute equally, so each sublattice carries

<!-- eq 6: eq-ex-afm-moment -->

$$
S-\langle\hat S\rangle=\langle\hat n\rangle
=\frac12\Big\langle\frac{1}{\sqrt{1-\gamma_{\mathbf k}^2}}\Big\rangle-\frac12=0.196602. \tag{6}
$$

The reduction does not depend on $S$, so it is a correction of relative order $1/S$ to the classical moment. At $S=1/2$ the LSWT moment is $\langle\hat S\rangle=0.303398$, about sixty percent of the classical value [@andersonApproximate1952]. The integrand of (6) diverges as $1/|\mathbf k|$ at the Goldstone points. The singularity is integrable in two dimensions, but on a uniform mesh of $n\times n$ momenta that avoids the singular points the average converges only as $1/n$. Two meshes $n$ and $2n$ then extrapolate as $2\,\langle\cdot\rangle_{2n}-\langle\cdot\rangle_n$, which is how the value in (6) is obtained. At $T>0$ the thermal part of the boson number diverges logarithmically at the Goldstone modes, as described under Zero Modes in research note *LSWT magnon thermodynamics*, so (6) is a zero-temperature result.

## Numerical Check

For $J=1$ the computed bands agree with (3) along $\Gamma$–$X$–$M$–$\Gamma$ within $10^{-7}$, the deviation at the Goldstone points being the square-root rounding of a zero mode. The slope at $\Gamma$ is $1.4142$ for $S=1/2$ and $2.8284$ for $S=1$, as (4) gives, and $\varepsilon_X=4JS$. The ground-state energy on a $96\times96$ mesh of the MBZ agrees with (5) to six digits. The ordered moment on that mesh is $0.3061$ at $S=1/2$, and the extrapolation from the $48\times48$ and $96\times96$ meshes gives $0.3034$, as (6) gives.

## References

### Internal Documents

- Example: Square-Lattice Néel Antiferromagnet in research note *LSWT boson Hamiltonian*: the BdG blocks (2) and the zero-point energy (5).
- Worked Example in research note *LSWT paraunitary diagonalization*: the single-sublattice dispersion and the vacuum boson number.
- Boson Number and Reduced Moment, and Zero Modes, in research note *LSWT magnon thermodynamics*: the reduced moment and its behavior at a Goldstone mode.
- `examples/worked_examples/square_antiferromagnet.py` and `code-space/tests/test_worked_examples.py`: the numbers of Numerical Check.

### External Sources

- [@andersonApproximate1952]: spin-wave theory of the antiferromagnetic ground state, its energy and the reduction of the sublattice moment.
- [@zhengSpinwave1993]: the $1/S$ series of the square-lattice ground-state energy, whose first two terms are (5).
