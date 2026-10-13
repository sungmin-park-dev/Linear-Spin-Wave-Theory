<!-- Written on 2026-10-12 as a worked example. Review status: draft; awaiting the user's physics and mathematics review. -->

# Triangular-Lattice Antiferromagnet

The nearest-neighbor Heisenberg antiferromagnet on the triangular lattice is the standard example of a noncollinear order in linear spin-wave theory (LSWT). Its classical ground state is the coplanar $120^\circ$ state, a single-$Q$ spiral, so the rotating frame of research note *LSWT classical order* reduces the problem to one sublattice. We use the primitive vectors $\mathbf a_1=(1,0)$ and $\mathbf a_2=(1/2,\sqrt3/2)$, one spin $S$ per site, and

<!-- eq 1: eq-ex-tri-hamiltonian -->

$$
\hat H=J\sum_{\langle ij\rangle}\hat{\mathbf S}_i\cdot\hat{\mathbf S}_j,
\qquad J>0, \tag{1}
$$

with $3N$ nearest-neighbor bonds for $N$ sites along $\boldsymbol\delta\in\{\mathbf a_1,\mathbf a_2,\mathbf a_2-\mathbf a_1\}$ and no field. Energies are in units of $J$ in the numerical values below.

## The 120° State in the Rotating Frame

The spins lie in a plane, which we take as the $xy$ plane, and rotate about $\mathbf c=\hat z$ with the wave vector $\mathbf Q=\mathbf K=(4\pi/3,0)$: $\mathbf n_i=\mathsf R_{\hat z}(\mathbf Q\cdot\mathbf r_i)\,\hat x$. Each bond direction has $\cos(\mathbf Q\cdot\boldsymbol\delta)=-1/2$, so neighboring spins are $120^\circ$ apart and the classical energy per site is

<!-- eq 2: eq-ex-tri-classical-energy -->

$$
\frac{E_{\mathrm{cl}}}{N}=3JS^2\cos(\mathbf Q\cdot\boldsymbol\delta)=-\tfrac32JS^2. \tag{2}
$$

The Heisenberg exchange commutes with rotations about $\hat z$, so Single-Q Spirals in a Rotating Frame in research note *LSWT classical order* applies: in the frame that rotates with the spins, every site has the same direction and every bond carries the real matrix $\mathsf J'_{\boldsymbol\delta}=J\,\mathsf R_{\hat z}(\mathbf Q\cdot\boldsymbol\delta)$.

## Boson Hamiltonian

We take the local axes $\mathbf e^0$ along the spin, $\mathbf e^1$ in the plane, and $\mathbf e^2=\hat z$. In these axes a bond with relative angle $\theta=\mathbf Q\cdot\boldsymbol\delta$ reads

$$
\hat{\mathbf S}_i\cdot\hat{\mathbf S}_j
=\cos\theta\big(\hat{\widetilde S}_i^0\hat{\widetilde S}_j^0+\hat{\widetilde S}_i^1\hat{\widetilde S}_j^1\big)
+\hat{\widetilde S}_i^2\hat{\widetilde S}_j^2
+\sin\theta\big(\hat{\widetilde S}_i^1\hat{\widetilde S}_j^0-\hat{\widetilde S}_i^0\hat{\widetilde S}_j^1\big).
$$

With $\hat{\widetilde S}^0=S-\hat n$, $\hat{\widetilde S}^1=\sqrt{S/2}\,(\hat a+\hat a^\dagger)$ and $\hat{\widetilde S}^2=-\mathrm i\sqrt{S/2}\,(\hat a-\hat a^\dagger)$, the terms with $\sin\theta$ contribute to quadratic order only through boson-linear terms, which cancel in the sum over the six neighbors because the state is stationary. The remaining terms give, per bond,

$$
JS\Big[-\cos\theta\,(\hat n_i+\hat n_j)
+\tfrac12(1+\cos\theta)\big(\hat a_i^\dagger\hat a_j+\hat a_j^\dagger\hat a_i\big)
+\tfrac12(\cos\theta-1)\big(\hat a_i\hat a_j+\hat a_j^\dagger\hat a_i^\dagger\big)\Big]
$$

above the classical energy. With $\cos\theta=-1/2$, six bonds per site, and the bond sums $\sum_{\langle ij\rangle}(\hat a_i^\dagger\hat a_j+\mathrm{h.c.})=\sum_{\mathbf k}6\gamma_{\mathbf k}\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}$ and $\sum_{\langle ij\rangle}(\hat a_i\hat a_j+\mathrm{h.c.})=\sum_{\mathbf k}3\gamma_{\mathbf k}(\hat a_{-\mathbf k}\hat a_{\mathbf k}+\mathrm{h.c.})$, the Hamiltonian takes the single-sublattice form of Worked Example in research note *LSWT paraunitary diagonalization* with

<!-- eq 3: eq-ex-tri-coefficients -->

$$
A_{\mathbf k}=3JS\Big(1+\frac{\gamma_{\mathbf k}}2\Big),
\qquad
B_{\mathbf k}=-\frac92JS\,\gamma_{\mathbf k},
\qquad
\gamma_{\mathbf k}=\frac13\sum_{\boldsymbol\delta}\cos(\mathbf k\cdot\boldsymbol\delta). \tag{3}
$$

The momentum $\mathbf k$ is the momentum in the rotating frame.

## Magnon Dispersion

The single-sublattice energy $\varepsilon_{\mathbf k}=\sqrt{A_{\mathbf k}^2-B_{\mathbf k}^2}$ is

<!-- eq 4: eq-ex-tri-dispersion -->

$$
\varepsilon_{\mathbf k}=3JS\sqrt{\big(1-\gamma_{\mathbf k}\big)\big(1+2\gamma_{\mathbf k}\big)}. \tag{4}
$$

The band vanishes where $\gamma_{\mathbf k}=1$, at $\Gamma$, and where $\gamma_{\mathbf k}=-1/2$, at $\pm\mathbf K$. The $120^\circ$ order breaks all three spin rotations, and these three zeros are the three Goldstone modes. Near $\Gamma$, $1-\gamma_{\mathbf k}=|\mathbf k|^2/4+O(|\mathbf k|^4)$; near $\mathbf K$, $\gamma_{\mathbf K+\mathbf q}=-1/2+|\mathbf q|^2/8+O(|\mathbf q|^3)$. The Goldstone modes are therefore linear, with the velocities

<!-- eq 5: eq-ex-tri-velocities -->

$$
c_\Gamma=\frac{3\sqrt3}{2}\,JS,
\qquad
c_{\mathbf K}=\frac{3\sqrt3}{2\sqrt2}\,JS. \tag{5}
$$

The cubic term of $\gamma_{\mathbf K+\mathbf q}$ makes the mode at $\mathbf K$ trigonally warped: the slope along a fixed direction differs from $c_{\mathbf K}$ at order $|\mathbf q|$, with opposite sign at $\mathbf K$ and $-\mathbf K$. At $M=(\pi,\pi/\sqrt3)$, $\gamma_M=-1/3$ and $\varepsilon_M=2JS$.

On the three-site magnetic cell of the $120^\circ$ state in the laboratory frame, the same spectrum appears as three bands at each momentum, $\varepsilon_{\mathbf k}$ and $\varepsilon_{\mathbf k\pm\mathbf Q}$, because $\pm\mathbf Q$ are reciprocal vectors of that cell. All three Goldstone modes then sit at $\Gamma$ of the magnetic zone. Which of the folded bands carries neutron weight at a given momentum follows from the laboratory spin correlations, which combine rotating-frame correlations at $\mathbf k$ and $\mathbf k\pm\mathbf Q$ (Single-Q Spirals in a Rotating Frame).

## Ground-State Energy

The zero-point correction $\frac12\sum_{\mathbf k}(\varepsilon_{\mathbf k}-A_{\mathbf k})$ uses $\langle\gamma_{\mathbf k}\rangle=0$, and with (2) the energy per site is

<!-- eq 6: eq-ex-tri-energy -->

$$
\frac{E_{\mathrm{GS}}}{N}=-\frac32JS^2-\frac32JS\,(1-I),
\qquad
I=\Big\langle\sqrt{(1-\gamma_{\mathbf k})(1+2\gamma_{\mathbf k})}\Big\rangle=0.781588, \tag{6}
$$

that is, $E_{\mathrm{GS}}/N=-\frac32JS\,(S+0.218412)$. At $S=1/2$ it gives $E_{\mathrm{GS}}/N=-0.538809\,J$. These are the harmonic results of the rotating-frame treatment in [@chernyshevSpin2009].

## Reduced Ordered Moment

The vacuum boson number $\frac12\langle A_{\mathbf k}/\varepsilon_{\mathbf k}-1\rangle$ of the single-sublattice problem gives

<!-- eq 7: eq-ex-tri-moment -->

$$
S-\langle\hat S\rangle=\frac12\Big\langle\frac{1+\gamma_{\mathbf k}/2}{\sqrt{(1-\gamma_{\mathbf k})(1+2\gamma_{\mathbf k})}}\Big\rangle-\frac12=0.261303, \tag{7}
$$

so the LSWT moment at $S=1/2$ is $\langle\hat S\rangle=0.238697$, less than half the classical value [@chernyshevSpin2009]. As for the square lattice, the integrand diverges as $1/|\mathbf k|$ at the Goldstone points, a uniform $n\times n$ mesh that avoids them converges as $1/n$, and the value in (7) is the extrapolation $2\,\langle\cdot\rangle_{2n}-\langle\cdot\rangle_n$.

## Numerical Check

For $J=1$ and $S=1/2$, the bands computed on the three-site magnetic cell along $\Gamma$–$\mathbf K$–$M$–$\Gamma$ agree with the three copies $\varepsilon_{\mathbf k}$, $\varepsilon_{\mathbf k\pm\mathbf Q}$ of (4) within $10^{-7}$, with $\varepsilon_M=1$. The slopes at $\Gamma$ of the magnetic zone are $1.2990$ for the copy $\varepsilon_{\mathbf k}$ and $0.9186$ for the average of the two copies $\varepsilon_{\mathbf k\pm\mathbf Q}$, as (5) gives; averaging over $\pm\mathbf q$ and over the two copies removes the warping term. The ground-state energy on a $96\times96$ mesh agrees with (6) to six digits. The ordered moment on that mesh is $0.2411$, and the extrapolation from the $48\times48$ and $96\times96$ meshes gives $0.2387$, as (7) gives.

## References

### Internal Documents

- Single-Q Spirals in a Rotating Frame in research note *LSWT classical order*: the rotating frame, the condition on the exchange matrix, and the combination of momenta in laboratory correlations.
- Worked Example in research note *LSWT paraunitary diagonalization*: the single-sublattice dispersion and the vacuum boson number.
- Research note *LSWT example: square-lattice antiferromagnet*: the collinear case and the mesh extrapolation.
- Research note *LSWT nonlinear spin waves*: the cubic vertices that the $\sin\theta$ terms produce in a noncollinear state, which this harmonic note leaves out.
- `code-space/examples/worked_examples/triangular_antiferromagnet.py` and `code-space/tests/test_worked_examples.py`: the numbers of Numerical Check.

### External Sources

- [@chernyshevSpin2009]: spin-wave theory of the triangular-lattice antiferromagnet in the rotating frame, its harmonic spectrum, ground-state energy and ordered moment.
