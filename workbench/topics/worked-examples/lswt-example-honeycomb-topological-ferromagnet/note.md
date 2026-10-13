<!-- Written on 2026-10-12 as a worked example. Review status: draft; awaiting the user's physics and mathematics review. -->

# Honeycomb Ferromagnet with Topological Magnon Bands

A ferromagnet on the honeycomb lattice has two magnon bands that touch at Dirac points, as the electron bands of graphene do. A Dzyaloshinskii–Moriya (DM) interaction between second neighbors acts on the magnons as the complex second-neighbor hopping of the Haldane model [@haldaneModel1988]: it opens a gap at the Dirac points and gives the two bands Chern numbers $\pm1$ [@owerreFirst2016; @kimRealization2016]. The magnon number is conserved, so the example tests the band topology of research note *LSWT magnon topology* without anomalous terms. We use the triangular Bravais lattice $\mathbf a_1=(1,0)$, $\mathbf a_2=(1/2,\sqrt3/2)$ with sublattice $A$ at the cell origin and $B$ at $\boldsymbol\tau=(\mathbf a_1+\mathbf a_2)/3$, one spin $S$ per site, and

<!-- eq 1: eq-ex-hc-hamiltonian -->

$$
\hat H=-J\sum_{\langle ij\rangle}\hat{\mathbf S}_i\cdot\hat{\mathbf S}_j
+D\sum_{\langle\langle ij\rangle\rangle}\nu_{ij}\,\hat z\cdot\big(\hat{\mathbf S}_i\times\hat{\mathbf S}_j\big),
\qquad J>0. \tag{1}
$$

The nearest-neighbor vectors from $A$ to $B$ are $\boldsymbol\delta\in\{\boldsymbol\tau,\boldsymbol\tau-\mathbf a_1,\boldsymbol\tau-\mathbf a_2\}$. Each second-neighbor pair is counted once, from $\mathbf r_i$ to $\mathbf r_j=\mathbf r_i+\mathbf b$ with $\mathbf b\in\{\mathbf a_1,\mathbf a_2-\mathbf a_1,-\mathbf a_2\}$, and $\nu_{ij}=+1$ on $A$ and $-1$ on $B$, so that the two sublattices have opposite chirality. Energies are in units of $J$ in the numerical values below.

## Polarized State

We take all spins along $\hat z$. The DM term vanishes for parallel spins whatever their common direction, so the classical energy per site, $E_{\mathrm{cl}}/N=-\frac32JS^2$, does not select a direction; an infinitesimal field along $\hat z$ selects this one, and the field is set to zero below. The state is stationary, because the DM contribution $D\,\mathbf n_j\times\hat z$ to the local field vanishes for $\mathbf n_j=\hat z$ and the exchange field is parallel to every spin.

## Boson Hamiltonian

The exchange part gives, as in research note *LSWT example: square-lattice ferromagnet*, $JS(\hat n_i+\hat n_j-\hat a_i^\dagger\hat a_j-\hat a_j^\dagger\hat a_i)$ per bond. For the DM part, $\hat z\cdot(\hat{\mathbf S}_i\times\hat{\mathbf S}_j)=\frac{\mathrm i}2(\hat S_i^+\hat S_j^--\hat S_i^-\hat S_j^+)$, and with $\hat S^+=\sqrt{2S}\,\hat a$, $\hat S^-=\sqrt{2S}\,\hat a^\dagger$ one pair contributes $\mathrm iD\nu_{ij}S(\hat a_j^\dagger\hat a_i-\hat a_i^\dagger\hat a_j)$. In the full-position Fourier convention of Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*, this pair term becomes $2D\nu S\sin(\mathbf k\cdot\mathbf b)\,\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}$ on its sublattice. The anomalous block vanishes, and the normal block is

<!-- eq 2: eq-ex-hc-block -->

$$
\mathsf A_{\mathbf k}=S\begin{pmatrix}3J+2Ds_{\mathbf k} & -Jf_{\mathbf k}\\ -Jf_{\mathbf k}^* & 3J-2Ds_{\mathbf k}\end{pmatrix},
\qquad
f_{\mathbf k}=\sum_{\boldsymbol\delta}\mathrm e^{\mathrm i\mathbf k\cdot\boldsymbol\delta},
\qquad
s_{\mathbf k}=\sum_{\mathbf b}\sin(\mathbf k\cdot\mathbf b). \tag{2}
$$

Written as $\mathsf A_{\mathbf k}=3JS\,\mathsf I_2+\mathbf d_{\mathbf k}\cdot\boldsymbol\sigma$, the matrix has $d^x_{\mathbf k}-\mathrm id^y_{\mathbf k}=-JSf_{\mathbf k}$ and the mass term $d^z_{\mathbf k}=2DSs_{\mathbf k}$, the form of the Haldane model with nearest-neighbor hopping $JS$ and a second-neighbor phase of $\pm\pi/2$.

## Magnon Bands and the DM Gap

The two bands are the eigenvalues of (2),

<!-- eq 3: eq-ex-hc-bands -->

$$
\varepsilon_{\mp,\mathbf k}=3JS\mp S\sqrt{J^2|f_{\mathbf k}|^2+4D^2s_{\mathbf k}^2}. \tag{3}
$$

At $\Gamma$, $|f_\Gamma|=3$ and $s_\Gamma=0$, so the lower band has the Goldstone mode $\varepsilon_{-,\Gamma}=0$ and the upper band starts at $6JS$. At the corners $\pm\mathbf K$ with $\mathbf K=(4\pi/3,0)$, $f_{\pm\mathbf K}=0$ and $s_{\pm\mathbf K}=\mp3\sqrt3/2$. Without DM the two bands touch there at $3JS$; the DM term opens the gap

<!-- eq 4: eq-ex-hc-gap -->

$$
\varepsilon_{+,\mathbf K}-\varepsilon_{-,\mathbf K}=6\sqrt3\,DS, \tag{4}
$$

with a mass $d^z$ of opposite sign at $\mathbf K$ and $-\mathbf K$. The polarized state is locally stable when $\varepsilon_{-,\mathbf k}\ge0$ on the whole zone. At $\mathbf K$ this requires $3\sqrt3\,|D|<3J$, that is, $|D|<J/\sqrt3$. For $D=0.1\,J$ the lower band of (3) stays nonnegative on the whole zone.

## Chern Numbers

When $D\ne0$ the two bands are separated everywhere: their separation $2S\sqrt{J^2|f_{\mathbf k}|^2+4D^2s_{\mathbf k}^2}$ vanishes only where $f_{\mathbf k}=0$, which happens only at $\pm\mathbf K$, where $s_{\mathbf k}\ne0$. At $M=(\pi,\pi/\sqrt3)$, $|f_M|=1$ and $s_M=0$, so the separation there is $2JS$ for every $D$, and the corner gap (4) is the smaller of the two when $|D|<J/(3\sqrt3)$. For $D=0.1\,J$ the corner gap is the smallest separation on the whole zone, which we checked on a $600\times600$ mesh. The Chern numbers of Chern Number of an Isolated Magnon Band in research note *LSWT magnon topology* are then defined. The Chern number cannot change while the bands stay separated, so it can be evaluated for small $|D|$, where the Berry curvature is concentrated near the corners. There each massive Dirac point contributes $\pm\frac12$ to the Chern number of a band, with a sign fixed by the product of the sign of its mass and the winding of the phase of $f_{\mathbf k}$ around it. At $\mathbf K$ and $-\mathbf K$ both the mass and the winding have opposite signs, so the two contributions add, and each band has $|C|=1$ [@haldaneModel1988]. The overall sign depends on the sign of $D$ and on the orientation convention for the Berry curvature. In the convention of research note *LSWT magnon topology*, the computation in Numerical Check gives $C_-=+1$ and $C_+=-1$ for $D>0$, and the signs reverse for $D<0$. The two Chern numbers sum to zero, as the sum rule of that note requires. The thermal Hall conductivity of these bands follows from the formulas of Magnon Thermal Hall Conductivity in the same note and is not repeated here.

## Numerical Check

For $J=1$, $D=0.1$ and $S=1/2$, the computed bands agree with (3) along $\Gamma$–$\mathbf K$–$M$–$\Gamma$ to machine precision. At $\mathbf K$ the energies are $1.2402$ and $1.7598$, so the gap is $0.5196=6\sqrt3\,DS$, and the upper band at $\Gamma$ is $3.0000=6JS$. On a $30\times30$ mesh the Kubo integral and the link-variable construction agree on the Chern numbers $(C_-,C_+)=(+1,-1)$ for $D=+0.1$ and $(-1,+1)$ for $D=-0.1$.

## References

### Internal Documents

- Momentum-Space BdG Hamiltonian in research note *LSWT boson Hamiltonian*: the full-position Fourier convention used for (2).
- Chern Number of an Isolated Magnon Band and Magnon Thermal Hall Conductivity in research note *LSWT magnon topology*: the definition and sign convention of the Chern numbers, and the sum rule.
- Research note *LSWT example: square-lattice ferromagnet*: the exchange part of the boson Hamiltonian.
- `code-space/examples/worked_examples/honeycomb_topological_ferromagnet.py` and `code-space/tests/test_worked_examples.py`: the numbers of Numerical Check.

### External Sources

- [@haldaneModel1988]: the two-band model with a staggered mass of opposite sign at the two Dirac points and its Chern numbers.
- [@owerreFirst2016]: magnon bands of the honeycomb ferromagnet with second-neighbor DM interaction and their Chern numbers.
- [@kimRealization2016]: the same magnon bands as a bosonic realization of the Haldane model.
