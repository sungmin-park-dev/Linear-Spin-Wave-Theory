<!-- Written on 2026-10-12 as a worked example. Review status: draft; awaiting the user's physics and mathematics review. -->

# Square-Lattice Ferromagnet in a Field

The Heisenberg ferromagnet is the simplest model in which every step of linear spin-wave theory (LSWT) can be done by hand. Its Hamiltonian conserves the number of Holstein–Primakoff bosons about the polarized state, so the BdG matrix has no anomalous block, the zero-point correction vanishes, and the one-magnon energies are exact. We use the square lattice with unit lattice constant, one spin $S$ per site, and

<!-- eq 1: eq-ex-fm-hamiltonian -->

$$
\hat H=-J\sum_{\langle ij\rangle}\hat{\mathbf S}_i\cdot\hat{\mathbf S}_j-h\sum_i\hat S_i^z,
\qquad J>0,\quad h\ge0, \tag{1}
$$

where $\langle ij\rangle$ runs over the $2N$ nearest-neighbor bonds of $N$ sites and $h$ is the Zeeman energy per unit spin along $\hat z$. Energies and the temperature $T$ (meaning $k_{\mathrm B}T$) are in units of $J$ in the numerical values below.

## Polarized State

All spins along $\hat z$ form the classical ground state of (1) for $J>0$ and $h\ge0$. The configuration is stationary because the local field $\mathbf H_i=JS\sum_{\delta}\mathbf n_{i+\delta}+h\hat z=(4JS+h)\hat z$ is parallel to every spin. The classical energy per site is

<!-- eq 2: eq-ex-fm-classical-energy -->

$$
\frac{E_{\mathrm{cl}}}{N}=-2JS^2-hS. \tag{2}
$$

The local frame coincides with the laboratory frame, $\mathbf e^0=\hat z$ and $\mathbf e^\pm=(\hat x\pm\mathrm i\hat y)/\sqrt2$, on every site.

## Boson Hamiltonian

With $\hat S_i^z=S-\hat n_i$, $\hat S_i^+=\sqrt{2S}\,\hat a_i$ and $\hat S_i^-=\sqrt{2S}\,\hat a_i^\dagger$ to the order needed, one bond contributes

$$
\hat{\mathbf S}_i\cdot\hat{\mathbf S}_j
=\hat S_i^z\hat S_j^z+\tfrac12\big(\hat S_i^+\hat S_j^-+\hat S_i^-\hat S_j^+\big)
=S^2-S(\hat n_i+\hat n_j)+S\big(\hat a_i^\dagger\hat a_j+\hat a_j^\dagger\hat a_i\big)+O(S^0),
$$

where $O(S^0)$ collects the quartic terms of the next order. No product $\hat a_i\hat a_j$ appears: in the language of Real-Space Boson Hamiltonian in research note *LSWT boson Hamiltonian*, the anomalous component $\widetilde J^{++}=-J\,\mathbf e^+\cdot\mathbf e^+$ vanishes because $\mathbf e^+\cdot\mathbf e^+=0$ for parallel frames. Each site has four bonds, and the Fourier sum over the two bond directions $\hat x,\hat y$ gives $\sum_{\langle ij\rangle}(\hat a_i^\dagger\hat a_j+\mathrm{h.c.})=\sum_{\mathbf k}4\gamma_{\mathbf k}\,\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k}$, so the quadratic Hamiltonian is

<!-- eq 3: eq-ex-fm-quadratic -->

$$
\hat H_2=\sum_{\mathbf k}\varepsilon_{\mathbf k}\,\hat a_{\mathbf k}^\dagger\hat a_{\mathbf k},
\qquad
\varepsilon_{\mathbf k}=h+4JS\big(1-\gamma_{\mathbf k}\big),
\qquad
\gamma_{\mathbf k}=\tfrac12(\cos k_x+\cos k_y). \tag{3}
$$

The BdG blocks are $A_{\mathbf k}=\varepsilon_{\mathbf k}$ and $B_{\mathbf k}=0$, and the magnon operators are the Holstein–Primakoff bosons themselves.

## Magnon Dispersion

The band (3) runs from $\varepsilon_\Gamma=h$ at $\Gamma$ through $\varepsilon_X=h+4JS$ at $X=(\pi,0)$ to $\varepsilon_M=h+8JS$ at $M=(\pi,\pi)$. Near $\Gamma$, $1-\gamma_{\mathbf k}=|\mathbf k|^2/4+O(|\mathbf k|^4)$ and

<!-- eq 4: eq-ex-fm-small-k -->

$$
\varepsilon_{\mathbf k}=h+JS|\mathbf k|^2+O(|\mathbf k|^4). \tag{4}
$$

At $h=0$ the mode at $\Gamma$ is the Goldstone mode of the broken spin-rotation symmetry, and its dispersion is quadratic, as for any ferromagnet, because the order parameter is itself a conserved quantity. The field breaks the rotation symmetry explicitly and opens the gap $h$; the polarized state keeps the rotation symmetry about $\hat z$, so no gapless mode remains.

The dispersion (3) is exact, not only harmonic. The Hamiltonian (1) conserves $\sum_i\hat S_i^z$, and on the states with one spin lowered by one unit it acts as a hopping problem: lowering spin $i$ changes the $zz$ energy of each of its four bonds by $+JS$ and adds $h$, and each transverse term moves the lowered spin to a neighbor with amplitude $-JS$. The plane wave of this hopping problem has energy (3) above the polarized state.

## Ground-State Energy

Because $B_{\mathbf k}=0$, the zero-point correction $\Delta E_{\mathrm{zp}}=\frac12\sum_{\mathbf k}(\varepsilon_{\mathbf k}-A_{\mathbf k})$ of Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization* vanishes, and $E_{\mathrm{GS}}=E_{\mathrm{cl}}$. This is again exact: the polarized state is an eigenstate of (1) with the largest total $S^z$, and its energy (2) is the lowest one.

## Thermal Reduction of the Moment

With $B_{\mathbf k}=0$ the boson number of Boson Number and Reduced Moment in research note *LSWT magnon thermodynamics* reduces to the momentum average of the Bose occupation,

<!-- eq 5: eq-ex-fm-moment -->

$$
S-\langle\hat S^z\rangle=\langle\hat n\rangle=\frac1N\sum_{\mathbf k}\frac{1}{\exp(\varepsilon_{\mathbf k}/T)-1}. \tag{5}
$$

At $T\ll JS$ only the quadratic part (4) of the band is occupied. Replacing the average by $\int d^2k/(2\pi)^2$ over the plane, with $x=JS|\mathbf k|^2$ and $d^2k=\pi\,d(|\mathbf k|^2)$,

<!-- eq 6: eq-ex-fm-moment-low-t -->

$$
\langle\hat n\rangle\simeq\frac{1}{4\pi JS}\int_0^\infty\frac{dx}{\exp[(h+x)/T]-1}
=-\frac{T}{4\pi JS}\,\ln\!\big(1-\mathrm e^{-h/T}\big). \tag{6}
$$

For $h\ll T$ the right-hand side grows as $(T/4\pi JS)\ln(T/h)$ and diverges at $h=0$. A two-dimensional isotropic ferromagnet therefore has no spontaneous moment at $T>0$, in agreement with the Mermin–Wagner theorem [@merminAbsence1966]. Within LSWT this appears as a reduction (5) that exceeds $S$, and the harmonic theory is no longer valid once $\langle\hat n\rangle$ is comparable to $S$, that is, for fields below $h\sim T\exp(-4\pi JS^2/T)$.

## Numerical Check

For $J=1$, $S=1$ and $h=0.1$, the polarized state has $E_{\mathrm{cl}}/N=-2.1$ and $\Delta E_{\mathrm{zp}}=0$, and the computed band agrees with (3) along $\Gamma$–$X$–$M$–$\Gamma$ to machine precision, with $\varepsilon_\Gamma=0.1$ and $\varepsilon_M=8.1$. On a $96\times96$ mesh the moment reduction (5) is $5.821\times10^{-4}$ at $T=0.05$ and $3.692\times10^{-3}$ at $T=0.1$, and the low-temperature form (6) gives $5.786\times10^{-4}$ and $3.650\times10^{-3}$. The difference of about one percent at $T=0.1$ comes from the $O(|\mathbf k|^4)$ terms of the band that (6) neglects.

## References

### Internal Documents

- Real-Space Boson Hamiltonian in research note *LSWT boson Hamiltonian*: circular exchange components, of which the anomalous one vanishes here.
- Paraunitary Diagonalization in research note *LSWT paraunitary diagonalization*: the zero-point correction.
- Boson Number and Reduced Moment, and Zero Modes, in research note *LSWT magnon thermodynamics*: the moment reduction (5) and its divergence for a gapless band in two dimensions.
- `code-space/examples/worked_examples/square_ferromagnet.py` and `code-space/tests/test_worked_examples.py`: the numbers of Numerical Check.

### External Sources

- [@holsteinField1940]: the boson representation of the spin operators.
- [@merminAbsence1966]: absence of spontaneous magnetization in the two-dimensional isotropic Heisenberg model at $T>0$.
