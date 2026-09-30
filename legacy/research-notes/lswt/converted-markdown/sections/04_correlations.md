# Correlations in Linear Spin Wave Theory

In this section, we discuss the physical quantities related to spin-spin
correlation functions. Specifically, from real-time (or dynamical)
spin-spin correlation function to structure factor and spectral
functions, these are quantities of fundamental interest in understanding
magnetic excitations.

- **Real-time correlation function:** The real-time spin-spin
  correlation function is a $3 \times 3$ matrix as a function of
  momentum and time: $$
\begin{equation}
          \mathcal{C}^{\alpha\beta} (\mathbf{k}, t) = \frac{1}{L} \sum_{i, j} e^{- i \mathbf{k} \cdot (\mathbf{r}_{i} - \mathbf{r}_{j})} \langle \hat{S}_{i}^{\alpha}(t) \hat{S}_{j}^{\beta}(0) \rangle,
      
  \end{equation}
$$ where $\alpha, \beta = x, y, z$ or $\pm, 0$ and $L$
  is the number of lattice sites for spins.

  - Note that the real-time spin-spin correlation function is
    equivalently defined as $$
\begin{equation}
                \mathcal{C}^{\alpha\beta} (\mathbf{k}, t) 
                = \big\langle \hat{S}_{+ \mathbf{k}}^{\alpha}(t) \hat{S}_{- \mathbf{k}}^{\beta}(0) \big\rangle,  
            
    \end{equation}
$$ by employing the spin Fourier transformation
    $S_{\mathbf{k}}^{\alpha} = \frac{1}{\sqrt{N}} \sum_{i} e^{-i \mathbf{k} \cdot \mathbf{r}_{i}} S_{i}^{\alpha}$.

  - The spin-spin correlation function under complex conjugation is
    $$
\begin{equation}
                [\mathcal{C}^{\alpha\beta}(\mathbf{k}, t)]^{*}
                = \langle [S_{\mathbf{k}}^{\alpha} ( t ) S_{-\mathbf{k}}^{\beta} ( 0 )]^{\dagger} \rangle
                = \langle (S_{-\mathbf{k}}^{\beta} ( 0 ))^{\dagger} (S_{\mathbf{k}}^{\alpha} ( t ))^{\dagger} \rangle.
            
    \end{equation}
$$ At this point, the spin-spin correlation transforms
    differently depending on indices $\alpha, \beta$:

    $$
\begin{equation}
                \langle (S_{-\mathbf{k}}^{\beta} ( 0 ))^{\dagger} (S_{\mathbf{k}}^{\alpha} ( t ))^{\dagger} \rangle
                = 
                \begin{cases}
                    \langle S_{\mathbf{k}}^{\bar{\beta}} (0) S_{-\mathbf{k}}^{\bar{\alpha}} (t) \rangle
                    = \langle S_{\mathbf{k}}^{\bar{\beta}} (-t) S_{-\mathbf{k}}^{\bar{\alpha}} (0) \rangle
                    = \mathcal{C}^{\bar{\beta}\bar{\alpha}}(-\mathbf{k}, -t), 
                    & \text{if } \alpha,\beta = \pm, 0, \\[1em]
                    \langle S_{\mathbf{k}}^{\beta} (0) S_{-\mathbf{k}}^{\alpha} (t) \rangle
                    = \langle S_{\mathbf{k}}^{\beta} (-t) S_{-\mathbf{k}}^{\alpha} (0) \rangle 
                    = \mathcal{C}^{\beta\alpha}(-\mathbf{k}, -t), 
                    & \text{if } \alpha, \beta = x, y, z.
                \end{cases}
            
    \end{equation}
$$ where in the second equality we use the identity
    $\langle \hat{A}(0) \hat{B}(t) \rangle = \langle \hat{A}(-t) \hat{B}(0) \rangle$
    for thermal states $\rho = e^{-\beta H}/Z$: $$
\begin{equation}
                \langle \hat{A}(0) \hat{B}(t) \rangle
                = \Tr[\rho \hat{A} \hat{U}^{\dagger}_{t} \hat{B} \hat{U}_{t}] 
                = \Tr[\rho \hat{U}_{t} \hat{A} \hat{U}^{\dagger}_{t} \hat{B}(0)]  
                = \langle \hat{A}(-t) \hat{B}(0) \rangle.
            
    \end{equation}
$$

- **Static structure factor:** The static structure factor is defined as
  $$
\begin{equation}
          \mathcal{S}^{\alpha\beta}(\mathbf{k}) 
          \equiv
          \frac{1}{N} \sum_{i,j} e^{-i \mathbf{k} \cdot (\mathbf{r}_{i} - \mathbf{r}_{j})} 
          \big\langle \hat{S}^{\alpha}_{i} \hat{S}^{\beta}_{j} \big\rangle
          = \big\langle \hat{S}_{+ \mathbf{k}}^{\alpha} \hat{S}_{- \mathbf{k}}^{\beta} \big\rangle 
          = \mathcal{C}^{\alpha\beta}(\mathbf{k}, 0).
      
  \end{equation}
$$ This equals the equal-time spin-spin correlation
  function in momentum space, and can also be viewed as its
  time-average.

- **Dynamical structure factor:** The dynamic structure factor is
  defined as $$
\begin{equation}
          \mathcal{S}^{\alpha\beta}(\mathbf{k}, \omega) 
          \equiv \frac{1}{2\pi} \int_{-\infty}^{\infty} dt \, \mathcal{C}^{\alpha\beta}(\mathbf{k}, t) e^{i \omega t}.
      
  \end{equation}
$$ This quantity is directly related to the scattering
  cross-section in neutron scattering experiments.

- **Real space correlation function in the reference frame:** The
  spin-spin correlation in the local reference frame is:
  $$
\begin{equation}
          \langle \widetilde{S}^{\alpha}_{i} \widetilde{S}_{j}^{\beta} \rangle,
      
  \end{equation}
$$ where $\widetilde{S}^{\alpha}_{i}$ represents the
  spin component in the local frame.

- **Spectral function:** The spectral function is defined as:
  $$
\begin{equation}
          \mathcal{A}^{\alpha\beta}(\mathbf{k}, \omega) = -\frac{1}{\pi} \mathrm{Im}[G^{\alpha\beta}_{\rm R}(\mathbf{k}, \omega)],
      
  \end{equation}
$$ where $G^{\alpha\beta}_{\rm R}(\mathbf{k}, \omega)$
  is the retarded Green's function: $$
\begin{equation}
          G^{\alpha\beta}_{\rm R}(\mathbf{k}, \omega) 
          \equiv \frac{1}{i} \int_{-\infty}^{\infty} dt \, e^{i \omega t} \theta(t) \big\langle [S_{\mathbf{k}}^{\alpha}(t), S_{-\mathbf{k}}^{\beta}(0)] \big\rangle.
      
  \end{equation}
$$ The spectral function provides information about the
  density of states of the magnetic excitations.

## Real-Time (Dynamical) Spin-Spin Correlation Function {#subsec: Real-Time (Dynamical) Spin-Spin Correlation Function}

In this section, we derive the spin-spin correlation function in linear
spin-wave theory (LSWT). The real-time spin-spin correlation function is
a $3 \times 3$ matrix as a function of momentum and time:
$$
\begin{equation}
    \mathcal{C}^{\alpha\beta}(\mathbf{k}, t) = \frac{1}{L} \sum_{i, j} e^{-i \mathbf{k} \cdot (\mathbf{r}_{i} - \mathbf{r}_{j})} \langle \hat{S}_{i}^{\alpha}(t) \hat{S}_{j}^{\beta}(0) \rangle,
\end{equation}
$$ where $\alpha, \beta = x, y, z$ or $\pm, 0$ and $L$ is
the number of lattice sites for spins. To proceed, we express the
$\alpha$-component of the spin operator as $$
\begin{equation}
    \hat{S}^{\alpha}_{i} 
    = \hat{\mathbf{e}}^{\alpha} \cdot \left( \frac{\widetilde{S}_{i}^{+} \hat{\mathbf{e}}^{-}_{i} + \widetilde{S}_{i}^{-} \hat{\mathbf{e}}^{+}_{i}}{\sqrt{2}} + \widetilde{S}^{0}_{i} \hat{\mathbf{e}}^{0}_{i} \right) 
    = \frac{1}{\sqrt{2}} \left( \bar{u}^{\alpha}_{i} \widetilde{S}_{i}^{+} + u^{\alpha}_{i} \widetilde{S}_{i}^{-} \right) + v^{\alpha}_{i} \widetilde{S}^{0}_{i},
\end{equation}
$$ where
$u^{\alpha}_{i} = \hat{\mathbf{e}}^{\alpha} \cdot \hat{\mathbf{e}}^{+}_{i}$,
$\bar{u}^{\alpha}_{i} = (u^{\alpha}_{i})^*$, and
$v^{\alpha}_{i} = \hat{\mathbf{e}}^{\alpha} \cdot \hat{\mathbf{e}}^{0}_{i}$
are components of the spin basis vectors. When the index $\alpha$ are
$\pm,0$, then the
$(u^{\alpha}_{i}, \bar{u}^{\alpha}_{i}, v^{\alpha}_{i})$ can be obtained
from $\mathbf{Q}$: $$
\begin{equation}
    [\mathbf{Q}]_{\alpha\beta} = 
    \hat{\bf e}^{\alpha} \cdot \hat{\bf e}^{\beta}_{i} = 
    \hat{\bf e}^{\alpha} \cdot \mathbf{R}_{i} \hat{\bf e}^{\beta}: 
    \qquad
    \mathbf{Q}
    =
    \begin{pmatrix}
            u^{-}_{i} &   \bar{u}^{-}_{i}     &   v_{i}^{-}   \\
            u^{+}_{i} &   \bar{u}^{+}_{i}     &   v_{i}^{+}   \\
            u^{0}_{i} &   \bar{u}^{0}_{i}     &   v_{i}^{0}   \\
    \end{pmatrix}
    = 
    \begin{pmatrix}
        (\hat{\bf e}^{-})^{\rm T} \\  
        (\hat{\bf e}^{+})^{\rm T} \\  
        (\hat{\bf e}^{0})^{\rm T} 
    \end{pmatrix}
    \mathbf{R}_{i}
    \begin{pmatrix}
        \hat{\bf e}^{+} & \hat{\bf e}^{-} & \hat{\bf e}^{0}
    \end{pmatrix}
    = \mathbf{C}^{\dagger} \mathbf{R}_{i} \mathbf{C},
\end{equation}
$$ If the indices $\alpha$ are $x,y,z$,
$\mathbf{U}_{i} = \mathbf{R}_{i} \mathbf{C}$. Substituting into the
spin-spin correlation function, we obtain $$
\begin{align}
    \langle S_i^{\alpha}(t) S_j^{\beta}(0) \rangle
    &= \frac{1}{2} \Bigg\langle 
        \big( \bar{u}^{\alpha}_{i} \widetilde{S}_{i}^{+}(t) + u^{\alpha}_{i} \widetilde{S}_{i}^{-}(t) \big) 
        \big( \bar{u}^{\beta}_{j} \widetilde{S}_{j}^{+}(0) + u^{\beta}_{j} \widetilde{S}_{j}^{-}(0) \big) 
    \Bigg\rangle 
    + v^{\alpha}_{i} v^{\beta}_{j} \langle \widetilde{S}^{0}_{i}(t) \widetilde{S}^{0}_{j}(0) \rangle \nonumber \\
    &\quad + \frac{1}{\sqrt{2}} \Bigg[ 
        v^{\alpha}_{i} \Bigg\langle 
        \widetilde{S}^{0}_{i}(t) 
        \big( \bar{u}^{\beta}_{j} \widetilde{S}_{j}^{+}(0) + u^{\beta}_{j} \widetilde{S}_{j}^{-}(0) \big) 
        \Bigg\rangle 
        + v^{\beta}_{j} \Bigg\langle 
        \big( \bar{u}^{\alpha}_{i} \widetilde{S}_{i}^{+}(t) + u^{\alpha}_{i} \widetilde{S}_{i}^{-}(t) \big) 
        \widetilde{S}^{0}_{j}(0) 
        \Bigg\rangle 
    \Bigg].
\end{align}
$$ In LSWT, which is a quadratic bosonic system, odd-order
bosonic correlation functions vanish, so the terms in the second line
are zero. Thus, $$
\begin{align}
    \langle S_i^{\alpha}(t) S_j^{\beta}(0) \rangle
    &= \frac{1}{2} \Bigg\langle 
        \big( \bar{u}^{\alpha}_{i} \widetilde{S}_{i}^{+}(t) + u^{\alpha}_{i} \widetilde{S}_{i}^{-}(t) \big) 
        \big( \bar{u}^{\beta}_{j} \widetilde{S}_{j}^{+}(0) + u^{\beta}_{j} \widetilde{S}_{j}^{-}(0) \big) 
    \Bigg\rangle 
    + v^{\alpha}_{i} v^{\beta}_{j} \langle \widetilde{S}^{0}_{i}(t) \widetilde{S}^{0}_{j}(0) \rangle \\
    &= \frac{1}{2} \sum_{m,n} \left[ 
    \begin{pmatrix}
        \bar{u}^{\alpha}_{i} & 0 \\ 0 & u^{\alpha}_{i}
    \end{pmatrix}
    \begin{pmatrix}
        \langle \widetilde{S}^{+}_{i}(t) \widetilde{S}^{-}_{j}(0) \rangle &
        \langle \widetilde{S}^{+}_{i}(t) \widetilde{S}^{+}_{j}(0) \rangle \\[0.25em]
        \langle \widetilde{S}^{-}_{i}(t) \widetilde{S}^{-}_{j}(0) \rangle &
        \langle \widetilde{S}^{-}_{i}(t) \widetilde{S}^{+}_{j}(0) \rangle
    \end{pmatrix}
    \begin{pmatrix}
        u^{\beta}_{j} & 0 \\ 0 & \bar{u}^{\beta}_{j}
    \end{pmatrix}
    \right]_{m,n}
    + v^{\alpha}_{i} v^{\beta}_{j} \langle \widetilde{S}^{0}_{i}(t) \widetilde{S}^{0}_{j}(0) \rangle.
\end{align}
$$

Now, we define two matrix quantities: $\mathsf{U}^{\alpha}_{i}$, which
maps the spin operators from the reference frame to the laboratory
frame, and the correlation matrix between spin ladder operators (or
transversal fluctuations). $$
\begin{equation}
    \mathsf{U}^{\alpha}_{i} 
    \equiv 
    \begin{pmatrix}
         \bar{u}^{\alpha}_{i} & 0 \\ 0 & u^{\alpha}_{i} 
    \end{pmatrix},
    \qquad \text{and} \qquad
    \big\langle \widetilde{\mathbf{S}}^{\pm}_{i}(t) \widetilde{\mathbf{S}}^{\mp}_{j}(0) \big\rangle
    = 
    \begin{pmatrix}
        \langle \widetilde{S}^{+}_{i}(t) \widetilde{S}^{-}_{j}(0) \rangle &
        \langle \widetilde{S}^{+}_{i}(t) \widetilde{S}^{+}_{j}(0) \rangle \\[0.25em]
        \langle \widetilde{S}^{-}_{i}(t) \widetilde{S}^{-}_{j}(0) \rangle &
        \langle \widetilde{S}^{-}_{i}(t) \widetilde{S}^{+}_{j}(0) \rangle
    \end{pmatrix}
\end{equation}
$$ The Holstein-Primakoff (HP) operators are
$$
\begin{equation}
    \widetilde{S}_{i}^{+} = \sqrt{2 S_i} \left( 1 - \frac{\hat{n}_i}{2 S_i} \right)^{1/2} \hat{b}_i, 
    \qquad \widetilde{S}_{i}^{-} = \sqrt{2 S_i} \hat{b}_i^{\dagger} \left( 1 - \frac{\hat{n}_i}{2 S_i} \right)^{1/2}, 
    \qquad \widetilde{S}_{i}^{0} = S_i - \hat{n}_i,
\end{equation}
$$ with $\hat{n}_i = \hat{b}_i^{\dagger} \hat{b}_i$ being
the magnon number operator. We approximate
$\widetilde{S}_{i}^{+} \approx \sqrt{2 S_i} \hat{b}_i$ and
$\widetilde{S}_{i}^{-} \approx \sqrt{2 S_i} \hat{b}_i^{\dagger}$,
yielding $$
\begin{equation}
    \langle S_i^{\alpha}(t) S_j^{\beta}(0) \rangle
    = \sqrt{S_i S_j} \sum_{m,n} \left[ 
    \mathsf{U}^{\alpha}_{i}
    \big\langle \Psi_{i}(t) \Psi_{j}^{\dagger}(0) \big\rangle
    \mathsf{U}^{\beta}_{j}
    \right]_{m,n}
    + v^{\alpha}_{i} v^{\beta}_{j} \big( S_i S_j -  S_i \langle \hat{n}_j(0) \rangle - S_j \langle \hat{n}_i(t) \rangle \big),
\end{equation}
$$ where
$\big\langle \Psi_{i}(t) \Psi_{j}^{\dagger}(0) \big\rangle$ is the
matrix form of the two-point correlation function, defined as
$$
\begin{equation}
    \big\langle \Psi_{i}(t) \Psi_{j}^{\dagger}(0) \big\rangle
    \equiv
    \begin{pmatrix}
        \langle \hat{b}_i(t) \hat{b}_j^{\dagger}(0) \rangle &
        \langle \hat{b}_i(t) \hat{b}_j(0) \rangle \\
        \langle \hat{b}_i^{\dagger}(t) \hat{b}_j^{\dagger}(0) \rangle &
        \langle \hat{b}_i^{\dagger}(t) \hat{b}_j(0) \rangle
    \end{pmatrix}
    = \frac{1}{\sqrt{S_{i}S_{j}}}
    \big\langle \widetilde{\mathbf{S}}^{\pm}_{i}(t) \widetilde{\mathbf{S}}^{\mp}_{j}(0) \big\rangle
\end{equation}
$$

The expectation value of the number operator satisfies
$\langle \hat{n}_i(t) \rangle = \langle \hat{n}_i(0) \rangle$ because,
for a time-independent Hamiltonian $\hat{H}$, $$
\begin{equation}
    \langle \hat{A}(t) \rangle = 
    \begin{cases}
        \langle \psi_n | e^{i \hat{H} t} \hat{A} e^{-i \hat{H} t} | \psi_n \rangle = \langle \hat{A} \rangle, & \text{if } \hat{H} | \psi_n \rangle = E_n | \psi_n \rangle, \\
        \mathrm{Tr} [ \rho \hat{U}^{\dagger}(t) \hat{A} \hat{U}(t) ] = \mathrm{Tr} [ \rho \hat{A} ], & \text{if } \rho = e^{-\beta \hat{H}} / Z,
    \end{cases}
\end{equation}
$$ where $\hat{U}(t) = e^{-i \hat{H} t}$ is the
time-evolution operator and $\rho = e^{-\beta \hat{H}} / Z$ is the
thermal density matrix. In the Gibbs state, this follows from the
commutation of $\rho$ with $\hat{U}(t)$ and the cyclic property of the
trace.

### Sublattice Spin-Spin Correlation Function

Given $L$ spins (or magnetic ions), there could be $m_{s}$ magnetic
sublattices and $N$ unit cells with $L = m_{s}N$. In this case, we
introduce a composite index notation: $I = (i, \mu)$ and $J = (j, \nu)$,
where $\mu, \nu = a, b, \dots$ denote the magnetic sublattices and
$i, j$ represent the unit cell indices. The position in the lattice is
given by $\mathbf{R}_{I} = \mathbf{r}_{i} + \boldsymbol{\delta}_{\mu}$,
where $\mathbf{r}_i$ is the position of the $i$-th unit cell and
$\boldsymbol{\delta}_\mu$ is the position of the $\mu$-th sublattice
within the unit cell.

Explicitly, for a spin at site $I = (i, \mu)$, its position vector is
$$
\begin{equation}
    \mathbf{R}_{I} = \mathbf{R}_{i\mu} = \mathbf{r}_{i} + \boldsymbol{\delta}_{\mu}.
\end{equation}
$$

To compute the real-time spin-spin correlation function in momentum
space using Linear Spin-Wave Theory (LSWT), we express it in terms of
magnetic sublattices: $$
\begin{align}
    \mathcal{C}^{\alpha\beta}(\mathbf{k}, t) 
    &= \frac{1}{Nm_{s}} \sum_{i\mu,j\nu} e^{-i\mathbf{k} \cdot [(\mathbf{r}_{i} + \boldsymbol{\delta}_{\mu}) - (\mathbf{r}_{j} + \boldsymbol{\delta}_{\nu})]} \langle \hat{S}_{i\mu}^{\alpha}(t) \hat{S}_{j\nu}^{\beta}(0) \rangle \\
    &= \frac{1}{m_{s}} \sum_{\mu,\nu} e^{-i\mathbf{k} \cdot (\boldsymbol{\delta}_{\mu} - \boldsymbol{\delta}_{\nu})} 
    \left[ 
    \frac{1}{N} \sum_{i,j} e^{-i\mathbf{k} \cdot (\mathbf{r}_{i} - \mathbf{r}_{j})} 
    \langle \hat{S}_{i\mu}^{\alpha}(t) \hat{S}_{j\nu}^{\beta}(0) \rangle \right] 
\end{align}
$$

This allows us to express the total correlation function as an average
over sublattice correlation functions: $$
\begin{align}
    \mathcal{C}^{\alpha\beta}(\mathbf{k}, t) 
    = \frac{1}{m_s} \sum_{\mu, \nu = a, b, \dots} 
    \mathcal{C}_{\mu\nu}^{\alpha\beta}(\mathbf{k}, t),
\end{align}
$$ where $m_s$ is the number of magnetic sublattices in the
unit cell, and $\mu, \nu$ denote sublattice indices.

The key insight of this section is to demonstrate that the sublattice
real-time spin-spin correlation function
$\mathcal{C}_{\mu\nu}^{\alpha\beta}(\mathbf{k}, t)$ can be obtained from
a partial sum of the $2m_s \times 2m_s$ matrix
$\mathsf{C}^{\alpha\beta}(\mathbf{k}, t)$: $$
\begin{equation}
    \begin{split}
        \mathcal{C}_{\mu\nu}^{\alpha\beta}(\mathbf{k}, t) 
        &= \sum_{(m,n)_{\mu,\nu}} 
        \big[ \mathsf{C}^{\alpha\beta}(\mathbf{k}, t) \big]_{m,n} \\
        &= \big[ \mathsf{C}^{\alpha\beta}(\mathbf{k}, t) \big]_{\mu, \nu} 
        + \big[ \mathsf{C}^{\alpha\beta}(\mathbf{k}, t) \big]_{m_s + \mu, \nu} 
        + \big[ \mathsf{C}^{\alpha\beta}(\mathbf{k}, t) \big]_{\mu, m_s + \nu} 
        + \big[ \mathsf{C}^{\alpha\beta}(\mathbf{k}, t) \big]_{m_s + \mu, m_s + \nu},
    \end{split}
\end{equation}
$$ where the index set
$(m,n)_{\mu,\nu} = \{ (\mu, \nu), (m_s + \mu, \nu), (\mu, m_s + \nu), (m_s + \mu, m_s + \nu) \}$
represents all possible combinations of indices in the extended Nambu
space. The matrix $\mathsf{C}^{\alpha\beta}$ is defined as
$$
\begin{equation}
    \boxed{
        \mathsf{C}^{\alpha\beta}(\mathbf{k}, t) 
        = 
        \mathsf{R}^{\alpha}_{\mathbf{k}} 
        \mathsf{T}_{\mathbf{k}} 
        \mathsf{N}_{\mathbf{k}}(t) 
        \mathsf{T}_{\mathbf{k}}^{\dagger}
        [\mathsf{R}^{\beta}_{\mathbf{k}}]^{\dagger},
    }
\end{equation}
$$ where $\mathsf{T}_{\mathbf{k}}$ is a para-unitary
matrix that diagonalizes the Hamiltonian matrix, satisfying
$\mathsf{T}_{\mathbf{k}}^{\dagger} \mathsf{H}_{\mathbf{k}} \mathsf{T}_{\mathbf{k}} = \mathsf{E}_{\mathbf{k}}$,
and $\mathsf{N}_{\mathbf{k}}(t)$ is the time-dependent correlation
matrix in the magnon basis.

The matrix $\mathsf{R}^{\alpha}_{\mathbf{k}}$ is a $2m_s \times 2m_s$
block-diagonal matrix that encodes both the spin components and
sublattice structure. It is defined as
$\mathsf{R}^{\alpha}_{\mathbf{k}} = \mathsf{U}^{\alpha} \mathsf{S}_{\mathbf{k}}$,
which can be expressed explicitly as: $$
\begin{equation}
    \mathsf{R}^{\alpha}_{\mathbf{k}} 
    = \mathsf{U}^{\alpha} \mathsf{S}_{\mathbf{k}}
    =
    \begin{pmatrix}
        \bar{\mathsf{u}}^{\alpha} \mathsf{s}_{\mathbf{k}} & \mathbf{0} \\
        \mathbf{0} & \mathsf{u}^{\alpha} \mathsf{s}_{\mathbf{k}} 
    \end{pmatrix},
\end{equation}
$$ where the diagonal matrices $\mathsf{s}$ and
$\mathsf{u}^{\alpha}$ are defined as: $$
\begin{equation}
    \begin{cases}
        \mathsf{s} = \text{diag} \big[ e^{-i \mathbf{k} \cdot \boldsymbol{\delta}_a} \sqrt{S_a}, e^{-i \mathbf{k} \cdot \boldsymbol{\delta}_b} \sqrt{S_b}, \dots, e^{-i \mathbf{k} \cdot \boldsymbol{\delta}_\mu} \sqrt{S_\mu}, \dots \big], \\
        \mathsf{u}^{\alpha} = \text{diag} \big[ u_a^{\alpha}, u_b^{\alpha}, \dots, u_\mu^{\alpha}, \dots \big],
    \end{cases}
\end{equation}
$$ where $\mathsf{u}^{\alpha}$ contains the
$\alpha$-component of the spin basis vector $\hat{\mathbf{e}}^-$ for
each sublattice, i.e.,
$\mathsf{u}^{\alpha}_\mu = [\hat{\mathbf{e}}^+_\mu]^{\alpha}$, and
$\bar{\mathsf{u}}^{\alpha} = [\mathsf{u}^{\alpha}]^* = [\mathsf{u}^{\alpha}]^{\dagger}$.
The matrix $\mathsf{s}$ encodes both the phase factors due to sublattice
positions and the square root of spin magnitudes $S_\mu$.

The matrix $\mathsf{N}_{\mathbf{k}}(t)$ encodes the time evolution of
magnon correlations and is given by: $$
\begin{equation}
    \mathsf{N}_{\mathbf{k}}(t) 
    = 
    \begin{pmatrix}
        \langle \hat{\boldsymbol{\beta}}_{\mathbf{k}}(t) \hat{\boldsymbol{\beta}}_{\mathbf{k}}^{\dagger}(0) \rangle & \langle \hat{\boldsymbol{\beta}}_{\mathbf{k}}(t) \hat{\boldsymbol{\beta}}_{-\mathbf{k}}(0) \rangle \\
        \langle \hat{\boldsymbol{\beta}}_{-\mathbf{k}}^{\dagger}(t) \hat{\boldsymbol{\beta}}_{\mathbf{k}}^{\dagger}(0) \rangle & \langle \hat{\boldsymbol{\beta}}_{-\mathbf{k}}^{\dagger}(t) \hat{\boldsymbol{\beta}}_{-\mathbf{k}}(0) \rangle
    \end{pmatrix}
    = 
    \begin{pmatrix}
        (\mathsf{I} + \mathsf{n}_{\mathbf{k}})e^{-it\mathsf{E}_{\mathbf{k}}} & \mathbf{0} \\
        \mathbf{0} & \mathsf{n}_{-\mathbf{k}}e^{it\mathsf{E}_{-\mathbf{k}}}
    \end{pmatrix},
\end{equation}
$$ where $\mathsf{I}$ is the $m_s \times m_s$ identity
matrix, $\mathsf{n}_{\mathbf{k}}$ and $\mathsf{E}_{\mathbf{k}}$ are
diagonal matrices containing the Bose-Einstein distribution
$[\mathsf{n}_{\mathbf{k}}]_{\mu\nu} = \delta_{\mu\nu}/[e^{\beta E_\mu(\mathbf{k})} - 1]$
and the magnon energies
$[\mathsf{E}_{\mathbf{k}}]_{\mu\nu} = \delta_{\mu\nu}E_{\mu}(\mathbf{k})$,
respectively. The time dependence arises from the energy eigenstates:
$e^{i\hat{H}t}\hat{\beta}_{\mathbf{k},\mu}e^{-i\hat{H}t} = e^{-iE_{\mathbf{k},\mu}t}\hat{\beta}_{\mathbf{k},\mu}$.

To derive this, we assume a spin system with magnetic sublattice
structures within the unit cell. The sublattice correlation function is
expressed as $$
\begin{align}
    \mathcal{C}^{\alpha\beta}_{\mu\nu} (\mathbf{k}, t) 
    = & \frac{1}{N} \sum_{i,j} e^{ - i \mathbf{k} \cdot [(\mathbf{r}_{i} + \mathbf{\delta}_{\mu}) - (\mathbf{r}_{j} + \mathbf{\delta}_{\nu})]} \langle S_{i\mu}^{\alpha}(t) S_{j\nu}^{\beta}(0) \rangle \\
    = & e^{ - i \mathbf{k} \cdot (\mathbf{\delta}_{\mu} - \mathbf{\delta}_{\nu})}  
    \frac{1}{N} \sum_{i,j} e^{ - i \mathbf{k} \cdot (\mathbf{r}_{i} - \mathbf{r}_{j} )} 
    \langle S_{i\mu}^{\alpha}(t) S_{j\nu}^{\beta}(0) \rangle \\
    = & e^{ - i \mathbf{k} \cdot (\mathbf{\delta}_{\mu} - \mathbf{\delta}_{\nu})}  
    \frac{1}{N}  \\
    & \times \sum_{i,j} e^{ - i \mathbf{k} \cdot (\mathbf{r}_{i} - \mathbf{r}_{j} )} 
    \bigg[
    \sqrt{S_{\mu}S_{\nu}} \sum_{m,n}
    \Big[ \mathsf{U}^{\alpha}_{\mu}
    \big\langle \Psi_{i\mu}^{\phantom \dagger}(t) \Psi_{j\nu}^{\dagger}(0) \big\rangle
    \mathsf{U}^{\beta}_{\nu} \Big]_{m,n} 
    + v_{\mu}^{\alpha} v_{\nu}^{\beta}
    \Big( S_{\nu} S_{\mu} - S_{\nu}\langle \hat{n}_{i\mu} \rangle - S_{\mu}\langle \hat{n}_{j\nu} \rangle \Big)
    \bigg], 
    \nonumber
\end{align}
$$ where the spin operators are transformed into bosonic
operators via the Holstein-Primakoff transformation, and the correlation
is computed using Fourier transforms. The first term, involving the
bosonic correlation matrix, yields $$
\begin{align}
        \frac{1}{N} \sum_{i,j}   
        e^{ - i \mathbf{k} \cdot (\mathbf{r}_{i} - \mathbf{r}_{j} )} 
        \big\langle \Psi_{i}^{\phantom \dagger}(t) \Psi_{j}^{\dagger}(0) \big\rangle 
        = &
        \big\langle \Psi_{\mathbf{k}}^{\phantom \dagger}(t) \Psi_{\mathbf{k}}^{\dagger}(0) \big\rangle,
        \\
        \frac{1}{N} \sum_{\mathbf{k}} 
        e^{ + i \mathbf{k} \cdot (\mathbf{r}_{i} - \mathbf{r}_{j} )} 
        \big\langle \Psi_{\mathbf{k}}^{\phantom \dagger}(t) \Psi_{\mathbf{k}}^{\dagger}(0) \big\rangle 
        = &
        \big\langle \Psi_{i}^{\phantom \dagger}(t) \Psi_{j}^{\dagger}(0) \big\rangle.
    \end{align}
$$ Similarly, the spin-spin correlation function in real
space is obtained as $$
\begin{equation}
    \big\langle \widetilde{\mathbf{S}}_{i}^{\pm}(t) \widetilde{\mathbf{S}}_{j}^{\mp}(0) \big\rangle
    = \frac{1}{N} \sum_{\mathbf{k}, \mu, \nu } 
    e^{ + i \mathbf{k} \cdot (\mathbf{r}_{i} - \mathbf{r}_{j} )} 
    \sqrt{S_{\mu} S_{\nu}} 
    e^{ i \mathbf{k} \cdot (\mathbf{\delta}_{\mu} - \mathbf{\delta}_{\nu}) }
    \big\langle \Psi_{\mathbf{k}, \mu}^{\phantom \dagger}(t) \Psi_{\mathbf{k}, \nu}^{\dagger}(0) \big\rangle
    = \frac{1}{N} \sum_{\mathbf{k}} 
    e^{ + i \mathbf{k} \cdot (\mathbf{r}_{i} - \mathbf{r}_{j} )} 
    \bar{\mathsf{S}}_{\mathbf{k}}^{\phantom \dagger} 
    \mathsf{T}_{\mathbf{k}}^{\phantom \dagger}
    \mathsf{N}_{\mathbf{k}}^{\phantom \dagger} (t)
    \mathsf{T}_{\mathbf{k}}^{\dagger} 
    \mathsf{S}_{\mathbf{k}}^{\phantom \dagger},
\end{equation}
$$ derived by applying the Fourier transform to the
bosonic operators and computing the correlation matrix elements:
$$
\begin{align}
    \frac{1}{N} \sum_{i,j} e^{ - i \mathbf{k} \cdot (\mathbf{r}_{i} - \mathbf{r}_{j} )} 
    \big\langle \Psi_{i\mu}^{\phantom \dagger}(t) \Psi_{j\nu}^{\dagger}(0) \big\rangle 
    = & 
    \begin{pmatrix}
        \big\langle \hat{b}_{+\mathbf{k}, \mu}^{\phantom \dagger} (t) \hat{b}_{+\mathbf{k}, \nu}^{\dagger}          (0) \big\rangle &
        \big\langle \hat{b}_{+\mathbf{k}, \mu}^{\phantom \dagger} (t) \hat{b}_{-\mathbf{k}, \nu}^{\phantom \dagger} (0) \big\rangle \\
        \big\langle \hat{b}_{-\mathbf{k}, \mu}^{\dagger}          (t) \hat{b}_{+\mathbf{k}, \nu}^{\dagger}          (0) \big\rangle &
        \big\langle \hat{b}_{-\mathbf{k}, \mu}^{\phantom \dagger} (t) \hat{b}_{-\mathbf{k}, \nu}^{\dagger}          (0) \big\rangle 
    \end{pmatrix}
    = \mathsf{T}_{\mathbf{k}}^{\phantom\dagger} \mathsf{N}_{\mathbf{k}}^{\phantom\dagger} (t) \mathsf{T}_{\mathbf{k}}^{\dagger},
\end{align}
$$ where $\mathsf{T}_{\mathbf{k}}$ and $\mathsf{N}_{\mathbf{k}}(t)$ are
defined above. The second term, representing the density operator in
momentum space, is $$
\begin{equation}
    \frac{1}{N} \sum_{i,j} e^{- i \mathbf{k} \cdot (\mathbf{r}_{i} - \mathbf{r}_{j})} \langle \hat{n}_{j\nu} \rangle
    = \Big( \sum_{\kappa \in \mathbf{G}} \delta(\mathbf{k} - \kappa) \Big) \langle \hat{n}_{\mathbf{k}, \mu} \rangle,
\end{equation}
$$ where $\hat{n}_{\mathbf{k}, \mu}$, the density fluctuation
operator at momentum $\mathbf{k}$, is $$
\begin{align*}
    \hat{n}_{\mathbf{k}} 
    = & \frac{1}{N} 
    \sum_{i} e^{i \mathbf{k} \cdot \mathbf{r}_{i}} \hat{a}^{\dagger}_{i} \hat{a}_{i}^{\phantom \dagger} 
    = \frac{1}{N}  \sum_{\mathbf{q}} \hat{a}_{\mathbf{q} + \mathbf{k}}^{\dagger} \hat{a}_{\mathbf{q}}^{\phantom \dagger},
\end{align}
$$ which vanishes for a free system unless
$\mathbf{k} = \kappa \in \mathbf{G}$ due to momentum conservation.

The correlation function is expressed in matrix form as
$$
\begin{equation}
    \mathcal{C}^{\alpha\beta}_{\mu\nu} (\mathbf{k}, t)
    = e^{ - i \mathbf{k} \cdot (\mathbf{\delta}_{\mu} - \mathbf{\delta}_{\nu})}  
    \left[
    \sqrt{S_{\mu}S_{\nu}} \sum_{m,n}
    \Big[ \mathsf{U}^{\alpha}_{\mu}
    \mathsf{T}_{\mathbf{k}}^{\phantom\dagger} \mathsf{N}_{\mathbf{k}}^{\phantom\dagger} (t) \mathsf{T}_{\mathbf{k}}^{\dagger}
    \mathsf{U}^{\beta}_{\nu} \Big]_{m,n}
    + v_{\mu}^{\alpha} v_{\nu}^{\beta}
    \Big( \sum_{\kappa \in \mathbf{G}} \delta(\mathbf{k} - \kappa) \Big)
    \Big( S_{\nu}\langle \hat{n}_{\mathbf{0}, \mu} \rangle + S_{\mu}\langle \hat{n}_{\mathbf{0}, \nu} \rangle \Big)
    \right],
\end{equation}
$$ where the first term describes time-dependent magnon
scattering, and the second term accounts for the static ordered moment
reduction due to magnon population. The matrix form is
$$
\begin{equation}
\label{eq: matrix form of spin-spin correlation matrix}
    \mathsf{C}^{\alpha\beta} (\mathbf{k}, t)
    = 
    \mathsf{U}^{\alpha} 
    \mathsf{S}_{\mathbf{k}}^{\phantom \dagger} 
    \mathsf{T}_{\mathbf{k}}^{\phantom \dagger} 
    \mathsf{N}_{\mathbf{k}}^{\phantom \dagger}  (t)
    \mathsf{T}_{\mathbf{k}}^{         \dagger}
    \mathsf{S}_{\mathbf{k}}^{         \dagger}
    \mathsf{U}^{\beta},
\end{equation}
$$ with elements computed as $$
\begin{align*}
    \mathsf{C}^{\alpha\beta} (\mathbf{k}, t) = & 
    \begin{pmatrix}
        \bar{\mathsf{u}}^{\alpha} \mathsf{s}_{\mathbf{k}}  & \mathbf{0} \\
        \mathbf{0} & \mathsf{u}^{\alpha} \mathsf{s}_{\mathbf{k}} \\
    \end{pmatrix}
    \begin{pmatrix}     
        \langle \mathbf{b}_{ \mathbf{k}\mu}           (t)    \mathbf{b}_{ \mathbf{k}\nu}^{\dagger} (0)    \rangle & 
        \langle \mathbf{b}_{ \mathbf{k}\mu}           (t)    \mathbf{b}_{-\mathbf{k}\nu}           (0)    \rangle \\
        \langle \mathbf{b}_{-\mathbf{k}\mu}^{\dagger} (t)    \mathbf{b}_{ \mathbf{k}\nu}^{\dagger} (0)    \rangle & 
        \langle \mathbf{b}_{-\mathbf{k}\mu}^{\dagger} (t)    \mathbf{b}_{-\mathbf{k}\nu}           (0)    \rangle
    \end{pmatrix}
    \begin{pmatrix}
        \bar{\mathsf{s}}_{\mathbf{k}} \mathsf{u}^{\beta} & \mathbf{0} \\
        \mathbf{0} & \bar{\mathsf{s}}_{\mathbf{k}} \bar{\mathsf{u}}^{\beta} \\
    \end{pmatrix}.
\end{align}
$$ The elements of $\mathsf{C}^{\alpha\beta}(\mathbf{k}, t)$ for
indices $(m,n)_{\mu,\nu}$ are $$
\begin{align*}
    \begin{array}{crl}
        {[\mathsf{C}^{\alpha\beta}(\mathbf{k}, t)]_{(1,1)_{\mu\nu}}} : \qquad\qquad
        &
        \big[
            \mathsf{s}_{\mathbf{k}} \bar{\mathsf{u}}^{\alpha} 
            \langle \mathbf{b}_{+\mathbf{k}\mu}           (t)    \mathbf{b}_{+\mathbf{k}\nu}^{\dagger} (0)    \rangle 
            \mathsf{u}^{\beta} \bar{\mathsf{s}}_{\mathbf{k}} 
        \big]_{\mu\nu}
        = &
        \sqrt{S_{\mu}S_{\nu}}e^{-i \mathbf{k} \cdot (\bm{\delta}_{\mu} - \bm{\delta}_{\nu})} 
        \big\langle b_{+\mathbf{k}\mu} (t) b_{+\mathbf{k}\nu}^{\dagger} (0) \big\rangle,
        \\
        {[\mathsf{C}^{\alpha\beta}(\mathbf{k}, t)]_{(1,2)_{\mu\nu}}} : \qquad\qquad
        &
        \big[
            \mathsf{s}_{\mathbf{k}} \bar{\mathsf{u}}^{\alpha} 
            \langle \mathbf{b}_{+\mathbf{k}\mu}           (t)    \mathbf{b}_{-\mathbf{k}\nu}           (0)    \rangle 
            \bar{\mathsf{u}}^{\beta} \bar{\mathsf{s}}_{\mathbf{k}}
        \big]_{\mu\nu}
        = & \sqrt{S_{\mu}S_{\nu}}e^{-i \mathbf{k} \cdot (\bm{\delta}_{\mu} - \bm{\delta}_{\nu})} 
        \big\langle b_{+\mathbf{k}\mu} (t) b_{-\mathbf{k}\nu} (0) \big\rangle,
        \\
        {[\mathsf{C}^{\alpha\beta}(\mathbf{k}, t)]_{(2,1)_{\mu\nu}}} : \qquad\qquad
        & 
        \big[
            \mathsf{s}_{\mathbf{k}} \mathsf{u}^{\alpha}
            \langle \mathbf{b}_{-\mathbf{k}\mu}^{\dagger} (t)    \mathbf{b}_{+\mathbf{k}\nu}^{\dagger} (0)    \rangle  
            \mathsf{u}^{\beta} \bar{\mathsf{s}}_{\mathbf{k}}  
        \big]_{\mu\nu}
        = & \sqrt{S_{\mu}S_{\nu}}e^{-i \mathbf{k} \cdot (\bm{\delta}_{\mu} - \bm{\delta}_{\nu})} 
        \big\langle b_{-\mathbf{k}\mu}^{\dagger} (t) b_{+\mathbf{k}\nu}^{\dagger} (0) \big\rangle,
        \\
        {[\mathsf{C}^{\alpha\beta}(\mathbf{k}, t)]_{(2,2)_{\mu\nu}}} : \qquad\qquad
        &
        \big[
            \mathsf{s}_{\mathbf{k}} \mathsf{u}^{\alpha}
            \langle \mathbf{b}_{-\mathbf{k}\mu}^{\dagger} (t)    \mathbf{b}_{-\mathbf{k}\nu}           (0)    \rangle
            \bar{\mathsf{u}}^{\beta} \bar{\mathsf{s}}_{\mathbf{k}}
        \big]_{\mu\nu}
        = & \sqrt{S_{\mu}S_{\nu}}e^{-i \mathbf{k} \cdot (\bm{\delta}_{\mu} - \bm{\delta}_{\nu})} 
        \big\langle b_{-\mathbf{k}\mu}^{\dagger} (t) b_{-\mathbf{k}\nu} (0) \big\rangle.
    \end{array}
\end{align}
$$ Summing $\mathsf{C}^{\alpha\beta}(\mathbf{k}, t)$ over
$(m,n)_{\mu,\nu}$ yields the sublattice spin-spin correlation function
$$
\begin{equation}
    \mathcal{C}_{\mu\nu}^{\alpha\beta} (\mathbf{k}, t) = \sum_{(m,n)_{\mu,\nu}} [\mathsf{C}^{\alpha\beta}(\mathbf{k}, t)]_{m,n}.
\end{equation}
$$ The spin structure factor is obtained by summing over
all sublattices:
$\mathcal{S}^{\alpha\beta} (\mathbf{k}, t) = \mathcal{C}^{\alpha\beta} (\mathbf{k}, t) = \frac{1}{m_s} \sum_{\mu,\nu} \mathcal{C}_{\mu\nu}^{\alpha\beta} (\mathbf{k}, t)$.

## Structure Factor {#subsec: Structure Factor}

In this section, we derive the formulas for the static and dynamic
structure factors in Linear Spin-Wave Theory.

### Static Structure Factor

The static structure factor in momentum space is given by
$$
\begin{equation}
    \mathcal{S}^{\alpha\beta} (\mathbf{k}) 
    \equiv \frac{1}{L} 
    \sum_{I, J}
    e^{- i \mathbf{k} \cdot ( \mathbf{r}_{I} -  \mathbf{r}_{J} ) }
    \langle 
    S_{I}^{\alpha}
    S_{J}^{\beta} 
    \rangle 
    = \big\langle S^{\alpha}_{\mathbf{k}} S^{\beta}_{-\mathbf{k}} \big\rangle.
\end{equation}
$$ This quantity $\mathcal{S}^{\alpha\beta} (\mathbf{k})$
is equal to $\mathcal{C}^{\alpha\beta}(\mathbf{k}, t)$, which we've found the
matrix form in a previous equation: $$
\begin{equation}
    \mathcal{S}^{\alpha\beta} (\mathbf{k}) = 
    \sum_{m,n} 
    \big[ \mathsf{U}^{\alpha} 
    \mathsf{S}_{\mathbf{k}}^{\phantom \dagger} 
    \mathsf{T}_{\mathbf{k}}^{\phantom \dagger} 
    \mathsf{N}_{\mathbf{k}}^{\phantom \dagger}
    \mathsf{T}_{\mathbf{k}}^{         \dagger}
    \mathsf{S}_{\mathbf{k}}^{         \dagger}
    \mathsf{U}^{\beta} \big]_{mn},
\end{equation}
$$ where $\mathsf{U}^{\alpha}$ and $\mathsf{U}^{\beta}$
are spin projection operators, $\mathsf{S}_{\mathbf{k}}$ and
$\mathsf{T}_{\mathbf{k}}$ are transformation matrices, and
$\mathsf{N}_{\mathbf{k}}$ accounts for the diagonalized magnon correlation
functions. In the reference frame, the structure factor is expressed as
$$
\begin{align*}
    \widetilde{\mathcal{S}}^{\alpha\beta} (\mathbf{k}) 
    \equiv &
    \frac{1}{L} 
    \sum_{I, J}
    e^{- i \mathbf{k} \cdot ( \mathbf{r}_{I} -  \mathbf{r}_{J} ) }
    \langle \widetilde{S}_{I}^{\alpha} \widetilde{S}_{J}^{\beta} \rangle 
    = 
    \sum_{\mu, \nu} 
    \left[
    \begin{pmatrix}
        \langle \widetilde{S}^{+}_{\mathbf{k}} \widetilde{S}^{-}_{-\mathbf{k}} \rangle &
        \langle \widetilde{S}^{+}_{\mathbf{k}} \widetilde{S}^{+}_{-\mathbf{k}} \rangle \\[0.25em]
        \langle \widetilde{S}^{-}_{\mathbf{k}} \widetilde{S}^{-}_{-\mathbf{k}} \rangle &
        \langle \widetilde{S}^{-}_{\mathbf{k}} \widetilde{S}^{+}_{-\mathbf{k}} \rangle
    \end{pmatrix} 
    \right]_{(\alpha, \beta)_{\mu\nu}}
    \\ = &
    \sum_{\mu, \nu} 
    \big[
    \mathsf{S}_{\mathbf{k}}^{\phantom \dagger} 
    \mathsf{T}_{\mathbf{k}}^{\phantom \dagger} 
    \mathsf{N}_{\mathbf{k}}^{\phantom \dagger}  
    \mathsf{T}_{\mathbf{k}}^{         \dagger}
    \mathsf{S}_{\mathbf{k}}^{         \dagger}
    \big]_{(\alpha, \beta)_{\mu\nu}},
\end{align}
$$ where the matrix elements are evaluated in the rotated
spin basis.

The quantity relevant to neutron scattering experiments is defined as:
$$
\begin{equation}
    \mathcal{S} (\mathbf{q}) 
    \equiv 
    \frac{1}{N} 
    \sum_{\substack{\alpha, \beta \\ I, J}}
    \Big(
    \delta_{\alpha\beta} - \frac{q_{\alpha} q_{\beta}}{q^{2}} 
    \Big)
    e^{i \mathbf{q} \cdot ( \mathbf{r}_{I} - \mathbf{r}_{J} ) }
    \langle 
    S_{i}^{\alpha}
    S_{j}^{\beta} 
    \rangle
    = 
    \sum_{\alpha, \beta} 
    \Big( \delta_{\alpha\beta} - \frac{q_{\alpha} q_{\beta}}{q^{2}} \Big)
    \mathcal{S}^{\alpha\beta} (\mathbf{q}) ,
\end{equation}
$$ This formulation respects the directional constraints
intrinsic to neutron scattering physics.

### Dynamic structure factor

The dynamic structure factor is defined as $$
\begin{equation}
    \mathcal{S}^{\alpha\beta}(\mathbf{k}, \omega) 
    = \frac{1}{2\pi} \int_{-\infty}^{\infty} d t \, e^{i \omega t} \mathcal{C}^{\alpha\beta} (\mathbf{k}, t).
\end{equation}
$$ From the LSWT, we consider $$
\begin{align*}
    \mathcal{S}^{\alpha\beta}_{\mu\nu} (\mathbf{k}, \omega) 
    = & \sum_{(m,n)_{\mu,\nu}} \int_{-\infty}^{\infty} dt \, [\mathsf{C}^{\alpha\beta} (\mathbf{k}, t)]_{m,n} e^{i \omega t}
    = \sum_{(m,n)_{\mu,\nu}} 
    \left[ 
    \int_{-\infty}^{\infty} dt \, e^{i \omega t} 
    \mathsf{U}^{\alpha} 
    \mathsf{S}_{\mathbf{k}}^{\phantom \dagger} 
    \mathsf{T}_{\mathbf{k}}^{\phantom \dagger} 
    \mathsf{N}_{\mathbf{k}}^{\phantom \dagger}  (t)
    \mathsf{T}_{\mathbf{k}}^{         \dagger}
    \mathsf{S}_{\mathbf{k}}^{         \dagger}
    \mathsf{U}^{\beta}
    \right]_{m,n}  \\
    = & \sum_{(m,n)_{\mu,\nu}} 
    \left[  \mathsf{R}^{\alpha}_{\mathbf{k}} \mathsf{T}_{\mathbf{k}}
    \left( \int_{-\infty}^{\infty} dt \, e^{i \omega t} \mathsf{N}_{\mathbf{k}} (t) \right)
    \mathsf{T}_{\mathbf{k}}^{\dagger} [\mathsf{R}^{\beta}_{\mathbf{k}}]^{\dagger} \right]_{m,n}.
\end{align}
$$ However, the integral yields $\delta$-function. So we
introduce the decay factor $\eta > 0$ so that
$\omega \rightarrow \omega + i \eta$: $$
\begin{equation}
    g_{\mathbf{k}} ( \omega ; \eta ) 
    \equiv \int_{0}^{\infty} dt \, e^{ - [ \eta  + i( E_{\mathbf{k}} - \omega ) ] t} 
    = \frac{1}{\eta + i ( E_{\mathbf{k}} - \omega )}
\end{equation}
$$ Using this notation, we have $$
\begin{align}
    \mathcal{F}_{\mathbf{k}} (\omega; \eta) = & \int_{-\infty}^{\infty} dt\, e^{-i E_{\mathbf{k}} t} e^{ i\omega t - \eta |t|} 
    =  \int_{0}^{\infty} dt \, 
    \big( e^{ - [ \eta  + i( E_{\mathbf{k}} - \omega ) ] t} + e^{ - [ \eta  - i( E_{\mathbf{k}} - \omega ) ] t} \big) 
    =  g_{\mathbf{k}} ( \omega ; \eta ) + g_{\mathbf{k}}^{*} ( \omega ; \eta ) \\
    = & 2 \mathrm{Re} [ g_{\mathbf{k}} ( \omega ; \eta ) ]
    = \frac{2\eta}{\eta^{2} + ( E_{\mathbf{k}} - \omega )^{2}}
\end{align}
$$ where $\mathcal{F}_{\mathbf{k}}(\omega; \eta)$ is Lorentzian
and its FWHM is $2\eta$. $$
\begin{equation}
    \begin{split}
        \int_{-\infty}^{\infty} dt \, e^{i \omega t - \eta |t|} \mathsf{N}_{\mathbf{k}} (t) 
        = &
        \int_{-\infty}^{\infty} dt \, e^{i \omega t  - \eta |t|}
        \begin{pmatrix}
            ( \mathsf{I} + \mathsf{n}_{\mathbf{k}} ) e^{-{ i E}_{\mathbf{k},\mu}t}  & 0 \\
            0                         & \mathsf{n}_{-\mathbf{k}}  e^{+{  i  E}_{-\mathbf{k},\mu}t}
        \end{pmatrix} \\
        = &
        \begin{pmatrix}
            ( \mathsf{I} + \mathsf{n}_{\mathbf{k}} ) 
            \int_{-\infty}^{\infty} dt \, e^{i \omega t  - \eta |t|} 
            e^{-{ i E}_{\mathbf{k},\mu}t}  & 0 \\
            0                         & \mathsf{n}_{-\mathbf{k}}  
            \int_{-\infty}^{\infty} dt \, e^{i \omega t  - \eta |t|}  e^{+{  i  E}_{-\mathbf{k},\mu}t}
        \end{pmatrix} \\
        = & 
        \begin{pmatrix}
            ( \mathsf{I} + \mathsf{n}_{\mathbf{k}} ) \mathcal{F}_{\mathbf{k}} (\omega; \eta)  & 0 \\
            0   &   \mathsf{n}_{-\mathbf{k}} \mathcal{F}_{- \mathbf{k}} (-\omega; \eta)
        \end{pmatrix} 
    \end{split}
\end{equation}
$$ Thus, in LSWT, the dynamical spin structure factor can
be computed from the matrix multiplication: $$
\begin{equation}
    \mathcal{S}^{\alpha\beta}_{\mu\nu} (\mathbf{k}, \omega)
    =
    \sum_{(m,n)_{\mu,\nu}} 
    \left[  \mathsf{R}^{\alpha}_{\mathbf{k}} \mathsf{T}_{\mathbf{k}}
    \mathcal{F} [\mathsf{N}_{\mathbf{k}}](\omega; \eta)
    \mathsf{T}_{\mathbf{k}}^{\dagger} [\mathsf{R}^{\beta}_{\mathbf{k}}]^{\dagger} \right]_{m,n}.
\end{equation}
$$ Here, the fourier transformation of diagonal matrix
$\mathsf{N}_{\mathbf{k}} (t)$ is given by $$
\begin{equation}
    \mathcal{F} [\mathsf{N}_{\mathbf{k}}](\omega; \eta)
    \equiv 
    \begin{pmatrix}
        ( \mathsf{I} + \mathsf{n}_{\mathbf{k}} )    \mathcal{F}_{ \mathbf{k}} (  \omega; \eta  )  & 0 \\
        0   &   \mathsf{n}_{-\mathbf{k}} \mathcal{F}_{-\mathbf{k}} ( -\omega; \eta  )
    \end{pmatrix} 
\end{equation}
$$

## Spectral Function {#subsec: Spectral Function}

In this section, we compute the spectral function within Linear
Spin-Wave Theory (LSWT): $$
\begin{equation}
    \mathcal{A}^{\alpha\beta}(\mathbf{k}, \omega) = - \frac{1}{\pi} \mathrm{Im} [G^{\alpha\beta}_{\rm R} (\mathbf{k}, \omega)],
\end{equation}
$$ where $G^{\alpha\beta}_{\rm R} (\mathbf{k}, \omega)$ is the
retarded Green's function, defined as $$
\begin{equation}
    G^{\alpha\beta}_{\rm R} (\mathbf{k}, \omega) 
    \equiv  \frac{1}{i} \int_{-\infty}^{\infty} dt \, e^{i\omega t} \theta(t) \big\langle [S_{\mathbf{k}}^{\alpha} (t), S_{-\mathbf{k}}^{\beta} (0)] \big\rangle . 
\end{equation}
$$ The ratared Green's function can be rewritten in terms
of the correlation function $$
\begin{equation}
    G^{\alpha\beta}_{\rm R} (\mathbf{k}, \omega) 
    = \frac{1}{i} \int_{0}^{\infty} dt \, \Big[ \big\langle S_{\mathbf{k}}^{\alpha} (t) S_{-\mathbf{k}}^{\beta} (0) \big\rangle - \big\langle S_{-\mathbf{k}}^{\beta} (0) S_{\mathbf{k}}^{\alpha} (t) \big\rangle \Big] e^{i \omega t} 
    = \frac{1}{i} \int_{0}^{\infty} dt \, \big[ \mathcal{C}^{\alpha\beta}(\mathbf{k}, t) - \mathcal{C}^{\beta\alpha}(-\mathbf{k}, -t) \big] e^{i \omega t}.
\end{equation}
$$ For spin components $\alpha, \beta = x, y, z$, the
retarded Green's function is expressed as the Fourier transform of the
imaginary part of the correlation function: $$
\begin{equation}
    G^{\alpha\beta}_{\rm R} (\mathbf{k}, \omega) 
    = 2 \int_{0}^{\infty} dt \, \mathrm{Im} [\mathcal{C}^{\alpha\beta}(\mathbf{k}, t) ] e^{\ri \omega t}.
\end{equation}
$$ The spectral function is then obtained as
$$
\begin{equation}
    \begin{split}
        \mathcal{A}^{\alpha\beta}(\mathbf{k}, \omega) 
        % = & - \frac{1}{\pi} \mathrm{Im} [G^{\alpha\beta}_{\rm R} (\mathbf{k}, \omega)] \\
        = & - \frac{2}{\pi} \int_{0}^{\infty} dt \, \mathrm{Im} \big[ \mathrm{Im} \big[ \mathcal{C}^{\alpha\beta}(\mathbf{k}, t) \big] e^{i \omega t} \big] \\
        = & - \frac{2}{\pi} \int_{0}^{\infty} dt \, \mathrm{Im} \big[ \mathcal{C}^{\alpha\beta}(\mathbf{k}, t) \sin (\omega t) \big] \\
        = & - \frac{2}{\pi} \mathrm{Im} 
        \sum_{m,n}
        \left[ 
        \mathsf{U}^{\alpha} \mathsf{S}_{\mathbf{k}}^{\phantom \dagger} \mathsf{T}_{\mathbf{k}}^{\phantom \dagger}
        \left( \int_{0}^{\infty} dt \, 
        \begin{pmatrix}
            (\mathsf{I} + \mathsf{n}_{\mathbf{k}}) e^{- i \mathsf{E}_{\mathbf{k}} t} & 0 \\
            0 & \mathsf{n}_{-\mathbf{k}} e^{i \mathsf{E}_{-\mathbf{k}} t}
        \end{pmatrix}
        \sin (\omega t)
        \right)
        \mathsf{T}^{\dagger}_{\mathbf{k}} \bar{\mathsf{S}}_{\mathbf{k}}^{\phantom \dagger} \bar{\mathsf{U}}^{\beta}
        \right]_{mn} \\
        = & -  \frac{2}{\pi} \mathrm{Im} 
        \sum_{m,n}
        \left[ 
            \mathsf{U}^{\alpha} \mathsf{S}_{\mathbf{k}}^{\phantom \dagger} \mathsf{T}_{\mathbf{k}}^{\phantom \dagger}
            \begin{pmatrix}
                (\mathsf{I + n}_{  \mathbf{k}}) \mathcal{G}_{\mathbf{k}} (\omega ; \eta) & 0 \\
                0 &  \mathsf{n}_{- \mathbf{k}} \mathcal{G}_{-\mathbf{k}}^{*} (\omega ; \eta)
            \end{pmatrix}
            \mathsf{T}^{\dagger}_{\mathbf{k}} \bar{\mathsf{S}}_{\mathbf{k}}^{\phantom \dagger}
            \bar{\mathsf{U}}^{\beta}
        \right]_{m,n}
    \end{split}
\end{equation}
$$ where the function $\mathcal{G}_{\mathbf{k}}(\omega ; \eta)$
is defined as $$
\begin{align*}
     \mathcal{G}_{\mathbf{k}}(\omega ; \eta)
     \equiv & \int_{0}^{\infty} dt \,  e^{- \ri E_{\mathbf{k}} t} \sin (\omega t) e^{-\eta |t|} 
     = \int_{0}^{\infty} dt \,  e^{-\ri E_{\mathbf{k}} t} \bigg( \frac{e^{\ri \omega t} - e^{- \ri \omega t}}{2\ri} \bigg) e^{-\eta t} \\
     = & \frac{1}{2 \ri} 
     \left( 
     \int_{0}^{\infty} dt \,
     e^{-(\eta + \ri(E_{\mathbf{k}} - \omega))t} - \int_{0}^{\infty} dt \, e^{-(\eta + \ri(E_{\mathbf{k}} + \omega))t} 
     \right) 
     = \frac{1}{2 \ri} \Big( \frac{1}{\eta + \ri(E_{\mathbf{k}} - \omega)} - \frac{1}{\eta + \ri(E_{\mathbf{k}} + \omega)} \Big) \\
     = & \frac{1}{2i} \big( g_{\eta}(\mathbf{k}, \omega) + g_{\eta} (\mathbf{k}, - \omega) \big).
\end{align}
$$ Thus, the spectral function in LSWT can be computed from
the product of the matrices. $$
\begin{equation}
    \mathcal{A}^{\alpha\beta}(\mathbf{k}, \omega) 
    = - \frac{2}{\pi} \mathrm{Im} 
    \sum_{m,n}
    \left[ 
    \mathsf{U}^{\alpha} \mathsf{S}_{\mathbf{k}}^{\phantom \dagger} \mathsf{T}_{\mathbf{k}}^{\phantom \dagger}
    \mathcal{G} [ \mathsf{N}_{\mathbf{k}} ] (\omega ; \eta)
    \mathsf{T}^{\dagger}_{\mathbf{k}} \bar{\mathsf{S}}_{\mathbf{k}}^{\phantom \dagger} \bar{\mathsf{U}}^{\beta}
    \right]_{m,n},
\end{equation}
$$ where the matrix
$\mathcal{G} [ \mathsf{N}_{\mathbf{k}} ] (\omega ; \eta)$ is defined as
$$
\begin{equation}
    \mathcal{G} [ \mathsf{N}_{\mathbf{k}} ] (\omega ; \eta)
    = 
    \begin{pmatrix}
        (\mathsf{I} + \mathsf{n}_{\mathbf{k}}) \mathcal{G}_{\mathbf{k}} (\omega; \eta) & 0 \\
        0 & \mathsf{n}_{-\mathbf{k}} \mathcal{G}_{-\mathbf{k}}^{*} (\omega; \eta).
    \end{pmatrix}.
\end{equation}
$$
