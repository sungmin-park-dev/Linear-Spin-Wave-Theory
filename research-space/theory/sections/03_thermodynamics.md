# Thermodynamics in Linear Spin Wave Theory

## Partition function {#subsec: Partition Function}

In this section, we derive the partition function for linear spin wave
theory. Recall that the Hamiltonian for the spin system with normal
bosons is written as $$
\begin{align}
    \hat{H} = E_{\rm cl} + \sum_{\mathbf{k}, \mu} E_{\mathbf{k}, \mu} \Big( \beta^{\dagger}_{\mathbf{k}, \mu} \beta_{\mathbf{k}, \mu} + \frac{1}{2} \Big) - \frac{1}{2} \Tr [\mathsf{H}_{\mathbf{k}}] 
    = E_{0} + \sum_{\mathbf{k} \in \mathrm{FBZ}} \sum_{\mu = a, b, \cdots} E_{\mathbf{k}, \mu} \hat{n}_{\mathbf{k}, \mu},
\end{align}
$$ where
$E_{0} = E_{\rm cl} + \frac{1}{2} \sum_{\mathbf{k}} \big( \sum_{\mu} E_{\mathbf{k}, \mu} - \Tr [\mathsf{A}_{\mathbf{k}}] \big)$
represents the zero-point energy
\[Eqn. (Eq. \ref{eq: def. of zero point energy})\] and
$\hat{n}_{\mu}(\mathbf{k}) = \beta^{\dagger}_{\mathbf{k}, \mu} \beta_{\mathbf{k}, \mu}$
is the magnon number operator. Here, $E_{\mathbf{k}, \mu}$ are the magnon
energy eigenvalues with band index $\mu$.

The partition function is given by $$
\begin{equation}
    Z_{\beta} \equiv \Tr \big( e^{- \beta \hat{H}} \big) 
    = e^{- \beta E_{0}}  \Tr \big( e^{- \beta \sum_{\mathbf{k}, \mu} E_{\mathbf{k}, \mu} \beta^{\dagger}_{\mathbf{k}, \mu} \beta_{\mathbf{k}, \mu}} \big) 
    = e^{- \beta E_{0}} \Tr \big( e^{- \beta \sum_{\mathbf{k}, \mu} E_{\mathbf{k}, \mu} \hat{n}_{\mathbf{k}, \mu}} \big),
\end{equation}
$$ Here, the constant term $E_0$ contributes to the
internal energy but is not important for thermodynamic quantities that
involve derivatives of $\ln \tilde{Z}_{\beta}$, where
$\tilde{Z}_{\beta}$ represents the partition function without the
zero-point energy contribution.

The partition function $Z_{\beta}$ can be evaluated as
$$\label{eq: explicit expressions for Z and log-Z}
    \begin{align}
        \Tr \big( e^{- \beta \sum_{\mathbf{k}, \mu} E_{\mathbf{k}, \mu} \hat{n}_{\mu}(\mathbf{k})} \big) 
        = & \Tr \Big[ \prod_{\mathbf{k}, \mu} e^{- \beta E_{\mathbf{k}, \mu} \hat{n}_{\mathbf{k}, \mu}} \Big]
        = \prod_{\mathbf{k}, \mu} \Tr \Big[ e^{- \beta E_{\mathbf{k}, \mu} \hat{n}_{\mathbf{k}, \mu}} \Big] 
        =  \prod_{\mathbf{k}, \mu}  \frac{1}{1 - e^{- \beta E_{\mathbf{k}, \mu}}} \\
        - \ln Z_{\beta} &= \beta E_{0} + \sum_{\mathbf{k}, \mu} \ln \big( 1 - e^{- \beta E_{\mathbf{k}, \mu}} \big).
    \end{align}
$$ The final expression for $-\ln Z_{\beta}$ is
particularly useful for calculating thermodynamic quantities such as
free energy, internal energy, specific heat, and entropy in the linear
spin-wave approximation.

## Internal Energy {#subsec: Internal Energy}

The internal energy $U$ is the expectation value of the Hamiltonian
$\hat{H}$, computed as: $$
\begin{equation}
    U \equiv \langle \hat{H} \rangle  = -\frac{\partial \ln Z_{\beta}}{\partial \beta},
\end{equation}
$$ where
$Z_{\beta} = e^{- \beta E_{0}} \tilde{Z}_{\beta}$, with $E_0$ being the
zero-point energy and $\tilde{Z}_{\beta}$ the partition function
excluding the zero-point energy contribution.

Computing the derivative, we get: $$
\begin{equation}
     - \frac{\partial \ln Z_{\beta}}{\partial \beta} 
    = E_{0} + \frac{\partial}{\partial \beta} \Big[  \sum_{\mathbf{k}, \mu} \ln \big( 1 - e^{- \beta E_{\mathbf{k}, \mu}} \big) \Big]
    = E_{0} + \sum_{\mathbf{k}, \mu} \frac{E_{\mathbf{k}, \mu} e^{-\beta E_{\mathbf{k}, \mu}}}{1 - e^{-\beta E_{\mathbf{k}, \mu}}}  
    = E_{0} + \sum_{\mathbf{k}, \mu} E_{\mathbf{k}, \mu} \frac{1}{e^{\beta E_{\mathbf{k}, \mu}} - 1},
\end{equation}
$$ where we identify the Bose-Einstein distribution
function: $$
\begin{equation}
    n_{\mu} (\mathbf{k}) = \frac{1}{e^{\beta E_{\mathbf{k}, \mu}} - 1},
\end{equation}
$$ which represents the thermal expectation value of the
magnon number operator
$\langle\hat{n}_{\mathbf{k}, \mu}\rangle = \langle \hat{\beta}^{\dagger}_{\mathbf{k}, \mu} \hat{\beta}_{\mathbf{k},\mu} \rangle$.
Therefore, the internal energy can be expressed as: $$
\begin{align}
    U &= E_0 + \sum_{\mathbf{k}, \mu} E_{\mathbf{k}, \mu} n_{\mu} (\mathbf{k}) \nonumber \\
    &= E_{\rm cl} + \frac{1}{2} \sum_{\mathbf{k}} \big( \sum_{\mu} E_{\mathbf{k}, \mu} - \Tr [\mathsf{A}_{\mathbf{k}}] \big) + \sum_{\mathbf{k}, \mu} E_{\mathbf{k}, \mu} n_{\mu}(\mathbf{k})
\end{align}
$$ This result expresses the internal energy as the sum of
the classical ground state energy, the zero-point quantum fluctuations,
and the thermal contribution from magnon excitations.

## Free Energy {#subsec: Free Energy}

The Helmholtz free energy ($F$) is defined as $$
\begin{equation}
\label{eq: def. of Helmholtz free energy}
    Z_{\beta} \equiv e^{-\beta F}
\end{equation}
$$ where $Z_{\beta} = e^{- \beta E_0} \tilde{Z}_{\beta}$
is the partition function, with $\tilde{Z}_{\beta}$ being the partition
function excluding the zero-point energy contribution. From
(Eq. \ref{eq: explicit expressions for Z and log-Z}) and
Eq. (Eq. \ref{eq: def. of Helmholtz free energy}), we get: $$
\begin{align}
    F &= -\frac{1}{\beta} \ln(e^{- \beta E_0} \tilde{Z}_{\beta}) \nonumber \\
    &= E_0 - \frac{1}{\beta} \ln \tilde{Z}_{\beta} \nonumber \\
    &= E_{\rm cl} + \frac{1}{2} \sum_{\mathbf{k}} \big( \sum_{\mu} E_{\mathbf{k}, \mu} - \Tr [\mathsf{A}_{\mathbf{k}}] \big) + \frac{1}{\beta} \sum_{\mathbf{k}, \mu} \ln \left( 1 - e^{-\beta E_{\mathbf{k}, \mu}} \right),
\end{align}
$$ where we use
$-\ln \tilde{Z}_{\beta} = \sum_{\mathbf{k}, \mu} \ln \left( 1 - e^{-\beta E_{\mathbf{k}, \mu}} \right)$
from the partition function section, and
$E_0 = E_{\rm cl} + \frac{1}{2} \sum_{\mathbf{k}} \big( \sum_{\mu} E_{\mathbf{k}, \mu} - \Tr [\mathsf{A}_{\mathbf{k}}] \big)$
is the zero-point energy.

## Entropy Expression {#subsec: Entropy}

In this section, we derive the expression for the thermal entropy in
LSWT. The entropy is defined as $$
\begin{equation}
    S \equiv -k_{\rm B}\Tr \big( \rho \ln \rho \big),
\end{equation}
$$ where $\rho$ is the density operator, which is Gibbs
states ($\rho = e^{-\beta \hat{H}}/Z_{\beta}$) especially in thermal
equilibrium. This definition coincides with the conventional definition
of entropy in thermal physics: $$
\begin{align}
    - \frac{\partial F}{\partial T}
    = \frac{1}{T} \Big(-\frac{\partial}{\partial \beta} \ln Z_{\beta} \Big) - \big(-k_{\rm B} \ln Z_{\beta} \big)
    = \frac{U - F}{T}.
\end{align}
$$ We can derive the same expression using Gibbs states:
$$
\begin{align}
    S = & -k_{\rm B}\Tr \big( \rho \ln \rho \big)
    = -k_{\rm B}\Tr \big( \rho (-\beta \hat{H} - \ln Z_{\beta}) \big) \nonumber \\
    = &  \frac{ \Tr (\rho \hat{H}) - (-k_{\rm B}T\ln Z_{\beta})}{T} 
    = \frac{U - F}{T}.
\end{align}
$$ Using our previous expressions for $U$ and $F$, we have:
$$
\begin{align}
    U - F &= E_0 + \sum_{\mathbf{k}, \mu} E_{\mathbf{k}, \mu} n_{\mu}(\mathbf{k}) - \left( E_0 + \frac{1}{\beta} \sum_{\mathbf{k}, \mu} \ln \left( 1 - e^{-\beta E_{\mathbf{k}, \mu}} \right) \right) \nonumber \\ 
    &= \sum_{\mathbf{k}, \mu} E_{\mathbf{k}, \mu} n_{\mu}(\mathbf{k}) - \frac{1}{\beta} \sum_{\mathbf{k}, \mu} \ln \left( 1 - e^{-\beta E_{\mathbf{k}, \mu}} \right).
\end{align}
$$

Since $S = k_{\rm B} \beta (U - F)$ (noting $\beta = \frac{1}{k_B T}$
and adjusting units appropriately): $$
\begin{equation}
    S = k_{\rm B} \sum_{\mathbf{k}, \mu} \beta E_{\mathbf{k}, \mu} n_{\mu} (\mathbf{k}) - k_{\rm B} \sum_{\mathbf{k}, \mu} \ln \left( 1 - e^{-\beta E_{\mathbf{k}, \mu}} \right).
\end{equation}
$$

Now, we can rewrite this expression using the properties of the
Bose-Einstein distribution function $n_{\mathbf{k}, \mu}$. From the
definition $n_{\mathbf{k}, \mu} = \frac{1}{e^{\beta E_{\mathbf{k}, \mu}} - 1}$, we
can derive: $$
\begin{equation}
    \beta E_{\mathbf{k}, \mu} = \ln \Big( \frac{1 + n_{\mu}(\mathbf{k})}{n_{\mu}(\mathbf{k})} \Big), 
    \quad \text{and} \quad
    \ln \left( 1 - e^{-\beta E_{\mathbf{k}, \mu}} \right) 
    = -\ln \big( 1 + n_{\mu} (\mathbf{k}) \big), 
\end{equation}
$$ Substituting these relations into our entropy
expression, we obtain: $$
\begin{equation}
    S = k_{\rm B} \sum_{\mathbf{k}, \mu} \big[ (1 + n_{\mu}(\mathbf{k})) \ln (1 + n_{\mu}(\mathbf{k})) - n_{\mu}(\mathbf{k}) \ln n_{\mu} (\mathbf{k}) \big]
\end{equation}
$$ This is the standard entropy formula for a system of
non-interacting bosons, which emerges naturally from our linear spin
wave theory.

## Specific Heat {#subsec: Specific Heat}

The specific heat $C$ is defined as the temperature derivative of the
internal energy: $$
\begin{equation}
    C = \frac{\partial U}{\partial T} = \sum_{\mathbf{k}, \mu} E_{\mathbf{k}, \mu} \frac{\partial n_{\mu}(\mathbf{k})}{\partial T},
\end{equation}
$$ where $n_{\mu}(\mathbf{k})$ is the Bose-Einstein
distribution for magnons $\mu$. The derivative of the distribution with
respect to temperature is $$
\begin{equation}
    \frac{\partial n_{\mu}(\mathbf{k})}{\partial T} 
    =  \frac{\partial n_{\mu}(\mathbf{k})}{\partial \beta} \frac{\partial \beta}{\partial T} \\
    =  \Big( - \frac{E_{\mathbf{k}, \mu} e^{\beta E_{\mathbf{k}, \mu}}}{\left( e^{\beta E_{\mathbf{k}, \mu}} - 1 \right)^2} \Big) \Big( - \frac{\beta}{T}\Big) 
    =  \frac{\beta}{T} \frac{E_{\mathbf{k}, \mu} e^{\beta E_{\mathbf{k}, \mu}}}{\left( e^{\beta E_{\mathbf{k}, \mu}} - 1 \right)^2}.
\end{equation}
$$ Therefore, the specific heat is given by:
$$
\begin{equation}
    \begin{split}
        C = & \sum_{\mathbf{k}, \mu} E_{\mathbf{k}, \mu} \frac{\partial n_{\mu}(\mathbf{k})}{\partial T} 
        = \frac{\beta}{T} \sum_{\mathbf{k}, \mu} \frac{E_{\mathbf{k}, \mu}^2 e^{\beta E_{\mathbf{k}, \mu}}}{\left( e^{\beta E_{\mathbf{k}, \mu}} - 1 \right)^2}
        = k_{\rm B} \sum_{\mathbf{k}, \mu} \frac{(\beta E_{\mathbf{k}, \mu})^2 e^{\beta E_{\mathbf{k}, \mu}}}{\left( e^{\beta E_{\mathbf{k}, \mu}} - 1 \right)^2}
        =  k_{\rm B} \sum_{\mathbf{k}, \mu} \frac{ (\beta E_{\mathbf{k}, \mu})^2 }{\left( e^{\beta E_{\mathbf{k}, \mu}/2} - e^{-\beta E_{\mathbf{k}, \mu}/2} \right)^2} \\
        = &  k_{\rm B} \sum_{\mathbf{k}, \mu} \frac{ (\beta E_{\mathbf{k}, \mu})^2 }{4 \sinh^{2} (\beta E_{\mathbf{k}, \mu}/2 )}
    \end{split}
\end{equation}
$$ This expression represents the specific heat
contribution from magnon excitations in the system, which dominates the
low-temperature thermal properties of magnetic insulators.

## Number and Spin moment from Correlation Matrix {#subsec: Number expectation value/Spin moment}

Finally, we discuss the number expectation and spin moment. Linear
spin-wave theory gives the leading correction of the sublattice
magnetization. The spin moments are $$
\begin{equation}
    \langle \hat{\bf S}_{\mu} \rangle 
    \equiv \frac{1}{L} \sum_{i} \langle \hat{\bf S}_{i, \mu} \rangle
\end{equation}
$$ where the spin moment at site $i=(i,\mu)$ can be
obtained from
$\langle \hat{\bf S}_{i, \mu} \rangle = \mathbf{R}_{i,\mu} \langle \widetilde{\bf S}_{i,\mu} \rangle$
and $\widetilde{\bf S}_{i,\mu} = (0, 0, \widetilde{S}^{0}_{i,\mu})$.
This is the spin moment in the reference frame. The average sublattice
magnetization along the magnetization axis $\widetilde{S}^{0}_{i,\mu}$
is given $$
\begin{align*}
    \frac{1}{L} \sum_{i} \langle \widetilde{S}^{0}_{i, \mu} \rangle 
    = & \frac{1}{L} \sum_{i} \langle S_{\mu} -  \hat{n}_{i, \mu} \rangle 
    = S_{\mu} - \frac{1}{L} \sum_{\mathbf{k} \in \mathrm{FBZ}}  \langle \hat{n}_{\mathbf{k}, \mu}  \rangle
    = S_{\mu} - n_{\mu},
\end{align}
$$ where $i$ runs over the lattice, $\mu$ denotes the
sublattice index, and $S_{\mu}$ is the spin moment of the spin. For
example, for the spin 1/2 system, $S_\mu = 1/2$. $$
\begin{align}
    n_{\mu} \equiv & \frac{1}{L} \sum_{\mathbf{k}}  \langle \hat{n}_{\mathbf{k}, \mu} \rangle 
    = \frac{1}{L} \sum_{\mathbf{k} \in \mathrm{FBZ}} 
    = \langle b^{\dagger}_{\mathbf{k}, \mu} b_{\mathbf{k}, \mu}^{\phantom \dagger} \rangle  
\end{align}
$$

To proceed, it is useful to introduce the correlation matrix. The
correlation matrix is defined as the expectation of outer product of the
BdG vector $\Psi_{\mathbf{k}}$. $$
\begin{equation}
    \langle b^{\dagger}_{\mathbf{k}, \mu} b_{\mathbf{k}, \mu}^{\phantom \dagger} \rangle  
    = \Big[ \big\langle \Psi_{\mathbf{k}}^{\phantom \dagger} \Psi_{\mathbf{k}}^{\dagger} \big\rangle \Big]_{m_{s} + \mu, m_{s} + \mu} 
    = \left[
        \left\langle 
        \begin{pmatrix}
            {\bm b}_{\mathbf{k}} \\ {\bm b}^{\dagger}_{- \mathbf{k}}
        \end{pmatrix}
        \begin{pmatrix}
            {\bm b}_{\mathbf{k}}^{\dagger} & {\bm b}_{- \mathbf{k}}
        \end{pmatrix}
        \right\rangle
    \right]_{m_{s} + \mu, m_{s} + \mu} 
    = \left[
        \begin{pmatrix}
            \langle  {\bm b}_{\mathbf{k}}^{\phantom \dagger} {\bm b}_{\mathbf{k}}^{\dagger}  \rangle
            & \langle  {\bm b}_{\mathbf{k}}^{\phantom \dagger} {\bm b}_{- \mathbf{k}}^{\phantom \dagger}  \rangle
            \\ 
            \langle  {\bm b}^{\dagger}_{- \mathbf{k}} {\bm b}_{\mathbf{k}}^{\dagger}  \rangle
            & \langle  {\bm b}^{\dagger}_{- \mathbf{k}} {\bm b}_{- \mathbf{k}}^{\phantom \dagger}  \rangle
        \end{pmatrix}
    \right]_{m_{s} + \mu, m_{s} + \mu}
\end{equation}
$$ where
$\langle \hat{\bm b}_{\pm \mathbf{k}}^{(\dagger)} \hat{\bm b}_{\pm \mathbf{k}}^{(\dagger)} \rangle$
denotes the block matrix whose elements is the expectation for two
bosonic operator
($\hat{\bm b}_{\pm \mathbf{k}}^{(\dagger)} \hat{\bm b}_{\pm \mathbf{k}}^{(\dagger)}$).
The expectation value is calculated for the state of the interest and
our main interests are the ground states and the Gibbs states.

The bosonic operator
$(\hat{b}_{\mathbf{k}\sigma}^{\phantom \dagger}, \hat{b}_{\mathbf{k}\sigma}^{\dagger} )$
at $\mathbf{k}$ are related to the eigenmodes
$(\hat{\beta}_{\mathbf{k}\sigma}^{\phantom \dagger}, \hat{\beta}_{\mathbf{k}\sigma}^{\dagger} )$
at momentum $\mathbf{k}$ through the para-unitary $\mathsf{T}_{\mathbf{k}}$.
$$
\begin{equation}
    \Psi_{\mathbf{k}} = \mathsf{T}_{\mathbf{k}} \widetilde{\Psi}_{\mathbf{k}}: \qquad
    \begin{pmatrix}
        \hat{\bm b}_{\mathbf{k}} \\ 
        \hat{\bm b}^{\dagger}_{-\mathbf{k}}
    \end{pmatrix}
    = 
    \begin{pmatrix}
        \mathsf{P}_{\mathbf{k}}           & \mathsf{Q}_{-\mathbf{k}}        \\
        \mathsf{Q}_{\mathbf{k}}^{*}       & \mathsf{P}_{-\mathbf{k}}^{*}
    \end{pmatrix}
    \begin{pmatrix}
        \hat{\bm\beta}_{\mathbf{k}} \\
        \hat{\bm\beta}_{-\mathbf{k}}^{\dagger}
    \end{pmatrix},
\end{equation}
$$ where
$\hat{\bm\beta}_{\mathbf{k}}^{T} = (\hat{\beta}_{\mathbf{k},a}, \hat{\beta}_{\mathbf{k},b}, \cdots)$.
Then, the correlation matrix can be obtained using unitary
transformation. $$
\begin{equation}
    \big 
    \langle \Psi_{\mathbf{k}}^{\phantom \dagger} \Psi_{\mathbf{k}}^{\dagger} 
    \big \rangle 
    = 
    \big\langle 
    \mathsf{T}_{\mathbf{k}}^{\phantom\dagger} \widetilde{\Psi}_{\mathbf{k}}^{\phantom\dagger} \widetilde{\Psi}_{\mathbf{k}}^{\dagger} \mathsf{T}_{\mathbf{k}}^{\dagger} 
    \big\rangle
    = 
    \mathsf{T}_{\mathbf{k}} \big\langle \widetilde{\Psi}_{\mathbf{k}}^{\phantom\dagger} \widetilde{\Psi}_{\mathbf{k}}^{\dagger}  \big \rangle \mathsf{T}_{\mathbf{k}}^{\dagger}
    = 
    \mathsf{T}_{\mathbf{k}} 
    \begin{pmatrix}
            \langle \hat{\bm \beta}_{ \mathbf{k}}^{\phantom \dagger} \hat{\bm \beta}_{ \mathbf{k}}^{\dagger}          \rangle            
        &   \langle \hat{\bm \beta}_{ \mathbf{k}}^{\phantom \dagger} \hat{\bm \beta}_{-\mathbf{k}}^{\phantom \dagger} \rangle\\
            \langle \hat{\bm \beta}_{ \mathbf{k}}^{\dagger}          \hat{\bm \beta}_{-\mathbf{k}}^{\dagger}          \rangle  
        &   \langle \hat{\bm \beta}_{-\mathbf{k}}^{\dagger}          \hat{\bm \beta}_{-\mathbf{k}}^{\phantom \dagger} \rangle
    \end{pmatrix}
    \mathsf{T}_{\mathbf{k}}^{\dagger}
    =\mathsf{T}_{\mathbf{k}}^{\phantom \dagger} \mathsf{N}_{\mathbf{k}}^{\phantom \dagger} \mathsf{T}_{\mathbf{k}}^{\dagger},
\end{equation}
$$ where $\mathsf{N}_{\mathbf{k}}$ is the
$2m_{s} \times 2m_{s}$ Block matrix, For the Gibbs states, all elements
for the correlation matrices can be obtained from the partition
function. $$
\begin{equation}
    \begin{array}{rlrl}
        \big[ \langle \hat{\bm \beta}_{ \mathbf{k}}^{\phantom \dagger} \hat{\bm \beta}_{ \mathbf{k}}^{\dagger} \rangle \big]_{\mu\nu} 
        & = \delta_{\mu\nu} \big( 1 + n_{\mu}(\mathbf{k}) \big) , 
        \qquad & \qquad 
        \big[ \langle \hat{\bm\beta}_{-\mathbf{k}}^{\dagger} \hat{\bm\beta}_{\mathbf{k}}^{\dagger} \rangle \big]_{\mu\nu} & = 0  ,
        \\[0.5em]
        \big[ \langle \hat{\bm\beta}_{-\mathbf{k}}^{\dagger} \hat{\bm\beta}_{- \mathbf{k}}^{\phantom \dagger} \rangle \big]_{\mu\nu}
        & = \delta_{\mu\nu} n_{\mu}(-\mathbf{k}) ,
        \qquad & \qquad 
        \big[ \langle \hat{\bm\beta}_{\mathbf{k}} \hat{\bm\beta}_{-\mathbf{k}} \rangle \big]_{\mu\nu} & = 0.
    \end{array}
\end{equation}
$$ where $n_{\mu}(\mathbf{k}) = 1/(e^{\beta E_{\mathbf{k}}} - 1)$ is
the Bose-Einstein distribution at energy $E_{\mathbf{k}}$.
