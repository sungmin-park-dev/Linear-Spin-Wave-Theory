# Skyrmions, Topological Magnons, and Hall Effects

## Skyrmion number {#subsec: Skyrmion number}

The skyrmion number itself does not require the LSWT, but introducing it
here would be useful for a later purpose. ho in the
lattice [@BERG1981412] However, The skyrmion number $Q_{\rm sk}$
quantifies the topological charge of a magnetic skyrmion in a lattice,
computed as the sum of solid angles over triangular plaquettes
($\triangle$): $$
\begin{equation}
    Q_{\rm skyr} 
    = \frac{1}{4\pi} \sum_{\triangle} \chi_{\triangle}.
\end{equation}
$$ For example, when the subsystem has four magnetic
sublattices ($\mu = a, b, c, d$), the summation runs over the triangles
$abc, acd, abd, bcd$. Here, $\chi_{\triangle}$ is the solid angle
subtended by normalized magnetization vectors $\mathbf{m}_i$,
$\mathbf{m}_j$, and $\mathbf{m}_k$ at the vertices of a plaquette, where
$\mathbf{m}_i = \mathbf{S}_i / S_i$ with $|\mathbf{m}_i| = 1$. The solid
angle is given by: $$
\begin{equation}
    \chi_{\triangle} 
    = 2 \arctan 
    \left( 
    \frac{|\mathbf{m}_i \cdot (\mathbf{m}_j \times \mathbf{m}_k)|}{1 + \mathbf{m}_i \cdot \mathbf{m}_j + \mathbf{m}_j \cdot \mathbf{m}_k + \mathbf{m}_k \cdot \mathbf{m}_i} 
    \right).
\end{equation}
$$

## Chern Number {#subsec: Chern number}

The Chern number of the $n$th band, $C_n$, is defined as the integral of
the Berry curvature $\Omega_{n, \mathbf{k}}$ over the first Brillouin zone
(FBZ): $$
\begin{equation}
    C_n = \frac{1}{2\pi} \int_{\text{FBZ}} \Omega_{n, \mathbf{k}} \, d^2\mathbf{k} 
    \approx \frac{1}{2\pi} \sum_{\mathbf{k} \in \text{FBZ}} \Omega_{n, \mathbf{k}} \Delta k_x \Delta k_y.
\end{equation}
$$ The Berry curvature for the $n$th band is calculated as
$$
\begin{equation}
    \Omega_{n, \mathbf{k}}^{\mu\nu} = 
    - 2 \, \mathrm{Im} \sum_{\substack{m = 1 \\ m \neq n}}^{2N}
    \frac{
        \big[ \mathsf{J} \mathsf{T}^{\dagger}_{\mathbf{k}} (\partial_{\mu} \mathsf{H}_{\mathbf{k}}) \mathsf{T}_{\mathbf{k}} \big]_{nm}
        \big[ \mathsf{J} \mathsf{T}^{\dagger}_{\mathbf{k}} (\partial_{\nu} \mathsf{H}_{\mathbf{k}}) \mathsf{T}_{\mathbf{k}} \big]_{mn}
    }{\big( (\mathsf{J} \mathsf{E}_{\mathbf{k}})_{nn} - (\mathsf{J} \mathsf{E}_{\mathbf{k}})_{mm} \big)^2},
\end{equation}
$$ where $\mathsf{J}$, $\mathsf{H}_{\mathbf{k}}$,
$\mathsf{T}_{\mathbf{k}}$, and $\mathsf{E}_{\mathbf{k}}$ are $2N \times 2N$
matrices, defined earlier $$
\begin{equation}
    \mathsf{J} = 
    \begin{pmatrix}
        \mathsf{I} & 0 \\ 0 & -\mathsf{I}
    \end{pmatrix}, 
    \qquad \text{and} \qquad
    \mathsf{T}^{\dagger}_{\mathbf{k}} \mathsf{H}_{\mathbf{k}} \mathsf{T}_{\mathbf{k}} = \mathsf{E}_{\mathbf{k}},
\end{equation}
$$ with $\mathsf{T}_{\mathbf{k}}$ diagonalizing the Hamiltonian
matrix $\mathsf{H}_{\mathbf{k}}$ to yield the energy matrix
$\mathsf{E}_{\mathbf{k}}$.

## Thermal Hall Conductance {#subsec: Thermal Hall Conductance}

The thermal Hall conductivity (THC), while it's hard to observe in
experiments, is related to the topology of magnon
bands [@PhysRevLett.128.117201]. It is computed using the Berry
curvature as $$
\begin{equation}
    \kappa_{xy} = - \frac{k_B^2 T}{V} \sum_{\mathbf{k} \in \mathrm{FBZ}} \sum_{n=1}^{N} \left\{ c_2[g(\varepsilon_{n, \mathbf{k}})] - \frac{\pi^2}{3} \right\} \Omega_{n, \mathbf{k}},
\end{equation}
$$ where $V$ is the system volume,
$g(\varepsilon_{n, \mathbf{k}}) = [e^{\varepsilon_{n, \mathbf{k}}/k_B T} - 1]^{-1}$
is the Bose-Einstein distribution, and $c_2(x)$ is the Spence function,
defined as $$
\begin{equation}
    c_2(x) 
    = \int_{0}^{x} dt \, \left( \ln \frac{1 + t}{t} \right)^2 
    = (1 + x) \left( \ln \frac{1 + x}{x} \right)^2 - (\ln x)^2 - 2 \mathrm{Li}_2(-x), 
    \quad \text{with} \quad 
    \mathrm{Li}_2(z) = - \int_{0}^{z} \frac{\ln (1 - t)}{t} \, dt.
\end{equation}
$$
