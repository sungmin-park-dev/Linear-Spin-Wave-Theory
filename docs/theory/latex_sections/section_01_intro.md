# Introduction to Spin Wave Theory (SWT)

For the spin wave theory, the spin model of our interest is
$$ \hat{H} = \sum_{ij} \sum_{\alpha, \beta} \hat{S}^{\alpha}_{i} J_{ij}^{\alpha\beta}  \hat{S}^{\beta}_{j} -  \sum_{j, \alpha} h_{j}^{\alpha} \hat{S}^{\alpha}_{j}, $$ 
where ${\bf S}_{i} = (S^{x}_{i}, S^{y}_{i}, S_{i}^{z})$
is the spin operator at site $i$, with $S^\alpha_i$ representing the
$\alpha$-direction component, and $ij$ denotes the *link*, not the
lattice indices[^1]. The first term is called the exchange Hamiltonian,
and the second term is the Zeeman term. Note that many Mott insulators
can be generally modeled as in
Eqn. [\[eq: General model Hamiltonian for spin wave theory\]](#eq: General model Hamiltonian for spin wave theory){reference-type="eqref"
reference="eq: General model Hamiltonian for spin wave theory"}.

1.  **Spin-Exchange interactions:** 
$$ \sum_{ij} \sum_{\alpha, \beta} \hat{S}^{\alpha}_{i} J_{ij}^{\alpha\beta}  \hat{S}^{\beta}_{j}  = \sum_{i \neq j} \hat{\bf S}_{i} \cdot {\bf J}_{ij} \cdot \hat{\bf S}_{j} + \sum_{i} \hat{\bf S}_{i} \cdot {\bf A}_{i} \cdot \hat{\bf S}_{i} $$ 
where the exchange energy is composed of inter-site interactions (first term) and single-ion anisotropies (second term).
The exchange matrix ${\bf J}_{ij}$ can represent various types of interactions including isotropic exchange, Dzyaloshinskii-Moriya interactions, and anisotropic exchanges such as Kitaev interactions. 
The single-ion anisotropy term, for example $- A \sum_{i} (S^{x}_{i})^{2}$, describes easy-axis anisotropy along $x$.

2.  **External magnetic fields and $g$-tensors:** 
$$  \sum_{j} {\bf h}_{j} \cdot \hat{\bf S}_{j} = \mu_B \sum_{j} \sum_{\alpha, \beta} B_{j}^{\alpha} g_{j}^{\alpha\beta} \hat{S}^{\beta}_{j} $$ 
where $\mu_B = 0.057883$ meV/T is the Bohr
    magneton. In magnetic materials, the $g$-tensor
    $g_{j}^{\alpha\beta}$ mediates the interaction between spins and
    external magnetic fields and can exhibit significant anisotropy due
    to spin-orbit coupling and crystal field effects, unlike the
    isotropic $g$-factor of free electrons ($g_e \approx 2.002$).

## Spin to Boson Transformations {#subsec: spin-to-boson transformation}

A spin wave theory for the model
\[Eqn. [\[eq: General model Hamiltonian for spin wave theory\]](#eq: General model Hamiltonian for spin wave theory){reference-type="eqref"
reference="eq: General model Hamiltonian for spin wave theory"}\] can be
formulated by assuming the quantum spin exhibits small fluctuations
around its classical spin configuration: $$  
    \hat{\mathbf{S}}_{j} = \mathbf{S}_{j} + \delta \hat{\mathbf{S}}_{j},
  $$ where $\mathbf{S}_{j}$ is the classical spin and the
operator $\delta \hat{\mathbf{S}}_{j}$ represents the quantum
fluctuation. Substituting this into the Hamiltonian $H$, we obtain:
$$\begin{align*}
    \hat{H} 
    = & \sum_{ij} \big( \mathbf{S}_{i} + \delta \hat{\mathbf{S}}_{i} \big) \cdot \mathbf{J}_{ij} \cdot \big( \mathbf{S}_{j} + \delta \hat{\mathbf{S}}_{j} \big) - \sum_{i} \mathbf{h}_{i} \cdot \big( \mathbf{S}_{i} + \delta \hat{\mathbf{S}}_{i} \big) \\
    = & E_{\rm cl}
    + \Big[ \sum_{ij} \big( \delta \hat{\mathbf{S}}_{i} \cdot \mathbf{J}_{ij} \cdot \mathbf{S}_{j} + \mathbf{S}_{i} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j} \big) - \sum_{i} \mathbf{h}_{i} \cdot \delta \hat{\mathbf{S}}_{i} \Big] 
    + \sum_{ij} \delta \hat{\mathbf{S}}_{i} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j} \\
    = & E_{\rm cl}
    + \sum_{i} \Big( \sum_{j \in \mathsf{L}(i)} \mathbf{S}_{j} \cdot \mathbf{J}_{ji} - \mathbf{h}_{i} \Big) \cdot \delta \hat{\mathbf{S}}_{i}
    + \sum_{i,j} \delta \hat{\mathbf{S}}_{i} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j},
\end{align*}$$ where
$E_{\rm cl} = \sum_{i,j} \mathbf{S}_{i} \cdot \mathbf{J}_{ij} \cdot \mathbf{S}_{j} - \sum_{i} \mathbf{h}_{i} \cdot \mathbf{S}_{i}$
is the classical energy, and the summation runs over the links, not the
lattice indices[^2], and
$\sum_{i} \sum_{j \in \mathsf{L}(i)} = \sum_{\langle i, j \rangle}$. In
the second line, we use
$\delta \hat{\mathbf{S}}_{i} \cdot \mathbf{J}_{ij} \cdot \mathbf{S}_{j} = \mathbf{S}_{j} \cdot \mathbf{J}_{ji} \cdot \delta \hat{\mathbf{S}}_{i}$,
which follows from $J_{ij}^{\alpha\beta} = J_{ji}^{\beta\alpha}$.

The fluctuation operator $\delta \hat{\mathbf{S}}_{j}$ can be further
decomposed into components perpendicular and parallel to the classical
spin orientation: $$  
    \delta \hat{\mathbf{S}}_{j} = \delta \hat{\mathbf{S}}_{j}^{\perp} + \delta \hat{\mathbf{S}}_{j}^{\parallel} 
    %= \big( \widetilde{S}_{j}^{+} \hat{\bf e}_{j}^{-} + \widetilde{S}_{j}^{-} \hat{\bf e}_{j}^{+} \big) + \widetilde{S}^{z}_{j} \hat{\bf e}_{j}^{0}.
  $$ Formally, this can be done using the
*Holstein-Primakoff* transformation [@PhysRev.58.1098], which represents
the spin operator in the laboratory frame in terms of bosonic operators:
$$  
    \hat{\bf S}_{j} 
    = \sqrt{S_{j}} 
    \left[ 
    \Big( 1 - \frac{\hat{n}_{j}}{2S_{j}} \Big)^{1/2} \hat{a}_{j} \hat{\bf e}_{j}^{-} + 
    \hat{a}_{j}^{\dagger} \Big( 1 - \frac{\hat{n}_{j}}{2S_{j}} \Big)^{1/2} \hat{\bf e}_{j}^{+}
    \right]
    + (S_{j} - \hat{n}_{j}) \hat{\bf e}_{j}^{0},
  $$ where the operators $\hat{a}_{i}$ and
$\hat{a}_{i}^{\dagger}$ are the bosonic annihilation and creation
operators that satisfy the canonical commutation relation
$[\hat{a}_{i}, \hat{a}_{j}^{\dagger}] = \delta_{ij}$, and
$\hat{n}_{j} \equiv \hat{a}^{\dagger}_{j} \hat{a}_{j}$ is the boson
number operator. The vectors $\hat{\bf e}_{j}^{\alpha}$
($\alpha = \pm, 0$) define a local frame of reference. In a more
conventional Cartesian basis, one defines
$\hat{\bf e}_{j}^{\pm} = (\hat{\bf x}_{j} \pm i \hat{\bf y}_{j})/\sqrt{2}$
and $\hat{\bf e}_{j}^{0} = \hat{\bf z}_{j}$ with
$\hat{\bf e}_{j}^{-} \times \hat{\bf e}_{j}^{+} = i \hat{\bf e}_{j}^{0}$,
where $\hat{\bf e}^{0}_{j} = {\bf S}_{j} / {S_{j}} = \hat{\bf n}_{j}$
denotes the classical spin orientation.

The perpendicular component $\delta \hat{\mathbf{S}}_{j}^{\perp}$
satisfies
$\hat{\bf e}_{j}^{0} \cdot \delta \hat{\mathbf{S}}_{j}^{\perp} = 0$
(where $\hat{\bf e}_{j}^{0}$ is the unit vector along the classical spin
direction) and contains only odd-order bosonic operators:
$$  
    \delta \hat{\mathbf{S}}_{j}^{\perp} = \sqrt{S_{j}}\left[ 
    \Big( 1 - \frac{\hat{n}_{j}}{2S_{j}} \Big)^{1/2} \hat{a}_{j} \hat{\bf e}_{j}^{-} + 
    \hat{a}_{j}^{\dagger} \Big( 1 - \frac{\hat{n}_{j}}{2S_{j}} \Big)^{1/2} \hat{\bf e}_{j}^{+}
    \right] 
    = \delta \hat{\mathbf{S}}_{j}^{(1)} + \delta \hat{\mathbf{S}}_{j}^{(3)} + \cdots 
  $$ The parallel component,
$\delta \hat{\mathbf{S}}_{j}^{\parallel} = \delta \hat{\mathbf{S}}_{j}^{(2)} = -\hat{n}_{j} \hat{\bf e}_{j}^{0}$,
involves the number operator, where $\hat{\bf e}_{j}^{0}$ is the unit
vector along the classical spin.

Based on the above equations, the Hamiltonian can then be rewritten as:
$$  
    \begin{split}
        H = & E_{\rm cl}
        + \sum_{i} \Big( \sum_{j} \mathbf{S}_{j} \cdot \mathbf{J}_{ji} - \mathbf{h}_{i} \Big) \cdot \delta \hat{\mathbf{S}}_{i}
        + \sum_{i,j} \delta \hat{\mathbf{S}}_{i} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j}, \\
        = & E_{\rm cl} + H_{1} + H_{2} + H_{3} + H_{4} + \cdots,
    \end{split}
  $$ where the terms could be obtained by plugging
$\delta \hat{\bf S}_{j}^{\perp} = \delta \hat{\mathbf{S}}_{j}^{(1)} + \delta \hat{\mathbf{S}}_{j}^{(3)} + \cdots$,
and $\delta \hat{\bf S}_{j}^{\parallel} = \delta \hat{\bf S}_{j}^{(2)}$.
$$  
    \begin{split}
        E_{\rm cl} 
        = & \sum_{ij} \mathbf{S}_i \cdot \mathbf{J}_{ij} \mathbf{S}_j - \sum_{i} \mathbf{h}_{i} \cdot \mathbf{S}_{i}
        = \sum_{ij}  \widetilde{J}_{ij}^{00} S_{i} S_{j} - \sum_{i} S_{i}^{0} h_{i}, \\
        H_{1} = & \sum_{i} \Big( \sum_{j \in \mathsf{L}(i)} \mathbf{S}_{j} \cdot \mathbf{J}_{ji} - \mathbf{h}_{i} \Big) \cdot \delta \hat{\mathbf{S}}_{i}^{(1)} 
        = \sum_{i} \frac{\partial E_{\rm cl}}{\partial \mathbf{S}_{i}} \cdot \delta \hat{\mathbf{S}}_{i}^{(1)}, \\
        H_{2} = & \sum_{ij} \delta \hat{\mathbf{S}}_{i}^{(1)} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j}^{(1)} 
        + \sum_{i} \Big( \sum_{j\in \mathsf{L}(i)} \mathbf{S}_{j} \cdot \mathbf{J}_{ji} - \mathbf{h}_{i} \Big) \cdot \delta \hat{\mathbf{S}}_{i}^{(2)}, \\
        H_{3} = & \sum_{ij} \big( \delta \hat{\mathbf{S}}_{i}^{(1)} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j}^{(2)} + \delta \hat{\mathbf{S}}_{i}^{(2)} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j}^{(1)} \big), 
        \quad \text{and so on.} 
    \end{split}
  $$ Note that the linear boson term vanishes when the spin
configuration corresponds to a classical extremum: $$  
    \sum_{i} \frac{\partial E_{\rm cl}}{\partial \mathbf{S}_{i}} \cdot \delta \hat{\mathbf{S}}_{i}^{(1)} = 0.
  $$ To demonstrate this, consider the Lagrangian with a
Lagrange multiplier enforcing the spin magnitude constraint:
$$  
    \mathcal{L} = E_{\rm cl} - \sum_{j} \frac{\lambda_{j}}{2} \big( |\mathbf{S}_{j}|^{2} - S_{j}^{2} \big).
  $$ Taking the derivative with respect to $\mathbf{S}_{i}$,
we get: $$  
    \frac{\partial \mathcal{L}}{\partial \mathbf{S}_{i}} = \frac{\partial E_{\rm cl}}{\partial \mathbf{S}_{i}} - \lambda_{i} \mathbf{S}_{i} = 0,
  $$ where
$\frac{\partial E_{\rm cl}}{\partial \mathbf{S}_{i}} = \sum_{j} \mathbf{S}_{j} \cdot \mathbf{J}_{ji} - \mathbf{h}_{i}$.
Since $\delta \hat{\mathbf{S}}_{i}^{\perp}$ is perpendicular to
$\mathbf{S}_{i}$, it follows that: $$  
    \boxed{\frac{\partial E_{\rm cl}}{\partial \mathbf{S}_{i}} \cdot \delta \hat{\mathbf{S}}_{i}^{\perp} = \lambda_{i} \mathbf{S}_{i} \cdot \delta \hat{\mathbf{S}}_{i}^{\perp} = 0, \text{ for all } i.} 
  $$ Thus, the linear boson term, as well as any odd terms
related to classical energy differentiation, vanishes when the spin
configuration is a classical stable point.

##### Comment ---

Alternatively, *Dyson-Maleev*
transformation [@PhysRev.102.1217; @maleev1958scattering] could map spin
operators onto bosonic creation and annihilation operators
($a_{i}^{\dagger}, a_{i}$) as well: $$  
\label{eq: spin-to-boson mapping}
        S^{+}_{i} = \sqrt{2S_{i}} \Big( 1 - \frac{a^{\dagger}_{i}a_{i}}{2S_{i}} \Big)^{\xi} a_{i}, \quad
        S^{-}_{i} = \sqrt{2S_{i}} \, a_{i}^{\dagger} \Big( 1 - \frac{a^{\dagger}_{i}a_{i}}{2S_{i}} \Big)^{1 - \xi}, \quad
        S_{i}^{z} = S_{i} - a^{\dagger}_{i} a_{i},
  $$ where $\xi = 1/2$ for the Holstein-Primakoff
transformation, and $\xi = 1$ for the Dyson-Maleev transformation. While
both transformations map spin operators onto bosonic operators, the
Dyson-Maleev transformation does not preserve Hermiticity. For instance,
$(S^{+}_{i})^{\dagger}$ is not equal to $S^{-}_{i}$: $$  
        (S^{+}_{i})^{\dagger} = \sqrt{2S_{i}} a^{\dagger}_{i} \Big( 1 - \frac{a^{\dagger}_{i} a_{i}}{2S_{i}} \Big)^{\xi} = S^{-}_{i} \Big( 1 - \frac{a^{\dagger}_{i} a_{i}}{2S_{i}} \Big)^{2\xi - 1},
  $$ whereas for $\xi = 1/2$ (the Holstein-Primakoff
transformation), Hermiticity of the spin operators is preserved.

## Rotations for Spin Models

To analyze an ordered spin system using linear spin-wave theory (LSWT),
it is useful to rotate the spins at each site from the *laboratory*
reference frame $\{\hat{\bf x}, \hat{\bf y}, \hat{\bf z}\}$ to a *local*
reference frame $\{\hat{\bf x}_{j}, \hat{\bf y}_{j}, \hat{\bf z}_{j}\}$
(aligned with the classical spin ordering, where $\hat{{\bf z}}_{j}$
coincides with the spin's quantization axis). The transformation is
given by: $$  
    \label{eq: Spin rotation}
    \hat{\bf S}_j = \mathbf{R}_j \, \widetilde{\mathbf{S}}_j 
    = \widetilde{S}^{x}_{j} \hat{\bf x}_{j} + \widetilde{S}^{y}_{j} \hat{\bf y}_{j} + \widetilde{S}^{z}_{j} \hat{\bf z}_{j},
  $$ where $\widetilde{\mathbf{S}}_j$ is the spin operator
in the local reference frame at site $j$. By applying the rotation to
all spins in a unit cell, the spin Hamiltonian is transformed from the
laboratory frame to the local frame as: $$  
\label{eq: Rotated Hamiltonian}
    \begin{split}
        \hat{H} = & \sum_{ij} \sum_{\alpha, \beta} S^{\alpha}_{i} J_{ij}^{\alpha\beta} S^{\beta}_{j} - \sum_{j, \alpha} h_{j}^{\alpha} S^{\alpha}_{j}
        = \sum_{ij} \sum_{\alpha, \beta} \widetilde{S}^{\alpha}_{i} \widetilde{J}_{ij}^{\alpha\beta} \widetilde{S}^{\beta}_{j} - \sum_{j, \alpha} \widetilde{h}_{j}^{\alpha} \widetilde{S}^{\alpha}_{j},
    \end{split}
  $$ where the rotated couplings are defined as:
$$  
    \widetilde{J}_{ij}^{\alpha\beta}  = [\widetilde{\mathbf{J}}_{ij}]^{\alpha\beta} = [\mathbf{R}_i^{\rm T} \mathbf{J}_{ij} \mathbf{R}_j]^{\alpha\beta}, 
    \quad \text{ and } \quad
     \widetilde{h}_{j}^{\alpha} = [\widetilde{\mathbf{h}}_j]^{\alpha} = [\mathbf{h}_{j}^{\rm T} \mathbf{R}_{j}]^{\alpha}.
  $$ In practice, the rotation from the crystal frame to the
local frame is performed in two stages, expressed as
$\mathbf{R}_i = \mathbf{R}^{\varphi} \cdot \mathbf{R}^{\theta}$. The
combined rotation matrix is: $$  
    \mathbf{R} (\theta, \varphi) = \mathbf{R}^{\varphi} \cdot \mathbf{R}^{\theta} = 
    \begin{pmatrix} 
        \cos \theta \cos \varphi & -\sin \varphi & \sin \theta \cos \varphi \\
        \cos \theta \sin \varphi & \cos \varphi & \sin \theta \sin \varphi \\
        -\sin \theta & 0 & \cos \theta
    \end{pmatrix},
    \label{eq_frametransform_gen}
  $$ where $\theta$ is the angle relative to the $z$-axis in
lab frame and $\varphi$ is the azimuthal angle in the $xy$-plane. Note
that the classical spin orientation $\mathbf{S}_j$ satisfies:
$$  
    \mathbf{S}_{j} 
    = \mathbf{R}_{j} \, \hat{\bf z} 
    = \mathbf{R}^{\varphi_{j}} \mathbf{R}^{\theta_{j}} 
    \begin{pmatrix} 
        0 \\ 0 \\ 1 
    \end{pmatrix} 
    = \mathbf{R}^{\varphi_j} 
    \begin{pmatrix} 
        \sin \theta_{j} \\ 0 \\ \cos \theta_{j} 
    \end{pmatrix} 
    = 
    \begin{pmatrix} 
        \sin \theta_{j} \cos \varphi_{j} \\ \sin \theta_{j} \sin \varphi_{j} \\ \cos \theta_{j} 
    \end{pmatrix}.
  $$

For later uses, it is also convenient to express the spin operator using
ladder operators with complex vector: $$  
\label{eq: Spin rotation-ladder}
    \hat{\bf S}_j = \mathbf{R}_j \, \widetilde{\mathbf{S}}_j
    = \frac{\widetilde{S}_{j}^{+} \hat{\bf e}^{-}_{j} + \widetilde{S}_{j}^{-} \hat{\bf e}^{+}_{j} }{\sqrt{2}} + \widetilde{S}^{0}_{j} \hat{\bf e}^{0}_{j},
  $$ where the spin operators are
$\widetilde{S}_{j}^{\pm} \equiv \widetilde{S}_{j}^{x} \pm i \widetilde{S}_{j}^{y}$,
$\widetilde{S}^{0}_{j} = \widetilde{S}^{z}_{j}$, and
$[\widetilde{S}^{z}_{j}, \widetilde{S}^{\pm}_{j}] = \pm \widetilde{S}^{\pm}_{j}$.
The complex vectors are $$  
    \hat{\bf e}_{j}^{\alpha} \equiv \mathbf{R}_{j} \hat{\bf e}^{\alpha}: \quad 
    \hat{\bf e}^{\pm}_{j} = \mathbf{R}_{j} \Big( \frac{ \hat{\bf x} \pm i \hat{\bf y} }{\sqrt{2}} \Big)
    \text{ and }
    \hat{\bf e}_{j}^{0} \equiv \mathbf{R}_{j} \hat{\bf z},
  $$ where $\alpha = \pm, 0, (x,y,z)$ and the
$\hat{\bf e}_{j}^{0}$-direction represents the classical magnetic
orientation at site $j$. These complex vectors
$\hat{\bf e}_{j}^{\alpha}$ introduce a rotated Hamiltonian similar to
the previous one: $$  
    \hat{H} 
    = \sum_{ij} \sum_{\alpha, \beta = 0, \pm} 
    \widetilde{S}^{\alpha}_{i} \widetilde{J}^{\bar{\alpha}\bar{\beta}}_{ij} \widetilde{S}^{\beta}_{j} 
    - \sum_{j} \sum_{\alpha = 0, \pm} \widetilde{h}^{\alpha}_{j} \widetilde{S}^{\alpha}_{j}.
  $$ where $\bar{(\cdot)} = - ( \cdot )$, for example
$\alpha = \pm, (0)$ and $\bar{\alpha} = \mp, (0)$. The couplings
$\widetilde{J}^{\bar{\alpha}\bar{\beta}}_{ij}$ and magnetic fields
$\widetilde{h}_{j}^{\alpha}$ are defined as $$  
    \widetilde{J}^{\alpha\beta}_{ij} \equiv \hat{\bf e}_{i}^{\alpha} \cdot \widetilde{\bf J}_{ij} \cdot \hat{\bf e}_{j}^{\beta}
    \quad \text{ and } \quad
    \widetilde{h}^{\alpha}_{j} \equiv \widetilde{\bf h}_{j} \cdot \hat{\bf e}^{\alpha}_{j}.
  $$

The transformation from $xyz$ to $\pm0$ basis is given by:
$$  
    \mathbf{C} = 
    \begin{pmatrix}
        \hat{\bf e}^{+} & \hat{\bf e}^{-} & \hat{\bf e}^{0}
    \end{pmatrix}
    = 
    \begin{pmatrix}
        [\hat{\mathbf{e}}^{+}]^{x} & [\hat{\mathbf{e}}^{-}]^{x} & [\hat{\mathbf{e}}^{0}]^{x} \\[0.1em]
        [\hat{\mathbf{e}}^{+}]^{y} & [\hat{\mathbf{e}}^{-}]^{y} & [\hat{\mathbf{e}}^{0}]^{y} \\[0.1em]
        [\hat{\mathbf{e}}^{+}]^{z} & [\hat{\mathbf{e}}^{-}]^{z} & [\hat{\mathbf{e}}^{0}]^{z} 
    \end{pmatrix}
    = 
    \begin{pmatrix}
        \frac{1}{\sqrt{2}}  &   \frac{1}{\sqrt{2}}  &   0   \\[0.1em]
        \frac{i}{\sqrt{2}}  & - \frac{i}{\sqrt{2}}  &   0   \\[0.1em]
                    0       &           0           &   1   
    \end{pmatrix} 
  $$ where $[\hat{\mathbf{e}}_{i}^{\alpha}]^{\beta}$ denotes
the $\beta = x, y, z$ component of the basis vector
$\hat{\mathbf{e}}_{i}^{\alpha}$ (with $\alpha = +, -, 0$). These
components are obtained from: $$  
    \mathbf{U}_{j} = \mathbf{R}_{j} \mathbf{C} =
    \begin{pmatrix}
        [\hat{\mathbf{e}}_{j}^{+}]^{x} & [\hat{\mathbf{e}}_{j}^{-}]^{x} & [\hat{\mathbf{e}}_{j}^{0}]^{x} \\[0.25em]
        [\hat{\mathbf{e}}_{j}^{+}]^{y} & [\hat{\mathbf{e}}_{j}^{-}]^{y} & [\hat{\mathbf{e}}_{j}^{0}]^{y} \\[0.25em]
        [\hat{\mathbf{e}}_{j}^{+}]^{z} & [\hat{\mathbf{e}}_{j}^{-}]^{z} & [\hat{\mathbf{e}}_{j}^{0}]^{z} 
    \end{pmatrix}
    = 
    \begin{pmatrix}
        R^{xx}_{j}   &   R^{xy}_{j}   &   R^{xz}_{j}   \\[0.25em]
        R^{yx}_{j}   &   R^{yy}_{j}   &   R^{yz}_{j}   \\[0.25em]
        R^{zx}_{j}   &   R^{zy}_{j}   &   R^{zz}_{j}   
    \end{pmatrix}
    \begin{pmatrix}
        \frac{1}{\sqrt{2}}   &   \frac{1}{\sqrt{2}}   &   0   \\[0.25em]
        \frac{i}{\sqrt{2}}   & - \frac{i}{\sqrt{2}}   &   0   \\[0.25em]
                0     &         0       &   1   
    \end{pmatrix} 
  $$

The transformed coupling tensor
$\mathbf{C}^{\dagger} \mathbf{J} \mathbf{C} =$ is expressed as:
$$  
    \begin{split}
    \begin{pmatrix}
        J^{-+}_{ij} & J^{--}_{ij} & J^{-0}_{ij} \\[0.3em]
        J^{++}_{ij} & J^{+-}_{ij} & J^{+0}_{ij} \\[0.3em]
        J^{0+}_{ij} & J^{0-}_{ij} & J^{00}_{ij}
    \end{pmatrix}
    = &
    \begin{pmatrix}
        \frac{1}{\sqrt{2}}   & - \frac{i}{\sqrt{2}}   &   0   \\[0.3em]
        \frac{1}{\sqrt{2}}   &   \frac{i}{\sqrt{2}}   &   0   \\[0.3em]
                0     &         0       &   1   
    \end{pmatrix} 
    \begin{pmatrix}
        J^{xx}_{ij}   &   J^{xy}_{ij}   &   J^{xz}_{ij}   \\[0.3em]
        J^{yx}_{ij}   &   J^{yy}_{ij}   &   J^{yz}_{ij}   \\[0.3em]
        J^{zx}_{ij}   &   J^{zy}_{ij}   &   J^{zz}_{ij}   
    \end{pmatrix}
    \begin{pmatrix}
        \frac{1}{\sqrt{2}}   &   \frac{1}{\sqrt{2}}   &   0   \\[0.3em]
        \frac{i}{\sqrt{2}}   & - \frac{i}{\sqrt{2}}   &   0   \\[0.3em]
                0     &         0       &   1   
    \end{pmatrix} 
    % \\ 
    % \begin{pmatrix}
    %     \frac{1}{2} \left( J^{xx} + J^{yy} + i (J^{xy} - J^{yx}) \right) & 
    %     \frac{1}{2} \left( J^{xx} - J^{yy} - i (J^{xy} + J^{yx}) \right) & 
    %     \frac{1}{\sqrt{2}} \left( J^{xz} - i J^{yz} \right) \\
    %     \frac{1}{2} \left( J^{xx} - J^{yy} + i (J^{xy} + J^{yx}) \right) & 
    %     \frac{1}{2} \left( J^{xx} + J^{yy} - i (J^{xy} - J^{yx}) \right) & 
    %     \frac{1}{\sqrt{2}} \left( J^{xz} + i J^{yz} \right) \\
    %     \frac{1}{\sqrt{2}} \left( J^{zx} + i J^{zy} \right) & 
    %     \frac{1}{\sqrt{2}} \left( J^{zx} - i J^{zy} \right) & 
    %     J^{zz}
    % \end{pmatrix}
    \end{split}
  $$ with elements defined as
$\widetilde{J}_{ij}^{\alpha\beta} = \hat{\mathbf{e}}_i^{\alpha} \cdot \widetilde{\mathbf{J}}_{ij} \cdot \hat{\mathbf{e}}_j^{\beta}$:
$$  
\label{eq: expressions for exchange couplings}
    \begin{array}{rlrlrl}
        J_{ij}^{--} &= \frac{1}{2} \big( J_{ij}^{xx} - J_{ij}^{yy} - i (J_{ij}^{xy} + J_{ij}^{yx}) \big), \qquad \qquad &
        J_{ij}^{++} &= \frac{1}{2} \big( J_{ij}^{xx} - J_{ij}^{yy} + i (J_{ij}^{xy} + J_{ij}^{yx}) \big), &&\\[0.25em]
        J_{ij}^{-+} &= \frac{1}{2} \big( J_{ij}^{xx} + J_{ij}^{yy} + i (J_{ij}^{xy} - J_{ij}^{yx}) \big), \qquad &
        J_{ij}^{+-} &= \frac{1}{2} \big( J_{ij}^{xx} + J_{ij}^{yy} - i (J_{ij}^{xy} - J_{ij}^{yx}) \big), &&\\[0.25em]
        J_{ij}^{0\pm} &= \frac{1}{\sqrt{2}} \big( J_{ij}^{zx} \pm i J_{ij}^{zy} \big), &
        J_{ij}^{\pm0} &= \frac{1}{\sqrt{2}} \big( J_{ij}^{xz} \pm i J_{ij}^{yz} \big), & \qquad
        J_{ij}^{00} &= J_{ij}^{zz}.
    \end{array}
  $$

## Bosonic representations for spin Hamiltonian {#subsec: Bosonic representations for spin exchange Hamiltonian}

In this section, we derive the explicit bosonic Hamiltonian for the spin
model. $$  
    \begin{split}
        \hat{H} 
        &= E_{\rm cl}
        + \sum_{i} \Big( \sum_{j \in \mathsf{L}(ij)} \mathbf{S}_{j} \cdot \mathbf{J}_{ji} - \mathbf{h}_{i} \Big) \cdot \delta \hat{\mathbf{S}}_{i}
        + \sum_{ij \in \mathsf{L}} \delta \hat{\mathbf{S}}_{i} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j},
    \end{split}
  $$ where $E_{\rm cl}$ is the classical energy,
$\mathbf{S}_{j}$ represents the spin vector at site $j$,
$\mathbf{J}_{ji}$ is the exchange interaction tensor, $\mathbf{h}_{i}$
is the external field, and $\delta \hat{\mathbf{S}}_{i}$ denotes the
spin fluctuation operator.

Let us divide the spin model into even and odd bosonic operator terms,
$$\label{eq: LSWT boson expansion even/odd}
    \begin{align}
        H_{\rm even} &= \sum_{ij\in\mathsf{L}} 
        \Big( \delta \hat{\mathbf{S}}_{i}^{\perp} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j}^{\perp} 
        + \delta \hat{\mathbf{S}}_{i}^{\parallel} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j}^{\parallel} \Big)
        + \sum_{i} \Big( \sum_{j \in \mathsf{L}(ij)} \mathbf{S}_{j} \cdot \mathbf{J}_{ji} - \mathbf{h}_{i} \Big) \cdot \delta \hat{\mathbf{S}}_{i}^{\parallel}, 
        \label{subeq: even bosonic terms in SWT} \\
        H_{\rm odd}^{(m \ge 3)} &= \sum_{ij\in\mathsf{L}} 
        \Big( \delta \hat{\mathbf{S}}_{i}^{\perp} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j}^{\parallel} 
        + \delta \hat{\mathbf{S}}_{i}^{\parallel} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j}^{\perp} \Big).
        \label{subeq: odd bosonic terms in SWT} 
    \end{align}$$ and substituting the following spin-to-boson
transformation: $$  
    \delta \hat{\mathbf{S}}_{i}^{\perp} = \frac{\widetilde{S}_{i}^{+} \hat{\mathbf{e}}^{-}_{i} + \widetilde{S}_{i}^{-} \hat{\mathbf{e}}^{+}_{i} }{\sqrt{2}} = \sqrt{S_{i}} \left[ \Big( 1 - \frac{\hat{a}^{\dagger}_{i}\hat{a}_{i}}{2S_{i}} \Big)^{1/2} \hat{a}_{i} \hat{\mathbf{e}}_{i}^{-} + \hat{a}_{i}^{\dagger} \Big( 1 - \frac{\hat{a}^{\dagger}_{i}\hat{a}_{i}}{2S_{i}} \Big)^{1/2} \hat{\mathbf{e}}_{i}^{+} \right], 
    \quad \text{ and } \quad
    \delta \hat{\mathbf{S}}^{\parallel}_{i} =  - \hat{n}_{i} \hat{\mathbf{e}}_{i},
  $$ where $\hat{\mathbf{e}}_{i}^{\pm}$ are the raising and
lowering unit vectors, $\hat{\mathbf{e}}_{i}$ is the unit vector along
the local quantization axis, and
$\hat{n}_{i} = \hat{a}^{\dagger}_{i}\hat{a}_{i}$ is the boson number
operator. This boson mapping results in the SWT Hamiltonian. While the
odd terms can also be obtained easily, we will focus on the even terms
as they are of primary interest for the linear spin wave theory.

The even Hamiltonian term, $H_{\rm even}$, consists of terms involving
an even number of bosonic operators: $$  
    \delta \hat{\mathbf{S}}_{i}^{\perp} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j}^{\perp}  
    = \frac{1}{2} \Big( \widetilde{S}_{i}^{+} \widetilde{J}_{ij}^{--} \widetilde{S}_{j}^{+}  
    + \widetilde{S}_{i}^{+} \widetilde{J}_{ij}^{-+} \widetilde{S}_{j}^{-}
    + \widetilde{S}_{i}^{-} \widetilde{J}_{ij}^{+-} \widetilde{S}_{j}^{+} 
    + \widetilde{S}_{i}^{-} \widetilde{J}_{ij}^{++} \widetilde{S}_{j}^{-} \Big),
  $$ This becomes: $$\begin{align*}
    \frac{1}{2} \widetilde{S}_{i}^{+} \widetilde{J}_{ij}^{-+} \widetilde{S}_{j}^{-} 
    = & 
    \sqrt{S_{i}S_{j}} \widetilde{J}_{ij}^{-+} \Big( 1 - \frac{\hat{n}_{i}}{2S_i} \Big)^{1/2} \hat{a}_{i} \hat{a}_{j}^{\dagger}  \Big( 1 - \frac{\hat{n}_{j}}{2S_j} \Big)^{1/2} 
    = t_{ij}^{-+} \Big[ \hat{a}_{i} \hat{a}_{j}^{\dagger}  - \tfrac{1}{4} \Big( \tfrac{\hat{n}_{i} \hat{a}_{i} \hat{a}_{j}^{\dagger}}{S_{i}} + \tfrac{\hat{a}_{i} \hat{a}_{j}^{\dagger} \hat{n}_{j}}{S_{j}} \Big) + \mathcal{O} (S^{-2}) \Big]
    \\
    \frac{1}{2} \widetilde{S}_{i}^{-} \widetilde{J}_{ij}^{+-} \widetilde{S}_{j}^{+} = &
    \sqrt{S_i S_j} \widetilde{J}_{ij}^{+-} \hat{a}_i^{\dagger} \Big( 1 - \frac{\hat{n}_i}{2 S_i} \Big)^{1/2} \Big( 1 - \frac{\hat{n}_j}{2 S_j} \Big)^{1/2} \hat{a}_j 
    = t_{ij}^{+-} \Big[ \hat{a}_{i}^{\dagger} \hat{a}_{j}  - \tfrac{1}{4} \Big( \tfrac{\hat{a}_{i}^{\dagger} \hat{n}_{i} \hat{a}_{j}}{S_{i}} + \tfrac{\hat{a}_{i}^{\dagger} \hat{n}_{j} \hat{a}_{j}}{S_{j}} \Big) + \mathcal{O} (S^{-2}) \Big]
    \\
    \frac{1}{2} \widetilde{S}_{i}^{+} \widetilde{J}_{ij}^{--} \widetilde{S}_{j}^{+} = & 
    \sqrt{S_i S_j} \widetilde{J}_{ij}^{--} \Big( 1 - \frac{\hat{n}_i}{2 S_i} \Big)^{1/2} \hat{a}_i \Big( 1 - \frac{\hat{n}_j}{2 S_j} \Big)^{1/2} \hat{a}_j 
    = t_{ij}^{--} \Big[ \hat{a}_{i} \hat{a}_{j} - \tfrac{1}{4} \Big( \tfrac{\hat{n}_{i} \hat{a}_{i} \hat{a}_{j}}{S_{i}} + \tfrac{\hat{a}_{i} \hat{n}_{j} \hat{a}_{j}}{S_{j}} \Big) + \mathcal{O} (S^{-2}) \Big]
    \\
    \frac{1}{2} \widetilde{S}_{i}^{-} \widetilde{J}_{ij}^{++} \widetilde{S}_{j}^{-} = & 
    \sqrt{S_i S_j} \widetilde{J}_{ij}^{++} \hat{a}_i^{\dagger} \Big( 1 - \frac{\hat{n}_i}{2 S_i} \Big)^{1/2} \hat{a}_j^{\dagger} \Big( 1 - \frac{\hat{n}_j}{2 S_j} \Big)^{1/2}
    = t_{ij}^{++} \Big[ \hat{a}_{i}^{\dagger} \hat{a}_{j}^{\dagger} - \tfrac{1}{4} \Big( \tfrac{\hat{a}_{i}^{\dagger} \hat{n}_{i} \hat{a}_{j}^{\dagger}}{S_{i}} + \tfrac{\hat{a}_{i}^{\dagger} \hat{a}_{j}^{\dagger} \hat{n}_{j}}{S_{j}} \Big) + \mathcal{O} (S^{-2}) \Big],
\end{align*}$$ where we define
$t^{\alpha\beta}_{ij} = \sqrt{S_{i}S_{j}} \widetilde{J}^{\alpha\beta}_{ij}$.
Next, the other terms in
Eqn. [\[subeq: even bosonic terms in SWT\]](#subeq: even bosonic terms in SWT){reference-type="eqref"
reference="subeq: even bosonic terms in SWT"} are: $$  
    \begin{split}
    \sum_{i} \Big( \sum_{j \in \mathsf{L}(i)} \mathbf{S}_{j} \cdot \mathbf{J}_{ji} - \mathbf{h}_{i} \Big) \cdot \delta \hat{\mathbf{S}}_{i}^{\parallel} 
    &= 
    \sum_{i} \Big( \widetilde{h}_{i}^{0} - \sum_{j \in \mathsf{L}(i)} S_{j} \cdot \widetilde{J}_{ji}^{00} \Big)  \hat{n}_{i}, \\
    \delta \hat{\mathbf{S}}_{i}^{\parallel} \cdot \mathbf{J}_{ij} \cdot \delta \hat{\mathbf{S}}_{j}^{\parallel}
    &= \hat{n}_{i} \widetilde{J}_{ij}^{00} \hat{n}_{j},
    \end{split}
  $$ where $\widetilde{h}_{i}^{0}$ is the magnetic field
along the local quantization axis and $\widetilde{J}_{ij}^{00}$
represents the longitudinal component of the exchange interaction.

Combining all terms, we can express the quadratic and quartic
Hamiltonian contributions, $H_{2}$ and $H_{4}$, as: $$\begin{align}
    H_{2} &= \sum_{ij} \Big[ t_{ij}^{--} \hat{a}_{i} \hat{a}_{j} + t_{ij}^{-+} \hat{a}_{i}\hat{a}_{j}^{\dagger} + t_{ij}^{+-} \hat{a}_{i}^{\dagger} \hat{a}_{j} + t_{ij}^{++} \hat{a}^{\dagger}_{i} \hat{a}^{\dagger}_{j} - \widetilde{J}_{ij}^{00} \big( S_{i} \hat{n}_{j} + S_{j} \hat{n}_{i} \big) \Big] + \sum_{i} h_{i} \hat{n}_{i}, 
    \label{eq: SWT bosonic H2 in real space}\\
    H_{4} &= \sum_{ij} \widetilde{J}_{ij}^{00} \hat{n}_{i} \hat{n}_{j} - \Big[ t_{ij}^{--} \big( \tfrac{\hat{n}_{i}}{S_{i}} + \tfrac{\hat{n}_{j}}{S_{j}} \big) \hat{a}_{i} \hat{a}_{j} + t_{ij}^{++} \hat{a}^{\dagger}_{i} \hat{a}^{\dagger}_{j} \big( \tfrac{\hat{n}_{i}}{S_{i}} + \tfrac{\hat{n}_{j}}{S_{j}} \big) + t_{ij}^{-+} \big( \tfrac{\hat{n}_{i} \hat{a}_{i} \hat{a}_{j}^{\dagger}}{S_{i}} + \tfrac{\hat{a}_{i} \hat{a}_{j}^{\dagger} \hat{n}_{j}}{S_{j}} \big) + t_{ij}^{+-} \big( \tfrac{\hat{a}_{i}^{\dagger} \hat{n}_{i} \hat{a}_{j}}{S_{i}} + \tfrac{\hat{a}_{i}^{\dagger} \hat{n}_{j} \hat{a}_{j}}{S_{j}} \big) \Big]. 
    \label{eq: SWT bosonic H4 in real space}
\end{align}$$ Note that if the anisotropic couplings are absent in the
rotated exchange matrix $\widetilde{J}^{xy} = \widetilde{J}^{yx} = 0$,
and $\widetilde{J}^{xx} = \widetilde{J}^{yy}$ are equal, then the
anomalous contribution, such as $a^{\dagger}a^{\dagger}$, vanishes, see
Eq. [\[eq: expressions for exchange couplings\]](#eq: expressions for exchange couplings){reference-type="eqref"
reference="eq: expressions for exchange couplings"}. Furthermore, when
the spins' magnitudes equal $S_{i} = S$, the expansions
$$  
    H = S^{2} \tilde{E}_{cl} + S^{1} \tilde{H}_2 + S^{1/2} \tilde{H}_3 + S^{0} \tilde{H}_4 + \cdots.
  $$

## Momentum Space Representations

In this section, we shall find the momentum space representation for the
quadratic spin wave Hamiltonian ($H_{2}$). The quadratic bosonic
Hamiltonian we have in
Eqn. [\[eq: SWT bosonic H2 in real space\]](#eq: SWT bosonic H2 in real space){reference-type="eqref"
reference="eq: SWT bosonic H2 in real space"} is $$  
    \begin{split}
        H_{2} = & \sum_{ij}
        \Big[ t_{ij}^{--} \hat{a}_{i} \hat{a}_{j} + t_{ij}^{-+} \hat{a}_{i}\hat{a}_{j}^{\dagger} + t_{ij}^{+-} \hat{a}_{i}^{\dagger} \hat{a}_{j} + t_{ij}^{++} \hat{a}^{\dagger}_{i} \hat{a}^{\dagger}_{j} - \widetilde{J}_{ij}^{00} \big( S_{i} \hat{a}_{j}^{\dagger} \hat{a}_{j} + S_{j} \hat{a}_{i}^{\dagger} \hat{a}_{i} \big) \Big]  + \sum_{i} \widetilde{h}^{0}_{i} \hat{a}^{\dagger}_{i}\hat{a}_{i} \\
        = & \sum_{ij} 
        \Big[ t_{ij}^{--} \hat{a}_{i} \hat{a}_{j} + t_{ij}^{-+} \hat{a}_{i}\hat{a}_{j}^{\dagger} + t_{ij}^{+-} \hat{a}_{i}^{\dagger} \hat{a}_{j} + t_{ij}^{++} \hat{a}^{\dagger}_{i} \hat{a}^{\dagger}_{j} \Big] 
        + \sum_{i} \mu_{i} \hat{a}^{\dagger}_{i} \hat{a}_{i}, 
    \end{split}
  $$ where the (effective) chemical potential $\mu_{i}$ is
$$  
    \mu_{i} = \widetilde{h}^{0}_{i} - \sum_{j \in \mathsf{L}(i)} S_{j} \widetilde{J}^{00}_{ij} = \frac{\partial E_{\rm cl}}{\partial \mathbf{S}_{i}} \cdot \hat{\mathbf{e}}_{i}^{0}.
  $$

We assume translational invariance. In general, the spin system has
magnetic sublattices $\sigma = a, b, c, \dots$. The bosonic annihilation
operator $\hat{a}_{i}$ is defined with two indices $i = (i, \sigma)$,
where $\sigma$ denotes the magnetic sublattice and $i$ is the unit-cell
index. For example, $\hat{b}_{(i, a)} = \hat{a}_{i}$ and
$\hat{b}_{(i, b)} = \hat{b}_{i}$. This allows the operators to be
transformed into their momentum-space representations:
$$  
    \hat{b}_{i\sigma} = \frac{1}{\sqrt{L}} \sum_{\mathbf{k} \in \mathrm{FBZ}} e^{i \mathbf{k} \cdot \mathbf{r}_{i\sigma}} \hat{b}_{\mathbf{k}\sigma},
  $$ where $L$ is the number of unit cells in the system and
FBZ denotes the first Brillouin zone. Then, the Hamiltonian becomes:
$$\begin{align}
    H_{2} 
    = & \sum_{\mathbf{k}} 
    \Big[ \sum_{\mathsf{L}(i\sigma, j\rho)}  
        t^{+-}_{ij} e^{-i \mathbf{k} \cdot \bm{\delta}_{ij}} \hat{a}_{\mathbf{k} \sigma}^{\dagger}          \hat{a}_{ \mathbf{k} \rho }^{\phantom \dagger}   +
        t^{-+}_{ij} e^{ i \mathbf{k} \cdot \bm{\delta}_{ij}} \hat{a}_{\mathbf{k} \sigma}^{\phantom \dagger} \hat{a}_{ \mathbf{k} \rho }^{\dagger}  +
        t^{++}_{ij} e^{-i \mathbf{k} \cdot \bm{\delta}_{ij}} \hat{a}_{\mathbf{k} \sigma}^{\dagger}          \hat{a}_{-\mathbf{k} \rho }^{\dagger} + t^{--}_{ij} e^{ i \mathbf{k} \cdot \bm{\delta}_{ij}} \hat{a}_{\mathbf{k} \sigma}^{\phantom \dagger} \hat{a}_{-\mathbf{k} \rho }^{\phantom \dagger} \Big] 
    + \sum_{\mathbf{k}, \sigma} \mu_{\sigma} \hat{a}_{\mathbf{k}\sigma}^{\dagger} \hat{a}_{\mathbf{k}\sigma}^{\phantom\dagger} \nonumber \\
    = & \frac{1}{2} \sum_{\mathbf{k} \in \mathrm{FBZ}} 
    \begin{pmatrix}
        \psi_{  \mathbf{k}}^{\dagger} &  
        \psi_{-\mathbf{k}}
    \end{pmatrix}
    \begin{pmatrix}
        \mathsf{A}_{\mathbf{k}}          & \mathsf{B}_{\mathbf{k}} \\
        \mathsf{B}_{\mathbf{k}}^{\dagger}& \mathsf{A}_{-\mathbf{k}}^{*}
    \end{pmatrix}
    \begin{pmatrix}
        \psi_{  \mathbf{k}} \\
        \psi_{-\mathbf{k}}^{\dagger} 
    \end{pmatrix}
    + \frac{1}{2} \sum_{\mathbf{k} \in \mathrm{FBZ}}  C_{\mathbf{k}},
\end{align}$$ where
$\psi^{\dagger}_{\mathbf{k}} = (a^{\dagger}_{\mathbf{k}}, b^{\dagger}_{\mathbf{k}}, \cdots)$
is a vector of bosonic creation operators for each sublattice, and the
constant $C_{\mathbf{k}}$ follows from the normal ordering
$\hat{a}_{\mathbf{k}\rho}^{\dagger} \hat{a}_{\mathbf{k}\sigma} = \frac{1}{2} (\hat{a}_{\mathbf{k}\rho}^{\dagger} \hat{a}_{ \mathbf{k}\sigma} + \hat{a}_{ \mathbf{k}\rho} \hat{a}_{\mathbf{k}\sigma}^{\dagger} ) - \frac{1}{2} \delta_{\rho\sigma}$.
$$\begin{align}
    \sum_{\mathbf{k} \in \mathrm{FBZ}} C_{\mathbf{k}} 
    = & \sum_{\mathbf{k} \in \mathrm{FBZ}} \Big[
    \sum_{\mathsf{L}(i\sigma,j\rho)} \big( t^{-+}_{ij} e^{i \mathbf{k} \cdot \bm{\delta}_{ij}} - t^{+-}_{ij} e^{-i \mathbf{k} \cdot \bm{\delta}_{ij}}\big) \delta_{\rho\sigma}    - \sum_{\sigma} \mu_{\sigma} \Big]
    = - L \sum_{\sigma = a, b, \cdots} \mu_{\sigma} \nonumber \\
    = & L \sum_{\sigma = a, b, \cdots} \Big( \sum_{\rho \in \mathsf{L}(\sigma)} S_{\rho} \widetilde{J}_{\rho\sigma}^{00} - \widetilde{h}_{\sigma}^{0} \Big) 
    = \sum_{i} \frac{\partial E_{\rm cl}}{\partial \mathbf{S}_{i}} \cdot \hat{\mathbf{e}}_{i}^{0} 
\end{align}$$ In the first line, the first term always vanishes if
$\rho \neq \sigma$. Even when $\rho = \sigma$ for hopping within the
same sublattice but between different unit cells, these terms vanish in
the summation over the FBZ. Note that the constant term can be expressed
as
$\frac{1}{2} \sum_{\mathbf{k} \in \mathrm{FBZ}}  C_{\mathbf{k}}  = - \frac{1}{4} \Tr[\mathsf{H}_{\mathbf{k}}]$
and $$  
    \frac{1}{2} \sum_{\mathbf{k} \in \mathrm{FBZ}}  C_{\mathbf{k}} 
    = \frac{1}{2} \sum_{i} \frac{\partial E_{\rm cl}}{\partial \mathbf{S}_{i}} \cdot \hat{\mathbf{e}}_{i}^{0} 
    = \frac{1}{2} \sum_{i} \Big( \widetilde{h}_{i}^{0} - \sum_{j \in \mathsf{L}(i)} S_{j} \cdot \widetilde{J}_{ji}^{00} \Big)
    \xrightarrow{ S_{i} = S}  
    \frac{1}{S} \big( E_{\rm exc} + \frac{1}{2}  E_{\rm zm} \big),
  $$ where
$E_{\rm exc} = \sum_{ij} \mathbf{S}_i \cdot \mathbf{J}_{ij} \cdot \mathbf{S}_j$
is the exchange energy and
$E_{\rm zm} = - \sum_{i} \mathbf{h}_{i} \cdot \mathbf{S}_{i}$ is the
Zeeman energy. This is the reason why the quantum correction reduces the
classical energy. Clearly, when $\mathbf{h} = 0$, we have
$$  
    \hat{H} = S(S+1) E_{\rm cl} + S H_{2} + S^{1/2} H_{3} + \cdots .
  $$

To summarize, the LSWT Hamiltonian can be expressed in the following
form: $$\begin{align}
    H^{(2)} 
    = & \frac{1}{2}\sum_{\mathbf{k}} \Big(     
        \Psi_{\mathbf{k}}^\dagger \hat{\mathsf{H}}_{\mathbf{k}}^{\phantom{\dagger}} \Psi_{\mathbf{k}}^{\phantom{\dagger}} 
    - \Tr [\mathsf{A}_{\mathbf{k}}] \Big)
\end{align}$$ where
$\Psi^\dagger_{\mathbf{k}}=\left( a^\dagger_{\mathbf{k}}, b^\dagger_{\mathbf{k}},\dots, a^{\phantom \dagger}_{-\mathbf{k}},b^{\phantom \dagger}_{-\mathbf{k}},\dots\right)$
is a vector of length $2 M_s$, with $M_s$ being the number of magnetic
sublattices, and $\hat{\mathsf{H}}_{\mathbf{k}}$ is a $2M_s\times 2M_s$
matrix given by: $$  
    \hat{\mathsf{H}}_{\mathbf{k}} = 
    \begin{pmatrix}
        \hat{\mathsf{A}}^{\phantom \dagger}_{\mathbf{k}} &  \hat{\mathsf{B}}^{\phantom \dagger}_{\mathbf{k}}\\[0.5ex] 
        \hat{\mathsf{B}}^\dagger_{\mathbf{k}}  & \hat{\mathsf{A}}^*_{-\mathbf{k}}
    \end{pmatrix},
  $$ where
$\hat{\mathsf{A}}^{\dagger}_{\mathbf{k}} = \hat{\mathsf{A}}^{\phantom \dagger}_{\mathbf{k}}$
are Hermitian, which follows from the Hermiticity of the Hamiltonian,
and
$\hat{\mathsf{B}}^{\phantom *}_{-\mathbf{k}} = \hat{\mathsf{B}}^{\rm T}_{\mathbf{k}}$,
which follows from the bosonic commutation relations.

Finally, we provide explicit expressions for $\mathsf{H}_{\mathbf{k}}$
for a two-sublattice system $(\rho, \sigma) = (a, b)$ for the reader's
convenience. Let us first consider the case where $a \neq b$, which can
be written in matrix form as:

[(SJ: I think $t^{--}_{ij}$ should be replaced by $(t^{--}_{ij})^{*}$ in
equation 46, to satisfy the constraint on $B$,
$\hat{\mathsf{B}}^{\phantom *}_{-\mathbf{k}} = \hat{\mathsf{B}}^{\rm T}_{\mathbf{k}}$)
]{style="color: blue"}

[ (SP): I agree and this is a typo. The code is implemented without this
typo. ]{style="color: red"} $$  
    \mathsf{A}_{\mathbf{k}} = 
    \begin{pmatrix}
        \widetilde{h}_{a} - \sum_{\mathsf{L}(i,j)} \widetilde{J}_{ij}^{00} S_{b} \quad
        & \sum_{\mathsf{L}(i,j)}t^{+-}_{ij} e^{-i \mathbf{k} \cdot \bm{\delta}_{ij}} \\[0.5em]
        \sum_{\mathsf{L}(i,j)}t^{-+}_{ij} e^{i \mathbf{k} \cdot \bm{\delta}_{ij}} \quad &
        \widetilde{h}_{b} - \sum_{\mathsf{L}(i,j)} \widetilde{J}_{ij}^{00} S_{a}
    \end{pmatrix},
    \quad  \mathsf{B}_{\mathbf{k}} =
    \begin{pmatrix}
        0   & \sum_{\mathsf{L}(i,j)}t^{++}_{ij} e^{-i \mathbf{k} \cdot \bm{\delta}_{ij}} \\[0.5em]
        \sum_{\mathsf{L}(i,j)}t^{++}_{ij} e^{i \mathbf{k} \cdot \bm{\delta}_{ij}} &    0
    \end{pmatrix},
  $$ If the two sublattices are identical ($a = b$), the
matrices in the Hamiltonian are given as: [(SJ: I think the factor 2 in
$A_k$ (in equation 47) is unnecessary; the chemical potential is
doublely counted for $A_k$ and $A_{-k}$, and divided by 2 in total LSWT
Hamiltonian )]{style="color: blue"} $$  
    \begin{split}
        \mathsf{A}_{\mathbf{k}} 
        = & 2 \Big( \widetilde{h}_{a} - \sum_{\mathsf{L}(ij)} \widetilde{J}_{ij}^{00} S_{a} \Big)
        + \sum_{\mathsf{L}(i,j)}(t_{ij}^{+-} e^{-i \mathbf{k} \cdot \bm{\delta}_{ij}} + t^{-+}_{ij} e^{+i \mathbf{k} \cdot \bm{\delta}_{ij}}), \\
        \mathsf{B}_{\mathbf{k}} 
        = & \sum_{\mathsf{L}(i,j)}t^{++}_{ij} (e^{-i \mathbf{k} \cdot \bm{\delta}_{ij}} + e^{+i \mathbf{k} \cdot \bm{\delta}_{ij}})
        = 2 \sum_{\mathsf{L}(i,j)}t^{++}_{ij} \cos ( \mathbf{k} \cdot \bm{\delta}_{ij}) 
    \end{split}
  $$ This momentum-space formulation simplifies the analysis
of bosonic excitations and provides a basis for studying dispersion
relations and system stability.

## Diagonalization of Quadratic Boson Hamiltonian

In this section, we discuss the diagonalization of the boson
Hamiltonian. The quadratic boson Hamiltonian can be diagonalized by the
Bogoliubov transformation. Consider the following Bogoliubov
transformation: $$  
    \hat{b}_{\mathbf{k}, \mu} 
    = \big[ \mathsf{P}_{\mathbf{k}}^{\phantom \dagger} \hat{\bm \beta}_{\mathbf{k}}^{\phantom \dagger} + \mathsf{Q}_{-\mathbf{k}}^{\phantom \dagger} \hat{\bm \beta}_{-\mathbf{k}}^{\dagger} \big]_{\mu}
    = \sum_{\nu = 1}^{M_s} [\mathsf{P}_{\mathbf{k}}]_{\mu\nu}^{\phantom \dagger} \hat{\beta}_{\mathbf{k}, \nu}^{\phantom \dagger} + [\mathsf{Q}_{-\mathbf{k}}]_{\mu\nu}^{\phantom \dagger} \hat{\beta}_{-\mathbf{k}, \nu}^{\dagger},
  $$ where the indices $\mu, \nu$ are defined over
$\mu, \nu = 1, \cdots, M_s$. In order to preserve the bosonic
commutation relations, we require the following identities:
$$\begin{align*}
    [\hat{b}_{  \mathbf{k}, \mu}^{\phantom \dagger}, \hat{b}_{  \mathbf{q}, \nu}^{\dagger}] 
    = & \big[ \mathsf{P}_{\mathbf{k}}\mathsf{P}_{\mathbf{k}}^{\dagger} - \mathsf{Q}_{-\mathbf{k}}\mathsf{Q}_{-\mathbf{k}}^{\dagger} \big]_{\mu\nu} \delta_{\mathbf{k},\mathbf{q}}
    = + \delta_{\mu,\nu}    \delta_{\mathbf{k}, \mathbf{q}} ,
    \qquad \quad
    [\hat{b}_{  \mathbf{k}, \mu}, \hat{b}_{-\mathbf{q}, \nu}] 
    = \big[ \mathsf{P}_{\mathbf{k}}\mathsf{Q}_{\mathbf{k}}^{\mathrm{T}} - \mathsf{Q}_{-\mathbf{k}}\mathsf{P}_{-\mathbf{k}}^{\mathrm{T}} \big]_{\mu\nu} \delta_{\mathbf{k},\mathbf{q}} = 0, \\
    [\hat{b}_{-\mathbf{k}, \mu}^{\dagger}, \hat{b}_{-\mathbf{q}, \nu}^{\phantom \dagger}] 
    = & \big[ \mathsf{P}_{-\mathbf{k}}^{*}\mathsf{P}_{-\mathbf{k}}^{\mathrm{T}} - \mathsf{Q}_{\mathbf{k}}^{*}\mathsf{Q}_{\mathbf{k}}^{\mathrm{T}} \big]_{\mu\nu} \delta_{\mathbf{k},\mathbf{q}}
    = - \delta_{\mu,\nu}    \delta_{\mathbf{k}, \mathbf{q}}, 
    \qquad \quad
    [\hat{b}_{-\mathbf{k}, \mu}^{\dagger}, \hat{b}_{  \mathbf{q}, \nu}^{\dagger}] 
    = \big[ \mathsf{Q}_{\mathbf{k}}^{*}\mathsf{P}_{\mathbf{k}}^{\dagger} - \mathsf{P}_{-\mathbf{k}}^{*}\mathsf{Q}_{-\mathbf{k}}^{\dagger} \big]_{\mu\nu} \delta_{\mathbf{k},\mathbf{q}}
    = 0.
\end{align*}$$ These identities can be represented in a compact matrix
form: $$  
    \begin{pmatrix}
       \mathsf{P}_{\mathbf{k}} & \mathsf{Q}_{-\mathbf{k}} \\ 
       \mathsf{Q}_{\mathbf{k}}^{*} & \mathsf{P}_{-\mathbf{k}}^{*}
    \end{pmatrix}
    \begin{pmatrix}
        1 & 0 \\ 0 & -1
    \end{pmatrix}
    \begin{pmatrix}
       \mathsf{P}_{\mathbf{k}}^{\dagger} & \mathsf{Q}_{\mathbf{k}}^{\rm T} \\ 
       \mathsf{Q}_{-\mathbf{k}}^{\dagger} & \mathsf{P}_{-\mathbf{k}}^{\rm T}
    \end{pmatrix} 
    = 
    \begin{pmatrix} 
        \mathsf{P}_{\mathbf{k}}\mathsf{P}_{\mathbf{k}}^{\dagger} - \mathsf{Q}_{-\mathbf{k}}\mathsf{Q}_{-\mathbf{k}}^{\dagger} & 
        \mathsf{P}_{\mathbf{k}}\mathsf{Q}_{\mathbf{k}}^{\rm T}   - \mathsf{Q}_{-\mathbf{k}}\mathsf{P}_{-\mathbf{k}}^{\rm T}     \\
        \mathsf{Q}_{\mathbf{k}}^{*}\mathsf{P}_{\mathbf{k}}^{\dagger} - \mathsf{P}_{-\mathbf{k}}^{*}\mathsf{Q}_{-\mathbf{k}}^{\dagger} & 
            \mathsf{Q}_{\mathbf{k}}^{*}\mathsf{Q}_{\mathbf{k}}^{\rm T} - \mathsf{P}_{-\mathbf{k}}^{*}\mathsf{P}_{-\mathbf{k}}^{\rm T}
    \end{pmatrix}
    = 
    \begin{pmatrix}
        1 & 0 \\ 0 & -1
    \end{pmatrix}
  $$

By adopting the above convention (i.e., the transformation of
$\hat{\bm b}_{\mathbf{k}} \rightarrow \hat{\bm b}^{\dagger}_{-\mathbf{k}}$),
the (para)-unitary transformation matrix $\mathsf{T}_{\mathbf{k}}$ has
the following form: $$  
    \Psi_{\mathbf{k}} = \mathsf{T}_{\mathbf{k}} \widetilde{\Psi}_{\mathbf{k}}: \quad
    \begin{pmatrix}
        \hat{\bm b}_{\mathbf{k}} \\
        \hat{\bm b}_{-\mathbf{k}}^{\dagger}
    \end{pmatrix}
    =
    \begin{pmatrix}
        \mathsf{P}_{\mathbf{k}}       & \mathsf{Q}_{-\mathbf{k}} \\ 
        \mathsf{Q}_{\mathbf{k}}^{*}   & \mathsf{P}_{-\mathbf{k}}^{*}
    \end{pmatrix}
    \begin{pmatrix}
        \hat{\bm \beta}_{ \mathbf{k}} \\
        \hat{\bm \beta}_{-\mathbf{k}}^{\dagger}
    \end{pmatrix}.
  $$ Suppose the unitary transformation
$\mathsf{T}_{\mathbf{k}}$ diagonalizes the Hamiltonian matrix
$\mathsf{H}_{\mathbf{k}}$: $$  
    \mathsf{T}_{\mathbf{k}}^{\dagger} \mathsf{H}_{\mathbf{k}}^{\phantom \dagger} \mathsf{T}_{\mathbf{k}} = \mathsf{E}_{\mathbf{k}}: \qquad
        \begin{pmatrix}
        \mathsf{P}_{\mathbf{k}}^{\dagger}     & \mathsf{Q}_{  \mathbf{k}}^{\rm T} \\[0.3em] 
        \mathsf{Q}_{-\mathbf{k}}^{\dagger}    & \mathsf{P}_{- \mathbf{k}}^{\rm T}
    \end{pmatrix}
    \begin{pmatrix}
        \mathsf{A}^{\phantom \dagger}_{\mathbf{k}}      &   \mathsf{B}^{\phantom \dagger}_{\mathbf{k}}\\[0.3em] 
        \mathsf{B}^\dagger_{\mathbf{k}}                 &   \mathsf{A}^*_{-\mathbf{k}}
    \end{pmatrix}
    \begin{pmatrix}
        \mathsf{P}_{\mathbf{k}}       & \mathsf{Q}_{-\mathbf{k}} \\[0.3em] 
        \mathsf{Q}_{\mathbf{k}}^{*}   & \mathsf{P}_{-\mathbf{k}}^{*}
    \end{pmatrix}
    =
    \begin{pmatrix}
        \mathsf{E}_{\mathbf{k}} & 0 \\ 0 & \widetilde{\mathsf{E}}_{-\mathbf{k}}
    \end{pmatrix},
  $$ The two diagonal matrices $\mathsf{E}_{\mathbf{k}}$ and
$\widetilde{\mathsf{E}}_{-\mathbf{k}}$ are real-valued, and their
explicit forms are written as: $$\begin{align*}
    \mathsf{E}_{\mathbf{k}} = & 
    \mathsf{P}_{\mathbf{k}}^{\dagger} \mathsf{A}_{\mathbf{k}} \mathsf{P}_{\mathbf{k}} + 
    \mathsf{P}_{\mathbf{k}}^{\dagger} \mathsf{B}_{\mathbf{k}} \mathsf{Q}_{\mathbf{k}}^{*} + 
    \mathsf{Q}_{\mathbf{k}}^{\rm T} \mathsf{B}_{\mathbf{k}}^{\dagger} \mathsf{P}_{\mathbf{k}} + 
    \mathsf{Q}_{\mathbf{k}}^{\rm T} \mathsf{A}_{-\mathbf{k}}^{*} \mathsf{Q}_{\mathbf{k}}^{*},
    \\
    \widetilde{\mathsf{E}}_{-\mathbf{k}} = & 
    \mathsf{Q}_{-\mathbf{k}}^{\dagger} \mathsf{A}_{\mathbf{k}} \mathsf{Q}_{-\mathbf{k}} + 
    \mathsf{Q}_{-\mathbf{k}}^{\dagger} \mathsf{B}_{\mathbf{k}} \mathsf{P}_{-\mathbf{k}}^{*} + 
    \mathsf{P}_{-\mathbf{k}}^{\rm T} \mathsf{B}_{\mathbf{k}}^{\dagger} \mathsf{Q}_{-\mathbf{k}} + 
    \mathsf{P}_{-\mathbf{k}}^{\rm T} \mathsf{A}_{-\mathbf{k}}^{*} \mathsf{P}_{-\mathbf{k}}^{*}.
\end{align*}$$ Note that the two diagonal matrices can be mapped to each
other under conjugation and momentum inversion. Thus, we obtain the
energy bands ($E_{\mathbf{k}}$ and $E_{-\mathbf{k}}$) when we
diagonalize the quadratic boson Hamiltonian via Colpa's
method [@Colpa1978Diagonalization]: $$  
    \widetilde{\mathsf{E}}_{-\mathbf{k}} 
    \xrightarrow[\mathbf{k} \rightarrow -\mathbf{k}]{\rm conj} 
    \widetilde{\mathsf{E}}_{\mathbf{k}}^{*}:  \qquad
    \widetilde{\mathbf{E}}_{\mathbf{k}}^{*} =
    \mathsf{P}_{\mathbf{k}}^{\dagger}   \mathsf{A}_{\mathbf{k}}             \mathsf{P}_{\mathbf{k}}   +
    \mathsf{P}_{\mathbf{k}}^{\dagger}   \mathsf{B}^{\rm T}_{-\mathbf{k}}    \mathsf{Q}_{\mathbf{k}}^{*} +
    \mathsf{Q}_{\mathbf{k}}^{\rm T}     \mathsf{B}^{*}_{-\mathbf{k}}        \mathsf{P}_{\mathbf{k}}     +
    \mathsf{Q}_{\mathbf{k}}^{\rm T}     \mathsf{A}_{-\mathbf{k}}^{*}        \mathsf{Q}_{\mathbf{k}}^{*}
    = \mathbf{E}_{\mathbf{k}},
  $$ where
$\mathsf{B}^{\rm T}_{-\mathbf{k}} = \mathsf{B}_{\mathbf{k}}^{\phantom T}$
and $\mathsf{B}^{*}_{-\mathbf{k}} = \mathsf{B}_{\mathbf{k}}^{\dagger}$.
Since $\mathsf{E}_{\mathbf{k}}$ is a *real* diagonal matrix, we conclude
$\widetilde{\mathsf{E}}_{\mathbf{k}} = \mathsf{E}_{\mathbf{k}}$. Thus,
the canonical form of the quadratic bosonic Hamiltonian reads:
$$\begin{align}
    \hat{H}_{2}
    = & \frac{1}{2} \sum_{\mathbf{k}} 
    \Big( (\widetilde{\Psi}^{\dagger}_{\mathbf{k}} \mathsf{T}_{\mathbf{k}}^{\dagger}) \mathsf{H}_{\mathbf{k}}^{\phantom \dagger} (\mathsf{T}_{\mathbf{k}}^{\phantom \dagger} \widetilde{\Psi}_{\mathbf{k}}^{\phantom \dagger} )- \Tr [\mathsf{A}_{\mathbf{k}}] \Big) 
    = \frac{1}{2} \sum_{\mathbf{k}} 
    \Big( \widetilde{\Psi}^{\dagger}_{\mathbf{k}} \mathsf{E}_{\mathbf{k}} \widetilde{\Psi}_{\mathbf{k}} - \Tr [\mathsf{A}_{\mathbf{k}}] \Big) \nonumber \\
    = & \frac{1}{2} \sum_{\mathbf{k}} 
    \begin{pmatrix}
        \hat{\bm \beta}_{\mathbf{k}} \\
        \hat{\bm \beta}_{-\mathbf{k}}^{\dagger}
    \end{pmatrix}^{\dagger}
    \begin{pmatrix}
        \mathbf{E}_{\mathbf{k}} & 0 \\ 0 & \widetilde{\mathbf{E}}_{-\mathbf{k}}
    \end{pmatrix}
    \begin{pmatrix}
        \hat{\bm \beta}_{\mathbf{k}} \\
        \hat{\bm \beta}_{-\mathbf{k}}^{\dagger}
    \end{pmatrix} 
    - \frac{1}{2} \Tr [\mathsf{A}_{\mathbf{k}}]
    \nonumber \\
    = & \frac{1}{2} \sum_{\mathbf{k}, \sigma} 
    \big( 
        E_{\mathbf{k}, \sigma} \hat{\beta}^{\dagger}_{\mathbf{k}, \sigma}   \hat{\beta}^{\phantom \dagger}_{\mathbf{k}, \sigma} +
        E_{-\mathbf{k}, \sigma} \hat{\beta}^{\phantom \dagger}_{-\mathbf{k}, \sigma} \hat{\beta}^{\dagger}_{-\mathbf{k}, \sigma}
    \big) 
    - \frac{1}{2} \Tr [\mathsf{A}_{\mathbf{k}}] \\
    = & \sum_{\mathbf{k}, \sigma} E_{\mathbf{k}, \sigma} \hat{\beta}_{\mathbf{k}, \sigma}^{\dagger} \hat{\beta}_{\mathbf{k}, \sigma}^{\phantom \dagger} 
    + \frac{1}{2} \sum_{\mathbf{k}, \sigma} \big( E_{\mathbf{k}, \sigma} - \Tr [\mathsf{A}_{\mathbf{k}}] \big),
\end{align}$$ where $E_{\mathbf{k}} \ge 0$ for all
$\mathbf{k} \in \mathrm{BZ}$. The constant term is called the
"zero-point energy," as it represents the ground state energy of the
quantum system at zero temperature (the Gibbs state at $T=0$). This
gives the quantum correction to the classical energy in the LSWT
framework: $$  
\label{eq: def. of zero point energy}
    E_{0} = E_{\rm cl} + \frac{1}{2} \sum_{\mathbf{k} \in \mathrm{FBZ}} \Big( \sum_{\sigma=a,b,\cdots} E_{\mathbf{k}, \sigma} -  \Tr[\mathsf{A}_{\mathbf{k}}] \Big),
  $$ and the energy density per spin site is:
$$  
    \mathcal{E} 
    = \frac{E_{0}}{N} 
    = \frac{1}{N} \Big[ E_{\rm cl} + \frac{1}{2} \sum_{\mathbf{k} \in \mathrm{FBZ}} \Big( \sum_{\sigma=a,b,\cdots} E_{\mathbf{k}, \sigma} -  \Tr[\mathsf{A}_{\mathbf{k}}] \Big) \Big],
  $$ where $N=Lm_{s}$ is the total number of spin sites in
the system, with $L$ being the number of unit cells and $m_s$ the number
of magnetic sublattices per unit cell.

## Paraunitary Diagonalization

Now, the remaining problem is to find a paraunitary matrix
$\mathsf{T}_{\kvec}$ (we will suppress the $\kvec$ subscript in this
section for clarity, so $\mathsf{H}_{\kvec} \rightarrow \mathsf{H}$,
etc.) such that $$  
    \mathsf{T}^{\dagger}\mathsf{HT} = \mathsf{E}, \quad \text{with}
    \quad \mathsf{TJT}^{\dagger} = \mathsf{J},
  $$ where $\mathsf{E}$ is a diagonal matrix containing the
magnon energies, and $\mathsf{J}$ is the symplectic metric tensor. The
paraunitary transformation $\mathsf{T}$ exists if and only if
$\mathsf{H}$ is positive definite.

The explicit form of the paraunitary matrix is given by
$$  
    \mathsf{T} = (\mathsf{K}^{\dagger})^{-1} \mathsf{V} \mathsf{(\Lambda J)}^{1/2},
  $$ where $\mathsf{K}$ comes from the Cholesky
decomposition of the Hamiltonian. Since the Hamiltonian matrix
$\mathsf{H}$ is positive definite, the Cholesky decomposition exists,
satisfying: $$  
    \mathsf{H} = \mathsf{KK}^{\dagger},
  $$ and $\mathsf{V}$ is the unitary matrix that
diagonalizes $\mathsf{K}^{\dagger} \mathsf{JK}$: $$  
    \mathsf{V \Lambda V}^{\dagger} = \mathsf{K}^{\dagger} \mathsf{JK},
  $$ where $\mathsf{\Lambda}$ is a diagonal matrix.

We will now verify that this construction satisfies the required
paraunitary conditions. First, let us check
$\mathsf{TJT}^{\dagger} = \mathsf{J}$: $$\begin{align*}
    \mathsf{TJT}^{\dagger} &= 
    \big( (\mathsf{K}^{\dagger})^{-1} \mathsf{V} \mathsf{(\Lambda J)}^{1/2} \big) \mathsf{J} \big( \mathsf{(\Lambda J)}^{1/2} \big)^{\dagger} \mathsf{V}^{\dagger} \mathsf{K}^{-1} \\
    &= (\mathsf{K}^{\dagger})^{-1} \mathsf{V} \mathsf{(\Lambda J)}^{1/2} \mathsf{J} \mathsf{(J\Lambda)}^{1/2} \mathsf{V}^{\dagger} \mathsf{K}^{-1} \\
    &= (\mathsf{K}^{\dagger})^{-1} \mathsf{V} \mathsf{\Lambda} \mathsf{V}^{\dagger} \mathsf{K}^{-1} \\
    &= (\mathsf{K}^{\dagger})^{-1} (\mathsf{K}^{\dagger} \mathsf{JK}) \mathsf{K}^{-1} = \mathsf{J}
\end{align*}$$ where we can also verify that
$\mathsf{T}^{\dagger} \mathsf{JT} = \mathsf{J}$.

Next, let us check that $\mathsf{T}$ diagonalizes the Hamiltonian as
required: $$\begin{align*}
    \mathsf{T}^{\dagger} \mathsf{HT}
    &= \big( \mathsf{(\Lambda J)}^{1/2} \big)^{\dagger} \mathsf{V}^{\dagger} \mathsf{K}^{-1} \cdot \mathsf{KK}^{\dagger} \cdot (\mathsf{K}^{\dagger})^{-1} \mathsf{V} \mathsf{(\Lambda J)}^{1/2} \\
    &= \mathsf{(J\Lambda)}^{1/2} \mathsf{V}^{\dagger} \mathsf{V} \mathsf{(\Lambda J)}^{1/2} \\
    &= \mathsf{(J\Lambda)}^{1/2} \mathsf{(\Lambda J)}^{1/2} \\
    &= \mathsf{J\Lambda} = \mathsf{\Lambda J} = \mathsf{E}
\end{align*}$$ where we identify the diagonal energy matrix as
$\mathsf{E} = \mathsf{J\Lambda} = \mathsf{\Lambda J}$. This equality
holds because both $\mathsf{J}$ and $\mathsf{\Lambda}$ are diagonal
matrices, with $\mathsf{J}$ having half positive and half negative
entries, and $\mathsf{\Lambda}$ arranged to ensure that $\mathsf{E}$ is
always positive definite.

[^1]: If we count $i$ and $j$ as independent variables, we change the
    exchange Hamiltonian:
    $\sum_{ij} \sum_{\alpha, \beta} S^{\alpha}_{i} J_{ij}^{\alpha\beta}  S^{\beta}_{j} \rightarrow \frac{1}{2} \sum_{ij} \sum_{\alpha, \beta} S^{\alpha}_{i} J_{ij}^{\alpha\beta} S^{\beta}_{j}$.

[^2]: If we count the indices independently, this becomes
    $\sum_{i} \Big( 2 \sum_{j} \mathbf{S}_{j} \cdot \mathbf{J}_{ji} - \mathbf{h}_{i} \Big) \cdot \delta \hat{\mathbf{S}}_{i}$.
