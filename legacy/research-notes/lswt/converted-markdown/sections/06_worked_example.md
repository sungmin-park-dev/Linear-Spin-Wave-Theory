---
bibliography:
- Setting/citation.bib
---

# Example: Solving spin system with linear spin wave theory

In this example, we solve a quadratic boson Hamiltonian. Consider the
following Hamiltonian in momentum space, rewritten in Nambu form:
$$
\begin{equation}
    H = \sum_{\mathbf{k}} A_{\mathbf{k}} a^{\dagger}_{\mathbf{k}} a_{\mathbf{k}} + \frac{1}{2} \Big( B_{\mathbf{k}} a^{\dagger}_{\mathbf{k}} a^{\dagger}_{-\mathbf{k}} + B_{\mathbf{k}}^{*} a_{\mathbf{k}} a_{-\mathbf{k}} \Big) 
    = \frac{1}{2} \sum_{\mathbf{k}}
    \begin{pmatrix}
        a^{\dagger}_{\mathbf{k}} & a_{-\mathbf{k}}
    \end{pmatrix}
    \begin{pmatrix}
        A_{\mathbf{k}} & B_{\mathbf{k}} \\
        B_{\mathbf{k}}^{*} & A_{\mathbf{k}}
    \end{pmatrix}
    \begin{pmatrix}
        a_{\mathbf{k}} \\
        a^{\dagger}_{-\mathbf{k}}
    \end{pmatrix},
\end{equation}
$$ where $B_{\mathbf{k}}$ is real without loss of
generality. This assumption holds because we can redefine the bosonic
operator with a phase factor
$a_{\mathbf{k}} \to a_{\mathbf{k}} e^{-i\phi}$, where
$B_{\mathbf{k}} = e^{i\phi} |B_{\mathbf{k}}|$, making $B_{\mathbf{k}}$
real by choosing an appropriate $\phi$. Note that the condition to be
Hamiltonian density positive is $$\begin{equation*}
    \lambda(\mathsf{H}_{\mathbf{k}}) = A_{\mathbf{k}} \pm B_{\mathbf{k}} \ge 0, 
    \qquad \rightarrow \qquad
    | A_{\mathbf{k}} | \ge | B_{\mathbf{k}} |.
\end{equation*}$$ This condition corresponds to the positivity of the
the magnon energy
$\omega_{\mathbf{k}} = \sqrt{A_{\mathbf{k}}^{2} - B_{\mathbf{k}}^{2}}$.

To diagonalize the Hamiltonian, we introduce the Bogoliubov
transformation: $$
\begin{equation}
    \begin{pmatrix}
        a_{\mathbf{k}} \\
        a^{\dagger}_{-\mathbf{k}}
    \end{pmatrix}
    =
    \begin{pmatrix}
        \cosh \theta_{\mathbf{k}} & \sinh \theta_{-\mathbf{k}} \\
        \sinh \theta_{\mathbf{k}} & \cosh \theta_{-\mathbf{k}}
    \end{pmatrix}
    \begin{pmatrix}
        \alpha_{\mathbf{k}} \\
        \alpha^{\dagger}_{-\mathbf{k}}
    \end{pmatrix}
\end{equation}
$$ where $\theta_{\mathbf{k}}$ is real and even under
$\mathbf{k} \to -\mathbf{k}$, so
$\theta_{\mathbf{k}} = \theta_{-\mathbf{k}}$. The bosonic commutation
relations $[a_{\mathbf{k}}, a^{\dagger}_{\mathbf{k}}] = 1$ and
$[\alpha_{\mathbf{k}}, \alpha^{\dagger}_{\mathbf{k}}] = 1$ must be
preserved. Define the transformation matrix as: $$
\begin{equation}
    \mathsf{T}_{\mathbf{k}} = 
    \begin{pmatrix}
        \cosh \theta_{\mathbf{k}} & \sinh \theta_{\mathbf{k}} \\
        \sinh \theta_{\mathbf{k}} & \cosh \theta_{\mathbf{k}}
    \end{pmatrix}
\end{equation}
$$ Check the commutation relation for the new operators:
$$
\begin{align}
    [\alpha_{\mathbf{k}}, \alpha^{\dagger}_{\mathbf{k}}] 
    &= [\cosh \theta_{\mathbf{k}} a_{\mathbf{k}} + \sinh \theta_{\mathbf{k}} a^{\dagger}_{-\mathbf{k}}, \cosh \theta_{\mathbf{k}} a^{\dagger}_{\mathbf{k}} + \sinh \theta_{\mathbf{k}} a_{-\mathbf{k}}]   \\
    &= \cosh^2 \theta_{\mathbf{k}} [a_{\mathbf{k}}, a^{\dagger}_{\mathbf{k}}] + \sinh^2 \theta_{\mathbf{k}} [a^{\dagger}_{-\mathbf{k}}, a_{-\mathbf{k}}]   \\
    &= \cosh^2 \theta_{\mathbf{k}} \cdot 1 + \sinh^2 \theta_{\mathbf{k}} \cdot (-1)   \\
    &= \cosh^2 \theta_{\mathbf{k}} - \sinh^2 \theta_{\mathbf{k}} = 1
\end{align}
$$ This holds due to the hyperbolic identity, confirming that
the transformation is canonical.

## Diagonalizing the Hamiltonian {#diagonalizing-the-hamiltonian .unnumbered}

Substitute the transformation into the Hamiltonian. First, the conjugate
transformation is: $$
\begin{equation}
    \begin{pmatrix}
        a^{\dagger}_{\mathbf{k}} & a_{-\mathbf{k}}
    \end{pmatrix}
    =
    \begin{pmatrix}
        \alpha^{\dagger}_{\mathbf{k}} & \alpha_{-\mathbf{k}}
    \end{pmatrix}
    \begin{pmatrix}
        \cosh \theta_{\mathbf{k}} & \sinh \theta_{\mathbf{k}} \\
        \sinh \theta_{\mathbf{k}} & \cosh \theta_{\mathbf{k}}
    \end{pmatrix}
\end{equation}
$$ Thus, the Hamiltonian becomes: $$
\begin{equation}
    H = \frac{1}{2} \sum_{\mathbf{k}}
    \begin{pmatrix}
        \alpha^{\dagger}_{\mathbf{k}} & \alpha_{-\mathbf{k}}
    \end{pmatrix}
    T_{\mathbf{k}}
    \begin{pmatrix}
        A_{\mathbf{k}} & B_{\mathbf{k}} \\
        B_{\mathbf{k}}^{*} & A_{\mathbf{k}}
    \end{pmatrix}
    T_{\mathbf{k}}
    \begin{pmatrix}
        \alpha_{\mathbf{k}} \\
        \alpha^{\dagger}_{-\mathbf{k}}
    \end{pmatrix}
\end{equation}
$$ Define the matrix to be diagonalized as:
$$
\begin{equation}
    \mathsf{E}_{\mathbf{k}} = \mathsf{T}_{\mathbf{k}}^{\dagger}
    \begin{pmatrix}
        A_{\mathbf{k}} & B_{\mathbf{k}} \\
        B_{\mathbf{k}}^{*} & A_{\mathbf{k}}
    \end{pmatrix}
    \mathsf{T}_{\mathbf{k}}
    =
    \begin{pmatrix}
        \cosh \theta_{\mathbf{k}} & \sinh \theta_{\mathbf{k}} \\
        \sinh \theta_{\mathbf{k}} & \cosh \theta_{\mathbf{k}}
    \end{pmatrix}
    \begin{pmatrix}
        A_{\mathbf{k}} & B_{\mathbf{k}} \\
        B_{\mathbf{k}}^{*} & A_{\mathbf{k}}
    \end{pmatrix}
    \begin{pmatrix}
        \cosh \theta_{\mathbf{k}} & \sinh \theta_{\mathbf{k}} \\
        \sinh \theta_{\mathbf{k}} & \cosh \theta_{\mathbf{k}}
    \end{pmatrix}
\end{equation}
$$ Since $B_{\mathbf{k}}$ is real,
$B_{\mathbf{k}}^{*} = B_{\mathbf{k}}$. Compute $M_{\mathbf{k}}$:
$$
\begin{equation}
    \mathsf{E}_{\mathbf{k}}
    = 
    \begin{pmatrix}
        A_{\mathbf{k}} \cosh 2\theta_{\mathbf{k}} + B_{\mathbf{k}} \sinh 2\theta_{\mathbf{k}} 
        & A_{\mathbf{k}} \sinh 2\theta_{\mathbf{k}} + B_{\mathbf{k}} \cosh 2\theta_{\mathbf{k}}\\
        A_{\mathbf{k}} \sinh 2\theta_{\mathbf{k}} + B_{\mathbf{k}} \cosh 2\theta_{\mathbf{k}} 
        & A_{\mathbf{k}} \cosh 2\theta_{\mathbf{k}} + B_{\mathbf{k}} \sinh 2\theta_{\mathbf{k}} 
    \end{pmatrix}
    = 
    \begin{pmatrix}
        \omega_{\mathbf{k}}
        & 0 \\ 0 & 
        \omega_{\mathbf{k}}
    \end{pmatrix}
\end{equation}
$$ Thus, we have the diagonalization condition
$$
\begin{equation}
    [\mathsf{E}_{\mathbf{k}}]_{12} = A_{\mathbf{k}} \sinh 2\theta_{\mathbf{k}} + B_{\mathbf{k}} \cosh 2\theta_{\mathbf{k}} = 0,
    \quad \rightarrow \quad
    \tanh 2\theta_{\mathbf{k}} = -\frac{B_{\mathbf{k}}}{A_{\mathbf{k}}}
\end{equation}
$$ The diagonal element $M_{\mathbf{k}}^{(11)}$ gives the
eigenenergy: $$
\begin{equation}
    \omega_{\mathbf{k}} 
    = A_{\mathbf{k}} \cosh 2\theta_{\mathbf{k}} + B_{\mathbf{k}} \sinh 2\theta_{\mathbf{k}} 
    = A_{\mathbf{k}} \cdot \frac{A_{\mathbf{k}}}{\sqrt{A_{\mathbf{k}}^2 - B_{\mathbf{k}}^2}} + B_{\mathbf{k}} \cdot \left(-\frac{B_{\mathbf{k}}}{\sqrt{A_{\mathbf{k}}^2 - B_{\mathbf{k}}^2}}\right) 
    = \sqrt{A_{\mathbf{k}}^2 - B_{\mathbf{k}}^2}.
\end{equation}
$$ To express this in terms of $A_{\mathbf{k}}$ and
$B_{\mathbf{k}}$, use
$\cosh 2\theta_{\mathbf{k}} = \frac{1}{\sqrt{1 - \tanh^2 2\theta_{\mathbf{k}}}}$
and
$\sinh 2\theta_{\mathbf{k}} = \tanh 2\theta_{\mathbf{k}} \cosh 2\theta_{\mathbf{k}}$:
$$\begin{equation*}
    \cosh 2\theta_{\mathbf{k}} = \frac{1}{\sqrt{1 - \left(-B_{\mathbf{k}} / A_{\mathbf{k}} \right)^2}} = \frac{A_{\mathbf{k}}}{\omega_{\mathbf{k}}},  
    \qquad 
    \sinh 2\theta_{\mathbf{k}} = -\frac{B_{\mathbf{k}}}{A_{\mathbf{k}}} \cdot \frac{A_{\mathbf{k}}}{\sqrt{A_{\mathbf{k}}^2 - B_{\mathbf{k}}^2}} = -\frac{B_{\mathbf{k}}}{\omega_{\mathbf{k}}}.
\end{equation*}$$

**Number Expectation in Ground State:** $$
\begin{align}
    \langle n_{\mathbf{k}} \rangle 
    = & \langle a_{\mathbf{k}}^{\dagger} a_{\mathbf{k}} \rangle 
    =  \langle 
    ( \cosh \theta_{\mathbf{k}} \alpha_{\mathbf{k}}^{\dagger} + \sinh \theta_{\mathbf{k}} \alpha_{-\mathbf{k}} ) 
    ( \cosh \theta_{\mathbf{k}} \alpha_{\mathbf{k}} + \sinh \theta_{\mathbf{k}} \alpha_{-\mathbf{k}}^{\dagger} ) 
    \rangle \\
    = & \cosh^{2} \theta_{\mathbf{k}} \langle  \alpha_{\mathbf{k}}^{\dagger} \alpha_{\mathbf{k}} \rangle 
    + \sinh^{2} \theta_{\mathbf{k}} \langle \alpha_{-\mathbf{k}} \alpha_{-\mathbf{k}}^{\dagger} \rangle \
    + \sinh \theta_{\mathbf{k}} \cosh \theta_{\mathbf{k}} 
    \big(\langle \alpha_{-\mathbf{k}}^{\dagger} \alpha_{\mathbf{k}}^{\dagger} \rangle + \langle \alpha_{\mathbf{k}} \alpha_{-\mathbf{k}} \rangle \big) \\
    = & \sinh^{2} \theta_{\mathbf{k}} 
    = \frac{\cosh 2 \theta_{\mathbf{k}} - 1}{2} 
    = \frac{A_{\mathbf{k}} / \omega_{\mathbf{k}} - 1}{2}.
\end{align}
$$

- If
  $\omega_{\mathbf{k}} = \sqrt{A^{2}_{\mathbf{k}} - |B_{\mathbf{k}}|^{2}} \rightarrow 0^{+}$,
  then the number $$
\begin{equation}
          \langle a_{k}^{\dagger} a_{k} \rangle = |v_k|^2 = \frac{A_{\mathbf{k}}/\omega_{\mathbf{k}} - 1}{2} \approx \frac{A_{\mathbf{k}}}{\omega_{\mathbf{k}}} \rightarrow \infty
      
  \end{equation}
$$

- If $B_{\mathbf{k}} = 0$: $$
\begin{equation}
          \langle n_{\mathbf{k}} \rangle = 0
      
  \end{equation}
$$

- If $A_{\mathbf{k}} \rightarrow A_{\mathbf{k}} + \mu$, which corresponds to
  adding on-site chemical potential amount to
  $\mu \sum_{\mathbf{k}} a_{\mathbf{k}}^{\dagger}a_{\mathbf{k}} = \frac{\mu}{2} \sum_{\mathbf{k}} (a_{\mathbf{k}}^{\dagger}a_{\mathbf{k}} + a_{-\mathbf{k}}a^{\dagger}_{-\mathbf{k}} )$
  is the quantum correction $$
\begin{align*}
          E_{\rm qu}^{(\mu)}
          = & \frac{1}{2} \sum_{\mathbf{k}} \Big( \sum_{\mu} E_{\mathbf{k}, \mu} - \Tr[\mathsf{A}_{\mathbf{k}}]\Big)
          = \frac{1}{2}\sum_{\mathbf{k}}  \Big(  ((A_{\mathbf{k}} + \mu)^{2} - B_{\mathbf{k}}^{2})^{1/2} - (A_{\mathbf{k}} + \mu) \Big) \\
          = & \frac{1}{2} \sum_{\mathbf{k}} \big( (A_{\mathbf{k}}^{2} - B_{\mathbf{k}})^{1/2} - A_{\mathbf{k}} \big) 
          + \Big( \frac{A_{\mathbf{k}}}{\omega_{\mathbf{k}}} - 1\Big) \mu 
          + \frac{B_{\mathbf{k}}^{2}}{2\omega^{3}_{\mathbf{k}}} \mu^{2} + \cdots  \\
          = & \sum_{\mathbf{k}} \frac{\omega_{\mathbf{k}}}{2} + \langle a_{\mathbf{k}}^{\dagger} a_{\mathbf{k}} \rangle \mu +  \frac{\langle a_{\mathbf{k}}^{\dagger} a_{\mathbf{k}}^{\dagger} a_{\mathbf{k}} a_{\mathbf{k}} \rangle}{\omega_{\mathbf{k}}} \mu^{2} + \cdots.
      
  \end{align}
$$ $$
\begin{align*}
          \langle n_{\mathbf{k}}^{(\mu)} \rangle - \langle n_{\mathbf{k}}^{(0)} \rangle 
          = & \frac{A_{\mathbf{k}} + \mu - \sqrt{(A_{\mathbf{k}} + \mu)^{2} - B_{\mathbf{k}}^{2}}}{2 \sqrt{(A_{\mathbf{k}} + \mu)^{2} - B_{\mathbf{k}}^{2}} } 
          - \frac{A_{\mathbf{k}} - \sqrt{(A_{\mathbf{k}})^{2} - B_{\mathbf{k}}^{2}}}{2 \sqrt{(A_{\mathbf{k}})^{2} - B_{\mathbf{k}}^{2}} } \\
          = & - \frac{B_{\mathbf{k}}^{2}}{2 \omega^{3}_{\mathbf{k}}} \mu - \frac{3}{4} \frac{A_{\mathbf{k}}B_{\mathbf{k}}^{2}}{\omega_{\mathbf{k}}^{5}} \mu^{2} + \cdots
      
  \end{align}
$$
