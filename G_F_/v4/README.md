# Two-field false-vacuum decay — fluctuation traces (pre-renormalization)

This repository computes the **one-loop fluctuation traces** around the bounce for vacuum decay in a two-field scalar model. They are the raw input to the fluctuation determinant in the semiclassical decay rate.

> **Rendering note.** GitHub renders the LaTeX below natively. In a local VS Code preview you need a math extension (e.g. *Markdown+Math*) to see the equations typeset.

## Physics in brief

The decay rate per unit volume is controlled by the bounce action $S$ and the ratio of fluctuation determinants between the bounce and the false vacuum:

$$ \Gamma \;\propto\; e^{-S}\,\left(\frac{\det \mathcal{M}}{\det \mathcal{M}_{\mathrm{FV}}}\right)^{-1/2}. $$

That determinant ratio splits into partial waves $n$ (degeneracy $(n+1)^2$ on the 3-sphere) and an integral over a spectral parameter $s^2$. For each $n$ this code computes the trace quantity $\bar g_n(s^2)$ on the bounce and subtracts the false-vacuum reference:

$$ -\ln\frac{\det \mathcal{M}}{\det \mathcal{M}_{\mathrm{FV}}} \;\sim\; \int ds^2 \sum_{n\ge 0}\Big[\,\bar g_n(s^2) - \bar g_n^{\mathrm{FV}}(s^2)\,\Big], \qquad \bar g_n \equiv (n+1)^2 I_n. $$

Three regimes need three numerical methods:

| Partial wave | Physical role | Method |
|---|---|---|
| $n=0$ | negative (Coleman) mode | finite differences (FD) |
| $n=1$ | translation zero mode | finite differences (FD) |
| $n\ge 2$ | ordinary modes | Runge–Kutta Green's function (RK) |

**Scope.** This is the *pre-renormalization* stage only. The Born subtractions, counterterms, and the final renormalized $\ln\det$ are computed downstream and are **not** included here.

## Requirements

```
numpy
scipy
sympy
mpmath
matplotlib
cosmoTransitions
```

Install with `pip install -r requirements.txt`.

## Data directory

Point at your data folder once via an environment variable; otherwise the code falls back to a local `./data` (handled by `config.py`, the same convention as `G_F_/v2/`):

```bash
export G_PROJECT_DATA=/path/to/your/data_folder
```

---

## The codes, in pipeline order

### Stage 0 — Model and bounce

#### 1. `potential_git.py`

Defines the two-field scalar potential V(x,y) used in the false-vacuum-decay study, together with its gradient and 2x2 Hessian (mass matrix), all built symbolically with SymPy and then turned into fast NumPy functions. It finds the potential's critical points by Newton-style root-finding of the gradient on a grid of seed points, classifies each as minimum/maximum/saddle from the sign of the Hessian eigenvalues, and lists the true minima (vacua) sorted by energy. Finally it implements a "primed" frame attached to a chosen false vacuum: the field is shifted so the false vacuum sits at the origin, the potential is lifted so V=0 there, and the axes are rotated by the orthogonal matrix that diagonalizes the Hessian at the false vacuum, so that the quadratic (mass) part is diagonal at the origin.

**Key equations**

*Two-field potential*

$$ V(x,y) = a_0 + a_{xy^2}\, x\, y^2 + a_{y^4}\,(y^2-1)^2 + a_{x^2y^2}\, x^2 y^2 + a_{x^4}\,(x^2-1)^2 $$

*Default couplings*

$$ (a_0, a_{xy^2}, a_{y^4}, a_{x^2y^2}, a_{x^4}) = (-0.8,\ 1.0,\ 1.0,\ 0.15,\ 1.0) $$

*Gradient and Hessian (mass matrix)*

$$ \nabla V = \left(\frac{\partial V}{\partial x},\ \frac{\partial V}{\partial y}\right), \qquad H_{ij} = \frac{\partial^2 V}{\partial \phi_i \partial \phi_j} $$

*Critical points and classification*

$$ \nabla V(\phi_*) = 0; \quad \text{minimum if } \mathrm{eig}(H) > 0,\ \text{maximum if } \mathrm{eig}(H) < 0,\ \text{saddle otherwise} $$

*Hessian diagonalization at false vacuum*

$$ H_F = L\,\Lambda\,L^{T}, \qquad L^{T}L = \mathbb{1}, \qquad \Lambda = \mathrm{diag}(\lambda_1,\lambda_2) $$

*Shift / rotate coordinate maps*

$$ \phi = \phi_F + L\,\phi', \qquad \phi' = L^{T}(\phi - \phi_F) $$

*Primed potential, gradient, Hessian*

$$ V'(\phi') = V(\phi_F + L\phi') - V(\phi_F), \qquad \nabla' V' = L^{T}\nabla V, \qquad H'(\phi') = L^{T} H L $$

**Key objects**

- $V(x,y)$ — Two-field quartic potential with quadratic and quartic cross-couplings
- $\phi = (x,y)$ — Two real scalar fields
- $(a_0, a_{xy^2}, a_{y^4}, a_{x^2y^2}, a_{x^4})$ — Five real couplings parameterizing the potential
- $H = \partial^2 V / \partial\phi_i\partial\phi_j$ — 2x2 Hessian / mass matrix of the potential
- $\phi_F$ — False vacuum (chosen local minimum, here the highest-energy one)
- $H_F = L\Lambda L^{T}$ — Hessian at the false vacuum and its orthogonal eigendecomposition
- $L$ — Orthogonal rotation whose columns are eigenvectors of H_F (maps primed to original frame)
- $\phi'$ — Primed coordinates: field shifted to put the false vacuum at the origin and rotated to diagonalize its mass matrix

#### 2. `bounce_git.py`

For each downward false-to-true vacuum pair of a two-field potential, this file computes the Euclidean O(4) bounce: the radially symmetric field profile that interpolates from near the true vacuum at the center to the false vacuum at large radius. The bounce and its Euclidean action are obtained by calling CosmoTransitions' full path-deformation tunneling solver (with O(4) symmetry, i.e. three damping dimensions), working in a shifted-and-rotated "primed" coordinate frame whose origin sits at the false vacuum and whose axes are the eigenvectors of the Hessian there. Results (the radial grid, the two-component profile in both primed and original coordinates, and the action) are saved per pair and plotted in field space.

**Key equations**

*O(4) radial bounce equation of motion*

$$ \frac{d^2 \phi'_a}{d\rho^2} + \frac{3}{\rho}\frac{d\phi'_a}{d\rho} = \frac{\partial V}{\partial \phi'_a}, \qquad a = x,y $$

*Bounce boundary conditions*

$$ \left.\frac{d\phi'}{d\rho}\right|_{\rho=0} = 0, \qquad \lim_{\rho\to\infty}\phi'(\rho) = \phi'_F = 0 $$

*Euclidean O(4) bounce action*

$$ S = 2\pi^2 \int_0^\infty d\rho \; \rho^3 \left[\frac{1}{2}\left(\frac{d\phi'}{d\rho}\right)^2 + V(\phi')\right] $$

*Shift-and-rotate to primed coordinates*

$$ \phi' = L^{T}\,(\phi - \phi_F), \qquad \phi = \phi_F + L\,\phi' $$

*False-vacuum Hessian diagonalization defining the rotation*

$$ H_F \equiv \nabla\nabla V\big|_{\phi_F} = L\,\Lambda\,L^{T}, \qquad L^{T}L = \mathbb{1} $$

*Lifted potential in primed frame*

$$ V(\phi') = V_{\mathrm{orig}}\big(\phi_F + L\phi'\big) - V_{\mathrm{orig}}(\phi_F) $$

**Key objects**

- $\phi'(\rho)$ — Two-component bounce profile in primed coordinates as a function of Euclidean radius
- $\rho$ — Euclidean O(4) radial coordinate of the bounce
- $L$ — Orthogonal rotation whose columns are eigenvectors of the false-vacuum Hessian
- $\phi_F$ — False-vacuum point in original field coordinates (origin of primed frame)
- $S$ — Euclidean O(4) action of the bounce, controlling the decay exponent
- $\alpha = 3$ — Number of extra damping dimensions, selecting O(4) symmetry in the EOM
- $V(\phi')$ — Potential in primed frame, shifted so it vanishes at the false vacuum

### Stage 1 — Low modes $n=0,1$ (finite differences)

#### 3. `fd_builder_n0_git.py`

For the n=0 (negative/Coleman mode) angular sector of a two-field bounce, this file discretizes the radial fluctuation operator on a uniform grid as a sparse symmetric matrix, finds its lowest eigenpair (the negative mode), and stochastically estimates the trace of the resolvent (operator-plus-shift inverse) using the Hutchinson method. It works in the Liouville-transformed variable that removes the first-derivative term, so the matrix is symmetric and the inner product is standard Euclidean. The trace is estimated both raw and with the negative-mode direction projected out, the latter via a rank-1 correction that reuses a single sparse LU factorization. Boundary conditions are Dirichlet, enforced symmetrically by zeroing boundary rows/columns and placing a unit on the diagonal.

**Key equations**

*Liouville transformation*

$$ \tilde u(r) = r^{3/2}\, u_{\mathrm{raw}}(r), \qquad u_{\mathrm{raw}}(r) = r^{-3/2}\, \tilde u(r) $$

*Liouville-transformed radial fluctuation operator*

$$ \tilde M_n = -\frac{d^2}{dr^2} + \frac{n(n+2) + \tfrac{3}{4}}{r^2} + U''(\phi(r)) $$

*Two-field block matrix with Hessian coupling*

$$ \tilde M = \begin{pmatrix} -\partial_r^2 + V_{\mathrm{rad}} + U_{11} & U_{12} \\ U_{12} & -\partial_r^2 + V_{\mathrm{rad}} + U_{22} \end{pmatrix}, \quad V_{\mathrm{rad}}(r) = \frac{n(n+2)+\tfrac34}{r^2} $$

*Second-derivative finite-difference stencil*

$$ (-\partial_r^2 u)_i \approx -\frac{u_{i-1} - 2u_i + u_{i+1}}{(\Delta r)^2} $$

*Hutchinson stochastic trace estimator*

$$ tr(A^{-1}) \approx \frac{1}{K}\sum_{k=1}^{K} v_k^{\mathsf T} A^{-1} v_k, \qquad A = \tilde M + s^2 I, \quad (v_k)_i \in \{-1, +1\} $$

*Rank-1 projected-trace identity (negative mode removed)*

$$ tr(P A^{-1} P) = tr(A^{-1}) - \chi^{\mathsf T} A^{-1} \chi, \qquad P = I - \chi\chi^{\mathsf T}, \ \lVert\chi\rVert = 1 $$

**Key objects**

- $\tilde M_n$ — Liouville-transformed radial fluctuation operator for angular index n, built as a sparse symmetric 2N x 2N matrix
- $U''(\phi(r)) = U_{ij}$ — 2x2 Hessian of the potential evaluated along the bounce profile; supplies on-site terms U11, U22 and off-diagonal coupling U12
- $(\lambda, \chi)$ — lowest eigenpair of M-tilde for n=0; the negative eigenvalue lambda<0 and its unit-normalized Coleman negative-mode eigenvector chi
- $A = \tilde M + s^2 I$ — shifted operator whose resolvent trace is estimated per spectral parameter s^2 via sparse LU factorization
- $P = I - \chi\chi^{\mathsf T}$ — rank-1 projector that removes the negative-mode direction from the trace
- $v_k$ — Rademacher probe vector (entries +/-1, zeroed at boundary indices) used in the Hutchinson trace estimate
- $\bar g(s^2)$ — per-s^2 trace quantity returned raw (Tr A^{-1}) and subtracted (Tr P A^{-1} P), with Monte-Carlo mean and SEM

#### 4. `fd_builder_n1_git.py`

This file builds the discretized radial fluctuation operator for the n=1 angular sector (the channel that contains the translation zero mode) of a two-field bounce, and uses it to estimate the trace of its inverse via stochastic (Hutchinson) sampling. After a Liouville change of variables that removes the first-derivative term and makes the inner product Euclidean, the 2x2-coupled radial Schrodinger-like operator is assembled as a sparse symmetric matrix on a uniform grid with Dirichlet boundary conditions. The near-zero eigenvector (discrete translation zero mode) is found by shift-invert and projected out, and the regularized resolvent trace is estimated by averaging Rademacher quadratic forms over many right-hand-side solves of a single sparse LU factorization. The projected (zero-mode-subtracted) trace is obtained cheaply from a rank-one identity rather than a second Monte Carlo pass.

**Key equations**

*Liouville transformation*

$$ \tilde{u}(r) = r^{3/2}\, u_{\text{raw}}(r), \qquad u_{\text{raw}}(r) = r^{-3/2}\,\tilde{u}(r) $$

*Liouville-transformed radial operator (n=1)*

$$ \tilde{M}_n = -\frac{d^2}{dr^2} + \frac{n(n+2) + \tfrac34}{r^2} + U''(\phi(r)) $$

*Coupled 2x2 block structure with Hessian potential*

$$ \tilde{M}_n = \begin{pmatrix} -\partial_r^2 + V(r) + U_{11} & U_{12} \\ U_{12} & -\partial_r^2 + V(r) + U_{22} \end{pmatrix}, \quad V(r) = \frac{n(n+2)+\tfrac34}{r^2}, \quad U_{ab} = \partial_a\partial_b U(\phi(r)) $$

*Second-derivative finite-difference stencil*

$$ (-\partial_r^2 u)_i \approx \frac{-u_{i-1} + 2u_i - u_{i+1}}{(\Delta r)^2} $$

*Analytic continuum translation zero mode (Liouville basis)*

$$ \chi_{\text{zm}}(r) \propto r^{3/2}\,\frac{d\phi}{dr}, \qquad \big(\chi_x,\chi_y\big) = \big(r^{3/2} x'(r),\; r^{3/2} y'(r)\big) $$

*Hutchinson trace estimator of the resolvent*

$$ tr\!\big[(\tilde{M}_n + s^2 I)^{-1}\big] \approx \frac{1}{K}\sum_{k=1}^{K} v_k^{\top} (\tilde{M}_n + s^2 I)^{-1} v_k, \qquad (v_k)_i \in \{-1,+1\} $$

*Rank-one projected (zero-mode-subtracted) trace identity*

$$ tr\!\big[P A^{-1} P\big] = tr\!\big[A^{-1}\big] - \chi^{\top} A^{-1}\chi, \qquad P = I - \chi\chi^{\top}, \;\; \|\chi\| = 1, \;\; A = \tilde{M}_n + s^2 I $$

**Key objects**

- $\tilde{M}_n$ — Liouville-transformed coupled radial fluctuation operator for sector n, stored as a sparse symmetric 2N x 2N matrix
- $U_{ab} = \partial_a\partial_b U(\phi(r))$ — 2x2 Hessian of the potential evaluated along the bounce profile, the mass-matrix term of the operator
- $\chi$ — unit-normalized discrete (or analytic) translation zero mode that is projected out of the resolvent
- $P = I - \chi\chi^{\top}$ — orthogonal projector removing the zero-mode direction
- $A = \tilde{M}_n + s^2 I$ — regularized operator at virtual mass-squared s^2, factorized once by sparse LU per s^2 value
- $v_k \in \{-1,+1\}^{2N}$ — Rademacher random probe vectors used in the Hutchinson trace estimator
- $\lambda_{\text{zm}}$ — eigenvalue of the discrete zero mode, shifted from 0 by O(\Delta r^2) plus bounce EOM-violation error

#### 5. `compute_gbar_n0_fd_git.py`

This driver computes the n=0 (negative-mode) contribution to the trace of the inverse fluctuation operator of a two-field Euclidean bounce, as a function of a spectral shift s^2. It builds the Liouville-transformed radial fluctuation operator on a uniform finite-difference grid, finds its single negative (Coleman) eigenvalue and eigenvector, and then scans s^2 over an adaptive grid that is refined near the pole at s^2 = -lambda_neg. At each s^2 it forms the shifted operator M_tilde + s^2 I, factorizes it once by sparse LU, and uses a Hutchinson stochastic estimator to obtain both the raw trace gbar_raw = Tr[(M_tilde + s^2 I)^-1] and the zero-mode-subtracted trace gbar_sub = Tr[P(M_tilde + s^2 I)^-1 P], where P projects out the negative mode. Results (means and Monte-Carlo standard errors) are saved to an .npz file.

**Key equations**

*Liouville-transformed radial fluctuation operator (n=0)*

$$ \tilde{M}_n = -\frac{d^2}{dr^2} + \frac{n(n+2) + \tfrac{3}{4}}{r^2} + U''(\phi(r)), \qquad n=0 $$

*Two-field block structure with potential Hessian*

$$ \tilde{M} = \begin{pmatrix} -\partial_r^2 + \frac{3/4}{r^2} + U_{xx} & U_{xy} \\ U_{xy} & -\partial_r^2 + \frac{3/4}{r^2} + U_{yy} \end{pmatrix}, \qquad U_{ab} = \frac{\partial^2 U}{\partial\phi_a \partial\phi_b} $$

*Liouville change of variable*

$$ u_{\text{raw}}(r) = r^{-3/2}\, \tilde{u}(r), \qquad \tilde{u}(r) = r^{3/2}\, u_{\text{raw}}(r) $$

*Second-derivative finite-difference stencil*

$$ \big(\partial_r^2 u\big)_i \approx \frac{u_{i-1} - 2u_i + u_{i+1}}{(\Delta r)^2} $$

*Negative mode and pole location*

$$ \tilde{M}\,\chi = \lambda_{\mathrm{neg}}\,\chi, \quad \lambda_{\mathrm{neg}} < 0, \qquad s^2_{\text{pole}} = -\lambda_{\mathrm{neg}} $$

*Raw and zero-mode-subtracted traces (rank-1 identity)*

$$ \bar{g}_{\text{raw}}(s^2) = tr\!\big[(\tilde{M} + s^2 I)^{-1}\big], \qquad \bar{g}_{\text{sub}}(s^2) = tr\!\big[P(\tilde{M}+s^2 I)^{-1}P\big] = \bar{g}_{\text{raw}}(s^2) - \chi^{T}(\tilde{M}+s^2 I)^{-1}\chi, \qquad P = I - \chi\chi^{T} $$

*Hutchinson stochastic trace estimator*

$$ tr[A^{-1}] \approx \frac{1}{K}\sum_{k=1}^{K} v_k^{T} A^{-1} v_k, \qquad (v_k)_j \in \{-1,+1\}, \quad \mathrm{SEM} = \frac{\mathrm{std}(v_k^T A^{-1} v_k)}{\sqrt{K}} $$

**Key objects**

- $\tilde{M}$ — Liouville-transformed two-field radial fluctuation operator, a sparse symmetric 2N x 2N matrix (Dirichlet BCs)
- $\lambda_{\mathrm{neg}}, \chi$ — lowest (negative) eigenvalue and unit-normalized eigenvector: the Coleman negative mode of the n=0 sector
- $s^2_{\text{pole}} = -\lambda_{\mathrm{neg}}$ — shift value at which (M_tilde + s^2 I) becomes singular; the adaptive grid is refined and clipped around it
- $P = I - \chi\chi^{T}$ — Feshbach projector removing the negative-mode direction from the resolvent
- $\bar{g}_{\text{raw}}(s^2)$ — raw trace of the shifted inverse operator at spectral shift s^2
- $\bar{g}_{\text{sub}}(s^2)$ — zero-mode-subtracted trace, the physical n=0 contribution free of the negative-mode pole
- $K$ — number of Rademacher Hutchinson probe vectors per s^2 (default 100000)
- $U''(\phi(r))$ — 2x2 field-space Hessian of the potential evaluated along the bounce profile

#### 6. `compute_gbar_n1_fd_git.py`

This driver computes the n=1 partial-wave contribution to a two-field fluctuation determinant as a function of the spectral parameter s^2. It builds the Liouville-transformed radial fluctuation operator on a finite-difference grid, finds the discrete near-zero (translation) mode of that operator by diagonalization, and then scans a logarithmically-near-zero / linear-far grid of s^2 values, estimating at each point the trace of the resolvent (M+s^2)^{-1} and its zero-mode-projected version via stochastic (Hutchinson) trace estimation with sparse LU solves. Both the raw and zero-mode-subtracted traces are multiplied by the angular degeneracy (n+1)^2 = 4 and written, together with the zero-mode eigenvalue and pole location, to an .npz file. The file also validates the numerically found zero mode against the analytic translation zero mode by an overlap check.

**Key equations**

*Liouville-transformed radial fluctuation operator (n=1)*

$$ \tilde{M}_n = -\frac{d^2}{dr^2} + \frac{n(n+2) + \tfrac{3}{4}}{r^2} + U''(\phi_b(r)), \qquad n=1 $$

*Liouville substitution*

$$ u_{\mathrm{raw}}(r) = r^{-3/2}\,\tilde{u}(r), \qquad \tilde{u}(r) = r^{3/2}\,u_{\mathrm{raw}}(r) $$

*Two-field block structure of the operator*

$$ \tilde{M} = \begin{pmatrix} -\partial_r^2 + V_{\mathrm{rad}} + U''_{11} & U''_{12} \\ U''_{12} & -\partial_r^2 + V_{\mathrm{rad}} + U''_{22} \end{pmatrix}, \quad V_{\mathrm{rad}}(r) = \frac{n(n+2)+\tfrac34}{r^2} $$

*Raw resolvent trace (per partial wave)*

$$ \bar{g}_{\mathrm{raw}}(s^2) = tr\big[(\tilde{M} + s^2 I)^{-1}\big] $$

*Zero-mode-subtracted trace via rank-1 identity*

$$ \bar{g}_{\mathrm{sub}}(s^2) = tr\big[P\,(\tilde{M}+s^2 I)^{-1} P\big] = tr\big[(\tilde{M}+s^2 I)^{-1}\big] - \chi_{\mathrm{zm}}^{\mathsf T} (\tilde{M}+s^2 I)^{-1}\chi_{\mathrm{zm}}, \quad P = I - \chi_{\mathrm{zm}}\chi_{\mathrm{zm}}^{\mathsf T} $$

*Degeneracy-weighted output and Hutchinson estimator*

$$ \bar{g}_{n=1} = (n+1)^2\,\bar{g}, \qquad tr\,A^{-1} \approx \frac{1}{K}\sum_{k=1}^{K} v_k^{\mathsf T} A^{-1} v_k, \quad (v_k)_i \in \{-1,+1\} $$

**Key objects**

- $\tilde{M}$ — Sparse symmetric 2N x 2N Liouville-transformed radial fluctuation operator for the two coupled fields (n=1)
- $U''(\phi_b(r))$ — 2x2 Hessian of the potential evaluated along the bounce profile, providing the diagonal and off-diagonal mass terms
- $\chi_{\mathrm{zm}}$ — Discrete translation zero mode of M-tilde (eigenvector with eigenvalue closest to zero, from shift-invert eigsh)
- $\lambda_{\mathrm{zm}}$ — Near-zero eigenvalue of the discrete operator; the resolvent pole sits at s^2 = -lambda_zm
- $P = I - \chi_{\mathrm{zm}}\chi_{\mathrm{zm}}^{\mathsf T}$ — Projector removing the zero-mode direction from the resolvent trace
- $s^2$ — Spectral / mass-shift parameter scanned on a log-near-zero, linear-far grid; the resolvent trace is sampled as a function of it
- $(n+1)^2 = 4$ — Angular degeneracy factor for the n=1 partial wave applied to both raw and subtracted traces
- $\chi_{\mathrm{zm}}^{\mathrm{an}} \propto r^{3/2}\,\phi_b'(r)$ — Analytic continuum translation zero mode used as a sanity-check overlap reference

#### 7. `compute_gbar_n0_fd_wkb_vfinal4_git.py`

This file is a pure I/O pass-through wrapper around the n=0 dense-grid generator and contains no physics or numerical math of its own. It mirrors the command-line options of the original driver (bounce file, grid size N, radial range, number of stochastic probes K, the s^2 sampling/tail schedule, RNG seed, and the choice of subtracted-trace estimator), resolves where the output should be written (using an explicit data directory or environment variables, falling back to the project data volume), and retags the output filename with the suffix _wkb_vfinal4 so it fits the WKB pipeline naming convention. It then delegates all actual computation to compute_gbar_n0_fd.main() by overriding the argument list; the underlying finite-difference / Hutchinson-Schrodinger trace method is used unchanged because it does not rely on Bessel kernels affected by the Stage-A WKB step.

#### 8. `compute_gbar_n1_fd_wkb_vfinal4_git.py`

This is a pure command-line/I-O wrapper and contains no mathematics of its own. It parses run options (bounce-data file, radial grid size and range, number of Hutchinson trace samples K, the spectral parameter s^2 grid bounds and spacing, the random seed, and the choice of subtracted-trace estimator), decides where the output .npz should be written, builds the corresponding argument list, and then calls the main() routine of the underlying n=1 finite-difference driver. All physics (the discrete fluctuation operator, the translational zero-mode projection, the subtracted radial Green's function, and the stochastic trace) is performed in that driver and its helper modules, not here. The only file-specific behavior is retagging the output filename with the "_wkb_vfinal4" suffix and setting the v-final-4 defaults (e.g. K = 100000).

**Key objects**

- $n = 1$ — Partial-wave (angular momentum) sector this wrapper drives, which contains the translational zero mode of the bounce
- $K = 10^{5}$ — Default number of Hutchinson stochastic-trace samples forwarded to the n=1 determinant builder
- $s^2 \in [10^{-6}, 10^{3}]$ — Default range of the spectral shift parameter over which the subtracted partial-wave trace is evaluated

### Stage 2 — False-vacuum subtraction for $n=0,1$

#### 9. `add_fv_fd_to_dense_vfinal4_git.py`

This script appends the false-vacuum (FV) side of the fluctuation-determinant subtraction to existing finite-difference output files for the lowest two angular sectors (n=0, n=1). For each sector it rebuilds the same Liouville-transformed radial fluctuation operator used on the bounce, but with the field profile set to the false vacuum (constant background), so the operator's mass matrix is just the potential's Hessian evaluated at the false-vacuum minimum. It then estimates the trace of the resolvent (M+s^2)^(-1) at every s^2 in the existing grid using a Rademacher Hutchinson stochastic trace estimator, with the negative mode (n=0) or translational zero mode (n=1) projected out by the same rank-1 projector used on the bounce side, and multiplies the result by the angular degeneracy (n+1)^2. The purpose is purely consistency: subtracting bounce-FD minus FV-FD in an identical discretization scheme cancels a spurious boundary/discretization tail in the per-sector renormalized determinant.

**Key equations**

*Liouville-transformed radial fluctuation operator*

$$ \tilde{M}_n = -\frac{d^2}{dr^2} + \frac{n(n+2) + \tfrac{3}{4}}{r^2} + U''(\phi(r)) $$

*False-vacuum operator (zero profile)*

$$ \tilde{M}_n^{\mathrm{FV}} = -\frac{d^2}{dr^2} + \frac{n(n+2)+\tfrac{3}{4}}{r^2}\,\mathbb{1}_{2\times 2} + U''(\phi'=0) $$

*Potential Hessian in primed (false-vacuum) coordinates*

$$ U''(\phi') = L^{T}\,\mathrm{Hess}\,V\big(\phi_{\mathrm{false}} + L\,\phi'\big)\,L $$

*Second-difference Laplacian discretization*

$$ \big(L_2 u\big)_i = \frac{u_{i-1} - 2u_i + u_{i+1}}{(\Delta r)^2} $$

*Projected Hutchinson resolvent trace*

$$ \bar{g}_n(s^2) = \frac{1}{K}\sum_{k=1}^{K} v_k^{T}\,P\,(\tilde{M}_n^{\mathrm{FV}} + s^2 I)^{-1}\,P\,v_k,\qquad P = I - \chi\chi^{T} $$

*Degeneracy-weighted FV contribution*

$$ \bar{g}_n^{\mathrm{FV}} \longrightarrow (n+1)^2\,\bar{g}_n^{\mathrm{FV}} $$

**Key objects**

- $\tilde{M}_n^{\mathrm{FV}}$ — False-vacuum radial fluctuation operator, sparse 2N x 2N symmetric matrix with zero background profile
- $U''(\phi'=0)$ — Constant 2x2 potential Hessian at the false vacuum, the mass matrix of the FV operator
- $P = I - \chi\chi^{T}$ — Rank-1 projector removing the bounce-side negative mode (n=0) or zero mode (n=1)
- $\chi$ — Unit-normalized mode vector (chi_neg for n=0, chi_zm for n=1) taken from the bare-side file, zeroed at boundary indices
- $v_k$ — Rademacher random probe vector (+/-1 entries, zero at Dirichlet boundaries) for the Hutchinson estimator
- $(n+1)^2$ — Angular partial-wave degeneracy weight applied to match the bounce-side trace
- $s^2$ — Spectral shift parameter; trace of resolvent is evaluated over the existing s2_grid
- $K$ — Number of Hutchinson probes (default 100000); Monte-Carlo error scales as 1/sqrt(K)

### Stage 3 — Higher modes $n\ge 2$ (Runge–Kutta)

#### 10. `wkb_bessel_vfinal4_git.py`

This file is a thin evaluator for the modified Bessel functions of the first and second kind and their first derivatives. It exposes four functions that return I_nu(z), K_nu(z), and their z-derivatives, each obtained by directly calling the corresponding scipy.special routines (which delegate to the AMOS/Slatec library) and casting the result to a plain float. Unlike the earlier pipeline version, it performs no regime dispatch: a single backend is used for every (nu, z) in the production range (nu up to 31, z up to about 222), with the former Hankel/Olver threshold constants retained but set to infinity so those branches are never taken. It is essentially a numerical-library wrapper, so the only equations are the defining identities for the special functions it returns.

**Key equations**

*Modified Bessel functions returned*

$$ \texttt{iv}(\nu,z)=I_\nu(z),\qquad \texttt{kv}(\nu,z)=K_\nu(z) $$

*First derivatives in z returned*

$$ \texttt{ivp}(\nu,z)=I_\nu'(z)=\frac{d}{dz}I_\nu(z),\qquad \texttt{kvp}(\nu,z)=K_\nu'(z)=\frac{d}{dz}K_\nu(z) $$

**Key objects**

- $I_\nu(z)$ — Modified Bessel function of the first kind, order nu, argument z (regular solution).
- $K_\nu(z)$ — Modified Bessel function of the second kind, order nu, argument z (decaying solution).
- $I_\nu'(z),\ K_\nu'(z)$ — First derivatives of the modified Bessel functions with respect to the argument z.
- $(\nu,z)$ — Bessel order nu (up to 31) and argument z (up to ~222 at s^2=1000) spanning the production regime.

#### 11. `rk_builder_adapt_wkb_vfinal4_git.py`

On the bounce background of a two-field theory, this file builds the coupled-channel radial fluctuation Green function in a fixed partial-wave (angular) sector. It writes each regular/irregular solution as a free modified-Bessel basis function times a correction h, solves the resulting linear correction ODE for h with a stiff implicit Radau integrator (shooting inward and outward), and forms a 2x2 Wronskian-like matrix whose r^3-weighted version reaches a constant plateau. The inverse of that plateau matrix is contracted with the inward and outward basis solutions to produce, for every radius, the diagonal trace of the Green function that downstream code integrates into the fluctuation determinant.

**Key equations**

*Free mass matrix at the false vacuum*

$$ M_{\mathrm{free}} = H'(0) = L^{T}\, \partial^2 V\,L\big|_{\phi'=0}, \qquad m_i^2 = (M_{\mathrm{free}})_{ii} $$

*Channel decay constant*

$$ \kappa_i = \sqrt{s^2 + m_i^2} $$

*Partial-wave index (4D)*

$$ \nu = n + 1 $$

*Free Bessel basis (irregular '+' and regular '-')*

$$ B_i^{+}(r) = \frac{K_{\nu}(\kappa_i r)}{r}, \qquad B_i^{-}(r) = \frac{I_{\nu}(\kappa_i r)}{r} $$

*Interaction (potential minus free mass)*

$$ U_{ij}(r) = H'\big(\phi'(r)\big)_{ij} - (M_{\mathrm{free}})_{ij} $$

*Channel-coupling and centrifugal coefficients*

$$ K_{ij}(r) = U_{ij}(r)\,\frac{B_j(\kappa_j r)}{B_i(\kappa_i r)}, \qquad A_i(r) = 2\kappa_i\,\frac{B'_{\nu}(\kappa_i r)}{B_{\nu}(\kappa_i r)} $$

*Correction ODE for h (source column = sol_index s)*

$$ h_i'' = -\Big(\tfrac{1}{r} + A_i(r)\Big)h_i' + \sum_{j} K_{ij}(r)\,h_j + K_{is}(r) $$

**Key objects**

- $f^{\pm}_{i\alpha}(r) = B_i(r)\,(\delta_{i\alpha} + h_{i\alpha})$ — Full radial mode functions: free Bessel basis dressed by the solved correction h, '+' (irregular, large-r decaying) and '-' (regular at origin)
- $W_{\alpha\beta}(r) = -\sum_i \big(f^{-}_{i\alpha}\,{f^{+}_{i\beta}}' - f^{+}_{i\beta}\,{f^{-}_{i\alpha}}'\big)$ — 2x2 Wronskian matrix between regular and irregular solution sets
- $\Omega = \langle r^3 W(r)\rangle_{\text{plateau}}$ — r^3-weighted Wronskian averaged over the constant plateau window; its inverse normalizes the Green function
- $\mathrm{tr}\,G_{kk} = \sum_{i,b,c} f^{+}_{ib}(r_k)\,(\Omega^{-1})_{bc}\,f^{-}_{ic}(r_k)$ — Diagonal trace of the radial Green function at each radius, the file's main output

#### 12. `compute_gbar_npos_wkb_vfinal4_git.py`

This driver computes, for each angular mode n >= 2 and each value of the spectral parameter s^2, a radial integral of the trace of the WKB Green's-function kernel of the fluctuation operator on the bounce background, then forms a per-mode summary quantity gbar_n weighted by the angular degeneracy. For each requested s^2 on a hybrid grid it invokes a builder that produces the per-mode Green's function on a radial grid, reads back the diagonal trace, integrates it against r^3 by the trapezoidal rule, multiplies by (n+1)^2, and sums the results over the mode range. The output is a set of .npz summary files containing I_n, gbar_n, and their sum for each s^2. The numerical methods used are direct trapezoidal quadrature on a stored radial grid and accumulation over modes; no new physics formula is derived here beyond the integral and the degeneracy weight.

**Key equations**

*Radial trace integral*

$$ I_n(s^2) = \int_0^\infty r^3 \, \mathrm{tr}\, G_n(r,r;s^2)\, dr $$

*Diagonal trace of the Green's-function block*

$$ \mathrm{tr}\, G_n(r,r) = \sum_{a} \big[G_n(r,r)\big]_{aa} $$

*Trapezoidal quadrature of the integrand*

$$ I_n \approx \sum_{k} \tfrac{1}{2}\big(f_{k+1}+f_k\big)\big(r_{k+1}-r_k\big), \qquad f_k = r_k^3 \, \mathrm{tr}\, G_n(r_k,r_k) $$

*Degeneracy-weighted mode summary*

$$ \bar{g}_n = (n+1)^2 \, I_n $$

*Total sum over modes*

$$ S(s^2) = \sum_{n=n_{\min}}^{n_{\max}} \bar{g}_n, \qquad n_{\min}=2,\ n_{\max}=18 $$

**Key objects**

- $G_n(r,r';s^2)$ — WKB Green's-function kernel block of the fluctuation operator for angular mode n at spectral parameter s^2, built on the bounce background.
- $I_n(s^2)$ — Radial integral of r^3 times the diagonal trace of the mode-n Green's function.
- $\bar{g}_n$ — Degeneracy-weighted per-mode summary, (n+1)^2 I_n, used in the determinant/effective-action sum.
- $(n+1)^2$ — Angular-mode degeneracy weight for mode n in the four-dimensional partial-wave decomposition.
- $S(s^2)$ — Sum of gbar_n over the mode range n = 2..18 at fixed s^2.
- $s^2$ — Spectral/mass-shift parameter sampled on a hybrid grid (fine step below s^2=10, coarse tail step above).

### Stage 4 — False-vacuum reference traces (Runge–Kutta)

#### 13. `rk_builder_fv_wkb_vfinal4_git.py`

For a given partial-wave index n and Euclidean momentum-squared s^2, this file builds the free (false-vacuum) two-field fluctuation basis from modified Bessel functions and assembles the radial Green function on a grid, then extracts only the diagonal trace needed downstream. The two free masses come from the Hessian of the potential at the false vacuum (diagonalized so the two fields decouple). The regular and irregular solutions are the modified Bessel functions I and K of order n+1 evaluated at kappa_i r and divided by r (the d=4 radial reduction), with all Bessel evaluations done through a numerically stable WKB/asymptotic dispatcher. A Wronskian-type bilinear form of the two solution sets is averaged over a tail window to fix the normalization matrix Omega, whose inverse sandwiched between the regular and irregular solutions gives the diagonal Green-function trace that is saved.

**Key equations**

*Free masses from false-vacuum Hessian*

$$ m_1^2 = \big[H(\phi_{\mathrm{false}})\big]_{00}, \qquad m_2^2 = \big[H(\phi_{\mathrm{false}})\big]_{11}, \qquad H = L^{\mathsf T}\, \partial^2 V\, L $$

*Radial wavenumber per field*

$$ \kappa_i = \sqrt{s^2 + m_i^2}, \qquad i = 1,2 $$

*Free basis (regular / irregular solutions), order nu = n+1*

$$ B^{-}_{\nu}(r) = \frac{I_{\nu}(\kappa_i r)}{r}, \qquad B^{+}_{\nu}(r) = \frac{K_{\nu}(\kappa_i r)}{r}, \qquad \nu = n+1 $$

*Radial derivative of a basis function*

$$ \frac{d}{dr}\!\left(\frac{B_{\nu}(\kappa_i r)}{r}\right) = \frac{\kappa_i\, B_{\nu}'(\kappa_i r)\, r - B_{\nu}(\kappa_i r)}{r^{2}} $$

*Wronskian-type bilinear form*

$$ W_{\alpha\beta}(r) = -\sum_{i=1}^{2}\Big( f^{-}_{i\alpha}\, \frac{d f^{+}_{i\beta}}{dr} - f^{+}_{i\beta}\, \frac{d f^{-}_{i\alpha}}{dr} \Big) $$

*Tail-averaged normalization matrix*

$$ \Omega_{\alpha\beta} = \big\langle\, r^{3}\, W_{\alpha\beta}(r) \,\big\rangle_{r \in [0.05,\, 0.9\, r_{\max}]} $$

*Diagonal Green-function trace*

$$ \mathrm{tr}\, G(r,r) = \sum_{i,b,c} f^{+}_{ib}(r)\, (\Omega^{-1})_{bc}\, f^{-}_{ic}(r) $$

**Key objects**

- $\nu = n+1$ — order of the modified Bessel functions for partial wave n (d=4 radial reduction)
- $\kappa_i$ — radial decay constant sqrt(s^2 + m_i^2) for free field i at the false vacuum
- $m_1^2,\, m_2^2$ — diagonal free masses-squared from the potential Hessian at the false vacuum
- $f^{+}_{i\alpha},\ f^{-}_{i\alpha}$ — irregular (K) and regular (I) free solution components, diagonal in field index
- $W_{\alpha\beta}(r)$ — Wronskian-type bilinear of the two free solution sets at radius r
- $\Omega^{-1}$ — inverse of the tail-averaged, r^3-scaled normalization matrix
- $\mathrm{tr}\,G(r,r)$ — diagonal trace of the free radial Green function, the only quantity consumed downstream

#### 14. `compute_gbar_fv_wkb_vfinal4_git.py`

This driver computes the false-vacuum-side per-mode determinant contributions for a two-field bounce, looping over all angular-momentum modes n (default 0 to 18) and over a grid of squared spectral parameters s^2. For each (n, s^2) it loads (or builds) the radial-RK Green's function, forms the radially weighted trace integral over the bubble profile, multiplies by the four-dimensional rotational degeneracy of mode n, and sums the result over all n. Outputs are per-s^2 NPZ summaries containing the per-mode integrals, the degeneracy-weighted gbar_n, and their total. It is a numerically-stable (WKB-Bessel) variant whose actual radial trace and degeneracy weighting are the only physics done in this file.

**Key equations**

*Radially weighted trace integral*

$$ I_n(s^2) = \int r^3 \, \mathrm{tr}\,G_n(r,r;s^2)\, dr $$

*Degeneracy-weighted mode contribution*

$$ \bar g_n(s^2) = (n+1)^2 \, I_n(s^2) $$

*Summed determinant contribution*

$$ \bar g_{\mathrm{tot}}(s^2) = \sum_{n=n_{\min}}^{n_{\max}} \bar g_n(s^2) $$

*Diagonal trace of the matrix Green function*

$$ \mathrm{tr}\,G_n(r,r;s^2) = \sum_{a} \big[G_n(r,r;s^2)\big]_{aa} $$

**Key objects**

- $I_n(s^2)$ — Radial trace integral of the FV Green function for angular mode n at spectral parameter s^2
- $\bar g_n(s^2)$ — Degeneracy-weighted per-mode contribution, (n+1)^2 times I_n
- $(n+1)^2$ — Four-dimensional O(4) rotational multiplicity (degeneracy) of angular mode n
- $G_n(r,r';s^2)$ — False-vacuum-side radial matrix Green function for mode n (built by the WKB RK helper)
- $r$ — Euclidean radial coordinate along the bounce profile, with r^3 the 4D radial measure
- $s^2$ — Squared spectral / mass parameter scanned on a hybrid fine-plus-tail grid

---

## Where this stops

The outputs of Stages 1–4 are the bare and false-vacuum $\bar g_n(s^2)$ traces. Turning these into a finite, renormalized $\ln\det$ requires the Born subtractions, the tadpole/fish counterterms, and the high-$n$ tail — performed in a separate downstream stage that is intentionally **not** part of this repository.
