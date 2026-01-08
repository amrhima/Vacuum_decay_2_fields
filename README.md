G_project

Description:  

A Python project for computing O(4)-bounce solutions and Green's functions for a 2-field potential using the Runge-Kutta (RK) method with linear field redefinition.

Overview:

This project implements a computational pipeline for:
1. Potential: Symbolic and numeric evaluation of a 2-field potential with automatic gradient and Hessian computation
2. Bounce-solution: Finding O(4)-symmetric bounce solutions between false and true vacua
3. Green's function construction: Building RK Green's functions for fluctuation operators in primed coordinates
4. Green's function-visualization: Plotting bounce profiles, Green functions, and diagnostic plots
5. Test: Testing jump condition


Files:

*potential.py

- Purpose: 
  Defines the 2-field potential V(x,y) with symbolic computation using SymPy
- Workflow:
  1. Automatic gradient and Hessian computation via symbolic differentiation
  2. Critical point finder (minima, maxima, saddles)
  3. Linear field redefinition class "CTShiftedLiftedPotential" for primed coordinates
  4. Vacuum finder for locating all local minima

- Main functions:
  - V_numeric(X, params): Evaluate potential at point(s) X
  - gradV_numeric(X, params): Compute gradient
  - H_numeric(point, params): Compute Hessian matrix
  - find_all_minima(params): Find all local minima
  - CTShiftedLiftedPotential: Class for primed coordinate system:
    1. Choose a false vacuum φ_F (minimum of original potential).
    2. Compute Hessian H_F = H_orig(φ_F).
    3. Diagonalise H_F = L Λ L^T,  with L orthogonal, Λ = diag(λ1, λ2).
    4. Define primed fields (column-vector convention):
           φ = φ_F + L φ' --> φ' = L^T (φ - φ_F).
    5. Define shifted+lifted potential in primed coords:
           V'(φ') = V_orig(φ_F + L φ') - V_orig(φ_F)
       Then V'(0) = 0 and
           H'(φ') = L^T H_orig(φ_F + L φ') L,
           H'(0)  = Λ  (diagonal).

- Potential-Form:
  V(x,y) = a0 + a_xy2 * x y^2 + a_y4 * (y^2 - 1)^2 
           + a_x2y2 * x^2 y^2 + a_x4 * (x^2 - 1)^2
  

*bounce.py

- Purpose: 
  Compute O(4)-symmetric bounce solutions for all true-to-false vacuum-pairs in primed coordinates using CosmoTransitions
- Workflow:
  1. Find all local minima of the potential
  2. Identify pairs (F, T) where V_F > V_T
  3. For each pair, build primed coordinate system around false vacuum
  4. Compute bounce profile using cosmoTransitions.pathDeformation
  5. Save bounce data to bounce_data_F*_T*.npz files

- Output files: bounce_data_F{iF}_T{iT}.npz containing:
  - R_bounce: Radial grid
  - X_bounce_prime, Y_bounce_prime: Bounce in primed coordinates
  - X_bounce_orig, Y_bounce_orig: Bounce in original coordinates
  - params, false_vac, true_vac, S_E = O(4)-Euclidean bounce action


*rk_builder.py

- Purpose: 
  Construct Runge-Kutta Green's functions for fluctuation operators
- Workflow:
  1. Load bounce data from bounce_data_F*_T*.npz
  2. Build fluctuation data in primed coordinates:
     - Free Bessel basis functions B_i^{±}(r)
     - h-basis functions via RK4 integration
     - Full modes f_i^{±,α}(r)
     - Wronskian computation
  3. Construct Green function G_rk(r, r')
  4. Save to rk_green_data_F*_T*.npz

- Main Functions:
  - build_fluctuation_data_prime(): Build all fluctuation objects
  - rk4_h(): RK4-integrator for h-equation
  - build_rk_green_for_bounce(): Main builder for one bounce

- Convention:
  - "+" = decaying solution at infinity (uses K_l)
  - "-" = regular solution at r=0 (uses I_l)

- Output files: rk_green_data_F*_T*.npz containing:
  - r_grid, G_rk: Green function
  - f_plus, f_minus, df_plus, df_minus: Full modes
  - h_plus, h_minus: h-basis functions
  - B_plus, B_minus: Free Bessel modes
  - W_scaled, Omega_inv: Wronskian data
  - M_free: Free mass matrix
  - Bounce data 

*green_plots.py

- Purpose: 
  Visualize all computed data
- Workflow:
  - 3D surface plots of Green's function components (11, 12, 21, 22)
  - Plots for RK, FD, and spectral Green's functions (if available)
  - h-basis, full modes, and free modes visualization
  - Hessian eigenvalues along bounce vs M_free eigenvalues
  - r^3-scaled Wronskian components with plateau values

- Usage: 
  Run after rk_builder.py to visualize data
- Input: 
  Automatically finds all rk_green_*.npz files and matching fd_green_*.npz, spec_green_*.npz if present


Test Files:

*jump_test.py

- Purpose: 
  Checks the local jump condition at r = r', which is: r_prime^3 [G'(r+0,r) - G'(r-0,r)] = I_2, which is derived from the Green's equation after integrating over a small intervall around r = r'
- Method: 
  Compute one-sided finite differences for the derivative of G_rk with respect to r along the diagonal r_grid[k] = r_grid[l], where k-index means: position on the radial grid of r and l-index means: position on the radial grid of r_prime.
- Output: 
  If the Green's function is normalized correctly, jump condition is fulfilled with J[k] = I_2, for all k. Plots of J_11, J_22, J_12, J_21 vs radius are shown.


*green_test.py 

- Purpose: 
  Test the Green's equation
- Method:
  Apply the full radial fluctuation operator to G_rk and check if the delta function appears on the right hand side. To see that, used centered finite differences to approximate the derivatives of the Green's function and a simple test function, which is phi[r] = 1, for all r. Then multiplied the whole equation by r^3 and integrated over r from 0 to infinity. 
- Output:
  On the grid, the integral becomes a sum over k and for each fixed r' the test results are stored in S_slice. Plots for S_slice are shown. For a sufficient result this should equal to the unit matrix. Also the average is taken, which is stored in S_avg. 

