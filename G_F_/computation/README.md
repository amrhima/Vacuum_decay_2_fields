# Coupled toy-model computation pipeline

This directory contains the numerical implementation of the Baacke-style
renormalized fluctuation determinant for the two-field toy model.  The
procedure is based on `../renormalization_baacke_procedure.pdf`.  The
production determinant calculation uses the Green-function/resolvent method;
the determinant method is not a production route in these scripts.

The pipeline should be read and run in the order below.  A later stage should
use the `.npz` output of the preceding stage with the same `tag` and the same
`bounce_sha` provenance.  All command-line programs provide `--help`; the
commands below are therefore schematic and omit run-specific numerical
options.

## Calculation order

```text
potential + shared numerical modules
                  |
stage 0: bounce -> negative mode and zero mode
                  |
stage 1: adaptive residual band generation
                  |
stage 3: n=0 and n=1 continuum sectors (can run independently)
                  |
stage 4: finite-s2 band integrals + high-s2 completion
                  |
stage 5: high-n partial-wave tail
                  |
stage 6: tadpole counterterm    stage 7: fish counterterm
                  \                 /
                   stage 8: assemble ln(D)^ren
```

There is no separate stage 2 executable: the residual band engine is invoked
by the stage-1 adaptive driver.  The sector calculations can be done before or
after stage 1, because they depend on the bounce and their own eigenvalue
file, not on the band tables.

### 0. Define the model and compute the bounce

#### 1. `potential_coupled_toy_model.py` — model and background

This file defines the fixed two-field quartic
   potentials (`F2_T0` coupled and `bdet` decoupled), their known vacua,
   gradients/Hessians, and the shifted/rotated background used by the
   calculation.  It is imported by essentially every physics stage; it does
   not discover vacua dynamically.

   ##### Functions/classes — model setup order

   - `model_names()` / `get_model()` select and return the hardcoded model.
   - `couplings_to_params()` converts couplings to the stored 15-coefficient
     polynomial vector.
   - `V_numeric()`, `gradV_numeric()`, `H_numeric()`, and
     `H_numeric_batch()` evaluate the potential, gradient, and Hessian.
   - `is_decoupled()` / `tunneling_fields()` choose the one- or two-field
     bounce route.
   - `guard_vacua()` validates the declared false and true vacua.
   - `CTShiftedLiftedPotential` exposes the shifted/lifted potential to the
     bounce solver.
   - The private `_unpack()` helper decodes the coefficient vector for the
     evaluators.

#### 2. `bounce_coupled_toy_model.py` — stage 0: bounce

This file solves or reuses the O(4)-symmetric
   bounce, using the two-field CosmoTransitions path-deformation solver for
   the coupled model and the one-dimensional instanton solver for the
   decoupled model.  It applies the endpoint/EOM checks and measures the
   remaining bounce-profile error.  It writes
   `bounce_data_<tag>.npz`, the primary input to all later stages.

   Typical first command:

   ```bash
   python bounce_coupled_toy_model.py --help
   python bounce_coupled_toy_model.py --model F2_T0 --tag coupled_toy_model
   ```

   ##### Functions — bounce calculation order

   - `parse_pair()` resolves the requested vacuum pair.
   - `compute_bounce_coupled()` and `compute_bounce_decoupled()` solve the
     coupled and one-dimensional bounce problems, respectively.
   - `solve_bounce()` selects the solver and handles cached profiles.
   - `assert_bounce_endpoints()`, `eom_residual_rms()`, and `_check_eom()`
     validate the returned profile.
   - `profile_moments()` and `s2c_of_bounce()` calculate accuracy diagnostics.
   - `bounce_accuracy_watcher()` and `_improve_and_store()` test the profile
     against the accuracy ladder; `_quantify_lnD_error()` estimates its
     determinant error.
   - `bounce_npz_payload()` defines the saved schema; `convert_legacy_npz()`
     imports an older compatible profile.
   - Supporting helpers `_common_radial_grid()`, `single_field_V_dV()`,
     `_npz_dict()`, `_changed_settings()`, `_resample_onto()`,
     `_reference_profile()`, `_write_candidate()`, and `_walk_ct_ladder()`
     prepare grids, migrate/cache candidates, compare settings, resample
     profiles, and advance the accuracy ladder.

#### 3. `eigenvalue_neg_coupled_toy_model.py` — stage 0a: negative mode

This file locates the n=0 negative mode
   from the zero-crossing of the solution-matching determinant.  It also
   checks that the bounce has exactly one negative mode.  It writes
   `eig_n0_<tag>.npz`.

   ##### Functions — negative-mode calculation order

   - `find_crossing_eigenvalue()` scans, brackets, and root-finds the first
     n=0 matching-determinant crossing, then checks its multiplicity and the
     Coleman one-negative-mode condition.
   - `_parking_report()` prints the bounce-quality context for the result.

#### 4. `eigenvalue_zero_coupled_toy_model.py` — stage 0b: zero mode

This file locates the n=1 translational
   zero-mode crossing.  It checks the zero-mode residual and the n=1 mode
   census, then writes `eig_n1_<tag>.npz`.

   ##### Functions — zero-mode calculation order

   - `find_zero_mode_crossing()` scans and refines the n=1 crossing near
     `s2=0`, then applies the residual and zero-mode tolerances.
   - `_measured_bounce_error()` reads the stage-0 profile-quality diagnostics.

The two eigenvalue programs require `bounce_data_<tag>.npz` and are
independent of one another.  They must nevertheless be run after the bounce,
because both use that exact profile.

### 1. Generate the residual partial-wave band

#### 5. `band_adaptive_coupled_toy_model.py` — stage 1: adaptive driver

This file is the stage-1 orchestration
   driver.  It chooses the required `s2_max`, `n_max`, and high-n fit onset
   from the actual bounce and computed data.  Its watchers stop the s2
   direction when the Baacke law is reached and stop the n direction when the
   asymptotic tail is bounded.  It writes `band_cutoffs_<tag>.npz`, a
   manifest, and the residual-band slice files.

   ##### Functions — adaptive-band calculation order

   - `cutoff_first_estimate()` makes the initial s2 and partial-wave targets
     from the bounce moments.
   - `s2_ladder()` / `ladder_snap()` construct the certified s2 grid.
   - `plan_missing()` finds missing wave/s2 blocks; `_runs()` groups contiguous
     wave ranges for workers.
   - `engine_cmd()` builds worker commands and `run_jobs()` runs them.
   - `evaluate_watchers()` applies the s2-plateau, n-onset, and n-tail-bound
     decisions to the accumulated tables.
   - `main()` repeats these steps until the adaptive cutoffs pass and writes
     the cutoffs plus manifest.

#### 6. `delta_g_bar_greater_equal_3_coupled_toy_model.py` — band worker

This file is the numerical
   band worker called by the adaptive driver.  For each partial wave and
   deformation parameter `s2`, it solves the coupled h-ODE and returns the
   Born-subtracted residual trace containing orders `O(U^3)` and higher,
   `delta_n(s2)`.  It normally should be run through
   `band_adaptive_coupled_toy_model.py`, not manually; direct execution is
   useful for a single diagnostic slice or self-test.

   ##### Class/functions — residual-trace calculation order

   - `load_background()` reconstructs the bounce, masses, Hessian insertion,
     and radial grid.
   - `ResidualBand` solves the coupled h-ODE and forms the Born-subtracted
     `delta_n(s2)` residual containing orders `O(U^3)` and higher.
     Within it, `h_inf()` obtains asymptotic mode corrections;
     `_rhs()`/`_rhs_scalar()` define the matrix/scalar ODEs;
     `_solve_on_grid()` integrates them; and `Umat()`/`Uscal()` evaluate the
     matrix or scalar insertion. `_band_grid()` builds the radial grid.
     `_eval_point_scalar()`, `eval_point()`, and `_deg_weighted_trace()` turn
     the solutions into trace and Born-order contributions.
   - `_graded_residual_scalar()` and `_graded_residual_vec()` perform the
     scalar and coupled Born-subtraction algebra; `_seg()` extracts the
     relevant radial segment.
   - `_run_slice()` packages one requested slice or wave block as an output
     file.
   - `selftest_fastpath()` compares the optimized decoupled path with the
     general calculation; `main()` provides direct worker execution.

### 3. Compute the special n=0 and n=1 sectors

#### 7. `gbar_n0_coupled_toy_model.py` — stage 3a

This file computes the n=0 continuum resolvent
   integral.  It reads the negative eigenvalue, subtracts its pole, performs
   the adaptive s2 quadrature, and writes `sector_n0_<tag>.npz`.

   ##### Function — sector calculation order

   - `main()` loads the negative eigenvalue, evaluates the pole-subtracted
     n=0 resolvent on the adaptive s2 grid, checks sector convergence and
     cutoff guards, and writes the sector file.

#### 8. `gbar_n1_coupled_toy_model.py` — stage 3b

This file performs the corresponding n=1
   continuum resolvent calculation.  It reads the zero-mode eigenvalue,
   subtracts the four-fold zero-mode pole, and writes
   `sector_n1_<tag>.npz`.

   ##### Function — sector calculation order

   - `main()` loads the four-fold zero-mode eigenvalue, subtracts its pole,
     performs the adaptive n=1 resolvent integral, checks the guards, and
     writes the sector file.

These are the special sectors because the negative and translational modes
must be treated explicitly.  They are not included in the ordinary n>=2 band
sum.

### 4. Complete the finite-s2 band integrals

#### 9. `tail_s2_completion_coupled_toy_model.py` — stage 4

This file reads the residual-band
   slices and integrates each n>=2 wave from s2=0 to the delivered cutoff.
   It then adds the calibrated analytic high-s2 completion, producing
   `I_n(infinity)` and writing `band_integrals_<tag>.npz`.

   ##### Function — completion calculation order

   - `main()` validates the slice manifest and provenance, integrates each
     finite-s2 wave with the shared quadrature, adds the calibrated high-s2
     completion, checks S2-STEP/S2-BAACKE, and writes `band_integrals`.

### 5. Complete the high-n partial-wave sum

#### 10. `tail_high_n_zeta_coupled_toy_model.py` — stage 5

This file fits the completed waves to
    the odd large-n form
    `a/nu^3 + c/nu^5 + e/nu^7`, with `nu=n+1`, and sums all waves above
    `n_max` using Hurwitz zeta functions.  It writes
    `tail_highn_<tag>.npz`.  Only the summed tail enters the final determinant;
   the fit coefficients are also retained as convergence diagnostics.

   ##### Function — high-n calculation order

   - `main()` reads the completed waves, performs the odd inverse-power fit and
     Hurwitz-zeta sum through `odd_tail_fit()`, checks the fit and tail guards,
     and writes `tail_highn_<tag>.npz`.

### 6–7. Renormalization counterterms

#### 11. `counterterm_tadpole_coupled_toy_model.py` — stage 6

This file computes the finite
    one-insertion (tadpole) counterterm `A1_fin` in the MS-bar prescription.
    It writes `ct_tadpole_<tag>.npz`.  This term enters the final assembly with
    a plus sign.

    ##### Functions — tadpole calculation order

    - `compute_A1_fin()` forms the diagonal insertion and evaluates the finite
      MS-bar one-insertion radial integral.
    - `main()` loads the bounce, adds metadata, and writes the counterterm file.

#### 12. `counterterm_fish_coupled_toy_model.py` — stage 7

This file computes the finite
    two-insertion bubble/fish counterterm `A2_fin`, using the radial Hankel
    transform of the Hessian insertion and an adaptive momentum cutoff.  It
    writes `ct_fish_<tag>.npz`.  This term enters the final assembly as
   `-A2_fin/2`; an unconverged adaptive cutoff is rejected by the assembler.

   ##### Functions — fish calculation order

   - `_compute_A2_fin_at()` evaluates the fish integral at one momentum cutoff.
   - `compute_A2_fin()` grows the cutoff until successive values converge and
     records the adaptive trace.
   - `main()` loads the bounce, calls the adaptive calculation, and writes
     `ct_fish_<tag>.npz`.

The two counterterm stages only require the bounce and can be run in either
order.  They are placed here in the documentation because their contributions
are added after the determinant pieces have been computed.

### 8. Assemble the renormalized determinant

#### 13. `assemble_lnD_coupled_toy_model.py` — stage 8

This file combines the n=0 and n=1
    resolvent sectors, the ordinary band sum, the high-n zeta tail, and the
    counterterms:

    ```text
    lnD_ren = sector block + band sum + high-n tail
              + A1_fin - A2_fin/2
    ```

    It rechecks provenance, sector-cutoff convergence, the fictitious-scale
    identity, and counterterm convergence before writing
    `D_integral_<tag>.npz`.  This is the final numerical output used by the
    plotting/decay-rate analysis.

    ##### Functions — assembly calculation order

    - `_load_stage()` loads and validates each stage output.
    - `fict_block_terms()` evaluates the closed-form fictitious-mode and
      sector-tail terms.
    - `main()` checks provenance and convergence guards, adds the sector block,
      band sum, high-n tail, and counterterms in the certified order, then
      writes `D_integral_<tag>.npz`.

## Shared implementation modules

These files are libraries rather than independent calculation stages:

### `pipeline_helpers_coupled_toy_model.py` — shared plumbing

Common file naming and CLI
  options, bounce loading, provenance hashes, potential insertion, matching
  determinants, band-table merging, and validated I/O.

  ##### Functions — shared data flow

  `bounce_arrays()` / `load_bounce()` read and reconstruct the common
  background; `potential_insertion_V()` builds the Hessian insertion;
  `sector_matching_matrix()` / `sector_matching_det()` locate spectral
  crossings; and `read_band_tables()` merges and validates adaptive slices.
  `wall_radius()` supplies the centrifugal scale. `require_finite()` and
  `atomic_savez()` protect numerical outputs, while `stage_paths()`,
  `add_standard_cli()`, `bounce_sha256()`, `potential_id()`, and
  `provenance_stamp()` keep stages reproducible and cross-consistent.
  `crossing_multiplicity()`, `_scan_det_svmin()`, and
  `stable_crossing_scan()` diagnose and robustly locate matching-determinant
  crossings.
### `fv_analytic_coupled_toy_model.py` — false-vacuum functions

Analytic false-vacuum Bessel
  functions and their stable derivatives/products.  The false-vacuum Green
  function is never discretized.

  ##### Functions — false-vacuum evaluation order

  `IK_prod()` evaluates the stable analytic Bessel product. `dlogI_fast()` and
  `dlogK_fast()` provide stable logarithmic derivatives;
  `dlogI_and_logI_fast()` and `dlogK_and_logK_fast()` return derivatives and
  logarithms from one scaled-Bessel evaluation. Private fallback helpers select
  uniform-asymptotic forms when direct evaluation is unsafe.
### `pipeline_quadrature_coupled_toy_model.py` — shared quadrature

The shared log/linear
  Simpson s2 quadrature, sector grids, high-s2 shape completion, odd-family
  fit, and Hurwitz-zeta tail utilities.

  ##### Functions — quadrature calculation order

  `sector_s2_grid()` creates the pole-aware sector grid;
  `s2_integral()` performs the production integral;
  `pole_subtracted_sector_integral()` and `adaptive_sector_integral()` apply
  it to the n=0/1 sectors and refine until converged.
  `s2_tail_shape_ratio()` supplies the high-s2 completion;
  `odd_tail_fit()` supplies the high-n fit and zeta sum; and
  `sector_tail_error()` estimates unresolved sector-cutoff error.
  `_sector_simpson_split()` performs the pole-aware split Simpson step used by
  the sector integrator.
### `watchers_guards_coupled_toy_model.py` — convergence decisions

Convergence watchers used to
  decide how much data to generate and guards used to recheck it.  Watchers
  and guards only control acceptance/cutoffs; they do not contribute numbers
  to `lnD_ren`.

  ##### Functions — convergence decision order

  `analytic_moments()` computes the shared third-Born moments;
  `band_candidate_sums()` prepares adaptive measurements.
  `watch_s2_plateau()`, `watch_n_onset()`, and `watch_n_max_needed()` choose
  the required cutoffs. `guard_s2_baacke()`, `guard_n_asymptote()`, and
  `guard_n_tail_bound()` recheck those choices on final data.

## Plotting and outputs

### `plot_bounces_coupled_toy_model.py` — presentation-only

This module plots all
downward bounce pairs of the hardcoded model and marks the pair used by the
pipeline.  It does not contribute to `lnD_ren` and may be run independently
once the model/bounce data are available.

##### Functions — plotting order

`load_or_solve()` obtains cached or fresh profiles; `_landscape()` builds the
potential plot and `_mark_vacua()` labels endpoints. `plot_gallery()` creates
the overview, `plot_one()` creates an individual figure, and `main()` selects
the model and output directory.

Generated `.npz`, `.json`, slice, and plot files should normally be placed in
the selected data/output directory rather than mixed with these source files.
Keep the same tag and bounce provenance across a complete run; the stages are
designed to fail loudly when files from different bounces are combined.
