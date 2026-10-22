# add_on_greensfct -- the Green's-function matrices of the bounce

This folder computes the Green's function of the fluctuation operator `M_b` around the bounce as a
matrix in field space, `[G(x, x', s^2)]_ij`, component by component.  Its end product is the
renormalised coincident Green's function `[G_ren(x, x, s^2)]_ij` (dimensional regularisation, MSbar),
the second end product of the pipeline next to `lnD_ren`.  The other codes compute the objects
around it: one radial wave `G_n(r, r')`, the position-space matrix off the coincident point, the
unrenormalised coincident matrix, their false-vacuum counterparts, and the pole terms of the two
dangerous modes.

The folder has no physics of its own; every object is assembled from pipeline objects.  From the
production engine (`GreensFunction` in `computation/greens_function_vgeneral_phi_v2.py`), solved on
the run's bounce background, come the two h-branches `h^-` and `h^+` of each wave, the constant
Wronskian `C = 1 + h^-(inf)`, read off at `r_max`, and the coincident bracket
`G_bracket` with its Dyson orders.  From the two MSbar add-back codes of the determinant level
(stages 5 and 6) come the closed forms: `tau_fin_fv` from
`computation/addback_tadpole_vgeneral_phi_v2.py`, and the Fourier-transformed insertion `Utilde`
with the finite bubble `Pi_fin_fv` from `computation/addback_fish_vgeneral_phi_v2.py`.

Every matrix built from the h-branches is the full resolvent.  In the waves `n = 0` (negative mode) and `n = 1`
(translation modes) it contains the pole terms of these modes; the pipeline removes them only at the
determinant level (the sector stages 2a and 2b).  `greensfct_dangerous_modes_v2.py` computes the pole
terms themselves.

## Where the folder hangs in the workflow

The codes read these outputs of the main computation (`run_vgeneral_phi_v2_production.sh`), all in
the run's data directory and named with the run tag `<tag>`:

* `bounce_data_<tag>.npz` (stage 0, level 1): the bounce and the model.  Every code rebuilds from it
  the radial grid `R`, the bounce `phi_b` in the primed frame, the channel masses `m2`, the scale
  `mu = sum_i m_i` and the interaction Hessian `U_int(r) = H'(phi_b(r)) - diag(m2)`, the Hessian of
  the potential on the bounce (primed frame) minus its false-vacuum value.
* `band_cutoffs_<tag>.npz` (stage 1, level 2): the wave cutoff `n_max`, where the wave sums stop.
* `eig_n0_<tag>.npz`, `eig_n1_<tag>.npz` (stages 0a and 0b, level 2): the eigenvalues `omega2_00` and
  `omega2_10` of the two dangerous modes.

A stage file is opened only if it carries the fingerprint of the same bounce.  The codes write their
results into the same data directory; the main computation reads none of them, and no code of this
folder reads the output of another.

## The tree

Levels are the downward steps (a code can run once the trunk files it reads exist); the codes of one
level are independent branches that can run at the same time.  The letters are those of the
workflow tree in the main README, where the stages hold the other letters of each level; `2(f)`
is code (f) of level 2.

```
TRUNK OUTPUTS READ
  level 1   bounce_data_<tag>.npz                      stage 0
  level 2   eig_n0_<tag>.npz, eig_n1_<tag>.npz          stages 0a, 0b   (omega2_00, omega2_10)
            band_cutoffs_<tag>.npz                      stage 1         (n_max)
                |
                v
LEVEL 2   needs only the bounce -- two independent branches
  (f) greensfct_off_coincident_radial_v2.py     [G_n(r, r', s2)]_ij, one wave n
  (g) greensfct_fv_coincident_renorm_v2.py      G0_fv_ren_addback = delta_ij tau_fin_fv(kappa_i^2)
                |
                v
LEVEL 3   needs the bounce and a level-2 trunk file -- seven independent branches
  n_max from band_cutoffs_<tag>.npz:
  (d) greensfct_off_coincident_v2.py            [G(x, x', s2)]_ij at one angle Theta
  (e) greensfct_off_coincident_vs_theta_v2.py   [G(x, x', s2)]_ij against Theta, two fixed radii
  (f) greensfct_coincident_unrenorm_v2.py       [G(x, x, s2)]_ij, cut off at n_max
  (g) greensfct_fv_off_coincident_v2.py         [G_FV(x, x', s2)]_ij
  (h) greensfct_fv_coincident_v2.py             the waves of [G_FV(x, x, s2)]_ij
  (j) greensfct_coincident_renorm_v2.py         *** END PRODUCT ***  [G_ren(x, x, s2)]_ij
  omega2_00, omega2_10 from eig_n0_<tag>.npz, eig_n1_<tag>.npz:
  (i) greensfct_dangerous_modes_v2.py           pole terms u_i u_j / (omega2 + s2), n = 0, 1
```

Codes 3(d), 3(f), 3(g), 3(h) and 3(j) take `--n-max`; when it is given, the code does not read
`band_cutoffs_<tag>.npz` and needs only the bounce.  Code 3(e) always reads `n_max` from that
file.  Three codes reuse a function of another code, not its file: 3(d) and 3(e) build every wave
with `build_G_n` of 2(f), and 3(j) takes its Dyson-order-0 add-back from the function
`fv_coincident_renorm_matrix` of 2(g).  All nine codes use the shared toolbox
`greensfct_toolbox_v2.py`.

How the end product `G_ren` is built, inside code 3(j):

```
bounce_data_<tag>.npz --> R, U_int, m2, mu
  --> for every wave n = 0..n_max: the branches h^-, h^+ and the constant Wronskian C = 1 + h^-(inf)
  --> the coincident bracket G_bracket, split into its Dyson orders 0, 1, 2, >= 3
G_remainder       = sum_{n=0}^{n_max} (n+1)^2/(2 pi^2) K_nu(kappa_i r) I_nu(kappa_j r)/r^2
                                      [Dyson orders >= 2 of G_bracket]_ij
                    (every wave without its two subtraction waves, the free wave of Dyson order 0
                     and the tadpole wave of Dyson order 1, whose wave sums diverge)
G0_fv_ren_addback = delta_ij tau_fin_fv(kappa_i^2)                                 Dyson order 0, MSbar
G1_ren_addback    = - inverse Hankel transform of Utilde_ij(q) Pi_fin_fv_ij(q^2)   Dyson order 1, MSbar
G_ren             = G0_fv_ren_addback + G1_ren_addback + G_remainder
```

How `G_ren` is tied to `lnD_ren`: the determinant does not read these files; the engine traces the
same waves (`GreensFunction.gbar_traces`).  Traced over the fields and space,
`sum_i Int dr 2 pi^2 r^3 [.]_ii`, the weighted wave `(n+1)^2/(2 pi^2) G_n(r, r)` minus its free wave
is the engine's `Gbar_minus_fv(n, s2)`, and its Dyson-order `>= 2` part is `Gbar_2 + delta_geq3`; the
determinant also subtracts the fish `Gbar_2` (its finite part returns as `-A2_fin/2`) and integrates
`delta_geq3` over `s2`.  The traced Dyson-order-1 add-back, integrated as `-Int_0^inf ds2` in dim reg,
is the tadpole add-back `A1_fin`; the Dyson-order-0 add-back cancels in the ratio to the false-vacuum
determinant.

## Running the codes

Run every code from the pipeline folder, once the trunk files it reads exist (see the tree).  The options
shared by all codes:

* `--s2` (required): the spectral parameter `s^2`; every channel mass is shifted,
  `kappa_i = sqrt(m_i^2 + s^2)`.  A value at which a wave has a bound state (`det C = 0`) aborts with
  the request to choose another `s2`.  A value at or below `-min_i m_i^2` (the lightest `kappa_i`
  would not be positive) aborts at once, before any wave is solved, and so does a negative `--n`
  or `--n-max` (`check_cli` of the toolbox).
* `--data-dir`: the run's data directory (default `$VGENERAL_PHI_DATA`, else the current folder); the output is written there.
* `--bounce-npz`: the bounce file (default `$VGENERAL_PHI_BOUNCE_NPZ`, else the one
  `bounce_data_*.npz` in the data directory; if the directory holds several, pass one).
* `--tag`: the run tag (default: read from the bounce file name `bounce_data_<tag>.npz`).
* `--n-r`: the number of points of the radial grid, which runs from the engine's `r_start` to
  `r_max` (default 300); not taken by 2(g), whose matrix does not depend on `r`, and 3(e), whose
  radial samples are its two radii.

Every output file is named after the code that wrote it, followed by the run tag, the wave `n`
(radial code 2(f) only) and `s^2`:

    <code name>_<tag>[_n<k>]_s2<v>.npz

with `<v>` the value of `s^2` to six decimals and the decimal point written `p`
(`s^2 = 0.5` gives `s20p500000`).  Nothing else is in the name: `--n-max`, `--mu`, `--n-r` and the
other options are not, so a rerun with other values into the same `--data-dir` overwrites the
file.  Besides the keys listed below, every file stores `s2` and the
provenance stamps `bounce_sha`, `potential_id`, `code_version`: the fingerprint of the bounce it
was computed from.

## The codes

**`greensfct_toolbox_v2.py`** -- the shared toolbox of the folder; it is imported by every code and
writes no file.
* `solve_branches`: solves the two h-branches of one wave `n` at one `s2` through the engine's
  assembly `GreensFunction.branch_objects`, giving `h^-`, `h^+`, the constant Wronskian `C`, its inverse
  `C_inv`, the coincident bracket `G_bracket` and its Dyson orders 1, 2 and `>= 3`.
* `log_bessel_tables`, `fv_weight_matrix`: the free Bessel functions in log space, and the
  false-vacuum (free Bessel) weight `K_nu(kappa_i r) I_nu(kappa_j r)/r^2` (`nu = n + 1`) that every
  coincident wave carries, entry by entry; its diagonal is the free coincident wave.
* `coincident_wave`: one wave of the coincident matrix, in full and as its part of Dyson order
  `>= 2`.
* `chebyshev_U`: the angular weights `U_n(cos Theta)` of the wave sum.
* the shared command-line options, the bounce and tag lookup (`resolve_bounce`), `read_n_max` (the
  cutoff `n_max` from `band_cutoffs_<tag>.npz`), and the writer of every output file
  (`save_greensfct_npz`).

**`greensfct_off_coincident_radial_v2.py`** -- the radial Green's function `[G_n(r, r', s^2)]_ij` of
one wave `n`.
* It computes the radial master formula (`build_G_n`): the regular free solution `I_nu(kappa_i r)/r`
  dressed by `1 + h^-`, the decaying one `K_nu(kappa_i r)/r` dressed by `1 + h^+`, matched through
  `C^-1`; for `r > r'`
  `[G_n]_ij = K_nu(kappa_i r) I_nu(kappa_j r')/(r r') sum_ab [1 + h^+(r)]_ia [C^-1]_ba [1 + h^-(r')]_jb`,
  and the case `r < r'` follows by exchanging `(r, i)` with `(r', j)`.
* Input: the bounce; `--n` and `--s2` (both required).
* Output: `greensfct_off_coincident_radial_v2_<tag>_n<k>_s2<v>.npz` with `r_grid`, `G_n` (axes `r`,
  `r'`, `i`, `j`), `C`, `kappa_i`, the branches `h_minus` and `h_plus`, the logarithms of the two
  free solutions `log_B_minus_fv`, `log_B_plus_fv`, and `coincident_sym_dev`: the largest asymmetry
  `|G_ij(r, r) - G_ji(r, r)|` relative to `G_n(r, r)`, which measures the continuity of `G_n` at
  `r = r'`.
* For `n = 0, 1` the wave contains the pole terms of the negative mode and the translation modes.

```
python3 add_on_greensfct/greensfct_off_coincident_radial_v2.py --s2 <s2> --n <n> --data-dir results
```

**`greensfct_off_coincident_v2.py`** -- the position-space matrix `[G(x, x', s^2)]_ij` for one angle
`Theta` between `x` and `x'`, as a function of `r = |x|` and `r' = |x'|`.
* It computes the wave sum `1/(2 pi^2) sum_{n=0}^{n_max} (n+1) U_n(cos Theta) G_n(r, r')`, every wave
  from `build_G_n`.
* Input: the bounce, `n_max`; `--theta` (default 0), `--n-max`.
* Output: `greensfct_off_coincident_v2_<tag>_s2<v>.npz` with `r_grid`, `G_off_coincident` (axes `r`,
  `r'`, `i`, `j`), `theta`, `n_max`, and `last_wave_magnitude` (the size of the last wave added).
  On the diagonal `r = r'` the wave sum does not converge; there the matrix is set to NaN.

```
python3 add_on_greensfct/greensfct_off_coincident_v2.py --s2 <s2> --theta <Theta> --data-dir results
```

**`greensfct_off_coincident_vs_theta_v2.py`** -- the same matrix `[G(x, x', s^2)]_ij` as a function of
the angle `Theta` in `[0, pi]`, at one fixed pair of radii `r0`, `r0'`.
* It computes `G_n(r0, r0')` of every wave with `build_G_n`, then the wave sum at every `Theta`.
* Input: the bounce, `n_max` (always from `band_cutoffs_<tag>.npz`); `--r` and `--r-prime` (default:
  the wall radius `r_W` and `r_W/2`; the two must differ), `--n-theta` (number of angles, default
  361).
* Output: `greensfct_off_coincident_vs_theta_v2_<tag>_s2<v>.npz` with `theta_grid`, `G_theta`, the
  single waves `G_n_at_r0_pair`, the partial sums at `Theta = 0` `G_partial_sums_theta0` (the
  convergence in `n`), `r0`, `r0_prime`, `r_W`, `n_max` and `last_wave_magnitude`.

```
python3 add_on_greensfct/greensfct_off_coincident_vs_theta_v2.py --s2 <s2> --data-dir results
```

**`greensfct_coincident_unrenorm_v2.py`** -- the unrenormalised coincident matrix `[G(x, x, s^2)]_ij`,
which diverges as the wave cutoff grows.
* It computes the sum of the coincident waves
  `(n+1)^2/(2 pi^2) K_nu(kappa_i r) I_nu(kappa_j r)/r^2 [G_bracket]_ij` over `n = 0..n_max`; here the
  cutoff `n_max` is an ultraviolet regulator.
* Input: the bounce, `n_max`; `--n-max`.
* Output: `greensfct_coincident_unrenorm_v2_<tag>_s2<v>.npz` with `r_grid`, `G_unren` (sum to
  `n_max`), `G_unren_half` (sum to `n_max/2`, which shows the dependence on the cutoff), `n_max`,
  `n_half`.

```
python3 add_on_greensfct/greensfct_coincident_unrenorm_v2.py --s2 <s2> --data-dir results
```

**`greensfct_fv_off_coincident_v2.py`** -- the false-vacuum matrix `[G_FV(x, x', s^2)]_ij`, finite and
diagonal in the fields.
* It computes the same angular wave sum over the free waves
  `delta_ij I_nu(kappa_i r_<) K_nu(kappa_i r_>)/(r r')` (closed Bessel forms, no h-branches), with
  `r_<` and `r_>` the smaller and the larger of `r`, `r'`, and compares it with the free propagator
  in four dimensions, `kappa_i K_1(kappa_i d)/(4 pi^2 d)`, `d = |x - x'|`.
* Input: the masses and the radial box of the bounce, `n_max`; `--theta` (default 0), `--n-max`.
* Output: `greensfct_fv_off_coincident_v2_<tag>_s2<v>.npz` with `r_grid`, `G_fv` (NaN on `r = r'`),
  `theta`, `n_max`, `kappa_i`, and `free_propagator_fv_dev` (the largest relative deviation from the
  free propagator).

```
python3 add_on_greensfct/greensfct_fv_off_coincident_v2.py --s2 <s2> --theta <Theta> --data-dir results
```

**`greensfct_fv_coincident_v2.py`** -- the false-vacuum coincident matrix, wave by wave; its wave sum
diverges.
* It computes the free waves `delta_ij I_nu(kappa_i r) K_nu(kappa_i r)/r^2` (the diagonal of
  `fv_weight_matrix`, no h-branches).  These are the Dyson-order-0 waves that the renormalised code
  3(j) leaves out of its sum.
* Input: the masses and the radial box of the bounce, `n_max`; `--n-max`.
* Output: `greensfct_fv_coincident_v2_<tag>_s2<v>.npz` with `r_grid`, the unweighted waves
  `G_n_fv_coinc_diag`, the weighted partial sums `G_fv_sum` (to `n_max`) and `G_fv_sum_half` (to
  `n_max/2`), `kappa_i`, `n_max`, `n_half`.

```
python3 add_on_greensfct/greensfct_fv_coincident_v2.py --s2 <s2> --data-dir results
```

**`greensfct_fv_coincident_renorm_v2.py`** -- the renormalised false-vacuum coincident matrix
`[G0_fv_ren_addback]_ij = delta_ij tau_fin_fv(kappa_i^2)`, with
`tau_fin_fv(W^2) = W^2/(16 pi^2) [ln(W^2/mu^2) - 1]`.
* It computes the closed form (dimensional regularisation, MSbar) at the shifted masses
  `kappa_i^2 = m_i^2 + s^2`, with `tau_fin_fv` taken from the tadpole add-back code
  `computation/addback_tadpole_vgeneral_phi_v2.py`.  It solves no h-branch.
* Its function `fv_coincident_renorm_matrix` is the Dyson-order-0 add-back `G0_fv_ren_addback` of the
  renormalised code 3(j).
* Input: the masses of the bounce; `--mu` (default `mu = sum_i m_i`).
* Output: `greensfct_fv_coincident_renorm_v2_<tag>_s2<v>.npz` with `G0_fv_ren_addback` (an
  `N x N` matrix, independent of `r`), `kappa_i`, `s2`, `mu`.

```
python3 add_on_greensfct/greensfct_fv_coincident_renorm_v2.py --s2 <s2> --data-dir results
```

**`greensfct_coincident_renorm_v2.py`** -- the end product: the renormalised coincident matrix
`[G_ren(x, x, s^2)]_ij` (dimensional regularisation, MSbar).
* It computes `G_ren = G0_fv_ren_addback + G1_ren_addback + G_remainder`.
  * `G_remainder`: the sum over `n = 0..n_max` of the weighted coincident waves of Dyson order
    `>= 2` only (`coincident_wave`).  The divergent Dyson orders 0 and 1 are left out wave by wave, so
    the wave sum converges; every wave is solved once.
  * `G0_fv_ren_addback = delta_ij tau_fin_fv(kappa_i^2)`, the finite part of Dyson order 0, from
    `greensfct_fv_coincident_renorm_v2.py`.
  * `G1_ren_addback`, the finite part of Dyson order 1: minus the inverse Hankel transform of
    `Utilde_ij(q) Pi_fin_fv_ij(q^2)`, the Fourier transform of the insertion times the finite bubble,
    both taken from the fish add-back code `computation/addback_fish_vgeneral_phi_v2.py`, the bubble
    evaluated at the shifted masses `kappa_i`.
* Input: the bounce, `n_max`; `--n-max`, `--mu` (default `mu = sum_i m_i`).
* Output: `greensfct_coincident_renorm_v2_<tag>_s2<v>.npz` with `r_grid`, `G_ren`, its three pieces
  `G0_fv_ren_addback`, `G1_ren_addback`, `G_remainder`, and `n_max`, `s2`, `mu`.

```
python3 add_on_greensfct/greensfct_coincident_renorm_v2.py --s2 <s2> --data-dir results
```

**`greensfct_dangerous_modes_v2.py`** -- the pole terms of the two dangerous modes in the radial
Green's function: the negative mode (wave `n = 0`) and the translation modes (wave `n = 1`).
* It reads the eigenvalues `omega2_00` and `omega2_10` from `eig_n0_<tag>.npz` and
  `eig_n1_<tag>.npz`.  At `s^2 = -omega^2` the matrix `C` is singular, and its null vector gives the
  radial eigenfunction `u_n(r)`: the regular branch inside, the decaying branch outside, normalised
  to `sum_i Int dr r^3 u_i^2 = 1`.  From it the code forms the coincident term
  `u_i(r) u_j(r)/(omega^2 + s^2)` and the off-coincident term `u_i(r) u_j(r')/(omega^2 + s^2)`.  The
  radial terms carry no degeneracy factor; the `(n+1)^2` copies are angular.
* Input: the bounce, the two eigenvalue files; `--s2`, `--n-r` (the same radial grid as the radial
  code, so the two outputs can be subtracted point by point).
* Output: `greensfct_dangerous_modes_v2_<tag>_s2<v>.npz` with `r_grid`, `omega2_00`, `omega2_10`, the
  eigenfunctions `u_n0`, `u_n1`, the coincident terms `pole_term_coinc_n0`, `pole_term_coinc_n1`, the
  off-coincident terms `pole_term_offc_n0`, `pole_term_offc_n1`, and three measures of the construction:
  `null_quality_n0/n1`, the smallest singular value of `C` relative to `1 + max|C - 1|` (above
  `VERIFY_FRAC` the code aborts, because the stored eigenvalue does not make `C` singular here),
  `match_dev_n0/n1`, the relative jump of the stitched eigenfunction just outside the matching
  radius, and `overlap_dphi_n1`, the overlap of `u_1` with the normalised bounce slope `phi_b'`.

```
python3 add_on_greensfct/greensfct_dangerous_modes_v2.py --s2 <s2> --data-dir results
```
