# pipeline_rk_vgeneral_phi_v2

## 1. How to install and run

### Install

```bash
git clone https://github.com/amrhima/Vacuum_decay_2_fields.git
cd Vacuum_decay_2_fields/G_F_/v5
python3 -m venv env
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Activate the environment with `source env/bin/activate` in every new shell: every
command in this README and in the folder READMEs assumes it.  Without activation,
write `env/bin/python` for `python3` and put `PY=env/bin/python` in front of the
driver, `run_vgeneral_phi_v2_production.sh`.  The pipeline runs on Linux
and macOS with bash (on Windows inside WSL; on Debian and Ubuntu, `python3 -m venv`
needs the package `python3-venv`); no compiler is needed.  Any Python >= 3.11 works
(matplotlib 3.11 needs it).  `requirements.txt` lists the minimum package versions,
explains why each package is needed and names the versions the pipeline was tested
with.

### Step 1: enter your potential

Every letter you list in `--fields` is one field component: `x,y` means two fields,
`x,y,z` three, and so on; the letters are distinct lowercase letters other than `c`.
The potential is a polynomial of total degree <= 4, entered as monomial coefficients.
A monomial name joins the field letters in the order of `--fields`, each with its
power, and power 1 is omitted: `x2` = x^2, `xy2` = x*y^2 (with `--fields x,y`, `xy2`
and not `y2x`), `c` = the constant.  The worked example:

```bash
./run_vgeneral_phi_v2_production.sh \
    --fields x,y \
    --couplings 'c=1.2,x2=-2,y2=-2,xy2=1,x2y2=0.15,x4=1,y4=1' \
    --select-bounce --data-dir results
```

The pipeline finds the minima of V (a root search from a grid of seeds inside a box that
holds every stationary point; a minimum whose basin holds no seed is missed) and solves
the bounce of every downward vacuum pair: every bounce the potential allows.  Before it
computes anything else, it draws all of them in the bounce gallery `results/gallery/`
(the folder also opens in the file browser), two pictures per bounce:
`bounce_<name>_potential.png`, the bounce path from the true vacuum to the false vacuum on
the potential, in the original fieldspace frame, with every vacuum named
(for two fields the surface V next to the landscape -V, for three fields the path in
field space; V is drawn without its constant term c), and `bounce_<name>_profiles.png`,
the bounce profiles of the bounce from the true vacuum to the false vacuum, in the primed
fieldspace frame, with the action S_E
(for one field and for more than three fields the profiles alone).  Then it prints a menu of all bounces, each with its name and S_E, and waits for your
choice (Step 2): the rest of the run computes the fluctuation determinant of the one
bounce you choose.  A bounce whose census is not the accepted one (one negative mode, one
translation zero mode) is drawn dashed and marked SPECIAL; you can look at it, but not
choose it.  The pipeline has no built-in potential: a run without `--fields` and
`--couplings` stops with an error.

### Step 2: choose the bounce by name

The vacua are named so that every bounce has a name: the names of the two vacua it
connects.  The minima are sorted by V, which is the height of the hilltops of the inverted
potential -V.  The deepest minimum, the highest hilltop of -V, is **TV0** (the global true
vacuum); the second deepest is **M1**, the next one **M2**, and so on; the highest
minimum, the lowest hilltop of -V, is **FV0** (the global false vacuum).  Minima of equal
V are ordered by their coordinates.  Every vacuum keeps its label in every bounce name,
in every picture and in the file names.  A bounce is named `<falseLabel>_<trueLabel>`,
for example `FV0_TV0`, `M2_TV0` or `FV0_M1`.
The index form `F<i>_T<j>`, which the file names of the bounce data use, is accepted as
well.  Look at the gallery, pick the bounce you want and type its name (the `<name>` in
the file names of its two pictures, the name in the menu) at the prompt:

```
Write the name of the bounce solution to use (e.g. FV0_TV0): <bounce-name>
```

This name fixes the bounce of the run: every later stage of the downward flow, down to
the fluctuation determinant lnD_ren (and the renormalized Green's function G_ren of the
add-on), is computed for this bounce.  For a run without a terminal, replace
`--select-bounce` with `--pair <bounce-name>`: a first run with `--select-bounce` (or the
gallery of an earlier run of the same potential) shows the names, and the bounces already
solved are reused.
`python3 computation/bounce_vgeneral_phi_v2.py --fields ... --couplings ... --list-bounces`
prints the vacua and all bounce names and then exits.  Every run needs a bounce choice
(`--select-bounce` or `--pair`), otherwise it stops.

The run then works down the levels of the workflow tree (section 2) and ends with the
number `lnD_ren`.  This number and every part it is assembled from are stored in
`<data-dir>/lnD_ren_<tag>.npz`.  The band stage (stage 1) takes by far the longest.

**Check values for the example.**  Choose `M2_TV0` at the prompt (or run with
`--pair M2_TV0` instead of `--select-bounce`).  Stage 0 gives this bounce the action
S_E = 282.8624, and the run ends with lnD_ren = +26.3372 at the scale mu = 4.3624 (other
package versions agree to about 1e-4); the watchers stop at n_max = 80 and
s2_max = 220.  With the default `BAND_JOBS=2` the run takes about one hour on a laptop
and writes about 6 MB into `results/` (the add-on files of Step 3 add about
3 MB, all nine add-on codes about 15 MB).  A thinner
wall needs many more waves and takes hours.  Set `BAND_JOBS` to the number of free
cores; each worker uses one core.

**Reading the result.**  The last lines of the run, also in `<data-dir>/run.log`, list
the seven terms of the assembly and `lnD_ren`.  In Python:

```python
import glob, numpy as np
f = sorted(glob.glob('results/lnD_ren_*.npz'))[-1]
d = np.load(f, allow_pickle=True)
b = np.load(f.replace('lnD_ren_', 'bounce_data_'), allow_pickle=True)
print(float(d['lnD_ren']), float(d['mu']), float(b['S_E']))
```

lnD_ren is the MS-bar value at mu = sum_i m_i; S_E is stored with the bounce.  The
accuracy targets `EPS_S2` and `EPS_N` (1e-2 by default) set the cutoffs, and the run
prints the bounce sensitivity `eps_bounce` at its end.

**Rerunning.**  You can rerun the driver at any point with the same command.  A stage is
skipped when its output exists and still matches the bounce, the cutoffs and the
tolerances of the run, and the band computes only its missing slices.  Without `--tag`,
the tag carries the bounce name (`rk_vgeneral_phi_v2_M2_TV0_...` for the example) and is
stored per bounce, `--fields` and `--couplings`: a rerun continues under the same tag
when `--fields` and `--couplings` are written the same way, with `--pair` as the name or
in the file form.  A `--select-bounce` run goes on as a `--pair` run of the bounce it
selected (`[select] chosen bounce: F2_T0`, the bounce `M2_TV0`); to continue it without a
terminal, rerun with `--pair M2_TV0`.

**Good to know.**
- Run every command from the pipeline folder.  The driver reads a relative `--data-dir`
  from the pipeline folder, every other code from the current folder.
- Give every potential its own `--data-dir` (without the option all runs share
  `./results`): a data folder holds the bounces of one potential, and a bounce file of
  another potential in it stops stage 0.
- After updating the code, start new runs in a new `--data-dir`: stored bounces and most
  stage files of an older build are reused on resume.
- Write the values of `--fields`, `--couplings` and `--seed-range` without spaces; a
  space stops stage 0 with 'unrecognized arguments'.

### Step 3: the Green's functions

The add-on `add_on_greensfct/` computes the Green's function of the fluctuation operator around
the bounce as a matrix in field space; its end product is the renormalised coincident Green's
function G_ren, the second end product of the pipeline next to lnD_ren
(`add_on_greensfct/README.md` describes every add-on code).  For the example run above (data
folder `results`, environment active) the block below makes the coincident Green's functions at
s2 = 0.5 and the radial wave n = 3 at the same s2 (the add-on codes re-solve the waves and take
minutes each).  In it, `D` is the run's data folder, `B` its bounce file (set it by hand when the
folder holds several runs) and `T` its tag.

```bash
D=results
B=$(ls $D/bounce_data_*.npz | head -1)
T=$(basename $B .npz); T=${T#bounce_data_}
for c in greensfct_coincident_unrenorm_v2 greensfct_coincident_renorm_v2; do
  python3 add_on_greensfct/$c.py --s2 0.5 --data-dir $D --bounce-npz $B --tag $T
done
python3 add_on_greensfct/greensfct_off_coincident_radial_v2.py --s2 0.5 --n 3 --data-dir $D --bounce-npz $B --tag $T
```

### Driver options and settings

| option | meaning |
|---|---|
| `--fields x,y,...` | the field letters (required; no spaces) |
| `--couplings 'x2=-2,...'` | the monomial coefficients of V (required; no spaces) |
| `--pair NAME` / `--select-bounce` | the bounce: given by name, or chosen from the menu (one of them is required) |
| `--seed-range LO,HI` | the search box for the vacua (no spaces; by default it is derived from the couplings) |
| `--data-dir DIR` | where all output goes (default `./results` next to the pipeline); one folder per potential |
| `--tag TAG` | the run tag in every file name (default: automatic); it must contain `vgeneral_phi_v2` |

Every option can also come from an environment variable: `FIELDS_ARG='--fields x,y'`,
`COUPLINGS_ARG`, `PAIR`, `SELECT_BOUNCE=1`, `SEEDRANGE_ARG`, `DATA_DIR` and `TAG`; the
command line wins.  A variable left exported in the shell therefore takes part in every
run.

Settings are given as environment variables in front of the command, like `PY` above:

| variable | default | meaning |
|---|---|---|
| `PY` | `python3` | the Python interpreter |
| `BAND_JOBS` | 2 | number of band workers that run in parallel |
| `EPS_S2`, `EPS_N` | 1e-2 | accuracy targets for the s2 cutoff and for the wave cutoff |
| `ONSET_TOL`, `A3_TOL` | 0.2, 0.10 | tolerances for the start of the fit window and for the fitted leading coefficient |
| `S2_STEP_TOL`, `SECTOR_STEP_TOL` | 0.005, 1e-3 | the step-size limits of the s2 quadratures: the change of a band integral relative to the largest band integral (stage 3) and the absolute change of a sector term Delta_n (stages 2a, 2b) when every second s2 node is dropped |
| `SECTOR_S2_LO` | derived | the lower end of the sector s2 grid, an absolute s2; the sectors derive it from the model, never above the lowest band node, so set it only to repeat a stored run |
| `PIN_NMAX`, `PIN_S2MAX`, `PIN_FITLO` | unset | fix the cutoffs instead of letting the watchers find them, to repeat a stored run with the same cutoffs: `PIN_NMAX` and `PIN_S2MAX` fix n_max and s2_max and must be set together; `PIN_FITLO` fixes the start of the fit window and may be set alone |

The add-on is not run by the driver.  Run each of its codes by hand once the files it
reads exist; the workflow tree of section 2 shows which files these are.  The commands of
the add-on are in `add_on_greensfct/README.md`.

## 2. The workflow tree

```text
KEY  LEVEL     a step of the downward flow; it starts once every file it reads exists
     (a), (b)  branches: the computations of one level; they read none of each other's
               files, so they can run at the same time
     [0]..[7]  stage of run_vgeneral_phi_v2_production.sh, which runs 0, 0a, 0b, 1, 2a, 2b,
               3, 4, 5, 6, 7 one after another; the add-on codes are run by hand
     <-  ->    file read from / written into the data directory; "of 2(c)": made by branch
               (c) of LEVEL 2; "uses": a function imported from another code
     *_vgeneral_phi_v2.py: computation/; greensfct_*_v2.py:
     add_on_greensfct/, each add-on code writing <code name>_<tag>[_n<k>]_s2<v>.npz

LEVEL 0  THE POTENTIAL
         potential_vgeneral_phi_v2.py (library: built in [0], rebuilt by every later code)
         V(phi), a polynomial of degree <= 4 in N fields (--fields, --couplings); its minima
         TV0, M1, M2, ..., FV0; for a false vacuum the primed frame: the masses m_i^2
         (channel 1 the heaviest), the Hessian H', the cubic couplings g_ijk
            |
            v
LEVEL 1  THE BOUNCE AND THE MODE CENSUS                                                     [0]
         bounce_vgeneral_phi_v2.py with bounce_polish_vgeneral_phi_v2.py (the solve),
         mode_census_vgeneral_phi_v2.py (the census) and bounce_gallery_vgeneral_phi_v2.py
         every downward vacuum pair: seed -> boundary-value solve of phi_b(r) and S_E ->
         census of the waves n = 0, 1 -> reference profile; gallery; selection gate: one
         negative mode and one dangerous n = 1 crossing (the translation zero mode)
         -> bounce_gen/bounce_data_F<i>_T<j>.npz, gallery/       every solved pair
         -> bounce_data_<tag>.npz                               the chosen bounce
            |
            |  every code below reads bounce_data_<tag>.npz and rebuilds the background from
            |  it (pipeline_helpers_vgeneral_phi_v2.py): R, phi_b, m_i^2, mu = sum_i m_i and the
            |  interaction Hessian U_int(r) = H'(phi_b(r)) - diag(m_i^2) of the fluctuation
            |  operator M_b = -Laplacian + H'(phi_b).  The Green's function G(x, x', s2) of
            |  M_b + s2 comes from one engine, greens_function_vgeneral_phi_v2.py: for one wave n
            |  and one s2 the h-ODE for the branches h^-, h^+, the constant Wronskian C, the
            |  coincident bracket G_bracket and its traces (used by [0], [0a], [0b], [1], [2a],
            |  [2b] and the add-on), with the free Bessel solutions I_nu(kappa_i r)/r,
            |  K_nu(kappa_i r)/r of fv_analytic_toolbox_vgeneral_phi_v2.py
            v
LEVEL 2  NEEDS ONLY THE BOUNCE                                        (branches, independent)
  (a) [0a] eigenvalue_neg_vgeneral_phi_v2.py
           omega2_00, the eigenvalue of the negative mode (wave n = 0), read from the census;
           stops the run unless there is exactly one such mode            -> eig_n0_<tag>.npz
  (b) [0b] eigenvalue_zero_vgeneral_phi_v2.py
           omega2_10, the eigenvalue of the translation zero mode (wave n = 1), read from the
           census; stops the run unless there is exactly one dangerous n = 1 crossing
                                                                           -> eig_n1_<tag>.npz
  (c) [1]  band_adaptive_vgeneral_phi_v2.py, running the engine as parallel workers
           the band: the Born-subtracted remainder delta_geq3(n, s2) of the waves
           n = 2..n_max at 0 < s2 <= s2_max; the watchers of watchers_guards_vgeneral_phi_v2.py
           fix s2_max, n_max and the fit-window start fit_lo
           -> residual_band_<tag>_s2<v>[_n<lo>to<hi>].npz (the slices), band_manifest_<tag>.json,
              band_cutoffs_<tag>.npz (s2_max, s2_min, n_max, fit_lo, A3, baacke_c, targets)
  (d) [5]  addback_tadpole_vgeneral_phi_v2.py
           tadpole moments M_i and A1_fin = sum_i M_i tau_fin_fv(m_i^2)
                                                                   -> addback_tadpole_<tag>.npz
  (e) [6]  addback_fish_vgeneral_phi_v2.py
           A2_fin, the MS-bar finite part of the fish                -> addback_fish_<tag>.npz
  add-on codes (f), (g), through greensfct_toolbox_v2.py (no stage reads add-on files):
  (f)      greensfct_off_coincident_radial_v2.py
           G_n(r, r', s2), the radial Green's function of one wave n
  (g)      greensfct_fv_coincident_renorm_v2.py
           G0_fv_ren_addback = delta_ij tau_fin_fv(kappa_i^2), kappa_i^2 = m_i^2 + s2; uses
           tau_fin_fv of 2(d)
            |
            v
LEVEL 3  NEEDS LEVEL-2 FILES                                          (branches, independent)
         the s2 integrals of (a), (b), (c) use pipeline_quadrature_vgeneral_phi_v2.py
  (a) [2a] gbar_sector_vgeneral_phi_v2.py --sector-n 0
           Delta_n = -Int_0^s2_max [delta_geq3 - 1/(omega2_00 + s2)] ds2, wave n = 0
           <- eig_n0_<tag>.npz of 2(a), band_cutoffs_<tag>.npz of 2(c) (s2_max, s2_min)
           -> sector_n0_<tag>.npz
  (b) [2b] gbar_sector_vgeneral_phi_v2.py --sector-n 1
           the same for wave n = 1, with the pole 4/(omega2_10 + s2)
           <- eig_n1_<tag>.npz of 2(b), band_cutoffs_<tag>.npz of 2(c) (s2_max, s2_min)
           -> sector_n1_<tag>.npz
           (a), (b): delta_geq3 of the waves n = 0, 1 above s2_max is dropped; the
           SECTOR-CUTOFF watcher of [1] and guard of [7] check that it is small
  (c) [3]  band_integrals_vgeneral_phi_v2.py
           every band wave: I_n(s2_max) = -Int_0^s2_max delta_geq3 ds2, its high-s2 tail
           T_n(s2_max) (an estimate) and I_n(inf) = I_n(s2_max) + T_n(s2_max)
           <- the slices, band_manifest_<tag>.json and band_cutoffs_<tag>.npz, all of 2(c)
           -> band_integrals_<tag>.npz
  add-on codes (d)-(j), through greensfct_toolbox_v2.py; (d)-(h) and (j) read n_max
  <- band_cutoffs_<tag>.npz of 2(c) unless --n-max is given ((e) always reads it)
  (d)      greensfct_off_coincident_v2.py
           G(x, x', s2) at one angle Theta between x and x'; uses build_G_n of 2(f)
  (e)      greensfct_off_coincident_vs_theta_v2.py
           G(x, x', s2) against Theta at one pair of radii; uses build_G_n of 2(f)
  (f)      greensfct_coincident_unrenorm_v2.py
           G(x, x, s2) unrenormalised: the wave sum up to n_max, which regulates it
  (g)      greensfct_fv_off_coincident_v2.py
           G_FV(x, x', s2) from the free waves, compared with the free propagator
  (h)      greensfct_fv_coincident_v2.py
           the free coincident waves of G_FV(x, x, s2)
  (i)      greensfct_dangerous_modes_v2.py
           the pole terms u_i(r) u_j(r')/(omega2 + s2) of the waves n = 0, 1
           <- eig_n0_<tag>.npz of 2(a), eig_n1_<tag>.npz of 2(b)
  (j)      greensfct_coincident_renorm_v2.py         *** END PRODUCT: G_ren(x, x, s2) ***
           the renormalised coincident Green's function (dim reg, MS-bar at mu):
           G_ren = G0_fv_ren_addback + G1_ren_addback + G_remainder, the MS-bar finite parts
           of Dyson orders 0 and 1 and the waves n = 0..n_max of Dyson orders >= 2;
           uses fv_coincident_renorm_matrix of 2(g), Utilde_matrix and Pi_fin_fv_q of 2(e)
            |
            v
LEVEL 4  THE WAVES n > n_max                                                                [4]
         tail_high_n_zeta_vgeneral_phi_v2.py
         one fit I_n(inf) ~ a/nu^3 + b/nu^5 + c/nu^7 (nu = n + 1) on [fit_lo, n_max];
         Zcal = a zeta(3, n_max+2) + b zeta(5, n_max+2) + c zeta(7, n_max+2)
         <- band_integrals_<tag>.npz of 3(c), band_cutoffs_<tag>.npz of 2(c)
         -> tail_high_n_<tag>.npz
            |
            v
LEVEL 5  *** END PRODUCT: lnD_ren ***                                                      [7]
         assemble_lnD_vgeneral_phi_v2.py
         lnD_ren = ln|det' M_b / det M_FV|_ren (det' without the four translation zero modes),
         MS-bar at mu
         <- sector_n0 of 3(a), sector_n1 of 3(b), band_integrals of 3(c), tail_high_n of
            LEVEL 4, addback_tadpole of 2(d), addback_fish of 2(e) (each _<tag>.npz) and
            bounce_data_<tag>.npz of LEVEL 1 (census verdict, reference profile)
         uses tau_fin_fv of 2(d) for (T2), (T4), and for the bounce sensitivity indicator
         eps_bounce the A1_fin, A2_fin and A3 routines of 2(d), 2(e) and watchers_guards on
         the reference profile
         lnD_ren =   (T1)        Delta_0 + Delta_1 + ln|omega2_00|
                                 - ln(omega2_00 + s2_max) - 4 ln(omega2_10 + s2_max)
                   + (T2)+(T4)   A1_fin, split at s2_max
                   + (T3)+(T5)   sum_{n=2..n_max} [I_n(s2_max) + T_n(s2_max)]
                   + (T6)        Zcal
                   + (T7)        -A2_fin/2
         -> lnD_ren_<tag>.npz
         G_ren and lnD_ren come from the same engine: lnD_ren integrates over s2 the traces
         of Born orders >= 3 and adds back the MS-bar finite parts of Born orders 1 and 2.
         -Int_0^inf ds2 (dim reg) of the trace of G1_ren_addback is (T2)+(T4) = A1_fin; the
         fish converges at fixed s2 (it stays in G_remainder), but its s2 integral diverges,
         so its add-back -A2_fin/2 exists only here; G0_fv_ren_addback cancels in the ratio
         to the false vacuum.  No add-on file is read here.
```

## 3. The computational codes

- **`run_vgeneral_phi_v2_production.sh`** (the driver): from the potential (`--fields`,
  `--couplings`) and the bounce (`--pair` or `--select-bounce`) it runs the stages 0, 0a, 0b,
  1, 2a, 2b, 3, 4, 5, 6, 7 in one data directory, reusing every stage file that still matches
  the bounce, the cutoffs and the tolerances, and ends with `lnD_ren_<tag>.npz`.
- **`computation/potential_vgeneral_phi_v2.py`** (level 0, library): V of the fields and
  couplings with its derivatives, its minima (`find_all_minima`) and the primed frame of a false
  vacuum (`PrimedPotential`: masses m_i^2, heaviest first, Hessian H', cubic couplings g_ijk); it
  writes nothing, every later code rebuilds V from the bounce file.
- **`computation/bounce_vgeneral_phi_v2.py`** (level 1, stage 0): for every downward vacuum pair
  the seed (CosmoTransitions path deformation), the boundary-value solve, the census of the waves
  n = 0, 1 and a tighter reference solve, stored in `bounce_gen/bounce_data_F<i>_T<j>.npz`; the
  chosen pair must pass the selection gate and becomes `bounce_data_<tag>.npz`.
- **`computation/bounce_polish_vgeneral_phi_v2.py`** (library of stage 0): the bounce solve, one
  `solve_bvp` of d^2phi'/dr^2 + (3/r) dphi'/dr = dV'/dphi' with dphi'/dr(0) = 0 and the tail
  condition of the decaying K_1(m_i r)/r at r_max, r_W plus a fixed number of decay lengths
  1/m_min (at least the seed's end); it returns phi_b on its grid R and S_E.
- **`computation/mode_census_vgeneral_phi_v2.py`** (library of stages 0, 0a, 0b, 7 and the
  add-on): with the engine it forms the constant Wronskian C(n, s2) = 1 + h^-(inf) and searches
  for every s2 at which an eigenvalue of C vanishes, a mode with omega^2 = -s2; it also reads a
  stored census and decides which n = 1 crossings are dangerous.
- **`computation/bounce_gallery_vgeneral_phi_v2.py`** (part of stage 0): the pictures of every
  solved bounce in `<data-dir>/gallery/` (the primed profiles with S_E and, for two fields,
  the path on V, for three the path in field space; one picture per bounce for one field
  or more than three fields; a SPECIAL bounce is drawn dashed and marked, and the census
  eigenvalues are printed on the terminal, not drawn) and, with `--select-bounce`, the menu
  of bounce names to choose from.  Nothing it makes enters lnD_ren.
- **`computation/pipeline_helpers_vgeneral_phi_v2.py`** (library of the stages and the
  add-on): `load_background` rebuilds from a bounce file R, phi_b, m_i^2, mu = sum_i m_i,
  U_int = H'(phi_b) - diag(m_i^2) and tr U_int^3; it also names, reads and writes every stage
  file, stamped with its bounce, and holds the index sums of the Green's-function bracket.
- **`computation/greens_function_vgeneral_phi_v2.py`** (the engine; band worker of stage 1): for
  one wave n and one s2 the branches h^-, h^+ of the h-ODE h'' + P h' = K (1 + h), split exactly
  into Born orders h1 + h2 + h_geq3, then C, G_bracket and, with the FV weight r I_nu K_nu, the
  traces `Gbar_minus_fv` (of G - G_FV), `Gbar_1`, `Gbar_2` and `delta_geq3`.
- **`computation/fv_analytic_toolbox_vgeneral_phi_v2.py`** (library of the engine and the
  add-on): I_nu and K_nu of the free Bessel solutions I_nu(kappa_i r)/r, K_nu(kappa_i r)/r
  (kappa_i = sqrt(m_i^2 + s2), nu = n + 1): their logarithms, log-derivatives and product
  I_nu K_nu, from scipy's scaled values or, beyond their range, a large-order formula.
- **`computation/eigenvalue_neg_vgeneral_phi_v2.py`** (level 2, stage 0a): omega2_00, the
  negative-mode eigenvalue of wave n = 0, read from the stored census; it stops unless there is
  exactly one n = 0 mode and writes `eig_n0_<tag>.npz` (with a det C scan as a record) for
  stage 2a and `greensfct_dangerous_modes_v2.py`.
- **`computation/eigenvalue_zero_vgeneral_phi_v2.py`** (level 2, stage 0b): the same for wave
  n = 1: the translation zero-mode crossing s2_c (nearest s2 = 0) and omega2_10 = -s2_c; it stops
  unless there is exactly one dangerous n = 1 crossing, with s2_c <= the zero-mode level, and
  writes `eig_n1_<tag>.npz` for stage 2b and `greensfct_dangerous_modes_v2.py`.
- **`computation/band_adaptive_vgeneral_phi_v2.py`** (level 2, stage 1): the band, delta_geq3 of
  the waves n = 2..n_max on an s2 ladder up to s2_max, from parallel engine workers in rounds
  until the watchers accept s2_max, n_max and fit_lo; written are the slices
  `residual_band_<tag>_s2<v>.npz` (by the workers), `band_manifest_<tag>.json` and
  `band_cutoffs_<tag>.npz`.
- **`computation/watchers_guards_vgeneral_phi_v2.py`** (library of stages 1, 3, 4 and 7): the
  moments A3 = (1/16) Int dr r^5 tr U_int^3 and baacke_c = -(1/48) Int dr r^3 tr U_int^3, the
  watchers that fix s2_max, fit_lo and n_max in stage 1 from the large-s2 and large-n laws, the
  guards that recheck them in stages 3 and 4, and the SECTOR-CUTOFF limit.
- **`computation/addback_tadpole_vgeneral_phi_v2.py`** (level 2, stage 5): the tadpole add-back
  A1_fin = sum_i M_i tau_fin_fv(m_i^2), with the tadpole moments M_i = 2 pi^2 Int dr r^3
  [U_int]_ii (the bounce tail beyond r_max in closed form) and tau_fin_fv(W^2) =
  W^2/(16 pi^2) (ln(W^2/mu^2) - 1); it writes `addback_tadpole_<tag>.npz`.
- **`computation/addback_fish_vgeneral_phi_v2.py`** (level 2, stage 6): A2_fin, the MS-bar finite
  part of the fish, 1/(8 pi^2) Int dq q^3 sum_ij Utilde_ij(q)^2 Pi_fin_fv_ij(q), from the Hankel
  transform Utilde of U_int and the finite two-mass bubble, converged in momentum cutoff and
  spacing; it writes `addback_fish_<tag>.npz` (lnD_ren receives -A2_fin/2).
- **`add_on_greensfct/greensfct_toolbox_v2.py`** (library of every add-on code): finds the bounce
  and its tag, builds the engine and gives for one (n, s2) the branches, C, G_bracket with its
  Dyson orders, the FV weight K_nu(kappa_i r) I_nu(kappa_j r)/r^2, one coincident wave, the
  angular weights U_n(cos Theta) and n_max; it stops at a crossing.
- **`add_on_greensfct/greensfct_off_coincident_radial_v2.py`** (level 2): the radial Green's
  function [G_n(r, r', s2)]_ij of one wave n, the free Bessel solutions dressed by 1 + h^- and
  1 + h^+ and joined through C^-1 (`build_G_n`); for n = 0, 1 it contains the pole terms of the
  dangerous modes.
- **`add_on_greensfct/greensfct_fv_coincident_renorm_v2.py`** (level 2): the renormalised
  false-vacuum matrix G0_fv_ren_addback = delta_ij tau_fin_fv(kappa_i^2), kappa_i^2 = m_i^2 + s2, in
  closed form: the Dyson-order-0 add-back of G_ren; it cancels in the ratio of lnD_ren.
- **`computation/pipeline_quadrature_vgeneral_phi_v2.py`** (library: the integration rules of
  stages 1 to 4): the s2 quadrature (Simpson in ln s2, the strip below the first node from a
  quadratic), the high-s2 tail of a band wave, the three-term large-n fit with its Hurwitz-zeta
  sum, and the sector s2 grids with the pole-subtracted sector integral.
- **`computation/gbar_sector_vgeneral_phi_v2.py`** (level 3, stages 2a/2b, `--sector-n 0` or
  `1`): the sector integral `Delta_n` = -Int_0^{s2_max} [delta_geq3 -
  (n+1)^2/(omega2 + s2)] ds2 of wave n, delta_geq3 from the engine on the sector's own s2 grid,
  omega2 from `eig_n<n>`, s2_max from `band_cutoffs`; it writes `sector_n<n>_<tag>.npz`.
- **`computation/band_integrals_vgeneral_phi_v2.py`** (level 3, stage 3): for every band wave
  I_n(s2_max) = -Int_0^{s2_max} delta_geq3 ds2, the high-s2 tail T_n(s2_max) (an estimate: the
  large-s2 shape of Born order 3 scaled to the computed delta_geq3(n, s2_max), or its analytic
  amplitude where that ratio cancels) and I_n(inf) = I_n(s2_max) + T_n(s2_max); it writes
  `band_integrals_<tag>.npz`.
- **`add_on_greensfct/greensfct_off_coincident_v2.py`** (level 3): [G(x, x', s2)]_ij at one
  angle Theta between x and x', 1/(2 pi^2) sum_{n=0..n_max} (n+1) U_n(cos Theta) G_n(r, r'),
  set to NaN on r = r', where the wave sum does not converge.
- **`add_on_greensfct/greensfct_off_coincident_vs_theta_v2.py`** (level 3): the same matrix
  against Theta in [0, pi] at one pair of radii (default r_W and r_W/2), with every wave and the
  partial sums at Theta = 0.
- **`add_on_greensfct/greensfct_coincident_unrenorm_v2.py`** (level 3): the unrenormalised
  [G(x, x, s2)]_ij, the sum of the waves (n+1)^2/(2 pi^2) K_nu(kappa_i r) I_nu(kappa_j r)/r^2
  [G_bracket]_ij to n_max and to n_max/2; it diverges, so n_max is a regulator here.
- **`add_on_greensfct/greensfct_fv_off_coincident_v2.py`** (level 3): [G_FV(x, x', s2)]_ij as
  the same angular wave sum over the free waves, compared with the free propagator
  kappa_i K_1(kappa_i d)/(4 pi^2 d).
- **`add_on_greensfct/greensfct_fv_coincident_v2.py`** (level 3): the free coincident waves
  delta_ij I_nu K_nu(kappa_i r)/r^2 and their weighted sums to n_max and n_max/2: the
  Dyson-order-0 waves that G_ren leaves out.
- **`add_on_greensfct/greensfct_dangerous_modes_v2.py`** (level 3): the pole terms of the
  negative mode (n = 0) and of the translation zero modes (n = 1): with omega2_00 and omega2_10
  it builds the eigenfunction u_n(r) from the null vector of C at s2 = -omega^2, normalised to
  sum_i Int dr r^3 u_i^2 = 1, and forms u_i(r) u_j(r')/(omega^2 + s2).
- **`add_on_greensfct/greensfct_coincident_renorm_v2.py`** (level 3, end product): the
  renormalised coincident Green's function (dim reg, MS-bar) [G_ren(x, x, s2)]_ij =
  `G0_fv_ren_addback` (delta_ij tau_fin_fv(kappa_i^2)) + `G1_ren_addback` (minus the inverse
  Hankel transform of Utilde_ij Pi_fin_fv_ij at the masses kappa_i) + `G_remainder` (the waves
  n = 0..n_max of Dyson orders >= 2).
- **`computation/tail_high_n_zeta_vgeneral_phi_v2.py`** (level 4, stage 4): the high-n tail, one
  least-squares fit I_n(inf) ~ a/nu^3 + b/nu^5 + c/nu^7 on [fit_lo, n_max] summed over n > n_max
  with Hurwitz zeta functions, with the guards N-ASYMPTOTE (a against A3) and N-TAIL-BOUND; it
  writes `tail_high_n_<tag>.npz`.
- **`computation/assemble_lnD_vgeneral_phi_v2.py`** (level 5, stage 7, end product): lnD_ren as
  the sum of the seven terms (T1)-(T7) (the fictitious-mode terms of (T1) in closed form, (T2)
  and (T4) from the tadpole moments), after checking that the six stage files share the bounce,
  s2_max and n_max, and the bounce sensitivity indicator eps_bounce; it writes
  `lnD_ren_<tag>.npz`.
