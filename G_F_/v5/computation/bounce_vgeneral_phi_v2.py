#!/usr/bin/env python3
"""bounce_vgeneral_phi_v2.py -- Level 1, driver stage 0: the O(4) bounce
phi_b(r) of every downward vacuum pair of the potential, the mode census of
its waves n = 0 and n = 1, and the ONE selected bounce every later level
reads.

READS   the model from the command line (--fields, --couplings -> params,
        potential_vgeneral_phi_v2.couplings_to_params) and the vacuum
        catalogue from find_all_minima (level 0, in memory); on a resume, its
        own earlier outputs below, checked against that model and reused (a
        file of another potential stops the run: one data folder holds the
        bounces of ONE potential).
WRITES  bounce_gen/bounce_data_F<i>_T<j>.npz, one per pair (bounce_gallery
        draws each one into gallery/: bounce_<name>_profiles.png, the
        primed profiles phi_b with S_E in the legend, and for two
        or three fields bounce_<name>_potential.png, the path
        phi_b_orig with the potential V; a SPECIAL bounce is drawn
        dashed); and the tagged copy of the selected
        pair, bounce_data_<tag>.npz (pipeline_helpers.stage_paths['bounce'])
        -- the input of every later code (schema at the end of this
        docstring): load_background rebuilds the background R, phi_b, m2,
        U_int from it for every stage, eigenvalue_neg / eigenvalue_zero read
        its census_* keys, and assemble_lnD reads phi_b_ref with
        bounce_ref_tol for eps_bounce and reports census_accept, census_s2_c,
        census_zero_mode_trust and bounce_method.
HOW     THE SPINE (generate_all_bounces, below):
    take all minima of the potential -> all downward (false, true) pairs ->
    FOR EACH PAIR:
      1. PRIMED FRAME  rotate + shift so THIS pair's false vacuum is at the
                       origin (PrimedPotential).  Steps 2-4 ALL run there, in
                       that one frame; nothing converts back and forth.
      2. SEED          one CosmoTransitions fullTunneling call at LIBRARY
                       DEFAULTS on the primed potential, except the profile
                       tolerance SEED_PHITOL, which places a thin wall close
                       enough for step 3 to converge (alpha=3 is the O(4)
                       friction, not a tuning)
      3. SOLVE         scipy.integrate.solve_bvp of the O(4) EOM, seeded
                       DIRECTLY by step 2 (bounce_polish_vgeneral_phi_v2 --
                       carries ALL accuracy); guards run on the FINAL profile
      4. CENSUS        the eigenvalues omega^2 = -s2 of the waves n = 0 and
                       n = 1: the s2 with det C(n, s2) = 0, for the constant
                       Wronskian C = 1 + h^-(inf), from the value and slope
                       of the regular branch at r_max
                       (GreensFunction.h_minus_inf); mode_census finds
                       every s2 at which an eigenvalue of C is zero.
      4b. REFERENCE    the same boundary-value solve at a tighter tolerance,
                       on the kept domain and grid (phi_b_ref):
                       the reference of the bounce sensitivity indicator
                       eps_bounce that assemble_lnD_vgeneral_phi_v2 computes
      5. SAVE          one npz per bounce: the primed profile AND the ONE
                       return to the unprimed frame (phi_b_orig)
    -> back to the unprimed system, next pair.
main() then plots the whole gallery (bounce_gallery_vgeneral_phi_v2) and
selects ONE bounce (--pair or --select-bounce); the SELECTION GATE
(assert_census_accepted) is the only census abort: only a bounce with ONE
n=0 negative mode and ONE n=1 translation zero mode (positive n=1 bound
states are allowed) may enter the lnD computation; anything else (a SPECIAL
bounce) prints its eigenvalues and aborts with the open-problem message.

Files of this stage (Level 1):
    bounce_vgeneral_phi_v2.py    THIS file -- the pair-loop spine, guards,
                                 census, naming, CLI (main())
    bounce_polish_vgeneral_phi_v2.py   the solve core (solve_bvp, analytic
                                 Jacobian, singular origin, tail condition,
                                 radius rules)
    mode_census_vgeneral_phi_v2.py     the census library (shared with the
                                 eigenvalue stages 0a / 0b)
    bounce_gallery_vgeneral_phi_v2.py  all plotting + the interactive menu

Governing equation, solved in the primed frame phi' = L^T (phi - phi_F)
(L orthogonal; the prime marks the primed frame ONLY, radial derivatives are
written d/dr):
    d^2 phi'_i/dr^2 + (3/r) d phi'_i/dr = dV'/dphi'_i ,
    d phi'_i/dr (0) = 0 ,
    d phi'_i/dr (r_max) = -m_i K_2(m_i r_max)/K_1(m_i r_max) phi'_i(r_max)
    (the false vacuum is the origin; at r_max the profile is its decaying
    tail A_i K_1(m_i r)/r, whose shape the outer condition imposes;
    bounce_polish).

Naming (used by every menu, picture and file name): catalogue =
find_all_minima, sorted V ascending, i.e. by the height of the hilltops of the
inverted potential -V.  Every minimum has ONE label, the same in every bounce
name and every picture: TV0 = the deepest minimum (the highest hilltop of -V),
M1 = the second deepest, M2 = the third, ..., FV0 = the highest minimum (the
lowest hilltop of -V); minima of equal V are ordered by their coordinates
(potential_vgeneral_phi_v2._canonical_vacuum_order).  Bounce NAME =
<falseLabel>_<trueLabel> (FV0_TV0, M2_TV0, FV0_M1, M2_M1, ...); file name
bounce_data_F<i>_T<j>.npz (i, j the catalogue indices of the two labels),
vacuum COORDINATES stored inside as the stable identity.

npz schema of both files (ONE write per bounce, never rewritten):
  params, fields, false_vac, true_vac, R, phi_b, phi_b_orig,
  S_E, false_index, true_index, tag (run tag), pair_name, mass_basis_L,
  seed_range (the vacuum-search box LO, HI; empty = derived from the
  couplings),
  census_n0_s2, census_n1_s2 (the located crossings, ascending; each one is an
  eigenvalue omega^2 = -s2), census_n0_modes, census_n1_modes (how many modes
  sit at each of them), census_n0_resid, census_n1_resid (|det C| at the
  production crossing of each sector, reported by the eigenvalue stages),
  census_n0_unresolved, census_n1_unresolved (near-zero regions the locator
  could not settle), census_symmetry (the field symmetry profile_symmetry
  detected, or ''; quoted beside a shared crossing), census_s2_c (the
  translation crossing: the n=1 crossing nearest s2 = 0),
  census_zero_mode_tol, census_zero_mode_trust (the two derived zero-mode
  levels), census_accept, bounce_method,
  phi_b_ref, bounce_ref_tol (the reference profile of the
  bounce sensitivity indicator on the grid R and its solve_bvp tolerance, NaN when
  no reference converged; assemble_lnD_vgeneral_phi_v2 computes eps_bounce
  from it), code_version, potential_id.
"""
import argparse
import os
import re
import shutil
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from potential_vgeneral_phi_v2 import (
    PrimedPotential,
    V_numeric,
    couplings_to_params,
    find_all_minima,
    is_decoupled,
    params_no_const,
    validate_fields,
)
from pipeline_helpers_vgeneral_phi_v2 import (
    FALSE_VAC_OFFSET_MAX,
    ROUNDOFF_REL,
    VGENERAL_PHI_PIPELINE_VERSION,
    add_standard_cli,
    atomic_savez,
    channel_masses_and_mu,
    fv_newton_step,
    insertion,
    potential_id,
    stage_paths,
)
from bounce_polish_vgeneral_phi_v2 import (
    POLISH_DECAY_LENGTHS_PAST_WALL,
    POLISH_MAX_NODES,
    POLISH_TOL_REF,
    bvp_polish,
)
# Step 4 of the spine: the engine (GreensFunction) and the census library --
# the scan window and the zero-mode levels, the two sector censuses, the
# dangerous-crossing rule, the stored-census reader, the inventory printer,
# the evidence lines and the gate messages.
from greens_function_vgeneral_phi_v2 import GreensFunction
from mode_census_vgeneral_phi_v2 import (
    LEG_POINTER_MSG,
    OPEN_PROBLEM_MSG,
    census_violations,
    dangerous_n1_mask,
    negative_mode_census,
    print_census_inventory,
    negative_mode_window,
    stored_census,
    zero_mode_census,
    zero_mode_levels,
)


# --------------------------------------------------------------------------- #
#  Named tolerances (dimensionless or scaled by the problem's own scales)      #
# --------------------------------------------------------------------------- #
# BOUNCE-EOM gross-failure gate on the FINAL profile: RMS EOM residual
# relative to |dV|_rms (a converged solve sits far below 1%).
EOM_RESIDUAL_MAX = 0.01
# The seed's relative profile tolerance (CosmoTransitions findProfile phitol,
# library default 1e-4; overridable): the solve of step 3 converges only
# from a seed whose wall lies within a small fraction of a wall width of the
# true wall, and at the library default a thin wall is misplaced by more.
SEED_PHITOL = float(os.environ.get('VGENERAL_PHI_SEED_PHITOL', '1e-8'))
# FALSE_VAC_OFFSET_MAX (the false-vacuum displacement limit, a fraction of the
# vacuum separation) lives in pipeline_helpers_vgeneral_phi_v2: load_background
# gates every stored bounce by the same rule.


# --------------------------------------------------------------------------- #
#  THE PAIR LOOP (the spine of bounce_vgeneral_phi_v2)                         #
# --------------------------------------------------------------------------- #
def generate_all_bounces(params, fields, minima, run_tag, out_dir, seed_range):
    """Solve EVERY downward vacuum pair of the catalogue `minima` (main()'s
    one Newton sweep, minima_catalogue) and write one
    bounce_data_F<i>_T<j>.npz per pair into out_dir
    (_reuse_stored resumes an existing file); `seed_range` is the vacuum-search
    box of the catalogue, stored with every bounce.  A failing pair is
    recorded and reported by name; abort only when NO pair produced a
    bounce."""
    print("Fields:", tuple(fields))
    print("Using parameters:", params)

    # ---- bounce_gen/ holds the pairs of ONE potential (its files are named
    #      by catalogue indices only, so two potentials would share names):
    #      a file of another potential stops the run before anything is
    #      solved (_assert_same_potential) --------------------------------
    for gen_name in sorted(f for f in os.listdir(out_dir)
                           if f.startswith('bounce_data_') and f.endswith('.npz')):
        _assert_same_potential(os.path.join(out_dir, gen_name), params,
                               tuple(fields), 'a pair file of this data folder')

    # ---- the catalogue -> all downward pairs --------------------------------
    print_vacuum_catalogue(minima)
    if len(minima) < 2:
        raise SystemExit(
            f'[ABORT BOUNCE-GEN] only {len(minima)} minimum(a) found in the '
            f'vacuum-search box -- a bounce needs at least two.  If more '
            f'minima exist, widen the box with --seed-range LO,HI.')
    pairs = enumerate_bounce_pairs(minima)
    if not pairs:
        raise SystemExit(
            f'[ABORT BOUNCE-GEN] no downward (false,true) pair: the '
            f'{len(minima)} minima are V-degenerate (no V gap above 1e-12 of '
            f'the potential energy scale, the rule of enumerate_bounce_pairs) '
            f'-- degenerate vacua get no bounce (catalogue above).')
    n_minima = len(minima)
    print("\nDownward pairs to solve (EVERY pair, not just the least-action "
          "one):")
    for (iF, iT) in pairs:
        print(f"  {bounce_display_name(iF, iT, n_minima):<12s} "
              f"(file F{iF}_T{iT}): V_F={minima[iF]['V']: .6f} -> "
              f"V_T={minima[iT]['V']: .6f}")

    solved, skipped, failures = 0, 0, []
    for (iF, iT) in pairs:
        false_vac = minima[iF]['point']
        true_vac = minima[iT]['point']
        name = bounce_display_name(iF, iT, n_minima)
        pair_name = f"{name} (file F{iF}_T{iT})"
        pair_npz = os.path.join(out_dir, f"bounce_data_F{iF}_T{iT}.npz")

        if os.path.exists(pair_npz):
            # the same potential (checked above) must also have the same
            # vacua (a different seed box renumbers the pairs)
            _assert_requested_vacua(pair_npz, false_vac, true_vac, pair_name,
                                    'existing pair being resumed')
            if _reuse_stored(pair_npz, name):
                skipped += 1
                continue

        print("\n========================================================")
        print("Bounce pair", pair_name)
        print("  fields           =", tuple(fields))
        print("  false_vac (orig) =", np.asarray(false_vac, float))
        print("  true_vac  (orig) =", np.asarray(true_vac, float))
        print("========================================================")
        try:
            # 1. PRIMED FRAME
            pot_prime = PrimedPotential(params, false_vac, fields)
            assert_false_vacuum(pot_prime, false_vac, true_vac, pair_name)

            # 2. SEED
            R_seed, phi_seed = seed_profile(pot_prime, true_vac)

            # 3. SOLVE (guards on the FINAL profile)
            R, phi_b, S_E = polish_profile(R_seed, phi_seed, pot_prime,
                                           pair_name)

            # 4. CENSUS, from the background of THIS bounce formed as
            # load_background forms it for every later stage: the channel
            # masses m2 and the insertion U_int = H'(phi_b) - diag(m2) feed the
            # engine (GreensFunction) inside bounce_census.  The profile
            # returns to the unprimed frame ONCE, here: the symmetry test in
            # bounce_census and the stored phi_b_orig read it.
            phi_b_orig = pot_prime.to_original(phi_b)
            m2, _ = channel_masses_and_mu(pot_prime)
            U_int = insertion(pot_prime, phi_b)
            census = bounce_census(R, phi_b, U_int, m2, name, params, fields,
                                   phi_b_orig)

            # 4b. REFERENCE: the profile the bounce sensitivity indicator is taken
            # against (assemble_lnD_vgeneral_phi_v2 computes eps_bounce).
            phi_b_ref, bounce_ref_tol = reference_profile(
                R, phi_b, pot_prime, pair_name, name)
        except Exception as e:
            print(f"\n[BOUNCE-GEN FAILED] {name} ({pair_name}): "
                  f"{type(e).__name__}: {e}")
            failures.append((name, pair_name, f'{type(e).__name__}: {e}'))
            continue

        # 5. SAVE -- then back to the unprimed system, next pair.
        atomic_savez(pair_npz, **bounce_npz_payload(
            R=R, phi_b=phi_b, phi_b_orig=phi_b_orig, params=params,
            fields=fields, false_vac=false_vac, true_vac=true_vac, S_E=S_E,
            iF=iF, iT=iT, tag=run_tag, pair_name=pair_name,
            mass_basis_L=pot_prime.L, census=census, phi_b_ref=phi_b_ref,
            bounce_ref_tol=bounce_ref_tol, m2=m2, seed_range=seed_range))
        print(f"\nSaved bounce {name} -> {pair_npz}  (S_E = {S_E:.6f}, "
              f"census {'ACCEPT' if census['census_accept'] else 'SPECIAL'})")
        solved += 1

    if solved == 0 and skipped == 0:
        detail = '\n'.join(f'    {name}  ({t})\n        {why}'
                           for name, t, why in failures)
        raise SystemExit(
            f"[ABORT BOUNCE-GEN] no bounce solved: all {len(failures)} "
            f"attempted pair(s) failed, so no bounce_data_F<i>_T<j>.npz was "
            f"written and nothing downstream can run.\n"
            f"  fields = {tuple(fields)}\n"
            f"  params = {np.array2string(params, precision=6)}\n"
            f"  WHY, per pair, in the words of the solver or guard that "
            f"rejected it:\n{detail}")
    print(f"\nBounce gallery: {solved} solved, {skipped} reused"
          + (f", {len(failures)} failed (reported above)" if failures else "")
          + ".")


def _reuse_stored(pair_npz, name):
    """Resume of the existing bounce npz `pair_npz` of THIS potential and
    pair (the caller checked both): True = reuse (census re-printed), False
    = re-solve -- a bounce_gen file that lacks a census key
    (mode_census.stored_census returns None)."""
    with np.load(pair_npz, allow_pickle=True) as bounce_npz_data:
        census = stored_census(bounce_npz_data)
        if census is None:
            print(f"\n[RE-SOLVE] {name}: {pair_npz} matches the potential "
                  f"but lacks a census key -- re-solving it.")
            return False
        phi_b = np.asarray(bounce_npz_data['phi_b'], float)
        solved_by = str(bounce_npz_data['code_version'])
    print(f"\n[SKIP] {name}: {pair_npz} already solved for this "
          f"potential by build {solved_by} (this code: "
          f"{VGENERAL_PHI_PIPELINE_VERSION}; census stored) -- reused.")
    # the census eigenvalues are REPORTED on every run's terminal, including
    # a resumed one -- a resume must never hide the mode inventory (n <= 1).
    print_census_inventory(census, f'{name} (stored census, reused)',
                           phi_b=phi_b)
    print(f"[CENSUS {name}] stored verdict: "
          f"{'ACCEPT' if census['census_accept'] else 'SPECIAL'}")
    return True


# --------------------------------------------------------------------------- #
#  Step 2: SEED (CosmoTransitions at library defaults except phitol)          #
# --------------------------------------------------------------------------- #
def seed_profile(pot_prime, true_vac_orig):
    """Spine step 2 (module docstring): ONE CosmoTransitions fullTunneling
    call at library defaults on the primed potential (alpha = 3 = d-1 is the
    O(4) friction), seeding THIS pair in ITS primed frame; one default is
    tightened, the profile tolerance phitol = SEED_PHITOL (its comment
    among the named tolerances says why).  HOW: path deformation -- along a
    path Phi(x) in field space (x = the path distance from the true vacuum)
    the one-field profile x(r) is shot by the overshoot/undershoot method,
    the path is then deformed by the normal force Nbar, and the two steps
    alternate until the path stops moving; the profile Phi(x(r)) along the
    final path is the seed.  The same call serves any number of fields,
    coupled or not.  The seed is the initial guess of step 3: it selects
    the solution step 3 converges to, and step 3 alone sets its accuracy.
    Returns (R_seed, phi_seed), primed frame; R_seed starts at exactly
    r = 0, as bvp_polish requires (solve_bvp puts its singular point there)."""
    # deferred: only the seed step needs CosmoTransitions
    from cosmoTransitions import pathDeformation as pd
    true_prime = pot_prime.to_prime(np.asarray(true_vac_orig, float))
    # the initial path (2, N): the straight line from the true vacuum to the
    # false vacuum, the primed origin (GUARD FALSE-VACUUM checked it)
    path_guess = np.vstack([true_prime, np.zeros(pot_prime.N)])
    t0 = time.perf_counter()
    try:
        ct_result = pd.fullTunneling(path_guess, pot_prime.V, pot_prime.dV,
                                     tunneling_init_params={"alpha": 3},   # O(4)
                                     tunneling_findProfile_params={"phitol": SEED_PHITOL})
    except ValueError as e:
        if 'must have different signs' not in str(e):
            raise
        raise RuntimeError(
            'CosmoTransitions found no undershoot/overshoot bracket for this '
            'pair.  Usual cause: no direct barrier between the two vacua (the '
            'deformation path slides into another minimum) or a wall too thin '
            'for the default bracket; the failure is per pair -- other pairs '
            'of the potential still solve.') from e
    seed_seconds = time.perf_counter() - t0
    # ct_result.Phi = Phi(x(r)) on the radii profile1D.R (profile1D.Phi is x(r))
    R_seed = np.asarray(ct_result.profile1D.R, float)
    phi_seed = np.asarray(ct_result.Phi, float)
    # fRatio = max|Nbar| / max|grad V| on the final path, with the normal force
    # Nbar = (d^2 Phi/dx^2)(dx/dr)^2 - grad_perp V (zero on the bounce path)
    print(f"  [SEED] fullTunneling (CT defaults except phitol = {SEED_PHITOL:g}): S_seed = "
          f"{float(ct_result.action):.6f}, fRatio = {float(ct_result.fRatio):.2e}, "
          f"{len(R_seed)} pts, R_seed = {R_seed[-1]:.4f}, {seed_seconds:.2f} s  "
          f"(the boundary-value solve below carries the accuracy)")
    return R_seed, phi_seed


# --------------------------------------------------------------------------- #
#  Step 3: SOLVE (the production solve) + gates on the FINAL profile           #
# --------------------------------------------------------------------------- #
def polish_profile(R_seed, phi_seed, pot_prime, pair_name):
    """Spine step 3 (module docstring): bvp_polish on the primed seed, in the
    primed frame, then the BOUNCE-ENDPOINT and BOUNCE-EOM gates.
    Returns (R, phi_b, S_E): the kept grid, the primed profile, the action."""
    polish_result = bvp_polish(R_seed, phi_seed, pot_prime)
    print(f"  [SOLVE] solve_bvp: success={polish_result['success']} ({polish_result['message']}); "
          f"{polish_result['niter']} mesh passes, {polish_result['nodes']} nodes, worst relative "
          f"collocation residual {polish_result['rms_max']:.2e}, "
          f"{polish_result['solve_seconds']:.2f} s")
    print(f"  [SOLVE] solved and stored to R = {polish_result['r_max']:.4f} (m_min*R = "
          f"{polish_result['m_min'] * polish_result['r_max']:.1f}, wall at m_min*r_W = "
          f"{polish_result['m_min'] * polish_result['r_W']:.2f})")
    if not polish_result['success']:
        # solve_bvp's own message names the failure; the mesh cap (status 1)
        # is the one failure with a remedy of its own
        remedy = ''
        if polish_result['status'] == 1:
            remedy = (f"  The collocation mesh hit POLISH_MAX_NODES = "
                      f"{POLISH_MAX_NODES} before reaching tol = {polish_result['tol']:g} "
                      f"(m_min*r_W = {polish_result['m_min'] * polish_result['r_W']:.1f}).  "
                      f"For a very thin wall (m_min*r_W above about 50) this means "
                      f"the seed's wall lies too far from the true one for the "
                      f"solve to converge: a larger node cap does not help, a "
                      f"smaller seed tolerance does (VGENERAL_PHI_SEED_PHITOL, "
                      f"{SEED_PHITOL:g} in this run).  "
                      f"Otherwise either the model is written in large units "
                      f"(e.g. GeV-scale fields and couplings): enter it in "
                      f"units of a mass Lambda that makes its masses of order "
                      f"one (lnD_ren in the original units is the computed "
                      f"value minus 8 ln Lambda); or tol is so tight that the "
                      f"collocation residual meets round-off: loosen "
                      f"VGENERAL_PHI_POLISH_TOL.")
        raise RuntimeError(
            f"[ABORT BOUNCE-SOLVE] bounce {pair_name}: the boundary-value "
            f"solve did not converge (solve_bvp status {polish_result['status']}: "
            f"{polish_result['message']})." + remedy)
    R = np.asarray(polish_result['R'], float)
    phi_b = np.asarray(polish_result['phi_b'], float)
    assert_bounce_endpoints(phi_b, pot_prime, pair_name)
    eom_resid = eom_residual_rms(R, phi_b, pot_prime)
    print(f"  [BOUNCE-EOM] final-profile EOM residual (RMS, normalized) = "
          f"{100*eom_resid:.4f}%  (gross-failure limit {100*EOM_RESIDUAL_MAX:.1f}%)")
    if not np.isfinite(eom_resid) or eom_resid > EOM_RESIDUAL_MAX:
        raise RuntimeError(
            f"[ABORT BOUNCE-EOM] bounce {pair_name}: the FINAL profile's EOM "
            f"residual {100*eom_resid:.2f}% is non-finite or exceeds "
            f"{100*EOM_RESIDUAL_MAX:.1f}% -- the boundary-value solve did not "
            f"actually converge.")
    print(f"  [ACTION] S_E = {polish_result['S_E']:.6f}")
    return R, phi_b, polish_result['S_E']


# --------------------------------------------------------------------------- #
#  Guards (none enters lnD): step 1 false vacuum, step 3 FINAL profile         #
# --------------------------------------------------------------------------- #
def eom_residual_rms(R, phi_b, pot_prime):
    """RMS of the O(4) EOM residual d^2phi/dr^2 + (3/r) dphi/dr - dV(phi),
    all N components, normalized by |dV|_rms.  Gross-failure gate only --
    not an accuracy meter (the spline re-differentiation dominates the
    residual, so it overstates the true profile error by orders of
    magnitude)."""
    # ONE cubic spline through all N components (axis 0 = r): phi_spline(r, k)
    # = d^k phi_b/dr^k; the r = R[0] = 0 node is skipped (3/r undefined there)
    phi_spline = CubicSpline(R, phi_b)
    r = R[1:]
    dV = pot_prime.dV(phi_b[1:])
    eom_residual = phi_spline(r, 2) + 3.0 / r[:, None] * phi_spline(r, 1) - dV
    return float(np.sqrt(np.mean(np.sum(eom_residual ** 2, axis=1))
                         / np.mean(np.sum(dV ** 2, axis=1))))


def assert_bounce_endpoints(phi_b, pot_prime, pair_name):
    """GUARD BOUNCE-ENDPOINT (aborts): the profile is a NONTRIVIAL bounce
    that has reached its tail.  Its core phi_b(0) is a RELEASE POINT, not a
    boundary condition (a thick-wall bounce releases far from the true
    vacuum), and it must lie BELOW the false-vacuum energy,
    V(phi_b(0)) < V(phi_F) = 0: the Euclidean energy
    E = |dphi/dr|^2/2 - V only decreases, dE/dr = -3 |dphi/dr|^2 / r, from
    E(0) = -V(phi_b(0)) to E(inf) = -V(phi_F) = 0.  This excludes the
    trivial solution phi == phi_F of the same BVP.  Its outer end
    phi_b(r_max) is left to the solve (the outer condition fixes only the
    slope-to-value ratio of the tail, bounce_polish._bvp_system), so it is
    checked here: a converged bounce has decayed to about e^{-K} of its
    amplitude after the K decay lengths of the domain rule, and the guard
    demands e^{-K/2} at most."""
    V_core = float(pot_prime.V(phi_b[0]))      # the primed V is 0 at phi_F
    if not np.isfinite(V_core) or V_core >= 0.0:
        raise RuntimeError(
            f"[ABORT BOUNCE-ENDPOINT] bounce {pair_name}: the core phi_b(0) = "
            f"{np.round(pot_prime.to_original(phi_b[0]), 4)} (original "
            f"coordinates) has V(core) - V(FV) = {V_core:+.6e}, not below the "
            f"false-vacuum energy -- not the release point of a bounce (the "
            f"trivial phi == phi_F solves the same BVP).")
    amplitude = float(np.max(np.abs(phi_b)))
    end = float(np.max(np.abs(phi_b[-1])))
    if not end < np.exp(-0.5 * POLISH_DECAY_LENGTHS_PAST_WALL) * amplitude:
        raise RuntimeError(
            f"[ABORT BOUNCE-ENDPOINT] bounce {pair_name}: |phi_b(r_max)| = "
            f"{end:.3e} = {end / amplitude:.2e} of the amplitude, not below "
            f"e^(-{POLISH_DECAY_LENGTHS_PAST_WALL:g}/2) -- the profile has not "
            f"decayed onto its tail at r_max.")


def assert_false_vacuum(pot_prime, false_vac_orig, true_vac_orig, pair_name):
    """GUARD FALSE-VACUUM (aborts): the frame origin really is a stationary
    point of V -- ONE rule, pipeline_helpers.fv_newton_step: one Newton step
    |H'(0)^-1 dV'(0)| against the vacuum separation (scale-free;
    load_background applies the same rule to every stored bounce)."""
    false_vac = np.asarray(false_vac_orig, float)
    true_vac = np.asarray(true_vac_orig, float)
    sep = float(np.linalg.norm(true_vac - false_vac))
    off = fv_newton_step(pot_prime)
    print(f"  [FALSE-VACUUM] one Newton step moves phi_F by {off:.2e} = "
          f"{off/sep:.1e} of the vacuum separation {sep:.6f} "
          f"(limit {FALSE_VAC_OFFSET_MAX:g})")
    if off > FALSE_VAC_OFFSET_MAX * sep:
        raise RuntimeError(
            f"[ABORT FALSE-VACUUM] bounce {pair_name}: the requested false "
            f"vacuum {np.round(false_vac, 8)} does not solve gradV = 0 "
            f"(Newton step {off:.3e} = {off/sep:.2e} of the separation, above "
            f"{FALSE_VAC_OFFSET_MAX:g}).  Re-enumerate the vacua "
            f"(find_all_minima).")


# --------------------------------------------------------------------------- #
#  Step 4: CENSUS (spectral acceptance), both sectors, per bounce              #
# --------------------------------------------------------------------------- #
def profile_symmetry(params, fields, phi_b_orig):
    """The NAME of a field-space symmetry that maps this POTENTIAL to itself
    AND leaves THIS profile untouched -- or '' when the cheap test finds none.
    Such a symmetry commutes with the fluctuation operator and labels its
    invariant subspaces. A field permutation alone need not force equal
    eigenvalues; the census measures any degeneracy independently.

    Two families, both tested numerically on random field points (no algebra):
      * a PERMUTATION of two fields (V unchanged when they are swapped, and the
        two profile components equal all along r);
      * a ROTATION of their plane, i.e. the two fields entering only through
        phi_i^2 + phi_j^2 (V unchanged under one generic angle -- for a
        polynomial V that already forces the whole SO(2) -- and both profile
        components zero all along r).
    Both tolerances are relative -- to the spread of V over the sample points
    and to the size of the profile itself -- so no absolute field or energy
    value enters."""
    # the field scale of THIS profile; never zero, because the BOUNCE-ENDPOINT
    # guard has already placed its core below the false-vacuum energy
    scale = float(np.max(np.abs(phi_b_orig)))
    # 40 random points of field space at that scale, and V on them, without
    # the constant monomial c: a large c would swallow the spread of V in
    # round-off (the test compares V at each point with V at its image)
    params = params_no_const(params, fields)
    points = np.random.default_rng(0).normal(size=(40, len(fields))) * scale
    V_points = np.asarray(V_numeric(points, params, fields), float)
    # tolerance from the SPREAD of V over the sample points, not its size
    v_tol = 1e-10 * float(np.ptp(V_points))
    def same_V(mapped):
        """True when V is unchanged on the sample points under the map."""
        return bool(np.max(np.abs(
            np.asarray(V_numeric(mapped, params, fields), float) - V_points))
            < v_tol)

    # one generic angle, fixed like the rng seed above: the answer may not
    # vary between runs
    cos_angle, sin_angle = np.cos(0.7), np.sin(0.7)
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            # the ROTATION first: it is the stronger statement, and a
            # rotation-invariant pair is always permutation-invariant too
            mapped = points.copy()
            mapped[:, i] = cos_angle * points[:, i] - sin_angle * points[:, j]
            mapped[:, j] = sin_angle * points[:, i] + cos_angle * points[:, j]
            if (same_V(mapped)
                    and np.max(np.abs(phi_b_orig[:, [i, j]]))
                    < ROUNDOFF_REL * scale):
                return (f'rotation of the ({fields[i]}, {fields[j]}) plane -- '
                        f'they enter V only as {fields[i]}^2 + {fields[j]}^2')
            mapped = points.copy()
            mapped[:, [i, j]] = points[:, [j, i]]
            if (same_V(mapped)
                    and np.max(np.abs(phi_b_orig[:, i] - phi_b_orig[:, j]))
                    < ROUNDOFF_REL * scale):
                return f'field permutation {fields[i]} <-> {fields[j]}'
    return ''


def bounce_census(R, phi_b, U_int, m2, name, params, fields, phi_b_orig):
    """Spine step 4 (module docstring, where C is defined): locate every zero
    crossing of an eigenvalue of the constant Wronskian C(n, s2) in the
    sectors n=0 and n=1 -- det C = 0 exactly at the sector eigenvalues, so ONE
    crossing is ONE eigenvalue omega^2 = -s2 (the negative modes in n=0; in
    n=1 the translation zero mode and every bound state, positive ones
    included) -- and count the modes sitting on each crossing AT the root
    (mode_census.modes_at_root).
    ACCEPT iff n=0 has exactly ONE crossing, n=1 has exactly ONE DANGEROUS
    crossing (mode_census.dangerous_n1_mask), ONE mode sits on each, and
    the locator left NO unresolved near-zero region; otherwise SPECIAL
    (computed, plotted and reported; SELECTING it for lnD aborts,
    assert_census_accepted).  zero_mode_tol and zero_mode_trust are the
    zero-mode levels of THIS bounce (mode_census.zero_mode_levels; s2 carries
    mass^2 units).
    Returns the census dict under the npz key names census_* -- what the
    bounce npz stores verbatim."""
    engine = GreensFunction(R, U_int, m2)
    s2_lo, s2_hi = negative_mode_window(engine, U_int, m2)
    # the derived zero-mode levels (mode_census.zero_mode_levels)
    zero_mode_tol, zero_mode_trust = zero_mode_levels(s2_hi, float(np.min(m2)))

    print(f"\n[CENSUS {name}] zero-crossing mode census, sectors n=0 and n=1")
    s2_roots_n0, n_modes_n0, crossing_resid_n0, n_unresolved_n0 =\
        negative_mode_census(engine, s2_lo, s2_hi)
    s2_c, s2_roots_n1, n_modes_n1, crossing_resid_n1, n_unresolved_n1 =\
        zero_mode_census(engine, s2_hi, zero_mode_tol)

    census = dict(
        # the located crossings, ascending in s2; each IS an eigenvalue
        # omega^2 = -s2
        census_n0_s2=np.array(s2_roots_n0, float),
        census_n1_s2=np.array(s2_roots_n1, float),
        # how many modes sit on each crossing (mode_census.modes_at_root)
        census_n0_modes=np.array(n_modes_n0, int),
        census_n1_modes=np.array(n_modes_n1, int),
        # |det C| at the production crossing of each sector (reported by
        # eigenvalue_neg/eigenvalue_zero)
        census_n0_resid=float(crossing_resid_n0),
        census_n1_resid=float(crossing_resid_n1),
        # near-zero regions the refinement could not settle (a place a mode
        # could hide): stored, printed, and enough on their own to make the
        # bounce SPECIAL
        census_n0_unresolved=int(n_unresolved_n0),
        census_n1_unresolved=int(n_unresolved_n1),
        census_s2_c=float(s2_c),
        # the field symmetry profile_symmetry detected ('' when none); the
        # legend and the inventory quote it next to a shared crossing, whose
        # multiplicity census_n<k>_modes measures independently.
        census_symmetry=profile_symmetry(params, fields, phi_b_orig),
        census_zero_mode_tol=float(zero_mode_tol),      # DERIVED (stored as data)
        census_zero_mode_trust=float(zero_mode_trust),  # DERIVED (stored as data)
    )
    print_census_inventory(census, name, phi_b=phi_b)

    # the dangerous n=1 crossings (mode_census.dangerous_n1_mask, the one
    # owner of the rule): the translation zero mode, a second zero mode or a
    # negative mode; a positive bound state (s2 < -zero_mode_tol) stays in the
    # resolvent
    n1_dangerous = dangerous_n1_mask(s2_roots_n1, s2_c, zero_mode_tol)
    n1_dangerous_modes = np.asarray(n_modes_n1, int)[n1_dangerous]
    n_positive_n1 = int(np.count_nonzero(~n1_dangerous))
    lone = (len(s2_roots_n0) == 1 and int(n_modes_n0[0]) == 1
            and n1_dangerous_modes.size == 1 and int(n1_dangerous_modes[0]) == 1)
    accept = bool(lone and not (n_unresolved_n0 or n_unresolved_n1))
    if accept:
        print(f"[CENSUS {name}] ACCEPT: 1 negative mode in n=0 and 1 translation "
              f"zero mode in n=1 (four angular copies)"
              + (f", plus {n_positive_n1} positive n=1 bound state(s) that "
                 f"stay inside the resolvent" if n_positive_n1 else "")
              + " -- the index-1 bounce the lnD assembly is written for.  "
              "(The sector operators differ only by the centrifugal term "
              "n(n+2)/r^2, which grows with n, so a clean n=1 sector pins "
              "every n >= 2 sector clean.)")
    else:
        print(f"[CENSUS {name}] SPECIAL bounce ({len(s2_roots_n0)} n=0 "
              f"crossing(s) sharing {list(n_modes_n0)} mode(s), "
              f"{int(np.count_nonzero(n1_dangerous))} dangerous n=1 "
              f"crossing(s) sharing {n1_dangerous_modes.tolist()} mode(s), "
              f"{n_positive_n1} positive n=1 bound state(s), "
              f"{n_unresolved_n0} + {n_unresolved_n1} unresolved near-zero "
              f"region(s); computed, plotted and reported, but NOT selectable "
              f"for the lnD computation -- selecting it aborts with the "
              f"open-problem message).")
    census['census_accept'] = bool(accept)
    return census


def assert_census_accepted(path, params, fields, n_minima):
    """THE SELECTION GATE (module docstring; the only census abort) on the
    bounce npz at `path` (the freshly selected bounce and a resumed tagged
    one): it enters lnD iff its stored census is the accepted index-1 one; a
    SPECIAL census aborts with its evidence lines (census_violations) and
    the open-problem message.  A DECOUPLED potential gets the per-leg pointer
    appended: there the rejected bounce is several one-field bounces at
    once (one negative mode, hence one n=0 crossing, per tunnelling field),
    and its one-field legs are the bounces to select instead.
    A bounce npz that lacks a census key (mode_census.stored_census: None)
    has no census to gate on: it passes untouched, and eigenvalue_neg /
    eigenvalue_zero census it again with the same locator (their
    selection-gate backstop decides)."""
    with np.load(path, allow_pickle=True) as bounce_npz_data:
        census = stored_census(bounce_npz_data)
        b = {k: bounce_npz_data[k] for k in bounce_npz_data.files}
    name = bounce_display_name(int(b['false_index']), int(b['true_index']),
                               n_minima)
    if census is None:
        print(f"[SELECT {name}] {path} lacks a census key -- resumed "
              f"untouched; eigenvalue_neg / eigenvalue_zero census it again with "
              f"the same locator.")
        return
    if census['census_accept']:
        print(f"[SELECT {name}] census ACCEPTED (index-1 bounce) -- entering "
              f"the lnD computation.")
        return
    lines = '\n'.join(census_violations(b)) or '    (see the stored census)'
    pointer = LEG_POINTER_MSG if is_decoupled(params, fields) else ''
    raise SystemExit(
        f"[ABORT SELECTION {name}] the selected bounce FAILED the census "
        f"acceptance rule\n(ONE n=0 negative mode and ONE n=1 translation zero "
        f"mode).  Dangerous crossings beyond those two:\n{lines}\n"
        + OPEN_PROBLEM_MSG + pointer)


# --------------------------------------------------------------------------- #
#  Step 4b: the REFERENCE profile of the bounce sensitivity indicator          #
# --------------------------------------------------------------------------- #
def reference_profile(R, phi_b, pot_prime, pair_name, name):
    """The REFERENCE profile the bounce sensitivity indicator is evaluated
    against: the same boundary-value solve (bvp_polish) at the tighter
    tolerances POLISH_TOL_REF, tightest first, on the kept domain and
    tabulated on the kept grid R (eval_R=R), so the two profiles differ by
    SOLUTION error only, and gated like the kept one
    (assert_bounce_endpoints).  Stored as phi_b_ref with its
    tolerance bounce_ref_tol; assemble_lnD_vgeneral_phi_v2 computes eps_bounce
    from it (A1_fin, A2_fin and A3 of both profiles by the production
    add-back and moment code).  Returns (phi_b_ref (nR, N), tol_ref);
    NaN-filled and NaN when no tolerance converged (the assembler then
    prints that the indicator could not be evaluated, with the reason; it is
    never set to zero)."""
    for tol_ref in POLISH_TOL_REF:
        try:
            polish_result_ref = bvp_polish(R, phi_b, pot_prime, tol=tol_ref, eval_R=R)
            if not polish_result_ref['success']:
                raise RuntimeError(f"solve_bvp: {polish_result_ref['message']}")
            assert_bounce_endpoints(polish_result_ref['phi_b'], pot_prime,
                                    pair_name)
        except Exception as exc:
            print(f"  [BOUNCE-REFERENCE] {name}: reference solve at "
                  f"tol={tol_ref:g} failed ({type(exc).__name__}: {exc}) -- "
                  f"trying the next tolerance")
            continue
        print(f"  [BOUNCE-REFERENCE] {name}: reference profile at "
              f"tol={tol_ref:g} on the kept grid ({len(R)} nodes) stored as "
              f"phi_b_ref")
        return np.asarray(polish_result_ref['phi_b'], float), float(tol_ref)
    print(f"  [BOUNCE-REFERENCE] {name}: no reference profile at ANY "
          f"tolerance in {POLISH_TOL_REF} -- phi_b_ref stored NaN; "
          f"assemble_lnD_vgeneral_phi_v2 reports the bounce sensitivity "
          f"indicator as not evaluated, never zero")
    return np.full(np.shape(phi_b), np.nan), float('nan')


# --------------------------------------------------------------------------- #
#  Step 5: the npz schema (ONE write per bounce, never rewritten)              #
# --------------------------------------------------------------------------- #
def bounce_npz_payload(R, phi_b, phi_b_orig, params, fields,
                       false_vac, true_vac, S_E, iF, iT, tag, pair_name,
                       mass_basis_L, census, phi_b_ref, bounce_ref_tol, m2,
                       seed_range):
    """The ONE npz schema (module docstring), written ONCE so its bytes
    (= its provenance fingerprint) never change.  `tag` is the RUN tag;
    `pair_name` the human pair string; `census` the bounce_census dict
    (already under its npz key names census_*)."""
    return dict(
        params=np.asarray(params, float),
        fields=np.array(list(fields)),
        false_vac=np.asarray(false_vac, float),
        true_vac=np.asarray(true_vac, float),
        R=np.asarray(R, float),
        phi_b=np.asarray(phi_b, float),
        phi_b_orig=np.asarray(phi_b_orig, float),
        S_E=float(S_E),
        false_index=int(iF),
        true_index=int(iT),
        tag=tag,
        pair_name=pair_name,
        mass_basis_L=np.asarray(mass_basis_L, float),
        # the vacuum-search box (LO, HI) of the catalogue; empty = the box
        # derived from the couplings (the gallery plot rebuilds the same one)
        seed_range=np.asarray(seed_range or (), float),
        # ---- census (bounce_census: the located crossings, each an
        # eigenvalue omega^2 = -s2, ascending in s2; keys census_*) ----
        **census,
        # ---- the reference profile of the bounce sensitivity indicator
        # (assemble_lnD_vgeneral_phi_v2 computes eps_bounce from it) ----
        phi_b_ref=np.asarray(phi_b_ref, float),
        bounce_ref_tol=float(bounce_ref_tol),
        # ---- provenance ----
        bounce_method=(f'CT seed (phitol {SEED_PHITOL:g}) + solve_bvp '
                       f'(seed->solve->gate)'),
        code_version=VGENERAL_PHI_PIPELINE_VERSION,
        # stamped with THIS run's model (params, fields), exactly as every
        # downstream stage stamps its own output -- params + fields always
        # come from this run's explicit model, so the bounce can never
        # disagree with anything descended from it.
        potential_id=potential_id(m2, params, fields),
    )


# --------------------------------------------------------------------------- #
#  Vacuum catalogue, pair enumeration, naming                                  #
# --------------------------------------------------------------------------- #
def minima_catalogue(params, fields, seed_range=None):
    """THE vacuum catalogue of a model: find_all_minima (V ascending) under
    the seed box `seed_range` = (LO, HI) -- None = the box build_guesses
    derives from the couplings (Cauchy root bound of grad V).  ONE Newton
    sweep per run: main() calls it once and hands `minima` down to the menu,
    the pair loop, the gallery and the selection (the gallery plot code calls
    it for its own run)."""
    return find_all_minima(params, fields, seed_range)


def enumerate_bounce_pairs(minima):
    """Every (iF, iT) of the catalogue with V(minima[iF]) GENUINELY above
    V(minima[iT]) -- the degeneracy tolerance is 1e-12 of the potential's own
    energy scale, so symmetry-degenerate vacua get no bounce."""
    vac_values = [m["V"] for m in minima]
    if not vac_values:
        return []
    V_scale = max(max(vac_values) - min(vac_values),
                  max(abs(v) for v in vac_values))
    V_gap_tol = 1e-12 * V_scale
    pairs = []
    for iF in range(len(minima)):
        for iT in range(len(minima)):
            if iF != iT and vac_values[iF] - vac_values[iT] > V_gap_tol:
                pairs.append((iF, iT))
    return pairs


def vacuum_label(i, n_minima):
    """The label of catalogue minimum i (naming spec in the module
    docstring): TV0 for the deepest (i = 0), FV0 for the highest
    (i = n_minima - 1), M<i> in between (the M-number is the catalogue
    index).  A vacuum keeps its label in every bounce it belongs to."""
    i, n = int(i), int(n_minima)
    if i == 0:
        return 'TV0'
    if i == n - 1:
        return 'FV0'
    return f'M{i}'


def bounce_display_name(iF, iT, n_minima):
    """The user-facing bounce NAME <falseLabel>_<trueLabel>: the labels of
    its false and its true vacuum."""
    return vacuum_label(iF, n_minima) + '_' + vacuum_label(iT, n_minima)


def pair_from_name(name, n_minima):
    """Parse a bounce name into catalogue indices (iF, iT): the NAME
    '<falseLabel>_<trueLabel>' of two vacuum labels TV0 / M<i> / FV0
    (`n_minima`, the catalogue size, inverts FV0 -> n-1) or the raw FILE form
    'F<i>_T<j>'.  Whether the pair is a downward bounce the caller decides
    (enumerate_bounce_pairs).  Raises ValueError (the menu re-prompts on it;
    the CLI turns it into its own abort)."""
    s = str(name).strip().upper()
    m = re.fullmatch(r'F(\d+)_T(\d+)', s)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r'(TV0|FV0|M\d+)_(TV0|FV0|M\d+)', s)
    if not m:
        raise ValueError(
            f"bounce name {name!r} is not '<falseLabel>_<trueLabel>' with both "
            f"labels in {{TV0, M<i>, FV0}} (e.g. FV0_TV0, M2_TV0, FV0_M1), nor "
            f"the raw file form F<i>_T<j>.")
    n = int(n_minima)
    indices = []
    for label in m.groups():
        if label == 'TV0':
            indices.append(0)
        elif label == 'FV0':
            indices.append(n - 1)
        elif 1 <= int(label[1:]) <= n - 2:
            indices.append(int(label[1:]))
        else:
            raise ValueError(
                f"bounce name {name!r}: the label {label} does not exist ({n} "
                f"minima: {', '.join(vacuum_label(k, n) for k in range(n))}).")
    return indices[0], indices[1]


def catalogue_indices(minima, false_vac, true_vac, pair_npz):
    """Resolve a stored bounce's endpoints to catalogue indices (iF, iT) by
    COORDINATES -- the stable identity across runs (a changed seed box
    renumbers the minima).  Tolerance: FALSE_VAC_OFFSET_MAX of the vacuum
    separation; `pair_npz` names the file in the abort text."""
    minima_points = np.array([np.asarray(m['point'], float) for m in minima])
    false_vac = np.asarray(false_vac, float)
    true_vac = np.asarray(true_vac, float)
    sep = float(np.linalg.norm(true_vac - false_vac))
    tol = FALSE_VAC_OFFSET_MAX * sep
    indices = []
    for role, pt in (('false', false_vac), ('true', true_vac)):
        i = int(np.argmin(np.linalg.norm(minima_points - pt[None, :], axis=1)))
        d = float(np.linalg.norm(minima_points[i] - pt))
        if d > tol:
            raise RuntimeError(
                f'[ABORT GALLERY-PLOT] {os.path.basename(str(pair_npz))}: its '
                f'stored {role} vacuum {np.round(pt, 6)} matches NO minimum '
                f'of the current catalogue (nearest is {np.round(minima_points[i], 6)} '
                f'at distance {d:.3e} > {tol:.3e}).  The catalogue was built '
                f'with a different --seed-range than the bounce was solved '
                f'with -- rerun with the seed range of the solve.')
        indices.append(i)
    return indices[0], indices[1]


def _fmt_point(pt):
    return '(' + ', '.join(f'{v: .6f}' for v in np.asarray(pt).ravel()) + ')'


def print_vacuum_catalogue(minima):
    """Print the catalogue with its global labels (start of every run)."""
    n = len(minima)
    print("Local minima of the potential, sorted by V ascending; V is quoted "
          "without the constant monomial c.  Labels: TV0 = the deepest minimum "
          "(the highest hilltop of -V), M1, M2, ... the next ones upwards, "
          "FV0 = the highest minimum (the lowest hilltop of -V); a bounce is "
          "named <falseLabel>_<trueLabel>:")
    for i, m in enumerate(minima):
        print(f"  {vacuum_label(i, n):<4s} (index {i}): phi = "
              f"{_fmt_point(m['point'])}   V = {m['V']: .6f}")


def list_bounces(fields, minima):
    """Print the menu of every downward bounce pair of the catalogue, named
    per the labeling spec, then return without computing anything."""
    pairs = enumerate_bounce_pairs(minima)
    print(f"Fields: {fields}")
    print_vacuum_catalogue(minima)
    if not pairs:
        print("\nNo downward (false, true) vacuum pairs "
              "(need two minima with V_false > V_true).")
        return
    print("\nAvailable bounces (rerun with --pair <name>; the raw file form "
          "F<i>_T<j> works too):")
    for (iF, iT) in pairs:
        name = bounce_display_name(iF, iT, len(minima))
        print(f"  {name:<12s} (file F{iF}_T{iT}): "
              f"false={_fmt_point(minima[iF]['point'])} "
              f"V={minima[iF]['V']: .6f}, "
              f"true={_fmt_point(minima[iT]['point'])} "
              f"V={minima[iT]['V']: .6f}")


# --------------------------------------------------------------------------- #
#  CLI parsers (terminal potential entry: --fields and --couplings)            #
# --------------------------------------------------------------------------- #
def parse_fields(spec):
    """'x,y,z' -> validated fields tuple (distinct single lowercase letters,
    'c' reserved)."""
    return validate_fields([t.strip() for t in spec.split(',') if t.strip()])


def parse_couplings(spec):
    """'x2=-2.0,y2=-2.0,...' -> {monomial_name: float}.  Names are validated
    downstream by couplings_to_params against the declared fields."""
    couplings = {}
    for term in spec.split(','):
        term = term.strip()
        if not term:
            continue
        if '=' not in term:
            raise RuntimeError(
                f"[ABORT] --couplings term {term!r} is not 'name=value' "
                f"(e.g. 'x2=-2.0,y2=-2.0,xy2=1.0,x4=1.0,y4=1.0').")
        name, val = term.split('=', 1)
        name = name.strip()
        try:
            couplings[name] = float(val.strip())
        except ValueError:
            raise RuntimeError(
                f"[ABORT] --couplings value for {name!r} is not a number: "
                f"{val.strip()!r}.")
    return couplings


# --------------------------------------------------------------------------- #
#  Orchestration (CLI, resume, selection)                                      #
# --------------------------------------------------------------------------- #
def _assert_requested_vacua(path, req_fv, req_tv, pair_name, how):
    """Fail loud unless the bounce npz at `path` was solved for the vacuum
    COORDINATES the requested pair resolves to under the CURRENT
    --seed-range.  `req_fv` is None when no pair was requested.  Both callers
    make sure first that the file is of the requested potential, so it is a
    bounce npz of the one schema (bounce_npz_payload: vacua AND catalogue
    indices) with the requested field count."""
    if req_fv is None:
        return
    with np.load(path, allow_pickle=True) as bounce_npz_data:
        have_fv = np.asarray(bounce_npz_data['false_vac'], float)
        have_tv = np.asarray(bounce_npz_data['true_vac'], float)
        have = (f"F{int(bounce_npz_data['false_index'])}"
                f"_T{int(bounce_npz_data['true_index'])}")
    sep = float(np.linalg.norm(req_tv - req_fv))
    tol = FALSE_VAC_OFFSET_MAX * sep
    d_fv = float(np.linalg.norm(have_fv - req_fv))
    d_tv = float(np.linalg.norm(have_tv - req_tv))
    if d_fv > tol or d_tv > tol:
        raise SystemExit(
            f'[ABORT BOUNCE-PAIR] {path} ({how}) was solved for DIFFERENT '
            f'vacua than the requested pair {pair_name} resolves to under the '
            f'current --seed-range.\n'
            f'  stored    false_vac = {np.round(have_fv, 8)}, '
            f'true_vac = {np.round(have_tv, 8)}   [stored pair {have}]\n'
            f'  requested false_vac = {np.round(req_fv, 8)}, '
            f'true_vac = {np.round(req_tv, 8)}   [requested {pair_name}]\n'
            f'  distances {d_fv:.3e} / {d_tv:.3e} exceed {tol:.3e} '
            f'(= {FALSE_VAC_OFFSET_MAX:g} of the vacuum separation '
            f'{sep:.6f}).\n'
            f'  A different --seed-range finds a different set of minima and '
            f'RENUMBERS them, so the SAME pair name means a different pair.  '
            f'Delete {path} (and any bounce_gen/ files of the old seed box), '
            f'or use a fresh --data-dir (bounce_gen/ is shared by every tag).')


def same_potential(bounce_npz_data, params, fields):
    """True when the bounce file `bounce_npz_data` (an open npz, or a dict of
    its arrays) was solved for the potential of this run: the same field
    names and the same couplings.  The couplings are compared EXACTLY: both
    vectors are float parses of --couplings text (couplings_to_params), so
    the same model gives bit-identical floats; a tolerance here would be an
    absolute coupling scale of its own.  A file without `fields` cannot be
    checked and counts as a different potential."""
    return ('fields' in bounce_npz_data
            and tuple(str(v) for v in np.asarray(bounce_npz_data['fields']).ravel())
            == tuple(fields)
            and np.array_equal(np.asarray(bounce_npz_data['params'], float), params))


def _assert_same_potential(path, params, fields, what):
    """Fail loud unless the bounce npz at `path` was solved for the potential
    of this run (same_potential).  A data folder holds the bounces of ONE
    potential, so this runs on the tagged bounce being resumed (main) and on
    every pair file of bounce_gen before the pair loop
    (generate_all_bounces): a file of another potential is never reused,
    removed or overwritten."""
    with np.load(path, allow_pickle=True) as bounce_npz_data:
        if 'fields' not in bounce_npz_data.files:
            raise SystemExit(
                f'[ABORT BOUNCE-SCHEMA] {path} stores no `fields` key -- a '
                f'pre-schema bounce npz whose model cannot be verified.  Run '
                f'this potential with a new --data-dir.')
        same = same_potential(bounce_npz_data, params, fields)
        have = np.asarray(bounce_npz_data['params'], float)
        have_fields = tuple(str(f) for f in np.asarray(bounce_npz_data['fields']).ravel())
    if not same:
        raise SystemExit(
            f'[ABORT BOUNCE-PARAMS] {path} ({what}) was solved for a '
            f'DIFFERENT potential (fields/couplings) than requested.  A data '
            f'folder holds the bounces of ONE potential: run this potential '
            f'with a new --data-dir.\n'
            f'  stored fields:    {have_fields}\n'
            f'  requested fields: {fields}\n'
            f'  stored params:    {have}\n'
            f'  requested params: {params}')


def main():
    ap = argparse.ArgumentParser(
        description='vgeneral_phi_v2 stage 0: terminal potential entry + the '
                    'pair-loop bounce gallery + selection')
    add_standard_cli(ap)   # the uniform CLI prefix; --bounce-npz is unused
    #                        here (this stage WRITES the bounce)
    ap.add_argument('--pair', default=None,
                    help='bounce NAME <falseLabel>_<trueLabel> (FV0_TV0, '
                         'M1_TV0, FV0_M2, ...) or the raw file form F<i>_T<j>; '
                         'see --list-bounces / --select-bounce.  Without it '
                         '(and without --select-bounce) the pair menu is '
                         'printed and the stage aborts.')
    ap.add_argument('--pair-name', action='store_true',
                    help='print the bounce NAME of --pair (given as a name or '
                         'in the file form) and EXIT; the production driver '
                         'names its automatic data tag with it')
    ap.add_argument('--select-bounce', action='store_true',
                    help='INTERACTIVE bounce selection: after solving ALL '
                         'downward pairs and plotting the gallery, ask which '
                         'bounce to use (type its NAME).  Ignores --pair.')
    ap.add_argument('--fields', default=None,
                    help="comma-separated scalar field names, e.g. 'x,y,z' "
                         "(distinct single lowercase letters; 'c' reserved). "
                         "REQUIRED: the pipeline ships no built-in model.")
    ap.add_argument('--couplings', default=None,
                    help="the potential couplings as 'name=value,...' over "
                         "the degree-<=4 monomials of the declared --fields "
                         "(e.g. 'x2=-2,y2=-2,xy2=1,x2y2=0.15,x4=1,y4=1'); "
                         "unlisted monomials are 0, an unknown name raises.  "
                         "REQUIRED: the pipeline ships no built-in "
                         "potential.")
    ap.add_argument('--list-bounces', action='store_true',
                    help='print the vacuum catalogue (global labels) and the '
                         'menu of every downward pair, then EXIT')
    ap.add_argument('--seed-range', default=None, metavar='LO,HI',
                    help='vacuum-search seed box [LO,HI]^N (default: derived '
                         'from the couplings by build_guesses, the Cauchy '
                         'root bound of grad V).')
    args = ap.parse_args()

    # the vacuum-search box of THIS run (None = derived from the couplings)
    seed_range = None
    if args.seed_range is not None:
        try:
            lo, hi = (float(v) for v in args.seed_range.split(','))
        except ValueError:
            raise SystemExit(f'[ABORT] --seed-range must be LO,HI '
                             f'(got {args.seed_range!r})')
        if not lo < hi:
            raise SystemExit(f'[ABORT] --seed-range needs LO < HI '
                             f'(got {lo}, {hi})')
        seed_range = (lo, hi)

    # ---- TERMINAL POTENTIAL ENTRY: the model must be given EXPLICITLY -----
    if args.fields is None or args.couplings is None:
        raise SystemExit(
            "[ABORT] the model must be given EXPLICITLY -- this pipeline "
            "ships NO built-in potential.  Pass BOTH --fields and "
            "--couplings, e.g.\n"
            "  --fields x,y --couplings "
            "'c=1.2,x2=-2,y2=-2,xy2=1,x2y2=0.15,x4=1,y4=1'\n"
            "(the example model of the README), or run "
            "run_vgeneral_phi_v2_production.sh with the same flags.")
    fields = parse_fields(args.fields)
    params = couplings_to_params(parse_couplings(args.couplings), fields)

    # THE VACUUM CATALOGUE of this run: the one Newton sweep, handed down to
    # the menu, the pair loop, the gallery and the selection.
    try:
        minima = minima_catalogue(params, fields, seed_range)
    except RuntimeError as e:
        raise SystemExit(str(e))        # e.g. [ABORT SEED-BOX]
    n_minima = len(minima)

    if args.list_bounces:
        list_bounces(fields, minima)
        return

    # ---- --pair policy: no default pair -- every run without --pair /
    #      --select-bounce gets the menu, then aborts.
    if args.pair is None and not args.select_bounce:
        list_bounces(fields, minima)
        raise SystemExit('[ABORT] choose a pair: rerun with --pair <name> '
                         '(menu above) or --select-bounce.')

    # normalize --pair to catalogue indices (iF, iT), the FILE form F<i>_T<j>
    # and COORDINATES (read by the resume check); --select-bounce sets the
    # indices and the file form below.
    req_fv = req_tv = None
    if args.pair and not args.select_bounce:
        try:
            iF, iT = pair_from_name(args.pair, n_minima=n_minima)
        except ValueError as e:
            raise SystemExit(f'[ABORT] {e}  Run --list-bounces to see the '
                             f'menu.')
        if (iF, iT) not in enumerate_bounce_pairs(minima):
            raise SystemExit(
                f'[ABORT] --pair {args.pair!r} (file form F{iF}_T{iT}) is not '
                f'a downward (false, true) pair of the {n_minima} minima found '
                f'under --seed-range {seed_range or "(default)"} -- run '
                f'--list-bounces to see the pairs that exist.')
        if args.pair_name:
            print(bounce_display_name(iF, iT, n_minima))
            return
        args.pair = f'F{iF}_T{iT}'
        req_fv = np.asarray(minima[iF]['point'], float)
        req_tv = np.asarray(minima[iT]['point'], float)

    # the tagged bounce npz of THIS run, the file every later stage reads;
    # its name comes from stage_paths (the one naming rule every stage reads)
    tagged_npz = stage_paths(args.data_dir, args.tag)['bounce']

    # ---- RESUME: the tagged bounce already exists ---------------------------
    if os.path.isfile(tagged_npz):
        _assert_same_potential(tagged_npz, params, fields,
                               'existing bounce being resumed')
        _assert_requested_vacua(tagged_npz, req_fv, req_tv, args.pair,
                                'existing bounce being resumed')
        print(f'[bounce] {tagged_npz} exists with the requested '
              f'fields+couplings -- skip (resume).')
        if args.select_bounce:
            print(f'NOTE: --select-bounce ignored -- {tagged_npz} already '
                  f'exists; delete it (or use a new --tag) to re-select.')
        assert_census_accepted(tagged_npz, params, fields, n_minima)
        return

    # ---- THE PAIR LOOP: solve EVERY downward pair ---------------------------
    gen_dir = os.path.join(args.data_dir, 'bounce_gen')
    os.makedirs(gen_dir, exist_ok=True)
    generate_all_bounces(params=params, fields=fields, minima=minima,
                         run_tag=args.tag, out_dir=gen_dir,
                         seed_range=seed_range)

    # ---- PLOT the whole gallery (ALL solved bounces, never a selection) -----
    # deferred: bounce_gallery imports this module, so a header import would be circular
    from bounce_gallery_vgeneral_phi_v2 import (bounce_gallery,
                                                select_bounce_interactive)
    gallery_dir = os.path.join(args.data_dir, 'gallery')
    gallery_rows = bounce_gallery(gen_dir, params, fields, gallery_dir,
                                  minima, show=args.select_bounce)

    # ---- SELECT (by NAME) ---------------------------------------------------
    if args.select_bounce:
        iF, iT = select_bounce_interactive(gallery_rows, minima, gallery_dir)
        args.pair = f'F{iF}_T{iT}'
    gen_npz = os.path.join(gen_dir, f'bounce_data_{args.pair}.npz')
    if not os.path.isfile(gen_npz):
        have = sorted(f for f in os.listdir(gen_dir)
                      if f.startswith('bounce_data_') and f.endswith('.npz'))
        raise SystemExit(
            f'[ABORT BOUNCE-GEN] the requested pair {args.pair} has NO '
            f'bounce: its solve was rejected above (see the '
            f'[BOUNCE-GEN FAILED] line naming it for the reason) and '
            f'{gen_npz} does not exist.\n'
            f'  bounces that DID solve in {gen_dir}: '
            f'{have if have else "none"}')

    # ---- THE SELECTION GATE (the only census abort); then the selected
    #      pair's npz becomes the tagged bounce of the run -------------------
    assert_census_accepted(gen_npz, params, fields, n_minima)
    shutil.copyfile(gen_npz, tagged_npz)
    print(f'[bounce] SUMMARY: selected {bounce_display_name(iF, iT, n_minima)} '
          f'(file F{iF}_T{iT}), fields {fields} -> {tagged_npz}')


if __name__ == '__main__':
    main()
