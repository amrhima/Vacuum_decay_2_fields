#!/usr/bin/env python3
"""bounce_coupled_toy_model.py -- stage 0: solve (or reuse) the O(4) bounce of
the hardcoded two-field toy potential, then WATCH its accuracy.

WHAT THIS FILE REPLACES
-----------------------
In the general pipeline stage 0 was four files: a solver, an orchestrator, a
selection gallery and a boundary-value polisher.  Here it is one file, because
three of the four jobs have gone away:

  * NO VACUUM SEARCH and NO PAIR ENUMERATION.  The vacua of both hardcoded
    models are written down in potential_coupled_toy_model.MODELS, and so is
    THE PAIR this pipeline runs.  Nothing is looked for.
  * NO INTERACTIVE GALLERY.  The bounce is fixed, so there is nothing to
    choose.  Plotting moved to the standalone plot_bounces_coupled_toy_model.py
    (which draws every downward pair and marks the one used here).
  * NO BOUNCE POLISH.  By design this pipeline uses the bounce EXACTLY AS
    COSMOTRANSITIONS RETURNS IT -- the "less accurate" bounce -- with the
    accuracy-critical CosmoTransitions SETTINGS of the general pipeline kept
    verbatim (BOUNCE_FINDPROFILE / BOUNCE_DEFORM below).  The general pipeline
    additionally refined the converged profile with a direct boundary-value
    solve of the bounce EOM; that refinement is deliberately absent here, and
    the price it costs is MEASURED and reported rather than hidden (see the
    BOUNCE-ACCURACY block).

WHAT IS KEPT, UNCHANGED
-----------------------
  * the fixed CosmoTransitions settings (thinCutoff = 1e-4 is the one that
    matters -- see the block above the constants);
  * the two solver routes and the single switch between them:
      COUPLED potential   -> pathDeformation.fullTunneling (the path curves in
                             field space, so both fields must be solved at once)
      DECOUPLED potential -> tunneling1D.SingleFieldInstanton per moving field
                             (the path is a straight line along that field's
                             axis; this is the 1-D bounce BubbleDet consumes)
  * the GUARDS: BOUNCE-ENDPOINT (the profile must connect the requested vacua)
    and BOUNCE-EOM (gross non-convergence), both fail-loud;
  * the WATCHER BOUNCE-ACCURACY: measure |s2_c|, improve the bounce by walking
    the CosmoTransitions escalation ladder if it misses the trust level, keep
    the better profile, and ALWAYS quantify what the kept bounce costs in lnD
    units.  It never aborts.

WHAT STAGE 0 PRODUCES
---------------------
The O(4)-symmetric two-field bounce phi'_b(r) in the primed (shifted + lifted +
rotated) frame, plus the original-frame profile and the Euclidean action.

FLOW (main()):
  1. if <data-dir>/bounce_data_<tag>.npz exists -> skip (resume; value-matched
     on the model's params);
  2. else if the canonical shipped profile computation/bounce_data_<pair>.npz
     exists for this model -> COPY it, converting to this pipeline's npz schema
     (needs no cosmoTransitions);
  3. else -> SOLVE it from scratch with CosmoTransitions;
  4. ALWAYS -> the BOUNCE-ACCURACY watcher.

OUTPUT   bounce_data_<tag>.npz with keys
         R (nR,), Phi_bounce_prime (nR,2), Phi_bounce_orig (nR,2),
         fields (2,) str, params (15,), false_vac/true_vac (2,),
         S_CT, false_index/true_index, tag, bounce_solver,
         bounce_s2c / bounce_zero_crossing_ok / bounce_trust / bounce_method /
         bounce_dA1_abs / bounce_dA2_abs / bounce_dA3_rel (the BOUNCE-ACCURACY
         verdict; stage 8 turns it into the printed bounce-error term)
         [+ the legacy X_/Y_bounce_prime/orig keys, so older plot tools read it].
"""
import argparse
import os
import shutil
import sys
from collections import namedtuple

import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from potential_coupled_toy_model import (                                # noqa: E402
    FIELDS,
    MODEL_DEFAULT,
    CTShiftedLiftedPotential,
    V_numeric,
    get_model,
    gradV_numeric,
    guard_vacua,
    is_decoupled,
    model_names,
    tunneling_fields,
)
from pipeline_helpers_coupled_toy_model import (add_standard_cli,        # noqa: E402
                                                atomic_savez, load_bounce,
                                                stage_paths)


# --------------------------------------------------------------------------- #
#  COSMOTRANSITIONS ACCURACY SETTINGS -- the fixed settings of the general      #
#  pipeline, carried over verbatim.  CosmoTransitions' OWN defaults             #
#  (fRatioConv = 0.02, thinCutoff = 0.01) stop deforming the path too early and #
#  hand over from the analytic core solution far too late: the four n=1         #
#  translation modes then drift to lambda ~ -0.008..-0.07, i.e. FOUR SPURIOUS   #
#  NEGATIVE modes where the exact bounce has four ZERO modes.  The settings     #
#  below fix the PATH; thinCutoff (next block) fixes the CORE.                  #
#                                                                              #
#  THE ONE GAUGE OF BOUNCE QUALITY is the n=1 crossing s2_c.  The four          #
#  translation modes d_mu phi_b are EXACT zero modes of M_1 for ANY true        #
#  solution of the bounce equation, so s2_c -> 0 is a property with a KNOWN     #
#  answer and every nonzero value IS bounce error -- measured with no reference #
#  solution (s2c_of_bounce below).  The BOUNCE-ACCURACY watcher gates on it,    #
#  and the stage-0b ZERO-MODE guard gates on it again.                          #
#                                                                              #
#  WHAT THIS PIPELINE DOES WHEN THE GAUGE FAILS depends on the potential:       #
#    DECOUPLED -> the error is in the 1-D SHOOTING bisection, and tolerance     #
#                 removes it: the watcher walks BOUNCE_ESCALATION_LADDER rung   #
#                 by rung until |s2_c| clears the trust level (measured on the  #
#                 bdet model: 9.4e-7 -> 3.3e-9 on the first rung).              #
#    COUPLED   -> the error is in the PATH DEFORMATION, and NO CosmoTransitions #
#                 setting removes it (measured floor |s2_c| ~ 6e-5: deformPath  #
#                 stalls in a limit cycle, its spline-projected force 2.9e-4    #
#                 while the true normal force saturates at 1.65e-3).  This      #
#                 pipeline ACCEPTS that floor -- that is what "the less         #
#                 accurate, purely CosmoTransitions bounce" means -- and        #
#                 reports what it costs instead of removing it.                 #
#  TWO quantities are NOT usable as accuracy checks (both measured): the action #
#  S settles to 7 digits while the core offset is still 1.5% wrong, and the EOM #
#  residual is non-monotone in |s2_c| and overstates the error by ~1000x.  The  #
#  BOUNCE-EOM guard below is therefore exactly what its name says -- a          #
#  gross-failure gate (a solve that did not converge at all) -- and never a     #
#  quality meter; nothing in the pipeline decides anything from its value.      #
# --------------------------------------------------------------------------- #
# thinCutoff is the ACCURACY-CRITICAL setting, and CosmoTransitions' default
# (0.01) is far too coarse for the determinant.  CT cannot start integrating at
# r = 0 (the O(4) equation has a 3/r singularity there), so it carries the field
# outward from the origin with the ANALYTIC solution of the LINEARIZED equation
# (exact when V is quadratic: modified Bessel functions), and only hands over to
# the numerical integrator once the field has moved
#     delta_phi_cutoff = thinCutoff * |phi_false - phi_true|
# away from the true vacuum.  thinCutoff is thus a dimensionless FRACTION of the
# field range -- the same number transfers across potentials.  At CT's default
# 0.01 the quadratic approximation is stretched over the first 1% of the field
# range, where the cubic term is no longer negligible; the shooting bisection
# then absorbs that handover error into a BIASED core offset
# a = phi(0) - phi_true, with relative error ~ 0.5|V'''/V''|*Delta_phi*thinCutoff.
# MEASURED: 1.53e-2 at thinCutoff = 1e-2 vs 1.3e-4 at 1e-4 (a is exponentially
# small, so a tiny ABSOLUTE profile error is a large RELATIVE one).  That bias is
# invisible in the action (S converges to 7 digits while a is still 1.5% wrong)
# but it moves lnD_ren by ~0.4 through the tadpole moment A1_fin.  With
# thinCutoff = 1e-4 the profile reproduces a purpose-built high-precision solve
# to 2.5e-8, for ~20 s of extra solve time.
# Do NOT push it far below 1e-4: if delta_phi_cutoff drops BELOW a, CT's handover
# collapses to its rmin floor and accuracy degrades again (measured), so 1e-4 sits
# in the middle of the usable window for every potential tested.
BOUNCE_FINDPROFILE = dict(xtol=1e-9, phitol=1e-9, npoints=1200,
                          thinCutoff=1e-4)                        # 1D shooting
# THE ESCALATION LADDER of the BOUNCE-ACCURACY watcher, walked ONE RUNG AT A
# TIME and stopped as soon as |s2_c| clears the trust level.  Each rung tightens
# the shooting bisection tolerance by ~10^-3 and doubles the stored grid.
# THE LADDER ENDS AT RUNG 2 BY ARITHMETIC, NOT BY CHOICE.  CosmoTransitions
# shoots on x, defined by phi(0) = phi_true + e^{-x}(phi_false - phi_true), and
# stops when the bracket (xmax - xmin) < xtol.  Since x = -ln(a/Delta_phi) with
# a = phi(0) - phi_true, an ABSOLUTE tolerance on x is exactly a RELATIVE
# tolerance on the core offset a -- so xtol = 1e-14 already asks for a to 14
# significant digits, within two decades of what a double can represent, and a
# third rung could not deliver a tighter bracket.  phitol is a different knob:
# it is the fractional/absolute error tolerance of the adaptive RK integrator,
# stepped with xtol so the integration error never limits the bracket.
# thinCutoff is deliberately CONSTANT down the ladder: 1e-4 is already optimal
# (it degrades in EITHER direction), so it is not a dial.
BOUNCE_ESCALATION_LADDER = (
    dict(xtol=1e-12, phitol=1e-12, npoints=2400, thinCutoff=1e-4),   # rung 1
    dict(xtol=1e-14, phitol=1e-14, npoints=4800, thinCutoff=1e-4),   # rung 2
)
BOUNCE_DEFORM      = dict(fRatioConv=3e-4, startstep=5e-4, maxiter=500)  # path
# There is NO path-deformation escalation rung.  A COUPLED bounce's error lives
# in the deformation, and tightening fRatioConv does not remove it: measured
# fRatioConv 3e-4 -> 3e-5 (with fRatioIncrease = 100, ~16 min) moved |s2_c| only
# 7.17e-5 -> 5.92e-5, because deformPath is in a limit cycle rather than short
# of iterations.  So the ladder is a genuine improvement route only for the
# DECOUPLED model; on the coupled one it is expected to change little, and the
# watcher will say so.
BOUNCE_DEFORM_INIT = dict(nb=40)                                  # path spline pts
BOUNCE_VSPLINE     = 1000                                         # V-spline samples
BOUNCE_MAXITER     = 120                                          # outer iters (capped)
EOM_RESIDUAL_MAX   = 0.01     # GROSS-FAILURE gate only (not an accuracy dial):
                              # refuse a solve whose RMS EOM residual is >= 1%


# --------------------------------------------------------------------------- #
#  HOW THE BOUNCE ERROR IS MEASURED (the two functions the watcher runs on)     #
# --------------------------------------------------------------------------- #
def profile_moments(R, DW, trDW3):
    """The TWO moments of a bounce profile through which a profile error
    reaches lnD_ren:

        M1 = Int dr r^3 tr U(r)          (U = DW = H(phi_b(r)) - H_FV)
        A3 = (1/16) Int dr r^5 tr U^3

    WHY THESE TWO.  The tadpole counterterm
    A1_fin = -(1/8) sum_i m_i^2 [1 - ln(m_i^2/mu^2)] Int dr r^3 U_ii(r)
    (stage 6) is a FIXED-weight LINEAR functional of U, so a change of the
    profile moves it in exact proportion to the change of M1, and A1_fin is the
    largest single term of the assembly -- the tadpole DOMINATES the bounce
    error.  The second contributor is the residual band, whose per-wave
    asymptote is I_n(inf) -> A3/nu^3 with exactly this A3 (the same moment the
    band's N-ONSET/N-ASYMPTOTE watcher/guard pair uses), so the band moves by
    band_sum * dA3/A3.  Everything else is a much weaker functional of the
    profile.  Nothing computed here enters lnD_ren -- stage 8 uses the changes
    only, to quote an error bar."""
    R = np.asarray(R, float)
    trDW = np.einsum('kii->k', np.asarray(DW, float))
    M1 = float(np.trapezoid(R ** 3 * trDW, R))
    A3 = float(np.trapezoid(R ** 5 * np.asarray(trDW3, float), R)) / 16.0
    return M1, A3


def s2c_of_bounce(bounce_npz):
    """|s2_c| of a solved bounce npz: the n=1 det(1+M) zero-mode crossing,
    located by the SAME recipe stage 0b uses (find_zero_mode_crossing --
    imported, never duplicated).  The exact answer is 0 (translation
    invariance), so the returned number IS the bounce error in the one place
    where the true value is known.

    The imports are LOCAL because they pull in the band engine (ResidualBand),
    which nothing else in this module needs; they are not circular
    (eigenvalue_zero imports the band engine and the helpers, never this
    file)."""
    from delta_g_bar_greater_equal_3_coupled_toy_model import ResidualBand
    from eigenvalue_zero_coupled_toy_model import find_zero_mode_crossing
    bg = load_bounce(bounce_npz)
    band = ResidualBand(bg['R'], bg['DW'], bg['m2'])      # DEFAULT tolerances
    w_zero2_cont, _resid = find_zero_mode_crossing(band)
    return abs(float(w_zero2_cont))


def eom_residual_rms(R, Phi_prime, pot_prime):
    """RMS of the O(4) bounce-EOM residual  phi'' + (3/r) phi' - dV(phi),
    summed over BOTH field components in the primed frame (the equation
    CosmoTransitions solved), normalized by |dV|_rms.  A converged bounce has
    << 1%; CosmoTransitions' default tolerances leave ~5-10%.  Self-contained
    (no band engine) -- this feeds the BOUNCE-EOM GROSS-FAILURE guard (aborts;
    does NOT enter lnD) and NOTHING else: it is not an accuracy gauge
    (measured: non-monotone in |s2_c| and overstating the error by ~1000x), so
    no decision in this pipeline is taken on its value.  The accuracy gauge is
    |s2_c| (s2c_of_bounce)."""
    from scipy.interpolate import CubicSpline
    R = np.asarray(R, float)
    Phi_prime = np.asarray(Phi_prime, float)
    m = R > 1e-3
    g = pot_prime.dV(Phi_prime)          # vectorised over the radial axis
    res2 = np.zeros(m.sum())
    for i in range(Phi_prime.shape[1]):
        s = CubicSpline(R, Phi_prime[:, i])
        ri = s(R, 2)[m] + 3.0 / R[m] * s(R, 1)[m] - g[m, i]
        res2 = res2 + ri ** 2
    scale = np.sqrt(np.mean(np.sum(g[m] ** 2, axis=1)))
    return float(np.sqrt(np.mean(res2)) / scale)


def assert_bounce_endpoints(Phi_orig, true_vac_orig, false_vac_orig, tag,
                            tail_frac=0.20, core_frac=0.50):
    """GUARD BOUNCE-ENDPOINT (aborts; does NOT enter lnD): the bounce must
    connect the REQUESTED vacua -- tail (r -> inf) close to the false vacuum,
    core (r = 0) on the true-vacuum side.  Distances are scaled by the vacuum
    separation, so the check is potential-independent; the core tolerance is
    loose because a thick wall offsets the core from the exact true vacuum.
    Catches CosmoTransitions silently under/overshooting, e.g. handing back a
    shorter leg whose tail parks at one of the OTHER minima of the toy
    potential instead of the requested false vacuum."""
    true_vac = np.asarray(true_vac_orig, float)
    false_vac = np.asarray(false_vac_orig, float)
    Phi = np.asarray(Phi_orig, float)
    sep = np.linalg.norm(true_vac - false_vac)
    if sep <= 0:
        return
    d_tail = np.linalg.norm(Phi[-1] - false_vac)
    d_core = np.linalg.norm(Phi[0] - true_vac)
    if d_tail > tail_frac * sep or d_core > core_frac * sep:
        raise RuntimeError(
            f"[ABORT BOUNCE-ENDPOINT] bounce {tag} does not connect the "
            f"requested vacua: core Phi(0)={np.round(Phi[0], 4)} (requested "
            f"TV={np.round(true_vac, 4)}), tail Phi(inf)={np.round(Phi[-1], 4)} "
            f"(requested FV={np.round(false_vac, 4)}).  CosmoTransitions likely "
            f"under/overshot (e.g. returned a shorter leg parking at an "
            f"intermediate minimum of the toy potential).")


def _check_eom(R, Phi_prime, pot_prime, tag, what):
    """GUARD BOUNCE-EOM (aborts; does NOT enter lnD): reject a solve that did
    not converge at all.  Shared by both solver routes so there is one gate,
    not two copies of one."""
    eom = eom_residual_rms(R, Phi_prime, pot_prime)
    if not np.isfinite(eom):
        raise RuntimeError(
            f"[ABORT BOUNCE-EOM] {what} bounce {tag} EOM residual is "
            f"non-finite (NaN/inf) -- corrupt profile or zero |dV| "
            f"normalization.  A NaN would silently PASS the 'eom > limit' "
            f"comparison, so fail loud.")
    print(f"  [BOUNCE-EOM] EOM residual (RMS, normalized) = {100*eom:.3f}%  "
          f"(limit {100*EOM_RESIDUAL_MAX:.1f}%)")
    if eom > EOM_RESIDUAL_MAX:
        raise RuntimeError(
            f"[ABORT BOUNCE-EOM] {what} bounce {tag} under-converged: EOM "
            f"residual {100*eom:.2f}% > {100*EOM_RESIDUAL_MAX:.1f}%.  This "
            f"gate is gross NON-CONVERGENCE, not the residual "
            f"path-deformation floor this pipeline deliberately accepts.")


# --------------------------------------------------------------------------- #
#  SOLVER 1: the COUPLED bounce (pathDeformation.fullTunneling, both fields)    #
# --------------------------------------------------------------------------- #
def compute_bounce_coupled(pot_prime, false_vac_orig, true_vac_orig, tag="",
                           findprofile=None, deform=None):
    """The two-field bounce of the COUPLED toy potential.  Because the cross
    terms make the Hessian non-diagonal, the bounce path CURVES in field space:
    the two radial equations cannot be separated and CosmoTransitions' path
    deformation must solve them together (alpha = 3 for O(4)).

    `findprofile` = the 1-D shooting settings dict (None -> BOUNCE_FINDPROFILE),
    `deform` = the path-deformation settings (None -> BOUNCE_DEFORM).  The
    watcher re-solves through here with a BOUNCE_ESCALATION_LADDER rung as
    `findprofile`; `deform` is never escalated (see the constants block)."""
    from cosmoTransitions import pathDeformation as pd

    findprofile = BOUNCE_FINDPROFILE if findprofile is None else findprofile
    deform = BOUNCE_DEFORM if deform is None else deform
    false_vac_orig = np.asarray(false_vac_orig, dtype=float)
    true_vac_orig = np.asarray(true_vac_orig, dtype=float)

    false_prime = pot_prime.to_prime(false_vac_orig)   # should be ~0
    true_prime = pot_prime.to_prime(true_vac_orig)

    print("\n========================================================")
    print("Computing COUPLED two-field bounce for pair", tag)
    print("  fields           =", FIELDS)
    print("  false_vac (orig) =", false_vac_orig)
    print("  true_vac  (orig) =", true_vac_orig)
    print("  false'           =", false_prime)
    print("  true'            =", true_prime)
    print("  solver: cosmoTransitions pathDeformation.fullTunneling "
          "(alpha=3, O(4))")
    print("========================================================")
    print("  |false'| = ", np.linalg.norm(false_prime))
    print("  Check: V'(false') =", float(pot_prime.V(false_prime)),
          " (should be ~0)")

    path_guess_prime = np.vstack([true_prime, false_prime])   # (2, 2)

    Y = pd.fullTunneling(
        path_guess_prime,
        pot_prime.V,
        pot_prime.dV,
        maxiter=BOUNCE_MAXITER,
        verbose=True,
        V_spline_samples=BOUNCE_VSPLINE,
        tunneling_init_params={"alpha": 3},  # O(4)
        tunneling_findProfile_params=findprofile,
        deformation_init_params=BOUNCE_DEFORM_INIT,
        deformation_deform_params=deform,
    )

    print("CosmoTransitions action S =", Y.action)
    print("Final fRatio =", Y.fRatio)

    R = Y.profile1D.R
    Phi_prime = np.asarray(Y.Phi, float)      # (nR, 2)
    Phi_orig = pot_prime.to_original(Phi_prime)

    assert_bounce_endpoints(Phi_orig, true_vac_orig, false_vac_orig, tag)
    _check_eom(R, Phi_prime, pot_prime, tag, 'coupled')
    return true_vac_orig, false_vac_orig, R, Phi_prime, Phi_orig, Y.action


# --------------------------------------------------------------------------- #
#  SOLVER 2: the DECOUPLED bounce (1-D SingleFieldInstanton per moving field)   #
#                                                                              #
#  When the potential has no cross terms (V = V_1(x) + V_2(y), Hessian          #
#  diagonal everywhere) the vector bounce EOM SEPARATES EXACTLY: each component #
#  of any vector solution solves its own radial equation.  The configuration    #
#  assembled below is therefore ONE exact solution of the FULL two-field EOM    #
#  (validated by the BOUNCE-EOM guard; it coincides with what path deformation  #
#  converges to, action equal to 7 digits) -- solving per component is a        #
#  solution METHOD made exact by separability, NOT an approximation.            #
#  WHY 1-D IS EXACT HERE: the one-flip decoupled bounce path is a STRAIGHT LINE #
#  along the tunneling field's axis -- the transverse gradient vanishes         #
#  identically on it (the frozen field sits at a stationary point of its own    #
#  potential), so path deformation has nothing to deform and the problem        #
#  reduces EXACTLY to the 1-D radial instanton.  This is the same reduction     #
#  BubbleDet makes, which is why the decoupled model is the BubbleDet reference.#
#  MODE-CENSUS CAVEAT: a pair with EXACTLY ONE moving field is a genuine        #
#  whole-vector Coleman bounce (1 negative mode, 4 zero modes d_mu Phi_b, the   #
#  frozen component contributing 0 to d_mu Phi_b) and runs through the pipeline #
#  unchanged.  A pair where BOTH fields flip is a PRODUCT SADDLE: the           #
#  block-diagonal operator has 1 negative + 4 zero modes PER moving block       #
#  (Morse index >= 2), so it is NOT a decay-rate bounce -- the COLEMAN-N0/N1    #
#  guards REJECT it downstream (exit 3 -> clean skip) rather than compute a     #
#  lnD_ren for it.  A frozen component has U_ii(r) = 0 -> contributes 0 to lnD  #
#  and enters only through mu = sum_i m_i.                                      #
# --------------------------------------------------------------------------- #
def single_field_V_dV(params, base_point, idx):
    """The 1-D potential V(phi) and derivative dV(phi) of field `idx`, with the
    other field frozen at base_point.  For a decoupled potential this is
    exactly V_idx(phi) + const (V_idx depends on phi_idx only, and the constant
    does not affect the instanton).  Both callables accept a scalar or a 1-D
    array and return the matching shape (SingleFieldInstanton calls them with
    both)."""
    base = np.asarray(base_point, dtype=float)

    def _points(phi):
        phi_arr = np.atleast_1d(np.asarray(phi, dtype=float))
        pts = np.tile(base, (phi_arr.size, 1))
        pts[:, idx] = phi_arr
        return pts

    def V1(phi):
        out = np.atleast_1d(V_numeric(_points(phi), params))
        return float(out[0]) if np.ndim(phi) == 0 else out

    def dV1(phi):
        g = gradV_numeric(_points(phi), params)[:, idx]
        return float(g[0]) if np.ndim(phi) == 0 else g

    return V1, dV1


def _common_radial_grid(grids, rel_tol=1e-7):
    """A clean common radial grid from several per-field instanton grids: the
    sorted union with near-duplicate radii MERGED (points closer than
    rel_tol * span are collapsed).  Without the merge, similar fields
    contribute near-coincident nodes (differing at ~1e-15) that would make the
    downstream CubicSpline second derivative numerically singular; genuinely
    distinct nodes are kept, so each field keeps its native resolution around
    its own wall."""
    allr = np.unique(np.concatenate([np.asarray(g, float) for g in grids]))
    if allr.size < 3:
        return allr
    mindx = rel_tol * (allr[-1] - allr[0])
    keep = [allr[0]]
    for r in allr[1:-1]:
        if r - keep[-1] > mindx:
            keep.append(r)
    keep.append(allr[-1])                     # always preserve the endpoint
    return np.array(keep)


def compute_bounce_decoupled(pot_prime, false_vac_orig, true_vac_orig, params,
                             tun_idx, tag="", findprofile=None):
    """The DECOUPLED two-field bounce, assembled from one 1-D
    SingleFieldInstanton per moving field in `tun_idx` (alpha = 3, O(4)) -- the
    exact per-field input BubbleDet consumes.  Each field is solved on its own
    native grid, then interpolated onto the UNION radial grid; a frozen field
    stays at its vacuum value.  The total action is the sum of the per-field
    actions.  `findprofile` = the shooting settings (None ->
    BOUNCE_FINDPROFILE); the watcher passes a BOUNCE_ESCALATION_LADDER rung
    here to re-solve tighter -- this is THE route by which a decoupled bounce
    is improved.  Same return shape as compute_bounce_coupled."""
    from cosmoTransitions.tunneling1D import SingleFieldInstanton
    from scipy.interpolate import CubicSpline

    findprofile = BOUNCE_FINDPROFILE if findprofile is None else findprofile
    false_vac_orig = np.asarray(false_vac_orig, dtype=float)
    true_vac_orig = np.asarray(true_vac_orig, dtype=float)
    tun_idx = [int(i) for i in tun_idx]

    print("\n========================================================")
    print("Computing DECOUPLED bounce for pair", tag)
    print("  fields            =", FIELDS)
    print("  moving field(s)   =", [FIELDS[i] for i in tun_idx],
          "(each solved as its own 1-D instanton; the other is frozen)")
    print("  false_vac (orig)  =", false_vac_orig)
    print("  true_vac  (orig)  =", true_vac_orig)
    print("  solver: cosmoTransitions SingleFieldInstanton (alpha=3, O(4))")
    print("          -- the SAME 1-D solver BubbleDet consumes")
    print("========================================================")

    solves = {}                                   # idx -> (R_i, Phi_i, S_i)
    for idx in tun_idx:
        V1, dV1 = single_field_V_dV(params, false_vac_orig, idx)
        inst = SingleFieldInstanton(
            phi_absMin=float(true_vac_orig[idx]),
            phi_metaMin=float(false_vac_orig[idx]),
            V=V1, dV=dV1,
            alpha=3,                              # alpha = d - 1 for d = 4
        )
        # pass the WHOLE settings dict: naming individual keys would silently
        # drop npoints (falling back to CT's default 500) and the
        # accuracy-critical thinCutoff.  The coupled route forwards the same
        # dict via fullTunneling(tunneling_findProfile_params=...), so both
        # solver routes run at identical accuracy settings.
        prof = inst.findProfile(**findprofile)
        R_i = np.asarray(prof.R, dtype=float)
        Phi_i = np.asarray(prof.Phi, dtype=float)
        S_i = float(inst.findAction(prof))
        solves[idx] = (R_i, Phi_i, S_i)
        print(f"  field '{FIELDS[idx]}': S={S_i:.6f}, {len(R_i)} pts, "
              f"r in [{R_i[0]:.3e}, {R_i[-1]:.4f}]")

    R = _common_radial_grid([solves[i][0] for i in tun_idx])

    # Assemble the ORIGINAL-frame profile: both fields at their false-vacuum
    # value, each moving field overwritten by its instanton resampled onto the
    # common grid.  The resampling is CUBIC (not linear) so no curvature kink
    # is introduced at the interleaved nodes -- a linear resample would spike
    # the downstream second-derivative EOM residual.  Outside a field's own
    # [R_i[0], R_i[-1]] it is clamped: the r=0 escape value below, the
    # (relaxed) false vacuum beyond its wall.
    Phi_orig = np.tile(false_vac_orig, (len(R), 1))
    for idx in tun_idx:
        R_i, Phi_i, _ = solves[idx]
        vals = CubicSpline(R_i, Phi_i, extrapolate=False)(R)
        vals = np.where(R < R_i[0], Phi_i[0], vals)
        vals = np.where(R > R_i[-1], float(false_vac_orig[idx]), vals)
        Phi_orig[:, idx] = vals
    Phi_prime = pot_prime.to_prime(Phi_orig)
    S = float(sum(solves[i][2] for i in tun_idx))

    print(f"Total decoupled bounce action S = {S:.6f}   "
          f"(union grid: {len(R)} points, r in [{R[0]:.3e}, {R[-1]:.4f}])")

    assert_bounce_endpoints(Phi_orig, true_vac_orig, false_vac_orig, tag)
    _check_eom(R, Phi_prime, pot_prime, tag, 'decoupled')
    return true_vac_orig, false_vac_orig, R, Phi_prime, Phi_orig, S


# --------------------------------------------------------------------------- #
#  THE SWITCH: coupled -> path deformation, decoupled -> 1-D instanton          #
# --------------------------------------------------------------------------- #
def solve_bounce(pot_prime, false_vac_orig, true_vac_orig, params, tag="",
                 findprofile=None, deform=None):
    """THE one place where the coupled/decoupled distinction is made.  Below
    this function the two cases are indistinguishable: the same npz schema, the
    same watcher, the same eigenvalue/band/sector/counterterm/assembly stages.

      DECOUPLED potential (no cross terms) -> one 1-D SingleFieldInstanton per
          moving field (exact; the per-field BubbleDet input);
      COUPLED potential (any cross term)   -> pathDeformation.fullTunneling.

    `findprofile` (None -> BOUNCE_FINDPROFILE) and `deform` (None ->
    BOUNCE_DEFORM) are forwarded UNCHANGED, so a watcher re-solve on an
    escalation rung takes the identical code path.  The 1-D route never deforms
    a path, so `deform` simply does not reach it.
    Returns (true_vac, false_vac, R, Phi_prime, Phi_orig, S, solver_label)."""
    false_vac_orig = np.asarray(false_vac_orig, dtype=float)
    true_vac_orig = np.asarray(true_vac_orig, dtype=float)
    tun = tunneling_fields(false_vac_orig, true_vac_orig)
    if is_decoupled(params):
        res = compute_bounce_decoupled(pot_prime, false_vac_orig,
                                       true_vac_orig, params, tun, tag,
                                       findprofile=findprofile)
        names = ",".join(FIELDS[i] for i in tun)
        return res + (f"SingleFieldInstanton 1D per field (decoupled; "
                      f"moving field(s) {names})",)
    res = compute_bounce_coupled(pot_prime, false_vac_orig, true_vac_orig, tag,
                                 findprofile=findprofile, deform=deform)
    return res + ("pathDeformation.fullTunneling (two coupled fields)",)


# --------------------------------------------------------------------------- #
#  npz schema                                                                  #
# --------------------------------------------------------------------------- #
def bounce_npz_payload(R, Phi_prime, Phi_orig, params, false_vac, true_vac,
                       S_CT, iF, iT, tag,
                       solver="pathDeformation.fullTunneling (two coupled fields)",
                       mass_basis_L=None):
    """The bounce npz payload: the two-field schema keys plus the legacy X_/Y_
    keys (kept so older plot tools keep working; every reader in this pipeline
    uses Phi_bounce_prime).  `bounce_solver` records which solver produced it
    (provenance only).  `mass_basis_L` (when given) persists the false-vacuum
    Hessian eigenbasis the primed profile was written in (columns =
    eigenvectors): PROVENANCE for the BOUNCE-BASIS guard -- eigh's sign/order
    choices are not unique across platforms, and storing the actual L used
    makes the stored primed profile auditable against the canonical
    original-frame one."""
    payload = dict(
        params=np.asarray(params, float),
        fields=np.array(list(FIELDS)),
        false_vac=np.asarray(false_vac, float),
        true_vac=np.asarray(true_vac, float),
        R=np.asarray(R, float),
        Phi_bounce_prime=np.asarray(Phi_prime, float),
        Phi_bounce_orig=np.asarray(Phi_orig, float),
        S_CT=S_CT,
        false_index=iF,
        true_index=iT,
        tag=tag,
        bounce_solver=solver,
        X_bounce_prime=np.asarray(Phi_prime[:, 0], float),
        Y_bounce_prime=np.asarray(Phi_prime[:, 1], float),
        X_bounce_orig=np.asarray(Phi_orig[:, 0], float),
        Y_bounce_orig=np.asarray(Phi_orig[:, 1], float),
    )
    if mass_basis_L is not None:
        payload['mass_basis_L'] = np.asarray(mass_basis_L, float)
    return payload


def convert_legacy_npz(src, dst):
    """Copy a canonical shipped bounce npz (legacy X_/Y_ keys, no `fields`) to
    dst in this pipeline's schema (adds Phi_bounce_prime/orig + fields).  All
    numeric content is byte-preserved -- only keys are added."""
    b = np.load(src, allow_pickle=True)
    if 'Phi_bounce_prime' in b.files:            # already new-schema: plain copy
        shutil.copyfile(src, dst)
        return
    Phi_prime = np.stack([np.asarray(b['X_bounce_prime'], float),
                          np.asarray(b['Y_bounce_prime'], float)], axis=1)
    Phi_orig = np.stack([np.asarray(b['X_bounce_orig'], float),
                         np.asarray(b['Y_bounce_orig'], float)], axis=1)
    solver = (str(b['bounce_solver']) if 'bounce_solver' in b.files
              else "pathDeformation.fullTunneling (two coupled fields)")
    np.savez(dst, **bounce_npz_payload(
        np.asarray(b['R'], float), Phi_prime, Phi_orig,
        np.asarray(b['params'], float),
        np.asarray(b['false_vac'], float), np.asarray(b['true_vac'], float),
        b['S_CT'], int(b['false_index']), int(b['true_index']), str(b['tag']),
        solver=solver))


# --------------------------------------------------------------------------- #
#  WATCHER BOUNCE-ACCURACY (stage 0)                                           #
#                                                                              #
#  THE GAUGE.  |s2_c| is the n=1 det(1+M) crossing of the solved bounce.  The   #
#  four n=1 modes are the translation modes d_mu phi_b, EXACT zero modes of the #
#  fluctuation operator for ANY true solution of the bounce equation, so the    #
#  exact answer is s2_c = 0 and every nonzero value IS bounce error -- measured #
#  with no reference solution.  --bounce-trust is the level below which that    #
#  error does not move lnD_ren measurably.                                      #
#                                                                              #
#  THE FLOW:                                                                   #
#    1. CosmoTransitions solves the bounce with the FIXED settings              #
#       BOUNCE_FINDPROFILE.  This always happens first.                         #
#    2. |s2_c| is measured on that profile.                                     #
#    3. If it MISSES the trust level, the CosmoTransitions escalation ladder is #
#       walked one rung at a time and the profile with the smaller |s2_c| is    #
#       kept.  That is the ONLY improvement route in this pipeline: it stays    #
#       purely CosmoTransitions, by design.  (On the DECOUPLED model it works   #
#       -- measured 9.4e-7 -> 3.3e-9 on rung 1.  On the COUPLED model it will   #
#       barely move: the error is in the path deformation, which no             #
#       CosmoTransitions setting reaches, and this pipeline accepts that floor.)#
#    4. EVERY kept bounce is MEASURED (_quantify_lnD_error): its |dA1_fin| and
#       |dA2_fin|/2 in lnD units, plus dA3/A3 for the band, against ANOTHER
#       CosmoTransitions solve at different settings, on the SAME radial grid.
#       When the ladder ran, one of its own candidates IS that solve and is
#       reused (a coupled solve costs minutes); only a bounce that cleared the
#       trust level as solved pays for a fresh reference.  Passing the
#       zero-crossing condition does NOT exempt a bounce from this.            #
#                                                                              #
#  WHY |s2_c| CANNOT CERTIFY ACCURACY.  It is the first-order shift of the      #
#  translation mode, so it weights the profile error by chi = d_r phi_b, which  #
#  peaks at the WALL.  A CosmoTransitions error lives in the exponentially      #
#  small CORE OFFSET, where chi is tiny, so the gauge under-reports it by ~1e3. #
#  So |s2_c| decides whether to IMPROVE a bounce, and gates the stage-0b abort  #
#  on a QUALITATIVELY wrong one (spurious negative modes); it never decides     #
#  whether the error is small.                                                  #
#                                                                              #
#  WHAT THE MEASURED NUMBER DOES AND DOES NOT COVER -- read this before quoting #
#  it.  The reference is a CosmoTransitions solve at the tightest settings, so  #
#  the number is a SELF-CONVERGENCE estimate: it bounds the part of the bounce  #
#  error that CosmoTransitions' own tolerances control.  On the DECOUPLED model #
#  that is the whole error (the 1-D shooting converges to the true solution).   #
#  On the COUPLED model it is NOT: the path-deformation limit cycle biases      #
#  every rung the same way, so that shared component cancels out of the         #
#  difference and is invisible here.  Removing it needs a boundary-value        #
#  refinement of the bounce EOM, which this pipeline deliberately does not do   #
#  -- that is exactly the accuracy this toy model trades away.                  #
#                                                                              #
#  The watcher NEVER aborts: a bounce that cannot clear the trust level is      #
#  still a usable bounce, it just comes with a measured error.  Aborting is the #
#  stage-0b ZERO-MODE guard's job, at a much looser tolerance.                  #
# --------------------------------------------------------------------------- #

# One comparison profile.  `rung` orders the candidates by how tight the
# CosmoTransitions settings that produced them were: -1 = the default
# BOUNCE_FINDPROFILE ("as solved"), 0..len(ladder)-1 = that rung of
# BOUNCE_ESCALATION_LADDER.  The accuracy reference is picked with it.
_Cand = namedtuple('_Cand', 'path s2c label rung')


def _npz_dict(path):
    """Every key of an npz as a plain dict (so it can be re-written with the
    accuracy keys added, nothing else changed)."""
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def _changed_settings(new, old):
    """'key=value, ...' for every setting a ladder rung actually CHANGES
    relative to the defaults (used only for the printed message, so the message
    can never drift from the code)."""
    return ', '.join(f'{k}={v:g}' for k, v in new.items() if old.get(k) != v)


def _lnD_terms_of(bounce_npz):
    """(A1_fin, A2_fin, A3) of a bounce npz -- the three objects a profile
    error reaches lnD_ren through, each computed by the SAME production code
    that computes it for the assembly:

      A1_fin  the tadpole counterterm (counterterm_tadpole.compute_A1_fin,
              ~0.5 ms).  Computed DIRECTLY rather than inferred from the
              unweighted moment M1: A1_fin weights each channel by
              K_i(mu) = -(1/8) m_i^2 [1 - ln(m_i^2/mu^2)], and on a genuinely
              two-channel COUPLED bounce those weights differ enough that M1 is
              a poor proxy.  A1_fin IS the K-weighted moment, so computing it
              directly removes the proxy entirely.
      A2_fin  the fish counterterm (counterterm_fish.compute_A2_fin, ~10 s).
              It enters lnD as -A2_fin/2.  dA2 is relatively tiny, but the
              omitted TERM was measured to be a systematic ~25% understatement
              of the total error, so it is included.
      A3      the band's leading 1/nu^3 coefficient.  Returned as a moment, not
              in lnD units, because the band sum it scales is not known until
              stage 8 -- stage 8 multiplies.

    Both A1_fin and A2_fin come back in ABSOLUTE lnD units, so no
    proportionality assumption is needed for either."""
    from counterterm_fish_coupled_toy_model import compute_A2_fin
    from counterterm_tadpole_coupled_toy_model import compute_A1_fin
    from pipeline_helpers_coupled_toy_model import potential_insertion_V
    bg = load_bounce(bounce_npz)
    V_full, V_diag = potential_insertion_V(bg['R'], bg['Phi'], bg['pot'],
                                           bg['Hfv'])
    A1 = compute_A1_fin(bg['R'], V_diag, bg['masses'], bg['mu'])
    A2, _conv = compute_A2_fin(bg['R'], V_full, bg['masses'], bg['mu'],
                               verbose=False)
    _M1, A3 = profile_moments(bg['R'], bg['DW'], bg['trDW3'])
    return float(A1), float(A2), float(A3)


def _resample_onto(R_ref, Phi_ref, R_target, false_vac):
    """Tabulate a reference profile on the KEPT profile's radial grid, so the
    two counterterm integrals below run over IDENTICAL nodes and their
    difference is a pure SOLUTION difference with no quadrature or domain gap
    folded in.  Cubic spline inside the reference's own radial range, clamped
    to its core value below it and to the false vacuum beyond its wall."""
    from scipy.interpolate import CubicSpline
    R_ref = np.asarray(R_ref, float)
    Phi_ref = np.asarray(Phi_ref, float)
    R_target = np.asarray(R_target, float)
    out = np.empty((R_target.size, Phi_ref.shape[1]))
    for i in range(Phi_ref.shape[1]):
        vals = CubicSpline(R_ref, Phi_ref[:, i], extrapolate=False)(R_target)
        vals = np.where(R_target < R_ref[0], Phi_ref[0, i], vals)
        vals = np.where(R_target > R_ref[-1], float(false_vac[i]), vals)
        out[:, i] = vals
    return out


def _reference_profile(kept, cands, pot_prime, k, name):
    """(R, Phi_orig, S, what) of the profile the kept bounce is measured
    AGAINST, or None if none could be produced.

    THE REFERENCE IS ANOTHER COSMOTRANSITIONS SOLVE AT DIFFERENT SETTINGS, and
    the cheapest correct one is reused rather than re-solved: if the watcher
    already walked the ladder, its candidates are exactly such solves, so the
    one with the TIGHTEST settings that is not the kept profile itself is used
    (a coupled path-deformation solve costs minutes, so this matters).  Only
    when the ladder never ran -- the bounce cleared the trust level as solved
    -- is a reference solved fresh, at the tightest rung.

    Note the difference is symmetric, so using a LOOSER candidate (which
    happens when the kept profile is itself the tightest one) measures the same
    self-convergence gap; it is not a weaker measurement."""
    others = [c for c in cands if c.path != kept.path]
    if others:
        ref = max(others, key=lambda c: c.rung)
        d = _npz_dict(ref.path)
        what = (f"the '{ref.label}' profile already solved by the escalation "
                f"ladder")
        print(f"[BOUNCE-ACCURACY] {name}: measuring the kept bounce against "
              f"{what} (no extra solve needed)")
        return (np.asarray(d['R'], float),
                np.asarray(d['Phi_bounce_orig'], float), d['S_CT'], what)

    rung = BOUNCE_ESCALATION_LADDER[-1]
    what = (f"a fresh CosmoTransitions solve at "
            f"{_changed_settings(rung, BOUNCE_FINDPROFILE)}")
    print(f"[BOUNCE-ACCURACY] {name}: the bounce cleared the trust level as "
          f"solved, so no ladder profile exists to compare it with -- solving "
          f"{what} as the accuracy reference")
    try:
        (_tv, _fv, R_ref, _Pp, Phi_ref, S_ref, _solver) = solve_bounce(
            pot_prime, np.asarray(k['false_vac'], float),
            np.asarray(k['true_vac'], float), np.asarray(k['params'], float),
            f"{name} [accuracy reference]", findprofile=rung)
    except Exception as exc:
        print(f"[BOUNCE-ACCURACY] {name}: the reference solve FAILED "
              f"({type(exc).__name__}: {exc}) -- the bounce error cannot be "
              f"quantified for this bounce and is reported as UNQUANTIFIED, "
              f"never as zero")
        return None
    return R_ref, Phi_ref, S_ref, what


def _quantify_lnD_error(kept, cands, pot_prime, name):
    """The bounce error of the KEPT profile, IN lnD UNITS, measured against a
    reference rather than inferred from a proxy.

    The reference (see _reference_profile) is another CosmoTransitions solve at
    different settings, tabulated on the KEPT profile's own radial grid.
    Passing the grid through _resample_onto is what makes the comparison exact:
    both profiles are integrated over identical nodes, so the difference of
    their counterterms is a pure SOLUTION difference with no quadrature or
    domain gap folded in.  Read the "WHAT THE MEASURED NUMBER DOES AND DOES NOT
    COVER" paragraph in the block above before quoting it: on the decoupled
    model this is the whole bounce error, on the coupled one it is the
    settings-controlled part only.

    Returns (dA1_abs, dA2_abs, dA3_rel):
        dA1_abs  |A1_fin(kept) - A1_fin(ref)|         lnD units, exact
        dA2_abs  |A2_fin(kept) - A2_fin(ref)| / 2     lnD units, exact
                 (A2 enters the assembly as -A2_fin/2)
        dA3_rel  |A3(kept) - A3(ref)| / |A3(kept)|    RELATIVE -- the band sum
                 it scales is not known until stage 8, which multiplies.
    All three are NaN if no reference could be built; the caller stores that
    and stage 8 reports the error as unquantified rather than as zero."""
    _nan3 = (float('nan'),) * 3
    k = _npz_dict(kept.path)
    fv = np.asarray(k['false_vac'], float)
    R_keep = np.asarray(k['R'], float)

    got = _reference_profile(kept, cands, pot_prime, k, name)
    if got is None:
        return _nan3
    R_ref, Phi_ref, S_ref, what = got

    Phi_ref = _resample_onto(R_ref, Phi_ref, R_keep, fv)
    ref_path = os.path.splitext(kept.path)[0] + '_ref.npz'
    _write_candidate(ref_path, k, pot_prime, R_keep,
                     pot_prime.to_prime(Phi_ref), Phi_ref, S_ref,
                     f'accuracy reference: {what}')
    try:
        A1k, A2k, A3k = _lnD_terms_of(kept.path)
        A1r, A2r, A3r = _lnD_terms_of(ref_path)
    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)

    dA1_abs = abs(A1k - A1r)
    dA2_abs = abs(A2k - A2r) / 2.0
    dA3_rel = abs(A3k - A3r) / abs(A3k) if A3k != 0.0 else float('nan')
    print(f"[BOUNCE-ACCURACY] {name}: bounce error measured against {what}, "
          f"on the SAME radial grid -- "
          f"|dA1_fin| = {dA1_abs:.3e} (tadpole, lnD units), "
          f"|dA2_fin|/2 = {dA2_abs:.3e} (fish, lnD units), "
          f"dA3/A3 = {dA3_rel:.3e} (band; stage 8 scales it by the band sum)")
    return dA1_abs, dA2_abs, dA3_rel


def _write_candidate(path, b, pot_prime, R, Phi_prime, Phi_orig, S, solver):
    """Write one comparison profile as a schema-complete bounce npz.  Every
    candidate goes through this, so |s2_c| and the moments of a candidate are
    measured by exactly the same pipeline code that reads the production
    bounce -- there is no separate measurement path to drift."""
    atomic_savez(path, **bounce_npz_payload(
        R, Phi_prime, Phi_orig,
        np.asarray(b['params'], float),
        np.asarray(b['false_vac'], float), np.asarray(b['true_vac'], float),
        S, int(b['false_index']), int(b['true_index']), str(b['tag']),
        solver=solver, mass_basis_L=pot_prime.L))


def _walk_ct_ladder(b, target, pot_prime, trust, name, cands):
    """Re-solve with CosmoTransitions one rung of BOUNCE_ESCALATION_LADDER at a
    time, stopping at the FIRST rung whose |s2_c| clears `trust` (there is no
    point paying for a tighter rung once the condition is met).  `cands` is
    filled IN PLACE so the caller's cleanup sees every file written."""
    params = np.asarray(b['params'], float)
    for k, rung in enumerate(BOUNCE_ESCALATION_LADDER, 1):
        # A rung that silently DROPPED a key would fall back to
        # CosmoTransitions' own default for it -- in particular thinCutoff 0.01
        # instead of 1e-4, which is the accuracy-critical setting.  The whole
        # dict is forwarded, so check the key set rather than trusting it.
        if set(rung) != set(BOUNCE_FINDPROFILE):
            raise RuntimeError(
                f'[ABORT BOUNCE-ACCURACY] escalation rung {k} sets '
                f'{sorted(rung)} but the defaults set '
                f'{sorted(BOUNCE_FINDPROFILE)} -- a rung must give EVERY '
                f'setting, or CosmoTransitions silently uses its own default '
                f'for the missing one.')
        print(f"[BOUNCE-ACCURACY] {name}: re-solving with tightened "
              f"CosmoTransitions settings because the bounce error is above "
              f"the trust level: {_changed_settings(rung, BOUNCE_FINDPROFILE)} "
              f"  (rung {k} of {len(BOUNCE_ESCALATION_LADDER)})")
        # A rung CosmoTransitions cannot deliver must not kill stage 0: the
        # profiles already in hand are still usable, so drop this rung and keep
        # them.  SystemExit is deliberately NOT caught -- the n=1 census raises
        # it (exit 3) to make the driver skip a multi-wall bounce cleanly.
        path = os.path.splitext(target)[0] + f'_rung{k}.npz'
        try:
            (true_arr, false_arr, R2, Phi_p2, Phi_o2, S2,
             solver2) = solve_bounce(
                pot_prime, np.asarray(b['false_vac'], float),
                np.asarray(b['true_vac'], float), params, str(b['tag']),
                findprofile=rung)
            _write_candidate(path, b, pot_prime, R2, Phi_p2, Phi_o2, S2,
                             solver2)
            s2c_k = s2c_of_bounce(path)
        except Exception as exc:
            print(f"[BOUNCE-ACCURACY] rung {k} FAILED ({type(exc).__name__}: "
                  f"{exc}) -- skipping this rung and keeping the profiles "
                  f"already obtained")
            if os.path.exists(path):
                os.remove(path)
            continue
        cands.append(_Cand(path, s2c_k, f'rung {k}', k - 1))
        print(f"[BOUNCE-ACCURACY] rung {k}: |s2_c| = {s2c_k:.3e} "
              f"(was {cands[-2].s2c:.3e})")
        if s2c_k < trust:
            break


def bounce_accuracy_watcher(target, trust, resumed=False, invalidates=()):
    """Run the BOUNCE-ACCURACY flow documented in the block above on the tagged
    bounce and record the verdict in its npz.

    Written keys (read by stage 8):
      bounce_s2c               |s2_c| of the profile finally KEPT
      bounce_zero_crossing_ok  did it clear `trust`?
      bounce_trust             the level it was judged against
      bounce_method            what was done: 'CosmoTransitions bounce kept as
                               solved' (the condition was already met) or
                               'CosmoTransitions escalation ladder'
      bounce_dA1_abs           |dA1_fin| and |dA2_fin|/2 of the kept profile
      bounce_dA2_abs           against another CosmoTransitions solve at
                               different settings, ALREADY IN lnD UNITS, and
      bounce_dA3_rel           the RELATIVE change of the band moment A3, which
                               stage 8 scales by the band sum.
                               MEASURED FOR EVERY BOUNCE, including one that
                               satisfies the zero-crossing condition.

    IDEMPOTENT: a bounce npz that already carries the verdict is reported from
    the stored values and never re-measured, so a resumed run neither pays for
    the measurement again nor rewrites the bounce file (which would change its
    provenance sha and invalidate every stage output computed against it).
    `resumed` only selects the extra warning printed when the verdict has to be
    added to a bounce file this run did not produce."""
    b = _npz_dict(target)

    def _store(payload, s2c_kept, dA1_abs, dA2_abs, dA3_rel, method, ok):
        """Write the verdict into the bounce npz (nothing else changes)."""
        payload.update(bounce_s2c=s2c_kept, bounce_dA1_abs=dA1_abs,
                       bounce_dA2_abs=dA2_abs, bounce_dA3_rel=dA3_rel,
                       bounce_method=method, bounce_zero_crossing_ok=ok,
                       bounce_trust=trust)
        atomic_savez(target, **payload)
        # Writing the verdict changes the bounce bytes, hence its provenance
        # sha, hence every stage output computed against the OLD bytes.  Most
        # of those are sha-checked when they are read, but the driver's stage
        # 0a/0b resume skips are file-existence / s2_sector_max tests that do
        # NOT look at the sha -- a resumed run could therefore reuse stale
        # eigenvalues and only discover it hours later, after the band.  Delete
        # them here so 0a/0b simply recompute (they cost seconds).
        for stale in invalidates:
            if os.path.exists(stale):
                os.remove(stale)
                print(f"[BOUNCE-ACCURACY] removed {os.path.basename(stale)}: "
                      f"it was computed against the pre-verdict bounce bytes "
                      f"and the driver's stage-0a/0b resume skip does not "
                      f"check the provenance sha")
        if resumed:
            print("[BOUNCE-ACCURACY] the verdict was written into an EXISTING "
                  "bounce file (this run resumed it), so that file's "
                  "provenance sha changed: any OTHER stage output of this tag "
                  "left from an earlier run will now fail its bounce_sha check "
                  "and must be deleted and recomputed.")

    # The resume skip is VALUE-MATCHED on the trust level, like every other
    # resume skip in this pipeline: a verdict recorded against a DIFFERENT
    # --bounce-trust answers a different question and must be re-taken, or a
    # loose earlier run would silently license a tight one.
    if ('bounce_zero_crossing_ok' in b
            and float(b['bounce_trust']) == float(trust)):
        print(f"[BOUNCE-ACCURACY] |s2_c| = {float(b['bounce_s2c']):.3e} was "
              f"already measured for this bounce file (method: "
              f"{str(b['bounce_method'])}, zero-crossing condition "
              f"{'satisfied' if bool(b['bounce_zero_crossing_ok']) else 'NOT satisfied'})"
              f" -- not re-measured")
        return
    if 'bounce_zero_crossing_ok' in b:
        print(f"[BOUNCE-ACCURACY] this bounce carries a verdict taken against "
              f"--bounce-trust {float(b['bounce_trust']):g}, but this run asks "
              f"for {trust:g} -- re-taking it")

    params = np.asarray(b['params'], float)
    iF, iT = int(b['false_index']), int(b['true_index'])
    name = f'F{iF}_T{iT}'

    # ---- the zero-crossing condition on the bounce as solved ----------------
    s2c0 = s2c_of_bounce(target)
    pot_prime = CTShiftedLiftedPotential(params,
                                         np.asarray(b['false_vac'], float))
    decoupled = is_decoupled(params)
    print(f"[BOUNCE-ACCURACY] {name}: |s2_c| = {s2c0:.3e} vs trust {trust:g} "
          f"-- the zero-crossing condition is "
          f"{'SATISFIED' if s2c0 < trust else 'NOT satisfied'}; the potential "
          f"is {'DECOUPLED' if decoupled else 'COUPLED'}")

    cands = [_Cand(target, s2c0, 'as solved', -1)]
    try:
        _improve_and_store(b, target, pot_prime, s2c0, trust, name, decoupled,
                           cands, _store)
    finally:
        # EVERY exit path cleans up the candidate npz files, including the
        # multi-wall sys.exit(3) raised inside the n=1 census (a SystemExit,
        # which no `except Exception` would catch) and any solver abort.
        for c in cands:
            if c.path != target and os.path.exists(c.path):
                os.remove(c.path)


def _improve_and_store(b, target, pot_prime, s2c0, trust, name, decoupled,
                       cands, _store):
    """The improve-keep-report half of the watcher, split out so the caller can
    wrap it in the try/finally that guarantees candidate cleanup.

    ORDER OF OPERATIONS: CosmoTransitions has ALREADY solved this bounce with
    the fixed settings BOUNCE_FINDPROFILE before this function is reached.  The
    ladder below only re-solves it tighter, and only if it has to."""
    method = 'CosmoTransitions bounce kept as solved'
    if s2c0 >= trust:
        _walk_ct_ladder(b, target, pot_prime, trust, name, cands)
        method = 'CosmoTransitions escalation ladder'
        if not decoupled and len(cands) > 1 and cands[-1].s2c >= trust:
            print(f"[BOUNCE-ACCURACY] {name}: the ladder did not clear the "
                  f"trust level, which is EXPECTED on the coupled model -- the "
                  f"residual error is in the path deformation, a limit-cycle "
                  f"floor no CosmoTransitions setting reaches.  This pipeline "
                  f"accepts that floor by design and quantifies it below "
                  f"instead of removing it.")

    # ---- keep the best profile ---------------------------------------------
    best = min(range(len(cands)), key=lambda i: cands[i].s2c)
    kept = cands[best]
    ok = kept.s2c < trust
    payload = b if kept.path == target else _npz_dict(kept.path)
    print(f"[BOUNCE-ACCURACY] {name}: KEPT the '{kept.label}' profile "
          f"(|s2_c| = {kept.s2c:.3e}); the zero-crossing condition is "
          f"{'SATISFIED' if ok else 'still NOT satisfied'}")

    # ---- ALWAYS quantify the error, whether or not the gate passed ---------
    # The zero-crossing condition is NOT a sufficient proxy for the bounce
    # error, and that is measured, not feared: |s2_c| weights the profile error
    # by chi = d_r phi_b, which peaks at the wall, while a CosmoTransitions
    # error lives in the exponentially small core offset -- the gauge
    # under-reports such an error by ~10^3.  Reporting the error as zero would
    # make it the one term in the whole assembly that is silently set to 0
    # rather than quantified, so it is always measured.
    dA1_abs, dA2_abs, dA3_rel = _quantify_lnD_error(kept, cands, pot_prime,
                                                    name)
    _store(payload, kept.s2c, dA1_abs, dA2_abs, dA3_rel, method, ok)


# --------------------------------------------------------------------------- #
#  stage-0 flow                                                                #
# --------------------------------------------------------------------------- #
def parse_pair(spec, model):
    """Resolve a --pair string 'F<i>_T<j>' against the model's HARDCODED
    vacua.  Only a downward (V_false > V_true) pair of THIS model is accepted;
    everything else fails loud with the list of the ones that exist.  Nothing
    is searched for -- the pairs come from potential_coupled_toy_model.MODELS."""
    s = str(spec).strip().upper()
    ok = False
    if s.startswith('F') and '_T' in s:
        a, b = s.split('_T', 1)
        a = a[1:]
        ok = a.isdigit() and b.isdigit()
    if not ok:
        raise SystemExit(f"[ABORT] --pair {spec!r} is not of the form "
                         f"F<i>_T<j> (indices into this model's vacuum list).")
    pair = (int(a), int(b))
    if pair not in model['pairs']:
        raise SystemExit(
            f"[ABORT] F{pair[0]}_T{pair[1]} is not a downward vacuum pair of "
            f"model {model['name']} -- it has "
            f"{['F%d_T%d' % p for p in model['pairs']]}.")
    return pair


def main():
    ap = argparse.ArgumentParser(
        description='coupled toy model stage 0: the two-field bounce of the '
                    'hardcoded potential')
    add_standard_cli(ap)          # --bounce-npz accepted for CLI uniformity
    ap.add_argument('--model', default=os.environ.get(
                        'COUPLED_TOY_MODEL_MODEL', MODEL_DEFAULT),
                    choices=model_names(),
                    help=f'which HARDCODED potential to use: {model_names()}.  '
                         f'{MODEL_DEFAULT} is the coupled toy model (curved '
                         f'path -> CosmoTransitions path deformation); bdet is '
                         f'its decoupled companion (straight path -> the 1-D '
                         f'CosmoTransitions instanton, the BubbleDet input).')
    ap.add_argument('--pair', default=None,
                    help="which of the model's downward vacuum pairs to "
                         "compute, as F<i>_T<j>.  Default: the pair the model "
                         "declares (F2_T0 for the coupled model, F3_T1 for "
                         "bdet).  Only pairs of the hardcoded vacuum list are "
                         "accepted -- run --list-bounces to see them.")
    ap.add_argument('--list-bounces', action='store_true',
                    help="print the model's hardcoded vacua and downward "
                         "pairs, then EXIT without computing anything")
    ap.add_argument('--bounce-src', default=None,
                    help='override the canonical shipped profile path')
    ap.add_argument('--bounce-trust', type=float, default=1e-8,
                    help='the ZERO-CROSSING CONDITION of the BOUNCE-ACCURACY '
                         'watcher: |s2_c| (the n=1 translation-mode crossing, '
                         'EXACTLY 0 for a true bounce, so every nonzero value '
                         'is bounce error) below this means the bounce error '
                         'does not move lnD_ren measurably.  Above it the '
                         'bounce is re-solved on the CosmoTransitions '
                         'escalation ladder first.  Either way the residual '
                         'error is quantified in lnD units and reported by '
                         'stage 8.  The watcher NEVER aborts.')
    args = ap.parse_args()

    # GUARD VACUA: the hardcoded minima must still be minima of the hardcoded
    # couplings (they are written down, not searched for, so nothing else in
    # the code would notice if the two drifted apart).
    model = guard_vacua(args.model)
    params = model['params']
    pair = model['pair'] if args.pair is None else parse_pair(args.pair, model)

    if args.list_bounces:
        print(f"model {model['name']}  "
              f"({'DECOUPLED' if model['decoupled'] else 'COUPLED'}; "
              f"couplings {model['couplings']})")
        print("hardcoded vacua (V ascending):")
        for i, (pt, Vv) in enumerate(zip(model['minima'], model['V_min'])):
            print(f"   M{i}: phi = {np.array2string(pt, precision=8)}   "
                  f"V = {Vv:+.6f}")
        print("downward (false, true) pairs:")
        for (iF, iT) in model['pairs']:
            moving = [FIELDS[i] for i in tunneling_fields(model['minima'][iF],
                                                          model['minima'][iT])]
            mark = '  <== THE PAIR THIS PIPELINE RUNS' if (iF, iT) == \
                model['pair'] else ''
            print(f"   F{iF}_T{iT}: M{iF} -> M{iT}, moving field(s) "
                  f"{moving}{mark}")
        print(f"why: {model['why']}")
        return

    iF, iT = pair
    false_vac = model['minima'][iF]
    true_vac = model['minima'][iT]
    target = os.path.join(args.data_dir, f'bounce_data_{args.tag}.npz')

    if os.path.isfile(target):
        # resume skip is VALUE-MATCHED: the existing bounce must have been
        # solved with the params requested NOW and for the SAME vacuum pair --
        # a same-tag bounce from the other model (or the other pair) must never
        # be silently reused.
        have_npz = np.load(target, allow_pickle=True)
        have = np.asarray(have_npz['params'], float)
        if (have.shape != params.shape
                or not np.allclose(have, params, rtol=0, atol=1e-12)):
            raise SystemExit(
                f'[ABORT BOUNCE-PARAMS] {target} exists but was solved with a '
                f'DIFFERENT potential than model {model["name"]} -- delete it '
                f'or use a different --tag.\n  stored:    {have}\n'
                f'  requested: {params}')
        if (int(have_npz['false_index']), int(have_npz['true_index'])) != pair:
            raise SystemExit(
                f'[ABORT BOUNCE-PAIR] {target} exists but is a DIFFERENT '
                f'vacuum pair than requested -- stored '
                f'F{int(have_npz["false_index"])}_T{int(have_npz["true_index"])}'
                f', requested F{iF}_T{iT}.  Use a --tag that is unique per '
                f'pair, or delete {target}.')
        print(f'[bounce] {target} exists for model {model["name"]} pair '
              f'F{iF}_T{iT} -- skip (resume).')
        have_npz.close()
        _p = stage_paths(args.data_dir, args.tag)
        # the watcher runs on a RESUMED bounce too (it is idempotent: a bounce
        # that already carries the measurement is only reported, not re-solved)
        bounce_accuracy_watcher(target, args.bounce_trust, resumed=True,
                                invalidates=(_p['eig_n0'], _p['eig_n1']))
        return
    os.makedirs(args.data_dir, exist_ok=True)

    # A canonical profile is shipped next to this file for the COUPLED model's
    # four pairs (they were solved with exactly these settings and these
    # vacua); copying it costs nothing and needs no cosmoTransitions install.
    src = args.bounce_src or os.path.join(_HERE, f'bounce_data_F{iF}_T{iT}.npz')
    use_src = os.path.isfile(src)
    if use_src:
        stored = np.load(src, allow_pickle=True)
        if not np.allclose(np.asarray(stored['params'], float), params,
                           rtol=0, atol=1e-12):
            print(f'[bounce] the shipped profile {os.path.basename(src)} was '
                  f'solved for the OTHER model -- ignoring it and solving from '
                  f'scratch')
            use_src = False
        stored.close()

    if use_src:
        convert_legacy_npz(src, target)
        print(f'[bounce] copied the canonical shipped bounce: {src} -> {target}')
    else:
        print(f'[bounce] no shipped profile for model {model["name"]} pair '
              f'F{iF}_T{iT} -> solving from scratch with CosmoTransitions')
        pot_prime = CTShiftedLiftedPotential(params, false_vac)
        (true_arr, false_arr, R, Phi_prime, Phi_orig,
         S_CT, solver) = solve_bounce(pot_prime, false_vac, true_vac, params,
                                      f'F{iF}_T{iT}')
        atomic_savez(target, **bounce_npz_payload(
            R, Phi_prime, Phi_orig, params, false_arr, true_arr, S_CT, iF, iT,
            args.tag, solver=solver, mass_basis_L=pot_prime.L))
        print(f'[bounce] solved + wrote -> {target}   [solver: {solver}]')

    # WATCHER BOUNCE-ACCURACY: measure |s2_c| of the bounce just produced and,
    # if it is above the trust level, re-solve on the CosmoTransitions ladder
    # and keep the better profile (never aborts; see the flow block above).
    _p = stage_paths(args.data_dir, args.tag)
    bounce_accuracy_watcher(target, args.bounce_trust,
                            invalidates=(_p['eig_n0'], _p['eig_n1']))
    print(f'[bounce] SUMMARY: model {model["name"]}, pair F{iF}_T{iT} '
          f'({"decoupled" if model["decoupled"] else "coupled"}), '
          f'fields {FIELDS} -> {target}')


if __name__ == '__main__':
    main()
