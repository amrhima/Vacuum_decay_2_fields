#!/usr/bin/env python3
"""eigenvalue_zero_coupled_toy_model.py -- stage 0b: locate the n = 1 ZERO-mode
eigenvalue w_zero2 as the det(1+M) zero-crossing near s2 = 0 (the
det-crossing eigenvalue condition, kept in its OWN file, separate from
the RK resolvent of gbar_n1 and from the determinant validator in crosschecks/).

METHOD MANDATE (user, standing): ONE method per code.  The det-crossing IS
the eigenvalue condition -- no determinant VALUE is
formed here; only the ZERO of the 2x2 solution-matching determinant
(pipeline_helpers_coupled_toy_model.sector_matching_det) is used to LOCATE the zero-mode pole.

WHAT IT COMPUTES
----------------
The four n=1 modes are the TRANSLATION zero modes (d_mu phi_b): for the exact
bounce they sit at lambda_1 = 0, i.e. the crossing s2_c = -lambda_1 = 0.  A
numerical bounce shifts them slightly; the crossing is found by a SCAN (no
hardcoded bracket), so it works whatever the bounce quality.  Measured levels,
AFTER the stage-0 BOUNCE-ACCURACY watcher has improved the bounce:
  - the DECOUPLED (bdet) bounce on the CosmoTransitions escalation ladder:
    |s2_c| ~ 1e-9 (rung 1 measured 9.4e-7 -> 3.3e-9);
  - the COUPLED (F2_T0) bounce: |s2_c| floors near 6e-5.  That is a
    path-deformation limit cycle no CosmoTransitions setting removes, and THIS
    PIPELINE ACCEPTS IT BY DESIGN -- it runs the bounce exactly as
    CosmoTransitions returns it.  (The general pipeline removed this floor with
    a boundary-value refinement of the bounce EOM, reaching ~1e-11; that
    refinement is deliberately absent here.)  Such a bounce is still perfectly
    usable: it lands far below the abort tolerance, the run continues with a
    loud warning, and stage 8 quotes what it costs in lnD units;
  - an UNDER-CONVERGED bounce: s2_c can be + and O(1e-2..1e-1), i.e.
    lambda_1 < 0 -- FOUR SPURIOUS NEGATIVE modes (a bounce must have exactly
    one negative mode, in n=0).
The same crossing is what the stage-0 BOUNCE-ACCURACY watcher measures (it
calls find_zero_mode_crossing below): stage 0 uses it to decide whether to
re-solve the bounce, this stage uses it to GATE (ZERO-MODE guard).
det(1+M)(s2) is scanned upward from a small negative s2 (the zero modes are the
lowest n=1 modes); the FIRST sign change is bracketed and rooted with brentq
(xtol 1e-12).  No polynomial fit: production = the bracketed brentq root.

GUARDS (none enters lnD)
  COLEMAN-N1    the scan must find a crossing (part 1; else a hard error), and
                no further det(1+M) crossing more than 0.02 above the zero mode
                (part 2).  Such an extra crossing is a genuine negative n=1 mode
                = a lifted relative-wall mode of a MULTI-WALL bounce (Morse index
                >= 2): it does not contribute to the decay rate, so it is
                REJECTED -- the stage prints "multiple zero crossings in sector
                n=1" and exits with code 3 so the driver skips this bounce and
                continues.
  EIG-RESIDUAL  |det(1+M)(s2_c)| <= --crossing-resid-tol (root on the zero;
                hard error).
  ZERO-MODE     |s2_c| <= --zero-mode-tol; a larger crossing means the bounce
                is under-converged (a hard error -- regenerate it), NOT a
                multi-wall mode census.

INPUTS   --bounce-npz (the potential is rebuilt from the coupling vector
         stored in it); no other stage output.
OUTPUT   eig_n1_<tag>.npz (stage_paths key 'eig_n1') with keys
    w_zero2_cont   ()   brentq crossing root s2_c (PRODUCTION; ~0 for a good bounce)
    lam_zm_cont    ()   = -w_zero2_cont = lambda_1 (the pole-subtraction eigenvalue)
    crossing_resid ()   |det(1+M)(s2_c)| (the EIG-RESIDUAL guard value)
    bounce_sha, potential_id, code_version   metadata
"""
import argparse
import os
import sys
import numpy as np
from scipy.optimize import brentq

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from delta_g_bar_greater_equal_3_coupled_toy_model import ResidualBand                          # noqa: E402
from pipeline_helpers_coupled_toy_model import (add_standard_cli, atomic_savez,          # noqa: E402
                                 crossing_multiplicity, load_bounce,
                                 provenance_stamp, sector_matching_det,
                                 stable_crossing_scan, stage_paths)


def _measured_bounce_error(bounce_npz):
    """The stage-0 BOUNCE-ACCURACY measurement carried by a bounce npz, as
    (|dA1_fin|, |dA2_fin|/2) in lnD units, or None if the file carries no
    measurement (it predates the current stage 0).  The band term is NOT
    included: it is stored relative and only stage 8 knows the band sum to
    scale it by."""
    with np.load(bounce_npz, allow_pickle=True) as d:
        if 'bounce_dA1_abs' not in d.files:
            return None
        return float(d['bounce_dA1_abs']), float(d['bounce_dA2_abs'])


def find_zero_mode_crossing(band, s2_scan_max=0.15):
    """Locate the n=1 zero-mode crossing det(1+M)(s2)=0 by a SCAN (no hardcoded
    bracket).  Grid: dense on [-0.01, 0.01] (straddle a near-zero crossing
    finely) then coarser out to s2_scan_max (catch an under-converged bounce
    whose modes drifted to s2_c ~ +0.07).  Take the FIRST sign change scanning
    upward (the zero modes are the lowest n=1 modes) and root it with brentq
    (xtol 1e-12).  Returns (w_zero2_cont, crossing_resid)."""
    def _det1(s2):
        return sector_matching_det(band, 1, float(s2))

    # Self-certifying census (stable_crossing_scan): the two-piece window --
    # dense on [-0.01, 0.01] to straddle the near-zero translation crossing,
    # coarser out to s2_scan_max -- is refined by density DOUBLING until the
    # crossing count stabilizes, so two nearby n=1 roots inside one interval
    # cannot cancel out of the count.
    def _make_scan(npts):
        n_lo = (4 * npts) // 7 or 2      # keep the original 41:30 proportions
        n_hi = npts - n_lo + 1
        return np.unique(np.concatenate([
            np.linspace(-0.01, 0.01, n_lo),
            np.linspace(0.01, s2_scan_max, n_hi)]))

    scan, dets, flips, n_hidden = stable_crossing_scan(band, 1, _make_scan,
                                                       n0=71)
    # ---- GUARD COLEMAN-N1, part 1 (aborts; does NOT enter lnD) -----------
    # the n=1 spectrum must contain the translation zero modes: the scan must
    # find at least one crossing.
    if flips.size == 0:
        raise RuntimeError(
            f'[ABORT COLEMAN-N1] no n=1 det(1+M) sign change on '
            f'[-0.01, {s2_scan_max:g}] -- no zero-mode crossing found '
            f'(wrong potential/bounce?).')
    # hidden roots (svmin dips the det sign missed -- a near-split or
    # tangential even pair anywhere in the window) are EXTRA n=1 modes on top
    # of the single translation crossing: reject exactly like part 2 below.
    if n_hidden > 0:
        print(f"[REJECTED] multiple zero crossings in sector n=1: {n_hidden} "
              f"hidden root(s) beyond the translation zero mode (svmin rank-"
              f"loss dips the det sign could not see -- a near-split or "
              f"tangential pair of extra n=1 modes; a multi-wall bounce, "
              f"Morse index >= 2).  This bounce does NOT contribute to the "
              f"decay rate; skipping it (no lnD_ren).")
        sys.exit(3)
    i0 = int(flips[0])
    w_zero2_cont = float(brentq(_det1, scan[i0], scan[i0 + 1], xtol=1e-12))
    crossing_resid = abs(_det1(w_zero2_cont))
    # ---- GUARD COLEMAN-N1, MULTIPLICITY --------------------------------------
    # The O(4) translation zero mode is a SINGLE radial eigenvector (the angular
    # degeneracy 4 is carried by the (l+1)^2 factor, not by extra radial modes),
    # so a genuine index-1 bounce has a SIMPLE n=1 crossing.  A DEGENERATE n=1
    # crossing (several coincident radial zero modes at the same s2) is a
    # multi-wall composite's extra relative-translation modes -- which a single
    # sign change cannot see -- so REJECT it (exit 3).
    mult = crossing_multiplicity(band, 1, w_zero2_cont)
    if mult > 1:
        print(f"[REJECTED] multiple zero crossings in sector n=1: the n=1 "
              f"det(1+M) crossing at s2={w_zero2_cont:.3e} is {mult}-fold "
              f"DEGENERATE = {mult} coincident radial zero modes -- a multi-wall "
              f"bounce (Morse index >= 2, extra relative-translation modes).  "
              f"This bounce does NOT contribute to the decay rate; skipping it "
              f"(no lnD_ren).")
        sys.exit(3)
    # ---- GUARD COLEMAN-N1, part 2 --------------------------------------------
    # the n=1 wave is exactly the four (degenerate) translation zero modes -> ONE
    # crossing near s2=0.  A further crossing well ABOVE it (separated by more
    # than 0.02, clear of the near-zero cluster and its under-convergence drift
    # band) is a genuine EXTRA negative n=1 mode -- i.e. a lifted relative-wall
    # mode of a MULTI-WALL bounce (Morse index >= 2 in the n=1 sector).  Such a
    # bounce does NOT contribute to the decay rate, so we REJECT it and exit with
    # code 3 ("rejected", NOT an error), so the driver skips it and continues.
    extra = [float(scan[i]) for i in flips[1:] if scan[i] > w_zero2_cont + 0.02]
    if extra:
        print(f"[REJECTED] multiple zero crossings in sector n=1: extra "
              f"crossing(s) at s2={extra} beyond the translation zero mode at "
              f"s2={w_zero2_cont:.3e} -- a genuine negative n=1 mode (a lifted "
              f"relative-wall mode of a multi-wall bounce, Morse index >= 2).  "
              f"This bounce does NOT contribute to the decay rate; skipping it "
              f"(no lnD_ren).  Only index-1 bounces are computed.")
        sys.exit(3)
    print(f"[COLEMAN-N1] 0 extra det(1+M) crossings above "
          f"s2_c + 0.02 on [-0.01, {s2_scan_max:g}] (limit: 0)")
    return w_zero2_cont, crossing_resid


def main():
    ap = argparse.ArgumentParser(
        description='coupled toy model stage 0b: n=1 zero-mode eigenvalue '
                    '(det-crossing locator)')
    add_standard_cli(ap)
    ap.add_argument('--crossing-resid-tol', type=float, default=1e-8,
                    help='EIG-RESIDUAL guard: |det(1+M)(s2_c)| at the located '
                         'crossing must not exceed this (else the brentq root '
                         'is not on the zero)')
    ap.add_argument('--zero-mode-tol', type=float, default=1e-3,
                    help='ZERO-MODE guard, ABORT level: |w_zero2| = |s2_c,n=1| '
                         'must be < this.  The four n=1 modes are EXACT zeros; '
                         'a large crossing (e.g. +0.008) is an under-converged '
                         'bounce with spurious negative modes -- regenerate the '
                         'bounce.')
    ap.add_argument('--zero-mode-trust', type=float, default=1e-8,
                    help='ZERO-MODE guard, TRUST level: |s2_c| below this means '
                         'the bounce is converged enough that its error does not '
                         'move lnD_ren measurably.  Between trust and abort the '
                         'run CONTINUES but prints a loud [ZERO-MODE WARNING] '
                         'pointing at the stage-8 bounce-error term.  Default '
                         '1e-8 is deliberately the SAME value as stage 0\'s '
                         '--bounce-trust, so the two stages never disagree in the '
                         'same log about whether one bounce is converged.  It '
                         'sits above what BOTH stage-0 routes reach: a '
                         'decoupled solve on the escalation ladder measures '
                         '5e-10..3e-9.  The COUPLED bounce floors near 6e-5 '
                         '(the path-deformation limit cycle this pipeline '
                         'accepts by design), so it warns here every time -- '
                         'that warning is expected, not a fault.')
    args = ap.parse_args()

    bg = load_bounce(args.bounce_npz)
    band = ResidualBand(bg['R'], bg['DW'], bg['m2'])   # DEFAULT tolerances

    w_zero2_cont, crossing_resid = find_zero_mode_crossing(band)
    # ---- GUARD EIG-RESIDUAL (aborts; does NOT enter lnD) ---------------------
    # the located root really sits on the zero
    if crossing_resid > args.crossing_resid_tol:
        raise RuntimeError(
            f'[ABORT EIG-RESIDUAL] n=1 det-crossing residual too large: '
            f'|det(1+M)(s2_c)| = {crossing_resid:.2e} > '
            f'{args.crossing_resid_tol:g} -- the brentq root at '
            f'w_zero2 = {w_zero2_cont:+.6e} is not on the det zero.')
    print(f"[EIG-RESIDUAL] |det(1+M)(s2_c)| = {crossing_resid:.2e} "
          f"(limit {args.crossing_resid_tol:g})")
    # ---- GUARD ZERO-MODE (does NOT enter lnD) --------------------------------
    # THE GAUGE OF BOUNCE QUALITY.  The four n=1 translation modes d_mu phi_b are
    # EXACT zero modes of M_1 for ANY true solution of the bounce equation (that
    # is translation invariance, not an approximation), so the exact answer is
    # s2_c = 0 and EVERY nonzero value is bounce error -- measured with no
    # reference solution needed.  Two levels:
    #   |s2_c| <= --zero-mode-trust : converged; bounce error does not move lnD_ren
    #   trust < |s2_c| <= --zero-mode-tol : run CONTINUES with a loud warning
    #                                       pointing at the stage-8 error term
    #   |s2_c| >  --zero-mode-tol  : ABORT -- lambda_1 < 0 would be FOUR SPURIOUS
    #                                NEGATIVE modes (a bounce has exactly one, in
    #                                n=0), so the result cannot be trusted at all.
    # NO lnD-unit number is printed HERE: how much |s2_c| costs depends on THIS
    # potential's tadpole moment (A1_fin ranges over orders of magnitude between
    # models), and that conversion is already done properly elsewhere -- the
    # stage-0 BOUNCE-ACCURACY watcher measures the profile change between two
    # solves and stage 8 prints the resulting error in lnD units next to
    # lnD_ren.  This guard therefore reports |s2_c| and gates on it, nothing
    # more.
    lam_zm_cont = -w_zero2_cont
    s2c = abs(w_zero2_cont)
    _REMEDY = (
        "  WHAT HAS ALREADY BEEN TRIED, AND WHAT IS LEFT:\n"
        "   1. the stage-0 BOUNCE-ACCURACY watcher has ALREADY tried to improve\n"
        "      this bounce, on every rung of the CosmoTransitions escalation\n"
        "      ladder (bounce_coupled_toy_model.BOUNCE_ESCALATION_LADDER:\n"
        "      xtol/phitol down to 1e-14, npoints up to 4800).  The ladder ends\n"
        "      there because a relative bisection tolerance cannot be tightened\n"
        "      past the double-precision floor.  The better profile was kept, so\n"
        "      re-solving is not the fix here.\n"
        "   2. thinCutoff stays at 1e-4 and is NOT an escalation dial: it is\n"
        "      already optimal (measured -- it degrades in EITHER direction\n"
        "      away from that value; see BOUNCE_FINDPROFILE).\n"
        "   3. on the COUPLED model the ladder cannot go below |s2_c| ~ 6e-5:\n"
        "      the residual error is in the PATH DEFORMATION (a limit cycle),\n"
        "      which no CosmoTransitions setting reaches.  This pipeline uses\n"
        "      the purely-CosmoTransitions bounce on purpose and accepts that\n"
        "      floor; removing it would need a boundary-value refinement of the\n"
        "      bounce EOM, which is exactly what this toy model leaves out.\n"
        "   4. a bounce that lands between the trust level and this abort\n"
        "      tolerance is still usable -- stage 8 prints what it costs in lnD\n"
        "      and marks the zero-crossing condition unsatisfied.\n"
        "  NOT usable as accuracy checks (both measured): the action S (settles\n"
        "  to 7 digits while the core offset is still 1.5% wrong) and the\n"
        "  BOUNCE-EOM residual (non-monotone in |s2_c|, overstates by ~1000x).")
    if s2c > args.zero_mode_tol:
        raise RuntimeError(
            f'[ABORT ZERO-MODE] the bounce is not converged enough to trust the '
            f'numerical result of ln D_ren.\n'
            f'  n=1 translation modes are not at zero: s2_c = {w_zero2_cont:+.6e} '
            f'(lambda_1 = {lam_zm_cont:+.6e}), |s2_c| = {s2c:.2e} > '
            f'{args.zero_mode_tol:g}.\n'
            f'  A positive crossing (lambda_1 < 0) means FOUR SPURIOUS NEGATIVE '
            f'modes -- a genuine bounce has exactly one negative mode, in n=0.\n'
            + _REMEDY)
    # ---- the MEASURED error, not an inference from |s2_c| -------------------
    # |s2_c| is reported and gated on above, but it is NOT used to claim the
    # bounce error is small.  It cannot be: |s2_c| is the first-order shift of
    # the translation mode, so it weights the profile error by chi = d_r phi_b,
    # which peaks at the WALL, while a CosmoTransitions error lives in the
    # exponentially small CORE OFFSET -- measured, the gauge under-reports such
    # an error by ~1e3, and a bounce can clear |s2_c| < 1e-8 by four decades
    # while carrying 7.7e-4 in lnD.  The stage-0 BOUNCE-ACCURACY watcher
    # therefore MEASURES the error of every bounce against another
    # CosmoTransitions solve at different settings (see the stage-0 module for
    # what that does and does not cover); this guard reports it.  The two
    # terms below are already in lnD units; the band term needs the band sum,
    # which only stage 8 has, so the total is printed there.
    _lo = _measured_bounce_error(args.bounce_npz)
    if _lo is None:
        print("[ZERO-MODE] this bounce carries no BOUNCE-ACCURACY measurement "
              "(it did not go through the current stage 0), so its error in lnD "
              "units is UNKNOWN -- |s2_c| alone does not establish it")
    else:
        _dA1, _dA2 = _lo
        print(f"[ZERO-MODE] MEASURED bounce error (stage-0 BOUNCE-ACCURACY, "
              f"against another CosmoTransitions solve at different "
              f"settings): |dA1_fin| = {_dA1:.3e} "
              f"+ |dA2_fin|/2 = {_dA2:.3e} = {_dA1+_dA2:.3e} in lnD units, "
              f"before the band term -- stage 8 adds that and prints the total")
    if s2c > args.zero_mode_trust:
        print(f"[ZERO-MODE WARNING] |s2_c| = {s2c:.2e} exceeds the trust level "
              f"{args.zero_mode_trust:g} (abort level {args.zero_mode_tol:g}).\n"
              f"  The exact value is 0, so this is PURE BOUNCE ERROR, and stage "
              f"0 was unable to improve the bounce past this level.  The run "
              f"CONTINUES -- quote the result with the measured error term "
              f"above.\n"
              + _REMEDY)
    else:
        print(f"[ZERO-MODE] |s2_c| = {s2c:.2e} is within the trust level "
              f"{args.zero_mode_trust:g}: the translation modes are where they "
              f"should be, so the bounce is QUALITATIVELY right (no spurious "
              f"negative modes).  That is all this gauge establishes -- what "
              f"the bounce costs in lnD is the measured number above")
    print(f"[ZERO-MODE] w_zero2 = s2_c = {w_zero2_cont:+.6e}, "
          f"lambda_1 = {lam_zm_cont:+.6e} (abort level "
          f"{args.zero_mode_tol:g})")

    out = stage_paths(args.data_dir, args.tag)['eig_n1']
    atomic_savez(out,
             w_zero2_cont=w_zero2_cont,
             lam_zm_cont=lam_zm_cont, crossing_resid=crossing_resid,
             **provenance_stamp(args.bounce_npz, bg['m2']))
    print(f"[EIG n=1] lam_zm_cont = -w_zero2 = {lam_zm_cont:+.6e} "
          f"(brentq det-crossing zero-mode eigenvalue)")
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
