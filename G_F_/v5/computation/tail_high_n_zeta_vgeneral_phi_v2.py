#!/usr/bin/env python3
"""tail_high_n_zeta_vgeneral_phi_v2.py -- level 4, stage 4: the high-n tail
of lnD_ren, Zcal = sum_{n > n_max} I_n(inf), the closed-form sum of ONE
odd-family fit of the s2-completed per-wave numbers I_n(inf) of stage 3 --
the (T6) summand of lnD_ren (stage 7).

READS   band_integrals_<tag>.npz  (stage 3: band_n, I_n_inf)
        band_cutoffs_<tag>.npz    (stage 1: fit_lo, n_max, A3, a3_tol,
                                   eps_n -- the fit window, the analytic
                                   moment and the targets the watchers
                                   stopped on, their one creation point;
                                   LOADED, never re-derived here)
        --bounce-npz              (the provenance stamp only)
WRITES  tail_high_n_<tag>.npz with keys
    fit_a, fit_b, fit_c  ()  the odd-family coefficients of the ONE fit
                             (stage 7 copies them into the lnD_ren npz,
                             from which the lnD_ren part figures draw the
                             high-n tail)
    Zcal                 ()  the Hurwitz-zeta sum over n > n_max
                             (enters lnD_ren: stage 7, term (T6))
    fit_lo, n_max        ()  the fit window (echoed from band_cutoffs;
                             the assembler reads them)
    A3                   ()  the analytic moment the N-ASYMPTOTE guard used
                             (echoed from band_cutoffs; the assembler's
                             eps_bounce reads it)
    a_over_A3_dev, subleading_zeta_bound  the two guard measures (record only, no reader)
    a3_tol, eps_n            the guard limits (echoed from band_cutoffs; the
                             driver's stage-4 skip-guard value-matches them
                             on a resumed run)
    bounce_sha, potential_id, code_version    metadata

HOW
---
lnD_ren regrouped (the assembler's seven terms (T1)-(T7), with
sum_n I_n(inf) = (T3) + (T5) and A1_fin = (T2) + (T4)):

    lnD_ren  = Sigma_dang + sum_{n=2..n_max} I_n(inf) + Zcal
               + A1_fin - A2_fin/2,
    I_n(inf) = I_n(s2_max) + T_n(s2_max)   (band integral + high-s2 tail),
    Zcal     = sum_{n > n_max} I_n(inf) .

Beyond the computed waves I_n(inf) follows the ODD-FAMILY law in nu = n+1,

    I_n(inf) ~ a/nu^3 + b/nu^5 + c/nu^7 ,

because Born orders 1 and 2 are subtracted, the j-th Born order of the
s2-integrated trace starts at nu^(3-2j) (j = 3, 4, 5 -> nu^-3, nu^-5,
nu^-7), and its corrections come in relative powers 1/nu^2; the leading
coefficient has the analytic value A3 (third Born order).

1. ONE least-squares fit of (a, b, c) on the fit window [fit_lo, n_max] that
   the band driver placed (fit_lo = the sustained match of nu^3 I_n(inf)
   to A3; n_max = where the N-TAIL-BOUND watcher stopped) --
   pipeline_quadrature.odd_family_high_n_fit, the same function the band
   watchers call.  Three terms: dropping c/nu^7 would push the window's
   subleading curvature into the fitted (a, b) and through them into the
   summed tail.
2. The sum over ALL remaining waves in CLOSED FORM: nu runs over n_max+2,
   n_max+3, ... and sum 1/nu^p = zeta(p, n_max+2) (Hurwitz zeta), so

       Zcal = a zeta(3,n_max+2) + b zeta(5,n_max+2) + c zeta(7,n_max+2).

   Only this summed tail enters lnD_ren; the fitted (a, b, c) are
   window-dependent interpolation coefficients, never quoted as physics.

Because the fit acts on the s2-COMPLETED numbers I_n(inf), the region
n > n_max AND s2 > s2_max is inside the tail automatically.

GUARDS (abort-only; nothing they compute enters lnD_ren)
--------------------------------------------------------
N-WINDOW      the window must be contiguous and hold >= N_WINDOW_MIN_WAVES
              waves (3-parameter fit) -- inside odd_family_high_n_fit.
N-ASYMPTOTE   the fitted leading a must agree with the ANALYTIC third-Born
              moment A3 = (1/16) Int dr r^5 tr(U_int^3) to a3_tol: the
              check that the window sits on the I_n(inf) asymptote.
N-TAIL-BOUND  |b| zeta5 + |c| zeta7 <= eps_n at n_max: the fitted
              subleading part (the b and c terms) of the tail is negligible,
              so n_max is deep enough (same rule the watcher stopped on).
"""
import argparse
import os
import sys

import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline_helpers_vgeneral_phi_v2 import (add_standard_cli, atomic_savez,
                                 load_background, load_stage_npz,
                                 provenance_stamp, require_finite, stage_paths)
from pipeline_quadrature_vgeneral_phi_v2 import odd_family_high_n_fit
from watchers_guards_vgeneral_phi_v2 import (guard_n_asymptote,
                                          guard_n_tail_bound)


def main():
    ap = argparse.ArgumentParser(
        description='vgeneral_phi_v2 stage 4: the high-n tail of lnD_ren -- '
                    'one odd-family fit of the s2-completed I_n(inf) on the '
                    'fit window [fit_lo, n_max] + its Hurwitz-zeta sum over '
                    'n > n_max')
    add_standard_cli(ap)
    args = ap.parse_args()

    # the loaded background serves only the provenance stamp of the output
    # (m2, params, fields -> potential_id)
    background = load_background(args.bounce_npz)
    # ---- the fit window [fit_lo, n_max], the analytic guard object A3 and
    # the guard targets a3_tol, eps_n: ONE creation point (the band driver,
    # band_cutoffs_<tag>.npz), LOADED, never re-derived
    paths = stage_paths(args.data_dir, args.tag)
    band_cutoffs_npz = load_stage_npz(paths['band_cutoffs'], args.bounce_npz,
                            'band_adaptive_vgeneral_phi_v2')
    fit_lo, n_max = int(band_cutoffs_npz['fit_lo']), int(band_cutoffs_npz['n_max'])
    A3 = float(band_cutoffs_npz['A3'])
    a3_tol, eps_n = float(band_cutoffs_npz['a3_tol']), float(band_cutoffs_npz['eps_n'])
    print(f"[MOMENTS] A3 = {A3:+.5g}   (loaded from band_cutoffs -- its one "
          f"creation point)")

    # ---- the s2-completed per-wave numbers of band_integrals ------------
    band_integrals_npz = load_stage_npz(paths['band_integrals'], args.bounce_npz,
                        'band_integrals_vgeneral_phi_v2')
    band_n = np.asarray(band_integrals_npz['band_n'], int)
    I_n_inf = np.asarray(band_integrals_npz['I_n_inf'], float)
    require_finite(I_n_inf, '[ABORT] non-finite I_n(inf) in the band '
                            'integrals npz')

    # ---- the ONE fit + the Hurwitz-zeta sum (N-WINDOW gate inside the fit) --
    fit_a, fit_b, fit_c, Zcal, subleading_zeta_bound = odd_family_high_n_fit(
        band_n, I_n_inf, fit_lo, n_max)
    # A3 != 0: band_adaptive's MOMENT-A3 abort ran before any band existed
    a_over_A3_dev = abs(fit_a / A3 - 1.0)
    print(f"[TAIL-FIT] odd-family fit of I_n(inf) on n = [{fit_lo},{n_max}]: "
          f"a = {fit_a:+.5g} (A3 {A3:+.5g}, |a/A3 - 1| = {100*a_over_A3_dev:.1f}%)  "
          f"b = {fit_b:+.4g}  c = {fit_c:+.4g}")
    print(f"[TAIL] Zcal(n > {n_max}) = a zeta(3,{n_max+2}) + b zeta(5,{n_max+2})"
          f" + c zeta(7,{n_max+2}) = {Zcal:+.6f}")

    # ===== GUARD N-ASYMPTOTE (aborts; does NOT enter lnD) =====================
    guard_n_asymptote(fit_a, A3, tol=a3_tol)
    # ===== GUARD N-TAIL-BOUND (aborts; does NOT enter lnD) ====================
    print(f"[N-TAIL-BOUND] interpolation part |b|zeta5+|c|zeta7 = "
          f"{subleading_zeta_bound:.2e} (limit {eps_n:g})")
    guard_n_tail_bound(subleading_zeta_bound, eps_n)
    # ==========================================================================

    atomic_savez(paths['tail_high_n'],
             fit_a=fit_a, fit_b=fit_b, fit_c=fit_c, Zcal=Zcal,
             fit_lo=int(fit_lo), n_max=int(n_max),
             A3=A3,
             a_over_A3_dev=a_over_A3_dev, subleading_zeta_bound=subleading_zeta_bound,
             a3_tol=a3_tol, eps_n=eps_n,
             **provenance_stamp(args.bounce_npz, background['m2'],
                                background['params'], background['fields']))
    print(f"[TAIL] SUMMARY: Zcal(n > {n_max}) = {Zcal:+.6f} enters lnD_ren "
          f"as the closed-form sum of the fitted I_n(inf) beyond n_max.")
    print(f"[OK] wrote {paths['tail_high_n']}")


if __name__ == '__main__':
    main()
