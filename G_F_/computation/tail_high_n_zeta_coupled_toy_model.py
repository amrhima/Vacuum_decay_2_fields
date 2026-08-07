#!/usr/bin/env python3
"""tail_high_n_zeta_coupled_toy_model.py -- stage 5: the high-n (deep-wave) tail of the
partial-wave sum, n > n_max.

WHAT IT COMPUTES
----------------
The completed per-wave integrals I_n(inf) fall on the ODD asymptotic
family in nu = n+1 (notes/F2_T0.pdf eq. 5):

    I_n(inf)  ~  a / nu^3  +  c / nu^5  +  e / nu^7

1. Least-squares fit of (a, c, e) on the window [fit_lo, n_max] that the
   N-ONSET watcher placed (fit_lo = the sustained match of nu^3 I_n(inf) to
   the analytic A3; n_max = where the N-TAIL-BOUND watcher stopped).  Three
   terms: the two-term fit leaves a 2.9e-2 (bdet) / 4.8e-3 (F2) tail error
   at the same window where three terms leave ~1e-4 (measured; F2_T0.pdf
   Part III) -- the e/nu^7 term stays.
2. The tail over ALL remaining waves in CLOSED FORM via Hurwitz zeta:
       tail = a zeta(3,n_max+2) + c zeta(5,n_max+2) + e zeta(7,n_max+2).
   Only this summed tail enters lnD_ren; the fitted (a, c, e) are
   window-dependent interpolation coefficients.

GUARDS IN THIS STAGE (abort-only; nothing they compute enters lnD_ren)
----------------------------------------------------------------------
N-WINDOW      the window must be contiguous and hold >= 12 waves
              (3-parameter fit) -- inside pipeline_quadrature.odd_tail_fit.
N-ASYMPTOTE   the fitted leading a must agree with the ANALYTIC third-Born
              moment A3 = (1/16) Int dr r^5 tr(DW^3) to --a3-tol (10%): the
              certificate that the window sits on the I_n(inf) asymptote.
N-TAIL-BOUND  |c| zeta5 + |e| zeta7 <= --eps-n at n_max: the certificate
              that the interpolation part of the tail is negligible, so
              n_max is deep enough (same rule the watcher stopped on).

INPUTS   band_integrals_<tag>.npz   (stage 4: band_n, I_inf)
         --bounce-npz                        (tr(DW^3) for the analytic A3)
OUTPUT   tail_highn_<tag>.npz with keys
    fit_a, fit_c, fit_e  ()  odd-family lstsq coefficients (nu^-3, -5, -7)
    tail            ()    Hurwitz-zeta sum over n > n_max (enters lnD_ren)
    fit_window      (2,)  [fit_lo, n_max] (int)
    A3_analytic     ()    (1/16) Int dr r^5 tr(DW^3)
    baacke_c        ()    -(1/48) Int dr r^3 tr(DW^3) (reference echo)
    a_over_A3_dev   ()    |a/A3 - 1| (the N-ASYMPTOTE guard value)
    sub_bound       ()    |c| zeta5 + |e| zeta7 (the N-TAIL-BOUND guard value)
    a3_tol, eps_n   ()    the gates they were checked against
    bounce_sha, potential_id, code_version    metadata
"""
import argparse
import os
import sys
import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline_helpers_coupled_toy_model import (add_standard_cli, atomic_savez,        # noqa: E402
                                 load_bounce, provenance_stamp, require_finite,
                                 stage_paths)
from pipeline_quadrature_coupled_toy_model import odd_tail_fit                  # noqa: E402
from watchers_guards_coupled_toy_model import (analytic_moments, guard_n_asymptote,  # noqa: E402
                                          guard_n_tail_bound)


def main():
    ap = argparse.ArgumentParser(
        description='coupled_toy_model stage 5: high-n odd-family zeta tail')
    add_standard_cli(ap)
    ap.add_argument('--fit-window', type=int, nargs=2, required=True,
                    metavar=('FIT_LO', 'N_MAX'),
                    help='[fit_lo, n_max] from band_cutoffs_<tag> (the '
                         'N-ONSET and N-TAIL-BOUND watchers chose them; no '
                         'default -- nothing model-dependent is hardcoded)')
    ap.add_argument('--a3-tol', type=float, default=0.10,
                    help='N-ASYMPTOTE guard: max fractional deviation of the '
                         'fitted a from the analytic A3')
    ap.add_argument('--eps-n', type=float, default=1e-2,
                    help='N-TAIL-BOUND guard: ceiling on the interpolation '
                         'part of the zeta tail (same target the watcher '
                         'stopped on)')
    args = ap.parse_args()
    fit_lo, n_max = args.fit_window

    # ---- analytic guard objects from the bounce ------------------------------
    bg = load_bounce(args.bounce_npz)
    mom = analytic_moments(bg['R'], bg['trDW3'], bg['Phi'], bg['m2'])
    A3, baacke_c = mom['A3'], mom['c']
    print(f"[MOMENTS] analytic A3 = {A3:+.5g}   Baacke c = {baacke_c:+.5g}   "
          f"mbar2 = {bg['mbar2']:.5g}   mu = sum_i m_i = {bg['mu']:.6f}")

    # ---- completed per-wave integrals from stage 4 ---------------------------
    paths = stage_paths(args.data_dir, args.tag)
    bi = paths['band_integrals']
    if not os.path.isfile(bi):
        raise RuntimeError(f'[ABORT] stage-4 output missing: {bi} -- run '
                           f'tail_s2_completion_coupled_toy_model first.')
    d = np.load(bi, allow_pickle=True)
    ns = np.asarray(d['band_n'])
    I_inf = np.asarray(d['I_inf'], float)
    require_finite(I_inf, '[ABORT] non-finite I_n(inf) in the band '
                          'integrals npz')

    # ---- the odd-family fit + zeta tail (N-WINDOW gate inside odd_tail_fit) --
    a, c, e, tail, sub = odd_tail_fit(ns, I_inf, fit_lo, n_max)
    dev = abs(a / A3 - 1.0) if A3 != 0.0 else float('nan')
    print(f"[TAIL-FIT] odd family on [{fit_lo},{n_max}]: a={a:+.5g} "
          f"(A3 {A3:+.5g}, {100*dev:+.1f}%)  c={c:+.4g}  e={e:+.4g}  "
          f"tail(n>{n_max}) = {tail:+.4f}")

    # ===== GUARD N-ASYMPTOTE (aborts; does NOT enter lnD) =====================
    guard_n_asymptote(a, A3, tol=args.a3_tol)
    # ===== GUARD N-TAIL-BOUND (aborts; does NOT enter lnD) ====================
    print(f"[N-TAIL-BOUND] interpolation part of the tail |c|zeta5+|e|zeta7 "
          f"= {sub:.2e} (limit {args.eps_n:g})")
    guard_n_tail_bound(sub, args.eps_n)
    # ==========================================================================

    out = paths['tail_highn']
    atomic_savez(out,
             fit_a=a, fit_c=c, fit_e=e, tail=tail,
             fit_window=np.asarray(args.fit_window, int),
             fit_lo=int(fit_lo), n_max=int(n_max),
             A3_analytic=A3, baacke_c=baacke_c,
             a_over_A3_dev=dev,
             sub_bound=sub, a3_tol=float(args.a3_tol),
             eps_n=float(args.eps_n),
             **provenance_stamp(args.bounce_npz, bg['m2']))
    print(f"[TAIL] SUMMARY: fit window n = {fit_lo}..{n_max}; "
          f"tail(n > {n_max}) = {tail:+.6f} (a within {100*dev:.1f}% of the "
          f"analytic A3)")
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
