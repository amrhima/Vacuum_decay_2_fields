#!/usr/bin/env python3
"""gbar_n0_coupled_toy_model.py -- the n = 0 sector (stage 3a): continuum RESOLVENT
s2-integral of the negative-mode partial wave.

METHOD MANDATE (user, standing): production = RESOLVENT (Green's-function)
method ONLY; the fixed-s2 determinant method is a CROSSCHECK, never the
production path.  Carosi (arXiv:2601.08686): the two are the same object.
ONE method per code: the negative-mode EIGENVALUE is located in its own
stage (eigenvalue_neg_coupled_toy_model); this file is PURE RK resolvent -- it READS
the eigenvalue and does only the pole subtraction + log-Simpson s2 integral.

WHAT IT COMPUTES (Carosi Eq. 5.10 bookkeeping, n = 0, deg = 1)
--------------------------------------------------------------
1. delta_0(s2): the Born-subtracted coincident trace difference (all orders
   >= 3) of the SAME third-order-Born h-ODE the n >= 2 residual band
   integrates (delta_g_bar_greater_equal_3_coupled_toy_model.eval_point; the false-vacuum trace
   enters ONLY in its analytic Bessel form), evaluated on the dedicated
   geometric resolvent s2 grid [--sector-s2-lo, --s2-sector-max]
   (pipeline_quadrature_coupled_toy_model.adaptive_sector_integral: pole-aware, odd node
   count so the [::2] half grid shares both endpoints, node count DOUBLED
   from --sector-nodes until the SECTOR-STEP meter passes).
2. Pole subtraction + s2 quadrature (pipeline_quadrature_coupled_toy_model.adaptive_
   sector_integral): the negative-mode eigenvalue lam_neg = w_neg2 is READ
   from the eig_n0 npz (stage 0a), then delta_0 -> delta_0 - 1/(lam_neg +
   s2), then the SAME log-Simpson scheme as the band (Simpson per side if
   the grid is pole-clustered), with the SECTOR-STEP every-2nd-point
   convergence meter (limit --sector-conv-tol):
       I8res_0 = -Int_0^{s2_sector_max} [delta_0(s2) - 1/(lam_neg+s2)] ds2 .

INPUTS   --bounce-npz (R, X'/Y', params, false_vac) + the injected
         potential module + the eig_n0 npz (--eig-npz; run
         eigenvalue_neg_coupled_toy_model first).
OUTPUT   sector_n0_<tag>.npz with keys (N = final adaptive node count)
    s2_sector      (N,)    the geometric resolvent s2 grid
    delta_raw      (N,)    raw delta_0(s2)
    delta_sub      (N,)    pole-subtracted delta_0(s2) - 1/(lam_neg+s2)
    lam_neg_cont   ()      the crossing eigenvalue (= w_neg2, brentq
                           det-crossing root; passed through from eig_n0)
    crossing_resid ()      |det(1+M)(-w_neg2)| det-residual guard value
                           (diagnostic, from eig_n0)
    I8res          ()      the n=0 resolvent integral (enters lnD_ren)
    conv_meter     ()      |I(full) - I(every-2nd)| (SECTOR-STEP value)
    conv_tol       ()      the limit it was checked against
    deg            ()      1.0 (n=0 degeneracy (n+1)^2)
    s2_sector_max, sector_s2_lo, sector_nodes, runtime_s   grid/provenance
    bounce_sha, potential_id, code_version                 metadata
"""
import argparse
import os
import sys
import time

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from delta_g_bar_greater_equal_3_coupled_toy_model import ResidualBand                          # noqa: E402
from pipeline_helpers_coupled_toy_model import (add_standard_cli, atomic_savez,         # noqa: E402
                                 load_bounce, load_eig_npz,
                                 provenance_stamp, stage_paths)
from pipeline_quadrature_coupled_toy_model import adaptive_sector_integral    # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description='coupled_toy_model stage 3a: n=0 sector (continuum resolvent)')
    add_standard_cli(ap)
    ap.add_argument('--eig-npz', default=None,
                    help='eig_n0 npz with the negative-mode eigenvalue '
                         '(default: <data-dir>/eig_n0_<tag>.npz; '
                         'run eigenvalue_neg_coupled_toy_model first)')
    ap.add_argument('--s2-sector-max', type=float, default=300.0,
                    help='upper end of the resolvent sector s2 grid (the '
                         'closed-form T_sec8 exact-log tail carries the '
                         'sector integrals beyond it)')
    ap.add_argument('--sector-s2-lo', type=float, default=1e-4,
                    help='low end of the PRODUCTION resolvent sector s2 grid '
                         '(geometric to s2_sec_max; the [0, s2_lo] sliver is the '
                         'log-Simpson rectangle of the SUBTRACTED smooth '
                         'integrand)')
    ap.add_argument('--sector-nodes', type=int, default=321,
                    help='STARTING node count of the geometric resolvent '
                         'sector s2 grid (min 61; forced odd so the '
                         'every-2nd-point half grid shares both endpoints); '
                         'adaptive_sector_integral doubles it until the '
                         'SECTOR-STEP meter passes')
    ap.add_argument('--sector-conv-tol', type=float, default=1e-3,
                    help='SECTOR-STEP limit: |I(full grid) - I(every-2nd-'
                         'point grid)| of the resolvent s2 integral; nodes '
                         'are doubled until the meter passes it')
    args = ap.parse_args()

    bg = load_bounce(args.bounce_npz)
    band = ResidualBand(bg['R'], bg['DW'], bg['m2'])   # DEFAULT tolerances
    s2_sec_max = float(args.s2_sector_max)

    # ---- READ the negative-mode eigenvalue from the eig_n0 npz -------------
    # eig_n0 is located in stage 0a, BEFORE the band, so its det-crossing SCAN
    # range (the stored s2_sector_max = the guard scan S2_GUARD) legitimately
    # DIFFERS from this sector integration cutoff s2_sec_max (the band-certified
    # S2_SEC_MAX).  That is fine and requires no cutoff-equality check: the
    # eigenvalue is the FIRST det(1+M) zero (a root -- independent of the scan
    # range that bracketed it), load_eig_npz already aborts on a bounce_sha
    # mismatch (so this eig is for THIS bounce), and eigenvalue_neg's
    # EIG-RESIDUAL guard already certified the root.  We read w_neg2 and use
    # THIS stage's --s2-sector-max (the band cutoff) for the integral.
    eig = load_eig_npz(args.eig_npz, args.data_dir, args.tag, 'eig_n0',
                       args.bounce_npz, 'eigenvalue_neg_coupled_toy_model')
    lam_neg_cont = float(eig['lam_neg_cont'])          # = w_neg2 (brentq root)
    crossing_resid = float(eig['crossing_resid'])
    print(f"       lam_neg_cont (READ from eig_n0) = {lam_neg_cont:+.10e} "
          f"(det residual {crossing_resid:.2e})")

    # ---- PRODUCTION resolvent integral --------------------------------------
    t_sec0 = time.time()
    s2_sec, delta_raw, delta_sub, I8res, conv = adaptive_sector_integral(
        band, 0, lam_neg_cont, args.sector_s2_lo, s2_sec_max,
        args.sector_nodes, args.sector_conv_tol)
    runtime_s = time.time() - t_sec0

    out = stage_paths(args.data_dir, args.tag)['sector_n0']
    atomic_savez(out,
             s2_sector=s2_sec,
             delta_raw=delta_raw, delta_sub=delta_sub,
             lam_neg_cont=lam_neg_cont,
             crossing_resid=crossing_resid,
             I8res=I8res,
             conv_meter=conv, conv_tol=float(args.sector_conv_tol),
             deg=1.0,  # (0+1)^2 = 1 partial-wave degeneracy
             s2_sector_max=s2_sec_max,
             sector_s2_lo=float(args.sector_s2_lo),
             sector_nodes=int(len(s2_sec)),
             runtime_s=runtime_s,
             **provenance_stamp(args.bounce_npz, bg['m2']))
    print(f"[SECTOR n=0] SUMMARY: geometric grid [{args.sector_s2_lo:g}, "
          f"{s2_sec_max:g}], {len(s2_sec)} nodes; runtime {runtime_s:.1f} s")
    print(f"       lam_neg_cont = {lam_neg_cont:+.10e} (pole-subtraction "
          f"eigenvalue, continuum crossing; READ from eig_n0)")
    print(f"       I8res_0 = {I8res:+.6f} (SECTOR-STEP meter {conv:.2e}; "
          f"limit {args.sector_conv_tol:g})")
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
