#!/usr/bin/env python3
"""tail_s2_completion_coupled_toy_model.py -- stage 4: per-wave band s2 integrals
I_n(L2) + the analytic high-s2 completion T_n, for every band wave
n = 2..n_max.

WHAT IT COMPUTES
----------------
For each partial wave n (nu = n+1) it reads delta_n(s2) (all Born orders
>= 3, deg-weighted) from the residual_band slice npz set and evaluates

1. I_n(L2) = -Int_0^{L2} delta_n(s2) ds2 with the production W3_B
   log-Simpson scheme (pipeline_quadrature.s2_integral): log-Simpson on
   [s2_min, 50] + linear Simpson on [50, L2] + the [0, s2_min] rectangle
   sliver.  L2 = Lam2 = the last band grid value (the grid MUST end at
   --s2-max, else abort -- one uniform cutoff for the whole band).

2. The Method-I calibrated per-wave completion:
       T_n = -delta_n(Lam2) * R_n,
       R_n = s2_tail_shape_ratio(nu, Lam2, mbar2, R, trDW3)
   (pipeline_quadrature) -- the EXACT s2-integral of the ASSUMED uniform
   O(U^3) tail shape from Lam2 to infinity, normalised to that shape's value
   at Lam2.  Then  I_n(inf) = I_n(Lam2) + T_n .  "Exact" qualifies the
   integral of the assumed shape, NOT the claim that the true determinant
   tail has that shape: the shape uses a SINGLE mbar2 = min(m_i^2), so for
   unequal-mass coupled N it is an equal-mass/leading calibration whose
   residual is bounded by the S2-BAACKE cutoff certificate (eps_s2), not an
   exact per-channel unequal-mass completion.

GUARDS IN THIS STAGE (abort-only; nothing they compute enters lnD_ren)
----------------------------------------------------------------------
MANIFEST   the globbed slice set must equal band_manifest_<tag>.json (the
           adaptive driver wrote it); a foreign same-tag slice aborts.
S2-STEP    per-wave grid-halving quadrature meter: the same integral on
           every 2nd grid point must agree to --quad-meter-tol relative --
           certifies the s2 STEP SIZE (not the cutoff).
S2-BAACKE  the s2 CUTOFF certificate (watchers_guards_coupled_toy_model.guard_s2_baacke):
           re-measures, on the delivered tables, exactly what the
           S2-PLATEAU watcher stopped on -- the raw band truncation must
           follow the Baacke law c/(Lam2+mbar2) (eq. 6, analytic c from
           this bounce) and the predicted remaining error shape_miss x
           envelope must be <= --eps-s2.

INPUTS   residual_band_<tag>_s2*.npz  (base slices + the driver's
                                       _n<lo>to<hi> wave-extension files,
                                       merged by read_band_tables)
         band_manifest_<tag>.json     (the exact expected slice set)
         --bounce-npz                 (mbar2 + tr(DW^3) for R_n and the
                                       S2-BAACKE analytic c)
OUTPUT   band_integrals_<tag>.npz with keys
    band_n     (n_max-1,)  n = 2..n_max
    s2_grid    (nslices,)  sorted slice s2 values;  Lam2 () = s2_grid[-1]
    I_data     (n_max-1,)  I_n(Lam2), W3_B log-Simpson
    T, R_shape (n_max-1,)  completion T_n and the shape ratio R_n
    I_inf      (n_max-1,)  I_n(inf) = I_data + T
    qmeter     (n_max-1,)  S2-STEP per-wave halving meter;  qmeter_max ()
    guard_E_pred, guard_c_eff_ratio   the S2-BAACKE guard measurements
    s2_max, n_max, completion(='mi-calibrated')   config echo
    bounce_sha, potential_id, code_version        metadata
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
                                 bounce_sha256, load_bounce, provenance_stamp,
                                 read_band_tables, stage_paths)
from pipeline_quadrature_coupled_toy_model import s2_integral, s2_tail_shape_ratio  # noqa: E402
from watchers_guards_coupled_toy_model import (analytic_moments, band_candidate_sums,  # noqa: E402
                               guard_s2_baacke, watch_s2_plateau)


def main():
    ap = argparse.ArgumentParser(
        description='coupled_toy_model stage 4: band s2 integrals + Method-I completion')
    add_standard_cli(ap)
    ap.add_argument('--s2-max', type=float, required=True,
                    help='band cutoff Lam2 (from band_cutoffs_<tag>: the '
                         'S2-PLATEAU watcher chose it; no default -- nothing '
                         'model-dependent is hardcoded)')
    ap.add_argument('--n-max', type=int, required=True,
                    help='band top wave (from band_cutoffs_<tag>: the '
                         'N-TAIL-BOUND watcher chose it; no default)')
    ap.add_argument('--eps-s2', type=float, default=1e-2,
                    help='S2-BAACKE guard: ceiling on the predicted remaining '
                         's2-truncation error (same target the S2-PLATEAU '
                         'watcher stopped on)')
    ap.add_argument('--quad-meter-tol', type=float, default=5e-3,
                    help='S2-STEP guard: max relative |I_half-I_full| of the '
                         'grid-halving quadrature meter (loosen ONLY for '
                         'coarse smoke grids)')
    ap.add_argument('--band-manifest', default=None,
                    help='band slice manifest json (default: '
                         '<data-dir>/band_manifest_<tag>.json); the globbed '
                         'slice set must equal it exactly')
    args = ap.parse_args()

    # ---- bounce background: mbar2 + tr(DW^3) for R_n and the guard ----------
    bg = load_bounce(args.bounce_npz)
    R, trDW3, mbar2 = bg['R'], bg['trDW3'], bg['mbar2']

    # ---- band tables (MANIFEST guard inside read_band_tables) ---------------
    manifest = args.band_manifest or os.path.join(
        args.data_dir, f'band_manifest_{args.tag}.json')
    if not os.path.isfile(manifest):
        raise RuntimeError(f'[ABORT MANIFEST] {manifest} not found -- run the '
                           f'adaptive band driver (band_adaptive_coupled_toy_model) '
                           f'first; this stage refuses an unmanifested glob.')
    s2s, tables, files = read_band_tables(args.data_dir, args.tag,
                                          expect_sha=bounce_sha256(args.bounce_npz),
                                          manifest_path=manifest,
                                          s2_cap=args.s2_max)
    print(f'[MANIFEST] OK: {len(files)} slice files match '
          f'{os.path.basename(manifest)}')
    if abs(s2s[-1] - args.s2_max) > 1e-6:
        raise RuntimeError(f'[ABORT S2-STEP] band s2 coverage ends at '
                           f'{s2s[-1]}, expected s2_max={args.s2_max} '
                           f'(one uniform cutoff for the whole band).')
    Lam2 = float(s2s[-1])          # completion cutoff = last s2 grid value

    # the band must cover n = 2..n_max completely (sectors own n = 0, 1)
    ns = np.arange(2, args.n_max + 1)
    for s2 in s2s:
        missing = [int(n) for n in ns if int(n) not in tables[s2]]
        if missing:
            raise RuntimeError(f'[ABORT] slice s2={s2:g} is missing waves '
                               f'{missing[:6]}{"..." if len(missing) > 6 else ""} '
                               f'(fail-loud; rerun the adaptive driver).')
    D = np.vstack([[tables[s2][int(n)] for s2 in s2s] for n in ns])

    # every-2nd-point sub-grid for the S2-STEP grid-halving meter (endpoint
    # forced in so I_half and I_full cover the same [s2_min, Lam2] interval)
    half_idx = np.unique(np.r_[np.arange(0, len(s2s), 2), len(s2s) - 1])

    I_data = np.empty(len(ns)); I_inf = np.empty(len(ns))
    T_arr = np.empty(len(ns)); R_arr = np.empty(len(ns))
    qmeter = np.empty(len(ns))
    for k, n in enumerate(ns):
        nu = n + 1.0
        dvals = D[k]
        # LOG-SIMPSON band scheme (W3_B): log-Simpson [s2_min,50] + linear
        # Simpson [50,Lam2] + rectangle sliver for the omitted [0,s2_min]
        I_data[k] = s2_integral(s2s, dvals)

        # ---- GUARD S2-STEP (aborts; does NOT enter lnD): step-size meter ----
        I_half = s2_integral(s2s[half_idx], dvals[half_idx])
        qmeter[k] = abs(I_half - I_data[k]) / max(abs(I_data[k]), 1e-12)
        if qmeter[k] > args.quad_meter_tol:
            raise RuntimeError(f'[ABORT S2-STEP] grid-halving meter '
                               f'|I_half-I_full|={abs(I_half-I_data[k]):.2e} > '
                               f'{args.quad_meter_tol:g}*|I_full| '
                               f'at n={n}: refine the s2 grid.')

        # Method-I calibrated uniform shape: T_n = -delta_n(Lam2) * R_n
        R_arr[k] = s2_tail_shape_ratio(nu, Lam2, mbar2, R, trDW3)
        T_arr[k] = -dvals[-1] * R_arr[k]
        I_inf[k] = I_data[k] + T_arr[k]
    qmeter_max = float(np.max(qmeter))
    print(f"[S2-STEP] grid-halving quadrature meter: max q_n = {qmeter_max:.2e} "
          f"(limit {args.quad_meter_tol:g})")

    # ===== GUARD S2-BAACKE (aborts; does NOT enter lnD) ======================
    # The s2 CUTOFF certificate: on the delivered tables, re-measure exactly
    # what the S2-PLATEAU watcher stopped on -- the raw band truncation must
    # follow the analytic Baacke law and the predicted remaining error must
    # be inside --eps-s2 (watchers_guards_coupled_toy_model holds the one implementation).
    mom = analytic_moments(R, trDW3, bg['Phi'], bg['m2'])
    A3, c_analytic = mom['A3'], mom['c']     # A3 printed for reference
    L2c, Braw, Bc, _ = band_candidate_sums(s2s, D, ns, mbar2, R, trDW3)
    meas = watch_s2_plateau(L2c, Bc, Braw, c_analytic, mbar2)
    print(f"[S2-BAACKE] raw truncation vs law c/(Lam2+mbar2): c_eff/c = "
          f"{meas['c_eff_ratio']:.3f}; predicted remaining error "
          f"{meas['shape_miss']:.2e} x {meas['envelope']:.3f} = "
          f"{meas['E_pred']:.2e} (limit {args.eps_s2:g})")
    guard_s2_baacke(meas, args.eps_s2)
    # ==========================================================================

    out = stage_paths(args.data_dir, args.tag)['band_integrals']
    atomic_savez(out,
             band_n=ns, s2_grid=s2s, Lam2=Lam2,
             I_data=I_data, T=T_arr, R_shape=R_arr, I_inf=I_inf,
             qmeter=qmeter, qmeter_max=qmeter_max,
             guard_E_pred=meas['E_pred'],
             guard_c_eff_ratio=meas['c_eff_ratio'],
             s2_max=args.s2_max, n_max=int(args.n_max),
             completion='mi-calibrated',
             **provenance_stamp(args.bounce_npz, bg['m2']))
    print(f"[BAND] SUMMARY: {len(ns)} waves (n = {ns[0]}..{ns[-1]}), "
          f"{len(s2s)} s2 slices to Lam2 = {Lam2:g}; sum I_n(inf) over the "
          f"band = {float(np.sum(I_inf)):+.6f}   (A3={A3:+.5g})")
    print(f"[OK] wrote {out}")


if __name__ == '__main__':
    main()
