#!/usr/bin/env python3
"""band_integrals_vgeneral_phi_v2.py -- level 3, stage 3: for every band wave
n = 2..n_max the s2 integral and its high-s2 tail (HOW 2),
    I_n(s2_max) = -Int_0^{s2_max} delta_geq3(n, s2) ds2,
    T_n(s2_max) = -Int_{s2_max}^inf delta_geq3(n, s2) ds2,
    I_n(inf)    = I_n(s2_max) + T_n(s2_max).
Summed over n, I_n(s2_max) and T_n(s2_max) are the terms (T3) and (T5) of
lnD_ren (stage 7); the sequence I_n(inf) is what the high-n tail (stage 4)
is fitted on.

READS   residual_band_<tag>_s2*.npz  the stage-1 slices (base slices + the
                                     driver's _n<lo>to<hi> wave-extension
                                     files, merged by read_band_tables):
                                     delta_geq3(n, s2), the engine's stored
                                     trace of all Born orders >= 3,
                                     deg-weighted
        band_manifest_<tag>.json     the exact expected slice set (stage 1)
        band_cutoffs_<tag>.npz       (stage 1) the certified cutoffs n_max,
                                     s2_max and the S2-BAACKE inputs baacke_c,
                                     baacke_c_scale, eps_s2
        --bounce-npz                 through load_background: R, m_min2,
                                     tr(U_int^3) for the high-s2 tail and the
                                     Baacke law; params, fields, m2 for the
                                     provenance (slice check, output stamp)
WRITES  band_integrals_<tag>.npz (stage_paths key 'band_integrals') with keys
    band_n     (n_max-1,)  n = 2..n_max (stages 4 and 7)
    s2_values  (nslices,)  sorted slice s2 values (the band s2 lattice;
                           record only, no reader)
    s2_max     ()          the band cutoff = s2_values[-1] (stage 7, the plot
                           layer, the driver's skip-guard)
    I_n_s2_max (n_max-1,)  I_n(s2_max), log-Simpson (stage 7, term (T3);
                           copied into the lnD_ren npz by stage 7, where
                           the lnD_ren part figures re-check its sum over n)
    T_n_s2_max (n_max-1,)  T_n(s2_max), the high-s2 tail (stage 7, term
                           (T5); copied into the lnD_ren npz by stage 7)
    I_n_inf    (n_max-1,)  I_n(inf) = I_n_s2_max + T_n_s2_max (stage 4
                           fits it; stage 7)
    s2_step_meter (n_max-1,)  S2-STEP per-wave halving meter
                              |I_n_half_grid - I_n_s2_max| / max_n |I_n_s2_max|:
                              each wave's change in units of the band's
                              LARGEST |I_n| (record only, no reader);
    s2_step_meter_max ()      its maximum over the waves (the S2-STEP guard
                              value; record only, no reader)
    guard_s2_trunc_err_pred, guard_baacke_c_miss   the S2-BAACKE guard
                              measurements (the predicted error and
                              |baacke_c_measured - baacke_c| / baacke_c_scale;
                              record only, no reader)
    eps_s2, s2_step_tol       the guard limits (eps_s2 echoed from
                              band_cutoffs; both value-matched by the driver's
                              stage-3 skip-guard)
    n_max      ()          the band's last wave (stage 7, the plot layer, the
                           driver's skip-guard)
    high_s2_tail_method    'trU_int3-shape-anchored', the name of the tail rule
                           (stage 7 prints it and copies it into lnD_ren)
    bounce_sha, potential_id, code_version        metadata

HOW
---
For each partial wave n (nu = n+1) the stored delta_geq3(n, s2) of the
slice set is integrated and completed, all waves at once:

1. I_n(s2_max) = -Int_0^{s2_max} delta_geq3(n, s2) ds2 with the production
   log-Simpson scheme (pipeline_quadrature_vgeneral_phi_v2.I_n_from_delta,
   called by band_candidate_sums of the same module -- its
   full-cutoff candidate IS this stage's production computation, the
   trailing candidates below it feed the S2-BAACKE guard):
   Simpson in ln s2 on [s2_min, s2_max] + the [0, s2_min] strip closed by
   the quadratic through the three lowest nodes.
   s2_max = the last band grid value, the largest s2 with data (the grid MUST
   end at the band_cutoffs s2_max, else abort -- one uniform cutoff for the
   whole band).

2. The high-s2 tail (pipeline_quadrature high_s2_tail): above
   s2_max each wave keeps its computed value at s2_max and takes the
   s2-dependence of its leading large-s2 term (third Born order),
       delta_geq3(n, s2) ~ delta_geq3(n, s2_max) shape(s2) / shape(s2_max),
       shape(s2) = Int dr r^7 tr_U_int3 chi^-7,
       chi = sqrt(nu^2 + (s2 + m_min2) r^2),
   whose s2-integral is exact:
       T_n(s2_max) = -delta_geq3(n, s2_max) * shape_ratio,
       shape_ratio = (2/5) Int dr r^5 tr_U_int3 chi^-5
                     / Int dr r^7 tr_U_int3 chi^-7    (chi at s2 = s2_max).
   The shape carries ONE mass, m_min2 = min(m_i^2), for every channel; the
   error this leaves falls as s2_max grows; the S2-BAACKE guard estimates
   it, with every other cutoff effect, from the drift of the completed band
   sum over the candidate cutoffs and requires the estimate <= eps_s2.

   The high-s2 tail acts ONLY on the Born-order >= 3 traces read here.  The
   Born-order-1 add-back's own high-s2 part is the assembler's closed form
   G1_addback_high_s2 (term (T4); (T2) + (T4) = A1_fin) and is never
   shape-completed.

GUARDS IN THIS STAGE (abort-only, except SHAPE-RATIO, which corrects: its
repaired high-s2 tail enters lnD; each guard is explained where it runs)
----------------------------------------------------------------------
MANIFEST      the slice files must be exactly those of band_manifest_<tag>.json
S2-COVERAGE   the band grid must end at the certified s2_max of band_cutoffs
N-COVERAGE    every slice must hold every band wave n = 2..n_max
S2-STEP       the s2 STEP SIZE: the integrals on every 2nd grid node must agree
              to --s2-step-tol, relative to the band's largest |I_n|
SHAPE-RATIO   (inside high_s2_tail) a wave whose tr(U_int^3) denominator
              moment cancels gets the analytic amplitude of the same shape
REPAIR-SHARE  the SHAPE-RATIO-repaired waves must carry <= 5% of the
              tail sum (the repair is safe only while it sets little)
S2-BAACKE     the s2 CUTOFF (guard_s2_baacke): the raw band truncation follows
              the Baacke law baacke_c/(s2_max + m_min2) and the predicted
              remaining error is <= eps_s2 -- what the S2-PLATEAU watcher
              stopped on,
              re-measured on the delivered tables
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
                                 bounce_fingerprint, load_background,
                                 load_stage_npz, provenance_stamp,
                                 read_band_tables, stage_paths)
from pipeline_quadrature_vgeneral_phi_v2 import (I_n_from_delta,
                                                 band_candidate_sums)
from watchers_guards_vgeneral_phi_v2 import (guard_s2_baacke, watch_s2_plateau,
                               BAACKE_MISS_TOL)


def main():
    ap = argparse.ArgumentParser(
        description='vgeneral_phi_v2 stage 3: band s2 integrals + trU_int3-shape high-s2 tail')
    add_standard_cli(ap)
    ap.add_argument('--s2-step-tol', type=float, default=5e-3,
                    help='S2-STEP guard: max |I_n_half_grid-I_n_s2_max| / max_n |I_n_s2_max| of the '
                         'grid-halving quadrature meter (loosen ONLY for '
                         'coarse smoke grids)')
    args = ap.parse_args()

    # ---- the inputs from stage 1 (band_cutoffs): the certified cutoffs n_max,
    # s2_max and the S2-BAACKE inputs baacke_c, baacke_c_scale, eps_s2 ---------
    paths = stage_paths(args.data_dir, args.tag)
    band_cutoffs_npz = load_stage_npz(paths['band_cutoffs'], args.bounce_npz,
                            'band_adaptive_vgeneral_phi_v2')
    n_max, s2_max = int(band_cutoffs_npz['n_max']), float(band_cutoffs_npz['s2_max'])
    baacke_c, eps_s2 = float(band_cutoffs_npz['baacke_c']), float(band_cutoffs_npz['eps_s2'])
    baacke_c_scale = float(band_cutoffs_npz['baacke_c_scale'])

    # ---- bounce background: R, m_min2, tr(U_int^3) for the high-s2 tail's shape
    # and the Baacke law ---------------------------------------------------------
    background = load_background(args.bounce_npz)
    R, tr_U_int3, m_min2 = (background['R'], background['tr_U_int3'],
                          background['m_min2'])

    # ---- band tables (MANIFEST guard inside read_band_tables) ---------------
    manifest = os.path.join(args.data_dir, f'band_manifest_{args.tag}.json')
    if not os.path.isfile(manifest):
        raise RuntimeError(f'[ABORT MANIFEST] {manifest} not found -- run the '
                           f'adaptive band driver (band_adaptive_vgeneral_phi_v2) '
                           f'first; this stage refuses an unmanifested glob.')
    s2_values, tables, files = read_band_tables(
        args.data_dir, args.tag,
        expect_bounce_fingerprint=bounce_fingerprint(args.bounce_npz),
        manifest_path=manifest, s2_cap=s2_max,
        params=background['params'], fields=background['fields'])
    print(f'[MANIFEST] OK: {len(files)} slice files match '
          f'{os.path.basename(manifest)}')
    # the delivered grid must end at the certified cutoff s2_max (band_cutoffs,
    # its one creation point).  Relative tolerance (s2 carries mass^2 units):
    # 1e-9 is far below any grid spacing and far above float rounding.
    if abs(s2_values[-1] - s2_max) > 1e-9 * s2_max:
        raise RuntimeError(f'[ABORT S2-COVERAGE] band s2 coverage ends at '
                           f'{s2_values[-1]}, expected s2_max={s2_max} '
                           f'(one uniform cutoff for the whole band).')

    # the band wave list band_n = 2..n_max (sectors own n = 0, 1): every
    # slice must cover it completely
    band_n = np.arange(2, n_max + 1)
    for s2 in s2_values:
        missing = [int(n) for n in band_n if int(n) not in tables[s2]]
        if missing:
            raise RuntimeError(f'[ABORT N-COVERAGE] slice s2={s2:g} is missing waves '
                               f'{missing[:6]}{"..." if len(missing) > 6 else ""} '
                               f'(fail-loud; rerun the adaptive driver).')
    # delta_table[j, k] = delta_geq3(band_n[j], s2_values[k]): one row per band
    # wave, one column per s2 node (the degeneracy (n+1)^2 is included)
    delta_table = np.vstack([[tables[s2][int(n)] for s2 in s2_values]
                             for n in band_n])

    # ---- the band integrals and their high-s2 tails, all waves at once ------
    # band_candidate_sums, shared with the stage-1 s2 watcher (which therefore
    # stopped on exactly these numbers), returns per wave of band_n:
    #   I_n_s2_max  = -Int_0^{s2_max} delta_geq3 ds2      (log-Simpson)
    #   T_n_s2_max  = -Int_{s2_max}^inf delta_geq3 ds2    (anchored shape, GUARD
    #                 SHAPE-RATIO inside: repaired waves in shape_ratio_repaired)
    #   I_n_inf     = I_n_s2_max + T_n_s2_max
    # and the band sums at the trailing candidate cutoffs (S2-BAACKE below)
    s2_cutoff_candidates, band_sum_to_cutoff, band_sum_with_high_s2_tail, per_wave_at_s2_max = band_candidate_sums(
        s2_values, delta_table, band_n, m_min2, R, tr_U_int3)
    I_n_s2_max = per_wave_at_s2_max['I_n_s2_max']
    T_n_s2_max = per_wave_at_s2_max['T_n_s2_max']
    I_n_inf = per_wave_at_s2_max['I_n_inf']
    shape_ratio_repaired = per_wave_at_s2_max['shape_ratio_repaired']

    # ---- GUARD S2-STEP (aborts; does NOT enter lnD): per-wave step-size
    # meter on the every-2nd-point sub-grid (endpoint forced in so I_n_half_grid and
    # I_n_s2_max cover the same [s2_min, s2_max] interval); I_n_from_delta
    # integrates every wave at once along the s2 axis, as for I_n_s2_max.
    # Each wave's change is measured against the band's LARGEST |I_n|, one
    # scale for all waves: a wave whose I_n passes through zero (channels of
    # opposite sign cancelling) keeps an ordinary quadrature change, and
    # dividing by its own tiny |I_n| would abort for nothing.
    half_grid_idx = np.unique(np.r_[np.arange(0, len(s2_values), 2),
                               len(s2_values) - 1])
    I_n_half_grid = I_n_from_delta(s2_values[half_grid_idx], delta_table[:, half_grid_idx])
    s2_step_meter = np.abs(I_n_half_grid - I_n_s2_max) / np.max(np.abs(I_n_s2_max))
    s2_step_meter_max = float(np.max(s2_step_meter))
    print(f"[S2-STEP] grid-halving quadrature meter: max over the waves of "
          f"|I_n_half_grid - I_n_s2_max| / max_n |I_n_s2_max| = "
          f"{s2_step_meter_max:.2e} (limit {args.s2_step_tol:g})")
    if s2_step_meter_max > args.s2_step_tol:
        raise RuntimeError(f'[ABORT S2-STEP] grid-halving meter '
                           f'{s2_step_meter_max:.2e} > {args.s2_step_tol:g} at '
                           f'n={int(band_n[np.argmax(s2_step_meter)])}: refine '
                           f'the s2 grid.')
    # ---- GUARD REPAIR-SHARE (aborts; does NOT enter lnD): the SHAPE-RATIO repair is
    # a stand-in for a broken ansatz, harmless only while the repaired waves
    # carry a negligible slice of the high-s2 tail -- so MEASURE that share
    # (ceiling 5%) instead of assuming it.
    repaired_high_s2_share = (float(np.abs(T_n_s2_max[shape_ratio_repaired]).sum())
                              / max(abs(float(T_n_s2_max.sum())), 1e-300))
    if shape_ratio_repaired.any():
        print(f'[GUARD REPAIR-SHARE] {int(shape_ratio_repaired.sum())} wave(s) {band_n[shape_ratio_repaired].tolist()} '
              f'used a REPAIRED shape ratio; their magnitudes sum to '
              f'{np.abs(T_n_s2_max[shape_ratio_repaired]).sum():.3e} against sum T_n(s2_max) = '
              f'{T_n_s2_max.sum():+.6f} '
              f'({100*repaired_high_s2_share:.3f}% -- tolerable while this is small)')
    if repaired_high_s2_share > 0.05:
        raise RuntimeError(
            f'[ABORT REPAIR-SHARE] the SHAPE-RATIO-repaired waves {band_n[shape_ratio_repaired].tolist()} '
            f'carry {100*repaired_high_s2_share:.2f}% of the high-s2 tail (ceiling 5%): '
            f'the uniform O(U_int^3) shape ansatz cannot complete waves that '
            f'still carry weight, so the uncalibrated repair would set a real '
            f'part of lnD -- inspect tr(U_int^3) of this bounce.')

    # ===== GUARD S2-BAACKE (aborts; does NOT enter lnD) ======================
    # The s2 CUTOFF guard: on the delivered tables, re-measure exactly
    # what the S2-PLATEAU watcher stopped on -- the raw band truncation must
    # follow the analytic Baacke law and the predicted remaining error must
    # be inside eps_s2 (watchers_guards_vgeneral_phi_v2 holds the one implementation).
    plateau = watch_s2_plateau(s2_cutoff_candidates, band_sum_with_high_s2_tail, band_sum_to_cutoff,
                               baacke_c, baacke_c_scale, m_min2)
    print(f"[S2-BAACKE] raw truncation vs law baacke_c/(s2_max+m_min2): Baacke-law miss "
          f"|baacke_c_measured - baacke_c| / baacke_c_scale = "
          f"{plateau['baacke_c_miss']:.3f} (limit {BAACKE_MISS_TOL:g}); predicted remaining error "
          f"{plateau['completed_sum_drift']:.2e} x {plateau['baacke_envelope']:.3f} = "
          f"{plateau['s2_trunc_err_pred']:.2e} (limit {eps_s2:g})")
    guard_s2_baacke(plateau, eps_s2)
    # ==========================================================================

    atomic_savez(paths['band_integrals'],
             band_n=band_n, s2_values=s2_values, s2_max=s2_max,
             I_n_s2_max=I_n_s2_max, T_n_s2_max=T_n_s2_max,
             I_n_inf=I_n_inf,
             s2_step_meter=s2_step_meter, s2_step_meter_max=s2_step_meter_max,
             eps_s2=eps_s2, s2_step_tol=args.s2_step_tol,
             guard_s2_trunc_err_pred=plateau['s2_trunc_err_pred'],
             guard_baacke_c_miss=plateau['baacke_c_miss'],
             n_max=n_max,
             high_s2_tail_method='trU_int3-shape-anchored',
             **provenance_stamp(args.bounce_npz, background['m2'],
                                background['params'], background['fields']))
    print(f"[BAND] SUMMARY: {len(band_n)} waves (n = {band_n[0]}..{band_n[-1]}), "
          f"{len(s2_values)} s2 slices to s2_max = {s2_max:g}; sum I_n(inf) over the "
          f"band = {float(np.sum(I_n_inf)):+.6f}")
    print(f"[OK] wrote {paths['band_integrals']}")


if __name__ == '__main__':
    main()
