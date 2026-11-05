#!/usr/bin/env python3
"""band_adaptive_vgeneral_phi_v2.py -- level 2, stage 1: the residual band
delta_geq3(n, s2) = Gbar_minus_fv - Gbar_1 - Gbar_2, the traced Green's
function of wave n at Born order >= 3 (weighted with the degeneracy
deg = (n+1)^2), for the waves n = 2..n_max on the s2 ladder up to s2_max.
The band is generated in ROUNDS by the h-ODE engine
(greens_function_vgeneral_phi_v2, one subprocess per worker), and the
WATCHERS (watchers_guards_vgeneral_phi_v2) decide, per direction, when the
band has reached the asymptotic regime in which the two analytic
tails hold (the high-s2 tail of stage 3, the high-n tail of
stage 4) and generation stops: they find the band cutoffs s2_max, n_max and
the fit-window start fit_lo of stage 4.  No cutoff is hardcoded: every
model-dependent number below is derived from the bounce being integrated.

THE (n, s2) PLANE of lnD_ren, region by region:
  n = 0, 1      all s2        the sectors: pole-subtracted integral up to s2_max
                              (stages 2a/2b), pole and fictitious-mode terms in
                              closed form (stage 7); their raw trace above
                              s2_max is dropped (bound: WATCH SECTOR-CUTOFF)
  n = 2..n_max  s2 <= s2_max  THE BAND: delta_geq3 here, I_n(s2_max) in stage 3
  n = 2..n_max  s2 > s2_max   the high-s2 tail T_n(s2_max) (stage 3)
  n > n_max     all s2        the high-n tail Zcal (stage 4)

READS   --bounce-npz through load_background: R and phi_b (the wall radius
        r_W), m_min2 = min m_i^2 and tr_U_int3 = tr(U_int^3)(r) (the two
        analytic moments A3 and baacke_c), params/fields (with the bounce
        fingerprint, the provenance every slice must match) and m2 (the
        provenance stamp of band_cutoffs); the slice npz of this tag,
        re-read in every round for the watchers (on a rerun the existing
        ones are kept, RESUME below).
WRITES  residual_band_<tag>_s2<v>.npz   one slice per band s2 node (v = s2 with
            7 significant digits, '.' -> 'p'), written by the engine workers
            this driver starts: n_values and, per wave, delta_geq3 and the Gbar
            traces; a base slice also the n = 0, 1 traces Gbar_minus_fv_n0/_n1
            (plot layer) and delta_geq3_n0/_n1 (WATCH SECTOR-CUTOFF).  Read
            through read_band_tables by this driver's watchers, stage 3 and the
            plot layer.
        band_cutoffs_<tag>.npz   the certified cutoffs, moments and targets, with
            their readers:
              n_max                     stages 3, 4, the add-on, the driver
                                        script's resume skip-guards (the plot
                                        codes take n_max from band_integrals,
                                        lnD_ren or the add-on npz instead)
              s2_max                    stages 2a/2b, 3, the skip-guards
              s2_min (lowest band node) stages 2a/2b
              fit_lo                    stage 4, the skip-guards
              baacke_c, baacke_c_scale, eps_s2   stage 3 (guard S2-BAACKE)
              A3, a3_tol, eps_n         stage 4 (guards N-ASYMPTOTE, N-TAIL-BOUND)
            record only, no reader: the watcher measurements watch_s2_trunc_err_pred,
            watch_baacke_c_miss, watch_subleading_zeta_bound, watch_fit_a/b/c and the
            echoes m_min2, r_W, onset_tol, rounds, rtol, atol
        band_manifest_<tag>.json   the slice files of the final rectangle (the
            MANIFEST guard of stage 3 and of the plot layer's band read refuses
            any other same-tag slice)

HOW (main, in this order)
-------------------------
0.  The analytic guard objects of THIS bounce are computed once: A3,
    baacke_c (watchers_guards_vgeneral_phi_v2.analytic_moments, each with its
    unsigned cancellation scale -- a cancelled moment aborts here, before any
    band is generated), with m_min2 and r_W (the wall radius; the
    centrifugal edge is n*(s2) = r_W sqrt(s2 + m_min2)).
1.  First estimate (cutoff_first_estimate), one analytic equation per
    direction: s2_first = m_min2 + |baacke_c|/START_ENVELOPE and the waves up
    to the centrifugal edge there, n = r_W sqrt(s2_first + m_min2) (at least
    the first band wave n = 2) -- a deliberately small first rectangle that the
    watchers only extend upward.  (A regression pin, --pin-n-max with
    --pin-s2-max, fixes the rectangle instead and skips the extension of
    step 3.)
2.  Each round computes the (slice, wave-block) pieces missing from the
    current target rectangle (plan_missing; run_jobs starts the engine
    workers), then re-reads the tables and evaluates the four watchers on
    the ACTUAL production data (evaluate_watchers); the s2 direction stops
    when S2-PLATEAU and SECTOR-CUTOFF both pass, the n direction when
    N-TAIL-BOUND passes:
      WATCH S2-PLATEAU   the band sums at the trailing candidate cutoffs L2
                         of the s2 grid.  Passes when the RAW sum
                         band_sum_to_cutoff = sum_n I_n(L2) depends on L2 as
                         const + baacke_c/(L2 + m_min2), the Baacke law
                         (fitted slope against 1/(L2 + m_min2) within
                         BAACKE_MISS_TOL x baacke_c_scale of baacke_c:
                         baacke_law_holds),
                         AND the predicted remaining error of the COMPLETED
                         sum band_sum_with_high_s2_tail = sum_n [I_n(L2) +
                         its high-s2 tail above L2],
                         s2_trunc_err_pred = completed_sum_drift x envelope
                         |baacke_c|/(s2_max + m_min2), is <= eps_s2 (the S2-BAACKE
                         conditions stage 3 re-verifies).
      WATCH SECTOR-CUTOFF the raw traces delta_geq3 of the sectors n = 0, 1
                         above s2_max are dropped by stages 2a/2b: passes
                         when their extrapolated remainder, summed over
                         n = 0, 1, is <= SECTOR_REMAINDER_TOL (the limit
                         and its estimator sector_high_s2_remainder_bound
                         live in watchers_guards).  The estimator
                         runs here on the n = 0, 1 traces stored in every
                         base slice, i.e. on the band s2 grid; the
                         SECTOR-CUTOFF guard of stage 7 (assemble_lnD) runs
                         the same estimator on the sector grids.
      WATCH N-ONSET      pointwise nu^3 I_n(inf) against the analytic A3
                         (the leading 1/nu^3 coefficient of the high-n
                         tail): the sustained match defines fit_lo.
      WATCH N-TAIL-BOUND the fit I_n(inf) ~ fit_a/nu^3 + fit_b/nu^5 +
                         fit_c/nu^7 on [fit_lo, n_max] and its subleading
                         tail bound |fit_b| zeta(5, n_max+2) + |fit_c|
                         zeta(7, n_max+2): stop the n direction when the
                         bound is <= eps_n (then even an O(1) misfit of the
                         interpolation coefficients cannot move lnD_ren by
                         more than eps_n) AND the fitted leading fit_a
                         agrees with the analytic A3 within a3_tol (the
                         N-ASYMPTOTE condition stage 4 re-verifies).
3.  Directions that have not passed are extended (s2: to the ladder point
    >= 1.25 x the current top; n: by 25%, or to the N-TAIL-BOUND prediction
    when that is larger; a raw truncation off the Baacke law extends both)
    and the loop repeats; --max-rounds is the runaway guard.
4.  On success the manifest and band_cutoffs of the final rectangle are
    written (WRITES above).  The downstream stages re-verify the same
    conditions as GUARDS and abort if they fail: S2-BAACKE (stage 3) and
    N-ASYMPTOTE, N-TAIL-BOUND (stage 4) on the same band data, so they
    repeat the numbers the watchers stopped on; SECTOR-CUTOFF (stage 7) on
    the sector grids instead of the band grid, so its number differs
    slightly and a run near SECTOR_REMAINDER_TOL can pass the watcher and
    still abort in stage 7.

THE s2 LADDER (s2_ladder)
-------------------------
A fixed grid-density template in units of the mass^2 scale
max(1, m_min2/10), continued geometrically below it (down to 1e-2 of the
model's lowest s2 scale) and above it (up to s2_max).  The density is a
quadrature CONVENTION (like an ODE tolerance), checked per run by the
scale-free S2-STEP halving guard in band_integrals; the ladder's EXTENT is
what the S2-PLATEAU watcher decides.

RESUME: existing slice npz with matching provenance (bounce fingerprint,
engine tolerances, potential_id, engine build -- checked by
read_band_tables) are kept and only the missing pieces are computed; a
mismatching same-tag slice aborts (delete or retag it).
Partial-wave extension files are written next to their base slice as
residual_band_<tag>_s2<v>_n<lo>to<hi>.npz and merged by read_band_tables.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from greens_function_vgeneral_phi_v2 import LSODA_ATOL, LSODA_RTOL
from pipeline_helpers_vgeneral_phi_v2 import (add_standard_cli, atomic_savez,
                                 band_s2_text, bounce_fingerprint, load_background, provenance_stamp,
                                 read_band_tables, stage_paths, wall_radius)
from pipeline_quadrature_vgeneral_phi_v2 import (N_WINDOW_MIN_WAVES,
                                                 band_candidate_sums,
                                                 odd_family_high_n_fit)
from watchers_guards_vgeneral_phi_v2 import (analytic_moments,
                               baacke_law_holds, watch_n_onset, watch_n_max_needed,
                               watch_s2_plateau, BAACKE_MISS_TOL,
                               SECTOR_REMAINDER_TOL, sector_high_s2_remainder_bound)


# --------------------------------------------------------------------------- #
#  The s2 ladder (single owner of the band grid rule)                          #
# --------------------------------------------------------------------------- #
def s2_ladder(s2_max, m_min2, r_W):
    """The band s2 grid: the fixed density template in units of
    ladder_mass2_unit = max(1, m_min2/10), continued geometrically below it
    down to 1e-2 min(m_min2, 1/r_W^2) and above it up to (and including) the
    first node >= s2_max; every node rounded to 7 significant digits."""
    # ladder_mass2_unit >= 1: a model with m_min2 <= 10 keeps the template in
    # absolute units (bottom 1e-3; its lower s2 scales are reached by the
    # continuation below), a heavier model moves the fine region up with its
    # thresholds
    ladder_mass2_unit = max(1.0, m_min2 / 10.0)
    ladder_segments = [np.geomspace(1e-3, 10.0, 49) * ladder_mass2_unit,
            np.arange(12.5, 50.0001, 2.5) * ladder_mass2_unit,
            np.arange(55.0, 300.0001, 5.0) * ladder_mass2_unit,
            np.arange(310.0, 500.0001, 10.0) * ladder_mass2_unit,
            np.arange(525.0, 1000.0001, 25.0) * ladder_mass2_unit]
    s2_grid = np.concatenate(ladder_segments)
    # below the template: its geometric segment continues (ratio 10^(1/12),
    # 12 nodes per decade) until the first node lies a hundred times below the
    # model's lowest s2 scale -- the lightest channel mass^2 or the wall scale
    # 1/r_W^2; two nodes at a time, so every Simpson panel (a pair of
    # intervals, I_n_from_delta) of the grid above keeps its nodes.  Why a
    # hundred: the strip [0, first node] is closed by extrapolation
    # (low_s2_strip), and no grid comparison (S2-STEP included) can see
    # structure below the first node, so delta_geq3 must be flat there: it
    # varies on the lowest s2 scale, so across the strip it changes by about
    # 1% (a hundred rather than ten is a safety margin, 12 more nodes)
    while s2_grid[0] > 1e-2 * min(m_min2, 1.0 / r_W**2):
        s2_grid = np.concatenate([s2_grid[0] / 10**(np.array([2.0, 1.0]) / 12), s2_grid])
    # above the template: geometric, ratio 1.055, up to the first node >= s2_max
    while s2_grid[-1] < s2_max:
        s2_grid = np.append(s2_grid, s2_grid[-1] * 1.055)
    # 7 significant digits: the values the engine CLI and the slice file names
    # carry (band_s2_text), exact at every scale the continuation reaches.
    # The grid ends at the first node at or above s2_max: a pinned s2_max is
    # itself such a node value, so no tolerance is needed
    s2_grid = np.array([float(band_s2_text(v)) for v in s2_grid])
    return s2_grid[:int(np.searchsorted(s2_grid, s2_max)) + 1]


# --------------------------------------------------------------------------- #
#  HOW 0-1: the moment-cancellation limit and the first estimate              #
# --------------------------------------------------------------------------- #
MOMENT_REL_TOL = 1e-6   # HOW 0: a moment counts as CANCELLED (abort before any
                        # band is generated) when |A3| <= MOMENT_REL_TOL A3_scale
                        # or |baacke_c| <= MOMENT_REL_TOL baacke_c_scale, the
                        # _scale being the same integral with |tr U_int^3|.
                        # Relative, because the moments carry the model's units:
                        # a plain "== 0.0" test would let a cancelled moment through
START_ENVELOPE = 10.0   # bound on the Baacke envelope |baacke_c|/(s2 + m_min2),
                        # the size of the raw s2-truncation of lnD, at the first
                        # estimate's s2: far above any accuracy target, so
                        # round 1 starts small (a small start costs a few
                        # cheap rounds, an oversized one wastes generation)


def cutoff_first_estimate(bounce_moments):
    """First estimate of the two cutoffs (where round 1 begins), ONE
    analytic equation per direction, both from the bounce moments alone:
        s2_first = m_min2 + |baacke_c| / START_ENVELOPE   (there the
                   Baacke-law envelope |baacke_c|/(s2 + m_min2) =
                   |baacke_c|/(|baacke_c|/START_ENVELOPE + 2 m_min2) is at
                   most START_ENVELOPE, and s2_first > 0 for any baacke_c);
        n_first  = max(2, ceil(r_W sqrt(s2_first + m_min2))), the
                   centrifugal estimate with at least the first band wave.
    A first estimate only -- the watchers own the real cutoffs and refine
    strictly upward from here."""
    s2_first = bounce_moments['m_min2'] + abs(bounce_moments['baacke_c']) / START_ENVELOPE
    # the centrifugal edge n*(s2) = r_W sqrt(s2 + m_min2), the wave at which
    # the band integrand turns over at this s2; the band starts at n = 2, so
    # even a very small edge needs one band wave
    n_first = max(2, int(np.ceil(bounce_moments['r_W']
                                 * np.sqrt(s2_first + bounce_moments['m_min2']))))
    return s2_first, n_first


# --------------------------------------------------------------------------- #
#  Engine subprocess plumbing                                                  #
# --------------------------------------------------------------------------- #
def engine_cmd(args, s2_list, n_block_lo, n_block_hi, suffix):
    """The band-engine invocation for one worker chunk: the engine file next to
    this one, run directly, at the engine's own LSODA tolerances LSODA_RTOL /
    LSODA_ATOL (the same the sector stages and the census solve with)."""
    return [sys.executable,
            os.path.join(_HERE, 'greens_function_vgeneral_phi_v2.py'),
            '--s2-list', ','.join(band_s2_text(v) for v in s2_list),
            '--n-min', str(n_block_lo), '--n-max', str(n_block_hi),
            '--data-dir', args.data_dir, '--tag', args.tag,
            '--file-suffix', suffix]


def run_jobs(args, jobs, round_no):
    """Run the round's engine jobs: each job is (s2_values, n_block_lo, n_block_hi,
    suffix).  Jobs run ONE AFTER ANOTHER so total concurrency never exceeds
    --jobs; within a job the slices, which all carry the same wave block,
    are dealt out to the --jobs workers in turn in s2 order, so neighbouring
    slices (of similar cost) go to different workers and every worker gets
    an equal share (the split decides only which worker computes a slice,
    never a result).  Workers get
    VGENERAL_PHI_BOUNCE_NPZ set explicitly to the driver's --bounce-npz (never
    an inherited stale value); output goes to band_worker<k>_<tag>.log.  A
    non-zero worker exit terminates the job's remaining workers and aborts
    the driver with the log tail (fail-loud)."""
    env = dict(os.environ,
               VGENERAL_PHI_BOUNCE_NPZ=os.path.abspath(args.bounce_npz))
    n_workers = max(1, args.jobs)
    for (s2v, n_block_lo, n_block_hi, suffix) in jobs:
        s2_sorted = sorted(s2v)
        chunks = [s2_sorted[k::n_workers] for k in range(n_workers)]
        procs, logs = [], []
        for k, ch in enumerate(c for c in chunks if c):
            log = os.path.join(args.data_dir, f'band_worker{k}_{args.tag}.log')
            fh = open(log, 'a')
            fh.write(f'\n===== round {round_no}: {len(ch)} slices '
                     f'n={n_block_lo}..{n_block_hi} suffix="{suffix}" =====\n')
            fh.flush()
            procs.append(subprocess.Popen(
                engine_cmd(args, ch, n_block_lo, n_block_hi, suffix),
                stdout=fh, stderr=subprocess.STDOUT, env=env))
            logs.append((log, fh))
        failed = None
        for p, (log, fh) in zip(procs, logs):
            p.wait()
            fh.close()
            if p.returncode != 0 and failed is None:
                failed = (p.returncode, log)
                for q in procs:                # stop the still-running siblings
                    if q.poll() is None:
                        q.terminate()
        if failed:
            rc, log = failed
            tail = ''.join(open(log).readlines()[-25:])
            raise RuntimeError(
                f'[ABORT BAND-WORKER] engine worker failed (exit {rc}); '
                f'log tail of {log}:\n{tail}')


def plan_missing(tables, s2_grid, n_max_target):
    """The (slice, wave-block) pieces missing from the target rectangle
    waves 2..n_max_target x s2_grid, as engine jobs grouped by identical wave block.
    Missing waves per slice are grouped into contiguous runs."""
    groups = {}
    for s2 in s2_grid:
        have = set(tables.get(float(s2), {}).keys())
        missing = [n for n in range(2, n_max_target + 1) if n not in have]
        for lo, hi in _runs(missing):
            groups.setdefault((lo, hi), []).append(float(s2))
    jobs = []
    for (lo, hi), s2v in sorted(groups.items()):
        # base slices (full range, no file yet) carry no suffix; partial
        # wave-extensions on existing slices get a block suffix
        suffix = '' if lo == 2 else f'_n{lo}to{hi}'
        jobs.append((s2v, lo, hi, suffix))
    return jobs


def _runs(sorted_ints):
    """Contiguous runs [(lo, hi), ...] of an ascending integer list."""
    runs = []
    for v in sorted_ints:
        if runs and v == runs[-1][1] + 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return [(a, b) for a, b in runs]


# --------------------------------------------------------------------------- #
#  Watcher evaluation on the current tables                                    #
# --------------------------------------------------------------------------- #
def evaluate_watchers(s2_values, tables, files, band_n, bounce_moments, onset_tol,
                      R, tr_U_int3, fit_lo_pin=None):
    """Evaluate the watchers of both directions on the current rectangle.
    Returns dict(plateau, sector_remainder_bound, fit_lo_onset, tail_fit, fit_lo):
    the S2-PLATEAU measurement, the SECTOR-CUTOFF bound (the dropped remainder
    of the sectors n = 0, 1 above s2_max, summed), the N-ONSET wave, the
    tail_fit with its subleading bound (None where not yet
    determinable) and fit_lo, the fit-window start = fit_lo_pin (the
    regression pin, so a pinned run's recorded tail_fit is the one the downstream
    stage will use) or else the fit_lo_onset.  `files` are the slice files of
    the tables (their base slices carry the n = 0, 1 traces the SECTOR-CUTOFF
    bound reads, delta_geq3_n0_n1_on_band_grid)."""
    delta_table = np.vstack([[tables[s2][n] for s2 in s2_values]
                             for n in band_n])              # (n_waves, n_slices)
    # the full-cutoff candidate of band_candidate_sums is the production
    # I_n(inf) of band_integrals: the n-direction watchers read it
    s2_cutoff_candidates, band_sum_to_cutoff, band_sum_with_high_s2_tail, per_wave_at_s2_max = band_candidate_sums(
        s2_values, delta_table, band_n, bounce_moments['m_min2'], R, tr_U_int3)
    plateau = watch_s2_plateau(s2_cutoff_candidates, band_sum_with_high_s2_tail, band_sum_to_cutoff,
                               bounce_moments['baacke_c'], bounce_moments['baacke_c_scale'],
                               bounce_moments['m_min2'])
    # the sectors' raw traces on the band grid, each through the estimator
    # the assembly's SECTOR-CUTOFF guard applies to the sector grids
    sector_remainder_bound = sum(
        sector_high_s2_remainder_bound(s2_values, delta_geq3_n)[0]
        for delta_geq3_n in delta_geq3_n0_n1_on_band_grid(files, s2_values))
    fit_lo_onset = watch_n_onset(band_n, per_wave_at_s2_max['I_n_inf'],
                                 bounce_moments['A3'], onset_tol)
    fit_lo = fit_lo_pin if fit_lo_pin is not None else fit_lo_onset
    tail_fit = None
    # the fit is tried once the window holds the N-WINDOW minimum that
    # stage 4's fit requires as well
    if fit_lo is not None and band_n[-1] - fit_lo + 1 >= N_WINDOW_MIN_WAVES:
        fit_a, fit_b, fit_c, _Zcal, subleading_zeta_bound = odd_family_high_n_fit(
            band_n, per_wave_at_s2_max['I_n_inf'], fit_lo, int(band_n[-1]))
        tail_fit = dict(a=fit_a, b=fit_b, c=fit_c,
                        subleading_zeta_bound=subleading_zeta_bound)
    return dict(plateau=plateau, sector_remainder_bound=sector_remainder_bound,
                fit_lo_onset=fit_lo_onset, tail_fit=tail_fit, fit_lo=fit_lo)


def delta_geq3_n0_n1_on_band_grid(files, s2_values):
    """delta_geq3 of the sectors n = 0 and n = 1 at every band s2, as the base
    slices store them (greens_function._run_slice: delta_geq3_n0, delta_geq3_n1)."""
    delta_geq3_n0_n1_by_s2 = {}
    for f in files:
        with np.load(f) as slice_npz:
            if 'delta_geq3_n0' in slice_npz.files:          # base slices only
                delta_geq3_n0_n1_by_s2[float(slice_npz['s2'])] = (float(slice_npz['delta_geq3_n0']),
                                                 float(slice_npz['delta_geq3_n1']))
    return [np.array([delta_geq3_n0_n1_by_s2[s2][nd] for s2 in s2_values]) for nd in (0, 1)]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description='vgeneral_phi_v2 stage 1: adaptive band driver (the '
                    'watchers decide the cutoffs from the band data itself; '
                    'nothing model-dependent is hardcoded)')
    add_standard_cli(ap)
    ap.add_argument('--eps-s2', type=float, default=1e-2,
                    help='S2-PLATEAU target: predicted remaining s2-truncation '
                         'error in lnD (default 1e-2)')
    ap.add_argument('--eps-n', type=float, default=1e-2,
                    help='N-TAIL-BOUND target: ceiling on the interpolation '
                         'part of the zeta tail (default 1e-2)')
    ap.add_argument('--onset-tol', type=float, default=0.2,
                    help='N-ONSET: sustained |nu^3 I_n(inf)/A3 - 1| defining '
                         'the fit-window start (default 0.2)')
    ap.add_argument('--a3-tol', type=float, default=0.10,
                    help='N-ASYMPTOTE: max |fitted a / A3 - 1| of the fit '
                         'window, the tolerance the stage-4 guard re-verifies '
                         '(default 0.10)')
    ap.add_argument('--jobs', type=int,
                    default=int(os.environ.get('BAND_JOBS', '2')),
                    help='parallel engine workers (default $BAND_JOBS or 2)')
    ap.add_argument('--max-rounds', type=int, default=40,
                    help='runaway guard: abort when the watchers have not '
                         'passed after this many rounds (default 40)')
    ap.add_argument('--pin-n-max', type=int, default=None,
                    help='REGRESSION PIN: fix n_max instead of letting the '
                         'watchers extend (requires --pin-s2-max; watchers '
                         'still print, the downstream guards still verify)')
    ap.add_argument('--pin-s2-max', type=float, default=None,
                    help='REGRESSION PIN: fix s2_max (requires --pin-n-max)')
    ap.add_argument('--pin-fit-lo', type=int, default=None,
                    help='REGRESSION PIN: fix fit_lo, with or without the '
                         'rectangle pins (default: the N-ONSET watcher '
                         'finding)')
    args = ap.parse_args()
    pinned = (args.pin_n_max is not None) or (args.pin_s2_max is not None)
    if pinned and (args.pin_n_max is None or args.pin_s2_max is None):
        ap.error('--pin-n-max and --pin-s2-max must be given together')

    background = load_background(args.bounce_npz)
    # ---- HOW 0: the analytic guard objects of THIS bounce -- the tr(U_int^3)
    # moments A3 and baacke_c with their cancellation scales, m_min2 and the
    # wall radius r_W (the prediction side of every watcher; stored in
    # band_cutoffs)
    bounce_moments = dict(analytic_moments(background['R'], background['tr_U_int3']),
                          m_min2=background['m_min2'],
                          r_W=wall_radius(background['R'], background['phi_b']))
    # FAIL FAST on a cancelled moment (MOMENT_REL_TOL): the N-ONSET watcher,
    # the n stop test and the N-ASYMPTOTE guard divide by A3, the S2-PLATEAU
    # watcher and the S2-BAACKE guard fit against baacke_c/(s2 + m_min2).
    # (With A3 = 0 the high-n form starts at 1/nu^5; only the stopping
    # criterion is lost.)  "<=" refuses an exactly vanishing moment
    # (A3 = A3_scale = 0) too; the aborts run before the pin branch, so the
    # pins cannot bypass them.
    if abs(bounce_moments['A3']) <= MOMENT_REL_TOL * bounce_moments['A3_scale']:
        raise RuntimeError(
            f"[ABORT MOMENT-A3] the analytic moment A3 = (1/16) Int r^5 tr U^3 = "
            f"{bounce_moments['A3']:+.6g} is at most MOMENT_REL_TOL = "
            f"{MOMENT_REL_TOL:g} of its unsigned moment (1/16) Int r^5 |tr U^3| = "
            f"{bounce_moments['A3_scale']:.6g}: the sign-indefinite trace cancels.  "
            f"The N-ONSET watcher and the N-ASYMPTOTE guard both divide by "
            f"A3 and are undefined here; the band driver has no n-direction "
            f"convergence criterion for such a model, and the regression pins "
            f"do not bypass this abort.")
    # the same test for the s2-direction moment baacke_c
    if abs(bounce_moments['baacke_c']) <= MOMENT_REL_TOL * bounce_moments['baacke_c_scale']:
        raise RuntimeError(
            f"[ABORT MOMENT-BAACKE-C] the analytic Baacke moment baacke_c = -(1/48) Int "
            f"r^3 tr U^3 = {bounce_moments['baacke_c']:+.6g} is at most "
            f"MOMENT_REL_TOL = {MOMENT_REL_TOL:g} of its unsigned moment "
            f"(1/48) Int r^3 |tr U^3| = {bounce_moments['baacke_c_scale']:.6g}: "
            f"the sign-indefinite trace cancels.  The S2-PLATEAU watcher and the "
            f"S2-BAACKE guard fit the band sums against the law "
            f"baacke_c/(s2 + m_min2) and its envelope |baacke_c|/(s2 + m_min2), which vanish "
            f"here; the band driver has no s2-direction convergence criterion "
            f"for such a model, and the regression pins do not bypass this abort.")
    print(f"[MOMENTS] analytic guard objects of THIS bounce: "
          f"A3={bounce_moments['A3']:+.6g} (|A3| is "
          f"{abs(bounce_moments['A3'])/bounce_moments['A3_scale']:.3f} of its unsigned moment)  "
          f"baacke_c={bounce_moments['baacke_c']:+.6g} (|baacke_c| is "
          f"{abs(bounce_moments['baacke_c'])/bounce_moments['baacke_c_scale']:.3f} of its "
          f"unsigned moment)  m_min2={bounce_moments['m_min2']:.6g}  "
          f"r_W={bounce_moments['r_W']:.4f}", flush=True)

    # ---- HOW 1: the first estimate (one analytic equation per direction, see
    # cutoff_first_estimate); a regression pin fixes the rectangle instead
    # and skips the watcher extension loop.  Every round's s2_ladder call
    # snaps s2_target to the first ladder point >= it.
    if pinned:
        s2_target, n_max_target = args.pin_s2_max, args.pin_n_max
        print(f"[PIN] regression rectangle fixed: n_max={n_max_target} "
              f"s2_max = the ladder point >= {s2_target:g} (watchers report, "
              f"guards verify)", flush=True)
    else:
        s2_target, n_max_target = cutoff_first_estimate(bounce_moments)
        print(f"[FIRST-ESTIMATE] s2_first = m_min2 + |baacke_c|/{START_ENVELOPE:g} -> "
              f"s2 ~ {s2_target:g}; centrifugal edge there -> n ~ {n_max_target} "
              f"(watchers refine upward from here)", flush=True)
    fingerprint = bounce_fingerprint(args.bounce_npz)
    t0 = time.time()

    # remove stale dot-prefixed atomic-write temp files a killed worker may
    # have left (they never match the slice glob, but they must not pile up)
    for stale in glob.glob(os.path.join(
            args.data_dir, f'.residual_band_{args.tag}_s2*.tmp*.npz')):
        os.remove(stale)
        print(f'[BAND-DRIVER] removed stale worker temp '
              f'{os.path.basename(stale)}', flush=True)

    # the provenance every slice must carry (read_band_tables checks it).  Both
    # reads of a round pass it and cap the read at the ladder top (s2_cap), so
    # a round sees exactly its target rectangle: slices beyond it (a resumed
    # run with deeper data) are never read
    expected_slice_stamps = dict(expect_bounce_fingerprint=fingerprint,
                                 expect_tol=(LSODA_RTOL, LSODA_ATOL),
                                 params=background['params'],
                                 fields=background['fields'])
    # ---- HOW 2-3: the rounds -- complete the target rectangle, evaluate the
    # watchers on it, extend the directions that have not passed
    for rnd in range(1, args.max_rounds + 1):
        s2_grid = s2_ladder(s2_target, bounce_moments['m_min2'],
                            bounce_moments['r_W'])
        s2_values, tables, files = read_band_tables(
            args.data_dir, args.tag, s2_cap=s2_grid[-1], **expected_slice_stamps)
        jobs = plan_missing(tables, s2_grid, n_max_target)
        todo = sum(len(s2v) * (n_block_hi - n_block_lo + 1)
                   for s2v, n_block_lo, n_block_hi, _suffix in jobs)
        print(f"[ROUND {rnd}] target: {len(s2_grid)} slices to s2={s2_grid[-1]:g}, "
              f"waves 2..{n_max_target}  ({todo} missing wave-points)", flush=True)
        if jobs:
            run_jobs(args, jobs, rnd)
            s2_values, tables, files = read_band_tables(
                args.data_dir, args.tag, s2_cap=s2_grid[-1], **expected_slice_stamps)
        # the watcher rectangle = the target rectangle, freshly complete
        band_n = np.arange(2, n_max_target + 1)
        for s2 in s2_values:
            missing = [n for n in band_n if n not in tables[s2]]
            if missing:
                raise RuntimeError(f'[ABORT BAND-DRIVER] slice s2={s2:g} still '
                                   f'missing waves {missing[:6]}... after its '
                                   f'round -- engine did not deliver.')

        watch = evaluate_watchers(s2_values, tables, files, band_n, bounce_moments,
                      args.onset_tol,
                      background['R'], background['tr_U_int3'],
                      fit_lo_pin=args.pin_fit_lo)
        plateau = watch['plateau']
        fit_lo_onset = watch['fit_lo_onset']
        tail_fit = watch['tail_fit']
        print(f"[WATCH S2-PLATEAU] s2_trunc_err_pred = completed_sum_drift x envelope = "
              f"{plateau['completed_sum_drift']:.2e} x {plateau['baacke_envelope']:.3f} = "
              f"{plateau['s2_trunc_err_pred']:.2e} (target {args.eps_s2:g}); "
              f"Baacke-law miss |baacke_c_measured - baacke_c| / baacke_c_scale = "
              f"{plateau['baacke_c_miss']:.3f} (limit {BAACKE_MISS_TOL:g})", flush=True)
        print(f"[WATCH SECTOR-CUTOFF] dropped remainder of the sectors "
              f"{watch['sector_remainder_bound']:.2e} (limit {SECTOR_REMAINDER_TOL:g})",
              flush=True)
        if fit_lo_onset is None:
            print(f"[WATCH N-ONSET] no sustained {args.onset_tol:g}-match of "
                  f"nu^3 I_n(inf) to A3 yet (deepest wave "
                  f"{n_max_target})", flush=True)
        else:
            print(f"[WATCH N-ONSET] onset = {fit_lo_onset} (fit window "
                  f"[{watch['fit_lo']},{n_max_target}])", flush=True)
        if tail_fit:
            print(f"[WATCH N-TAIL-BOUND] |b|zeta5+|c|zeta7 = "
                  f"{tail_fit['subleading_zeta_bound']:.2e} "
                  f"(target {args.eps_n:g}); fitted a/A3-1 = "
                  f"{tail_fit['a']/bounce_moments['A3']-1.0:+.4f}", flush=True)

        # the s2 stop = the SAME conditions the S2-BAACKE guard
        # (watchers_guards.guard_s2_baacke) and the SECTOR-CUTOFF guard
        # (assemble_lnD) verify: predicted remaining error inside the target,
        # the raw truncation on the Baacke law (baacke_law_holds), and the
        # sectors' dropped remainder inside SECTOR_REMAINDER_TOL
        s2_ok = (plateau['s2_trunc_err_pred'] <= args.eps_s2 and baacke_law_holds(plateau)
                 and watch['sector_remainder_bound'] <= SECTOR_REMAINDER_TOL)
        # the n stop = the SAME two conditions the N-TAIL-BOUND and
        # N-ASYMPTOTE guards verify (watchers_guards.guard_n_tail_bound,
        # guard_n_asymptote): subleading tail bound inside the target AND
        # the fitted leading a on the analytic A3
        n_ok = (tail_fit is not None
                and tail_fit['subleading_zeta_bound'] <= args.eps_n
                and abs(tail_fit['a'] / bounce_moments['A3'] - 1.0) <= args.a3_tol)
        # a regression pin fixes the rectangle: accepted after its one
        # complete round (the downstream guards do the verifying)
        if pinned or (s2_ok and n_ok):
            break                    # this round's rectangle is the band
        # extend each direction that has not passed by a scale-free step:
        # s2 by the factor 1.25 (s2_ladder snaps it to the next ladder point),
        # n by 25%, or to the N-TAIL-BOUND prediction when that is larger.
        # A raw truncation off the Baacke law extends BOTH directions -- the
        # two remedies the S2-BAACKE guard names: the band may stop below the
        # centrifugal edge r_W sqrt(s2_max + m_min2), which a higher s2 alone
        # only moves further out, or s2_max may not be asymptotic yet
        if not s2_ok:
            s2_target = 1.25 * s2_values[-1]
        if not n_ok or not baacke_law_holds(plateau):
            n_max_target = int(np.ceil(1.25 * n_max_target))
            if tail_fit:
                n_max_target = max(n_max_target, watch_n_max_needed(
                    tail_fit['b'], tail_fit['c'], args.eps_n))
    else:                            # every round ran without a break
        raise RuntimeError(f'[ABORT BAND-DRIVER] no watcher convergence in '
                           f'{args.max_rounds} rounds.')

    # after the break, s2_values, band_n, files, rnd, plateau and tail_fit are
    # those of the converged (or pinned) round
    fit_lo = watch['fit_lo']
    if fit_lo is None or tail_fit is None:
        raise RuntimeError('[ABORT BAND-DRIVER] the pinned rectangle is too '
                           'small for the N-ONSET watcher to place a fit '
                           'window -- give --pin-fit-lo or enlarge the pin.')
    # ---- HOW 4: the outputs of the final rectangle -- the slice manifest and
    # band_cutoffs (the certified cutoffs, moments and targets the later
    # stages read; WRITES in the module docstring)
    paths = stage_paths(args.data_dir, args.tag)
    # the manifest lists ONLY the final rectangle's slice files (the capped
    # read of its round): data beyond the watcher-chosen cutoff stays on disk
    # but is excluded there and by every downstream read (s2_cap)
    manifest = os.path.join(args.data_dir, f'band_manifest_{args.tag}.json')
    with open(manifest, 'w') as fh:
        json.dump({'slices': sorted(os.path.basename(f)
                                    for f in files)}, fh, indent=1)
    atomic_savez(paths['band_cutoffs'],
                 n_max=int(band_n[-1]), s2_max=float(s2_values[-1]),
                 s2_min=float(s2_values[0]),
                 fit_lo=int(fit_lo),
                 A3=bounce_moments['A3'], baacke_c=bounce_moments['baacke_c'],
                 baacke_c_scale=bounce_moments['baacke_c_scale'],
                 m_min2=bounce_moments['m_min2'], r_W=bounce_moments['r_W'],
                 eps_s2=args.eps_s2, eps_n=args.eps_n,
                 onset_tol=args.onset_tol, a3_tol=args.a3_tol,
                 # the watcher measurements of the final round (record only);
                 # watch_baacke_c_miss = |baacke_c_measured - baacke_c| / baacke_c_scale
                 watch_s2_trunc_err_pred=plateau['s2_trunc_err_pred'],
                 watch_baacke_c_miss=plateau['baacke_c_miss'],
                 watch_subleading_zeta_bound=tail_fit['subleading_zeta_bound'],
                 watch_fit_a=tail_fit['a'], watch_fit_b=tail_fit['b'],
                 watch_fit_c=tail_fit['c'],
                 rounds=rnd,
                 rtol=LSODA_RTOL, atol=LSODA_ATOL,
                 **provenance_stamp(args.bounce_npz, background['m2'],
                                    background['params'], background['fields']))
    print(f"[BAND-DRIVER] converged in {rnd} round(s), "
          f"{time.time()-t0:.0f}s: n_max={band_n[-1]}  s2_max={s2_values[-1]:g}  "
          f"fit_lo..n_max=[{fit_lo},{band_n[-1]}]")
    print(f"[OK] wrote {paths['band_cutoffs']} and {manifest}")


if __name__ == '__main__':
    main()
