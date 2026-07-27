#!/usr/bin/env python3
"""band_adaptive_coupled_toy_model.py -- stage 1: the adaptive band driver.  Generates
the residual-band data in ROUNDS and lets the WATCHERS (watchers_guards_coupled_toy_model)
decide, per direction, when the asymptotic regime is reached and generation
stops.  There is NO pre-scan and NO hardcoded cutoff: every model-dependent
number below is derived from the bounce being integrated.

HOW IT WORKS
------------
0.  The analytic guard objects of THIS bounce are computed once
    (watchers_guards_coupled_toy_model.analytic_moments): A3, c, mbar2, r_W, edge(s2).
1.  First estimate (cutoff_first_estimate): ONE analytic equation
    per direction -- s2 from the Baacke-law envelope |c|/(s2+mbar2) set to
    START_ENVELOPE = 10 (one order of magnitude above the O(1)-O(few)
    envelope every stop lands at, so the estimate always errs SMALL and the
    watchers extend upward), waves up to the centrifugal edge at that s2,
    n = r_W*sqrt(s2+mbar2).  A first estimate only; the watchers own the
    real cutoffs.
2.  Each round computes the (slice, wave-block) pieces missing from the
    current target rectangle -- with the UNCHANGED band engine
    (delta_g_bar_greater_equal_3_coupled_toy_model), one subprocess per worker -- then
    re-reads the tables and evaluates the two watchers on the ACTUAL
    production data:
      WATCH S2-PLATEAU   completed band sum B_c(Lam2) against the Baacke law
                         c/(Lam2+mbar2) (eq. 6): stop the s2 direction when
                         the predicted remaining error E_pred =
                         shape_miss x envelope <= eps_s2.
      WATCH N-ONSET      pointwise nu^3 I_n(inf) against the analytic A3
                         (eq. 5): the sustained match defines fit_lo.
      WATCH N-TAIL-BOUND the fitted subleading tail |c_fit| zeta5 +
                         |e_fit| zeta7 at the current top wave: stop the n
                         direction when it is <= eps_n (then even an O(1)
                         misfit of the interpolation coefficients cannot move
                         lnD_ren by more than eps_n).
3.  Directions that have not passed are extended (s2: ladder target x1.5;
    n: to the N-TAIL-BOUND prediction) and the loop repeats.
4.  On success it writes
      band_cutoffs_<tag>.npz   (n_max, s2_max, fit_lo + the watcher
                                         measurements + the analytic moments)
      band_manifest_<tag>.json          (the exact slice-file set; the
                                         MANIFEST guard in tail_s2_completion
                                         refuses any other same-tag file)
    The downstream stages re-verify the same conditions as GUARDS
    (S2-BAACKE, N-ASYMPTOTE, N-TAIL-BOUND) and abort if they fail.

THE s2 LADDER
-------------
One fixed grid-density template (the certified production grid-density rule (verified per-run by the scale-free S2-STEP halving guard in tail_s2_completion)),
expressed in the mass unit sigma = max(1, mbar2/10) so it transfers across
mass scales:  geomspace(1e-3, 10, 49) + 12.5..50 @2.5 + 55..300 @5 +
310..500 @10 + 525..1000 @25, all x sigma, then a geometric continuation of
ratio 1.055 above.  The density is a quadrature CONVENTION (like an ODE
tolerance), certified per-run by the scale-free S2-STEP halving guard in
tail_s2_completion; the ladder's EXTENT is what the S2-PLATEAU watcher
decides.  For mbar2 <= 10 (sigma = 1) the template reproduces the certified
production grids exactly.

RESUME: existing slice npz with matching provenance (bounce sha, tolerances,
engine version, no thinning) are kept and only the missing pieces are
computed; a mismatching same-tag slice aborts (delete or retag it).
Partial-wave extension files are written next to their base slice as
residual_band_<tag>_s2<v>_n<lo>to<hi>.npz and merged by read_band_tables.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pipeline_helpers_coupled_toy_model import (add_standard_cli, atomic_savez,         # noqa: E402
                                 bounce_sha256, load_bounce, provenance_stamp,
                                 read_band_tables, stage_paths)
from pipeline_quadrature_coupled_toy_model import odd_tail_fit                   # noqa: E402
from watchers_guards_coupled_toy_model import (analytic_moments, band_candidate_sums,          # noqa: E402
                               watch_n_onset, watch_n_max_needed,
                               watch_s2_plateau)


# --------------------------------------------------------------------------- #
#  The s2 ladder (single owner of the band grid rule)                          #
# --------------------------------------------------------------------------- #
def s2_ladder(s2_max, mbar2):
    """The band s2 grid up to (and including) the first template point
    >= s2_max, from the fixed density template scaled by
    sigma = max(1, mbar2/10); see the module docstring."""
    sig = max(1.0, mbar2 / 10.0)
    segs = [np.geomspace(1e-3, 10.0, 49) * sig,
            np.arange(12.5, 50.0001, 2.5) * sig,
            np.arange(55.0, 300.0001, 5.0) * sig,
            np.arange(310.0, 500.0001, 10.0) * sig,
            np.arange(525.0, 1000.0001, 25.0) * sig]
    grid = np.concatenate(segs)
    while grid[-1] < s2_max:
        grid = np.append(grid, grid[-1] * 1.055)
    grid = np.round(grid, 6)
    return grid[:int(np.searchsorted(grid, s2_max - 1e-9)) + 1]


def ladder_snap(s2_target, mbar2):
    """The smallest ladder point >= s2_target (the ladder extends forever)."""
    return float(s2_ladder(s2_target, mbar2)[-1])


# --------------------------------------------------------------------------- #
#  The first estimate (where round 1 begins)                                  #
# --------------------------------------------------------------------------- #
START_ENVELOPE = 10.0   # Baacke envelope |c|/(s2+mbar2) at the start: one
                        # order of magnitude above the O(1)-O(few) envelope
                        # every certified stop lands at, so the first
                        # estimate always errs SMALL (the watchers only ever
                        # extend upward; a small start costs a few cheap
                        # early rounds, an oversized one wastes generation)


def cutoff_first_estimate(mom):
    """First estimate of the two cutoffs (where round 1 begins).  ONE
    analytic equation per direction, both from the bounce moments alone:
        s2 side:  |c| / (s2 + mbar2) = START_ENVELOPE
                  =>  s2_first = mbar2 + |c| / START_ENVELOPE
                  (the +mbar2 keeps degenerate small-|c| potentials above
                  zero without a branch)
        n side:   the centrifugal edge at that s2,
                  n_first = ceil( r_W * sqrt(s2_first + mbar2) ).
    A first estimate only -- the watchers own the real cutoffs and refine
    strictly upward from here."""
    s2_first = mom['mbar2'] + abs(mom['c']) / START_ENVELOPE
    n_first = int(np.ceil(mom['edge'](s2_first)))
    return s2_first, n_first


# --------------------------------------------------------------------------- #
#  Engine subprocess plumbing                                                  #
# --------------------------------------------------------------------------- #
def engine_cmd(args, s2_list, n_lo, n_hi, suffix):
    """The band-engine invocation for one worker chunk.  The engine file runs
    directly -- the potential is hardcoded, so there is no module-injection
    launcher to route it through."""
    base = [sys.executable,
            os.path.join(_HERE,
                         'delta_g_bar_greater_equal_3_coupled_toy_model.py')]
    return base + ['--s2-list', ','.join(f'{v:.6f}' for v in s2_list),
                   '--n-min', str(n_lo), '--n-max', str(n_hi),
                   '--out-dir', args.data_dir, '--tag', args.tag,
                   '--file-suffix', suffix,
                   '--rtol', str(args.rtol), '--atol', str(args.atol)]


def run_jobs(args, jobs, round_no):
    """Run the round's engine jobs: each job is (s2_values, n_lo, n_hi,
    suffix).  Jobs run ONE AFTER ANOTHER so total concurrency never exceeds
    --jobs; within a job the slices are split over --jobs workers, balanced
    by the per-point cost model sqrt(s2+mbar2)*(n_hi-n_lo+1).  Workers get
    COUPLED_TOY_MODEL_BOUNCE_NPZ set explicitly to the driver's --bounce-npz (never
    an inherited stale value); output goes to band_worker<k>_<tag>.log.  A
    non-zero worker exit terminates the job's remaining workers and aborts
    the driver with the log tail (fail-loud)."""
    env = dict(os.environ,
               COUPLED_TOY_MODEL_BOUNCE_NPZ=os.path.abspath(args.bounce_npz))
    for (s2v, n_lo, n_hi, suffix) in jobs:
        order = np.argsort([-math.sqrt(v) for v in s2v])
        chunks = [[] for _ in range(max(1, args.jobs))]
        load = [0.0] * len(chunks)
        for i in order:                                  # greedy cost balance
            k = int(np.argmin(load))
            chunks[k].append(s2v[i])
            load[k] += math.sqrt(s2v[i] + 1.0) * (n_hi - n_lo + 1)
        procs, logs = [], []
        for k, ch in enumerate(c for c in chunks if c):
            log = os.path.join(args.data_dir, f'band_worker{k}_{args.tag}.log')
            fh = open(log, 'a')
            fh.write(f'\n===== round {round_no}: {len(ch)} slices '
                     f'n={n_lo}..{n_hi} suffix="{suffix}" =====\n')
            fh.flush()
            procs.append(subprocess.Popen(
                engine_cmd(args, sorted(ch), n_lo, n_hi, suffix),
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


def plan_missing(tables, s2_grid, n_hi):
    """The (slice, wave-block) pieces missing from the target rectangle
    waves 2..n_hi x s2_grid, as engine jobs grouped by identical wave block.
    Missing waves per slice are grouped into contiguous runs."""
    groups = {}
    for s2 in s2_grid:
        have = set(tables.get(float(s2), {}).keys())
        missing = [n for n in range(2, n_hi + 1) if n not in have]
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
def evaluate_watchers(s2s, tables, ns, mom, args, R, trDW3, fit_lo_pin=None):
    """Evaluate both watcher directions on the current rectangle.
    Returns dict with the S2-PLATEAU measure, the onset, the fit and its
    subleading bound (None where not yet determinable).  fit_lo_pin (the
    regression pin) overrides the N-ONSET window start for the fit, so a
    pinned run's recorded fit is the one the downstream stage will use."""
    D = np.vstack([[tables[s2][n] for s2 in s2s] for n in ns])   # (N, nsl)
    L2c, Braw, Bc, Iinf_last = band_candidate_sums(
        s2s, D, ns, mom['mbar2'], R, trDW3)
    meas = watch_s2_plateau(L2c, Bc, Braw, mom['c'], mom['mbar2'])
    onset = watch_n_onset(ns, Iinf_last, mom['A3'], args.onset_tol)
    fl = fit_lo_pin if fit_lo_pin is not None else onset
    fit = None
    if fl is not None and ns[-1] - fl + 1 >= args.min_fit_points:
        a, cf, ef, tail, sub = odd_tail_fit(ns, Iinf_last, fl, int(ns[-1]),
                                            min_points=args.min_fit_points)
        fit = dict(a=a, c=cf, e=ef, tail=tail, sub=sub)
    return dict(measure=meas, onset=onset, fit=fit, I_inf=Iinf_last)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description='coupled_toy_model stage 1: adaptive band driver (watchers decide '
                    'the cutoffs; no pre-scan, no hardcoded model knowledge)')
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
    ap.add_argument('--min-fit-points', type=int, default=12,
                    help='minimum waves in the fit window (3-parameter fit)')
    ap.add_argument('--jobs', type=int,
                    default=int(os.environ.get('BAND_JOBS', '2')),
                    help='parallel engine workers (default $BAND_JOBS or 2)')
    ap.add_argument('--n-cap', type=int, default=700,
                    help='hard sanity cap on n_max (abort above)')
    ap.add_argument('--s2-cap', type=float, default=2e4,
                    help='hard sanity cap on s2_max (abort above)')
    ap.add_argument('--rtol', type=float, default=1e-11,
                    help='band-engine LSODA rtol (passed through)')
    ap.add_argument('--atol', type=float, default=1e-13,
                    help='band-engine LSODA atol (passed through)')
    ap.add_argument('--max-rounds', type=int, default=40)
    ap.add_argument('--pin-n-max', type=int, default=None,
                    help='REGRESSION PIN: fix n_max instead of letting the '
                         'watchers extend (requires --pin-s2-max; watchers '
                         'still print, the downstream guards still verify)')
    ap.add_argument('--pin-s2-max', type=float, default=None,
                    help='REGRESSION PIN: fix s2_max (requires --pin-n-max)')
    ap.add_argument('--pin-fit-lo', type=int, default=None,
                    help='REGRESSION PIN: fix fit_lo (default: the N-ONSET '
                         'watcher finding, even in pinned mode)')
    args = ap.parse_args()
    pinned = (args.pin_n_max is not None) or (args.pin_s2_max is not None)
    if pinned and (args.pin_n_max is None or args.pin_s2_max is None):
        ap.error('--pin-n-max and --pin-s2-max must be given together')

    bg = load_bounce(args.bounce_npz)
    mom = analytic_moments(bg['R'], bg['trDW3'], bg['Phi'], bg['m2'])
    if mom['A3'] == 0.0:
        raise RuntimeError(
            '[ABORT N-ONSET] the analytic moment A3 = (1/16) Int r^5 tr U^3 '
            'vanishes for this potential -- the odd-family tail machinery '
            'does not apply; inspect the model by hand.')
    print(f"[MOMENTS] analytic guard objects of THIS bounce: "
          f"A3={mom['A3']:+.6g}  c={mom['c']:+.6g}  mbar2={mom['mbar2']:.6g}  "
          f"r_W={mom['r_W']:.4f}", flush=True)

    # first estimate (one analytic equation per direction, see
    # cutoff_first_estimate); a regression pin fixes the rectangle instead
    # and skips the watcher extension loop
    if pinned:
        s2_target = ladder_snap(args.pin_s2_max, mom['mbar2'])
        n_hi = args.pin_n_max
        print(f"[PIN] regression rectangle fixed: n_max={n_hi} "
              f"s2_max={s2_target:g} (watchers report, guards verify)",
              flush=True)
    else:
        s2_first, n_first = cutoff_first_estimate(mom)
        s2_target = ladder_snap(s2_first, mom['mbar2'])
        n_hi = n_first
        print(f"[FIRST-ESTIMATE] Baacke envelope {START_ENVELOPE:g} -> "
              f"s2 ~ {s2_target:g}; centrifugal edge there -> n ~ {n_hi} "
              f"(watchers refine upward from here)", flush=True)
    sha = bounce_sha256(args.bounce_npz)
    t0 = time.time()

    # remove stale dot-prefixed atomic-write temp files a killed worker may
    # have left (they never match the slice glob, but they must not pile up)
    import glob as _glob
    for stale in _glob.glob(os.path.join(
            args.data_dir, f'.residual_band_{args.tag}_s2*.tmp*.npz')):
        os.remove(stale)
        print(f'[BAND-DRIVER] removed stale worker temp '
              f'{os.path.basename(stale)}', flush=True)

    result = None
    for rnd in range(1, args.max_rounds + 1):
        grid = s2_ladder(s2_target, mom['mbar2'])
        s2s, tables, files = read_band_tables(
            args.data_dir, args.tag, expect_sha=sha,
            expect_tol=(args.rtol, args.atol))
        jobs = plan_missing(tables, grid, n_hi)
        todo = sum(len(j[0]) * (j[2] - j[1] + 1) for j in jobs)
        print(f"[ROUND {rnd}] target: {len(grid)} slices to s2={grid[-1]:g}, "
              f"waves 2..{n_hi}  ({todo} missing wave-points)", flush=True)
        if jobs:
            run_jobs(args, jobs, rnd)
            s2s, tables, files = read_band_tables(
                args.data_dir, args.tag, expect_sha=sha,
                expect_tol=(args.rtol, args.atol))
        # the watcher rectangle = the target rectangle, freshly complete
        s2s = np.asarray([s2 for s2 in s2s if s2 <= grid[-1] + 1e-9])
        ns = np.arange(2, n_hi + 1)
        for s2 in s2s:
            missing = [n for n in ns if n not in tables[s2]]
            if missing:
                raise RuntimeError(f'[ABORT BAND-DRIVER] slice s2={s2:g} still '
                                   f'missing waves {missing[:6]}... after its '
                                   f'round -- engine did not deliver.')

        ev = evaluate_watchers(s2s, tables, ns, mom, args,
                      bg['R'], bg['trDW3'],
                      fit_lo_pin=(args.pin_fit_lo if pinned else None))
        meas, onset, fit = ev['measure'], ev['onset'], ev['fit']
        print(f"[WATCH S2-PLATEAU] E_pred = shape_miss x envelope = "
              f"{meas['shape_miss']:.2e} x {meas['envelope']:.3f} = "
              f"{meas['E_pred']:.2e} (target {args.eps_s2:g}); "
              f"c_eff/c = {meas['c_eff_ratio']:.3f}", flush=True)
        if onset is None:
            print(f"[WATCH N-ONSET] no sustained {args.onset_tol:g}-match of "
                  f"nu^3 I_n(inf) to A3 yet (deepest wave {n_hi})", flush=True)
        else:
            print(f"[WATCH N-ONSET] onset = {onset} (fit window "
                  f"[{onset},{n_hi}])", flush=True)
        if fit:
            print(f"[WATCH N-TAIL-BOUND] |c|zeta5+|e|zeta7 = {fit['sub']:.2e} "
                  f"(target {args.eps_n:g}); fitted a/A3-1 = "
                  f"{fit['a']/mom['A3']-1.0:+.4f}", flush=True)

        if pinned:
            # regression pin: the rectangle is fixed -- accept after the one
            # complete round; the downstream guards do the verifying
            result = dict(ev=ev, s2s=s2s, ns=ns, rounds=rnd)
            break
        # the s2 stop = the SAME two conditions the S2-BAACKE guard verifies:
        # predicted remaining error inside target AND the raw truncation on
        # the Baacke law (NaN ratio = vanishing analytic c: E_pred-only)
        ratio = meas['c_eff_ratio']
        s2_ok = (meas['E_pred'] <= args.eps_s2
                 and (np.isnan(ratio) or 0.5 <= ratio <= 1.5))
        n_ok = fit is not None and fit['sub'] <= args.eps_n
        if s2_ok and n_ok:
            result = dict(ev=ev, s2s=s2s, ns=ns, rounds=rnd)
            break
        if not s2_ok:
            s2_target = ladder_snap(min(args.s2_cap, 1.5 * s2s[-1]),
                                    mom['mbar2'])
            if s2s[-1] >= args.s2_cap:
                raise RuntimeError(f'[ABORT BAND-DRIVER] s2 cap '
                                   f'{args.s2_cap:g} reached without an '
                                   f'S2-PLATEAU pass.')
        if not n_ok:
            if fit:
                n_hi = max(n_hi + 10,
                           watch_n_max_needed(fit['c'], fit['e'],
                                                     args.eps_n))
            else:
                n_hi = int(np.ceil(1.25 * n_hi))
            if n_hi > args.n_cap:
                raise RuntimeError(f'[ABORT BAND-DRIVER] n cap {args.n_cap} '
                                   f'reached without an N-TAIL-BOUND pass.')
    if result is None:
        raise RuntimeError(f'[ABORT BAND-DRIVER] no watcher convergence in '
                           f'{args.max_rounds} rounds.')

    ev, s2s, ns = result['ev'], result['s2s'], result['ns']
    meas, onset, fit = ev['measure'], ev['onset'], ev['fit']
    paths = stage_paths(args.data_dir, args.tag)
    # the manifest lists ONLY the final rectangle's slice files: data beyond
    # the watcher-chosen cutoff (a resumed dir with deeper slices) stays on
    # disk but is excluded here and by every downstream read (s2_cap)
    _, _, rect_files = read_band_tables(args.data_dir, args.tag,
                                        expect_sha=sha,
                                        expect_tol=(args.rtol, args.atol),
                                        s2_cap=float(s2s[-1]))
    manifest = os.path.join(args.data_dir, f'band_manifest_{args.tag}.json')
    with open(manifest, 'w') as fh:
        json.dump({'slices': sorted(os.path.basename(f)
                                    for f in rect_files)}, fh, indent=1)
    fit_lo = (args.pin_fit_lo if pinned and args.pin_fit_lo is not None
              else onset)
    if fit_lo is None or fit is None:
        raise RuntimeError('[ABORT BAND-DRIVER] the pinned rectangle is too '
                           'small for the N-ONSET watcher to place a fit '
                           'window -- give --pin-fit-lo or enlarge the pin.')
    atomic_savez(paths['band_cutoffs'],
                 n_max=int(ns[-1]), s2_max=float(s2s[-1]), fit_lo=int(fit_lo),
                 A3=mom['A3'], baacke_c=mom['c'], mbar2=mom['mbar2'],
                 r_W=mom['r_W'],
                 eps_s2=args.eps_s2, eps_n=args.eps_n,
                 onset_tol=args.onset_tol,
                 watch_E_pred=meas['E_pred'],
                 watch_c_eff_ratio=meas['c_eff_ratio'],
                 watch_sub_bound=fit['sub'],
                 fit_a=fit['a'], fit_c=fit['c'], fit_e=fit['e'],
                 rounds=result['rounds'],
                 rtol=args.rtol, atol=args.atol,
                 **provenance_stamp(args.bounce_npz, bg['m2']))
    print(f"[BAND-DRIVER] converged in {result['rounds']} round(s), "
          f"{time.time()-t0:.0f}s: n_max={ns[-1]}  s2_max={s2s[-1]:g}  "
          f"fit_window=[{fit_lo},{ns[-1]}]")
    print(f"[OK] wrote {paths['band_cutoffs']} and {manifest}")


if __name__ == '__main__':
    main()
