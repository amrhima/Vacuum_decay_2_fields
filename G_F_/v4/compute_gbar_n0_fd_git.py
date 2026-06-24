#!/usr/bin/env python3
"""
compute_gbar_n0_fd_git.py  --  Compute the n=0 (s-wave) subtracted
fluctuation propagator trace gbar(n=0, s^2) for a two-field
false-vacuum-decay bounce, using the finite-difference (FD) operator
and a sparse-LU / Hutchinson estimator.

Input
-----
A bounce .npz file (default: bounce_data_F2_T0.npz) holding the radial
bounce profile and metadata: R, X_prime, Y_prime, params, false_vac,
false_index, true_index.

Method
------
  1. Build the FD operator M_tilde for n=0 on a radial grid.
  2. Diagonalize to locate the single negative mode; its eigenvalue
     fixes the pole at s^2 = -lambda_neg.
  3. Build a 4-scale adaptive s^2 grid (refined around the pole, coarse
     in the tail) and scan it.
  4. At each s^2 estimate the raw trace Tr(A^-1) and the subtracted
     trace Tr(P A^-1 P) (negative mode projected out) with K Hutchinson
     samples (default K = 100000). The LU is well-conditioned at every
     sampled s^2 because the adaptive grid excludes the immediate pole.

Output
------
gbar_n0_fd_ver2_F{F}_T{T}.npz with fields:
    s2_grid, gbar_n0_raw, gbar_n0_raw_err, gbar_n0_sub,
    gbar_n0_sub_err, lambda_neg, pole_s2, chi_neg, r, dr, N, N2, K,
    estimator, skip_radius, false_index, true_index, bounce_file.
"""

import argparse
import os
import sys
import time

import numpy as np

from config import DATA_DIR

# [vx-opt] gbar_raw_and_sub_fd: one factorization per s^2 serves raw + sub.
from fd_builder_n0_git import (
    build_M_tilde_clean,
    find_negative_mode,
    gbar_raw_fd,
    gbar_sub_fd,
    gbar_raw_and_sub_fd,
    load_bounce,
)
from potential_git import CTShiftedLiftedPotential


# --------------------------------------------------------------------------- #
#   s^2 grid builder (adaptive around the pole)                               #
# --------------------------------------------------------------------------- #

def build_s2_grid(alpha_est, s2_min=0.01, s2_max=10.0,
                  coarse_step=0.1, medium_step=0.005, fine_step=0.001,
                  very_fine_step=0.0002,
                  skip_radius=3e-4,
                  delta_wide=0.15, delta_medium=0.05, delta_narrow=0.01,
                  tail_start=10.0, tail_step=5.0):
    """
    Same 4-scale adaptive s^2 grid as compute_gbar_n0.py (RK method),
    plus a coarse tail (step = tail_step) for s^2 in (tail_start, s2_max]
    so that extending s2_max from 10 to 50 does not multiply the grid
    size by 50.
    """
    if s2_min >= s2_max:
        raise ValueError(f"s2_min ({s2_min}) must be < s2_max ({s2_max})")
    for name, val in (("coarse_step", coarse_step), ("medium_step", medium_step),
                      ("fine_step", fine_step), ("very_fine_step", very_fine_step)):
        if val <= 0:
            raise ValueError(f"{name} must be > 0, got {val}")
    if skip_radius < 0:
        raise ValueError(f"skip_radius must be >= 0, got {skip_radius}")

    vals = []

    def add_range(lo, hi, step):
        if hi > lo and step > 0:
            vals.extend(np.arange(lo, hi + 0.5 * step, step).tolist())

    # below pole
    add_range(s2_min, alpha_est - delta_wide, coarse_step)
    add_range(alpha_est - delta_wide, alpha_est - delta_medium, medium_step)
    add_range(alpha_est - delta_medium, alpha_est - delta_narrow, fine_step)
    add_range(alpha_est - delta_narrow, alpha_est - skip_radius, very_fine_step)

    # above pole
    add_range(alpha_est + skip_radius, alpha_est + delta_narrow, very_fine_step)
    add_range(alpha_est + delta_narrow, alpha_est + delta_medium, fine_step)
    add_range(alpha_est + delta_medium, alpha_est + delta_wide, medium_step)
    # Fine portion of the high-s^2 region runs up to tail_start (default 10)
    high_fine_upper = min(tail_start, s2_max)
    add_range(alpha_est + delta_wide,
              high_fine_upper + 0.5 * coarse_step, coarse_step)
    # Coarse tail beyond tail_start (default step 5, for s2_max=50 -> 8 points)
    if s2_max > tail_start:
        add_range(tail_start + tail_step,
                  s2_max + 0.5 * tail_step, tail_step)

    vals.append(s2_min)
    vals.append(s2_max)

    arr = np.array(sorted(v for v in vals if s2_min <= v <= s2_max))
    tol = 0.5 * min(very_fine_step, skip_radius) if skip_radius > 0 else 0.5 * very_fine_step
    if arr.size > 1:
        keep = np.concatenate([[True], np.diff(arr) > tol])
        arr = arr[keep]
    if arr.size == 0:
        raise ValueError("s^2 grid ended up empty -- check step sizes and bounds.")
    return arr


# --------------------------------------------------------------------------- #
#   Bounce file resolver                                                      #
# --------------------------------------------------------------------------- #

def resolve_bounce(path, script_dir):
    if os.path.isfile(path):
        return path
    alt = os.path.join(script_dir, os.path.basename(path))
    if os.path.isfile(alt):
        return alt
    dp = os.path.join(DATA_DIR, os.path.basename(path))
    if os.path.isfile(dp):
        return dp
    raise FileNotFoundError(f"Bounce file not found: {path}")


# --------------------------------------------------------------------------- #
#   Main                                                                      #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Compute gbar(n=0, s^2) via the FD operator (K=100000 default)."
    )
    parser.add_argument("--bounce", default="bounce_data_F2_T0.npz")
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--r-min", type=float, default=1e-4)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--K", type=int, default=100000,
                        help="Hutchinson samples per s^2 (vfinal4 default: 100000).")
    parser.add_argument("--s2-min", type=float, default=0.01)
    parser.add_argument("--s2-max", type=float, default=1000.0,
                        help="Upper s^2 cutoff (default: 1000.0).")
    parser.add_argument("--s2-tail-start", type=float, default=10.0,
                        help="s^2 above which the coarse tail step is used.")
    parser.add_argument("--s2-tail-step", type=float, default=5.0,
                        help="Coarse step in the high-s^2 tail.")
    parser.add_argument("--skip-radius", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--estimator", choices=["rank1", "projected"],
                        default="rank1",
                        help="FD subtracted-trace estimator: 'rank1' (vx-opt: "
                             "Tr A^-1 - chi^T A^-1 chi, fast but high variance "
                             "near a zero/negative mode) or 'projected' (OLD: "
                             "Hutchinson Tr(P A^-1 P), low variance). Default rank1.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bounce_path = resolve_bounce(args.bounce, script_dir)
    print(f"[INFO] Bounce file: {bounce_path}")

    data_dir = args.data_dir or DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    print(f"[INFO] Output directory: {data_dir}")
    print(f"[INFO] settings: K = {args.K}")

    b = load_bounce(bounce_path)
    pot_lin = CTShiftedLiftedPotential(b["params"], b["false_vac"])

    # build operator
    print(f"\n[STEP 1] Building M_tilde (n=0) on {args.N} radial points ...")
    t0 = time.time()
    M_tilde, r, dr, U_pp = build_M_tilde_clean(
        n=0,
        R_bounce=b["R"],
        X_prime_bounce=b["X_prime"],
        Y_prime_bounce=b["Y_prime"],
        pot_lin=pot_lin,
        N=args.N,
        r_min=args.r_min,
        r_max=args.r_max,
        fd_order=2,
    )
    N2 = M_tilde.shape[0]
    print(f"  matrix shape: {M_tilde.shape}, nnz: {M_tilde.nnz}, dr: {dr:.6f}")
    print(f"  built in {time.time() - t0:.1f}s")

    boundary_indices = np.array([0, args.N - 1, args.N, 2 * args.N - 1])

    # find negative mode
    print(f"\n[STEP 2] Diagonalizing M_tilde to find negative mode ...")
    t0 = time.time()
    lambda_neg, chi_neg = find_negative_mode(M_tilde, verbose=True)
    print(f"  found in {time.time() - t0:.1f}s")
    if lambda_neg >= 0:
        print("[ERROR] No negative eigenvalue found. Aborting.")
        sys.exit(1)

    # clamp eigensolver noise at boundary DOFs and renormalize
    chi_neg[boundary_indices] = 0.0
    chi_neg = chi_neg / np.linalg.norm(chi_neg)

    pole_s2 = -lambda_neg
    print(f"  pole at s^2 = -lambda_neg = {pole_s2:.8f}")

    s2_grid = build_s2_grid(pole_s2,
                            s2_min=args.s2_min,
                            s2_max=args.s2_max,
                            skip_radius=args.skip_radius,
                            tail_start=args.s2_tail_start,
                            tail_step=args.s2_tail_step)
    n_s2 = len(s2_grid)
    print(f"\n[STEP 3] s^2 grid: {n_s2} points in [{s2_grid[0]:.3f}, {s2_grid[-1]:.3f}]")

    # main scan
    gbar_raw = np.full(n_s2, np.nan)
    gbar_raw_err = np.full(n_s2, np.nan)
    gbar_sub = np.full(n_s2, np.nan)
    gbar_sub_err = np.full(n_s2, np.nan)

    rng = np.random.default_rng(args.seed)

    print(f"\n[STEP 4] Scan over s^2 (K = {args.K}) ...")
    t_scan_start = time.time()
    for i, s2 in enumerate(s2_grid):
        t0 = time.time()
        # [vx-opt] Outside skip_radius the raw trace is well-defined, so one
        # combined call factorizes A once and returns both raw and subtracted
        # (rank-1 identity Tr(P A^-1 P) = Tr(A^-1) - chi^T A^-1 chi). Inside
        # skip_radius the raw trace is NaN by design, so keep the
        # gbar_sub_fd / hutchinson_trace_projected fallback for the subtracted
        # value. NaNs propagate (fail-loud), never masked.
        if args.estimator == "projected" or abs(s2 - pole_s2) < args.skip_radius:
            raw_val, raw_err = np.nan, np.nan
            sub_val, sub_err = gbar_sub_fd(
                M_tilde, s2, dr, N2, chi_neg, K=args.K, rng=rng,
                boundary_indices=boundary_indices,
            )
        else:
            (raw_val, raw_err), (sub_val, sub_err) = gbar_raw_and_sub_fd(
                M_tilde, s2, dr, N2, chi_neg, K=args.K, rng=rng,
                boundary_indices=boundary_indices,
            )

        gbar_raw[i], gbar_raw_err[i] = raw_val, raw_err
        gbar_sub[i], gbar_sub_err[i] = sub_val, sub_err

        dt = time.time() - t0
        raw_str = "nan" if np.isnan(raw_val) else f"{raw_val:+.3e}"
        sub_str = "nan" if np.isnan(sub_val) else f"{sub_val:+.3e}"
        print(f"  [{i+1:3d}/{n_s2}]  s2={s2:.6f}  "
              f"raw={raw_str}  sub={sub_str}  ({dt:.1f}s)")

    print(f"\n[INFO] Scan finished in {time.time() - t_scan_start:.0f}s")

    out = args.out
    if out is None:
        out = os.path.join(
            data_dir,
            f"gbar_n0_fd_ver2_F{b['false_index']}_T{b['true_index']}.npz",
        )
    np.savez(
        out,
        s2_grid=s2_grid,
        gbar_n0_raw=gbar_raw,
        gbar_n0_raw_err=gbar_raw_err,
        gbar_n0_sub=gbar_sub,
        gbar_n0_sub_err=gbar_sub_err,
        lambda_neg=lambda_neg,
        pole_s2=pole_s2,
        chi_neg=chi_neg,
        r=r,
        dr=dr,
        N=args.N,
        N2=N2,
        K=args.K,
        estimator=args.estimator,
        skip_radius=args.skip_radius,
        false_index=b['false_index'],
        true_index=b['true_index'],
        bounce_file=str(bounce_path),
    )
    print(f"[SAVE] {out}")
    print("[INFO] To visualize, run: python3 plot_gbar_n0_fd_ver2.py")


if __name__ == "__main__":
    main()
