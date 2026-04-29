#!/usr/bin/env python3
"""
compute_gbar_n1_fd_ver2.py  --  Same as compute_gbar_n1_fd.py with two v2
modifications:

  * The translation zero mode chi_zm is obtained from eigsh on the discrete
    operator M_tilde (via find_discrete_zero_mode -- closest-to-zero
    eigenvalue), instead of the analytic formula r^(3/2)*phi_b'(r). This
    makes the projector match the discrete operator exactly, removing the
    residual hairpin in G_bar^sub_n=1 right at the (discrete) zero-mode
    pole.
  * K = 400 by default (4x more Hutchinson samples -> ~2x smaller SEM).

The gbar_sub_fd code path itself is identical to v1 (sparse LU).
The script also performs a sanity check on the returned chi_zm: it
overlaps with the analytic r^(3/2)*phi_b' mode and warns if the
overlap is low, to catch the case of a spurious non-translational
mode being picked.

Output: gbar_n1_fd_ver2_F{F}_T{T}.npz
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fd_builder_n1_v2 import (
    build_M_tilde_clean,
    build_translation_zero_mode,
    find_discrete_zero_mode,
    gbar_raw_fd,
    gbar_sub_fd,
    load_bounce,
)
from potential import CTShiftedLiftedPotential
from config import DATA_DIR


# --------------------------------------------------------------------------- #
#   s^2 grid (log-spaced near 0, linear far from 0)                           #
# --------------------------------------------------------------------------- #

def build_s2_grid_n1(s2_min=1e-6, s2_max=10.0,
                     log_ppd=8, s2_transition=0.1, linear_step=0.1):
    """
    Same grid as compute_gbar_n1.py (RK method).
    """
    if s2_min <= 0:
        raise ValueError(f"s2_min must be > 0 for geomspace, got {s2_min}")
    if s2_min >= s2_max:
        raise ValueError(f"s2_min ({s2_min}) must be < s2_max ({s2_max})")
    if linear_step <= 0:
        raise ValueError(f"linear_step must be > 0, got {linear_step}")
    if log_ppd <= 0:
        raise ValueError(f"log_ppd must be > 0, got {log_ppd}")

    parts = [np.array([s2_min, s2_max])]
    if s2_transition <= s2_min:
        parts.append(np.arange(s2_min, s2_max + 0.5 * linear_step, linear_step))
    elif s2_transition >= s2_max:
        n_decades = np.log10(s2_max / s2_min)
        n_log = max(int(n_decades * log_ppd), 10)
        parts.append(np.geomspace(s2_min, s2_max, n_log))
    else:
        n_decades = np.log10(s2_transition / s2_min)
        n_log = max(int(n_decades * log_ppd), 10)
        parts.append(np.geomspace(s2_min, s2_transition, n_log))
        parts.append(np.arange(s2_transition + linear_step,
                               s2_max + 0.5 * linear_step, linear_step))

    grid = np.unique(np.concatenate(parts))
    grid = grid[(grid >= s2_min) & (grid <= s2_max)]
    if grid.size == 0:
        raise ValueError("s^2 grid ended up empty -- check bounds.")
    return grid


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
        description="Compute gbar(n=1, s^2) (v2: discrete chi_zm via eigsh, K=400 default)."
    )
    parser.add_argument("--bounce", default="bounce_data_F2_T0.npz")
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--r-min", type=float, default=1e-4)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--K", type=int, default=400,
                        help="Hutchinson samples per s^2 (v2 default: 400).")
    parser.add_argument("--zm-overlap-warn", type=float, default=0.95,
                        help="Warn if |<chi_zm_eigsh | chi_zm_analytic>| "
                             "drops below this (default 0.95) -- safeguard "
                             "against picking a non-translational mode.")
    parser.add_argument("--zm-eigvalue-warn", type=float, default=0.1,
                        help="Warn if |lambda_zm| > this (default 0.1) -- "
                             "the discrete zero-mode eigenvalue should be "
                             "small (O(dr^2)).")
    parser.add_argument("--strict-zm", action="store_true",
                        help="Promote the chi_zm sanity warnings to hard "
                             "errors: abort the run if the picked mode's "
                             "overlap with the analytic translation mode is "
                             "below --zm-overlap-warn or |lambda_zm| exceeds "
                             "--zm-eigvalue-warn. Recommended for production.")
    parser.add_argument("--s2-min", type=float, default=1e-6)
    parser.add_argument("--s2-max", type=float, default=10.0)
    parser.add_argument("--s2-transition", type=float, default=0.1)
    parser.add_argument("--log-ppd", type=int, default=8)
    parser.add_argument("--linear-step", type=float, default=0.1)
    parser.add_argument("--skip-radius", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=23456)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bounce_path = resolve_bounce(args.bounce, script_dir)
    print(f"[INFO] Bounce file: {bounce_path}")

    data_dir = args.data_dir
    if data_dir is None:
        dp = DATA_DIR
        data_dir = dp if os.path.isdir(dp) else script_dir
    os.makedirs(data_dir, exist_ok=True)
    print(f"[INFO] Output directory: {data_dir}")
    print(f"[INFO] v2 settings: K = {args.K}, chi_zm via eigsh "
          f"(closest-to-zero eigenvalue)")

    b = load_bounce(bounce_path)
    pot_lin = CTShiftedLiftedPotential(b["params"], b["false_vac"])

    print(f"\n[STEP 1] Building M_tilde (n=1) on {args.N} radial points ...")
    t0 = time.time()
    M_tilde, r, dr, U_pp = build_M_tilde_clean(
        n=1,
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

    # v2: use eigsh-derived discrete near-zero mode instead of analytic chi_zm
    print(f"\n[STEP 2] Diagonalizing M_tilde to find discrete zero mode ...")
    t0 = time.time()
    lambda_zm, chi_zm = find_discrete_zero_mode(M_tilde, verbose=True)
    print(f"  found in {time.time() - t0:.1f}s")
    print(f"  discrete zero-mode eigenvalue lambda_zm = {lambda_zm:+.6e}")
    print(f"  -> pole expected at s^2 = -lambda_zm = {-lambda_zm:+.6e}")

    # clamp eigsh boundary noise and renormalize
    chi_zm[boundary_indices] = 0.0
    chi_zm = chi_zm / np.linalg.norm(chi_zm)

    # validation: cross-check against analytic translation zero mode
    chi_zm_analytic = build_translation_zero_mode(
        r, b["R"], b["X_prime"], b["Y_prime"]
    )
    chi_zm_analytic[boundary_indices] = 0.0
    chi_zm_analytic = chi_zm_analytic / np.linalg.norm(chi_zm_analytic)
    overlap = float(abs(chi_zm @ chi_zm_analytic))
    print(f"  |<chi_zm_eigsh | chi_zm_analytic>| = {overlap:.6f}  "
          f"(expect ~1)")
    bad_overlap = overlap < args.zm_overlap_warn
    bad_eigval  = abs(lambda_zm) > args.zm_eigvalue_warn
    if bad_overlap:
        msg = (f"  overlap = {overlap:.4f} below threshold "
               f"{args.zm_overlap_warn}: the picked eigenvector may not "
               f"be a translation mode. Inspect the eigsh diagnostic above.")
        if args.strict_zm:
            print(f"  [ERROR-STRICT] {msg}")
        else:
            print(f"  [WARN] {msg}  (continuing; pass --strict-zm to abort.)")
    if bad_eigval:
        msg = (f"  |lambda_zm| = {abs(lambda_zm):.3e} exceeds threshold "
               f"{args.zm_eigvalue_warn}: the picked eigenvalue is not "
               f"close to zero. Inspect the eigsh diagnostic above.")
        if args.strict_zm:
            print(f"  [ERROR-STRICT] {msg}")
        else:
            print(f"  [WARN] {msg}  (continuing; pass --strict-zm to abort.)")
    if args.strict_zm and (bad_overlap or bad_eigval):
        print("[ABORT] --strict-zm set and chi_zm sanity check(s) failed. "
              "Re-run without --strict-zm to override, or fix the inputs.")
        sys.exit(1)

    pole_s2 = -lambda_zm

    s2_grid = build_s2_grid_n1(s2_min=args.s2_min,
                               s2_max=args.s2_max,
                               s2_transition=args.s2_transition,
                               log_ppd=args.log_ppd,
                               linear_step=args.linear_step)
    n_s2 = len(s2_grid)
    print(f"\n[STEP 3] s^2 grid: {n_s2} points in [{s2_grid[0]:.2e}, {s2_grid[-1]:.3f}]")

    gbar_raw = np.full(n_s2, np.nan)
    gbar_raw_err = np.full(n_s2, np.nan)
    gbar_sub = np.full(n_s2, np.nan)
    gbar_sub_err = np.full(n_s2, np.nan)

    rng = np.random.default_rng(args.seed)
    degeneracy = (1 + 1) ** 2

    print(f"\n[STEP 4] Scan over s^2 (K = {args.K}) ...")
    t_scan_start = time.time()
    for i, s2 in enumerate(s2_grid):
        t0 = time.time()

        # raw: skip if too close to discrete zero-mode pole
        if abs(s2 - pole_s2) < args.skip_radius:
            raw_val, raw_err = np.nan, np.nan
        else:
            raw_val, raw_err = gbar_raw_fd(
                M_tilde, s2, dr, N2, K=args.K, rng=rng,
                boundary_indices=boundary_indices,
            )
        # subtracted: discrete chi_zm projected analytically; sparse LU.
        sub_val, sub_err = gbar_sub_fd(
            M_tilde, s2, dr, N2, chi_zm, K=args.K, rng=rng,
            boundary_indices=boundary_indices,
        )

        if np.isfinite(raw_val):
            raw_val *= degeneracy
            raw_err *= degeneracy
        if np.isfinite(sub_val):
            sub_val *= degeneracy
            sub_err *= degeneracy

        gbar_raw[i], gbar_raw_err[i] = raw_val, raw_err
        gbar_sub[i], gbar_sub_err[i] = sub_val, sub_err

        dt = time.time() - t0
        raw_str = "nan" if np.isnan(raw_val) else f"{raw_val:+.3e}"
        sub_str = "nan" if np.isnan(sub_val) else f"{sub_val:+.3e}"
        print(f"  [{i+1:3d}/{n_s2}]  s2={s2:.6e}  "
              f"raw={raw_str}  sub={sub_str}  ({dt:.1f}s)")

    print(f"\n[INFO] Scan finished in {time.time() - t_scan_start:.0f}s")

    out = args.out
    if out is None:
        out = os.path.join(
            data_dir,
            f"gbar_n1_fd_ver2_F{b['false_index']}_T{b['true_index']}.npz",
        )
    np.savez(
        out,
        s2_grid=s2_grid,
        gbar_n1_raw=gbar_raw,
        gbar_n1_raw_err=gbar_raw_err,
        gbar_n1_sub=gbar_sub,
        gbar_n1_sub_err=gbar_sub_err,
        chi_zm=chi_zm,
        lambda_zm=lambda_zm,
        pole_s2=pole_s2,
        degeneracy=degeneracy,
        r=r,
        dr=dr,
        N=args.N,
        N2=N2,
        K=args.K,
        zm_overlap=overlap,
        skip_radius=args.skip_radius,
        false_index=b['false_index'],
        true_index=b['true_index'],
        bounce_file=str(bounce_path),
    )
    print(f"[SAVE] {out}")
    print("[INFO] To visualize, run: python3 plot_gbar_n1_fd_ver2.py")


if __name__ == "__main__":
    main()
