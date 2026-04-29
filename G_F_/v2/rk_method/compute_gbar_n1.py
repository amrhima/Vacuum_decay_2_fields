#!/usr/bin/env python3
"""
compute_gbar_n1.py -- Compute G_bar(n=1, s2) on the bounce background.

The n=1 sector contains the 4 translation zero modes (degeneracy (1+1)^2 = 4).
G_bar(n=1, s2) diverges as 4/s2 when s2 -> 0.

Computes:
  - G_bar(n=1, s2) unsubtracted
  - det(Omega) and cond(Omega) vs s2  (Wronskian diagnostics)
  - s_star: the fitted pole shift from s2=0
  - G_bar_sub = G_bar - 4/(s2 - s_star)  (zero mode subtracted)

Uses the lightweight RK approach from rk_builder_adapt_v2 (no full G matrix saved).

Output:  gbar_n1_scan_F{F}_T{T}.npz
"""

import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rk_builder_adapt_v2 import build_fluctuation_data_prime, solve_branch
from potential import CTShiftedLiftedPotential
from config import DATA_DIR


# ------------------------------------------------------------------ #
#  Reuse the same lightweight G_bar engine from compute_gbar_n0       #
# ------------------------------------------------------------------ #

def compute_gbar_single(pot_lin, r_bounce, x_prime, y_prime,
                        s2, n_mode=1, n_eval=2000, r0=1e-4):
    """
    Compute G_bar and det(Omega) for a single s2 value at given n_mode.
    Returns (gbar, det_omega, cond).
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        (B, dB, K_matrix, A_i,
         _U, _Mfree, _xp, _yp) = build_fluctuation_data_prime(
            pot_lin, r_bounce, x_prime, y_prime, s2, n_mode)

    r_start = max(float(r_bounce[0]), r0)
    r_max   = float(r_bounce[-1])
    r_grid  = np.linspace(r_start, r_max, n_eval)
    y0      = np.zeros(4)

    _, y_m1 = solve_branch(r_start, r_max, y0, "-", 0, K_matrix, A_i, r_grid)
    _, y_m2 = solve_branch(r_start, r_max, y0, "-", 1, K_matrix, A_i, r_grid)

    r_desc = r_grid[::-1]
    _, y_p1 = solve_branch(r_max, r_start, y0, "+", 0, K_matrix, A_i, r_desc)
    _, y_p2 = solve_branch(r_max, r_start, y0, "+", 1, K_matrix, A_i, r_desc)
    y_p1 = y_p1[::-1, :]
    y_p2 = y_p2[::-1, :]

    nr = len(r_grid)

    h_m  = np.zeros((nr, 2, 2))
    dh_m = np.zeros((nr, 2, 2))
    h_p  = np.zeros((nr, 2, 2))
    dh_p = np.zeros((nr, 2, 2))

    h_m[:, 0, 0], h_m[:, 1, 0]  = y_m1[:, 0], y_m1[:, 1]
    dh_m[:, 0, 0], dh_m[:, 1, 0] = y_m1[:, 2], y_m1[:, 3]
    h_m[:, 0, 1], h_m[:, 1, 1]  = y_m2[:, 0], y_m2[:, 1]
    dh_m[:, 0, 1], dh_m[:, 1, 1] = y_m2[:, 2], y_m2[:, 3]

    h_p[:, 0, 0], h_p[:, 1, 0]  = y_p1[:, 0], y_p1[:, 1]
    dh_p[:, 0, 0], dh_p[:, 1, 0] = y_p1[:, 2], y_p1[:, 3]
    h_p[:, 0, 1], h_p[:, 1, 1]  = y_p2[:, 0], y_p2[:, 1]
    dh_p[:, 0, 1], dh_p[:, 1, 1] = y_p2[:, 2], y_p2[:, 3]

    f_p  = np.zeros((nr, 2, 2))
    f_m  = np.zeros((nr, 2, 2))
    df_p = np.zeros((nr, 2, 2))
    df_m = np.zeros((nr, 2, 2))

    for k, r in enumerate(r_grid):
        for i in range(2):
            for a in range(2):
                d = 1.0 if i == a else 0.0
                bp  = B(i, r, "+")
                dbp = dB(i, r, "+")
                f_p[k, i, a]  = bp * (d + h_p[k, i, a])
                df_p[k, i, a] = dbp * (d + h_p[k, i, a]) + bp * dh_p[k, i, a]
                bm  = B(i, r, "-")
                dbm = dB(i, r, "-")
                f_m[k, i, a]  = bm * (d + h_m[k, i, a])
                df_m[k, i, a] = dbm * (d + h_m[k, i, a]) + bm * dh_m[k, i, a]

    w_scaled = np.zeros((nr, 2, 2))
    for idx in range(nr):
        r = r_grid[idx]
        w = np.zeros((2, 2))
        for a in range(2):
            for b in range(2):
                s = 0.0
                for i in range(2):
                    s += (f_m[idx, i, a] * df_p[idx, i, b]
                          - f_p[idx, i, b] * df_m[idx, i, a])
                w[a, b] = -s
        w_scaled[idx] = r**3 * w

    r_lo = 0.05
    r_hi = 0.9 * r_grid[-1]
    i_lo = np.searchsorted(r_grid, r_lo)
    i_hi = np.searchsorted(r_grid, r_hi)
    omega = np.mean(w_scaled[i_lo:i_hi + 1], axis=0)

    det_omega = float(np.linalg.det(omega))
    cond = float(np.linalg.cond(omega))

    if cond > 1e14:
        return np.nan, det_omega, cond

    omega_inv = np.linalg.inv(omega)

    trace_diag = np.zeros(nr)
    for k in range(nr):
        G_diag = f_p[k] @ omega_inv @ f_m[k].T
        trace_diag[k] = np.trace(G_diag)

    integrand = r_grid**3 * trace_diag
    gbar = float(np.trapezoid(integrand, r_grid))

    # degeneracy factor (n+1)^2 = 4 for n=1
    gbar *= 4.0

    return gbar, det_omega, cond


# ------------------------------------------------------------------ #
#  s2 grid for n=1: log-spaced near s2=0, linear for larger s2       #
# ------------------------------------------------------------------ #

def build_s2_grid_n1(s2_min=1e-6, s2_max=10.0,
                     log_points_per_decade=8,
                     s2_transition=0.1,
                     linear_step=0.1):
    """
    Build s2 grid with log spacing near 0 (zero mode region)
    and linear spacing for larger s2.
    """
    n_decades = np.log10(s2_transition / s2_min)
    n_log = max(int(n_decades * log_points_per_decade), 10)
    log_part = np.geomspace(s2_min, s2_transition, n_log)

    lin_part = np.arange(s2_transition + linear_step,
                         s2_max + 0.5 * linear_step, linear_step)

    grid = np.unique(np.concatenate([log_part, lin_part, [s2_min, s2_max]]))
    return grid


# ------------------------------------------------------------------ #
#  Resolve bounce file                                                #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Main                                                              #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Compute G_bar(n=1, s2) on the bounce background."
    )
    parser.add_argument("--bounce", default="bounce_data_F2_T0.npz")
    parser.add_argument("--s2-min", type=float, default=1e-6)
    parser.add_argument("--s2-max", type=float, default=10.0)
    parser.add_argument("--s2-transition", type=float, default=0.1,
                        help="Switch from log to linear spacing at this s2.")
    parser.add_argument("--log-ppd", type=int, default=8,
                        help="Log-spaced points per decade below s2-transition.")
    parser.add_argument("--linear-step", type=float, default=0.1)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--r0", type=float, default=1e-4)
    parser.add_argument("--fit-min", type=float, default=0.005,
                        help="Lower bound of s2 window for pole fit.")
    parser.add_argument("--fit-max", type=float, default=0.05,
                        help="Upper bound of s2 window for pole fit.")
    parser.add_argument("--sub-threshold", type=float, default=1e-7,
                        help="|s2 - s_star| below which subtracted G_bar is NaN.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bounce_path = resolve_bounce(args.bounce, script_dir)
    print(f"[INFO] Bounce file: {bounce_path}")

    data_dir = args.data_dir or os.environ.get("TRG_DATA_DIR")
    if data_dir is None:
        dp = DATA_DIR
        if os.path.isdir(dp):
            data_dir = dp
        else:
            data_dir = script_dir
    os.makedirs(data_dir, exist_ok=True)
    print(f"[INFO] Output directory: {data_dir}")

    # --- load bounce data ---
    bdata = np.load(bounce_path, allow_pickle=True)
    params    = bdata["params"]
    false_vac = np.asarray(bdata["false_vac"], dtype=float)
    r_bounce  = bdata["R"]
    x_prime   = bdata["X_bounce_prime"]
    y_prime   = bdata["Y_bounce_prime"]
    false_idx = int(bdata["false_index"])
    true_idx  = int(bdata["true_index"])

    pot_lin = CTShiftedLiftedPotential(params, false_vac)

    # --- build s2 grid ---
    s2_grid = build_s2_grid_n1(args.s2_min, args.s2_max,
                                args.log_ppd, args.s2_transition,
                                args.linear_step)
    n_s2 = len(s2_grid)
    print(f"[INFO] {n_s2} s2 values in [{s2_grid[0]:.2e}, {s2_grid[-1]:.2e}]")
    print(f"[INFO] Zero mode pole expected near s2 = 0")

    # --- scan ---
    gbar_arr      = np.full(n_s2, np.nan)
    det_omega_arr = np.full(n_s2, np.nan)
    cond_arr      = np.full(n_s2, np.nan)

    t0 = time.time()
    for idx, s2 in enumerate(s2_grid):
        t1 = time.time()
        try:
            gb, det_om, cond = compute_gbar_single(
                pot_lin, r_bounce, x_prime, y_prime,
                s2, n_mode=1, n_eval=args.n_eval, r0=args.r0,
            )
            gbar_arr[idx]      = gb
            det_omega_arr[idx] = det_om
            cond_arr[idx]      = cond
            dt = time.time() - t1
            print(f"  [{idx+1:3d}/{n_s2}]  s2={s2:.6e}  "
                  f"G_bar={gb: .6e}  det(Om)={det_om: .6e}  "
                  f"cond={cond:.2e}  ({dt:.1f}s)")
        except Exception as exc:
            dt = time.time() - t1
            print(f"  [{idx+1:3d}/{n_s2}]  s2={s2:.6e}  FAILED: {exc}  ({dt:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n[INFO] Scan finished in {elapsed:.0f}s")

    # --- fit pole model: G_bar ~ A/(s2 - s_star) + B ---
    fit_mask = ((s2_grid >= args.fit_min) & (s2_grid <= args.fit_max)
                & np.isfinite(gbar_arr))
    if np.sum(fit_mask) < 3:
        raise RuntimeError(
            f"Only {np.sum(fit_mask)} points in fit window "
            f"[{args.fit_min}, {args.fit_max}]. "
            "Cannot fit pole model. Adjust --fit-min / --fit-max "
            "or check that the scan covers this range."
        )

    if np.sum(fit_mask) >= 3:
        s2_fit = s2_grid[fit_mask]
        gb_fit = gbar_arr[fit_mask]

        # fit G_bar = A/(s2 - s_star) + B  by least squares
        # rewrite as: G_bar * (s2 - s_star) = A + B*(s2 - s_star)
        # use iterative approach: start with s_star=0, fit A and B,
        # then refine s_star
        best_rss = np.inf
        for s_trial in np.linspace(-0.01, 0.01, 201):
            denom = s2_fit - s_trial
            if np.any(np.abs(denom) < 1e-12):
                continue
            # fit: gb = A/denom + B  =>  gb*denom = A + B*denom
            X = np.column_stack([np.ones_like(denom), denom])
            y = gb_fit * denom
            try:
                coeffs, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
                A_try, B_try = coeffs
                pred = A_try / denom + B_try
                rss = np.sum((pred - gb_fit)**2)
                if rss < best_rss:
                    best_rss = rss
                    s_star = s_trial
                    A_fit = A_try
                    B_fit = B_try
            except Exception:
                continue

        print(f"[FIT] s_star = {s_star:.8f}")
        print(f"[FIT] A = {A_fit:.6f}  (expected ~4 for zero mode)")
        print(f"[FIT] B = {B_fit:.4f}")
        print(f"[FIT] RSS = {best_rss:.6e}")

    # --- subtracted G_bar: G_bar_sub = G_bar - A_fit/(s2 - s_star) ---
    gbar_sub = np.full(n_s2, np.nan)
    for idx in range(n_s2):
        s2 = s2_grid[idx]
        if np.isfinite(gbar_arr[idx]) and abs(s2 - s_star) > args.sub_threshold:
            gbar_sub[idx] = gbar_arr[idx] - A_fit / (s2 - s_star)

    n_valid     = np.sum(np.isfinite(gbar_arr))
    n_valid_sub = np.sum(np.isfinite(gbar_sub))
    print(f"[INFO] Valid G_bar points: {n_valid}/{n_s2}")
    print(f"[INFO] Valid G_bar_sub points: {n_valid_sub}/{n_s2}")

    # --- save ---
    out_name = args.out
    if out_name is None:
        out_name = os.path.join(
            data_dir,
            f"gbar_n1_scan_F{false_idx}_T{true_idx}.npz",
        )
    out_dir_check = os.path.dirname(out_name) or "."
    if not os.path.isdir(out_dir_check):
        fallback = os.path.join(script_dir,
                                f"gbar_n1_scan_F{false_idx}_T{true_idx}.npz")
        print(f"[WARN] {out_dir_check} not available, saving to {fallback}")
        out_name = fallback

    np.savez(
        out_name,
        s2_grid=s2_grid,
        gbar_n1=gbar_arr,
        gbar_n1_sub=gbar_sub,
        det_omega=det_omega_arr,
        cond_omega=cond_arr,
        s_star=s_star,
        A_fit=A_fit,
        B_fit=B_fit,
        bounce_file=str(bounce_path),
        false_index=false_idx,
        true_index=true_idx,
        n_eval=args.n_eval,
        r0=args.r0,
        fit_min=args.fit_min,
        fit_max=args.fit_max,
        sub_threshold=args.sub_threshold,
    )
    print(f"[SAVE] {out_name}")
    print("[DONE]")


if __name__ == "__main__":
    main()
