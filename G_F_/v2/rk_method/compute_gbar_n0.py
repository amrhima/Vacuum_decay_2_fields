#!/usr/bin/env python3
"""
compute_gbar_n0.py -- Compute G_bar(n=0, s2) on the bounce background.

For each s2 value, solves the h-basis ODE via rk_builder_adapt_v2,
builds the radial Green's function diagonal G(r,r), and integrates:

    G_bar(n=0, s2) = integral_0^Rmax dr r^3 Tr[G_0(r,r,s2)]

Also computes:
  - det(Omega(s2)) to locate the negative eigenvalue  (pole at s2 = alpha)
  - Subtracted:  G_bar_sub = G_bar - 1/(s2 - alpha)   (pole removed)

Output:  gbar_n0_scan_F{F}_T{T}.npz  on the DP drive (or locally).
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
#  Lightweight G_bar computation (no full G matrix stored)           #
# ------------------------------------------------------------------ #

def compute_gbar_single(pot_lin, r_bounce, x_prime, y_prime,
                        s2, n_mode=0, n_eval=2000, r0=1e-4):
    """
    Compute G_bar(n=0, s2) and det(Omega) for one s2 value.

    Uses the same ODE solver as rk_builder_adapt_v2 but only
    computes the diagonal Tr[G(r,r)] -- avoids storing the full
    (nr x nr x 2 x 2) Green's-function matrix.

    Returns
    -------
    gbar      : float   (or NaN if omega is singular)
    det_omega : float
    cond      : float   (condition number of Omega)
    """
    # --- fluctuation data (suppressed stdout) ---
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        (B, dB, K_matrix, A_i,
         _U, _Mfree, _xp, _yp) = build_fluctuation_data_prime(
            pot_lin, r_bounce, x_prime, y_prime, s2, n_mode)

    r_start = max(float(r_bounce[0]), r0)
    r_max   = float(r_bounce[-1])
    r_grid  = np.linspace(r_start, r_max, n_eval)
    y0      = np.zeros(4)

    # --- solve 4 ODE branches ---
    _, y_m1 = solve_branch(r_start, r_max, y0, "-", 0, K_matrix, A_i, r_grid)
    _, y_m2 = solve_branch(r_start, r_max, y0, "-", 1, K_matrix, A_i, r_grid)

    r_desc = r_grid[::-1]
    _, y_p1 = solve_branch(r_max, r_start, y0, "+", 0, K_matrix, A_i, r_desc)
    _, y_p2 = solve_branch(r_max, r_start, y0, "+", 1, K_matrix, A_i, r_desc)
    y_p1 = y_p1[::-1, :]
    y_p2 = y_p2[::-1, :]

    nr = len(r_grid)

    # --- assemble h matrices ---
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

    # --- build f_plus (decaying) and f_minus (regular) ---
    f_p  = np.zeros((nr, 2, 2))
    f_m  = np.zeros((nr, 2, 2))
    df_p = np.zeros((nr, 2, 2))
    df_m = np.zeros((nr, 2, 2))

    for k, r in enumerate(r_grid):
        for i in range(2):
            for a in range(2):
                d = 1.0 if i == a else 0.0
                # plus (decaying, K-Bessel)
                bp  = B(i, r, "+")
                dbp = dB(i, r, "+")
                f_p[k, i, a]  = bp * (d + h_p[k, i, a])
                df_p[k, i, a] = dbp * (d + h_p[k, i, a]) + bp * dh_p[k, i, a]
                # minus (regular, I-Bessel)
                bm  = B(i, r, "-")
                dbm = dB(i, r, "-")
                f_m[k, i, a]  = bm * (d + h_m[k, i, a])
                df_m[k, i, a] = dbm * (d + h_m[k, i, a]) + bm * dh_m[k, i, a]

    # --- Wronskian  C^{alpha beta} = r^3 W^{alpha beta} ---
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

    # --- Omega = plateau average of r^3 W ---
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

    # --- diagonal trace:  Tr[G(r,r)] = Tr[ f_plus(r) @ omega_inv @ f_minus(r)^T ] ---
    trace_diag = np.zeros(nr)
    for k in range(nr):
        G_diag = f_p[k] @ omega_inv @ f_m[k].T
        trace_diag[k] = np.trace(G_diag)

    integrand = r_grid**3 * trace_diag
    gbar = float(np.trapezoid(integrand, r_grid))

    return gbar, det_omega, cond


# ------------------------------------------------------------------ #
#  Adaptive s2 grid builder                                          #
# ------------------------------------------------------------------ #

def build_adaptive_s2_grid(alpha,
                           s2_min=0.01, s2_max=5.0,
                           coarse_step=0.1,
                           medium_step=0.005,
                           fine_step=0.001,
                           very_fine_step=0.0002,
                           delta_wide=0.15,
                           delta_medium=0.05,
                           delta_narrow=0.01,
                           skip_radius=0.0003):
    """
    Build an s2 grid that is very dense around alpha (the negative-mode
    pole) and coarse far away.

    Regions (schematic, all symmetric around alpha):
      [s2_min, alpha - delta_wide]          coarse_step
      [alpha - delta_wide, alpha - delta_medium]  medium_step
      [alpha - delta_medium, alpha - delta_narrow] fine_step
      [alpha - delta_narrow, alpha - skip_radius]  very_fine_step
      --- gap: |s2 - alpha| < skip_radius  (excluded) ---
      [alpha + skip_radius, alpha + delta_narrow]  very_fine_step
      [alpha + delta_narrow, alpha + delta_medium] fine_step
      [alpha + delta_medium, alpha + delta_wide]  medium_step
      [alpha + delta_wide, s2_max]          coarse_step
    """
    vals = set()

    def add_range(lo, hi, step):
        if hi > lo and step > 0:
            vals.update(np.arange(lo, hi + 0.5 * step, step).tolist())

    # below the pole
    add_range(s2_min, alpha - delta_wide, coarse_step)
    add_range(alpha - delta_wide, alpha - delta_medium, medium_step)
    add_range(alpha - delta_medium, alpha - delta_narrow, fine_step)
    add_range(alpha - delta_narrow, alpha - skip_radius, very_fine_step)

    # above the pole
    add_range(alpha + skip_radius, alpha + delta_narrow, very_fine_step)
    add_range(alpha + delta_narrow, alpha + delta_medium, fine_step)
    add_range(alpha + delta_medium, alpha + delta_wide, medium_step)
    add_range(alpha + delta_wide, s2_max + 0.5 * coarse_step, coarse_step)

    vals.add(s2_min)
    vals.add(s2_max)

    grid = sorted(v for v in vals if s2_min <= v <= s2_max)
    return np.array(grid)


# ------------------------------------------------------------------ #
#  Helper: resolve bounce file path                                  #
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
        description="Compute G_bar(n=0, s2) on the bounce background."
    )
    parser.add_argument("--bounce",
                        default="bounce_data_F2_T0.npz",
                        help="Bounce .npz file.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Negative-mode location in s2. "
                             "If omitted, read from neg_mode_alpha.npz "
                             "(produced by negative_mode_diagnostic.py).")
    parser.add_argument("--s2-min", type=float, default=0.01)
    parser.add_argument("--s2-max", type=float, default=10.0)
    parser.add_argument("--n-eval", type=int, default=2000,
                        help="Radial grid points for the ODE solver.")
    parser.add_argument("--r0", type=float, default=1e-4)
    parser.add_argument("--skip-radius", type=float, default=0.0003,
                        help="Half-width of the exclusion zone around alpha.")
    parser.add_argument("--sub-threshold", type=float, default=2e-3,
                        help="|s2 - alpha| below which subtracted G_bar is set to NaN.")
    parser.add_argument("--fit-min", type=float, default=0.001,
                        help="Minimum |s2 - alpha| used in the A/(s2 - s_star) + B fit.")
    parser.add_argument("--fit-max", type=float, default=0.03,
                        help="Maximum |s2 - alpha| used in the A/(s2 - s_star) + B fit.")
    parser.add_argument("--s-star-half-width", type=float, default=0.005,
                        help="Half-width of s_star search grid around alpha_refined.")
    parser.add_argument("--s-star-n", type=int, default=401,
                        help="Number of s_star trial values in the grid search.")
    parser.add_argument("--no-fit", action="store_true",
                        help="Skip the pole fit; use fixed A=1 and alpha from det=0 crossing.")
    parser.add_argument("--no-det-refine", action="store_true",
                        help="Skip the det(Omega)=0 linear interpolation step; use the "
                             "initial alpha as the grid-search anchor directly "
                             "(matches the approach in compute_gbar_n1.py).")
    parser.add_argument("--out", default=None,
                        help="Output .npz filename (auto-generated if omitted).")
    parser.add_argument("--data-dir", default=None,
                        help="Output directory (default: $G_PROJECT_DATA or local).")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bounce_path = resolve_bounce(args.bounce, script_dir)
    print(f"[INFO] Bounce file: {bounce_path}")

    # --- output directory ---
    data_dir = args.data_dir or os.environ.get("TRG_DATA_DIR")
    if data_dir is None:
        dp = DATA_DIR
        if os.path.isdir(dp):
            data_dir = dp
        else:
            data_dir = script_dir
    os.makedirs(data_dir, exist_ok=True)
    print(f"[INFO] Output directory: {data_dir}")

    # --- load bounce data once ---
    bdata = np.load(bounce_path, allow_pickle=True)
    params    = bdata["params"]
    false_vac = np.asarray(bdata["false_vac"], dtype=float)
    r_bounce  = bdata["R"]
    x_prime   = bdata["X_bounce_prime"]
    y_prime   = bdata["Y_bounce_prime"]
    false_idx = int(bdata["false_index"])
    true_idx  = int(bdata["true_index"])

    pot_lin = CTShiftedLiftedPotential(params, false_vac)

    # --- resolve alpha: CLI flag > neg_mode_alpha.npz > error ---
    if args.alpha is not None:
        alpha = args.alpha
        print(f"[INFO] alpha from CLI: {alpha:.8f}")
    else:
        alpha_file = None
        for d in [data_dir, script_dir, DATA_DIR]:
            candidate = os.path.join(d, "neg_mode_alpha.npz")
            if os.path.isfile(candidate):
                alpha_file = candidate
                break
        if alpha_file is None:
            raise FileNotFoundError(
                "neg_mode_alpha.npz not found. "
                "Run negative_mode_diagnostic.py first, "
                "or pass --alpha VALUE explicitly."
            )
        adat = np.load(alpha_file, allow_pickle=True)
        alpha = float(adat["alpha"])
        print(f"[INFO] alpha from {alpha_file}: {alpha:.8f}")
    s2_grid = build_adaptive_s2_grid(alpha, args.s2_min, args.s2_max,
                                     skip_radius=args.skip_radius)
    n_s2 = len(s2_grid)
    print(f"[INFO] {n_s2} s2 values in [{s2_grid[0]:.6f}, {s2_grid[-1]:.6f}]")
    print(f"[INFO] Expected pole at s2 ~ {alpha:.6f}")

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
                s2, n_mode=0, n_eval=args.n_eval, r0=args.r0,
            )
            gbar_arr[idx]      = gb
            det_omega_arr[idx] = det_om
            cond_arr[idx]      = cond
            dt = time.time() - t1
            print(f"  [{idx+1:3d}/{n_s2}]  s2={s2:.6f}  "
                  f"G_bar={gb: .6e}  det(Om)={det_om: .6e}  "
                  f"cond={cond:.2e}  ({dt:.1f}s)")
        except Exception as exc:
            dt = time.time() - t1
            print(f"  [{idx+1:3d}/{n_s2}]  s2={s2:.6f}  FAILED: {exc}  ({dt:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n[INFO] Scan finished in {elapsed:.0f}s")

    # --- refine alpha from det(Omega) sign change (optional) ---
    # When --no-det-refine is set, skip this step entirely and use the initial
    # alpha as the anchor for the fit window / grid search (matches the n=1
    # approach in compute_gbar_n1.py; s_star then comes purely from the fit).
    alpha_refined = alpha
    if args.no_det_refine:
        print(f"[INFO] Skipping det(Omega)=0 refinement (--no-det-refine). "
              f"Using initial alpha = {alpha_refined:.10f} as fit anchor; "
              f"s_star will be determined entirely by the pole fit.")
    else:
        for k in range(1, n_s2):
            d0 = det_omega_arr[k - 1]
            d1 = det_omega_arr[k]
            if np.isfinite(d0) and np.isfinite(d1) and d0 * d1 < 0:
                s0, s1 = s2_grid[k - 1], s2_grid[k]
                alpha_refined = s0 - d0 * (s1 - s0) / (d1 - d0)
                break
        print(f"[RESULT] Refined negative-mode location (det=0 crossing): "
              f"alpha = {alpha_refined:.8f}")

    # --- fit pole model:  G_bar ~ A / (s2 - s_star) + B ---
    # Same approach as compute_gbar_n1.py:
    #   grid-search s_star around alpha_refined; at each trial, linear lstsq
    #   for (A, B) using the identity  gbar * (s2 - s_star) = A + B*(s2 - s_star).
    # Keep the fitted A (not forced to 1) and s_star for the subtraction.
    A_fit = 1.0
    s_star = alpha_refined
    B_fit = 0.0
    best_rss = np.nan
    fit_ok = False
    n_fit_points = 0

    if not args.no_fit:
        delta = np.abs(s2_grid - alpha_refined)
        fit_mask = (np.isfinite(gbar_arr)
                    & (delta >= args.fit_min)
                    & (delta <= args.fit_max))
        n_fit_points = int(np.sum(fit_mask))

        if n_fit_points < 3:
            print(f"[WARN] Only {n_fit_points} points in fit window "
                  f"[{args.fit_min}, {args.fit_max}]; "
                  f"skipping fit, falling back to A=1, s_star=alpha_refined.")
        else:
            s2_fit = s2_grid[fit_mask]
            gb_fit = gbar_arr[fit_mask]

            s_trials = np.linspace(
                alpha_refined - args.s_star_half_width,
                alpha_refined + args.s_star_half_width,
                args.s_star_n,
            )

            best_rss = np.inf
            for s_trial in s_trials:
                denom = s2_fit - s_trial
                if np.any(np.abs(denom) < 1e-12):
                    continue
                # gbar = A/denom + B  =>  gbar*denom = A + B*denom
                X = np.column_stack([np.ones_like(denom), denom])
                y = gb_fit * denom
                try:
                    coeffs, _res, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
                    A_try, B_try = coeffs
                    pred = A_try / denom + B_try
                    rss = float(np.sum((pred - gb_fit) ** 2))
                    if rss < best_rss:
                        best_rss = rss
                        s_star = float(s_trial)
                        A_fit = float(A_try)
                        B_fit = float(B_try)
                        fit_ok = True
                except Exception:
                    continue

            if fit_ok:
                print(f"[FIT] Pole fit to A/(s2 - s_star) + B "
                      f"using {n_fit_points} points:")
                print(f"       A      = {A_fit:.8f}  (expected ~1 for n=0 neg. mode)")
                print(f"       s_star = {s_star:.10f}")
                print(f"       B      = {B_fit:.6e}")
                print(f"       RSS    = {best_rss:.6e}")
                print(f"[FIT] shift s_star - alpha_refined = {s_star - alpha_refined:+.2e}")
                print(f"[FIT] residue A departure from 1   = {A_fit - 1.0:+.6f}")
            else:
                print(f"[WARN] Pole fit did not converge; "
                      f"falling back to A=1, s_star=alpha_refined.")

    # --- subtracted G_bar:  G_bar_sub = G_bar - A_fit / (s2 - s_star) ---
    print(f"[SUB] Subtracting A/(s2 - s_star) with A = {A_fit:.6f}, "
          f"s_star = {s_star:.10f}")

    gbar_sub = np.full(n_s2, np.nan)
    for idx in range(n_s2):
        s2 = s2_grid[idx]
        if np.isfinite(gbar_arr[idx]) and abs(s2 - s_star) > args.sub_threshold:
            gbar_sub[idx] = gbar_arr[idx] - A_fit / (s2 - s_star)

    # count valid
    n_valid     = np.sum(np.isfinite(gbar_arr))
    n_valid_sub = np.sum(np.isfinite(gbar_sub))
    print(f"[INFO] Valid G_bar points: {n_valid}/{n_s2}")
    print(f"[INFO] Valid G_bar_sub points: {n_valid_sub}/{n_s2}")

    # --- save (with fallback if data_dir disappeared, e.g. USB unmounted) ---
    out_name = args.out
    if out_name is None:
        out_name = os.path.join(
            data_dir,
            f"gbar_n0_scan_F{false_idx}_T{true_idx}.npz",
        )
    out_dir_check = os.path.dirname(out_name) or "."
    if not os.path.isdir(out_dir_check):
        fallback = os.path.join(script_dir,
                                f"gbar_n0_scan_F{false_idx}_T{true_idx}.npz")
        print(f"[WARN] {out_dir_check} not available, saving to {fallback}")
        out_name = fallback
    np.savez(
        out_name,
        s2_grid=s2_grid,
        gbar_n0=gbar_arr,
        gbar_n0_sub=gbar_sub,
        det_omega=det_omega_arr,
        cond_omega=cond_arr,
        alpha_initial=alpha,
        alpha_refined=alpha_refined,
        # pole fit outputs (used in the subtraction)
        A_fit=A_fit,
        s_star=s_star,
        B_fit=B_fit,
        rss=best_rss,
        fit_ok=fit_ok,
        n_fit_points=n_fit_points,
        fit_min=args.fit_min,
        fit_max=args.fit_max,
        bounce_file=str(bounce_path),
        false_index=false_idx,
        true_index=true_idx,
        n_eval=args.n_eval,
        r0=args.r0,
        sub_threshold=args.sub_threshold,
    )
    print(f"[SAVE] {out_name}")
    print("[INFO] To visualize, run:  python3 plot_gbar_n0.py")
    print("[DONE]")


if __name__ == "__main__":
    main()
