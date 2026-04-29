#!/usr/bin/env python3
"""
compute_D_integral.py -- Compute the fluctuation determinant ratio D
by integrating G_bar over s2 for each partial wave sector.

Three sectors with independent s2 grids:
  n=0:   reads gbar_n0_scan_*.npz   (negative mode subtracted)
  n=1:   reads gbar_n1_scan_*.npz   (zero mode subtracted)
  n>1:   reads gbar_bounce_*_npos_s2*.npz and gbar_fv_*_npos_s2*.npz

For each sector the integrand is:
  G_bar_bounce(n, s2) - G_bar_FV(n, s2)

with additional pole subtractions for n=0 (negative mode) and n=1 (zero mode).

The FV contribution is interpolated onto each sector's native s2 grid.

Output: D_integral_F{F}_T{T}.npz
"""

import argparse
import glob
import os
import sys

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def find_file(pattern, dirs):
    """Search for a file matching pattern in multiple directories."""
    for d in dirs:
        matches = sorted(glob.glob(os.path.join(d, pattern)))
        if matches:
            return matches[-1]
    return None


def load_fv_gbar_at_n(fv_files, n_target):
    """
    From a list of FV npos summary files, extract G_bar for a single n
    at each s2.  Returns (s2_arr, gbar_arr).
    """
    s2_list = []
    gb_list = []
    for f in sorted(fv_files):
        dat = np.load(f, allow_pickle=True)
        s2_val = float(dat["s2"])
        n_vals = dat["n_values"].astype(int)
        gbar_n = dat["gbar_n"].astype(float)
        idx = np.where(n_vals == n_target)[0]
        if len(idx) == 0:
            continue
        s2_list.append(s2_val)
        gb_list.append(float(gbar_n[idx[0]]))
    return np.array(s2_list), np.array(gb_list)


def load_fv_gbar_sum(fv_files, n_min=2, n_max=50):
    """
    From FV npos files, sum G_bar over n in [n_min, n_max] at each s2.
    Returns (s2_arr, gbar_sum_arr).
    """
    s2_list = []
    gb_list = []
    for f in sorted(fv_files):
        dat = np.load(f, allow_pickle=True)
        s2_val = float(dat["s2"])
        n_vals = dat["n_values"].astype(int)
        gbar_n = dat["gbar_n"].astype(float)
        mask = (n_vals >= n_min) & (n_vals <= n_max)
        if not np.any(mask):
            continue
        s2_list.append(s2_val)
        gb_list.append(float(np.sum(gbar_n[mask])))
    return np.array(s2_list), np.array(gb_list)


def cumulative_integral(s2, integrand):
    """Cumulative trapezoidal integral from s2[0] to each s2[k]."""
    result = np.zeros_like(s2)
    for k in range(1, len(s2)):
        result[k] = result[k-1] + 0.5 * (integrand[k-1] + integrand[k]) * (s2[k] - s2[k-1])
    return result


# ------------------------------------------------------------------ #
#  Main                                                              #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Compute the fluctuation determinant D from G_bar data."
    )
    parser.add_argument("--bounce-tag", default="F2_T0",
                        help="Tag for bounce pair, e.g. F2_T0.")
    parser.add_argument("--n-max", type=int, default=50)
    parser.add_argument("--data-dir", default=None,
                        help="Directory with G_bar data files.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir
    if data_dir is None:
        dp = DATA_DIR
        if os.path.isdir(dp):
            data_dir = dp
        else:
            data_dir = script_dir

    search_dirs = [data_dir, script_dir,
                   os.path.join(script_dir, "data_files"),
                   DATA_DIR]
    tag = args.bounce_tag

    # ==================================================================
    #  LOAD n=0 data
    # ==================================================================
    n0_file = find_file(f"gbar_n0_scan_{tag}.npz", search_dirs)
    if n0_file is None:
        print("[ERROR] gbar_n0_scan_*.npz not found. Run compute_gbar_n0.py first.")
        sys.exit(1)
    print(f"[LOAD] n=0: {n0_file}")

    d0 = np.load(n0_file, allow_pickle=True)
    s2_n0       = d0["s2_grid"]
    gbar_n0     = d0["gbar_n0"]
    gbar_n0_sub = d0["gbar_n0_sub"]
    alpha       = float(d0["alpha_refined"])
    print(f"  {len(s2_n0)} points, alpha = {alpha:.8f}")

    # ==================================================================
    #  LOAD n=1 data
    # ==================================================================
    n1_file = find_file(f"gbar_n1_scan_{tag}.npz", search_dirs)
    if n1_file is None:
        print("[ERROR] gbar_n1_scan_*.npz not found. Run compute_gbar_n1.py first.")
        sys.exit(1)
    print(f"[LOAD] n=1: {n1_file}")

    d1 = np.load(n1_file, allow_pickle=True)
    s2_n1       = d1["s2_grid"]
    gbar_n1     = d1["gbar_n1"]
    gbar_n1_sub = d1["gbar_n1_sub"]
    s_star      = float(d1["s_star"])
    A_fit       = float(d1["A_fit"])
    print(f"  {len(s2_n1)} points, s_star = {s_star:.8f}, A = {A_fit:.4f}")

    # ==================================================================
    #  LOAD n>1 bounce data
    # ==================================================================
    npos_bounce_files = []
    for d in search_dirs:
        npos_bounce_files += glob.glob(os.path.join(d, f"gbar_bounce_{tag}_npos_s2*.npz"))
    npos_bounce_files = sorted(set(npos_bounce_files))
    if not npos_bounce_files:
        print("[ERROR] No gbar_bounce_*_npos_s2*.npz found. "
              "Run build_gbar_bounce.py first.")
        sys.exit(1)
    print(f"[LOAD] n>1 bounce: {len(npos_bounce_files)} files")

    s2_npos_b = []
    gbar_npos_b = []
    gbar_per_n = {}  # {s2: {n: gbar_value}}
    for f in npos_bounce_files:
        dat = np.load(f, allow_pickle=True)
        s2_val = float(dat["s2"])
        n_vals = dat["n_values"].astype(int)
        gbar_n = dat["gbar_n"].astype(float)
        mask = (n_vals >= 2) & (n_vals <= args.n_max)
        if not np.any(mask):
            continue
        s2_npos_b.append(s2_val)
        gbar_npos_b.append(float(np.sum(gbar_n[mask])))
        gbar_per_n[s2_val] = dict(zip(n_vals[mask].tolist(), gbar_n[mask].tolist()))
    s2_npos_b = np.array(s2_npos_b)
    gbar_npos_b = np.array(gbar_npos_b)
    order = np.argsort(s2_npos_b)
    s2_npos_b = s2_npos_b[order]
    gbar_npos_b = gbar_npos_b[order]
    print(f"  {len(s2_npos_b)} s2 points, n range [2, {args.n_max}]")

    # ==================================================================
    #  LOAD FV data
    # ==================================================================
    fv_files = []
    for d in search_dirs:
        fv_files += glob.glob(os.path.join(d, f"gbar_fv_{tag}_npos_s2*.npz"))
    fv_files = sorted(set(fv_files))
    if not fv_files:
        print("[ERROR] No gbar_fv_*_npos_s2*.npz found. "
              "Run build_gbar_fv.py first.")
        sys.exit(1)
    print(f"[LOAD] FV: {len(fv_files)} files")

    # Extract FV contributions for n=0, n=1, and n>1
    s2_fv_n0, gbar_fv_n0 = load_fv_gbar_at_n(fv_files, 0)
    s2_fv_n1, gbar_fv_n1 = load_fv_gbar_at_n(fv_files, 1)
    s2_fv_npos, gbar_fv_npos = load_fv_gbar_sum(fv_files, 2, args.n_max)
    print(f"  FV n=0: {len(s2_fv_n0)} points")
    print(f"  FV n=1: {len(s2_fv_n1)} points")
    print(f"  FV n>1: {len(s2_fv_npos)} points")

    # ==================================================================
    #  Interpolate FV onto each bounce s2 grid
    # ==================================================================
    def interp_fv(s2_fv, gbar_fv, s2_target, label):
        if len(s2_fv) < 2:
            print(f"[WARN] FV {label}: only {len(s2_fv)} points, using zero.")
            return np.zeros_like(s2_target)
        f = interp1d(s2_fv, gbar_fv, kind="linear", fill_value="extrapolate")
        return f(s2_target)

    gbar_fv_n0_on_n0grid   = interp_fv(s2_fv_n0, gbar_fv_n0, s2_n0, "n=0")
    gbar_fv_n1_on_n1grid   = interp_fv(s2_fv_n1, gbar_fv_n1, s2_n1, "n=1")
    gbar_fv_npos_on_nposgrid = interp_fv(s2_fv_npos, gbar_fv_npos,
                                          s2_npos_b, "n>1")

    # ==================================================================
    #  Compute integrands (bounce - FV) on native grids
    # ==================================================================

    # n=0: use subtracted G_bar (negative mode removed)
    integrand_n0 = np.where(
        np.isfinite(gbar_n0_sub),
        gbar_n0_sub - gbar_fv_n0_on_n0grid,
        np.nan,
    )

    # n=1: use subtracted G_bar (zero mode removed)
    integrand_n1 = np.where(
        np.isfinite(gbar_n1_sub),
        gbar_n1_sub - gbar_fv_n1_on_n1grid,
        np.nan,
    )

    # n>1: no subtraction needed
    integrand_npos = gbar_npos_b - gbar_fv_npos_on_nposgrid

    # ==================================================================
    #  Integrate each sector over s2 (cumulative)
    # ==================================================================

    def safe_cumulative(s2, integrand):
        ok = np.isfinite(integrand)
        if np.sum(ok) < 2:
            return np.full_like(s2, np.nan)
        s2_ok = s2[ok]
        ig_ok = integrand[ok]
        return cumulative_integral(s2_ok, ig_ok), s2_ok

    I_n0_cum, s2_n0_ok   = safe_cumulative(s2_n0, integrand_n0)
    I_n1_cum, s2_n1_ok   = safe_cumulative(s2_n1, integrand_n1)
    I_npos_cum, s2_npos_ok = safe_cumulative(s2_npos_b, integrand_npos)

    # Final values at largest s2
    I_n0_final   = I_n0_cum[-1] if len(I_n0_cum) > 0 else 0.0
    I_n1_final   = I_n1_cum[-1] if len(I_n1_cum) > 0 else 0.0
    I_npos_final = I_npos_cum[-1] if len(I_npos_cum) > 0 else 0.0

    # Analytical contributions
    ln_alpha = np.log(abs(alpha))  # negative mode: ln|w^2_{0,1,0}|
    # zero mode: 4 * ln(m^2) would come from Carosi/Baacke procedure
    # but with finite cutoff this is absorbed; we report the integral only

    # Total D-integral (without counterterm -- that comes from renormalization)
    I_total = I_n0_final + I_n1_final + I_npos_final

    # ln|D| = -I_total   (minus sign from the Green's function method)
    # S_eff = (1/2) ln|D| = -(1/2) I_total
    # Here we report the pieces

    print("\n" + "=" * 60)
    print("D-INTEGRAL RESULTS")
    print("=" * 60)
    print(f"  n=0  integral:   {I_n0_final: .8f}  (neg mode subtracted)")
    print(f"  n=1  integral:   {I_n1_final: .8f}  (zero mode subtracted)")
    print(f"  n>1  integral:   {I_npos_final: .8f}  (n=2..{args.n_max})")
    print(f"  ---")
    print(f"  Total integral:              {I_total: .8f}")
    print(f"  ln|D| = -total integral:     {-I_total: .8f}")
    print(f"  S_eff (no ct) = (1/2)ln|D|:  {-0.5 * I_total: .8f}")
    print("=" * 60)

    # ==================================================================
    #  Interpolate all to a union grid for combined plotting
    # ==================================================================
    s2_union = np.unique(np.concatenate([s2_n0_ok, s2_n1_ok, s2_npos_ok]))
    s2_union = np.sort(s2_union)

    def interp_cum(s2_native, cum_native, s2_target):
        if len(s2_native) < 2:
            return np.zeros_like(s2_target)
        f = interp1d(s2_native, cum_native, kind="linear",
                     bounds_error=False, fill_value=(0.0, cum_native[-1]))
        return f(s2_target)

    I_n0_union   = interp_cum(s2_n0_ok, I_n0_cum, s2_union)
    I_n1_union   = interp_cum(s2_n1_ok, I_n1_cum, s2_union)
    I_npos_union = interp_cum(s2_npos_ok, I_npos_cum, s2_union)
    I_total_union = I_n0_union + I_n1_union + I_npos_union

    # ==================================================================
    #  Save
    # ==================================================================
    out_name = args.out
    if out_name is None:
        out_name = os.path.join(data_dir, f"D_integral_{tag}.npz")
    out_dir_check = os.path.dirname(out_name) or "."
    if not os.path.isdir(out_dir_check):
        out_name = os.path.join(script_dir, f"D_integral_{tag}.npz")
        print(f"[WARN] Saving to {out_name}")

    np.savez(
        out_name,
        # native grids and cumulative integrals
        s2_n0=s2_n0_ok,
        s2_n1=s2_n1_ok,
        s2_npos=s2_npos_ok,
        I_n0_cum=I_n0_cum,
        I_n1_cum=I_n1_cum,
        I_npos_cum=I_npos_cum,
        # integrands on native grids
        integrand_n0=integrand_n0[np.isfinite(integrand_n0)],
        integrand_n1=integrand_n1[np.isfinite(integrand_n1)],
        integrand_npos=integrand_npos,
        # union grid
        s2_union=s2_union,
        I_n0_union=I_n0_union,
        I_n1_union=I_n1_union,
        I_npos_union=I_npos_union,
        I_total_union=I_total_union,
        # final values
        I_n0_final=I_n0_final,
        I_n1_final=I_n1_final,
        I_npos_final=I_npos_final,
        I_total=I_total,
        ln_alpha=ln_alpha,
        alpha=alpha,
        s_star=s_star,
        A_fit=A_fit,
        n_max=args.n_max,
        bounce_tag=tag,
        # per-n data for n>1 (for G_bar vs n plot)
        s2_npos_grid=s2_npos_b,
        gbar_per_n_s2_values=np.array(list(gbar_per_n.keys())),
    )

    # save per-n data separately (dict not saveable in npz directly)
    per_n_file = out_name.replace(".npz", "_per_n.npz")
    n_range = np.arange(2, args.n_max + 1)
    gbar_matrix = np.full((len(s2_npos_b), len(n_range)), np.nan)
    for i, s2_val in enumerate(s2_npos_b):
        if s2_val in gbar_per_n:
            for j, n in enumerate(n_range):
                if n in gbar_per_n[s2_val]:
                    gbar_matrix[i, j] = gbar_per_n[s2_val][n]
    np.savez(per_n_file,
             s2_values=s2_npos_b,
             n_values=n_range,
             gbar_bounce_per_n=gbar_matrix)
    print(f"[SAVE] {out_name}")
    print(f"[SAVE] {per_n_file}")
    print("[DONE]")


if __name__ == "__main__":
    main()
