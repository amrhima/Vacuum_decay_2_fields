#!/usr/bin/env python3
"""
plot_gbar_n0.py -- Plot the n=0 pole-fit diagnostic from saved scan data.

Reads gbar_n0_scan_F*_T*.npz (output of compute_gbar_n0.py) and produces a
three-panel figure:

  1. G_bar unsubtracted with fit overlay  (A / (s2 - s_star) + B)
  2. G_bar subtracted WITHOUT excluding noisy near-pole points  (spike visible)
  3. G_bar subtracted WITH noisy points excluded  (clean)

Run with no CLI args to auto-detect the latest scan and save
gbar_n0_fit.png next to the script.
"""

import argparse
import os
import sys

import matplotlib
# prefer interactive backends so plt.show() opens a window;
# fall back to Agg (headless) only if none of them are available.
for _be in ("MacOSX", "TkAgg", "Qt5Agg", "Agg"):
    try:
        matplotlib.use(_be)
        break
    except Exception:
        continue
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR


def locate_scan_file(explicit_path, script_dir):
    """Return a path to an existing gbar_n0_scan_*.npz, auto-detecting if needed."""
    if explicit_path is not None:
        if os.path.isfile(explicit_path):
            return explicit_path
        print(f"[ERROR] File not found: {explicit_path}")
        sys.exit(1)

    candidates = []
    for d in [DATA_DIR, script_dir, "."]:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.startswith("gbar_n0_scan_") and f.endswith(".npz"):
                    candidates.append(os.path.join(d, f))
    if not candidates:
        print("[ERROR] No gbar_n0_scan_*.npz found. Run compute_gbar_n0.py first.")
        sys.exit(1)
    # prefer the data directory (first entry in the search list)
    for c in candidates:
        if c.startswith(DATA_DIR):
            return c
    return candidates[-1]


def make_three_panel_plot(dat, zoom_half_width=0.05):
    """Build and return a 3-panel figure from the saved scan dict."""
    s2 = dat["s2_grid"]
    g = dat["gbar_n0"]
    gs = dat["gbar_n0_sub"]

    A = float(dat["A_fit"])
    s_star = float(dat["s_star"])
    B = float(dat["B_fit"])
    alpha_refined = float(dat["alpha_refined"])
    fit_min = float(dat["fit_min"])
    fit_max = float(dat["fit_max"])
    n_fit_points = int(dat["n_fit_points"])
    sub_thresh = float(dat["sub_threshold"])
    F = int(dat["false_index"])
    T = int(dat["true_index"])

    hw = zoom_half_width
    zmask = np.abs(s2 - s_star) <= hw
    s2z, gz, gsz = s2[zmask], g[zmask], gs[zmask]

    # --- subtraction without NaN exclusion (shows spike) ---
    gs_nomask = np.where(np.isfinite(g), g - A / (s2 - s_star), np.nan)
    gs_nomask_z = gs_nomask[zmask]

    # --- fit curve on a dense grid ---
    s2_dense = np.linspace(s2z.min(), s2z.max(), 5000)
    s2_dense = s2_dense[np.abs(s2_dense - s_star) > 1e-5]
    fit_curve = A / (s2_dense - s_star) + B

    # --- which points were actually used in the fit ---
    fit_mask = (np.isfinite(g)
                & (np.abs(s2 - alpha_refined) >= fit_min)
                & (np.abs(s2 - alpha_refined) <= fit_max))
    fit_in_zoom = fit_mask & zmask

    # --- exclusion info for the bottom panel ---
    excl_lo = s_star - sub_thresh
    excl_hi = s_star + sub_thresh
    n_excluded = int(np.sum(np.isfinite(g) & ~np.isfinite(gs)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    ax1, ax2, ax3 = axes

    # --- Panel 1: G_bar unsubtracted with fit overlay ---
    # z-order: fit curve (bottom) → red-square fit markers → blue dots (top)
    # so every raw data point is visible; the red square is a frame around it
    # when the point was used in the fit.
    ax1.plot(s2_dense, fit_curve, "-", color="darkorange", linewidth=1.5, zorder=2,
             label=f"fit: A/(s2 - s_star) + B\n"
                   f"A={A:.5f}, s_star={s_star:.8f}\n"
                   f"B={B:.2e}")
    ax1.plot(s2[fit_in_zoom], g[fit_in_zoom], "s",
             markersize=9, markerfacecolor="none", markeredgecolor="red",
             markeredgewidth=1.2, alpha=0.9, zorder=3,
             label=f"points used in fit ({n_fit_points} total)")
    ax1.plot(s2z, gz, "o", markersize=4, color="steelblue",
             alpha=1.0, zorder=4,
             label=f"G_bar unsubtracted (bounce F{F}_T{T})")
    ax1.axvline(s_star, color="red", linestyle=":", alpha=0.7)
    ax1.axvspan(alpha_refined - fit_max, alpha_refined - fit_min,
                color="gray", alpha=0.08, label="fit window")
    ax1.axvspan(alpha_refined + fit_min, alpha_refined + fit_max,
                color="gray", alpha=0.08)
    ymax = np.nanmax(np.abs(gz)) * 1.05
    ax1.set_ylim(-ymax, ymax)
    ax1.set_ylabel("G_bar(s2)")
    ax1.set_title("G_bar unsubtracted with fit overlay")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(alpha=0.3)

    # --- Panel 2: subtraction WITHOUT exclusion ---
    ax2.plot(s2z, gs_nomask_z, "o-", markersize=4, color="crimson",
             alpha=0.8, linewidth=0.8,
             label="G_bar subtracted (no sub_threshold exclusion)")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax2.axvline(s_star, color="red", linestyle=":", alpha=0.7)
    ax2.set_ylabel("G_bar_sub(s2)")
    ax2.set_title("G_bar subtracted WITHOUT excluding noisy near-pole points  "
                  "(spike visible)")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(alpha=0.3)

    # --- Panel 3: subtraction WITH exclusion ---
    ax3.plot(s2z, gsz, "o-", markersize=4, color="seagreen",
             alpha=0.8, linewidth=0.8,
             label=(f"G_bar subtracted (sub_threshold = {sub_thresh:g})\n"
                    f"excluded range: [{excl_lo:.6f}, {excl_hi:.6f}]\n"
                    f"number of points excluded: {n_excluded}"))
    ax3.axvspan(excl_lo, excl_hi, color="crimson", alpha=0.12,
                label="excluded (noisy) band")
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax3.axvline(s_star, color="red", linestyle=":", alpha=0.7)
    ax3.set_xlabel("s2")
    ax3.set_ylabel("G_bar_sub(s2)")
    ax3.set_title("G_bar subtracted WITH noisy points excluded  (clean)")
    ax3.legend(fontsize=8, loc="best")
    ax3.grid(alpha=0.3)

    fig.suptitle(f"G_bar(n=0) pole fit  --  bounce F{F}_T{T}  "
                 f"(zoom |s2 - s_star| <= {hw};  middle: no exclusion  vs  "
                 f"bottom: sub_threshold = {sub_thresh:g})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot the 3-panel n=0 pole-fit diagnostic from saved scan data."
    )
    parser.add_argument("data", nargs="?", default=None,
                        help="Path to gbar_n0_scan_F*_T*.npz  "
                             "(auto-detected if omitted).")
    parser.add_argument("--save", default="gbar_n0_fit.png",
                        help="Output PNG filename (empty = don't save).")
    parser.add_argument("--zoom-half-width", type=float, default=0.05,
                        help="Half-width in s2 around s_star for the plot zoom.")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open a plot window (headless).")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = locate_scan_file(args.data, script_dir)
    print(f"[INFO] Reading: {data_path}")

    dat = np.load(data_path, allow_pickle=True)
    # sanity check: new-format file with fit keys present?
    required = {"A_fit", "s_star", "B_fit", "alpha_refined",
                "fit_min", "fit_max", "n_fit_points", "sub_threshold"}
    missing = required - set(dat.files)
    if missing:
        print(f"[ERROR] Scan file is missing fit keys: {sorted(missing)}")
        print("  Re-run compute_gbar_n0.py to regenerate with fit data.")
        sys.exit(1)

    fig = make_three_panel_plot(dat, zoom_half_width=args.zoom_half_width)

    if args.save:
        save_path = args.save
        if not os.path.isabs(save_path):
            save_path = os.path.join(script_dir, save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {save_path}")

    if not args.no_show:
        # plt.show() only makes sense on an interactive backend
        if matplotlib.get_backend().lower() != "agg":
            plt.show()
        else:
            print("[INFO] Non-interactive backend (Agg) -- skipping plt.show(). "
                  "Open the saved PNG to view the plot.")


if __name__ == "__main__":
    main()
