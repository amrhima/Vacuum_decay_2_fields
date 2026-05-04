#!/usr/bin/env python3
"""
plot_gbar_n0_fd_ver2.py  --  Plot the v2 FD result for gbar(n=0, s^2).

Reads gbar_n0_fd_ver2_F*_T*.npz produced by compute_gbar_n0_fd_ver2.py.

Same 2x2 layout as plot_gbar_n0_fd.py:
  Left column: full s^2 range (raw on top, sub on bottom)
  Right column: zoomed pole region (default ±0.05)
"""

import argparse
import os
import sys

import matplotlib
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


def locate_file(explicit_path, script_dir):
    if explicit_path is not None:
        if os.path.isfile(explicit_path):
            return explicit_path
        print(f"[ERROR] File not found: {explicit_path}")
        sys.exit(1)
    # search order: script directory first (most-recent locally generated
    # files), then the data drive, then cwd. Within each directory we
    # prefer the most recently modified file so re-runs don't surprise.
    candidates = []
    for d in [script_dir, DATA_DIR, "."]:
        if os.path.isdir(d):
            files = [
                os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.startswith("gbar_n0_fd_ver2_") and f.endswith(".npz")
            ]
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            candidates.extend(files)
    if not candidates:
        print("[ERROR] No gbar_n0_fd_ver2_*.npz found. "
              "Run compute_gbar_n0_fd_ver2.py first.")
        sys.exit(1)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default=None)
    parser.add_argument("--save", default="gbar_n0_fd_ver2.png")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--zoom-half-width", type=float, default=0.01,
                        help="Half-width (in s² units) of the zoomed pole "
                             "panel (default 0.01 — the very-fine-grid "
                             "region). Set to 0 to disable the zoom column.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = locate_file(args.data, script_dir)
    print(f"[INFO] Reading: {path}")

    d = np.load(path, allow_pickle=True)
    s2 = d["s2_grid"]
    raw = d["gbar_n0_raw"]
    sub = d["gbar_n0_sub"]
    raw_err = d["gbar_n0_raw_err"]
    sub_err = d["gbar_n0_sub_err"]
    lambda_neg = float(d["lambda_neg"])
    pole_s2 = float(d["pole_s2"])
    skip_radius = float(d["skip_radius"]) if "skip_radius" in d.files else 0.0
    F = int(d["false_index"])
    T = int(d["true_index"])
    K = int(d["K"])
    N = int(d["N"])

    ok_raw = np.isfinite(raw)
    ok_sub = np.isfinite(sub)

    plot_kw = dict(fmt='o-', markersize=1.5, linewidth=0.5, elinewidth=0.4,
                   alpha=0.8)

    show_zoom = args.zoom_half_width > 0
    if show_zoom:
        fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                                 gridspec_kw={'width_ratios': [2, 1]})
        (ax_raw, ax_raw_z), (ax_sub, ax_sub_z) = axes
    else:
        fig, (ax_raw, ax_sub) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax_raw_z = ax_sub_z = None

    # Panel: raw, full range
    ax_raw.errorbar(s2[ok_raw], raw[ok_raw], yerr=raw_err[ok_raw],
                    color='steelblue', label="G bar n=0 raw", **plot_kw)
    ax_raw.axvline(pole_s2, color="red", linestyle=":", alpha=0.7,
                   label=f"pole at s² = {pole_s2:.6f}")
    if skip_radius > 0:
        ax_raw.axvspan(pole_s2 - skip_radius, pole_s2 + skip_radius,
                       color="red", alpha=0.18, linewidth=0,
                       label=f"excluded |s²−α| < {skip_radius:.0e}")
    ax_raw.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_raw.set_ylabel("gbar_n0(s²)")
    ax_raw.set_title("Unsubtracted G bar (n=0)")
    ax_raw.legend(fontsize=9)
    ax_raw.grid(alpha=0.3)

    # Panel: sub, full range  -- pole line included, no legend
    ax_sub.errorbar(s2[ok_sub], sub[ok_sub], yerr=sub_err[ok_sub],
                    color='seagreen', **plot_kw)
    ax_sub.axvline(pole_s2, color="red", linestyle=":", alpha=0.7)
    if skip_radius > 0:
        ax_sub.axvspan(pole_s2 - skip_radius, pole_s2 + skip_radius,
                       color="red", alpha=0.18, linewidth=0)
    ax_sub.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax_sub.set_xlabel("s²")
    ax_sub.set_ylabel("gbar_n0_sub(s²)")
    ax_sub.set_title("Subtracted G bar (n=0)")
    ax_sub.grid(alpha=0.3)

    if show_zoom:
        hw = args.zoom_half_width
        zoom_mask = np.abs(s2 - pole_s2) <= hw
        zr = ok_raw & zoom_mask
        zs = ok_sub & zoom_mask
        zoom_title = f"Zoom: pole ± {hw}"

        for ax_z, color, mask in (
            (ax_raw_z, 'steelblue', zr),
            (ax_sub_z, 'seagreen',  zs),
        ):
            ax_z.errorbar(s2[mask],
                          (raw if ax_z is ax_raw_z else sub)[mask],
                          yerr=(raw_err if ax_z is ax_raw_z else sub_err)[mask],
                          color=color, **plot_kw)
            ax_z.axvline(pole_s2, color="red", linestyle=":", alpha=0.7)
            if skip_radius > 0:
                ax_z.axvspan(pole_s2 - skip_radius, pole_s2 + skip_radius,
                             color="red", alpha=0.18, linewidth=0)
            ax_z.axhline(0, color="gray", linestyle="--",
                         linewidth=0.8, alpha=0.5)
            ax_z.set_xlim(pole_s2 - hw, pole_s2 + hw)
            ax_z.set_title(zoom_title)
            ax_z.grid(alpha=0.3)
            ax_z.tick_params(axis='x', labelsize=8)

        ax_sub_z.set_xlabel("s²")

        # Highlight the zoom window on the full-range panels (no legend entry)
        for ax_full in (ax_raw, ax_sub):
            ax_full.axvspan(pole_s2 - hw, pole_s2 + hw,
                            color='red', alpha=0.07)
        ax_raw.legend(fontsize=9)

    fig.suptitle(f"G_bar(n=0) v2  --  bounce F{F}_T{T}  "
                 f"(N={N}, K={K})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()

    save_path = args.save
    if not os.path.isabs(save_path):
        save_path = os.path.join(script_dir, save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[SAVE] {save_path}")

    if not args.no_show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    elif matplotlib.get_backend().lower() == "agg":
        print("[INFO] Non-interactive backend -- open the PNG to view.")


if __name__ == "__main__":
    main()
