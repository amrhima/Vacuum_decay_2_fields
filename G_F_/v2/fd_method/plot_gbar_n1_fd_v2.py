#!/usr/bin/env python3
"""
plot_gbar_n1_fd_ver2.py  --  Plot the v2 FD result for gbar(n=1, s^2).

Reads gbar_n1_fd_ver2_F*_T*.npz produced by compute_gbar_n1_fd_ver2.py.
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
    # search order: script directory first, then data drive, then cwd;
    # within each directory we prefer the most recently modified file.
    candidates = []
    for d in [script_dir, DATA_DIR, "."]:
        if os.path.isdir(d):
            files = [
                os.path.join(d, f) for f in sorted(os.listdir(d))
                if f.startswith("gbar_n1_fd_ver2_") and f.endswith(".npz")
            ]
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            candidates.extend(files)
    if not candidates:
        print("[ERROR] No gbar_n1_fd_ver2_*.npz found. "
              "Run compute_gbar_n1_fd_ver2.py first.")
        sys.exit(1)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs="?", default=None)
    parser.add_argument("--save", default="gbar_n1_fd_ver2.png")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--linear-x", action="store_true",
                        help="Use linear x-axis (default: log).")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = locate_file(args.data, script_dir)
    print(f"[INFO] Reading: {path}")

    d = np.load(path, allow_pickle=True)
    s2 = d["s2_grid"]
    raw = d["gbar_n1_raw"]
    sub = d["gbar_n1_sub"]
    raw_err = d["gbar_n1_raw_err"]
    sub_err = d["gbar_n1_sub_err"]
    F = int(d["false_index"])
    T = int(d["true_index"])
    K = int(d["K"])
    N = int(d["N"])
    degeneracy = int(d["degeneracy"])
    lambda_zm = float(d["lambda_zm"]) if "lambda_zm" in d.files else None
    pole_s2 = float(d["pole_s2"]) if "pole_s2" in d.files else None
    skip_radius = float(d["skip_radius"]) if "skip_radius" in d.files else 0.0

    ok_raw = np.isfinite(raw)
    ok_sub = np.isfinite(sub)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: raw
    ax1.errorbar(s2[ok_raw], raw[ok_raw], yerr=raw_err[ok_raw],
                 fmt='o-', markersize=1.5, linewidth=0.5, elinewidth=0.4,
                 color='steelblue', alpha=0.8, label="G bar n=1 raw")
    if pole_s2 is not None and pole_s2 > 0:
        ax1.axvline(pole_s2, color="red", linestyle=":", alpha=0.7,
                    label=f"discrete pole at s² = {pole_s2:.6f}")
        if skip_radius > 0:
            ax1.axvspan(max(pole_s2 - skip_radius, 1e-300),
                        pole_s2 + skip_radius,
                        color="red", alpha=0.18, linewidth=0,
                        label=f"excluded |s²−pole| < {skip_radius:.0e}")
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("gbar_n1(s²)")
    ax1.set_title("Unsubtracted G bar (n=1)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, which='both')
    if not args.linear_x:
        ax1.set_xscale('log')

    # Panel 2: sub  -- pole line included, no legend
    ax2.errorbar(s2[ok_sub], sub[ok_sub], yerr=sub_err[ok_sub],
                 fmt='o-', markersize=1.5, linewidth=0.5, elinewidth=0.4,
                 color='seagreen', alpha=0.8)
    if pole_s2 is not None and pole_s2 > 0:
        ax2.axvline(pole_s2, color="red", linestyle=":", alpha=0.7)
        if skip_radius > 0:
            ax2.axvspan(max(pole_s2 - skip_radius, 1e-300),
                        pole_s2 + skip_radius,
                        color="red", alpha=0.18, linewidth=0)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("s²")
    ax2.set_ylabel("gbar_n1_sub(s²)")
    ax2.set_title("Subtracted G bar (n=1)")
    ax2.grid(alpha=0.3, which='both')
    if not args.linear_x:
        ax2.set_xscale('log')

    fig.suptitle(f"G_bar(n=1) v2  --  bounce F{F}_T{T}  "
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
