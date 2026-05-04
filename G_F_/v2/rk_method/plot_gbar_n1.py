#!/usr/bin/env python3
"""
plot_gbar_n1.py -- Plot G_bar(n=1, s2) diagnostics.

Reads output of compute_gbar_n1.py and produces 4 panels:
  1. G_bar(n=1, s2) unsubtracted  (shows 4/s2 divergence at zero mode)
  2. G_bar_sub(n=1, s2) subtracted  (zero mode removed)
  3. det(Omega) vs s2  (Wronskian singularity at zero mode)
  4. cond(Omega) vs s2  (condition number spike)
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


def main():
    parser = argparse.ArgumentParser(
        description="Plot G_bar(n=1) diagnostics: unsubtracted, subtracted, "
                    "det(Omega), cond(Omega)."
    )
    parser.add_argument(
        "data", nargs="?", default=None,
        help="Path to gbar_n1_scan_F*_T*.npz (auto-detected if omitted).",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save", default="",
                        help="Save figure to this filename.")
    parser.add_argument("--xlim", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))
    parser.add_argument("--ylim-unsub", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))
    parser.add_argument("--ylim-sub", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"))
    args = parser.parse_args()

    # --- locate data ---
    data_path = args.data
    if data_path is None:
        candidates = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for d in [".", script_dir,
                   os.path.join(script_dir, "data_files"),
                   DATA_DIR]:
            if os.path.isdir(d):
                for f in sorted(os.listdir(d)):
                    if f.startswith("gbar_n1_scan_") and f.endswith(".npz"):
                        candidates.append(os.path.join(d, f))
        if not candidates:
            print("[ERROR] No gbar_n1_scan_*.npz found. "
                  "Run compute_gbar_n1.py first.")
            sys.exit(1)
        data_path = candidates[-1]
        print(f"[INFO] Auto-detected: {data_path}")

    dat = np.load(data_path, allow_pickle=True)
    s2        = dat["s2_grid"]
    gbar      = dat["gbar_n1"]
    gbar_sub  = dat["gbar_n1_sub"]
    det_omega = dat["det_omega"]
    cond_om   = dat["cond_omega"]
    s_star    = float(dat["s_star"])
    A_fit     = float(dat["A_fit"])
    B_fit     = float(dat["B_fit"])

    print(f"[INFO] {len(s2)} points, s_star = {s_star:.8f}, A = {A_fit:.4f}, B = {B_fit:.4f}")

    ok     = np.isfinite(gbar)
    ok_sub = np.isfinite(gbar_sub)
    ok_det = np.isfinite(det_omega)
    ok_con = np.isfinite(cond_om)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: G_bar unsubtracted + fit overlay ---
    ax1 = axes[0, 0]
    ax1.plot(s2[ok], gbar[ok], "o-", markersize=2, linewidth=0.8, color="C0",
             label="G_bar(n=1, s2)")
    # overlay fitted pole model
    s2_fit_curve = np.geomspace(max(s2[ok].min(), 1e-7), s2[ok].max(), 500)
    denom = s2_fit_curve - s_star
    denom[np.abs(denom) < 1e-15] = np.nan
    gbar_fit_curve = A_fit / denom + B_fit
    ax1.plot(s2_fit_curve, gbar_fit_curve, "--", linewidth=1.2, color="red",
             label=f"Fit: {A_fit:.2f}/(s2 - {s_star:.2e}) + {B_fit:.2f}")
    ax1.set_xlabel("s2")
    ax1.set_ylabel("G_bar(n=1, s2)")
    ax1.set_title("Unsubtracted G_bar(n=1) with pole fit")
    ax1.set_xscale("log")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    if args.ylim_unsub:
        ax1.set_ylim(args.ylim_unsub)

    # --- Panel 2: G_bar subtracted ---
    ax2 = axes[0, 1]
    ax2.plot(s2[ok_sub], gbar_sub[ok_sub], "o-", markersize=2, linewidth=0.8,
             color="C1",
             label=f"G_bar - {A_fit:.2f}/(s2 - {s_star:.6f})")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("s2")
    ax2.set_ylabel("G_bar_sub(n=1, s2)")
    ax2.set_title("Subtracted G_bar(n=1)  (zero mode removed)")
    ax2.set_xscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    if args.ylim_sub:
        ax2.set_ylim(args.ylim_sub)

    # --- Panel 3: det(Omega) vs s2 ---
    ax3 = axes[1, 0]
    ax3.plot(s2[ok_det], det_omega[ok_det], "o-", markersize=2, linewidth=0.8,
             color="C2")
    ax3.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    if abs(s_star) > 1e-10:
        ax3.axvline(s_star, color="red", linestyle=":", alpha=0.7,
                    label=f"s_star = {s_star:.6f}")
        ax3.legend(fontsize=8)
    ax3.set_xlabel("s2")
    ax3.set_ylabel("det(Omega)")
    ax3.set_title("det(Omega) vs s2  (zero at zero mode)")
    ax3.set_xscale("log")
    ax3.grid(alpha=0.3)

    # --- Panel 4: cond(Omega) vs s2 ---
    ax4 = axes[1, 1]
    ax4.semilogy(s2[ok_con], cond_om[ok_con], "o-", markersize=2, linewidth=0.8,
                 color="C3")
    if abs(s_star) > 1e-10:
        ax4.axvline(s_star, color="red", linestyle=":", alpha=0.7,
                    label=f"s_star = {s_star:.6f}")
        ax4.legend(fontsize=8)
    ax4.set_xlabel("s2")
    ax4.set_ylabel("cond(Omega)")
    ax4.set_title("Condition number vs s2  (spike at zero mode)")
    ax4.set_xscale("log")
    ax4.grid(alpha=0.3)

    if args.xlim:
        for ax in axes.flat:
            ax.set_xlim(args.xlim)

    fig.suptitle(f"G_bar  n=1  |  s_star = {s_star:.6e}  |  A = {A_fit:.4f}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    if args.save:
        save_path = args.save
        if not os.path.isabs(save_path):
            save_path = os.path.join(os.path.dirname(data_path), save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {save_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
