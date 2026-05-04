#!/usr/bin/env python3
"""
plot_D_integral.py -- Visualize the fluctuation determinant D.

Reads output of compute_D_integral.py and produces:
  Figure 1: Each partial-wave contribution vs s2 (n=0, n=1, n>1)
  Figure 2: Total D-integral vs s2
  Figure 3: G_bar_bounce(n, s2) as a function of n for several s2 values
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
        description="Plot D-integral: contributions, total, and G_bar vs n."
    )
    parser.add_argument("data", nargs="?", default=None,
                        help="Path to D_integral_*.npz (auto-detected if omitted).")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--save-prefix", default="",
                        help="Save figures with this prefix (e.g. 'D_plots').")
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
                    if f.startswith("D_integral_") and f.endswith(".npz") and "_per_n" not in f:
                        candidates.append(os.path.join(d, f))
        if not candidates:
            print("[ERROR] No D_integral_*.npz found. "
                  "Run compute_D_integral.py first.")
            sys.exit(1)
        data_path = candidates[-1]
        print(f"[INFO] Auto-detected: {data_path}")

    dat = np.load(data_path, allow_pickle=True)

    s2_n0      = dat["s2_n0"]
    s2_n1      = dat["s2_n1"]
    s2_npos    = dat["s2_npos"]
    I_n0_cum   = dat["I_n0_cum"]
    I_n1_cum   = dat["I_n1_cum"]
    I_npos_cum = dat["I_npos_cum"]

    s2_union      = dat["s2_union"]
    I_n0_union    = dat["I_n0_union"]
    I_n1_union    = dat["I_n1_union"]
    I_npos_union  = dat["I_npos_union"]
    I_total_union = dat["I_total_union"]

    I_n0_final   = float(dat["I_n0_final"])
    I_n1_final   = float(dat["I_n1_final"])
    I_npos_final = float(dat["I_npos_final"])
    I_total      = float(dat["I_total"])
    ln_alpha     = float(dat["ln_alpha"])
    alpha        = float(dat["alpha"])
    n_max        = int(dat["n_max"])

    print(f"[INFO] n=0: {I_n0_final:.6f},  n=1: {I_n1_final:.6f},  "
          f"n>1: {I_npos_final:.6f},  total: {I_total:.6f}")

    save_dir = os.path.dirname(data_path) if args.save_prefix else None

    # ==================================================================
    #  Figures 1a, 1b, 1c: Each contribution in its own plot
    # ==================================================================

    # n=0
    fig1a, ax1a = plt.subplots(figsize=(10, 5))
    ax1a.plot(s2_n0, I_n0_cum, "-", linewidth=1.5, color="C0")
    ax1a.axhline(I_n0_final, color="red", linestyle=":", alpha=0.5,
                 label=f"final = {I_n0_final:.4f}")
    ax1a.set_xlabel("s2")
    ax1a.set_ylabel("Cumulative integral")
    ax1a.set_title("n=0 contribution (neg mode subtracted)")
    ax1a.legend(fontsize=9)
    ax1a.grid(alpha=0.3)
    ax1a.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    fig1a.tight_layout()
    if args.save_prefix:
        path = os.path.join(save_dir, f"{args.save_prefix}_n0.png")
        fig1a.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {path}")

    # n=1
    fig1b, ax1b = plt.subplots(figsize=(10, 5))
    ax1b.plot(s2_n1, I_n1_cum, "-", linewidth=1.5, color="C1")
    ax1b.axhline(I_n1_final, color="red", linestyle=":", alpha=0.5,
                 label=f"final = {I_n1_final:.4f}")
    ax1b.set_xlabel("s2")
    ax1b.set_ylabel("Cumulative integral")
    ax1b.set_title("n=1 contribution (zero mode subtracted)")
    ax1b.legend(fontsize=9)
    ax1b.grid(alpha=0.3)
    ax1b.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    fig1b.tight_layout()
    if args.save_prefix:
        path = os.path.join(save_dir, f"{args.save_prefix}_n1.png")
        fig1b.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {path}")

    # n>1
    fig1c, ax1c = plt.subplots(figsize=(10, 5))
    ax1c.plot(s2_npos, I_npos_cum, "-", linewidth=1.5, color="C2")
    ax1c.axhline(I_npos_final, color="red", linestyle=":", alpha=0.5,
                 label=f"final = {I_npos_final:.4f}")
    ax1c.set_xlabel("s2")
    ax1c.set_ylabel("Cumulative integral")
    ax1c.set_title(f"n=2..{n_max} contribution")
    ax1c.legend(fontsize=9)
    ax1c.grid(alpha=0.3)
    ax1c.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    fig1c.tight_layout()
    if args.save_prefix:
        path = os.path.join(save_dir, f"{args.save_prefix}_npos.png")
        fig1c.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {path}")

    # ==================================================================
    #  Figure 2: Total D-integral vs s2 (on union grid)
    # ==================================================================
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # top: individual contributions on union grid
    ax2a.plot(s2_union, I_n0_union, "-", linewidth=1.0, color="C0", label="n=0")
    ax2a.plot(s2_union, I_n1_union, "-", linewidth=1.0, color="C1", label="n=1")
    ax2a.plot(s2_union, I_npos_union, "-", linewidth=1.0, color="C2",
              label=f"n=2..{n_max}")
    ax2a.set_ylabel("Cumulative integral")
    ax2a.set_title("D-integral components (union grid)")
    ax2a.legend(fontsize=9)
    ax2a.grid(alpha=0.3)
    ax2a.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    # bottom: total
    ax2b.plot(s2_union, I_total_union, "-", linewidth=2.0, color="black",
              label=f"Total  = {I_total:.6f}")
    ax2b.axhline(I_total, color="red", linestyle=":", alpha=0.5,
                 label=f"Final value = {I_total:.6f}")
    ax2b.set_xlabel("s2")
    ax2b.set_ylabel("Total cumulative integral")
    ax2b.set_title("Total D-integral vs s2")
    ax2b.legend(fontsize=9)
    ax2b.grid(alpha=0.3)

    fig2.suptitle(f"D-integral  |  S_eff(no ct) = (1/2)ln|D| = {-0.5*I_total:.6f}",
                  fontsize=12, fontweight="bold")
    fig2.tight_layout()

    if args.save_prefix:
        path = os.path.join(save_dir, f"{args.save_prefix}_total.png")
        fig2.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {path}")

    # ==================================================================
    #  Figure 3: G_bar_bounce vs n for several s2 values
    # ==================================================================
    per_n_path = data_path.replace(".npz", "_per_n.npz")
    if os.path.isfile(per_n_path):
        pn = np.load(per_n_path, allow_pickle=True)
        s2_vals = pn["s2_values"]
        n_vals  = pn["n_values"]
        gbar_matrix = pn["gbar_bounce_per_n"]  # shape (n_s2, n_modes)

        # pick a few s2 values to plot
        n_show = min(6, len(s2_vals))
        indices = np.linspace(0, len(s2_vals) - 1, n_show, dtype=int)

        fig3, ax3 = plt.subplots(figsize=(10, 6))
        for idx in indices:
            s2_val = s2_vals[idx]
            gb = gbar_matrix[idx, :]
            ok = np.isfinite(gb)
            if np.any(ok):
                ax3.plot(n_vals[ok], gb[ok], "o-", markersize=3, linewidth=0.8,
                         label=f"s2 = {s2_val:.2f}")

        ax3.set_xlabel("n (partial wave)")
        ax3.set_ylabel("G_bar_bounce(n, s2)  =  (n+1)^2 * I_n")
        ax3.set_title(f"G_bar bounce vs partial wave n  (n=2..{n_max})")
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)
        ax3.axhline(0, color="gray", linestyle="--", linewidth=0.5)

        if args.save_prefix:
            path = os.path.join(save_dir, f"{args.save_prefix}_gbar_vs_n.png")
            fig3.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[SAVE] {path}")
        # ==============================================================
        #  Figure 4: 3D surface — G_bar_bounce(n, s2) including n=0,1
        # ==============================================================
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from scipy.interpolate import interp1d

        # Extend n range to include n=0 and n=1
        n_all = np.concatenate([[0, 1], n_vals])
        Z_all = np.full((len(s2_vals), len(n_all)), np.nan)

        # Fill n>=2 from existing data
        Z_all[:, 2:] = gbar_matrix

        # Load n=0 subtracted and interpolate onto npos s2 grid
        search_dirs_scan = [os.path.dirname(data_path),
                            os.path.dirname(os.path.abspath(__file__)),
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "data_files"),
                            DATA_DIR]
        for d in search_dirs_scan:
            if not os.path.isdir(d):
                continue
            import glob as _gl
            n0_files = sorted(_gl.glob(os.path.join(d, "gbar_n0_scan_*.npz")))
            if n0_files:
                d0 = np.load(n0_files[-1], allow_pickle=True)
                s2_n0_scan = d0["s2_grid"]
                gb_n0_sub  = d0["gbar_n0_sub"]
                ok0 = np.isfinite(gb_n0_sub)
                if np.sum(ok0) >= 2:
                    f0 = interp1d(s2_n0_scan[ok0], gb_n0_sub[ok0],
                                  kind="linear", fill_value="extrapolate")
                    Z_all[:, 0] = f0(s2_vals)
                break

        # Load n=1 subtracted and interpolate onto npos s2 grid
        for d in search_dirs_scan:
            if not os.path.isdir(d):
                continue
            n1_files = sorted(_gl.glob(os.path.join(d, "gbar_n1_scan_*.npz")))
            if n1_files:
                d1 = np.load(n1_files[-1], allow_pickle=True)
                s2_n1_scan = d1["s2_grid"]
                gb_n1_sub  = d1["gbar_n1_sub"]
                ok1 = np.isfinite(gb_n1_sub)
                if np.sum(ok1) >= 2:
                    f1 = interp1d(s2_n1_scan[ok1], gb_n1_sub[ok1],
                                  kind="linear", fill_value="extrapolate")
                    Z_all[:, 1] = f1(s2_vals)
                break

        N_grid, S2_grid = np.meshgrid(n_all, s2_vals)
        Z_bounce = np.copy(Z_all)
        Z_bounce[~np.isfinite(Z_bounce)] = 0.0

        fig4 = plt.figure(figsize=(12, 7))
        ax4 = fig4.add_subplot(111, projection="3d")
        ax4.plot_surface(N_grid, S2_grid, Z_bounce,
                         cmap="viridis", edgecolor="none", alpha=0.8)
        ax4.set_xlabel("n")
        ax4.set_ylabel("s2")
        ax4.set_zlabel("G_bar_bounce")
        ax4.set_title(f"G_bar bounce(n, s2)  surface  (n=0..{n_max}, subtracted for n=0,1)")
        fig4.tight_layout()

        if args.save_prefix:
            path = os.path.join(save_dir, f"{args.save_prefix}_gbar_3d_bounce.png")
            fig4.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[SAVE] {path}")

        # ==============================================================
        #  Figure 5: 3D surface — G_bar_FV(n, s2) if available
        # ==============================================================
        # Try to load FV per-n data from the same search dirs
        fv_per_n_loaded = False
        for d in [os.path.dirname(data_path),
                  os.path.dirname(os.path.abspath(__file__)),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_files"),
                  DATA_DIR]:
            if not os.path.isdir(d):
                continue
            import glob as _glob
            fv_files = sorted(_glob.glob(os.path.join(d, "gbar_fv_*_npos_s2*.npz")))
            if not fv_files:
                continue

            # build FV matrix matching the bounce grid (n=0..n_max)
            Z_fv = np.full((len(s2_vals), len(n_all)), np.nan)
            for fv_f in fv_files:
                fd = np.load(fv_f, allow_pickle=True)
                fv_s2 = float(fd["s2"])
                fv_nv = fd["n_values"].astype(int)
                fv_gb = fd["gbar_n"].astype(float)
                s2_idx = np.where(np.abs(s2_vals - fv_s2) < 1e-6)[0]
                if len(s2_idx) == 0:
                    continue
                si = s2_idx[0]
                for ni, gv in zip(fv_nv, fv_gb):
                    n_idx = np.where(n_all == ni)[0]
                    if len(n_idx) > 0:
                        Z_fv[si, n_idx[0]] = gv

            Z_fv[~np.isfinite(Z_fv)] = 0.0
            if np.any(Z_fv != 0):
                fig5 = plt.figure(figsize=(12, 7))
                ax5 = fig5.add_subplot(111, projection="3d")
                ax5.plot_surface(N_grid, S2_grid, Z_fv,
                                 cmap="plasma", edgecolor="none", alpha=0.8)
                ax5.set_xlabel("n")
                ax5.set_ylabel("s2")
                ax5.set_zlabel("G_bar_FV")
                ax5.set_title(f"G_bar FV(n, s2)  surface  (n=0..{n_max})")
                fig5.tight_layout()

                if args.save_prefix:
                    path = os.path.join(save_dir,
                                        f"{args.save_prefix}_gbar_3d_fv.png")
                    fig5.savefig(path, dpi=150, bbox_inches="tight")
                    print(f"[SAVE] {path}")
                fv_per_n_loaded = True
            break

        if not fv_per_n_loaded:
            print("[WARN] FV per-n data not found, skipping FV 3D surface plot.")

    else:
        print(f"[WARN] Per-n data not found at {per_n_path}, skipping G_bar vs n plot.")

    # ==================================================================
    #  Figure 6: ln|D| vs n  (cumulative partial wave sum)
    # ==================================================================
    # For each n_max, ln|D|(n_max) = -(I_n0 + I_n1 + sum_{n=2}^{n_max} I_n)
    # We have per-n data from the per_n file to build this
    if os.path.isfile(per_n_path):
        pn = np.load(per_n_path, allow_pickle=True)
        pn_s2 = pn["s2_values"]
        pn_nvals = pn["n_values"]
        pn_gbar = pn["gbar_bounce_per_n"]  # (n_s2, n_modes)

        # Load FV per-n to subtract
        fv_per_n_matrix = np.zeros_like(pn_gbar)
        for d in [os.path.dirname(data_path),
                  os.path.dirname(os.path.abspath(__file__)),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data_files"),
                  DATA_DIR]:
            if not os.path.isdir(d):
                continue
            import glob as _g2
            fv_fs = sorted(_g2.glob(os.path.join(d, "gbar_fv_*_npos_s2*.npz")))
            if not fv_fs:
                continue
            for fv_f in fv_fs:
                fd = np.load(fv_f, allow_pickle=True)
                fv_s2 = float(fd["s2"])
                fv_nv = fd["n_values"].astype(int)
                fv_gb = fd["gbar_n"].astype(float)
                s2_idx = np.where(np.abs(pn_s2 - fv_s2) < 1e-6)[0]
                if len(s2_idx) == 0:
                    continue
                si = s2_idx[0]
                for ni, gv in zip(fv_nv, fv_gb):
                    n_idx = np.where(pn_nvals == ni)[0]
                    if len(n_idx) > 0:
                        fv_per_n_matrix[si, n_idx[0]] = gv
            break

        # Compute integrand per n: integrate (bounce_n - fv_n) over s2
        # using trapezoidal rule on the coarse s2 grid
        diff_per_n = pn_gbar - fv_per_n_matrix
        diff_per_n[~np.isfinite(diff_per_n)] = 0.0

        # Integrate over s2 for each n
        I_per_n = np.zeros(len(pn_nvals))
        for j in range(len(pn_nvals)):
            col = diff_per_n[:, j]
            I_per_n[j] = np.trapezoid(col, pn_s2)

        # Cumulative sum starting from n=0:
        # n=0: ln|D| = -I_n0
        # n=1: ln|D| = -(I_n0 + I_n1)
        # n=2..n_max: ln|D| = -(I_n0 + I_n1 + sum_{n=2}^{n_max})
        n_plot = [0, 1] + pn_nvals.tolist()
        cum_lnD_list = []
        cum_lnD_list.append(-I_n0_final)                      # n_max=0
        cum_lnD_list.append(-(I_n0_final + I_n1_final))       # n_max=1
        running = -(I_n0_final + I_n1_final)
        for j in range(len(pn_nvals)):
            running -= I_per_n[j]
            cum_lnD_list.append(running)

        n_plot = np.array(n_plot)
        cum_lnD = np.array(cum_lnD_list)

        fig6, ax6 = plt.subplots(figsize=(10, 6))
        ax6.plot(n_plot, cum_lnD, "o-", markersize=4, linewidth=1.2, color="C0")
        ax6.set_xlabel("n_max")
        ax6.set_ylabel("ln|D|  (cumulative)")
        ax6.set_title("ln|D| vs n_max  (= - total integral up to n_max, no counterterm)")
        ax6.grid(alpha=0.3)
        ax6.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        fig6.tight_layout()

        if args.save_prefix:
            path = os.path.join(save_dir, f"{args.save_prefix}_lnD_vs_n.png")
            fig6.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[SAVE] {path}")
    else:
        print("[WARN] Per-n data not found, skipping ln|D| vs n plot.")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
