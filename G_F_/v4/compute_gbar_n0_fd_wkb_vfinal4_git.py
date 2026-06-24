#!/usr/bin/env python3
"""
compute_gbar_n0_fd_wkb_vfinal4.py -- thin filename wrapper around the n=0
dense-grid driver.

This script performs no computation of its own. It parses the same command
line as the n=0 driver `compute_gbar_n0_fd_git.main()`, forwards every
argument straight through to that driver, and only re-tags the default output
filename with the `_wkb_vfinal4` suffix so it lines up with the rest of the
pipeline's file naming. The dense n=0 grid is produced entirely by the driver
(an FD / Hutchinson-Schrodinger trace estimator); nothing about the numerics
is altered here.

Because this is a pure pass-through, any new CLI flag added to the underlying
driver must also be added to the parser below AND mirrored into the rebuilt
sys.argv near the end of main(), or it will be silently dropped.
"""

import os
import sys
import argparse

from config import DATA_DIR

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import compute_gbar_n0_fd_git as _orig


def main():
    parser = argparse.ArgumentParser(
        description="Thin filename wrapper for the n=0 dense-grid driver; "
                    "forwards all args to compute_gbar_n0_fd_git and retags "
                    "the default output filename _wkb_vfinal4.")
    # Mirror the original's argument list
    parser.add_argument("--bounce", default="bounce_data_F2_T0.npz")
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--r-min", type=float, default=1e-4)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--K", type=int, default=100000)  # vfinal4 default
    parser.add_argument("--s2-min", type=float, default=0.01)
    parser.add_argument("--s2-max", type=float, default=1000.0)
    parser.add_argument("--s2-tail-start", type=float, default=10.0)
    parser.add_argument("--s2-tail-step", type=float, default=5.0)
    parser.add_argument("--skip-radius", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--estimator", choices=["rank1", "projected"],
                        default="rank1",
                        help="FD subtracted-trace estimator forwarded to the "
                             "builder; use 'projected' for n=0,1 (low variance "
                             "near the zero/negative mode).")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Resolve the data directory (where the file should be written).
    # Without this, args.out defaults to a relative filename and the
    # original main() would write to the CWD (the project root in
    # `run_extend_to_s2_500.sh`).  The downstream pipeline expects the
    # file in the shared data directory, so we anchor it there.
    # --data-dir on the CLI overrides DATA_DIR from config.
    data_dir = args.data_dir or DATA_DIR

    # Default output: gbar_n0_fd_ver2_<TAG>_wkb_vfinal4.npz, written into
    # data_dir if known.
    if args.out is None:
        bdata_name = os.path.basename(args.bounce)
        # bounce_data_F2_T0.npz -> F2_T0
        tag = bdata_name.replace("bounce_data_", "").replace(".npz", "")
        out_basename = f"gbar_n0_fd_ver2_{tag}_wkb_vfinal4.npz"
        args.out = (os.path.join(data_dir, out_basename)
                    if data_dir else out_basename)
    elif data_dir is not None and not os.path.isabs(args.out):
        # User supplied a bare filename: anchor it in data_dir.
        args.out = os.path.join(data_dir, args.out)

    # Forward to the original driver by rebuilding sys.argv and calling
    # _orig.main().  NOTE: any new flag added to compute_gbar_n0_fd_git must
    # be added to the parser above AND appended here, or it will be dropped.
    fwd_args = [
        "compute_gbar_n0_fd.py",
        "--bounce", args.bounce,
        "--N", str(args.N),
        "--r-min", str(args.r_min),
        "--K", str(args.K),
        "--s2-min", str(args.s2_min),
        "--s2-max", str(args.s2_max),
        "--s2-tail-start", str(args.s2_tail_start),
        "--s2-tail-step", str(args.s2_tail_step),
        "--skip-radius", str(args.skip_radius),
        "--seed", str(args.seed),
        "--estimator", args.estimator,
        "--out", args.out,
    ]
    if args.r_max is not None:
        fwd_args += ["--r-max", str(args.r_max)]
    if args.data_dir is not None:
        fwd_args += ["--data-dir", args.data_dir]

    print("[compute_gbar_n0_fd_wkb_vfinal4] Forwarding to original "
          "compute_gbar_n0_fd with output:", args.out)
    sys.argv = fwd_args
    _orig.main()


if __name__ == "__main__":
    main()
