#!/usr/bin/env python3
"""
compute_gbar_n1_fd_wkb_vfinal4.py -- Pass-through WKB v1 wrapper for the
dense n=1 grid generator.

This is a thin filename-wrapper: it parses the same CLI arguments as
compute_gbar_n1_fd_git (including --strict-zm), retags the default output
filename with the _wkb_vfinal4 suffix, and forwards every argument to
compute_gbar_n1_fd_git.main(). No numerical or physics logic lives here.

Argument forwarding is done by rebuilding sys.argv from the parsed args and
calling _orig.main(); the original module reads its options from sys.argv.
"""

import os
import sys
import argparse

from config import DATA_DIR

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import compute_gbar_n1_fd_git as _orig


def main():
    parser = argparse.ArgumentParser(
        description="Dense n=1 grid (FD method, no Bessel changes for "
                    "Stage A; output retagged _wkb_vfinal4).")
    parser.add_argument("--bounce", default="bounce_data_F2_T0.npz")
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--r-min", type=float, default=1e-4)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--K", type=int, default=100000)  # vfinal4 default
    parser.add_argument("--zm-overlap-warn", type=float, default=0.95)
    parser.add_argument("--zm-eigvalue-warn", type=float, default=0.1)
    parser.add_argument("--strict-zm", action="store_true")
    parser.add_argument("--s2-min", type=float, default=1e-6)
    parser.add_argument("--s2-max", type=float, default=1000.0)
    parser.add_argument("--s2-transition", type=float, default=0.1)
    parser.add_argument("--log-ppd", type=int, default=8)
    parser.add_argument("--linear-step", type=float, default=0.1)
    parser.add_argument("--s2-tail-start", type=float, default=10.0)
    parser.add_argument("--s2-tail-step", type=float, default=5.0)
    parser.add_argument("--skip-radius", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=23456)
    parser.add_argument("--estimator", choices=["rank1", "projected"],
                        default="rank1",
                        help="FD subtracted-trace estimator forwarded to the "
                             "builder; use 'projected' for n=0,1 (low variance "
                             "near the zero/negative mode).")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Resolve data directory so the FD output lands next to the rest
    # of the pipeline outputs (otherwise main() writes to CWD).
    data_dir = args.data_dir or DATA_DIR

    if args.out is None:
        bdata_name = os.path.basename(args.bounce)
        tag = bdata_name.replace("bounce_data_", "").replace(".npz", "")
        out_basename = f"gbar_n1_fd_ver2_{tag}_wkb_vfinal4.npz"
        args.out = (os.path.join(data_dir, out_basename)
                    if data_dir else out_basename)
    elif data_dir is not None and not os.path.isabs(args.out):
        args.out = os.path.join(data_dir, args.out)

    fwd_args = [
        "compute_gbar_n1_fd.py",
        "--bounce", args.bounce,
        "--N", str(args.N),
        "--r-min", str(args.r_min),
        "--K", str(args.K),
        "--zm-overlap-warn", str(args.zm_overlap_warn),
        "--zm-eigvalue-warn", str(args.zm_eigvalue_warn),
        "--s2-min", str(args.s2_min),
        "--s2-max", str(args.s2_max),
        "--s2-transition", str(args.s2_transition),
        "--log-ppd", str(args.log_ppd),
        "--linear-step", str(args.linear_step),
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
    if args.strict_zm:
        fwd_args += ["--strict-zm"]

    print("[compute_gbar_n1_fd_wkb_vfinal4] Forwarding to original "
          "compute_gbar_n1_fd with output:", args.out)
    sys.argv = fwd_args
    _orig.main()


if __name__ == "__main__":
    main()
