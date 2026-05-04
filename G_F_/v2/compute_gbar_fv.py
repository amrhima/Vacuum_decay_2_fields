#!/usr/bin/env python3
"""
Build Gbar summaries for the false-vacuum background on specified s2 grids.
"""

import argparse
import os
import numpy as np

from rk_builder_fv import build_rk_green_fv_for_bounce
from config import DATA_DIR


def s2_tag(s2: float, digits: int = 6) -> str:
    return f"{s2:.{digits}f}".replace(".", "p")


def integrated_trace_from_rk(rk_filename: str) -> float:
    data = np.load(rk_filename, allow_pickle=True)
    r_grid = data["r_grid"]
    G_rk = data["G_rk"]

    nr = len(r_grid)
    trace_diag = np.empty(nr, dtype=float)
    for k in range(nr):
        trace_diag[k] = np.trace(G_rk[k, k, :, :])

    integrand = (r_grid ** 3) * trace_diag
    return float(np.trapezoid(integrand, r_grid))


def build_s2_grid_coarse(s2_min, s2_max, step):
    values = np.arange(0.0, s2_max + 0.5 * step, step)
    values = values[values >= s2_min]
    values = np.append(values, [s2_min, s2_max])
    values = np.unique(np.round(values, 12))
    return values.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Gbar summaries for the false-vacuum background."
    )
    parser.add_argument(
        "--bounce",
        default="bounce_data_F2_T0.npz",
        help="Bounce file, e.g. bounce_data_F2_T0.npz",
    )
    parser.add_argument("--s2-min", type=float, default=1e-3)
    parser.add_argument("--s2-max", type=float, default=10.0)
    parser.add_argument("--s2-step-fine", type=float, default=0.001)
    parser.add_argument("--s2-step-coarse", type=float, default=0.5)
    parser.add_argument(
        "--s2-log-max",
        type=float,
        default=0.1,
        help="Upper limit for log-spaced IR grid (used for n>=1).",
    )
    parser.add_argument(
        "--s2-log-spaced-points",
        type=int,
        default=20,
        help="Log-grid points per decade in the IR.",
    )
    parser.add_argument("--alpha", type=float, default=0.3115)
    parser.add_argument("--delta", type=float, default=0.001)
    parser.add_argument("--n-min-npos", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--r0", type=float, default=1e-4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite summary outputs if they already exist.",
    )
    merge_group = parser.add_mutually_exclusive_group()
    merge_group.add_argument(
        "--merge",
        action="store_true",
        help="Merge into existing summaries when present (default: on).",
    )
    merge_group.add_argument(
        "--no-merge",
        action="store_false",
        dest="merge",
        help="Do not merge; skip existing summaries.",
    )
    build_group = parser.add_mutually_exclusive_group()
    build_group.add_argument(
        "--build-missing-rk",
        action="store_true",
        help="Build RK data files if they are missing (default: on).",
    )
    build_group.add_argument(
        "--no-build-missing-rk",
        action="store_false",
        dest="build_missing_rk",
        help="Do not build missing RK data files.",
    )
    parser.add_argument(
        "--overwrite-rk",
        action="store_true",
        help="Rebuild RK data files even if they exist.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Output directory (defaults to TRG_DATA_DIR/TRG_OUT_DIR or $G_PROJECT_DATA).",
    )
    parser.set_defaults(build_missing_rk=True)
    parser.set_defaults(merge=True)
    args = parser.parse_args()
    if args.n_min_npos != 0:
        raise ValueError(
            "FV summaries must include n=0 and n=1. Use --n-min-npos 0."
        )
    if args.overwrite and args.merge:
        raise ValueError("Use either --overwrite or --merge, not both.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bounce_path = args.bounce
    if not os.path.isfile(bounce_path):
        for search_dir in [script_dir,
                           os.path.join(script_dir, "data_files"),
                           DATA_DIR]:
            alt_path = os.path.join(search_dir, os.path.basename(args.bounce))
            if os.path.isfile(alt_path):
                bounce_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Bounce file not found: {args.bounce}")

    default_data_dir = DATA_DIR
    data_dir = (
        args.data_dir
        or os.environ.get("TRG_DATA_DIR")
        or os.environ.get("TRG_OUT_DIR")
    )
    if data_dir is None and os.path.isdir(default_data_dir):
        data_dir = default_data_dir
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        os.chdir(data_dir)
        print(f"[INFO] Writing outputs to {data_dir}")

    bdata = np.load(bounce_path, allow_pickle=True)
    false_index = int(bdata["false_index"])
    true_index = int(bdata["true_index"])

    s2_values_npos = build_s2_grid_coarse(
        args.s2_min,
        args.s2_max,
        args.s2_step_coarse,
    )
    built_rk = set()

    for s2 in s2_values_npos:
        tag = s2_tag(s2, 6)
        summary_name = f"gbar_fv_F{false_index}_T{true_index}_npos_s2{tag}.npz"
        if os.path.exists(summary_name) and not args.overwrite:
            if not args.merge:
                print(f"[SKIP] {summary_name} already exists.")
                continue
            prev = np.load(summary_name, allow_pickle=True)
            prev_n = set(prev["n_values"].astype(int).tolist())
            needed_n = set(range(args.n_min_npos, args.n_max + 1))
            if needed_n.issubset(prev_n):
                print(f"[SKIP] {summary_name} already complete.")
                continue

        merged = {}
        if args.merge and os.path.exists(summary_name):
            prev = np.load(summary_name, allow_pickle=True)
            prev_n = prev["n_values"].astype(int)
            prev_i = prev["I_n"].astype(float)
            if "gbar_n" in prev.files:
                prev_g = prev["gbar_n"].astype(float)
            elif "contrib" in prev.files:
                prev_g = prev["contrib"].astype(float)
            else:
                raise KeyError(f"{summary_name} missing 'gbar_n' or 'contrib'")
            for n_val, i_val, g_val in zip(prev_n, prev_i, prev_g):
                n_int = int(n_val)
                if args.n_min_npos <= n_int <= args.n_max:
                    merged[n_int] = (float(i_val), float(g_val))

        for n_mode in range(args.n_min_npos, args.n_max + 1):
            rk_name = f"rk_green_data_FV_n{n_mode}_s2{tag}.npz"
            if os.path.exists(rk_name):
                if args.overwrite_rk:
                    if rk_name not in built_rk:
                        build_rk_green_fv_for_bounce(
                            bounce_path,
                            s2=s2,
                            n_mode=n_mode,
                            n_eval=args.n_eval,
                            r0=args.r0,
                            out_fname=rk_name,
                            overwrite=True,
                        )
                        built_rk.add(rk_name)
                i_n = integrated_trace_from_rk(rk_name)
                gbar = (n_mode + 1) ** 2 * i_n
                merged[n_mode] = (i_n, gbar)
            elif args.build_missing_rk:
                if rk_name not in built_rk:
                    build_rk_green_fv_for_bounce(
                        bounce_path,
                        s2=s2,
                        n_mode=n_mode,
                        n_eval=args.n_eval,
                        r0=args.r0,
                        out_fname=rk_name,
                        overwrite=False,
                    )
                    built_rk.add(rk_name)
                i_n = integrated_trace_from_rk(rk_name)
                gbar = (n_mode + 1) ** 2 * i_n
                merged[n_mode] = (i_n, gbar)
            elif n_mode not in merged:
                raise FileNotFoundError(
                    f"Missing RK file {rk_name} and no existing summary entry. "
                    "Use --build-missing-rk to create it."
                )

        for required_n in (0, 1):
            if required_n not in merged:
                raise RuntimeError(
                    f"FV summary missing n={required_n} at s2={s2:.6f}. "
                    "Ensure RK files exist or rerun without --no-build-missing-rk."
                )

        n_values = np.arange(args.n_min_npos, args.n_max + 1, dtype=int)
        i_n_values = np.array([merged[n][0] for n in n_values], dtype=float)
        gbar_values = np.array([merged[n][1] for n in n_values], dtype=float)
        total_sum = float(np.sum(gbar_values))

        np.savez(
            summary_name,
            n_values=np.array(n_values, dtype=int),
            I_n=np.array(i_n_values, dtype=float),
            gbar_n=np.array(gbar_values, dtype=float),
            total_sum=total_sum,
            bounce_file=bounce_path,
            s2=s2,
            n_min=args.n_min_npos,
            n_max=args.n_max,
        )
        print(f"[SAVE] {summary_name}")


if __name__ == "__main__":
    main()
