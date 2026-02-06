#!/usr/bin/env python3
"""
Build n=1 Gbar summaries on a log-spaced s2 grid with high RK resolution,
optionally extending with a coarse linear tail up to s2_max.

Default: s2 in [0.0007, 0.1], n_eval in {2000}.
"""

import argparse
import os
import numpy as np

from rk_builder_adapt import build_rk_green_for_bounce


def s2_tag(s2: float, digits: int = 6) -> str:
    return f"{s2:.{digits}f}".replace(".", "p")


def apply_suffix(name: str, suffix: str) -> str:
    if not suffix:
        return name
    if not suffix.startswith("_"):
        suffix = "_" + suffix
    if name.endswith(".npz"):
        return name[:-4] + suffix + ".npz"
    return name + suffix


def build_s2_grid_log(s2_min, s2_log_max, s2_max, log_spaced_points, step_coarse):
    if s2_min <= 0.0:
        raise ValueError("s2_min must be > 0 for log-spaced grid.")
    log_max_eff = min(s2_log_max, s2_max)
    values = []
    if s2_min < log_max_eff:
        decades = np.log10(log_max_eff / s2_min)
        n_log = max(2, int(np.ceil(decades * log_spaced_points)) + 1)
        values.extend(np.logspace(np.log10(s2_min), np.log10(log_max_eff), n_log))
    else:
        values.append(s2_min)

    if s2_max > log_max_eff and step_coarse > 0:
        values.extend(np.arange(log_max_eff, s2_max + 0.5 * step_coarse, step_coarse))

    values.extend([s2_min, log_max_eff, s2_max])
    values = np.unique(np.round(values, 12))
    return values


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build high-res n=1 Gbar summaries on a log-spaced s2 grid."
    )
    parser.add_argument(
        "--bounce",
        default="bounce_data_F2_T0.npz",
        help="Bounce file, e.g. bounce_data_F2_T0.npz",
    )
    parser.add_argument("--s2-min", type=float, default=0.0007)
    parser.add_argument("--s2-log-max", type=float, default=0.1)
    parser.add_argument("--s2-max", type=float, default=0.1)
    parser.add_argument("--s2-log-spaced-points", type=int, default=20)
    parser.add_argument(
        "--s2-step-coarse",
        type=float,
        default=0.5,
        help="Coarse step size used from s2-log-max to s2-max.",
    )
    parser.add_argument(
        "--n-evals",
        default="2000",
        help="Comma-separated n_eval values to build (default: 2000).",
    )
    parser.add_argument("--r0", type=float, default=1e-4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite summary outputs if they already exist.",
    )
    parser.add_argument(
        "--overwrite-rk",
        action="store_true",
        help="Rebuild RK data files even if they exist.",
    )
    parser.add_argument(
        "--suffix-template",
        default="neval{n_eval}_spike",
        help="Suffix template for outputs (use {n_eval} if desired).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Output directory (defaults to TRG_DATA_DIR/TRG_OUT_DIR or /Volumes/DP/G_project_data).",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bounce_path = args.bounce
    if not os.path.isfile(bounce_path):
        alt_path = os.path.join(script_dir, args.bounce)
        if os.path.isfile(alt_path):
            bounce_path = alt_path
        else:
            raise FileNotFoundError(f"Bounce file not found: {args.bounce}")

    default_data_dir = "/Volumes/DP/G_project_data"
    data_dir = (
        args.data_dir
        or os.environ.get("TRG_DATA_DIR")
        or os.environ.get("TRG_OUT_DIR")
    )
    if data_dir is None and os.path.isdir(default_data_dir):
        data_dir = default_data_dir
    if data_dir is None:
        data_dir = script_dir

    bdata = np.load(bounce_path, allow_pickle=True)
    false_index = int(bdata["false_index"])
    true_index = int(bdata["true_index"])

    s2_values = build_s2_grid_log(
        args.s2_min,
        args.s2_log_max,
        args.s2_max,
        args.s2_log_spaced_points,
        args.s2_step_coarse,
    )
    n_evals = [int(x) for x in args.n_evals.split(",") if x.strip()]
    for n_eval in n_evals:
        suffix = args.suffix_template.format(n_eval=n_eval).strip()
        for s2 in s2_values:
            tag = s2_tag(s2, 6)
            summary_name = apply_suffix(
                f"gbar_bounce_F{false_index}_T{true_index}_n1_s2{tag}.npz",
                suffix,
            )
            summary_path = os.path.join(data_dir, summary_name)
            if os.path.exists(summary_path) and not args.overwrite:
                print(f"[SKIP] {summary_path} already exists.")
                continue

            rk_name = apply_suffix(f"rk_green_data_n1_s2{tag}.npz", suffix)
            rk_path = os.path.join(data_dir, rk_name)
            if (not os.path.exists(rk_path)) or args.overwrite_rk:
                build_rk_green_for_bounce(
                    bounce_path,
                    s2=s2,
                    n_mode=1,
                    n_eval=n_eval,
                    r0=args.r0,
                    out_fname=rk_path,
                    overwrite=True,
                )

            i_n = integrated_trace_from_rk(rk_path)
            gbar = (1 + 1) ** 2 * i_n

            np.savez(
                summary_path,
                n_mode=1,
                I_n=i_n,
                gbar=gbar,
                bounce_file=bounce_path,
                s2=s2,
            )
            print(f"[SAVE] {summary_path}")


if __name__ == "__main__":
    main()
