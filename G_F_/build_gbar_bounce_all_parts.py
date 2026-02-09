#!/usr/bin/env python3

import argparse
import os
import numpy as np

from rk_builder_adapt import build_rk_green_for_bounce


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


def load_trace_safe(rk_filename: str, rebuild_fn=None) -> float:
    try:
        return integrated_trace_from_rk(rk_filename)
    except Exception as exc:
        print(f"[WARN] Failed to read {rk_filename}: {exc}")
        if os.path.exists(rk_filename):
            os.remove(rk_filename)
            print(f"[WARN] Removed corrupted file {rk_filename}")
        if rebuild_fn is not None:
            print(f"[INFO] Rebuilding {rk_filename}")
            rebuild_fn()
            return integrated_trace_from_rk(rk_filename)
        raise


def build_s2_grid_n0(
    s2_min,
    s2_max,
    alpha,
    delta,
    step1_max,
    step1,
    step2,
    coarse_step,
):
    if s2_min <= 0.0:
        raise ValueError("s2_min must be > 0 for n=0 grid.")

    values = []
    gap_lo = alpha - delta
    gap_hi = alpha + delta

    end1 = min(step1_max, gap_lo, s2_max)
    if end1 > s2_min:
        values.extend(np.arange(s2_min, end1 + 0.5 * step1, step1))

    start2 = max(end1, s2_min)
    end2 = min(gap_lo, s2_max)
    if end2 > start2:
        values.extend(np.arange(start2, end2 + 0.5 * step2, step2))

    start3 = max(gap_hi, s2_min)
    if s2_max > start3:
        values.extend(np.arange(start3, s2_max + 0.5 * coarse_step, coarse_step))

    values.extend([s2_min, s2_max, gap_lo, gap_hi, end1, end2])
    values = [v for v in values if s2_min <= v <= s2_max]
    values = np.unique(np.round(values, 12))
    return values.tolist()


def build_s2_grid_log(s2_min, s2_max, step, log_max, log_spaced_points):
    if s2_min <= 0.0:
        raise ValueError("s2_min must be > 0 for log-spaced IR grid.")

    values = []
    log_max_eff = min(log_max, s2_max)
    if s2_min < log_max_eff:
        decades = np.log10(log_max_eff / s2_min)
        n_log = max(2, int(np.ceil(decades * log_spaced_points)) + 1)
        values.extend(np.logspace(np.log10(s2_min), np.log10(log_max_eff), n_log))
        lin_start = log_max_eff
    else:
        lin_start = s2_min

    values.extend(np.arange(lin_start, s2_max + 0.5 * step, step))
    values = np.append(values, [s2_min, s2_max, log_max_eff])
    if s2_min <= 1.0 <= s2_max:
        values = np.append(values, 1.0)
    values = np.unique(np.round(values, 12))
    return values.tolist()


def build_s2_grid_coarse(s2_min, s2_max, step, include_one=True):
    values = np.arange(0.0, s2_max + 0.5 * step, step)
    values = values[values >= s2_min]
    values = np.append(values, [s2_min, s2_max])
    values = np.unique(np.round(values, 12))
    if (not include_one) and (s2_min < 1.0 < s2_max):
        values = values[np.abs(values - 1.0) > 1e-12]
    return values.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Gbar summaries for the bounce background."
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
        "--s2-n0-step1-max",
        type=float,
        default=0.3,
        help="Upper limit for the 0.1 linear grid (n=0).",
    )
    parser.add_argument(
        "--s2-n0-step1",
        type=float,
        default=0.1,
        help="Step size for the first linear grid (n=0).",
    )
    parser.add_argument(
        "--s2-n0-step2",
        type=float,
        default=0.01,
        help="Step size for the fine grid up to alpha-delta (n=0).",
    )
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
    parser.add_argument("--n-min-npos", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument(
        "--n1-s2-min",
        type=float,
        default=0.0007,
        help="n=1 log grid minimum (high-res).",
    )
    parser.add_argument(
        "--n1-s2-log-max",
        type=float,
        default=0.1,
        help="n=1 log grid max (high-res).",
    )
    parser.add_argument(
        "--n1-s2-max",
        type=float,
        default=0.1,
        help="n=1 grid max (high-res).",
    )
    parser.add_argument(
        "--n1-s2-log-spaced-points",
        type=int,
        default=20,
        help="n=1 log-grid points per decade (high-res).",
    )
    parser.add_argument(
        "--n1-s2-step-coarse",
        type=float,
        default=0.5,
        help="n=1 coarse step from n1-s2-log-max to n1-s2-max.",
    )
    parser.add_argument(
        "--n1-n-evals",
        default="2000",
        help="Comma-separated n_eval values for n=1 high-res builds.",
    )
    parser.add_argument(
        "--n1-suffix-template",
        default="neval{n_eval}_spike",
        help="Suffix template for n=1 outputs.",
    )
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
        help="Output directory (defaults to TRG_DATA_DIR/TRG_OUT_DIR or /Volumes/DP/G_project_data).",
    )
    parser.set_defaults(build_missing_rk=True)
    parser.set_defaults(merge=True)
    args = parser.parse_args()
    if args.overwrite and args.merge:
        raise ValueError("Use either --overwrite or --merge, not both.")

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
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        os.chdir(data_dir)
        print(f"[INFO] Writing outputs to {data_dir}")

    bdata = np.load(bounce_path, allow_pickle=True)
    false_index = int(bdata["false_index"])
    true_index = int(bdata["true_index"])

    s2_values_n0 = build_s2_grid_n0(
        args.s2_min,
        args.s2_max,
        args.alpha,
        args.delta,
        args.s2_n0_step1_max,
        args.s2_n0_step1,
        args.s2_n0_step2,
        args.s2_step_coarse,
    )
    s2_values_n1 = build_s2_grid_log(
        args.n1_s2_min,
        args.n1_s2_max,
        args.n1_s2_step_coarse,
        args.n1_s2_log_max,
        args.n1_s2_log_spaced_points,
    )
    s2_values_nsum = build_s2_grid_coarse(
        args.s2_min,
        args.s2_max,
        args.s2_step_coarse,
        include_one=False,
    )
    built_rk = set()
    n_min_effective = max(args.n_min_npos, 2)

    for s2 in s2_values_n0:
        tag = s2_tag(s2, 6)
        summary_name = f"gbar_bounce_F{false_index}_T{true_index}_n0_s2{tag}.npz"
        if os.path.exists(summary_name) and not args.overwrite and not args.merge:
            print(f"[SKIP] {summary_name} already exists.")
            continue

        existing = None
        if args.merge and os.path.exists(summary_name):
            prev = np.load(summary_name, allow_pickle=True)
            existing = {
                "I_n": float(prev["I_n"]),
                "gbar": float(prev["gbar"]),
            }

        rk_name = f"rk_green_data_n0_s2{tag}.npz"
        def rebuild_rk(n_mode=0, rk_name=rk_name, s2=s2):
            build_rk_green_for_bounce(
                bounce_path,
                s2=s2,
                n_mode=n_mode,
                n_eval=args.n_eval,
                r0=args.r0,
                out_fname=rk_name,
                overwrite=True,
            )
            built_rk.add(rk_name)
        i_n = None
        gbar = None
        if os.path.exists(rk_name):
            if args.overwrite_rk:
                if rk_name not in built_rk:
                    build_rk_green_for_bounce(
                        bounce_path,
                        s2=s2,
                        n_mode=0,
                        n_eval=args.n_eval,
                        r0=args.r0,
                        out_fname=rk_name,
                        overwrite=True,
                    )
                    built_rk.add(rk_name)
            i_n = load_trace_safe(rk_name, rebuild_rk)
            gbar = (0 + 1) ** 2 * i_n
        elif args.build_missing_rk:
            if rk_name not in built_rk:
                build_rk_green_for_bounce(
                    bounce_path,
                    s2=s2,
                    n_mode=0,
                    n_eval=args.n_eval,
                    r0=args.r0,
                    out_fname=rk_name,
                    overwrite=False,
                )
                built_rk.add(rk_name)
            i_n = load_trace_safe(rk_name, rebuild_rk)
            gbar = (0 + 1) ** 2 * i_n
        elif existing is not None:
            i_n = existing["I_n"]
            gbar = existing["gbar"]
        else:
            raise FileNotFoundError(
                f"Missing RK file {rk_name}. Use --build-missing-rk to create it."
            )

        np.savez(
            summary_name,
            n_mode=0,
            I_n=i_n,
            gbar=gbar,
            bounce_file=bounce_path,
            s2=s2,
        )
        print(f"[SAVE] {summary_name}")

    n1_n_evals = [int(x) for x in args.n1_n_evals.split(",") if x.strip()]
    for n_eval in n1_n_evals:
        suffix = args.n1_suffix_template.format(n_eval=n_eval).strip()
        if suffix and not suffix.startswith("_"):
            suffix = "_" + suffix
        for s2 in s2_values_n1:
            tag = s2_tag(s2, 6)
            summary_name = f"gbar_bounce_F{false_index}_T{true_index}_n1_s2{tag}{suffix}.npz"
            if os.path.exists(summary_name) and not args.overwrite:
                print(f"[SKIP] {summary_name} already exists.")
                continue

            rk_name = f"rk_green_data_n1_s2{tag}{suffix}.npz"
            def rebuild_rk(n_mode=1, rk_name=rk_name, s2=s2, n_eval=n_eval):
                build_rk_green_for_bounce(
                    bounce_path,
                    s2=s2,
                    n_mode=n_mode,
                    n_eval=n_eval,
                    r0=args.r0,
                    out_fname=rk_name,
                    overwrite=True,
                )
                built_rk.add(rk_name)

            if os.path.exists(rk_name):
                if args.overwrite_rk:
                    if rk_name not in built_rk:
                        build_rk_green_for_bounce(
                            bounce_path,
                            s2=s2,
                            n_mode=1,
                            n_eval=n_eval,
                            r0=args.r0,
                            out_fname=rk_name,
                            overwrite=True,
                        )
                        built_rk.add(rk_name)
            elif args.build_missing_rk:
                if rk_name not in built_rk:
                    build_rk_green_for_bounce(
                        bounce_path,
                        s2=s2,
                        n_mode=1,
                        n_eval=n_eval,
                        r0=args.r0,
                        out_fname=rk_name,
                        overwrite=False,
                    )
                    built_rk.add(rk_name)
            else:
                raise FileNotFoundError(
                    f"Missing RK file {rk_name}. Use --build-missing-rk to create it."
                )

            i_n = load_trace_safe(rk_name, rebuild_rk)
            gbar = (1 + 1) ** 2 * i_n

            np.savez(
                summary_name,
                n_mode=1,
                I_n=i_n,
                gbar=gbar,
                bounce_file=bounce_path,
                s2=s2,
            )
            print(f"[SAVE] {summary_name}")

    for s2 in s2_values_nsum:
        tag = s2_tag(s2, 6)
        summary_name = f"gbar_bounce_F{false_index}_T{true_index}_npos_s2{tag}.npz"
        if os.path.exists(summary_name) and not args.overwrite and not args.merge:
            print(f"[SKIP] {summary_name} already exists.")
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
                if n_min_effective <= n_int <= args.n_max:
                    merged[n_int] = (float(i_val), float(g_val))

        for n_mode in range(n_min_effective, args.n_max + 1):
            rk_name = f"rk_green_data_n{n_mode}_s2{tag}.npz"
            if n_mode in merged and not args.overwrite_rk:
                continue
            def rebuild_rk(n_mode=n_mode, rk_name=rk_name, s2=s2):
                build_rk_green_for_bounce(
                    bounce_path,
                    s2=s2,
                    n_mode=n_mode,
                    n_eval=args.n_eval,
                    r0=args.r0,
                    out_fname=rk_name,
                    overwrite=True,
                )
                built_rk.add(rk_name)
            if os.path.exists(rk_name):
                if args.overwrite_rk:
                    if rk_name not in built_rk:
                        build_rk_green_for_bounce(
                            bounce_path,
                            s2=s2,
                            n_mode=n_mode,
                            n_eval=args.n_eval,
                            r0=args.r0,
                            out_fname=rk_name,
                            overwrite=True,
                        )
                        built_rk.add(rk_name)
                i_n = load_trace_safe(rk_name, rebuild_rk)
                gbar = (n_mode + 1) ** 2 * i_n
                merged[n_mode] = (i_n, gbar)
            elif args.build_missing_rk:
                if rk_name not in built_rk:
                    build_rk_green_for_bounce(
                        bounce_path,
                        s2=s2,
                        n_mode=n_mode,
                        n_eval=args.n_eval,
                        r0=args.r0,
                        out_fname=rk_name,
                        overwrite=False,
                    )
                    built_rk.add(rk_name)
                i_n = load_trace_safe(rk_name, rebuild_rk)
                gbar = (n_mode + 1) ** 2 * i_n
                merged[n_mode] = (i_n, gbar)
            elif n_mode not in merged:
                raise FileNotFoundError(
                    f"Missing RK file {rk_name} and no existing summary entry. "
                    "Use --build-missing-rk to create it."
                )

        n_values = np.arange(n_min_effective, args.n_max + 1, dtype=int)
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
            n_min=n_min_effective,
            n_max=args.n_max,
        )
        print(f"[SAVE] {summary_name}")


if __name__ == "__main__":
    main()
