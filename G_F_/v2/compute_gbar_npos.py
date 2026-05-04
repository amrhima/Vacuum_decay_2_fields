#!/usr/bin/env python3
"""
compute_gbar_npos.py -- Build bounce-side G_bar summaries for n >= 2.

For each s2 on a coarse grid, solves the RK system for n=2..n_max
and saves per-n G_bar values.

Output: gbar_bounce_F{F}_T{T}_npos_s2{tag}.npz  (one file per s2)
"""

import argparse
import os
import numpy as np

from rk_builder_adapt_v2 import build_rk_green_for_bounce
from config import DATA_DIR


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


def integrated_trace_from_rk(rk_filename: str) -> float:
    data = np.load(rk_filename, allow_pickle=True)
    r_grid = data["r_grid"]
    g_rk = data["G_rk"]
    nr = len(r_grid)
    trace_diag = np.empty(nr, dtype=float)
    for k in range(nr):
        trace_diag[k] = np.trace(g_rk[k, k, :, :])
    integrand = (r_grid ** 3) * trace_diag
    return float(np.trapezoid(integrand, r_grid))


def build_s2_grid_coarse(s2_min, s2_max, step):
    values = np.arange(0.0, s2_max + 0.5 * step, step)
    values = values[values >= s2_min]
    values = np.append(values, [s2_min, s2_max])
    values = np.unique(np.round(values, 12))
    return values.tolist()


def resolve_bounce(path, script_dir):
    if os.path.isfile(path):
        return path
    for search_dir in [script_dir,
                       os.path.join(script_dir, "data_files"),
                       DATA_DIR]:
        alt = os.path.join(search_dir, os.path.basename(path))
        if os.path.isfile(alt):
            return alt
    raise FileNotFoundError(f"Bounce file not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build bounce G_bar summaries for n >= 2."
    )
    parser.add_argument("--bounce", default="bounce_data_F2_T0.npz")
    parser.add_argument("--s2-min", type=float, default=1e-3)
    parser.add_argument("--s2-max", type=float, default=10.0)
    parser.add_argument("--s2-step", type=float, default=0.5)
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=50)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--r0", type=float, default=1e-4)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-rk", action="store_true")
    parser.add_argument("--merge", action="store_true", default=True)
    parser.add_argument("--no-merge", action="store_false", dest="merge")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    if args.n_min < 2:
        raise ValueError("--n-min must be >= 2.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bounce_path = resolve_bounce(args.bounce, script_dir)
    print(f"[INFO] Bounce file: {bounce_path}")

    data_dir = args.data_dir
    if data_dir is None:
        dp = DATA_DIR
        if os.path.isdir(dp):
            data_dir = dp
        else:
            data_dir_candidate = os.path.join(script_dir, "data_files")
            if os.path.isdir(data_dir_candidate):
                data_dir = data_dir_candidate
            else:
                data_dir = script_dir
    os.makedirs(data_dir, exist_ok=True)
    os.chdir(data_dir)
    print(f"[INFO] Writing outputs to {data_dir}")

    bdata = np.load(bounce_path, allow_pickle=True)
    false_index = int(bdata["false_index"])
    true_index = int(bdata["true_index"])

    s2_values = build_s2_grid_coarse(args.s2_min, args.s2_max, args.s2_step)

    print(f"[INFO] {len(s2_values)} s2 values in "
          f"[{s2_values[0]:.4f}, {s2_values[-1]:.4f}]")
    print(f"[INFO] n range: [{args.n_min}, {args.n_max}]")

    built_rk = set()

    for s2 in s2_values:
        tag = s2_tag(s2, 6)
        summary_name = apply_suffix(
            f"gbar_bounce_F{false_index}_T{true_index}_npos_s2{tag}.npz",
            args.suffix,
        )
        if os.path.exists(summary_name) and not args.overwrite:
            if not args.merge:
                print(f"[SKIP] {summary_name} already exists.")
                continue
            # merge mode: check if all n values already present
            prev = np.load(summary_name, allow_pickle=True)
            prev_n = set(prev["n_values"].astype(int).tolist())
            needed_n = set(range(args.n_min, args.n_max + 1))
            if needed_n.issubset(prev_n):
                print(f"[SKIP] {summary_name} already complete (n={args.n_min}..{args.n_max}).")
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
                if args.n_min <= n_int <= args.n_max:
                    merged[n_int] = (float(i_val), float(g_val))

        for n_mode in range(args.n_min, args.n_max + 1):
            rk_name = apply_suffix(f"rk_green_data_n{n_mode}_s2{tag}.npz",
                                   args.suffix)
            if os.path.exists(rk_name):
                if args.overwrite_rk:
                    if rk_name not in built_rk:
                        build_rk_green_for_bounce(
                            bounce_path, s2=s2, n_mode=n_mode,
                            n_eval=args.n_eval, r0=args.r0,
                            out_fname=rk_name, overwrite=True,
                        )
                        built_rk.add(rk_name)
            elif n_mode not in merged:
                if rk_name not in built_rk:
                    build_rk_green_for_bounce(
                        bounce_path, s2=s2, n_mode=n_mode,
                        n_eval=args.n_eval, r0=args.r0,
                        out_fname=rk_name, overwrite=False,
                    )
                    built_rk.add(rk_name)

            if os.path.exists(rk_name):
                i_n = integrated_trace_from_rk(rk_name)
                gbar = (n_mode + 1) ** 2 * i_n
                merged[n_mode] = (i_n, gbar)

        n_values = np.arange(args.n_min, args.n_max + 1, dtype=int)
        i_values = np.array([merged[n][0] for n in n_values], dtype=float)
        gbar_values = np.array([merged[n][1] for n in n_values], dtype=float)
        total_sum = float(np.sum(gbar_values))

        np.savez(
            summary_name,
            n_values=n_values,
            I_n=i_values,
            gbar_n=gbar_values,
            total_sum=total_sum,
            bounce_file=bounce_path,
            s2=s2,
            n_min=args.n_min,
            n_max=args.n_max,
        )
        print(f"[SAVE] {summary_name}  total_sum={total_sum:.6f}")

    print("[DONE]")


if __name__ == "__main__":
    main()
