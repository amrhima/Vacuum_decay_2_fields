#!/usr/bin/env python3
"""
compute_gbar_fv_wkb_vfinal4.py -- WKB Stage-A version of
compute_gbar_fv.py (FV-side G_bar summaries for n >= 0).

This is the FALSE-VACUUM side of the two-field fluctuation-determinant
pipeline.  The G_bar summaries produced here are the homogeneous false-vacuum
reference whose contribution is *subtracted* from the corresponding
bounce-side G_bar summaries, so that the (bounce - false-vacuum) difference
yields the finite renormalised fluctuation determinant.  Because the
subtraction is performed mode-by-mode in the angular index n, this FV side
MUST supply the lowest modes n=0 and n=1 for every s2 value -- if either is
absent the bounce-side modes have nothing to cancel against.  That is exactly
why the per-s2 loop enforces the "FV summary missing n=0/1" guard below
(a RuntimeError is raised rather than silently writing an incomplete file).

Output files: gbar_fv_F{F}_T{T}_npos_s2{tag}_wkb_vfinal4.npz
"""

import argparse
import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from config import DATA_DIR
from rk_builder_fv_wkb_vfinal4_git import build_rk_green_fv_for_bounce_wkb


def s2_tag(s2: float, digits: int = 6) -> str:
    return f"{s2:.{digits}f}".replace(".", "p")


def integrated_trace_from_rk(rk_filename: str) -> float:
    data = np.load(rk_filename, allow_pickle=True)
    r_grid = data["r_grid"]
    # [vx-opt] Prefer the precomputed diagonal trace if present; fall
    # back to the old dense G_rk diagonal loop for legacy npz files.
    if "trace_diag" in data.files:
        trace_diag = data["trace_diag"]
    else:
        G_rk = data["G_rk"]
        nr = len(r_grid)
        trace_diag = np.empty(nr, dtype=float)
        for k in range(nr):
            trace_diag[k] = np.trace(G_rk[k, k, :, :])
    integrand = (r_grid ** 3) * trace_diag
    return float(np.trapezoid(integrand, r_grid))


def build_s2_grid_hybrid(s2_min, s2_max, fine_step, tail_step, tail_start):
    fine = np.arange(0.0, tail_start + 0.5 * fine_step, fine_step)
    fine = fine[fine >= s2_min]
    if s2_max <= tail_start:
        values = fine
    else:
        tail = np.arange(tail_start + tail_step,
                         s2_max + 0.5 * tail_step, tail_step)
        tail = tail[tail <= s2_max]
        values = np.concatenate([fine, tail])
    values = np.append(values, [s2_min, s2_max])
    values = np.unique(np.round(values, 12))
    return values.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FV G_bar summaries (WKB Stage A)."
    )
    parser.add_argument("--bounce", default="bounce_data_F2_T0.npz")
    parser.add_argument("--s2-min", type=float, default=1e-3)
    parser.add_argument("--s2-max", type=float, default=1000.0)
    parser.add_argument("--s2-step-coarse", type=float, default=0.5)
    parser.add_argument("--s2-tail-step", type=float, default=5.0)
    parser.add_argument("--s2-tail-start", type=float, default=10.0)
    parser.add_argument("--n-min-npos", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=18,
                        help="WKB Stage A: matches v2_fit n_cut=18.")
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--r0", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--tag", default=None,
                        help="Explicit tag for output filenames (e.g. "
                             "'F2_T0').  See compute_gbar_npos_wkb_vfinal4.py "
                             "for resolution rules.")
    args = parser.parse_args()

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

    data_dir = args.data_dir or DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    print(f"[INFO] Writing outputs to {data_dir}")

    bdata = np.load(bounce_path, allow_pickle=True)
    false_index = int(bdata["false_index"])
    true_index = int(bdata["true_index"])
    # Output-filename label resolution (priority):
    #   1. --tag CLI argument (explicit override)
    #   2. bdata["tag"] if alphanumeric+underscore (safe filename)
    #   3. fallback F<false_index>_T<true_index>
    # Prevents using a verbose physics label like
    # 'F(M2, V=-0.0036) -> T(M0, V=-1.8854)' as a filename component.
    import re
    if args.tag is not None:
        ft_label = args.tag
    elif "tag" in bdata.files:
        raw_tag = str(bdata["tag"])
        if re.fullmatch(r"[A-Za-z0-9_]+", raw_tag):
            ft_label = raw_tag
        else:
            print(f"[INFO] bdata['tag']={raw_tag!r} is not a clean filename; "
                  f"falling back to F{false_index}_T{true_index}.")
            ft_label = f"F{false_index}_T{true_index}"
    else:
        ft_label = f"F{false_index}_T{true_index}"
    print(f"[INFO] output filename tag: '{ft_label}'")

    s2_values_npos = build_s2_grid_hybrid(
        args.s2_min, args.s2_max,
        fine_step=args.s2_step_coarse,
        tail_step=args.s2_tail_step,
        tail_start=args.s2_tail_start,
    )
    print(f"[INFO] WKB Stage A active: n_max={args.n_max}, "
          f"s2_max={args.s2_max}, {len(s2_values_npos)} s2 values")

    for s2 in s2_values_npos:
        tag = s2_tag(s2, 6)
        summary_name = os.path.join(
            data_dir,
            f"gbar_fv_{ft_label}_npos_s2{tag}_wkb_vfinal4.npz")

        if os.path.exists(summary_name) and not args.overwrite:
            prev = np.load(summary_name, allow_pickle=True)
            prev_n = set(prev["n_values"].astype(int).tolist())
            needed_n = set(range(args.n_min_npos, args.n_max + 1))
            if needed_n.issubset(prev_n):
                print(f"[SKIP] {summary_name} already complete.")
                continue

        merged = {}
        if os.path.exists(summary_name):
            prev = np.load(summary_name, allow_pickle=True)
            prev_n = prev["n_values"].astype(int)
            prev_i = prev["I_n"].astype(float)
            prev_g = prev["gbar_n"].astype(float)
            for n_val, i_val, g_val in zip(prev_n, prev_i, prev_g):
                n_int = int(n_val)
                if args.n_min_npos <= n_int <= args.n_max:
                    merged[n_int] = (float(i_val), float(g_val))

        for n_mode in range(args.n_min_npos, args.n_max + 1):
            rk_name = os.path.join(
                data_dir,
                f"rk_green_data_FV_{ft_label}_n{n_mode}_s2{tag}_wkb_vfinal4.npz")
            if not os.path.exists(rk_name) and n_mode not in merged:
                build_rk_green_fv_for_bounce_wkb(
                    bounce_path, s2=s2, n_mode=n_mode,
                    n_eval=args.n_eval, r0=args.r0,
                    out_fname=rk_name, overwrite=False,
                )

            if os.path.exists(rk_name):
                i_n = integrated_trace_from_rk(rk_name)
                gbar = (n_mode + 1) ** 2 * i_n
                merged[n_mode] = (i_n, gbar)

        for required_n in (0, 1):
            if required_n not in merged:
                raise RuntimeError(
                    f"FV summary missing n={required_n} at s2={s2:.6f}.")

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
            wkb_stage="A",
        )
        print(f"[SAVE] {summary_name}")

    print("[DONE WKB Stage-A FV]")


if __name__ == "__main__":
    main()
