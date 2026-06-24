#!/usr/bin/env python3
"""
compute_gbar_npos_wkb_vfinal4.py -- WKB Stage-A version of
compute_gbar_npos.py (bounce-side G_bar summaries for n >= 2).

Calls rk_builder_adapt_wkb_vfinal4.build_rk_green_for_bounce_wkb instead
of the non-WKB version.  Output files get `_wkb_vfinal4` suffix:

  gbar_bounce_F{F}_T{T}_npos_s2{tag}_wkb_vfinal4.npz

Hybrid grid (s^2 up to 50 with step 5 above 10) and n_max=18 are the
defaults, matching the non-WKB version.

Quantities per partial wave n at fixed spectral parameter s^2:
  I_n     = integral_0^inf r^3 tr[G_rk(r,r)] dr -- the integrated
            (radially traced) bounce-side rk-Green's-function diagonal.
  gbar_n  = (n+1)^2 * I_n -- the I_n weighted by the SO(4) angular
            multiplicity (n+1)^2 of the n-th partial wave.
  s^2     = the spectral / mass-shift parameter at which each radial
            Green's function is evaluated.
n starts at 2 here because the n=0 and n=1 partial waves are handled
separately on the finite-difference (FD) side (where the zero/translation
modes and collective coordinates require dedicated treatment); this WKB
bounce-side script supplies only the n >= 2 contributions.
"""

import argparse
import os
import sys
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from rk_builder_adapt_wkb_vfinal4_git import build_rk_green_for_bounce_wkb

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
    # [vx-opt] Prefer the precomputed diagonal trace if present; fall
    # back to the old dense G_rk diagonal loop for legacy npz files.
    if "trace_diag" in data.files:
        trace_diag = data["trace_diag"]
    else:
        g_rk = data["G_rk"]
        nr = len(r_grid)
        trace_diag = np.empty(nr, dtype=float)
        for k in range(nr):
            trace_diag[k] = np.trace(g_rk[k, k, :, :])
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
        description="Build bounce G_bar summaries for n >= 2 (WKB Stage A)."
    )
    parser.add_argument("--bounce", default="bounce_data_F2_T0.npz")
    parser.add_argument("--s2-min", type=float, default=1e-3)
    parser.add_argument("--s2-max", type=float, default=1000.0)
    parser.add_argument("--s2-step", type=float, default=0.5)
    parser.add_argument("--s2-tail-step", type=float, default=5.0)
    parser.add_argument("--s2-tail-start", type=float, default=10.0)
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=18,
                        help="WKB Stage A: same default as v2_fit (n=18).")
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--r0", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--tag", default=None,
                        help="Explicit tag for output filenames (e.g. "
                             "'F2_T0').  If omitted, falls back to "
                             "F<false_index>_T<true_index>; the "
                             "bdata['tag'] field is used only if it is "
                             "alphanumeric+underscore (avoids garbage like "
                             "'F(M2, V=-0.0036) -> T(M0, ...)' from "
                             "physics-label tags).")
    args = parser.parse_args()

    if args.n_min < 2:
        raise ValueError("--n-min must be >= 2.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bounce_path = resolve_bounce(args.bounce, script_dir)
    print(f"[INFO] Bounce file: {bounce_path}")
    print(f"[INFO] WKB Stage A active: n_max={args.n_max}, s2_max={args.s2_max}")

    data_dir = args.data_dir or DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    print(f"[INFO] Writing outputs to {data_dir}")

    bdata = np.load(bounce_path, allow_pickle=True)
    false_index = int(bdata["false_index"])
    true_index = int(bdata["true_index"])
    # Output-filename label resolution (priority):
    #   1. --tag CLI argument (explicit override)
    #   2. bdata["tag"] if it is alphanumeric+underscore (safe filename)
    #   3. fallback F<false_index>_T<true_index>
    # The bdata["tag"] check prevents using a verbose physics label like
    # 'F(M2, V=-0.0036) -> T(M0, V=-1.8854)' (which would produce
    # invalid filenames with spaces, parentheses, commas).
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

    s2_values = build_s2_grid_hybrid(
        args.s2_min, args.s2_max,
        fine_step=args.s2_step,
        tail_step=args.s2_tail_step,
        tail_start=args.s2_tail_start)

    print(f"[INFO] {len(s2_values)} s2 values in "
          f"[{s2_values[0]:.4f}, {s2_values[-1]:.4f}]")
    print(f"[INFO] n range: [{args.n_min}, {args.n_max}]")

    for s2 in s2_values:
        tag = s2_tag(s2, 6)
        summary_name = os.path.join(
            data_dir,
            f"gbar_bounce_{ft_label}_npos_s2{tag}_wkb_vfinal4.npz")

        if os.path.exists(summary_name) and not args.overwrite:
            prev = np.load(summary_name, allow_pickle=True)
            prev_n = set(prev["n_values"].astype(int).tolist())
            needed_n = set(range(args.n_min, args.n_max + 1))
            if needed_n.issubset(prev_n):
                print(f"[SKIP] {summary_name} already complete "
                      f"(n={args.n_min}..{args.n_max}).")
                continue

        merged = {}
        if os.path.exists(summary_name):
            prev = np.load(summary_name, allow_pickle=True)
            prev_n = prev["n_values"].astype(int)
            prev_i = prev["I_n"].astype(float)
            prev_g = prev["gbar_n"].astype(float)
            for n_val, i_val, g_val in zip(prev_n, prev_i, prev_g):
                n_int = int(n_val)
                if args.n_min <= n_int <= args.n_max:
                    merged[n_int] = (float(i_val), float(g_val))

        for n_mode in range(args.n_min, args.n_max + 1):
            rk_name = os.path.join(
                data_dir,
                f"rk_green_data_{ft_label}_n{n_mode}_s2{tag}_wkb_vfinal4.npz")
            if not os.path.exists(rk_name) and n_mode not in merged:
                build_rk_green_for_bounce_wkb(
                    bounce_path, s2=s2, n_mode=n_mode,
                    n_eval=args.n_eval, r0=args.r0,
                    out_fname=rk_name, overwrite=False,
                )

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
            wkb_stage="A",
        )
        print(f"[SAVE] {summary_name}  total_sum={total_sum:.6f}")

    print("[DONE WKB Stage-A npos]")


if __name__ == "__main__":
    main()
