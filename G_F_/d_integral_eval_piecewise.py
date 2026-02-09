#!/usr/bin/env python3

#Evaluate the D integral using precomputed Gbar summaries (default n_max=50).

import argparse
import os
import numpy as np
import glob


def s2_tag(s2: float, digits: int = 6) -> str:
    return f"{s2:.{digits}f}".replace(".", "p")


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
    gap_low = alpha - delta
    gap_high = alpha + delta

    end1 = min(step1_max, gap_low, s2_max)
    if end1 > s2_min:
        values.extend(np.arange(s2_min, end1 + 0.5 * step1, step1))

    start2 = max(end1, s2_min)
    end2 = min(gap_low, s2_max)
    if end2 > start2:
        values.extend(np.arange(start2, end2 + 0.5 * step2, step2))

    start3 = max(gap_high, s2_min)
    if s2_max > start3:
        values.extend(np.arange(start3, s2_max + 0.5 * coarse_step, coarse_step))

    values.extend([s2_min, s2_max, gap_low, gap_high, end1, end2])
    values = [v for v in values if s2_min <= v <= s2_max]
    values = np.unique(np.round(values, 12))
    return values


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
    return values


def build_s2_grid_coarse(s2_min, s2_max, step, include_one=True):
    values = np.arange(0.0, s2_max + 0.5 * step, step)
    values = values[values >= s2_min]
    values = np.append(values, [s2_min, s2_max])
    values = np.unique(np.round(values, 12))
    if (not include_one) and (s2_min < 1.0 < s2_max):
        values = values[np.abs(values - 1.0) > 1e-12]
    return values


def load_gbar_n0(prefix, false_index, true_index, s2, data_dir):
    tag = s2_tag(s2, 6)
    name = f"gbar_{prefix}_F{false_index}_T{true_index}_n0_s2{tag}.npz"
    path = os.path.join(data_dir, name) if data_dir else name
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {prefix} n0 summary: {path}")
    data = np.load(path, allow_pickle=True)
    if "gbar" not in data.files:
        raise KeyError(f"{path} missing 'gbar'")
    return float(data["gbar"])


def load_gbar_n1(false_index, true_index, s2, data_dir):
    tag = s2_tag(s2, 6)
    name = f"gbar_bounce_F{false_index}_T{true_index}_n1_s2{tag}.npz"
    path = os.path.join(data_dir, name) if data_dir else name
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing bounce n1 summary: {path}")
    data = np.load(path, allow_pickle=True)
    if "gbar" not in data.files:
        raise KeyError(f"{path} missing 'gbar'")
    return float(data["gbar"])


def s2_from_path(path: str) -> float:
    base = os.path.basename(path)
    tag = base.split("_s2", 1)[1].rsplit(".npz", 1)[0]
    tag = tag.split("_", 1)[0]
    return float(tag.replace("p", "."))


def load_gbar_n1_records(search_dirs, false_index, true_index, suffix):
    suffix = suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    paths = []
    seen = set()
    for data_dir in search_dirs:
        pattern = os.path.join(
            data_dir,
            f"gbar_bounce_F{false_index}_T{true_index}_n1_s2*{suffix}.npz",
        )
        for path in glob.glob(pattern):
            if path in seen:
                continue
            seen.add(path)
            paths.append(path)
    if not paths:
        return None
    records = []
    for path in paths:
        s2 = s2_from_path(path)
        gbar = float(np.load(path, allow_pickle=True)["gbar"])
        records.append((s2, gbar))
    records.sort(key=lambda x: x[0])
    s2_vals = np.array([r[0] for r in records], dtype=float)
    gbar_vals = np.array([r[1] for r in records], dtype=float)
    return s2_vals, gbar_vals


def cumulative_trapezoid(x, y):
    out = np.zeros_like(y, dtype=float)
    for i in range(1, len(x)):
        out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1])
    return out


def load_gbar_npos(prefix, false_index, true_index, s2, data_dir):
    tag = s2_tag(s2, 6)
    name = f"gbar_{prefix}_F{false_index}_T{true_index}_npos_s2{tag}.npz"
    path = os.path.join(data_dir, name) if data_dir else name
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {prefix} npos summary: {path}")
    data = np.load(path, allow_pickle=True)
    n_values = data["n_values"].astype(int)
    if "gbar_n" in data.files:
        gbar_n = data["gbar_n"].astype(float)
    elif "contrib" in data.files:
        gbar_n = data["contrib"].astype(float)
    else:
        raise KeyError(f"{path} missing 'gbar_n' or 'contrib'")
    return n_values, gbar_n


def extract_sum_range(n_values, gbar_n, n_min, n_max):
    n_values = np.asarray(n_values, dtype=int)
    gbar_n = np.asarray(gbar_n, dtype=float)
    if np.min(n_values) > n_min or np.max(n_values) < n_max:
        raise ValueError(
            f"npos summaries must cover n={n_min}..{n_max} (found {n_values.min()}..{n_values.max()}). "
            "Rebuild FV summaries with --n-min-npos 0."
        )
    if not np.any(n_values == n_min):
        raise ValueError(
            f"npos summaries missing n={n_min}. Rebuild FV summaries with --n-min-npos 0."
        )
    mask = (n_values >= n_min) & (n_values <= n_max)
    return float(np.sum(gbar_n[mask])) if np.any(mask) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute D using precomputed Gbar summaries."
    )
    parser.add_argument(
        "--bounce",
        default="bounce_data_F2_T0.npz",
        help="Bounce file, e.g. bounce_data_F2_T0.npz",
    )
    parser.add_argument("--s2-min", type=float, default=1e-3)
    parser.add_argument("--s2-max", type=float, default=10.0)
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
    parser.add_argument("--n-max", type=int, default=50)
    parser.add_argument(
        "--n-sum-min",
        type=int,
        default=2,
        help="Start n for the summed n>=2 contribution.",
    )
    parser.add_argument(
        "--disable-uv-term",
        action="store_true",
        help="Disable the -4/s2 * Heaviside(1-s2) term.",
    )
    parser.add_argument(
        "--ct-coeff",
        type=float,
        default=4.0,
        help="Coefficient for the n=1 counterterm (default 4).",
    )
    parser.add_argument(
        "--ct-form",
        choices=["inv", "inv-shift"],
        default="inv",
        help="Counterterm form: inv=ct_coeff/s2, inv-shift=ct_coeff/(s2+ct_s2_shift).",
    )
    parser.add_argument(
        "--ct-s2-shift",
        type=float,
        default=0.0,
        help="Shift for inv-shift counterterm (s2 -> s2 + shift).",
    )
    parser.add_argument(
        "--ct-auto-fit",
        action="store_true",
        help="Auto-fit ct-coeff from median(s2 * gbar_n1) in the counterterm window.",
    )
    parser.add_argument(
        "--ct-s2-min",
        type=float,
        default=1e-3,
        help="Turn on the n=1 counterterm above this s2 (window lower bound).",
    )
    parser.add_argument(
        "--ct-s2-max",
        type=float,
        default=1.0,
        help="Turn off the n=1 counterterm above this s2 (window upper bound).",
    )
    parser.add_argument(
        "--n1-suffix",
        default="neval2000_spike",
        help="Suffix for n=1 summaries (aligned with split n=1 code).",
    )
    parser.add_argument(
        "--n1-peak-min",
        type=float,
        default=0.007,
        help="Lower s2 bound for n=1 peak search.",
    )
    parser.add_argument(
        "--n1-peak-max",
        type=float,
        default=0.01,
        help="Upper s2 bound for n=1 peak search.",
    )
    parser.add_argument(
        "--n1-integral-end",
        type=float,
        default=0.1,
        help="Upper bound s2_end for n=1 log segment.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Input directory (defaults to TRG_DATA_DIR/TRG_OUT_DIR or /Volumes/DP/G_project_data).",
    )
    args = parser.parse_args()

    if args.s2_min <= 0.0 and not args.disable_uv_term:
        raise ValueError("s2_min must be > 0 when using the 4/s2 UV term.")
    if not args.disable_uv_term and args.ct_s2_min >= args.ct_s2_max:
        raise ValueError("--ct-s2-min must be less than --ct-s2-max.")
    if args.ct_s2_shift < 0.0:
        raise ValueError("--ct-s2-shift must be >= 0.")

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
        print(f"[INFO] Reading data from {data_dir}")

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
    s2_values_nsum = build_s2_grid_coarse(
        args.s2_min,
        args.s2_max,
        args.s2_step_coarse,
        include_one=False,
    )

    gbar_n0 = np.array(
        [
            load_gbar_n0("bounce", false_index, true_index, s2, data_dir)
            for s2 in s2_values_n0
        ],
        dtype=float,
    )

    # n=1 alignment with split code: use summary records + target-based grid.
    search_dirs = [d for d in (data_dir, default_data_dir, script_dir) if d and os.path.isdir(d)]
    n1_records = load_gbar_n1_records(search_dirs, false_index, true_index, args.n1_suffix)
    if n1_records is None:
        raise FileNotFoundError("Missing n=1 summaries for requested suffix.")
    s2_vals_n1_raw, gbar_vals_n1_raw = n1_records

    gbar_sum = []
    gbar_sum_fv_all = []
    for s2 in s2_values_nsum:
        n_vals, gbar_vals = load_gbar_npos(
            "bounce", false_index, true_index, s2, data_dir
        )
        n_vals_fv, gbar_vals_fv = load_gbar_npos(
            "fv", false_index, true_index, s2, data_dir
        )
        gsum = extract_sum_range(n_vals, gbar_vals, args.n_sum_min, args.n_max)
        gsum_fv = extract_sum_range(n_vals_fv, gbar_vals_fv, 0, args.n_max)
        gbar_sum.append(gsum)
        gbar_sum_fv_all.append(gsum_fv)

    gbar_sum = np.array(gbar_sum, dtype=float)
    gbar_sum_fv_all = np.array(gbar_sum_fv_all, dtype=float)

    gap_low = args.alpha - args.delta
    gap_high = args.alpha + args.delta

    def integrate_n0(s2_max):
        mask1 = (s2_values_n0 >= args.s2_min) & (s2_values_n0 <= min(gap_low, s2_max))
        mask2 = (s2_values_n0 >= max(gap_high, args.s2_min)) & (s2_values_n0 <= s2_max)
        val = 0.0
        if np.any(mask1):
            val += np.trapezoid(
                gbar_n0[mask1],
                s2_values_n0[mask1],
            )
        if np.any(mask2):
            val += np.trapezoid(
                gbar_n0[mask2],
                s2_values_n0[mask2],
            )
        return val

    # Native n=0 cumulative values on the n=0 grid (same construction as split n=0 code).
    i_n0_native = np.array([integrate_n0(s2) for s2 in s2_values_n0], dtype=float)

    # n=1 target from peak window (same logic as split n=1 code)
    mask_peak = (s2_vals_n1_raw >= args.n1_peak_min) & (s2_vals_n1_raw <= args.n1_peak_max)
    if not np.any(mask_peak):
        raise ValueError("No n=1 data in peak window.")
    gbar_win = gbar_vals_n1_raw[mask_peak]
    pos_mask = gbar_win > 0.0
    neg_mask = gbar_win < 0.0
    if not np.any(pos_mask) or not np.any(neg_mask):
        raise ValueError("Peak window lacks positive/negative peaks.")
    gmax = float(np.max(gbar_win[pos_mask]))
    gmin = float(np.min(gbar_win[neg_mask]))
    delta_n1 = abs(gmax + gmin)
    if delta_n1 == 0.0:
        raise ValueError("Delta is zero for n=1 target.")
    s2_target = abs(args.ct_coeff) / delta_n1
    s2_end = args.n1_integral_end
    if s2_end <= s2_target:
        raise ValueError("--n1-integral-end must be > s2_target.")

    # Reverse-style counterterm window from unshifted Gbar integral on [s2_target, s2_end]
    s2_cond = s2_vals_n1_raw[(s2_vals_n1_raw >= s2_target) & (s2_vals_n1_raw <= s2_end)]
    if s2_cond.size < 2:
        raise ValueError("Not enough n=1 points in [s2_target, s2_end].")
    s2_cond = np.unique(np.round(s2_cond, 12))
    if s2_cond[0] > s2_target:
        s2_cond = np.insert(s2_cond, 0, s2_target)
    if s2_cond[-1] < s2_end:
        s2_cond = np.append(s2_cond, s2_end)
    gbar_cond = np.interp(s2_cond, s2_vals_n1_raw, gbar_vals_n1_raw)
    gbar_int = float(np.trapezoid(gbar_cond, s2_cond))
    ct_coeff_abs = abs(args.ct_coeff)
    ct_coeff_signed = ct_coeff_abs if gbar_int >= 0.0 else -ct_coeff_abs
    k = gbar_int / ct_coeff_signed
    exp_k = float(np.exp(k))
    r2 = 0.0 if exp_k == 1.0 else (exp_k * s2_target - s2_end) / (1.0 - exp_k)
    ct_start = s2_target + r2
    ct_end = s2_end + r2
    ct_end_eff = min(ct_end, args.ct_s2_max)

    # n=1 native grid: log s2_target->s2_end, then coarse to s2_max
    decades_n1 = np.log10(s2_end / s2_target)
    n_log_n1 = max(2, int(np.ceil(decades_n1 * args.s2_log_spaced_points)) + 1)
    s2_n1_log = np.logspace(np.log10(s2_target), np.log10(s2_end), n_log_n1)
    if args.s2_max > s2_end:
        s2_n1_tail = np.arange(s2_end, args.s2_max + 0.5 * args.s2_step_coarse, args.s2_step_coarse)
    else:
        s2_n1_tail = np.array([], dtype=float)
    s2_values_n1 = np.unique(np.round(np.concatenate([s2_n1_log, s2_n1_tail]), 12))
    gbar_n1 = np.interp(s2_values_n1, s2_vals_n1_raw, gbar_vals_n1_raw)
    gbar_n1_cum = cumulative_trapezoid(s2_values_n1, gbar_n1)

    ct_part = np.zeros_like(s2_values_n1)
    if (not args.disable_uv_term) and (ct_end_eff > ct_start) and (ct_start > 0.0):
        decades_ct = np.log10(ct_end_eff / ct_start)
        n_log_ct = max(2, int(np.ceil(decades_ct * args.s2_log_spaced_points)) + 1)
        s2_ct = np.logspace(np.log10(ct_start), np.log10(ct_end_eff), n_log_ct)
        ct_vals = ct_coeff_signed / s2_ct
        ct_cum = cumulative_trapezoid(s2_ct, ct_vals)
        mask_ct = s2_values_n1 >= ct_start
        ct_part[mask_ct] = np.interp(
            np.minimum(s2_values_n1[mask_ct], ct_end_eff), s2_ct, ct_cum
        )

    i_n1_native = gbar_n1_cum - ct_part

    def integrate_n1(s2_max):
        if s2_max <= s2_target:
            return 0.0
        s2_eval_n1 = min(s2_max, s2_values_n1[-1])
        return float(np.interp(s2_eval_n1, s2_values_n1, i_n1_native))

    integrand_sum = gbar_sum - gbar_sum_fv_all
    i_sum_native = cumulative_trapezoid(s2_values_nsum, integrand_sum)

    def integrate_sum(s2_max):
        if s2_max <= s2_values_nsum[0]:
            return 0.0
        s2_eval_sum = min(s2_max, s2_values_nsum[-1])
        return float(np.interp(s2_eval_sum, s2_values_nsum, i_sum_native))

    s2_eval_grid = np.unique(
        np.round(np.concatenate([s2_values_n0, s2_values_n1, s2_values_nsum]), 12)
    )
    uv_cutoff = s2_eval_grid[-1]
    prev_s2 = None
    logd_values = []
    i_total_values = []
    i_n0_values = []
    i_n1_values = []
    i_sum_values = []
    s2_eval = []

    for s2_max in s2_eval_grid:
        i_n0 = integrate_n0(s2_max)
        i_n1 = integrate_n1(s2_max)
        i_sum = integrate_sum(s2_max)
        i_total = i_n0 + i_n1 + i_sum
        if not np.isfinite(i_total):
            if prev_s2 is not None:
                uv_cutoff = prev_s2
            break
        prev_s2 = s2_max
        s2_eval.append(s2_max)
        i_total_values.append(i_total)
        i_n0_values.append(i_n0)
        i_n1_values.append(i_n1)
        i_sum_values.append(i_sum)
        logd_values.append(float(-i_total))

    print(f"[INFO] UV cutoff used: {uv_cutoff:.6f}")
    if logd_values:
        print(f"[INFO] logD at UV cutoff: {logd_values[-1]:.6e}")

    out_name = f"d_integral_piecewise_F{false_index}_T{true_index}.npz"
    np.savez(
        out_name,
        s2_values_n0=s2_values_n0,
        s2_values_n1=s2_values_n1,
        s2_values_nsum=s2_values_nsum,
        gbar_n0=gbar_n0,
        i_n0_native=i_n0_native,
        gbar_n1=gbar_n1,
        i_n1_native=i_n1_native,
        gbar_sum=gbar_sum,
        gbar_sum_fv_all=gbar_sum_fv_all,
        i_sum_native=i_sum_native,
        s2_eval=np.array(s2_eval, dtype=float),
        i_total=np.array(i_total_values, dtype=float),
        i_n0=np.array(i_n0_values, dtype=float),
        i_n1=np.array(i_n1_values, dtype=float),
        i_sum=np.array(i_sum_values, dtype=float),
        logD=np.array(logd_values, dtype=float),
        alpha=args.alpha,
        delta=args.delta,
        uv_cutoff=uv_cutoff,
        s2_min=args.s2_min,
        s2_max=args.s2_max,
        s2_step_coarse=args.s2_step_coarse,
        n_sum_min=args.n_sum_min,
        n_max=args.n_max,
        disable_uv_term=args.disable_uv_term,
        n1_suffix=args.n1_suffix,
        s2_target=s2_target,
        s2_end=s2_end,
        ct_start=ct_start,
        ct_end=ct_end,
        ct_coeff=ct_coeff_signed,
        r2=r2,
    )
    print(f"[SAVE] Wrote D integral data to {out_name}")

    # Optional split outputs for n=0, n=1, and n>=2 contributions.
    n0_name = f"D_int_contribution_n_0_F{false_index}_T{true_index}.npz"
    np.savez(
        n0_name,
        s2_eval=s2_values_n0,
        gbar_n0=gbar_n0,
        i_n0=i_n0_native,
        alpha=args.alpha,
        delta=args.delta,
        s2_min=args.s2_min,
        s2_max=args.s2_max,
        s2_n0_step1_max=args.s2_n0_step1_max,
        s2_n0_step1=args.s2_n0_step1,
        s2_n0_step2=args.s2_n0_step2,
        s2_step_coarse=args.s2_step_coarse,
    )
    print(f"[SAVE] Wrote n=0 contribution to {n0_name}")

    n1_name = f"D_int_contribution_n_1_F{false_index}_T{true_index}.npz"
    np.savez(
        n1_name,
        s2_eval=s2_values_n1,
        gbar_n1=gbar_n1,
        i_n1=i_n1_native,
        s2_target=s2_target,
        s2_end=s2_end,
        ct_coeff=ct_coeff_signed,
        ct_start=ct_start,
        ct_end=ct_end,
        ct_s2_max=args.ct_s2_max,
        r2=r2,
        s2_step_coarse=args.s2_step_coarse,
        s2_log_spaced_points=args.s2_log_spaced_points,
        n1_suffix=args.n1_suffix,
    )
    print(f"[SAVE] Wrote n=1 contribution to {n1_name}")

    nsum_name = f"D_int_contribution_nsum_F{false_index}_T{true_index}.npz"
    np.savez(
        nsum_name,
        s2_eval=s2_values_nsum,
        gbar_sum=gbar_sum,
        gbar_sum_fv_all=gbar_sum_fv_all,
        i_sum=i_sum_native,
        n_sum_min=args.n_sum_min,
        n_max=args.n_max,
        s2_min=args.s2_min,
        s2_max=args.s2_max,
        s2_step_coarse=args.s2_step_coarse,
    )
    print(f"[SAVE] Wrote n>=2 contribution to {nsum_name}")


if __name__ == "__main__":
    main()
