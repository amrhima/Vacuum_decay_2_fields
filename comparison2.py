"""Compare the decoupled G_Y_WKB and v5 one-loop determinants.

The v5 production pipeline writes ``lnD_ren`` in the determinant convention

    lnD_ren = ln |det' M_b / det M_FV|_ren,

where the four translational zero modes are omitted.  ``G_Y_WKB`` returns the
corresponding one-loop action, so the directly comparable quantity is

    DeltaS_GY = 1/2 * lnD_ren.

This script deliberately does not apply BubbleDet's zero-mode Jacobian.  The
v5 decay-rate formula supplies that prefactor separately, just as the raw
G_Y_WKB decay-rate formula does.

The v5 result and bounce files must belong to the same run.  The comparison is
currently restricted to a decoupled two-field background, because the WKB tail
used by G_Y_WKB is scalar/diagonal.

Example
-------
    python comparison2.py \
        --v5-result results/lnD_ren_<tag>.npz \
        --v5-bounce results/bounce_data_<tag>.npz \
        --out-dir comparison2_output --n-max 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
V5_COMPUTATION_DIR = PROJECT_ROOT / "G_F_" / "v5" / "computation"
if str(V5_COMPUTATION_DIR) not in sys.path:
    sys.path.insert(0, str(V5_COMPUTATION_DIR))

from potential_vgeneral_phi_v2 import PrimedPotential  # noqa: E402
from G_Y_WKB.one_loop_renormalised import (  # noqa: E402
    dm_renormalized_one_loop_action,
    zero_mode_jacobian,
)
from G_Y_WKB.WKB_terms import build_W_profile  # noqa: E402


def _scalar(npz, key):
    """Read a scalar npz entry with a useful error message."""
    if key not in npz.files:
        raise KeyError(f"{npz.filename} does not contain required key {key!r}")
    return float(np.asarray(npz[key]).reshape(()))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_v5_inputs(result_path: Path, bounce_path: Path):
    result = np.load(result_path, allow_pickle=True)
    bounce = np.load(bounce_path, allow_pickle=True)

    required_bounce = ("R", "phi_b", "params", "false_vac", "fields", "S_E")
    missing = [key for key in required_bounce if key not in bounce.files]
    if missing:
        raise RuntimeError(
            f"v5 bounce file is missing required keys: {', '.join(missing)}"
        )
    if "lnD_ren" not in result.files or "mu" not in result.files:
        raise RuntimeError("v5 result must contain lnD_ren and mu")

    if "bounce_sha" in result.files and "bounce_sha" in bounce.files:
        result_sha = str(np.asarray(result["bounce_sha"]).reshape(()))
        bounce_sha = str(np.asarray(bounce["bounce_sha"]).reshape(()))
        if result_sha != bounce_sha:
            raise RuntimeError(
                "v5 result and bounce have different bounce_sha values; "
                "refusing to compare different backgrounds."
            )

    R = np.asarray(bounce["R"], dtype=float)
    phi_b = np.asarray(bounce["phi_b"], dtype=float)
    params = np.asarray(bounce["params"], dtype=float)
    false_vac = np.asarray(bounce["false_vac"], dtype=float)
    fields = tuple(str(x) for x in np.asarray(bounce["fields"]).ravel())

    if len(fields) != 2 or phi_b.ndim != 2 or phi_b.shape[1] != 2:
        raise RuntimeError(
            "comparison2.py currently requires a two-field v5 bounce."
        )
    if R.ndim != 1 or len(R) != len(phi_b) or np.any(np.diff(R) <= 0.0):
        raise RuntimeError("v5 R and phi_b do not define a valid radial grid.")

    pot = PrimedPotential(params, false_vac, fields)
    W_profile, H_fv = build_W_profile(
        R_bounce=R,
        X_bounce=phi_b[:, 0],
        Y_bounce=phi_b[:, 1],
        phi_fv=np.zeros(2),
        pot=pot,
    )
    offdiag = W_profile.copy()
    offdiag[:, 0, 0] = 0.0
    offdiag[:, 1, 1] = 0.0
    scale = max(float(np.max(np.abs(W_profile))), 1.0)
    if float(np.max(np.abs(offdiag))) > 1e-10 * scale:
        raise RuntimeError(
            "the supplied v5 background is coupled in the mass basis; "
            "comparison2.py currently supports only the decoupled case."
        )

    return result, bounce, R, phi_b, pot, H_fv


def run_comparison(result_path: Path, bounce_path: Path, n_max: int,
                   wkb_start_n: int):
    result, bounce, R, phi_b, pot, H_fv = load_v5_inputs(
        result_path, bounce_path
    )
    mu = _scalar(result, "mu")
    S_E = _scalar(bounce, "S_E")

    delta_s_gy = float(dm_renormalized_one_loop_action(
        R_bounce=R,
        X_bounce=phi_b[:, 0],
        Y_bounce=phi_b[:, 1],
        phi_fv=np.zeros(2),
        pot=pot,
        r_min=max(float(R[0]), 1e-4),
        Rmax=float(R[-1]),
        n_max=int(n_max),
        mu_ren=mu,
        wkb_start_n=int(wkb_start_n),
    ))

    lnd_v5 = _scalar(result, "lnD_ren")
    action_v5 = 0.5 * lnd_v5
    J_zero = zero_mode_jacobian(S_E)

    return {
        "v5_result": str(result_path),
        "v5_bounce": str(bounce_path),
        "mu": mu,
        "S_E": S_E,
        "n_max_gy": int(n_max),
        "wkb_start_n": int(wkb_start_n),
        "v5_lnD_ren": lnd_v5,
        "v5_one_loop_action": action_v5,
        "G_Y_WKB_delta_S_reg": delta_s_gy,
        "difference_GY_minus_v5_action": delta_s_gy - action_v5,
        "relative_difference_to_v5_action": (
            (delta_s_gy - action_v5) / max(abs(action_v5), 1.0)
        ),
        "zero_mode_jacobian": J_zero,
        # One-loop prefactors, excluding the common exp(-S_E) and the
        # negative-mode factor 1/2.  Here J_zero=(S_E/2pi)^2 is the full
        # four-translation Jacobian in both conventions.
        "G_Y_WKB_one_loop_prefactor": float(
            J_zero * np.exp(-delta_s_gy)
        ),
        "v5_one_loop_prefactor": float(
            J_zero * np.exp(-action_v5)
        ),
        "note": (
            "Comparison is in the raw determinant convention: "
            "G_Y_WKB delta_S_reg = 1/2 v5 lnD_ren."
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare G_Y_WKB with a v5 MS-bar lnD_ren result."
    )
    parser.add_argument("--v5-result", type=Path, required=True)
    parser.add_argument("--v5-bounce", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("comparison2_output"))
    parser.add_argument("--n-max", type=int, default=40)
    parser.add_argument("--wkb-start-n", type=int, default=2)
    args = parser.parse_args()

    if args.n_max < 1:
        parser.error("--n-max must be positive")
    if args.wkb_start_n < 2 or args.wkb_start_n > args.n_max + 1:
        parser.error("--wkb-start-n must satisfy 2 <= wkb-start-n <= n-max+1")

    data = run_comparison(
        args.v5_result,
        args.v5_bounce,
        args.n_max,
        args.wkb_start_n,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "comparison2_results.json"
    with output_path.open("w") as stream:
        json.dump(_jsonable(data), stream, indent=2)
        stream.write("\n")

    print(f"[OK] wrote {output_path}")
    print(f"  mu = {data['mu']:.12g}")
    print(f"  v5  1/2 lnD_ren = {data['v5_one_loop_action']:+.12e}")
    print(f"  GY  DeltaS      = {data['G_Y_WKB_delta_S_reg']:+.12e}")
    print(f"  difference      = {data['difference_GY_minus_v5_action']:+.12e}")


if __name__ == "__main__":
    main()
