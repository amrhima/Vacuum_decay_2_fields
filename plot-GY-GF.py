"""Plot GY_WKB versus the ``G_F_``/v5 determinant for two-field backgrounds.

The G_F_ production pipeline writes one ``lnD_ren`` result file and one
matching bounce file per parameter point.  This script reads those pairs,
recomputes GY_WKB on the same bounce, and makes the same three plots as
``plot-GY.py``: scans versus a three-point coupling, a four-point coupling,
and a mass parameter.

Manifest format
---------------
The command-line interface takes a JSON manifest containing three lists:

    {
      "cubic":  [{"value": 2.5, "result": "...npz", "bounce": "...npz"}],
      "quartic": [{"value": 1.0, "result": "...npz", "bounce": "...npz"}],
      "mass":    [{"value": 0.5, "result": "...npz", "bounce": "...npz"}]
    }

Paths are resolved relative to the manifest.  The result file must contain
``lnD_ren`` and ``mu``; the bounce file must be one of the matching v5 bounce
files accepted by ``comparison2.load_v5_inputs``.

Examples
--------
    python plot-GY-GF.py --manifest gf_scans.json --scan cubic \
        --out plots/one_loop_vs_cubic_gf.png

The G_F_ value plotted is ``0.5 * lnD_ren``.  GY_WKB is plotted as its raw
renormalised determinant action ``delta_S_reg`` so both curves use the same
zero-mode convention as the v5 comparison pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
V5_COMPUTATION_DIR = PROJECT_ROOT / "G_F_" / "v5" / "computation"
if str(V5_COMPUTATION_DIR) not in sys.path:
    sys.path.insert(0, str(V5_COMPUTATION_DIR))

from G_Y_WKB.one_loop_renormalised import dm_renormalized_one_loop_action
from comparison2 import load_v5_inputs


def _load_plot_gy_module():
    """Load the shared parameterized-bounce helpers from plot-GY.py."""
    path = PROJECT_ROOT / "plot-GY.py"
    name = "project_plot_gy_helpers"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_point(result_path, bounce_path, l_max=12, wkb_start_l=2):
    """Compare one matching G_F_ result/bounce pair with GY_WKB.

    Parameters may be strings or ``pathlib.Path`` objects.  The returned
    actions are in the raw determinant convention:

        G_F_:     1/2 * lnD_ren
        GY_WKB:   delta_S_reg
    """
    result_path = Path(result_path)
    bounce_path = Path(bounce_path)
    result, bounce, R, phi_b, pot, _ = load_v5_inputs(
        result_path, bounce_path
    )
    lnD_ren = float(np.asarray(result["lnD_ren"]).reshape(()))
    mu = float(np.asarray(result["mu"]).reshape(()))

    delta_s_gy = dm_renormalized_one_loop_action(
        R_bounce=R,
        X_bounce=phi_b[:, 0],
        Y_bounce=phi_b[:, 1],
        phi_fv=np.zeros(2),
        pot=pot,
        r_min=max(float(R[0]), 1e-4),
        Rmax=float(R[-1]),
        l_max=int(l_max),
        mu_ren=mu,
        wkb_start_l=int(wkb_start_l),
    )
    return {
        "gf": 0.5 * lnD_ren,
        "gy_wkb": float(delta_s_gy),
        "difference": float(delta_s_gy - 0.5 * lnD_ren),
    }


def _resolve_entry(entry, base_dir):
    required = ("value", "result", "bounce")
    missing = [key for key in required if key not in entry]
    if missing:
        raise ValueError(f"Manifest entry is missing {missing}")
    result = Path(entry["result"])
    bounce = Path(entry["bounce"])
    if not result.is_absolute():
        result = base_dir / result
    if not bounce.is_absolute():
        bounce = base_dir / bounce
    return float(entry["value"]), result, bounce


def _scan(entries, base_dir, l_max, wkb_start_l):
    values = []
    gf = []
    gy = []
    for i, entry in enumerate(entries, start=1):
        value, result_path, bounce_path = _resolve_entry(entry, base_dir)
        comparison = evaluate_point(
            result_path, bounce_path, l_max=l_max, wkb_start_l=wkb_start_l
        )
        values.append(value)
        gf.append(comparison["gf"])
        gy.append(comparison["gy_wkb"])
        print(
            f"{i}/{len(entries)}: parameter={value:g}, "
            f"G_F_={gf[-1]:+.8e}, GY_WKB={gy[-1]:+.8e}"
        )
    return np.asarray(values), np.asarray(gf), np.asarray(gy)


def _plot_scan(values, gf, gy, xlabel, out=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(values, gf, "o-", label="G_F_")
    ax.plot(values, gy, "s--", label="GY_WKB")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("one-loop correction to effective action")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=180)
    return fig, ax


def plot_varying_three_point(
    entries, manifest_dir=".", l_max=12, wkb_start_l=2, out=None
):
    """Plot the comparison versus the three-point/cubic parameter."""
    values, gf, gy = _scan(entries, Path(manifest_dir), l_max, wkb_start_l)
    return _plot_scan(values, gf, gy, r"three-point coupling $g$", out)


def plot_varying_four_point(
    entries, manifest_dir=".", l_max=12, wkb_start_l=2, out=None
):
    """Plot the comparison versus the four-point/quartic parameter."""
    values, gf, gy = _scan(entries, Path(manifest_dir), l_max, wkb_start_l)
    return _plot_scan(values, gf, gy, r"four-point coupling $\lambda$", out)


def plot_varying_mass(
    entries, manifest_dir=".", l_max=12, wkb_start_l=2, out=None
):
    """Plot the comparison versus the mass parameter."""
    values, gf, gy = _scan(entries, Path(manifest_dir), l_max, wkb_start_l)
    return _plot_scan(values, gf, gy, r"mass parameter $m^2$", out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scan", choices=("cubic", "quartic", "mass"),
                        default="cubic")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--l-max", type=int, default=12)
    parser.add_argument("--wkb-start-l", type=int, default=2)
    args = parser.parse_args()

    if args.l_max < 1:
        parser.error("--l-max must be positive")
    if args.wkb_start_l < 2 or args.wkb_start_l > args.l_max + 1:
        parser.error("--wkb-start-l must satisfy 2 <= start <= l-max+1")

    with args.manifest.open() as stream:
        manifest = json.load(stream)
    if args.scan not in manifest:
        parser.error(f"manifest has no {args.scan!r} scan")

    kwargs = dict(
        entries=manifest[args.scan],
        manifest_dir=args.manifest.parent,
        l_max=args.l_max,
        wkb_start_l=args.wkb_start_l,
        out=args.out,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.scan == "cubic":
        plot_varying_three_point(**kwargs)
    elif args.scan == "quartic":
        plot_varying_four_point(**kwargs)
    else:
        plot_varying_mass(**kwargs)


if __name__ == "__main__":
    main()
