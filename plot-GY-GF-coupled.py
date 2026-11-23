"""Plot coupled-field ``G_F_/v5`` versus ``G_Y_WKB`` results.

The manifest contains matching v5 ``lnD_ren`` and bounce files.  Each entry
also carries the coupling value used on the horizontal axis::

    {"xy2": [{"value": 1.0, "result": "...", "bounce": "..."}]}

The actual determinant evaluation is shared with ``plot-GY-GF.py``; this
wrapper gives the coupled scans their own names and output directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _load_generic_module():
    path = PROJECT_ROOT / "plot-GY-GF.py"
    spec = importlib.util.spec_from_file_location("plot_gy_gf_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _plot(entries, manifest_dir, xlabel, out, l_max=12, wkb_start_l=2):
    shared = _load_generic_module()
    values, gf, gy = shared._scan(
        entries, Path(manifest_dir), l_max, wkb_start_l
    )
    return shared._plot_scan(values, gf, gy, xlabel, out)


def plot_varying_xy2(entries, manifest_dir=".", l_max=12,
                     wkb_start_l=2, out=None):
    """Plot the comparison while varying the coupled ``x*y^2`` coupling."""
    return _plot(entries, manifest_dir, r"coupling $x y^2$", out,
                 l_max, wkb_start_l)


def plot_varying_x2y2(entries, manifest_dir=".", l_max=12,
                      wkb_start_l=2, out=None):
    """Plot the comparison while varying the coupled ``x^2*y^2`` coupling."""
    return _plot(entries, manifest_dir, r"coupling $x^2 y^2$", out,
                 l_max, wkb_start_l)


def plot_varying_mass(entries, manifest_dir=".", l_max=12,
                      wkb_start_l=2, out=None):
    """Plot the comparison while varying a quadratic mass coefficient."""
    return _plot(entries, manifest_dir, r"quadratic coupling $m^2$", out,
                 l_max, wkb_start_l)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scan", choices=("xy2", "x2y2", "mass"),
                        default="xy2")
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
    args.out.parent.mkdir(parents=True, exist_ok=True)

    function = {
        "xy2": plot_varying_xy2,
        "x2y2": plot_varying_x2y2,
        "mass": plot_varying_mass,
    }[args.scan]
    function(
        manifest[args.scan],
        manifest_dir=args.manifest.parent,
        l_max=args.l_max,
        wkb_start_l=args.wkb_start_l,
        out=args.out,
    )


if __name__ == "__main__":
    main()
