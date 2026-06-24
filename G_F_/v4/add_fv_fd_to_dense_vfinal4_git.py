#!/usr/bin/env python3
"""
add_fv_fd_to_dense_vfinal4.py -- post-hoc patch for option A (FD-FV consistency).

NOTE: this script MUTATES the input dense FD npz IN PLACE.  It reloads each
existing dense FD output, computes two new FV-side fields, and re-saves the
file at the same path with all old fields preserved plus the new ones added.

For each existing dense FD output
    gbar_n0_fd_ver2_<TAG>_wkb_vfinal4.npz  (and the n=1 analogue)
this script:

  1. Reloads the bounce data and rebuilds the FV-side operator
     M_tilde_FV using the SAME finite-difference discretization as the
     bounce-side operator (build_M_tilde_clean with X_prime, Y_prime
     identically zero -- those are the false-vacuum profile in the
     potential's "primed" coordinates).
  2. For every s^2 already present in the existing file's s2_grid,
     runs the same Hutchinson trace estimator on M_tilde_FV instead
     of on M_tilde_bounce, and stores the result as a new field
         gbar_n0_FV_fd       (or gbar_n1_FV_fd)
         gbar_n0_FV_fd_err   (or gbar_n1_FV_fd_err)
     in the npz output.

The downstream finalizer / sweep can then subtract bare_FD - FV_FD
in the same scheme for n=0,1, eliminating the FD-vs-RK boundary-
discretization mismatch that produced the slow tail in delta_n
diagnosed on 2026-05-08.

USAGE:
    python3 add_fv_fd_to_dense_vfinal4.py
        --bounce bounce_data_F2_T0.npz
        --data-dir /path/to/your/G_project_data
        --K 100000 --N 2000 --r-min 1e-4 --seed 12345
"""

import argparse
import contextlib
import io
import os
import sys
import time

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from config import DATA_DIR
from fd_builder_n0_git import (
    build_M_tilde_clean as build_M_tilde_clean_n0,
    gbar_raw_fd as gbar_raw_fd_n0,
    gbar_sub_fd as gbar_sub_fd_n0,
    load_bounce as load_bounce_n0,
)
from fd_builder_n1_git import (
    build_M_tilde_clean as build_M_tilde_clean_n1,
    gbar_raw_fd as gbar_raw_fd_n1,
    gbar_sub_fd as gbar_sub_fd_n1,
)
from potential_git import CTShiftedLiftedPotential


def patch_one(n, dense_path, bounce_path, data_dir, K, N, r_min, r_max,
              seed, overwrite):
    """Compute and append gbar_n{n}_FV_fd to the dense FD npz at dense_path."""
    if not os.path.isfile(dense_path):
        # [vfinal4 C4] a MISSING dense FD target is a hard error, not a skip.
        # Returning success here lets the run (under set -e) march on and only
        # fail much later in the finalizer's FD-Born guard; abort immediately.
        sys.exit(f"[ABORT C4] dense FD target not found: {dense_path}  "
                 f"(run the FD n={n} stage first).")

    print(f"\n=== Patching n={n} dense FD file ===")
    print(f"  file: {dense_path}")
    d = np.load(dense_path, allow_pickle=True)
    fv_field = f"gbar_n{n}_FV_fd"
    fv_err_field = f"gbar_n{n}_FV_fd_err"
    if (fv_field in d.files) and not overwrite:
        print(f"  [SKIP] {fv_field} already present "
              f"(use --overwrite to recompute)")
        return

    s2_grid = np.asarray(d["s2_grid"], dtype=float)
    print(f"  s2_grid: {len(s2_grid)} pts in "
          f"[{s2_grid.min():.4f}, {s2_grid.max():.4f}]")

    # rebuild bounce data and FV-side operator
    b = load_bounce_n0(bounce_path)
    with contextlib.redirect_stdout(io.StringIO()):
        pot_lin = CTShiftedLiftedPotential(b["params"], b["false_vac"])

    # FV-side: same discretization, but X_prime = Y_prime = 0 everywhere
    # (false vacuum is at the origin in the "primed" coordinates).
    R_bounce = np.asarray(b["R"], dtype=float)
    zeros_R = np.zeros_like(R_bounce)

    builder = build_M_tilde_clean_n0 if n == 0 else build_M_tilde_clean_n1
    sub_fn = gbar_sub_fd_n0 if n == 0 else gbar_sub_fd_n1

    print(f"  building M_tilde_FV (n={n}) on N={N} pts ...")
    t0 = time.time()
    M_tilde_FV, r, dr, _ = builder(
        n=n,
        R_bounce=R_bounce,
        X_prime_bounce=zeros_R,
        Y_prime_bounce=zeros_R,
        pot_lin=pot_lin,
        N=N, r_min=r_min, r_max=r_max, fd_order=2,
    )
    N2 = M_tilde_FV.shape[0]
    boundary_indices = np.array([0, N - 1, N, 2 * N - 1])
    print(f"  M_tilde_FV: shape {M_tilde_FV.shape}, dr={dr:.6f}, "
          f"built in {time.time() - t0:.1f}s")

    # FV-PROJECTION FIX (2026-05-08).  The bare-side trace gbar_n{0,1}_sub
    # projects out the negative mode (n=0, chi_neg) or zero mode
    # (n=1, chi_zm) -- this gives gbar_n{0,1}_sub a -1/(s^2+lambda)
    # tail at large s^2.  The corresponding FV trace MUST carry the
    # SAME projection (using the bare's chi vector) for the slow tail
    # to cancel pairwise in delta_n = bare - FV - g1 - g2.  This is
    # an ALGEBRAIC bookkeeping projection, not a physical mode removal
    # (chi_neg is not an FV eigenvector): we reuse the bare side's projection
    # vector solely so the matching -1/(s^2+lambda) tails cancel term-by-term.
    chi_field = "chi_neg" if n == 0 else "chi_zm"
    chi_proj = np.asarray(d[chi_field], dtype=float)
    chi_proj = chi_proj / np.linalg.norm(chi_proj)
    chi_proj[boundary_indices] = 0.0
    chi_proj = chi_proj / np.linalg.norm(chi_proj)
    print(f"  using chi_{'neg' if n==0 else 'zm'} from bare-side npz "
          f"(||chi||={np.linalg.norm(chi_proj):.6f}) for FV projection")

    # The bare-side compute_gbar_n{0,1}_fd.py applies the
    # (n+1)^2 partial-wave degeneracy weight to the Hutchinson trace
    # (compute_gbar_n0_fd.py: implicit, weight = 1; compute_gbar_n1_fd.py:
    # explicit `degeneracy = (1+1)**2`, lines 247, 269-273).  We must
    # apply the SAME weight on the FV side or the bare-FD - FV-FD
    # subtraction will mis-cancel by a factor of (n+1)^2.
    degeneracy = (n + 1) ** 2

    # Hutchinson scan on FV-side operator with projection by bare's chi.
    print(f"\n  scanning s^2 (K={K}) on FV-side WITH projection  "
          f"((n+1)^2 weight = {degeneracy}) ...")
    rng = np.random.default_rng(seed)
    n_s2 = len(s2_grid)
    fv_val = np.full(n_s2, np.nan)
    fv_err = np.full(n_s2, np.nan)
    t_scan = time.time()
    for i, s2 in enumerate(s2_grid):
        t = time.time()
        v, e = sub_fn(M_tilde_FV, s2, dr, N2, chi_proj, K=K, rng=rng,
                      boundary_indices=boundary_indices)
        if v is not None and np.isfinite(v):
            v *= degeneracy
            e *= degeneracy
        fv_val[i] = v
        fv_err[i] = e
        if (i + 1) % 50 == 0 or i == n_s2 - 1:
            print(f"    [{i+1:4d}/{n_s2}]  s2={s2:.4f}  "
                  f"FV_fd_proj={v:+.4e}  ({time.time()-t:.1f}s)")
    print(f"  scan finished in {time.time() - t_scan:.0f}s")

    # write the patched npz: keep all old fields, add the two new ones.
    out = {k: d[k] for k in d.files}
    out[fv_field] = fv_val
    out[fv_err_field] = fv_err
    out["fv_fd_K"] = np.array(K, dtype=int)
    out["fv_fd_seed"] = np.array(seed, dtype=int)
    np.savez(dense_path, **out)
    print(f"  [SAVE] wrote {fv_field}, {fv_err_field} into "
          f"{os.path.basename(dense_path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounce", default="bounce_data_F2_T0.npz")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--tag", default="F2_T0")
    parser.add_argument("--K", type=int, default=100000)  # vfinal4: 100000;
    # the FD FV-subtraction Hutchinson noise scales as 1/sqrt(K) and (n+1)^2-weighted
    # FV dominates the n=0,1 error -> K=100000 cuts it ~16x vs the old K=400 default.
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--r-min", type=float, default=1e-4)
    parser.add_argument("--r-max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only-n", type=int, choices=(0, 1), default=None,
                        help="If set, patch only n=0 or only n=1.")
    args = parser.parse_args()

    data_dir = args.data_dir or DATA_DIR
    bounce_path = args.bounce
    if not os.path.isabs(bounce_path):
        for d in [_here, os.path.join(_here, "data_files"), data_dir]:
            alt = os.path.join(d, os.path.basename(bounce_path))
            if os.path.isfile(alt):
                bounce_path = alt
                break

    print(f"data_dir   = {data_dir}")
    print(f"bounce     = {bounce_path}")
    print(f"K          = {args.K}")
    print(f"N          = {args.N}")
    print(f"seed       = {args.seed}")

    targets = [(0, f"gbar_n0_fd_ver2_{args.tag}_wkb_vfinal4.npz"),
               (1, f"gbar_n1_fd_ver2_{args.tag}_wkb_vfinal4.npz")]
    if args.only_n is not None:
        targets = [t for t in targets if t[0] == args.only_n]

    for n, basename in targets:
        path = os.path.join(data_dir, basename)
        patch_one(n, path, bounce_path, data_dir,
                  args.K, args.N, args.r_min, args.r_max,
                  args.seed, args.overwrite)


if __name__ == "__main__":
    main()
