"""
rk_builder_fv_wkb_vfinal4.py -- WKB Stage-A version of rk_builder_fv.py
for the false-vacuum RK Green's function builder.

Replaces every modified-Bessel evaluation with the numerically stable
forms from wkb_bessel_vfinal4.py.

Output filename: rk_green_data_FV_..._wkb_vfinal4.npz

Used by: compute_gbar_fv_wkb_vfinal4.py.
"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import wkb_bessel_vfinal4_git as wkb
from base.green_function_constructor import (
    average_wronskian_plateau,
    diagonal_trace_from_fundamentals,
    weighted_wronskian_profile,
)
from potential_git import CTShiftedLiftedPotential


def build_free_basis_functions_wkb(params, false_vac, s2, n_mode):
    """WKB-stable replacement for build_free_basis_functions."""
    pot_lin = CTShiftedLiftedPotential(params, false_vac)
    phi_prime_false = np.zeros(2, dtype=float)
    m_free = pot_lin.H(phi_prime_false)
    m1_sq_free = m_free[0, 0]
    m2_sq_free = m_free[1, 1]

    nu = n_mode + 1
    r_eps = 1e-8

    def kappa(i):
        m_sq = m1_sq_free if i == 0 else m2_sq_free
        return np.sqrt(s2 + m_sq)

    # This is the FREE / false-vacuum reference operator: the background
    # is the constant false vacuum, so the fluctuation operator is
    # constant-coefficient and its modes are exactly the modified Bessel
    # functions in a diagonal basis -- no radial ODE is integrated here,
    # in contrast to the bounce-side builder which solves the coupled
    # fluctuation ODE along the bounce profile.
    #
    # b(i, r, sign) returns the *true* (unscaled) value
    # B^{+/-}_nu(kappa_i r) / r, with kappa_i set by the constant false-
    # vacuum masses; wkb.iv / wkb.kv are the scipy-backed evaluators.
    def b(i, r, sign):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        if sign == "+":
            return wkb.kv(nu, z) / r_eff
        return wkb.iv(nu, z) / r_eff

    def db(i, r, sign):
        r_eff = max(r, r_eps)
        kap = kappa(i)
        z = kap * r_eff
        if sign == "+":
            bc = wkb.kv(nu, z)
            dbc_dz = wkb.kvp(nu, z)
        else:
            bc = wkb.iv(nu, z)
            dbc_dz = wkb.ivp(nu, z)
        dbc_dr = kap * dbc_dz
        return (dbc_dr * r_eff - bc) / (r_eff * r_eff)

    return b, db, m_free


def build_rk_green_fv_for_bounce_wkb(bounce_npz_filename, s2, n_mode,
                                     n_eval=2000, r0=1e-4,
                                     out_fname=None, overwrite=False):
    """WKB Stage-A FV-side RK Green's function builder.

    Drop-in replacement for build_rk_green_fv_for_bounce, except:
      - Bessel evaluations use wkb_bessel (numerically stable)
      - Output filename gets `_wkb_vfinal4` suffix by default
    """
    data = np.load(bounce_npz_filename, allow_pickle=True)

    params = data["params"]
    false_vac = np.asarray(data["false_vac"], dtype=float)
    r_bounce = data["R"]

    false_index = int(data["false_index"]) if "false_index" in data.files else -1
    true_index = int(data["true_index"]) if "true_index" in data.files else 0

    if out_fname is None:
        out_fname = (f"rk_green_data_FV_F{false_index}_T{true_index}"
                     f"_n{n_mode}_wkb_vfinal4.npz")
    if (not overwrite) and os.path.exists(out_fname):
        print(f"[SKIP] {out_fname} already exists "
              f"(set overwrite=True to rebuild).")
        return out_fname

    b, db, m_free = build_free_basis_functions_wkb(
        params, false_vac, s2, n_mode)

    r_start = max(float(r_bounce[0]), r0)
    r_max = float(r_bounce[-1])
    r_grid = np.linspace(r_start, r_max, n_eval)

    nr = len(r_grid)
    f_plus = np.zeros((nr, 2, 2))
    f_minus = np.zeros((nr, 2, 2))
    df_plus = np.zeros((nr, 2, 2))
    df_minus = np.zeros((nr, 2, 2))

    for k, r in enumerate(r_grid):
        for i in range(2):
            for alpha in range(2):
                delta = 1.0 if i == alpha else 0.0
                f_plus[k, i, alpha] = b(i, r, "+") * delta
                f_minus[k, i, alpha] = b(i, r, "-") * delta
                df_plus[k, i, alpha] = db(i, r, "+") * delta
                df_minus[k, i, alpha] = db(i, r, "-") * delta

    _, w_scaled = weighted_wronskian_profile(
        f_minus,
        df_minus,
        f_plus,
        df_plus,
        r_grid,
        weight_power=3,
    )
    omega, omega_inv, (i_min, i_max) = average_wronskian_plateau(
        w_scaled,
        r_grid,
        r_min_tail=0.05,
        r_max_tail_fraction=0.9,
    )

    print("\n[r^3 Wronskian plateau]")
    print("  r_min_tail =", r_grid[i_min], " r_max_tail =", r_grid[i_max])
    print("  Omega (no symmetrization) =\n", omega)

    # [vx-opt] The downstream consumer only reads the diagonal trace
    # trace(G_rk[k,k]); at k==l the (r_grid[k] >= r_grid[l]) branch is
    # taken, so G_rk[k,k] = f_plus[k] @ omega_inv @ f_minus[k].T.  We
    # compute its trace directly and skip the O(nr^2) dense G_rk build.
    # trace_diag[k] = sum_{i,b,c} f_plus[k,i,b] omega_inv[b,c] f_minus[k,i,c]
    # (VERIFIED bit-identical to the old np.trace(g_rk[k,k]) diagonal).
    trace_diag = diagonal_trace_from_fundamentals(f_plus, f_minus, omega_inv)

    np.savez(
        out_fname,
        r_grid=r_grid,
        trace_diag=trace_diag,
        f_plus=f_plus,
        f_minus=f_minus,
        df_plus=df_plus,
        df_minus=df_minus,
        Omega_inv=omega_inv,
        M_free=m_free,
        W_scaled=w_scaled,
        false_vac=false_vac,
        params=params,
        s2=s2,
        n_mode=n_mode,
        false_index=false_index,
        true_index=true_index,
        wkb_stage="A",
    )
    print(f"[SAVE] Wrote FV RK Green data (WKB Stage A) to {out_fname}")
    return out_fname
