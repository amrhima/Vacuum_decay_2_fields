#!/usr/bin/env python3
"""
Build RK Green's function in the false-vacuum background using free mode functions.
This uses only the free Hessian at the false vacuum and does not solve for h.
"""

import argparse
import os
import numpy as np
from scipy.special import iv, ivp, kv, kvp

from potential import CTShiftedLiftedPotential
from config import DATA_DIR


def build_free_basis_functions(params, false_vac, s2, n_mode):
    pot_lin = CTShiftedLiftedPotential(params, false_vac)
    phi_prime_false = np.zeros(2, dtype=float)
    m_free = pot_lin.H(phi_prime_false)
    m1_sq_free = m_free[0, 0]
    m2_sq_free = m_free[1, 1]

    ell_bessel = n_mode + 1
    r_eps = 1e-8

    def kappa(i):
        m_sq = m1_sq_free if i == 0 else m2_sq_free
        return np.sqrt(s2 + m_sq)

    def bcore_plus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return kv(ell_bessel, z)

    def dbcore_plus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return kvp(ell_bessel, z)

    def bcore_minus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return iv(ell_bessel, z)

    def dbcore_minus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return ivp(ell_bessel, z)

    def b(i, r, sign):
        r_eff = max(r, r_eps)
        if sign == "+":
            return bcore_plus(i, r) / r_eff
        return bcore_minus(i, r) / r_eff

    def db(i, r, sign):
        r_eff = max(r, r_eps)
        k = kappa(i)
        if sign == "+":
            bc = bcore_plus(i, r)
            dbc_dz = dbcore_plus(i, r)
        else:
            bc = bcore_minus(i, r)
            dbc_dz = dbcore_minus(i, r)
        dbc_dr = k * dbc_dz
        return (dbc_dr * r_eff - bc) / (r_eff ** 2)

    return b, db, m_free


def build_rk_green_fv_for_bounce(bounce_npz_filename, s2, n_mode,
                                 n_eval=2000, r0=1e-4,
                                 out_fname=None, overwrite=False):
    data = np.load(bounce_npz_filename, allow_pickle=True)

    params = data["params"]
    false_vac = np.asarray(data["false_vac"], dtype=float)
    r_bounce = data["R"]

    false_index = int(data["false_index"]) if "false_index" in data.files else -1
    true_index = int(data["true_index"]) if "true_index" in data.files else 0

    if out_fname is None:
        out_fname = f"rk_green_data_FV_F{false_index}_T{true_index}_n{n_mode}.npz"
    if (not overwrite) and os.path.exists(out_fname):
        print(f"[SKIP] {out_fname} already exists (set overwrite=True to rebuild).")
        return out_fname

    b, db, m_free = build_free_basis_functions(params, false_vac, s2, n_mode)

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

    w_raw = np.zeros((nr, 2, 2))
    for idx in range(nr):
        w = np.zeros((2, 2))
        for alpha in range(2):
            for beta in range(2):
                s = 0.0
                for i in range(2):
                    fm_a = f_minus[idx, i, alpha]
                    fp_b = f_plus[idx, i, beta]
                    dfm_a = df_minus[idx, i, alpha]
                    dfp_b = df_plus[idx, i, beta]
                    s += fm_a * dfp_b - fp_b * dfm_a
                w[alpha, beta] = -s
        w_raw[idx] = w

    w_scaled = np.zeros_like(w_raw)
    for idx, r in enumerate(r_grid):
        w_scaled[idx] = (r ** 3) * w_raw[idx]

    r_min_tail = 0.05
    r_max_tail = 0.9 * r_grid[-1]
    i_min = np.searchsorted(r_grid, r_min_tail)
    i_max = np.searchsorted(r_grid, r_max_tail)
    w_tail = w_scaled[i_min:i_max + 1, :, :]
    omega = np.mean(w_tail, axis=0)
    omega_inv = np.linalg.inv(omega)

    g_rk = np.zeros((nr, nr, 2, 2))
    for k in range(nr):
        f_dec_r = f_plus[k]
        f_reg_r = f_minus[k]
        for l in range(nr):
            f_dec_rp = f_plus[l]
            f_reg_rp = f_minus[l]
            if r_grid[k] >= r_grid[l]:
                f_plus_use = f_dec_r
                f_minus_use = f_reg_rp
            else:
                f_plus_use = f_dec_rp
                f_minus_use = f_reg_r
            g_rk[k, l] = f_plus_use @ omega_inv @ f_minus_use.T

    np.savez(
        out_fname,
        r_grid=r_grid,
        G_rk=g_rk,
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
    )
    print(f"[SAVE] Wrote FV RK Green data to {out_fname}")
    return out_fname


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FV RK Green's function for one (n, s2)."
    )
    parser.add_argument(
        "--bounce",
        default="bounce_data_F2_T0.npz",
        help="Bounce file, e.g. bounce_data_F2_T0.npz",
    )
    parser.add_argument(
        "--s2",
        type=float,
        default=0.0,
        help="Spectral parameter s2.",
    )
    parser.add_argument(
        "--n-mode",
        type=int,
        default=0,
        help="Partial wave index n.",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=2000,
        help="Number of r samples.",
    )
    parser.add_argument(
        "--r0",
        type=float,
        default=1e-4,
        help="Small-r cutoff.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    bounce_path = args.bounce
    if not os.path.isfile(bounce_path):
        alt_path = os.path.join(script_dir, args.bounce)
        if os.path.isfile(alt_path):
            bounce_path = alt_path
        else:
            raise FileNotFoundError(f"Bounce file not found: {args.bounce}")

    output_dir = DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    print(f"[INFO] Writing outputs to {output_dir}")

    build_rk_green_fv_for_bounce(
        bounce_path,
        s2=args.s2,
        n_mode=args.n_mode,
        n_eval=args.n_eval,
        r0=args.r0,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
