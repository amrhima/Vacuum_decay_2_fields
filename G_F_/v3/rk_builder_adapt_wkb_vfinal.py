"""
rk_builder_adapt_wkb_vfinal.py -- bounce-side coupled-channel
RK ODE builder.

NOTE (pipeline_fd_rk_wkb_vfinal / scipy-only variant):
In this v3 pipeline, wkb_bessel_vfinal.py is a scipy-only stub --
iv/kv/ivp/kvp delegate directly to scipy.special and pick_regime
always returns 'scipy'.  No Hankel/Olver regime dispatch is active.
The original docstring described the v2 pipeline's three-branch
dispatch; that text is preserved below for historical context but
ALL Bessel evaluations in this file now go through scipy.

ORIGINAL (v2 pipeline):
Replaces every modified-Bessel evaluation (I_nu, K_nu, I'_nu, K'_nu)
with the regime-aware wrappers in wkb_bessel_vfinal.py:
  - bare scipy.special.iv/kv at moderate (nu, z),
  - Hankel asymptotic series for nu < 10 and z > max(30, 5*nu),
  - Olver uniform asymptotic for nu >= 10.
All three branches return the *true* (unscaled) function value.
The bare-scipy branch matches the existing non-WKB cache exactly,
so the new code path agrees with the cache wherever the cache is
in the bare-scipy regime.

Output: rk_green_data_n{N}_s2{S2}_wkb_vfinal.npz (same shape as the
non-WKB version, just tagged for tracking).

Used by: compute_gbar_npos_wkb_vfinal.py.
"""

import os
import numpy as np
from scipy.integrate import solve_ivp

import wkb_bessel_vfinal as wkb
from potential import CTShiftedLiftedPotential


def build_fluctuation_data_prime_wkb(pot_lin,
                                     R_bounce,
                                     X_prime_bounce,
                                     Y_prime_bounce,
                                     s2,
                                     n_mode):
    """WKB-stable replacement for build_fluctuation_data_prime.

    Same signature, same return tuple
    (B, dB, K_matrix, A_i, U_int_pp, M_free, X', Y') as the original,
    but every Bessel evaluation goes through wkb_bessel.
    """

    X_prime_bounce = np.asarray(X_prime_bounce, dtype=float)
    Y_prime_bounce = np.asarray(Y_prime_bounce, dtype=float)
    R_bounce = np.asarray(R_bounce, dtype=float)

    def x_prime_of_r(r):
        return np.interp(r, R_bounce, X_prime_bounce)

    def y_prime_of_r(r):
        return np.interp(r, R_bounce, Y_prime_bounce)

    def H_prime(phi_prime):
        return pot_lin.H(phi_prime)

    phi_prime_false = np.zeros(2, dtype=float)
    M_free = H_prime(phi_prime_false)

    m1_sq_free = M_free[0, 0]
    m2_sq_free = M_free[1, 1]

    print("\n[build_fluctuation_data_prime_wkb]")
    print("  M_free = H'(0) =\n", M_free)
    print(f"  m1_free^2 = {m1_sq_free:.6f}, m2_free^2 = {m2_sq_free:.6f}")
    print(f"  s2 = {s2}, n_mode = {n_mode}")
    print(f"  WKB Bessel routines from wkb_bessel_vfinal.py (Stage A)")

    nu = n_mode + 1   # the user's notation: nu = n+1 in 4D
    r_eps = 1e-8

    def kappa(i):
        m_sq = m1_sq_free if i == 0 else m2_sq_free
        return np.sqrt(s2 + m_sq)

    # Regime-aware WKB:  B(i, r, sign) returns the *true* (unscaled)
    # value B^{+/-}_nu(kappa_i r) / r.  Internally `wkb.iv` / `wkb.kv`
    # dispatch on (nu, z = kappa_i r):
    #     - bare scipy iv/kv at moderate (nu, z),
    #     - Hankel asymptotic series for nu < 10 and z > max(30, 5*nu),
    #     - Olver uniform asymptotic for nu >= 10.
    # The bare-scipy branch is the same evaluator the existing
    # non-WKB cache used; the asymptotic branches take over only
    # where bare scipy starts losing precision or overflows.
    def B(i, r, sign):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        if sign == "+":
            return wkb.kv(nu, z) / r_eff
        return wkb.iv(nu, z) / r_eff

    def dB(i, r, sign):
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

    def H_full_prime_r(r):
        xp = x_prime_of_r(r)
        yp = y_prime_of_r(r)
        phi_prime = np.array([xp, yp], dtype=float)
        return H_prime(phi_prime)

    def U_int_pp(r):
        return H_full_prime_r(r) - M_free

    # ===== K_matrix and A_i, regime-aware (bare scipy / Hankel / Olver) =====
    def K_matrix(r, sign):
        """K_ij(r) = U_int''_ij(r) * B_j(kappa_j r) / B_i(kappa_i r).

        Uses regime-aware `wkb.iv` / `wkb.kv` -- the values are TRUE
        (unscaled) and the ratio is taken directly.  No 'scaled' form
        anywhere in the call chain.
        """
        r_eff = max(r, r_eps)
        U = U_int_pp(r)
        K = np.zeros((2, 2))
        z = [kappa(0) * r_eff, kappa(1) * r_eff]
        if sign == "+":
            B = [wkb.kv(nu, z[0]), wkb.kv(nu, z[1])]
        else:
            B = [wkb.iv(nu, z[0]), wkb.iv(nu, z[1])]
        for i in range(2):
            for j in range(2):
                K[i, j] = U[i, j] * B[j] / (B[i] + 1e-300)
        return K

    def A_i(i, r, sign):
        """A_i(r) = 2*kappa_i * B'_nu(kappa_i r) / B_nu(kappa_i r),
        regime-aware via wkb.ivp/wkb.kvp."""
        r_eff = max(r, r_eps)
        kap = kappa(i)
        z = kap * r_eff
        if sign == "+":
            Bv = wkb.kv(nu, z)
            Bp = wkb.kvp(nu, z)
        else:
            Bv = wkb.iv(nu, z)
            Bp = wkb.ivp(nu, z)
        return 2.0 * kap * Bp / (Bv + 1e-300)

    return (
        B, dB, K_matrix, A_i, U_int_pp, M_free,
        X_prime_bounce, Y_prime_bounce
    )


def rhs_h_ivp(r, y, sign, sol_index, K_matrix, A_i):
    h1, h2, v1, v2 = y
    k = K_matrix(r, sign)
    invr = 0.0 if r == 0.0 else 1.0 / r

    s0 = k[0, sol_index]
    s1 = k[1, sol_index]

    dv1 = (-(invr + A_i(0, r, sign)) * v1
           + k[0, 0] * h1 + k[0, 1] * h2 + s0)
    dv2 = (-(invr + A_i(1, r, sign)) * v2
           + k[1, 0] * h1 + k[1, 1] * h2 + s1)
    return np.array([v1, v2, dv1, dv2])


def solve_branch(r_start, r_end, y0, sign, sol_index, K_matrix, A_i, r_eval):
    def capture(r, y):
        return rhs_h_ivp(r, y, sign, sol_index, K_matrix, A_i)

    sol = solve_ivp(
        capture,
        (r_start, r_end),
        y0,
        method="Radau",
        t_eval=r_eval,
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    return sol.t, sol.y.T


def build_rk_green_for_bounce_wkb(bounce_npz_filename, s2, n_mode,
                                  n_eval=2000, r0=1e-4,
                                  out_fname=None, overwrite=False):
    """WKB Stage-A bounce-side RK Green's function builder.

    Drop-in replacement for build_rk_green_for_bounce, except:
      - Bessel evaluations use wkb_bessel (numerically stable)
      - Output filename gets `_wkb_vfinal` suffix by default
    """
    data = np.load(bounce_npz_filename, allow_pickle=True)

    params = data["params"]
    false_vac = np.asarray(data["false_vac"], dtype=float)
    true_vac = np.asarray(data["true_vac"], dtype=float)
    r_bounce = data["R"]

    if "X_bounce_prime" not in data.files or "Y_bounce_prime" not in data.files:
        raise RuntimeError(
            f"{bounce_npz_filename} does not contain X_bounce_prime/Y_bounce_prime."
        )
    x_prime = data["X_bounce_prime"]
    y_prime = data["Y_bounce_prime"]

    tag = str(data["tag"]) if "tag" in data.files else ""
    false_index = int(data["false_index"]) if "false_index" in data.files else -1
    true_index = int(data["true_index"]) if "true_index" in data.files else 0

    if out_fname is None:
        out_fname = (f"rk_green_data_F{false_index}_T{true_index}"
                     f"_n{n_mode}_s2{s2:.6f}_wkb_vfinal.npz".replace(".", "p", 1))
    if (not overwrite) and os.path.exists(out_fname):
        print(f"\n[SKIP] {out_fname} already exists "
              f"(set overwrite=True to rebuild).")
        return out_fname

    print("\n===============================================")
    print("Building RK Green (Stage-A WKB v1) for bounce file:",
          bounce_npz_filename)
    print("  tag        =", tag)
    print("  R range    =", r_bounce[0], "→", r_bounce[-1])
    print("  s2 = ", s2, ", n_mode = ", n_mode)
    print("===============================================")

    pot_lin = CTShiftedLiftedPotential(params, false_vac)
    L = getattr(pot_lin, "L", None)

    (B, dB, K_matrix, A_i, _Q_matrix, M_free,
     x_prime_bounce, y_prime_bounce) = build_fluctuation_data_prime_wkb(
        pot_lin, r_bounce, x_prime, y_prime, s2, n_mode,
    )

    r_start = max(float(r_bounce[0]), r0)
    r_max = float(r_bounce[-1])
    r_grid = np.linspace(r_start, r_max, n_eval)
    y0 = np.array([0.0, 0.0, 0.0, 0.0])

    _r_minus, y_minus_1 = solve_branch(
        r_start, r_max, y0, "-", 0, K_matrix, A_i, r_grid
    )
    _r_minus2, y_minus_2 = solve_branch(
        r_start, r_max, y0, "-", 1, K_matrix, A_i, r_grid
    )

    r_grid_desc = r_grid[::-1]
    _r_plus, y_plus_1 = solve_branch(
        r_max, r_start, y0, "+", 0, K_matrix, A_i, r_grid_desc
    )
    _r_plus2, y_plus_2 = solve_branch(
        r_max, r_start, y0, "+", 1, K_matrix, A_i, r_grid_desc
    )
    y_plus_1 = y_plus_1[::-1, :]
    y_plus_2 = y_plus_2[::-1, :]

    nr = len(r_grid)
    h_minus = np.zeros((nr, 2, 2))
    dh_minus = np.zeros((nr, 2, 2))
    h_plus = np.zeros((nr, 2, 2))
    dh_plus = np.zeros((nr, 2, 2))

    h_minus[:, 0, 0], h_minus[:, 1, 0] = y_minus_1[:, 0], y_minus_1[:, 1]
    dh_minus[:, 0, 0], dh_minus[:, 1, 0] = y_minus_1[:, 2], y_minus_1[:, 3]
    h_minus[:, 0, 1], h_minus[:, 1, 1] = y_minus_2[:, 0], y_minus_2[:, 1]
    dh_minus[:, 0, 1], dh_minus[:, 1, 1] = y_minus_2[:, 2], y_minus_2[:, 3]

    h_plus[:, 0, 0], h_plus[:, 1, 0] = y_plus_1[:, 0], y_plus_1[:, 1]
    dh_plus[:, 0, 0], dh_plus[:, 1, 0] = y_plus_1[:, 2], y_plus_1[:, 3]
    h_plus[:, 0, 1], h_plus[:, 1, 1] = y_plus_2[:, 0], y_plus_2[:, 1]
    dh_plus[:, 0, 1], dh_plus[:, 1, 1] = y_plus_2[:, 2], y_plus_2[:, 3]

    print("\n[h-basis] solved adaptive system (WKB Stage A):")
    print(f"  r range: {r_grid[0]:.4e} -> {r_grid[-1]:.4f}, Nr={nr}")

    def build_f_df(sign):
        f = np.zeros((nr, 2, 2))
        df = np.zeros((nr, 2, 2))
        for k, r in enumerate(r_grid):
            for i in range(2):
                for alpha in range(2):
                    delta = 1.0 if i == alpha else 0.0
                    if sign == "+":
                        h = h_plus[k, i, alpha]
                        dh = dh_plus[k, i, alpha]
                        bi, dbi = B(i, r, "+"), dB(i, r, "+")
                    else:
                        h = h_minus[k, i, alpha]
                        dh = dh_minus[k, i, alpha]
                        bi, dbi = B(i, r, "-"), dB(i, r, "-")
                    f[k, i, alpha] = bi * (delta + h)
                    df[k, i, alpha] = dbi * (delta + h) + bi * dh
        return f, df

    f_plus, df_plus = build_f_df("+")
    f_minus, df_minus = build_f_df("-")

    B_plus = np.zeros((nr, 2))
    B_minus = np.zeros((nr, 2))
    for k, r in enumerate(r_grid):
        for i in range(2):
            B_plus[k, i] = B(i, r, "+")
            B_minus[k, i] = B(i, r, "-")

    w_raw = np.zeros((nr, 2, 2))
    for idx, _r in enumerate(r_grid):
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

    print("\n[r^3 Wronskian plateau]")
    print("  r_min_tail =", r_grid[i_min], " r_max_tail =", r_grid[i_max])
    print("  Omega (no symmetrization) =\n", omega)

    omega_inv_T = omega_inv.T

    # [vx-opt] The downstream consumer only reads the diagonal trace
    # trace(G_rk[k,k]); at k==l the (r>=rp) branch is taken, so
    # G_rk[k,k] = f_plus[k] @ omega_inv @ f_minus[k].T.  We compute its
    # trace directly and skip the O(nr^2) dense G_rk build entirely.
    # trace_diag[k] = sum_{i,b,c} f_plus[k,i,b] omega_inv[b,c] f_minus[k,i,c]
    # (VERIFIED bit-identical to the old np.trace(g_rk[k,k]) diagonal).
    trace_diag = np.einsum('kib,bc,kic->k', f_plus, omega_inv, f_minus)

    print("\n[RK Green WKB-A] computed trace_diag with shape", trace_diag.shape)

    np.savez(
        out_fname,
        r_grid=r_grid,
        trace_diag=trace_diag,
        f_plus=f_plus,
        f_minus=f_minus,
        df_plus=df_plus,
        df_minus=df_minus,
        h_plus=h_plus,
        h_minus=h_minus,
        B_plus=B_plus,
        B_minus=B_minus,
        Omega_inv=omega_inv,
        M_free=M_free,
        W_scaled=w_scaled,
        R_bounce=r_bounce,
        X_bounce_prime=x_prime_bounce,
        Y_bounce_prime=y_prime_bounce,
        false_vac=false_vac,
        true_vac=true_vac,
        L=L,
        params=params,
        s2=s2,
        n_mode=n_mode,
        tag=tag,
        false_index=false_index,
        true_index=true_index,
        wkb_stage="A",
    )

    print(f"\n[OK] Saved RK Green data (WKB Stage A) to {out_fname}")
    return out_fname
