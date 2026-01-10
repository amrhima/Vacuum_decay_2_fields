import glob
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import iv, ivp, kv, kvp
from potential import CTShiftedLiftedPotential


# ================================================================
# 1. Fluctuation data in PRIMED coordinates for one bounce
# ================================================================
def build_fluctuation_data_prime(pot_lin,
                                 R_bounce,
                                 X_prime_bounce,
                                 Y_prime_bounce,
                                 s2,
                                 n_mode):

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

    print("\n[build_fluctuation_data_prime]")
    print("  M_free = H'(0) =\n", M_free)
    print("  m1_free^2 =", m1_sq_free, "  m2_free^2 =", m2_sq_free)
    print("  s2 =", s2, "  n_mode =", n_mode)

    ell_bessel = n_mode + 1
    r_eps = 1e-8

    def kappa(i):
        m_sq = m1_sq_free if i == 0 else m2_sq_free
        return np.sqrt(s2 + m_sq)

    def Bcore_plus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return kv(ell_bessel, z)

    def dBcore_plus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return kvp(ell_bessel, z)

    def Bcore_minus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return iv(ell_bessel, z)

    def dBcore_minus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return ivp(ell_bessel, z)

    def B(i, r, sign):
        r_eff = max(r, r_eps)
        if sign == "+":
            return Bcore_plus(i, r) / r_eff
        return Bcore_minus(i, r) / r_eff

    def dB(i, r, sign):
        r_eff = max(r, r_eps)
        k = kappa(i)
        if sign == "+":
            bc = Bcore_plus(i, r)
            dbc_dz = dBcore_plus(i, r)
        else:
            bc = Bcore_minus(i, r)
            dbc_dz = dBcore_minus(i, r)
        dbc_dr = k * dbc_dz
        return (dbc_dr * r_eff - bc) / (r_eff ** 2)

    def H_full_prime_r(r):
        xp = x_prime_of_r(r)
        yp = y_prime_of_r(r)
        phi_prime = np.array([xp, yp], dtype=float)
        return H_prime(phi_prime)

    def U_int_pp(r):
        return H_full_prime_r(r) - M_free

    def K_matrix(r, sign):
        U = U_int_pp(r)
        K = np.zeros((2, 2))
        for i in range(2):
            bi = B(i, r, sign)
            for j in range(2):
                bj = B(j, r, sign)
                K[i, j] = U[i, j] * bj / (bi + 1e-30)
        return K

    def A_i(i, r, sign):
        if sign == "+":
            bc = Bcore_plus(i, r)
            dbc_dz = dBcore_plus(i, r)
        else:
            bc = Bcore_minus(i, r)
            dbc_dz = dBcore_minus(i, r)
        return 2.0 * kappa(i) * dbc_dz / (bc + 1e-30)

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


def build_rk_green_for_bounce(bounce_npz_filename, s2, n_mode,
                              n_eval=2000, r0=1e-4):
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

    print("\n===============================================")
    print("Building RK Green (adaptive) for bounce file:", bounce_npz_filename)
    print("  tag        =", tag)
    print("  false_vac  (orig) =", false_vac)
    print("  true_vac   (orig) =", true_vac)
    print("  R range    =", r_bounce[0], "→", r_bounce[-1])
    print("===============================================")

    pot_lin = CTShiftedLiftedPotential(params, false_vac)
    L = getattr(pot_lin, "L", None)

    (B, dB, K_matrix, A_i, Q_matrix, M_free,
     x_prime_bounce, y_prime_bounce) = build_fluctuation_data_prime(
        pot_lin,
        r_bounce,
        x_prime,
        y_prime,
        s2,
        n_mode,
    )

    r_start = max(float(r_bounce[0]), r0)
    r_max = float(r_bounce[-1])
    r_grid = np.linspace(r_start, r_max, n_eval)
    y0 = np.array([0.0, 0.0, 0.0, 0.0])

    r_minus_1, y_minus_1 = solve_branch(
        r_start, r_max, y0, "-", 0, K_matrix, A_i, r_grid
    )
    _r_minus, y_minus_2 = solve_branch(
        r_start, r_max, y0, "-", 1, K_matrix, A_i, r_grid
    )

    r_grid_desc = r_grid[::-1]
    r_plus_1, y_plus_1 = solve_branch(
        r_max, r_start, y0, "+", 0, K_matrix, A_i, r_grid_desc
    )
    _r_plus, y_plus_2 = solve_branch(
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

    print("\n[h-basis] solved adaptive system:")
    print("  r range:", r_grid[0], "→", r_grid[-1])
    print("  Nr     =", nr)

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

    g_rk = np.zeros((nr, nr, 2, 2))
    for k, r in enumerate(r_grid):
        f_dec_r = f_plus[k]
        f_reg_r = f_minus[k]
        for l, rp in enumerate(r_grid):
            f_dec_rp = f_plus[l]
            f_reg_rp = f_minus[l]
            if r >= rp:
                f_plus_use = f_dec_r
                f_minus_use = f_reg_rp
            else:
                f_plus_use = f_dec_rp
                f_minus_use = f_reg_r
            g_rk[k, l] = f_plus_use @ omega_inv @ f_minus_use.T

    print("\n[RK Green] built G_rk with shape", g_rk.shape)

    out_fname = f"rk_green_data_F{false_index}_T{true_index}.npz"

    np.savez(
        out_fname,
        r_grid=r_grid,
        G_rk=g_rk,
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
    )

    print(f"\n[OK] Saved RK Green data to {out_fname}")


if __name__ == "__main__":
    s2 = 2.8
    n_mode = 3
    bounce_files = sorted(glob.glob("bounce_data_F*_T*.npz"))

    if not bounce_files:
        print("[ERROR] No bounce_data_F*_T*.npz files found.")
        raise SystemExit

    print("[INFO] Found bounce files:")
    for bf in bounce_files:
        print("  ", bf)

    for bf in bounce_files:
        build_rk_green_for_bounce(bf, s2=s2, n_mode=n_mode)

    print("\n[INFO] Finished building RK Greens (adaptive) for all bounces.")
