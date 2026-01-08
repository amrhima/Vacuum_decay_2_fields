import glob
import numpy as np
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

    # ---- Bounce interpolation in PRIMED coordinates --------------------
    X_prime_bounce = np.asarray(X_prime_bounce, dtype=float)
    Y_prime_bounce = np.asarray(Y_prime_bounce, dtype=float)
    R_bounce       = np.asarray(R_bounce, dtype=float)

    def x_prime_of_r(r):
        return np.interp(r, R_bounce, X_prime_bounce)

    def y_prime_of_r(r):
        return np.interp(r, R_bounce, Y_prime_bounce)

    # ---- Primed Hessian from the linear-redefined potential -----------
    def H_prime(phi_prime):
        # CTShiftedLiftedPotential.H already applies H' = L^T H L
        return pot_lin.H(phi_prime)

    # ---- Free mass matrix in primed system (at false' = 0) ------------
    phi_prime_false = np.zeros(2, dtype=float)
    M_free = H_prime(phi_prime_false)      # 2×2 mass matrix at false vac

    # Diagonal entries as "free masses"
    m1_sq_free = M_free[0, 0]
    m2_sq_free = M_free[1, 1]

    print("\n[build_fluctuation_data_prime]")
    print("  M_free = H'(0) =\n", M_free)
    print("  m1_free^2 =", m1_sq_free, "  m2_free^2 =", m2_sq_free)
    print("  s2 =", s2, "  n_mode =", n_mode)

    ell_bessel = n_mode + 1      # From O(4), the Bessel order is: l_bessel = n+1
    r_eps = 1e-8

    # ---------- κ_i from free masses -----------------------------------
    def kappa(i):
        m_sq = m1_sq_free if i == 0 else m2_sq_free
        return np.sqrt(s2 + m_sq)

    # ---------- core Bessel (no 1/r) -----------------------------------
    # IMPORTANT: plus = decaying (K), minus = regular (I)
    def Bcore_plus(i, r):
        # decaying at large r (but singular at r→0)
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return kv(ell_bessel, z)

    def dBcore_plus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return kvp(ell_bessel, z)

    def Bcore_minus(i, r):
        # regular at the origin
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return iv(ell_bessel, z)

    def dBcore_minus(i, r):
        r_eff = max(r, r_eps)
        z = kappa(i) * r_eff
        return ivp(ell_bessel, z)

    # ---------- full free solutions B_i^{±} = Bcore / r ----------------
    def B(i, r, sign):
        r_eff = max(r, r_eps)
        if sign == "+":
            # decaying at infinity
            return Bcore_plus(i, r) / r_eff
        else:
            # regular at r=0
            return Bcore_minus(i, r) / r_eff

    def dB(i, r, sign):
        r_eff = max(r, r_eps)
        k = kappa(i)

        if sign == "+":
            Bc     = Bcore_plus(i, r)
            dBc_dz = dBcore_plus(i, r)
        else:
            Bc     = Bcore_minus(i, r)
            dBc_dz = dBcore_minus(i, r)

        dBc_dr = k * dBc_dz
        return (dBc_dr * r_eff - Bc) / (r_eff**2)

    # ---------- full Hessian along bounce in PRIMED coords -------------
    def H_full_prime_r(r):
        xp = x_prime_of_r(r)
        yp = y_prime_of_r(r)
        phi_prime = np.array([xp, yp], dtype=float)
        return H_prime(phi_prime)

    # ---------- Q(r) = H_full'(r) - M_free ------------------------------
    def U_int_pp(r):
        return H_full_prime_r(r) - M_free

    # ---------- K_matrix and A_i entering h-equation -------------------
    def K_matrix(r, sign):
        U = U_int_pp(r)
        K = np.zeros((2, 2))
        for i in range(2):
            Bi = B(i, r, sign)
            for j in range(2):
                Bj = B(j, r, sign)
                K[i, j] = U[i, j] * Bj / (Bi + 1e-30)
        return K

    def A_i(i, r, sign):
        if sign == "+":
            Bc     = Bcore_plus(i, r)
            dBc_dz = dBcore_plus(i, r)
        else:
            Bc     = Bcore_minus(i, r)
            dBc_dz = dBcore_minus(i, r)
        return 2.0 * kappa(i) * dBc_dz / (Bc + 1e-30)

    return (
        B, dB, K_matrix, A_i, U_int_pp, M_free,
        X_prime_bounce, Y_prime_bounce
    )


# ================================================================
# 2. h-basis RK integrator
# ================================================================
def rhs_h(r, y, sign, sol_index, K_matrix, A_i):

    # y = (h1, h2, v1, v2),  v_i = h_i'
    # Source: S_i(r) = K_{i, alpha}(r).
    h1, h2, v1, v2 = y
    K = K_matrix(r, sign)
    invr = 0.0 if r == 0.0 else 1.0 / r

    S0 = K[0, sol_index]
    S1 = K[1, sol_index]

    dv1 = (-(invr + A_i(0, r, sign)) * v1
           + K[0, 0] * h1 + K[0, 1] * h2 + S0)
    dv2 = (-(invr + A_i(1, r, sign)) * v2
           + K[1, 0] * h1 + K[1, 1] * h2 + S1)

    return np.array([v1, v2, dv1, dv2])


def rk4_h(r0, r1, y0, N, sign, sol_index,
          K_matrix, A_i):

    r_grid = np.linspace(r0, r1, N + 1)
    dr = (r1 - r0) / N
    Y = np.zeros((N + 1, len(y0)))
    Y[0] = y0

    for k in range(N):
        r = r_grid[k]
        y = Y[k]
        k1 = rhs_h(r,           y,             sign, sol_index, K_matrix, A_i)
        k2 = rhs_h(r + dr/2.0,  y + dr*k1/2.0, sign, sol_index, K_matrix, A_i)
        k3 = rhs_h(r + dr/2.0,  y + dr*k2/2.0, sign, sol_index, K_matrix, A_i)
        k4 = rhs_h(r + dr,      y + dr*k3,     sign, sol_index, K_matrix, A_i)
        Y[k+1] = y + (dr/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    return r_grid, Y


# ================================================================
# 3. Build RK Green for ONE bounce (no plotting)
# ================================================================
def build_rk_green_for_bounce(bounce_npz_filename,
                              s2, n_mode):

    data = np.load(bounce_npz_filename, allow_pickle=True)

    params    = data["params"]
    false_vac = np.asarray(data["false_vac"], dtype=float)   # Original coord.
    true_vac  = np.asarray(data["true_vac"],  dtype=float)   # Original coord.
    R_bounce  = data["R"]

    # primed bounce fields (these are what bounce.py now saves)
    if "X_bounce_prime" not in data.files or "Y_bounce_prime" not in data.files:
        raise RuntimeError(
            f"{bounce_npz_filename} does not contain X_bounce_prime/Y_bounce_prime."
        )
    X_prime = data["X_bounce_prime"]
    Y_prime = data["Y_bounce_prime"]

    # optional metadata
    tag = str(data["tag"]) if "tag" in data.files else ""
    false_index = int(data["false_index"]) if "false_index" in data.files else -1
    true_index  = int(data["true_index"])  if "true_index"  in data.files else 0

    print("\n===============================================")
    print("Building RK Green for bounce file:", bounce_npz_filename)
    print("  tag        =", tag)
    print("  false_vac  (orig) =", false_vac)
    print("  true_vac   (orig) =", true_vac)
    print("  R range    =", R_bounce[0], "→", R_bounce[-1])
    print("===============================================")

    # linear field–redefined + lifted potential in primed coordinates
    pot_lin = CTShiftedLiftedPotential(params, false_vac)
    L = getattr(pot_lin, "L", None)  # stored for plotting

    # build fluctuation data in primed system
    (B, dB, K_matrix, A_i, Q_matrix, M_free,
     X_prime_bounce, Y_prime_bounce) = build_fluctuation_data_prime(
        pot_lin,
        R_bounce,
        X_prime,
        Y_prime,
        s2,
        n_mode
    )

    # solve h-equation via RK for ± branches, α = 0,1
    r0   = max(R_bounce[0], 1e-4)
    Rmax = R_bounce[-1]
    Nsteps = 2000
    y0 = np.array([0.0, 0.0, 0.0, 0.0])

    # minus branch ("regular", I_ν) from small r outwards
    r_minus_1, Y_minus_1 = rk4_h(r0,   Rmax, y0, Nsteps, "-", 0, K_matrix, A_i)
    r_minus_2, Y_minus_2 = rk4_h(r0,   Rmax, y0, Nsteps, "-", 1, K_matrix, A_i)

    # plus branch ("decaying", K_ν) from large r inwards
    r_plus_1,  Y_plus_1  = rk4_h(Rmax, r0,   y0, Nsteps, "+", 0, K_matrix, A_i)
    r_plus_2,  Y_plus_2  = rk4_h(Rmax, r0,   y0, Nsteps, "+", 1, K_matrix, A_i)

    # r-grid is the regular branch's ascending grid
    r_grid = r_minus_1
    Nr = len(r_grid)

    # "-" branch (regular) from r0 → Rmax
    h_minus = np.zeros((Nr, 2, 2))
    dh_minus = np.zeros((Nr, 2, 2))
    h_minus[:, 0, 0], h_minus[:, 1, 0] = Y_minus_1[:, 0], Y_minus_1[:, 1]
    dh_minus[:, 0, 0], dh_minus[:, 1, 0] = Y_minus_1[:, 2], Y_minus_1[:, 3]
    h_minus[:, 0, 1], h_minus[:, 1, 1] = Y_minus_2[:, 0], Y_minus_2[:, 1]
    dh_minus[:, 0, 1], dh_minus[:, 1, 1] = Y_minus_2[:, 2], Y_minus_2[:, 3]

    # "+" branch (decaying) from Rmax → r0, reversed to increasing r
    r_plus_inc = r_plus_1[::-1]
    assert np.allclose(r_plus_inc, r_grid, rtol=1e-6, atol=1e-8)
    Y_plus_1_rev = Y_plus_1[::-1, :]
    Y_plus_2_rev = Y_plus_2[::-1, :]

    h_plus = np.zeros((Nr, 2, 2))
    dh_plus = np.zeros((Nr, 2, 2))
    h_plus[:, 0, 0], h_plus[:, 1, 0] = Y_plus_1_rev[:, 0], Y_plus_1_rev[:, 1]
    dh_plus[:, 0, 0], dh_plus[:, 1, 0] = Y_plus_1_rev[:, 2], Y_plus_1_rev[:, 3]
    h_plus[:, 0, 1], h_plus[:, 1, 1] = Y_plus_2_rev[:, 0], Y_plus_2_rev[:, 1]
    dh_plus[:, 0, 1], dh_plus[:, 1, 1] = Y_plus_2_rev[:, 2], Y_plus_2_rev[:, 3]

    print("\n[h-basis] solved RK system:")
    print("  r range:", r_grid[0], "→", r_grid[-1])
    print("  Nr     =", Nr)

    # reconstruct full modes f^{±,α} and f'^{±,α}
    def build_f_df(sign):
        f = np.zeros((Nr, 2, 2))
        df = np.zeros((Nr, 2, 2))
        for k, r in enumerate(r_grid):
            for i in range(2):
                for alpha in range(2):
                    delta = 1.0 if i == alpha else 0.0
                    if sign == "+":
                        h  = h_plus[k, i, alpha]
                        dh = dh_plus[k, i, alpha]
                        Bi, dBi = B(i, r, "+"), dB(i, r, "+")
                    else:
                        h  = h_minus[k, i, alpha]
                        dh = dh_minus[k, i, alpha]
                        Bi, dBi = B(i, r, "-"), dB(i, r, "-")
                    f[k, i, alpha]  = Bi * (delta + h)
                    df[k, i, alpha] = dBi * (delta + h) + Bi * dh
        return f, df

    # f_plus  = decaying branch
    # f_minus = regular branch
    f_plus, df_plus = build_f_df("+")
    f_minus, df_minus = build_f_df("-")

    # free modes B^± on r_grid 
    # B_plus  = decaying, B_minus = regular
    B_plus = np.zeros((Nr, 2))
    B_minus = np.zeros((Nr, 2))
    for k, r in enumerate(r_grid):
        for i in range(2):
            B_plus[k, i]  = B(i, r, "+")  # decaying
            B_minus[k, i] = B(i, r, "-")  # regular

    # Wronskian Ω and Ω^{-1}
    # Canonical choice for the Green's function:
    # W_{αβ}(r) = sum_i [ f_i^{-,α} f_i^{+,β}' - f_i^{+,β} f_i^{-,α}' ]
    r_eps_const = 1e-8
    W_raw = np.zeros((Nr, 2, 2))
    for idx, r in enumerate(r_grid):
        W = np.zeros((2, 2))
        for alpha in range(2):
            for beta in range(2):
                s = 0.0
                for i in range(2):
                    fm_a  = f_minus[idx, i, alpha]   
                    fp_b  = f_plus[idx, i, beta]     
                    dfm_a = df_minus[idx, i, alpha]
                    dfp_b = df_plus[idx, i, beta]
                    s += fm_a * dfp_b - fp_b * dfm_a
                W[alpha, beta] = -s
        W_raw[idx] = W

    W_scaled = np.zeros_like(W_raw)
    for idx, r in enumerate(r_grid):
        r_eff = max(r, r_eps_const)
        W_scaled[idx] = (r_eff**3) * W_raw[idx]

    r_min_tail = 0.05
    r_max_tail = 0.9 * r_grid[-1]
    i_min = np.searchsorted(r_grid, r_min_tail)
    i_max = np.searchsorted(r_grid, r_max_tail)

    W_tail = W_scaled[i_min:i_max+1, :, :]

# Plateau estimate of C_{αβ} = r^3 W_{αβ}
    Omega = np.mean(W_tail, axis=0)
    Omega_inv = np.linalg.inv(Omega)

    print("\n[r^3 Wronskian plateau]")
    print("  r_min_tail =", r_grid[i_min], " r_max_tail =", r_grid[i_max])
    print("  Omega (no symmetrization) =\n", Omega)

    # Build RK-Green's function G_rk(r,r'):
    G_rk = np.zeros((Nr, Nr, 2, 2))
    for k, r in enumerate(r_grid):
        F_dec_r = f_plus[k]    # decaying
        F_reg_r = f_minus[k]   # regular
        for l, rp in enumerate(r_grid):
            F_dec_rp = f_plus[l]
            F_reg_rp = f_minus[l]
            if r >= rp:
                F_plus   = F_dec_r
                F_minus = F_reg_rp
            else:
                F_plus   = F_dec_rp
                F_minus = F_reg_r
            G_rk[k, l] = F_plus @ Omega_inv @ F_minus.T

    print("\n[RK Green] built G_rk with shape", G_rk.shape)

    # save everything for this bounce
    out_fname = f"rk_green_data_F{false_index}_T{true_index}.npz"

    np.savez(
        out_fname,
        r_grid=r_grid,
        G_rk=G_rk,
        f_plus=f_plus,
        f_minus=f_minus,
        df_plus=df_plus,
        df_minus=df_minus,
        h_plus=h_plus,
        h_minus=h_minus,
        B_plus=B_plus,
        B_minus=B_minus,
        Omega_inv=Omega_inv,
        M_free=M_free,
        W_scaled=W_scaled,
        # bounce data in primed coords:
        R_bounce=R_bounce,
        X_bounce_prime=X_prime_bounce,
        Y_bounce_prime=Y_prime_bounce,
        # some metadata:
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


# ================================================================
# 4. Main: loop over all bounce_data_F*_T*.npz
# ================================================================
if __name__ == "__main__":

    # choose fluctuation parameters (same for all bounces)
    s2    = 2.8
    n_mode = 3

    # process ALL bounces, independent of T index 
    bounce_files = sorted(glob.glob("bounce_data_F*_T*.npz"))

    if not bounce_files:
        print("[ERROR] No bounce_data_F*_T*.npz files found.")
        raise SystemExit

    print("[INFO] Found bounce files:")
    for bf in bounce_files:
        print("  ", bf)

    for bf in bounce_files:
        build_rk_green_for_bounce(bf, s2=s2, n_mode=n_mode)

    print("\n[INFO] Finished building RK Greens for all bounces.")